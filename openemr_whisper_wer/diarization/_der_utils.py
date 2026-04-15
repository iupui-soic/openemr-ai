"""
Shared utilities for Diarization Error Rate (DER) computation.

Contains:
- `load_pyannote_pipeline()` — pyannote 3.1 speaker-diarization pipeline (cached)
- `textgrid_to_annotation()` — PriMock57 TextGrid (Praat) -> pyannote.core.Annotation
- `whisperx_align_transcript()` — forced-align a known transcript to audio,
    returns word-level intervals tagged with the speaker label parsed from
    the transcript's `D:` / `P:` prefix; used for Fareez + institutional where
    no gold timestamps exist
- `compute_der()` — wrapper around pyannote.metrics.DiarizationErrorRate
- `pyannote_diarize()` — run the pipeline on an audio file, return Annotation
- `audio_duration()` — quick read of file duration via torchaudio metadata
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Lazy imports inside functions for everything heavy (torch / pyannote /
# whisperx) so test harnesses can import the module cheaply.

_PIPELINE = None


def load_pyannote_pipeline(use_gpu: bool = True):
    """Return a cached pyannote.audio Pipeline for speaker-diarization-3.1."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    from pyannote.audio import Pipeline
    import torch

    hf_token = os.environ.get("HF_TOKEN") or _read_hf_token_file()
    if not hf_token:
        raise RuntimeError(
            "No HF_TOKEN found. Either set env var or run "
            "`huggingface-cli login` (which writes ~/.cache/huggingface/token)."
        )

    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    if use_gpu and torch.cuda.is_available():
        pipe = pipe.to(torch.device("cuda"))
        print(f"  pyannote pipeline on cuda")
    else:
        print(f"  pyannote pipeline on cpu")
    _PIPELINE = pipe
    return pipe


def _read_hf_token_file() -> Optional[str]:
    p = Path.home() / ".cache" / "huggingface" / "token"
    if p.exists():
        return p.read_text().strip()
    return None


def textgrid_to_annotation(textgrid_path: str, speaker_label: str):
    """
    Parse a Praat TextGrid (PriMock57 format: one IntervalTier of utterances)
    into a pyannote Annotation with all intervals labelled with `speaker_label`.

    Empty / silence intervals are skipped.
    """
    import textgrid as tg
    from pyannote.core import Annotation, Segment

    grid = tg.TextGrid.fromFile(textgrid_path)
    ann = Annotation()
    for tier in grid.tiers:
        for interval in tier.intervals:
            txt = (interval.mark or "").strip()
            if not txt:
                continue
            seg = Segment(float(interval.minTime), float(interval.maxTime))
            ann[seg] = speaker_label
    return ann


def merge_annotations(annotations: list):
    """Merge a list of pyannote Annotations into one."""
    from pyannote.core import Annotation
    out = Annotation()
    for ann in annotations:
        for segment, _, label in ann.itertracks(yield_label=True):
            out[segment] = label
    return out


def pyannote_diarize(audio_path: str, num_speakers: Optional[int] = 2):
    """
    Run the cached pyannote pipeline on an audio file and return its Annotation.

    We pre-load audio with soundfile -> torch.Tensor and pass it as a dict to
    avoid pyannote's torchcodec-based decoder (torchcodec needs libavutil
    shared libs that aren't installed alongside our static ffmpeg).
    """
    import soundfile as sf
    import torch
    import numpy as np

    pipeline = load_pyannote_pipeline()
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    # pyannote expects (channel, time) torch.Tensor
    waveform = torch.from_numpy(data.T).contiguous()
    out = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
    # pyannote 4.x returns DiarizeOutput; older versions return Annotation directly
    if hasattr(out, "speaker_diarization"):
        return out.speaker_diarization
    return out


def compute_der(reference, hypothesis, audio_duration: Optional[float] = None) -> Dict[str, float]:
    """
    Returns dict with DER and its components (false_alarm, missed_detection,
    confusion). Relies on pyannote.metrics' optimal speaker mapping.
    """
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.core import Segment, Timeline

    metric = DiarizationErrorRate(skip_overlap=False, collar=0.25)
    components = metric.compute_components(
        reference, hypothesis,
        uem=Timeline([Segment(0, audio_duration)]) if audio_duration else None,
    )
    rate = metric.compute_metric(components)
    total = components.get("total", 0)
    return {
        "der": rate,
        "false_alarm_s": components.get("false alarm", 0.0),
        "missed_detection_s": components.get("missed detection", 0.0),
        "confusion_s": components.get("confusion", 0.0),
        "total_speech_s": total,
    }


def audio_duration(audio_path: str) -> float:
    """Return audio duration in seconds via soundfile (no torchcodec required)."""
    import soundfile as sf
    info = sf.info(audio_path)
    return info.frames / info.samplerate


# ---------------------------------------------------------------------------
# WhisperX forced alignment for transcripts that lack timestamps
# ---------------------------------------------------------------------------

_ALIGN_MODEL = None


def _load_align_model(language_code: str = "en"):
    global _ALIGN_MODEL
    if _ALIGN_MODEL is not None:
        return _ALIGN_MODEL
    import whisperx
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    _ALIGN_MODEL = (model, metadata, device)
    return _ALIGN_MODEL


_DP_RE = re.compile(r"^\s*(D|P)\s*[:.\-]\s*(.*)", re.IGNORECASE)


def parse_dp_transcript(transcript_text: str) -> List[Tuple[str, str]]:
    """
    Parse a Fareez/institutional D:/P: transcript into a list of (speaker, text)
    tuples, where speaker is "D" or "P". Lines without a prefix are appended
    to the previous turn.
    """
    turns: List[Tuple[str, str]] = []
    current_speaker = None
    current_text: List[str] = []

    for line in transcript_text.splitlines():
        m = _DP_RE.match(line)
        if m:
            if current_speaker is not None and current_text:
                turns.append((current_speaker, " ".join(current_text).strip()))
            current_speaker = m.group(1).upper()
            current_text = [m.group(2).strip()]
        else:
            stripped = line.strip()
            if stripped:
                current_text.append(stripped)
    if current_speaker is not None and current_text:
        turns.append((current_speaker, " ".join(current_text).strip()))
    return [(spk, txt) for (spk, txt) in turns if txt]


def whisperx_align_transcript(audio_path: str, transcript_text: str):
    """
    Forced-align a known D:/P: transcript to audio using whisperx's wav2vec2
    alignment model. Returns a pyannote Annotation with each utterance labelled
    "DOCTOR" or "PATIENT".

    Implementation:
    1. Parse transcript into D:/P: turns.
    2. Form whisperx-compatible "transcription" segments by spacing turns evenly
       across the audio's duration as initial guesses (alignment refines them).
    3. Run whisperx.align() to recover word-level timestamps.
    4. Group words back into utterances using the original turn boundaries
       (cumulative word count) and emit one Annotation segment per utterance,
       labelled with the speaker.
    """
    import whisperx

    model_a, metadata, device = _load_align_model("en")
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / 16000.0

    turns = parse_dp_transcript(transcript_text)
    if not turns:
        from pyannote.core import Annotation
        return Annotation(), duration

    # Build initial whisperx segments: split the audio time evenly across turns
    # weighted by word count. Alignment refines start/end per word.
    word_counts = [max(1, len(t.split())) for _, t in turns]
    total_words = sum(word_counts)
    cum = 0
    segments = []
    for (spk, text), wc in zip(turns, word_counts):
        start = duration * cum / total_words
        end = duration * (cum + wc) / total_words
        segments.append({"start": start, "end": end, "text": text})
        cum += wc

    aligned = whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Group words back into utterances. whisperx returns aligned["segments"]
    # (one per input segment) each containing words[] with start/end.
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for (spk, _), seg in zip(turns, aligned["segments"]):
        words = seg.get("words", [])
        # Find the actual extent from word timestamps
        starts = [w["start"] for w in words if "start" in w]
        ends = [w["end"] for w in words if "end" in w]
        if not starts or not ends:
            # Fall back to segment-level start/end if alignment failed
            starts = [seg.get("start", 0)]
            ends = [seg.get("end", 0)]
        s, e = min(starts), max(ends)
        if e <= s:
            continue
        speaker_label = "DOCTOR" if spk == "D" else "PATIENT"
        ann[Segment(float(s), float(e))] = speaker_label
    return ann, duration


def map_pyannote_to_dp(hyp_annotation, ref_annotation):
    """
    Convenience: given a pyannote diarization (SPEAKER_00, SPEAKER_01, ...) and
    a reference Annotation labelled DOCTOR/PATIENT, find the optimal mapping
    of hypothesis labels to DOCTOR/PATIENT and return a relabelled hypothesis.

    This is useful for human inspection but NOT required for DER (pyannote.metrics
    does optimal mapping internally).
    """
    from pyannote.metrics.diarization import DiarizationErrorRate
    metric = DiarizationErrorRate()
    mapping = metric.optimal_mapping(reference=ref_annotation, hypothesis=hyp_annotation)
    return hyp_annotation.rename_labels(mapping=mapping)
