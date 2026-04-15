#!/usr/bin/env python
"""
Compute DER on the 6 institutional conversations.

Same silver-standard approach as Fareez: WhisperX wav2vec2 forced alignment of
the institutional transcripts against the audio (transcoded m4a -> 16 kHz mono
wav by `scripts/revision_setup.sh`), with speaker labels propagated from D:/P:
prefixes in the transcript. Compared against pyannote 3.1 diarization.

Six cases: anemia, arthritis, copd, dka, gallstones, gerd.

Note: the institutional transcript .txt files in `openemr_whisper_wer/data/`
are mostly empty (anemia.txt is 0 bytes; only the *_edge.txt variants have
content per `wc -l`). If a transcript is empty for a case, we fall back to a
reduced metric: speaker-count-only diarization quality (no DER).

Usage:
    python run_der_institutional.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

WER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WER_ROOT / "diarization"))

from _der_utils import (   # noqa: E402
    audio_duration,
    compute_der,
    pyannote_diarize,
    whisperx_align_transcript,
)

INST_WAV_DIR = WER_ROOT / "data" / "institutional" / "wav"
INST_DATA_DIR = WER_ROOT / "data"   # transcripts live in data/{case}.txt or data/{case}_edge.txt
OUT_CSV = WER_ROOT / "results" / "institutional_der.csv"

CASES = ["anemia", "arthritis", "copd", "dka", "gallstones", "gerd"]


def find_transcript(case: str) -> Path | None:
    """Pick the most usable transcript for a case (prefer non-empty)."""
    candidates = [
        INST_DATA_DIR / f"{case}_edge.txt",
        INST_DATA_DIR / f"{case}.txt",
        INST_DATA_DIR / f"{case}1.txt",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 50:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(OUT_CSV))
    args = ap.parse_args()

    rows = []
    t_global = time.time()
    for i, case in enumerate(CASES, 1):
        audio = INST_WAV_DIR / f"{case}.wav"
        transcript = find_transcript(case)
        if not audio.exists():
            print(f"  [{i}] {case}: missing audio at {audio}")
            continue

        t0 = time.time()
        try:
            duration = audio_duration(str(audio))
            hyp = pyannote_diarize(str(audio), num_speakers=2)
            n_hyp_speakers = len(hyp.labels())

            if transcript is None:
                # No usable transcript -> can't build silver reference. Report
                # diarization-only stats.
                row = {
                    "case": case,
                    "duration_s": round(duration, 2),
                    "der": None,
                    "false_alarm_s": None,
                    "missed_detection_s": None,
                    "confusion_s": None,
                    "total_speech_s": None,
                    "n_hyp_speakers": n_hyp_speakers,
                    "transcript_used": None,
                    "compute_s": round(time.time() - t0, 2),
                    "note": "no transcript with D:/P: turns -> DER not computable",
                }
            else:
                text = transcript.read_text(encoding="utf-8-sig", errors="replace")
                # Check whether transcript has D:/P: speaker labels we can use as reference.
                from _der_utils import parse_dp_transcript
                turns = parse_dp_transcript(text)
                if not turns:
                    # No speaker prefixes -> can't build silver reference. Diarization-only stats.
                    # Compute total hypothesis-detected speech and per-speaker durations.
                    speaker_durations = {}
                    for seg, _, lab in hyp.itertracks(yield_label=True):
                        speaker_durations[lab] = speaker_durations.get(lab, 0) + seg.duration
                    total_speech = sum(speaker_durations.values())
                    row = {
                        "case": case,
                        "duration_s": round(duration, 2),
                        "der": None,
                        "false_alarm_s": None,
                        "missed_detection_s": None,
                        "confusion_s": None,
                        "total_speech_s": round(total_speech, 2),
                        "n_hyp_speakers": n_hyp_speakers,
                        "transcript_used": transcript.name,
                        "compute_s": round(time.time() - t0, 2),
                        "note": ("transcript has no D:/P: speaker labels -> diarization-only "
                                 "stats reported; DER not computable. Speaker durations: "
                                 + "; ".join(f"{k}={v:.1f}s" for k, v in
                                             sorted(speaker_durations.items()))),
                    }
                else:
                    ref, _ = whisperx_align_transcript(str(audio), text)
                    metrics = compute_der(ref, hyp, audio_duration=duration)
                    row = {
                        "case": case,
                        "duration_s": round(duration, 2),
                        "der": round(metrics["der"], 4),
                        "false_alarm_s": round(metrics["false_alarm_s"], 2),
                        "missed_detection_s": round(metrics["missed_detection_s"], 2),
                        "confusion_s": round(metrics["confusion_s"], 2),
                        "total_speech_s": round(metrics["total_speech_s"], 2),
                        "n_hyp_speakers": n_hyp_speakers,
                        "transcript_used": transcript.name,
                        "compute_s": round(time.time() - t0, 2),
                        "note": "",
                    }
            rows.append(row)
            der_str = f"{row['der']:.3f}" if row['der'] is not None else "  ---"
            print(f"  [{i}] {case:>11s}  DER={der_str}  speakers={n_hyp_speakers}  "
                  f"({row['compute_s']:.1f}s)")
        except Exception as e:
            print(f"  [{i}] {case}: ERROR {str(e)[:120]}")
            rows.append({
                "case": case, "duration_s": 0, "der": None,
                "false_alarm_s": None, "missed_detection_s": None,
                "confusion_s": None, "total_speech_s": 0,
                "n_hyp_speakers": None, "transcript_used": None,
                "compute_s": round(time.time() - t0, 2),
                "note": f"error: {str(e)[:80]}",
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}  ({len(df)} rows)")

    valid = df.dropna(subset=["der"])
    if len(valid):
        print(f"Mean DER:   {valid['der'].mean():.3f}  (n={len(valid)})")
    print(f"Total compute: {(time.time() - t_global)/60:.1f} min")


if __name__ == "__main__":
    main()
