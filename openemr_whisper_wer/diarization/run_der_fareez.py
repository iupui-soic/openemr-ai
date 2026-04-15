#!/usr/bin/env python
"""
Compute DER on Fareez OSCE.

Fareez transcripts have D:/P: speaker labels but NO timestamps. Our reference is
therefore a *silver-standard* derived from WhisperX wav2vec2 forced alignment of
the original transcript text against the audio, with the speaker label propagated
from the D:/P: prefix. This reference is then compared against pyannote 3.1
speaker-diarization output to produce DER.

Caveat: errors in the forced-alignment will show up as DER, so the absolute
number is an upper bound on the true (unobservable) DER. We acknowledge this in
the paper limitations and report PriMock57 DER (true gold) alongside.

Usage:
    python run_der_fareez.py
    python run_der_fareez.py --limit 5
    python run_der_fareez.py --subset 40   # only the 40 cases used for clinician eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WER_ROOT / "diarization"))

from _der_utils import (   # noqa: E402
    audio_duration,
    compute_der,
    pyannote_diarize,
    whisperx_align_transcript,
)

FAREEZ_AUDIO_DIR = WER_ROOT / "data" / "fareez_osce" / "wav_16khz"
FAREEZ_TRANSCRIPT_DIR = WER_ROOT / "data" / "fareez_osce" / "Data" / "Clean Transcripts"
SELECTED_40_JSON = REPO_ROOT / "rag_models" / "data" / "fareez_selected_40.json"
OUT_CSV = WER_ROOT / "results" / "fareez_der.csv"


def list_files(subset_40: bool = False) -> list[str]:
    if subset_40:
        with open(SELECTED_40_JSON) as f:
            sel = json.load(f)
        return sorted(item["file_name"] for item in sel)
    return sorted(p.stem for p in FAREEZ_AUDIO_DIR.glob("*.wav"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subset", choices=["all", "40"], default="all",
                    help="Run on all 272 or only the 40 used for clinician eval")
    ap.add_argument("--output", default=str(OUT_CSV))
    args = ap.parse_args()

    files = list_files(subset_40=(args.subset == "40"))
    if args.limit:
        files = files[: args.limit]
    print(f"Fareez DER: {len(files)} files (subset={args.subset})")

    rows = []
    t_global = time.time()
    for i, base in enumerate(files, 1):
        audio = FAREEZ_AUDIO_DIR / f"{base}.wav"
        transcript = FAREEZ_TRANSCRIPT_DIR / f"{base}.txt"
        if not audio.exists() or not transcript.exists():
            print(f"  [{i:3d}] {base}: missing audio or transcript")
            continue

        t0 = time.time()
        try:
            text = transcript.read_text(encoding="utf-8-sig", errors="replace")
            ref, duration = whisperx_align_transcript(str(audio), text)
            hyp = pyannote_diarize(str(audio), num_speakers=2)
            metrics = compute_der(ref, hyp, audio_duration=duration)

            row = {
                "file": base,
                "duration_s": round(duration, 2),
                "der": round(metrics["der"], 4),
                "false_alarm_s": round(metrics["false_alarm_s"], 2),
                "missed_detection_s": round(metrics["missed_detection_s"], 2),
                "confusion_s": round(metrics["confusion_s"], 2),
                "total_speech_s": round(metrics["total_speech_s"], 2),
                "compute_s": round(time.time() - t0, 2),
            }
            rows.append(row)
            print(f"  [{i:3d}/{len(files)}] {base:>10s}  DER={row['der']:.3f} "
                  f"({row['compute_s']:.1f}s)")
        except Exception as e:
            print(f"  [{i:3d}] {base}: ERROR {str(e)[:120]}")
            rows.append({
                "file": base, "duration_s": 0, "der": None,
                "false_alarm_s": None, "missed_detection_s": None,
                "confusion_s": None, "total_speech_s": 0,
                "compute_s": round(time.time() - t0, 2),
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}  ({len(df)} rows)")

    valid = df.dropna(subset=["der"])
    if len(valid):
        print(f"Mean DER:   {valid['der'].mean():.3f}")
        print(f"Median DER: {valid['der'].median():.3f}")
        print(f"Total compute: {(time.time() - t_global)/60:.1f} min")


if __name__ == "__main__":
    main()
