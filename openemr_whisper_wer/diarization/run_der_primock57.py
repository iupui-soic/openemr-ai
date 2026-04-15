#!/usr/bin/env python
"""
Compute Diarization Error Rate (DER) on PriMock57.

Uses TextGrid ground-truth speaker intervals (one per speaker per consultation)
and pyannote 3.1 speaker-diarization on the mixed_audio stereo file.

Output: openemr_whisper_wer/results/primock57_der.csv
        with per-consultation DER, missed-speech, false-alarm, speaker-confusion.

Usage:
    python run_der_primock57.py
    python run_der_primock57.py --limit 5    # smoke test on 5 consultations
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WER_ROOT / "diarization"))

from _der_utils import (   # noqa: E402
    audio_duration,
    compute_der,
    merge_annotations,
    pyannote_diarize,
    textgrid_to_annotation,
)

PRIMOCK_DIR = WER_ROOT / "data" / "primock57"
MIXED_AUDIO_DIR = PRIMOCK_DIR / "mixed_audio"
TRANSCRIPT_DIR = PRIMOCK_DIR / "transcripts"
OUT_CSV = WER_ROOT / "results" / "primock57_der.csv"


def list_consultations() -> list[str]:
    """Return sorted list of base names like 'day1_consultation01'."""
    bases = set()
    for p in TRANSCRIPT_DIR.glob("*_doctor.TextGrid"):
        bases.add(p.stem.replace("_doctor", ""))
    return sorted(bases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", default=str(OUT_CSV))
    args = ap.parse_args()

    consultations = list_consultations()
    if args.limit:
        consultations = consultations[: args.limit]
    print(f"PriMock57: {len(consultations)} consultations")

    rows = []
    t_global = time.time()
    for i, base in enumerate(consultations, 1):
        mixed_audio = MIXED_AUDIO_DIR / f"{base}_mixed.wav"
        doc_tg = TRANSCRIPT_DIR / f"{base}_doctor.TextGrid"
        pat_tg = TRANSCRIPT_DIR / f"{base}_patient.TextGrid"
        if not mixed_audio.exists():
            print(f"  [{i:2d}] {base}: missing audio")
            continue
        if not (doc_tg.exists() and pat_tg.exists()):
            print(f"  [{i:2d}] {base}: missing TextGrid")
            continue

        t0 = time.time()
        try:
            ref = merge_annotations([
                textgrid_to_annotation(str(doc_tg), "DOCTOR"),
                textgrid_to_annotation(str(pat_tg), "PATIENT"),
            ])
            duration = audio_duration(str(mixed_audio))
            hyp = pyannote_diarize(str(mixed_audio), num_speakers=2)
            metrics = compute_der(ref, hyp, audio_duration=duration)

            row = {
                "consultation": base,
                "duration_s": round(duration, 2),
                "der": round(metrics["der"], 4),
                "false_alarm_s": round(metrics["false_alarm_s"], 2),
                "missed_detection_s": round(metrics["missed_detection_s"], 2),
                "confusion_s": round(metrics["confusion_s"], 2),
                "total_speech_s": round(metrics["total_speech_s"], 2),
                "compute_s": round(time.time() - t0, 2),
            }
            rows.append(row)
            print(f"  [{i:2d}/{len(consultations)}] {base:>26s}  "
                  f"DER={row['der']:.3f}  ({row['compute_s']:.1f}s compute)")
        except Exception as e:
            print(f"  [{i:2d}] {base}: ERROR {str(e)[:100]}")
            rows.append({
                "consultation": base, "duration_s": 0, "der": None,
                "false_alarm_s": None, "missed_detection_s": None,
                "confusion_s": None, "total_speech_s": 0,
                "compute_s": round(time.time() - t0, 2),
            })

    # Save
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
