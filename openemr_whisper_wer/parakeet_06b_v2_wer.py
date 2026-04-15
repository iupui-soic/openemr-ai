#!/usr/bin/env python
"""
NVIDIA Parakeet TDT 0.6B v2 — local WER evaluation on PriMock57 + Fareez OSCE.

Why local: the existing `parakeet_wer.py` runs on Modal, but this is a one-shot
benchmark of a much smaller model (0.6B vs 1.1B). Local RTX 6000 Ada (49 GB) can
host it comfortably and avoids Modal cold-start overhead.

Usage:
    python parakeet_06b_v2_wer.py --dataset primock57
    python parakeet_06b_v2_wer.py --dataset fareez
    python parakeet_06b_v2_wer.py --dataset both         # default
    python parakeet_06b_v2_wer.py --dataset primock57 --limit 5

Outputs:
    results/primock57-parakeet-tdt-06b-v2.csv
    results/fareez-parakeet-tdt-06b-v2.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import jiwer
import pandas as pd
import soundfile as sf

WER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WER_ROOT))

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
SHORT_NAME = "parakeet-tdt-06b-v2"


def load_model():
    import nemo.collections.asr as nemo_asr
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    # Parakeet TDT models use EncDecRNNTBPEModel
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        model_name=MODEL_ID,
        map_location=device,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    return model


def transcribe_file(model, audio_path: str) -> str:
    """Resample to 16 kHz mono if needed, then transcribe."""
    import numpy as np
    import tempfile
    from scipy import signal

    data, sr = sf.read(audio_path, always_2d=True)
    # mono
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    # 16 kHz
    if sr != 16000:
        n = int(len(data) * 16000 / sr)
        data = signal.resample(data, n).astype("float32")
        sr = 16000
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, data, sr)
        tmp = f.name
    try:
        result = model.transcribe([tmp])
        if not result:
            return ""
        out = result[0]
        return out.text.strip() if hasattr(out, "text") else str(out).strip()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def calc_wer(reference: str, hypothesis: str) -> dict:
    transform = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
    ])
    ref = transform(reference)
    hyp = transform(hypothesis)
    out = jiwer.process_words(ref, hyp)
    return {
        "wer": out.wer, "mer": out.mer, "wil": out.wil,
        "insertions": out.insertions, "deletions": out.deletions,
        "substitutions": out.substitutions, "hits": out.hits,
    }


def run_primock57(model, output_csv: Path, limit: int | None):
    from primock57_utils import merge_channels

    audio_dir = WER_ROOT / "data" / "primock57" / "mixed_audio"
    tg_dir = WER_ROOT / "data" / "primock57" / "transcripts"
    bases = sorted({p.stem.replace("_doctor", "")
                    for p in tg_dir.glob("*_doctor.TextGrid")})
    if limit:
        bases = bases[:limit]

    rows = []
    for i, base in enumerate(bases, 1):
        audio = audio_dir / f"{base}_mixed.wav"
        doc_tg = tg_dir / f"{base}_doctor.TextGrid"
        pat_tg = tg_dir / f"{base}_patient.TextGrid"
        if not audio.exists() or not doc_tg.exists() or not pat_tg.exists():
            print(f"  [{i:2d}] {base}: missing files")
            continue
        ref = merge_channels(str(doc_tg), str(pat_tg))
        t0 = time.time()
        try:
            hyp = transcribe_file(model, str(audio))
            metrics = calc_wer(ref, hyp)
            row = {"name": base, "ground_truth": ref, "transcript": hyp,
                   **metrics, "compute_s": round(time.time() - t0, 2)}
        except Exception as e:
            print(f"  [{i:2d}] {base}: ERROR {str(e)[:100]}")
            row = {"name": base, "ground_truth": ref, "transcript": "",
                   "wer": None, "mer": None, "wil": None,
                   "insertions": None, "deletions": None,
                   "substitutions": None, "hits": None,
                   "compute_s": round(time.time() - t0, 2),
                   "error": str(e)[:200]}
        rows.append(row)
        if row.get("wer") is not None:
            print(f"  [{i:2d}/{len(bases)}] {base:>26s}  WER={row['wer']:.3f}  ({row['compute_s']:.1f}s)")

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    valid = [r for r in rows if r.get("wer") is not None]
    if valid:
        mean_wer = sum(r["wer"] for r in valid) / len(valid)
        print(f"\nPriMock57 mean WER ({len(valid)} files): {mean_wer:.4f}")
    print(f"Wrote {output_csv}")


def run_fareez(model, output_csv: Path, limit: int | None):
    audio_dir = WER_ROOT / "data" / "fareez_osce" / "wav_16khz"
    transcript_dir = WER_ROOT / "data" / "fareez_osce" / "Data" / "Clean Transcripts"
    files = sorted(p.stem for p in audio_dir.glob("*.wav"))
    if limit:
        files = files[:limit]

    rows = []
    for i, base in enumerate(files, 1):
        audio = audio_dir / f"{base}.wav"
        tx = transcript_dir / f"{base}.txt"
        if not audio.exists() or not tx.exists():
            print(f"  [{i:3d}] {base}: missing files")
            continue
        # Strip D:/P: prefixes for the reference transcript
        ref = []
        for line in tx.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            # Drop leading "D:" / "P:" prefix if present
            if len(line) >= 2 and line[1] == ":" and line[0] in "DPdp":
                line = line[2:].strip()
            ref.append(line)
        ref = " ".join(ref)
        t0 = time.time()
        try:
            hyp = transcribe_file(model, str(audio))
            metrics = calc_wer(ref, hyp)
            row = {"name": base, "ground_truth": ref, "transcript": hyp,
                   **metrics, "compute_s": round(time.time() - t0, 2)}
        except Exception as e:
            print(f"  [{i:3d}] {base}: ERROR {str(e)[:100]}")
            row = {"name": base, "ground_truth": ref, "transcript": "",
                   "wer": None, "mer": None, "wil": None,
                   "insertions": None, "deletions": None,
                   "substitutions": None, "hits": None,
                   "compute_s": round(time.time() - t0, 2),
                   "error": str(e)[:200]}
        rows.append(row)
        if row.get("wer") is not None:
            print(f"  [{i:3d}/{len(files)}] {base:>10s}  WER={row['wer']:.3f}  ({row['compute_s']:.1f}s)")

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    valid = [r for r in rows if r.get("wer") is not None]
    if valid:
        mean_wer = sum(r["wer"] for r in valid) / len(valid)
        print(f"\nFareez mean WER ({len(valid)}/{len(files)} files): {mean_wer:.4f}")
    print(f"Wrote {output_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["primock57", "fareez", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", default=str(WER_ROOT / "results"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model()

    if args.dataset in ("primock57", "both"):
        run_primock57(model, out_dir / f"primock57-{SHORT_NAME}.csv", args.limit)
    if args.dataset in ("fareez", "both"):
        run_fareez(model, out_dir / f"fareez-{SHORT_NAME}.csv", args.limit)


if __name__ == "__main__":
    main()
