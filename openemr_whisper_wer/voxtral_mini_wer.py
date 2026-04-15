#!/usr/bin/env python
"""
Mistral Voxtral Mini 3B — local WER evaluation on PriMock57 + Fareez OSCE.

Uses transformers' Voxtral integration. Runs locally on RTX 6000.

Usage:
    python voxtral_mini_wer.py --dataset both
    python voxtral_mini_wer.py --dataset primock57 --limit 5

Outputs:
    results/primock57-voxtral-mini.csv
    results/fareez-voxtral-mini.csv
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

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
SHORT_NAME = "voxtral-mini"


def load_model():
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    import torch

    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    print(f"  loaded in {time.time() - t0:.1f}s on {device}")
    return model, processor, device


def transcribe_file(model, processor, device: str, audio_path: str) -> str:
    """Voxtral transcription via apply_transcription_request helper."""
    inputs = processor.apply_transcription_request(
        language="en",
        audio=audio_path,
        model_id=MODEL_ID,
    )
    inputs = inputs.to(device, dtype=model.dtype)
    out = model.generate(**inputs, max_new_tokens=2048)
    decoded = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return (decoded[0] if decoded else "").strip()


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


def run_primock57(model, processor, device, output_csv: Path, limit: int | None):
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
            hyp = transcribe_file(model, processor, device, str(audio))
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


def run_fareez(model, processor, device, output_csv: Path, limit: int | None):
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
        ref_lines = []
        for line in tx.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            if len(line) >= 2 and line[1] == ":" and line[0] in "DPdp":
                line = line[2:].strip()
            ref_lines.append(line)
        ref = " ".join(ref_lines)
        t0 = time.time()
        try:
            hyp = transcribe_file(model, processor, device, str(audio))
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
    model, processor, device = load_model()

    if args.dataset in ("primock57", "both"):
        run_primock57(model, processor, device,
                      out_dir / f"primock57-{SHORT_NAME}.csv", args.limit)
    if args.dataset in ("fareez", "both"):
        run_fareez(model, processor, device,
                   out_dir / f"fareez-{SHORT_NAME}.csv", args.limit)


if __name__ == "__main__":
    main()
