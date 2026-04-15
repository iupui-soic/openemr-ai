#!/usr/bin/env python
"""
Parakeet TDT 0.6B v2 with chunked transcription — Fareez OSCE.

The naive `model.transcribe([file])` API allocates intermediate RNN-T tensors
proportional to audio duration; long Fareez files (>10 min) trigger a CUDA
illegal-memory-access core dump. This wrapper splits each audio file into
60-second chunks (with 1-second overlap), transcribes each, concatenates.

Usage: python parakeet_06b_v2_wer_chunked.py
Output: results/fareez-parakeet-tdt-06b-v2.csv
"""
from __future__ import annotations
import os, sys, time, warnings, tempfile
from pathlib import Path
import jiwer, pandas as pd, soundfile as sf, numpy as np

WER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WER_ROOT))
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
SHORT = "parakeet-tdt-06b-v2"
CHUNK_SEC = 60.0
OVERLAP_SEC = 1.0


def load_model():
    import nemo.collections.asr as nemo_asr
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    m = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_ID, map_location="cuda")
    m.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    return m


def chunk_and_transcribe(model, audio_path: str) -> str:
    """Return concatenated transcript over ~60s chunks of the input audio."""
    from scipy import signal
    import torch
    data, sr = sf.read(audio_path, always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    if sr != 16000:
        n = int(len(data) * 16000 / sr)
        data = signal.resample(data, n).astype("float32")
        sr = 16000
    data = data.astype(np.float32)

    chunk_len = int(CHUNK_SEC * sr)
    overlap = int(OVERLAP_SEC * sr)
    transcripts = []
    start = 0
    while start < len(data):
        end = min(start + chunk_len, len(data))
        chunk = data[start:end]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, chunk, sr)
            tmp = f.name
        try:
            out = model.transcribe([tmp])
            text = (out[0].text if out and hasattr(out[0], "text") else str(out[0])).strip()
            if text:
                transcripts.append(text)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        # release GPU memory between chunks
        torch.cuda.empty_cache()
        if end >= len(data):
            break
        start = end - overlap
    return " ".join(transcripts)


def calc_wer(ref: str, hyp: str) -> dict:
    t = jiwer.Compose([jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                       jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    o = jiwer.process_words(t(ref), t(hyp))
    return {"wer": o.wer, "mer": o.mer, "wil": o.wil,
            "insertions": o.insertions, "deletions": o.deletions,
            "substitutions": o.substitutions, "hits": o.hits}


def main():
    audio_dir = WER_ROOT / "data" / "fareez_osce" / "wav_16khz"
    txt_dir = WER_ROOT / "data" / "fareez_osce" / "Data" / "Clean Transcripts"
    out_csv = WER_ROOT / "results" / f"fareez-{SHORT}.csv"
    files = sorted(p.stem for p in audio_dir.glob("*.wav"))
    print(f"Fareez (chunked): {len(files)} files")

    model = load_model()
    rows = []
    t_total = time.time()
    for i, base in enumerate(files, 1):
        audio = audio_dir / f"{base}.wav"
        tx = txt_dir / f"{base}.txt"
        if not audio.exists() or not tx.exists():
            continue
        ref_lines = []
        for line in tx.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = line.strip()
            if not line: continue
            if len(line) >= 2 and line[1] == ":" and line[0] in "DPdp":
                line = line[2:].strip()
            ref_lines.append(line)
        ref = " ".join(ref_lines)
        t0 = time.time()
        try:
            hyp = chunk_and_transcribe(model, str(audio))
            metrics = calc_wer(ref, hyp)
            row = {"name": base, "ground_truth": ref, "transcript": hyp,
                   **metrics, "compute_s": round(time.time() - t0, 2)}
            print(f"  [{i:3d}/{len(files)}] {base:>10s}  WER={row['wer']:.3f}  ({row['compute_s']:.1f}s)")
        except Exception as e:
            print(f"  [{i:3d}] {base}: ERROR {str(e)[:120]}")
            row = {"name": base, "ground_truth": ref, "transcript": "",
                   "wer": None, "mer": None, "wil": None,
                   "insertions": None, "deletions": None,
                   "substitutions": None, "hits": None,
                   "compute_s": round(time.time() - t0, 2),
                   "error": str(e)[:200]}
        rows.append(row)
        # Save partial every 25 files in case of crash
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  ... partial saved ({i} rows)")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    valid = [r for r in rows if r.get("wer") is not None]
    if valid:
        m = sum(r["wer"] for r in valid) / len(valid)
        print(f"\nMean WER: {m:.4f}  (n={len(valid)}/{len(files)})")
    print(f"Wrote {out_csv}  ({(time.time() - t_total)/60:.1f} min)")


if __name__ == "__main__":
    main()
