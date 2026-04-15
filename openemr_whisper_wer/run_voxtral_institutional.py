#!/usr/bin/env python
"""Run Voxtral Mini on the 6 institutional WAVs. Reference = *_edge.txt transcripts."""
import sys, time
from pathlib import Path
import jiwer, pandas as pd

WER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WER_ROOT))
from voxtral_mini_wer import load_model, transcribe_file, calc_wer  # noqa

INST_WAV_DIR = WER_ROOT / "data" / "institutional" / "wav"
INST_DATA_DIR = WER_ROOT / "data"
OUT = WER_ROOT / "results" / "institutional-voxtral-mini.csv"

CASES = ["anemia", "arthritis", "copd", "dka", "gallstones", "gerd"]


def find_transcript(case: str) -> Path | None:
    for c in [INST_DATA_DIR / f"{case}_edge.txt",
              INST_DATA_DIR / f"{case}.txt",
              INST_DATA_DIR / f"{case}1.txt"]:
        if c.exists() and c.stat().st_size > 50:
            return c
    return None


def main():
    model, proc, device = load_model()
    rows = []
    for case in CASES:
        audio = INST_WAV_DIR / f"{case}.wav"
        tx = find_transcript(case)
        if not audio.exists() or tx is None:
            print(f"  [{case}] missing audio or transcript")
            continue
        ref = tx.read_text(encoding="utf-8-sig", errors="replace")
        # Strip D:/P: prefixes if present
        ref_lines = []
        for line in ref.splitlines():
            line = line.strip()
            if not line: continue
            if len(line) >= 2 and line[1] == ":" and line[0] in "DPdp":
                line = line[2:].strip()
            ref_lines.append(line)
        ref = " ".join(ref_lines)
        t0 = time.time()
        try:
            hyp = transcribe_file(model, proc, device, str(audio))
            metrics = calc_wer(ref, hyp)
            row = {"case": case, "ground_truth": ref, "transcript": hyp,
                   **metrics, "transcript_used": tx.name,
                   "compute_s": round(time.time() - t0, 2)}
            print(f"  {case:>11s}  WER={row['wer']:.3f}  ({row['compute_s']:.1f}s)")
        except Exception as e:
            print(f"  {case}: ERROR {str(e)[:120]}")
            row = {"case": case, "ground_truth": ref, "transcript": "",
                   "wer": None, "compute_s": round(time.time() - t0, 2),
                   "transcript_used": tx.name, "error": str(e)[:200]}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    valid = df.dropna(subset=["wer"])
    if len(valid):
        print(f"\nMean WER: {valid['wer'].mean()*100:.2f}% (n={len(valid)}/{len(df)})")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
