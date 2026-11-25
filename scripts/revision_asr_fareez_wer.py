"""
Fareez WER with speaker labels removed from the reference for every ASR model.

The first analysis removed the "D:" and "P:" speaker labels from the reference only for
Voxtral Mini 3B (voxtral_mini_wer.py); the other models' references kept about 100 labels per
file, each counted as a deletion. Two references (RES0002, RES0054) were read from UTF-16 files
with the wrong encoding and are stored as 3 characters, so every model scores 100% on them.

This script recomputes per-file WER from the stored reference and hypothesis text with the labels
removed for every model, using the same normalization (remove punctuation, lower-case,
collapse spaces), and excludes the two corrupt files. Re-read those two transcripts as UTF-16 and
rerun the ASR scripts to include them.

Run: python revision_asr_fareez_wer.py --repo /path/to/openemr-ai     (numpy, pandas; about 3 min)
"""
import argparse, re, sys, unicodedata
import numpy as np, pandas as pd

NAMES = {"voxtral-mini": "Voxtral Mini 3B", "parakeet": "Parakeet TDT 1.1B", "canary": "Canary 1B-v2",
         "groq": "Groq Whisper v3 Turbo", "whisper-turbo": "Whisper Large v3 Turbo", "whisper-v3": "Whisper Large v3",
         "parakeet-tdt-06b-v2": "Parakeet TDT 0.6B v2", "medasr": "MedASR"}
CORRUPT = {"RES0002", "RES0054"}
PUNCT = {i: None for i in range(sys.maxunicode + 1) if unicodedata.category(chr(i)).startswith("P")}
TAG = re.compile(r"(?:(?<=\s)|^)[DP]:(?=\s|$)")

def words(s):
    return str(s).translate(PUNCT).lower().split()

def word_edits(ref, hyp):
    """Levenshtein distance between word lists (row-wise vectorized dynamic programming)."""
    if not ref: return len(hyp)
    if not hyp: return len(ref)
    h = np.array(hyp, dtype=object); idx = np.arange(len(hyp) + 1); prev = idx.copy()
    for i, w in enumerate(ref, 1):
        a = np.empty(len(hyp) + 1, dtype=np.int64); a[0] = i
        a[1:] = np.minimum(prev[:-1] + (h != w), prev[1:] + 1)
        prev = np.minimum.accumulate(a - idx) + idx
    return int(prev[-1])

def completed(d):
    ok = d.wer.notna() & d.transcript.notna() & (d.transcript.astype(str).str.strip() != "")
    if "error" in d: ok &= d.error.isna()
    return d[ok]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); a = ap.parse_args()
    res = a.repo.rstrip("/") + "/openemr_whisper_wer/results"
    per = {}
    for k in NAMES:
        d = completed(pd.read_csv(f"{res}/fareez-{k}.csv")).set_index("name")
        d = d[~d.index.isin(CORRUPT)]
        per[k] = pd.Series({n: word_edits(words(TAG.sub(" ", str(r.ground_truth))), words(r.transcript)) /
                               max(1, len(words(TAG.sub(" ", str(r.ground_truth))))) for n, r in d.iterrows()})
        print(f"done {NAMES[k]}", flush=True)
    P = pd.DataFrame(per)
    all8 = P.dropna().index; seven = P.drop(columns="whisper-v3").dropna().index
    rows = [dict(model=NAMES[k], files=int(P[k].notna().sum()), wer=round(100 * P[k].mean(), 2),
                 wer_common_7=(round(100 * P.loc[seven, k].mean(), 2) if k != "whisper-v3" else None),
                 wer_common_8=round(100 * P.loc[all8, k].mean(), 2)) for k in NAMES]
    print(f"\nFiles common to all 8 models: {len(all8)}; to all except Whisper Large v3: {len(seven)}")
    print(pd.DataFrame(rows).sort_values("wer").to_string(index=False))
    P.to_csv("fareez_wer_labels_removed.csv")

if __name__ == "__main__":
    main()
