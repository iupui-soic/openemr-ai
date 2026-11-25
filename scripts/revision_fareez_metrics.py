"""
Per-note automated metrics for the 159 production Fareez notes,
each scored against its source transcript.

Settings are those of rag_models/evaluation/compute_fareez_metrics.py: NLTK sentence BLEU with
smoothing method 1 on whitespace tokens, rouge-score ROUGE-L F with stemming, bert-score lang="en"
defaults, all-MiniLM-L6-v2 cosine. The failed generation (MedGemma-4B, MSK0037) is skipped.

Transcripts: the Fareez "Clean Transcripts" folder if present, otherwise the reference text stored
in openemr_whisper_wer/results/fareez-groq.csv (the same files, read the same way, then stripped).

Output: rag_models/results/revision/fareez_automated_metrics.csv
  conversation, model, bleu, rouge_l, sbert_similarity, bert_f1, summary_words, summary_tokens
  summary_words  = whitespace-separated words of the note as shown in the rating packet
  summary_tokens = output tokens logged by the provider at generation time

Run: python scripts/revision_fareez_metrics.py --repo .
"""
import argparse, json, os, re
import numpy as np, pandas as pd

MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3-32b", "medgemma-4b"]
FAILED = {("MSK0037", "medgemma-4b")}


def load_summary(path):
    lines = open(path, encoding="utf-8").readlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("=" * 10):
            start = i + 1
            break
    return "".join(lines[start:]).strip()


def load_transcripts(repo, names):
    tdir = f"{repo}/openemr_whisper_wer/data/fareez_osce/Data/Clean Transcripts"
    out = {}
    if os.path.isdir(tdir):
        for n in names:
            try:
                out[n] = open(f"{tdir}/{n}.txt", encoding="utf-8-sig").read()
            except UnicodeDecodeError:
                out[n] = open(f"{tdir}/{n}.txt", encoding="latin-1").read()
        return out, "Clean Transcripts folder"
    d = pd.read_csv(f"{repo}/openemr_whisper_wer/results/fareez-groq.csv").set_index("name")
    return {n: str(d.loc[n, "ground_truth"]) for n in names}, "fareez-groq.csv ground_truth column"


def packet_words(repo):
    """Words of each note exactly as printed in the rating packet (same parse as revision_rating_analyses.py)."""
    R = f"{repo}/rag_models/rating_packets"
    txt = open(f"{R}/packet_rater_1.txt", encoding="utf-8").read()
    parts = re.split(r"\n#{20,}\n# SUMMARY (S\d+)", txt)
    words = {}
    for i in range(1, len(parts), 2):
        body = parts[i + 1].split("SECTION C: AI-GENERATED SUMMARY")[1].split("\n", 2)[2]
        words[parts[i]] = len(re.split(r"\n-{40,}\n", body)[0].split())
    key = pd.read_csv(f"{R}/ANSWER_KEY_DO_NOT_SHARE.csv")
    key = key[key.rater == "rater_1"]
    return {(r.conversation, r.model): words[r.summary_id] for r in key.itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    repo = a.repo.rstrip("/")
    R = f"{repo}/rag_models"
    convs = [s["file_name"] for s in json.load(open(f"{R}/data/fareez_selected_40.json"))]
    transcripts, source = load_transcripts(repo, convs)
    print(f"Transcript source: {source}")
    pw = packet_words(repo)
    logged = pd.concat([pd.read_csv(f"{R}/results/fareez/evaluation_results_{m}.csv") for m in MODELS])
    logged = {(r.patient_name, r.model): r.output_tokens for r in logged.itertuples()}

    items = []
    for m in MODELS:
        for c in convs:
            if (c, m) in FAILED:
                continue
            s = load_summary(f"{R}/results/fareez/fareez_summaries/{m}/{c}.txt")
            assert s and not s.startswith("Error generating summary"), (c, m)
            items.append(dict(conversation=c, model=m, summary=s, reference=transcripts[c]))
    print(f"Notes: {len(items)} (expected 159)")
    assert len(items) == 159

    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    from bert_score import score as bert_score_fn
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    summ = [i["summary"] for i in items]
    refs = [i["reference"] for i in items]
    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bleu = [sentence_bleu([r.split()], s.split(), smoothing_function=smooth) for s, r in zip(summ, refs)]
    rl = [rouge.score(r, s)["rougeL"].fmeasure for s, r in zip(summ, refs)]
    sb = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    se, re_ = sb.encode(summ, batch_size=32), sb.encode(refs, batch_size=32)
    sbert = [float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(se, re_)]
    _, _, F1 = bert_score_fn(summ, refs, lang="en", batch_size=16, device=device)

    rows = []
    for k, it in enumerate(items):
        key = (it["conversation"], it["model"])
        rows.append(dict(conversation=key[0], model=key[1], bleu=bleu[k], rouge_l=rl[k],
                         sbert_similarity=sbert[k], bert_f1=float(F1[k]),
                         summary_words=pw[key], summary_tokens=logged.get(key, np.nan)))
        assert pw[key] == len(it["summary"].split()), f"packet text differs from stored note for {key}"
    df = pd.DataFrame(rows)
    out = f"{R}/results/revision/fareez_automated_metrics.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    print(df.groupby("model")[["bleu", "rouge_l", "sbert_similarity", "bert_f1", "summary_words", "summary_tokens"]]
            .mean().reindex(MODELS).round(3).to_string())


if __name__ == "__main__":
    main()
