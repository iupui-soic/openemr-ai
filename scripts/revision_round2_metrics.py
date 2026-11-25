"""
Automated metrics for the models added in the second rating round (same definitions as
revision_fareez_metrics.py and revision_entity_metrics.py; transcript as reference).
Writes rag_models/results/revision/fareez_round2_metrics.csv and leaves the round-one file untouched.

Run from the repo root: python scripts/revision_round2_metrics.py --repo . --models qwen3.8-27b
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

TOOLS = ["medcat", "scispacy"]
COLS = [f"{t}_entity_{k}" for t in TOOLS for k in ("recall", "precision", "f1")] + \
       [f"{t}_{k}" for t in TOOLS for k in ("n_note_cuis", "n_ref_cuis", "n_matched_cuis")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="."); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--skip-entities", action="store_true")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo); R = f"{repo}/rag_models"
    sys.path.insert(0, f"{repo}/scripts"); sys.path.insert(0, f"{R}/evaluation")
    from revision_fareez_metrics import load_summary, load_transcripts
    convs = [s["file_name"] for s in json.load(open(f"{R}/data/fareez_selected_40.json"))]
    transcripts, source = load_transcripts(repo, convs)
    print(f"Transcript source: {source}")
    items = []
    for m in a.models:
        for c in convs:
            p = f"{R}/results/fareez/fareez_summaries/{m}/{c}.txt"
            if os.path.exists(p) and load_summary(p):
                items.append(dict(conversation=c, model=m, summary=load_summary(p), reference=transcripts[c]))
    print(f"Notes: {len(items)}")
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    from bert_score import score as bert_score_fn
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summ = [i["summary"] for i in items]; refs = [i["reference"] for i in items]
    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sb = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    se, re_ = sb.encode(summ, batch_size=32), sb.encode(refs, batch_size=32)
    _, _, F1 = bert_score_fn(summ, refs, lang="en", batch_size=16, device=device)
    df = pd.DataFrame([dict(
        conversation=it["conversation"], model=it["model"],
        bleu=sentence_bleu([it["reference"].split()], it["summary"].split(), smoothing_function=smooth),
        rouge_l=rouge.score(it["reference"], it["summary"])["rougeL"].fmeasure,
        sbert_similarity=float(np.dot(se[k], re_[k]) / (np.linalg.norm(se[k]) * np.linalg.norm(re_[k]))),
        bert_f1=float(F1[k]), summary_words=len(it["summary"].split())) for k, it in enumerate(items)])
    out = f"{R}/results/revision/fareez_round2_metrics.csv"
    if not a.skip_entities:
        import modal
        from entity_evaluator_service import app, EntityEvaluator
        with modal.enable_output(), app.run():
            res = list(EntityEvaluator().evaluate.map(summ, refs, return_exceptions=True))
        bad = sum(isinstance(r, Exception) for r in res)
        print(f"entity extraction failed for {bad} notes")
        ent = pd.DataFrame([{c: (float('nan') if isinstance(r, Exception) else r[c]) for c in COLS} for r in res])
        df = pd.concat([df, ent], axis=1)
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    show = [c for c in ["bleu", "rouge_l", "bert_f1", "sbert_similarity", "medcat_entity_recall", "medcat_entity_precision",
                        "medcat_entity_f1", "scispacy_entity_recall", "scispacy_entity_precision", "scispacy_entity_f1",
                        "medcat_n_note_cuis", "summary_words"] if c in df.columns]
    print(df.groupby("model")[show].mean().round(3).T.to_string())


if __name__ == "__main__":
    main()
