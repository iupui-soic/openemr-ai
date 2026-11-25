"""
MedCAT and scispaCy entity recall, precision, and F1
for the 159 production Fareez notes, each against its source transcript.

Runs rag_models/evaluation/entity_evaluator_service.py as an ephemeral Modal app (self-hosted
models; nothing is sent to an external model API) and merges the result into
rag_models/results/revision/fareez_automated_metrics.csv (written by revision_fareez_metrics.py).

Run: python scripts/revision_entity_metrics.py --repo .
"""
import argparse, json, os, sys
import pandas as pd

MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3-32b", "medgemma-4b"]
TOOLS = ["medcat", "scispacy"]
COLS = [f"{t}_entity_{k}" for t in TOOLS for k in ("recall", "precision", "f1")] + \
       [f"{t}_{k}" for t in TOOLS for k in ("n_note_cuis", "n_ref_cuis", "n_matched_cuis")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); a = ap.parse_args()
    repo = os.path.abspath(a.repo); R = f"{repo}/rag_models"
    sys.path.insert(0, f"{repo}/scripts"); sys.path.insert(0, f"{R}/evaluation")
    from revision_fareez_metrics import load_summary, load_transcripts
    import modal
    from entity_evaluator_service import app, EntityEvaluator

    out = f"{R}/results/revision/fareez_automated_metrics.csv"
    met = pd.read_csv(out)
    assert len(met) == 159, "run revision_fareez_metrics.py first"
    convs = [s["file_name"] for s in json.load(open(f"{R}/data/fareez_selected_40.json"))]
    transcripts, source = load_transcripts(repo, convs)
    print(f"Transcript source: {source}")
    keys = list(zip(met.conversation, met.model))
    notes = [load_summary(f"{R}/results/fareez/fareez_summaries/{m}/{c}.txt") for c, m in keys]
    refs = [transcripts[c] for c, _ in keys]

    with modal.enable_output(), app.run():
        results = list(EntityEvaluator().evaluate.map(notes, refs, return_exceptions=True))
    failed = [(k, repr(r)[:200]) for k, r in zip(keys, results) if isinstance(r, Exception)]
    for k, e in failed:
        print(f"FAILED {k}: {e}")
    rows = [{c: (float("nan") if isinstance(r, Exception) else r[c]) for c in COLS} for r in results]
    ent = pd.DataFrame(rows)
    met = pd.concat([met.drop(columns=[c for c in COLS if c in met.columns]), ent], axis=1)
    met.to_csv(out, index=False)
    print(f"Saved {out}; notes failed: {len(failed)}")
    show = [f"{t}_entity_{k}" for t in TOOLS for k in ("recall", "precision", "f1")]
    print(met.groupby("model")[show].agg(["mean", "std"]).reindex(MODELS).round(3).T.to_string())
    print(met.groupby("model")[[c for c in COLS if "_n_" in c]].mean().reindex(MODELS).round(1).T.to_string())


if __name__ == "__main__":
    main()
