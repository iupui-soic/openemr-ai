#!/usr/bin/env python
"""
Consolidate text + entity metrics over the Fareez ablation cells.

Walks `results/fareez/ablations/<cell>/<model>/*.txt`, computes the same metric
suite as `compute_fareez_metrics.py` (BLEU, ROUGE-L, SBERT coherence, BERTScore F1),
optionally MedCAT/scispaCy via the Modal entity-evaluator service, and emits one
long-format CSV that downstream notebooks (and Table 2c in the paper) read.

Usage:
    python compute_ablation_metrics.py
    python compute_ablation_metrics.py --with-entities
    python compute_ablation_metrics.py --cells norag,k1,k3,k5 --models gpt-oss-120b,gpt-oss-20b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAG_ROOT = Path(__file__).resolve().parent.parent
ABLATIONS_DIR = RAG_ROOT / "results" / "fareez" / "ablations"
SELECTION = RAG_ROOT / "data" / "fareez_selected_40.json"
TRANSCRIPT_DIR = (RAG_ROOT.parent / "openemr_whisper_wer" / "data" /
                  "fareez_osce" / "Data" / "Clean Transcripts").resolve()
OUTPUT_TEXT = RAG_ROOT / "results" / "fareez" / "ablations" / "fareez_ablation_metrics.csv"
OUTPUT_ENTITY = RAG_ROOT / "results" / "fareez" / "ablations" / "fareez_ablation_entity_metrics.csv"


def load_transcript(name: str) -> str:
    p = TRANSCRIPT_DIR / f"{name}.txt"
    try:
        return p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1")


def load_summary(path: Path) -> str | None:
    """Strip the metadata header that ablation drivers prepend before '======...'."""
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("=" * 10):
            start = i + 1
            break
    text = "".join(lines[start:]).strip()
    if not text or text.startswith("Error generating summary") or text.startswith("[No summary"):
        return None
    return text


def discover(cells_filter: list[str] | None, models_filter: list[str] | None) -> list[dict]:
    """
    Walk ablations/<cell>/<model>/*.txt and return per-summary dicts:
        {cell, model, conversation, summary_path, summary_text}
    Skips cells/models not in filter (if provided).
    """
    if not ABLATIONS_DIR.exists():
        print(f"No ablations dir at {ABLATIONS_DIR} — nothing to do.")
        return []

    items = []
    for cell_dir in sorted(ABLATIONS_DIR.iterdir()):
        if not cell_dir.is_dir():
            continue
        cell = cell_dir.name
        if cells_filter and cell not in cells_filter:
            continue
        for model_dir in sorted(cell_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if models_filter and model not in models_filter:
                continue
            for txt in sorted(model_dir.glob("*.txt")):
                summary = load_summary(txt)
                if summary is None:
                    continue
                items.append({
                    "cell": cell,
                    "model": model,
                    "conversation": txt.stem,
                    "summary": summary,
                })
    return items


def compute_text_metrics(items: list[dict]) -> list[dict]:
    """BLEU, ROUGE-L, SBERT coherence, BERTScore F1, all against transcript reference."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    from bert_score import score as bert_score_fn

    print("Loading evaluation models (SBERT, ROUGE, BLEU, BERTScore)...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1

    # Hydrate references once per conversation
    refs_cache: dict[str, str] = {}
    for it in items:
        refs_cache.setdefault(it["conversation"], load_transcript(it["conversation"]))
    refs = [refs_cache[it["conversation"]] for it in items]
    summaries = [it["summary"] for it in items]

    print(f"  computing BLEU + ROUGE-L on {len(items)} pairs...")
    for i, (s, r) in enumerate(zip(summaries, refs)):
        items[i]["bleu"] = sentence_bleu([r.split()], s.split(), smoothing_function=smoother)
        items[i]["rouge_l"] = rouge.score(r, s)["rougeL"].fmeasure
        items[i]["summary_chars"] = len(s)
        items[i]["summary_tokens"] = len(s.split())

    print("  encoding SBERT (batched)...")
    s_emb = sbert.encode(summaries, show_progress_bar=False, batch_size=32)
    r_emb = sbert.encode(refs, show_progress_bar=False, batch_size=32)
    for i, (se, re) in enumerate(zip(s_emb, r_emb)):
        items[i]["sbert_coherence"] = float(np.dot(se, re) /
                                            (np.linalg.norm(se) * np.linalg.norm(re)))

    print("  computing BERTScore F1...")
    P, R, F1 = bert_score_fn(summaries, refs, lang="en", verbose=False, batch_size=8, device="cuda" if _has_cuda() else "cpu")
    for i, f in enumerate(F1.tolist()):
        items[i]["bert_f1"] = f
    return items


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def compute_entity_metrics(items: list[dict]) -> list[dict]:
    """MedCAT + scispaCy via the existing Modal entity-evaluator-service."""
    import modal
    print("Connecting to Modal entity-evaluator-service...")
    EntityEvaluator = modal.Cls.from_name("entity-evaluator-service", "EntityEvaluator")
    evaluator = EntityEvaluator()

    refs_cache: dict[str, str] = {}
    for it in items:
        refs_cache.setdefault(it["conversation"], load_transcript(it["conversation"]))

    for i, it in enumerate(items):
        try:
            r = evaluator.evaluate.remote(
                generated=it["summary"],
                reference=refs_cache[it["conversation"]],
            )
            it["scispacy_entity_recall"] = r.get("scispacy_entity_recall", 0.0)
            it["medcat_entity_recall"] = r.get("medcat_entity_recall", 0.0)
        except Exception as e:
            print(f"  entity error item {i} ({it['cell']}/{it['model']}/{it['conversation']}): {e}")
            it["scispacy_entity_recall"] = None
            it["medcat_entity_recall"] = None
        if (i + 1) % 20 == 0:
            print(f"  entity progress: {i+1}/{len(items)}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None,
                    help="Comma-separated cell filter (e.g., norag,k1,k3)")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model filter")
    ap.add_argument("--with-entities", action="store_true",
                    help="Also compute MedCAT/scispaCy entity recall via Modal")
    ap.add_argument("--text-output", default=str(OUTPUT_TEXT))
    ap.add_argument("--entity-output", default=str(OUTPUT_ENTITY))
    args = ap.parse_args()

    cells = args.cells.split(",") if args.cells else None
    models = args.models.split(",") if args.models else None

    items = discover(cells, models)
    if not items:
        print("No summaries found. Exiting.")
        return

    cells_present = sorted({i["cell"] for i in items})
    models_present = sorted({i["model"] for i in items})
    print(f"Found {len(items)} summaries across {len(cells_present)} cells x {len(models_present)} models")
    print(f"  cells:  {cells_present}")
    print(f"  models: {models_present}")

    t0 = time.time()
    items = compute_text_metrics(items)
    print(f"Text metrics in {time.time() - t0:.1f}s")

    if args.with_entities:
        t0 = time.time()
        items = compute_entity_metrics(items)
        print(f"Entity metrics in {time.time() - t0:.1f}s")

    # Persist text-metric CSV
    keep = ["cell", "model", "conversation",
            "bleu", "rouge_l", "sbert_coherence", "bert_f1",
            "summary_chars", "summary_tokens"]
    df = pd.DataFrame([{k: it.get(k) for k in keep} for it in items])
    Path(args.text_output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.text_output, index=False)
    print(f"Wrote {args.text_output}  ({len(df)} rows)")

    if args.with_entities:
        ek = ["cell", "model", "conversation",
              "scispacy_entity_recall", "medcat_entity_recall"]
        ed = pd.DataFrame([{k: it.get(k) for k in ek} for it in items])
        ed.to_csv(args.entity_output, index=False)
        print(f"Wrote {args.entity_output}  ({len(ed)} rows)")


if __name__ == "__main__":
    main()
