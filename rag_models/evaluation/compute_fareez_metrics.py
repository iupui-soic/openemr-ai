"""
Compute automated evaluation metrics on 160 Fareez RAG summaries.

Since no gold-standard reference summaries exist for the Fareez transcripts,
we use the transcript as the reference for content coverage metrics:
- ROUGE-L: measures how much of the transcript's content appears in the summary
- BERTScore: contextual semantic overlap between summary and transcript
- SBERT Coherence: overall semantic alignment between summary and transcript

We also compute summary-only metrics:
- Summary length (tokens, characters)
- BLEU (against transcript — measures n-gram overlap)

Entity-based metrics (scispaCy, MedCAT) are run via Modal evaluator services
if available, since they require heavy NLP models.

Usage:
    python compute_fareez_metrics.py                    # Text metrics only (local)
    python compute_fareez_metrics.py --with-entities    # Also run entity metrics via Modal
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTION_FILE = os.path.join(RAG_ROOT, "data", "fareez_selected_40.json")
SUMMARIES_DIR = os.path.join(RAG_ROOT, "results", "fareez", "fareez_summaries")
TRANSCRIPT_DIR = os.path.normpath(os.path.join(
    RAG_ROOT, "..", "openemr_whisper_wer", "data",
    "fareez_osce", "Data", "Clean Transcripts"
))
OUTPUT_DIR = os.path.join(RAG_ROOT, "results", "fareez")

MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3-32b", "medgemma-4b"]


def load_transcript(file_name):
    path = os.path.join(TRANSCRIPT_DIR, f"{file_name}.txt")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def load_summary(model, file_name):
    path = os.path.join(SUMMARIES_DIR, model, f"{file_name}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Skip metadata header (before === separator)
    content_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("=" * 10):
            content_start = i + 1
            break
    text = "".join(lines[content_start:]).strip()
    # Skip error summaries
    if text.startswith("Error generating summary"):
        return None
    return text


def compute_text_metrics(summaries_with_refs):
    """Compute BLEU, ROUGE-L, SBERT, BERTScore for all (summary, reference) pairs."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    from bert_score import score as bert_score_fn

    print("Loading evaluation models...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1

    results = []
    summaries = [s["summary"] for s in summaries_with_refs]
    references = [s["reference"] for s in summaries_with_refs]

    # BLEU + ROUGE-L (per-item)
    print("Computing BLEU and ROUGE-L...")
    bleu_scores = []
    rouge_scores = []
    for summ, ref in zip(summaries, references):
        # BLEU
        ref_tokens = ref.split()
        summ_tokens = summ.split()
        bleu = sentence_bleu([ref_tokens], summ_tokens, smoothing_function=smoother)
        bleu_scores.append(bleu)

        # ROUGE-L
        rl = rouge.score(ref, summ)
        rouge_scores.append(rl["rougeL"].fmeasure)

    # SBERT Coherence (batch)
    print("Computing SBERT coherence...")
    summ_embeddings = sbert_model.encode(summaries, show_progress_bar=True, batch_size=32)
    ref_embeddings = sbert_model.encode(references, show_progress_bar=True, batch_size=32)
    sbert_scores = []
    for se, re in zip(summ_embeddings, ref_embeddings):
        cos_sim = np.dot(se, re) / (np.linalg.norm(se) * np.linalg.norm(re))
        sbert_scores.append(float(cos_sim))

    # BERTScore (batch)
    print("Computing BERTScore (this may take a few minutes)...")
    P, R, F1 = bert_score_fn(summaries, references, lang="en", verbose=True,
                              batch_size=16, device="cpu")
    bertscore_f1 = F1.tolist()

    # Combine
    for i, item in enumerate(summaries_with_refs):
        item["bleu"] = bleu_scores[i]
        item["rouge_l"] = rouge_scores[i]
        item["sbert_coherence"] = sbert_scores[i]
        item["bert_f1"] = bertscore_f1[i]
        item["summary_chars"] = len(summaries[i])
        item["summary_tokens"] = len(summaries[i].split())

    return summaries_with_refs


def compute_entity_metrics_modal(summaries_with_refs):
    """Compute entity metrics via Modal evaluator services."""
    import modal

    print("\nConnecting to Modal entity evaluator...")
    try:
        EntityEvaluator = modal.Cls.from_name("entity-evaluator-service", "EntityEvaluator")
        evaluator = EntityEvaluator()
        print("Connected to entity evaluator service")
    except Exception as e:
        print(f"Could not connect to entity evaluator: {e}")
        print("Deploy first: modal deploy entity_evaluator_service.py")
        return summaries_with_refs

    for i, item in enumerate(summaries_with_refs):
        try:
            result = evaluator.evaluate.remote(
                generated=item["summary"],
                reference=item["reference"],
            )
            item["scispacy_entity_recall"] = result.get("scispacy_entity_recall", 0.0)
            item["medcat_entity_recall"] = result.get("medcat_entity_recall", 0.0)
            if (i + 1) % 10 == 0:
                print(f"  Entity metrics: {i+1}/{len(summaries_with_refs)}")
        except Exception as e:
            print(f"  Error on item {i}: {e}")
            item["scispacy_entity_recall"] = 0.0
            item["medcat_entity_recall"] = 0.0

    return summaries_with_refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-entities", action="store_true",
                        help="Also compute entity metrics via Modal")
    args = parser.parse_args()

    # Load selection
    with open(SELECTION_FILE) as f:
        selection = json.load(f)
    conversations = {s["file_name"]: s for s in selection}

    # Build (summary, reference=transcript) pairs
    print("Loading summaries and transcripts...")
    items = []
    for model in MODELS:
        for conv_name, conv_info in conversations.items():
            summary = load_summary(model, conv_name)
            if summary is None:
                continue
            transcript = load_transcript(conv_name)

            items.append({
                "conversation": conv_name,
                "model": model,
                "specialty": conv_info.get("category", ""),
                "condition": conv_info.get("detected_condition", ""),
                "summary": summary,
                "reference": transcript,
            })

    print(f"Loaded {len(items)} (summary, transcript) pairs")
    print(f"  Models: {sorted(set(i['model'] for i in items))}")
    print(f"  Per model: {pd.Series([i['model'] for i in items]).value_counts().to_dict()}")

    # Text metrics
    print("\n--- Computing text-based metrics ---")
    start = time.time()
    items = compute_text_metrics(items)
    print(f"Text metrics completed in {time.time()-start:.1f}s")

    # Entity metrics (optional)
    if args.with_entities:
        print("\n--- Computing entity-based metrics (Modal) ---")
        items = compute_entity_metrics_modal(items)

    # Build results DataFrame
    metric_cols = ["bleu", "rouge_l", "sbert_coherence", "bert_f1",
                   "summary_chars", "summary_tokens"]
    if args.with_entities:
        metric_cols += ["scispacy_entity_recall", "medcat_entity_recall"]

    rows = []
    for item in items:
        row = {
            "conversation": item["conversation"],
            "model": item["model"],
            "specialty": item["specialty"],
            "condition": item["condition"],
        }
        for col in metric_cols:
            row[col] = item.get(col, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save full results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_path = os.path.join(OUTPUT_DIR, "fareez_automated_metrics.csv")
    df.to_csv(full_path, index=False)
    print(f"\nSaved: {full_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("AUTOMATED METRICS — Mean (SD) per Model")
    print("=" * 80)

    summary_rows = []
    for model in MODELS:
        md = df[df["model"] == model]
        row = {"Model": model, "n": len(md)}
        for col in metric_cols:
            if col in ["summary_chars", "summary_tokens"]:
                row[col] = f"{md[col].mean():.0f} ({md[col].std():.0f})"
            else:
                row[col] = f"{md[col].mean():.4f} ({md[col].std():.4f})"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("Model")
    print(summary_df.to_string())

    # Save model comparison
    comp_path = os.path.join(OUTPUT_DIR, "fareez_automated_metrics_comparison.csv")
    model_comp = df.groupby("model")[metric_cols].agg(["mean", "std"]).round(4)
    model_comp.to_csv(comp_path)
    print(f"\nSaved: {comp_path}")

    # Per-specialty breakdown
    print("\n" + "=" * 80)
    print("AUTOMATED METRICS — Mean per Model per Specialty")
    print("=" * 80)
    for spec in ["RES", "MSK", "GAS", "CAR"]:
        sd = df[df["specialty"] == spec]
        if len(sd) == 0:
            continue
        print(f"\n  {spec} (n={sd['conversation'].nunique()} conversations):")
        for model in MODELS:
            md = sd[sd["model"] == model]
            if len(md) == 0:
                continue
            vals = "  ".join(f"{col[:6]}={md[col].mean():.3f}" for col in metric_cols[:4])
            print(f"    {model:<18s}  {vals}")


if __name__ == "__main__":
    main()
