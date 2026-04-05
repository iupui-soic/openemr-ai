"""
Compute entity metrics (scispaCy + MedCAT recall) via Modal for 159 Fareez summaries.
Saves incrementally to CSV so progress isn't lost on interruption.
"""

import os
import sys
import json
import time
import pandas as pd
import modal

RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTION_FILE = os.path.join(RAG_ROOT, "data", "fareez_selected_40.json")
SUMMARIES_DIR = os.path.join(RAG_ROOT, "results", "fareez", "fareez_summaries")
TRANSCRIPT_DIR = os.path.normpath(os.path.join(
    RAG_ROOT, "..", "openemr_whisper_wer", "data",
    "fareez_osce", "Data", "Clean Transcripts"
))
OUTPUT_CSV = os.path.join(RAG_ROOT, "results", "fareez", "fareez_entity_metrics.csv")
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
    content_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("=" * 10):
            content_start = i + 1
            break
    text = "".join(lines[content_start:]).strip()
    if text.startswith("Error generating summary"):
        return None
    return text


def main():
    # Load existing results to resume
    done = set()
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        done = set(zip(existing["conversation"], existing["model"]))
        print(f"Resuming: {len(done)} already done")

    # Load selection
    with open(SELECTION_FILE) as f:
        selection = json.load(f)

    # Connect to Modal entity evaluator
    print("Connecting to Modal entity evaluator...")
    EntityEvaluator = modal.Cls.from_name("entity-evaluator-service", "EntityEvaluator")
    evaluator = EntityEvaluator()
    print("Connected.")

    # Process
    total = 0
    for model in MODELS:
        for s in selection:
            conv = s["file_name"]
            if (conv, model) in done:
                continue

            summary = load_summary(model, conv)
            if summary is None:
                continue

            transcript = load_transcript(conv)
            total += 1

            try:
                result = evaluator.evaluate.remote(
                    generated=summary,
                    reference=transcript,
                )
                row = {
                    "conversation": conv,
                    "model": model,
                    "specialty": s.get("category", ""),
                    "scispacy_entity_recall": result.get("scispacy_entity_recall", 0.0),
                    "medcat_entity_recall": result.get("medcat_entity_recall", 0.0),
                }

                # Append to CSV
                row_df = pd.DataFrame([row])
                if not os.path.exists(OUTPUT_CSV):
                    row_df.to_csv(OUTPUT_CSV, index=False)
                else:
                    row_df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

                sci = row["scispacy_entity_recall"]
                med = row["medcat_entity_recall"]
                print(f"  [{total:3d}] {conv}/{model}: scispaCy={sci:.4f} MedCAT={med:.4f}")

            except Exception as e:
                print(f"  [{total:3d}] {conv}/{model}: ERROR {str(e)[:80]}")

    # Print summary
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"\nTotal: {len(df)} results")
        print("\nMean entity recall by model:")
        print(df.groupby("model")[["scispacy_entity_recall", "medcat_entity_recall"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
