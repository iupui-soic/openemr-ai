"""
Compute entity metrics in parallel using Modal .map() for ~10x speedup.
Fans out across multiple Modal containers simultaneously.
"""

import os
import json
import pandas as pd
import modal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SELECTION_FILE = os.path.join(BASE_DIR, "fareez_selected_40.json")
SUMMARIES_DIR = os.path.join(BASE_DIR, "results", "fareez", "fareez_summaries")
TRANSCRIPT_DIR = os.path.normpath(os.path.join(
    BASE_DIR, "..", "..", "openemr_whisper_wer", "data",
    "fareez_osce", "Data", "Clean Transcripts"
))
OUTPUT_CSV = os.path.join(BASE_DIR, "results", "fareez", "fareez_entity_metrics.csv")
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
    with open(SELECTION_FILE) as f:
        selection = json.load(f)

    # Load existing results to skip
    done = set()
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        done = set(zip(existing["conversation"], existing["model"]))
        print(f"Already done: {len(done)}")

    # Build work items
    items = []
    for model in MODELS:
        for s in selection:
            conv = s["file_name"]
            if (conv, model) in done:
                continue
            summary = load_summary(model, conv)
            if summary is None:
                continue
            transcript = load_transcript(conv)
            items.append({
                "conversation": conv,
                "model": model,
                "specialty": s.get("category", ""),
                "summary": summary,
                "transcript": transcript,
            })

    print(f"Remaining: {len(items)} items to process in parallel")
    if not items:
        print("Nothing to do.")
        return

    # Connect to Modal
    EntityEvaluator = modal.Cls.from_name("entity-evaluator-service", "EntityEvaluator")
    evaluator = EntityEvaluator()

    # Use .map() for parallel execution
    summaries = [item["summary"] for item in items]
    transcripts = [item["transcript"] for item in items]

    print("Running entity evaluation in parallel via Modal .map()...")
    results = list(evaluator.evaluate.map(summaries, transcripts))

    # Combine with metadata and save
    rows = []
    for item, result in zip(items, results):
        rows.append({
            "conversation": item["conversation"],
            "model": item["model"],
            "specialty": item["specialty"],
            "scispacy_entity_recall": result.get("scispacy_entity_recall", 0.0),
            "medcat_entity_recall": result.get("medcat_entity_recall", 0.0),
        })

    new_df = pd.DataFrame(rows)

    # Append to existing
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(combined)} total results to {OUTPUT_CSV}")

    # Summary
    print("\nMean entity recall by model:")
    print(combined.groupby("model")[["scispacy_entity_recall", "medcat_entity_recall"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
