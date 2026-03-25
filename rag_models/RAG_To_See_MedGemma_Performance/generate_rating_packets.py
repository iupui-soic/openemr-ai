"""
Generate randomized, blinded rating packets for 3 clinician raters.

For each rater, produces a single PDF-ready text file containing all summaries
in randomized order with:
- Original transcript
- OpenEMR data extract
- AI-generated summary (blinded — model replaced with summary ID)
- Rating sheet

Randomization rules:
- Each rater gets a unique random order (seeds: 100, 200, 300)
- The same conversation's model outputs are never consecutive
- Model identity is replaced with an opaque Summary ID (e.g., S001)

Also generates:
- answer_key.csv: Maps summary IDs to model names (for analysis after collection)
- rating_template.csv: Empty CSV raters can fill in instead of paper forms

Usage:
    python generate_rating_packets.py
    python generate_rating_packets.py --output-dir rating_packets
"""

import os
import json
import csv
import random
import argparse
from pathlib import Path


BASE_DIR = os.path.dirname(__file__)
SELECTION_FILE = os.path.join(BASE_DIR, "fareez_selected_40.json")
EXTRACTS_DIR = os.path.join(BASE_DIR, "fareez_openemr_extracts")
SUMMARIES_DIR = os.path.join(BASE_DIR, "results", "fareez", "fareez_summaries")
TRANSCRIPT_DIR = os.path.normpath(os.path.join(
    BASE_DIR, "..", "..", "openemr_whisper_wer", "data",
    "fareez_osce", "Data", "Clean Transcripts"
))

MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3-32b", "medgemma-4b"]
RATER_SEEDS = {"rater_1": 100, "rater_2": 200, "rater_3": 300}

RATING_DIMENSIONS = [
    "Accuracy",
    "Completeness",
    "Organization",
    "Conciseness",
    "Clinical Utility",
    "Overall Quality",
]


def load_transcript(file_name):
    path = os.path.join(TRANSCRIPT_DIR, f"{file_name}.txt")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def load_openemr_extract(file_name):
    path = os.path.join(EXTRACTS_DIR, f"{file_name}_openemr.txt")
    if not os.path.exists(path):
        return "[No OpenEMR data available]"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_summary(model, file_name):
    path = os.path.join(SUMMARIES_DIR, model, f"{file_name}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Skip the metadata header (lines before the === separator)
    content_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("=" * 10):
            content_start = i + 1
            break
    return "".join(lines[content_start:]).strip()


def generate_shuffled_order(conversations, models, seed):
    """Generate a randomized order ensuring no consecutive same-conversation items."""
    rng = random.Random(seed)

    # Create all (conversation, model) pairs
    items = []
    for conv in conversations:
        for model in models:
            items.append((conv, model))

    # Shuffle with constraint: same conversation not consecutive
    for _ in range(1000):  # Try up to 1000 times
        rng.shuffle(items)
        valid = True
        for i in range(1, len(items)):
            if items[i][0] == items[i - 1][0]:
                valid = False
                break
        if valid:
            return items

    # Fallback: interleave by model to guarantee no consecutive same-conversation
    by_model = {m: [] for m in models}
    for conv in conversations:
        for m in models:
            by_model[m].append(conv)
    for m in models:
        rng.shuffle(by_model[m])

    result = []
    for i in range(len(conversations)):
        for m in models:
            if by_model[m]:
                result.append((by_model[m].pop(0), m))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="rating_packets")
    args = parser.parse_args()

    output_dir = os.path.join(BASE_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load selection
    with open(SELECTION_FILE, "r") as f:
        selection = json.load(f)
    conversations = [s["file_name"] for s in selection]

    # Check which models have results
    available_models = []
    for m in MODELS:
        model_dir = os.path.join(SUMMARIES_DIR, m)
        if os.path.exists(model_dir) and len(os.listdir(model_dir)) >= 40:
            available_models.append(m)
        else:
            count = len(os.listdir(model_dir)) if os.path.exists(model_dir) else 0
            print(f"  Warning: {m} has {count}/40 summaries — skipping")

    print(f"Available models: {available_models}")
    num_models = len(available_models)
    total_summaries = len(conversations) * num_models
    print(f"Total summaries per rater: {total_summaries}")

    # Create global answer key (summary_id -> model, conversation)
    # Use consistent IDs across raters for the answer key
    answer_key = []
    summary_counter = 1

    for rater_id, seed in RATER_SEEDS.items():
        print(f"\nGenerating packet for {rater_id} (seed={seed})...")

        order = generate_shuffled_order(conversations, available_models, seed)

        # Build packet text
        packet_lines = []
        packet_lines.append("=" * 80)
        packet_lines.append("CLINICIAN RATING PACKET")
        packet_lines.append(f"Rater: {rater_id}")
        packet_lines.append(f"Total Summaries: {len(order)}")
        packet_lines.append(f"Conversations: {len(conversations)} | Models: {num_models} (blinded)")
        packet_lines.append("=" * 80)
        packet_lines.append("")
        packet_lines.append("INSTRUCTIONS:")
        packet_lines.append("1. For each summary, read the TRANSCRIPT and OPENEMR DATA first")
        packet_lines.append("2. Then read the AI-GENERATED SUMMARY")
        packet_lines.append("3. Rate 6 dimensions on a 1-5 scale (see rating instrument)")
        packet_lines.append("4. Record ratings in the provided CSV template or on paper")
        packet_lines.append("5. Take a 10-minute break every 15-20 summaries")
        packet_lines.append("")

        # CSV template rows for this rater
        csv_rows = []

        for idx, (conv, model) in enumerate(order):
            summary_id = f"S{summary_counter:03d}"
            summary_counter += 1

            transcript = load_transcript(conv)
            openemr = load_openemr_extract(conv)
            summary = load_summary(model, conv)

            if summary is None:
                print(f"  Warning: No summary for {conv}/{model}")
                continue

            # Get specialty from selection
            specialty = ""
            condition = ""
            for s in selection:
                if s["file_name"] == conv:
                    specialty = s.get("category", "")
                    condition = s.get("detected_condition", "")
                    break

            # Answer key entry
            answer_key.append({
                "summary_id": summary_id,
                "rater": rater_id,
                "conversation": conv,
                "model": model,
                "specialty": specialty,
                "condition": condition,
                "order_position": idx + 1,
            })

            # CSV template row
            csv_rows.append({
                "summary_id": summary_id,
                "accuracy": "",
                "completeness": "",
                "organization": "",
                "conciseness": "",
                "clinical_utility": "",
                "overall_quality": "",
                "comments": "",
            })

            # Packet entry
            packet_lines.append("")
            packet_lines.append("#" * 80)
            packet_lines.append(f"# SUMMARY {summary_id}  ({idx + 1}/{len(order)})")
            packet_lines.append(f"# Conversation: {conv} | Specialty: {specialty} | Condition: {condition}")
            packet_lines.append("#" * 80)
            packet_lines.append("")

            # Section 1: Transcript
            packet_lines.append("-" * 60)
            packet_lines.append("SECTION A: ORIGINAL TRANSCRIPT")
            packet_lines.append("-" * 60)
            packet_lines.append("")
            packet_lines.append(transcript)
            packet_lines.append("")

            # Section 2: OpenEMR Data
            packet_lines.append("-" * 60)
            packet_lines.append("SECTION B: OPENEMR DATA EXTRACT")
            packet_lines.append("-" * 60)
            packet_lines.append("")
            packet_lines.append(openemr)
            packet_lines.append("")

            # Section 3: AI Summary (BLINDED)
            packet_lines.append("-" * 60)
            packet_lines.append(f"SECTION C: AI-GENERATED SUMMARY  [{summary_id}]")
            packet_lines.append("-" * 60)
            packet_lines.append("")
            packet_lines.append(summary)
            packet_lines.append("")

            # Rating box
            packet_lines.append("-" * 60)
            packet_lines.append(f"RATING SHEET — {summary_id}")
            packet_lines.append("-" * 60)
            packet_lines.append("")
            packet_lines.append(f"Summary ID: {summary_id}")
            packet_lines.append(f"Rater: {rater_id}")
            packet_lines.append("")
            packet_lines.append("Accuracy:         [ 1 | 2 | 3 | 4 | 5 ]")
            packet_lines.append("Completeness:     [ 1 | 2 | 3 | 4 | 5 ]")
            packet_lines.append("Organization:     [ 1 | 2 | 3 | 4 | 5 ]")
            packet_lines.append("Conciseness:      [ 1 | 2 | 3 | 4 | 5 ]")
            packet_lines.append("Clinical Utility: [ 1 | 2 | 3 | 4 | 5 ]")
            packet_lines.append("Overall Quality:  [ 1 | 2 | 3 | 4 | 5 ]")
            packet_lines.append("")
            packet_lines.append("Comments: ___________________________________________")
            packet_lines.append("")

        # Save packet
        packet_path = os.path.join(output_dir, f"packet_{rater_id}.txt")
        with open(packet_path, "w", encoding="utf-8") as f:
            f.write("\n".join(packet_lines))
        print(f"  Saved: {packet_path} ({len(order)} summaries, {len(packet_lines)} lines)")

        # Save CSV template for this rater
        csv_path = os.path.join(output_dir, f"ratings_{rater_id}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "summary_id", "accuracy", "completeness", "organization",
                "conciseness", "clinical_utility", "overall_quality", "comments"
            ])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  Saved: {csv_path}")

    # Save answer key (researcher only — do NOT share with raters)
    answer_key_path = os.path.join(output_dir, "ANSWER_KEY_DO_NOT_SHARE.csv")
    with open(answer_key_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "summary_id", "rater", "conversation", "model",
            "specialty", "condition", "order_position"
        ])
        writer.writeheader()
        writer.writerows(answer_key)
    print(f"\nAnswer key: {answer_key_path}")
    print("WARNING: Do NOT share the answer key with raters!")

    print(f"\nDone. Rating packets saved to {output_dir}/")
    print(f"  - 3 packet files (packet_rater_1.txt, etc.)")
    print(f"  - 3 CSV templates (ratings_rater_1.csv, etc.)")
    print(f"  - 1 answer key (ANSWER_KEY_DO_NOT_SHARE.csv)")


if __name__ == "__main__":
    main()
