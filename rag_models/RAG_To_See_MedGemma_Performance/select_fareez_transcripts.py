"""
Step 4.9.1: Select 40 Fareez OSCE transcripts balanced by medical specialty.

Selection strategy:
- 20 RES (respiratory) from 213 available
- 10 MSK (musculoskeletal) from 46 available
- 5 GAS (gastroenterology) from 6 available
- 4 CAR (cardiology) from 5 available
- 1 DER (dermatology) from 1 available

Selection criteria:
- All transcripts >= 2000 characters (all qualify)
- Deterministic selection using seed=42 for reproducibility
- Sorted by filename within each specialty before random sampling

Output: fareez_selected_40.json
"""

import os
import json
import random

TRANSCRIPT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "openemr_whisper_wer", "data", "fareez_osce", "Data", "Clean Transcripts"
)

TARGET_COUNTS = {
    "RES": 20,
    "MSK": 10,
    "GAS": 5,
    "CAR": 4,
    "DER": 1,
}

SEED = 42
MIN_CHARS = 2000


def load_all_transcripts(transcript_dir):
    """Load all transcript files and return list of dicts."""
    entries = []
    for f in sorted(os.listdir(transcript_dir)):
        if not f.endswith(".txt"):
            continue
        path = os.path.join(transcript_dir, f)
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                text = fh.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as fh:
                text = fh.read()
        name = f.replace(".txt", "")
        category = name[:3]
        entries.append({
            "file_name": name,
            "category": category,
            "char_count": len(text),
            "transcript": text,
        })
    return entries


def select_transcripts(entries, target_counts, seed=42, min_chars=2000):
    """Select balanced set of transcripts across specialties."""
    rng = random.Random(seed)

    # Group by category
    by_category = {}
    for e in entries:
        if e["char_count"] < min_chars:
            continue
        by_category.setdefault(e["category"], []).append(e)

    selected = []
    for cat, count in target_counts.items():
        pool = by_category.get(cat, [])
        if len(pool) <= count:
            chosen = pool
        else:
            chosen = rng.sample(pool, count)
        chosen.sort(key=lambda x: x["file_name"])
        selected.extend(chosen)

    return selected


def main():
    transcript_dir = os.path.normpath(TRANSCRIPT_DIR)
    print(f"Loading transcripts from: {transcript_dir}")

    entries = load_all_transcripts(transcript_dir)
    print(f"Loaded {len(entries)} total transcripts")

    selected = select_transcripts(entries, TARGET_COUNTS, SEED, MIN_CHARS)
    print(f"\nSelected {len(selected)} transcripts:")

    # Print selection summary
    cat_counts = {}
    for s in selected:
        cat_counts[s["category"]] = cat_counts.get(s["category"], 0) + 1
    for cat in sorted(cat_counts):
        print(f"  {cat}: {cat_counts[cat]}")

    # Save selection JSON (without full transcript text for the index file)
    output = []
    for s in selected:
        output.append({
            "file_name": s["file_name"],
            "category": s["category"],
            "char_count": s["char_count"],
            "detected_condition": "",  # Filled in step 4.9.2
            "matched_openemr_pid": None,  # Filled in step 4.9.3
        })

    output_path = os.path.join(os.path.dirname(__file__), "fareez_selected_40.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved selection to: {output_path}")

    # Print selected file names
    print("\nSelected transcripts:")
    for s in output:
        print(f"  {s['file_name']} ({s['category']}, {s['char_count']} chars)")


if __name__ == "__main__":
    main()
