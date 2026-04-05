"""
Step 4.9.2: Detect primary medical condition from each of the 40 selected Fareez transcripts.

Uses Groq API with GPT-OSS-20B (fast, 100% ELM accuracy) to extract the primary condition
from each transcript. Updates fareez_selected_40.json with detected_condition field.

Usage:
    python detect_conditions.py
"""

import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

TRANSCRIPT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "openemr_whisper_wer", "data", "fareez_osce", "Data", "Clean Transcripts"
))

RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTION_FILE = os.path.join(RAG_ROOT, "data", "fareez_selected_40.json")
MODEL = "llama-3.3-70b-versatile"


def load_transcript(file_name):
    """Load a transcript file by name."""
    path = os.path.join(TRANSCRIPT_DIR, f"{file_name}.txt")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def detect_condition(client, transcript_text):
    """Use Groq to detect the primary medical condition from a transcript."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical expert. Identify the primary disease or medical "
                    "condition discussed in a clinical conversation. "
                    "Respond with ONLY the disease/condition name (1-4 words). "
                    "Examples: COPD, Asthma, Osteoarthritis, Chest Pain, Acid Reflux, "
                    "Pneumonia, Back Pain, Eczema, Angina, Irritable Bowel Syndrome."
                ),
            },
            {
                "role": "user",
                "content": f"""Read the following doctor-patient conversation and identify the PRIMARY medical condition being discussed.

Return ONLY the condition name (1-4 words, no explanation).

Transcript:
{transcript_text[:3000]}

Primary Condition:""",
            },
        ],
        temperature=0.1,
        max_tokens=20,
    )
    raw = response.choices[0].message.content
    if raw:
        condition = raw.strip().split("\n")[0].strip().strip('"').strip("'")
        return condition
    return "General"


def main():
    client = Groq()

    with open(SELECTION_FILE, "r") as f:
        selected = json.load(f)

    print(f"Detecting conditions for {len(selected)} transcripts using {MODEL}...")
    print(f"Transcript dir: {TRANSCRIPT_DIR}\n")

    for i, entry in enumerate(selected):
        file_name = entry["file_name"]
        transcript = load_transcript(file_name)

        condition = detect_condition(client, transcript)
        entry["detected_condition"] = condition

        print(f"  [{i+1:2d}/40] {file_name} ({entry['category']}): {condition}")

        # Rate limiting - Groq free tier
        time.sleep(0.5)

    # Save updated JSON
    with open(SELECTION_FILE, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\nUpdated {SELECTION_FILE} with detected conditions")

    # Summary
    conditions = {}
    for e in selected:
        c = e["detected_condition"]
        conditions[c] = conditions.get(c, 0) + 1
    print("\nCondition distribution:")
    for c, count in sorted(conditions.items(), key=lambda x: -x[1]):
        print(f"  {c}: {count}")


if __name__ == "__main__":
    main()
