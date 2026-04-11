#!/usr/bin/env python3
"""Run prompt-mode comparison (5 strategies × 3 models) on the 41-case
benchmark via Groq. Mirrors the behaviour of run_multi_trial.py but does
single-run per (model, prompt) combination and writes one CSV per pair into
results/prompts/."""

import csv
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from modal_app import (build_prompt, parse_response, extract_embedded_errors,
                       MAX_PROMPT_CHARS_GROQ)

TEST_DATA_DIR = BASE_DIR / "test_data"
GROUND_TRUTH_FILE = TEST_DATA_DIR / "ground_truth.json"
OUTPUT_DIR = BASE_DIR / "results" / "prompts"

TEMPERATURE = 0.1

GROQ_MODELS = {
    "qwen3-32b": "qwen/qwen3-32b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "llama-3.3-70b": "llama-3.3-70b-versatile",
}

PROMPT_MODES = ["standard", "cot", "structured", "minimal", "few-shot"]


def load_test_data():
    with open(GROUND_TRUTH_FILE) as f:
        gt = json.load(f)
    items = []
    for fname, tc in gt["test_cases"].items():
        elm_path = TEST_DATA_DIR / fname
        if not elm_path.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        with open(elm_path) as f:
            elm_json = json.load(f)
        cpg_content = None
        cpg_file = tc.get("cpg_file")
        if cpg_file:
            cpg_path = TEST_DATA_DIR / cpg_file
            if cpg_path.exists():
                with open(cpg_path) as f:
                    cpg_content = f.read()
        library = elm_json.get("library", {}).get("identifier", {}).get("id", fname)
        items.append({
            "file_name": fname, "elm_json": elm_json, "library_name": library,
            "cpg_content": cpg_content, "cpg_file": cpg_file,
            "expected_valid": tc["valid"],
        })
    return items


def run_one(client, model_id, groq_model, items, prompt_mode):
    results = []
    for i, item in enumerate(items, 1):
        print(f"    [{i}/{len(items)}] {item['file_name']}", end="", flush=True)
        embedded = extract_embedded_errors(item["elm_json"])
        if embedded:
            r = {"file": item["file_name"], "library": item["library_name"],
                 "model": model_id, "valid": False, "errors": embedded,
                 "warnings": [], "source": "embedded", "time_seconds": 0}
        else:
            prompt = build_prompt(item["elm_json"], item["library_name"],
                                  item["cpg_content"],
                                  max_chars=MAX_PROMPT_CHARS_GROQ,
                                  prompt_mode=prompt_mode)
            start = time.time()
            try:
                resp = client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE, max_tokens=4096)
                answer = resp.choices[0].message.content.strip()
                parsed = parse_response(answer)
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "valid": parsed["valid"],
                     "errors": parsed["errors"], "warnings": parsed["warnings"],
                     "source": "groq", "time_seconds": time.time() - start}
            except Exception as e:
                print(f"  ERROR: {e}")
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "valid": False,
                     "errors": [f"API error: {str(e)[:200]}"],
                     "warnings": [], "source": "error",
                     "time_seconds": time.time() - start}
        r["cpg_file"] = item.get("cpg_file")
        r["has_cpg"] = item["cpg_content"] is not None
        r["expected_valid"] = item["expected_valid"]
        r["correct"] = r["valid"] == item["expected_valid"]
        r["has_ground_truth"] = True
        results.append(r)
        status = "OK" if r["correct"] else "WRONG"
        print(f"  {status} ({r['time_seconds']:.1f}s)")
    correct = sum(1 for r in results if r["correct"])
    print(f"    => {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    return results


def save_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "library", "model", "valid", "time_seconds",
                  "errors", "warnings", "source", "cpg_file", "has_cpg",
                  "has_ground_truth", "correct", "expected_valid"]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = r.copy()
            row["errors"] = "; ".join(r.get("errors", []))
            row["warnings"] = "; ".join(r.get("warnings", []))
            writer.writerow(row)


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        env_file = BASE_DIR.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["GROQ_API_KEY"] = api_key
                    break
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    from groq import Groq
    client = Groq(api_key=api_key)

    items = load_test_data()
    print(f"Loaded {len(items)} test cases")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_id, groq_model in GROQ_MODELS.items():
        for mode in PROMPT_MODES:
            output_path = OUTPUT_DIR / f"results-{model_id}-{mode}.csv"
            if output_path.exists():
                print(f"\n  Skipping {model_id} {mode} (already exists)")
                continue
            print(f"\n{'='*60}\n  {model_id}  |  prompt={mode}\n{'='*60}")
            results = run_one(client, model_id, groq_model, items, mode)
            save_results(results, output_path)
            time.sleep(2)
    print("\nDone.")


if __name__ == "__main__":
    main()
