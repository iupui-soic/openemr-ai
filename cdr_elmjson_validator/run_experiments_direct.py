#!/usr/bin/env python3
"""
Direct experiment runner — calls Groq API without Modal.

Runs ablation and prompt engineering experiments for a Groq-hosted model
by calling the API directly with the local GROQ_API_KEY.

Usage:
    GROQ_API_KEY=xxx python run_experiments_direct.py
"""

import os
import sys
import json
import csv
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from modal_app import (build_prompt, parse_response, extract_embedded_errors,
                       ABLATION_MODES, PROMPT_MODES, MAX_PROMPT_CHARS_GROQ,
                       FEW_SHOT_EXEMPLARS)

BASE_DIR = Path(__file__).parent
TEST_DATA_DIR = BASE_DIR / "test_data"
GROUND_TRUTH_FILE = TEST_DATA_DIR / "ground_truth.json"

GROQ_MODELS = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-32b": "qwen/qwen3-32b",
    "llama-3.3-70b": "llama-3.3-70b-versatile",
}

# Defaults (overridden by --model CLI flag)
GROQ_MODEL = "openai/gpt-oss-20b"
MODEL_ID = "gpt-oss-20b"


def load_test_data():
    """Load all test cases with ground truth."""
    with open(GROUND_TRUTH_FILE) as f:
        gt = json.load(f)
    test_cases = gt["test_cases"]

    items = []
    for fname, tc in test_cases.items():
        elm_path = TEST_DATA_DIR / fname
        if not elm_path.exists():
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
            "file_name": fname,
            "elm_json": elm_json,
            "library_name": library,
            "cpg_content": cpg_content,
            "cpg_file": cpg_file,
            "expected_valid": tc["valid"],
        })

    return items


def run_single(client, item, groq_model, model_id,
               ablation_mode="full", prompt_mode="standard"):
    """Run a single validation via Groq API."""
    elm_json = item["elm_json"]
    library_name = item["library_name"]
    cpg_content = item["cpg_content"]

    # Check embedded errors
    embedded_errors = extract_embedded_errors(elm_json)
    if embedded_errors:
        return {
            "file": item["file_name"], "library": library_name,
            "model": model_id, "valid": False, "errors": embedded_errors,
            "warnings": [], "source": "embedded", "time_seconds": 0,
            "cpg_file": item.get("cpg_file"), "has_cpg": cpg_content is not None,
        }

    prompt = build_prompt(elm_json, library_name, cpg_content,
                          max_chars=MAX_PROMPT_CHARS_GROQ,
                          ablation_mode=ablation_mode, prompt_mode=prompt_mode)

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        elapsed = time.time() - start
        return {
            "file": item["file_name"], "library": library_name,
            "model": model_id, "valid": False,
            "errors": [f"Groq API error: {str(e)[:200]}"],
            "warnings": [], "source": "error", "time_seconds": elapsed,
            "cpg_file": item.get("cpg_file"), "has_cpg": cpg_content is not None,
        }

    elapsed = time.time() - start
    result = parse_response(answer)
    result.update({
        "file": item["file_name"], "library": library_name,
        "model": model_id, "source": "groq", "time_seconds": elapsed,
        "cpg_file": item.get("cpg_file"), "has_cpg": cpg_content is not None,
    })
    return result


def add_ground_truth(result, item):
    """Add ground truth comparison."""
    expected = item["expected_valid"]
    actual = result["valid"]
    result["expected_valid"] = expected
    result["correct"] = expected == actual
    result["has_ground_truth"] = True
    return result


def save_results(results, output_path):
    """Save results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file", "library", "model", "valid", "time_seconds",
        "errors", "warnings", "source", "cpg_file", "has_cpg",
        "has_ground_truth", "correct", "expected_valid",
        "error_match", "warning_match"
    ]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = r.copy()
            row["errors"] = "; ".join(r.get("errors", []))
            row["warnings"] = "; ".join(r.get("warnings", []))
            writer.writerow(row)
    print(f"  Saved: {output_path}")


def run_experiment_set(client, items, modes, mode_type, output_dir,
                       groq_model, model_id):
    """Run a set of experiments (ablation or prompt)."""
    for mode in modes:
        if mode_type == "ablation":
            ablation_mode, prompt_mode = mode, "standard"
        else:
            ablation_mode, prompt_mode = "full", mode

        output_path = output_dir / f"results-{model_id}-{mode}.csv"
        if output_path.exists():
            print(f"\n  Skipping {mode} — already exists")
            continue

        print(f"\n  --- {mode_type}: {mode} ---")
        results = []
        for i, item in enumerate(items, 1):
            print(f"  [{i}/{len(items)}] {item['file_name']}", end="")
            result = run_single(client, item, groq_model, model_id,
                                ablation_mode=ablation_mode,
                                prompt_mode=prompt_mode)
            result = add_ground_truth(result, item)
            results.append(result)
            status = "correct" if result["correct"] else "WRONG"
            print(f"  -> {status} ({result['time_seconds']:.1f}s)")

        correct = sum(1 for r in results if r["correct"])
        print(f"  Accuracy: {correct}/{len(results)} ({correct/len(results):.1%})")
        save_results(results, output_path)


def main():
    import argparse
    from groq import Groq

    parser = argparse.ArgumentParser(description="Run ablation/prompt experiments via Groq API")
    parser.add_argument("--model", default="gpt-oss-20b",
                       choices=list(GROQ_MODELS.keys()),
                       help="Model to run experiments with")
    parser.add_argument("--ablation-only", action="store_true",
                       help="Run only ablation experiments")
    parser.add_argument("--prompt-only", action="store_true",
                       help="Run only prompt experiments")
    args = parser.parse_args()

    model_id = args.model
    groq_model = GROQ_MODELS[model_id]

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY environment variable")
        sys.exit(1)

    client = Groq(api_key=api_key)
    items = load_test_data()
    print(f"Loaded {len(items)} test cases")
    print(f"Model: {model_id} ({groq_model})")

    run_ablation = not args.prompt_only
    run_prompts = not args.ablation_only

    if run_ablation:
        print("\n" + "=" * 70)
        print(f"ABLATION EXPERIMENTS — {model_id}")
        print("=" * 70)
        ablation_dir = BASE_DIR / "results" / "ablation"
        run_experiment_set(client, items, list(ABLATION_MODES), "ablation",
                           ablation_dir, groq_model, model_id)

    if run_prompts:
        print("\n" + "=" * 70)
        print(f"PROMPT ENGINEERING EXPERIMENTS — {model_id}")
        print("=" * 70)
        prompt_dir = BASE_DIR / "results" / "prompts"
        run_experiment_set(client, items, list(PROMPT_MODES), "prompt",
                           prompt_dir, groq_model, model_id)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
