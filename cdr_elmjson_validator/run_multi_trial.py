#!/usr/bin/env python3
"""
Run multiple trials of frontier models via Groq API for reproducibility analysis.

Runs N trials of the 4 frontier models at temperature=0.1 (matching the original
experiment), saves per-trial results, and computes per-case stability and
aggregate statistics (mean, std, min, max accuracy).
"""

import os
import sys
import json
import csv
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from modal_app import (build_prompt, parse_response, extract_embedded_errors,
                       MAX_PROMPT_CHARS_GROQ)

BASE_DIR = Path(__file__).parent
TEST_DATA_DIR = BASE_DIR / "test_data"
GROUND_TRUTH_FILE = TEST_DATA_DIR / "ground_truth.json"
RESULTS_DIR = BASE_DIR / "results" / "multi_trial"

N_TRIALS = 5
TEMPERATURE = 0.1

GROQ_MODELS = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-32b": "qwen/qwen3-32b",
    "llama-3.3-70b": "llama-3.3-70b-versatile",
}


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


def run_single_trial(client, model_id, groq_model, items, trial_num,
                     prompt_mode="standard"):
    """Run one trial of a single model on all test cases."""
    results = []
    for i, item in enumerate(items, 1):
        print(f"    [{i}/{len(items)}] {item['file_name']}", end="", flush=True)
        embedded = extract_embedded_errors(item["elm_json"])
        if embedded:
            r = {"file": item["file_name"], "library": item["library_name"],
                 "model": model_id, "trial": trial_num,
                 "valid": False, "errors": embedded,
                 "warnings": [], "source": "embedded", "time_seconds": 0}
        else:
            prompt = build_prompt(item["elm_json"], item["library_name"],
                                  item["cpg_content"], max_chars=MAX_PROMPT_CHARS_GROQ,
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
                     "model": model_id, "trial": trial_num,
                     "valid": parsed["valid"],
                     "errors": parsed["errors"], "warnings": parsed["warnings"],
                     "source": "groq", "time_seconds": time.time() - start}
            except Exception as e:
                print(f"  ERROR: {e}")
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "trial": trial_num,
                     "valid": False,
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
    print(f"    => Trial {trial_num}: {correct}/{len(results)} "
          f"({correct/len(results)*100:.1f}%)")
    return results


def save_trial_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "library", "model", "trial", "valid", "time_seconds",
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


def analyze_results(all_results, n_trials=N_TRIALS):
    """Compute per-model and per-case stability statistics."""
    import collections

    # Group by model
    by_model = collections.defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    print("\n" + "=" * 70)
    print("MULTI-TRIAL REPRODUCIBILITY ANALYSIS")
    print(f"Trials: {n_trials}, Temperature: {TEMPERATURE}")
    print("=" * 70)

    summary_rows = []

    for model_id in GROQ_MODELS:
        model_results = by_model[model_id]
        if not model_results:
            continue

        # Group by trial
        by_trial = collections.defaultdict(list)
        for r in model_results:
            by_trial[r["trial"]].append(r)

        trial_accuracies = []
        for t in sorted(by_trial.keys()):
            trial_res = by_trial[t]
            acc = sum(1 for r in trial_res if r["correct"]) / len(trial_res)
            trial_accuracies.append(acc)

        import numpy as np
        accs = np.array(trial_accuracies)

        print(f"\n{model_id}:")
        print(f"  Per-trial accuracy: {[f'{a*100:.1f}%' for a in accs]}")
        print(f"  Mean: {accs.mean()*100:.1f}%  Std: {accs.std()*100:.1f}%  "
              f"Min: {accs.min()*100:.1f}%  Max: {accs.max()*100:.1f}%")
        n_correct_per_trial = [int(a * 31) for a in accs]
        print(f"  Correct counts: {n_correct_per_trial}")

        summary_rows.append({
            "model": model_id,
            "n_trials": n_trials,
            "temperature": TEMPERATURE,
            "mean_accuracy": accs.mean(),
            "std_accuracy": accs.std(),
            "min_accuracy": accs.min(),
            "max_accuracy": accs.max(),
            "per_trial_correct": n_correct_per_trial,
            "per_trial_accuracy": [round(a, 4) for a in accs.tolist()],
        })

        # Per-case stability
        by_case = collections.defaultdict(list)
        for r in model_results:
            by_case[r["file"]].append(r["correct"])

        unstable = []
        for case, correctness_list in sorted(by_case.items()):
            if len(set(correctness_list)) > 1:
                n_correct = sum(correctness_list)
                unstable.append((case, n_correct, len(correctness_list)))

        if unstable:
            print(f"  Unstable cases ({len(unstable)}):")
            for case, nc, nt in unstable:
                print(f"    {case}: {nc}/{nt} correct")
        else:
            print(f"  All 31 cases stable across {n_trials} trials")

    # Save summary CSV
    summary_path = RESULTS_DIR / "multi_trial_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "n_trials", "temperature",
            "mean_accuracy", "std_accuracy", "min_accuracy", "max_accuracy",
            "per_trial_correct", "per_trial_accuracy"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"\nSummary saved: {summary_path}")

    # Save per-case stability CSV
    case_stability_path = RESULTS_DIR / "per_case_stability.csv"
    case_rows = []
    for model_id in GROQ_MODELS:
        model_results = by_model[model_id]
        by_case = collections.defaultdict(list)
        for r in model_results:
            by_case[r["file"]].append(r["correct"])
        for case in sorted(by_case.keys()):
            vals = by_case[case]
            case_rows.append({
                "model": model_id,
                "case": case,
                "n_trials": len(vals),
                "n_correct": sum(vals),
                "agreement_rate": sum(vals) / len(vals) if all(vals) else
                                  (0.0 if not any(vals) else sum(vals) / len(vals)),
                "stable": len(set(vals)) == 1,
            })
    with open(case_stability_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "case", "n_trials", "n_correct", "agreement_rate", "stable"])
        writer.writeheader()
        writer.writerows(case_rows)
    print(f"Per-case stability saved: {case_stability_path}")

    return summary_rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-trial frontier model evaluation")
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help=f"Number of trials (default: {N_TRIALS})")
    parser.add_argument("--model", help="Run specific model only")
    parser.add_argument("--prompt-mode", default="standard",
                        choices=["standard", "few-shot", "cot", "structured", "minimal"],
                        help="Prompt strategy (default: standard)")
    args = parser.parse_args()

    num_trials = args.trials

    items = load_test_data()
    print(f"Loaded {len(items)} test cases")
    print(f"Running {num_trials} trials per model at temperature={TEMPERATURE}")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    from groq import Groq
    client = Groq(api_key=api_key)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    models_to_run = GROQ_MODELS
    if args.model:
        if args.model not in GROQ_MODELS:
            print(f"ERROR: Unknown model '{args.model}'. Available: {list(GROQ_MODELS.keys())}")
            sys.exit(1)
        models_to_run = {args.model: GROQ_MODELS[args.model]}

    for model_id, groq_model in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"  {model_id} — {num_trials} trials")
        print(f"{'='*60}")

        for trial in range(1, num_trials + 1):
            print(f"\n  --- Trial {trial}/{num_trials} ---")
            results = run_single_trial(client, model_id, groq_model, items, trial,
                                       prompt_mode=args.prompt_mode)

            # Save individual trial
            suffix = f"-{args.prompt_mode}" if args.prompt_mode != "standard" else ""
            trial_path = RESULTS_DIR / f"results-{model_id}{suffix}-trial{trial}.csv"
            save_trial_results(results, trial_path)
            all_results.extend(results)

            # Brief pause between trials to avoid rate limiting
            if trial < num_trials:
                time.sleep(2)

    # Analyze
    analyze_results(all_results, num_trials)
    print("\nDone.")


if __name__ == "__main__":
    main()
