#!/usr/bin/env python3
"""
Run the full ablation study (4 conditions × 4 frontier models) with multiple
trials for reproducibility analysis.

Conditions:
  full            – Simplified ELM + CPG
  no_cpg          – Simplified ELM only
  no_simplify     – Raw truncated JSON + CPG  (was "no_simplify" in original)
  no_cpg_no_simplify – Raw truncated JSON only (was "neither" in original)
"""

import os
import sys
import csv
import json
import time
import collections
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from modal_app import (build_prompt, parse_response, extract_embedded_errors,
                       MAX_PROMPT_CHARS_GROQ)

BASE_DIR = Path(__file__).parent
TEST_DATA_DIR = BASE_DIR / "test_data"
GROUND_TRUTH_FILE = TEST_DATA_DIR / "ground_truth.json"
RESULTS_DIR = BASE_DIR / "results" / "ablation_multi_trial"

TEMPERATURE = 0.1

GROQ_MODELS = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-32b": "qwen/qwen3-32b",
    "llama-3.3-70b": "llama-3.3-70b-versatile",
}

ABLATION_CONDITIONS = ["full", "no_cpg", "no_simplify", "no_cpg_no_simplify"]

CONDITION_LABELS = {
    "full": "Full (simpl.+CPG)",
    "no_cpg": "No CPG",
    "no_simplify": "No simplification",
    "no_cpg_no_simplify": "Neither",
}


def load_test_data():
    with open(GROUND_TRUTH_FILE) as f:
        gt = json.load(f)
    items = []
    for fname, tc in gt["test_cases"].items():
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
            "file_name": fname, "elm_json": elm_json, "library_name": library,
            "cpg_content": cpg_content, "cpg_file": cpg_file,
            "expected_valid": tc["valid"],
        })
    return items


def run_single_trial(client, model_id, groq_model, items, ablation_mode, trial_num):
    results = []
    for i, item in enumerate(items, 1):
        print(f"      [{i}/{len(items)}] {item['file_name'][:40]}", end="", flush=True)
        embedded = extract_embedded_errors(item["elm_json"])
        if embedded:
            r = {"file": item["file_name"], "library": item["library_name"],
                 "model": model_id, "trial": trial_num,
                 "ablation": ablation_mode,
                 "valid": False, "errors": embedded,
                 "warnings": [], "source": "embedded", "time_seconds": 0}
        else:
            prompt = build_prompt(item["elm_json"], item["library_name"],
                                  item["cpg_content"], max_chars=MAX_PROMPT_CHARS_GROQ,
                                  ablation_mode=ablation_mode)
            start = time.time()
            try:
                resp = client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE, max_tokens=500)
                answer = resp.choices[0].message.content.strip()
                parsed = parse_response(answer)
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "trial": trial_num,
                     "ablation": ablation_mode,
                     "valid": parsed["valid"],
                     "errors": parsed["errors"], "warnings": parsed["warnings"],
                     "source": "groq", "time_seconds": time.time() - start}
            except Exception as e:
                print(f"  ERROR: {e}")
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "trial": trial_num,
                     "ablation": ablation_mode,
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
        status = "OK" if r["correct"] else "XX"
        print(f"  {status} ({r['time_seconds']:.1f}s)")
    correct = sum(1 for r in results if r["correct"])
    print(f"      => {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    return results


def save_trial_csv(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "library", "model", "trial", "ablation", "valid",
              "time_seconds", "errors", "warnings", "source", "cpg_file",
              "has_cpg", "has_ground_truth", "correct", "expected_valid"]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in results:
            row = r.copy()
            row["errors"] = "; ".join(r.get("errors", []))
            row["warnings"] = "; ".join(r.get("warnings", []))
            w.writerow(row)


def analyze(all_results, n_trials):
    by_model_cond = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in all_results:
        by_model_cond[r["model"]][r["ablation"]].append(r)

    print("\n" + "=" * 78)
    print("ABLATION MULTI-TRIAL RESULTS")
    print(f"Trials: {n_trials}, Temperature: {TEMPERATURE}")
    print("=" * 78)

    # Build summary table: model × condition → mean ± std
    summary_rows = []
    for model_id in GROQ_MODELS:
        print(f"\n  {model_id}:")
        for cond in ABLATION_CONDITIONS:
            results = by_model_cond[model_id][cond]
            if not results:
                continue
            by_trial = collections.defaultdict(list)
            for r in results:
                by_trial[r["trial"]].append(r)
            accs = []
            for t in sorted(by_trial.keys()):
                trial_res = by_trial[t]
                acc = sum(1 for r in trial_res if r["correct"]) / len(trial_res)
                accs.append(acc)
            accs = np.array(accs)
            counts = [int(round(a * 31)) for a in accs]
            label = CONDITION_LABELS[cond]
            print(f"    {label:25s}  {accs.mean()*100:5.1f} ± {accs.std()*100:4.1f}%  "
                  f"[{accs.min()*100:.1f}, {accs.max()*100:.1f}]  counts={counts}")
            summary_rows.append({
                "model": model_id, "condition": cond,
                "mean_accuracy": round(accs.mean(), 4),
                "std_accuracy": round(accs.std(), 4),
                "min_accuracy": round(accs.min(), 4),
                "max_accuracy": round(accs.max(), 4),
                "per_trial_correct": str(counts),
            })

    # Print LaTeX-ready table
    print("\n" + "=" * 78)
    print("LaTeX-ready ablation table (mean ± SD):")
    print("=" * 78)
    for cond in ABLATION_CONDITIONS:
        label = CONDITION_LABELS[cond]
        cells = []
        for model_id in ["gpt-oss-20b", "gpt-oss-120b", "qwen3-32b", "llama-3.3-70b"]:
            row = [r for r in summary_rows
                   if r["model"] == model_id and r["condition"] == cond]
            if row:
                m = row[0]["mean_accuracy"] * 100
                s = row[0]["std_accuracy"] * 100
                cells.append(f"{m:.1f}±{s:.1f}")
            else:
                cells.append("---")
        print(f"  {label:25s} & {'  &  '.join(cells)} \\\\")

    # Save CSV
    summary_path = RESULTS_DIR / "ablation_multi_trial_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "condition", "mean_accuracy", "std_accuracy",
            "min_accuracy", "max_accuracy", "per_trial_correct"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved: {summary_path}")

    # Compute deltas vs full
    print("\n" + "=" * 78)
    print("Deltas vs Full (mean):")
    print("=" * 78)
    for model_id in GROQ_MODELS:
        full_row = [r for r in summary_rows
                    if r["model"] == model_id and r["condition"] == "full"]
        if not full_row:
            continue
        full_mean = full_row[0]["mean_accuracy"]
        print(f"  {model_id}:")
        for cond in ABLATION_CONDITIONS:
            row = [r for r in summary_rows
                   if r["model"] == model_id and r["condition"] == cond]
            if row:
                delta = (row[0]["mean_accuracy"] - full_mean) * 100
                print(f"    {CONDITION_LABELS[cond]:25s}  Δ = {delta:+.1f} pp")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--model", help="Run specific model only")
    parser.add_argument("--condition", help="Run specific condition only",
                        choices=ABLATION_CONDITIONS)
    args = parser.parse_args()
    n_trials = args.trials

    items = load_test_data()
    print(f"Loaded {len(items)} test cases")
    print(f"Running {n_trials} trials × 4 conditions × "
          f"{len(GROQ_MODELS) if not args.model else 1} models")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    from groq import Groq
    client = Groq(api_key=api_key)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    models = {args.model: GROQ_MODELS[args.model]} if args.model else GROQ_MODELS
    conditions = [args.condition] if args.condition else ABLATION_CONDITIONS

    for model_id, groq_model in models.items():
        for cond in conditions:
            print(f"\n{'='*60}")
            print(f"  {model_id} | {CONDITION_LABELS[cond]} | {n_trials} trials")
            print(f"{'='*60}")
            for trial in range(1, n_trials + 1):
                print(f"\n    --- Trial {trial}/{n_trials} ---")
                results = run_single_trial(
                    client, model_id, groq_model, items, cond, trial)
                trial_path = (RESULTS_DIR /
                              f"results-{model_id}-{cond}-trial{trial}.csv")
                save_trial_csv(results, trial_path)
                all_results.extend(results)
                if trial < n_trials:
                    time.sleep(1)

    analyze(all_results, n_trials)
    print("\nDone.")


if __name__ == "__main__":
    main()
