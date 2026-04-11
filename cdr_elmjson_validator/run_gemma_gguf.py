#!/usr/bin/env python3
"""
Run Gemma 4 26B A4B and Gemma 4 31B experiments via Unsloth GGUF Q4_K_M
on Modal L40S, replacing the earlier OpenRouter bf16 and vLLM MXFP4/NF4
attempts (both of which had model-specific blockers for 26B A4B MoE).

Precision: GGUF Q4_K_M (llama.cpp K-quant, 4-bit storage / fp16 compute).
This is the community-standard 4-bit format for Gemma 4 deployment —
what users actually run on consumer GPUs, Ollama, LM Studio, etc. Both
Gemma 4 variants at the same quantization and same hardware (Modal L40S)
for clean internal consistency.

Writes per-trial CSVs to:
  results/multi_trial/results-{model}-trial{N}.csv        (from "full" condition)
  results/ablation_multi_trial/results-{model}-{cond}-trial{N}.csv

Overwrites the prior OpenRouter bf16 results for these two models.

Usage:
    # Sanity: 2 cases × 1 trial × 1 condition
    python run_gemma_gguf.py --model 26b --sanity
    python run_gemma_gguf.py --model 31b --sanity

    # Full run per model: 820 inferences
    python run_gemma_gguf.py --model 26b
    python run_gemma_gguf.py --model 31b
    python run_gemma_gguf.py --all
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from modal_app import build_prompt, parse_response, extract_embedded_errors
from modal_gemma_gguf import run_batch as modal_run_batch, app as modal_app

BASE_DIR = Path(__file__).parent
TEST_DATA_DIR = BASE_DIR / "test_data"
GROUND_TRUTH_FILE = TEST_DATA_DIR / "ground_truth.json"
MULTI_DIR = BASE_DIR / "results" / "multi_trial"
ABL_DIR = BASE_DIR / "results" / "ablation_multi_trial"

N_TRIALS = 5
ABLATION_CONDITIONS = ["full", "no_cpg", "no_simplify", "no_cpg_no_simplify"]
MAX_PROMPT_CHARS = 24000

MODELS = {
    "26b": "gemma-4-26b-a4b",
    "31b": "gemma-4-31b",
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
            "file_name": fname,
            "elm_json": elm_json,
            "library_name": library,
            "cpg_content": cpg_content,
            "cpg_file": cpg_file,
            "expected_valid": tc["valid"],
        })
    return items


def build_all_prompts(items, n_trials, conditions):
    """Build every (trial, condition, case) prompt and return as a flat batch."""
    prompts = []
    skipped = {}
    for condition in conditions:
        for trial in range(1, n_trials + 1):
            for item in items:
                pid = f"{condition}|trial{trial}|{item['file_name']}"
                embedded = extract_embedded_errors(item["elm_json"])
                if embedded:
                    skipped[pid] = embedded
                    continue
                prompt = build_prompt(
                    item["elm_json"], item["library_name"], item["cpg_content"],
                    max_chars=MAX_PROMPT_CHARS, ablation_mode=condition,
                )
                prompts.append({"id": pid, "prompt": prompt})
    return prompts, skipped


def save_csv(rows, path, is_ablation):
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_ablation:
        fields = [
            "file", "library", "model", "trial", "ablation", "valid",
            "time_seconds", "errors", "warnings", "source", "cpg_file",
            "has_cpg", "has_ground_truth", "correct", "expected_valid",
        ]
    else:
        fields = [
            "file", "library", "model", "trial", "valid", "time_seconds",
            "errors", "warnings", "source", "cpg_file", "has_cpg",
            "has_ground_truth", "correct", "expected_valid",
        ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = r.copy()
            row["errors"] = "; ".join(r.get("errors", []))
            row["warnings"] = "; ".join(r.get("warnings", []))
            w.writerow(row)


def demux_results(items, answers_by_id, skipped_ids, model_id, n_trials, conditions):
    by_cond_trial = {}
    by_trial_full = {}

    for condition in conditions:
        for trial in range(1, n_trials + 1):
            trial_rows = []
            for item in items:
                pid = f"{condition}|trial{trial}|{item['file_name']}"
                base = {
                    "file": item["file_name"],
                    "library": item["library_name"],
                    "model": model_id,
                    "trial": trial,
                    "ablation": condition,
                    "cpg_file": item.get("cpg_file"),
                    "has_cpg": item["cpg_content"] is not None,
                    "expected_valid": item["expected_valid"],
                    "has_ground_truth": True,
                }
                if pid in skipped_ids:
                    base.update({
                        "valid": False,
                        "errors": skipped_ids[pid],
                        "warnings": [],
                        "source": "embedded",
                        "time_seconds": 0,
                    })
                else:
                    ans = answers_by_id.get(pid)
                    if ans is None:
                        base.update({
                            "valid": False,
                            "errors": ["Modal batch result missing"],
                            "warnings": [],
                            "source": "error",
                            "time_seconds": 0,
                        })
                    elif ans.get("error"):
                        base.update({
                            "valid": False,
                            "errors": [f"GGUF error: {ans['error']}"],
                            "warnings": [],
                            "source": "error",
                            "time_seconds": ans.get("time_seconds", 0),
                        })
                    else:
                        parsed = parse_response(ans["answer"])
                        base.update({
                            "valid": parsed["valid"],
                            "errors": parsed["errors"],
                            "warnings": parsed["warnings"],
                            "source": "modal-gguf-q4km",
                            "time_seconds": ans.get("time_seconds", 0),
                        })
                base["correct"] = base["valid"] == base["expected_valid"]
                trial_rows.append(base)
            by_cond_trial[(condition, trial)] = trial_rows
            if condition == "full":
                by_trial_full[trial] = [dict(r) for r in trial_rows]

    return by_cond_trial, by_trial_full


def summarize(by_cond_trial, model_id):
    print("\n" + "=" * 70)
    print(f"  {model_id} — Modal L40S GGUF Q4_K_M results")
    print("=" * 70)
    for condition in ABLATION_CONDITIONS:
        trial_accs = []
        trial_correct = []
        for trial in range(1, N_TRIALS + 1):
            rows = by_cond_trial.get((condition, trial), [])
            if not rows:
                continue
            c = sum(1 for r in rows if r["correct"])
            n = len(rows)
            trial_accs.append(c / n)
            trial_correct.append(c)
        if trial_accs:
            mean = sum(trial_accs) / len(trial_accs)
            sd = (sum((a - mean) ** 2 for a in trial_accs) / len(trial_accs)) ** 0.5
            print(f"  {condition:22s} {mean*100:5.1f}% ± {sd*100:.1f}%  {trial_correct}")


def run_model(model_key, sanity=False):
    model_id = MODELS[model_key]

    items = load_test_data()
    print(f"Loaded {len(items)} test cases")

    if sanity:
        items = items[:2]
        n_trials = 1
        conditions = ["full"]
    else:
        n_trials = N_TRIALS
        conditions = ABLATION_CONDITIONS

    prompts, skipped = build_all_prompts(items, n_trials, conditions)
    total_prompts = len(prompts) + len(skipped)
    print(f"Built {len(prompts)} prompts ({n_trials} trials x {len(conditions)} conditions x {len(items)} cases = {total_prompts})")
    print(f"Skipped (embedded compiler errors): {len(skipped)}")

    print(f"\nDispatching to Modal (modal_gemma_gguf::run_batch) for {model_id}...")
    dispatch_start = time.time()
    with modal_app.run():
        modal_results = modal_run_batch.remote(model_id, prompts)
    elapsed = time.time() - dispatch_start
    print(f"\nModal call complete in {elapsed/60:.1f} minutes")
    print(f"Received {len(modal_results)} results")

    n_errors = sum(1 for r in modal_results if r.get("error"))
    if n_errors:
        print(f"WARNING: {n_errors}/{len(modal_results)} inferences returned errors")
        for r in modal_results[:5]:
            if r.get("error"):
                print(f"  {r['id']}: {r['error'][:200]}")

    answers_by_id = {r["id"]: r for r in modal_results}
    by_cond_trial, by_trial_full = demux_results(
        items, answers_by_id, skipped, model_id, n_trials, conditions
    )

    if sanity:
        print("\n=== SANITY RESULTS ===")
        for (cond, trial), rows in by_cond_trial.items():
            for r in rows:
                print(
                    f"  {r['file']:60s} valid={r['valid']} "
                    f"correct={r['correct']} ({r['time_seconds']:.1f}s)"
                )
        return

    # Save ablation CSVs
    for (cond, trial), rows in by_cond_trial.items():
        path = ABL_DIR / f"results-{model_id}-{cond}-trial{trial}.csv"
        save_csv(rows, path, is_ablation=True)
    print(f"Saved {len(by_cond_trial)} ablation trial CSVs to {ABL_DIR}")

    # Save multi-trial CSVs (from "full" condition)
    for trial, rows in by_trial_full.items():
        path = MULTI_DIR / f"results-{model_id}-trial{trial}.csv"
        save_csv(rows, path, is_ablation=False)
    print(f"Saved {len(by_trial_full)} multi-trial CSVs to {MULTI_DIR}")

    summarize(by_cond_trial, model_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    if args.all:
        models = list(MODELS.keys())
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model or --all")

    for m in models:
        run_model(m, sanity=args.sanity)
        print(f"\n{m} done.\n")


if __name__ == "__main__":
    main()
