#!/usr/bin/env python3
"""
Run Gemma 4 26B A4B and Gemma 4 31B at NF4 via Modal H100 for the ELM
Validator paper. Replaces the OpenRouter bf16 results with deployment-
precision (4-bit) measurements to restore the compute equalizer comparison.

Flow:
1. Build all prompts locally (needs elm_simplifier imports).
2. Send one batch per model to Modal (H100 container, NF4 load).
3. Parse responses and write per-trial CSVs to:
     results/multi_trial/results-{model}-trial{N}.csv         (from "full" condition)
     results/ablation_multi_trial/results-{model}-{cond}-trial{N}.csv

Usage:
    # Sanity check (2 cases, 1 trial, 1 condition — fast smoke test)
    python run_gemma_nf4_modal.py --model gemma-4-26b-a4b --sanity

    # Full run (5 trials x 4 conditions x 41 cases = 820 inferences per model)
    python run_gemma_nf4_modal.py --model gemma-4-26b-a4b
    python run_gemma_nf4_modal.py --model gemma-4-31b

    # Both models sequentially
    python run_gemma_nf4_modal.py --all
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from modal_app import build_prompt, parse_response, extract_embedded_errors
from modal_gemma_nf4 import run_batch as modal_run_batch, app as modal_app

BASE_DIR = Path(__file__).parent
TEST_DATA_DIR = BASE_DIR / "test_data"
GROUND_TRUTH_FILE = TEST_DATA_DIR / "ground_truth.json"
MULTI_DIR = BASE_DIR / "results" / "multi_trial"
ABL_DIR = BASE_DIR / "results" / "ablation_multi_trial"

N_TRIALS = 5
ABLATION_CONDITIONS = ["full", "no_cpg", "no_simplify", "no_cpg_no_simplify"]
MAX_PROMPT_CHARS = 24000  # local-model budget (larger than Groq cap)

MODELS = {
    "gemma-4-26b-a4b": "google/gemma-4-26b-a4b-it",
    "gemma-4-31b": "google/gemma-4-31b-it",
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
            "file_name": fname,
            "elm_json": elm_json,
            "library_name": library,
            "cpg_content": cpg_content,
            "cpg_file": cpg_file,
            "expected_valid": tc["valid"],
        })
    return items


def build_all_prompts(items, n_trials, conditions):
    """Build every (trial, condition, case) prompt.

    Encodes trial/condition/file into a tuple id so we can demux the
    Modal response batch back into per-(trial, condition) CSVs.
    """
    prompts = []
    skipped = {}  # id -> embedded error list (pre-failed, no LLM call needed)
    for condition in conditions:
        for trial in range(1, n_trials + 1):
            for item in items:
                pid = f"{condition}|trial{trial}|{item['file_name']}"
                embedded = extract_embedded_errors(item["elm_json"])
                if embedded:
                    skipped[pid] = embedded
                    continue
                prompt = build_prompt(
                    item["elm_json"],
                    item["library_name"],
                    item["cpg_content"],
                    max_chars=MAX_PROMPT_CHARS,
                    ablation_mode=condition,
                )
                prompts.append({"id": pid, "prompt": prompt})
    return prompts, skipped


def save_csv(rows, path, is_ablation: bool):
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
    """Convert Modal batch results into per-(trial, condition) row lists."""
    by_cond_trial = {}
    by_trial_fullonly = {}  # mirror of "full" condition for multi_trial dir

    items_by_fname = {it["file_name"]: it for it in items}

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
                            "errors": [f"Modal NF4 error: {ans['error']}"],
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
                            "source": "modal-nf4",
                            "time_seconds": ans.get("time_seconds", 0),
                        })
                base["correct"] = base["valid"] == base["expected_valid"]
                trial_rows.append(base)
            by_cond_trial[(condition, trial)] = trial_rows

            if condition == "full":
                # Mirror for multi_trial dir (same rows, drop 'ablation' column later)
                by_trial_fullonly[trial] = [dict(r) for r in trial_rows]

    return by_cond_trial, by_trial_fullonly


def summarize(by_cond_trial, model_id):
    print("\n" + "=" * 70)
    print(f"  {model_id} — NF4 H100 results")
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
            sd = (
                sum((a - mean) ** 2 for a in trial_accs) / len(trial_accs)
            ) ** 0.5
            print(f"  {condition:22s} {mean*100:5.1f}% ± {sd*100:.1f}%  {trial_correct}")


def run_model(model_id, sanity=False):
    items = load_test_data()
    print(f"Loaded {len(items)} test cases")

    if sanity:
        items = items[:2]
        n_trials = 1
        conditions = ["full"]
        print(f"SANITY MODE: {len(items)} cases, {n_trials} trial, conditions={conditions}")
    else:
        n_trials = N_TRIALS
        conditions = ABLATION_CONDITIONS

    prompts, skipped = build_all_prompts(items, n_trials, conditions)
    print(f"Built {len(prompts)} prompts "
          f"({n_trials} trials x {len(conditions)} conditions x "
          f"{len(items) - len(skipped)//max(1, n_trials*len(conditions))} cases)")
    print(f"Skipped (embedded compiler errors): {len(skipped)}")

    hf_name = MODELS[model_id]
    print(f"\nDispatching to Modal: {hf_name} on H100 at NF4...")
    dispatch_start = time.time()

    with modal_app.run():
        modal_results = modal_run_batch.remote(hf_name, prompts)

    elapsed = time.time() - dispatch_start
    print(f"\nModal call complete in {elapsed/60:.1f} minutes")
    print(f"Received {len(modal_results)} results")

    # Report any errors in the batch
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
        print("\n=== SANITY CHECK RESULTS ===")
        for (cond, trial), rows in by_cond_trial.items():
            for r in rows:
                print(
                    f"  {r['file']:60s} valid={r['valid']} "
                    f"correct={r['correct']} ({r['time_seconds']:.1f}s)"
                )
                if r.get("errors"):
                    print(f"    errors: {r['errors'][:1]}")
        return

    # Save ablation CSVs
    for (cond, trial), rows in by_cond_trial.items():
        path = ABL_DIR / f"results-{model_id}-{cond}-trial{trial}.csv"
        save_csv(rows, path, is_ablation=True)
    print(f"Saved {len(by_cond_trial)} ablation trial CSVs to {ABL_DIR}")

    # Save multi-trial CSVs (mirror of "full" condition, without 'ablation' col)
    for trial, rows in by_trial_full.items():
        path = MULTI_DIR / f"results-{model_id}-trial{trial}.csv"
        save_csv(rows, path, is_ablation=False)
    print(f"Saved {len(by_trial_full)} multi-trial CSVs to {MULTI_DIR}")

    summarize(by_cond_trial, model_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument("--all", action="store_true",
                        help="Run both Gemma 4 variants sequentially")
    parser.add_argument("--sanity", action="store_true",
                        help="2 cases, 1 trial, 1 condition — smoke test only")
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
