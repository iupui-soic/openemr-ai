#!/usr/bin/env python3
"""
Prompt Engineering Experiments for ELM Validation.

Runs a model across all 4 prompt strategies and compares results:
  standard   - direct comparison instruction (default)
  cot        - chain-of-thought reasoning before verdict
  structured - category-by-category checklist
  minimal    - bare minimum instructions

Justifies the final prompt design choice with empirical data.

Usage:
    python run_prompt_experiments.py --model gpt-oss-20b
    python run_prompt_experiments.py --model qwen3-32b --output-dir results/prompts
    python run_prompt_experiments.py --analyze-only --results-dir results/prompts
"""

import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path(__file__).parent))

PROMPT_MODES = ["standard", "cot", "structured", "minimal", "few-shot"]

PROMPT_LABELS = {
    "standard": "Standard (direct comparison)",
    "cot": "Chain-of-thought",
    "structured": "Structured checklist",
    "minimal": "Minimal instruction",
    "few-shot": "Few-shot (2 exemplars)",
}


def run_prompt_experiments(model_id, data_dir, output_dir):
    """Run all 4 prompt modes for a model."""
    from run_validation import run_validation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode in PROMPT_MODES:
        output_file = output_dir / f"results-{model_id}-{mode}.csv"
        if output_file.exists():
            print(f"\n  Skipping {mode} — results already exist at {output_file}")
            continue

        print(f"\n{'='*70}")
        print(f"  PROMPT MODE: {mode} ({PROMPT_LABELS[mode]})")
        print(f"{'='*70}")
        run_validation(model_id, data_dir, output_file,
                       ablation_mode="full", prompt_mode=mode)

    print(f"\nAll prompt experiments complete. Results in {output_dir}/")


def load_prompt_results(results_dir, model_id=None):
    """Load prompt experiment CSVs into a single DataFrame."""
    results_dir = Path(results_dir)
    frames = []

    for mode in PROMPT_MODES:
        pattern = f"results-*-{mode}.csv"
        files = list(results_dir.glob(pattern))
        if model_id:
            files = [f for f in files if model_id in f.name]

        for fpath in files:
            df = pd.read_csv(fpath)
            df["prompt_mode"] = mode
            for col in ["valid", "correct", "expected_valid"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.lower() == "true"
            frames.append(df)

    if not frames:
        print(f"No prompt experiment results found in {results_dir}")
        return None

    return pd.concat(frames, ignore_index=True)


def analyze_prompt_results(df):
    """Analyze and compare prompt modes."""
    print("\n" + "=" * 78)
    print("PROMPT ENGINEERING — STRATEGY COMPARISON")
    print("=" * 78)

    models = sorted(df["model"].unique())
    print(f"\nModels: {models}")

    # 1. Accuracy per prompt mode
    print("\n" + "-" * 78)
    print("1. ACCURACY PER PROMPT MODE (Wilson 95% CI)")
    print("-" * 78)

    rows = []
    for model in models:
        print(f"\n  Model: {model}")
        print(f"  {'Prompt Mode':<32s} {'Correct':>8s} {'Acc':>7s} {'95% CI':>18s}")
        print("  " + "-" * 70)

        for mode in PROMPT_MODES:
            mdf = df[(df["model"] == model) & (df["prompt_mode"] == mode)]
            if len(mdf) == 0:
                continue
            n = len(mdf)
            k = int(mdf["correct"].sum())
            acc = k / n
            ci_lo, ci_hi = proportion_confint(k, n, alpha=0.05, method="wilson")
            label = PROMPT_LABELS[mode]
            rows.append({
                "model": model, "prompt_mode": mode, "label": label,
                "correct": k, "total": n, "accuracy": acc,
                "ci_low": ci_lo, "ci_high": ci_hi,
            })
            print(f"  {label:<32s} {k:>3d}/{n:<3d}  {acc:5.1%}  [{ci_lo:.3f}, {ci_hi:.3f}]")

    accuracy_df = pd.DataFrame(rows)

    # 2. Confusion matrix per prompt mode
    print("\n" + "-" * 78)
    print("2. CONFUSION MATRICES PER PROMPT MODE")
    print("-" * 78)

    cm_rows = []
    for model in models:
        print(f"\n  Model: {model}")
        print(f"  {'Mode':<15s} {'TP':>3s} {'FP':>3s} {'TN':>3s} {'FN':>3s} {'Sens':>6s} {'Spec':>6s} {'F1':>6s}")
        print("  " + "-" * 55)

        for mode in PROMPT_MODES:
            mdf = df[(df["model"] == model) & (df["prompt_mode"] == mode)]
            if len(mdf) == 0:
                continue
            tp = int((mdf["valid"] & mdf["expected_valid"]).sum())
            fp = int((mdf["valid"] & ~mdf["expected_valid"]).sum())
            tn = int((~mdf["valid"] & ~mdf["expected_valid"]).sum())
            fn = int((~mdf["valid"] & mdf["expected_valid"]).sum())
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            cm_rows.append({
                "model": model, "prompt_mode": mode,
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "sensitivity": sens, "specificity": spec, "F1": f1,
            })
            print(f"  {mode:<15s} {tp:3d} {fp:3d} {tn:3d} {fn:3d} {sens:6.2f} {spec:6.2f} {f1:6.2f}")

    cm_df = pd.DataFrame(cm_rows)

    # 3. McNemar tests: standard vs each variant
    print("\n" + "-" * 78)
    print("3. McNEMAR'S EXACT TESTS (standard vs each variant)")
    print("-" * 78)

    mcnemar_rows = []
    for model in models:
        std_df = df[(df["model"] == model) & (df["prompt_mode"] == "standard")].set_index("file")
        if len(std_df) == 0:
            continue

        print(f"\n  Model: {model}")
        for mode in ["cot", "structured", "minimal"]:
            mode_df = df[(df["model"] == model) & (df["prompt_mode"] == mode)].set_index("file")
            common = std_df.index.intersection(mode_df.index)
            if len(common) == 0:
                continue

            c_std = std_df.loc[common, "correct"]
            c_mode = mode_df.loc[common, "correct"]
            b = int((c_std & ~c_mode).sum())
            c = int((~c_std & c_mode).sum())
            n_disc = b + c

            if n_disc == 0:
                p_val = 1.0
            else:
                p_val = stats.binomtest(max(b, c), n_disc, 0.5).pvalue

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            label = PROMPT_LABELS[mode]
            mcnemar_rows.append({
                "model": model, "prompt_mode": mode, "b_std_only": b,
                "c_mode_only": c, "p_value": p_val,
            })
            print(f"  standard vs {label:<32s}  b={b} c={c}  p={p_val:.4f}  {sig}")

    mcnemar_df = pd.DataFrame(mcnemar_rows) if mcnemar_rows else pd.DataFrame()

    # 4. Inference time comparison
    print("\n" + "-" * 78)
    print("4. INFERENCE TIME COMPARISON")
    print("-" * 78)

    for model in models:
        print(f"\n  Model: {model}")
        print(f"  {'Mode':<15s} {'Mean(s)':>8s} {'Median(s)':>10s} {'Total(s)':>9s}")
        print("  " + "-" * 45)

        for mode in PROMPT_MODES:
            mdf = df[(df["model"] == model) & (df["prompt_mode"] == mode)]
            if len(mdf) == 0:
                continue
            times = mdf["time_seconds"].dropna()
            print(f"  {mode:<15s} {times.mean():8.2f} {times.median():10.2f} {times.sum():9.2f}")

    # 5. Recommendation
    print("\n" + "-" * 78)
    print("5. PROMPT SELECTION RECOMMENDATION")
    print("-" * 78)

    for model in models:
        mrows = accuracy_df[accuracy_df["model"] == model]
        if len(mrows) == 0:
            continue

        best = mrows.loc[mrows["accuracy"].idxmax()]
        std_row = mrows[mrows["prompt_mode"] == "standard"]
        std_acc = std_row["accuracy"].values[0] if len(std_row) > 0 else 0

        print(f"\n  Model: {model}")
        print(f"  Best prompt: {best['prompt_mode']} ({best['accuracy']:.1%})")
        print(f"  Standard:    {std_acc:.1%}")

        if best["accuracy"] > std_acc + 0.05:
            print(f"  -> {best['prompt_mode']} outperforms standard by {best['accuracy']-std_acc:+.1%}")
        elif best["accuracy"] < std_acc - 0.05:
            print(f"  -> Standard outperforms {best['prompt_mode']} by {std_acc-best['accuracy']:+.1%}")
        else:
            print(f"  -> No meaningful difference — standard is recommended (simpler, fewer tokens)")

    return accuracy_df, cm_df, mcnemar_df


def save_prompt_analysis(accuracy_df, cm_df, mcnemar_df, output_dir):
    """Save prompt analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc_path = output_dir / "prompt_accuracy.csv"
    accuracy_df.to_csv(acc_path, index=False)
    print(f"\n  Saved: {acc_path}")

    cm_path = output_dir / "prompt_confusion_matrices.csv"
    cm_df.to_csv(cm_path, index=False)
    print(f"  Saved: {cm_path}")

    if len(mcnemar_df) > 0:
        mcn_path = output_dir / "prompt_mcnemar.csv"
        mcnemar_df.to_csv(mcn_path, index=False)
        print(f"  Saved: {mcn_path}")


def main():
    parser = argparse.ArgumentParser(description="ELM Validation Prompt Engineering Experiments")
    parser.add_argument("--model", default="gpt-oss-20b",
                       help="Model to test prompts with (default: gpt-oss-20b)")
    parser.add_argument("--data-dir", type=Path,
                       default=Path(__file__).parent / "test_data",
                       help="Test data directory")
    parser.add_argument("--output-dir", type=Path,
                       default=Path(__file__).parent / "results" / "prompts",
                       help="Output directory for prompt experiment results")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Skip running experiments, just analyze existing results")
    parser.add_argument("--results-dir", type=Path, default=None,
                       help="Directory with existing results (for --analyze-only)")

    args = parser.parse_args()

    print("=" * 78)
    print("ELM VALIDATION — PROMPT ENGINEERING EXPERIMENTS")
    print("=" * 78)

    if not args.analyze_only:
        run_prompt_experiments(args.model, args.data_dir, args.output_dir)

    results_dir = args.results_dir or args.output_dir
    df = load_prompt_results(results_dir, model_id=args.model)

    if df is not None and len(df) > 0:
        accuracy_df, cm_df, mcnemar_df = analyze_prompt_results(df)
        analysis_dir = Path(__file__).parent / "results" / "analysis"
        save_prompt_analysis(accuracy_df, cm_df, mcnemar_df, analysis_dir)
    else:
        print("\nNo results to analyze. Run experiments first (without --analyze-only).")

    print("\nPrompt engineering analysis complete.")


if __name__ == "__main__":
    main()
