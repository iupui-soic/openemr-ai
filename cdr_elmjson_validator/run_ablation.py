#!/usr/bin/env python3
"""
Ablation Study for ELM Validation Pipeline.

Runs a model across all 4 ablation conditions and compares results:
  full              - simplified ELM + CPG (default pipeline)
  no_cpg            - simplified ELM only, no CPG reference
  no_simplify       - raw truncated JSON + CPG
  no_cpg_no_simplify - raw truncated JSON only

Analyzes the contribution of each pipeline component (CPG reference,
ELM simplification) to validation accuracy.

Usage:
    python run_ablation.py --model gpt-oss-20b
    python run_ablation.py --model qwen3-32b --output-dir results/ablation
    python run_ablation.py --analyze-only --results-dir results/ablation
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

ABLATION_MODES = ["full", "no_cpg", "no_simplify", "no_cpg_no_simplify"]

ABLATION_LABELS = {
    "full": "Simplified ELM + CPG",
    "no_cpg": "Simplified ELM only",
    "no_simplify": "Raw JSON + CPG",
    "no_cpg_no_simplify": "Raw JSON only",
}


def run_ablation_experiments(model_id, data_dir, output_dir):
    """Run all 4 ablation conditions for a model."""
    from run_validation import run_validation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode in ABLATION_MODES:
        output_file = output_dir / f"results-{model_id}-{mode}.csv"
        if output_file.exists():
            print(f"\n  Skipping {mode} — results already exist at {output_file}")
            continue

        print(f"\n{'='*70}")
        print(f"  ABLATION: {mode} ({ABLATION_LABELS[mode]})")
        print(f"{'='*70}")
        run_validation(model_id, data_dir, output_file,
                       ablation_mode=mode, prompt_mode="standard")

    print(f"\nAll ablation experiments complete. Results in {output_dir}/")


def load_ablation_results(results_dir, model_id=None):
    """Load ablation result CSVs into a single DataFrame."""
    results_dir = Path(results_dir)
    frames = []

    for mode in ABLATION_MODES:
        pattern = f"results-*-{mode}.csv"
        files = list(results_dir.glob(pattern))
        if model_id:
            files = [f for f in files if model_id in f.name]

        for fpath in files:
            df = pd.read_csv(fpath)
            # Extract model and mode from filename
            stem = fpath.stem  # e.g., results-gpt-oss-20b-no_cpg
            parts = stem.replace("results-", "")
            df["ablation_mode"] = mode
            for col in ["valid", "correct", "expected_valid"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.lower() == "true"
            frames.append(df)

    if not frames:
        print(f"No ablation results found in {results_dir}")
        return None

    return pd.concat(frames, ignore_index=True)


def analyze_ablation_results(df):
    """Analyze and compare ablation conditions."""
    print("\n" + "=" * 78)
    print("ABLATION STUDY — COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 78)

    models = sorted(df["model"].unique())
    print(f"\nModels: {models}")

    # 1. Accuracy per condition
    print("\n" + "-" * 78)
    print("1. ACCURACY PER ABLATION CONDITION (Wilson 95% CI)")
    print("-" * 78)

    rows = []
    for model in models:
        print(f"\n  Model: {model}")
        print(f"  {'Condition':<28s} {'Correct':>8s} {'Acc':>7s} {'95% CI':>18s}")
        print("  " + "-" * 65)

        for mode in ABLATION_MODES:
            mdf = df[(df["model"] == model) & (df["ablation_mode"] == mode)]
            if len(mdf) == 0:
                continue
            n = len(mdf)
            k = int(mdf["correct"].sum())
            acc = k / n
            ci_lo, ci_hi = proportion_confint(k, n, alpha=0.05, method="wilson")
            label = ABLATION_LABELS[mode]
            rows.append({
                "model": model, "ablation_mode": mode, "label": label,
                "correct": k, "total": n, "accuracy": acc,
                "ci_low": ci_lo, "ci_high": ci_hi,
            })
            print(f"  {label:<28s} {k:>3d}/{n:<3d}  {acc:5.1%}  [{ci_lo:.3f}, {ci_hi:.3f}]")

    accuracy_df = pd.DataFrame(rows)

    # 2. Component contributions (delta analysis)
    print("\n" + "-" * 78)
    print("2. COMPONENT CONTRIBUTION (delta from baseline)")
    print("-" * 78)

    for model in models:
        mrows = accuracy_df[accuracy_df["model"] == model]
        if len(mrows) < 2:
            continue

        accs = {r["ablation_mode"]: r["accuracy"] for _, r in mrows.iterrows()}
        full = accs.get("full", None)

        if full is None:
            continue

        print(f"\n  Model: {model} (full pipeline accuracy: {full:.1%})")
        print(f"  {'Component Removed':<30s} {'Accuracy':>8s} {'Delta':>8s} {'Impact'}")
        print("  " + "-" * 60)

        deltas = {}
        for mode in ["no_cpg", "no_simplify", "no_cpg_no_simplify"]:
            if mode not in accs:
                continue
            delta = accs[mode] - full
            impact = "HURTS" if delta < -0.05 else "HELPS" if delta > 0.05 else "NEUTRAL"
            removed = {
                "no_cpg": "CPG reference",
                "no_simplify": "ELM simplification",
                "no_cpg_no_simplify": "Both (CPG + simplification)",
            }[mode]
            deltas[mode] = delta
            print(f"  {removed:<30s} {accs[mode]:5.1%}   {delta:+5.1%}   {impact}")

        # Isolate individual contributions
        cpg_effect = accs.get("no_cpg_no_simplify", 0) - accs.get("no_simplify", 0) if "no_simplify" in accs and "no_cpg_no_simplify" in accs else None
        simplify_effect = accs.get("no_cpg_no_simplify", 0) - accs.get("no_cpg", 0) if "no_cpg" in accs and "no_cpg_no_simplify" in accs else None

        if cpg_effect is not None and simplify_effect is not None:
            print(f"\n  Isolated CPG contribution:          {-cpg_effect:+5.1%}")
            print(f"  Isolated simplification contribution: {-simplify_effect:+5.1%}")
            interaction = full - accs.get("no_cpg_no_simplify", 0) - (-cpg_effect) - (-simplify_effect)
            print(f"  Interaction effect:                   {interaction:+5.1%}")

    # 3. McNemar tests between conditions
    print("\n" + "-" * 78)
    print("3. McNEMAR'S EXACT TESTS (full vs each ablation condition)")
    print("-" * 78)

    mcnemar_rows = []
    for model in models:
        full_df = df[(df["model"] == model) & (df["ablation_mode"] == "full")].set_index("file")
        if len(full_df) == 0:
            continue

        print(f"\n  Model: {model}")
        for mode in ["no_cpg", "no_simplify", "no_cpg_no_simplify"]:
            mode_df = df[(df["model"] == model) & (df["ablation_mode"] == mode)].set_index("file")
            common = full_df.index.intersection(mode_df.index)
            if len(common) == 0:
                continue

            c_full = full_df.loc[common, "correct"]
            c_mode = mode_df.loc[common, "correct"]
            b = int((c_full & ~c_mode).sum())  # full correct, mode wrong
            c = int((~c_full & c_mode).sum())  # full wrong, mode correct
            n_disc = b + c

            if n_disc == 0:
                p_val = 1.0
            else:
                p_val = stats.binomtest(max(b, c), n_disc, 0.5).pvalue

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            label = ABLATION_LABELS[mode]
            mcnemar_rows.append({
                "model": model, "condition": mode, "b_full_only": b,
                "c_mode_only": c, "p_value": p_val,
            })
            print(f"  full vs {label:<28s}  b={b} c={c}  p={p_val:.4f}  {sig}")

    mcnemar_df = pd.DataFrame(mcnemar_rows) if mcnemar_rows else pd.DataFrame()

    # 4. Per-case analysis: which cases change across conditions?
    print("\n" + "-" * 78)
    print("4. CASES THAT CHANGE ACROSS CONDITIONS")
    print("-" * 78)

    for model in models:
        mdf = df[df["model"] == model]
        full_correct = set(mdf[(mdf["ablation_mode"] == "full") & (mdf["correct"])]["file"])

        print(f"\n  Model: {model}")
        for mode in ["no_cpg", "no_simplify", "no_cpg_no_simplify"]:
            mode_correct = set(mdf[(mdf["ablation_mode"] == mode) & (mdf["correct"])]["file"])
            gained = mode_correct - full_correct
            lost = full_correct - mode_correct
            if gained or lost:
                print(f"\n  {ABLATION_LABELS[mode]}:")
                for f in sorted(lost):
                    print(f"    LOST: {f}")
                for f in sorted(gained):
                    print(f"    GAINED: {f}")
            else:
                print(f"  {ABLATION_LABELS[mode]}: no change")

    return accuracy_df, mcnemar_df


def save_ablation_analysis(accuracy_df, mcnemar_df, output_dir):
    """Save ablation analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc_path = output_dir / "ablation_accuracy.csv"
    accuracy_df.to_csv(acc_path, index=False)
    print(f"\n  Saved: {acc_path}")

    if len(mcnemar_df) > 0:
        mcn_path = output_dir / "ablation_mcnemar.csv"
        mcnemar_df.to_csv(mcn_path, index=False)
        print(f"  Saved: {mcn_path}")


def main():
    parser = argparse.ArgumentParser(description="ELM Validation Ablation Study")
    parser.add_argument("--model", default="gpt-oss-20b",
                       help="Model to run ablation with (default: gpt-oss-20b)")
    parser.add_argument("--data-dir", type=Path,
                       default=Path(__file__).parent / "test_data",
                       help="Test data directory")
    parser.add_argument("--output-dir", type=Path,
                       default=Path(__file__).parent / "results" / "ablation",
                       help="Output directory for ablation results")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Skip running experiments, just analyze existing results")
    parser.add_argument("--results-dir", type=Path, default=None,
                       help="Directory with existing results (for --analyze-only)")

    args = parser.parse_args()

    print("=" * 78)
    print("ELM VALIDATION — ABLATION STUDY")
    print("=" * 78)

    if not args.analyze_only:
        run_ablation_experiments(args.model, args.data_dir, args.output_dir)

    results_dir = args.results_dir or args.output_dir
    df = load_ablation_results(results_dir, model_id=args.model)

    if df is not None and len(df) > 0:
        accuracy_df, mcnemar_df = analyze_ablation_results(df)
        analysis_dir = Path(__file__).parent / "results" / "analysis"
        save_ablation_analysis(accuracy_df, mcnemar_df, analysis_dir)
    else:
        print("\nNo results to analyze. Run experiments first (without --analyze-only).")

    print("\nAblation study complete.")


if __name__ == "__main__":
    main()
