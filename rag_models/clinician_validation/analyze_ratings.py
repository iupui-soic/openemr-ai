"""
Statistical analysis of clinician ratings for Fareez RAG summarization experiment.

Computes:
1. Inter-rater reliability (Gwet's AC2, ordinal weights)
2. Intraclass Correlation Coefficient (ICC, two-way random)
3. Model comparison (Friedman test)
4. Pairwise comparison (Wilcoxon signed-rank, Bonferroni-corrected)
5. Effect size (Kendall's W)
6. Descriptive statistics (mean +/- SD per model per dimension)
7. Per-specialty subanalysis (RES, MSK)

Usage:
    python analyze_ratings.py
"""

import os
import csv
import numpy as np
import pandas as pd
from scipy import stats
from irrCAC.raw import CAC
import pingouin as pg
from itertools import combinations

RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKETS_DIR = os.path.join(RAG_ROOT, "rating_packets")
ANSWER_KEY = os.path.join(PACKETS_DIR, "ANSWER_KEY_DO_NOT_SHARE.csv")

DIMENSIONS = ["accuracy", "completeness", "organization", "conciseness",
              "clinical_utility", "overall_quality"]

RATER_FILES = {
    "rater_1": os.path.join(PACKETS_DIR, "ratings_rater_1.csv"),
    "rater_2": os.path.join(PACKETS_DIR, "ratings_rater_2.csv"),
    "rater_3": os.path.join(PACKETS_DIR, "ratings_rater_3.csv"),
}


def load_data():
    """Load all rater data and merge with answer key."""
    # Load answer key
    key = pd.read_csv(ANSWER_KEY)

    # Load each rater
    frames = []
    for rater_id, path in RATER_FILES.items():
        df = pd.read_csv(path)
        df["rater"] = rater_id
        frames.append(df)

    ratings = pd.concat(frames, ignore_index=True)

    # Merge with answer key
    merged = ratings.merge(key[["summary_id", "conversation", "model", "specialty", "condition"]],
                           on="summary_id", how="left")

    # Ensure numeric ratings
    for dim in DIMENSIONS:
        merged[dim] = pd.to_numeric(merged[dim], errors="coerce")

    return merged


def compute_gwet_ac2(merged):
    """Compute Gwet's AC2 for each dimension (ordinal weights)."""
    print("\n" + "=" * 70)
    print("1. INTER-RATER RELIABILITY: Gwet's AC2 (ordinal weights)")
    print("=" * 70)
    print(f"   Target: >= 0.667 (acceptable), >= 0.80 (good)\n")

    results = {}
    # Create a unique item key per (conversation, model) pair
    merged["item"] = merged["conversation"] + "_" + merged["model"]
    for dim in DIMENSIONS:
        # Build item x rater matrix (irrCAC expects subjects as rows, raters as columns)
        pivot = merged.pivot_table(index="item", columns="rater", values=dim, aggfunc="first")
        pivot = pivot.dropna()
        cac = CAC(pivot, weights="ordinal")
        gwet_result = cac.gwet()
        coeff = gwet_result["est"]["coefficient_value"]
        ci_low = gwet_result["est"]["confidence_interval"][0]
        ci_high = gwet_result["est"]["confidence_interval"][1]
        results[dim] = {"ac2": coeff, "ci_low": ci_low, "ci_high": ci_high}
        status = "GOOD" if coeff >= 0.80 else "ACCEPTABLE" if coeff >= 0.667 else "LOW"
        print(f"   {dim:<20s}  AC2 = {coeff:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]  [{status}]")

    avg_ac2 = np.mean([r["ac2"] for r in results.values()])
    print(f"\n   {'AVERAGE':<20s}  AC2 = {avg_ac2:.4f}")
    return results


def compute_icc(merged):
    """Compute Intraclass Correlation Coefficient (two-way random, consistency)."""
    print("\n" + "=" * 70)
    print("2. SUPPLEMENTARY RELIABILITY: ICC (two-way random, consistency)")
    print("=" * 70)

    results = {}
    if "item" not in merged.columns:
        merged["item"] = merged["conversation"] + "_" + merged["model"]
    for dim in DIMENSIONS:
        subset = merged[["item", "rater", dim]].dropna()
        try:
            icc_result = pg.intraclass_corr(
                data=subset, targets="item", raters="rater", ratings=dim
            )
            # ICC3k = two-way mixed, consistency, average measures
            icc_row = icc_result[icc_result["Type"] == "ICC3k"]
            if len(icc_row) > 0:
                icc_val = icc_row["ICC"].values[0]
                ci_low = icc_row["CI95%"].values[0][0]
                ci_high = icc_row["CI95%"].values[0][1]
                results[dim] = {"icc": icc_val, "ci_low": ci_low, "ci_high": ci_high}
                print(f"   {dim:<20s}  ICC = {icc_val:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]")
            else:
                print(f"   {dim:<20s}  ICC = N/A")
        except Exception as e:
            print(f"   {dim:<20s}  ICC = ERROR ({e})")

    return results


def compute_descriptive_stats(merged):
    """Compute mean +/- SD per model per dimension."""
    print("\n" + "=" * 70)
    print("3. DESCRIPTIVE STATISTICS: Mean +/- SD per Model per Dimension")
    print("=" * 70)

    # Group by model
    model_stats = {}
    for model in sorted(merged["model"].unique()):
        model_data = merged[merged["model"] == model]
        stats_row = {}
        for dim in DIMENSIONS:
            vals = model_data[dim].dropna()
            stats_row[dim] = {"mean": vals.mean(), "std": vals.std(), "n": len(vals)}
        model_stats[model] = stats_row

    # Print table
    header = f"{'Model':<20s}"
    for dim in DIMENSIONS:
        header += f"  {dim[:12]:>12s}"
    print(f"\n   {header}")
    print("   " + "-" * len(header))

    for model in sorted(model_stats.keys()):
        row = f"   {model:<20s}"
        for dim in DIMENSIONS:
            m = model_stats[model][dim]["mean"]
            s = model_stats[model][dim]["std"]
            row += f"  {m:>5.2f}+/-{s:.2f}"
        print(row)

    # Overall averages across all dimensions
    print(f"\n   {'COMPOSITE':<20s}", end="")
    for model in sorted(model_stats.keys()):
        all_means = [model_stats[model][dim]["mean"] for dim in DIMENSIONS]
        composite = np.mean(all_means)
        print(f"  {model}: {composite:.2f}", end="")
    print()

    return model_stats


def compute_friedman_test(merged):
    """Friedman test for overall model differences per dimension."""
    print("\n" + "=" * 70)
    print("4. MODEL COMPARISON: Friedman Test (non-parametric)")
    print("=" * 70)
    print(f"   H0: No difference between models across conversations\n")

    results = {}
    models = sorted(merged["model"].unique())

    for dim in DIMENSIONS:
        # Need each conversation rated by all raters for all models
        # Average across raters first, then compare models
        avg_ratings = merged.groupby(["conversation", "model"])[dim].mean().reset_index()
        pivot = avg_ratings.pivot(index="conversation", columns="model", values=dim).dropna()

        if len(pivot) < 5:
            print(f"   {dim:<20s}  Insufficient data (n={len(pivot)})")
            continue

        groups = [pivot[m].values for m in models if m in pivot.columns]
        stat, p = stats.friedmanchisquare(*groups)

        # Kendall's W = chi2 / (n * (k-1))
        n = len(pivot)
        k = len(groups)
        w = stat / (n * (k - 1))

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        results[dim] = {"chi2": stat, "p": p, "w": w, "n": n, "k": k}
        print(f"   {dim:<20s}  chi2={stat:7.3f}  p={p:.6f}  W={w:.4f}  n={n}  {sig}")

    return results


def compute_pairwise_wilcoxon(merged):
    """Pairwise Wilcoxon signed-rank tests with Bonferroni correction."""
    print("\n" + "=" * 70)
    print("5. PAIRWISE COMPARISON: Wilcoxon Signed-Rank (Bonferroni)")
    print("=" * 70)

    models = sorted(merged["model"].unique())
    pairs = list(combinations(models, 2))
    n_comparisons = len(pairs)
    alpha_corrected = 0.05 / n_comparisons
    print(f"   {n_comparisons} comparisons, Bonferroni alpha = {alpha_corrected:.4f}\n")

    results = {}
    for dim in DIMENSIONS:
        print(f"   --- {dim} ---")
        avg_ratings = merged.groupby(["conversation", "model"])[dim].mean().reset_index()
        pivot = avg_ratings.pivot(index="conversation", columns="model", values=dim).dropna()

        dim_results = {}
        for m1, m2 in pairs:
            if m1 not in pivot.columns or m2 not in pivot.columns:
                continue
            v1 = pivot[m1].values
            v2 = pivot[m2].values

            try:
                stat, p = stats.wilcoxon(v1, v2, alternative="two-sided")
                sig = "***" if p < 0.001 else "**" if p < alpha_corrected else "*" if p < 0.05 else "ns"
                diff = np.mean(v1) - np.mean(v2)
                # Effect size r = Z / sqrt(N)
                n = len(v1)
                z = stats.norm.ppf(p / 2)
                r = abs(z) / np.sqrt(n)
                dim_results[(m1, m2)] = {"stat": stat, "p": p, "diff": diff, "r": r}
                print(f"   {m1:>18s} vs {m2:<18s}  W={stat:7.1f}  p={p:.6f}  diff={diff:+.3f}  r={r:.3f}  {sig}")
            except Exception as e:
                print(f"   {m1:>18s} vs {m2:<18s}  ERROR: {e}")

        results[dim] = dim_results
        print()

    return results


def compute_specialty_subanalysis(merged):
    """Per-specialty analysis for RES and MSK (largest subgroups)."""
    print("\n" + "=" * 70)
    print("6. SPECIALTY SUBANALYSIS: Friedman Test (RES, MSK)")
    print("=" * 70)

    models = sorted(merged["model"].unique())

    for specialty in ["RES", "MSK"]:
        spec_data = merged[merged["specialty"] == specialty]
        n_convos = spec_data["conversation"].nunique()
        print(f"\n   --- {specialty} ({n_convos} conversations) ---")

        for dim in DIMENSIONS:
            avg_ratings = spec_data.groupby(["conversation", "model"])[dim].mean().reset_index()
            pivot = avg_ratings.pivot(index="conversation", columns="model", values=dim).dropna()

            if len(pivot) < 5:
                print(f"   {dim:<20s}  Insufficient data (n={len(pivot)})")
                continue

            groups = [pivot[m].values for m in models if m in pivot.columns]
            stat, p = stats.friedmanchisquare(*groups)
            n = len(pivot)
            k = len(groups)
            w = stat / (n * (k - 1))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"   {dim:<20s}  chi2={stat:7.3f}  p={p:.6f}  W={w:.4f}  {sig}")


def generate_summary_table(merged, model_stats, ac2_results, friedman_results):
    """Generate a publication-ready summary table."""
    print("\n" + "=" * 70)
    print("7. PUBLICATION TABLE: Model Comparison Summary")
    print("=" * 70)

    models = sorted(merged["model"].unique())

    print(f"\n   {'Model':<20s}", end="")
    for dim in DIMENSIONS:
        short = dim[:4].upper() if len(dim) > 12 else dim.upper()
        print(f"  {short:>10s}", end="")
    print(f"  {'COMPOSITE':>10s}")

    print("   " + "-" * (20 + 12 * (len(DIMENSIONS) + 1)))

    for model in models:
        print(f"   {model:<20s}", end="")
        dim_means = []
        for dim in DIMENSIONS:
            m = model_stats[model][dim]["mean"]
            s = model_stats[model][dim]["std"]
            dim_means.append(m)
            print(f"  {m:5.2f}({s:.2f})", end="")
        composite = np.mean(dim_means)
        print(f"  {composite:10.2f}")

    # Friedman p-values row
    print(f"\n   {'Friedman p':<20s}", end="")
    for dim in DIMENSIONS:
        if dim in friedman_results:
            p = friedman_results[dim]["p"]
            print(f"  {p:10.4f}", end="")
        else:
            print(f"  {'N/A':>10s}", end="")
    print()

    # Gwet AC2 row
    print(f"   {'Gwet AC2':<20s}", end="")
    for dim in DIMENSIONS:
        if dim in ac2_results:
            a = ac2_results[dim]["ac2"]
            print(f"  {a:10.4f}", end="")
        else:
            print(f"  {'N/A':>10s}", end="")
    print()


def save_results(merged, model_stats, ac2_results, icc_results, friedman_results):
    """Save analysis results to CSV."""
    output_dir = os.path.join(RAG_ROOT, "results", "fareez")
    os.makedirs(output_dir, exist_ok=True)

    # 1. IRR results
    irr_rows = []
    for dim in DIMENSIONS:
        row = {"dimension": dim}
        ac2_data = ac2_results.get(dim, {})
        row["gwet_ac2"] = ac2_data.get("ac2", None)
        row["ac2_ci_low"] = ac2_data.get("ci_low", None)
        row["ac2_ci_high"] = ac2_data.get("ci_high", None)
        if dim in icc_results:
            row["icc"] = icc_results[dim]["icc"]
            row["icc_ci_low"] = icc_results[dim]["ci_low"]
            row["icc_ci_high"] = icc_results[dim]["ci_high"]
        if dim in friedman_results:
            row["friedman_chi2"] = friedman_results[dim]["chi2"]
            row["friedman_p"] = friedman_results[dim]["p"]
            row["kendall_w"] = friedman_results[dim]["w"]
        irr_rows.append(row)

    irr_df = pd.DataFrame(irr_rows)
    irr_path = os.path.join(output_dir, "fareez_irr_analysis.csv")
    irr_df.to_csv(irr_path, index=False)
    print(f"\n   Saved: {irr_path}")

    # 2. Model comparison
    rows = []
    for model in sorted(model_stats.keys()):
        row = {"model": model}
        for dim in DIMENSIONS:
            row[f"{dim}_mean"] = model_stats[model][dim]["mean"]
            row[f"{dim}_std"] = model_stats[model][dim]["std"]
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_path = os.path.join(output_dir, "fareez_model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"   Saved: {comp_path}")

    # 3. Full merged ratings
    ratings_path = os.path.join(output_dir, "fareez_clinician_ratings.csv")
    merged.to_csv(ratings_path, index=False)
    print(f"   Saved: {ratings_path}")


def main():
    print("=" * 70)
    print("CLINICIAN RATING ANALYSIS — Fareez RAG Summarization")
    print("=" * 70)

    merged = load_data()
    print(f"\nLoaded {len(merged)} ratings from {merged['rater'].nunique()} raters")
    print(f"Models: {sorted(merged['model'].unique())}")
    print(f"Conversations: {merged['conversation'].nunique()}")
    print(f"Specialties: {dict(merged.groupby('specialty')['conversation'].nunique())}")

    # Check for missing data
    for dim in DIMENSIONS:
        missing = merged[dim].isna().sum()
        if missing > 0:
            print(f"  Warning: {missing} missing values in {dim}")

    ac2_results = compute_gwet_ac2(merged)
    icc_results = compute_icc(merged)
    model_stats = compute_descriptive_stats(merged)
    friedman_results = compute_friedman_test(merged)
    pairwise_results = compute_pairwise_wilcoxon(merged)
    compute_specialty_subanalysis(merged)
    generate_summary_table(merged, model_stats, ac2_results, friedman_results)
    save_results(merged, model_stats, ac2_results, icc_results, friedman_results)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
