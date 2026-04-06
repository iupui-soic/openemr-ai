"""
Statistical analysis of ELM validation results across 12 LLMs.

Computes:
1. Per-model accuracy with Wilson confidence intervals
2. Confusion matrices (TP/FP/TN/FN) with sensitivity, specificity, PPV, NPV
3. McNemar's exact tests for pairwise model comparison
4. Post-hoc power analysis for n=22 binary classification
5. Model × test-case heatmap (saved as PNG)
6. Error disaggregation by type (age threshold, time interval, value, missing logic, crash)
7. Per-case difficulty analysis (which test cases are hardest)

Usage:
    python analyze_elm_results.py
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.power import GofChisquarePower
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis")

# Model display order: by parameter count descending
MODEL_ORDER = [
    "gpt-oss-120b", "llama-3.3-70b", "qwen3-32b", "gpt-oss-20b",
    "medgemma-4b", "medgemma-1.5-4b", "gemma-3-4b", "phi-3-mini",
    "qwen-2.5-3b", "llama-3.2-3b", "qwen-2.5-1.5b", "llama-3.2-1b",
]

# Model tiers for analysis
TIER_LABELS = {
    "gpt-oss-120b": "Frontier", "llama-3.3-70b": "Frontier",
    "qwen3-32b": "Frontier", "gpt-oss-20b": "Frontier",
    "medgemma-4b": "Mid-range", "medgemma-1.5-4b": "Mid-range",
    "gemma-3-4b": "Mid-range", "phi-3-mini": "Mid-range",
    "qwen-2.5-3b": "Failed", "llama-3.2-3b": "Failed",
    "qwen-2.5-1.5b": "Failed", "llama-3.2-1b": "Failed",
}

# Approximate parameter counts (billions) for display
MODEL_PARAMS = {
    "gpt-oss-120b": 120, "llama-3.3-70b": 70, "qwen3-32b": 32,
    "gpt-oss-20b": 20, "medgemma-4b": 4, "medgemma-1.5-4b": 4,
    "gemma-3-4b": 4, "phi-3-mini": 3.8, "qwen-2.5-3b": 3,
    "llama-3.2-3b": 3, "qwen-2.5-1.5b": 1.5, "llama-3.2-1b": 1,
}

# Error type patterns for disaggregation
ERROR_PATTERNS = [
    ("age_threshold", re.compile(r"age\s+(threshold|range|lower|upper|bound)", re.I)),
    ("time_interval", re.compile(r"(time\s+interval|lookback|period)\s+(mismatch|different)", re.I)),
    ("time_interval", re.compile(r"lookback\s+(period|mismatch)", re.I)),
    ("value_mismatch", re.compile(r"(value|threshold|BMI|HbA1c|LDL|risk|glucose)\s+(mismatch|instead|uses)", re.I)),
    ("missing_logic", re.compile(r"(does not|doesn't)\s+(explicitly|mention|state|include)", re.I)),
    ("missing_logic", re.compile(r"(not explicitly|missing|absent|omit)", re.I)),
    ("modal_crash", re.compile(r"Modal exit code", re.I)),
]


def load_all_results():
    """Load all result CSVs into a single DataFrame."""
    frames = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.startswith("results-") or not fname.endswith(".csv"):
            continue
        path = os.path.join(RESULTS_DIR, fname)
        df = pd.read_csv(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Normalize boolean columns
    for col in ["valid", "correct", "expected_valid", "has_cpg", "has_ground_truth"]:
        if col in combined.columns:
            combined[col] = combined[col].astype(str).str.strip().str.lower() == "true"

    # Clean up model names
    combined["model"] = combined["model"].str.strip()

    # Short filename for display
    combined["case"] = combined["file"].str.replace(".json", "", regex=False)

    return combined


def compute_accuracy_wilson_ci(df):
    """Compute per-model accuracy with Wilson score confidence intervals."""
    print("\n" + "=" * 78)
    print("1. PER-MODEL ACCURACY WITH WILSON 95% CONFIDENCE INTERVALS")
    print("=" * 78)

    rows = []
    n_total = df.groupby("model")["correct"].count().iloc[0]  # should be 22

    for model in MODEL_ORDER:
        mdf = df[df["model"] == model]
        n = len(mdf)
        k = mdf["correct"].sum()
        acc = k / n
        ci_low, ci_high = proportion_confint(k, n, alpha=0.05, method="wilson")
        tier = TIER_LABELS.get(model, "")
        params = MODEL_PARAMS.get(model, "?")
        rows.append({
            "model": model, "params_B": params, "tier": tier,
            "correct": int(k), "total": n, "accuracy": acc,
            "ci_low": ci_low, "ci_high": ci_high,
        })
        print(f"   {model:<20s} {params:>5.1f}B  {int(k):>2d}/{n}  "
              f"{acc:5.1%}  95% CI [{ci_low:.3f}, {ci_high:.3f}]  ({tier})")

    # Base rate for context
    n_valid = df[df["model"] == MODEL_ORDER[0]]["expected_valid"].sum()
    n_invalid = n_total - n_valid
    base_rate = max(n_valid, n_invalid) / n_total
    print(f"\n   Base rate (majority class): {base_rate:.1%} "
          f"({int(n_valid)} valid, {int(n_invalid)} invalid out of {n_total})")

    return pd.DataFrame(rows)


def build_confusion_matrices(df):
    """Build per-model confusion matrices with derived metrics."""
    print("\n" + "=" * 78)
    print("2. CONFUSION MATRICES (TP/FP/TN/FN) AND DERIVED METRICS")
    print("=" * 78)
    print(f"   Positive class = 'valid ELM', Negative class = 'invalid ELM'\n")

    header = (f"   {'Model':<20s} {'TP':>3s} {'FP':>3s} {'TN':>3s} {'FN':>3s}  "
              f"{'Sens':>5s} {'Spec':>5s} {'PPV':>5s} {'NPV':>5s} {'F1':>5s}")
    print(header)
    print("   " + "-" * (len(header) - 3))

    rows = []
    for model in MODEL_ORDER:
        mdf = df[df["model"] == model]
        tp = int(((mdf["valid"]) & (mdf["expected_valid"])).sum())
        fp = int(((mdf["valid"]) & (~mdf["expected_valid"])).sum())
        tn = int(((~mdf["valid"]) & (~mdf["expected_valid"])).sum())
        fn = int(((~mdf["valid"]) & (mdf["expected_valid"])).sum())

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        rows.append({
            "model": model, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "sensitivity": sens, "specificity": spec, "PPV": ppv, "NPV": npv, "F1": f1,
        })
        print(f"   {model:<20s} {tp:3d} {fp:3d} {tn:3d} {fn:3d}  "
              f"{sens:5.2f} {spec:5.2f} {ppv:5.2f} {npv:5.2f} {f1:5.2f}")

    return pd.DataFrame(rows)


def compute_mcnemar_tests(df):
    """Pairwise McNemar's exact tests between functioning models."""
    print("\n" + "=" * 78)
    print("3. McNEMAR'S EXACT TESTS (pairwise model comparison)")
    print("=" * 78)

    # Only compare models that actually ran (exclude crashed small models)
    functioning = [m for m in MODEL_ORDER if TIER_LABELS.get(m) != "Failed"]
    pairs = list(combinations(functioning, 2))
    n_comparisons = len(pairs)
    alpha_corrected = 0.05 / n_comparisons
    print(f"   {len(functioning)} functioning models, {n_comparisons} pairs, "
          f"Bonferroni alpha = {alpha_corrected:.4f}\n")

    results = []
    for m1, m2 in pairs:
        df1 = df[df["model"] == m1].set_index("file")["correct"]
        df2 = df[df["model"] == m2].set_index("file")["correct"]
        common = df1.index.intersection(df2.index)
        c1 = df1.loc[common]
        c2 = df2.loc[common]

        # McNemar contingency: b = m1 correct & m2 wrong, c = m1 wrong & m2 correct
        b = int((c1 & ~c2).sum())
        c = int((~c1 & c2).sum())
        n_disc = b + c

        if n_disc == 0:
            p_val = 1.0
        else:
            # Exact McNemar (mid-p): use binomial test
            p_val = stats.binomtest(max(b, c), n_disc, 0.5).pvalue

        sig = "***" if p_val < 0.001 else "**" if p_val < alpha_corrected else "*" if p_val < 0.05 else "ns"
        results.append({
            "model_1": m1, "model_2": m2,
            "b_m1_only": b, "c_m2_only": c, "n_discordant": n_disc,
            "p_value": p_val, "significant": sig,
        })
        print(f"   {m1:>20s} vs {m2:<20s}  b={b} c={c}  p={p_val:.4f}  {sig}")

    return pd.DataFrame(results)


def compute_power_analysis(df):
    """Post-hoc power analysis for the n=22 binary classification setup."""
    print("\n" + "=" * 78)
    print("4. POST-HOC POWER ANALYSIS (n=22 test cases)")
    print("=" * 78)

    n = 22
    alpha = 0.05
    power_analysis = GofChisquarePower()

    # What effect size can we detect at 80% power?
    # For chi-square GOF test: effect_size = w (Cohen's w)
    # df for 2x2 = 1
    w_detectable = power_analysis.solve_power(
        effect_size=None, nobs=n, alpha=alpha, power=0.80, n_bins=2
    )
    print(f"   Detectable effect size (w) at 80% power, alpha={alpha}: {w_detectable:.3f}")
    print(f"   Interpretation: w >= 0.1 small, >= 0.3 medium, >= 0.5 large")

    # Power at various effect sizes
    print(f"\n   {'Effect size (w)':<20s} {'Power':>8s}")
    print("   " + "-" * 30)
    power_rows = []
    for w in [0.1, 0.2, 0.3, 0.5, 0.8]:
        pwr = power_analysis.solve_power(
            effect_size=w, nobs=n, alpha=alpha, power=None, n_bins=2
        )
        power_rows.append({"effect_size_w": w, "power": pwr, "n": n, "alpha": alpha})
        label = {"0.1": "small", "0.2": "", "0.3": "medium", "0.5": "large", "0.8": "v.large"}.get(str(w), "")
        print(f"   w = {w:<5.1f} ({label:>8s})    {pwr:8.3f}")

    # What n would we need for 80% power at medium effect?
    n_needed = power_analysis.solve_power(
        effect_size=0.3, nobs=None, alpha=alpha, power=0.80, n_bins=2
    )
    print(f"\n   Sample size needed for 80% power at w=0.3 (medium): n={int(np.ceil(n_needed))}")

    # Actual observed effect: compare top model vs base rate
    n_valid = 15
    base_rate = n_valid / n  # 0.682
    # Cohen's w for the observed accuracy of top model (100%) vs base rate
    # w = sqrt(sum((O_i - E_i)^2 / E_i))
    top_acc = df[df["model"] == "gpt-oss-20b"]["correct"].mean()
    observed_w = np.sqrt(
        ((top_acc - base_rate) ** 2) / base_rate +
        (((1 - top_acc) - (1 - base_rate)) ** 2) / (1 - base_rate)
    )
    observed_power = power_analysis.solve_power(
        effect_size=observed_w, nobs=n, alpha=alpha, power=None, n_bins=2
    )
    print(f"\n   Observed: top model (gpt-oss-20b) accuracy = {top_acc:.1%}")
    print(f"   Observed Cohen's w vs base rate ({base_rate:.1%}): {observed_w:.3f}")
    print(f"   Observed power at this effect: {observed_power:.3f}")

    return pd.DataFrame(power_rows)


def generate_heatmap(df):
    """Generate model × test-case heatmap (correct/incorrect/crash)."""
    print("\n" + "=" * 78)
    print("5. GENERATING HEATMAP: Model x Test Case")
    print("=" * 78)

    # Build pivot: rows=cases, columns=models, values=correctness
    # 1 = correct, 0 = incorrect, -1 = crash
    pivot_data = []
    for _, row in df.iterrows():
        val = 1 if row["correct"] else 0
        if "Modal exit code" in str(row.get("errors", "")):
            val = -1
        pivot_data.append({
            "case": row["case"], "model": row["model"], "result": val,
        })

    pivot_df = pd.DataFrame(pivot_data)
    pivot = pivot_df.pivot(index="case", columns="model", values="result")

    # Sort cases by difficulty (number of models that got it right)
    case_difficulty = pivot.apply(lambda x: (x == 1).sum(), axis=1)
    pivot = pivot.loc[case_difficulty.sort_values().index]

    # Reorder columns by MODEL_ORDER
    cols = [m for m in MODEL_ORDER if m in pivot.columns]
    pivot = pivot[cols]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Custom colormap: -1=gray (crash), 0=red (wrong), 1=green (correct)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#999999", "#e74c3c", "#2ecc71"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    sns.heatmap(
        pivot, cmap=cmap, norm=norm, linewidths=0.5, linecolor="white",
        cbar_kws={"ticks": [-1, 0, 1], "label": ""},
        ax=ax, square=False,
    )

    # Fix colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(["Crash", "Incorrect", "Correct"])

    ax.set_title("ELM Validation Results: Model × Test Case", fontsize=14, pad=12)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Test Case (sorted by difficulty)", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    heatmap_path = os.path.join(OUTPUT_DIR, "elm_validation_heatmap.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {heatmap_path}")

    return pivot


def disaggregate_errors(df):
    """Classify errors by type from the error text."""
    print("\n" + "=" * 78)
    print("6. ERROR DISAGGREGATION BY TYPE")
    print("=" * 78)

    # Only look at incorrect predictions from functioning models
    functioning = [m for m in MODEL_ORDER if TIER_LABELS.get(m) != "Failed"]
    errors_df = df[(df["model"].isin(functioning)) & (~df["correct"])].copy()

    def classify_error(error_text):
        if pd.isna(error_text) or error_text == "":
            return ["no_error_text"]
        error_text = str(error_text)
        types = set()
        for label, pattern in ERROR_PATTERNS:
            if pattern.search(error_text):
                types.add(label)
        if not types:
            types.add("other")
        return sorted(types)

    errors_df["error_types"] = errors_df["errors"].apply(classify_error)

    # Count by error type
    type_counts = {}
    for _, row in errors_df.iterrows():
        for et in row["error_types"]:
            type_counts[et] = type_counts.get(et, 0) + 1

    print(f"\n   Total incorrect predictions (functioning models): {len(errors_df)}")
    print(f"\n   {'Error Type':<25s} {'Count':>6s} {'%':>8s}")
    print("   " + "-" * 42)
    total = sum(type_counts.values())
    type_rows = []
    for et, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        type_rows.append({"error_type": et, "count": count, "pct": pct})
        print(f"   {et:<25s} {count:>6d} {pct:>7.1f}%")

    # Per-model error breakdown
    print(f"\n   Per-model error type breakdown:")
    print(f"   {'Model':<20s} {'FP':>3s} {'FN':>3s} {'Error Types'}")
    print("   " + "-" * 70)
    for model in functioning:
        mdf = errors_df[errors_df["model"] == model]
        if len(mdf) == 0:
            continue
        fp = int(((mdf["valid"]) & (~mdf["expected_valid"])).sum())
        fn = int(((~mdf["valid"]) & (mdf["expected_valid"])).sum())
        all_types = set()
        for types_list in mdf["error_types"]:
            all_types.update(types_list)
        print(f"   {model:<20s} {fp:>3d} {fn:>3d} {', '.join(sorted(all_types))}")

    # Per-case error analysis: which cases are misclassified most?
    print(f"\n   Per-case misclassification (functioning models only):")
    case_errors = errors_df.groupby("case").agg(
        n_wrong=("correct", "count"),
        models_wrong=("model", lambda x: ", ".join(sorted(x))),
    ).sort_values("n_wrong", ascending=False)

    print(f"   {'Case':<50s} {'#Wrong':>6s} {'Models'}")
    print("   " + "-" * 90)
    for case, row in case_errors.iterrows():
        print(f"   {case:<50s} {row['n_wrong']:>6d} {row['models_wrong']}")

    return pd.DataFrame(type_rows), errors_df


def compute_case_difficulty(df):
    """Analyze per-case difficulty across all models."""
    print("\n" + "=" * 78)
    print("7. PER-CASE DIFFICULTY ANALYSIS")
    print("=" * 78)

    functioning = [m for m in MODEL_ORDER if TIER_LABELS.get(m) != "Failed"]
    fdf = df[df["model"].isin(functioning)]

    case_stats = fdf.groupby("case").agg(
        expected_valid=("expected_valid", "first"),
        n_correct=("correct", "sum"),
        n_total=("correct", "count"),
    )
    case_stats["accuracy"] = case_stats["n_correct"] / case_stats["n_total"]
    case_stats = case_stats.sort_values("accuracy")

    print(f"\n   {'Case':<50s} {'Truth':>6s} {'Correct':>8s} {'Acc':>6s}")
    print("   " + "-" * 75)
    for case, row in case_stats.iterrows():
        truth = "Valid" if row["expected_valid"] else "Invalid"
        print(f"   {case:<50s} {truth:>6s} {int(row['n_correct']):>3d}/{int(row['n_total'])}   "
              f"{row['accuracy']:5.1%}")

    # Summary: easiest vs hardest
    easy = case_stats[case_stats["accuracy"] == 1.0]
    hard = case_stats[case_stats["accuracy"] < 1.0]
    print(f"\n   Cases solved by ALL functioning models: {len(easy)}/{len(case_stats)}")
    print(f"   Cases with at least one error: {len(hard)}/{len(case_stats)}")

    return case_stats.reset_index()


def fisher_group_comparison(df):
    """Fisher's exact test comparing large (>=20B) vs small (<=4B) model groups."""
    print("\n" + "=" * 78)
    print("8. FISHER'S EXACT TEST: Large (>=20B) vs Small (<=4B) Models")
    print("=" * 78)

    functioning = [m for m in MODEL_ORDER if TIER_LABELS.get(m) != "Failed"]
    large = [m for m in functioning if MODEL_PARAMS.get(m, 0) >= 20]
    small = [m for m in functioning if MODEL_PARAMS.get(m, 999) <= 4]

    fdf = df[df["model"].isin(functioning)]

    large_correct = int(fdf[fdf["model"].isin(large)]["correct"].sum())
    large_total = int(fdf[fdf["model"].isin(large)]["correct"].count())
    small_correct = int(fdf[fdf["model"].isin(small)]["correct"].sum())
    small_total = int(fdf[fdf["model"].isin(small)]["correct"].count())

    # 2x2 table: [[large_correct, large_wrong], [small_correct, small_wrong]]
    table = [
        [large_correct, large_total - large_correct],
        [small_correct, small_total - small_correct],
    ]
    odds_ratio, p_value = stats.fisher_exact(table)

    print(f"\n   Large models ({', '.join(large)}): {large_correct}/{large_total} "
          f"({large_correct/large_total:.1%})")
    print(f"   Small models ({', '.join(small)}): {small_correct}/{small_total} "
          f"({small_correct/small_total:.1%})")
    print(f"\n   Fisher's exact test: OR = {odds_ratio:.2f}, p = {p_value:.6f}")
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"   Significance: {sig}")

    return {"large_correct": large_correct, "large_total": large_total,
            "small_correct": small_correct, "small_total": small_total,
            "odds_ratio": odds_ratio, "p_value": p_value}


def bootstrap_ci(df, n_boot=10000):
    """Bootstrapped CIs for error_match and warning_match scores."""
    print("\n" + "=" * 78)
    print("9. BOOTSTRAP 95% CIs FOR ERROR/WARNING MATCH SCORES")
    print("=" * 78)

    functioning = [m for m in MODEL_ORDER if TIER_LABELS.get(m) != "Failed"]
    fdf = df[df["model"].isin(functioning)]

    rng = np.random.default_rng(42)
    rows = []

    for model in [m for m in MODEL_ORDER if m in functioning]:
        mdf = fdf[fdf["model"] == model]
        for metric in ["error_match", "warning_match"]:
            if metric not in mdf.columns:
                continue
            vals = mdf[metric].dropna().values
            if len(vals) == 0:
                continue
            observed = vals.mean()
            boot_means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                                   for _ in range(n_boot)])
            ci_lo = np.percentile(boot_means, 2.5)
            ci_hi = np.percentile(boot_means, 97.5)
            rows.append({"model": model, "metric": metric, "mean": observed,
                         "ci_low": ci_lo, "ci_high": ci_hi})
            print(f"   {model:<20s} {metric:<15s}  {observed:.3f}  "
                  f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    return pd.DataFrame(rows)


def extract_qualitative_examples(df):
    """Extract 1 TP, 1 TN, 1 FP, 1 FN example from the best model."""
    print("\n" + "=" * 78)
    print("10. QUALITATIVE EXAMPLES (from best model)")
    print("=" * 78)

    # Use gpt-oss-120b as the "production" model
    best = "gpt-oss-120b"
    mdf = df[df["model"] == best].copy()

    examples = {}
    for _, row in mdf.iterrows():
        predicted_valid = row["valid"]
        actual_valid = row["expected_valid"]

        if predicted_valid and actual_valid:
            label = "TP"
        elif not predicted_valid and not actual_valid:
            label = "TN"
        elif predicted_valid and not actual_valid:
            label = "FP"
        else:
            label = "FN"

        if label not in examples:
            errors_text = str(row.get("errors", ""))[:300]
            examples[label] = {
                "type": label, "model": best, "file": row["file"],
                "expected": "Valid" if actual_valid else "Invalid",
                "predicted": "Valid" if predicted_valid else "Invalid",
                "errors": errors_text if errors_text and errors_text != "nan" else "(none)",
            }

    for label in ["TP", "TN", "FP", "FN"]:
        if label in examples:
            ex = examples[label]
            print(f"\n   [{label}] {ex['file']}")
            print(f"   Expected: {ex['expected']}, Predicted: {ex['predicted']}")
            print(f"   Errors: {ex['errors'][:200]}")
        else:
            print(f"\n   [{label}] No example found (model had no {label} cases)")

    return pd.DataFrame(list(examples.values()))


def save_results(accuracy_df, confusion_df, mcnemar_df, power_df, error_types_df, case_df,
                 fisher_results=None, bootstrap_df=None, examples_df=None):
    """Save all analysis results to CSV files."""
    print("\n" + "=" * 78)
    print("8. SAVING RESULTS")
    print("=" * 78)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = {
        "elm_accuracy_wilson_ci.csv": accuracy_df,
        "elm_confusion_matrices.csv": confusion_df,
        "elm_mcnemar_pairwise.csv": mcnemar_df,
        "elm_power_analysis.csv": power_df,
        "elm_error_types.csv": error_types_df,
        "elm_case_difficulty.csv": case_df,
    }
    if bootstrap_df is not None and len(bootstrap_df) > 0:
        files["elm_bootstrap_ci.csv"] = bootstrap_df
    if examples_df is not None and len(examples_df) > 0:
        files["elm_qualitative_examples.csv"] = examples_df

    for fname, data in files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        data.to_csv(path, index=False)
        print(f"   Saved: {path}")

    if fisher_results:
        import json
        path = os.path.join(OUTPUT_DIR, "elm_fisher_group.json")
        with open(path, 'w') as f:
            json.dump(fisher_results, f, indent=2)
        print(f"   Saved: {path}")


def print_publication_table(accuracy_df, confusion_df):
    """Print a publication-ready summary table."""
    print("\n" + "=" * 78)
    print("PUBLICATION TABLE: ELM Validation Model Comparison")
    print("=" * 78)

    merged = accuracy_df.merge(confusion_df[["model", "sensitivity", "specificity", "F1"]],
                               on="model")

    print(f"\n   {'Model':<20s} {'Params':>6s} {'Acc':>6s} {'95% CI':>16s} "
          f"{'Sens':>5s} {'Spec':>5s} {'F1':>5s} {'Tier'}")
    print("   " + "-" * 85)
    for _, r in merged.iterrows():
        ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        print(f"   {r['model']:<20s} {r['params_B']:>5.1f}B {r['accuracy']:5.1%} {ci_str:>16s} "
              f"{r['sensitivity']:5.2f} {r['specificity']:5.2f} {r['F1']:5.2f} {r['tier']}")

    # Base rate footnote
    print(f"\n   Note: 22 test cases (15 valid, 7 invalid). Base rate = 68.2%.")
    print(f"   4 small models (<=3B) failed to load on Modal — always predicted 'invalid'.")


def main():
    print("=" * 78)
    print("ELM VALIDATION RESULTS — STATISTICAL ANALYSIS")
    print("=" * 78)

    df = load_all_results()
    n_models = df["model"].nunique()
    n_cases = df["file"].nunique()
    print(f"\nLoaded {len(df)} observations: {n_models} models × {n_cases} test cases")
    print(f"Models: {sorted(df['model'].unique())}")

    # Ground truth distribution
    gt = df[df["model"] == MODEL_ORDER[0]]
    n_valid = gt["expected_valid"].sum()
    n_invalid = len(gt) - n_valid
    print(f"Ground truth: {int(n_valid)} valid, {int(n_invalid)} invalid")

    accuracy_df = compute_accuracy_wilson_ci(df)
    confusion_df = build_confusion_matrices(df)
    mcnemar_df = compute_mcnemar_tests(df)
    power_df = compute_power_analysis(df)
    heatmap_pivot = generate_heatmap(df)
    error_types_df, errors_detail = disaggregate_errors(df)
    case_df = compute_case_difficulty(df)
    fisher_results = fisher_group_comparison(df)
    bootstrap_df = bootstrap_ci(df)
    examples_df = extract_qualitative_examples(df)

    save_results(accuracy_df, confusion_df, mcnemar_df, power_df, error_types_df, case_df,
                 fisher_results=fisher_results, bootstrap_df=bootstrap_df,
                 examples_df=examples_df)
    print_publication_table(accuracy_df, confusion_df)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
