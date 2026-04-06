#!/usr/bin/env python3
"""
Generate Figure 3: Per-Case Heatmap for the ELM Simplifier paper.

Layout (top to bottom):
  Semantic Invalid (3 cases)   -- hardest, paper's key finding
  Parametric Invalid (13 cases)
  Valid (15 cases)             -- bottom block

X-axis: models grouped by tier (Frontier | Mid-range | Small)
        with vertical separators between tiers.

Colors: colorblind-safe blue (#4575b4) for correct,
        orange (#d73027) for incorrect, with hatching as
        secondary encoding for print/grayscale.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis")

# ── Case lists (canonical order within each group is overridden by difficulty) ─

VALID_CASES = [
    "Adult-Weight-Screening-and-Follow-Up-OpenEMR",
    "Breast-Cancer-Screening-OpenEMR",
    "Colon-Cancer-Screening-OpenEMR",
    "Prostate-Cancer-Screening-OpenEMR",
    "Depression_screening",
    "Cervical-Cancer-Screening",
    "Lung-Cancer-Screening",
    "AAA-Screening",
    "Alcohol-Misuse-Screening",
    "Falls-Prevention-Screening",
    "Tobacco-Use-Screening",
    "Osteoporosis screening",
    "USPSTFStatinUseForPrimaryPreventionOfCVDInAdultsSharedLogicFHIRv401",
    "Condition_and_Medication_Count_FHIRv401",
    "Chlamydia_Screening_Common_CQL",
]

PARAMETRIC_CASES = [
    "Hypertension-Screening",
    "Statin-Therapy-for-CVD-Prevention",
    "Type-2-Diabetes-Diagnosis",
    "Colorectal-Cancer-Average-Risk-Screening",
    "Anxiety-Screening",
    "HIV-Screening",
    "Prediabetes-Obesity-Screening",
    "Cervical-Cancer-Screening-WrongAge",
    "Falls-Prevention-WrongAge",
    "Alcohol-Misuse-WrongAge",
    "AAA-Screening-WrongAge",
    "Depression-Screening-WrongLookback",
    "Osteoporosis-WrongLookback",
]

SEMANTIC_CASES = [
    "Breast-Cancer-WrongOperator",
    "Tobacco-Missing-Exclusion",
    "Lung-Cancer-Missing-SubPopulation",
]

# ── Display names: all ≤25 characters, consistent hyphen style ──────────────

CASE_LABELS = {
    # Valid (15)
    "Adult-Weight-Screening-and-Follow-Up-OpenEMR": "Weight Screening",
    "Breast-Cancer-Screening-OpenEMR":              "Breast Cancer",
    "Colon-Cancer-Screening-OpenEMR":               "Colon Cancer",
    "Prostate-Cancer-Screening-OpenEMR":             "Prostate Cancer",
    "Depression_screening":                          "Depression",
    "Cervical-Cancer-Screening":                     "Cervical Cancer",
    "Lung-Cancer-Screening":                         "Lung Cancer",
    "AAA-Screening":                                 "AAA",
    "Alcohol-Misuse-Screening":                      "Alcohol Misuse",
    "Falls-Prevention-Screening":                    "Falls Prevention",
    "Tobacco-Use-Screening":                         "Tobacco Use",
    "Osteoporosis screening":                        "Osteoporosis",
    "USPSTFStatinUseForPrimaryPreventionOfCVDInAdultsSharedLogicFHIRv401":
                                                     "Statin-SharedLogic",
    "Condition_and_Medication_Count_FHIRv401":        "Condition-MedCount",
    "Chlamydia_Screening_Common_CQL":                "Chlamydia",
    # Parametric Invalid (13) – show the injected error
    "Hypertension-Screening":                        "Hypertension",
    "Statin-Therapy-for-CVD-Prevention":             "Statin-CVD",
    "Type-2-Diabetes-Diagnosis":                     "Type-2-Diabetes",
    "Colorectal-Cancer-Average-Risk-Screening":      "Colorectal-Cancer",
    "Anxiety-Screening":                             "Anxiety",
    "HIV-Screening":                                 "HIV",
    "Prediabetes-Obesity-Screening":                 "Prediabetes-Obesity",
    "Cervical-Cancer-Screening-WrongAge":            "Cervical-WrongAge",
    "Falls-Prevention-WrongAge":                     "Falls-WrongAge",
    "Alcohol-Misuse-WrongAge":                       "Alcohol-WrongAge",
    "AAA-Screening-WrongAge":                        "AAA-WrongAge",
    "Depression-Screening-WrongLookback":             "Depression-WrongLkbk",
    "Osteoporosis-WrongLookback":                    "Osteoporosis-WrongLkbk",
    # Semantic Invalid (3)
    "Breast-Cancer-WrongOperator":                   "Breast-WrongOp",
    "Tobacco-Missing-Exclusion":                     "Tobacco-MissingExcl",
    "Lung-Cancer-Missing-SubPopulation":              "Lung-MissingSubPop",
}

# ── Model tier ordering (matches Table I) ────────────────────────────────────

TIER_FRONTIER = ["gpt-oss-20b", "qwen3-32b", "gpt-oss-120b", "llama-3.3-70b"]
TIER_MIDRANGE = ["phi-3-mini", "medgemma-4b", "medgemma-1.5-4b", "gemma-3-4b"]
TIER_SMALL    = ["qwen-2.5-3b", "llama-3.2-3b", "qwen-2.5-1.5b", "llama-3.2-1b"]

MODEL_DISPLAY = {
    "gpt-oss-20b":    "GPT-OSS-20B",
    "qwen3-32b":      "Qwen3-32B",
    "gpt-oss-120b":   "GPT-OSS-120B",
    "llama-3.3-70b":  "Llama-3.3-70B",
    "phi-3-mini":     "Phi-3-Mini",
    "medgemma-4b":    "MedGemma-4B",
    "medgemma-1.5-4b":"MedGemma1.5-4B",
    "gemma-3-4b":     "Gemma-3-4B",
    "qwen-2.5-3b":    "Qwen-2.5-3B",
    "llama-3.2-3b":   "Llama-3.2-3B",
    "qwen-2.5-1.5b":  "Qwen-2.5-1.5B",
    "llama-3.2-1b":   "Llama-3.2-1B",
}

# ── Colorblind-safe palette ──────────────────────────────────────────────────

CLR_CORRECT   = "#4575b4"   # blue
CLR_INCORRECT = "#d73027"   # orange-red (distinguishable from blue in all CVD types)
CLR_BG_VALID  = "#f0f7f0"   # faint green tint for Valid band
CLR_BG_PARAM  = "#fff8ed"   # faint amber tint for Parametric band
CLR_BG_SEMAN  = "#fef0f0"   # faint red tint for Semantic band


def load_all_results():
    frames = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith("results-") and fname.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(RESULTS_DIR, fname)))
    return pd.concat(frames, ignore_index=True)


def generate_figure3():
    df = load_all_results()
    df["case"] = df["file"].str.replace(".json", "", regex=False)

    # Per-model accuracy (for label annotations)
    model_acc = df.groupby("model")["correct"].mean()

    # Fixed tier-based model order
    model_order = TIER_FRONTIER + TIER_MIDRANGE + TIER_SMALL
    n_frontier = len(TIER_FRONTIER)
    n_midrange = len(TIER_MIDRANGE)

    # Within each case group, sort by difficulty (easiest at bottom, hardest at top)
    # Since we draw bottom-up, the list order is easiest-first
    def sort_by_difficulty(cases):
        acc = df[df["case"].isin(cases)].groupby("case")["correct"].mean()
        return sorted(cases, key=lambda c: acc.get(c, 0), reverse=True)

    # Stack: Valid (bottom) → Parametric (middle) → Semantic (top)
    ordered_valid     = sort_by_difficulty(VALID_CASES)
    ordered_parametric = sort_by_difficulty(PARAMETRIC_CASES)
    ordered_semantic  = sort_by_difficulty(SEMANTIC_CASES)
    all_cases = ordered_valid + ordered_parametric + ordered_semantic

    n_valid    = len(ordered_valid)
    n_param    = len(ordered_parametric)
    n_semantic = len(ordered_semantic)
    n_cases    = len(all_cases)
    n_models   = len(model_order)

    # Pivot table
    pivot = df.pivot_table(index="case", columns="model", values="correct",
                           aggfunc="first")
    pivot = pivot.loc[all_cases, model_order].astype(float)

    # ── Figure ───────────────────────────────────────────────────────────────

    fig_w = 3.6 + n_models * 0.62
    fig_h = 1.8 + n_cases * 0.31
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})

    # ── Background shading per group ─────────────────────────────────────────

    ax.add_patch(plt.Rectangle((-0.02, 0), n_models + 0.04, n_valid,
                               facecolor=CLR_BG_VALID, edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((-0.02, n_valid), n_models + 0.04, n_param,
                               facecolor=CLR_BG_PARAM, edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((-0.02, n_valid + n_param), n_models + 0.04, n_semantic,
                               facecolor=CLR_BG_SEMAN, edgecolor="none", zorder=0))

    # ── Draw cells ───────────────────────────────────────────────────────────

    data = pivot.values
    for i in range(n_cases):
        for j in range(n_models):
            val = data[i, j]
            y = n_cases - 1 - i  # flip so first case is at top
            if val == 1.0:
                rect = plt.Rectangle((j, y), 1, 1, facecolor=CLR_CORRECT,
                                     edgecolor="white", linewidth=0.6, zorder=2)
                ax.add_patch(rect)
            else:
                # Incorrect: solid fill + diagonal hatch for secondary encoding
                rect = plt.Rectangle((j, y), 1, 1, facecolor=CLR_INCORRECT,
                                     edgecolor="white", linewidth=0.6, zorder=2)
                ax.add_patch(rect)
                hatch = plt.Rectangle((j, y), 1, 1, facecolor="none",
                                      edgecolor="white", linewidth=0,
                                      hatch="//", zorder=3, alpha=0.3)
                ax.add_patch(hatch)

    ax.set_xlim(0, n_models)
    ax.set_ylim(0, n_cases)

    # ── Vertical tier separators ─────────────────────────────────────────────

    vsep1 = n_frontier
    vsep2 = n_frontier + n_midrange
    ax.axvline(x=vsep1, color="black", linewidth=1.8, zorder=5)
    ax.axvline(x=vsep2, color="black", linewidth=1.8, zorder=5)

    # Tier labels above the heatmap
    tier_label_y = n_cases + 0.25
    ax.text(vsep1 / 2, tier_label_y, "Frontier",
            ha="center", va="bottom", fontsize=8, fontstyle="italic",
            clip_on=False)
    ax.text(vsep1 + n_midrange / 2, tier_label_y, "Mid-range",
            ha="center", va="bottom", fontsize=8, fontstyle="italic",
            clip_on=False)
    ax.text(vsep2 + len(TIER_SMALL) / 2, tier_label_y, "Small",
            ha="center", va="bottom", fontsize=8, fontstyle="italic",
            clip_on=False)

    # ── X-axis: model names + accuracy ───────────────────────────────────────

    x_labels = [f"{MODEL_DISPLAY[m]} ({model_acc[m]*100:.1f}%)" for m in model_order]
    ax.set_xticks([j + 0.5 for j in range(n_models)])
    ax.set_xticklabels(x_labels, fontsize=7.5, ha="right", rotation=40,
                       rotation_mode="anchor")
    ax.xaxis.set_ticks_position("bottom")

    # ── Y-axis ───────────────────────────────────────────────────────────────

    case_labels = [CASE_LABELS.get(c, c) for c in all_cases]
    y_positions = [n_cases - 1 - i + 0.5 for i in range(n_cases)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(case_labels, fontsize=7)

    # ── Horizontal group separators ──────────────────────────────────────────

    sep1_y = n_cases - n_valid           # between Valid and Parametric
    sep2_y = n_cases - n_valid - n_param # between Parametric and Semantic

    ax.axhline(y=sep1_y, color="black", linewidth=2.0, zorder=5)
    ax.axhline(y=sep2_y, color="black", linewidth=2.0, zorder=5)

    # ── Group bracket labels (right side) ────────────────────────────────────

    bracket_x = n_models + 0.15
    label_x   = n_models + 0.35

    valid_center    = n_cases - n_valid / 2
    param_center    = n_cases - n_valid - n_param / 2
    semantic_center = n_semantic / 2

    for center, top, bot, label in [
        (valid_center,    n_cases, sep1_y, f"Valid\n(n={n_valid})"),
        (param_center,    sep1_y, sep2_y,  f"Parametric\nInvalid\n(n={n_param})"),
        (semantic_center, sep2_y, 0,       f"Semantic\nInvalid\n(n={n_semantic})"),
    ]:
        margin = 0.3
        ax.plot([bracket_x, bracket_x], [bot + margin, top - margin],
                color="black", linewidth=1.0, clip_on=False)
        ax.plot([bracket_x, bracket_x - 0.08], [top - margin, top - margin],
                color="black", linewidth=1.0, clip_on=False)
        ax.plot([bracket_x, bracket_x - 0.08], [bot + margin, bot + margin],
                color="black", linewidth=1.0, clip_on=False)
        ax.text(label_x, center, label, fontsize=7.5, fontweight="bold",
                va="center", ha="left", clip_on=False)

    # ── Legend ───────────────────────────────────────────────────────────────

    correct_patch = mpatches.Patch(facecolor=CLR_CORRECT, edgecolor="gray",
                                   label="Correct")
    # Build an incorrect patch with hatch
    incorrect_patch = mpatches.Patch(facecolor=CLR_INCORRECT, edgecolor="gray",
                                    hatch="//", label="Incorrect")
    ax.legend(handles=[correct_patch, incorrect_patch], loc="upper left",
              bbox_to_anchor=(0.0, 1.08), ncol=2, fontsize=8,
              frameon=True, edgecolor="gray", fancybox=False)

    # ── Spines ───────────────────────────────────────────────────────────────

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    plt.subplots_adjust(right=0.78)

    # ── Save ─────────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, "figure3_per_case_heatmap.png")
    pdf_path = os.path.join(OUTPUT_DIR, "figure3_per_case_heatmap.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # ── Summary stats ────────────────────────────────────────────────────────

    print(f"\nModel order: {[MODEL_DISPLAY[m] for m in model_order]}")
    print(f"Cases: Valid={n_valid}, Parametric={n_param}, Semantic={n_semantic}")
    print("\nPer-group accuracy:")
    for m in model_order:
        v = pivot.loc[ordered_valid, m].mean() * 100
        p = pivot.loc[ordered_parametric, m].mean() * 100
        s = pivot.loc[ordered_semantic, m].mean() * 100
        print(f"  {MODEL_DISPLAY[m]:18s}  Valid={v:5.1f}%  Param={p:5.1f}%  Semantic={s:5.1f}%")

    # Verify all labels ≤25 chars
    for c, label in CASE_LABELS.items():
        if len(label) > 25:
            print(f"  WARNING: label too long ({len(label)}): {label}")

    # Suggested caption
    print("\n--- Suggested caption ---")
    print(
        "Figure 3. Per-case validation heatmap across 12 open-weight LLMs. "
        "Rows represent 31 test cases grouped into valid cases "
        f"(n={n_valid}, top), parametric invalid cases (n={n_param}, middle), "
        f"and semantic logic invalid cases (n={n_semantic}, bottom). "
        "Columns represent models ordered by tier: Frontier (>=20B parameters), "
        "Mid-range (3-4B), and Small (1-3B), separated by vertical lines. "
        "Blue cells indicate correct predictions; orange-red hatched cells "
        "indicate incorrect predictions. Within each block, cases are sorted "
        "by difficulty (easiest at top, hardest at bottom). "
        "All four frontier models achieve 100% accuracy on parametric invalid "
        "cases, but semantic logic errors remain challenging across all tiers."
    )


if __name__ == "__main__":
    generate_figure3()
