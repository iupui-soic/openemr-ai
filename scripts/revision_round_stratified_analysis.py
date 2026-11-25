"""
Statistically defensible round-stratified primary analysis + exploratory pooled analysis
for the OpenEMR-AI physician ratings and expanded automated metrics (legacy + SOTA).

Design rationale:
- Round 1 (4 models, n=159 notes, 477 ratings) and Round 2 (2 models, n=80 notes, 240 ratings)
  were rated several months apart. Although 15 repeated anchor notes per rater were embedded in
  Round 2, 266/270 repeated scores were identical to Round 1 (90/90 for Rater 1 and Rater 3),
  so the anchors cannot be treated as independent cross-round equating observations.
- Primary model comparisons are therefore performed WITHIN each interleaved rating round:
    * Round 1: Friedman tests (39 complete blocks, 3 df), 36 Bonferroni-corrected Wilcoxon pairs,
      and within-round CLMM (model + rater fixed, conversation random).
    * Round 2: Paired Wilcoxon tests (40 complete blocks, Qwen3.8-27B vs LLaMA-4-Scout, 6 tests
      Bonferroni-corrected) and within-round CLMM (model + rater fixed, conversation random).
- Exploratory pooled 6-model analyses (n=239 notes, 717 ratings) are reported alongside
  per-round analyses and explicitly labeled exploratory.
- Metric-vs-physician analyses cover ALL 10 legacy metrics, note length, AND the new locally
  runnable SOTA metrics (untruncated BERTScore, chunked SBERT/BGE-large/S-PubMedBERT, and
  RAG-retrieved DeBERTa-v3-large NLI entailment & contradiction) across Round 1, Round 2,
  and exploratory pooled scopes, including full CLMM odds ratios for every metric.

Run:  .venv/bin/python scripts/revision_round_stratified_analysis.py --repo .
"""
import argparse
import itertools
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_rating_analyses import gwet_ac2, kripp_alpha, qwk, fp
from revision_metric_vs_clinician import clmm_x, partial_spearman, cluster_boot, z

DIMS = ["accuracy", "completeness", "organization", "conciseness", "clinical_utility", "overall_quality"]
R1_MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3-32b", "medgemma-4b"]
R2_MODELS = ["qwen3.8-27b", "llama-4-scout"]
ALL_MODELS = ["qwen3.8-27b", "gpt-oss-120b", "gpt-oss-20b", "qwen3-32b", "medgemma-4b", "llama-4-scout"]
RATERS = ["rater_1", "rater_2", "rater_3"]

LEGACY_METRICS = [
    "bleu", "rouge_l", "bert_f1", "sbert_similarity",
    "medcat_entity_recall", "medcat_entity_precision", "medcat_entity_f1",
    "scispacy_entity_recall", "scispacy_entity_precision", "scispacy_entity_f1",
]
SOTA_METRICS = [
    "bert_f1_full", "bert_r_full", "bert_p_full",
    "sbert_full", "bge_large_full", "pubmedbert_full",
    "nli_entail_transcript", "nli_contra_transcript", "nli_supported_frac_transcript",
    "nli_entail_inputs", "nli_contra_inputs", "nli_supported_frac_inputs",
]
ALL_METRICS = LEGACY_METRICS + SOTA_METRICS + ["summary_words"]


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    return 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(a, b).pvalue)


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation for Wilcoxon signed-rank test (a minus b)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    r_pos = ranks[nz > 0].sum()
    r_neg = ranks[nz < 0].sum()
    return float((r_pos - r_neg) / (r_pos + r_neg))


def fast_cluster_boot_spearman(df, mc, d, B=2000, seed=2026):
    """Fast numpy conversation-clustered bootstrap for Spearman rho and length-adjusted partial Spearman rho."""
    rng = np.random.default_rng(seed)
    convs = df["conversation"].unique()
    # Pre-extract numpy arrays per conversation
    groups_x = []
    groups_y = []
    groups_z = []
    has_len = (mc != "summary_words")
    for c in convs:
        sub = df[df["conversation"] == c]
        groups_x.append(sub[mc].values.astype(float))
        groups_y.append(sub[d].values.astype(float))
        if has_len:
            groups_z.append(sub["summary_words"].values.astype(float))
    n_c = len(convs)
    idx_mat = rng.integers(0, n_c, size=(B, n_c))
    rhos = np.empty(B, dtype=float)
    prhos = np.empty(B, dtype=float) if has_len else None
    for b in range(B):
        idxs = idx_mat[b]
        x = np.concatenate([groups_x[i] for i in idxs])
        y = np.concatenate([groups_y[i] for i in idxs])
        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        rx_c = rx - rx.mean()
        ry_c = ry - ry.mean()
        denom = np.sqrt((rx_c * rx_c).sum() * (ry_c * ry_c).sum())
        rhos[b] = (rx_c * ry_c).sum() / denom if denom > 0 else np.nan
        if has_len:
            z_ = np.concatenate([groups_z[i] for i in idxs])
            rz = stats.rankdata(z_)
            rz_c = rz - rz.mean()
            zz = (rz_c * rz_c).sum()
            if zz > 0:
                ex = rx_c - ((rx_c * rz_c).sum() / zz) * rz_c
                ey = ry_c - ((ry_c * rz_c).sum() / zz) * rz_c
                pden = np.sqrt((ex * ex).sum() * (ey * ey).sum())
                prhos[b] = (ex * ey).sum() / pden if pden > 0 else np.nan
            else:
                prhos[b] = np.nan
    lo, hi = np.nanpercentile(rhos, [2.5, 97.5])
    if has_len:
        plo, phi = np.nanpercentile(prhos, [2.5, 97.5])
        return lo, hi, plo, phi
    return lo, hi, np.nan, np.nan


def within_conv_spearman(df, mc, d):
    x = df[mc] - df.groupby("conversation")[mc].transform("mean")
    y = df[d] - df.groupby("conversation")[d].transform("mean")
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    R = os.path.abspath(args.repo) + "/rag_models"
    out = f"{R}/results/revision"
    rng = np.random.default_rng(2026)

    allr = pd.read_csv(f"{out}/round2_ratings_long.csv")
    m1 = pd.read_csv(f"{out}/fareez_automated_metrics.csv").rename(columns={"sbert_coherence": "sbert_similarity"})
    m2 = pd.read_csv(f"{out}/fareez_round2_metrics.csv")
    met = pd.concat([m1, m2], ignore_index=True)
    sota = pd.read_csv(f"{out}/fareez_sota_metrics.csv")
    met = met.merge(sota, on=["conversation", "model"], how="inner", validate="one_to_one")
    assert len(met) == 239, f"Expected 239 merged metric rows, got {len(met)}"

    note = allr.groupby(["conversation", "model", "round"])[DIMS].mean().reset_index()
    note["composite"] = note[DIMS].mean(axis=1)
    note["composite_no_org"] = note[[d for d in DIMS if d != "organization"]].mean(axis=1)
    df = note.merge(met, on=["conversation", "model"], how="inner", validate="one_to_one")

    # ------------------------------------------------------------------
    # 1. PRIMARY WITHIN-ROUND MODEL COMPARISONS
    # ------------------------------------------------------------------
    print("=" * 80)
    print("1. PRIMARY WITHIN-ROUND MODEL COMPARISONS")
    print("=" * 80)

    # Round 1: 4 models, 39 complete blocks
    r1_df = allr[allr["round"] == 1]
    r1_note = note[note["round"] == 1]
    r1_blocks = r1_note.pivot_table(index="conversation", columns="model", values="accuracy")[R1_MODELS].dropna().index
    print(f"\n--- ROUND 1 (4 interleaved models, n={len(r1_note)} notes, {len(r1_blocks)} complete blocks, {len(r1_df)} ratings) ---")
    r1_summary_rows = []
    for d in DIMS + ["composite", "composite_no_org"]:
        w = r1_note.pivot_table(index="conversation", columns="model", values=d).loc[r1_blocks, R1_MODELS].values
        chi2 = float(stats.friedmanchisquare(*w.T).statistic)
        pval = float(stats.chi2.sf(chi2, len(R1_MODELS) - 1))
        W = chi2 / (len(r1_blocks) * (len(R1_MODELS) - 1))
        bs = []
        for _ in range(args.boot):
            s = w[rng.integers(0, len(r1_blocks), len(r1_blocks))]
            try:
                bs.append(stats.friedmanchisquare(*s.T).statistic / (len(r1_blocks) * (len(R1_MODELS) - 1)))
            except ValueError:
                pass
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  R1 {d:16s}: chi2(3)={chi2:5.1f}, P={fp(pval)}, W={W:.2f} ({lo:.2f} to {hi:.2f})")
        r1_summary_rows.append(dict(scope="round_1", dimension=d, n_blocks=len(r1_blocks),
                                    stat_name="Friedman_chi2_3df", stat=round(chi2, 2), p=pval,
                                    effect_name="Kendall_W", effect=round(W, 3),
                                    ci_lo=round(lo, 3), ci_hi=round(hi, 3)))

    # Round 1 pairwise (36 tests across 6 dims)
    pw_rows = []
    for d in DIMS + ["composite"]:
        w = r1_note.pivot_table(index="conversation", columns="model", values=d).loc[r1_blocks]
        for a, b in itertools.combinations(R1_MODELS, 2):
            p = wil(w[a], w[b])
            pw_rows.append(dict(scope="round_1", dimension=d, model_a=a, model_b=b,
                                mean_diff=round(float((w[a] - w[b]).mean()), 2),
                                r_rb=round(rank_biserial(w[a], w[b]), 2), p=p))
    pw_r1 = pd.DataFrame(pw_rows)
    pw_r1["p_bonf"] = np.where(pw_r1["dimension"] == "composite",
                               np.minimum(1.0, pw_r1["p"] * 6),
                               np.minimum(1.0, pw_r1["p"] * 36))

    # Round 2: 2 models, 40 complete blocks
    r2_df = allr[allr["round"] == 2]
    r2_note = note[note["round"] == 2]
    r2_blocks = r2_note.pivot_table(index="conversation", columns="model", values="accuracy")[R2_MODELS].dropna().index
    print(f"\n--- ROUND 2 (2 interleaved models: Qwen3.8-27B vs LLaMA-4-Scout, n={len(r2_note)} notes, {len(r2_blocks)} complete blocks, {len(r2_df)} ratings) ---")
    r2_pw_rows = []
    for d in DIMS + ["composite", "composite_no_org"]:
        w = r2_note.pivot_table(index="conversation", columns="model", values=d).loc[r2_blocks]
        a, b = w["qwen3.8-27b"].values, w["llama-4-scout"].values
        diff = a - b
        p = wil(a, b)
        rb = rank_biserial(a, b)
        bs_d = [float(np.mean(diff[rng.integers(0, len(diff), len(diff))])) for _ in range(args.boot)]
        lo, hi = np.percentile(bs_d, [2.5, 97.5])
        p_adj = min(1.0, p * 6) if d in DIMS else p
        print(f"  R2 {d:16s}: Qwen3.8={a.mean():.2f}, LLaMA4={b.mean():.2f}, diff={diff.mean():+.2f} ({lo:+.2f} to {hi:+.2f}), r_rb={rb:+.2f}, P={fp(p)}, adjP={fp(p_adj)}")
        r1_summary_rows.append(dict(scope="round_2", dimension=d, n_blocks=len(r2_blocks),
                                    stat_name="Wilcoxon_diff", stat=round(float(diff.mean()), 3), p=p,
                                    effect_name="rank_biserial", effect=round(rb, 3),
                                    ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
        if d != "composite_no_org":
            r2_pw_rows.append(dict(scope="round_2", dimension=d, model_a="qwen3.8-27b", model_b="llama-4-scout",
                                   mean_diff=round(float(diff.mean()), 2), r_rb=round(rb, 2), p=p, p_bonf=p_adj))
    pw_all = pd.concat([pw_r1, pd.DataFrame(r2_pw_rows)], ignore_index=True)
    pw_all.to_csv(f"{out}/stratified_pairwise.csv", index=False)

    # Within-rater tests in Round 1 and Round 2 separately
    print("\n--- WITHIN-RATER TESTS SEPARATELY BY ROUND ---")
    for d in DIMS:
        r1_ps = []
        r2_ps = []
        for r in RATERS:
            g1 = r1_df[r1_df.rater == r].pivot_table(index="conversation", columns="model", values=d)[R1_MODELS].dropna()
            try:
                p1 = fp(stats.friedmanchisquare(*g1.values.T).pvalue)
            except ValueError:
                p1 = "NA"
            g2 = r2_df[r2_df.rater == r].pivot_table(index="conversation", columns="model", values=d)[R2_MODELS].dropna()
            p2 = fp(wil(g2["qwen3.8-27b"], g2["llama-4-scout"]))
            diff2 = (g2["qwen3.8-27b"] - g2["llama-4-scout"]).mean()
            r1_ps.append(f"{r}:{p1}")
            r2_ps.append(f"{r}:{diff2:+.2f}(P={p2})")
        print(f"  {d:16s} | R1 Friedman: {', '.join(r1_ps)} | R2 Qwen3.8-LLaMA4: {', '.join(r2_ps)}")

    # ------------------------------------------------------------------
    # 2. WITHIN-ROUND CLMMs (NO INVALID ANCHOR EQUATING!)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. WITHIN-ROUND CLMMs (ROUND 1: 4 MODELS; ROUND 2: 2 MODELS)")
    print("=" * 80)
    clmm_rows = []
    # Round 1 CLMM (ref = gpt-oss-120b)
    Xr1 = np.column_stack([(r1_df.rater == r).astype(float).values for r in RATERS[1:]])
    Xm1 = np.column_stack([(r1_df.model == m).astype(float).values for m in R1_MODELS[1:]])
    for d in DIMS:
        y = r1_df[d].values
        full = clmm_x(y, np.column_stack([Xm1, Xr1]), r1_df.conversation.values)
        null = clmm_x(y, Xr1, r1_df.conversation.values)
        lr = 2 * (null["nll"] - full["nll"])
        row = dict(scope="round_1", dimension=d, n_ratings=len(r1_df), df=3,
                   lr_chi2=round(lr, 1), p=fp(stats.chi2.sf(max(lr, 0), 3)),
                   conv_sd=round(float(full["sigma"]), 2), converged=full["converged"])
        for j, m in enumerate(R1_MODELS[1:]):
            b, s = full["beta"][j], full["se"][j]
            row[f"or_{m}"] = "Not estimable" if d == "organization" else f"{np.exp(b):.2f} ({np.exp(b - 1.96 * s):.2f} to {np.exp(b + 1.96 * s):.2f})"
        clmm_rows.append(row)

    # Round 2 CLMM (Qwen3.8-27B vs LLaMA-4-Scout; ref = llama-4-scout)
    Xr2 = np.column_stack([(r2_df.rater == r).astype(float).values for r in RATERS[1:]])
    Xm2 = (r2_df.model == "qwen3.8-27b").astype(float).values[:, None]
    for d in DIMS:
        y = r2_df[d].values
        full = clmm_x(y, np.column_stack([Xm2, Xr2]), r2_df.conversation.values)
        null = clmm_x(y, Xr2, r2_df.conversation.values)
        lr = 2 * (null["nll"] - full["nll"])
        b, s = full["beta"][0], full["se"][0]
        row = dict(scope="round_2", dimension=d, n_ratings=len(r2_df), df=1,
                   lr_chi2=round(lr, 1), p=fp(stats.chi2.sf(max(lr, 0), 1)),
                   conv_sd=round(float(full["sigma"]), 2), converged=full["converged"],
                   or_qwen38_vs_llama4=f"{np.exp(b):.2f} ({np.exp(b - 1.96 * s):.2f} to {np.exp(b + 1.96 * s):.2f})")
        clmm_rows.append(row)
        print(f"  R2 CLMM {d:16s}: LR chi2(1)={lr:5.1f} (P={row['p']}), OR(Qwen3.8 vs LLaMA4)={row['or_qwen38_vs_llama4']}, conv_SD={row['conv_sd']}")

    pd.DataFrame(clmm_rows).to_csv(f"{out}/stratified_clmm.csv", index=False)
    pd.DataFrame(r1_summary_rows).to_csv(f"{out}/stratified_model_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 3. EXPANDED AUTOMATED METRICS BY MODEL + WITHIN-ROUND TESTS
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. EXPANDED AUTOMATED METRICS BY MODEL (LEGACY + SOTA)")
    print("=" * 80)
    by_m = df.groupby("model")[ALL_METRICS].agg(["mean", "std"]).reindex(ALL_MODELS)
    by_m.columns = [f"{a}_{b}" for a, b in by_m.columns]
    by_m.to_csv(f"{out}/stratified_metric_by_model.csv")

    show_sota = ["bert_f1", "bert_f1_full", "sbert_similarity", "sbert_full", "bge_large_full", "pubmedbert_full",
                 "nli_entail_transcript", "nli_contra_transcript", "nli_entail_inputs", "nli_contra_inputs",
                 "medcat_entity_recall", "scispacy_entity_recall", "summary_words", "composite"]
    print(df.groupby("model")[show_sota].mean().reindex(ALL_MODELS).round(3).to_string())

    # Round 2 head-to-head metric tests (LLaMA-4-Scout vs Qwen3.8-27B, 40 blocks)
    print("\n--- ROUND 2 HEAD-TO-HEAD METRIC TESTS (LLaMA-4-Scout minus Qwen3.8-27B, n=40 blocks) ---")
    mt_rows = []
    for mc in ALL_METRICS:
        w2 = df[df["round"] == 2].pivot_table(index="conversation", columns="model", values=mc).loc[r2_blocks]
        d2 = (w2["llama-4-scout"] - w2["qwen3.8-27b"]).mean()
        p2 = wil(w2["llama-4-scout"], w2["qwen3.8-27b"])
        w1 = df[df["round"] == 1].pivot_table(index="conversation", columns="model", values=mc).loc[r1_blocks, R1_MODELS]
        chi1 = float(stats.friedmanchisquare(*w1.values.T).statistic)
        p1 = float(stats.chi2.sf(chi1, 3))
        wp = df.pivot_table(index="conversation", columns="model", values=mc).loc[r1_blocks, ALL_MODELS]
        chip = float(stats.friedmanchisquare(*wp.values.T).statistic)
        pp = float(stats.chi2.sf(chip, 5))
        means = df.groupby("model")[mc].mean().reindex(ALL_MODELS)
        mt_rows.append(dict(metric=mc, r1_chi2=round(chi1, 1), r1_p=p1,
                            r2_llama_minus_qwen38=round(float(d2), 4), r2_p=p2,
                            pooled_chi2=round(chip, 1), pooled_p=pp,
                            pooled_W=round(chip / (len(r1_blocks) * 5), 2),
                            highest_model=means.idxmax(), lowest_model=means.idxmin()))
        if mc in show_sota:
            print(f"  {mc:26s}: LLaMA4={w2['llama-4-scout'].mean():.3f}, Qwen3.8={w2['qwen3.8-27b'].mean():.3f}, diff={d2:+.3f} (P={fp(p2)}) | pooled top={means.idxmax()}")
    pd.DataFrame(mt_rows).to_csv(f"{out}/stratified_metric_tests.csv", index=False)

    # ------------------------------------------------------------------
    # 4. NOTE-LEVEL CORRELATIONS: ROUND 1, ROUND 2, AND POOLED (EXPLORATORY)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. NOTE-LEVEL SPEARMAN CORRELATIONS (ROUND 1, ROUND 2, POOLED EXPLORATORY)")
    print("=" * 80)
    cor_rows = []
    scopes = [
        ("round one", df[df["round"] == 1]),
        ("round two", df[df["round"] == 2]),
        ("all six models", df),
    ]
    for sname, sub_all in scopes:
        for mc in ALL_METRICS:
            for d in DIMS + ["composite"]:
                cols = list(dict.fromkeys([mc, d, "conversation", "summary_words"]))
                sub = sub_all[cols].dropna()
                rho = float(stats.spearmanr(sub[mc], sub[d]).correlation)
                lo, hi, plo, phi = fast_cluster_boot_spearman(sub, mc, d, B=args.boot, seed=2026)
                wc = within_conv_spearman(sub, mc, d)
                row = dict(scope=sname, metric=mc, dimension=d, n=len(sub),
                           rho=round(rho, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                           rho_within_conv=round(wc, 2))
                if mc != "summary_words":
                    pr = float(partial_spearman(sub[mc].values, sub[d].values, sub["summary_words"].values))
                    row.update(partial_rho_len=round(pr, 2), p_ci_lo=round(plo, 2), p_ci_hi=round(phi, 2))
                cor_rows.append(row)
    cor_df = pd.DataFrame(cor_rows)
    cor_df.to_csv(f"{out}/stratified_metric_correlations.csv", index=False)

    # Print key summary table for accuracy, completeness, overall_quality, composite
    for sname in ["round one", "round two", "all six models"]:
        print(f"\n--- Scope: {sname} ---")
        subc = cor_df[(cor_df.scope == sname) & (cor_df.dimension.isin(["accuracy", "completeness", "overall_quality", "composite"]))]
        for mc in ALL_METRICS:
            parts = []
            for d in ["accuracy", "completeness", "composite"]:
                r = subc[(subc.metric == mc) & (subc.dimension == d)].iloc[0]
                parts.append(f"{d[:4]}={r.rho:+.2f}({r.ci_lo:+.2f},{r.ci_hi:+.2f})")
            rc = subc[(subc.metric == mc) & (subc.dimension == "composite")].iloc[0]
            ra = subc[(subc.metric == mc) & (subc.dimension == "accuracy")].iloc[0]
            padj = f"adjComp={rc.partial_rho_len:+.2f}({rc.p_ci_lo:+.2f},{rc.p_ci_hi:+.2f}) adjAcc={ra.partial_rho_len:+.2f}" if mc != "summary_words" else ""
            print(f"  {mc:28s}: " + " | ".join(parts) + f" | {padj}")

    # ------------------------------------------------------------------
    # 5. FULL CLMM ODDS RATIOS FOR ALL METRICS (POOLED + ROUND 1 + ROUND 2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. METRIC CLMM ODDS RATIOS (ALL LEGACY + SOTA METRICS)")
    print("=" * 80)
    long_df = allr.merge(met[["conversation", "model"] + ALL_METRICS], on=["conversation", "model"], how="inner")
    assert len(long_df) == 717
    mclmm_rows = []
    for sname, lsub, add_round in [
        ("all six models", long_df, True),
        ("round one", long_df[long_df["round"] == 1], False),
        ("round two", long_df[long_df["round"] == 2], False),
    ]:
        zlen = z(np.log(lsub["summary_words"])).values
        rdum = np.column_stack([(lsub.rater == r).astype(float).values for r in RATERS[1:]])
        base_X = np.column_stack([zlen, rdum, (lsub["round"] == 2).astype(float).values]) if add_round else np.column_stack([zlen, rdum])
        for mc in LEGACY_METRICS + SOTA_METRICS:
            zm = z(lsub[mc]).values
            for d in DIMS:
                f = clmm_x(lsub[d].values, np.column_stack([zm, base_X]), lsub.conversation.values)
                f0 = clmm_x(lsub[d].values, base_X, lsub.conversation.values)
                b, s = f["beta"][0], f["se"][0]
                lr = 2 * (f0["nll"] - f["nll"])
                mclmm_rows.append(dict(
                    scope=sname, metric=mc, dimension=d, n=len(lsub),
                    or_per_sd=round(float(np.exp(b)), 2),
                    ci_lo=round(float(np.exp(b - 1.96 * s)), 2),
                    ci_hi=round(float(np.exp(b + 1.96 * s)), 2),
                    p=float(stats.chi2.sf(max(lr, 0), 1)),
                    converged=f["converged"],
                ))
    mclmm_df = pd.DataFrame(mclmm_rows)
    mclmm_df.to_csv(f"{out}/stratified_metric_clmm.csv", index=False)

    print("\n--- Pooled CLMM OR per 1 SD (adjusted for log length, rater, round) ---")
    subm = mclmm_df[mclmm_df.scope == "all six models"]
    for mc in ["bert_f1", "bert_f1_full", "sbert_similarity", "sbert_full", "bge_large_full", "pubmedbert_full",
               "nli_entail_transcript", "nli_contra_transcript", "nli_entail_inputs", "nli_contra_inputs",
               "medcat_entity_recall", "scispacy_entity_recall"]:
        parts = []
        for d in ["accuracy", "completeness", "clinical_utility", "overall_quality"]:
            r = subm[(subm.metric == mc) & (subm.dimension == d)].iloc[0]
            parts.append(f"{d[:4]}={r.or_per_sd:.2f}({r.ci_lo:.2f}-{r.ci_hi:.2f})")
        print(f"  {mc:24s}: " + " | ".join(parts))

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
