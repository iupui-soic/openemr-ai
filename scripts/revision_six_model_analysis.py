"""
Six-model analysis of the physician ratings: the four models of round one and the two models
added in round two (LLaMA-4-Scout, Qwen3.8-27B), rated by the same three physicians.

Reads the long rating file written by revision_round2_analysis.py (run that first), the round-two
packets and returned rating files (for the 15 anchor notes per rater), and the per-note metric files.

Prints: ratings by model, anchor test-retest, Friedman tests over six models with Kendall W,
pairwise Wilcoxon tests (Bonferroni over 90), cumulative link mixed models with a round effect
identified by the anchors, agreement, per-rater results, between-model tests of each automated
metric, and the model-level comparison of metrics with ratings.

Run:  python scripts/revision_six_model_analysis.py --repo .
Output: rag_models/results/revision/six_model_output.txt (stdout), six_model_table.csv,
        six_model_pairwise.csv, six_model_clmm.csv
"""
import argparse, itertools, os, sys
import numpy as np, pandas as pd
from scipy import stats, optimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_rating_analyses import gwet_ac2, kripp_alpha, qwk, fp
from revision_metric_vs_clinician import clmm_x

DIMS = ["accuracy", "completeness", "organization", "conciseness", "clinical_utility", "overall_quality"]
MODELS = ["qwen3.8-27b", "gpt-oss-120b", "gpt-oss-20b", "qwen3-32b", "medgemma-4b", "llama-4-scout"]
REF = "gpt-oss-120b"
RATERS = ["rater_1", "rater_2", "rater_3"]
METRICS = ["bleu", "rouge_l", "bert_f1", "sbert_similarity", "medcat_entity_recall", "medcat_entity_precision",
           "medcat_entity_f1", "scispacy_entity_recall", "scispacy_entity_precision", "scispacy_entity_f1", "summary_words"]


def min_detectable_W(alpha, N, k, power=0.8):
    crit = stats.chi2.ppf(1 - alpha, k - 1)
    lam = optimize.brentq(lambda l: stats.ncx2.sf(crit, k - 1, l) - power, 1e-6, 500)
    return lam / (N * (k - 1))


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    return 1.0 if np.allclose(d, 0) else stats.wilcoxon(a, b).pvalue


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--ratings-dir", default=None)
    ap.add_argument("--boot", type=int, default=10000); a = ap.parse_args()
    R = a.repo.rstrip("/") + "/rag_models"; out = f"{R}/results/revision"
    rng = np.random.default_rng(2026)

    allr = pd.read_csv(f"{out}/round2_ratings_long.csv")          # round one (477) and new round-two notes (240)
    key2 = pd.read_csv(f"{R}/rating_packets_round2/ANSWER_KEY_DO_NOT_SHARE.csv")
    r2 = pd.concat([pd.read_csv(f"{a.ratings_dir or R + '/rating_packets_round2'}/ratings_rater_{i}.csv").assign(rater=f"rater_{i}") for i in (1, 2, 3)])
    r2 = r2.merge(key2, on=["summary_id", "rater"], validate="one_to_one")
    anc = r2[r2.is_anchor == 1].assign(round=2)
    print(f"Ratings: {len(allr)} of {allr.groupby(['conversation','model']).ngroups} notes; anchor re-ratings: {len(anc)}")

    # ---------- ratings by model ----------
    print("\n== Ratings by model: mean (SD) over ratings; composite is the mean of the six dimension means")
    rows = []
    for m in MODELS:
        g = allr[allr.model == m]
        row = dict(model=m, n_ratings=len(g), **{d: f"{g[d].mean():.2f} ({g[d].std():.2f})" for d in DIMS},
                   composite=round(np.mean([g[d].mean() for d in DIMS]), 2),
                   composite_no_org=round(np.mean([g[d].mean() for d in DIMS if d != "organization"]), 2))
        rows.append(row)
    tab = pd.DataFrame(rows); tab.to_csv(f"{out}/six_model_table.csv", index=False); print(tab.to_string(index=False))
    print("grand means:", allr[DIMS].mean().round(2).to_dict())

    # ---------- anchors ----------
    print("\n== Anchors: the same rater scoring the same note in both rounds (45 pairs)")
    am = anc.merge(allr[allr["round"] == 1][["rater", "conversation", "model"] + DIMS], on=["rater", "conversation", "model"],
                   suffixes=("_2", "_1"), validate="one_to_one")
    for d in DIMS:
        x1, x2 = am[d + "_1"].values, am[d + "_2"].values; diff = x2 - x1
        k = qwk(x1, x2) if len(set(x1) | set(x2)) > 1 else float("nan")
        print(f"{d:17s} shift {diff.mean():+.2f} (Wilcoxon P={fp(wil(x2, x1))}); exact {100*(diff == 0).mean():.0f}%, within 1 point {100*(abs(diff) <= 1).mean():.0f}%, weighted kappa {k:.2f}")
    c1 = am[[d + "_1" for d in DIMS]].mean(1).values; c2 = am[[d + "_2" for d in DIMS]].mean(1).values
    print(f"composite         shift {np.mean(c2 - c1):+.2f} (P={fp(wil(c2, c1))}); Spearman {stats.spearmanr(c1, c2).correlation:.2f}; Pearson {stats.pearsonr(c1, c2)[0]:.2f}")
    for r in RATERS:
        s = am[am.rater == r]; e1 = s[[d + "_1" for d in DIMS]].mean(1).values; e2 = s[[d + "_2" for d in DIMS]].mean(1).values
        print(f"   {r}: composite shift {np.mean(e2 - e1):+.2f} (P={fp(wil(e2, e1))}); by dimension " +
              ", ".join(f"{d[:5]} {np.mean(s[d+'_2'] - s[d+'_1']):+.2f}" for d in DIMS))

    # ---------- Friedman over six models ----------
    note = allr.groupby(["conversation", "model"])[DIMS].mean(); note["composite"] = note[DIMS].mean(1)
    blocks = note["accuracy"].unstack()[MODELS].dropna().index; N = len(blocks); k = len(MODELS)
    print(f"\n== Friedman over six models on rater-averaged scores ({N} complete conversation blocks)")
    for d in DIMS + ["composite"]:
        w = note[d].unstack().loc[blocks, MODELS].values; chi2 = stats.friedmanchisquare(*w.T).statistic; bs = []
        for _ in range(a.boot):
            s = w[rng.integers(0, N, N)]
            try: bs.append(stats.friedmanchisquare(*s.T).statistic / (N * (k - 1)))
            except ValueError: pass
        print(f"{d:17s} chi2={chi2:.1f} P={fp(stats.chi2.sf(chi2, k-1))} W={chi2/(N*(k-1)):.2f} (95% CI {np.percentile(bs,2.5):.2f} to {np.percentile(bs,97.5):.2f}) | means " +
              ", ".join(f"{m} {note[d].unstack()[m].mean():.2f}" for m in MODELS))
    print(f"minimal detectable W at 80% power, {N} blocks, 6 models: alpha .05 -> {min_detectable_W(.05, N, k):.3f}; alpha .05/6 -> {min_detectable_W(.05/6, N, k):.3f}")

    # ---------- pairwise ----------
    prow = []
    for d in DIMS + ["composite"]:
        w = note[d].unstack().loc[blocks]
        for x, y in itertools.combinations(MODELS, 2):
            p = wil(w[x], w[y]); prow.append(dict(dimension=d, model_a=x, model_b=y, mean_diff=round((w[x] - w[y]).mean(), 2), p=p))
    pw = pd.DataFrame(prow); nt = (pw.dimension != "composite").sum()
    pw["p_bonferroni_90"] = np.minimum(1, pw.p * nt); pw.to_csv(f"{out}/six_model_pairwise.csv", index=False)
    print(f"\n== Pairwise Wilcoxon signed-rank tests (Bonferroni over {nt} tests on the six dimensions)")
    for d in DIMS + ["composite"]:
        s = pw[pw.dimension == d]
        print(d + ": " + "; ".join(f"{r.model_a} vs {r.model_b} {r.mean_diff:+.2f} P={fp(r.p)} adj={fp(r.p_bonferroni_90)}" for r in s.itertuples()))

    # ---------- CLMM with round effect identified by anchors ----------
    print("\n== Cumulative link mixed models: model, rater and round fixed; conversation random. Anchor re-ratings included so that round is estimable")
    long = pd.concat([allr[["rater", "round", "conversation", "model"] + DIMS], anc[["rater", "round", "conversation", "model"] + DIMS]])
    others = [m for m in MODELS if m != REF]
    Xm = np.column_stack([(long.model == m).astype(float).values for m in others])
    Xr = np.column_stack([(long.rater == r).astype(float).values for r in RATERS[1:]])
    Xround = (long["round"] == 2).astype(float).values[:, None]
    crow = []
    for d in DIMS:
        y = long[d].values
        full = clmm_x(y, np.column_stack([Xm, Xr, Xround]), long.conversation.values)
        nomodel = clmm_x(y, np.column_stack([Xr, Xround]), long.conversation.values)
        noround = clmm_x(y, np.column_stack([Xm, Xr]), long.conversation.values)
        lr = 2 * (nomodel["nll"] - full["nll"]); lrr = 2 * (noround["nll"] - full["nll"])
        row = dict(dimension=d, n=len(long), lr_model=round(lr, 1), p_model=fp(stats.chi2.sf(max(lr, 0), len(others))),
                   or_round2=round(float(np.exp(full["beta"][-1])), 2),
                   round_ci=f"{np.exp(full['beta'][-1]-1.96*full['se'][-1]):.2f} to {np.exp(full['beta'][-1]+1.96*full['se'][-1]):.2f}",
                   p_round=fp(stats.chi2.sf(max(lrr, 0), 1)), conv_sd=round(float(full["sigma"]), 2), converged=full["converged"])
        for j, m in enumerate(others):
            b, s = full["beta"][j], full["se"][j]
            row[f"or_{m}"] = "not estimable" if d == "organization" else f"{np.exp(b):.2f} ({np.exp(b-1.96*s):.2f} to {np.exp(b+1.96*s):.2f})"
        crow.append(row)
    cl = pd.DataFrame(crow); cl.to_csv(f"{out}/six_model_clmm.csv", index=False); print(cl.T.to_string())

    # ---------- agreement ----------
    print("\n== Agreement: Gwet AC2 (ordinal weights) and Krippendorff alpha (ordinal), with pairwise exact agreement")
    for name, df in (("round one, 159 notes", allr[allr["round"] == 1]), ("round two, 80 notes", allr[allr["round"] == 2]), ("both rounds, 239 notes", allr)):
        for d in DIMS:
            piv = df.pivot_table(index=["conversation", "model"], columns="rater", values=d).dropna(); v = piv.values
            bs = [kripp_alpha(v[rng.integers(0, len(v), len(v))]) for _ in range(1000)]
            pr = "; ".join(f"{x[-1]}v{y[-1]} exact {100*(piv[x] == piv[y]).mean():.0f}% within1 {100*(abs(piv[x]-piv[y]) <= 1).mean():.0f}% qwk {qwk(piv[x].values, piv[y].values):.2f}"
                           for x, y in itertools.combinations(RATERS, 2))
            print(f"{name:24s} {d:17s} AC2={gwet_ac2(v):.2f} alpha={kripp_alpha(v):.2f} ({np.percentile(bs,2.5):.2f} to {np.percentile(bs,97.5):.2f}) | {pr}")

    # ---------- per rater ----------
    print("\n== Mean score by rater and model, with within-rater Friedman test over six models")
    for d in DIMS:
        for r in RATERS:
            g = allr[allr.rater == r]; w = g.pivot_table(index="conversation", columns="model", values=d)[MODELS].dropna()
            try: p = fp(stats.friedmanchisquare(*w.values.T).pvalue)
            except ValueError: p = "NA"
            print(f"{d:17s} {r} " + " ".join(f"{g[g.model == m][d].mean():.2f}" for m in MODELS) + f"  P={p}")
    print("\n== Score distribution by rater, round two notes (80 per rater)")
    for d in DIMS:
        for r in RATERS:
            g = allr[(allr.rater == r) & (allr["round"] == 2)][d]
            print(d, r, g.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).tolist(), f"{g.mean():.2f} ({g.std():.2f})")

    # ---------- automated metrics ----------
    m1 = pd.read_csv(f"{out}/fareez_automated_metrics.csv"); m2 = pd.read_csv(f"{out}/fareez_round2_metrics.csv")
    if "sbert_coherence" in m1.columns: m1 = m1.rename(columns={"sbert_coherence": "sbert_similarity"})
    met = pd.concat([m1, m2]).set_index(["conversation", "model"])
    print(f"\n== Automated metrics by model, mean (SD), and Friedman test over six models ({N} blocks)")
    mrow = []
    for mc in METRICS:
        w = met[mc].unstack().loc[blocks, MODELS]; chi2 = stats.friedmanchisquare(*w.values.T).statistic
        sig = sum(min(1, wil(w[x], w[y]) * 15) < .05 for x, y in itertools.combinations(MODELS, 2))
        top = met[mc].unstack()[MODELS].mean().idxmax()
        print(f"{mc:26s} " + " ".join(f"{met[mc].unstack()[m].mean():.3f} ({met[mc].unstack()[m].std():.3f})" for m in MODELS) +
              f" | chi2={chi2:.1f} P={fp(stats.chi2.sf(chi2, k-1))} W={chi2/(N*(k-1)):.2f}, pairs significant {sig}/15, highest {top}")
    print("\n== LLaMA-4-Scout against each other model: metric difference (Wilcoxon P, uncorrected)")
    for mc in ["bert_f1", "sbert_similarity", "rouge_l", "medcat_entity_recall", "medcat_entity_precision", "medcat_entity_f1", "scispacy_entity_recall"]:
        w = met[mc].unstack().loc[blocks]
        print(f"{mc:26s} " + "; ".join(f"vs {m} {(w['llama-4-scout'] - w[m]).mean():+.3f} P={fp(wil(w['llama-4-scout'], w[m]))}" for m in MODELS if m != "llama-4-scout"))
    print("\n== Model level: rank of each model (1 = highest) on metrics and on physician ratings")
    ml = pd.concat([met[METRICS].groupby("model").mean(), note.groupby("model")[DIMS + ["composite"]].mean()], axis=1).reindex(MODELS)
    print(ml.rank(ascending=False).astype(int).T.to_string())
    print("\nSpearman across the six model means (exact permutation P for n=6 not reported; descriptive):")
    for mc in METRICS:
        print(f"{mc:26s} " + ", ".join(f"{d[:6]} {stats.spearmanr(ml[mc], ml[d]).correlation:+.2f}" for d in DIMS + ["composite"]))


if __name__ == "__main__":
    main()
