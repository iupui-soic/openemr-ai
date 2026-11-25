"""
Note-level association between automated metrics and physician ratings on the 159 Fareez notes,
with note-length adjustment, between-model tests of each metric, and cumulative link mixed models.

Inputs (paths relative to the repo root):
  rag_models/results/revision/fareez_automated_metrics.csv   (revision_fareez_metrics.py, then revision_entity_metrics.py)
  rag_models/rating_packets/ratings_rater_{1,2,3}.csv
  rag_models/rating_packets/ANSWER_KEY_DO_NOT_SHARE.csv
The failed generation (MedGemma-4B, MSK0037) is excluded. Length is summary_words: whitespace-separated
words of the note as shown in the rating packet.

Run:  python scripts/revision_metric_vs_clinician.py --repo .
Outputs in rag_models/results/revision/:
  metric_clinician_correlations.csv   Spearman rho, conversation-clustered bootstrap 95% CI, partial rho adjusted for length
  metric_between_model_tests.csv      Friedman, Kendall W, six Wilcoxon pairs with Bonferroni correction
  metric_by_model.csv                 mean (SD) of every metric by model
  metric_clmm_odds_ratios.csv         CLMM on unaveraged ratings: OR per 1 SD of the metric, adjusted for log length and rater
Needs numpy, pandas, scipy.
"""
import argparse, numpy as np, pandas as pd
from scipy import stats, optimize, special

DIMS = ["accuracy","completeness","organization","conciseness","clinical_utility","overall_quality"]
MODELS = ["gpt-oss-120b","gpt-oss-20b","qwen3-32b","medgemma-4b"]
RATERS = ["rater_1","rater_2","rater_3"]
LENGTH = "summary_words"
METRICS = ["bleu","rouge_l","sbert_similarity","bert_f1",
           "medcat_entity_recall","medcat_entity_precision","medcat_entity_f1",
           "scispacy_entity_recall","scispacy_entity_precision","scispacy_entity_f1"]

def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)[0]

def cluster_boot(df, fn, B=2000, seed=2026):
    rng = np.random.default_rng(seed); convs = df.conversation.unique()
    groups = {c: df[df.conversation == c] for c in convs}; out = []
    for _ in range(B):
        s = pd.concat([groups[c] for c in rng.choice(convs, len(convs))])
        out.append(fn(s))
    return np.nanpercentile(out, [2.5, 97.5])

def clmm_x(y_raw, X, conv, nodes=40):
    """Logit cumulative link mixed model with fixed effects X and a random intercept for conversation.
    Same likelihood and quadrature as clmm() in revision_rating_analyses.py (maximum likelihood, Gauss-Hermite)."""
    gx, gw = special.roots_hermite(nodes); gz = np.sqrt(2)*gx; gw = gw/np.sqrt(np.pi)
    cats = sorted(np.unique(y_raw)); y = np.array([cats.index(v) for v in y_raw]); q = len(cats)
    p = X.shape[1]; conv = pd.factorize(conv)[0]; G = conv.max()+1
    def unpack(par): return np.cumsum(np.r_[par[0], np.exp(par[1:q-1])]), par[q-1:q-1+p], np.exp(par[-1])
    def nll(par):
        th, beta, s = unpack(par); eta = (X@beta)[:,None] + s*gz[None,:]
        up = np.where(y[:,None] < q-1, special.expit(th[np.minimum(y,q-2)][:,None]-eta), 1.0)
        lo = np.where(y[:,None] > 0, special.expit(th[np.maximum(y-1,0)][:,None]-eta), 0.0)
        S = np.zeros((G, len(gz))); np.add.at(S, conv, np.log(np.clip(up-lo, 1e-300, None)))
        mx = S.max(1, keepdims=True); return -(mx[:,0] + np.log((np.exp(S-mx)*gw).sum(1))).sum()
    x0 = np.r_[-1.5, np.zeros(q-2), np.zeros(p), np.log(0.5)]
    bounds = [(None,None)]*(q-1) + [(-20,20)]*p + [(np.log(1e-3), np.log(10))]
    r = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    h = 1e-4; n = len(r.x); H = np.zeros((n,n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i]*h, np.eye(n)[j]*h
            H[i,j] = H[j,i] = (nll(r.x+ei+ej)-nll(r.x+ei-ej)-nll(r.x-ei+ej)+nll(r.x-ei-ej))/(4*h*h)
    try: se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))[q-1:q-1+p]
    except np.linalg.LinAlgError: se = np.full(p, np.nan)
    th, beta, s = unpack(r.x)
    return dict(nll=r.fun, beta=beta, se=se, sigma=s, converged=bool(r.success))

def z(v): return (v - v.mean())/v.std()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--metrics", default=None)
    a = ap.parse_args(); R = a.repo.rstrip("/") + "/rag_models"; out = f"{R}/results/revision"
    met = pd.read_csv(a.metrics or f"{out}/fareez_automated_metrics.csv")
    if "sbert_coherence" in met.columns: met = met.rename(columns={"sbert_coherence": "sbert_similarity"})
    key = pd.read_csv(f"{R}/rating_packets/ANSWER_KEY_DO_NOT_SHARE.csv")
    rat = pd.concat([pd.read_csv(f"{R}/rating_packets/ratings_rater_{i}.csv").assign(rater=f"rater_{i}") for i in (1,2,3)])
    rat = rat.merge(key, on=["summary_id","rater"], validate="one_to_one")
    rat = rat[~((rat.conversation == "MSK0037") & (rat.model == "medgemma-4b"))].copy()
    met = met[~((met.conversation == "MSK0037") & (met.model == "medgemma-4b"))].copy()
    clin = rat.groupby(["conversation","model"])[DIMS].mean().reset_index()
    clin["composite"] = clin[DIMS].mean(axis=1)
    metric_cols = [c for c in METRICS if c in met.columns]
    df = clin.merge(met[["conversation","model"] + metric_cols + [LENGTH]], on=["conversation","model"], how="inner")
    print(f"Notes with both metrics and ratings: {len(df)} (expected 159); ratings: {len(rat)} (expected 477)")
    assert len(df) == 159 and len(rat) == 477

    by_model = met.groupby("model")[metric_cols + [LENGTH]].agg(["mean","std"]).reindex(MODELS).round(3)
    by_model.columns = [f"{m}_{s}" for m, s in by_model.columns]
    by_model.to_csv(f"{out}/metric_by_model.csv"); print("\n== Mean (SD) by model"); print(by_model.T.to_string())

    rows = []
    for mc in metric_cols + [LENGTH]:
        for d in DIMS + ["composite"]:
            cols = list(dict.fromkeys([mc, d, "conversation", LENGTH]))
            sub = df[cols].dropna()
            rho = stats.spearmanr(sub[mc], sub[d]).correlation
            lo, hi = cluster_boot(sub, lambda s: stats.spearmanr(s[mc], s[d]).correlation)
            row = dict(metric=mc, dimension=d, n=len(sub), rho=round(rho,2), ci_lo=round(lo,2), ci_hi=round(hi,2))
            if mc != LENGTH:
                pr = partial_spearman(sub[mc].values, sub[d].values, sub[LENGTH].values)
                plo, phi = cluster_boot(sub, lambda s: partial_spearman(s[mc].values, s[d].values, s[LENGTH].values))
                row.update(partial_rho_len=round(pr,2), p_ci_lo=round(plo,2), p_ci_hi=round(phi,2))
            rows.append(row)
    cor = pd.DataFrame(rows); cor.to_csv(f"{out}/metric_clinician_correlations.csv", index=False)
    print("\n== Spearman correlations (95% CI from 2,000 bootstrap resamples of conversations)"); print(cor.to_string(index=False))

    # Between-model differences in each metric: Friedman over conversations, then Bonferroni-corrected Wilcoxon
    tests = []
    for mc in metric_cols + [LENGTH]:
        w = df.pivot_table(index="conversation", columns="model", values=mc)[MODELS].dropna()
        chi2, p = stats.friedmanchisquare(*[w[c] for c in MODELS])
        tests.append(dict(metric=mc, test="Friedman", n_conv=len(w), stat=round(chi2,2), p=p, W=round(chi2/(len(w)*3),3)))
        pairs = [(x, y) for i, x in enumerate(MODELS) for y in MODELS[i+1:]]
        for x, y in pairs:
            st, pp = stats.wilcoxon(w[x], w[y])
            tests.append(dict(metric=mc, test=f"{x} vs {y}", n_conv=len(w), stat=st, p=pp, p_bonf=min(pp*len(pairs),1), mean_diff=round((w[x]-w[y]).mean(),4)))
    t = pd.DataFrame(tests); t.to_csv(f"{out}/metric_between_model_tests.csv", index=False)
    print("\n== Between-model tests of each metric"); print(t.to_string(index=False))

    # CLMM on the unaveraged ratings: standardized metric + standardized log length + rater (fixed), conversation (random)
    long = rat.merge(met[["conversation","model"] + metric_cols + [LENGTH]], on=["conversation","model"], how="inner")
    assert len(long) == 477
    zlen = z(np.log(long[LENGTH])).values
    rdum = np.column_stack([(long.rater == r).astype(float).values for r in RATERS[1:]])
    crow = []
    for mc in metric_cols:
        zm = z(long[mc]).values
        for d in DIMS:
            f = clmm_x(long[d].values, np.column_stack([zm, zlen, rdum]), long.conversation.values)
            f0 = clmm_x(long[d].values, np.column_stack([zlen, rdum]), long.conversation.values)
            b, s = f["beta"][0], f["se"][0]; lr = 2*(f0["nll"] - f["nll"])
            crow.append(dict(metric=mc, dimension=d, n=len(long), or_per_sd=round(np.exp(b),2), ci_lo=round(np.exp(b-1.96*s),2),
                             ci_hi=round(np.exp(b+1.96*s),2), lr_chisq_1df=round(lr,2), p=stats.chi2.sf(max(lr,0),1),
                             or_log_length_per_sd=round(np.exp(f["beta"][1]),2), conversation_sd=round(f["sigma"],2), converged=f["converged"]))
    for d in DIMS:   # length alone, for reference
        f = clmm_x(long[d].values, np.column_stack([zlen, rdum]), long.conversation.values); b, s = f["beta"][0], f["se"][0]
        crow.append(dict(metric="log_length_only", dimension=d, n=len(long), or_per_sd=round(np.exp(b),2), ci_lo=round(np.exp(b-1.96*s),2),
                         ci_hi=round(np.exp(b+1.96*s),2), conversation_sd=round(f["sigma"],2), converged=f["converged"]))
    c = pd.DataFrame(crow); c.to_csv(f"{out}/metric_clmm_odds_ratios.csv", index=False)
    print("\n== CLMM: odds ratio for a higher rating per 1 SD of the metric, adjusted for log length and rater"); print(c.to_string(index=False))

if __name__ == "__main__":
    main()
