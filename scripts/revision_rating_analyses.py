"""
Analyses of the released round-one physician ratings: model means, Friedman and pairwise Wilcoxon
tests, score distributions by rater, agreement, cumulative link mixed models, specialty subanalyses,
keyword flags, the power sensitivity analysis, and note-length correlations.

Run:  python revision_rating_analyses.py --repo /path/to/openemr-ai [--include-failed]
By default the 3 ratings of the failed MedGemma-4B generation (MSK0037) are excluded.
Also writes rag_models/results/revision/ratings_long.csv (one row per rating, read by revision_clmm_crossed.R).
Needs numpy, pandas, scipy only.
"""
import argparse, re
import numpy as np, pandas as pd
from scipy import stats, optimize, special

DIMS = ["accuracy","completeness","organization","conciseness","clinical_utility","overall_quality"]
MODELS = ["gpt-oss-120b","gpt-oss-20b","qwen3-32b","medgemma-4b"]
RATERS = ["rater_1","rater_2","rater_3"]

# ---------------- agreement ----------------
def ordinal_weights(q):
    n = np.abs(np.subtract.outer(np.arange(q), np.arange(q))) + 1
    W = n*(n-1)/2
    return 1 - W/W.max()
def gwet_ac2(mat):
    mat = np.asarray(mat); cats = sorted(np.unique(mat)); q = len(cats); idx = {c:i for i,c in enumerate(cats)}
    rik = np.zeros((len(mat), q))
    for i, row in enumerate(mat):
        for v in row: rik[i, idx[v]] += 1
    W = ordinal_weights(q); ri = rik.sum(1); rstar = rik @ W.T
    pa = np.mean((rik*(rstar-1)).sum(1)/(ri*(ri-1)))
    pik = (rik/ri[:,None]).mean(0); pe = W.sum()/(q*(q-1))*(pik*(1-pik)).sum()
    return (pa-pe)/(1-pe)
def kripp_alpha(mat, metric="ordinal"):
    mat = np.asarray(mat, dtype=float); vals = sorted(set(mat[~np.isnan(mat)].tolist())); q = len(vals); idx = {v:i for i,v in enumerate(vals)}
    o = np.zeros((q,q))
    for row in mat:
        r = row[~np.isnan(row)]; mu = len(r)
        if mu < 2: continue
        for a in range(mu):
            for b in range(mu):
                if a != b: o[idx[r[a]], idx[r[b]]] += 1.0/(mu-1)
    nc = o.sum(1); n = nc.sum(); d2 = np.zeros((q,q))
    for c in range(q):
        for k in range(q):
            if metric == "nominal": d2[c,k] = float(c != k)
            elif metric == "interval": d2[c,k] = (vals[c]-vals[k])**2
            else:
                lo, hi = min(c,k), max(c,k); d2[c,k] = (nc[lo:hi+1].sum() - (nc[c]+nc[k])/2.0)**2
    return 1 - ((o*d2).sum()/n) / ((np.outer(nc,nc)*d2).sum()/(n*(n-1)))
def qwk(a, b, cats=(1,2,3,4,5)):
    q = len(cats); idx = {c:i for i,c in enumerate(cats)}; O = np.zeros((q,q))
    for x, y in zip(a, b): O[idx[x], idx[y]] += 1
    O /= O.sum(); E = np.outer(O.sum(1), O.sum(0)); i, j = np.indices((q,q)); D = (i-j)**2/(q-1)**2
    return 1 - (O*D).sum()/(E*D).sum()

# ---------------- cumulative link mixed model ----------------
def clmm(df, y_col, with_model=True, nodes=40):
    """Logit CLMM: model + rater fixed effects, random intercept for conversation (Gauss-Hermite)."""
    gx, gw = special.roots_hermite(nodes); gz = np.sqrt(2)*gx; gw = gw/np.sqrt(np.pi)
    cats = sorted(df[y_col].unique()); y = df[y_col].map({c:i for i,c in enumerate(cats)}).values; q = len(cats)
    cols = ([(df.model == m).astype(float).values for m in MODELS[1:]] if with_model else []) + [(df.rater == r).astype(float).values for r in RATERS[1:]]
    X = np.column_stack(cols); p = X.shape[1]
    conv = pd.factorize(df.conversation)[0]; G = conv.max()+1
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
    return dict(nll=r.fun, beta=beta, se=se, sigma=s)

def min_detectable_W(alpha, N, k=4, power=0.8):
    crit = stats.chi2.ppf(1-alpha, k-1)
    lam = optimize.brentq(lambda l: stats.ncx2.sf(crit, k-1, l) - power, 1e-6, 500)
    return lam/(N*(k-1))

def fp(p):
    if p < .001: return "<.001"
    return (f"{p:.3f}" if p < .01 else f"{p:.2f}").lstrip("0") if p < 1 else "1.00"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--boot", type=int, default=10000); a = ap.parse_args()
    R = a.repo.rstrip("/") + "/rag_models"
    rng = np.random.default_rng(2026)   # agreement and length bootstraps
    rng42 = np.random.default_rng(42)   # Kendall W bootstrap
    key = pd.read_csv(f"{R}/rating_packets/ANSWER_KEY_DO_NOT_SHARE.csv")
    m = pd.concat([pd.read_csv(f"{R}/rating_packets/ratings_{r}.csv").assign(rater=r) for r in RATERS]).merge(key, on=["summary_id","rater"], validate="one_to_one")
    failed = (m.conversation == "MSK0037") & (m.model == "medgemma-4b")
    LONG = ["summary_id","rater","conversation","model","specialty","condition"] + DIMS
    m[LONG].assign(failed_generation=failed.astype(int)).to_csv(f"{R}/results/revision/ratings_long.csv", index=False)
    if not a.include_failed:
        m = m[~failed].copy()
    print(f"Ratings analysed: {len(m)}")
    avg = m.groupby(["conversation","model"])[DIMS].mean()
    blocks = avg[DIMS[0]].unstack().dropna().index; avgc = avg.loc[avg.index.get_level_values(0).isin(blocks)]; N = len(blocks)
    print(f"Complete conversation blocks: {N}")

    print("\n== Mean (SD), composite")
    for mm in MODELS:
        g = m[m.model == mm]; print(mm, " | ".join(f"{g[d].mean():.3f} ({g[d].std():.2f})" for d in DIMS), f"| composite {np.mean([g[d].mean() for d in DIMS]):.2f} | without organization {np.mean([g[d].mean() for d in DIMS if d!='organization']):.2f}")
    print("grand means:", m[DIMS].mean().round(2).to_dict())

    print("\n== Friedman on rater-averaged scores, Kendall W with bootstrap CI")
    for d in DIMS:
        w = avgc[d].unstack()[MODELS].values; chi2 = stats.friedmanchisquare(*w.T).statistic; bs = []
        for _ in range(a.boot):
            s = w[rng42.integers(0, N, N)]
            try: bs.append(stats.friedmanchisquare(*s.T).statistic/(N*3))
            except ValueError: pass
        print(f"{d:17s} chi2={chi2:.1f} P={fp(stats.chi2.sf(chi2,3))} W={chi2/(N*3):.3f} (95% CI {np.percentile(bs,2.5):.2f} to {np.percentile(bs,97.5):.2f})")
    print(f"minimal detectable W at 80% power: alpha .05 -> {min_detectable_W(.05, N):.3f}; alpha .05/6 -> {min_detectable_W(.05/6, N):.3f}")

    print("\n== Wilcoxon post hoc (Bonferroni over 36 tests)")
    for d in DIMS:
        w = avgc[d].unstack()
        for i, x in enumerate(MODELS):
            for y in MODELS[i+1:]:
                diff = w[x]-w[y]; p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(w[x], w[y]).pvalue
                print(f"{d:17s} {x} vs {y}: diff {diff.mean():.2f} P={fp(p)} Pbonf36={fp(min(1, p*36))}")

    print("\n== Score distribution by rater")
    for d in DIMS:
        for r in RATERS:
            g = m[m.rater == r][d]; print(d, r, g.value_counts().reindex([1,2,3,4,5], fill_value=0).tolist(), f"{g.mean():.2f} ({g.std():.2f})")
    print("\n== Rater x model means, within-rater Friedman")
    for d in DIMS:
        for r in RATERS:
            g = m[m.rater == r]; w = g.pivot_table(index="conversation", columns="model", values=d)[MODELS].dropna()
            print(d, r, g.groupby("model")[d].mean()[MODELS].round(2).tolist(), "P=" + fp(stats.friedmanchisquare(*w.values.T).pvalue))

    print("\n== Agreement")
    for d in DIMS:
        piv = m.pivot_table(index=["conversation","model"], columns="rater", values=d).dropna(); v = piv.values
        bs = [kripp_alpha(v[rng.integers(0, len(v), len(v))]) for _ in range(1000)]
        print(f"{d:17s} AC2={gwet_ac2(v):.3f} alpha={kripp_alpha(v):.3f} (95% CI {np.percentile(bs,2.5):.2f} to {np.percentile(bs,97.5):.2f})")
        for x, y in [("rater_1","rater_2"),("rater_1","rater_3"),("rater_2","rater_3")]:
            X, Y = piv[x].values, piv[y].values
            print(f"   {x} vs {y}: exact {100*(X==Y).mean():.1f}% within1 {100*(abs(X-Y)<=1).mean():.1f}% qwk {qwk(X,Y):.2f} AC2 {gwet_ac2(np.c_[X,Y]):.2f}")

    print("\n== CLMM (model + rater fixed, conversation random), ORs vs gpt-oss-120b")
    for d in DIMS:
        f1 = clmm(m, d, True); f0 = clmm(m, d, False); lr = 2*(f0["nll"]-f1["nll"])
        ors = [f"{np.exp(b):.3g} ({np.exp(b-1.96*s):.3g} to {np.exp(b+1.96*s):.3g})" for b, s in zip(f1["beta"][:3], f1["se"][:3])]
        print(f"{d:17s} LR={lr:.1f} P={fp(stats.chi2.sf(lr,3))} conv SD={f1['sigma']:.2f} OR 20b/qwen/medgemma: " + "; ".join(ors) + f" | rater2,3 logit: {np.round(f1['beta'][3:],2).tolist()}")

    print("\n== Specialty subanalyses")
    for sp in ["RES","MSK"]:
        sub = avgc[[c.startswith(sp) for c in avgc.index.get_level_values(0)]]
        for d in DIMS:
            w = sub[d].unstack()[MODELS]; r = stats.friedmanchisquare(*w.values.T)
            print(sp, len(w), d, w.mean().round(2).tolist(), f"W={r.statistic/(len(w)*3):.2f} P={fp(r.pvalue)}")

    print("\n== Note length (words, as rated) vs ratings, Spearman with conversation bootstrap")
    txt = open(f"{R}/rating_packets/packet_rater_1.txt", encoding="utf-8").read()
    parts = re.split(r"\n#{20,}\n# SUMMARY (S\d+)", txt); words = {}
    for i in range(1, len(parts), 2):
        body = parts[i+1].split("SECTION C: AI-GENERATED SUMMARY")[1].split("\n", 2)[2]
        words[parts[i]] = len(re.split(r"\n-{40,}\n", body)[0].split())
    k1 = key[key.rater == "rater_1"].assign(words=lambda x: x.summary_id.map(words)).set_index(["conversation","model"]).words
    A = avg.copy(); A["words"] = k1.reindex(A.index).values; A["composite"] = A[DIMS].mean(axis=1)
    convs = A.index.get_level_values(0).unique().values; groups = {c: A.xs(c, level=0) for c in convs}
    for d in DIMS + ["composite"]:
        rho = stats.spearmanr(A.words, A[d]).correlation; bs = []
        for _ in range(2000):
            s = pd.concat([groups[c] for c in rng.choice(convs, len(convs))]); bs.append(stats.spearmanr(s.words, s[d]).correlation)
        print(f"{d:17s} rho={rho:.2f} (95% CI {np.nanpercentile(bs,2.5):.2f} to {np.nanpercentile(bs,97.5):.2f})")
    print("mean words by model:", A.groupby(level=1).words.mean().round(1).to_dict())

    print("\n== Keyword flags (patterns from results/fareez/error_analysis.ipynb)")
    pats = dict(
        chart=r'ehr (inclusion|comorbidity|fact|history|conditions?)|imports? ehr|mismatched ehr (?!correctly|appropriately|but not)|extra info|unmentioned (additions?|plan|meds?|comorbidit)|(includes|adds|inserts)( |s )(unmentioned|chart|background)|pulled? from (the )?(chart|ehr)|chart content not (in|from) (the )?(transcript|encounter)',
        ignored=r'(correctly|appropriately) ignor(es|ed) (the )?(mismatched )?ehr|(correctly|appropriately) handle[ds]( the)? (mismatched )?(ehr|patient profile)|did not (import|include) (mismatched )?ehr|avoid(s|ed) ehr',
        fabrication=r'fabricat|invent(ed|s|ion)|hallucinat(ed|ion|es)|made.up|incorrect medication|wrong (med|drug|dose)|invented (exam|finding|plan)',
        gender=r'gender mismatch|mismatched patient (profile|record)|sex mismatch')
    c = m.comments.fillna("").str.lower()
    F = pd.DataFrame({k: c.apply(lambda s, p=v: bool(re.search(p, s, re.I))) for k, v in pats.items()}); F["model"] = m.model.values; F["conv"] = m.conversation.values
    print("per model, % of ratings:", F.groupby("model")[list(pats)].mean().mul(100).round(1).reindex(MODELS).to_dict("index"))
    print("all, % of ratings:", F[list(pats)].mean().mul(100).round(1).to_dict(), "| chart and fabrication both:", int((F.chart & F.fabrication).sum()))
    notes = F.groupby(["conv","model"])[["chart","fabrication"]].any()
    print("notes flagged by any rater, %:", notes.mean().mul(100).round(1).to_dict(), notes.groupby(level=1).mean().mul(100).round(1).reindex(MODELS).to_dict("index"))

if __name__ == "__main__":
    main()
