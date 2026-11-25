"""
Round-two analysis: LLaMA-4-Scout and Qwen3.8 27B notes rated by the same three physicians,
with 15 unmarked anchor notes from round one per rater.

Inputs (relative to the repo root):
  rag_models/rating_packets/                 round-one ratings and answer key
  rag_models/rating_packets_round2/ANSWER_KEY_DO_NOT_SHARE.csv
  rag_models/rating_packets_round2/ratings_rater_{1,2,3}.csv   returned scores (free-text comments are not released)
  rag_models/results/revision/fareez_automated_metrics.csv, fareez_round2_metrics.csv

A returned file whose summary IDs belong to another rater's packet is mapped through that packet's key
(the 95 notes are the same for every rater; only the order and the IDs differ) and is reported.

Run:  python scripts/revision_round2_analysis.py --repo . [--drop-rater rater_2]
Outputs in rag_models/results/revision/: round2_ratings_long.csv, round2_model_means.csv,
  round2_metric_clinician_correlations.csv, round2_metric_clmm_odds_ratios.csv
"""
import argparse, itertools, sys, os
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_rating_analyses import gwet_ac2, kripp_alpha
from revision_metric_vs_clinician import clmm_x, partial_spearman, cluster_boot, z

DIMS = ["accuracy","completeness","organization","conciseness","clinical_utility","overall_quality"]
R1 = ["gpt-oss-120b","gpt-oss-20b","qwen3-32b","medgemma-4b"]; R2 = ["qwen3.8-27b","llama-4-scout"]
MODELS = R1 + R2
RATERS = ["rater_1","rater_2","rater_3"]
METRICS = ["bleu","rouge_l","sbert_similarity","bert_f1","medcat_entity_recall","medcat_entity_precision","medcat_entity_f1",
           "scispacy_entity_recall","scispacy_entity_precision","scispacy_entity_f1"]
LENGTH = "summary_words"

def fp(p): return "<.001" if p < .001 else f"{p:.3f}".lstrip("0")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--ratings-dir", default=None)
    ap.add_argument("--drop-rater", default=None); a = ap.parse_args()
    R = a.repo.rstrip("/") + "/rag_models"; out = f"{R}/results/revision"
    tag = f"_without_{a.drop_rater}" if a.drop_rater else ""

    # ---------- load ----------
    key1 = pd.read_csv(f"{R}/rating_packets/ANSWER_KEY_DO_NOT_SHARE.csv")
    r1 = pd.concat([pd.read_csv(f"{R}/rating_packets/ratings_rater_{i}.csv").assign(rater=f"rater_{i}") for i in (1,2,3)])
    r1 = r1.merge(key1, on=["summary_id","rater"], validate="one_to_one")
    r1 = r1[~((r1.conversation == "MSK0037") & (r1.model == "medgemma-4b"))].assign(round=1)
    key2 = pd.read_csv(f"{R}/rating_packets_round2/ANSWER_KEY_DO_NOT_SHARE.csv")
    parts = []
    print("== Returned files")
    for i in (1,2,3):
        d = pd.read_csv(f"{a.ratings_dir or R + '/rating_packets_round2'}/ratings_rater_{i}.csv")
        owner = key2[key2.summary_id.isin(d.summary_id)].rater.unique()
        assert len(owner) == 1 and len(d) == 95 and d[DIMS].notna().all().all()
        print(f"rater_{i}: 95 rows, IDs belong to the packet of {owner[0]}" + ("" if owner[0] == f"rater_{i}" else "  <-- NOT THIS RATER'S PACKET"))
        k = key2[key2.rater == owner[0]].drop(columns="rater")
        parts.append(d.merge(k, on="summary_id", validate="one_to_one").assign(rater=f"rater_{i}", packet=owner[0]))
    r2 = pd.concat(parts).assign(round=2)
    for x, y in itertools.combinations((1,2,3), 2):
        m = parts[x-1].merge(parts[y-1], on=["conversation","model"], suffixes=("_a","_b"))
        same = (m[[d+"_a" for d in DIMS]].values == m[[d+"_b" for d in DIMS]].values).all(1)
        print(f"rater_{x} vs rater_{y}: identical six-score rows {same.sum()}/95")
    if a.drop_rater:
        r1 = r1[r1.rater != a.drop_rater]; r2 = r2[r2.rater != a.drop_rater]; print(f"\n*** {a.drop_rater} dropped from both rounds ***")
    raters = [r for r in RATERS if r != a.drop_rater]
    new = r2[r2.is_anchor == 0].copy(); anc = r2[r2.is_anchor == 1].copy()
    cols = ["rater","round","conversation","model","specialty","condition"] + DIMS
    allr = pd.concat([r1[cols], new[cols]]); allr.to_csv(f"{out}/round2_ratings_long{tag}.csv", index=False)
    print(f"\nRound one ratings {len(r1)}, round two new-note ratings {len(new)}, anchor ratings {len(anc)}")

    # ---------- anchors ----------
    print("\n== Anchors: the same rater scoring the same note in round one and round two (round two minus round one)")
    am = anc.merge(r1[["rater","conversation","model"] + DIMS], on=["rater","conversation","model"], suffixes=("_2","_1"), validate="one_to_one")
    for grp, sub in [("all raters", am)] + [(r, am[am.rater == r]) for r in raters]:
        line = []
        for d in DIMS:
            diff = sub[d+"_2"] - sub[d+"_1"]
            p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            line.append(f"{d[:6]} {diff.mean():+.2f} (exact {100*(diff == 0).mean():.0f}%, P={fp(p)})")
        comp = sub[[d+"_2" for d in DIMS]].mean(1).values - sub[[d+"_1" for d in DIMS]].mean(1).values
        print(f"{grp:11s} n={len(sub):2d} composite {comp.mean():+.2f} (P={fp(stats.wilcoxon(comp).pvalue if (comp != 0).any() else 1.0)}) | " + "; ".join(line))
    print("By round-one model (composite shift):", am.assign(s=am[[d+"_2" for d in DIMS]].mean(1).values - am[[d+"_1" for d in DIMS]].mean(1).values).groupby("model").s.mean().round(2).to_dict())

    # ---------- agreement ----------
    print("\n== Agreement on the 80 new notes (and round one for reference)")
    for name, df in (("round two", new), ("round one", r1)):
        for d in DIMS:
            mat = df.pivot_table(index=["conversation","model"], columns="rater", values=d).dropna().values
            print(f"{name} {d:17s} AC2={gwet_ac2(mat):.3f} alpha={kripp_alpha(mat):.3f}")

    # ---------- model means ----------
    note = allr.groupby(["round","conversation","model"])[DIMS].mean().reset_index(); note["composite"] = note[DIMS].mean(1)
    mm = note.groupby("model")[DIMS + ["composite"]].agg(["mean","std"]).reindex(MODELS).round(2)
    mm.columns = [f"{d}_{s}" for d, s in mm.columns]; mm.to_csv(f"{out}/round2_model_means{tag}.csv")
    print("\n== Mean rating by model (rater-averaged per note; round one models from round one)")
    print(note.groupby("model")[DIMS + ["composite"]].mean().reindex(MODELS).round(2).to_string())
    print("\n== Paired Wilcoxon over conversations, rater-averaged (not corrected)")
    pairs = [("llama-4-scout","qwen3.8-27b"), ("llama-4-scout","medgemma-4b"), ("llama-4-scout","gpt-oss-120b"), ("qwen3.8-27b","gpt-oss-120b"), ("qwen3.8-27b","qwen3-32b")]
    for x, y in pairs:
        line = []
        for d in DIMS + ["composite"]:
            w = note.pivot_table(index="conversation", columns="model", values=d)[[x, y]].dropna(); diff = w[x] - w[y]
            p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            line.append(f"{d[:6]} {diff.mean():+.2f} P={fp(p)}")
        print(f"{x} minus {y}{' (cross-round)' if (x in R2) != (y in R2) else ''}: " + "; ".join(line))

    # ---------- metrics vs ratings ----------
    m1 = pd.read_csv(f"{out}/fareez_automated_metrics.csv"); m2 = pd.read_csv(f"{out}/fareez_round2_metrics.csv")
    if "sbert_coherence" in m1.columns: m1 = m1.rename(columns={"sbert_coherence": "sbert_similarity"})
    met = pd.concat([m1, m2])[["conversation","model"] + METRICS + [LENGTH]]
    df = note.merge(met, on=["conversation","model"], validate="one_to_one"); print(f"\nNotes with metrics and ratings: {len(df)}")
    print("\n== Model-level: mean metric and mean rating (six models)")
    ml = df.groupby("model")[["bert_f1","sbert_similarity","rouge_l","medcat_entity_recall","medcat_entity_precision","medcat_entity_f1",LENGTH,"completeness","accuracy","conciseness","overall_quality","composite"]].mean().reindex(MODELS)
    print(ml.round(3).to_string())
    for mc in ("bert_f1","sbert_similarity","rouge_l","medcat_entity_recall","medcat_entity_precision","medcat_entity_f1",LENGTH):
        print(f"  rank correlation across 6 model means, {mc:24s}: " + ", ".join(f"{d[:6]} {stats.spearmanr(ml[mc], ml[d]).correlation:+.2f}" for d in ("completeness","accuracy","conciseness","overall_quality","composite")))

    rows = []
    for scope, sub0 in (("all six models", df), ("round two only", df[df["round"] == 2]), ("round one only", df[df["round"] == 1])):
        # within-conversation: subtract each conversation's mean so that easy and hard conversations do not drive the result
        for mc in METRICS + [LENGTH]:
            for d in DIMS + ["composite"]:
                sub = sub0[list(dict.fromkeys([mc, d, "conversation", LENGTH]))].dropna()
                rho = stats.spearmanr(sub[mc], sub[d]).correlation
                lo, hi = cluster_boot(sub, lambda s: stats.spearmanr(s[mc], s[d]).correlation, B=1000)
                dm = sub[[mc, d]] - sub.groupby("conversation")[[mc, d]].transform("mean")
                row = dict(scope=scope, metric=mc, dimension=d, n=len(sub), rho=round(rho,2), ci_lo=round(lo,2), ci_hi=round(hi,2),
                           rho_within_conversation=round(stats.spearmanr(dm[mc], dm[d]).correlation,2))
                if mc != LENGTH:
                    row["partial_rho_len"] = round(partial_spearman(sub[mc].values, sub[d].values, sub[LENGTH].values),2)
                    plo, phi = cluster_boot(sub, lambda s: partial_spearman(s[mc].values, s[d].values, s[LENGTH].values), B=1000)
                    row.update(p_ci_lo=round(plo,2), p_ci_hi=round(phi,2))
                rows.append(row)
    cor = pd.DataFrame(rows); cor.to_csv(f"{out}/round2_metric_clinician_correlations{tag}.csv", index=False)
    show = cor[cor.metric.isin(["bert_f1","sbert_similarity","rouge_l","medcat_entity_recall","medcat_entity_precision","medcat_entity_f1","scispacy_entity_recall",LENGTH])
               & cor.dimension.isin(["accuracy","completeness","conciseness","clinical_utility","overall_quality","composite"])]
    print("\n== Note-level Spearman (95% CI from conversation bootstrap), partial rho adjusts for note length"); print(show.to_string(index=False))

    # ---------- CLMM on unaveraged ratings ----------
    long = allr.merge(met, on=["conversation","model"]); zlen = z(np.log(long[LENGTH])).values
    fx = [(long.rater == r).astype(float).values for r in raters[1:]] + [(long["round"] == 2).astype(float).values]
    base = np.column_stack([zlen] + fx); crow = []
    for mc in ("bert_f1","sbert_similarity","rouge_l","medcat_entity_recall","medcat_entity_precision","medcat_entity_f1","scispacy_entity_recall"):
        zm = z(long[mc]).values
        for d in DIMS:
            f = clmm_x(long[d].values, np.column_stack([zm, base]), long.conversation.values); f0 = clmm_x(long[d].values, base, long.conversation.values)
            b, s = f["beta"][0], f["se"][0]; lr = 2*(f0["nll"] - f["nll"])
            crow.append(dict(metric=mc, dimension=d, n=len(long), or_per_sd=round(np.exp(b),2), ci_lo=round(np.exp(b-1.96*s),2), ci_hi=round(np.exp(b+1.96*s),2),
                             p=fp(stats.chi2.sf(max(lr,0),1)), or_round2=round(np.exp(f["beta"][-1]),2), converged=f["converged"]))
    c = pd.DataFrame(crow); c.to_csv(f"{out}/round2_metric_clmm_odds_ratios{tag}.csv", index=False)
    print("\n== CLMM: OR for a higher rating per 1 SD of the metric, adjusted for log length, rater and round; conversation random"); print(c.to_string(index=False))

if __name__ == "__main__":
    main()
