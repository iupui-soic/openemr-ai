"""
Retrieval similarity versus physician ratings.

Retrieval similarity is the cosine similarity (all-MiniLM-L6-v2) between the detected condition of a
conversation and the diagnosis list of the top retrieved schema: the same call as
GroqSummarizer._retrieve_schemas in rag_models/pipeline/run_fareez_summaries.py.

Needs the local Chroma store (credentialed, MIMIC-derived; not in git). Only the diagnosis-list
metadata is read, and nothing leaves the machine.

Outputs in rag_models/results/revision/:
  retrieval_similarity_by_conversation.csv   one row per conversation (no schema text)
  retrieval_similarity_vs_ratings.csv        Spearman rho with conversation-clustered bootstrap CIs

Run: python scripts/revision_retrieval_similarity.py --repo . --chroma /path/to/chroma_schema_improved
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats

DIMS = ["accuracy", "completeness", "organization", "conciseness", "clinical_utility", "overall_quality"]
MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3-32b", "medgemma-4b"]


def boot_ci(df, x, y, B=2000, seed=2026):
    rng = np.random.default_rng(seed); convs = df.conversation.unique()
    groups = {c: df[df.conversation == c] for c in convs}; out = []
    for _ in range(B):
        s = pd.concat([groups[c] for c in rng.choice(convs, len(convs))])
        out.append(stats.spearmanr(s[x], s[y]).correlation if s[x].nunique() > 1 and s[y].nunique() > 1 else np.nan)
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=".")
    ap.add_argument("--chroma", default=None, help="path to chroma_schema_improved (default rag_models/vectorDB/chroma_schema_improved)")
    a = ap.parse_args(); R = a.repo.rstrip("/") + "/rag_models"; out = f"{R}/results/revision"
    import chromadb
    from sentence_transformers import SentenceTransformer, util
    client = chromadb.PersistentClient(path=a.chroma or f"{R}/vectorDB/chroma_schema_improved")
    cols = client.list_collections(); assert len(cols) == 1, cols
    col = client.get_collection(cols[0].name if hasattr(cols[0], "name") else cols[0])
    metas = col.get(include=["metadatas"])["metadatas"]
    diseases = [m.get("diseases", "Unspecified") for m in metas]
    print(f"Schemas: {len(diseases)}; distinct diagnosis lists: {len(set(diseases))}")
    sb = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    cand = sb.encode(diseases, convert_to_tensor=True)
    sel = json.load(open(f"{R}/data/fareez_selected_40.json"))
    rows = []
    for s in sel:
        sc = util.cos_sim(sb.encode(s["detected_condition"], convert_to_tensor=True), cand)[0]
        top = sc.topk(2)
        rows.append(dict(conversation=s["file_name"], specialty=s["category"], detected_condition=s["detected_condition"],
                         sim_top1=float(top.values[0]), sim_top2=float(top.values[1])))
    sim = pd.DataFrame(rows); sim["sim_mean_top2"] = sim[["sim_top1", "sim_top2"]].mean(axis=1)
    sim.to_csv(f"{out}/retrieval_similarity_by_conversation.csv", index=False)
    u = sim.drop_duplicates("detected_condition")
    print(f"Unique conditions: {len(u)}; top-1 similarity mean {u.sim_top1.mean():.3f} (population SD {u.sim_top1.std(ddof=0):.3f}, sample SD {u.sim_top1.std():.3f}); "
          f">=0.95: {sorted(u[u.sim_top1 >= .95].detected_condition)}; <0.50: {int((u.sim_top1 < .5).sum())}")
    print("Check: mean 0.625 (SD 0.161) over 25 unique conditions")
    print(f"Over 40 conversations: mean {sim.sim_top1.mean():.3f} (SD {sim.sim_top1.std():.3f}), range {sim.sim_top1.min():.2f} to {sim.sim_top1.max():.2f}")

    key = pd.read_csv(f"{R}/rating_packets/ANSWER_KEY_DO_NOT_SHARE.csv")
    rat = pd.concat([pd.read_csv(f"{R}/rating_packets/ratings_rater_{i}.csv").assign(rater=f"rater_{i}") for i in (1, 2, 3)])
    rat = rat.merge(key, on=["summary_id", "rater"], validate="one_to_one")
    rat = rat[~((rat.conversation == "MSK0037") & (rat.model == "medgemma-4b"))]
    clin = rat.groupby(["conversation", "model"])[DIMS].mean().reset_index(); clin["composite"] = clin[DIMS].mean(axis=1)
    df = clin.merge(sim, on="conversation"); assert len(df) == 159
    res = []
    for scope in ["all"] + MODELS:
        sub = df if scope == "all" else df[df.model == scope]
        for d in DIMS + ["composite"]:
            if sub[d].nunique() < 2:
                res.append(dict(scope=scope, dimension=d, n=len(sub), rho=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)); continue
            r = stats.spearmanr(sub.sim_top1, sub[d]); lo, hi = boot_ci(sub, "sim_top1", d)
            # the P value ignores clustering in the pooled analysis; the bootstrap CI is the primary result
            res.append(dict(scope=scope, dimension=d, n=len(sub), rho=round(r.correlation, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2), p=r.pvalue))
    # conversation level: mean over the four models and three raters (one point per conversation)
    cm = df.groupby("conversation").agg({**{d: "mean" for d in DIMS + ["composite"]}, "sim_top1": "first"}).reset_index()
    for d in DIMS + ["composite"]:
        r = stats.spearmanr(cm.sim_top1, cm[d]); lo, hi = boot_ci(cm, "sim_top1", d)
        res.append(dict(scope="conversation_mean", dimension=d, n=len(cm), rho=round(r.correlation, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2), p=r.pvalue))
    res = pd.DataFrame(res); res.to_csv(f"{out}/retrieval_similarity_vs_ratings.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
