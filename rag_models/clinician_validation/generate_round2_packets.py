"""
Second rating round: blinded packets for the models added in revision, plus anchor notes.

Each rater receives every note from the new models (40 conversations each) and 15 anchor notes
that the same rater already scored in round one. Anchors are not marked in the packet. They are
used to check whether a rater's use of the scale has shifted between rounds and to estimate
test-retest agreement. Layout, sections and rating sheet are identical to round one.

Anchors: chosen once (seed 2026) from the 159 round-one notes, spread over the four round-one
models and over low, middle and high rater-averaged composite scores, one per conversation.

Usage (from rag_models/):
    python clinician_validation/generate_round2_packets.py
"""
import csv, json, os, random, re, sys
import pandas as pd

RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAG_ROOT, "clinician_validation"))
from generate_rating_packets import load_openemr_extract, load_summary, SUMMARIES_DIR  # noqa: E402

NEW_MODELS = ["qwen3.8-27b", "llama-4-scout"]
ROUND1_MODELS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3-32b", "medgemma-4b"]
ANCHORS_PER_MODEL = {"gpt-oss-120b": 4, "gpt-oss-20b": 4, "qwen3-32b": 4, "medgemma-4b": 3}
RATER_SEEDS = {"rater_1": 400, "rater_2": 500, "rater_3": 600}
FIRST_ID = 481  # round one used S001 to S480
DIMS = ["accuracy", "completeness", "organization", "conciseness", "clinical_utility", "overall_quality"]


def round1_transcripts():
    text = open(os.path.join(RAG_ROOT, "rating_packets", "packet_rater_1.txt"), encoding="utf-8").read()
    out = {}
    for item in re.split(r"#{80}\n# SUMMARY ", text)[1:]:
        conv = re.search(r"# Conversation: (\w+)", item).group(1)
        out[conv] = re.search(r"SECTION A: ORIGINAL TRANSCRIPT\n-+\n\n(.*?)\n\n-+\nSECTION B", item, re.S).group(1)
    return out


def choose_anchors():
    r = pd.read_csv(os.path.join(RAG_ROOT, "results", "revision", "ratings_long.csv"))
    r = r[r.failed_generation == 0]
    note = r.assign(comp=r[DIMS].mean(axis=1)).groupby(["conversation", "model"]).comp.mean().reset_index()
    rng = random.Random(2026)
    used, anchors = set(), []
    for m in ROUND1_MODELS:
        d = note[note.model == m].sort_values(["comp", "conversation"]).reset_index(drop=True)
        k = ANCHORS_PER_MODEL[m]
        bands = [d.iloc[i * len(d) // k:(i + 1) * len(d) // k] for i in range(k)]  # low to high composite
        for b in bands:
            cands = [c for c in b.conversation if c not in used]
            c = rng.choice(cands)
            used.add(c); anchors.append((c, m))
    return anchors


def shuffled(items, seed):
    rng = random.Random(seed)
    for _ in range(5000):
        rng.shuffle(items)
        if all(items[i][0] != items[i - 1][0] for i in range(1, len(items))):
            return list(items)
    raise RuntimeError("could not find an order without adjacent notes from one conversation")


def main():
    out_dir = os.path.join(RAG_ROOT, "rating_packets_round2"); os.makedirs(out_dir, exist_ok=True)
    selection = json.load(open(os.path.join(RAG_ROOT, "data", "fareez_selected_40.json")))
    meta = {s["file_name"]: s for s in selection}
    convs = list(meta)
    for m in NEW_MODELS:
        n = sum(load_summary(m, c) is not None for c in convs)
        assert n == 40, f"{m} has {n}/40 notes"
    transcripts = round1_transcripts()
    anchors = choose_anchors()
    key1 = pd.read_csv(os.path.join(RAG_ROOT, "rating_packets", "ANSWER_KEY_DO_NOT_SHARE.csv"))
    key1 = {(k.rater, k.conversation, k.model): k.summary_id for k in key1.itertuples()}
    items0 = [(c, m, False) for m in NEW_MODELS for c in convs] + [(c, m, True) for c, m in anchors]
    answer, counter = [], FIRST_ID
    for rater, seed in RATER_SEEDS.items():
        order = shuffled(list(items0), seed)
        L = ["=" * 80, "CLINICIAN RATING PACKET", f"Rater: {rater}", f"Total Summaries: {len(order)}",
             f"Conversations: {len(convs)} (models blinded)", "=" * 80, "", "INSTRUCTIONS:",
             "1. For each summary, read the TRANSCRIPT and OPENEMR DATA first",
             "2. Then read the AI-GENERATED SUMMARY",
             "3. Rate 6 dimensions on a 1-5 scale (see rating instrument)",
             "4. Record ratings in the provided CSV template or on paper",
             "5. Take a 10-minute break every 15-20 summaries", ""]
        rows = []
        for idx, (conv, model, is_anchor) in enumerate(order):
            sid = f"S{counter:03d}"; counter += 1
            answer.append(dict(summary_id=sid, rater=rater, conversation=conv, model=model,
                               specialty=meta[conv].get("category", ""), condition=meta[conv].get("detected_condition", ""),
                               order_position=idx + 1, is_anchor=int(is_anchor),
                               round1_summary_id=key1.get((rater, conv, model), "") if is_anchor else ""))
            rows.append(dict(summary_id=sid, **{d: "" for d in DIMS}, comments=""))
            L += ["", "#" * 80, f"# SUMMARY {sid}  ({idx + 1}/{len(order)})",
                  f"# Conversation: {conv} | Specialty: {meta[conv].get('category', '')} | Condition: {meta[conv].get('detected_condition', '')}",
                  "#" * 80, "", "-" * 60, "SECTION A: ORIGINAL TRANSCRIPT", "-" * 60, "", transcripts[conv], "",
                  "-" * 60, "SECTION B: OPENEMR DATA EXTRACT", "-" * 60, "", load_openemr_extract(conv), "",
                  "-" * 60, f"SECTION C: AI-GENERATED SUMMARY  [{sid}]", "-" * 60, "", load_summary(model, conv), "",
                  "-" * 60, f"RATING SHEET — {sid}", "-" * 60, "", f"Summary ID: {sid}", f"Rater: {rater}", "",
                  "Accuracy:         [ 1 | 2 | 3 | 4 | 5 ]", "Completeness:     [ 1 | 2 | 3 | 4 | 5 ]",
                  "Organization:     [ 1 | 2 | 3 | 4 | 5 ]", "Conciseness:      [ 1 | 2 | 3 | 4 | 5 ]",
                  "Clinical Utility: [ 1 | 2 | 3 | 4 | 5 ]", "Overall Quality:  [ 1 | 2 | 3 | 4 | 5 ]", "",
                  "Comments: ___________________________________________", ""]
        open(os.path.join(out_dir, f"packet_{rater}.txt"), "w", encoding="utf-8").write("\n".join(L))
        with open(os.path.join(out_dir, f"ratings_{rater}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["summary_id"] + DIMS + ["comments"]); w.writeheader(); w.writerows(rows)
        print(f"{rater}: {len(order)} items")
    pd.DataFrame(answer).to_csv(os.path.join(out_dir, "ANSWER_KEY_DO_NOT_SHARE.csv"), index=False)
    print("anchors:", anchors)


if __name__ == "__main__":
    main()
