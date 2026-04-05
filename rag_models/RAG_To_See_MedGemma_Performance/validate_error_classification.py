"""Validate keyword-based error classification against LLM classification on 60 stratified samples."""

import pandas as pd
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

key = pd.read_csv(os.path.join(BASE_DIR, "rating_packets/ANSWER_KEY_DO_NOT_SHARE.csv"))
frames = []
for r in ["rater_1", "rater_2", "rater_3"]:
    df = pd.read_csv(os.path.join(BASE_DIR, f"rating_packets/ratings_{r}.csv"))
    df["rater"] = r
    frames.append(df)
merged = pd.concat(frames, ignore_index=True)
merged = merged.merge(key[["summary_id", "conversation", "model"]], on="summary_id", how="left")

# Keyword classification
ehr_kw = ["mismatched", "mismatch", "conflat", "conflicting ehr", "unrelated ehr", "outdated",
           "elderly ehr", "female ehr", "gender mismatch", "age error", "109yo",
           "expired meds", "archived", "irrelevant ehr", "different patient"]
halluc_kw = ["fabricat", "hallucin", "invented", "unsupported", "unstated", "unmentioned",
             "added unmentioned", "unconfirmed", "speculative"]
correct_kw = ["correctly ignored", "correctly excluded", "correctly noted mismatch",
              "correctly identified", "appropriately ignore"]

merged["c_lower"] = merged["comments"].fillna("").str.lower()
merged["auto_ehr"] = merged["c_lower"].apply(lambda c: any(k in c for k in ehr_kw))
merged["auto_hal"] = merged["c_lower"].apply(lambda c: any(k in c for k in halluc_kw))
merged["auto_correct"] = merged["c_lower"].apply(lambda c: any(k in c for k in correct_kw))
merged["kw_class"] = "none"
merged.loc[merged["auto_ehr"] & ~merged["auto_hal"], "kw_class"] = "ehr_mismatch"
merged.loc[~merged["auto_ehr"] & merged["auto_hal"], "kw_class"] = "hallucination"
merged.loc[merged["auto_ehr"] & merged["auto_hal"], "kw_class"] = "both"
merged.loc[merged["auto_correct"], "kw_class"] = "correctly_handled"

# Stratified sample
sample = merged.groupby("kw_class", group_keys=False).apply(
    lambda x: x.sample(min(12, len(x)), random_state=42)
).reset_index(drop=True)

client = Groq()

prompt_template = """Classify this clinician comment about an AI-generated medical summary into exactly ONE category:

- ehr_mismatch: Error from importing data from a mismatched/different patient's EHR
- hallucination: Error from AI fabricating content not in transcript or EHR
- both: Both EHR mismatch AND hallucination errors present
- correctly_handled: AI correctly ignored mismatched EHR data
- none: No specific error type, or other issues (verbosity, omissions, formatting)

Comment: "{comment}"

Respond with ONLY the category name:"""

results = []
for _, row in sample.iterrows():
    comment = row["comments"]
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt_template.format(comment=comment)}],
            temperature=0.1, max_tokens=10,
        )
        llm_raw = r.choices[0].message.content.strip().lower()
        if "ehr" in llm_raw and "both" not in llm_raw:
            llm_class = "ehr_mismatch"
        elif "halluc" in llm_raw and "both" not in llm_raw:
            llm_class = "hallucination"
        elif "both" in llm_raw:
            llm_class = "both"
        elif "correct" in llm_raw:
            llm_class = "correctly_handled"
        else:
            llm_class = "none"
        results.append({
            "summary_id": row["summary_id"],
            "comment": comment[:80],
            "kw_class": row["kw_class"],
            "llm_class": llm_class,
        })
        time.sleep(0.3)
    except Exception as e:
        print(f"Error: {e}")

results_df = pd.DataFrame(results)
agreement = (results_df["kw_class"] == results_df["llm_class"]).mean()
print(f"Keyword vs LLM agreement on {len(results_df)} samples: {agreement:.1%}")
print()

# Confusion
print("CONFUSION MATRIX (keyword rows x LLM columns):")
ct = pd.crosstab(results_df["kw_class"], results_df["llm_class"], margins=True)
print(ct)
print()

# Per-category
print("PER-CATEGORY AGREEMENT:")
for cat in ["ehr_mismatch", "hallucination", "both", "correctly_handled", "none"]:
    sub = results_df[results_df["kw_class"] == cat]
    if len(sub) > 0:
        match = (sub["kw_class"] == sub["llm_class"]).sum()
        print(f"  {cat:20s}: {match}/{len(sub)} ({match/len(sub)*100:.0f}%)")

# Disagreements
print()
print("DISAGREEMENTS:")
disagree = results_df[results_df["kw_class"] != results_df["llm_class"]]
for _, row in disagree.iterrows():
    print(f'  [{row["summary_id"]}] kw={row["kw_class"]:18s} llm={row["llm_class"]:18s} "{row["comment"]}"')
