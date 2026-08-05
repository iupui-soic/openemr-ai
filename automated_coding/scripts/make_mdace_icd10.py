"""Build the ICD-10 gold set from MDACE Profee annotations + MIMIC-III NOTEEVENTS.

Mirrors the CPT benchmark's data shape (note_id, text, gold_codes) but filters
annotations to code_system == "ICD-10-CM" instead of "CPT".

Output:
  dataset/all_icd10.parquet   columns: note_id, text, icd10_codes
  dataset/codes_icd10.json    {code: description}
"""
import json
import sys
from pathlib import Path

import pandas as pd
import polars as pl

MDACE_PROFEE_ICD10 = Path.home() / "icd10-benchmark/explainable-medical-coding/data/raw/MDace/Profee/ICD-10/1.0"
NOTEEVENTS = Path("/data0/mimic-iii-1.4/NOTEEVENTS.csv.gz")
OUT_DIR = Path.home() / "icd10-benchmark/openemr-ai/automated_coding/dataset"

def collect_annotations():
    """Return {note_id: set(icd10_codes)} and {code: description} from MDACE JSONs."""
    note_codes: dict[str, set[str]] = {}
    descriptions: dict[str, str] = {}
    n_charts = 0
    for json_path in sorted(MDACE_PROFEE_ICD10.glob("*.json")):
        case = json.loads(json_path.read_text(encoding="utf8"))
        n_charts += 1
        for note in case["notes"]:
            note_id = str(note["note_id"])
            for ann in note["annotations"]:
                if ann["code_system"] != "ICD-10-CM":
                    continue
                code = ann["code"]
                note_codes.setdefault(note_id, set()).add(code)
                desc = ann.get("description", "").strip()
                if desc:
                    descriptions.setdefault(code, desc)
    print(f"Read {n_charts} charts, {len(note_codes)} notes with ICD-10 codes, "
          f"{len(descriptions)} unique ICD-10 codes")
    return note_codes, descriptions

def load_note_texts(needed_ids: set[str]) -> dict[str, str]:
    """Stream NOTEEVENTS, return {row_id: text} for the note_ids we need."""
    texts: dict[str, str] = {}
    reader = pd.read_csv(
        NOTEEVENTS, compression="gzip", chunksize=50000,
        usecols=["ROW_ID", "TEXT"], dtype={"ROW_ID": str},
    )
    for chunk in reader:
        hit = chunk[chunk["ROW_ID"].isin(needed_ids)]
        for _, r in hit.iterrows():
            texts[r["ROW_ID"]] = r["TEXT"]
        if len(texts) == len(needed_ids):
            break
    print(f"Matched {len(texts)}/{len(needed_ids)} note texts from NOTEEVENTS")
    return texts

def main():
    note_codes, descriptions = collect_annotations()
    texts = load_note_texts(set(note_codes.keys()))

    rows = []
    missing = []
    for note_id, codes in note_codes.items():
        if note_id not in texts:
            missing.append(note_id)
            continue
        rows.append({
            "note_id": note_id,
            "text": texts[note_id],
            "icd10_codes": sorted(codes),
        })
    if missing:
        print(f"WARNING: {len(missing)} notes had no text match: {missing[:10]}")

    df = pl.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_DIR / "all_icd10.parquet")
    (OUT_DIR / "codes_icd10.json").write_text(
        json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf8")

    all_codes = sorted({c for cs in note_codes.values() for c in cs})
    print(f"\nWrote {len(rows)} notes to all_icd10.parquet")
    print(f"Label space: {len(all_codes)} unique ICD-10 codes")
    print(f"Sample codes: {all_codes[:10]}")

if __name__ == "__main__":
    main()
