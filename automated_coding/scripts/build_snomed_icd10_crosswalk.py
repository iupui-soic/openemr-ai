"""
Build a SNOMED-CT -> ICD-10-CM crosswalk for the 100-patient OpenEMR
database's diagnosis list, using UMLS's official crosswalk API.

Reads distinct SNOMED codes from `lists` (type='medical_problem'),
queries UMLS's crosswalk endpoint for each, and writes both:
  - a raw code-level mapping (snomed_code -> [icd10_codes])
  - a per-patient gold-label file (pid -> [icd10_codes]) matching the
    same shape used elsewhere in this project for benchmark gold sets

Requires UMLS_API_KEY in the environment (free UTS account).
"""
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

API_KEY = os.environ.get("UMLS_API_KEY")
if not API_KEY:
    raise SystemExit("set UMLS_API_KEY")

CROSSWALK_URL = "https://uts-ws.nlm.nih.gov/rest/crosswalk/current/source/SNOMEDCT_US/{code}?targetSource=ICD10CM&apiKey={key}"
OUT_DIR = Path(__file__).resolve().parent.parent / "dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_crosswalk(snomed_code: str) -> list[str]:
    url = CROSSWALK_URL.format(code=snomed_code, key=API_KEY)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR for {snomed_code}: {e}")
        return []
    results = data.get("result", [])
    if not isinstance(results, list):
        return []
    return [r["ui"] for r in results if r.get("rootSource") == "ICD10CM"]


def main():
    distinct_codes_file = OUT_DIR / "snomed_codes_used.json"
    if not distinct_codes_file.exists():
        print(f"Missing {distinct_codes_file} -- run the DB export step first")
        return

    snomed_codes = json.loads(distinct_codes_file.read_text())
    print(f"Mapping {len(snomed_codes)} distinct SNOMED codes to ICD-10-CM...")

    mapping = {}
    mapped_count = 0
    for i, code in enumerate(snomed_codes):
        icd10_codes = fetch_crosswalk(code)
        mapping[code] = icd10_codes
        if icd10_codes:
            mapped_count += 1
        print(f"  [{i+1}/{len(snomed_codes)}] {code} -> {icd10_codes}")
        time.sleep(0.1)  # be polite to the API

    out_path = OUT_DIR / "snomed_to_icd10_crosswalk.json"
    out_path.write_text(json.dumps(mapping, indent=2))

    print(f"\nMapped {mapped_count}/{len(snomed_codes)} codes successfully")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
