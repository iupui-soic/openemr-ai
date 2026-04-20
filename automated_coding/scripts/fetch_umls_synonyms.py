"""Fetch UMLS synonyms for every CPT code in the benchmark label space.

For each CPT code:
  1) GET /rest/content/current/source/CPT/<code> → resolves CUI via concepts link.
  2) GET /rest/search/current?string=<code>&inputType=sourceUi&sabs=CPT
       &returnIdType=concept → CUI.
  3) GET /rest/content/current/CUI/<CUI>/atoms?pageSize=200 → all atoms.
  4) Filter to useful term types, dedup case-insensitively, write JSON.

Uses the UTS API key directly via ?apiKey=.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import urllib.request

ROOT = Path(__file__).resolve().parent.parent
CODES_JSON = ROOT / "dataset" / "codes.json"
OUT_JSON = ROOT / "dataset" / "cpt_synonyms.json"
API = "https://uts-ws.nlm.nih.gov/rest"

# Term-type filter: keep clinical/synonym/preferred variants; drop obsolete.
KEEP_TTY = {
    "PT",      # preferred term
    "SY",      # synonym
    "FN",      # full form
    "ETCF",    # entry term for clinical finding
    "ETCLIN",  # clinician-friendly entry term
    "AB",      # abbreviation
    "SYGB",    # British variant
    "MTH_SY",  # metathesaurus synonym
    "LO",      # lexical variant
    "LPN",     # preferred name of long common
    "DN",      # display name
    "CE",      # clinical entry
    "OAF",     # obsolete active fully-specified (sometimes still useful)
}

# Sources we *want* — clinical/procedure terminology rich with synonyms.
KEEP_SAB = {
    "CPT",
    "HCPT",        # CPT-HCPCS
    "HCPCS",
    "SNOMEDCT_US",
    "MEDCIN",
    "ICD10PCS",
    "ICD9CM",
    "MSH",         # MeSH
    "LNC",         # LOINC (for lab-ish procedures)
    "MTH",         # metathesaurus
}


def _get(url: str, apikey: str) -> dict:
    sep = "&" if "?" in url else "?"
    full = f"{url}{sep}apiKey={apikey}"
    with urllib.request.urlopen(full, timeout=30) as resp:
        return json.loads(resp.read().decode("utf8"))


def find_cui(code: str, apikey: str) -> str | None:
    q = urlencode(
        {
            "string": code,
            "inputType": "sourceUi",
            "sabs": "CPT",
            "returnIdType": "concept",
            "searchType": "exact",
        }
    )
    data = _get(f"{API}/search/current?{q}", apikey)
    results = data.get("result", {}).get("results", [])
    for r in results:
        if r.get("rootSource") == "CPT" and r.get("ui", "").startswith("C"):
            return r["ui"]
    if results:
        ui = results[0].get("ui", "")
        if ui.startswith("C"):
            return ui
    return None


def fetch_atoms(cui: str, apikey: str) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        url = f"{API}/content/current/CUI/{cui}/atoms?pageSize=200&pageNumber={page}"
        try:
            data = _get(url, apikey)
        except Exception as exc:
            print(f"  atoms page {page} for {cui} failed: {exc}", file=sys.stderr)
            break
        results = data.get("result", [])
        if not results:
            break
        out.extend(results)
        if len(results) < 200:
            break
        page += 1
        if page > 20:  # safety
            break
    return out


def clean_name(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" .,;:-")
    return s


def main() -> int:
    apikey = os.environ.get("UMLS_API_KEY") or sys.argv[1] if len(sys.argv) > 1 else None
    if not apikey:
        apikey = os.environ.get("UMLS_API_KEY")
    if not apikey:
        raise SystemExit("set UMLS_API_KEY or pass as first arg")

    codes = json.loads(CODES_JSON.read_text(encoding="utf8"))
    out: dict[str, dict] = {}

    for i, (code, desc) in enumerate(sorted(codes.items()), 1):
        print(f"[{i}/{len(codes)}] {code} — {desc[:60]}")
        cui = find_cui(code, apikey)
        if not cui:
            print("  no CUI found")
            out[code] = {"cui": None, "synonyms": []}
            continue
        atoms = fetch_atoms(cui, apikey)
        synonyms: list[str] = []
        seen: set[str] = set()
        for a in atoms:
            name = clean_name(a.get("name", ""))
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            sab = a.get("rootSource", "")
            tty = a.get("termType", "")
            if sab not in KEEP_SAB:
                continue
            if tty and tty not in KEEP_TTY:
                continue
            # UTS returns these as strings "true"/"false"
            if str(a.get("obsolete", "false")).lower() == "true":
                continue
            if str(a.get("suppressible", "false")).lower() == "true":
                continue
            seen.add(key)
            synonyms.append(name)
        out[code] = {"cui": cui, "synonyms": synonyms}
        print(f"  CUI={cui}  kept {len(synonyms)} synonyms")
        time.sleep(0.1)  # be polite

    OUT_JSON.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf8"
    )
    total = sum(len(v["synonyms"]) for v in out.values())
    print(f"\nwrote {OUT_JSON}  |  {total} synonyms across {len(out)} codes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
