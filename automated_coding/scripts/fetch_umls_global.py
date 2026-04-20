"""Richer CPT label enrichment via UMLS GLOBAL search.

Rationale: `fetch_umls_synonyms.py` goes CPT code → CPT-native CUI → atoms.
That only works if the CUI has cross-references to SNOMED/LOINC/MEDCIN etc.
For CPT codes introduced in 2018+ (many of our radiology codes), the
CPT-native CUI is THIN — UMLS hasn't back-mapped it yet.

Workaround: search UMLS globally by the CPT description text. The top hits
include not just the CPT-native CUI but also deprecated predecessors and
equivalent concepts in SNOMED/LOINC/MEDCIN that have been mapped since the
mid-2000s. Those "richer" CUIs have many more synonyms.

For each CPT code we:
  1. Search UMLS by description text (restricted to Diagnostic Procedure
     semantic types to reduce noise).
  2. Keep top-3 candidate CUIs whose name is substring-similar to the CPT
     description.
  3. Fetch all atoms for each candidate CUI and union the synonyms.
  4. Deduplicate and drop variants that are suspiciously unrelated
     (Jaccard word-overlap < 0.3 with original description).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import urllib.request

ROOT = Path(__file__).resolve().parent.parent
CODES_JSON = ROOT / "dataset" / "codes.json"
OUT_JSON = ROOT / "dataset" / "cpt_synonyms_global.json"
API = "https://uts-ws.nlm.nih.gov/rest"

KEEP_TTY = {
    "PT", "SY", "FN", "ETCF", "ETCLIN", "AB", "LO", "LPN",
    "MTH_SY", "OAP", "RPT", "SYGB", "CE",
}
KEEP_SAB = {
    "CPT", "HCPT", "HCPCS", "SNOMEDCT_US", "MEDCIN", "ICD10PCS", "ICD9CM",
    "MSH", "LNC", "MTH", "RADLEX", "NCI", "NCI_FDA", "NCI_NCPDP",
    "RXNORM",
}
SEMANTIC_TYPES_OK = {
    "Diagnostic Procedure", "Therapeutic or Preventive Procedure",
    "Laboratory Procedure", "Health Care Activity", "Medical Device",
}

MAX_CANDIDATE_CUIS = 3
MIN_NAME_JACCARD = 0.30

# Reuse the robust modifier taxonomy we already built for the disambiguator.
sys.path.insert(0, str(ROOT.parent))
from automated_coding.approaches.generic_modifiers import _modifier_labels_for  # noqa: E402


# Also handle billing-short-hand forms of contrast modifiers that aren't
# caught by generic_modifiers (which was designed for full English text).
# Order matters: most-specific patterns first so "W/O & W/" wins over "W/O".
_SHORTHAND_RX = [
    (re.compile(r"\bw/o\s*&\s*w/", re.I), "both"),
    (re.compile(r"\bw/o\s+and\s+w/", re.I), "both"),
    (re.compile(r"\bw/o\b", re.I), "without"),
    (re.compile(r"\bw/\s*contrast\b", re.I), "with"),
]


def _modifier_labels_with_shorthand(text: str) -> dict[str, str]:
    """Extend generic_modifiers with billing shorthand like 'w/o' and 'W/'."""
    labels = dict(_modifier_labels_for(text))
    if "contrast" not in labels:
        # Scan shorthand patterns in order (most specific first)
        for rx, label in _SHORTHAND_RX:
            if rx.search(text):
                labels["contrast"] = label
                break
    return labels


def _modifier_compatible(target_desc: str, candidate: str) -> bool:
    """Reject candidate if it has a DIFFERENT modifier label than the target
    in any modifier family. Missing modifier is OK (candidate is less specific)."""
    target = _modifier_labels_with_shorthand(target_desc)
    if not target:
        return True
    cand = _modifier_labels_with_shorthand(candidate)
    for fam, tlabel in target.items():
        clabel = cand.get(fam)
        if clabel is not None and clabel != tlabel:
            return False
    return True

logger = logging.getLogger(__name__)

_STOP = {"a", "an", "the", "and", "or", "of", "with", "without", "to", "for",
         "in", "on", "by", "from", "at"}


def _tokens(s: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", s)
            if w.lower() not in _STOP and len(w) > 2}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _get(url: str, apikey: str) -> dict:
    sep = "&" if "?" in url else "?"
    full = f"{url}{sep}apiKey={apikey}"
    with urllib.request.urlopen(full, timeout=30) as resp:
        return json.loads(resp.read().decode("utf8"))


def _clean_desc(desc: str) -> str:
    # Strip parentheticals and the part after ';' for a cleaner search query
    s = re.sub(r"\([^)]*\)", " ", desc)
    s = s.split(";", 1)[0]
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_candidate_cuis(code: str, desc: str, apikey: str) -> list[str]:
    """Search UMLS by description text, return top CUIs filtered by similarity."""
    clean = _clean_desc(desc)
    if not clean:
        return []
    url = (
        f"{API}/search/current?" +
        urlencode({
            "string": clean,
            "returnIdType": "concept",
            "pageSize": 20,
        })
    )
    try:
        data = _get(url, apikey)
    except Exception as exc:
        print(f"  search failed for {code}: {exc}", file=sys.stderr)
        return []
    results = data.get("result", {}).get("results", [])
    cuis: list[tuple[str, float]] = []
    for r in results:
        ui = r.get("ui", "")
        if not ui.startswith("C"):
            continue
        name = r.get("name", "")
        # Require semantic-type sanity
        sem = set(r.get("semanticTypes", []) or [])
        if sem and not (sem & SEMANTIC_TYPES_OK):
            continue
        sim = _jaccard(name, desc)
        if sim >= MIN_NAME_JACCARD:
            cuis.append((ui, sim))
    # Sort by similarity and keep distinct
    cuis.sort(key=lambda r: -r[1])
    seen: set[str] = set()
    top: list[str] = []
    for ui, _ in cuis:
        if ui in seen:
            continue
        seen.add(ui)
        top.append(ui)
        if len(top) >= MAX_CANDIDATE_CUIS:
            break
    return top


def fetch_atoms(cui: str, apikey: str) -> list[dict]:
    url = f"{API}/content/current/CUI/{cui}/atoms?pageSize=200"
    try:
        data = _get(url, apikey)
    except Exception as exc:
        print(f"  atoms failed for {cui}: {exc}", file=sys.stderr)
        return []
    return data.get("result", [])


def atom_is_ok(a: dict) -> bool:
    if str(a.get("obsolete", "false")).lower() == "true":
        return False
    if str(a.get("suppressible", "false")).lower() == "true":
        return False
    if a.get("rootSource") not in KEEP_SAB:
        return False
    tty = a.get("termType", "")
    if tty and tty not in KEEP_TTY:
        return False
    if a.get("language", "ENG") != "ENG":
        return False
    return True


def clean_name(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,;:-")


def main() -> int:
    apikey = os.environ.get("UMLS_API_KEY")
    if not apikey and len(sys.argv) > 1:
        apikey = sys.argv[1]
    if not apikey:
        raise SystemExit("set UMLS_API_KEY")

    codes = json.loads(CODES_JSON.read_text(encoding="utf8"))
    out: dict[str, dict] = {}

    for i, (code, desc) in enumerate(sorted(codes.items()), 1):
        print(f"[{i}/{len(codes)}] {code} — {desc[:60]}")
        cuis = find_candidate_cuis(code, desc, apikey)
        syns_seen: set[str] = set()
        syns: list[str] = []
        # Require high similarity between the synonym and the original desc
        # — prevents pulling unrelated concepts through loose sim
        for cui in cuis:
            atoms = fetch_atoms(cui, apikey)
            for a in atoms:
                if not atom_is_ok(a):
                    continue
                name = clean_name(a.get("name", ""))
                if not name:
                    continue
                k = name.lower()
                if k in syns_seen:
                    continue
                # Require word-overlap with the original description
                if _jaccard(name, desc) < MIN_NAME_JACCARD:
                    continue
                # Reject sibling-modifier clashes (e.g. "without" vs "with")
                if not _modifier_compatible(desc, name):
                    continue
                syns_seen.add(k)
                syns.append(name)
            time.sleep(0.05)
        out[code] = {"cuis": cuis, "synonyms": syns}
        print(f"  CUIs={cuis}  kept {len(syns)} synonyms")

    OUT_JSON.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf8"
    )
    total = sum(len(v["synonyms"]) for v in out.values())
    print(f"\nwrote {OUT_JSON}  |  {total} synonyms across {len(out)} codes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
