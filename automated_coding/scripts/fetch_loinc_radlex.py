"""Enrich CPT labels with LOINC (via UTS) and RadLex (via BioPortal).

LOINC:
  UMLS includes LOINC as SAB='LNC'. We search the LNC source by the CPT
  description text, collect the top matching LNC concept(s), and pull all
  their atoms. LOINC atoms include Long_Common_Name (LPDN), Shortname (LN),
  Related Names 2 (LNC_MTH names).

RadLex:
  Not in UMLS. Must go via BioPortal REST API. Requires a free API key set
  in BIOPORTAL_API_KEY env var (register at bioportal.bioontology.org).
  If key absent, RadLex step is skipped.

Output:
  dataset/cpt_synonyms_loinc.json          — LOINC synonyms per CPT code
  dataset/cpt_synonyms_radlex.json         — RadLex synonyms per CPT code
  dataset/cpt_synonyms_all.json            — merged (targeted + global + LOINC + RadLex)
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
OUT_LOINC = ROOT / "dataset" / "cpt_synonyms_loinc.json"
OUT_RADLEX = ROOT / "dataset" / "cpt_synonyms_radlex.json"
OUT_ALL = ROOT / "dataset" / "cpt_synonyms_all.json"
IN_TARGETED = ROOT / "dataset" / "cpt_synonyms.json"
IN_GLOBAL = ROOT / "dataset" / "cpt_synonyms_global.json"

UTS_API = "https://uts-ws.nlm.nih.gov/rest"
BIOPORTAL_API = "https://data.bioontology.org"

sys.path.insert(0, str(ROOT.parent))
from automated_coding.approaches.generic_modifiers import _modifier_labels_for  # noqa: E402

logger = logging.getLogger(__name__)

MIN_JACCARD = 0.25
LOINC_MAX_CUIS = 3
RADLEX_MAX_HITS = 5

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


_SHORTHAND_RX = [
    (re.compile(r"\bw/o\s*&\s*w/", re.I), ("contrast", "both")),
    (re.compile(r"\bw/o\s+and\s+w/", re.I), ("contrast", "both")),
    (re.compile(r"\bw/o\b", re.I), ("contrast", "without")),
    (re.compile(r"\bw/\s*contrast\b", re.I), ("contrast", "with")),
]


def _mods(text: str) -> dict[str, str]:
    out = dict(_modifier_labels_for(text))
    if "contrast" not in out:
        for rx, (fam, label) in _SHORTHAND_RX:
            if rx.search(text):
                out[fam] = label
                break
    return out


def _modifier_compatible(target: str, cand: str) -> bool:
    tm = _mods(target)
    if not tm:
        return True
    cm = _mods(cand)
    for fam, tlabel in tm.items():
        clabel = cm.get(fam)
        if clabel is not None and clabel != tlabel:
            return False
    return True


def _get(url: str, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf8"))


def _clean_query(desc: str) -> str:
    s = re.sub(r"\([^)]*\)", " ", desc)
    s = s.split(";", 1)[0]
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# -------- LOINC (via UTS) ---------------------------------------------------

def fetch_loinc_for_code(
    code: str, desc: str, apikey: str
) -> tuple[list[str], list[str]]:
    """Returns (cuis, synonyms) after search + atom fetch + filter."""
    q = _clean_query(desc)
    if not q:
        return [], []
    url = (
        f"{UTS_API}/search/current?"
        + urlencode({
            "string": q,
            "sabs": "LNC",
            "returnIdType": "concept",
            "pageSize": 20,
        })
        + f"&apiKey={apikey}"
    )
    try:
        data = _get(url)
    except Exception as exc:
        print(f"  loinc search {code}: {exc}", file=sys.stderr)
        return [], []
    candidates = data.get("result", {}).get("results", [])
    # keep top-K most similar
    ranked: list[tuple[str, float]] = []
    for r in candidates:
        ui = r.get("ui", "")
        if not ui.startswith("C"):
            continue
        sim = _jaccard(r.get("name", ""), desc)
        if sim >= MIN_JACCARD:
            ranked.append((ui, sim))
    ranked.sort(key=lambda r: -r[1])
    seen_cuis: set[str] = set()
    top_cuis: list[str] = []
    for ui, _ in ranked:
        if ui in seen_cuis:
            continue
        seen_cuis.add(ui)
        top_cuis.append(ui)
        if len(top_cuis) >= LOINC_MAX_CUIS:
            break

    # Fetch atoms for those CUIs
    syns_seen: set[str] = set()
    syns: list[str] = []
    for cui in top_cuis:
        aurl = (
            f"{UTS_API}/content/current/CUI/{cui}/atoms?pageSize=200&apiKey={apikey}"
        )
        try:
            adata = _get(aurl)
        except Exception as exc:
            print(f"  loinc atoms {cui}: {exc}", file=sys.stderr)
            continue
        for a in adata.get("result", []):
            if str(a.get("obsolete", "false")).lower() == "true":
                continue
            if str(a.get("suppressible", "false")).lower() == "true":
                continue
            if a.get("language", "ENG") != "ENG":
                continue
            name = re.sub(r"\s+", " ", a.get("name", "")).strip(" .,;:-")
            if not name:
                continue
            k = name.lower()
            if k in syns_seen:
                continue
            if _jaccard(name, desc) < MIN_JACCARD:
                continue
            if not _modifier_compatible(desc, name):
                continue
            syns_seen.add(k)
            syns.append(name)
        time.sleep(0.05)
    return top_cuis, syns


# -------- RadLex (via BioPortal) --------------------------------------------

def fetch_radlex_for_code(
    code: str, desc: str, apikey: str
) -> list[str]:
    q = _clean_query(desc)
    if not q:
        return []
    url = (
        f"{BIOPORTAL_API}/search?"
        + urlencode({
            "q": q,
            "ontologies": "RADLEX",
            "pagesize": 10,
            "include": "prefLabel,synonym,definition",
        })
    )
    try:
        data = _get(url, headers={"Authorization": f"apikey token={apikey}"})
    except Exception as exc:
        print(f"  radlex search {code}: {exc}", file=sys.stderr)
        return []

    syns_seen: set[str] = set()
    syns: list[str] = []
    for r in data.get("collection", [])[:RADLEX_MAX_HITS]:
        candidate_names: list[str] = []
        if pref := r.get("prefLabel"):
            candidate_names.append(pref)
        for s in r.get("synonym", []) or []:
            candidate_names.append(s)
        for name in candidate_names:
            name = re.sub(r"\s+", " ", str(name)).strip(" .,;:-")
            if not name:
                continue
            k = name.lower()
            if k in syns_seen:
                continue
            if _jaccard(name, desc) < MIN_JACCARD:
                continue
            if not _modifier_compatible(desc, name):
                continue
            syns_seen.add(k)
            syns.append(name)
    return syns


# -------- orchestrator -------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    umls_key = os.environ.get("UMLS_API_KEY")
    if not umls_key:
        raise SystemExit("UMLS_API_KEY missing")
    bio_key = os.environ.get("BIOPORTAL_API_KEY")
    if not bio_key:
        print("[warn] BIOPORTAL_API_KEY missing — RadLex step will be skipped",
              file=sys.stderr)

    codes = json.loads(CODES_JSON.read_text(encoding="utf8"))
    loinc: dict[str, dict] = {}
    radlex: dict[str, dict] = {}

    for i, (code, desc) in enumerate(sorted(codes.items()), 1):
        print(f"[{i}/{len(codes)}] {code} — {desc[:55]}")
        # LOINC
        cuis, syns = fetch_loinc_for_code(code, desc, umls_key)
        loinc[code] = {"cuis": cuis, "synonyms": syns}
        print(f"  LOINC  CUIs={cuis}  kept {len(syns)} synonyms")
        # RadLex
        if bio_key:
            rsyns = fetch_radlex_for_code(code, desc, bio_key)
            radlex[code] = {"synonyms": rsyns}
            print(f"  RADLEX kept {len(rsyns)} synonyms")
        else:
            radlex[code] = {"synonyms": []}

    OUT_LOINC.write_text(
        json.dumps(loinc, indent=2, ensure_ascii=False), encoding="utf8"
    )
    OUT_RADLEX.write_text(
        json.dumps(radlex, indent=2, ensure_ascii=False), encoding="utf8"
    )

    # Merge all synonym sources
    merged: dict[str, dict] = {}
    targeted = json.loads(IN_TARGETED.read_text(encoding="utf8")) if IN_TARGETED.exists() else {}
    glob_ = json.loads(IN_GLOBAL.read_text(encoding="utf8")) if IN_GLOBAL.exists() else {}
    for code in codes:
        seen: set[str] = set()
        out_syns: list[str] = []
        for src in (
            targeted.get(code, {}).get("synonyms", []),
            glob_.get(code, {}).get("synonyms", []),
            loinc.get(code, {}).get("synonyms", []),
            radlex.get(code, {}).get("synonyms", []),
        ):
            for s in src:
                k = s.lower()
                if k not in seen:
                    seen.add(k)
                    out_syns.append(s)
        merged[code] = {"synonyms": out_syns}
    OUT_ALL.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf8"
    )
    t_loinc = sum(len(v["synonyms"]) for v in loinc.values())
    t_radlex = sum(len(v["synonyms"]) for v in radlex.values())
    t_all = sum(len(v["synonyms"]) for v in merged.values())
    print(f"\nLOINC synonyms: {t_loinc}")
    print(f"RadLex synonyms: {t_radlex}")
    print(f"Merged total: {t_all}")
    print(f"Wrote {OUT_LOINC}, {OUT_RADLEX}, {OUT_ALL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
