"""Regex/dictionary baseline.

Patterns are derived from each CPT code's description plus a small hand-curated
abbreviation map. This is intentionally a naive keyword baseline — don't tune it
against gold, or the "baseline" leaks ground truth.
"""
from __future__ import annotations

import re

from .base import Predictor


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "with", "without", "for", "to", "in",
    "on", "by", "from", "per", "at", "least", "each", "any", "all", "single",
    "first", "second", "third", "subsequent", "initial", "complete", "limited",
    "report", "interpretation", "including", "included", "includes", "other",
    "otherwise", "specified", "unspecified", "only", "when", "performed", "as",
    "management", "evaluation", "examination",
}


# Hand-curated expansions keyed by a canonical lowercase stem so we can union
# pattern alternatives in for any CPT description that contains the stem.
ABBREVIATIONS: dict[str, list[str]] = {
    "electrocardiogram": ["electrocardiogram", "ECG", "EKG"],
    "radiologic examination, chest": [
        "chest x-ray", "CXR", "chest radiograph", "chest xray",
        "portable chest", "chest film", "AP chest", "PA chest",
    ],
    "chest": ["chest"],
    "computed tomography": [
        "computed tomography", "CT", "CAT scan", "CT scan",
    ],
    "magnetic resonance": [
        "magnetic resonance", "MRI", "MR imaging",
    ],
    "ultrasound": ["ultrasound", "US", "sonogram", "sonography"],
    "echocardiography": [
        "echocardiography", "echocardiogram", "echo", "TTE", "TEE",
    ],
    "angiography": ["angiography", "angiogram"],
    "ventilation": ["ventilation", "ventilator", "mechanical ventilation"],
    "hospital care": ["hospital care", "inpatient"],
    "critical care": ["critical care"],
    "catheter": ["catheter", "catheterization"],
    "biopsy": ["biopsy"],
    "drainage": ["drainage", "drain placement"],
    "intubation": ["intubation", "endotracheal"],
    "abdomen": ["abdomen", "abdominal"],
    "pelvis": ["pelvis", "pelvic"],
    "head": ["head", "skull", "brain"],
    "spine": ["spine", "spinal"],
    "swallowing": ["swallowing", "swallow study"],
    "cholangiography": ["cholangiography", "cholangiogram", "ERCP"],
}


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def _tokens(desc: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(desc)]


_SPECIFIC_UNIGRAMS_MIN_LEN = 9  # e.g. "echocardiography", "endoscopy"


def _base_terms(desc: str) -> list[str]:
    """Extract candidate keyword phrases from a description.

    Splits on common separators, drops stopwords, keeps bigrams and trigrams
    of content words. Unigrams are only kept if they are long enough to be
    reasonably specific (e.g. "echocardiography"), so generic words like
    "chest" or "report" don't fire across dozens of codes.
    """
    desc_lc = desc.lower()
    segments = re.split(r"[;,.:/()]", desc_lc)
    terms: list[str] = []
    for seg in segments:
        toks = [t for t in _TOKEN_RE.findall(seg) if t not in STOPWORDS]
        if not toks:
            continue
        for i in range(len(toks) - 1):
            terms.append(f"{toks[i]} {toks[i + 1]}")
        for i in range(len(toks) - 2):
            terms.append(f"{toks[i]} {toks[i + 1]} {toks[i + 2]}")
        for tok in toks:
            if len(tok) >= _SPECIFIC_UNIGRAMS_MIN_LEN:
                terms.append(tok)
    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        if term not in seen and len(term) >= 4:
            seen.add(term)
            ordered.append(term)
    return ordered


def _patterns_for(desc: str) -> list[re.Pattern[str]]:
    terms = _base_terms(desc)
    desc_lc = desc.lower()
    extras: list[str] = []
    for stem, aliases in ABBREVIATIONS.items():
        if stem in desc_lc:
            extras.extend(aliases)
    all_terms = sorted(set(terms + [e.lower() for e in extras]), key=len, reverse=True)
    return [
        re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in all_terms
    ]


class RegexPredictor:
    name = "regex"

    def __init__(self) -> None:
        self._patterns: dict[str, list[re.Pattern[str]]] = {}

    def prepare(
        self, label_space: list[str], descriptions: dict[str, str]
    ) -> None:
        self._patterns = {
            code: _patterns_for(descriptions[code]) for code in label_space
        }

    def predict(self, text: str) -> set[str]:
        return {
            code
            for code, pats in self._patterns.items()
            if any(p.search(text) for p in pats)
        }

    def close(self) -> None:
        self._patterns = {}


assert isinstance(RegexPredictor(), Predictor)
