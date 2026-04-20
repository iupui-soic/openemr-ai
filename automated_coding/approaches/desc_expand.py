"""Deterministic description expander — generalizable abbreviation & phrasing.

Takes a medical description and generates additional clinically-phrased variants
by applying a public, label-space-independent medical abbreviation and phrasing
table. Works on CPT, HCPCS, ICD-10-PCS, LOINC, or any procedure description —
no codes are referenced anywhere.

Design:
- Each rule is a bidirectional synonym set (canonical ↔ abbreviations)
- At expansion time, we find any canonical form in the input description and
  add a variant with each abbreviation substituted in
- Rules are cumulative — if a description matches multiple rules, we generate
  the cartesian product of variants (capped)

Sources: UMLS LRABR, Stedman's Medical Abbreviations, RadLex common terms.
"""
from __future__ import annotations

import re
from itertools import product


# Each entry: canonical regex pattern -> list of alternative surface forms.
# Ordered roughly by specificity (longer multi-word patterns first).
_ABBREVIATIONS: list[tuple[re.Pattern[str], list[str]]] = [
    # Imaging modalities
    (re.compile(r"\bradiologic\s+examination\b", re.I),
     ["X-ray exam", "radiograph", "plain film", "XR"]),
    (re.compile(r"\bcomputed\s+tomograph(?:y|ic)\b", re.I),
     ["CT", "CAT scan", "CT scan"]),
    (re.compile(r"\bmagnetic\s+resonance(?:\s+imaging)?\b", re.I),
     ["MRI", "MR", "MR imaging"]),
    (re.compile(r"\bmagnetic\s+resonance\s+angiography\b", re.I),
     ["MRA", "MR angiogram", "MR angiography"]),
    (re.compile(r"\bultrasound\b", re.I),
     ["US", "sonogram", "sonography", "ultrasonography"]),
    (re.compile(r"\becho(?:cardiograph(?:y|ic))?\b", re.I),
     ["echo", "TTE", "echocardiogram"]),
    (re.compile(r"\btransesophageal\s+echocardiograph(?:y|ic)?\b", re.I),
     ["TEE", "transesophageal echo"]),
    (re.compile(r"\btransthoracic\s+echocardiograph(?:y|ic)?\b", re.I),
     ["TTE", "transthoracic echo"]),
    (re.compile(r"\belectrocardiogra(?:m|ph(?:y|ic))\b", re.I),
     ["ECG", "EKG", "12-lead"]),
    (re.compile(r"\bduplex\s+scan\b", re.I),
     ["duplex US", "duplex ultrasound", "Doppler US"]),
    # Anatomy abbreviations
    (re.compile(r"\bchest\b", re.I),
     ["chest", "CXR (area)", "thorax"]),
    (re.compile(r"\babdomen\b", re.I),
     ["abdomen", "abd"]),
    (re.compile(r"\bpelvis\b", re.I),
     ["pelvis", "pelv"]),
    (re.compile(r"\bhead\b", re.I),
     ["head", "cranium", "skull"]),
    (re.compile(r"\bbrain\b", re.I),
     ["brain", "cerebral"]),
    # View count / laterality
    (re.compile(r"\bsingle\s+view\b", re.I),
     ["1 view", "single-view", "one view", "PA only"]),
    (re.compile(r"\b2\s+views?\b", re.I),
     ["two views", "2-view", "PA and lateral", "PA/lat"]),
    (re.compile(r"\b3\s+views?\b", re.I),
     ["three views", "3-view"]),
    (re.compile(r"\bbilateral\b", re.I),
     ["bilateral", "both sides", "B/L"]),
    (re.compile(r"\bunilateral\b", re.I),
     ["unilateral", "one side", "single-side"]),
    # Contrast phases
    (re.compile(r"\bwithout\s+contrast\b", re.I),
     ["without contrast", "non-contrast", "unenhanced", "w/o contrast", "no contrast"]),
    (re.compile(r"\bwith\s+contrast\b", re.I),
     ["with contrast", "contrast-enhanced", "w/ contrast", "post-contrast"]),
    (re.compile(r"\bfollowed\s+by\s+contrast\b", re.I),
     ["followed by contrast", "with and without contrast", "pre/post contrast"]),
    # Procedure verbs
    (re.compile(r"\binsertion\s+of\b", re.I),
     ["insertion of", "placement of", "insert"]),
    (re.compile(r"\bintroduction\s+of\b", re.I),
     ["introduction of", "insertion of", "placement of", "introduce"]),
    (re.compile(r"\bcatheteri[sz]ation\b", re.I),
     ["catheterization", "catheter insertion", "cath"]),
    (re.compile(r"\bendotracheal\s+intubation\b", re.I),
     ["endotracheal intubation", "ETT placement", "intubation", "ET tube insertion"]),
    (re.compile(r"\bcentral\s+venous\b", re.I),
     ["central venous", "central line", "CVL", "central catheter"]),
    (re.compile(r"\bbladder\s+catheter\b", re.I),
     ["bladder catheter", "Foley", "urinary catheter"]),
    # E/M terms
    (re.compile(r"\bcritical\s+care\b", re.I),
     ["critical care", "ICU care"]),
    (re.compile(r"\bhospital\s+care\b", re.I),
     ["hospital care", "inpatient care", "subsequent day care"]),
    (re.compile(r"\bcardiopulmonary\s+resuscitation\b", re.I),
     ["cardiopulmonary resuscitation", "CPR", "code blue"]),
    (re.compile(r"\bventilation\b", re.I),
     ["ventilation", "mechanical ventilation", "vent", "ventilator support"]),
    # Sedation / anesthesia
    (re.compile(r"\bmoderate\s+sedation\b", re.I),
     ["moderate sedation", "conscious sedation", "procedural sedation"]),
]

# Cap the total number of generated variants per description — cartesian
# products with multiple rules can explode quickly.
MAX_VARIANTS = 8


def expand_description(desc: str) -> list[str]:
    """Return a list of variant phrasings of `desc`, including the original.

    Applies each abbreviation rule that matches: for each matching rule, we
    branch the current candidate list by substituting every alternate form
    for the canonical form. The cartesian growth is capped at MAX_VARIANTS.
    """
    current: list[str] = [desc]
    for pat, alts in _ABBREVIATIONS:
        if not pat.search(desc):
            continue
        next_round: list[str] = []
        for candidate in current:
            # keep the original
            next_round.append(candidate)
            for alt in alts:
                # substitute only the first match to limit explosion
                new = pat.sub(alt, candidate, count=1)
                if new != candidate:
                    next_round.append(new)
        # dedup and cap
        seen: set[str] = set()
        deduped: list[str] = []
        for s in next_round:
            k = s.lower().strip()
            if k and k not in seen:
                seen.add(k)
                deduped.append(s.strip())
        current = deduped[:MAX_VARIANTS]
    return current
