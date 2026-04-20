"""Label-space-agnostic modifier disambiguator.

Reads the CPT description for every code in the label space, auto-discovers
"sibling groups" of codes whose descriptions differ only by a short modifier
phrase, and at prediction time picks the sibling whose modifier matches the
note text. Works on ANY label space — no hardcoded CPT codes, no assumptions
about what procedures are in scope.

Known modifier families (detected programmatically from description text):
  - view count:          "single view" / "1 view" / "2 views" / "3 views" / ...
  - contrast:            "with contrast" / "without contrast" / both phases
  - laterality:          "unilateral" / "bilateral" / "right" / "left"
  - complete/limited:    "complete" / "limited"
  - with/without report: "with report" / "without report"

At predict time, when multiple siblings of the same group are predicted, we
pick the one whose modifier phrase appears in the note (or default to a sane
fallback if the note is silent).
"""
from __future__ import annotations

import re
from typing import NamedTuple


# Modifier "token families" — each is a list of (regex, canonical_label).
# Canonical labels are used to bucket codes within a sibling group; at
# scoring time we count how many regexes from each family fire in the note.
_MODIFIER_FAMILIES: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "view_count": [
        (re.compile(r"\b(?:single\s+view|1\s*view|one\s+view)\b", re.I), "1"),
        (re.compile(r"\b(?:2\s+views?|two\s+views?|pa\s+(?:and|&)\s+lateral)\b", re.I), "2"),
        (re.compile(r"\b(?:3\s+views?|three\s+views?)\b", re.I), "3"),
        (re.compile(r"\b(?:4\s+views?|four\s+views?)\b", re.I), "4"),
        (re.compile(r"\b(?:minimum\s+of\s+2\s+views|minimum\s+2\s+views)\b", re.I), "2+"),
        (re.compile(r"\b(?:minimum\s+of\s+3\s+views|minimum\s+3\s+views)\b", re.I), "3+"),
        (re.compile(r"\b(?:minimum\s+of\s+4\s+views|minimum\s+4\s+views)\b", re.I), "4+"),
        (re.compile(r"\b(?:complete)\b", re.I), "complete"),
    ],
    "contrast": [
        (re.compile(r"\bwithout\s+(?:and\s+with\s+)?contrast\s+material.*?followed\s+by\s+contrast", re.I | re.DOTALL), "both"),
        (re.compile(r"\b(?:without\s+contrast|no\s+contrast|non[- ]?contrast|unenhanced)\b", re.I), "without"),
        (re.compile(r"\bwith\s+contrast\b", re.I), "with"),
    ],
    "laterality": [
        (re.compile(r"\bbilateral\b", re.I), "bilateral"),
        (re.compile(r"\bunilateral\b", re.I), "unilateral"),
    ],
    "completeness": [
        (re.compile(r"\bcomplete\b", re.I), "complete"),
        (re.compile(r"\blimited\b", re.I), "limited"),
    ],
    "with_report": [
        (re.compile(r"\bwith\s+(?:interpretation\s+and\s+)?report\b", re.I), "with_report"),
        (re.compile(r"\bwithout\s+(?:interpretation\s+or\s+)?report\b", re.I), "without_report"),
    ],
}


_MODIFIER_TOKEN_RE = re.compile(
    r"\b(single|1|2|two|3|three|4|four|complete|limited|bilateral|unilateral|"
    r"with|without|contrast|view|views|report|interpretation|right|left|"
    r"minimum|followed|and|or|further|sections|in|one|more|body|regions|"
    r"material|materials|acute|plus|by|of|a|the|on|at)\b",
    re.I,
)

# Common CPT stopwords and phrasal fillers.
_FILLER_RE = re.compile(
    r"\b(?:eg|e\.g\.|i\.e\.|s\))\b", re.I
)


def _strip_modifiers(desc: str) -> str:
    """Extract a code's 'stem' — the base procedure name without modifiers.

    CPT descriptions follow the convention "base procedure; modifier phrase"
    for most codes, so we truncate at the first `;` to get the base. Then we
    remove modifier tokens and punctuation.
    """
    # Strip parenthetical asides like "(eg, Foley)" which add noise
    desc = re.sub(r"\([^)]*\)", " ", desc)
    # Take the text before the first ';' as the base procedure
    base = desc.split(";", 1)[0]
    cleaned = _FILLER_RE.sub(" ", base)
    cleaned = _MODIFIER_TOKEN_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _modifier_labels_for(desc: str) -> dict[str, str]:
    """For a description, figure out which modifier bucket it falls into
    for each family. Returns {family: label}."""
    out: dict[str, str] = {}
    for family, rules in _MODIFIER_FAMILIES.items():
        for rx, label in rules:
            if rx.search(desc):
                out[family] = label
                break
    return out


def _modifier_labels_in_note(note: str) -> dict[str, set[str]]:
    """Which modifier labels appear in the note, per family."""
    out: dict[str, set[str]] = {}
    for family, rules in _MODIFIER_FAMILIES.items():
        labels = {label for rx, label in rules if rx.search(note)}
        if labels:
            out[family] = labels
    return out


class SiblingGroup(NamedTuple):
    stem: str
    codes: list[str]
    code_to_mods: dict[str, dict[str, str]]  # code -> family -> label


def build_sibling_groups(
    descriptions: dict[str, str],
) -> list[SiblingGroup]:
    """Auto-discover sibling groups from the label space."""
    by_stem: dict[str, list[str]] = {}
    code_to_mods: dict[str, dict[str, str]] = {}
    for code, desc in descriptions.items():
        stem = _strip_modifiers(desc)
        by_stem.setdefault(stem, []).append(code)
        code_to_mods[code] = _modifier_labels_for(desc)

    groups: list[SiblingGroup] = []
    for stem, codes in by_stem.items():
        if len(codes) < 2 or not stem:
            continue
        # Keep only groups where at least one modifier family varies across
        # siblings — otherwise they're just duplicate descriptions.
        varied_families: set[str] = set()
        for fam in _MODIFIER_FAMILIES:
            labels = {code_to_mods[c].get(fam) for c in codes}
            labels.discard(None)
            if len(labels) >= 2:
                varied_families.add(fam)
        if not varied_families:
            continue
        sub_map = {
            c: {f: code_to_mods[c][f] for f in varied_families if f in code_to_mods[c]}
            for c in codes
        }
        groups.append(SiblingGroup(stem=stem, codes=sorted(codes), code_to_mods=sub_map))
    return groups


def _code_compatible(
    code: str, group: SiblingGroup, note_mods: dict[str, set[str]]
) -> bool:
    mods = group.code_to_mods.get(code, {})
    for fam, label in mods.items():
        if fam not in note_mods:
            continue
        if label not in note_mods[fam]:
            return False
    return True


def apply_generic_rules(
    preds: set[str], note: str, groups: list[SiblingGroup]
) -> set[str]:
    """Label-space-agnostic sibling disambiguator.

    For any predicted code that belongs to a sibling group:
      - If the note explicitly attests a modifier (e.g. "2 views") and the
        predicted code's modifier doesn't match, SWAP to a sibling whose
        modifier does match (if any exists in the group).
      - If multiple siblings are predicted, keep only the note-compatible ones.
      - If the note attests nothing, leave predictions unchanged.
    """
    if not preds or not groups:
        return preds
    note_mods = _modifier_labels_in_note(note)
    if not note_mods:
        return preds   # nothing to disambiguate on
    new_preds = set(preds)

    code_to_group: dict[str, SiblingGroup] = {}
    for g in groups:
        for c in g.codes:
            code_to_group[c] = g

    for code in list(preds):
        group = code_to_group.get(code)
        if group is None:
            continue
        if _code_compatible(code, group, note_mods):
            continue
        # Predicted code contradicts the note — try to find a compatible sibling
        siblings = [c for c in group.codes if c != code]
        compatible = [
            c for c in siblings if _code_compatible(c, group, note_mods)
        ]
        if not compatible:
            continue   # no better sibling; leave predictions alone
        # If exactly one compatible sibling, swap to it.
        if len(compatible) == 1:
            new_preds.discard(code)
            new_preds.add(compatible[0])
            continue
        # If multiple compatible siblings, pick any one deterministically
        # (sort by code id — no peeking at downstream signals).
        new_preds.discard(code)
        new_preds.add(sorted(compatible)[0])

    # Also: if multiple siblings of the SAME group are in preds, keep only the
    # note-compatible subset (fallback to original if note contradicts all).
    for group in groups:
        in_preds = [c for c in group.codes if c in new_preds]
        if len(in_preds) < 2:
            continue
        keep = [c for c in in_preds if _code_compatible(c, group, note_mods)]
        if keep and len(keep) < len(in_preds):
            new_preds -= (set(in_preds) - set(keep))

    return new_preds
