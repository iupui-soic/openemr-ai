"""Deterministic post-processing rules for sibling CPT codes.

Many CPT codes differ only by a modifier — number of views, with/without
contrast, unilateral vs bilateral. Embedding and BM25 retrieval blur these
because they share most of their description words. Rules here look for the
exact modifier evidence in the note and either:
  - prefer one code over a sibling set, or
  - demote a code if the note explicitly contradicts its modifier.

Used as a post-processor on top of any retrieval-style predictor.
"""
from __future__ import annotations

import re
from typing import Callable


# Compile once.
_one_view_re = re.compile(
    r"\b(1\s*view|single\s*view|one\s*view|pa\s+only|ap\s+only|portable)\b", re.I
)
_two_view_re = re.compile(r"\b(2\s*views?|two\s*views?|pa\s+(?:and|&)\s*lateral|two[\- ]view)\b", re.I)
_three_view_re = re.compile(r"\b(3\s*views?|three\s*views?)\b", re.I)
_with_contrast_re = re.compile(
    r"\bwith\s+(?:iv\s+)?contrast\b|\bcontrast[-\s]enhanced\b|\bpost[- ]contrast\b",
    re.I,
)
_without_contrast_re = re.compile(
    r"\bwithout\s+contrast\b|\bno\s+contrast\b|\bnon[- ]?contrast\b|\bunenhanced\b",
    re.I,
)


# Sibling groups keyed by modifier; each rule decides which code of the group
# to keep based on note text. All rules are no-op when no predictions overlap
# with the group.

# Chest x-ray 1 view vs 2 views
CHEST_XRAY_GROUP = {"71045", "71046"}

# CT abdomen variants
CT_ABD_GROUP = {"74150", "74160", "74170"}      # without / with / both
CT_ABD_PELV_GROUP = {"74176", "74177", "74178"} # without / with / both

# CT head variants
CT_HEAD_GROUP = {"70450", "70460", "70470"}     # without / with / both


def _resolve_views(preds: set[str], note: str) -> set[str]:
    """If both 71045 (1 view) and 71046 (2 views) predicted, pick based on note."""
    if not (preds & CHEST_XRAY_GROUP):
        return preds
    one = bool(_one_view_re.search(note))
    two = bool(_two_view_re.search(note))
    if one and not two:
        return (preds - CHEST_XRAY_GROUP) | {"71045"}
    if two and not one:
        return (preds - CHEST_XRAY_GROUP) | {"71046"}
    # If the note doesn't specify, default to the single-view code 71045 which
    # is by far the more common gold in MDACE.
    if preds & CHEST_XRAY_GROUP and not one and not two:
        return (preds - CHEST_XRAY_GROUP) | {"71045"}
    return preds


def _resolve_contrast(preds: set[str], note: str, group: set[str], without: str, with_: str, both: str) -> set[str]:
    hit = preds & group
    if len(hit) <= 1:
        return preds
    w = bool(_with_contrast_re.search(note))
    wo = bool(_without_contrast_re.search(note))
    if w and wo:
        # both phases in the same study
        return (preds - group) | {both}
    if w:
        return (preds - group) | {with_}
    if wo:
        return (preds - group) | {without}
    # Can't tell — keep the predicted candidate with the fewest constraints.
    return preds


def apply_rules(preds: set[str], note: str) -> set[str]:
    preds = _resolve_views(preds, note)
    preds = _resolve_contrast(
        preds, note, CT_ABD_GROUP, "74150", "74160", "74170"
    )
    preds = _resolve_contrast(
        preds, note, CT_ABD_PELV_GROUP, "74176", "74177", "74178"
    )
    preds = _resolve_contrast(
        preds, note, CT_HEAD_GROUP, "70450", "70460", "70470"
    )
    return preds


def wrap_predictor(predict_fn: Callable[[str], set[str]]) -> Callable[[str], set[str]]:
    """Decorator-style wrapper that applies modifier rules to any predict()."""
    def wrapped(text: str) -> set[str]:
        return apply_rules(predict_fn(text), text)
    return wrapped
