"""Minimal NegEx-style negation detector (Chapman et al. 2001).

Checks whether a target phrase in a clinical chunk is inside a negation scope.
Not a full re-implementation — enough to catch the obvious cases that inflate
false positives in retrieval-style CPT coding.

Usage:
    from .negex import is_negated_in_chunk
    if is_negated_in_chunk(chunk_text, target_phrase):
        ...
"""
from __future__ import annotations

import re


# "Pre-target" negation triggers — appear BEFORE the target and negate it
# up to a termination (period, "but", "however", limited to N words after).
PRE_NEGATION = re.compile(
    r"\b("
    r"no\s+(?:evidence|sign|signs|indication|indications|history)?\s*(?:of)?"
    r"|not\s+(?:seen|performed|done|indicated|evident|obvious|demonstrated|noted|observed)?"
    r"|without"
    r"|denies"
    r"|denying"
    r"|never"
    r"|ruled\s+out"
    r"|r/o"
    r"|rule[ds]?\s*out"
    r"|negative\s+for"
    r"|free\s+of"
    r"|absent"
    r"|absence\s+of"
    r"|unremarkable\s+for"
    r"|no\s+acute"
    r"|no\s+\w+\s+acute"
    r")\b",
    re.I,
)

# Terminators: a negation scope ends at these words/punct.
TERMINATORS = re.compile(
    r"\b(but|however|although|except|nonetheless|nevertheless|yet|still|"
    r"despite|instead|aside\s+from|apart\s+from|other\s+than)\b|[.;]", re.I
)

# Hypothetical/temporal triggers — similar to negation for our purposes
HYPOTHETICAL = re.compile(
    r"\b("
    r"would\s+(?:consider|benefit|need|require)"
    r"|could\s+(?:consider|benefit|need|require)"
    r"|may\s+(?:need|require|benefit)"
    r"|if\s+(?:needed|necessary|worsens|progresses|indicated)"
    r"|planned\s+for"
    r"|scheduled\s+for"
    r"|recommend\w*"
    r"|recommended\s+to"
    r"|pending"
    r"|future"
    r"|considered"
    r")\b",
    re.I,
)

# Scope window (words between trigger and target)
SCOPE_WINDOW_WORDS = 8


def _word_positions(text: str, phrase: str) -> list[tuple[int, int]]:
    """Find char spans where `phrase` (case-insensitive) occurs in text."""
    pat = re.compile(re.escape(phrase), re.I)
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def _triggers_before(text: str, end: int) -> list[tuple[int, int, str]]:
    """Find negation/hypothetical triggers ending at-or-before `end`."""
    out: list[tuple[int, int, str]] = []
    for m in PRE_NEGATION.finditer(text[:end]):
        out.append((m.start(), m.end(), "neg"))
    for m in HYPOTHETICAL.finditer(text[:end]):
        out.append((m.start(), m.end(), "hyp"))
    return out


def is_negated_in_chunk(chunk: str, target_phrase: str) -> bool:
    """True if any mention of `target_phrase` in `chunk` falls in a negation
    scope triggered by a preceding pre-negation cue, with no terminator in
    between and within SCOPE_WINDOW_WORDS."""
    hits = _word_positions(chunk, target_phrase)
    if not hits:
        return False
    for h_start, h_end in hits:
        triggers = _triggers_before(chunk, h_start)
        for t_start, t_end, _kind in triggers:
            between = chunk[t_end:h_start]
            if len(between.split()) > SCOPE_WINDOW_WORDS:
                continue
            if TERMINATORS.search(between):
                continue
            return True
    return False


def any_hit_negated(chunk: str, phrases: list[str]) -> bool:
    """True if ANY of `phrases` is negated in `chunk` AND no non-negated hit
    of any phrase exists. Useful when we have several synonym variants per
    code — we only drop the prediction if every evidence form is negated."""
    found_any_positive = False
    found_any_negated = False
    for p in phrases:
        if not p:
            continue
        if _word_positions(chunk, p):
            if is_negated_in_chunk(chunk, p):
                found_any_negated = True
            else:
                found_any_positive = True
                break
    return found_any_negated and not found_any_positive
