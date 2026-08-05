"""Shared LLM prompt templates + output parser for the coding pipeline.

The same prompts are used regardless of which model backend runs them, so
switching models only means swapping the model file -- the pipeline logic
and prompt content stay identical.
"""
from __future__ import annotations

import json
import re


EXTRACTION_SYSTEM = "You are a certified professional medical coder."

EXTRACTION_TEMPLATE = (
    "Read the following clinical note and extract the diseases, diagnoses, "
    "and procedures mentioned -- as plain medical terms, not codes.\n\n"
    "CLINICAL NOTE:\n{note_text}\n\n"
    "CRITICAL RULES (FOLLOW EXACTLY):\n"
    "- DO NOT HALLUCINATE ANY FACT. Only extract diagnoses or procedures "
    "that explicitly appear in the note text above.\n"
    "- DO NOT infer, guess, or add anything based on what commonly "
    "co-occurs with what's mentioned.\n"
    "- If in doubt about whether something is stated, LEAVE IT OUT.\n\n"
    "INSTRUCTIONS:\n"
    "1. Extract every distinct billable diagnosis, test/procedure, "
    "medication action, vaccination, and the visit type itself if nothing "
    "else procedural happened.\n"
    "2. Convert abnormal vital-sign statements into a diagnosis (e.g. an "
    "elevated blood pressure reading becomes \"essential hypertension\", "
    "even if the clinician never used that word).\n"
    "3. Before including any item, check: can you point to the exact "
    "word, phrase, or vital sign in the note that supports it?\n"
    "4. Do NOT include related conditions or common comorbidities unless "
    "explicitly stated.\n"
    "5. Do NOT include inpatient/hospital-care concepts unless the note "
    "explicitly describes an inpatient stay.\n"
    "6. Use standard medical terminology, since these terms feed a "
    "similarity search against a code database.\n"
    "7. If two items describe the SAME underlying fact using different "
    "phrasing, consolidate them into a single item using the most "
    "specific phrasing.\n"
    "8. Output JSON only, no prose:\n"
    '   {{"diagnoses": ["<term>", ...], "procedures": ["<term>", ...]}}\n\n'
    "Return the JSON now:"
)

MATCHING_SYSTEM = "You are a certified professional medical coder."

MATCHING_TEMPLATE = (
    "Given a clinical summary, the clinical terms extracted from it, and a "
    "list of candidate codes retrieved for those terms, select the codes "
    "that actually apply.\n\n"
    "CLINICAL SUMMARY (context/disambiguation only -- do not extract new "
    "terms from this):\n{note_text}\n\n"
    "EXTRACTED TERMS:\n{extracted_terms}\n\n"
    "CANDIDATE CODES (code: description):\n{candidate_codes}\n\n"
    "INSTRUCTIONS:\n"
    "1. For each extracted term, select the single best-matching code, if "
    "one genuinely applies.\n"
    "2. Use the summary to disambiguate between close candidates (e.g. "
    "laterality, severity).\n"
    "3. Do NOT invent codes not in the candidate list.\n"
    "4. Do NOT force a match if none of the candidates fit.\n"
    "5. Do NOT select inpatient/hospital-care codes unless the summary "
    "explicitly describes an inpatient stay.\n"
    "6. Do NOT select \"adverse effect\" or drug-toxicity codes unless the "
    "summary explicitly describes a negative reaction. Starting or "
    "prescribing a drug is NOT itself an adverse effect.\n"
    "7. When several candidates describe the same entity at different "
    "severity/complexity levels (e.g. 99201-99205), select only the "
    "single best-fitting level.\n"
    "8. Output JSON only, no prose:\n"
    '   {{"matches": [{{"entity": "<term>", "code": "<code>", '
    '"code_type": "CPT4|ICD10"}}, ...]}}\n'
    "9. Output an empty list if nothing matches well.\n\n"
    "Return the JSON now:"
)

VALIDATION_SYSTEM = (
    "You are a certified professional medical coder performing claims validation."
)

VALIDATION_TEMPLATE = (
    "You are given a clinical note and ONE specific code that has already "
    "been entered. Judge whether THIS SPECIFIC code is actually supported "
    "by the note -- do not comment on what other codes might apply.\n\n"
    "CLINICAL NOTE:\n{note_text}\n\n"
    "CODE TO VALIDATE:\n{code}: {description}\n\n"
    "RULES:\n"
    "- Supported only if the note has explicit textual evidence for it.\n"
    "- Mentioning a condition as unrelated history does NOT support a "
    "code for it, unless the note shows it was addressed today.\n"
    "- A code for a different but related concept is NOT supported, even "
    "if something superficially similar is mentioned.\n"
    "- When supported, quote the exact phrase from the note.\n"
    "- When not supported, leave evidence empty and briefly explain why.\n\n"
    "Output JSON only, no prose:\n"
    '{{"supported": true or false, "evidence": "<exact quoted phrase, or '
    'empty string>", "reason": "<brief explanation>"}}\n\n'
    "Return the JSON now:"
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_extraction_messages(note_text: str) -> tuple[str, str]:
    return EXTRACTION_SYSTEM, EXTRACTION_TEMPLATE.format(note_text=note_text)


def build_matching_messages(
    note_text: str, terms: list[str], candidates: dict[str, str]
) -> tuple[str, str]:
    terms_block = "\n".join(f"- {t}" for t in terms)
    candidates_block = "\n".join(f"- {c}: {d}" for c, d in candidates.items())
    user = MATCHING_TEMPLATE.format(
        note_text=note_text,
        extracted_terms=terms_block,
        candidate_codes=candidates_block,
    )
    return MATCHING_SYSTEM, user


def build_validation_messages(
    note_text: str, code: str, description: str
) -> tuple[str, str]:
    user = VALIDATION_TEMPLATE.format(
        note_text=note_text, code=code, description=description
    )
    return VALIDATION_SYSTEM, user


def extract_json(raw: str) -> dict:
    """Best-effort JSON extraction: try a brace-matched span first, then the whole string."""
    match = _JSON_OBJECT_RE.search(raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# --- Legacy single-shot benchmark functions (used by automated_coding/approaches/llm.py) ---
# These predate the two-stage extraction/matching rewrite above and are kept
# because llm.py still depends on this exact single-shot shape.

_LEGACY_SYSTEM = (
    "You are a certified professional medical coder. Given a clinical note, "
    "identify which CPT procedure codes from the provided candidate list apply. "
    "Respond with JSON only: {\"codes\": [\"<code>\", ...]}. "
    "Output an empty list if none apply. Do not invent codes."
)

_LEGACY_USER_TEMPLATE = (
    "Candidate CPT codes (code: description):\n"
    "{code_block}\n\n"
    "Clinical note:\n"
    "<<<\n{text}\n>>>\n\n"
    "Return only the subset of candidate codes whose procedures are documented "
    "in the note. Output JSON only, no prose."
)

_LEGACY_JSON_SPAN_RE = re.compile(r"\{[^{}]*\"codes\"[^{}]*\}", re.DOTALL)


def build_code_block(descriptions: dict[str, str]) -> str:
    return "\n".join(f"- {code}: {desc}" for code, desc in descriptions.items())


def build_messages(
    text: str, descriptions: dict[str, str]
) -> tuple[str, list[dict[str, str]]]:
    """Return (system, messages) in chat format used by both HF and Anthropic."""
    user = _LEGACY_USER_TEMPLATE.format(
        code_block=build_code_block(descriptions), text=text
    )
    return _LEGACY_SYSTEM, [{"role": "user", "content": user}]


def parse_codes(raw: str, label_space: list[str]) -> set[str]:
    """Extract `codes` array from an LLM response. Returns `{}` on parse failure.

    Hallucinated codes not in `label_space` are dropped.
    """
    label_set = set(label_space)
    match = _LEGACY_JSON_SPAN_RE.search(raw)
    candidates: list[str] = []
    if match:
        try:
            obj = json.loads(match.group(0))
            codes = obj.get("codes", [])
            if isinstance(codes, list):
                candidates = [str(c).strip() for c in codes]
        except json.JSONDecodeError:
            candidates = []
    if not candidates:
        try:
            obj = json.loads(raw)
            codes = obj.get("codes", []) if isinstance(obj, dict) else []
            if isinstance(codes, list):
                candidates = [str(c).strip() for c in codes]
        except json.JSONDecodeError:
            candidates = []
    return {c for c in candidates if c in label_set}
