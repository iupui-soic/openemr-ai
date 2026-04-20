"""Shared LLM prompt template + output parser for the CPT benchmark.

The same prompt is reused across all LLM backends (HuggingFace transformers +
Anthropic API) so model-to-model comparisons are apples-to-apples.
"""
from __future__ import annotations

import json
import re


SYSTEM = (
    "You are a certified professional medical coder. Given a clinical note, "
    "identify which CPT procedure codes from the provided candidate list apply. "
    "Respond with JSON only: {\"codes\": [\"<code>\", ...]}. "
    "Output an empty list if none apply. Do not invent codes."
)


USER_TEMPLATE = (
    "Candidate CPT codes (code: description):\n"
    "{code_block}\n\n"
    "Clinical note:\n"
    "<<<\n{text}\n>>>\n\n"
    "Return only the subset of candidate codes whose procedures are documented "
    "in the note. Output JSON only, no prose."
)


_JSON_SPAN_RE = re.compile(r"\{[^{}]*\"codes\"[^{}]*\}", re.DOTALL)


def build_code_block(descriptions: dict[str, str]) -> str:
    return "\n".join(f"- {code}: {desc}" for code, desc in descriptions.items())


def build_messages(
    text: str, descriptions: dict[str, str]
) -> tuple[str, list[dict[str, str]]]:
    """Return (system, messages) in chat format used by both HF and Anthropic."""
    user = USER_TEMPLATE.format(
        code_block=build_code_block(descriptions), text=text
    )
    return SYSTEM, [{"role": "user", "content": user}]


def parse_codes(raw: str, label_space: list[str]) -> set[str]:
    """Extract `codes` array from an LLM response. Returns `{}` on parse failure.

    Hallucinated codes not in `label_space` are dropped.
    """
    label_set = set(label_space)
    match = _JSON_SPAN_RE.search(raw)
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
