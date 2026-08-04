"""
Shared code-validation logic, reused by every model backend. Given a note
and ONE already-entered code (from a human coder or from generate_codes()),
judges whether the note actually supports it.
"""
from typing import Dict, Any

from automated_cpt_and_icd_coding.pipeline import prompts


class ValidatorMixin:
    """Mix into any coder class that has a _call(system, user, max_tokens) method."""

    def validate_code(self, note_text: str, code: str, description: str) -> Dict[str, Any]:
        system, user = prompts.build_validation_messages(note_text, code, description)
        raw = self._call(system, user, max_tokens=256)
        obj = prompts.extract_json(raw)
        return {
            "code": code,
            "supported": bool(obj.get("supported", False)),
            "evidence": str(obj.get("evidence", "")).strip(),
            "reason": str(obj.get("reason", "")).strip(),
        }
