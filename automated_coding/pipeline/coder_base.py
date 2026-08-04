"""
Shared extraction + matching orchestration, reused by every model file in
automated_coding/models/. Each model file only implements _call() -- how to
send a system/user prompt to that specific backend and get text back.
"""
import time
from typing import Dict, List, Any

from automated_coding import prompts


class BaseCoder:
    def _call(self, system: str, user: str, max_tokens: int) -> str:
        raise NotImplementedError("Each model file must implement _call()")

    def _extract_entities(self, note_text: str) -> Dict[str, List[str]]:
        system, user = prompts.build_extraction_messages(note_text)
        raw = self._call(system, user, max_tokens=256)
        obj = prompts.extract_json(raw)
        return {
            "diagnoses": [str(x).strip() for x in obj.get("diagnoses", []) if str(x).strip()],
            "procedures": [str(x).strip() for x in obj.get("procedures", []) if str(x).strip()],
        }

    def _match_codes(
        self, note_text: str, entities: Dict[str, List[str]], candidates: Dict[str, str]
    ) -> List[Dict[str, str]]:
        if not candidates:
            return []
        all_terms = entities.get("diagnoses", []) + entities.get("procedures", [])
        system, user = prompts.build_matching_messages(note_text, all_terms, candidates)
        raw = self._call(system, user, max_tokens=512)
        obj = prompts.extract_json(raw)

        matches = []
        for m in obj.get("matches", []):
            code = str(m.get("code", "")).strip()
            if code in candidates:
                matches.append({
                    "entity": str(m.get("entity", "")).strip(),
                    "code": code,
                    "code_type": str(m.get("code_type", "")).strip(),
                    "description": candidates[code],
                })
        return matches

    def generate_codes(self, note_text: str, candidates: Dict[str, str]) -> Dict[str, Any]:
        start = time.time()
        entities = self._extract_entities(note_text)
        matches = self._match_codes(note_text, entities, candidates)
        return {
            "entities": entities,
            "candidates_considered": len(candidates),
            "matches": matches,
            "total_time": time.time() - start,
            "model": getattr(self, "MODEL_NAME", "unknown"),
        }
