"""
Automated CPT/ICD-10 coding pipeline.

Retrieval is done OUTSIDE this file, by coding_service.py querying a local
ChromaDB server. This file only does entity extraction and code matching,
so the GPU container has no vector-store dependency and no tie to any
individual's Modal workspace.

Prompts live in automated_coding/prompts.py, shared with the benchmark
harness, so every model variant uses identical prompt text.

Usage:
    modal run automated_coding_modal.py --note-file path/to/note.txt
"""

import modal
import os
import sys
from typing import Dict, List, Any

app = modal.App("automated-cpt-icd-coding")

MODEL_NAME = "google/gemma-4-26B-A4B-it"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "accelerate>=0.29.0",
        "huggingface_hub>=0.22.0",
    )
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), "..", "..", "automated_coding"),
        remote_path="/root/automated_coding",
    )
)


@app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
class Gemma4Coder:

    @modal.enter()
    def load_models(self):
        import torch
        from transformers import pipeline

        sys.path.insert(0, "/root")

        print(f"Loading {MODEL_NAME}...")
        self.pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        print("Coding pipeline loaded.")

    @modal.method()
    def validate_code(
        self, note_text: str, code: str, description: str
    ) -> Dict[str, Any]:
        """
        Given a note and a SINGLE already-entered code, judge whether the
        note actually supports it. Narrower than generation: a closed
        yes/no judgment with cited evidence.
        """
        from automated_coding import prompts

        system, user = prompts.build_validation_messages(note_text, code, description)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        output = self.pipe(messages, max_new_tokens=256, do_sample=False)
        raw = output[0]["generated_text"][-1]["content"]
        print(f"[validate_code] raw LLM output: {raw}")
        obj = prompts.extract_json(raw)
        result = {
            "code": code,
            "supported": bool(obj.get("supported", False)),
            "evidence": str(obj.get("evidence", "")).strip(),
            "reason": str(obj.get("reason", "")).strip(),
        }
        print(f"[validate_code] parsed: {result}")
        return result

    @modal.method()
    def wakeup(self):
        """Trivial method to force container/model load and reset the idle timer."""
        return {"status": "warm"}

    def _extract_entities(self, note_text: str) -> Dict[str, List[str]]:
        from automated_coding import prompts

        system, user = prompts.build_extraction_messages(note_text)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        output = self.pipe(messages, max_new_tokens=256, do_sample=False)
        raw = output[0]["generated_text"][-1]["content"]
        print(f"[extract_entities] raw LLM output: {raw}")
        obj = prompts.extract_json(raw)
        result = {
            "diagnoses": [str(x).strip() for x in obj.get("diagnoses", []) if str(x).strip()],
            "procedures": [str(x).strip() for x in obj.get("procedures", []) if str(x).strip()],
        }
        print(f"[extract_entities] parsed: {result}")
        return result

    def _match_codes(
        self, note_text: str, entities: Dict[str, List[str]], candidates: Dict[str, str]
    ) -> List[Dict[str, str]]:
        from automated_coding import prompts

        if not candidates:
            return []

        all_terms = entities.get("diagnoses", []) + entities.get("procedures", [])
        system, user = prompts.build_matching_messages(note_text, all_terms, candidates)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        output = self.pipe(messages, max_new_tokens=512, do_sample=False)
        raw = output[0]["generated_text"][-1]["content"]
        print(f"[match_codes] raw LLM output: {raw}")
        obj = prompts.extract_json(raw)
        print(f"[match_codes] parsed obj: {obj}")

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

    @modal.method()
    def generate_codes(
        self, note_text: str, candidates: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Given a note and a pre-retrieved candidate code set (produced
        outside Modal, by coding_service.py querying the local ChromaDB
        server), extract clinical entities and match them against the
        candidates.
        """
        import time

        start_total = time.time()

        t0 = time.time()
        entities = self._extract_entities(note_text)
        extraction_time = time.time() - t0

        t0 = time.time()
        matches = self._match_codes(note_text, entities, candidates)
        matching_time = time.time() - t0

        return {
            "entities": entities,
            "candidates_considered": len(candidates),
            "matches": matches,
            "extraction_time": extraction_time,
            "matching_time": matching_time,
            "total_time": time.time() - start_total,
            "model": MODEL_NAME,
        }


@app.local_entrypoint()
def main(note_file: str = ""):
    if not note_file:
        print("Usage: modal run automated_coding_modal.py --note-file path/to/note.txt")
        return

    with open(note_file) as f:
        note_text = f.read()

    coder = Gemma4Coder()
    result = coder.generate_codes.remote(note_text=note_text, candidates={})
    import json
    print(json.dumps(result, indent=2))
