"""
Gemma-4 26B on Modal -- thin model file, all extraction/matching logic
lives in pipeline/coder_base.py, shared with every other model file.

Retrieval is done OUTSIDE this file, by coding_service.py querying a local
ChromaDB server, so the GPU container has no vector-store dependency and
no tie to any individual's Modal workspace.
"""
import modal
import os
import sys
from typing import Dict, Any

sys.path.insert(0, "/root/project")

from automated_cpt_and_icd_coding.pipeline.coder_base import BaseCoder

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
        os.path.join(os.path.dirname(__file__), "..", ".."),
        remote_path="/root/project",
    )
)


@app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
class Gemma4Coder(BaseCoder):
    MODEL_NAME = MODEL_NAME

    @modal.enter()
    def load_model(self):
        import sys
        sys.path.insert(0, "/root/project")

        import torch
        from transformers import pipeline

        print(f"Loading {MODEL_NAME}...")
        self.pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Coding pipeline loaded.")

    def _call(self, system: str, user: str, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        output = self.pipe(messages, max_new_tokens=max_tokens, do_sample=False)
        return output[0]["generated_text"][-1]["content"]

    @modal.method()
    def generate_codes(self, note_text: str, candidates: Dict[str, str]) -> Dict[str, Any]:
        return BaseCoder.generate_codes(self, note_text, candidates)

    @modal.method()
    def validate_code(self, note_text: str, code: str, description: str) -> Dict[str, Any]:
        from automated_cpt_and_icd_coding.pipeline import prompts
        system, user = prompts.build_validation_messages(note_text, code, description)
        raw = self._call(system, user, max_tokens=256)
        obj = prompts.extract_json(raw)
        return {
            "code": code,
            "supported": bool(obj.get("supported", False)),
            "evidence": str(obj.get("evidence", "")).strip(),
            "reason": str(obj.get("reason", "")).strip(),
        }

    @modal.method()
    def wakeup(self):
        return {"status": "warm"}


@app.local_entrypoint()
def main(note_file: str = ""):
    if not note_file:
        print("Usage: modal run gemma4_26b_modal.py --note-file path/to/note.txt")
        return
    with open(note_file) as f:
        note_text = f.read()
    coder = Gemma4Coder()
    result = coder.generate_codes.remote(note_text=note_text, candidates={})
    import json
    print(json.dumps(result, indent=2))
