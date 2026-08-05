"""
Qwen3.6-35B-A3B on Modal -- thin model file, all extraction/matching logic
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
from automated_cpt_and_icd_coding.pipeline.validator import ValidatorMixin

app = modal.App("qwen3-6-35b-coder")

_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token:
    raise RuntimeError(
        "HF_TOKEN is not set in the local environment. Export it before "
        "running `modal deploy` -- an empty token will otherwise be "
        "silently passed into the container and fail at model load time."
    )

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.51.0",
        "accelerate>=0.29.0",
        "huggingface_hub>=0.22.0",
    )
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), "..", ".."),
        remote_path="/root/project",
        ignore=modal.FilePatternMatcher(
            "**/.env",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/chroma_data/**",
            "**/.idea/**",
            "**/*.parquet",
        ),
    )
)


@app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": _hf_token})],
)
class Qwen3Coder(BaseCoder, ValidatorMixin):
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

    def _call(self, system: str, user: str, max_tokens: int) -> tuple[str, bool]:
        """Returns (text, was_truncated). was_truncated is a heuristic: with
        greedy decoding (do_sample=False), a complete response ends via EOS
        before hitting max_new_tokens. If the generated token count is at or
        near the cap, that's a signal generation was cut off mid-response
        rather than finishing naturally.

        Qwen3 models default to a "thinking" mode that emits hidden <think>
        reasoning tokens before the actual answer, which was eating the
        entire max_tokens budget before reaching the JSON output. Building
        the prompt text explicitly with enable_thinking=False disables this.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_text = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        output = self.pipe(
            prompt_text,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_full_text=False,
        )
        text = output[0]["generated_text"]
        generated_token_count = len(self.pipe.tokenizer.encode(text, add_special_tokens=False))
        was_truncated = generated_token_count >= max_tokens
        return text, was_truncated

    @modal.method()
    def generate_codes(self, note_text: str, candidates: Dict[str, str]) -> Dict[str, Any]:
        return BaseCoder.generate_codes(self, note_text, candidates)

    @modal.method()
    def extract_entities(self, note_text: str) -> Dict[str, Any]:
        return BaseCoder.extract_entities(self, note_text)

    @modal.method()
    def match_codes(self, note_text: str, entities: Dict[str, Any], candidates: Dict[str, str]) -> Dict[str, Any]:
        return BaseCoder.match_codes(self, note_text, entities, candidates)

    @modal.method()
    def validate_code(self, note_text: str, code: str, description: str) -> Dict[str, Any]:
        return ValidatorMixin.validate_code(self, note_text, code, description)

    @modal.method()
    def wakeup(self):
        return {"status": "warm"}


@app.local_entrypoint()
def main(note_file: str = ""):
    if not note_file:
        print("Usage: modal run qwen3_6_35b_modal.py --note-file path/to/note.txt")
        return
    with open(note_file) as f:
        note_text = f.read()
    coder = Qwen3Coder()
    result = coder.generate_codes.remote(note_text=note_text, candidates={})
    import json
    print(json.dumps(result, indent=2))
