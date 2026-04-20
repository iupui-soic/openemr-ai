"""Unified LLM predictor supporting HuggingFace, Anthropic, and Groq.

Same prompt for every model (see `prompts.py`) so comparisons are fair.
Anthropic calls use prompt caching on the candidate-code block (identical across
all notes → cache hit rate approaches 100% after first call). Groq uses the
OpenAI-compatible Chat Completions API — no prompt caching exposed, but Groq's
very high throughput makes per-call latency low.
"""
from __future__ import annotations

import gc
import logging
import os
from typing import Any, Literal

from .. import prompts
from .base import Predictor


MAX_NEW_TOKENS = 512
HF_MAX_INPUT_TOKENS = 14000

logger = logging.getLogger(__name__)


class LLMPredictor:
    def __init__(
        self,
        model_id: str,
        backend: Literal["hf", "anthropic", "groq"],
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.name = f"llm:{backend}:{model_id}"
        self._tokenizer: Any = None
        self._model: Any = None
        self._client: Any = None
        self._label_space: list[str] = []
        self._descriptions: dict[str, str] = {}
        self._code_block: str = ""

    def prepare(
        self, label_space: list[str], descriptions: dict[str, str]
    ) -> None:
        self._label_space = list(label_space)
        self._descriptions = dict(descriptions)
        self._code_block = prompts.build_code_block(self._descriptions)
        if self.backend == "hf":
            self._load_hf()
        elif self.backend == "anthropic":
            self._load_anthropic()
        elif self.backend == "groq":
            self._load_groq()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _load_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def _load_anthropic(self) -> None:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set; export it before running the "
                "Anthropic backend."
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def _load_groq(self) -> None:
        # Groq exposes an OpenAI-compatible Chat Completions API.
        from openai import OpenAI

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self._client = OpenAI(
            api_key=api_key, base_url="https://api.groq.com/openai/v1"
        )

    def predict(self, text: str) -> set[str]:
        if self.backend == "hf":
            raw = self._generate_hf(text)
        elif self.backend == "anthropic":
            raw = self._generate_anthropic(text)
        elif self.backend == "groq":
            raw = self._generate_groq(text)
        else:
            raw = ""
        codes = prompts.parse_codes(raw, self._label_space)
        return codes

    def _generate_hf(self, text: str) -> str:
        import torch

        system, messages = prompts.build_messages(text, self._descriptions)
        chat = [{"role": "system", "content": system}] + messages
        try:
            prompt = self._tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"{system}\n\n{messages[0]['content']}\n\nAssistant:"
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=HF_MAX_INPUT_TOKENS,
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_anthropic(self, text: str) -> str:
        system_text, messages = prompts.build_messages(text, self._descriptions)
        system_blocks = [
            {
                "type": "text",
                "text": system_text + "\n\nCandidate CPT codes:\n" + self._code_block,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        user_content = (
            "Clinical note:\n<<<\n" + text + "\n>>>\n\n"
            "Return only the subset of candidate codes whose procedures are "
            "documented in the note. Output JSON only, no prose."
        )
        try:
            resp = self._client.messages.create(
                model=self.model_id,
                system=system_blocks,
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": user_content}],
            )
            parts = [
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            ]
            return "".join(parts)
        except Exception as exc:
            logger.warning("Anthropic call failed: %s", exc)
            return ""

    def _generate_groq(self, text: str) -> str:
        system_text, messages = prompts.build_messages(text, self._descriptions)
        system_msg = (
            system_text + "\n\nCandidate CPT codes:\n" + self._code_block
        )
        user_msg = (
            "Clinical note:\n<<<\n" + text + "\n>>>\n\n"
            "Return only the subset of candidate codes whose procedures are "
            "documented in the note. Output JSON only, no prose."
        )
        # gpt-oss reasoning tokens count against max_tokens; give plenty of headroom.
        # reasoning_effort='low' keeps latency down without sacrificing quality.
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max(self.max_new_tokens, 4096),
            temperature=0.0,
        )
        if "gpt-oss" in self.model_id.lower():
            kwargs["reasoning_effort"] = "low"
        try:
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("Groq call failed: %s", exc)
            return ""

    def close(self) -> None:
        if self._model is not None:
            try:
                import torch

                del self._model
                del self._tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._model = None
        self._tokenizer = None
        self._client = None
        gc.collect()


_ = Predictor
