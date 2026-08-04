"""Groq Llama 3.3 70B -- thin model file, all logic lives in pipeline/coder_base.py."""
import os

from automated_coding.pipeline.coder_base import BaseCoder

MODEL_NAME = "llama-3.3-70b-versatile"


class GroqLlamaModal(BaseCoder):
    MODEL_NAME = MODEL_NAME

    def __init__(self):
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)

    def _call(self, system: str, user: str, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
