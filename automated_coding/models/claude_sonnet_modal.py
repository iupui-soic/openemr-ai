"""Anthropic Claude -- thin model file, all logic lives in pipeline/coder_base.py."""
import os

from automated_coding.pipeline.coder_base import BaseCoder

MODEL_NAME = "claude-sonnet-4-6"


class ClaudeSonnetModal(BaseCoder):
    MODEL_NAME = MODEL_NAME

    def __init__(self):
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=api_key)

    def _call(self, system: str, user: str, max_tokens: int) -> str:
        resp = self.client.messages.create(
            model=MODEL_NAME,
            system=system,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "".join(parts)
