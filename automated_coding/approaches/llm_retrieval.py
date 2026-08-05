"""LLM predictor with BGE retrieval to shrink the candidate set per note."""
from __future__ import annotations

import logging
import os
from typing import Any

from .. import prompts
from .llm import LLMPredictor, HF_MAX_INPUT_TOKENS

logger = logging.getLogger(__name__)

RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "50"))
BGE_ENCODER = "BAAI/bge-base-en-v1.5"


class RetrievalLLMPredictor(LLMPredictor):
    def __init__(self, model_id: str, backend: str, max_new_tokens: int = 512) -> None:
        super().__init__(model_id=model_id, backend=backend, max_new_tokens=max_new_tokens)
        self.name = f"retrllm:{backend}:{model_id}"
        self._encoder: Any = None
        self._code_ids: list[str] = []
        self._code_embs: Any = None

    def prepare(self, label_space: list[str], descriptions: dict[str, str]) -> None:
        super().prepare(label_space, descriptions)
        from sentence_transformers import SentenceTransformer

        logger.info("Loading BGE encoder for retrieval: %s", BGE_ENCODER)
        self._encoder = SentenceTransformer(BGE_ENCODER, device="cuda")
        self._code_ids = list(label_space)
        texts = [f"{c}: {descriptions.get(c, '')}" for c in self._code_ids]
        self._code_embs = self._encoder.encode(
            texts, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        logger.info("Indexed %d code descriptions for retrieval", len(self._code_ids))

    def _retrieve(self, text: str) -> dict[str, str]:
        import numpy as np
        q = self._encoder.encode(
            [text[:4000]], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        sims = self._code_embs @ q
        k = min(RETRIEVE_TOP_K, len(self._code_ids))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        return {self._code_ids[i]: self._descriptions[self._code_ids[i]] for i in top_idx}

    def predict(self, text: str) -> set[str]:
        candidates = self._retrieve(text)
        full_desc, full_block = self._descriptions, self._code_block
        self._descriptions = candidates
        self._code_block = prompts.build_code_block(candidates)
        try:
            if self.backend == "hf":
                raw = self._generate_hf(text)
            elif self.backend == "anthropic":
                raw = self._generate_anthropic(text)
            elif self.backend == "groq":
                raw = self._generate_groq(text)
            else:
                raw = ""
        finally:
            self._descriptions, self._code_block = full_desc, full_block
        return prompts.parse_codes(raw, self._label_space)
