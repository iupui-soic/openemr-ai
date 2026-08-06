"""Benchmark predictor that calls a Modal-deployed Coder (Gemma4Coder/Qwen3Coder).

Does BGE retrieval locally to build the candidate shortlist, then calls the
Modal A100 for extraction+matching. Sidesteps local GPU memory limits.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "50"))
BGE_ENCODER = "BAAI/bge-base-en-v1.5"
MODAL_APP = os.environ.get("MODAL_APP", "automated-cpt-icd-coding")
MODAL_CLASS = os.environ.get("MODAL_CLASS", "Gemma4Coder")


class ModalCoderPredictor:
    def __init__(self, model_id: str = "", backend: str = "modal", **kw) -> None:
        self.name = f"modal:{MODAL_CLASS}"
        self.model_id = model_id or MODAL_CLASS
        self._encoder = None
        self._code_ids: list[str] = []
        self._code_embs = None
        self._descriptions: dict[str, str] = {}
        self._label_space: list[str] = []
        self._coder = None

    def prepare(self, label_space: list[str], descriptions: dict[str, str]) -> None:
        import modal
        from sentence_transformers import SentenceTransformer

        self._label_space = list(label_space)
        self._descriptions = dict(descriptions)
        logger.info("Loading BGE encoder: %s", BGE_ENCODER)
        self._encoder = SentenceTransformer(BGE_ENCODER, device="cpu")
        self._code_ids = list(label_space)
        texts = [f"{c}: {descriptions.get(c,'')}" for c in self._code_ids]
        self._code_embs = self._encoder.encode(
            texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        logger.info("Indexed %d codes. Connecting to Modal %s/%s",
                    len(self._code_ids), MODAL_APP, MODAL_CLASS)
        CoderCls = modal.Cls.from_name(MODAL_APP, MODAL_CLASS)
        self._coder = CoderCls()

    def _retrieve(self, text: str) -> dict[str, str]:
        import numpy as np
        q = self._encoder.encode([text[:4000]], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)[0]
        sims = self._code_embs @ q
        k = min(RETRIEVE_TOP_K, len(self._code_ids))
        top = np.argpartition(-sims, k - 1)[:k]
        return {self._code_ids[i]: self._descriptions[self._code_ids[i]] for i in top}

    def predict(self, text: str) -> set[str]:
        candidates = self._retrieve(text)
        try:
            result = self._coder.generate_codes.remote(note_text=text, candidates=candidates)
        except Exception as exc:
            logger.warning("Modal call failed: %s", exc)
            return set()
        preds = set()
        for m in result.get("matches", []):
            code = m.get("code") if isinstance(m, dict) else m
            if code in self._label_space:
                preds.add(code)
        return preds

    def close(self) -> None:
        pass
