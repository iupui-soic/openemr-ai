"""
Gemma4Coder two-stage pipeline (extract -> match), run against the benchmark's
fixed candidate set via Modal. Skips our own retrieval stage since the
benchmark harness already supplies a fixed candidate list per run.py's design
(same candidates given to every approach for a fair comparison).
"""
from __future__ import annotations

import logging

from .base import Predictor

logger = logging.getLogger(__name__)

MODAL_APP_NAME = "automated-cpt-icd-coding"
MODAL_CLASS_NAME = "Gemma4Coder"


class GemmaPipelinePredictor:
    name = "gemma_pipeline"

    def __init__(self) -> None:
        self._coder = None
        self._descriptions: dict[str, str] = {}

    def prepare(self, label_space: list[str], descriptions: dict[str, str]) -> None:
        import modal

        self._descriptions = dict(descriptions)
        Gemma4Coder = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
        self._coder = Gemma4Coder()

    def predict(self, text: str) -> set[str]:
        try:
            codes = self._coder.benchmark_predict.remote(
                note_text=text, candidates=self._descriptions
            )
            return set(codes)
        except Exception as exc:
            logger.warning("Gemma pipeline call failed: %s", exc)
            return set()

    def close(self) -> None:
        self._coder = None


_ = Predictor
