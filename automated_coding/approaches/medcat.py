"""MedCAT-based predictor.

MedCAT extracts UMLS/SNOMED concepts from text and links each mention to a CUI
with a confidence score. We pre-compute each CPT description's top-1 CUI at
`prepare()` time, then at `predict()` time declare a CPT code present if any
mention resolves to its CUI with score >= threshold.

Model pack path is passed via the `MEDCAT_MODEL_PACK` environment variable
because the pack is licensed separately (UMLS/SNOMED).
"""
from __future__ import annotations

import gc
import logging
import os
from typing import Any

from .base import Predictor


MEDCAT_SCORE_THRESHOLD = 0.3
MEDCAT_MODEL_PACK_ENV = "MEDCAT_MODEL_PACK"

logger = logging.getLogger(__name__)


class MedCATPredictor:
    name = "medcat"

    def __init__(
        self,
        model_pack_path: str | None = None,
        score_threshold: float = MEDCAT_SCORE_THRESHOLD,
    ) -> None:
        self.model_pack_path = model_pack_path or os.environ.get(
            MEDCAT_MODEL_PACK_ENV
        )
        self.score_threshold = score_threshold
        self._cat: Any = None
        self._code_to_cuis: dict[str, set[str]] = {}

    def prepare(
        self, label_space: list[str], descriptions: dict[str, str]
    ) -> None:
        if not self.model_pack_path:
            raise RuntimeError(
                f"MedCAT model pack path not set; export {MEDCAT_MODEL_PACK_ENV} "
                "to the path of a MedCAT model pack (see benchmark/README.md)."
            )
        from medcat.cat import CAT

        self._cat = CAT.load_model_pack(self.model_pack_path)
        for code in label_space:
            desc = descriptions[code]
            cuis = self._extract_cuis(desc)
            if not cuis:
                logger.warning("MedCAT: no CUIs for CPT %s description", code)
            self._code_to_cuis[code] = cuis

    def _extract_cuis(self, text: str) -> set[str]:
        doc = self._cat.get_entities(text)
        entities = doc.get("entities", {}) if isinstance(doc, dict) else {}
        if isinstance(entities, dict):
            items = entities.values()
        else:
            items = entities
        cuis: set[str] = set()
        for ent in items:
            score = ent.get("acc", ent.get("context_similarity", 0.0))
            if score < self.score_threshold:
                continue
            cui = ent.get("cui")
            if cui:
                cuis.add(str(cui))
        return cuis

    def predict(self, text: str) -> set[str]:
        detected = self._extract_cuis(text)
        if not detected:
            return set()
        return {
            code
            for code, desc_cuis in self._code_to_cuis.items()
            if desc_cuis and desc_cuis & detected
        }

    def close(self) -> None:
        self._cat = None
        self._code_to_cuis = {}
        gc.collect()


_ = Predictor
