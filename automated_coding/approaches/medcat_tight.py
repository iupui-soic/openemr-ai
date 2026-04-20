"""MedCAT predictor with tighter matching logic.

The base `medcat` approach uses *any-CUI-overlap* between a note and a CPT
description. That's too permissive: descriptions like "Radiologic examination,
chest; single view" extract both a specific CUI ("Plain X-ray of chest") and a
generic one ("View"), and the generic CUIs fire across unrelated notes.

This variant keeps only *discriminative* description CUIs: a CUI that appears
in the description of only one CPT code in the label space. Generic CUIs used
by many descriptions (View, Routine, Interpretation, contrast material, etc.)
are dropped. Everything else matches the base approach, so the comparison
isolates the effect of the filter.
"""
from __future__ import annotations

import gc
import logging
import os
from collections import Counter
from typing import Any

from .base import Predictor
from .medcat import MEDCAT_MODEL_PACK_ENV


SCORE_THRESHOLD = 0.5  # base was 0.3
MAX_CUI_DOC_FREQ = 1   # keep only CUIs that appear in at most 1 description

logger = logging.getLogger(__name__)


class MedCATTightPredictor:
    name = "medcat_tight"

    def __init__(
        self,
        model_pack_path: str | None = None,
        score_threshold: float = SCORE_THRESHOLD,
        max_cui_doc_freq: int = MAX_CUI_DOC_FREQ,
    ) -> None:
        self.model_pack_path = model_pack_path or os.environ.get(
            MEDCAT_MODEL_PACK_ENV
        )
        self.score_threshold = score_threshold
        self.max_cui_doc_freq = max_cui_doc_freq
        self._cat: Any = None
        self._code_to_cuis: dict[str, set[str]] = {}

    def prepare(
        self, label_space: list[str], descriptions: dict[str, str]
    ) -> None:
        if not self.model_pack_path:
            raise RuntimeError(
                f"MedCAT model pack path not set; export {MEDCAT_MODEL_PACK_ENV}"
            )
        from medcat.cat import CAT

        self._cat = CAT.load_model_pack(self.model_pack_path)

        raw = {
            code: self._extract_cuis(descriptions[code]) for code in label_space
        }
        freq: Counter[str] = Counter()
        for cuis in raw.values():
            freq.update(cuis)
        generic = {cui for cui, n in freq.items() if n > self.max_cui_doc_freq}
        logger.info(
            "medcat_tight: dropping %d generic CUIs (doc_freq > %d) out of %d total",
            len(generic),
            self.max_cui_doc_freq,
            len(freq),
        )

        for code, cuis in raw.items():
            specific = cuis - generic
            if not specific:
                logger.warning(
                    "medcat_tight: no discriminative CUIs for %s — falling back to all",
                    code,
                )
                specific = cuis
            self._code_to_cuis[code] = specific

    def _extract_cuis(self, text: str) -> set[str]:
        doc = self._cat.get_entities(text)
        entities = doc.get("entities", {}) if isinstance(doc, dict) else {}
        items = entities.values() if isinstance(entities, dict) else entities
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
