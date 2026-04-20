"""Bidirectional entity-set matching.

Run scispaCy NER on BOTH the note and each CPT code description to produce two
entity sets. Embed all entities with SapBERT. For each code, score it by the
fraction of its description-entities that have at least one semantically
similar entity in the note (cosine >= tau). Select top-K codes by score.

Contrast with `scispacy_sapbert`, which matches mentions in the note directly
to whole description embeddings (long-to-short mismatch; description
embeddings cluster in SapBERT space, giving poor discrimination).
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from .base import Predictor


SCISPACY_MODEL = "en_core_sci_lg"
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

DEFAULT_ENTITY_SIM = 0.80       # entity-entity match threshold
DEFAULT_MIN_COVERAGE = 0.50     # need >= this fraction of desc entities matched
DEFAULT_TOP_K = 2
DEFAULT_MAX_ENTITIES = 2048

SHORT_DESC_JSON = (
    Path(__file__).resolve().parent.parent / "dataset" / "cpt_short.json"
)

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def _fallback_entities(text: str) -> list[str]:
    """If scispaCy returns no entities for a description, chunk by punctuation."""
    segs = re.split(r"[;,./()]", text)
    out: list[str] = []
    for s in segs:
        toks = [t for t in _WORD_RE.findall(s) if len(t) >= 3]
        if toks:
            out.append(" ".join(toks))
    return out or [text.strip()]


class EntityMatchPredictor:
    name = "entity_match"

    def __init__(
        self,
        entity_sim: float = DEFAULT_ENTITY_SIM,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
        top_k: int = DEFAULT_TOP_K,
        scispacy_model: str = SCISPACY_MODEL,
        sapbert_model: str = SAPBERT_MODEL,
    ) -> None:
        self.entity_sim = float(os.environ.get("ENTITY_MATCH_SIM", entity_sim))
        self.min_coverage = float(
            os.environ.get("ENTITY_MATCH_MIN_COVERAGE", min_coverage)
        )
        self.top_k = int(os.environ.get("ENTITY_MATCH_TOP_K", top_k))
        self.scispacy_model_name = scispacy_model
        self.sapbert_model_name = sapbert_model
        self._nlp: Any = None
        self._tokenizer: Any = None
        self._encoder: Any = None
        self._device: Any = None
        self._label_space: list[str] = []
        self._desc_entity_emb: dict[str, Any] = {}   # code -> [n_ent, d] tensor

    def prepare(
        self, label_space: list[str], descriptions: dict[str, str]
    ) -> None:
        import spacy
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._nlp = spacy.load(self.scispacy_model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.sapbert_model_name)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._encoder = AutoModel.from_pretrained(self.sapbert_model_name).to(
            self._device
        )
        self._encoder.eval()
        self._label_space = list(label_space)

        short: dict[str, str] = {}
        if SHORT_DESC_JSON.exists():
            short = json.loads(SHORT_DESC_JSON.read_text(encoding="utf8"))

        for code in self._label_space:
            texts = [descriptions[code]]
            if code in short and short[code]:
                texts.append(short[code])
            ents: list[str] = []
            seen: set[str] = set()
            for t in texts:
                for e in self._nlp(t).ents:
                    name = e.text.strip().lower()
                    if name and name not in seen:
                        seen.add(name)
                        ents.append(e.text.strip())
            if not ents:
                for fe in _fallback_entities(descriptions[code]):
                    name = fe.lower()
                    if name not in seen:
                        seen.add(name)
                        ents.append(fe)
            self._desc_entity_emb[code] = self._embed(ents)

    def _embed(self, texts: list[str]) -> Any:
        import torch

        if not texts:
            return torch.empty(0, 768, device=self._device)
        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            out = self._encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled

    def predict(self, text: str) -> set[str]:
        import torch

        doc = self._nlp(text)
        note_ents: list[str] = []
        seen: set[str] = set()
        for e in doc.ents:
            name = e.text.strip().lower()
            if name and name not in seen:
                seen.add(name)
                note_ents.append(e.text.strip())
            if len(note_ents) >= DEFAULT_MAX_ENTITIES:
                break
        if not note_ents:
            return set()
        note_emb = self._embed(note_ents)   # [n_note_ent, d]

        scores: list[tuple[str, float]] = []
        for code in self._label_space:
            de = self._desc_entity_emb[code]
            if de.numel() == 0:
                continue
            sim = de @ note_emb.t()          # [n_desc_ent, n_note_ent]
            best_per_desc, _ = sim.max(dim=1)
            matched = (best_per_desc >= self.entity_sim).sum().item()
            coverage = matched / de.shape[0]
            if coverage >= self.min_coverage:
                scores.append((code, coverage))

        if not scores:
            return set()
        scores.sort(key=lambda r: -r[1])
        top = scores[: self.top_k] if self.top_k > 0 else scores
        return {code for code, _ in top}

    def close(self) -> None:
        import torch

        self._nlp = None
        self._tokenizer = None
        self._encoder = None
        self._desc_entity_emb = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_ = Predictor
