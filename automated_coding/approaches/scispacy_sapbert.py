"""scispaCy (en_core_sci_scibert) NER + SapBERT concept matching.

NER: scispaCy's SciBERT-backed model extracts biomedical mention spans.
Matching: SapBERT embeds each mention and each of the 61 CPT descriptions; a
mention maps to the highest-similarity description if similarity > τ.
"""
from __future__ import annotations

import gc
from typing import Any

from .base import Predictor


SCISPACY_MODEL = "en_core_sci_scibert"
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SIMILARITY_THRESHOLD = 0.72
MAX_ENTITIES_PER_NOTE = 512


class ScispacySapbertPredictor:
    name = "scispacy_sapbert"

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        scispacy_model: str = SCISPACY_MODEL,
        sapbert_model: str = SAPBERT_MODEL,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.scispacy_model_name = scispacy_model
        self.sapbert_model_name = sapbert_model
        self._nlp: Any = None
        self._tokenizer: Any = None
        self._encoder: Any = None
        self._desc_emb: Any = None
        self._label_space: list[str] = []
        self._device: Any = None

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
        self._desc_emb = self._embed(
            [descriptions[code] for code in self._label_space]
        )

    def _embed(self, texts: list[str]) -> Any:
        import torch

        if not texts:
            return torch.empty(0, 768, device=self._device)
        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            out = self._encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled

    def predict(self, text: str) -> set[str]:
        doc = self._nlp(text)
        mentions: list[str] = []
        seen: set[str] = set()
        for ent in doc.ents:
            key = ent.text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                mentions.append(ent.text.strip())
            if len(mentions) >= MAX_ENTITIES_PER_NOTE:
                break
        if not mentions:
            return set()
        ment_emb = self._embed(mentions)
        sim = ment_emb @ self._desc_emb.t()  # [n_mentions, n_codes]
        best_sim, best_idx = sim.max(dim=1)
        predicted: set[str] = set()
        for score, idx in zip(best_sim.tolist(), best_idx.tolist()):
            if score >= self.similarity_threshold:
                predicted.add(self._label_space[idx])
        return predicted

    def close(self) -> None:
        import torch

        self._nlp = None
        self._tokenizer = None
        self._encoder = None
        self._desc_emb = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _smoke() -> Predictor:
    return ScispacySapbertPredictor()


_ = _smoke  # silence unused warning in editors
