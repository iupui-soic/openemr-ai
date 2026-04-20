"""Two-stage retrieval: BGE bi-encoder shortlists, BGE cross-encoder reranks.

Stage 1 (retrieval): exactly what `embed_match` does — BGE base encodes note
chunks and candidate description variants, takes max cosine per code. Shortlist
the top-N codes as candidates.

Stage 2 (reranking): `BAAI/bge-reranker-v2-m3` is a cross-encoder that scores
(query, passage) pairs with full bidirectional attention. We pair every
shortlisted code's best-matching variant with the note's best-matching chunk
(identified by the cosine max) and score them. Predictions = top-K after
reranking.

This is the standard 2-stage IR pipeline (SOTA in MS MARCO and most retrieval
benchmarks). Cross-encoders beat bi-encoders for fine-grained scoring because
attention can relate specific query phrases to specific document phrases —
exactly the thing we need to distinguish sibling CPT codes.
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


BGE_ENCODER = "BAAI/bge-base-en-v1.5"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
DEFAULT_SHORTLIST_N = 10
DEFAULT_TOP_K = 1
DEFAULT_MIN_RERANK_SCORE = -10.0
DEFAULT_CHUNKS_PER_CODE = 5  # how many top BGE-matched chunks to rerank per code
DEFAULT_CHUNK_WORDS = 32
DEFAULT_CHUNK_STRIDE = 16
DEFAULT_MAX_CHUNKS = 1024
DEFAULT_BATCH = 64

QUERY_PREFIX = "Represent this procedure description for retrieval: "
PASSAGE_PREFIX = ""

SHORT_DESC_JSON = (
    Path(__file__).resolve().parent.parent / "dataset" / "cpt_short.json"
)
SYNONYMS_JSON = (
    Path(__file__).resolve().parent.parent / "dataset" / "cpt_synonyms.json"
)

logger = logging.getLogger(__name__)


def _chunk(text: str, window: int, stride: int) -> list[str]:
    toks = re.findall(r"\S+", text)
    if not toks:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(toks):
        chunk = " ".join(toks[i : i + window])
        if chunk:
            chunks.append(chunk)
        if i + window >= len(toks):
            break
        i += stride
    return chunks[:DEFAULT_MAX_CHUNKS]


def _augment_desc(
    code: str, long_desc: str, short: dict[str, str], synonyms: dict[str, dict]
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def push(s: str) -> None:
        s = s.strip()
        k = s.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s)

    push(long_desc)
    if code in short and short[code]:
        push(short[code])
    for syn in synonyms.get(code, {}).get("synonyms", []):
        push(syn)
    return out


class RerankMatchPredictor:
    name = "rerank_match"

    def __init__(
        self,
        shortlist_n: int = DEFAULT_SHORTLIST_N,
        top_k: int = DEFAULT_TOP_K,
        min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
        chunks_per_code: int = DEFAULT_CHUNKS_PER_CODE,
    ) -> None:
        self.shortlist_n = int(os.environ.get("RERANK_SHORTLIST_N", shortlist_n))
        self.top_k = int(os.environ.get("RERANK_TOP_K", top_k))
        self.min_rerank_score = float(
            os.environ.get("RERANK_MIN_SCORE", min_rerank_score)
        )
        self.chunks_per_code = int(
            os.environ.get("RERANK_CHUNKS_PER_CODE", chunks_per_code)
        )
        self._bge_tok: Any = None
        self._bge: Any = None
        self._rr_tok: Any = None
        self._rr: Any = None
        self._device: Any = None
        self._label_space: list[str] = []
        self._desc_variants: dict[str, list[str]] = {}
        self._variant_emb: Any = None
        self._variant_code_idx_t: Any = None

    def prepare(
        self, label_space: list[str], descriptions: dict[str, str]
    ) -> None:
        import torch
        from transformers import (
            AutoModel,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._bge_tok = AutoTokenizer.from_pretrained(BGE_ENCODER)
        self._bge = AutoModel.from_pretrained(BGE_ENCODER).to(self._device).eval()

        self._rr_tok = AutoTokenizer.from_pretrained(BGE_RERANKER)
        self._rr = (
            AutoModelForSequenceClassification.from_pretrained(BGE_RERANKER)
            .to(self._device)
            .eval()
        )

        short: dict[str, str] = {}
        if SHORT_DESC_JSON.exists():
            short = json.loads(SHORT_DESC_JSON.read_text(encoding="utf8"))
        synonyms: dict[str, dict] = {}
        if SYNONYMS_JSON.exists():
            synonyms = json.loads(SYNONYMS_JSON.read_text(encoding="utf8"))

        self._label_space = list(label_space)
        self._desc_variants = {
            code: _augment_desc(code, descriptions[code], short, synonyms)
            for code in self._label_space
        }

        variant_embs: list[Any] = []
        code_idx_list: list[int] = []
        for ci, code in enumerate(self._label_space):
            variants = self._desc_variants[code]
            ve = self._bge_embed([QUERY_PREFIX + v for v in variants])
            variant_embs.append(ve)
            code_idx_list.extend([ci] * ve.shape[0])
        self._variant_emb = torch.cat(variant_embs, dim=0)
        self._variant_code_idx_t = torch.tensor(
            code_idx_list, device=self._device, dtype=torch.long
        )
        logger.info(
            "rerank_match: %d codes, %d variants, shortlist_n=%d, top_k=%d",
            len(self._label_space),
            self._variant_emb.shape[0],
            self.shortlist_n,
            self.top_k,
        )

    def _bge_embed(self, texts: list[str]) -> Any:
        import torch

        if not texts:
            return torch.empty(0, 768, device=self._device)
        outs: list[Any] = []
        for i in range(0, len(texts), DEFAULT_BATCH):
            batch = self._bge_tok(
                texts[i : i + DEFAULT_BATCH],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                out = self._bge(**batch).last_hidden_state
            pooled = out[:, 0]
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            outs.append(pooled)
        return torch.cat(outs, dim=0)

    def _rerank_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Cross-encoder scores for (query, passage) pairs."""
        import torch

        if not pairs:
            return []
        scores: list[float] = []
        for i in range(0, len(pairs), DEFAULT_BATCH):
            batch = pairs[i : i + DEFAULT_BATCH]
            inp = self._rr_tok(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._rr(**inp).logits.view(-1).float()
            scores.extend(logits.tolist())
        return scores

    def predict(self, text: str) -> set[str]:
        import torch

        chunks = _chunk(text, DEFAULT_CHUNK_WORDS, DEFAULT_CHUNK_STRIDE)
        if not chunks:
            return set()

        # Stage 1: BGE retrieval — aggregate max variant similarity per code
        chunk_emb = self._bge_embed([PASSAGE_PREFIX + c for c in chunks])
        sim = chunk_emb @ self._variant_emb.t()              # [n_chunks, n_var]
        var_max = sim.max(dim=0).values                      # [n_var]
        n_codes = len(self._label_space)
        code_best = torch.full(
            (n_codes,), -1.0, device=self._device, dtype=var_max.dtype
        ).scatter_reduce(0, self._variant_code_idx_t, var_max, reduce="amax")

        n = min(self.shortlist_n, n_codes)
        top_vals, top_idx = torch.topk(code_best, n)
        shortlisted: list[int] = [int(i) for i in top_idx.tolist()]

        # Stage 2: for each shortlisted code, take the top-M BGE-ranked chunks
        # for that code's best-scoring variant. Pair each with the variant and
        # cross-encode. Score for the code = max rerank score.
        #
        # Picking top-M chunks (not just top-1) gives the reranker multiple
        # candidate evidence spans to choose from; picking ALL chunks is too
        # slow. M is controlled by chunks_per_code.
        all_pairs: list[tuple[str, str]] = []
        pair_slices: list[tuple[int, int, int]] = []
        m = min(self.chunks_per_code, len(chunks))
        for ci in shortlisted:
            variants = self._desc_variants[self._label_space[ci]]
            # sim for this code's variants = sim[:, variant_indices_for_ci]
            # recover variant indices belonging to this code
            var_mask = (self._variant_code_idx_t == ci)
            ci_variant_cols = var_mask.nonzero(as_tuple=True)[0]
            if ci_variant_cols.numel() == 0:
                continue
            # max chunk-variant sim for each chunk (across this code's variants)
            code_sim = sim[:, ci_variant_cols]              # [n_chunks, n_var_for_ci]
            chunk_scores, _ = code_sim.max(dim=1)           # [n_chunks]
            top_vals, top_chunk_idx = torch.topk(chunk_scores, m)
            # pick the best variant overall for this code as the description text
            best_var_local = int(code_sim.max(dim=0).values.argmax().item())
            variant_str = variants[best_var_local]
            start = len(all_pairs)
            for ci_idx in top_chunk_idx.tolist():
                all_pairs.append((chunks[int(ci_idx)], variant_str))
            pair_slices.append((ci, start, len(all_pairs)))

        scores = self._rerank_scores(all_pairs)
        code_scores: dict[int, float] = {}
        for ci, s, e in pair_slices:
            if s == e:
                continue
            code_scores[ci] = max(scores[s:e])

        scored = sorted(code_scores.items(), key=lambda r: -r[1])
        preds: set[str] = set()
        for ci, score in scored[: self.top_k]:
            if score >= self.min_rerank_score:
                preds.add(self._label_space[ci])
        return preds

    def _variant_code_variant(self, code_idx: int, variant_idx_global: int) -> str:
        """Resolve a global variant index back to its variant string."""
        # Walk the variant list for this code
        # (we didn't store a flat string list; rebuild here)
        offset = 0
        for ci in range(code_idx):
            offset += len(self._desc_variants[self._label_space[ci]])
        local = variant_idx_global - offset
        variants = self._desc_variants[self._label_space[code_idx]]
        if 0 <= local < len(variants):
            return variants[local]
        # Defensive fallback
        return variants[0]

    def close(self) -> None:
        import torch

        self._bge = None
        self._rr = None
        self._variant_emb = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_ = Predictor
