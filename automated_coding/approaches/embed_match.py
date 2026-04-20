"""Pure embedding-similarity predictor — no NER step.

Splits each note into sentence/window chunks, embeds each chunk with a
transformer encoder (default: SapBERT, which we already have), embeds each CPT
description (optionally augmented with a short clinical-shorthand variant from
`dataset/cpt_short.json`), and for every code takes the max cosine similarity
across all note chunks. A code fires if max-sim exceeds a threshold.

This directly tests the hypothesis "just match note text against the 61
descriptions by semantic similarity", bypassing the biomedical NER stage that
misses procedure mentions in the scispacy_sapbert pipeline.
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


DEFAULT_ENCODER = "BAAI/bge-base-en-v1.5"
DEFAULT_THRESHOLD = 0.60
DEFAULT_TOP_K = 2               # cap predictions at top-K; set 0 to disable
DEFAULT_APPLY_RULES = False     # post-process with negex + modifier rules
DEFAULT_CHUNK_WORDS = 32
DEFAULT_CHUNK_STRIDE = 16
DEFAULT_MAX_CHUNKS = 1024
DEFAULT_BATCH = 64
# BGE was trained with a "query:" prefix for retrieval queries (descriptions)
# and no prefix for passages (note chunks).
QUERY_PREFIX = "Represent this procedure description for retrieval: "
PASSAGE_PREFIX = ""

SHORT_DESC_JSON = (
    Path(__file__).resolve().parent.parent / "dataset" / "cpt_short.json"
)
SYNONYMS_JSON = (
    Path(__file__).resolve().parent.parent / "dataset" / "cpt_synonyms.json"
)

logger = logging.getLogger(__name__)


def _chunk_note(
    text: str, window: int = DEFAULT_CHUNK_WORDS, stride: int = DEFAULT_CHUNK_STRIDE
) -> list[str]:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i : i + window])
        if chunk:
            chunks.append(chunk)
        if i + window >= len(tokens):
            break
        i += stride
    return chunks[:DEFAULT_MAX_CHUNKS]


def _augment_desc(
    code: str,
    long_desc: str,
    short: dict[str, str],
    synonyms: dict[str, dict],
    expand: bool = False,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def push(s: str) -> None:
        s = s.strip()
        if not s:
            return
        k = s.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(s)

    push(long_desc)
    if code in short and short[code]:
        push(short[code])
    for syn in synonyms.get(code, {}).get("synonyms", []):
        push(syn)

    if expand:
        from .desc_expand import expand_description

        # Expand each original variant through the abbreviation table.
        # Only take the first ~5 expansions per variant to avoid explosion.
        base_variants = list(out)
        for v in base_variants:
            for ev in expand_description(v)[:5]:
                push(ev)
    return out


class EmbedMatchPredictor:
    name = "embed_match"

    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        chunk_words: int = DEFAULT_CHUNK_WORDS,
        chunk_stride: int = DEFAULT_CHUNK_STRIDE,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        self.encoder_name = os.environ.get("EMBED_MATCH_ENCODER", encoder_name)
        self.similarity_threshold = float(
            os.environ.get("EMBED_MATCH_THRESHOLD", similarity_threshold)
        )
        self.top_k = int(os.environ.get("EMBED_MATCH_TOP_K", top_k))
        self.apply_rules = bool(
            int(os.environ.get("EMBED_MATCH_APPLY_RULES", DEFAULT_APPLY_RULES))
        )
        self.expand_abbrevs = bool(
            int(os.environ.get("EMBED_MATCH_EXPAND_ABBREVS", 0))
        )
        self.chunk_words = chunk_words
        self.chunk_stride = chunk_stride
        self.batch_size = batch_size
        self._tokenizer: Any = None
        self._encoder: Any = None
        self._device: Any = None
        self._label_space: list[str] = []
        self._desc_emb: Any = None         # [n_codes, d] mean over desc variants
        self._desc_variants: dict[str, list[str]] = {}

    def prepare(
        self, label_space: list[str], descriptions: dict[str, str]
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._encoder = AutoModel.from_pretrained(self.encoder_name).to(self._device)
        self._encoder.eval()

        short: dict[str, str] = {}
        if SHORT_DESC_JSON.exists():
            short = json.loads(SHORT_DESC_JSON.read_text(encoding="utf8"))
            logger.info("loaded %d short CPT descriptions", len(short))

        synonyms: dict[str, dict] = {}
        if SYNONYMS_JSON.exists():
            synonyms = json.loads(SYNONYMS_JSON.read_text(encoding="utf8"))
            total = sum(len(v.get("synonyms", [])) for v in synonyms.values())
            logger.info(
                "loaded UMLS synonyms for %d codes (%d total)", len(synonyms), total
            )

        self._label_space = list(label_space)
        self._desc_variants = {
            code: _augment_desc(
                code, descriptions[code], short, synonyms,
                expand=self.expand_abbrevs,
            )
            for code in self._label_space
        }
        # Keep ALL variant embeddings per code; at predict time we take max
        # over variants so any one good match fires the code (no dilution).
        self._variant_emb: list[Any] = []
        self._variant_code_idx: list[int] = []
        for ci, code in enumerate(self._label_space):
            variants = [QUERY_PREFIX + v for v in self._desc_variants[code]]
            ve = self._embed(variants)  # already L2-normalized
            self._variant_emb.append(ve)
            self._variant_code_idx.extend([ci] * ve.shape[0])
        self._desc_emb = torch.cat(self._variant_emb, dim=0)  # [n_variants_total, d]
        self._variant_code_idx_t = torch.tensor(
            self._variant_code_idx, device=self._device, dtype=torch.long
        )
        avg_variants = sum(
            len(self._desc_variants[c]) for c in self._label_space
        ) / max(1, len(self._label_space))
        logger.info(
            "embedded %d codes (avg %.1f variants per code)",
            len(self._label_space),
            avg_variants,
        )

        # Auto-discover sibling groups (label-space-agnostic).
        if self.apply_rules:
            from .generic_modifiers import build_sibling_groups
            self._sibling_groups = build_sibling_groups(
                {c: descriptions[c] for c in self._label_space}
            )
            logger.info(
                "auto-discovered %d sibling groups for modifier disambig.",
                len(self._sibling_groups),
            )
        else:
            self._sibling_groups = []

    def _embed(self, texts: list[str]) -> Any:
        import torch

        if not texts:
            return torch.empty(0, 768, device=self._device)
        outs: list[Any] = []
        for i in range(0, len(texts), self.batch_size):
            batch_txt = texts[i : i + self.batch_size]
            batch = self._tokenizer(
                batch_txt,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                out = self._encoder(**batch).last_hidden_state
            # BGE uses CLS token pooling (not mean)
            pooled = out[:, 0]
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            outs.append(pooled)
        return torch.cat(outs, dim=0)

    def predict(self, text: str) -> set[str]:
        chunks = _chunk_note(text, self.chunk_words, self.chunk_stride)
        if not chunks:
            return set()
        import torch

        chunk_emb = self._embed([PASSAGE_PREFIX + c for c in chunks])
        sim = chunk_emb @ self._desc_emb.t()  # [n_chunks, n_variants]

        # Per-variant argmax chunk (needed later for NegEx on evidence span)
        var_max_val, var_argmax_chunk = sim.max(dim=0)  # [n_variants]

        # Per-code max variant + which variant won
        n_codes = len(self._label_space)
        max_per_code = torch.full(
            (n_codes,), -1.0, device=self._device, dtype=var_max_val.dtype
        ).scatter_reduce(
            0, self._variant_code_idx_t, var_max_val, reduce="amax"
        )
        # Recover winning variant per code
        code_to_best_var: dict[int, int] = {}
        for v_idx in range(var_max_val.shape[0]):
            ci = int(self._variant_code_idx_t[v_idx])
            if ci not in code_to_best_var or var_max_val[v_idx] > var_max_val[code_to_best_var[ci]]:
                code_to_best_var[ci] = v_idx

        above = (max_per_code >= self.similarity_threshold).nonzero(as_tuple=True)[0]
        if above.numel() == 0:
            return set()

        if self.top_k and above.numel() > self.top_k:
            scores = max_per_code[above]
            k = min(self.top_k, scores.numel())
            _, top_idx = torch.topk(scores, k)
            above = above[top_idx]

        preds: set[str] = {self._label_space[int(i)] for i in above.tolist()}

        if self.apply_rules:
            from .negex import is_negated_in_chunk
            from .generic_modifiers import apply_generic_rules

            drop: set[str] = set()
            for ci_t in above.tolist():
                ci = int(ci_t)
                code = self._label_space[ci]
                v_idx = code_to_best_var.get(ci)
                if v_idx is None:
                    continue
                chunk_idx = int(var_argmax_chunk[v_idx].item())
                chunk = chunks[chunk_idx]
                for phrase in self._desc_variants[code]:
                    anchor = " ".join(phrase.split()[-3:])
                    if is_negated_in_chunk(chunk, anchor):
                        drop.add(code)
                        break
            preds -= drop
            # Apply label-space-agnostic sibling disambiguator (auto-discovered)
            preds = apply_generic_rules(preds, text, self._sibling_groups)

        return preds

    def close(self) -> None:
        import torch

        self._tokenizer = None
        self._encoder = None
        self._desc_emb = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_ = Predictor
