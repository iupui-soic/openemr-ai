"""Hybrid BM25 + BGE retrieval with section-aware chunking.

Motivation
----------
1. Sibling codes like `71045 Radiologic examination, chest; single view` and
   `71046 Radiologic examination, chest; 2 views` are indistinguishable in BGE
   embedding space — the two descriptions and "chest x-ray" clinical text all
   live in the same neighbourhood. BM25 with proper tokenisation picks up the
   exact phrases ("single view" / "2 views") via IDF weighting, where dense
   models blur them.

2. MIMIC notes are long (often 5-10 KB) and most of the text is history /
   physical exam / lab results. Procedure evidence lives in a small number of
   sections: RADIOLOGY / IMAGING / FINDINGS / IMPRESSION / PROCEDURES / HOSPITAL
   COURSE. Filtering to those sections reduces noise.

Scoring
-------
For each code C with variants V_C (long desc + short desc + UMLS synonyms):
  s_bge(C)  = max over chunks c, variants v of cos(BGE(c), BGE(v))
  s_bm25(C) = max over variants v of BM25(note_tokens, v_tokens)
  s(C)      = alpha * rank_norm(s_bge) + (1-alpha) * rank_norm(s_bm25)

`rank_norm` converts raw scores to [0,1] by rank within the label space so the
two signals are comparable regardless of their native scale.

Prediction = codes whose fused score is in the top-K and exceeds a minimum
fused threshold.
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


# ------------------------------ config ---------------------------------------

DEFAULT_ENCODER = "BAAI/bge-base-en-v1.5"
DEFAULT_TOP_K = 1
DEFAULT_MIN_FUSED = 0.55        # fused min-score threshold (rank-normalized)
DEFAULT_ALPHA = 0.5             # weight on BGE in the fusion
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

# Section headers that contain procedure evidence in MIMIC notes.
RELEVANT_SECTION_RE = re.compile(
    r"(?im)^\s*("
    r"procedure|radiolog|imag|study|studies|finding|impression|"
    r"hospital\s+course|chest|exam|echo|cardiac catheter|ekg|ecg|"
    r"cat\s*scan|ct\b|mri|ultrasound|ventilat|intubat"
    r")[\s:]"
)

# Any colon-terminated header line counts as a section break.
SECTION_HEADER_RE = re.compile(
    r"(?m)^\s*([A-Z][A-Z \-\(\)/]{2,40}):\s*$"
)

logger = logging.getLogger(__name__)


# ------------------------------ helpers --------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")

# Short clinical stopwords + generic English words that poison BM25 on short
# CPT gist descriptions (e.g. "motion fluor eval swlng funcj rec" picks up
# "eval" from note chunks). Keep medical abbreviations (CXR/ECG/EKG/CT/MR).
_STOP = {
    "a", "an", "the", "of", "and", "or", "with", "without", "for", "to", "in",
    "on", "by", "from", "as", "at", "is", "was", "this", "that", "these",
    "those", "it", "its", "be", "been", "are", "per", "via", "any", "all",
    "other", "than", "then", "not", "no",
    "eval", "exam", "rec", "ser", "sep", "proc", "serv", "svc", "svcs",
    "req", "reqd", "addl", "add", "mgmt", "gen", "info",
}


def _tokens(text: str) -> list[str]:
    words = [t.lower() for t in _TOKEN_RE.findall(text)]
    words = [w for w in words if w not in _STOP and len(w) > 1]
    return words


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]


def _uni_and_bi(text: str) -> list[str]:
    uni = _tokens(text)
    return uni + _bigrams(uni)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split a note into (header, body) sections by ALL-CAPS headers ending in ':'."""
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    # Leading text before the first header
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((header, text[start:end]))
    return sections


def _relevant_text(text: str) -> str:
    """Keep only sections that look procedure-relevant. Fall back to full note."""
    sections = _split_sections(text)
    if len(sections) == 1:
        return text
    kept: list[str] = []
    for header, body in sections:
        if not header:
            # Leading orphan text — usually header metadata; keep it small.
            kept.append(body[:400])
            continue
        if RELEVANT_SECTION_RE.search(header + ":"):
            kept.append(f"{header}:\n{body}")
    joined = "\n\n".join(kept).strip()
    if len(joined) < 200:
        # Section filter removed almost everything — better to keep whole note.
        return text
    return joined


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


def _rank_norm(scores: list[float]) -> list[float]:
    """Map raw scores to [0,1] by rank (ties share the rank midpoint)."""
    n = len(scores)
    if n == 0:
        return []
    # sort indices by score ascending
    order = sorted(range(n), key=lambda i: scores[i])
    norm = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        rank = (i + j) / 2  # average rank for ties
        for k in range(i, j + 1):
            norm[order[k]] = rank / max(1, n - 1)
        i = j + 1
    return norm


def _augment_desc(
    code: str,
    long_desc: str,
    short: dict[str, str],
    synonyms: dict[str, dict],
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
    return out


# ------------------------------ predictor ------------------------------------


class HybridMatchPredictor:
    name = "hybrid_match"

    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        top_k: int = DEFAULT_TOP_K,
        min_fused: float = DEFAULT_MIN_FUSED,
        alpha: float = DEFAULT_ALPHA,
        chunk_words: int = DEFAULT_CHUNK_WORDS,
        chunk_stride: int = DEFAULT_CHUNK_STRIDE,
        use_sections: bool = True,
    ) -> None:
        self.encoder_name = os.environ.get("HYBRID_ENCODER", encoder_name)
        self.top_k = int(os.environ.get("HYBRID_TOP_K", top_k))
        self.min_fused = float(os.environ.get("HYBRID_MIN_FUSED", min_fused))
        self.alpha = float(os.environ.get("HYBRID_ALPHA", alpha))
        self.chunk_words = chunk_words
        self.chunk_stride = chunk_stride
        self.use_sections = bool(int(os.environ.get("HYBRID_USE_SECTIONS", int(use_sections))))
        self.apply_modifier_rules = bool(
            int(os.environ.get("HYBRID_MODIFIER_RULES", 0))
        )
        self._tokenizer: Any = None
        self._encoder: Any = None
        self._device: Any = None
        self._label_space: list[str] = []
        self._desc_variants: dict[str, list[str]] = {}
        self._variant_emb: Any = None            # [n_variants_total, d]
        self._variant_code_idx_t: Any = None
        self._bm25: Any = None                   # rank_bm25 BM25Okapi
        self._bm25_to_code: list[int] = []       # parallel to BM25 doc index

    # --- prepare -------------------------------------------------------------

    def prepare(
        self, label_space: list[str], descriptions: dict[str, str]
    ) -> None:
        import torch
        from rank_bm25 import BM25Okapi
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
        synonyms: dict[str, dict] = {}
        if SYNONYMS_JSON.exists():
            synonyms = json.loads(SYNONYMS_JSON.read_text(encoding="utf8"))

        self._label_space = list(label_space)
        self._desc_variants = {
            code: _augment_desc(code, descriptions[code], short, synonyms)
            for code in self._label_space
        }

        # ---- BGE: embed every variant separately (max-over-variants later)
        variant_embs: list[Any] = []
        self._bm25_docs: list[list[str]] = []
        self._bm25_to_code = []
        code_idx_list: list[int] = []
        for ci, code in enumerate(self._label_space):
            variants = self._desc_variants[code]
            ve = self._embed([QUERY_PREFIX + v for v in variants])
            variant_embs.append(ve)
            code_idx_list.extend([ci] * ve.shape[0])
            # BM25 doc per variant
            for v in variants:
                self._bm25_docs.append(_uni_and_bi(v))
                self._bm25_to_code.append(ci)
        self._variant_emb = torch.cat(variant_embs, dim=0)
        self._variant_code_idx_t = torch.tensor(
            code_idx_list, device=self._device, dtype=torch.long
        )

        self._bm25 = BM25Okapi(self._bm25_docs)

        logger.info(
            "hybrid: encoder=%s  codes=%d  variants=%d  sections=%s  alpha=%.2f",
            self.encoder_name,
            len(self._label_space),
            len(self._bm25_docs),
            self.use_sections,
            self.alpha,
        )

    # --- encoders -----------------------------------------------------------

    def _embed(self, texts: list[str]) -> Any:
        import torch

        if not texts:
            return torch.empty(0, 768, device=self._device)
        outs: list[Any] = []
        for i in range(0, len(texts), DEFAULT_BATCH):
            batch = self._tokenizer(
                texts[i : i + DEFAULT_BATCH],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                out = self._encoder(**batch).last_hidden_state
            pooled = out[:, 0]  # CLS for BGE
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            outs.append(pooled)
        return torch.cat(outs, dim=0)

    # --- predict -----------------------------------------------------------

    def predict(self, text: str) -> set[str]:
        import torch

        note_text = _relevant_text(text) if self.use_sections else text

        # ---- BGE: max over (chunks, variants) -> per code
        chunks = _chunk(note_text, self.chunk_words, self.chunk_stride)
        if not chunks:
            return set()
        chunk_emb = self._embed([PASSAGE_PREFIX + c for c in chunks])
        sim = chunk_emb @ self._variant_emb.t()           # [chunks, variants]
        max_per_variant, _ = sim.max(dim=0)               # [variants]
        n_codes = len(self._label_space)
        bge_per_code_t = torch.full(
            (n_codes,), -1.0, device=self._device, dtype=max_per_variant.dtype
        ).scatter_reduce(
            0, self._variant_code_idx_t, max_per_variant, reduce="amax"
        )
        bge_scores = bge_per_code_t.tolist()

        # ---- BM25: per-chunk query against variant docs, aggregate max per code.
        # Whole-note queries let common words in narrative boost long E/M
        # descriptions. Per-chunk queries keep BM25 local to procedure mentions.
        bm25_per_code = [-1e9] * n_codes
        for chunk in chunks:
            ctoks = _uni_and_bi(chunk)
            if not ctoks:
                continue
            scores = self._bm25.get_scores(ctoks)
            for di, ci in enumerate(self._bm25_to_code):
                s = float(scores[di])
                if s > bm25_per_code[ci]:
                    bm25_per_code[ci] = s

        # ---- Rank-normalize both, fuse
        bge_norm = _rank_norm(bge_scores)
        bm25_norm = _rank_norm(bm25_per_code)
        fused = [
            self.alpha * bge_norm[i] + (1 - self.alpha) * bm25_norm[i]
            for i in range(n_codes)
        ]

        # ---- Top-K gate + min fused score
        ranked = sorted(range(n_codes), key=lambda i: -fused[i])
        preds: set[str] = set()
        for i in ranked[: self.top_k]:
            if fused[i] >= self.min_fused:
                preds.add(self._label_space[i])
        return preds

    def close(self) -> None:
        import torch

        self._tokenizer = None
        self._encoder = None
        self._variant_emb = None
        self._bm25 = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_ = Predictor
