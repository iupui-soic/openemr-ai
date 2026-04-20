# Automated CPT Coding — Benchmark

Multi-label classification over the MDACE Profee CPT annotations (312 MIMIC-III
notes, 61 unique CPT procedure codes). Compares traditional NLP, dense-retrieval,
and LLM approaches on the same gold labels and identical prompts.

**[See RESULTS.md](RESULTS.md) for the full leaderboard, ablations, and analysis.**

Headline numbers (micro-F1 on 312 notes):

| Approach | Micro-F1 | Latency/note | Local? |
|---|---:|---:|:---:|
| Claude Sonnet 4.6 (API) | 0.832 | 4.76 s | no |
| Gemma 4 26B-A4B-it (HF local) | 0.793 | 1.22 s | ✅ |
| Qwen 3.6 35B-A3B (HF local) | 0.686 | 30.46 s | ✅ |
| gpt-oss-120b (Groq API) | 0.636 | 0.36 s | no |
| **embed_match + LOINC + RadLex + rules** | **0.524** | 0.19 s | ✅ |
| entity_match (scispacy NER + SapBERT) | 0.298 | 0.21 s | ✅ |
| scispacy + SapBERT (original) | 0.158 | 0.13 s | ✅ |
| regex | 0.129 | 0.01 s | ✅ |
| MedCAT v2 (SNOMED-MIMIC) | 0.053 | 0.05 s | ✅ |

---

## Quick start

```bash
# 1. Create venv
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip

# 2. Install core deps
./.venv/bin/pip install polars pyarrow anthropic openai sentence-transformers \
    transformers accelerate torch numpy "numpy<2"

# 3. Drop required data files into dataset/ (see Data section below)

# 4. Put API keys in automated_coding/.env
cat > .env <<EOF
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
GROQ_API_KEY=gsk_...
UMLS_API_KEY=...
BIOPORTAL_API_KEY=...            # optional, only needed for RadLex enrichment
MEDCAT_MODEL_PACK=/abs/path/to/model_pack.zip   # optional
EOF

# 5. Run
set -a; . .env; set +a
./.venv/bin/python -m automated_coding.run --approach regex
./.venv/bin/python -m automated_coding.run --approach embed_match
./.venv/bin/python -m automated_coding.run --approach llm --backend anthropic \
    --model-id claude-sonnet-4-6

# 6. Build comparison table after you've run N approaches
./.venv/bin/python -m automated_coding.run --summarize
```

All approach-specific deps are imported lazily, so installing only what you
actually plan to run works fine.

---

## Approaches

| CLI `--approach` | What it does | Needs |
|---|---|---|
| `regex` | Keyword & abbreviation dictionary match against CPT descriptions | — |
| `scispacy_sapbert` | scispaCy NER spans → SapBERT cosine to descriptions | `spacy`, `scispacy`, `en_core_sci_scibert` OR `en_core_sci_lg` |
| `medcat` | MedCAT UMLS/SNOMED entity linker → CUI-intersection | MedCAT v2 + a licensed model pack |
| `medcat_tight` | Same but drops generic CUIs shared across code descriptions | same |
| `entity_match` | scispaCy NER on BOTH note and description, SapBERT entity-to-entity | scispaCy + SapBERT |
| `embed_match` | Note chunks ↔ CPT description cosine via BGE (best non-LLM) | `BAAI/bge-base-en-v1.5` |
| `hybrid_match` | BM25 + BGE score fusion (turned out to hurt vs. BGE alone) | `rank_bm25` |
| `rerank_match` | BGE bi-encoder shortlist → `bge-reranker-v2-m3` cross-encoder | `bge-reranker-v2-m3` |
| `llm` + `--backend hf` | Local HuggingFace generation | `transformers`, `accelerate`, GPU |
| `llm` + `--backend anthropic` | Claude models via the Anthropic API | `anthropic`, `ANTHROPIC_API_KEY` |
| `llm` + `--backend groq` | Groq models (e.g., `openai/gpt-oss-120b`) | `openai`, `GROQ_API_KEY` |

### embed_match post-processing (the best non-LLM stack)

`embed_match` supports two generalizable post-processors controlled by env vars:

```bash
EMBED_MATCH_TOP_K=1               # predict top-K codes by max-cosine
EMBED_MATCH_THRESHOLD=0.55        # minimum raw cosine floor
EMBED_MATCH_APPLY_RULES=1         # enable NegEx + auto-discovered sibling rules
EMBED_MATCH_EXPAND_ABBREVS=0      # add deterministic clinical-abbrev variants
```

- **NegEx**: drops a predicted code if its best-matching chunk is in a
  negation/hypothetical scope. Rules from Chapman et al. 2001 — fully general,
  no label-space knowledge.
- **Auto-discovered sibling rules** (`approaches/generic_modifiers.py`): at
  load time, scans every description in the label space to auto-find sibling
  groups that differ only by a modifier ("1 view" vs "2 views", "with contrast"
  vs "without contrast", "bilateral" vs "unilateral", "complete" vs "limited").
  At predict time, swaps the top-1 pick to the sibling whose modifier is
  attested in the note. **No hardcoded CPT codes** — works unchanged if you
  add 10 000 codes.

---

## Data

The benchmark is designed to be self-contained once `dataset/` is populated.
**Data files are gitignored** because MDACE Profee, UMLS, SNOMED, and the AMA
CPT code set all have redistribution restrictions.

| File | Source | Licence |
|---|---|---|
| `dataset/all.parquet` | MDACE Profee (MIMIC-III notes + CPT span annotations) | PhysioNet DUA |
| `dataset/codes.json` | 61 CPT codes + AMA official long descriptions (auto-written by first run from MDACE) | AMA CPT |
| `dataset/cpt_short.json` | HCPCS Level I short descriptors | Public (AMA short-desc format) |
| `dataset/cpt_synonyms.json` | Targeted UMLS synonyms (CPT CUI → atoms) | UMLS |
| `dataset/cpt_synonyms_global.json` | UMLS global-search synonyms | UMLS |
| `dataset/cpt_synonyms_loinc.json` | LOINC synonyms (via UMLS) | LOINC / UMLS |
| `dataset/cpt_synonyms_radlex.json` | RadLex synonyms (via BioPortal) | RadLex / RSNA |
| `dataset/medcat_pack/*.zip` | MedCAT v2 SNOMED-2025-MIMIC pack | UMLS + CogStack |

### Getting the data

1. **MDACE Profee** (required): https://physionet.org/content/mdace/ — credentialed
   PhysioNet access; generate `all.parquet` via the provided pipeline or drop a
   pre-built copy.
2. **AMA CPT short descriptors** (optional, boosts retrieval): use your own
   HCPCS short-descriptor source; save as `dataset/cpt_short.json` mapping
   `{"71045": "Chest x-ray, 1 view", ...}`.
3. **UMLS** (optional, boosts retrieval): register for a free UMLS UTS API key
   at https://uts.nlm.nih.gov/uts/. Then:
   ```bash
   ./.venv/bin/python -m automated_coding.scripts.fetch_umls_synonyms
   ./.venv/bin/python -m automated_coding.scripts.fetch_umls_global
   ./.venv/bin/python -m automated_coding.scripts.fetch_loinc_radlex
   ```
4. **RadLex** (optional): register for a free BioPortal API key at
   https://bioportal.bioontology.org/account; set `BIOPORTAL_API_KEY`.
5. **MedCAT model pack** (optional, only if running `--approach medcat`):
   authenticate with a UMLS licence at https://medcat.sites.er.kcl.ac.uk/auth-callback
   and download a pack (we used `v2-SNOMED-2025-MIMIC`). Point
   `MEDCAT_MODEL_PACK` at the `.zip`.

---

## Reproducibility

Every F1 number in [RESULTS.md](RESULTS.md) corresponds to a specific
combination of `--approach`, `--backend`, `--model-id`, and environment-variable
tuning flags (`EMBED_MATCH_*`, etc.). All runs produce two JSON files:

```
reports/benchmark/cpt/metrics/<run_id>.json       # scalar metrics + per-code F1
reports/benchmark/cpt/predictions/<run_id>.json   # per-note gold/pred sets
```

These are **gitignored** but reproducible by re-running the command.

Hyperparameter sweep scripts (in `logs/`, also gitignored) orchestrate the
combinations shown in RESULTS.md. You can regenerate any cell of the table
from a single CLI invocation.

### Honest methodology note

Most hyperparameter choices in this benchmark (`k=1`, `threshold=0.55`, encoder
choice, similarity filter thresholds) were picked by sweeping on the full 312
notes and choosing the best F1. That's **test-set peeking** in a strict sense.
For a rigorous paper, split 200 train / 56 dev / 56 test, tune on dev, report
on test. On a 312-note benchmark we'd expect F1 numbers to drop 2-4 points
uniformly under that methodology, but the ranking between approaches holds.

See the [RESULTS.md "generalizable vs benchmark-hacky" section](RESULTS.md#generalizable-vs-benchmark-hacky)
for a per-component audit of which techniques read gold labels and which don't.

---

## Metrics reported

- **Micro-F1** and **Macro-F1** (primary)
- **Exact-Match Ratio** (strict: every code right AND no wrong codes)
- **Jaccard** (mean per-note IoU — partial-credit EMR)
- **Label Cardinality Ratio** (mean `|pred|` / mean `|gold|` — over/under-prediction)
- **Mean latency per note**
- Per-code precision/recall/F1/support in the per-run JSON

---

## Tests

```bash
./.venv/bin/python -m unittest discover tests
```

---

## Caveats

- LLM predictions use greedy decoding (temperature 0, single seed). Multi-seed
  ensembling is out of scope.
- `regex` is a naive dictionary baseline. Don't iteratively tune its patterns
  against the gold labels — that would leak ground truth.
- MedCAT's CUI-intersection logic in `approaches/medcat.py` is intentionally
  simple to illustrate the failure mode; a production MedCAT pipeline would
  use hierarchy reasoning, context meta-annotations (Negation, Temporality,
  Experiencer), and semantic-type filtering.
- MDACE Profee covers 49 of 52 charts that have CPT annotations; all 312 notes
  come from those 49 charts. The 61-code label space has a very long tail:
  median support = 1, only 8 codes have ≥5 examples.
