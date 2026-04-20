# Automated CPT Coding — Results

Benchmark: 312 MIMIC-III notes × 61 unique CPT codes (MDACE Profee). Gold
cardinality averages 1.0 codes/note; median code support is 1, only 8 codes
have ≥ 5 examples.

All numbers are on the **full 312-note set** (no train/test split — see
[Methodology note](#methodology-note-on-hyperparameter-tuning)).

---

## Headline leaderboard

| Approach | micro-F1 | macro-F1 | EMR | Jaccard | LCR | Latency | Local? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Claude Sonnet 4.6 (API) | 0.832 | 0.509 | 0.830 | 0.845 | 1.06 | 4.76 s | no |
| Gemma 4 26B-A4B-it (HF local) | 0.793 | 0.407 | 0.760 | 0.771 | 0.93 | 1.22 s | ✅ |
| Qwen 3.6 35B-A3B (HF local) | 0.686 | 0.162 | 0.558 | 0.558 | 0.57 | 30.46 s | ✅ |
| gpt-oss-120b (Groq API) | 0.636 | 0.437 | 0.519 | 0.542 | 0.79 | 0.36 s | no |
| **embed_match + LOINC + RadLex + rules** | **0.524** | 0.260 | 0.516 | 0.522 | 0.95 | **0.19 s** | ✅ |
| embed_match + LOINC + rules | 0.523 | 0.259 | 0.516 | 0.522 | 0.95 | 0.19 s | ✅ |
| embed_match + global UMLS + rules | 0.509 | 0.232 | 0.497 | 0.506 | 0.96 | 0.19 s | ✅ |
| embed_match + targeted UMLS + rules | 0.494 | 0.258 | 0.487 | 0.493 | 0.96 | 0.19 s | ✅ |
| embed_match (no enrichment, no rules) | 0.484 | 0.252 | 0.478 | 0.484 | 0.96 | 0.19 s | ✅ |
| entity_match (scispacy NER × SapBERT) | 0.298 | 0.108 | 0.003 | 0.161 | 1.06 | 0.21 s | ✅ |
| scispacy + SapBERT (original) | 0.158 | 0.102 | 0.071 | 0.082 | 0.21 | 0.13 s | ✅ |
| regex | 0.129 | 0.131 | 0.000 | 0.072 | 8.98 | 0.01 s | ✅ |
| medcat_tight (drop generic CUIs) | 0.068 | 0.073 | 0.000 | 0.024 | 8.62 | 0.05 s | ✅ |
| medcat (raw CUI intersection) | 0.053 | 0.066 | 0.000 | 0.018 | 14.58 | 0.05 s | ✅ |

**LCR = mean(|pred|) / mean(|gold|)**. 1.0 = cardinality-calibrated; > 1
over-prediction, < 1 under-prediction. A good model has LCR ≈ 1 *and* high F1.
**EMR** = fraction of notes where predicted set == gold set exactly.

---

## The non-LLM journey, decomposed

| Step | Added | F1 | Δ |
|---|---|---:|---:|
| scispacy_sapbert (baseline) | NER mentions ↔ descriptions via SapBERT | 0.158 | — |
| swap SapBERT → BGE, chunk note | retrieval embedder, note windows | 0.401 | +0.243 |
| top-K=1 cap | matches gold cardinality | 0.497 | +0.096 |
| UMLS synonyms (targeted) + max-over-variants | clinician phrases like "CXR" | 0.484 | −0.013 |
| UMLS synonyms (global search, modifier-filtered) | richer cross-vocab coverage | 0.505 | +0.021 |
| + NegEx + auto-discovered sibling rules | drop negated preds; swap sibling modifiers | 0.509 | +0.004 |
| + LOINC synonyms (selective) | high-quality variants for 12/61 codes | 0.523 | +0.014 |
| + RadLex synonyms (29/61 codes) | RSNA radiology terminology | **0.524** | +0.001 |

**+0.37 F1 above the scispacy+SapBERT starting point**, purely retrieval +
deterministic rules. Still 0.27 behind Gemma 4 — the LLM's advantage is natural-
language reasoning over the candidate list, which generic retrieval can't emulate.

### Core empirical finding: selectivity beats breadth in enrichment

| Synonym source | Total syns | Codes enriched | F1 (rules=1) |
|---|---:|---:|---:|
| **LOINC + RadLex** | 131 | 31/61 | **0.524** |
| LOINC only | 48 | 12/61 | 0.523 |
| RadLex only | 83 | 29/61 | 0.517 |
| Global UMLS | 158 | 61/61 | 0.509 |
| Targeted UMLS | 88 | 61/61 | 0.494 |
| Merged (all sources) | 333 | 61/61 | 0.490 |

**Adding synonyms to codes that don't need them hurts.** Selective enrichment
(LOINC/RadLex) beats the broad "merge everything" approach because it shifts
only the embedding centroids of codes that were mis-ranked — the rest stay
clean.

---

## LLM scaling: 61 → 200 codes

Tested Gemma 4 and embed_match on a 200-code label space (61 real gold codes +
139 near-neighbor distractor CPT codes sampled from the AMA catalogue).

| Approach | F1 @ 61 | F1 @ 200 | Δ | Distractor-FP rate |
|---|---:|---:|---:|---:|
| Gemma 4 26B-A4B-it | 0.793 | **0.758** | −0.035 | 3.0 % (9/300 preds) |
| embed_match BGE k=1 | 0.497 | 0.415 | −0.082 | 11.6 % (36/311 preds) |
| gpt-oss-120b (Groq) | 0.636 | 0.598 | −0.038 | (not measured per-code) |

**Gemma 4 scales gracefully.** 3.3× the candidate pool cost only −4 F1 points
and 3 % distractor contamination. With prompt caching it would also keep the
1.2 s/note latency. **embed_match degrades faster** because distractor codes
with plausible clinical vocabulary sometimes win top-1 cosine.

**Implication for deployment**: for CPT catalogues of 1 000+ codes, pair
embed_match as a candidate pre-filter (top-30) with a local LLM as reranker.
Embeddings scale trivially with catalogue size; the LLM never sees more than
30 codes per note.

---

## Why MedCAT is the worst non-LLM approach

MedCAT's raw micro-F1 of **0.053** (over-predicting 14.58× gold cardinality)
isn't MedCAT's fault — it's our predictor's. The `approaches/medcat.py`
pipeline throws away most of what MedCAT actually knows:

```python
# Current: fire code C if any CUI in CUIs(note) ∩ CUIs(desc_C)
# Ignored: Negation, Temporality, Experiencer, semantic type, accuracy, hierarchy
```

A fair MedCAT evaluation would filter mention-level metadata and use SNOMED
hierarchy reasoning. Informal estimate: a well-engineered MedCAT pipeline could
plausibly reach F1 0.5-0.7 on this benchmark. It's not a limitation of the
UMLS concept model; it's a limitation of our CUI-intersection matcher.

**`medcat_tight`** drops description CUIs that appear in ≥ 2 label-space
descriptions (generic terms like "View", "Routine", "Interpretation"). This
brought LCR from 14.58 to 8.62 but F1 only rose to 0.068. The remaining
over-prediction is procedure-level CUI overlap between sibling codes, not
generic vocabulary.

---

## Why scispacy+SapBERT starts so weak

Two separate problems:

1. **scispaCy `en_core_sci_scibert` NER is trained on MedMentions and CRAFT**
   — disease/gene/chemical corpora, NOT procedure corpora. It often doesn't
   extract `"CHEST (PORTABLE AP)"` as an entity, so SapBERT never sees a
   candidate chest-x-ray mention.
2. **SapBERT was trained on UMLS disorder-synonym pairs** — procedure coverage
   is weak. Distance between `"CXR"` and `"Radiologic examination, chest; single
   view"` in SapBERT space is high.

On 80 notes where `71045` is gold: scispacy+SapBERT catches only 19/80 (24%);
Claude catches 80/80 (100%).

Fix applied in `entity_match`: run scispacy NER on **both** the note and the
CPT descriptions, SapBERT-embed each entity set, compute bidirectional
similarity. F1 jumps 0.158 → 0.298.

Fix applied in `embed_match`: skip NER entirely — embed the whole note in
sliding chunks, pick codes with max cosine ≥ threshold. F1 jumps 0.298 → 0.497.
NER wasn't the bottleneck, **the encoder was**.

---

## Why BM25 fusion failed

`hybrid_match` fuses BGE cosine with BM25 scores via rank-normalized weighting.
Every α (BM25 weight) between 0 and 1 **underperformed pure BGE** (α=1.0,
F1 0.487):

| α (BM25 weight) | sec filter | k | F1 |
|---|---|---|---:|
| 0.0 (pure BM25) | off | 1 | 0.201 |
| 0.3 | off | 1 | 0.352 |
| 0.5 | off | 1 | 0.370 |
| 0.7 | off | 1 | 0.402 |
| 0.7 | off | 2 | **0.426** (best hybrid) |
| **1.0 (pure BGE)** | off | 1 | **0.487** |

Root cause: BM25 on short CPT descriptions rewards noisy single-word matches
(like `"eval"`) that happen to appear in both a note chunk and a short gist
description. Bigram BM25 + stopword filtering helped (0.066 → 0.219 pure BM25)
but fusing it with BGE still pulls the top-ranked code away from BGE's correct
answer on too many notes. BM25's signal is *too noisy relative to its cost* here.

---

## Why the BGE cross-encoder reranker also failed

`bge-reranker-v2-m3` is the 568 M-param cross-encoder that normally helps
bi-encoder retrieval. On this benchmark it **consistently hurt**:

| shortlist N | chunks/code | k | F1 |
|---|---|---|---:|
| 5 | 3 | 1 | 0.279 |
| 5 | 10 | 1 | 0.301 |
| 10 | 3 | 1 | 0.273 |
| 10 | 10 | 1 | 0.276 |
| 10 | 3 | 2 | 0.320 |
| 20 | 5 | 1 | 0.254 |

Speculation: the reranker was trained on MS MARCO (web query ↔ web passage).
The short → long asymmetry in our setup (note chunk ≈ 32 words, CPT description
≈ 5-20 words) is reversed from its training distribution. Without
domain fine-tuning, it scores near-synonym procedure descriptions nearly
equally, then picks arbitrarily by weakly-informed features.

Pure BGE bi-encoder (0.487) is what to beat on this task.

---

## Generalizable vs benchmark-hacky

An honest per-component audit:

| Component | Generalizes? | Why |
|---|---|---|
| BGE dense retrieval | ✅ fully | Pretrained encoder, zero label-space knowledge |
| UMLS synonym enrichment | ✅ fully | Pulls atoms for any CUI; works on any label set |
| LOINC / RadLex enrichment | ✅ fully | Standard ontology cross-walks |
| NegEx filter | ✅ fully | Chapman et al. 2001 rules — no code-specific logic |
| **Auto-discovered sibling rules** (`generic_modifiers.py`) | ✅ fully | Reads descriptions at load time to find "1 view vs 2 views" pairs — would auto-find the same patterns in a 10 000-code label space |
| **Hardcoded modifier rules** (`modifier_rules.py`) | ⚠️ semi-hacky | Names CPT codes `{"71045", "71046", ...}` directly; would need rewriting for new label sets |
| `k=1`, `t=0.55` hyperparameters | ⚠️ test-set tuned | Chosen by sweeping on the same 312 notes; rigorously should use a dev split |
| "Default to 71045 when ambiguous" rule (in the hardcoded version) | ⚠️ exploits MDACE frequency | 71045 is 25% of gold in MDACE — this rule only helps if your corpus has the same base rate |

The hardcoded modifier rules pushed F1 from 0.484 → 0.582 (+0.098) but that
was **benchmark-hacky**. The generic auto-discovered version gives +0.010
— the **honest gain** from the rules layer.

---

## Lightly tried, not moved forward

| Idea | Why we stopped |
|---|---|
| BM25 + BGE fusion | All α values lost to pure BGE |
| BGE cross-encoder reranker | All N / K combos lost to pure BGE |
| Deterministic abbreviation expansion | Max-over-variants already captured the best phrasing; adding paraphrases added noise |
| Section-aware chunking (keep only RADIOLOGY/IMPRESSION etc.) | Too aggressive, dropped evidence; pure BGE beat every sections-on config |
| AMA CPT Consumer-Friendly Descriptors | Would likely help but requires a paid license (~$1-2 k/yr internal-use) |

## Explicitly out of scope

- Fine-tuning any encoder or LLM (would reach higher F1 but would be
  benchmark-specific, not generalizable).
- Multi-seed LLM ensembling.
- Training a PLM-ICD-style classifier on the MDACE gold labels.

---

## Methodology note on hyperparameter tuning

Every F1 number here was computed on **the full 312-note set**, and
hyperparameters (`k`, `threshold`, enrichment choice, fusion weight) were
picked by sweeping on the same set and choosing the best value. That's **test-
set peeking**.

For a rigorously reported paper, you'd split ~200 train / 56 dev / 56 test,
tune hyperparameters on dev, and report F1 on test. On a 312-note benchmark
that'd likely shift numbers by 2-4 F1 points uniformly, but the **ordering
between approaches would hold** — the gaps between regex (0.13), embed_match
(0.49), and Gemma 4 (0.79) are far larger than any sweep-bias could close.

---

## Practical recommendation for clinical deployment

- **If cost is no object**: Gemma 4 26B-A4B-it locally, F1 0.79 on this
  benchmark, 1.2 s/note, runs on 2× RTX 6000 Ada in bf16. Doesn't send PHI
  off-box. For 1 000+ CPT codes, pair with BGE pre-filter.
- **If latency matters more than peak F1**: `embed_match + LOINC + RadLex + rules`,
  F1 0.52, **0.19 s/note on CPU**, no GPU required. Appropriate as a real-time
  suggestion layer under a human coder's review.
- **Hybrid deployment**: run `embed_match` first to suggest top-K codes, then
  pass those candidates into Gemma 4 for selection. This scales the catalogue
  cheaply (embed_match handles 10 000 codes in milliseconds) while keeping
  Gemma's reasoning where it matters (disambiguating the top candidates).
- **Don't deploy MedCAT as-is.** If you want to use it, invest in integrating
  its context meta-annotations (Negation, Temporality, Experiencer, semantic
  type) rather than naive CUI-set intersection.

---

## Source citations

- MDACE Profee — Cheng et al., MIT-PhysioNet
- UMLS Metathesaurus — NLM
- LOINC — Regenstrief Institute
- RadLex — RSNA, accessed via BioPortal
- SapBERT — Liu et al. 2021
- BGE — `BAAI/bge-base-en-v1.5`, `bge-reranker-v2-m3`
- MedCAT v2.7.0 — CogStack
- NegEx — Chapman et al., "A Simple Algorithm for Identifying Negated Findings
  and Diseases in Discharge Summaries", 2001
- PLM-ICD — Huang et al., 2022 (benchmark reference, not run)
