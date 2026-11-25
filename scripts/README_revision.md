# Rating, metric, ASR and cost analyses

Run every script from the repo root. Outputs go to `rag_models/results/revision/`.
The failed generation (MedGemma-4B, MSK0037) is excluded everywhere: 159 notes, 477 ratings, 39 complete conversation blocks in round one.
Round two adds 80 notes and 240 ratings (LLaMA-4-Scout and Qwen3.8-27B): 239 notes and 717 ratings in all. The round-two rating files in
`rag_models/rating_packets_round2/` hold scores only. Each round-two packet also contains 15 unmarked notes from round one.

| Script | Computes | Output files |
|---|---|---|
| `revision_rating_analyses.py` | Round-one ratings: model means, Friedman and pairwise Wilcoxon tests, score distributions by rater, agreement, mixed models, specialty subanalyses, keyword flags, power sensitivity, length correlations | `rating_analyses_output.txt` |
| `revision_asr_fareez_wer.py` | Fareez word error rate with speaker labels removed from the reference for every ASR model | `fareez_wer_labels_removed.csv`, `asr_fareez_wer_output.txt` |
| `revision_fareez_metrics.py` | BLEU, ROUGE-L, BERTScore and sentence-embedding similarity per note, against the source transcript | `fareez_automated_metrics.csv` |
| `revision_entity_metrics.py` | MedCAT and scispaCy entity recall, precision and F1 per note | adds entity columns to `fareez_automated_metrics.csv`, `entity_metrics_output.txt` |
| `revision_metric_vs_clinician.py` | Note-level association of each automated metric with the ratings, length-adjusted, with between-model tests and mixed models (round one) | `metric_clinician_correlations.csv`, `metric_between_model_tests.csv`, `metric_by_model.csv`, `metric_clmm_odds_ratios.csv`, `metric_vs_clinician_output.txt` |
| `revision_retrieval_similarity.py` | Retrieval similarity per conversation and its correlation with the ratings | `retrieval_similarity_by_conversation.csv`, `retrieval_similarity_vs_ratings.csv`, `retrieval_similarity_output.txt` |
| `revision_clmm_crossed.R` | Mixed models with rater as a random effect (sensitivity analysis) | `clmm_crossed.csv`, `clmm_crossed_output.txt` |
| `revision_cost.py` | Marginal compute cost per encounter and self-hosted total cost of ownership | `cost_model.csv`, `cost_output.txt` |
| `revision_round2_metrics.py` | Automated metrics for the two models added in rating round two | `fareez_round2_metrics.csv` |
| `revision_round2_analysis.py` | Merged ratings of both rounds; note-level metric versus rating correlations and mixed models for six models | `round2_ratings_long.csv`, `round2_model_means.csv`, `round2_metric_clinician_correlations.csv`, `round2_metric_clmm_odds_ratios.csv`, `round2_analysis_output.txt` |
| `revision_six_model_analysis.py` | Six-model ratings, repeated-note comparison, Friedman and pairwise tests, mixed models with a round effect, agreement, per-rater results, between-model metric tests | `six_model_output.txt`, `six_model_table.csv`, `six_model_pairwise.csv`, `six_model_clmm.csv` |

`ratings_long.csv` is the merged round-one rating file (one row per rating, with a `failed_generation` flag). `revision_rating_analyses.py` writes it and the R script reads it.

## Order

```
python scripts/revision_rating_analyses.py --repo .
python scripts/revision_asr_fareez_wer.py --repo .          # writes its CSV to the current directory
python scripts/revision_fareez_metrics.py --repo .
python scripts/revision_entity_metrics.py --repo .          # runs the entity evaluator on Modal
python scripts/revision_metric_vs_clinician.py --repo .
python scripts/revision_retrieval_similarity.py --repo . --chroma /path/to/chroma_schema_improved
Rscript scripts/revision_clmm_crossed.R
python scripts/revision_cost.py --repo .
python scripts/revision_round2_metrics.py --repo . --models qwen3.8-27b llama-4-scout   # runs the entity evaluator on Modal
python scripts/revision_round2_analysis.py --repo .
python scripts/revision_six_model_analysis.py --repo .       # reads round2_ratings_long.csv; about 15 minutes
```

## Requirements

- Python: numpy, pandas, scipy for the rating, ASR, correlation, and cost scripts. The metric script also needs nltk,
  rouge-score, bert-score, and sentence-transformers. The retrieval script needs chromadb and sentence-transformers.
  The entity script needs modal and a Modal account.
- R: package `ordinal` (the crossed model was fitted with ordinal 2026.7.26 on R 4.5.3).
- Transcripts: the Fareez "Clean Transcripts" folder under `openemr_whisper_wer/data/fareez_osce/Data/`, or, if that is
  absent, the reference text stored in `openemr_whisper_wer/results/fareez-groq.csv`.
- Schema library: the Chroma store is derived from credentialed MIMIC-IV notes and is not in git. Keep it outside the
  repo and pass its path with `--chroma`. The retrieval script reads only the diagnosis-list metadata.

## Data handling

No script sends MIMIC-IV text or schema library content to an external model API. The entity evaluator runs
self-hosted scispaCy and MedCAT models in a Modal container and receives only the generated notes and the public Fareez transcripts.

New scripts write the column `sbert_similarity`. Older files keep `sbert_coherence`; it is the same quantity.

## Full-precision MedGemma-27B

`rag_models/models/medgemma_27b_modal.py` runs `google/medgemma-27b-text-it` in bfloat16 on a Modal A100-80GB over the six
institutional cases. Three passes are in `rag_models/results/institutional/medgemma-27b/run{1,2,3}/`. The 4-bit run is the `medgemma-27b-4bit` row of
`rag_models/results/institutional/institutional_rag_6case.csv`.
