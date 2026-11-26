# openemr-ai

Artificial Intelligence tooling for OpenEMR. The repository holds three separate research areas. Each has its own code and its own data, and results from one area should not be read as part of another.

| Research area | What it studies | Code | Data |
|---|---|---|---|
| **1. Ambient listening and summarization** | Patient-provider conversation to transcript (ASR), transcript to SOAP note (retrieval-augmented LLM), and blinded clinician rating of the notes | `openemr_whisper_wer/`, `rag_models/`, `smart-ambient-listening/`, `scripts/revision_*` | See the data table below |
| **2. CQL clinical decision support (CQL-CDS)** | Whether LLMs can tell if an ELM JSON rule correctly implements a clinical practice guideline | `cdr_elmjson_validator/` | 41 hand-curated ELM JSON cases built from published guidelines. No patient data |
| **3. Automated coding** | Multi-label CPT and ICD-10-CM coding of clinical notes | `coding_bench/` | MDACE Profee annotations over MIMIC-III notes |

### Data used in the ambient listening and summarization work

| Experiment | Data | What is used |
|---|---|---|
| ASR word error rate | Six institutional conversations recorded at IU Indianapolis | Audio and reference transcripts |
| ASR word error rate | Kaggle Medical Speech, Transcription, and Intent | Short single-speaker utterances |
| ASR word error rate | PriMock57 | 57 mock primary care consultations, audio |
| ASR word error rate | Fareez OSCE | 272 simulated interviews, audio |
| Note schema library, built once offline | 1,000 MIMIC-IV notes | Structure-only outlines (section, test and diagnosis names, with no values, dates or narrative), extracted with a locally hosted MedGemma-27B. Not redistributed here |
| Seven-model reference-based note evaluation | Six institutional cases | Transcript, OpenEMR chart extract and a manually written reference SOAP note |
| Blinded clinician evaluation and automated metrics | 40 of the 272 Fareez OSCE conversations | Gold speaker-labelled transcript (not ASR output), a paired chart from the OpenEMR training instance (a de-identified MIMIC-IV-derived record, matched on condition) and two retrieved note outlines |
| RAG ablations | The six institutional cases and the 40 Fareez conversations | Same inputs with one factor changed |

MIMIC-IV has two roles: its notes supply the structure of the generated note, and MIMIC-IV-derived records in the OpenEMR training instance supply the chart paired with each Fareez conversation. The conversations themselves come from the institutional recordings and from Fareez OSCE. The OpenEMR training instance holds 11,030 patients: MIMIC-IV-derived records, Synthea-generated records added on top, and the six institutional case records. The CQL-CDS and automated coding sections further down are separate studies.

## Ambient listening: ASR Model Evaluation (Word Error Rate)

Comprehensive evaluation of Automatic Speech Recognition (ASR) models for medical transcription across four datasets:

1. **Institutional Dataset**: 6 custom simulated doctor-patient conversations recorded at IU Indianapolis
2. **Kaggle Dataset**: [Medical Speech, Transcription, and Intent](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent) (385 utterances)
3. **PriMock57**: 57 primary care mock consultations from [PriMock57](https://github.com/babylonhealth/primock57) (~9 hours, real clinicians, utterance-level TextGrid transcripts)
4. **Fareez OSCE**: 272 simulated patient-physician OSCE interviews from [Fareez et al.](https://springernature.figshare.com/collections/5545842) (51.9 hours of audio, 5 medical specialties)

### Models Evaluated

| Model | Provider | Notes |
|-------|----------|-------|
| Whisper Large v3 Turbo | OpenAI | Fast, accurate general-purpose ASR |
| Whisper Large v3 | OpenAI | Higher accuracy, slower than turbo |
| Canary-1B-v2 | NVIDIA | Multi-task ASR from NeMo toolkit |
| Parakeet-TDT-1.1B | NVIDIA | Fast Token-and-Duration Transducer |
| **Parakeet-TDT-0.6B-v2** | NVIDIA | Smaller TDT (added 2026-04); lowest Fareez WER (12.51%) on the 250 files it completed |
| **Voxtral Mini 3B** | Mistral | Multimodal speech-LM (added 2026-04); lowest WER on the institutional recordings (5.71%) |
| **Qwen3-ASR-1.7B** | Alibaba | Speech-LM (added 2026-11); lowest PriMock57 WER (14.66%) |
| Granite Speech 5.0 2B | IBM | Speech-LM (added 2026-11); long recordings are cut at quiet points into pieces of at most 100 s |
| **Ishiki ASR 1.0** | Ishiki Labs | Seven open speech models in one 13.3 B-parameter checkpoint, with a router that merges their outputs (added 2026-11); lowest Kaggle WER (10.98%). Its text is already normalized, see the note under the results |
| Groq Whisper | Groq | API-based Whisper Large v3 Turbo |
| MedASR | Google | Conformer architecture pre-trained for medical dictation |

### Latest Results

#### Average WER (%) across all 4 datasets

Bold = lowest in that column.

| Model | GPU used | Institutional (n=6) | Kaggle (n=385) | PriMock57 (n=57) | Fareez OSCE (n=270) |
|-------|----------|:-------------------:|:--------------:|:----------------:|:--------------------:|
| Voxtral Mini 3B           | L40S                  | **5.71**  | 15.33      | 25.15          | 13.87               |
| Granite Speech 5.0 2B     | A10G                  | 7.06      | 14.41      | 19.75          | 13.54               |
| Qwen3-ASR-1.7B            | A10G                  | 7.25      | 13.36      | **14.66**      | 14.55               |
| Parakeet TDT 1.1B         | A100, A10G for Kaggle | 8.30      | 16.28      | 17.06          | 14.19 †             |
| Parakeet TDT 0.6B v2      | A100, A10G for Kaggle | Not run   | 14.57      | 18.16          | **12.51** †         |
| Groq Whisper v3 Turbo     | None (Groq API)       | 8.48      | 15.05      | 18.87          | 13.61               |
| Canary 1B-v2              | A10G                  | 8.58      | 15.05      | 21.54          | 15.25               |
| Whisper Edge              | Not recorded          | 9.93      | 35.01      | Not run        | Not run             |
| Whisper Large v3 Turbo    | A100, A10G for Kaggle | 13.88     | 23.88      | 21.00          | 19.03               |
| Ishiki ASR 1.0 §          | L40S                  | 15.06     | **10.98**  | 22.17          | 20.83               |
| Whisper Large v3          | A100, A10G for Kaggle | 17.11 ‡   | 19.35      | 22.28          | 17.80 *             |
| MedASR                    | A10G                  | 43.94     | 39.25      | 64.59          | 43.95               |

† 250 of 270 files; 20 long recordings failed with GPU out-of-memory errors.
* 83 of 270 files; the run was stopped by a Modal billing limit.
‡ One recording (`gerd`) scored 54.65% with 360 inserted words and no deletions, a repetition loop; the other five average 9.61%. Earlier versions of this table gave 13.26%, which a bug in `whisper_wer.py` had produced with Whisper Large v3 Turbo.

§ Ishiki ASR 1.0 returns text that is already normalized, for example "i have" for "I've" and "3" for "three". The WER in this table removes only case and punctuation, so it counts each of those rewrites as an error. The table below scores it like for like.

#### WER (%) after text normalization

The three models run by `speech_llm_wer.py`, scored after the Whisper English text normalizer is applied to both the reference and the transcript (numbers as digits, contractions expanded, fillers such as "um" dropped; column `wer_norm` of their result files). Normalization lowers every model's WER, so these values are not comparable with the table above. Bold = lowest in that column.

| Model | Institutional (n=6) | Kaggle (n=385) | PriMock57 (n=57) | Fareez OSCE (n=270) |
|-------|:-------------------:|:--------------:|:----------------:|:--------------------:|
| Ishiki ASR 1.0            | **4.91**  | **10.05**  | **11.56**      | **8.47**            |
| Qwen3-ASR-1.7B            | 5.37      | 12.90      | 11.63          | 10.28               |
| Granite Speech 5.0 2B     | 5.47      | 13.59      | 15.10          | 10.88               |

#### GPU and cost

The GPU column is the GPU that each model's script requests on Modal. Modal's on-demand prices on 2026-11-26 were $1.10 per hour for an A10G (24 GB), $1.95 for an L40S (48 GB) and $2.10 for an A100 (40 GB); CPU and memory are billed on top. Groq Whisper is a hosted API and needs no GPU. Whisper Edge transcripts were produced outside these scripts and the hardware is not recorded here.

Four models recorded how fast they ran on the PriMock57 and Fareez recordings (8.6 and 51.9 hours of audio):

| Model | GPU | Speed (times real time) | Peak GPU memory | GPU cost per hour of audio |
|-------|-----|:-----------------------:|:---------------:|:--------------------------:|
| Granite Speech 5.0 2B     | A10G | 59 to 74 | 7.3 GiB      | about $0.02      |
| Ishiki ASR 1.0            | L40S | 53 to 58 | 34.5 GiB     | about $0.04      |
| Qwen3-ASR-1.7B            | A10G | 18 to 20 | 20.6 GiB     | about $0.06      |
| Voxtral Mini 3B           | L40S | 21 to 27 | Not recorded | $0.07 to $0.10   |

These figures count transcription time only. Starting a container, loading the model and idle time add to the bill. Ishiki ASR 1.0's weights alone are 24.8 GiB, so it does not fit a 24 GB GPU. Qwen3-ASR-1.7B is the slowest of the three because it transcribes each recording in one pass.

#### Where the values come from

The institutional and Kaggle values of the nine models in the WER workflow (Parakeet TDT 1.1B, Canary 1B-v2, Groq Whisper v3 Turbo, Whisper Large v3 Turbo, Whisper Large v3, MedASR, Qwen3-ASR-1.7B, Granite Speech 5.0 2B and Ishiki ASR 1.0) are from the workflow run of 2026-11-26 and are the means of the per-file scores in `openemr_whisper_wer/results/institutional-*.csv` and `kaggle-*.csv`. Values that changed from the runs before that date: Groq institutional 9.60 to 8.48 and Kaggle 14.91 to 15.05, Whisper Large v3 Turbo 14.02 to 13.88 and 23.92 to 23.88, MedASR institutional 44.01 to 43.94.

The six older models, Qwen3-ASR-1.7B and Granite Speech 5.0 2B were each run twice that day. The second run reproduced 11 of the 12 result files of the six older models exactly. Whisper Large v3's institutional mean moved from 17.13 to 17.11 and Qwen3-ASR-1.7B's Kaggle mean from 13.22 to 13.36 (6 of 385 files), so a difference of about 0.1 between two models is within run-to-run variation. Granite Speech 5.0 2B's institutional mean moved from 7.66 to 7.06 after the script was changed to transcribe the pieces of a recording as one batch; three of the six recordings changed, and its Kaggle files, which are one piece each, did not.

Qwen3-ASR-1.7B, Granite Speech 5.0 2B and Ishiki ASR 1.0 were added on 2026-11-26. Their PriMock57 and Fareez values are from the local dataset mode of `speech_llm_wer.py` (`primock57-*.csv` and `fareez-*.csv` for `qwen3-asr`, `granite-speech` and `ishiki-asr`). All 57 PriMock57 recordings and all 270 Fareez recordings were scored for each model. Their Fareez references have the speaker labels removed and leave out the two unreadable references, as for the other models.

Fareez WER is computed with the "D:" and "P:" speaker labels removed from every reference. Two references with text-encoding errors (RES0002, RES0054) are excluded, which leaves 270 of the 272 files. Earlier versions of this table removed the labels for Voxtral Mini 3B only, which inflated the Fareez WER of every other model; those values (for example 14.50% for Voxtral and 18.63% for Parakeet TDT 0.6B v2) are superseded. Per-file values are in `rag_models/results/revision/fareez_wer_labels_removed.csv` and are reproduced by `scripts/revision_asr_fareez_wer.py`.

On the 250 Fareez files completed by seven of the eight models, Parakeet TDT 0.6B v2 is still lowest (12.51%), followed by Voxtral Mini 3B (13.14%), Groq Whisper v3 Turbo (13.59%), and Parakeet TDT 1.1B (14.19%).

**Dataset-level headline numbers:**
- Institutional (6 custom medical recordings): Voxtral Mini 3B wins (5.71%), followed by Granite Speech 5.0 2B (7.06%) and Qwen3-ASR-1.7B (7.25%); all conversational ASRs ≤10% except Whisper Large variants and Ishiki ASR 1.0, whose 15.06% falls to 4.91% once its normalized text is scored like for like.
- Kaggle (385 short-form medical utterances, `validate` split): Ishiki ASR 1.0 wins (10.98%), ahead of Qwen3-ASR-1.7B (13.36%), Granite Speech 5.0 2B (14.41%) and Parakeet TDT 0.6B v2 (14.57%). Canary 1B-v2 and Groq Whisper follow (both 15.05%), then Voxtral Mini 3B (15.33%).
- PriMock57 (57 primary-care consultations, ~9h, real clinicians): Qwen3-ASR-1.7B wins (14.66%), ahead of Parakeet TDT 1.1B (17.06%) and Parakeet TDT 0.6B v2 (18.16%). Voxtral drops to 25.15% on these stereo-mixed doctor/patient recordings.
- Fareez OSCE (272 multi-specialty interviews, 51.9 h, mono; 270 scored): Parakeet TDT 0.6B v2 is lowest (12.51% on the 250 files it completed), followed by Granite Speech 5.0 2B (13.54%), Groq Whisper v3 Turbo (13.61%) and Voxtral Mini 3B (13.87%), all three on all 270 files.

**Key Findings (revised after the Fareez WER correction):**
- **No model is best on every dataset.** Voxtral Mini 3B has the lowest WER on the institutional recordings (5.71%), Ishiki ASR 1.0 on Kaggle (10.98%), Qwen3-ASR-1.7B on PriMock57 (14.66%), and Parakeet TDT 0.6B v2 on Fareez OSCE (12.51%).
- **The deployed transcription service runs Parakeet TDT 1.1B** on an A10G GPU (`smart-ambient-listening/transcription-service/modal_asr.py`). Any of the benchmarked models can be substituted without changes elsewhere.
- **Voxtral Mini 3B** takes about 25 s per Fareez file on an L40S GPU, against 1 to 3 s for the Parakeet and Whisper Turbo models, and has the highest WER of the general-purpose models on PriMock57 (25.15%).
- **Groq Whisper v3 Turbo** is within 4 percentage points of the best model on every dataset and completed every file.
- **Parakeet TDT 0.6B v2** has lower Fareez WER than its 1.1B parent (12.51% vs 14.19% on the same 250 files) at about half the parameter count.
- **Canary 1B-v2** completed every file, with WER from 8.58% to 21.54% across datasets.
- **MedASR** has the highest WER on every dataset (39.25% to 64.59%).

> Per-file results are in `openemr_whisper_wer/results/`. The [WER evaluation workflow](../../actions/workflows/wer-evaluation.yml) commits the institutional and Kaggle per-file scores there on each run.

## Ambient listening: SOAP Note Generation and Blinded Clinician Evaluation

This repository includes an evaluation of LLM models for generating medical discharge summaries in SOAP format using Retrieval-Augmented Generation (RAG). We test multiple models on their ability to create structured clinical summaries from doctor-patient transcripts and electronic health records.

**What is RAG?** Retrieval-Augmented Generation combines language models with a vector database to retrieve relevant context (SOAP note schemas from MIMIC Clinical Notes) before generating summaries. This ensures outputs follow proper clinical documentation structure.

**Vector Database**: ChromaDB hosted on Modal Volume containing SOAP note structures extracted from MIMIC Clinical Notes. Uses **all-MiniLM-L6-v2** for semantic retrieval of top 2 relevant schemas based on detected disease.

**Test Dataset**: 6 patient test cases with doctor-patient transcripts, OpenEMR extracts, and reference summaries for evaluation.

### Models Evaluated

| Model | Provider | Size | Infrastructure | Notes |
|-------|----------|------|----------------|-------|
| MedGemma 4B-IT | Google | 4B | Modal (A10G GPU) | Healthcare-specialized model |
| MedGemma 27B 4-bit | Google | 27B | Modal (A10G GPU) | Quantized healthcare model |
| Llama 3.1 8B Instruct | Meta | 8B | Modal (A10G GPU) | vLLM for fast inference |
| Qwen 3 32B | Alibaba | 32B | Modal (A10G GPU) | General-purpose |
| GPT-OSS 20B | Groq | 20B | Groq API | Fast inference |
| GPT-OSS 120B | Groq | 120B | Groq API | Large model |
| Llama 4 Scout | Groq | - | Groq API | Latest Llama generation |

### Results

Results are aggregated across 3 independent runs using `rag_models/evaluation/aggregate_results.py`.

**Evaluation Metrics:**
- **BLEU & ROUGE-L**: Lexical overlap and structural recall
- **SBERT Similarity**: Global semantic coherence (0.80-0.90 indicates clinical equivalence)
- **BERTScore F1**: Local semantic accuracy for clinical details
- **SciSpaCy/MedCAT Entity Recall**: Medical entity extraction coverage

### Fareez OSCE Experiment (40 Conversations x 6 Models, 239 Rated Notes)

**Dataset**: 40 conversations from [Fareez et al.](https://springernature.figshare.com/collections/5545842) balanced across 5 medical specialties (20 RES, 10 MSK, 5 GAS, 4 CAR, 1 DER). Each conversation paired with a condition-matched, MIMIC-IV-derived chart from the OpenEMR training instance (11,030 patients in total).

**Design**: All models use the same pre-detected conditions (via Llama 3.3 70B) for ChromaDB schema retrieval, isolating summary generation quality as the only variable.

| Model | Size | Infrastructure | Summaries | Avg Time | Avg Chars |
|-------|------|----------------|-----------|----------|-----------|
| GPT-OSS 120B | 120B | Groq (reasoning API) | 40/40 | 5.4s | 7,072 |
| Qwen3 32B | 32B | Groq API | 40/40 | 4.3s | 5,568 |
| GPT-OSS 20B | 20B | Groq (reasoning API) | 40/40 | 2.9s | 5,275 |
| MedGemma 4B-IT | 4B | Modal (A10G GPU) | 39/40 | 44.5s | 2,741 |
| Qwen3.8 27B (round two) | 27B | Groq API | 40/40 | 8.9s | 6,290 |
| LLaMA 4 Scout (round two) | 17B | Amazon Bedrock | 40/40 | 5.5s | 3,421 |

**Clinician Evaluation**: 3 physician fellows (GI, IR, ER) independently rated the notes on 6 dimensions using 5-point Likert scales adapted from PDQI-9 and QNOTE, in two blinded rounds. Round one covered four models (159 notes, 477 ratings; the one failed MedGemma 4B generation, MSK0037, is excluded). Round two added Qwen3.8 27B and LLaMA 4 Scout (80 notes, 240 ratings) and repeated 15 unmarked round-one notes per rater. The total is 239 notes and 717 ratings. Round-two rating files hold scores only. Inter-rater agreement uses Gwet's AC2 (ordinal weights) and Krippendorff's alpha. Models are compared with a Friedman test, Bonferroni-corrected pairwise Wilcoxon signed-rank tests, and cumulative link mixed models with a round effect.

#### Clinician Rating Results (Mean (SD), 5-point Likert)

| Model | Ratings | Accuracy | Completeness | Organization | Conciseness | Clinical Utility | Overall Quality | Composite |
|-------|---------|----------|--------------|--------------|-------------|------------------|-----------------|-----------|
| Qwen3.8 27B | 120 | 2.85 (0.83) | 4.22 (0.46) | 4.88 (0.33) | 3.28 (0.68) | 3.67 (0.80) | 3.54 (0.68) | 3.74 |
| GPT-OSS 120B | 120 | 2.70 (0.90) | 4.14 (0.57) | 5.00 (0.00) | 3.64 (0.99) | 3.54 (0.88) | 3.44 (0.86) | 3.74 |
| GPT-OSS 20B | 120 | 2.62 (0.92) | 3.92 (0.64) | 5.00 (0.00) | 3.92 (0.78) | 3.47 (0.85) | 3.35 (0.86) | 3.71 |
| Qwen3 32B | 120 | 2.50 (0.90) | 3.83 (0.62) | 4.94 (0.27) | 3.70 (0.71) | 3.42 (0.91) | 3.29 (0.90) | 3.61 |
| MedGemma 4B | 117 | 2.41 (0.79) | 3.02 (0.68) | 3.98 (1.02) | 3.39 (1.21) | 2.66 (0.81) | 2.55 (0.90) | 3.00 |
| LLaMA 4 Scout | 120 | 2.10 (0.59) | 3.11 (0.46) | 3.67 (0.92) | 3.21 (0.73) | 2.58 (0.57) | 2.52 (0.53) | 2.86 |

**Inter-rater agreement, both rounds (239 notes):** Gwet's AC2 ranged from 0.71 to 0.93 across dimensions and Krippendorff's alpha from 0.04 to 0.51 (accuracy 0.22). AC2 is high partly because scores cluster in a narrow range; alpha shows that raters agreed much less on which notes were better. Per-round and pairwise values are in `rag_models/results/revision/six_model_output.txt`.

**Friedman test over six models** (39 complete conversation blocks, all P<.001): Organization W=0.88, Completeness W=0.77, Clinical Utility W=0.52, Overall Quality W=0.51, Conciseness W=0.32, Accuracy W=0.26.

#### Automated Metrics (40 Conversations, transcript as reference)

| Model | BLEU | ROUGE-L | SBERT Similarity | BERTScore F1 | MedCAT Entity Recall | Avg Words |
|-------|------|---------|------------------|--------------|----------------------|-----------|
| Qwen3.8 27B | 0.025 | 0.143 | 0.504 | 0.803 | 0.489 | 927 |
| GPT-OSS 120B | 0.011 | 0.115 | 0.511 | 0.799 | 0.429 | 1,005 |
| GPT-OSS 20B | 0.009 | 0.118 | 0.529 | 0.799 | 0.396 | 756 |
| Qwen3 32B | 0.015 | 0.129 | 0.491 | 0.802 | 0.404 | 831 |
| MedGemma 4B | 0.011 | 0.145 | 0.506 | 0.801 | 0.306 | 565 |
| LLaMA 4 Scout | 0.014 | 0.147 | 0.534 | 0.807 | 0.359 | 529 |

**Key Findings:**
- **Qwen3.8 27B and GPT-OSS 120B** had the highest composite clinician rating (3.74/5), followed by GPT-OSS 20B (3.71) and Qwen3 32B (3.61). MedGemma 4B (3.00) and LLaMA 4 Scout (2.86) were rated lowest.
- **Accuracy was the weakest dimension for every model** (2.10 to 2.85), and it separated the models least (W=0.26).
- **LLaMA 4 Scout had the highest BERTScore F1, SBERT similarity and ROUGE-L but the lowest clinician ratings.** At the note level BERTScore was unrelated to the composite rating (Spearman rho -0.06, 95% CI -0.22 to 0.10).
- **Entity recall tracked completeness.** MedCAT entity recall correlated with rated completeness (rho 0.54, 95% CI 0.43 to 0.65; 0.36 after adjusting for note length).
- The scripts and outputs for these analyses are listed in [`scripts/README.md`](scripts/README.md).

**Full-precision MedGemma 27B (institutional six cases):** `google/medgemma-27b-text-it` in bfloat16 on a Modal A100-80GB, three passes, gave BLEU 0.092, ROUGE-L 0.353, SBERT 0.810, BERTScore F1 0.871, scispaCy recall 0.618 and MedCAT recall 0.629. Every metric is within 0.013 of the 4-bit run. Per-run files are in `rag_models/results/institutional/medgemma-27b/`.

### RAG Ablation (Institutional 6-Case)

Three ablations on the same six institutional conversations, 3 runs per LLM (7 LLMs × 2 conditions = 42 runs for the no-RAG baseline alone):

1. **No-RAG baseline** — schema retrieval disabled.
2. **Retrieval-depth sweep** — k ∈ {1, 2, 3, 5}.
3. **Embedding-model substitution** — all-MiniLM-L6-v2 vs. ClinicalBERT vs. PubMedBERT.

Canonical numeric outputs and a reproducibility notebook:

- [`rag_models/results/institutional/institutional_rag_6case.csv`](rag_models/results/institutional/institutional_rag_6case.csv) — RAG on (k=2).
- [`rag_models/results/institutional/institutional_norag_6case.csv`](rag_models/results/institutional/institutional_norag_6case.csv) — no-RAG baseline.
- [`rag_models/results/institutional/ablation_analysis.ipynb`](rag_models/results/institutional/ablation_analysis.ipynb) — loads both CSVs, reports deltas, and recomputes every ablation figure quoted below.

Findings: disabling retrieval degrades the entity- and structure-sensitive metrics across all seven LLMs (MedCAT −0.020 to −0.060, ROUGE-L −0.005 to −0.050, BERTScore F1 −0.020 to −0.060); BLEU and SBERT show the same directional pattern but several cells (e.g., GPT-OSS-120B BLEU 0.071→0.069) are within three-run variance. k=3 and k=5 degraded quality vs. k=2 on the gold-referenced n=6 cases; on the Fareez transcript-reference ablation (Table 2c) the same sweep fell within run-to-run variance. ClinicalBERT and PubMedBERT matched all-MiniLM-L6-v2 at higher latency. Full methodology in [`rag_models/README.md`](rag_models/README.md#rag-ablation-study-institutional-6-case-evaluation).

### RAG Ablation Replication on Fareez n=40

The n=6 ablation above was replicated on the 40 Fareez OSCE conversations used for clinician evaluation, with the four clinician-evaluated models (GPT-OSS-120B/20B, Qwen3-32B, MedGemma-4B) plus temperature/prompt sweeps on the top-2 models. **1,354 summaries across 12 cells × 4 models** (32 ablation cells in total).

Canonical artefacts in [`rag_models/results/fareez/ablations/`](rag_models/results/fareez/ablations/):

- `fareez_ablation_metrics.csv` — per-summary text + entity metrics (1,354 rows)
- `ablation_means.csv` — per-(cell, model) means
- `ablation_deltas.csv` — production-minus-cell deltas
- `ablation_bootstrap_ci.csv` — paired bootstrap 95% CIs (B=5000)
- `ablation_wilcoxon.csv` — Wilcoxon signed-rank p-values
- `ablation_analysis_n40.ipynb` — reproducibility notebook

**Headline result:** Of 136 production-vs-cell comparisons across 4 metrics, **only 26 (19%) had bootstrap CIs that excluded zero**, almost all in the prompt-variant cells (`prompt_minimal` 8/4, `prompt_hallucination_guarded` 6/4). The no-RAG, k-sweep, and embedding-substitution cells produced effects within run-to-run noise on the transcript-as-reference metric set. This is consistent with — and methodologically reinforces — the paper's central finding that automated metrics fail to discriminate the model-quality differences clinicians clearly identified (composite 2.95–3.74 on the same 40 conversations).

The n=6 institutional ablation results (evaluated against gold-standard SOAP references) remain the more sensitive indicator of the RAG pipeline's contribution; the n=40 replication is a sensitivity check rather than a replacement.

```bash
cd rag_models

# Run Groq models locally
python models/run_fareez_local.py --output-dir results/fareez

# Run MedGemma on Modal
HF_TOKEN=your_token modal run pipeline/run_fareez_summaries.py --output-dir results/fareez

# Generate randomized rating packets for clinician evaluation
python clinician_validation/generate_rating_packets.py --output-dir rating_packets

# Run clinician rating analysis
python clinician_validation/analyze_ratings.py

# Round two: models/build_fareez_prompts.py, models/run_fareez_round2_groq.py, models/run_fareez_round2_bedrock.py
# and clinician_validation/generate_round2_packets.py (arguments are in each docstring); analyses in scripts/README.md
```

## CQL-CDS: ELM JSON Validation (Clinical Decision Support Validation)

Evaluation of LLM models for validating clinical decision support logic. Tests whether LLMs can identify if ELM (Expression Logical Model) JSON implementations correctly implement Clinical Practice Guidelines (CPG).

**What is ELM?** ELM is the compiled JSON representation of Clinical Quality Language (CQL) - the standard for expressing clinical decision rules.

**Approach:**
- **ELM Simplifier**: Deterministic code extracts key values (age thresholds, time intervals, value sets) from complex nested JSON, reducing 500+ lines to ~20 lines of structured data
- **Step-by-step prompts**: Models compare extracted values against CPG requirements with explicit decision criteria
- **Batch processing**: Each model loads once and processes all test files efficiently

**Test Dataset**: 41 hand-curated, ground-truth annotated ELM JSON artifacts derived from published clinical practice guidelines (16 valid, 25 invalid). Invalid cases include 13 parametric errors (wrong age thresholds, lookback intervals, clinical values) and 12 semantic logic errors spanning 4 categories: missing condition, inverted logic, wrong nesting, and swapped references. See [`test_data/BENCHMARK.md`](cdr_elmjson_validator/test_data/BENCHMARK.md) for full methodology.

### Models Evaluated (15 open-weight models + 1 proprietary reference)

| Model | Provider | Params / Active | Infrastructure |
|-------|----------|-----------------|----------------|
| **Qwen 3 32B** | Alibaba | 32B dense | Groq API |
| **GPT-OSS 120B** | OpenAI | 120B / 5.1B active MoE | Groq API |
| **Qwen3.5 35B A3B** | Alibaba | 35B / 3B active MoE | OpenRouter API |
| **Llama 3.3 70B** | Meta | 70B dense | Groq API |
| **Gemma 4 26B A4B** | Google | 26B / 4B active MoE | Modal L40S (Unsloth GGUF Q4\_K\_M via llama.cpp) |
| **GPT-5.4-mini** (proprietary reference) | OpenAI | — | OpenAI API |
| **Gemma 4 31B** | Google | 31B dense | Modal A100 80GB (Unsloth GGUF Q4\_K\_M via llama.cpp) |
| **GPT-OSS 20B** | OpenAI | 20B / 3.6B active MoE | Groq API |
| MedGemma 4B / 1.5 4B | Google | 4B | Local RTX 6000 (bfloat16) |
| Gemma 3 4B | Google | 4B | OpenRouter API |
| Phi-3 Mini | Microsoft | 3.8B | Local RTX 6000 |
| Llama 3.2 1B/3B | Meta | 1-3B | Local RTX 6000 |
| Qwen 2.5 1.5B/3B | Alibaba | 1.5-3B | Local RTX 6000 |

### Latest Results (41 Test Cases, 16 Valid / 25 Invalid: 13 parametric, 12 semantic)

All results are means over 5 independent trials at temperature 0.1 (T=0.1 sampling for most models; greedy decoding for Gemma 4 GGUF which is deterministic across trials). Base rate: 39.0%. Each model is measured at its ecosystem-standard deployment precision: GPT-OSS at MXFP4 (native release via Groq LPU), Qwen3/Llama/Qwen3.5 at fp16 (Groq/OpenRouter default), Gemma 4 at Unsloth GGUF Q4\_K\_M via llama.cpp on Modal GPUs (the dominant 4-bit deployment format for edge inference), GPT-5.4-mini via OpenAI API, and mid/small models locally at bf16.

| Model | Params | Accuracy | ±SD | Sens. | Spec. | F1 | Param | Semantic |
|-------|--------|----------|-----|-------|-------|-----|-------|----------|
| **Qwen3 32B** | 32B | **90.7%** | 2.0 | 0.88 | 0.93 | **0.88** | 100% | 85% |
| **GPT-OSS 120B** | 120B (5.1B active MoE) | 85.9% | 2.0 | 0.64 | 1.00 | 0.78 | 100% | 100% |
| **Qwen3.5 35B A3B** | 35B (3B active MoE) | 85.4% | 3.4 | 0.69 | 0.96 | 0.78 | 100% | 92% |
| **Llama 3.3 70B** | 70B | 83.4% | 1.1 | 0.88 | 0.81 | 0.80 | 100% | 60% |
| **Gemma 4 31B** | 31B | 82.9% | 0.0 | 0.69 | 0.92 | 0.76 | 85% | **100%** |
| GPT-5.4-mini (proprietary) | — | 81.0% | 3.2 | 0.51 | 1.00 | 0.67 | 100% | 100% |
| **Gemma 4 26B A4B** | 26B (4B active MoE) | 80.5% | 0.0 | 0.62 | 0.92 | 0.71 | 100% | 83% |
| **GPT-OSS 20B** | 20B (3.6B active MoE) | 79.5% | 4.1 | 0.60 | 0.92 | 0.70 | 92% | 92% |
| MedGemma 1.5 4B | 4B | 70.2% | 6.1 | 0.64 | 0.74 | 0.62 | 83% | 65% |
| Phi-3 Mini | 3.8B | 50.7% | 2.7 | 0.99 | 0.20 | 0.61 | 37% | 2% |
| Gemma 3 4B | 4B | 48.8% | 0.0 | 1.00 | 0.16 | 0.60 | 31% | 0% |
| Llama-3.2-1B | 1B | 41.5% | 7.3 | 0.78 | 0.18 | 0.50 | 15% | 22% |
| MedGemma 4B | 4B | 39.0% | 0.0 | 1.00 | 0.00 | 0.56 | 0% | 0% |
| Qwen-2.5-3B | 3B | 39.0% | 0.0 | 1.00 | 0.00 | 0.56 | 0% | 0% |
| Llama-3.2-3B | 3B | 37.6% | 2.8 | 0.93 | 0.02 | 0.54 | 2% | 3% |
| Qwen-2.5-1.5B | 1.5B | 34.1% | 0.0 | 0.88 | 0.00 | 0.51 | 0% | 0% |

Benchmark expanded from 31 to 41 cases (added 2 IPF cases + 8 new semantic errors across 4 categories: missing condition, inverted logic, wrong nesting, swapped references). Semantic errors now comprise 12/25 (48%) of invalid cases, up from 3/16 (19%).

### Ablation Study (8 Frontier Models × 4 Conditions × 5 trials)

| Condition | Qwen3-32B | Llama 70B | Qwen3.5-35B-A3B | GPT-OSS-120B | Gemma 4 31B | GPT-5.4-mini | Gemma 4 26B A4B | GPT-OSS-20B |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | *32B dense* | *70B dense* | *3B active* | *5.1B active* | *31B dense Q4* | *proprietary* | *4B active Q4* | *3.6B active* |
| Full (simplified+CPG) | **88.3±2.7** | **84.9±1.1** | **83.9±3.3** | **83.9±2.8** | **82.9±0.0** | **82.9±2.4** | **80.5±0.0** | **80.0±2.0** |
| No CPG | 60.0 (Δ-28) | 51.7 (Δ-33) | 51.7 (Δ-32) | 62.4 (Δ-22) | 61.0 (Δ-22) | 69.3 (Δ-14) | 53.7 (Δ-27) | 62.4 (Δ-18) |
| No Simplification | 83.9 (Δ-4) | 82.9 (Δ-2) | 77.1 (Δ-7) | 80.5 (Δ-3) | **68.3 (Δ-15)** | 75.1 (Δ-8) | **82.4 (Δ+2)** | 70.7 (Δ-9) |
| Neither | 62.4 (Δ-26) | 45.9 (Δ-39) | 40.0 (Δ-44) | 68.8 (Δ-15) | 68.3 (Δ-15) | 71.2 (Δ-12) | 53.7 (Δ-27) | 58.0 (Δ-22) |

**Key Findings:**

- **Qwen3 32B leads overall (90.7% multi-trial, 88.3% full-ablation)** — dense 32B architecture with balanced sensitivity (0.88) and specificity (0.93), the highest F1 (0.88) of any model tested.
- **CPG reference is the dominant contributor**: removing it reduces accuracy by 14–33 pp across frontier models, larger than the effect of removing simplification in every case.
- **The ELM Simplifier acts as a quantization-compensation mechanism for dense architectures at deployment precision.** Gemma 4 31B (dense, Q4\_K\_M) drops 14.6 pp without simplification (82.9% → 68.3%), while Gemma 4 26B A4B (4B active MoE, same Q4\_K\_M) actually *gains* 2.0 pp (80.5% → 82.4%). The architecture-specific effect is striking: at 4-bit precision, sparse MoE experts are inherently more robust to quantization noise and can parse raw ELM JSON without assistance; dense layers lose attention-pattern fidelity and require the simplifier's structured output to recover accuracy.
- **Without the simplifier at Q4, 26B A4B beats 31B dense by 14 pp** (82.4% vs 68.3%) despite having only 4B active parameters. With the simplifier, the dense model recovers to roughly match the MoE (82.9% vs 80.5%). The simplifier is the compensating mechanism that closes the dense-vs-MoE gap at deployment precision.
- **Joint ablation is non-additive**: Qwen3.5-35B-A3B drops 44 pp with both scaffolds removed (vs. 32+7 = 39 pp additive), and Llama 70B drops 39 pp (vs. 33+2 = 35 pp). CPG and simplification interact.
- **Six of eight frontier models achieve 100% on parametric errors** with the full pipeline — the simplifier's Phase 1 (numeric extraction) ceiling-caps numeric comparison. The exceptions are Gemma 4 31B at Q4 (85%) and GPT-OSS-20B (92%); Gemma 4 31B's drop shows that dense 4-bit quantization affects even the simplest comparison task.
- **Semantic error detection varies**: GPT-OSS-120B, GPT-5.4-mini, and Gemma 4 31B achieve 100% on semantic errors; Qwen3.5-35B-A3B and GPT-OSS-20B are close at 92%; Llama 70B lags at 60%. The simplifier's Phase 2 (full logic tree) enables semantic detection.
- **Sensitivity–specificity tradeoff**: specificity-heavy models (GPT-OSS-120B, GPT-5.4-mini) miss 36–49% of valid cases; Qwen3-32B is the only model with both sensitivity ≥0.88 *and* specificity ≥0.93. Llama 70B is the next-best balance (0.88/0.81). The two Gemma 4 variants sit in the middle at ~0.62–0.69 sensitivity / ~0.92 specificity.
- **Medical pretraining (MedGemma)** shows no consistent advantage: MedGemma 4B matches the base rate (39.0%), while MedGemma 1.5 4B reaches 70.2% — the only medical model above mid-range, though still below frontier tier.
- All 4B-and-below models fall below the 48.4% naive baseline on the expanded 41-case benchmark, confirming a capability threshold around 20B total / 4B active parameters.

### Prompt Strategy Comparison (41 Test Cases, 4 Groq Frontier Models × 5 Strategies)

Single-run accuracy at temperature=0.1. Bold indicates the best (or tied-best) strategy per model. Generated by `run_prompt_experiments_41.py` (→ `results/prompts/`) and summarised by `analyze_prompt_41.py` (→ `results/analysis/prompt_accuracy_41.csv`).

| Strategy | Qwen3-32B | GPT-OSS-120B | GPT-OSS-20B | Llama 3.3 70B |
|---|:---:|:---:|:---:|:---:|
| **Few-shot** (2 exemplars) | **92.7** | **90.2** | **85.4** | **85.4** |
| **Standard** (direct comparison) | **92.7** | 85.4 | 75.6 | **85.4** |
| Chain-of-thought | 73.2 | 43.9 | 51.2 | 80.5 |
| Minimal | 70.7 | 39.0 | 39.0 | **85.4** |
| Structured checklist | 43.9 | 39.0 | 39.0 | 53.7 |

**Key Findings:**

- **Few-shot is strictly or tied-best for every model.** It matches standard for Qwen3-32B and Llama 3.3 70B, and improves both GPT-OSS models by 5–10 percentage points over standard.
- **Chain-of-thought is catastrophic for MoE models** (GPT-OSS-120B: 85.4 → 43.9; GPT-OSS-20B: 75.6 → 51.2; Qwen3-32B: 92.7 → 73.2). Llama 3.3 70B is the only model that tolerates CoT (80.5%). Verbose step-by-step reasoning dilutes the explicit key-value comparison that Phase 1 makes trivial.
- **Structured and minimal prompts collapse the MoE models to near base rate** (39%), showing that the clinical comparison task rewards concise, directive instructions anchored on the simplified ELM summary.
- Qwen3-32B's single-run accuracy (92.7%) at standard/few-shot is within sampling variance of its 5-trial mean (90.7% ± 2.0%) from the main table.

Detailed analysis available in the [Jupyter notebooks](cdr_elmjson_validator/notebooks/).

## Automated CPT and ICD-10 Coding

Multi-label CPT and ICD-10-CM coding on MDACE Profee: 312 MIMIC-III notes over 61 CPT codes, and 578 notes over 673 ICD-10 codes. Compares hosted LLMs (Claude Sonnet 5, GPT-OSS 120B, Qwen3.6 27B) against self-hosted quantised models (Muse Glimmer 30B, Gemma 4 26B-A4B) on the same gold labels, the same prompt and the same candidate lists.

The measurement it exists for is the **gold to full catalogue collapse**. Offering a model only the note's correct codes makes precision 1.000 by construction, so that number is a recall ceiling and not a deployment estimate; the `full` rows are the ones that predict real behaviour, and the two must never be compared against each other.

### Best model per condition

<!-- coding-bench-summary:start -->

<!-- Generated by coding_bench/bench/reporting.py. Edits between these markers are overwritten; CI fails if the committed copy is stale. -->

| Task | Candidates | Best model | n | Micro F1 [95% CI] | Latency |
|---|---|---|---:|---|---:|
| cpt | full | Qwen3.6 27B, Claude Sonnet 5 (tied) | 312 | 0.845 [0.802, 0.887] | 4.2s |
| icd10 | full | Qwen3.6 27B, Qwen3.8 27B, Claude Sonnet 5 (tied) | 578 | 0.528 [0.512, 0.542] | 11.9s |
| cpt | gold | Claude Sonnet 5 | 312 | 0.947 [0.928, 0.962] | 1.6s |
| icd10 | gold | Qwen3.6 27B, Claude Sonnet 5 (tied) | 578 | 0.815 [0.801, 0.829] | 6.6s |

Models marked tied are not separated from the first named by a paired bootstrap over the notes they share, so the ordering between them is resampling noise and not a result. The interval, n and latency in such a row describe the first named model only.

<!-- coding-bench-summary:end -->

**Full board, with bootstrap intervals, frequency bands, head-to-head significance and quarantined runs: [`coding_bench/results/LEADERBOARD.md`](coding_bench/results/LEADERBOARD.md).** Methodology, data tiers, cost and guard rails: [`coding_bench/README.md`](coding_bench/README.md). Both the board and the table above are generated from the committed run records, and CI fails if either is stale.

An earlier package, `automated_coding/`, carried the same CPT task plus traditional NLP baselines (regex, MedCAT v2, scispaCy+SapBERT) and its own ablations. It was retired in `9ef7086` once `coding_bench` covered the same ground with a gold-set contract, a prediction cache and per-run provenance. Its two baselines worth keeping were rewritten rather than moved; the rest, including the UMLS/LOINC/RadLex fetchers and the original ablation tables, stay in git history at commit `e460fab`. Numbers published from that package are not comparable with the board above: different gold set contract, different candidate handling, and no quarantine rule for failed runs.

## Running Experiments

### Prerequisites

- Python 3.11+
- [Modal](https://modal.com/) account (for GPU compute with small models)
- `git-lfs` (for PriMock57 dataset download)
- `ffmpeg` (for audio format conversion)
- Environment variables:
  - `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` - Modal authentication
  - `GROQ_API_KEY` - For Groq API evaluations
  - `GOOGLE_API_KEY` - For Gemini/Gemma API models (get at [Google AI Studio](https://aistudio.google.com/app/apikey))
  - `ANTHROPIC_API_KEY` - For Claude API models (get at [Anthropic Console](https://console.anthropic.com))
  - `NOTION_API_KEY` - For Notion dataset access
  - `HF_TOKEN` - For gated HuggingFace models (MedASR)
  - `KAGGLE_USERNAME` and `KAGGLE_KEY` - For Kaggle dataset setup

### ELM JSON Validation

```bash
cd cdr_elmjson_validator

# List available models and experiment modes
python run_validation.py --list-models

# Run a single model
python run_validation.py --model qwen3-32b --output results/results-qwen3-32b.csv

# Run with ablation mode (no CPG, no simplification, etc.)
python run_validation.py --model gpt-oss-20b --ablation-mode no_cpg

# Run with prompt mode (cot, few-shot, structured, minimal)
python run_validation.py --model gpt-oss-20b --prompt-mode few-shot

# Run ablation study across models
python run_ablation.py --model gpt-oss-20b

# Run prompt engineering experiments on the full 41-case benchmark
# (4 Groq frontier models x 5 strategies)
GROQ_API_KEY=your_key python run_prompt_experiments_41.py
python analyze_prompt_41.py  # Summarise results as accuracy table

# Run statistical analysis on existing results
python analyze_elm_results.py

# Run all models via Groq API (direct, no Modal)
GROQ_API_KEY=your_key python run_experiments_direct.py --model gpt-oss-20b
```

### RAG Medical Summarization

```bash
cd rag_models

# Deploy evaluation services (one-time)
modal deploy evaluation/shared_evaluator_service.py

# Run a model
modal run models/main_medgemma_4b_modal.py --output-dir results/run1

# Aggregate results across 3 runs
python evaluation/aggregate_results.py results/
```

### ASR Model Evaluation

```bash
cd openemr_whisper_wer
pip install -r requirements.txt
```

#### Notion Dataset (default)
```bash
python whisper_wer.py --output results.csv
python canary_wer.py --output results.csv
python parakeet_wer.py --output results.csv
python wlv3t_on_groq.py --output results.csv
python medasr_wer.py --output results.csv
python speech_llm_wer.py --model qwen3-asr --output results.csv
python speech_llm_wer.py --model granite-speech --output results.csv
python speech_llm_wer.py --model ishiki-asr --output results.csv
```

#### Kaggle Dataset
```bash
# One-time setup: Download dataset to Modal volume
KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx modal run kaggle_dataset.py

# Run evaluations
python whisper_wer.py --kaggle --output-dir ./results
python canary_wer.py --kaggle --output-dir ./results
python parakeet_wer.py --kaggle --output-dir ./results
python wlv3t_on_groq.py --kaggle --output-dir ./results
python medasr_wer.py --kaggle --output-dir ./results
python speech_llm_wer.py --model qwen3-asr --kaggle --output-dir ./results
python speech_llm_wer.py --model granite-speech --kaggle --output-dir ./results
python speech_llm_wer.py --model ishiki-asr --kaggle --output-dir ./results
```

#### PriMock57 and Fareez OSCE Datasets
```bash
# Download PriMock57 (~4GB, requires git-lfs)
cd data && git clone https://github.com/babylonhealth/primock57.git && cd ..

# Download Fareez OSCE (~1GB from Figshare)
# https://springernature.figshare.com/collections/5545842 ; unpack under openemr_whisper_wer/data/fareez_osce/

# Run evaluations on local datasets
python whisper_wer.py --local-dataset primock57 --output results/primock57-whisper-turbo.csv
python whisper_wer.py --local-dataset primock57 --output results/primock57-whisper-v3.csv --use-large-v3
python canary_wer.py --local-dataset primock57 --output results/primock57-canary.csv
python parakeet_wer.py --local-dataset primock57 --output results/primock57-parakeet.csv
python wlv3t_on_groq.py --local-dataset primock57 --output results/primock57-groq.csv
python medasr_wer.py --local-dataset primock57 --output results/primock57-medasr.csv

# Same for Fareez (replace primock57 with fareez)
python whisper_wer.py --local-dataset fareez --output results/fareez-whisper-turbo.csv

# Qwen3-ASR, Granite Speech and Ishiki ASR (--model qwen3-asr, granite-speech or ishiki-asr).
# A run that stops can be started again with the same command and continues where it stopped.
python speech_llm_wer.py --model qwen3-asr --local-dataset primock57 --workers 2 --output results/primock57-qwen3-asr.csv
python speech_llm_wer.py --model qwen3-asr --local-dataset fareez --workers 2 --output results/fareez-qwen3-asr.csv
```

#### Generate Summary Tables
```bash
cd .. && python scripts/generate_tables.py
```

### GitHub Actions

- **Coding Bench** runs on pushes to `coding_bench/**` and by manual dispatch.
- **RAG Medical Summarization**, **WER Evaluation** and **ELM JSON Validation** run by manual dispatch only, because each run spends Modal and API credit.

**Caching**: Results are cached based on script content hash. Evaluations are skipped if the model script and utilities haven't changed.

## Project Structure

```
cdr_elmjson_validator/
├── modal_app.py            # Modal/Groq inference with ablation_mode + prompt_mode
├── run_validation.py       # CLI runner (--ablation-mode, --prompt-mode flags)
├── elm_simplifier.py       # ELM JSON to simplified format converter
├── analyze_elm_results.py  # Statistical analysis (Wilson CI, McNemar, Fisher, heatmap)
├── run_ablation.py         # Ablation experiment runner + analysis
├── run_prompt_experiments.py    # Legacy prompt runner (31-case, via Modal)
├── run_prompt_experiments_41.py # 41-case prompt runner (4 Groq models x 5 modes)
├── analyze_prompt_41.py         # Summary analysis for 41-case prompt results
├── run_experiments_direct.py # Direct Groq API experiment runner
├── run_all_expanded.py     # Run all 12 models on expanded benchmark
├── run_multi_trial.py      # Multi-trial runner for Groq models (5 trials, T=0.1)
├── run_new_models_multi_trial.py  # Multi-trial for OpenRouter + OpenAI models
├── run_ablation_multi_trial.py    # Ablation multi-trial for Groq models
├── run_new_ablation_multi_trial.py # Ablation multi-trial for OpenRouter + OpenAI
├── create_expanded_cases.py # Script that generated 6 parametric variants
├── create_semantic_cases.py # Script that generated 8 semantic error cases
├── generate_figure3_heatmap.py   # Per-case heatmap
├── test_data/              # 41 ELM JSON files, CPG markdown, ground_truth.json, BENCHMARK.md
├── notebooks/              # 4 Jupyter notebooks with pre-rendered analysis
│   ├── 01_elm_validation_results.ipynb
│   ├── 02_error_analysis.ipynb
│   ├── 03_ablation_study.ipynb
│   └── 04_prompt_engineering.ipynb
└── results/
    ├── results-*.csv       # Legacy single-run results (31 cases)
    ├── multi_trial/        # 16 models x 5 trials x 41 cases
    ├── ablation_multi_trial/ # 8 frontier models x 4 conditions x 5 trials x 41 cases
    ├── ablation/           # Single-run ablation (legacy)
    ├── prompts/            # 4 Groq frontier models x 5 strategies = 20 CSVs (41 cases)
    │   └── _31cases_backup/  # Original 31-case prompt run (historical)
    └── analysis/           # Statistical outputs, Figure 3 heatmap PNG/PDF

openemr_whisper_wer/
├── whisper_wer.py          # OpenAI Whisper evaluation
├── canary_wer.py           # NVIDIA Canary evaluation
├── parakeet_wer.py         # NVIDIA Parakeet evaluation
├── medasr_wer.py           # Google MedASR evaluation
├── wlv3t_on_groq.py        # Groq API Whisper evaluation
├── speech_llm_wer.py       # Qwen3-ASR, Granite Speech and Ishiki ASR evaluation
├── wer_utils.py            # Shared utilities (WER calculation, Notion/Kaggle fetcher)
├── primock57_utils.py      # PriMock57 dataset loader + audio channel mixer
├── fareez_utils.py         # Fareez OSCE dataset loader + MP3-to-WAV converter
├── kaggle_dataset.py       # Modal volume setup for Kaggle data
├── results/                # WER result CSVs per model per dataset
└── requirements.txt        # Python dependencies

rag_models/
├── models/                          # Per-model inference scripts
│   ├── gpt_120b_modal.py            # GPT-OSS 120B via Groq
│   ├── gpt_oss_20b_modal.py         # GPT-OSS 20B via Groq
│   ├── main_medgemma_4b_modal.py    # MedGemma 4B-IT RAG pipeline
│   └── ...                          # Qwen, Llama, MedGemma 27B, etc.
├── pipeline/                        # Orchestration + RAG infrastructure
│   ├── run_fareez_summaries.py      # Main summarization pipeline
│   ├── fareez_rag_loader.py         # Data loader for Fareez transcripts
│   └── ...                          # Vector DB, condition detection, etc.
├── evaluation/                      # Automated metrics + evaluator services
│   ├── compute_fareez_metrics.py    # BLEU, ROUGE-L, SBERT, BERTScore
│   ├── shared_evaluator_service.py  # Modal evaluation service
│   └── aggregate_results.py         # Result aggregation (mean +/- SD)
├── clinician_validation/            # Clinician rating analysis
│   ├── analyze_ratings.py           # Gwet AC2, Friedman, Wilcoxon
│   └── clinician_rating_analysis.ipynb
├── data/                            # Static data (Fareez selections, OpenEMR extracts)
├── results/                         # Output CSVs + generated summaries
├── rating_packets/                  # Clinician evaluation packets
└── vectorDB/                        # ChromaDB vector database

scripts/
└── generate_tables.py      # Generate summary tables from all results

coding_bench/
├── data/                            # tiered gold set build, manifests, checksums
├── bench/                           # loaders, metrics, runner, cache, rescore, reporting
├── approaches/                      # llm, retrieval, embed_match, entity_match, retr_llm
├── adapters/                        # Groq, Anthropic, and llama.cpp models on Modal
├── run_chain*.py                    # detached multi-step runs that outlive their client
├── results/                         # committed run records + generated LEADERBOARD.md
└── README.md                        # data tiers, running, cost, guard rails
```

## License

See [LICENSE](LICENSE) for details.
