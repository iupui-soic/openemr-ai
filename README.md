# openemr-ai

Artificial Intelligence tooling for OpenEMR related to clinical decision rules, ambient listening, note summarization and automated coding.

## ELM JSON Validation (Clinical Decision Support Validation)

Evaluation of LLM models for validating clinical decision support logic. Tests whether LLMs can identify if ELM (Expression Logical Model) JSON implementations correctly implement Clinical Practice Guidelines (CPG).

**What is ELM?** ELM is the compiled JSON representation of Clinical Quality Language (CQL) - the standard for expressing clinical decision rules.

**Approach:**
- **ELM Simplifier**: Deterministic code extracts key values (age thresholds, time intervals, value sets) from complex nested JSON, reducing 500+ lines to ~20 lines of structured data
- **Step-by-step prompts**: Models compare extracted values against CPG requirements with explicit decision criteria
- **Batch processing**: Each model loads once and processes all test files efficiently

**Test Dataset**: 31 hand-curated, ground-truth annotated ELM JSON artifacts derived from published clinical practice guidelines (15 valid, 16 invalid). Invalid cases include 13 parametric errors (wrong age thresholds, lookback intervals, clinical values) and 3 semantic logic errors (Boolean operator swap, missing exclusion check, missing sub-population bound). See [`test_data/BENCHMARK.md`](cdr_elmjson_validator/test_data/BENCHMARK.md) for full methodology.

### Models Evaluated (16 open-weight models + 1 proprietary reference)

| Model | Provider | Params / Active | Infrastructure |
|-------|----------|-----------------|----------------|
| **Gemma 4 31B** | Google | 31B dense | Local RTX 6000 (4-bit) |
| **Gemma 4 26B A4B** | Google | 26B / 4B active MoE | Local RTX 6000 (4-bit) |
| **Qwen3.5 35B A3B** | Alibaba | 35B / 3B active MoE | OpenRouter API |
| **Qwen 3 32B** | Alibaba | 32B dense | Groq API |
| **GPT-OSS 20B/120B** | OpenAI | 20-120B / 3.6-5.1B active MoE | Groq API |
| **Llama 3.3 70B** | Meta | 70B dense | Groq API |
| **GPT-5.4-mini** (proprietary reference) | OpenAI | — | OpenAI API |
| MedGemma 4B / 1.5 4B | Google | 4B | Local RTX 6000 (bfloat16) |
| Gemma 3 4B | Google | 4B | OpenRouter API |
| Phi-3 Mini | Microsoft | 3.8B | Local RTX 6000 |
| Llama 3.2 1B/3B | Meta | 1-3B | Local RTX 6000 |
| Qwen 2.5 1.5B/3B | Alibaba | 1.5-3B | Local RTX 6000 |

### Latest Results (41 Test Cases, 16 Valid / 25 Invalid: 13 parametric, 12 semantic)

All results are means over 5 independent trials at temperature 0.1. Base rate: 39.0%.

| Model | Params | Accuracy | ±SD | Sens. | Spec. | F1 | Param | Semantic |
|-------|--------|----------|-----|-------|-------|-----|-------|----------|
| **Gemma 4 26B A4B** | 26B (4B active MoE) | **92.7%** | 0.0 | 0.88 | 0.96 | **0.90** | 100% | 92% |
| **Gemma 4 31B** | 31B | **92.7%** | 0.0 | 0.81 | **1.00** | **0.90** | 100% | 100% |
| **Qwen3 32B** | 32B | 90.7% | 2.0 | 0.88 | 0.93 | 0.88 | 100% | 85% |
| **GPT-OSS 120B** | 120B (5.1B active MoE) | 85.9% | 2.0 | 0.64 | 1.00 | 0.78 | 100% | 100% |
| **Qwen3.5 35B A3B** | 35B (3B active MoE) | 85.4% | 3.4 | 0.69 | 0.96 | 0.78 | 100% | 92% |
| **Llama 3.3 70B** | 70B | 83.4% | 1.1 | 0.88 | 0.81 | 0.80 | 100% | 60% |
| GPT-5.4-mini (proprietary) | — | 81.0% | 3.2 | 0.51 | 1.00 | 0.67 | 100% | 100% |
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

| Condition | Gemma 4 26B A4B | Gemma 4 31B | Qwen3-32B | Qwen3.5-35B-A3B | GPT-OSS-120B | Llama 70B | GPT-OSS-20B | GPT-5.4-mini |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | *4B active* | *31B dense* | *32B dense* | *3B active* | *5.1B active* | *70B dense* | *3.6B active* | *proprietary* |
| Full (simplified+CPG) | **90.2±0.0** | **92.7±0.0** | **88.3±2.7** | **83.9±3.3** | **83.9±2.8** | **84.9±1.1** | **80.0±2.0** | **82.9±2.4** |
| No CPG | 56.1 (Δ-34) | 70.7 (Δ-22) | 60.0 (Δ-28) | 51.7 (Δ-32) | 62.4 (Δ-22) | 51.7 (Δ-33) | 62.4 (Δ-18) | 69.3 (Δ-14) |
| No Simplification | 61.0 (Δ-29) | 61.0 (Δ-32) | 83.9 (Δ-4) | 77.1 (Δ-7) | 80.5 (Δ-3) | 82.9 (Δ-2) | 70.7 (Δ-9) | 75.1 (Δ-8) |
| Neither | 61.5 (Δ-29) | 65.9 (Δ-27) | 62.4 (Δ-26) | 40.0 (Δ-44) | 68.8 (Δ-15) | 45.9 (Δ-39) | 58.0 (Δ-22) | 71.2 (Δ-12) |

**Key Findings:**

- **Gemma 4 26B A4B and 31B tied for top accuracy (92.7%)** — both fully deterministic (±0.0% SD). The MoE variant with only 4B active parameters matches the 31B dense model.
- **ELM Simplifier as compute equalizer**: Models with small active parameter counts benefit most from simplification. Gemma 4 26B A4B drops from 90.2% → 61.0% (Δ=-29pp) without it, while Llama 70B drops only 2pp. The gradient tracks active parameters precisely.
- **CPG reference is universally essential**: removing it reduces accuracy by 14–34 pp across all frontier models.
- **Every frontier model achieves 100% on parametric errors** with the full pipeline — the simplifier's Phase 1 (numeric extraction) ceiling-caps numeric comparison.
- **Semantic error detection varies**: Gemma 4 31B, GPT-OSS-120B, Qwen3.5 and GPT-5.4-mini achieve 100% on semantic errors; Llama 70B lags at 60%. The simplifier's Phase 2 (full logic tree) enables semantic detection.
- **Sensitivity–specificity tradeoff**: High-specificity models (GPT-OSS-120B, GPT-5.4-mini) miss 36–49% of valid cases, while balanced models (Gemma 4 26B A4B, Qwen3-32B, Llama 70B) achieve sensitivity ≥0.88 and specificity ≥0.81.
- **Medical pretraining (MedGemma)** shows no consistent advantage: MedGemma 4B matches Gemma 3 4B at base rate (39.0%), while MedGemma 1.5 4B reaches 70.2% — the only medical model showing benefit, though still below frontier tier.
- All 4B-and-below models fall below the 48.4% naive baseline on the expanded 41-case benchmark, confirming a capability threshold around 20B total / 4B active parameters.

Detailed analysis available in the [Jupyter notebooks](cdr_elmjson_validator/notebooks/).

## RAG Medical Summarization (SOAP Note Generation)

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

### Fareez OSCE Experiment (40 Conversations x 4 Models = 160 Summaries)

**Dataset**: 40 conversations from [Fareez et al.](https://springernature.figshare.com/collections/5545842) balanced across 5 medical specialties (20 RES, 10 MSK, 5 GAS, 4 CAR, 1 DER). Each conversation paired with condition-matched EHR data from OpenEMR (11,030 synthetic patients).

**Design**: All models use the same pre-detected conditions (via Llama 3.3 70B) for ChromaDB schema retrieval, isolating summary generation quality as the only variable.

| Model | Size | Infrastructure | Summaries | Avg Time | Avg Chars |
|-------|------|----------------|-----------|----------|-----------|
| GPT-OSS 120B | 120B | Groq (reasoning API) | 40/40 | 5.3s | 7,072 |
| Qwen3 32B | 32B | Groq API | 40/40 | 4.2s | 5,568 |
| GPT-OSS 20B | 20B | Groq (reasoning API) | 40/40 | 2.8s | 5,275 |
| MedGemma 4B-IT | 4B | Modal (A10G GPU) | 39/40 | 42.3s | 2,741 |

**Clinician Evaluation**: 3 physician fellows (GI, IR, ER) independently rated all 160 summaries on 6 dimensions using 5-point Likert scales adapted from PDQI-9 and QNOTE. Inter-rater reliability via Gwet's AC2 (ordinal weights), model comparison via Friedman test with Bonferroni-corrected pairwise Wilcoxon signed-rank tests.

#### Clinician Rating Results (Mean(SD), 5-point Likert)

| Model | Accuracy | Completeness | Organization | Conciseness | Clinical Utility | Overall Quality | Composite |
|-------|----------|--------------|--------------|-------------|------------------|-----------------|-----------|
| GPT-OSS 120B | 2.70(0.90) | **4.14(0.57)** | **5.00(0.00)** | 3.64(0.99) | **3.54(0.88)** | **3.44(0.86)** | **3.74** |
| GPT-OSS 20B | 2.63(0.92) | 3.93(0.64) | **5.00(0.00)** | **3.92(0.78)** | 3.47(0.85) | 3.35(0.86) | 3.72 |
| Qwen3 32B | 2.50(0.90) | 3.83(0.62) | 4.94(0.27) | 3.70(0.71) | 3.42(0.91) | 3.29(0.90) | 3.61 |
| MedGemma 4B | 2.38(0.81) | 2.97(0.74) | 3.91(1.11) | 3.35(1.23) | 2.62(0.84) | 2.51(0.92) | 2.96 |

**Inter-Rater Reliability (Gwet's AC2, ordinal weights):**

| Dimension | AC2 | 95% CI | Interpretation |
|-----------|-----|--------|----------------|
| Organization | 0.971 | [0.960, 0.982] | Good |
| Completeness | 0.874 | [0.851, 0.898] | Good |
| Overall Quality | 0.770 | [0.729, 0.810] | Acceptable |
| Clinical Utility | 0.758 | [0.720, 0.795] | Acceptable |
| Accuracy | 0.708 | [0.663, 0.752] | Acceptable |
| Conciseness | 0.654 | [0.612, 0.695] | Low |

**Friedman Test** (significant dimensions): Organization (p<0.001, W=0.95), Completeness (p<0.001, W=0.74), Clinical Utility (p<0.001, W=0.43), Overall Quality (p<0.001, W=0.40)

#### Automated Metrics (40 Conversations)

| Model | BLEU | ROUGE-L | SBERT Coherence | BERTScore F1 | Avg Chars |
|-------|------|---------|-----------------|--------------|-----------|
| GPT-OSS 120B | 0.011 | 0.115 | 0.511 | 0.800 | 7,072 |
| GPT-OSS 20B | 0.009 | 0.118 | 0.529 | 0.800 | 5,275 |
| Qwen3 32B | 0.015 | 0.129 | 0.491 | 0.803 | 5,568 |
| MedGemma 4B | 0.011 | 0.145 | 0.506 | 0.803 | 3,729 |

**Key Findings:**
- **GPT-OSS 120B** achieved the highest composite clinician rating (3.74/5), with perfect organization scores and strongest completeness
- **Organization** showed near-perfect agreement across raters (AC2=0.971) and strongest model differentiation (W=0.95), with all three large models achieving near-perfect scores vs. MedGemma's 3.91
- **MedGemma 4B** scored significantly lower on completeness (2.97 vs 3.83-4.14) and clinical utility (2.62 vs 3.42-3.54), despite comparable BERTScore F1 (0.803) — automated metrics do not capture clinical completeness
- Automated text overlap metrics (BLEU, ROUGE-L) showed minimal variation across models, confirming clinician evaluation captures quality differences that automated metrics miss

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
```

## ASR Model Evaluation (Word Error Rate)

Comprehensive evaluation of Automatic Speech Recognition (ASR) models for medical transcription across four datasets:

1. **Notion Dataset**: 6 custom simulated doctor-patient conversations recorded at IU Indianapolis
2. **Kaggle Dataset**: [Medical Speech, Transcription, and Intent](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent) (~380 utterances)
3. **PriMock57**: 57 primary care mock consultations from [PriMock57](https://github.com/babylonhealth/primock57) (~9 hours, real clinicians, utterance-level TextGrid transcripts)
4. **Fareez OSCE**: 272 simulated patient-physician OSCE interviews from [Fareez et al.](https://springernature.figshare.com/collections/5545842) (~55 hours, 5 medical specialties)

### Models Evaluated

| Model | Provider | Notes |
|-------|----------|-------|
| Whisper Large v3 Turbo | OpenAI | Fast, accurate general-purpose ASR |
| Whisper Large v3 | OpenAI | Higher accuracy, slower than turbo |
| Canary-1B-v2 | NVIDIA | Multi-task ASR from NeMo toolkit |
| Parakeet-TDT-1.1B | NVIDIA | Fast Token-and-Duration Transducer |
| Groq Whisper | Groq | API-based Whisper Large v3 Turbo |
| MedASR | Google | Conformer architecture pre-trained for medical dictation |

### Latest Results

#### Notion Dataset (6 Custom Medical Recordings)

| Model | Avg WER | Status |
|-------|---------|--------|
| parakeet-tdt-1.1b | 8.30% | Run evaluation |
| canary-1b-v2 | 8.58% | Run evaluation |
| groq-whisper-large-v3-turbo | 9.60% | Run evaluation |
| whisper-large-v3 | 13.26% | Run evaluation |
| whisper-large-v3-turbo | 14.02% | Run evaluation |
| medasr | 44.01% | Run evaluation |

#### Kaggle Medical Speech Dataset (~380 utterances)

| Model | Avg WER | Status |
|-------|---------|--------|
| groq | 14.91% | Run evaluation |
| canary | 15.05% | Run evaluation |
| parakeet | 16.28% | Run evaluation |
| whisper-v3 | 19.35% | Run evaluation |
| whisper-turbo | 23.92% | Run evaluation |
| medasr | 39.25% | Run evaluation |

#### PriMock57 (57 Primary Care Consultations, ~9 hours)

| Model | Success Rate | Avg WER |
|-------|-------------|---------|
| Parakeet TDT 1.1B | 57/57 | **17.06%** |
| Groq Whisper | 57/57 | **18.87%** |
| Whisper Large v3 Turbo | 57/57 | 21.00% |
| Canary 1B v2 | 57/57 | 21.54% |
| Whisper Large v3 | 57/57 | 22.28% |
| MedASR | 57/57 | 64.59% |

#### Fareez OSCE (272 Multi-Specialty Interviews, ~55 hours)

| Model | Success Rate | Avg WER | Notes |
|-------|-------------|---------|-------|
| Groq Whisper | 272/272 | **19.54%** | FLAC compression for files >24MB |
| Parakeet TDT 1.1B | 251/272 | 20.17% | OOM on longest conversations (A100) |
| Canary 1B v2 | 272/272 | 21.29% | |
| Whisper Large v3 | 84/272 | 23.33% | Modal billing limit after 84 files |
| Whisper Large v3 Turbo | 272/272 | 24.27% | |
| MedASR | 272/272 | 47.53% | |

**Key Findings:**
- **Parakeet** and **Groq Whisper** are the top performers across both new datasets (17-20% WER)
- **Canary** delivers consistent ~21% WER across all datasets with zero failures
- **Whisper-turbo** and **Whisper-v3** perform similarly (21-24% WER) on medical conversations
- **MedASR** has significantly higher WER on conversational speech (47-65%) vs shorter medical dictation
- All models show higher WER on full conversations vs short utterances, as expected

> View detailed results in the [GitHub Actions workflow runs](../../actions/workflows/wer-evaluation.yml)

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

# Run prompt engineering experiments
python run_prompt_experiments.py --model gpt-oss-20b

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
```

#### PriMock57 and Fareez OSCE Datasets
```bash
# Download PriMock57 (~4GB, requires git-lfs)
cd data && git clone https://github.com/babylonhealth/primock57.git && cd ..

# Download Fareez OSCE (~1GB from Figshare)
# See manuscript/EXPERIMENT_IMPROVEMENT_PLAN.md for download instructions

# Run evaluations on local datasets
python whisper_wer.py --local-dataset primock57 --output results/primock57-whisper-turbo.csv
python whisper_wer.py --local-dataset primock57 --output results/primock57-whisper-v3.csv --use-large-v3
python canary_wer.py --local-dataset primock57 --output results/primock57-canary.csv
python parakeet_wer.py --local-dataset primock57 --output results/primock57-parakeet.csv
python wlv3t_on_groq.py --local-dataset primock57 --output results/primock57-groq.csv
python medasr_wer.py --local-dataset primock57 --output results/primock57-medasr.csv

# Same for Fareez (replace primock57 with fareez)
python whisper_wer.py --local-dataset fareez --output results/fareez-whisper-turbo.csv
```

#### Generate Publication Tables
```bash
cd .. && python scripts/generate_tables.py
```

### GitHub Actions

The evaluation workflows run automatically on:
- Push to `openemr_whisper_wer/**`, `cdr_elmjson_validator/**` files
- Manual trigger via workflow dispatch

**Caching**: Results are cached based on script content hash. Evaluations are skipped if the model script and utilities haven't changed.

## Project Structure

```
cdr_elmjson_validator/
├── modal_app.py            # Modal/Groq inference with ablation_mode + prompt_mode
├── run_validation.py       # CLI runner (--ablation-mode, --prompt-mode flags)
├── elm_simplifier.py       # ELM JSON to simplified format converter
├── analyze_elm_results.py  # Statistical analysis (Wilson CI, McNemar, Fisher, heatmap)
├── run_ablation.py         # Ablation experiment runner + analysis
├── run_prompt_experiments.py # Prompt engineering experiment runner + analysis
├── run_experiments_direct.py # Direct Groq API experiment runner
├── run_all_expanded.py     # Run all 12 models on expanded benchmark
├── create_expanded_cases.py # Script that generated the 9 new test cases
├── test_data/              # 31 ELM JSON files, CPG markdown, ground_truth.json, BENCHMARK.md
├── notebooks/              # 4 Jupyter notebooks with pre-rendered analysis
│   ├── 01_elm_validation_results.ipynb
│   ├── 02_error_analysis.ipynb
│   ├── 03_ablation_study.ipynb
│   └── 04_prompt_engineering.ipynb
└── results/
    ├── results-*.csv       # 12 model result CSVs (31 cases each)
    ├── ablation/           # 4 models x 4 conditions = 16 CSVs
    ├── prompts/            # 3 models x 5 strategies = 15 CSVs
    └── analysis/           # Statistical outputs, heatmap PNG, CSVs

openemr_whisper_wer/
├── whisper_wer.py          # OpenAI Whisper evaluation
├── canary_wer.py           # NVIDIA Canary evaluation
├── parakeet_wer.py         # NVIDIA Parakeet evaluation
├── medasr_wer.py           # Google MedASR evaluation
├── wlv3t_on_groq.py        # Groq API Whisper evaluation
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
└── generate_tables.py      # Generate publication tables from all results
```

## License

See [LICENSE](LICENSE) for details.
