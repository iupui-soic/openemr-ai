# openemr-ai

Artificial Intelligence tooling for OpenEMR related to clinical decision rules, ambient listening, note summarization and automated coding.

## ELM JSON Validation (Clinical Decision Support Validation)

Evaluation of LLM models for validating clinical decision support logic. Tests whether LLMs can identify if ELM (Expression Logical Model) JSON implementations correctly implement Clinical Practice Guidelines (CPG).

**What is ELM?** ELM is the compiled JSON representation of Clinical Quality Language (CQL) - the standard for expressing clinical decision rules.

**Approach:**
- **ELM Simplifier**: Deterministic code extracts key values (age thresholds, time intervals, value sets) from complex nested JSON, reducing 500+ lines to ~20 lines of structured data
- **Step-by-step prompts**: Models compare extracted values against CPG requirements with explicit decision criteria
- **Batch processing**: Each model loads once and processes all test files efficiently

**Test Dataset**: 22 hand-curated, ground-truth annotated ELM JSON artifacts derived from published clinical practice guidelines, including correct implementations and artifacts with intentionally introduced defects (incorrect age thresholds, missing exclusion criteria, wrong FHIR code bindings, logically contradictory conditions).

### Models Evaluated

| Model | Provider | Size | Infrastructure |
|-------|----------|------|----------------|
| **Gemini 3 Flash** | Google | - | Google AI Studio API |
| **Gemini 2.0 Flash** | Google | - | Google AI Studio API |
| **Gemma 3 27B** | Google | 27B | Google AI Studio API |
| **Claude Haiku 4** | Anthropic | - | Anthropic API |
| **Claude Sonnet 4** | Anthropic | - | Anthropic API |
| **Claude Opus 4** | Anthropic | - | Anthropic API |
| Llama 3.2 1B/3B | Meta | 1-3B | Modal T4 GPU |
| Qwen 2.5 1.5B/3B | Alibaba | 1.5-3B | Modal T4 GPU |
| Phi-3 Mini | Microsoft | 3.8B | Modal T4 GPU |
| Gemma 3 4B | Google | 4B | Modal T4 GPU |
| MedGemma 4B/1.5-4B | Google | 4B | Modal T4 GPU |
| GPT-OSS 20B/120B | OpenAI | 20-120B | Groq API |
| Llama 3.3 70B | Meta | 70B | Groq API |
| Qwen 3 32B | Alibaba | 32B | Groq API |

### Latest Results (22 Test Cases)

#### Frontier Models (via API)

These models require API keys (`GOOGLE_API_KEY` or `ANTHROPIC_API_KEY`) and can be run without Modal:

| Model | Accuracy | Correct/Total | Avg Time | Status |
|-------|----------|---------------|----------|--------|
| Gemini 3 Flash | - | -/22 | - | Requires `GOOGLE_API_KEY` |
| Gemini 2.0 Flash | - | -/22 | - | Requires `GOOGLE_API_KEY` |
| Gemma 3 27B | - | -/22 | - | Requires `GOOGLE_API_KEY` |
| Claude Haiku 4 | - | -/22 | - | Requires `ANTHROPIC_API_KEY` |
| Claude Sonnet 4 | - | -/22 | - | Requires `ANTHROPIC_API_KEY` |
| Claude Opus 4 | - | -/22 | - | Requires `ANTHROPIC_API_KEY` |

To run frontier models:
```bash
# Google models (requires GOOGLE_API_KEY from https://aistudio.google.com/app/apikey)
GOOGLE_API_KEY=your_key python run_validation.py --model gemini-3-flash --output results/results-gemini-3-flash.csv

# Anthropic models (requires ANTHROPIC_API_KEY from https://console.anthropic.com)
ANTHROPIC_API_KEY=your_key python run_validation.py --model claude-haiku --output results/results-claude-haiku.csv
```

#### Small/Medium Models (via Modal/Groq)

| Model | Accuracy | Correct/Total | Error Match | Avg Time | Status |
|-------|----------|---------------|-------------|----------|--------|
| GPT OSS 20B | **100.0%** | 22/22 | 73% | 3.67s | Excellent |
| GPT OSS 120B | 95.5% | 21/22 | 65% | 3.89s | Excellent |
| Llama 3.3 70B | 95.5% | 21/22 | 66% | 1.00s | Excellent |
| Qwen3 32B | 95.5% | 21/22 | 68% | 6.83s | Excellent |
| Phi-3 Mini | 68.2% | 15/22 | 55% | 21.35s | Needs Work |
| Gemma 3 4B | 68.2% | 15/22 | 68% | 34.94s | Needs Work |
| MedGemma 4B | 68.2% | 15/22 | 68% | 35.00s | Needs Work |
| MedGemma 1.5 4B | 68.2% | 15/22 | 68% | 37.14s | Needs Work |

**Key Findings:**
- **GPT OSS 20B** achieved perfect 100% accuracy (22/22) with the highest error-match rate (73%)
- **GPT OSS 120B, Llama 3.3 70B, and Qwen3 32B** all achieved 95.5% accuracy — a single misclassification separates them from perfect; interpret cautiously given the small benchmark size (~4.5 pp per case)
- **Llama 3.3 70B** improved from 40% (10-case pilot) to 95.5% on the full 22-case set, suggesting earlier poor performance reflected dataset size limitations rather than a capability gap
- Domain-specific **MedGemma** variants did not outperform general-purpose models of comparable scale (68.2%), consistent with prior observations that clinical fine-tuning alone is insufficient for structured logic validation tasks
- **GPT OSS 120B** (via Groq, 3.89s/artifact) offers a practical balance of accuracy and latency for deployment
- Frontier models (Gemini, Claude) can be evaluated with the appropriate API keys

View detailed results in the [workflow runs](../../actions/workflows/elm-validation.yml).

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

Results are aggregated across 3 independent runs using `rag_models/RAG_To_See_MedGemma_Performance/aggregate_results.py`.

**Evaluation Metrics:**
- **BLEU & ROUGE-L**: Lexical overlap and structural recall
- **SBERT Similarity**: Global semantic coherence (0.80-0.90 indicates clinical equivalence)
- **BERTScore F1**: Local semantic accuracy for clinical details
- **SciSpaCy/MedCAT Entity Recall**: Medical entity extraction coverage

> View detailed results in the [GitHub Actions workflow runs](../../actions/workflows/rag-summarization.yml)

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
pip install -r requirements.txt

# List available models
python run_validation.py --list-models

# Run a single model (Modal-based)
python run_validation.py --model qwen3-32b --output results/results-qwen3-32b.csv

# Run frontier models (no Modal required, uses API keys)
GOOGLE_API_KEY=your_key python run_validation.py --model gemini-3-flash --output results/results-gemini-3-flash.csv
ANTHROPIC_API_KEY=your_key python run_validation.py --model claude-haiku --output results/results-claude-haiku.csv

# Run all models
python run_validation.py --all-models --output-dir results/
```

### RAG Medical Summarization

```bash
cd rag_models/RAG_To_See_MedGemma_Performance

# Deploy evaluation services (one-time)
modal deploy shared_evaluator_service.py

# Run a model
modal run main_medgemma_4b_modal.py --output-dir results/run1

# Aggregate results across 3 runs
python aggregate_results.py results/
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
├── modal_app.py            # Modal functions for LLM validation
├── run_validation.py       # CLI runner for validation
├── elm_simplifier.py       # ELM JSON to simplified format converter
├── test_data/              # 22 ELM JSON files, CPG markdown, ground_truth.json
└── results/                # Validation result CSVs per model

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

rag_models/RAG_To_See_MedGemma_Performance/
├── main_medgemma_4b_modal.py    # MedGemma 4B-IT RAG pipeline
├── medgemma_27b_4bit_modal.py   # MedGemma 27B quantized
├── llama_3_1_8b_modal.py        # Llama 3.1 8B Instruct
├── qwen_32b_modal.py            # Qwen 3 32B
├── gpt_oss_20b_modal.py         # GPT-OSS 20B via Groq
├── gpt_120b_modal.py            # GPT-OSS 120B via Groq
├── groq_llama_4_scout_modal.py  # Llama 4 Scout via Groq
├── shared_evaluator_service.py  # Modal evaluation service
├── aggregate_results.py         # Result aggregation (mean +/- SD)
└── vectorDB/                    # ChromaDB vector database

scripts/
└── generate_tables.py      # Generate publication tables from all results
```

## License

See [LICENSE](LICENSE) for details.
