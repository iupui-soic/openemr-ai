# openemr-ai

Artificial Intelligence tooling for OpenEMR related to clinical decision rules, ambient listening, note summarization and automated coding.

## ELM JSON Validation (Clinical Decision Support Validation)

Evaluation of LLM models for validating clinical decision support logic. Tests whether LLMs can identify if ELM (Expression Logical Model) JSON implementations correctly implement Clinical Practice Guidelines (CPG).

**What is ELM?** ELM is the compiled JSON representation of Clinical Quality Language (CQL) - the standard for expressing clinical decision rules.

**Approach:**
- **ELM Simplifier**: Deterministic code extracts key values (age thresholds, time intervals, value sets) from complex nested JSON, reducing 500+ lines to ~20 lines of structured data
- **Step-by-step prompts**: Models compare extracted values against CPG requirements with explicit decision criteria
- **Batch processing**: Each model loads once and processes all test files efficiently

**Test Dataset**: 10 ground-truth annotated test cases (7 valid, 3 invalid) with ELM JSON and CPG markdown files.

### Models Evaluated

| Model | Provider | Size | Infrastructure |
|-------|----------|------|----------------|
| Llama 3.2 1B/3B | Meta | 1-3B | Modal T4 GPU |
| Llama 3.1 8B | Meta | 8B | Modal T4 GPU |
| Llama 3.3 70B | Meta | 70B | Groq API |
| Qwen 2.5 1.5B/3B | Alibaba | 1.5-3B | Modal T4 GPU |
| Phi-3 Mini | Microsoft | 3.8B | Modal T4 GPU |
| Gemma 3 270M/4B | Google | 270M-4B | Modal T4 GPU |
| MedGemma 4B/1.5-4B | Google | 4B | Modal T4 GPU |
| GPT-OSS 20B/120B | OpenAI | 20-120B | Groq API |

### Latest Results

Results updated via GitHub Actions. View detailed results in the [workflow runs](../../actions/workflows/elm-validation.yml).

## RAG Medical Summarization (SOAP Note Generation)

This repository includes evaluation of LLM models for generating medical discharge summaries in SOAP format using Retrieval-Augmented Generation (RAG). We test multiple models on their ability to create structured clinical summaries from doctor-patient transcripts and electronic health records.

**What is RAG?** Retrieval-Augmented Generation combines language models with a vector database to retrieve relevant context (SOAP note schemas from MIMIC Clinical Notes) before generating summaries. This ensures outputs follow proper clinical documentation structure.

**Vector Database**: ChromaDB hosted on Modal Volume containing SOAP note structures extracted from MIMIC Clinical Notes. Uses **all-MiniLM-L6-v2** for semantic retrieval of top 2 relevant schemas based on detected disease.

**Test Dataset**: 6 patient test cases with doctor-patient transcripts, OpenEMR extracts, and reference summaries for evaluation.

### Models Evaluated

| Model | Provider | Size | Infrastructure | Notes |
|-------|----------|------|----------------|-------|
| Llama 3.1 8B Instruct | Meta | 8B | Modal (A10G GPU) | vLLM for fast inference |
| MedGemma 4B-IT | Google | 4B | Modal (A10G GPU) | Healthcare-specialized model |
| Groq GPT-OSS-20B | Groq | 20B | API-based | Fast inference, no GPU needed |

### Latest Results

| Model | BLEU | ROUGE-L | SBERT | BERTScore F1 | Total Time | Status |
|-------|------|---------|-------|--------------|------------|--------|
| [Fill from workflow] | | | | | | |

**Evaluation Metrics:**
- **BLEU & ROUGE-L**: Lexical overlap and structural recall
- **SBERT Similarity**: Global semantic coherence (0.80-0.90 indicates clinical equivalence)
- **BERTScore F1**: Local semantic accuracy for clinical details

> View detailed results in the [GitHub Actions workflow runs](../../actions/workflows/rag-summarization.yml)

## ASR Model Evaluation (Word Error Rate)

This repository includes comprehensive evaluation of Automatic Speech Recognition (ASR) models for medical transcription. We evaluate multiple state-of-the-art models on two datasets:

1. **Notion Dataset**: Custom medical recordings from a Notion database
2. **Kaggle Dataset**: [Medical Speech, Transcription, and Intent](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent)

### Models Evaluated

| Model | Provider | Notes |
|-------|----------|-------|
| Whisper Large v3 Turbo | OpenAI | Fast, accurate general-purpose ASR |
| Whisper Large v3 | OpenAI | Higher accuracy, slower than turbo |
| Canary-1B-v2 | NVIDIA | Multi-task ASR from NeMo toolkit |
| Parakeet-TDT-1.1B | NVIDIA | Fast Token-and-Duration Transducer |
| Granite Speech 3.3 8B | IBM | Multimodal speech-to-text model |
| Phi-4 Multimodal | Microsoft | Multimodal instruct model |
| Groq Whisper | Groq | API-based Whisper Large v3 Turbo |
| MedASR | Google | A model based on the Conformer architecture pre-trained for medical dictation and transcription |

### Latest Results

#### Notion Dataset (Custom Medical Recordings)

| Model | Avg WER | Status |
|-------|---------|--------|
| parakeet-tdt-1.1b | 8.30% | Run evaluation |
| canary-1b-v2 | 8.58% | Run evaluation |
| groq-whisper-large-v3-turbo | 9.60% | Run evaluation |
| whisper-large-v3 | 13.26% | Run evaluation |
| whisper-large-v3-turbo | 14.02% | Run evaluation |
| medasr | 44.01% | Run evaluation |
| phi4-multimodal | 64.60% | Run evaluation |
| granite-speech-3.3-8b | 74.10% | Run evaluation |

#### Kaggle Medical Speech Dataset

| Model | Avg WER | Status |
|-------|---------|--------|
| groq | 14.91% | Run evaluation |
| canary | 15.05% | Run evaluation |
| parakeet | 16.28% | Run evaluation |
| granite | 17.70% | Run evaluation |
| whisper-v3 | 19.35% | Run evaluation |
| whisper-turbo | 23.92% | Run evaluation |
| medasr | 39.25% | Run evaluation |

> View detailed results in the [GitHub Actions workflow runs](../../actions/workflows/wer-evaluation.yml)

## Running Experiments

### Prerequisites

- Python 3.11+
- [Modal](https://modal.com/) account (for GPU compute)
- Environment variables:
  - `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` - Modal authentication
  - `NOTION_API_KEY` - For Notion dataset access
  - `GROQ_API_KEY` - For Groq API evaluations
  - `KAGGLE_USERNAME` and `KAGGLE_KEY` - For Kaggle dataset setup
  - `HF_TOKEN` - For ELM JSON validation experiments

### ELM JSON Validation

```bash
cd cdr_elmjson_validator
pip install -r requirements.txt

# List available models
python run_validation.py --list-models

# Run validation with specific model
python run_validation.py --model llama-3.2-1b --output results-llama-3.2-1b.csv

# Run validation with all models
python run_validation.py --all-models --output-dir results/
```

### RAG Medical Summarization

```bash
cd pranathi_RAG
pip install -r requirements.txt

# Run Llama 3.1 8B summarization
modal run main.py \
  --transcript-path="../rag_models/RAG_To_See_MedGemma_Performance/data/transcription_<patient>.txt" \
  --openemr-path="../rag_models/RAG_To_See_MedGemma_Performance/data/openemr_<patient>.txt" \
  --reference-path="../rag_models/RAG_To_See_MedGemma_Performance/data/reference_<patient>.txt" \
  --patient-name="<PatientName>" \
  --output-dir="results"

# Run Groq GPT-OSS-20B summarization
modal run main_groq.py \
  --transcript-path="../rag_models/RAG_To_See_MedGemma_Performance/data/transcription_<patient>.txt" \
  --patient-name="<PatientName>"

# Run MedGemma 4B-IT summarization
cd ../rag_models/RAG_To_See_MedGemma_Performance
modal run main_medgemma_4b_modal.py \
  --transcript-path="data/transcription_<patient>.txt" \
  --patient-name="<PatientName>"
```

### ASR Model Evaluation

```bash
cd openemr_whisper_wer
pip install -r requirements.txt
```

### Running Individual Model Evaluations (Notion Dataset)

```bash
# Whisper Large v3 Turbo
python whisper_wer.py --output results-whisper-turbo.csv

# Whisper Large v3
python whisper_wer.py --output results-whisper-v3.csv --use-large-v3

# NVIDIA Canary
python canary_wer.py --output results-canary.csv

# NVIDIA Parakeet
python parakeet_wer.py --output results-parakeet.csv

# IBM Granite
python granite_wer.py --output results-granite.csv

# Microsoft Phi-4
python phi4_wer.py --output results-phi4.csv

# Groq Whisper (API-based, no Modal required)
python wlv3t_on_groq.py --output results-groq.csv
```

### Running Kaggle Dataset Evaluation (Per-Model)

First, ensure the Kaggle dataset is downloaded to the Modal volume:

```bash
# One-time setup: Download dataset to Modal volume
KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx modal run kaggle_dataset.py
```

Then run evaluations per model using the `--kaggle` flag. Each model runs independently, so failures don't affect other models:

```bash
# Whisper Large v3 Turbo
python whisper_wer.py --kaggle --output-dir ./results

# Whisper Large v3
python whisper_wer.py --kaggle --use-large-v3 --output-dir ./results

# NVIDIA Canary
python canary_wer.py --kaggle --output-dir ./results

# NVIDIA Parakeet
python parakeet_wer.py --kaggle --output-dir ./results

# IBM Granite
python granite_wer.py --kaggle --output-dir ./results

# Groq Whisper (uses Modal to access volume, calls Groq API)
python wlv3t_on_groq.py --kaggle --output-dir ./results

# Use train split instead of validate
python whisper_wer.py --kaggle --split train --output-dir ./results
```

### GitHub Actions

The WER evaluation workflow runs automatically on:
- Push to `openemr_whisper_wer/**` files
- Manual trigger via workflow dispatch

**Caching**: Results are cached based on script content hash. Both Notion and Kaggle evaluations are cached independently per model. Evaluations are skipped if the model script and utilities haven't changed.

To manually run:
1. Go to **Actions** > **WER Evaluation**
2. Click **Run workflow**
3. Options:
   - **Force run all models**: Bypass cache and re-run all evaluations
   - **Run Kaggle dataset evaluation**: Include Kaggle dataset in evaluation (only runs if cache miss or forced)

## Project Structure

```
cdr_elmjson_validator/
├── modal_app.py            # Modal functions for LLM validation
├── run_validation.py       # CLI runner for validation
├── elm_simplifier.py       # ELM JSON to simplified format converter
└── test_data/              # ELM JSON files, CPG markdown, ground_truth.json

openemr_whisper_wer/
├── whisper_wer.py          # OpenAI Whisper evaluation (--kaggle for Kaggle dataset)
├── canary_wer.py           # NVIDIA Canary evaluation (--kaggle for Kaggle dataset)
├── parakeet_wer.py         # NVIDIA Parakeet evaluation (--kaggle for Kaggle dataset)
├── granite_wer.py          # IBM Granite evaluation (--kaggle for Kaggle dataset)
├── phi4_wer.py             # Microsoft Phi-4 evaluation
├── wlv3t_on_groq.py        # Groq API evaluation (--kaggle for Kaggle dataset)
├── kaggle_dataset.py       # Modal volume setup for Kaggle data
├── wer_utils.py            # Shared utilities (WER calculation, Notion/Kaggle fetcher)
└── requirements.txt        # Python dependencies
```

## License

See [LICENSE](LICENSE) for details.
