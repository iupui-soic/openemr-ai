# openemr-ai

Artificial Intelligence tooling for OpenEMR related to clinical decision rules, ambient listening, note summarization and automated coding.

## ELM JSON Validation (Clinical Decision Support Validation)

This repository includes comprehensive evaluation of LLM models for validating clinical decision support logic. We evaluate multiple language models on their ability to identify whether ELM (Expression Logical Model) JSON implementations correctly implement Clinical Practice Guidelines (CPG).

**What is ELM?** Expression Logical Model (ELM) is the compiled JSON representation of Clinical Quality Language (CQL) - the standard for expressing clinical decision rules in systems like OpenEMR. This evaluation tests if LLMs can act as automated reviewers of clinical logic by analyzing ELM files against medical guidelines.

**Test Dataset**: Ground truth annotated test cases with ELM JSON files and corresponding CPG markdown files. Each test case includes expected validation results (valid/invalid) and specific errors the LLM should identify.

### Models Evaluated

| Model | Provider | Size | Notes |
|-------|----------|------|-------|
| Llama 3.2 1B Instruct | Meta | 1B | Compact general-purpose LLM |
| Llama 3.2 3B Instruct | Meta | 3B | Larger general-purpose LLM |
| Llama 3.1 8B Instruct | Meta | 8B | High-capacity general-purpose |
| Qwen 2.5 1.5B Instruct | Alibaba | 1.5B | Efficient multilingual LLM |
| Qwen 2.5 3B Instruct | Alibaba | 3B | Balanced performance LLM |
| Phi-3 Mini | Microsoft | 3.8B | Compact reasoning model |
| Gemma 3 270M | Google | 270M | Ultra-compact model |
| Gemma 3 4B | Google | 4B | Efficient open model |
| MedGemma 4B | Google | 4B | Healthcare-specialized model |

### Latest Results

| Model | Accuracy | Correct/Total | Error Match | Avg Time | Status |
|-------|----------|---------------|-------------|----------|--------|
| Llama 3.2 1B | 75.0% | 3/4 | 75% | 4.73s | ✓ |
| Qwen 2.5 3B | 75.0% | 3/4 | 75% | 16.63s | ✓ |
| Gemma 3 270M | 75.0% | 3/4 | 75% | 22.76s | ✓ |
| MedGemma 4B | 75.0% | 3/4 | 75% | 35.44s | ✓ |
| Gemma 3 4B | 75.0% | 3/4 | 75% | 40.74s | ✓ |
| Llama 3.1 8B | 75.0% | 3/4 | 50% | 321.68s | ✓ |
| Llama 3.2 3B | 50.0% | 2/4 | 50% | 8.02s | ✓ |
| Phi-3 Mini | 50.0% | 2/4 | 25% | 27.05s | ✓ |
| Qwen 2.5 1.5B | 25.0% | 1/4 | 75% | 13.97s | ✓ |

> View detailed results in the [GitHub Actions workflow runs](../../actions/workflows/elm-validation.yml)

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

### Latest Results

#### Notion Dataset (Custom Medical Recordings)

| Model | Avg WER | Status |
|-------|---------|--------|
| parakeet-tdt-1.1b | 8.30% | Run evaluation |
| canary-1b-v2 | 8.58% | Run evaluation |
| groq-whisper-large-v3-turbo | 9.60% | Run evaluation |
| whisper-large-v3 | 13.26% | Run evaluation |
| whisper-large-v3-turbo | 14.02% | Run evaluation |
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