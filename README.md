# openemr-ai

Artificial Intelligence tooling for OpenEMR related to clinical decision rules, ambient listening, note summarization and automated coding.

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
| whisper-turbo | - | Run evaluation |
| whisper-v3 | - | Run evaluation |
| canary | - | Run evaluation |
| parakeet | - | Run evaluation |
| granite | - | Run evaluation |
| groq | - | Run evaluation |

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

### Installation

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