"""
Google MedASR WER Calculator

Fetches audio from Notion database, transcribes with Google MedASR on Modal (ephemeral),
calculates Word Error Rate, and generates detailed error analysis.

MedASR is Google's medical-domain ASR model based on Conformer architecture,
specifically trained for medical transcription tasks.

Usage:
    python medasr_wer.py --output results.csv
    python medasr_wer.py --kaggle  # Evaluate on Kaggle medical speech dataset
    python medasr_wer.py --kaggle --split validate --output-dir ./results

Requirements:
    pip install modal jiwer pandas requests notion-client httpx
"""

import os

import modal

# Local-only imports (not needed in Modal container)
if not os.environ.get("MODAL_IS_REMOTE"):
    from wer_utils import (
        NotionFetcher,
        WERCalculator,
        calculate_and_save_results,
    )

# ============================================================================
# Modal App - MedASR Transcription (Ephemeral)
# ============================================================================

app = modal.App("medasr-transcription")

MODEL_ID = "google/medasr"

# Reference the Kaggle dataset volume (for --kaggle mode)
kaggle_volume = modal.Volume.from_name("medical-speech-dataset")

medasr_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libsndfile1", "ffmpeg", "build-essential", "cmake")
    .pip_install(
        "torch",
        "torchaudio",
        "librosa",
        "soundfile",
        "scipy",
        "jiwer",
        "pandas",
        "huggingface_hub",
        # KenLM and pyctcdecode for language model decoding
        "kenlm==0.3.0",
        "git+https://github.com/mediacatch/pyctcdecode.git@ff49fc562bf8fc5d6697d4dcd34188dd630cc977",
        # Install transformers from specific commit for MedASR support
        "git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5",
    )
)


def _restore_text(text: str) -> str:
    """
    Restore text from pyctcdecode format.

    pyctcdecode converts "▁" to spaces internally, so by the time we get the text,
    all "▁" prefixes have become spaces between tokens.
    "#" was used to mark original word boundaries (original "▁" in vocabulary).

    So we:
    1. Remove all spaces (which came from our artificial "▁" prefixes)
    2. Convert "#" back to space (restore original word boundaries)
    3. Remove </s> end-of-sequence tokens
    """
    return text.replace(" ", "").replace("#", " ").replace("</s>", "").strip()


def _normalize_for_wer(s: str) -> str:
    """
    Normalize text for WER calculation - matches official MedASR notebook.

    https://github.com/Google-Health/medasr/blob/main/notebooks/quick_start_with_hugging_face.ipynb
    """
    import re
    s = s.lower()
    s = s.replace('</s>', '')  # Remove end-of-sequence tokens from MedASR
    s = re.sub(r"[^ a-z0-9']", ' ', s)  # Keep only alphanumeric, space, apostrophe
    s = ' '.join(s.split())  # Normalize whitespace
    return s


def _calculate_medasr_wer(reference: str, hypothesis: str) -> dict:
    """
    Calculate WER using MedASR notebook's normalization.
    """
    import jiwer

    ref = _normalize_for_wer(reference)
    hyp = _normalize_for_wer(hypothesis)
    output = jiwer.process_words(ref, hyp)

    return {
        "wer": output.wer,
        "mer": output.mer,
        "wil": output.wil,
        "insertions": output.insertions,
        "deletions": output.deletions,
        "substitutions": output.substitutions,
        "hits": output.hits,
    }


def _create_pipeline_with_lm(model_id: str, lm_path: str):
    """Create ASR pipeline with language model for better accuracy."""
    import dataclasses
    import pyctcdecode
    import transformers

    class LasrCtcBeamSearchDecoder:
        def __init__(self, tokenizer, kenlm_model_path=None, **kwargs):
            vocab = [None for _ in range(tokenizer.vocab_size)]
            for k, v in tokenizer.vocab.items():
                if v < tokenizer.vocab_size:
                    vocab[v] = k
            assert not [i for i in vocab if i is None]
            vocab[0] = ""
            for i in range(1, len(vocab)):
                piece = vocab[i]
                if not piece.startswith("<") and not piece.endswith(">"):
                    piece = "▁" + piece.replace("▁", "#")
                vocab[i] = piece
            self._decoder = pyctcdecode.build_ctcdecoder(vocab, kenlm_model_path, **kwargs)

        def decode_beams(self, *args, **kwargs):
            beams = self._decoder.decode_beams(*args, **kwargs)
            return [dataclasses.replace(i, text=_restore_text(i.text)) for i in beams]

    feature_extractor = transformers.LasrFeatureExtractor.from_pretrained(model_id)
    feature_extractor._processor_class = "LasrProcessorWithLM"
    pipe = transformers.pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        feature_extractor=feature_extractor,
        decoder=LasrCtcBeamSearchDecoder(
            transformers.AutoTokenizer.from_pretrained(model_id), lm_path
        ),
    )
    assert pipe.type == "ctc_with_lm"
    return pipe


@app.cls(
    image=medasr_image,
    gpu="A10G",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")]
)
class MedASRTranscriber:
    @modal.enter()
    def load_model(self):
        import huggingface_hub

        print(f"Loading {MODEL_ID} with language model...")

        # Download the KenLM language model
        lm_path = huggingface_hub.hf_hub_download(MODEL_ID, filename='lm_6.kenlm')
        print(f"Downloaded LM to: {lm_path}")

        # Create pipeline with language model
        self.pipe = _create_pipeline_with_lm(MODEL_ID, lm_path)
        print("Model loaded with LM!")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text using MedASR with language model.

        Args:
            audio_bytes: Raw audio file bytes (m4a, mp3, wav, etc.)

        Returns:
            Transcribed text
        """
        import tempfile
        import os
        import subprocess

        # Detect format from magic bytes
        if audio_bytes[:12].find(b'ftyp') >= 0:
            suffix = ".m4a"
        elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
            suffix = ".mp3"
        elif audio_bytes[:4] == b'RIFF':
            suffix = ".wav"
        elif audio_bytes[:4] == b'fLaC':
            suffix = ".flac"
        elif audio_bytes[:4] == b'OggS':
            suffix = ".ogg"
        else:
            suffix = ".m4a"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            input_path = f.name

        output_path = input_path.rsplit('.', 1)[0] + ".wav"

        try:
            # Convert to 16kHz mono WAV (required for MedASR)
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path
            ], check=True, capture_output=True)

            # Load audio to check duration
            import librosa
            audio, sr = librosa.load(output_path, sr=16000, mono=True)
            duration_s = len(audio) / sr

            # Use chunking only for audio longer than 10 seconds
            # Short audio can be processed directly without chunking
            if duration_s > 10:
                result = self.pipe(
                    output_path,
                    chunk_length_s=20,
                    stride_length_s=2,
                    decoder_kwargs=dict(beam_width=8),
                )
            else:
                # For short audio, process directly without chunking
                result = self.pipe(
                    {"raw": audio, "sampling_rate": sr},
                    decoder_kwargs=dict(beam_width=8),
                )

            return result['text'].strip()

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


# ============================================================================
# Kaggle Dataset Evaluator (Volume-based)
# ============================================================================

# Image with wer_utils.py for Kaggle evaluation
medasr_kaggle_image = medasr_image.add_local_file(
    os.path.join(os.path.dirname(__file__), "wer_utils.py"),
    "/root/wer_utils.py",
)

@app.cls(
    image=medasr_kaggle_image,
    gpu="A10G",
    timeout=1800,
    volumes={"/data": kaggle_volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
class MedASRKaggleEvaluator:
    """Evaluates MedASR on Kaggle medical speech dataset with language model."""

    @modal.enter()
    def load_model(self):
        import huggingface_hub

        print(f"Loading {MODEL_ID} with language model...")

        # Download the KenLM language model
        lm_path = huggingface_hub.hf_hub_download(MODEL_ID, filename='lm_6.kenlm')
        print(f"Downloaded LM to: {lm_path}")

        # Create pipeline with language model
        self.pipe = _create_pipeline_with_lm(MODEL_ID, lm_path)
        print("Model loaded with LM!")

    @modal.method()
    def evaluate_dataset(self, split: str = "validate") -> list[dict]:
        """Evaluate MedASR on Kaggle dataset using LM-based pipeline."""
        import sys
        import librosa
        sys.path.insert(0, "/root")
        from wer_utils import load_kaggle_dataset

        entries = load_kaggle_dataset(split)
        results = []

        for i, entry in enumerate(entries):
            print(f"[{i+1}/{len(entries)}] {entry['file_name']}")
            try:
                audio, sr = librosa.load(entry["path"], sr=16000, mono=True)
                duration_s = len(audio) / sr

                if duration_s > 10:
                    result = self.pipe(
                        audio,
                        chunk_length_s=20,
                        stride_length_s=2,
                        decoder_kwargs=dict(beam_width=8),
                    )
                else:
                    result = self.pipe(audio, decoder_kwargs=dict(beam_width=8))

                transcript = result['text'].strip()
                metrics = _calculate_medasr_wer(entry["transcript"], transcript)
                print(f"  WER: {metrics['wer']:.2%}")

                results.append({
                    "name": entry["file_name"],
                    "ground_truth": entry["transcript"],
                    "transcript": transcript,
                    **metrics
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "name": entry["file_name"],
                    "ground_truth": entry["transcript"],
                    "transcript": "",
                    "wer": 1.0,
                    "error": str(e)
                })

        return results


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(
        database_id: str,
        output_csv: str = "results.csv",
        error_report_path: str = "error_analysis.txt",
):
    """
    Run the complete MedASR WER pipeline.

    Args:
        database_id: Notion database ID
        output_csv: Path for results CSV
        error_report_path: Path for detailed error analysis report
    """
    print("=" * 60)
    print("Google MedASR WER Pipeline")
    print("=" * 60)

    print(f"Model: {MODEL_ID}")

    # Initialize
    print("\n[1/5] Fetching entries from Notion...")
    fetcher = NotionFetcher()
    entries = fetcher.get_entries(database_id)
    print(f"  Found {len(entries)} entries")

    if not entries:
        print("No entries found!")
        return

    wer_calc = WERCalculator()
    results = []

    # Run Modal transcription (ephemeral)
    print("\n[2/5] Starting MedASR transcription...")

    with app.run():
        transcriber = MedASRTranscriber()

        for i, entry in enumerate(entries):
            name = entry["name"]
            print(f"\n  [{i+1}/{len(entries)}] {name}")

            try:
                # Download
                print(f"    Downloading audio...")
                audio_bytes = fetcher.download_audio(entry["audio_url"])
                print(f"    Downloaded {len(audio_bytes):,} bytes")

                # Transcribe
                print(f"    Transcribing...")
                transcript = transcriber.transcribe.remote(audio_bytes)

                # Calculate WER
                metrics = wer_calc.calculate(entry["ground_truth"], transcript)
                print(f"    WER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")

                # Show quick error summary for high WER
                if metrics['wer'] > 0.15:
                    errors = wer_calc.get_error_details(entry["ground_truth"], transcript)
                    if errors["substitutions"]:
                        print(f"    Top substitutions: {errors['substitutions'][:3]}")

                results.append({
                    "name": name,
                    "ground_truth": entry["ground_truth"],
                    "transcript": transcript,
                    **metrics
                })

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({
                    "name": name,
                    "ground_truth": entry.get("ground_truth", ""),
                    "transcript": "",
                    "wer": 1.0,
                    "error": str(e)
                })

    # Calculate and save results
    return calculate_and_save_results(
        results=results,
        wer_calc=wer_calc,
        model_name=MODEL_ID,
        output_csv=output_csv,
        error_report_path=error_report_path,
    )


# ============================================================================
# Kaggle Pipeline
# ============================================================================

def run_kaggle_pipeline(
        split: str = "validate",
        output_dir: str = ".",
):
    """
    Run MedASR WER evaluation on Kaggle medical speech dataset.

    Args:
        split: Dataset split to use ('validate' or 'train')
        output_dir: Directory for output CSV files
    """
    import pandas as pd

    model_name = "medasr"

    print("=" * 60)
    print("MedASR Kaggle Dataset WER Evaluation")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Split: {split}")
    print()

    with app.run():
        evaluator = MedASRKaggleEvaluator()
        results = evaluator.evaluate_dataset.remote(split)

    # Save results
    output_csv = os.path.join(output_dir, f"kaggle-results-{model_name}.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Calculate summary
    valid = [r for r in results if "error" not in r or not r.get("error")]
    if valid:
        avg_wer = sum(r["wer"] for r in valid) / len(valid)
    else:
        avg_wer = 1.0

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY - {model_name}")
    print(f"{'=' * 60}")
    print(f"Entries: {len(results)} total, {len(valid)} successful")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Results saved to: {output_csv}")

    return {
        "model": model_name,
        "model_id": MODEL_ID,
        "avg_wer": avg_wer,
        "samples": len(valid),
        "total": len(results),
        "output_csv": output_csv,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate WER using Google MedASR with detailed error analysis"
    )
    parser.add_argument(
        "--database-id",
        default="294a6166c4978050930fea2073e66dc2",
        help="Notion database ID (for Notion mode)",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV path (for Notion mode)",
    )
    parser.add_argument(
        "--error-report",
        default="error_analysis.txt",
        help="Error analysis report path (for Notion mode)",
    )
    # Kaggle mode arguments
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Evaluate on Kaggle medical speech dataset instead of Notion",
    )
    parser.add_argument(
        "--split",
        default="validate",
        choices=["validate", "train"],
        help="Kaggle dataset split to use (default: validate)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for Kaggle results CSV",
    )

    args = parser.parse_args()

    if args.kaggle:
        run_kaggle_pipeline(
            split=args.split,
            output_dir=args.output_dir,
        )
    else:
        run_pipeline(
            database_id=args.database_id,
            output_csv=args.output,
            error_report_path=args.error_report,
        )


if __name__ == "__main__":
    main()