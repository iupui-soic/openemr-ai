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
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libsndfile1", "ffmpeg", "build-essential")
    .pip_install(
        "torch",
        "torchaudio",
        "librosa",
        "soundfile",
        "scipy",
        "jiwer",
        "pandas",
        "huggingface_hub",
        # Install transformers from specific commit for MedASR support
        "git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5",
    )
)


@app.cls(image=medasr_image, gpu="A10G", timeout=600)
class MedASRTranscriber:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        print(f"Loading {MODEL_ID}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForCTC.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}!")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file bytes (m4a, mp3, wav, etc.)

        Returns:
            Transcribed text
        """
        import tempfile
        import os
        import subprocess
        import librosa

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

            # Load audio with librosa
            speech, sample_rate = librosa.load(output_path, sr=16000)

            # Process with MedASR
            inputs = self.processor(
                speech,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate transcription
            outputs = self.model.generate(**inputs)
            decoded_text = self.processor.batch_decode(outputs)[0]

            return decoded_text.strip()

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
)
class MedASRKaggleEvaluator:
    """Evaluates MedASR on Kaggle medical speech dataset."""

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        print(f"Loading {MODEL_ID}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForCTC.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}!")

    def _load_audio_16k(self, audio_path: str):
        """Load audio and resample to 16kHz mono."""
        import librosa

        speech, _ = librosa.load(audio_path, sr=16000, mono=True)
        return speech

    @modal.method()
    def evaluate_dataset(self, split: str = "validate") -> list[dict]:
        """Evaluate MedASR on Kaggle dataset."""
        import sys
        sys.path.insert(0, "/root")
        from wer_utils import load_kaggle_dataset, calculate_wer_metrics

        entries = load_kaggle_dataset(split)
        results = []

        for i, entry in enumerate(entries):
            print(f"[{i+1}/{len(entries)}] {entry['file_name']}")
            try:
                # Load and process audio
                speech = self._load_audio_16k(entry["path"])

                inputs = self.processor(
                    speech,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate transcription
                outputs = self.model.generate(**inputs)
                transcript = self.processor.batch_decode(outputs)[0].strip()

                metrics = calculate_wer_metrics(entry["transcript"], transcript)
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