"""
NVIDIA Parakeet TDT WER Calculator

Fetches audio from Notion database, transcribes with NVIDIA Parakeet-TDT-1.1B on Modal (ephemeral),
calculates Word Error Rate, and generates detailed error analysis.

Parakeet-TDT-1.1B is NVIDIA's fast automatic speech recognition model using the TDT
(Token-and-Duration Transducer) architecture for efficient inference.

Usage:
    python parakeet_wer.py --output results.csv
    python parakeet_wer.py --kaggle  # Evaluate on Kaggle medical speech dataset
    python parakeet_wer.py --kaggle --split validate --output-dir ./results
    python parakeet_wer.py --local-dataset primock57 --output results/primock57-parakeet.csv
    python parakeet_wer.py --local-dataset fareez --output results/fareez-parakeet.csv

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
# Modal App - Parakeet Transcription (Ephemeral)
# ============================================================================

app = modal.App("parakeet-06b-v2-transcription")

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"

# Reference the Kaggle dataset volume (for --kaggle mode)
kaggle_volume = modal.Volume.from_name("medical-speech-dataset")

parakeet_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libsndfile1", "ffmpeg", "build-essential")
    .pip_install(
        "cython",
        "packaging",
        "torch",
        "torchaudio",
        "nemo_toolkit[asr]",
        "jiwer",
        "pandas",
        "scipy",
    )
)


@app.cls(image=parakeet_image, gpu="A100", timeout=600)
class ParakeetTranscriber:
    @modal.enter()
    def load_model(self):
        import nemo.collections.asr as nemo_asr

        print(f"Loading {MODEL_ID}...")
        # Parakeet TDT uses EncDecRNNTBPEModel (RNNT-based transducer)
        self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=MODEL_ID,
            map_location="cuda"
        )
        self.model.eval()
        print("Model loaded!")

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

        output_path = input_path.rsplit('.', 1)[0] + "_converted.wav"

        try:
            # Convert to 16kHz mono WAV (required for Parakeet)
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path
            ], check=True, capture_output=True)

            # Transcribe with Parakeet TDT
            result = self.model.transcribe([output_path])

            # Handle return format - Parakeet returns list of hypothesis objects or strings
            if result and len(result) > 0:
                text = result[0]
                if hasattr(text, 'text'):
                    return text.text.strip()
                return str(text).strip()
            return ""

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
parakeet_kaggle_image = parakeet_image.add_local_file(
    os.path.join(os.path.dirname(__file__), "wer_utils.py"),
    "/root/wer_utils.py",
)

@app.cls(
    image=parakeet_kaggle_image,
    gpu="A10G",
    timeout=1800,
    volumes={"/data": kaggle_volume},
)
class ParakeetKaggleEvaluator:
    """Evaluates Parakeet on Kaggle medical speech dataset."""

    @modal.enter()
    def load_model(self):
        import nemo.collections.asr as nemo_asr

        print(f"Loading {MODEL_ID}...")
        self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=MODEL_ID,
            map_location="cuda"
        )
        self.model.eval()
        print("Model loaded!")

    def _convert_to_mono_16k(self, audio_path: str) -> str:
        """Convert audio to mono 16kHz WAV (required by NeMo models)."""
        import soundfile as sf
        import numpy as np
        import tempfile
        from scipy import signal

        audio, sr = sf.read(audio_path)

        # Convert stereo to mono if needed
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            num_samples = int(len(audio) * 16000 / sr)
            audio = signal.resample(audio, num_samples)
            sr = 16000

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio, sr)
        return temp_file.name

    @modal.method()
    def evaluate_dataset(self, split: str = "validate") -> list[dict]:
        """Evaluate Parakeet on Kaggle dataset."""
        import sys
        import os
        sys.path.insert(0, "/root")
        from wer_utils import load_kaggle_dataset, calculate_wer_metrics

        entries = load_kaggle_dataset(split)
        results = []

        for i, entry in enumerate(entries):
            print(f"[{i+1}/{len(entries)}] {entry['file_name']}")
            mono_path = None
            try:
                # Convert to mono 16kHz (NeMo requires single channel audio)
                mono_path = self._convert_to_mono_16k(entry["path"])

                result = self.model.transcribe([mono_path])

                if result and len(result) > 0:
                    text = result[0]
                    transcript = text.text.strip() if hasattr(text, 'text') else str(text).strip()
                else:
                    transcript = ""

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
            finally:
                # Clean up temp file
                if mono_path and os.path.exists(mono_path):
                    os.unlink(mono_path)

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
    Run the complete Parakeet WER pipeline.

    Args:
        database_id: Notion database ID
        output_csv: Path for results CSV
        error_report_path: Path for detailed error analysis report
    """
    print("=" * 60)
    print("NVIDIA Parakeet TDT WER Pipeline")
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
    print("\n[2/5] Starting Parakeet transcription...")

    with app.run():
        transcriber = ParakeetTranscriber()

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
# Local Dataset Pipeline (PriMock57 / Fareez OSCE)
# ============================================================================

def run_local_pipeline(
        dataset_name: str,
        output_csv: str = "results.csv",
):
    """
    Run Parakeet WER evaluation on a local dataset (PriMock57 or Fareez OSCE).

    Args:
        dataset_name: 'primock57' or 'fareez'
        output_csv: Path for results CSV
    """
    import pandas as pd
    from pathlib import Path

    model_name = "parakeet-tdt-06b-v2"

    # Load dataset
    if dataset_name == "primock57":
        from primock57_utils import load_primock57_dataset
        entries = load_primock57_dataset("data/primock57")
    else:
        from fareez_utils import load_fareez_dataset
        entries = load_fareez_dataset("data/fareez_osce")

    print("=" * 60)
    print(f"Parakeet Local Dataset WER Evaluation ({dataset_name})")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Entries: {len(entries)}")
    print()

    wer_calc = WERCalculator()
    results = []

    with app.run():
        transcriber = ParakeetTranscriber()

        for i, entry in enumerate(entries):
            print(f"\n  [{i+1}/{len(entries)}] {entry['file_name']}")
            try:
                audio_bytes = Path(entry["path"]).read_bytes()
                print(f"    Read {len(audio_bytes):,} bytes")

                transcript = transcriber.transcribe.remote(audio_bytes)

                metrics = wer_calc.calculate(entry["transcript"], transcript)
                print(f"    WER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")

                results.append({
                    "name": entry["file_name"],
                    "ground_truth": entry["transcript"],
                    "transcript": transcript,
                    **metrics
                })
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({
                    "name": entry["file_name"],
                    "ground_truth": entry.get("transcript", ""),
                    "transcript": "",
                    "wer": 1.0,
                    "error": str(e)
                })

    # Save results
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    valid = [r for r in results if "error" not in r or not r.get("error")]
    avg_wer = sum(r["wer"] for r in valid) / len(valid) if valid else 1.0

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY - {model_name} on {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Entries: {len(results)} total, {len(valid)} successful")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Results saved to: {output_csv}")

    return {
        "model": model_name,
        "model_id": MODEL_ID,
        "dataset": dataset_name,
        "avg_wer": avg_wer,
        "samples": len(valid),
        "total": len(results),
        "output_csv": output_csv,
    }


# ============================================================================
# Kaggle Pipeline
# ============================================================================

def run_kaggle_pipeline(
        split: str = "validate",
        output_dir: str = ".",
):
    """
    Run Parakeet WER evaluation on Kaggle medical speech dataset.

    Args:
        split: Dataset split to use ('validate' or 'train')
        output_dir: Directory for output CSV files
    """
    import pandas as pd

    model_name = "parakeet-tdt-06b-v2"

    print("=" * 60)
    print("Parakeet Kaggle Dataset WER Evaluation")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Split: {split}")
    print()

    with app.run():
        evaluator = ParakeetKaggleEvaluator()
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
        description="Calculate WER using NVIDIA Parakeet TDT with detailed error analysis"
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
    # Local dataset mode
    parser.add_argument(
        "--local-dataset",
        choices=["primock57", "fareez"],
        help="Use a local dataset (PriMock57 or Fareez OSCE)",
    )

    args = parser.parse_args()

    if args.local_dataset:
        run_local_pipeline(
            dataset_name=args.local_dataset,
            output_csv=args.output,
        )
    elif args.kaggle:
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