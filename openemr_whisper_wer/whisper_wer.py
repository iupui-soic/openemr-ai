"""
Whisper WER Calculator

Fetches audio from Notion database, transcribes with Whisper on Modal (ephemeral),
calculates Word Error Rate, and generates detailed error analysis.

Usage:
    python whisper_wer.py --output results.csv
    python whisper_wer.py --output results.csv --use-large-v3  # More accurate, slower
    python whisper_wer.py --kaggle  # Evaluate on Kaggle medical speech dataset
    python whisper_wer.py --kaggle --split validate --output-dir ./results

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
# Modal App - Whisper Transcription (Ephemeral)
# ============================================================================

app = modal.App("whisper-transcription")

# Reference the Kaggle dataset volume (for --kaggle mode)
kaggle_volume = modal.Volume.from_name("medical-speech-dataset")

whisper_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "datasets[audio]",
        "soundfile",
        "librosa",
        "jiwer",
        "pandas",
    )
)


@app.cls(image=whisper_image, gpu="A10G", timeout=600)
class WhisperTranscriber:
    # Use modal.parameter() instead of __init__ (Modal deprecation fix)
    model_id: str = modal.parameter(default="openai/whisper-large-v3-turbo")

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Loading {self.model_id}...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
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

        output_path = input_path.rsplit('.', 1)[0] + ".wav"

        try:
            # Convert to 16kHz mono WAV for optimal Whisper performance
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path
            ], check=True, capture_output=True)

            result = self.pipe(output_path, return_timestamps=True)
            return result["text"].strip()

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

# Mount wer_utils.py for Kaggle evaluation
wer_utils_mount = modal.Mount.from_local_file(
    local_path=os.path.join(os.path.dirname(__file__), "wer_utils.py"),
    remote_path="/root/wer_utils.py",
)

@app.cls(
    image=whisper_image,
    gpu="A10G",
    timeout=1800,
    volumes={"/data": kaggle_volume},
    mounts=[wer_utils_mount],
)
class WhisperKaggleEvaluator:
    """Evaluates Whisper on Kaggle medical speech dataset."""

    # Use modal.parameter() instead of __init__ (Modal deprecation fix)
    model_id: str = modal.parameter(default="openai/whisper-large-v3-turbo")

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Loading {self.model_id}...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print("Model loaded!")

    @modal.method()
    def evaluate_dataset(self, split: str = "validate") -> list[dict]:
        """Evaluate Whisper on Kaggle dataset."""
        import sys
        sys.path.insert(0, "/root")
        from wer_utils import load_kaggle_dataset, calculate_wer_metrics

        entries = load_kaggle_dataset(split)
        results = []

        for i, entry in enumerate(entries):
            print(f"[{i+1}/{len(entries)}] {entry['file_name']}")
            try:
                result = self.pipe(entry["path"], return_timestamps=True)
                transcript = result["text"].strip()

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
        use_large_v3: bool = False,
        error_report_path: str = "error_analysis.txt",
):
    """
    Run the complete Whisper WER pipeline.

    Args:
        database_id: Notion database ID
        output_csv: Path for results CSV
        use_large_v3: Use whisper-large-v3 instead of turbo (slower, more accurate)
        error_report_path: Path for detailed error analysis report
    """
    print("=" * 60)
    print("Whisper WER Pipeline")
    print("=" * 60)

    model_id = "openai/whisper-large-v3" if use_large_v3 else "openai/whisper-large-v3-turbo"
    print(f"Model: {model_id}")

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
    print("\n[2/5] Starting Whisper transcription...")

    with app.run():
        transcriber = WhisperTranscriber()

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
        model_name=model_id,
        output_csv=output_csv,
        error_report_path=error_report_path,
    )


# ============================================================================
# Kaggle Pipeline
# ============================================================================

def run_kaggle_pipeline(
        split: str = "validate",
        output_dir: str = ".",
        use_large_v3: bool = False,
):
    """
    Run Whisper WER evaluation on Kaggle medical speech dataset.

    Args:
        split: Dataset split to use ('validate' or 'train')
        output_dir: Directory for output CSV files
        use_large_v3: Use whisper-large-v3 instead of turbo
    """
    import pandas as pd

    model_id = "openai/whisper-large-v3" if use_large_v3 else "openai/whisper-large-v3-turbo"
    model_name = "whisper-v3" if use_large_v3 else "whisper-turbo"

    print("=" * 60)
    print("Whisper Kaggle Dataset WER Evaluation")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Split: {split}")
    print()

    with app.run():
        evaluator = WhisperKaggleEvaluator(model_id=model_id)
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
        "model_id": model_id,
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
        description="Calculate WER using Whisper with detailed error analysis"
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
    parser.add_argument(
        "--use-large-v3",
        action="store_true",
        help="Use whisper-large-v3 (more accurate, slower) instead of turbo",
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
            use_large_v3=args.use_large_v3,
        )
    else:
        run_pipeline(
            database_id=args.database_id,
            output_csv=args.output,
            use_large_v3=args.use_large_v3,
            error_report_path=args.error_report,
        )


if __name__ == "__main__":
    main()
