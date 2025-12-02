"""
NVIDIA Parakeet TDT WER Calculator

Fetches audio from Notion database, transcribes with NVIDIA Parakeet-TDT-1.1B on Modal (ephemeral),
calculates Word Error Rate, and generates detailed error analysis.

Parakeet-TDT-1.1B is NVIDIA's fast automatic speech recognition model using the TDT
(Token-and-Duration Transducer) architecture for efficient inference.

Usage:
    python parakeet_wer.py --output results.csv

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

app = modal.App("parakeet-transcription")

MODEL_ID = "nvidia/parakeet-tdt-1.1b"

parakeet_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libsndfile1", "ffmpeg", "build-essential")
    .pip_install(
        "cython",
        "packaging",
        "torch",
        "torchaudio",
        "nemo_toolkit[asr]",
    )
)


@app.cls(image=parakeet_image, gpu="A10G", timeout=600)
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

        output_path = input_path.rsplit('.', 1)[0] + ".wav"

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
        help="Notion database ID",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--error-report",
        default="error_analysis.txt",
        help="Error analysis report path",
    )

    args = parser.parse_args()

    run_pipeline(
        database_id=args.database_id,
        output_csv=args.output,
        error_report_path=args.error_report,
    )


if __name__ == "__main__":
    main()