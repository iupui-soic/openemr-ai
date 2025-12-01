"""
Groq Whisper WER Calculator

Fetches audio from Notion database, transcribes with Groq's Whisper Large v3 Turbo,
calculates Word Error Rate, and generates detailed error analysis.

Usage:
    python groq_wer.py --output results.csv

Requirements:
    pip install groq jiwer pandas requests notion-client httpx python-dotenv
"""

import os
import argparse
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from groq import Groq
from wer_utils import (
    NotionFetcher,
    WERCalculator,
    calculate_and_save_results,
)

# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "whisper-large-v3-turbo"


# ============================================================================
# Groq Transcriber
# ============================================================================

class GroqTranscriber:
    """Transcribe audio using Groq's Whisper API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Set GROQ_API_KEY environment variable")
        self.client = Groq(api_key=self.api_key)

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file bytes (m4a, mp3, wav, etc.)

        Returns:
            Transcribed text
        """
        # Detect format from magic bytes for proper file extension
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

        # Write to temp file (Groq API needs a file path)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(temp_path), audio_file.read()),
                    model=MODEL_ID,
                    temperature=0,
                    response_format="verbose_json",
                )
            return transcription.text.strip()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(
        database_id: str,
        output_csv: str = "groq_results.csv",
        error_report_path: str = "groq_error_analysis.txt",
):
    """
    Run the complete Groq Whisper WER pipeline.

    Args:
        database_id: Notion database ID
        output_csv: Path for results CSV
        error_report_path: Path for detailed error analysis report
    """
    print("=" * 60)
    print("Groq Whisper WER Pipeline")
    print("=" * 60)

    print(f"Model: groq/{MODEL_ID}")

    # Initialize components
    print("\n[1/5] Initializing...")
    fetcher = NotionFetcher()
    transcriber = GroqTranscriber()
    wer_calc = WERCalculator()

    # Fetch entries from Notion
    print("\n[2/5] Fetching entries from Notion...")
    entries = fetcher.get_entries(database_id)
    print(f"  Found {len(entries)} entries")

    if not entries:
        print("No entries found!")
        return

    results = []

    # Process each entry
    print("\n[3/5] Transcribing audio files...")

    for i, entry in enumerate(entries):
        name = entry["name"]
        print(f"\n  [{i+1}/{len(entries)}] {name}")

        try:
            # Download audio
            print(f"    Downloading audio...")
            audio_bytes = fetcher.download_audio(entry["audio_url"])
            print(f"    Downloaded {len(audio_bytes):,} bytes")

            # Transcribe with Groq
            print(f"    Transcribing with Groq...")
            transcript = transcriber.transcribe(audio_bytes)
            print(f"    Transcribed: {len(transcript)} characters")

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

    # Calculate and save results using shared utility
    return calculate_and_save_results(
        results=results,
        wer_calc=wer_calc,
        model_name=f"groq/{MODEL_ID}",
        output_csv=output_csv,
        error_report_path=error_report_path,
    )


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate WER using Groq Whisper with detailed error analysis"
    )
    parser.add_argument(
        "--database-id",
        default="294a6166c4978050930fea2073e66dc2",
        help="Notion database ID",
    )
    parser.add_argument(
        "--output",
        default="groq_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--error-report",
        default="groq_error_analysis.txt",
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