"""
IBM Granite Speech WER Calculator

Fetches audio from Notion database, transcribes with IBM Granite Speech 3.3 8B on Modal (ephemeral),
calculates Word Error Rate, and generates detailed error analysis.

Granite Speech is IBM's multimodal speech-to-text model.

Usage:
    python granite_wer.py --output results.csv

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
# Modal App - Granite Speech Transcription (Ephemeral)
# ============================================================================

app = modal.App("granite-speech-transcription")

# Granite Speech model requires transformers with audio support + peft for LoRA
# Image version 2: fixed torchaudio backend
granite_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.4.0",  # Pin torch to avoid torchcodec default in torchaudio
        "torchaudio==2.4.0",
        "transformers>=4.52.0",
        "accelerate",
        "soundfile",
        "librosa",
        "peft==0.13.0",  # Pin to older version for compatibility
    )
)


@app.cls(image=granite_image, gpu="A10G", timeout=600)
class GraniteSpeechTranscriber:
    """IBM Granite Speech 3.3 8B transcriber."""

    # Use modal.parameter() instead of __init__ (Modal deprecation fix)
    model_id: str = modal.parameter(default="ibm-granite/granite-speech-3.3-8b")

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"Loading {self.model_id}...")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )

        print("Model loaded!")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text using Granite Speech.

        Args:
            audio_bytes: Raw audio file bytes (m4a, mp3, wav, etc.)

        Returns:
            Transcribed text
        """
        import tempfile
        import os
        import subprocess
        import torch
        import torchaudio

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
            # Convert to 16kHz mono WAV for optimal ASR performance
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path
            ], check=True, capture_output=True)

            # Load audio with torchaudio using soundfile backend (avoids torchcodec dependency)
            wav, sr = torchaudio.load(output_path, normalize=True, backend="soundfile")
            assert sr == 16000, f"Expected 16kHz, got {sr}Hz"

            # Create text prompt with audio placeholder (required by Granite Speech API)
            system_prompt = "You are a helpful AI assistant that transcribes speech to text accurately."
            user_prompt = "<|audio|>can you transcribe the speech into a written format?"
            chat = [
                dict(role="system", content=system_prompt),
                dict(role="user", content=user_prompt),
            ]
            prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

            # Process with Granite Speech (requires both text prompt and audio)
            model_inputs = self.processor(
                prompt,
                wav,
                device=self.device,
                return_tensors="pt",
            ).to(self.device)

            # Generate transcription
            with torch.no_grad():
                model_outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=448,
                    do_sample=False,
                    num_beams=1,
                )

            # Decode only the new tokens (skip input tokens)
            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
            transcription = self.tokenizer.batch_decode(
                new_tokens,
                add_special_tokens=False,
                skip_special_tokens=True,
            )[0]

            return transcription.strip()

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
    Run the complete Granite Speech WER pipeline.

    Args:
        database_id: Notion database ID
        output_csv: Path for results CSV
        error_report_path: Path for detailed error analysis report
    """
    print("=" * 60)
    print("IBM Granite Speech WER Pipeline")
    print("=" * 60)

    model_id = "ibm-granite/granite-speech-3.3-8b"
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
    print("\n[2/5] Starting Granite Speech transcription...")

    with app.run():
        transcriber = GraniteSpeechTranscriber()

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
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate WER using IBM Granite Speech with detailed error analysis"
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