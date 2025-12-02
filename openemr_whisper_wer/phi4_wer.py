"""
Microsoft Phi-4 Multimodal WER Calculator

Fetches audio from Notion database, transcribes with Phi-4 Multimodal Instruct on Modal (ephemeral),
calculates Word Error Rate, and generates detailed error analysis.

Phi-4 Multimodal is Microsoft's multimodal model that can process audio for ASR tasks.

Usage:
    python phi4_wer.py --output results.csv

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
# Modal App - Phi-4 Multimodal Transcription (Ephemeral)
# ============================================================================

app = modal.App("phi4-multimodal-transcription")

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

# Phi-4 Multimodal requires specific versions per official docs:
# https://huggingface.co/microsoft/Phi-4-multimodal-instruct
# Use prebuilt flash-attn wheel from Dao-AILab releases (much faster than building)
# See: https://modal.com/docs/examples/install_flash_attn
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

phi4_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.48.2",  # Official recommended version for Phi-4
        "accelerate==1.3.0",     # Official recommended version
        "soundfile",
        "librosa",
        "scipy",
        "peft==0.13.2",          # Official recommended version
        "backoff",
        "packaging",
        FLASH_ATTN_WHEEL,        # Prebuilt wheel - no compilation needed
    )
)

# Create persistent volume for model caching (Phi-4 is large)
model_volume = modal.Volume.from_name("phi4-model-cache", create_if_missing=True)


@app.cls(
    image=phi4_image,
    gpu="A100",  # Phi-4 benefits from more VRAM
    timeout=900,
    volumes={"/cache": model_volume},
    scaledown_window=300,
)
class Phi4Transcriber:
    """Microsoft Phi-4 Multimodal transcriber for ASR."""

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Use the mounted volume for caching
        cache_base = "/cache"
        hf_cache = os.path.join(cache_base, "huggingface")
        os.makedirs(hf_cache, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache
        os.environ["TORCH_HOME"] = cache_base

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Loading {MODEL_ID}...")
        print(f"Device: {self.device}")
        print(f"Cache: {hf_cache}")

        # Check if model is already cached
        model_marker = os.path.join(cache_base, ".phi4_downloaded")

        if os.path.exists(model_marker):
            print("Loading from cached model in volume...")
        else:
            print("First run: downloading model to volume (this may take 5-10 minutes)...")

        try:
            self.processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                cache_dir=hf_cache,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                cache_dir=hf_cache,
                device_map="auto",
                attn_implementation="flash_attention_2",  # Use flash attention for better performance
            )
            self.model.eval()

            # Mark as successfully downloaded
            if not os.path.exists(model_marker):
                with open(model_marker, 'w') as f:
                    f.write(f"Model {MODEL_ID} cached successfully\n")
                model_volume.commit()
                print("Model cached to volume successfully!")

            print("Model loaded and ready!")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text using Phi-4 Multimodal.

        Args:
            audio_bytes: Raw audio file bytes (m4a, mp3, wav, etc.)

        Returns:
            Transcribed text
        """
        import tempfile
        import os
        import subprocess
        import torch
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
            # Convert to 16kHz mono WAV
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path
            ], check=True, capture_output=True)

            # Load audio
            audio, sr = librosa.load(output_path, sr=16000)

            # Create ASR prompt for Phi-4 multimodal
            # Phi-4 uses a chat-style format with audio input
            # Audio placeholder must be <|audio_1|> (numbered) and wrapped in chat template
            user_message = "<|audio_1|>\nTranscribe the speech in this audio clip exactly as spoken."
            messages = [
                {"role": "user", "content": user_message},
            ]
            prompt = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs - Phi-4 expects audios as list of (audio, sample_rate) tuples
            inputs = self.processor(
                text=prompt,
                audios=[(audio, sr)],
                return_tensors="pt",
            ).to(self.model.device)

            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=448,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode - skip the input tokens
            input_len = inputs.get("input_ids", inputs.get("input_features")).shape[1]
            transcription = self.processor.decode(
                generated_ids[0][input_len:],
                skip_special_tokens=True,
            )

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
    Run the complete Phi-4 Multimodal WER pipeline.

    Args:
        database_id: Notion database ID
        output_csv: Path for results CSV
        error_report_path: Path for detailed error analysis report
    """
    print("=" * 60)
    print("Microsoft Phi-4 Multimodal WER Pipeline")
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
    print("\n[2/5] Starting Phi-4 Multimodal transcription...")
    print("Note: First run will download model to volume (5-10 min), then fast after that.")

    modal.enable_output()
    with app.run():
        transcriber = Phi4Transcriber()

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
        description="Calculate WER using Microsoft Phi-4 Multimodal with detailed error analysis"
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
