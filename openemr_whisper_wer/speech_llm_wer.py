"""
Speech LLM WER Calculator (Qwen3-ASR, Granite Speech)

Transcribes with an open-weight speech LLM on Modal (ephemeral) and calculates
Word Error Rate, in the same three modes as the other scripts in this folder:
the Notion recordings, a local dataset, or the Kaggle medical speech dataset.

Models (--model):
    qwen3-asr       Qwen/Qwen3-ASR-1.7B-hf           Alibaba, 2026-01, Apache 2.0
    granite-speech  ibm-granite/granite-speech-5.0-2b  IBM, 2026-10, Apache 2.0

Usage:
    python speech_llm_wer.py --model qwen3-asr --output results.csv
    python speech_llm_wer.py --model granite-speech --kaggle --output-dir ./results
    python speech_llm_wer.py --model qwen3-asr --kaggle --limit 20   # quick check
    python speech_llm_wer.py --model qwen3-asr --local-dataset primock57 --output results/primock57-qwen3-asr.csv

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
# Models
# ============================================================================

# max_chunk_s is the longest audio given to the model in one call. A longer
# recording is cut at quiet points and the pieces are transcribed in order.
#   qwen3-asr: 1200 s is the single-call limit of the model's own toolkit
#     (MAX_ASR_INPUT_SECONDS in QwenLM/Qwen3-ASR).
#   granite-speech: the model card asks for segments of up to about 2 minutes.
MODELS = {
    "qwen3-asr": {
        "model_id": "Qwen/Qwen3-ASR-1.7B-hf",
        "max_chunk_s": 1200,
    },
    "granite-speech": {
        "model_id": "ibm-granite/granite-speech-5.0-2b",
        "max_chunk_s": 100,
    },
}

GRANITE_PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."

# ============================================================================
# Modal App - Speech LLM Transcription (Ephemeral)
# ============================================================================

app = modal.App("speech-llm-transcription")

# Reference the Kaggle dataset volume (for --kaggle mode)
kaggle_volume = modal.Volume.from_name("medical-speech-dataset")

speech_llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers>=5.13.0",  # Qwen3-ASR needs 5.13 or newer
        "accelerate",
        "peft",
        "soundfile",
        "librosa",
        "sentencepiece",
        "protobuf",
        "numpy",
        "scipy",
        "jiwer",
        "pandas",
    )
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "wer_utils.py"),
        "/root/wer_utils.py",
    )
)


def split_at_quiet_points(audio, sample_rate: int, max_chunk_s: float, search_s: float = 15.0, frame_s: float = 0.3):
    """
    Cut audio into pieces no longer than max_chunk_s.

    Each cut is placed in the quietest frame of the last search_s seconds of the
    piece, so that it falls in a pause and not inside a word. The pieces, joined,
    are the original audio.
    """
    max_len = int(max_chunk_s * sample_rate)
    frame = int(frame_s * sample_rate)
    chunks = []
    start = 0
    while len(audio) - start > max_len:
        end = start + max_len
        lo = max(start + frame, end - int(search_s * sample_rate))
        window = audio[lo:end]
        n_frames = len(window) // frame
        energy = (window[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1)
        cut = lo + int(energy.argmin()) * frame + frame // 2
        chunks.append(audio[start:cut])
        start = cut
    chunks.append(audio[start:])
    return chunks


@app.cls(
    image=speech_llm_image,
    gpu="A10G",
    timeout=3600,
    volumes={"/data": kaggle_volume},
)
class SpeechLLM:
    """One of the models in MODELS, chosen by model_key."""

    model_key: str = modal.parameter()

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoProcessor

        self.config = MODELS[self.model_key]
        model_id = self.config["model_id"]
        print(f"Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        if self.model_key == "qwen3-asr":
            from transformers import AutoModelForMultimodalLM

            self.model = AutoModelForMultimodalLM.from_pretrained(
                model_id, device_map="cuda", dtype=torch.bfloat16
            )
        else:
            from transformers import AutoModelForSpeechSeq2Seq

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, device_map="cuda", dtype=torch.bfloat16
            )
        self.model.eval()
        print("Model loaded!")

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _load_mono_16k(self, audio_path: str):
        """Read an audio file as mono 16 kHz float32."""
        import numpy as np
        import soundfile as sf
        from scipy import signal

        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = signal.resample_poly(audio, 16000, sr)
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _decode_bytes(self, audio_bytes: bytes):
        """Decode audio file bytes (m4a, mp3, wav, ...) to mono 16 kHz float32."""
        import os
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(audio_bytes)
            input_path = f.name
        output_path = input_path + "_converted.wav"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path
            ], check=True, capture_output=True)
            return self._load_mono_16k(output_path)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")
        finally:
            for path in (input_path, output_path):
                if os.path.exists(path):
                    os.unlink(path)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _max_new_tokens(self, n_samples: int) -> int:
        # About 8 tokens per second of audio is well above fast speech, so the
        # limit does not cut a transcript short, and a model that starts to
        # repeat itself still stops.
        return min(8192, 128 + int(n_samples / 16000 * 8))

    def _transcribe_chunk(self, audio) -> str:
        """Transcribe one piece of mono 16 kHz audio, no longer than max_chunk_s."""
        import torch

        max_new_tokens = self._max_new_tokens(len(audio))

        if self.model_key == "qwen3-asr":
            import os
            import tempfile

            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            try:
                sf.write(wav_path, audio, 16000)
                inputs = self.processor.apply_transcription_request(
                    audio=wav_path, language="English",
                ).to(self.model.device, self.model.dtype)
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False
                    )
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                return self.processor.decode(generated, return_format="transcription_only")[0].strip()
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)

        # granite-speech
        tokenizer = self.processor.tokenizer
        chat = [{"role": "user", "content": GRANITE_PROMPT}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        wav = torch.from_numpy(audio).unsqueeze(0)
        inputs = self.processor(prompt, wav, device="cuda", return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:].unsqueeze(0)
        return tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )[0].strip()

    def _transcribe_audio(self, audio) -> str:
        """Transcribe mono 16 kHz audio of any length."""
        chunks = split_at_quiet_points(audio, 16000, self.config["max_chunk_s"])
        if len(chunks) > 1:
            print(f"  {len(audio) / 16000:.0f} s of audio in {len(chunks)} pieces")
        return " ".join(t for t in (self._transcribe_chunk(c) for c in chunks) if t)

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file bytes (m4a, mp3, wav, etc.)

        Returns:
            Transcribed text
        """
        return self._transcribe_audio(self._decode_bytes(audio_bytes))

    @modal.method()
    def evaluate_dataset(self, split: str = "validate", limit: int = 0) -> list[dict]:
        """Evaluate the model on the Kaggle dataset. limit > 0 takes the first files only."""
        import sys
        sys.path.insert(0, "/root")
        from wer_utils import load_kaggle_dataset, calculate_wer_metrics

        entries = load_kaggle_dataset(split)
        if limit > 0:
            entries = entries[:limit]
        results = []

        for i, entry in enumerate(entries):
            print(f"[{i+1}/{len(entries)}] {entry['file_name']}")
            try:
                transcript = self._transcribe_audio(self._load_mono_16k(entry["path"]))

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
        model_key: str,
        database_id: str,
        output_csv: str = "results.csv",
        error_report_path: str = "error_analysis.txt",
):
    """
    Run the complete WER pipeline on the Notion recordings.

    Args:
        model_key: A key of MODELS
        database_id: Notion database ID
        output_csv: Path for results CSV
        error_report_path: Path for detailed error analysis report
    """
    model_id = MODELS[model_key]["model_id"]

    print("=" * 60)
    print("Speech LLM WER Pipeline")
    print("=" * 60)

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
    print("\n[2/5] Starting transcription...")

    with app.run():
        transcriber = SpeechLLM(model_key=model_key)

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
# Local Dataset Pipeline (PriMock57 / Fareez OSCE)
# ============================================================================

def run_local_pipeline(
        model_key: str,
        dataset_name: str,
        output_csv: str = "results.csv",
):
    """
    Run WER evaluation on a local dataset (PriMock57 or Fareez OSCE).

    Args:
        model_key: A key of MODELS
        dataset_name: 'primock57' or 'fareez'
        output_csv: Path for results CSV
    """
    import pandas as pd
    from pathlib import Path

    model_id = MODELS[model_key]["model_id"]

    # Load dataset
    if dataset_name == "primock57":
        from primock57_utils import load_primock57_dataset
        entries = load_primock57_dataset("data/primock57")
    else:
        from fareez_utils import load_fareez_dataset
        entries = load_fareez_dataset("data/fareez_osce")

    print("=" * 60)
    print(f"Speech LLM Local Dataset WER Evaluation ({dataset_name})")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Entries: {len(entries)}")
    print()

    wer_calc = WERCalculator()
    results = []

    with app.run():
        transcriber = SpeechLLM(model_key=model_key)

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
    print(f"RESULTS SUMMARY - {model_key} on {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Entries: {len(results)} total, {len(valid)} successful")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Results saved to: {output_csv}")

    return {
        "model": model_key,
        "model_id": model_id,
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
        model_key: str,
        split: str = "validate",
        output_dir: str = ".",
        limit: int = 0,
):
    """
    Run WER evaluation on the Kaggle medical speech dataset.

    Args:
        model_key: A key of MODELS
        split: Dataset split to use ('validate' or 'train')
        output_dir: Directory for output CSV files
        limit: If > 0, evaluate the first files only
    """
    import pandas as pd

    model_id = MODELS[model_key]["model_id"]

    print("=" * 60)
    print("Speech LLM Kaggle Dataset WER Evaluation")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Split: {split}")
    print()

    with app.run():
        evaluator = SpeechLLM(model_key=model_key)
        results = evaluator.evaluate_dataset.remote(split, limit)

    # Save results
    output_csv = os.path.join(output_dir, f"kaggle-results-{model_key}.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Calculate summary
    valid = [r for r in results if "error" not in r or not r.get("error")]
    if valid:
        avg_wer = sum(r["wer"] for r in valid) / len(valid)
    else:
        avg_wer = 1.0

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY - {model_key}")
    print(f"{'=' * 60}")
    print(f"Entries: {len(results)} total, {len(valid)} successful")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Results saved to: {output_csv}")

    return {
        "model": model_key,
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
        description="Calculate WER using an open-weight speech LLM with detailed error analysis"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS),
        help="Model to evaluate",
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
        help="Output directory for Kaggle results (default: current directory)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Kaggle mode: evaluate the first N files only (default: all)",
    )
    # Local dataset mode
    parser.add_argument(
        "--local-dataset",
        choices=["primock57", "fareez"],
        help="Evaluate on a local dataset (primock57 or fareez)",
    )

    args = parser.parse_args()

    if args.local_dataset:
        run_local_pipeline(
            model_key=args.model,
            dataset_name=args.local_dataset,
            output_csv=args.output,
        )
    elif args.kaggle:
        run_kaggle_pipeline(
            model_key=args.model,
            split=args.split,
            output_dir=args.output_dir,
            limit=args.limit,
        )
    else:
        run_pipeline(
            model_key=args.model,
            database_id=args.database_id,
            output_csv=args.output,
            error_report_path=args.error_report,
        )


if __name__ == "__main__":
    main()
