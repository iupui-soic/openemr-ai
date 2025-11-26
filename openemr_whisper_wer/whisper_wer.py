"""
Whisper WER Calculator with Error Analysis

Fetches audio from Notion database, transcribes with Whisper v3 on Modal,
calculates Word Error Rate, and generates detailed error analysis.

Usage:
    python whisper_wer.py --output results.csv
    python whisper_wer.py --output results.csv --use-large-v3  # More accurate, slower
    python whisper_wer.py --output results.csv --medical-prompt  # Add medical context

Requirements:
    pip install modal jiwer pandas requests notion-client httpx
"""

import os
from typing import Optional
from collections import Counter

import modal

# Local-only imports (not needed in Modal container)
if not os.environ.get("MODAL_IS_REMOTE"):
    import requests
    import jiwer
    import pandas as pd
    from notion_client import Client as NotionClient

# ============================================================================
# Modal App - Whisper Transcription
# ============================================================================

app = modal.App("whisper-transcription")

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
    )
)


@app.cls(image=whisper_image, gpu="A10G", timeout=600)
class WhisperTranscriber:
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
        self.model_id = model_id

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
# Notion Fetcher
# ============================================================================

class NotionFetcher:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("NOTION_API_KEY")
        if not self.api_key:
            raise ValueError("Set NOTION_API_KEY environment variable")
        self.client = NotionClient(auth=self.api_key)

    def _format_uuid(self, raw_id: str) -> str:
        raw_id = raw_id.replace("-", "")
        if len(raw_id) == 32:
            return f"{raw_id[:8]}-{raw_id[8:12]}-{raw_id[12:16]}-{raw_id[16:20]}-{raw_id[20:]}"
        return raw_id

    def search_for_database(self) -> Optional[str]:
        """Search for a database with audio data."""
        results = self.client.search(filter={"property": "object", "value": "database"})
        for db in results.get("results", []):
            title = ""
            if db.get("title"):
                title = "".join(t.get("plain_text", "") for t in db["title"])
            print(f"  Found database: {db['id']} - {title}")
        return None

    def get_entries(self, database_id: str) -> list[dict]:
        """Fetch entries from Notion database."""
        import httpx

        database_id = self._format_uuid(database_id)
        entries = []
        has_more = True
        next_cursor = None

        try:
            self.client.databases.retrieve(database_id=database_id)
            print(f"  Database found: {database_id}")
        except Exception:
            print(f"  Database ID invalid, searching for databases...")
            self.search_for_database()
            raise ValueError(f"Invalid database ID: {database_id}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }

        while has_more:
            body = {}
            if next_cursor:
                body["start_cursor"] = next_cursor

            resp = httpx.post(
                f"https://api.notion.com/v1/databases/{database_id}/query",
                headers=headers,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            response = resp.json()

            for page in response["results"]:
                props = page.get("properties", {})
                entry = {"id": page["id"]}

                if "name" in props and props["name"].get("title"):
                    entry["name"] = "".join(
                        t.get("plain_text", "") for t in props["name"]["title"]
                    )

                if "original_script" in props and props["original_script"].get("rich_text"):
                    entry["ground_truth"] = "".join(
                        t.get("plain_text", "") for t in props["original_script"]["rich_text"]
                    )

                if "raw_audio" in props and props["raw_audio"].get("files"):
                    file_obj = props["raw_audio"]["files"][0]
                    if file_obj.get("type") == "file":
                        entry["audio_url"] = file_obj["file"]["url"]
                    elif file_obj.get("type") == "external":
                        entry["audio_url"] = file_obj["external"]["url"]

                if entry.get("name") and entry.get("ground_truth") and entry.get("audio_url"):
                    entries.append(entry)

            has_more = response.get("has_more", False)
            next_cursor = response.get("next_cursor")

        return entries

    def download_audio(self, url: str) -> bytes:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        return response.content


# ============================================================================
# WER Calculator with Error Analysis
# ============================================================================

class WERCalculator:
    """Calculate WER with detailed error analysis."""

    def __init__(self):
        # Standard WER normalization - no medical expansion
        # (expanding abbreviations was causing worse WER scores)
        self.transform = jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ToLowerCase(),
        ])

    def _prepare_text(self, text: str) -> str:
        """Apply normalization steps."""
        return self.transform(text)

    def calculate(self, reference: str, hypothesis: str) -> dict:
        """
        Calculate WER and related metrics.

        Args:
            reference: Ground truth transcript
            hypothesis: Model's transcription

        Returns:
            Dict with wer, mer, wil, insertions, deletions, substitutions, hits
        """
        ref = self._prepare_text(reference)
        hyp = self._prepare_text(hypothesis)
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

    def get_alignment(self, reference: str, hypothesis: str) -> str:
        """
        Get visual alignment of reference and hypothesis.

        Args:
            reference: Ground truth transcript
            hypothesis: Model's transcription

        Returns:
            String visualization of word alignment with measures
        """
        ref = self._prepare_text(reference)
        hyp = self._prepare_text(hypothesis)
        out = jiwer.process_words(ref, hyp)
        return jiwer.visualize_alignment(out, show_measures=True)

    def get_error_details(self, reference: str, hypothesis: str) -> dict:
        """
        Extract specific words that were wrong.

        Args:
            reference: Ground truth transcript
            hypothesis: Model's transcription

        Returns:
            Dict with lists of substitutions (tuples), deletions, insertions
        """
        ref = self._prepare_text(reference)
        hyp = self._prepare_text(hypothesis)
        out = jiwer.process_words(ref, hyp)

        ref_words = ref.split()
        hyp_words = hyp.split()

        errors = {
            "substitutions": [],  # (reference_word, hypothesis_word)
            "deletions": [],      # words in reference but missing
            "insertions": [],     # words in hypothesis but not in reference
        }

        for chunk in out.alignments[0]:
            if chunk.type == "substitute":
                for i, j in zip(
                        range(chunk.ref_start_idx, chunk.ref_end_idx),
                        range(chunk.hyp_start_idx, chunk.hyp_end_idx)
                ):
                    if i < len(ref_words) and j < len(hyp_words):
                        errors["substitutions"].append((ref_words[i], hyp_words[j]))
            elif chunk.type == "delete":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    if i < len(ref_words):
                        errors["deletions"].append(ref_words[i])
            elif chunk.type == "insert":
                for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    if j < len(hyp_words):
                        errors["insertions"].append(hyp_words[j])

        return errors


# ============================================================================
# Error Report Generator
# ============================================================================

def generate_error_report(
        results: list,
        wer_calc: WERCalculator,
        output_path: str = "error_analysis.txt"
):
    """
    Generate detailed error analysis report.

    Args:
        results: List of result dicts from pipeline
        wer_calc: WERCalculator instance
        output_path: Path to save report
    """
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("WHISPER TRANSCRIPTION ERROR ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        all_subs, all_dels, all_ins = [], [], []
        valid_results = [r for r in results if "error" not in r and r.get("transcript")]

        # Per-entry analysis
        for r in sorted(valid_results, key=lambda x: x["wer"], reverse=True):
            f.write(f"\n{'=' * 70}\n")
            f.write(f"Entry: {r['name']}\n")
            f.write(f"WER: {r['wer']*100:.2f}% | ")
            f.write(f"Insertions: {r['insertions']} | ")
            f.write(f"Deletions: {r['deletions']} | ")
            f.write(f"Substitutions: {r['substitutions']}\n")
            f.write(f"{'=' * 70}\n\n")

            # Show ground truth and transcript
            f.write("GROUND TRUTH:\n")
            f.write(f"{r['ground_truth'][:500]}{'...' if len(r['ground_truth']) > 500 else ''}\n\n")
            f.write("TRANSCRIPT:\n")
            f.write(f"{r['transcript'][:500]}{'...' if len(r['transcript']) > 500 else ''}\n\n")

            # Alignment visualization
            f.write("ALIGNMENT:\n")
            f.write("-" * 70 + "\n")
            f.write(wer_calc.get_alignment(r["ground_truth"], r["transcript"]))
            f.write("\n")

            # Error details
            errors = wer_calc.get_error_details(r["ground_truth"], r["transcript"])
            all_subs.extend(errors["substitutions"])
            all_dels.extend(errors["deletions"])
            all_ins.extend(errors["insertions"])

            if errors["substitutions"]:
                f.write(f"\nSUBSTITUTIONS ({len(errors['substitutions'])}):\n")
                for ref_w, hyp_w in errors["substitutions"][:20]:
                    f.write(f"  '{ref_w}' → '{hyp_w}'\n")
                if len(errors["substitutions"]) > 20:
                    f.write(f"  ... and {len(errors['substitutions']) - 20} more\n")

            if errors["deletions"]:
                f.write(f"\nDELETIONS ({len(errors['deletions'])}):\n")
                f.write(f"  {errors['deletions'][:30]}\n")
                if len(errors["deletions"]) > 30:
                    f.write(f"  ... and {len(errors['deletions']) - 30} more\n")

            if errors["insertions"]:
                f.write(f"\nINSERTIONS ({len(errors['insertions'])}):\n")
                f.write(f"  {errors['insertions'][:30]}\n")
                if len(errors["insertions"]) > 30:
                    f.write(f"  ... and {len(errors['insertions']) - 30} more\n")

        # Aggregate error patterns
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("AGGREGATE ERROR PATTERNS\n")
        f.write("=" * 80 + "\n")

        f.write(f"\nTotal errors across all entries:\n")
        f.write(f"  Substitutions: {len(all_subs)}\n")
        f.write(f"  Deletions: {len(all_dels)}\n")
        f.write(f"  Insertions: {len(all_ins)}\n")

        f.write(f"\n\nMOST COMMON SUBSTITUTIONS (what Whisper gets wrong):\n")
        f.write("-" * 50 + "\n")
        for (ref_w, hyp_w), count in Counter(all_subs).most_common(30):
            f.write(f"  '{ref_w}' → '{hyp_w}': {count}x\n")

        f.write(f"\n\nMOST COMMONLY DELETED WORDS (Whisper misses these):\n")
        f.write("-" * 50 + "\n")
        for word, count in Counter(all_dels).most_common(30):
            f.write(f"  '{word}': {count}x\n")

        f.write(f"\n\nMOST COMMONLY INSERTED WORDS (Whisper hallucinates these):\n")
        f.write("-" * 50 + "\n")
        for word, count in Counter(all_ins).most_common(30):
            f.write(f"  '{word}': {count}x\n")

        # Recommendations
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")

        if all_subs:
            common_subs = Counter(all_subs).most_common(10)
            medical_subs = [(r, h) for (r, h), _ in common_subs
                            if len(r) > 4 or len(h) > 4]  # Likely medical terms
            if medical_subs:
                f.write("1. Consider adding these terms to the medical prompt:\n")
                unique_terms = set()
                for ref_w, hyp_w in medical_subs:
                    unique_terms.add(ref_w)
                f.write(f"   {', '.join(unique_terms)}\n\n")

        f.write("2. If WER is high on specific entries, check:\n")
        f.write("   - Audio quality (background noise, multiple speakers)\n")
        f.write("   - Speaker accent or speech patterns\n")
        f.write("   - Domain-specific terminology\n\n")

        f.write("3. Consider post-processing corrections for common substitutions\n")

    print(f"Error analysis saved to: {output_path}")


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
    Run the complete pipeline.

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

    # Run Modal transcription
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

    # Calculate summary
    print("\n[3/5] Calculating statistics...")
    valid = [r for r in results if "error" not in r]

    if valid:
        avg_wer = sum(r["wer"] for r in valid) / len(valid)
        all_refs = " ".join(wer_calc._prepare_text(r["ground_truth"]) for r in valid)
        all_hyps = " ".join(wer_calc._prepare_text(r["transcript"]) for r in valid)
        agg_wer = jiwer.wer(all_refs, all_hyps)
    else:
        avg_wer = agg_wer = 1.0

    # Save CSV
    print("\n[4/5] Saving results...")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"  Results saved to: {output_csv}")

    # Generate error report
    print("\n[5/5] Generating error analysis report...")
    generate_error_report(results, wer_calc, error_report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Entries: {len(entries)} total, {len(valid)} successful")
    print(f"\nAverage WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Aggregate WER: {agg_wer:.4f} ({agg_wer*100:.2f}%)")

    print("\nPer-entry scores (sorted by WER):")
    for r in sorted(results, key=lambda x: x.get("wer", 1.0)):
        if "error" in r:
            print(f"  {r['name']}: ERROR - {r['error']}")
        else:
            print(f"  {r['name']}: {r['wer']:.4f} ({r['wer']*100:.2f}%)")

    print(f"\nDetailed error analysis: {error_report_path}")

    return {
        "average_wer": avg_wer,
        "aggregate_wer": agg_wer,
        "results": results,
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
    parser.add_argument(
        "--use-large-v3",
        action="store_true",
        help="Use whisper-large-v3 (more accurate, slower) instead of turbo",
    )

    args = parser.parse_args()

    run_pipeline(
        database_id=args.database_id,
        output_csv=args.output,
        use_large_v3=args.use_large_v3,
        error_report_path=args.error_report,
    )


if __name__ == "__main__":
    main()