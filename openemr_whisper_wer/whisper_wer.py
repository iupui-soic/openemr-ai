"""
Whisper WER Calculator

Fetches audio from Notion database, transcribes with Whisper v3 Turbo on Modal,
and calculates Word Error Rate.

Usage:
    python whisper_wer.py --output results.csv
"""

import os
from typing import Optional

import modal

# Local-only imports (not needed in Modal container)
if not os.environ.get("MODAL_IS_REMOTE"):
    import requests
    import jiwer
    import pandas as pd
    from notion_client import Client as NotionClient

# ============================================================================
# Modal App - Runs only during inference (not deployed)
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
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v3-turbo"

        print(f"Loading {model_id}...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)
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
        else:
            suffix = ".m4a"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            input_path = f.name

        output_path = input_path.replace(suffix, ".wav")

        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-f", "wav", output_path
            ], check=True, capture_output=True)

            result = self.pipe(output_path, return_timestamps=True)
            return result["text"]
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
        database_id = self._format_uuid(database_id)
        entries = []
        has_more = True
        next_cursor = None

        # First try to retrieve the database to check if it's valid
        try:
            db_info = self.client.databases.retrieve(database_id=database_id)
            print(f"  Database found: {database_id}")
        except Exception as e:
            print(f"  Database ID invalid, searching for databases...")
            self.search_for_database()
            raise ValueError(f"Invalid database ID: {database_id}. See above for available databases.")

        while has_more:
            # Use httpx directly since notion_client.request is having issues
            import httpx

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            }

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

                # Get name (title property)
                if "name" in props and props["name"].get("title"):
                    entry["name"] = "".join(
                        t.get("plain_text", "") for t in props["name"]["title"]
                    )

                # Get original_script (rich_text)
                if "original_script" in props and props["original_script"].get("rich_text"):
                    entry["ground_truth"] = "".join(
                        t.get("plain_text", "") for t in props["original_script"]["rich_text"]
                    )

                # Get raw_audio (files property)
                if "raw_audio" in props and props["raw_audio"].get("files"):
                    file_obj = props["raw_audio"]["files"][0]
                    if file_obj.get("type") == "file":
                        entry["audio_url"] = file_obj["file"]["url"]
                    elif file_obj.get("type") == "external":
                        entry["audio_url"] = file_obj["external"]["url"]

                # Only include if we have all required fields
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
# WER Calculator
# ============================================================================

class WERCalculator:
    def __init__(self):
        self.transform = jiwer.Compose([
            jiwer.RemovePunctuation(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])

    def calculate(self, reference: str, hypothesis: str) -> dict:
        ref = self.transform(reference)
        hyp = self.transform(hypothesis)
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


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(database_id: str, output_csv: str = "results.csv"):
    print("=" * 60)
    print("Whisper WER Pipeline")
    print("=" * 60)

    # Initialize
    print("\n[1/4] Fetching entries from Notion...")
    fetcher = NotionFetcher()
    entries = fetcher.get_entries(database_id)
    print(f"  Found {len(entries)} entries")

    if not entries:
        print("No entries found!")
        return

    wer_calc = WERCalculator()
    results = []

    # Run Modal app for transcription
    print("\n[2/4] Starting Whisper transcription (Modal will spin up GPU)...")

    with app.run():
        transcriber = WhisperTranscriber()

        for i, entry in enumerate(entries):
            name = entry["name"]
            print(f"\n  [{i+1}/{len(entries)}] {name}")

            try:
                # Download audio
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

    # Calculate summary
    print("\n[3/4] Calculating statistics...")
    valid = [r for r in results if "error" not in r]

    if valid:
        avg_wer = sum(r["wer"] for r in valid) / len(valid)
        all_refs = " ".join(wer_calc.transform(r["ground_truth"]) for r in valid)
        all_hyps = " ".join(wer_calc.transform(r["transcript"]) for r in valid)
        agg_wer = jiwer.wer(all_refs, all_hyps)
    else:
        avg_wer = agg_wer = 1.0

    # Save results
    print("\n[4/4] Saving results...")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model: openai/whisper-large-v3-turbo")
    print(f"Entries: {len(entries)} total, {len(valid)} successful")
    print(f"\nAverage WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Aggregate WER: {agg_wer:.4f} ({agg_wer*100:.2f}%)")
    print(f"\nResults saved to: {output_csv}")

    print("\nPer-entry scores:")
    for r in results:
        if "error" in r:
            print(f"  {r['name']}: ERROR - {r['error']}")
        else:
            print(f"  {r['name']}: {r['wer']:.4f} ({r['wer']*100:.2f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate WER using Whisper")
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
    args = parser.parse_args()

    run_pipeline(args.database_id, args.output)


if __name__ == "__main__":
    main()