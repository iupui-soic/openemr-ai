"""
Shared utilities for WER (Word Error Rate) calculation.

Contains:
- NotionFetcher: Fetch audio entries from Notion database
- KaggleFetcher: Fetch entries from Kaggle medical speech dataset
- WERCalculator: Calculate WER with detailed error analysis
- generate_error_report: Generate detailed error analysis report
"""

import os
from typing import Optional
from collections import Counter

# Local-only imports (not needed in Modal container)
if not os.environ.get("MODAL_IS_REMOTE"):
    import requests
    import jiwer
    import pandas as pd
    from notion_client import Client as NotionClient


# ============================================================================
# Kaggle Dataset Fetcher (runs inside Modal with volume mounted)
# ============================================================================

def load_kaggle_dataset(split: str = "validate", data_dir: str = "/data/Medical Speech, Transcription, and Intent") -> list[dict]:
    """
    Load audio files and transcripts from Kaggle medical speech dataset on Modal volume.

    This function is designed to run inside a Modal container with the dataset volume mounted.

    Args:
        split: Dataset split to use ('validate' or 'train')
        data_dir: Base directory of the dataset on the Modal volume

    Returns:
        List of dicts with: file_name, path, transcript, prompt
    """
    import pandas as pd
    from pathlib import Path

    data_path = Path(data_dir)
    recordings_dir = data_path / "recordings" / split
    csv_path = data_path / "overview-of-recordings.csv"

    if not recordings_dir.exists():
        raise ValueError(f"Split '{split}' not found at {recordings_dir}")

    # Get audio files
    audio_files = list(recordings_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {split} split")

    # Load CSV for transcripts
    df = pd.read_csv(csv_path)

    # Match audio files with transcripts
    results = []
    for audio_path in audio_files:
        file_name = audio_path.name
        transcript_row = df[df['file_name'] == file_name]

        if not transcript_row.empty:
            results.append({
                "file_name": file_name,
                "path": str(audio_path),
                "transcript": transcript_row['phrase'].iloc[0],
                "prompt": transcript_row['prompt'].iloc[0] if 'prompt' in transcript_row.columns else None
            })

    print(f"Matched {len(results)} files with transcripts")
    return results


def calculate_wer_metrics(reference: str, hypothesis: str) -> dict:
    """
    Calculate WER and related metrics (for use inside Modal containers).

    Args:
        reference: Ground truth transcript
        hypothesis: Model's transcription

    Returns:
        Dict with wer, mer, wil, insertions, deletions, substitutions, hits
    """
    import jiwer

    transform = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
    ])

    ref = transform(reference)
    hyp = transform(hypothesis)
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
        model_name: str = "unknown",
        output_path: str = "error_analysis.txt"
):
    """
    Generate detailed error analysis report.

    Args:
        results: List of result dicts from pipeline
        wer_calc: WERCalculator instance
        model_name: Name of the model used for transcription
        output_path: Path to save report
    """
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"TRANSCRIPTION ERROR ANALYSIS REPORT - {model_name}\n")
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

        f.write(f"\n\nMOST COMMON SUBSTITUTIONS (what the model gets wrong):\n")
        f.write("-" * 50 + "\n")
        for (ref_w, hyp_w), count in Counter(all_subs).most_common(30):
            f.write(f"  '{ref_w}' → '{hyp_w}': {count}x\n")

        f.write(f"\n\nMOST COMMONLY DELETED WORDS (model misses these):\n")
        f.write("-" * 50 + "\n")
        for word, count in Counter(all_dels).most_common(30):
            f.write(f"  '{word}': {count}x\n")

        f.write(f"\n\nMOST COMMONLY INSERTED WORDS (model hallucinates these):\n")
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
# Shared Pipeline Logic
# ============================================================================

def calculate_and_save_results(
        results: list,
        wer_calc: WERCalculator,
        model_name: str,
        output_csv: str,
        error_report_path: str,
):
    """
    Calculate statistics, save results, and generate reports.

    Args:
        results: List of result dicts from transcription
        wer_calc: WERCalculator instance
        model_name: Name of the model used
        output_csv: Path for results CSV
        error_report_path: Path for detailed error analysis report

    Returns:
        Dict with average_wer, aggregate_wer, results
    """
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
    generate_error_report(results, wer_calc, model_name, error_report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Entries: {len(results)} total, {len(valid)} successful")
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