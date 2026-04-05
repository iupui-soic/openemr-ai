"""
Utilities for patient summarization tasks, including Notion fetching.

This module provides the NotionFetcher class used by all model pipelines
to fetch patient data for medical transcript summarization.

Notion Database Schema (exact column names):
    - patient_name: Title field (required)
    - transcript: Rich text field containing doctor-patient conversation (required)
    - openemr_data: Rich text field containing EHR data (optional)
    - manual_reference_summary: Rich text field containing reference summary for evaluation (optional)

Usage:
    from summary_utils import NotionFetcher

    fetcher = NotionFetcher()
    patients = fetcher.get_entries()

    for patient in patients:
        print(patient["patient_name"])
        print(patient["transcript"])
        print(patient.get("openemr_data", ""))
        print(patient.get("manual_reference_summary", ""))
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from notion_client import Client as NotionClient
import httpx

# Load environment variables from .env
load_dotenv()


class NotionFetcher:
    """
    Fetch patient data from Notion database for summarization tasks.

    Environment Variables Required:
        VISHNU_NOTION: Notion API integration token
        VISHNU_NOTION_DB_ID: Notion database ID (32-char hex or UUID format)

    Returns List of Dicts with keys:
        - patient_name: str (from title field, e.g., "bhavana")
        - transcript: str (doctor-patient conversation)
        - openemr_data: str (EHR data, may be empty string)
        - manual_reference_summary: str (reference for evaluation, may be empty string)
        - To use add inside main call
    """

    def __init__(self, api_key: Optional[str] = None, database_id: Optional[str] = None):
        """
        Initialize NotionFetcher with API credentials.

        Args:
            api_key: Notion API token (defaults to VISHNU_NOTION env var)
            database_id: Notion database ID (defaults to VISHNU_NOTION_DB_ID env var)
        """
        self.api_key = api_key or os.environ.get("VISHNU_NOTION")
        self.database_id = database_id or os.environ.get("VISHNU_NOTION_DB_ID")

        if not self.api_key:
            raise ValueError(
                "Notion API key not found. Set VISHNU_NOTION environment variable or pass api_key parameter."
            )
        if not self.database_id:
            raise ValueError(
                "Notion database ID not found. Set VISHNU_NOTION_DB_ID environment variable or pass database_id parameter."
            )

        self.client = NotionClient(auth=self.api_key)
        self.database_id = self._format_uuid(self.database_id)

    def _format_uuid(self, raw_id: str) -> str:
        """Convert 32-char hex string to UUID format with dashes."""
        raw_id = raw_id.replace("-", "")
        if len(raw_id) == 32:
            return f"{raw_id[:8]}-{raw_id[8:12]}-{raw_id[12:16]}-{raw_id[16:20]}-{raw_id[20:]}"
        return raw_id

    def _extract_title(self, prop: Dict[str, Any]) -> str:
        """Extract plain text from a title property."""
        if prop.get("title"):
            return "".join(t.get("plain_text", "") for t in prop["title"])
        return ""

    def _extract_rich_text(self, prop: Dict[str, Any]) -> str:
        """Extract plain text from a rich_text property."""
        if prop.get("rich_text"):
            return "".join(t.get("plain_text", "") for t in prop["rich_text"])
        return ""

    def get_entries(self) -> List[Dict[str, Any]]:
        """
        Fetch all patient entries from the Notion database.

        Expected columns (exact names):
            - patient_name (title)
            - transcript (rich_text)
            - openemr_data (rich_text)
            - manual_reference_summary (rich_text)

        Returns:
            List of patient dictionaries
        """
        # Verify database exists
        try:
            db_info = self.client.databases.retrieve(database_id=self.database_id)
            db_title = ""
            if db_info.get("title"):
                db_title = "".join(t.get("plain_text", "") for t in db_info["title"])
            print(f"✅ Connected to database: {db_title or self.database_id}")
        except Exception as e:
            raise ValueError(f"Cannot access database {self.database_id}: {e}")

        entries = []
        has_more = True
        next_cursor = None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }

        while has_more:
            body = {"start_cursor": next_cursor} if next_cursor else {}

            try:
                resp = httpx.post(
                    f"https://api.notion.com/v1/databases/{self.database_id}/query",
                    headers=headers,
                    json=body,
                    timeout=30
                )
                resp.raise_for_status()
                response = resp.json()
            except httpx.HTTPStatusError as e:
                print(f"❌ Error fetching from Notion API: {e.response.text}")
                raise

            for page in response["results"]:
                props = page.get("properties", {})

                # Extract fields using exact column names
                entry = {
                    "patient_name": "",
                    "transcript": "",
                    "openemr_data": "",
                    "manual_reference_summary": "",
                }

                # patient_name (title field)
                if "patient_name" in props:
                    entry["patient_name"] = self._extract_title(props["patient_name"])

                # transcript (rich_text field)
                if "transcript" in props:
                    entry["transcript"] = self._extract_rich_text(props["transcript"])

                # openemr_data (rich_text field)
                if "openemr_data" in props:
                    entry["openemr_data"] = self._extract_rich_text(props["openemr_data"])

                # manual_reference_summary (rich_text field)
                if "manual_reference_summary" in props:
                    entry["manual_reference_summary"] = self._extract_rich_text(props["manual_reference_summary"])

                # Only include entries with required fields (patient_name and transcript)
                if entry["patient_name"] and entry["transcript"]:
                    entries.append(entry)

                    # Log with character counts
                    print(f"  📋 {entry['patient_name']}: "
                          f"transcript={len(entry['transcript']):,} chars, "
                          f"openemr={len(entry['openemr_data']):,} chars, "
                          f"reference={len(entry['manual_reference_summary']):,} chars")

            has_more = response.get("has_more", False)
            next_cursor = response.get("next_cursor")

        print(f"\n✅ Fetched {len(entries)} patient entries")
        return entries


def generate_consolidated_report(
        results_dir: str = "results",
        output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Generate consolidated comparison report from all model results.

    Scans results_dir for evaluation CSV files from all models,
    combines them, calculates averages, and generates comparison reports.

    Args:
        results_dir: Directory containing model result subdirectories
        output_dir: Directory to save consolidated reports

    Returns:
        Dict with:
            - model_summary: DataFrame with average metrics per model
            - consolidated: DataFrame with all patient results
            - best_models: Dict with best model per metric

    Output Files:
        - consolidated_results.csv: All patient results from all models
        - model_comparison.csv: Average metrics per model
        - comparison_report.md: Markdown report for GitHub Summary
    """
    import pandas as pd
    from pathlib import Path

    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all evaluation CSV files
    csv_files = list(results_path.glob("**/evaluation_results_*.csv"))

    print(f"🔍 Scanning {results_path} for result files...")
    print(f"   Found {len(csv_files)} CSV files")

    if not csv_files:
        print("❌ No evaluation CSV files found!")
        return {"error": "No CSV files found"}

    # Load and combine all results
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Filter out AVERAGE rows (we'll recalculate)
            df_patients = df[df['patient_name'] != 'AVERAGE'].copy()
            all_dfs.append(df_patients)
            print(f"   ✅ Loaded: {csv_file.name} ({len(df_patients)} patients)")
        except Exception as e:
            print(f"   ❌ Error loading {csv_file}: {e}")

    if not all_dfs:
        print("❌ No valid data loaded!")
        return {"error": "No valid data"}

    # Combine all results
    consolidated = pd.concat(all_dfs, ignore_index=True)

    # Calculate average metrics per model
    model_summary = consolidated.groupby('model').agg({
        'bleu': 'mean',
        'rouge_l': 'mean',
        'sbert_coherence': 'mean',
        'bert_f1': 'mean',
        'total_time_s': 'mean'
    }).round(4)

    # Sort by BERTScore F1 (descending)
    model_summary = model_summary.sort_values('bert_f1', ascending=False)

    # Find best model per metric
    best_models = {
        'bleu': {'model': model_summary['bleu'].idxmax(), 'score': model_summary['bleu'].max()},
        'rouge_l': {'model': model_summary['rouge_l'].idxmax(), 'score': model_summary['rouge_l'].max()},
        'sbert_coherence': {'model': model_summary['sbert_coherence'].idxmax(), 'score': model_summary['sbert_coherence'].max()},
        'bert_f1': {'model': model_summary['bert_f1'].idxmax(), 'score': model_summary['bert_f1'].max()},
        'fastest': {'model': model_summary['total_time_s'].idxmin(), 'score': model_summary['total_time_s'].min()},
    }

    # Print summary to console
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Average across all patients)")
    print("=" * 80)
    print(model_summary.to_string())
    print("\n" + "=" * 80)
    print("BEST MODELS BY METRIC")
    print("=" * 80)
    for metric, info in best_models.items():
        print(f"  {metric}: {info['model']} ({info['score']:.4f})")

    # Save CSV files
    consolidated_csv = output_path / "consolidated_results.csv"
    comparison_csv = output_path / "model_comparison.csv"

    consolidated.to_csv(consolidated_csv, index=False)
    model_summary.to_csv(comparison_csv)

    print(f"\n📁 Saved: {consolidated_csv}")
    print(f"📁 Saved: {comparison_csv}")

    # Generate Markdown report
    report_md = output_path / "comparison_report.md"

    with open(report_md, "w") as f:
        f.write("# 📊 Medical Summarization Model Comparison Report\n\n")

        # Model count and patient count
        num_models = len(model_summary)
        num_patients = consolidated['patient_name'].nunique()
        f.write(f"**Models evaluated:** {num_models}  \n")
        f.write(f"**Patients processed:** {num_patients}  \n\n")

        # Average metrics table
        f.write("## Average Metrics Across All Patients\n\n")
        f.write("| Model | BLEU | ROUGE-L | SBERT | BERTScore F1 | Avg Time (s) |\n")
        f.write("|-------|------|---------|-------|--------------|-------------|\n")

        for model, row in model_summary.iterrows():
            f.write(f"| {model} | {row['bleu']:.4f} | {row['rouge_l']:.4f} | ")
            f.write(f"{row['sbert_coherence']:.4f} | {row['bert_f1']:.4f} | {row['total_time_s']:.1f} |\n")

        # Best models
        f.write("\n## 🏆 Best Model by Metric\n\n")
        f.write(f"| Metric | Best Model | Score |\n")
        f.write(f"|--------|------------|-------|\n")
        f.write(f"| BLEU | {best_models['bleu']['model']} | {best_models['bleu']['score']:.4f} |\n")
        f.write(f"| ROUGE-L | {best_models['rouge_l']['model']} | {best_models['rouge_l']['score']:.4f} |\n")
        f.write(f"| SBERT Coherence | {best_models['sbert_coherence']['model']} | {best_models['sbert_coherence']['score']:.4f} |\n")
        f.write(f"| BERTScore F1 | {best_models['bert_f1']['model']} | {best_models['bert_f1']['score']:.4f} |\n")
        f.write(f"| Fastest | {best_models['fastest']['model']} | {best_models['fastest']['score']:.1f}s |\n")

        # Per-patient breakdown
        f.write("\n## Per-Patient Results\n\n")
        f.write("| Patient | Model | BLEU | ROUGE-L | SBERT | BERTScore F1 |\n")
        f.write("|---------|-------|------|---------|-------|-------------|\n")

        for _, row in consolidated.sort_values(['patient_name', 'model']).iterrows():
            f.write(f"| {row['patient_name']} | {row['model']} | {row['bleu']:.4f} | ")
            f.write(f"{row['rouge_l']:.4f} | {row['sbert_coherence']:.4f} | {row['bert_f1']:.4f} |\n")

        # Per-patient best model
        f.write("\n## Best Model Per Patient\n\n")
        f.write("| Patient | Best Model (by BERTScore F1) | Score |\n")
        f.write("|---------|------------------------------|-------|\n")

        for patient in consolidated['patient_name'].unique():
            patient_data = consolidated[consolidated['patient_name'] == patient]
            best_idx = patient_data['bert_f1'].idxmax()
            best_row = patient_data.loc[best_idx]
            f.write(f"| {patient} | {best_row['model']} | {best_row['bert_f1']:.4f} |\n")

    print(f"📁 Saved: {report_md}")

    return {
        "model_summary": model_summary,
        "consolidated": consolidated,
        "best_models": best_models,
        "files": {
            "consolidated_csv": str(consolidated_csv),
            "comparison_csv": str(comparison_csv),
            "report_md": str(report_md),
        }
    }


if __name__ == "__main__":
    import sys

    # Check if we're generating a report or fetching from Notion
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        # Generate consolidated report
        results_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
        output_dir = sys.argv[3] if len(sys.argv) > 3 else results_dir

        print("=" * 60)
        print("Generating Consolidated Report")
        print("=" * 60)

        result = generate_consolidated_report(results_dir, output_dir)

        if "error" not in result:
            print(f"\n✅ Report generated successfully!")
        else:
            print(f"\n❌ Error: {result['error']}")

    else:
        # Default: Fetch from Notion
        print("=" * 60)
        print("Notion Patient Database Fetcher")
        print("=" * 60)

        try:
            fetcher = NotionFetcher()
            entries = fetcher.get_entries()

            for entry in entries:
                print(f"\n📋 {entry['patient_name']}:")
                print(f"   Transcript:  {len(entry['transcript']):>6,} chars")
                print(f"   OpenEMR:     {len(entry['openemr_data']):>6,} chars")
                print(f"   Reference:   {len(entry['manual_reference_summary']):>6,} chars")

            print(f"\n✅ Total: {len(entries)} patients")

        except ValueError as e:
            print(f"❌ Error: {e}")