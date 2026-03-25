"""
Step 4.9.4: Fareez RAG Loader — drop-in replacement for NotionFetcher.

Loads the 40 selected Fareez OSCE transcripts paired with matched OpenEMR
patient data. Returns the same dict format as NotionFetcher for compatibility
with existing pipeline scripts.

Usage:
    from fareez_rag_loader import FareezLoader

    loader = FareezLoader()
    patients = loader.get_entries()

    for patient in patients:
        print(patient["patient_name"])       # e.g., "RES0007"
        print(patient["transcript"])          # D:/P: conversation text
        print(patient["openemr_data"])        # formatted EHR extract
        print(patient["category"])            # RES, MSK, GAS, CAR, DER
        print(patient["detected_condition"])  # e.g., "Asthma"
"""

import os
import json


class FareezLoader:
    """
    Load 40 selected Fareez OSCE transcripts with matched OpenEMR data.

    Returns list of dicts compatible with the existing pipeline interface
    (same keys as NotionFetcher: patient_name, transcript, openemr_data).

    Additional keys: category, detected_condition, matched_openemr_pid.
    """

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(__file__)
        self.base_dir = base_dir
        self.selection_file = os.path.join(base_dir, "fareez_selected_40.json")
        self.extracts_dir = os.path.join(base_dir, "fareez_openemr_extracts")
        self.transcript_dir = os.path.normpath(os.path.join(
            base_dir, "..", "..", "openemr_whisper_wer", "data",
            "fareez_osce", "Data", "Clean Transcripts"
        ))

    def _load_transcript(self, file_name):
        """Load transcript text from file."""
        path = os.path.join(self.transcript_dir, f"{file_name}.txt")
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                return f.read()

    def _load_openemr_extract(self, file_name):
        """Load OpenEMR extract text from file."""
        path = os.path.join(self.extracts_dir, f"{file_name}_openemr.txt")
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def get_entries(self):
        """
        Load all 40 selected entries with transcripts and OpenEMR data.

        Returns:
            List of dicts with keys: patient_name, transcript, openemr_data,
            manual_reference_summary, category, detected_condition, matched_openemr_pid
        """
        with open(self.selection_file, "r") as f:
            selection = json.load(f)

        entries = []
        for item in selection:
            file_name = item["file_name"]
            transcript = self._load_transcript(file_name)
            openemr_data = self._load_openemr_extract(file_name)

            entry = {
                "patient_name": file_name,
                "transcript": transcript,
                "openemr_data": openemr_data,
                "manual_reference_summary": "",  # No reference — using clinician eval instead
                "category": item.get("category", ""),
                "detected_condition": item.get("detected_condition", ""),
                "matched_openemr_pid": item.get("matched_openemr_pid"),
            }
            entries.append(entry)

        return entries


if __name__ == "__main__":
    loader = FareezLoader()
    entries = loader.get_entries()

    print(f"Loaded {len(entries)} Fareez entries\n")
    for e in entries:
        print(f"  {e['patient_name']} ({e['category']}, {e['detected_condition']})")
        print(f"    Transcript: {len(e['transcript']):,} chars")
        print(f"    OpenEMR:    {len(e['openemr_data']):,} chars")
