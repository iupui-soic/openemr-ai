"""
Utilities for patient summarization tasks, including Notion fetching.
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from notion_client import Client as NotionClient
import httpx

# Load environment variables from .env
load_dotenv()


class NotionFetcher:
    """Fetch patient data from Notion database for summarization tasks."""

    def __init__(self, api_key: Optional[str] = None, database_id: Optional[str] = None):
        self.api_key = api_key or os.environ.get("VISHNU_NOTION")
        self.database_id = database_id or os.environ.get("VISHNU_NOTION_DB_ID")

        if not self.api_key:
            raise ValueError("Set VISHNU_NOTION environment variable or pass api_key")
        if not self.database_id:
            raise ValueError("Set VISHNU_NOTION_DB_ID environment variable or pass database_id")

        self.client = NotionClient(auth=self.api_key)
        self.database_id = self._format_uuid(self.database_id)

    def _format_uuid(self, raw_id: str) -> str:
        """Convert 32-char hex string to UUID format with dashes."""
        raw_id = raw_id.replace("-", "")
        if len(raw_id) == 32:
            return f"{raw_id[:8]}-{raw_id[8:12]}-{raw_id[12:16]}-{raw_id[16:20]}-{raw_id[20:]}"
        return raw_id

    def get_entries(self) -> List[Dict[str, Any]]:
        """Fetch patient entries from the Notion database."""
        # Verify database exists
        try:
            db_info = self.client.databases.retrieve(database_id=self.database_id)
            db_title = ""
            if db_info.get("title"):
                db_title = "".join(t.get("plain_text", "") for t in db_info["title"])
            print(f"✅ Connected to database: {db_title or self.database_id}")
        except Exception as e:
            raise ValueError(f"Invalid database ID: {self.database_id}. Error: {e}")

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
                entry = {"id": page["id"]}

                # Auto-detect title field (patient_name)
                for key, prop in props.items():
                    if prop["type"] == "title":
                        entry["patient_name"] = "".join(t.get("plain_text", "") for t in prop["title"])

                # Extract other known fields
                for field in ["transcript", "openemr_data", "manual_reference_summary"]:
                    if field in props and props[field].get("rich_text"):
                        entry[field] = "".join(t.get("plain_text", "") for t in props[field]["rich_text"])

                # Only include entries with at least patient_name and transcript
                if entry.get("patient_name") and entry.get("transcript"):
                    entries.append(entry)
                    print(f"  📋 Found patient: {entry['patient_name']}")

            has_more = response.get("has_more", False)
            next_cursor = response.get("next_cursor")

        print(f"\n✅ Fetched {len(entries)} patient entries")
        return entries


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Notion Patient Database Fetcher")
    print("=" * 60)

    try:
        fetcher = NotionFetcher()
        entries = fetcher.get_entries()

        for entry in entries:
            print(f"\n{entry.get('patient_name')}:")
            print(f"  Transcript: {len(entry.get('transcript', ''))} chars")
            print(f"  OpenEMR: {len(entry.get('openemr_data', ''))} chars")
            print(f"  Reference: {len(entry.get('manual_reference_summary', ''))} chars")

        print(f"\nTotal entries fetched: {len(entries)}")
    except ValueError as e:
        print(f"Error: {e}")
