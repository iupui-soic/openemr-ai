"""
Fee-sheet insert logic: resolves patient/encounter UUIDs, enforces the
same-day-visit restriction, and writes matched codes into OpenEMR's
billing table as AI-suggested (pending human review).
"""
import datetime
import logging
import os
from typing import List, Optional, Tuple

import pymysql

logger = logging.getLogger(__name__)


def get_db_connection():
    return pymysql.connect(
        host=os.getenv("OPENEMR_DB_HOST", "127.0.0.1"),
        port=int(os.getenv("OPENEMR_DB_PORT", "3308")),
        user=os.getenv("OPENEMR_DB_USER", "openemr"),
        password=os.getenv("OPENEMR_DB_PASS", "openemr"),
        database=os.getenv("OPENEMR_DB_NAME", "openemr"),
    )


def _uuid_to_hex(uuid_str: str) -> str:
    """Strip dashes so it matches how MariaDB's UNHEX() expects a plain hex string."""
    return uuid_str.replace("-", "")


def resolve_patient_and_encounter(
    cursor, patient_uuid: str, encounter_uuid: str
) -> Tuple[Optional[int], Optional[int], Optional[datetime.datetime]]:
    """
    Resolve external UUIDs to OpenEMR's internal pid/encounter integers,
    and fetch the encounter date for the same-day-visit check.

    MariaDB has no UUID_TO_BIN/BIN_TO_UUID (those are MySQL 8.0+ only), so
    we convert manually with UNHEX() against the dash-stripped UUID string.
    """
    cursor.execute(
        "SELECT pid FROM patient_data WHERE uuid = UNHEX(%s)",
        (_uuid_to_hex(patient_uuid),),
    )
    row = cursor.fetchone()
    if not row:
        return None, None, None
    pid = row[0]

    cursor.execute(
        "SELECT encounter, date FROM form_encounter WHERE uuid = UNHEX(%s)",
        (_uuid_to_hex(encounter_uuid),),
    )
    row = cursor.fetchone()
    if not row:
        return pid, None, None
    encounter, encounter_date = row

    return pid, encounter, encounter_date


def is_same_day_visit(encounter_date: datetime.datetime) -> bool:
    today = datetime.date.today()
    encounter_day = encounter_date.date() if hasattr(encounter_date, "date") else encounter_date
    return encounter_day == today


def dedupe_matches(matches: List[dict]) -> List[dict]:
    """Drop matches with a duplicate (code_type, code) pair, keeping the first occurrence."""
    seen = set()
    deduped = []
    for m in matches:
        pair = (m.get("code_type", ""), m.get("code", ""))
        if pair not in seen:
            seen.add(pair)
            deduped.append(m)
    return deduped


def insert_billing_codes(cursor, pid: int, encounter: int, matches: List[dict]) -> List[int]:
    """Insert matched codes into the billing table as AI-suggested, pending review."""
    billing_ids = []
    for m in dedupe_matches(matches):
        code = m.get("code")
        code_type = m.get("code_type", "")
        desc = m.get("description", "")
        if code_type not in ("CPT4", "ICD10"):
            continue

        cursor.execute(
            """
            INSERT INTO billing
                (date, code_type, code, pid, provider_id, authorized,
                 encounter, code_text, billed, activity, units, notecodes, revenue_code)
            VALUES
                (NOW(), %s, %s, %s, 0, 0, %s, %s, 0, 1, 1, '', '')
            """,
            (code_type, code, pid, encounter, f"AI-SUGGESTED: {desc}"),
        )
        billing_ids.append(cursor.lastrowid)
    return billing_ids
