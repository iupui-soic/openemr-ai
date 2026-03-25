"""
Step 4.9.3: Match OpenEMR patients by detected condition and extract full EHR data.

For each of the 40 selected Fareez transcripts:
1. Search OpenEMR lists table for patients with matching/related conditions
2. Pick a patient with the richest data (most encounters, labs, meds)
3. Extract full EHR data using the retrieve_5555.py query patterns
4. Save formatted extracts to fareez_openemr_extracts/

Database: MariaDB on port 3307, user openemr/openemr
"""

import os
import json
import mysql.connector

SELECTION_FILE = os.path.join(os.path.dirname(__file__), "fareez_selected_40.json")
EXTRACTS_DIR = os.path.join(os.path.dirname(__file__), "fareez_openemr_extracts")

# Mapping from detected conditions to OpenEMR search terms
CONDITION_SEARCH_MAP = {
    "Asthma": ["asthma"],
    "Pneumonia": ["pneumonia"],
    "Chest Pain": ["chest pain", "angina", "cardiac"],
    "Acute Bronchitis": ["bronchitis"],
    "Loss of Smell": ["anosmia", "sinusitis", "respiratory"],
    "Upper Respiratory Infection": ["pharyngitis", "respiratory", "sinusitis"],
    "Sore Throat": ["pharyngitis", "sore throat", "streptococcal"],
    "Sinusitis": ["sinusitis"],
    "Sinus Infection": ["sinusitis"],
    "Common Cold": ["pharyngitis", "sinusitis", "respiratory"],
    "Back Pain": ["back pain", "lumbar"],
    "Knee Arthritis": ["osteoarthritis of knee", "knee"],
    "Knee Pain": ["knee", "osteoarthritis of knee"],
    "Carpal Tunnel Syndrome": ["carpal", "wrist", "hand"],
    "Tennis Elbow": ["elbow", "tendon", "epicondylitis"],
    "Bicep Tendonitis": ["tendon", "rotator cuff", "shoulder"],
    "Hip Fracture": ["fracture of hip", "hip"],
    "Wrist Injury": ["wrist", "sprain of wrist", "fracture"],
    "Dupuytren's Contracture": ["contracture", "hand", "localized"],
    "Foot Pain": ["foot", "laceration of foot", "plantar"],
    "Morning Sickness": ["nausea", "pregnancy"],
    "Abdominal Pain": ["abdominal", "gastro", "nausea"],
    "Acute Diarrhea": ["diarrhea"],
    "Gastroenteritis": ["gastro", "diarrhea", "nausea"],
    "Cellulitis": ["cellulitis", "skin", "infection"],
}


def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        port=3307,
        user="openemr",
        password="openemr",
        database="openemr",
    )


def find_patient_by_condition(cursor, condition, used_pids):
    """Find a patient with matching condition that has rich data.

    Uses a fast two-step approach:
    1. Get candidate PIDs from lists table (fast indexed lookup)
    2. Pick best candidate by checking encounter count (cheap scalar subqueries)
    """
    search_terms = CONDITION_SEARCH_MAP.get(condition, [condition.lower()])

    for term in search_terms:
        # Step 1: Get candidate PIDs with this condition (fast)
        excl = ",".join(["%s"] * len(used_pids)) if used_pids else "0"
        query = f"""
        SELECT DISTINCT l.pid
        FROM lists l
        WHERE l.type = 'medical_problem'
          AND l.title LIKE %s
          AND l.pid NOT IN ({excl})
        LIMIT 50
        """
        params = [f"%{term}%"] + list(used_pids)
        cursor.execute(query, params)
        candidates = [row["pid"] for row in cursor.fetchall()]

        if not candidates:
            continue

        # Step 2: Score candidates by encounter count (fast per-candidate)
        best_pid = None
        best_score = -1
        for pid in candidates:
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM form_encounter WHERE pid = %s", (pid,)
            )
            score = cursor.fetchone()["cnt"]
            if score > best_score:
                best_score = score
                best_pid = pid

        if best_pid and best_score > 0:
            return best_pid

    # Fallback: any patient with encounters not yet used
    excl = ",".join(["%s"] * len(used_pids)) if used_pids else "0"
    cursor.execute(f"""
        SELECT fe.pid, COUNT(*) as cnt
        FROM form_encounter fe
        WHERE fe.pid NOT IN ({excl})
        GROUP BY fe.pid
        ORDER BY cnt DESC
        LIMIT 1
    """, list(used_pids) if used_pids else [])
    row = cursor.fetchone()
    return row["pid"] if row else None


def format_val(value):
    """Format a value safely for display."""
    try:
        num = float(value)
        return f"{num:.2f}"
    except (TypeError, ValueError):
        return str(value) if value else "N/A"


def extract_patient_data(cursor, pid):
    """Extract full EHR data for a patient, returning formatted text."""
    sections = []

    # Demographics
    cursor.execute("""
        SELECT fname, lname, DOB, sex, pubpid AS unit_no,
               street, city, state, postal_code
        FROM patient_data WHERE pid = %s
    """, (pid,))
    row = cursor.fetchone()
    if row:
        sections.append(
            "Demographics\n"
            f"Name: {row.get('fname', '')} {row.get('lname', '')}\n"
            f"DOB: {row.get('DOB', '')}\n"
            f"Sex: {row.get('sex', '')}\n"
            f"Unit Number: {row.get('unit_no', '')}\n"
            f"Address: {row.get('street', '')}, {row.get('city', '')}, "
            f"{row.get('state', '')} {row.get('postal_code', '')}"
        )

    # Social History
    cursor.execute("""
        SELECT alcohol, tobacco, recreational_drugs, coffee,
               exercise_patterns, sleep_patterns
        FROM history_data WHERE pid = %s
    """, (pid,))
    row = cursor.fetchone()
    if row:
        sections.append(
            "Social History\n"
            f"Coffee: {row.get('coffee', 'N/A')}\n"
            f"Tobacco: {row.get('tobacco', 'N/A')}\n"
            f"Alcohol: {row.get('alcohol', 'N/A')}\n"
            f"Recreational Drugs: {row.get('recreational_drugs', 'N/A')}\n"
            f"Exercise: {row.get('exercise_patterns', 'N/A')}\n"
            f"Sleep: {row.get('sleep_patterns', 'N/A')}"
        )

    # Family History
    cursor.execute("""
        SELECT history_mother, history_father, history_siblings, history_offspring,
               relatives_cancer, relatives_diabetes, relatives_high_blood_pressure,
               relatives_heart_problems, relatives_stroke
        FROM history_data WHERE pid = %s
    """, (pid,))
    row = cursor.fetchone()
    if row:
        sections.append(
            "Family History\n"
            f"Mother: {row.get('history_mother', 'N/A')}\n"
            f"Father: {row.get('history_father', 'N/A')}\n"
            f"Siblings: {row.get('history_siblings', 'N/A')}\n"
            f"Cancer: {row.get('relatives_cancer', 'N/A')}\n"
            f"Diabetes: {row.get('relatives_diabetes', 'N/A')}\n"
            f"Heart Problems: {row.get('relatives_heart_problems', 'N/A')}\n"
            f"Stroke: {row.get('relatives_stroke', 'N/A')}"
        )

    # Allergies
    cursor.execute("""
        SELECT title AS allergy, diagnosis, severity_al, begdate AS noted_on, comments
        FROM lists WHERE pid = %s AND type = 'allergy'
        ORDER BY begdate DESC
    """, (pid,))
    rows = cursor.fetchall()
    if rows:
        allergy_lines = ["Allergies"]
        for r in rows:
            allergy_lines.append(
                f"- {r.get('allergy', 'Unknown')} (Severity: {r.get('severity_al', 'N/A')}, "
                f"Noted: {r.get('noted_on', 'N/A')})"
            )
        sections.append("\n".join(allergy_lines))

    # Vitals (most recent)
    cursor.execute("""
        SELECT date, weight, height, bps, bpd, pulse, respiration,
               temperature, oxygen_saturation, BMI
        FROM form_vitals WHERE pid = %s
        ORDER BY date DESC LIMIT 1
    """, (pid,))
    row = cursor.fetchone()
    if row:
        sections.append(
            "Vitals\n"
            f"Date: {row.get('date', 'N/A')}\n"
            f"Temperature: {format_val(row.get('temperature'))} C\n"
            f"Blood Pressure: {format_val(row.get('bps'))}/{format_val(row.get('bpd'))} mmHg\n"
            f"Heart Rate: {format_val(row.get('pulse'))} bpm\n"
            f"Respiration Rate: {format_val(row.get('respiration'))} breaths/min\n"
            f"SpO2: {format_val(row.get('oxygen_saturation'))} %\n"
            f"Weight: {format_val(row.get('weight'))} kg\n"
            f"Height: {format_val(row.get('height'))} cm\n"
            f"BMI: {format_val(row.get('BMI'))}"
        )

    # Get most recent encounter
    cursor.execute("""
        SELECT encounter, date, date_end FROM form_encounter
        WHERE pid = %s ORDER BY date DESC LIMIT 1
    """, (pid,))
    enc = cursor.fetchone()

    if enc:
        enc_id = enc["encounter"]

        # Encounter Info
        cursor.execute("""
            SELECT fe.encounter, fe.date AS admission_date, fe.date_end AS discharge_date,
                   fe.reason AS chief_complaint, fe.discharge_disposition,
                   u.fname AS attending_fname, u.lname AS attending_lname, u.specialty
            FROM form_encounter fe
            LEFT JOIN users u ON fe.provider_id = u.id
            WHERE fe.pid = %s AND fe.encounter = %s
        """, (pid, enc_id))
        row = cursor.fetchone()
        if row:
            sections.append(
                "Encounter Info\n"
                f"Encounter: {row.get('encounter', '')}\n"
                f"Admission: {row.get('admission_date', '')} -> Discharge: {row.get('discharge_date', '')}\n"
                f"Chief Complaint: {row.get('chief_complaint', '')}\n"
                f"Attending: Dr. {row.get('attending_fname', '')} {row.get('attending_lname', '')} "
                f"({row.get('specialty', '')})"
            )

        # HPI (Subjective SOAP)
        cursor.execute("""
            SELECT fs.subjective FROM form_encounter fe
            LEFT JOIN form_soap fs ON fs.pid = fe.pid AND DATE(fs.date) = DATE(fe.date)
            WHERE fe.pid = %s AND fe.encounter = %s
        """, (pid, enc_id))
        row = cursor.fetchone()
        if row and row.get("subjective"):
            sections.append(f"History of Present Illness\n{row['subjective']}")

        # Past Medical History
        cursor.execute("""
            SELECT title, diagnosis, begdate, enddate
            FROM lists WHERE pid = %s AND type = 'medical_problem'
        """, (pid,))
        rows = cursor.fetchall()
        if rows:
            pmh_lines = ["Past Medical History"]
            for r in rows:
                end = r.get("enddate") or "Ongoing"
                pmh_lines.append(
                    f"- {r.get('title', 'N/A')} ({r.get('diagnosis', 'N/A')}, "
                    f"Onset: {r.get('begdate', 'N/A')}, End: {end})"
                )
            sections.append("\n".join(pmh_lines))

        # Physical Exam (Objective SOAP)
        cursor.execute("""
            SELECT fs.objective FROM form_encounter fe
            LEFT JOIN form_soap fs ON fs.pid = fe.pid AND DATE(fs.date) = DATE(fe.date)
            WHERE fe.pid = %s AND fe.encounter = %s
        """, (pid, enc_id))
        row = cursor.fetchone()
        if row and row.get("objective"):
            sections.append(f"Physical Exam\n{row['objective']}")

        # Lab Results
        cursor.execute("""
            SELECT pr.result_text, pr.result, pr.units, pr.range, pr.abnormal, pr.result_code
            FROM procedure_order po
            JOIN procedure_report prt ON po.procedure_order_id = prt.procedure_order_id
            JOIN procedure_result pr ON prt.procedure_report_id = pr.procedure_report_id
            WHERE po.patient_id = %s
            ORDER BY pr.date DESC LIMIT 20
        """, (pid,))
        rows = cursor.fetchall()
        if rows:
            lab_lines = ["Lab Results"]
            for r in rows:
                abnormal = " [ABNORMAL]" if r.get("abnormal") else ""
                lab_lines.append(
                    f"- {r.get('result_text', '')} (LOINC: {r.get('result_code', '')}): "
                    f"{r.get('result', '')} {r.get('units', '')} "
                    f"(Range: {r.get('range', '')}){abnormal}"
                )
            sections.append("\n".join(lab_lines))

        # Medications
        cursor.execute("""
            SELECT drug, rxnorm_drugcode, dosage, route, start_date, end_date, active
            FROM prescriptions
            WHERE patient_id = %s AND (encounter = %s OR encounter IS NULL)
            ORDER BY date_added DESC LIMIT 15
        """, (pid, enc_id))
        rows = cursor.fetchall()
        if rows:
            med_lines = ["Medications"]
            for r in rows:
                status = "Active" if r.get("active") == 1 else "Inactive"
                med_lines.append(
                    f"- {r.get('drug', 'Unknown')} (RxNorm: {r.get('rxnorm_drugcode', 'N/A')})\n"
                    f"  Dosage: {r.get('dosage', 'N/A')} | Route: {r.get('route', 'N/A')}\n"
                    f"  Start: {r.get('start_date', 'N/A')} -> End: {r.get('end_date', 'Ongoing')} ({status})"
                )
            sections.append("\n".join(med_lines))

        # Hospital Course (Assessment + Plan)
        cursor.execute("""
            SELECT fs.assessment, fs.plan FROM form_encounter fe
            LEFT JOIN form_soap fs ON fs.pid = fe.pid AND DATE(fs.date) = DATE(fe.date)
            WHERE fe.pid = %s AND fe.encounter = %s
        """, (pid, enc_id))
        row = cursor.fetchone()
        if row:
            if row.get("assessment"):
                sections.append(f"Assessment\n{row['assessment']}")
            if row.get("plan"):
                sections.append(f"Plan\n{row['plan']}")

        # Care Plan
        cursor.execute("""
            SELECT date, codetext, description, note_related_to
            FROM form_care_plan WHERE pid = %s AND encounter = %s
        """, (pid, enc_id))
        rows = cursor.fetchall()
        if rows:
            cp_lines = ["Care Plan"]
            for r in rows:
                cp_lines.append(
                    f"- {r.get('codetext', '')} | {r.get('description', '')}\n"
                    f"  Notes: {r.get('note_related_to', '')}"
                )
            sections.append("\n".join(cp_lines))

        # Discharge Instructions
        cursor.execute("""
            SELECT instruction, date FROM form_clinical_instructions
            WHERE pid = %s AND encounter = %s
        """, (pid, enc_id))
        rows = cursor.fetchall()
        if rows:
            di_lines = ["Discharge Instructions"]
            for r in rows:
                di_lines.append(f"- {r.get('instruction', '')}")
            sections.append("\n".join(di_lines))

    return "\n\n".join(sections)


def main():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    with open(SELECTION_FILE, "r") as f:
        selected = json.load(f)

    os.makedirs(EXTRACTS_DIR, exist_ok=True)

    used_pids = set()
    print(f"Matching {len(selected)} transcripts to OpenEMR patients...\n")

    for i, entry in enumerate(selected):
        condition = entry["detected_condition"]
        pid = find_patient_by_condition(cursor, condition, used_pids)

        if pid is None:
            print(f"  [{i+1:2d}/40] {entry['file_name']}: NO MATCH for '{condition}'")
            continue

        used_pids.add(pid)
        entry["matched_openemr_pid"] = pid

        # Extract EHR data
        ehr_text = extract_patient_data(cursor, pid)

        # Save extract file
        extract_path = os.path.join(EXTRACTS_DIR, f"{entry['file_name']}_openemr.txt")
        with open(extract_path, "w", encoding="utf-8") as f:
            f.write(f"OpenEMR Patient Extract for {entry['file_name']}\n")
            f.write(f"Matched Condition: {condition}\n")
            f.write(f"Patient ID: {pid}\n")
            f.write("=" * 60 + "\n\n")
            f.write(ehr_text)

        print(f"  [{i+1:2d}/40] {entry['file_name']} ({condition}) -> PID {pid} ({len(ehr_text)} chars)")

    cursor.close()
    conn.close()

    # Save updated selection file
    with open(SELECTION_FILE, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\nUpdated {SELECTION_FILE} with matched PIDs")
    print(f"Extracts saved to {EXTRACTS_DIR}/")

    # Summary
    matched = sum(1 for e in selected if e.get("matched_openemr_pid"))
    print(f"\nMatched: {matched}/40 transcripts to unique OpenEMR patients")


if __name__ == "__main__":
    main()
