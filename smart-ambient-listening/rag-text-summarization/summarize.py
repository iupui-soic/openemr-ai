"""
SOAP Summarization Service

Authenticated service for RAG-based SOAP note generation.
Uses ChromaDB (local) + Groq API (external) with server-side credential retrieval.

SECURITY: Token-based authentication, server-side API key retrieval from OpenEMR.

Run: python summarize.py
"""
import base64
import json
import hashlib

import os
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import chromadb
import requests

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# OpenEMR server configuration (for token validation and API key retrieval)
OPENEMR_SERVER_URL = os.getenv("OPENEMR_SERVER_URL", "https://localhost:9300")
OPENEMR_FHIR_BASE = f"{OPENEMR_SERVER_URL}/apis/default/fhir"
OPENEMR_REST_BASE = f"{OPENEMR_SERVER_URL}/apis/default/api"


# Initialize FastAPI
app = FastAPI(
    title="SOAP Summarization Service",
    description="Authenticated RAG-based SOAP note generation using ChromaDB + Groq",
    version="6.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to ChromaDB (on startup)
try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = chroma_client.get_collection("medical-schemas")
    logger.info(f"✅ Connected to ChromaDB: {collection.count()} documents available")
except Exception as e:
    logger.error(f"❌ Failed to connect to ChromaDB: {e}")
    collection = None

# ============================================================================
# Request/Response Models
# ============================================================================

class SummarizeRequest(BaseModel):
    transcript_text: str
    openemr_text: str = ""
    patient_name: str = "Patient"

class SummarizeResponse(BaseModel):
    success: bool
    soap_note: Optional[str] = None
    detected_disease: Optional[str] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    total_time: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model: Optional[str] = None
    user_id: Optional[str] = None
    error: Optional[str] = None

class SaveSoapNoteRequest(BaseModel):
    patient_uuid: str
    encounter_uuid: str
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""

class SaveSoapNoteResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    soap_note_id: Optional[str] = None
    error: Optional[str] = None

# ============================================================================
# Authentication
# ============================================================================

def extract_token_from_header(authorization: str) -> str:
    """Extract bearer token from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    return parts[1]

def try_decode_jwt_subject(token: str) -> str | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        payload_b64 = parts[1]
        padding = "=" * (-len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8")
        payload = json.loads(payload_json)

        fhir_user = payload.get("fhirUser")
        sub = payload.get("sub")
        user_id = payload.get("user_id")
        username = payload.get("username")

        return fhir_user or sub or user_id or username
    except Exception:
        return None


async def validate_token_and_get_user(authorization: str) -> dict:
    token = extract_token_from_header(authorization)

    try:
        resp = requests.get(
            f"{OPENEMR_FHIR_BASE}/metadata",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
            verify=False
        )
        if resp.status_code != 200:
            logger.error(f"FHIR metadata token check failed: {resp.status_code} {resp.text}")
            raise HTTPException(status_code=401, detail="Invalid or expired access token")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to validate token against FHIR: {e}")
        raise HTTPException(status_code=503, detail="Failed to validate token with OpenEMR")

    extracted = try_decode_jwt_subject(token)
    if extracted:
        user_id = str(extracted).replace("/", "-")
        logger.info(f"✅ Authenticated token. Derived user id from JWT: {user_id}")
        return {"user_id": user_id, "username": None, "token": token}

    token_fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    user_id = f"token-{token_fingerprint}"
    logger.info(f"✅ Authenticated token. Derived user id from token hash: {user_id}")
    return {"user_id": user_id, "username": None, "token": token}


async def get_user_groq_key(user_token: str) -> str:
    try:
        response = requests.get(
            f"{OPENEMR_REST_BASE}/user/settings/custom",
            headers={"Authorization": f"Bearer {user_token}"},
            timeout=10,
            verify=False,
        )

        if response.status_code != 200:
            logger.error(f"Custom settings fetch failed: {response.status_code} {response.text}")
            raise HTTPException(status_code=500, detail="Failed to retrieve user settings from OpenEMR")

        data = response.json()

        # OpenEMR typically returns: {"data":[{field_id, field_label, field_name, field_value}, ...]}
        rows = data.get("data", [])
        if not isinstance(rows, list):
            rows = []

        # Build a lookup that supports multiple keys per row
        lookup = {}
        for row in rows:
            field_id = str(row.get("field_id", "")).strip()
            field_label = str(row.get("field_label", "")).strip()
            field_name = str(row.get("field_name", "")).strip()
            field_value = str(row.get("field_value", "")).strip()

            if field_id:
                lookup[field_id] = field_value
            if field_label:
                lookup[field_label.lower()] = field_value
            if field_name:
                lookup[field_name.lower()] = field_value

        # Try likely keys (case-insensitive)
        candidates = [
            "groq api token",
            "groq_api_token",
            "groq api key",
            "groq_api_key",
            "groqkey",
            "groq_token",
            # numeric fallback (only if you know it)
            "1",
        ]

        groq_key = None
        for k in candidates:
            groq_key = lookup.get(k.lower()) if not k.isdigit() else lookup.get(k)
            if groq_key:
                break

        if not groq_key or len(groq_key) < 20:
            raise HTTPException(
                status_code=400,
                detail="Groq API key not configured in OpenEMR. Add it in User Settings → Custom (GROQ API Token).",
            )

        logger.info(f"✅ Retrieved Groq key for user (prefix: {groq_key[:6]}...)")
        return groq_key

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get user settings: {e}")
        raise HTTPException(status_code=503, detail="Failed to retrieve user settings from OpenEMR")



async def get_user_groq_model(user_token: str) -> str:
    response = requests.get(
        f"{OPENEMR_REST_BASE}/user/settings/custom",
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=10,
        verify=False
    )
    if response.status_code != 200:
        return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    settings_data = response.json()
    settings = {}
    if 'data' in settings_data and isinstance(settings_data['data'], list):
        for setting in settings_data['data']:
            settings[setting['field_id']] = setting['field_value']

    model = (
            settings.get("groq_model") or
            settings.get("GROQ Model") or
            settings.get("groqModel") or
            os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    return model.strip()


# ============================================================================
# Helper Functions
# ============================================================================

def extract_disease(transcript: str, groq_api_key: str, groq_model: str) -> str:
    """Extract primary disease from transcript using Groq."""
    try:
        client = Groq(api_key=groq_api_key)

        disease_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a medical expert. Identify the primary disease or medical "
                    "condition discussed in a clinical conversation. "
                    "Respond with ONLY the disease name."
                ),
            },
            {
                "role": "user",
                "content": f"""
Read the following doctor-patient conversation and identify the PRIMARY medical condition.

Rules:
- Return ONLY the disease name
- If multiple conditions are mentioned, choose the most prominent
- If no disease is clearly stated, return "General"

Transcript:
{transcript[:2000]}

Primary Disease:"""
            }
        ]

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=disease_prompt,
            temperature=0.3,
            max_tokens=20,
        )

        raw_disease = response.choices[0].message.content
        detected_disease = raw_disease.strip() if raw_disease else "General"

        logger.info(f"✅ Detected Disease: {detected_disease}")
        return detected_disease

    except Exception as e:
        logger.error(f"Disease extraction failed: {e}")
        return "General"

def retrieve_schemas(disease: str, n_results: int = 2) -> str:
    """Retrieve relevant clinical schemas from ChromaDB."""
    if not collection:
        logger.warning("ChromaDB not available - skipping RAG")
        return "\n\n(No disease-specific schemas available - generating summary without RAG guidance)"

    try:
        results = collection.query(
            query_texts=[disease],
            n_results=n_results
        )

        if not results['documents'] or not results['documents'][0]:
            logger.warning(f"No schemas found for disease: {disease}")
            return "\n\n(No matching schemas found)"

        # Format retrieved schemas
        schema_context = ""
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            schema_context += f"\n\n=== SCHEMA {i+1} ({metadata.get('diseases', 'Unknown')}) ===\n{doc}"

        logger.info(f"✅ Retrieved {len(results['documents'][0])} schemas from ChromaDB")
        return schema_context

    except Exception as e:
        logger.error(f"Schema retrieval failed: {e}")
        return "\n\n(Schema retrieval failed)"

def generate_soap_note(
        transcript: str,
        openemr_text: str,
        patient_name: str,
        schemas: str,
        groq_api_key: str,
        groq_model: str
) -> dict:
    """Generate SOAP note using Groq with retrieved schemas."""
    try:
        client = Groq(api_key=groq_api_key)

        summary_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert medical scribe specialized in clinical documentation. "
                    "You generate SOAP notes by following a disease-specific SCHEMA TEMPLATE "
                    "and filling it with data extracted from the doctor-patient conversation "
                    "and the patient's electronic health record (EHR). "
                    "You write in professional medical prose — never JSON, markdown, or bullet points."
                )
            },
            {
                "role": "user",
                "content": f"""Generate a SOAP note for the following patient encounter.

PATIENT: {patient_name}

=== DATA SOURCE 1: TRANSCRIPT (Doctor-patient conversation) ===
{transcript}

=== DATA SOURCE 2: OPENEMR EHR EXTRACT (Existing medical record) ===
{openemr_text if openemr_text else "No OpenEMR data available."}

=== SCHEMA TEMPLATE (Disease-specific clinical note structure) ===
The following schema defines the STRUCTURE and FIELDS your SOAP note must cover.
Treat each field in the schema as a checklist item. For every field present in the schema,
look for matching data in the TRANSCRIPT and EHR EXTRACT and include it in the appropriate
SOAP section. The schema tells you WHAT to cover; the data sources tell you WHAT TO WRITE.
{schemas}

=== INSTRUCTIONS ===

STEP 1 — MAP SCHEMA FIELDS TO DATA:
For each section and field in the SCHEMA TEMPLATE above:
  a) First check the TRANSCRIPT for relevant information (current visit data)
  b) Then check the OPENEMR EHR EXTRACT for relevant information (historical data)
  c) Include the information in the appropriate SOAP section below

STEP 2 — WRITE THE SOAP NOTE with these exact four section headers:

Subjective:
  Fill using schema fields: chief_complaint, hpi (symptom_clusters, timeline_events,
  associated_symptoms), review_of_systems, allergies, past_medical_history,
  surgical_history, social_history, family_history, current_medications.
  Source: TRANSCRIPT for current complaints and symptoms. EHR for medical history,
  medications, allergies, and surgical history.

Objective:
  Fill using schema fields: vital_signs, physical_exam, labs, imaging, procedures.
  Source: TRANSCRIPT for exam findings discussed during the visit. EHR for vitals,
  lab results, and imaging data.

Assessment:
  Synthesize findings into clinical assessment. List diagnoses or differential diagnoses
  with reasoning. Reference schema disease context for condition-specific considerations.
  Source: Combine TRANSCRIPT clinical impression with EHR condition history.

Plan:
  Fill using schema fields: medications (new/changed), follow_up, referrals,
  patient_education, disposition.
  Source: TRANSCRIPT for treatment decisions made during the visit. EHR for
  ongoing medication management and follow-up needs.

STEP 3 — FORMATTING RULES:
- Write in narrative prose with complete sentences and paragraphs
- Use plain text only — NO markdown (no #, **, *, -, ```, etc.)
- Do NOT output JSON, XML, or structured data
- If a schema field has no matching data in either source, skip it silently (do NOT write "no information available" for every missing field — only mention notable gaps)
- If TRANSCRIPT and EHR conflict on current status, trust the TRANSCRIPT
- Do NOT hallucinate or invent information not present in the data sources
- Do NOT include meta-commentary, explanations, or references to this prompt
- Write as if this will be placed directly into the patient's medical chart

Generate the SOAP note now:"""
            }
        ]

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=summary_messages,
            temperature=0.3,
            max_tokens=4096,
        )

        soap_note = response.choices[0].message.content.strip()

        # Calculate tokens (rough estimate)
        input_tokens = len(summary_messages[0]["content"] + summary_messages[1]["content"]) // 4
        output_tokens = len(soap_note) // 4

        logger.info(f"✅ SOAP note generated: {len(soap_note)} chars")

        return {
            "success": True,
            "soap_note": soap_note,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": GROQ_MODEL
        }

    except Exception as e:
        logger.error(f"SOAP generation failed: {e}")
        return {
            "success": False,
            "error": f"SOAP generation failed: {str(e)}"
        }


def save_soap_note_to_openemr(
        patient_uuid: str,
        encounter_uuid: str,
        subjective: str,
        objective: str,
        assessment: str,
        plan: str,
        user_token: str
) -> dict:
    """
    Save a SOAP note to an OpenEMR encounter via direct database insert.

    The REST API has a bug in OpenEMR 8.0.1-dev where it stores incorrect
    pid and encounter values. This function bypasses the API and inserts
    directly into the MySQL database with correct values.
    """
    import pymysql

    # Database connection settings (same as OpenEMR's sqlconf.php)
    DB_HOST = os.getenv("OPENEMR_DB_HOST", "localhost")
    DB_PORT = int(os.getenv("OPENEMR_DB_PORT", "3306"))
    DB_USER = os.getenv("OPENEMR_DB_USER", "openemr")
    DB_PASS = os.getenv("OPENEMR_DB_PASS", "openemr")
    DB_NAME = os.getenv("OPENEMR_DB_NAME", "openemr")

    logger.info(f"💾 Saving SOAP note via direct DB insert")
    logger.info(f"   Patient UUID: {patient_uuid}")
    logger.info(f"   Encounter UUID: {encounter_uuid}")

    conn = None
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

        with conn.cursor() as cursor:
            # Step 1: Resolve patient UUID to pid
            cursor.execute(
                "SELECT pid, fname, lname FROM patient_data WHERE uuid = UNHEX(REPLACE(%s, '-', '')) LIMIT 1",
                (patient_uuid,)
            )
            patient_row = cursor.fetchone()
            if not patient_row:
                return {"success": False, "error": f"Patient UUID not found: {patient_uuid}"}

            pid = patient_row["pid"]
            patient_name = f"{patient_row.get('fname', '')} {patient_row.get('lname', '')}".strip()
            logger.info(f"   Resolved patient: {patient_name} (pid={pid})")

            # Step 2: Resolve encounter UUID to encounter number
            cursor.execute(
                "SELECT encounter, date FROM form_encounter WHERE uuid = UNHEX(REPLACE(%s, '-', '')) LIMIT 1",
                (encounter_uuid,)
            )
            encounter_row = cursor.fetchone()
            if not encounter_row:
                return {"success": False, "error": f"Encounter UUID not found: {encounter_uuid}"}

            encounter_id = encounter_row["encounter"]
            logger.info(f"   Resolved encounter: {encounter_id} (date={encounter_row['date']})")

            # Step 3: Resolve the username from the JWT token
            username = _resolve_username_from_token(user_token, cursor)
            logger.info(f"   Resolved user: {username}")

            # Step 4: Insert into form_soap
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                """INSERT INTO form_soap
                   (date, pid, user, groupname, authorized, activity, subjective, objective, assessment, plan)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (now, pid, username, "Default", 0, 1, subjective, objective, assessment, plan)
            )
            soap_id = cursor.lastrowid
            logger.info(f"   Inserted form_soap id={soap_id}")

            # Step 5: Register in forms table (links soap note to encounter)
            cursor.execute(
                """INSERT INTO forms
                   (date, encounter, form_name, form_id, pid, user, groupname, authorized, deleted, formdir)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (now, encounter_id, "SOAP", soap_id, pid, username, "Default", 1, 0, "soap")
            )
            forms_id = cursor.lastrowid
            logger.info(f"   Inserted forms id={forms_id}")

        conn.commit()
        logger.info(f"✅ SOAP note saved successfully (soap_id={soap_id}, forms_id={forms_id})")

        return {
            "success": True,
            "message": f"SOAP note saved to encounter {encounter_id} for {patient_name}",
            "soap_note_id": str(soap_id),
        }

    except pymysql.Error as e:
        logger.error(f"❌ Database error saving SOAP note: {e}")
        if conn:
            conn.rollback()
        return {
            "success": False,
            "error": f"Database error: {str(e)}",
        }
    except Exception as e:
        logger.error(f"❌ Error saving SOAP note: {e}")
        if conn:
            conn.rollback()
        return {
            "success": False,
            "error": f"Failed to save SOAP note: {str(e)}",
        }
    finally:
        if conn:
            conn.close()


def _resolve_username_from_token(token: str, cursor) -> str:
    """
    Resolve the OpenEMR username from the JWT token.
    Falls back to 'admin' if resolution fails.
    """
    try:
        # Decode JWT to get user identity
        parts = token.split(".")
        if len(parts) == 3:
            payload_b64 = parts[1]
            padding = "=" * (-len(payload_b64) % 4)
            payload_json = base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8")
            payload = json.loads(payload_json)

            # Try fhirUser first (e.g., "Practitioner/9d033ade-...")
            fhir_user = payload.get("fhirUser", "")
            if fhir_user and "/" in fhir_user:
                user_uuid = fhir_user.split("/")[-1]
                cursor.execute(
                    "SELECT username FROM users_secure WHERE id = (SELECT id FROM users WHERE uuid = UNHEX(REPLACE(%s, '-', '')) LIMIT 1) LIMIT 1",
                    (user_uuid,)
                )
                row = cursor.fetchone()
                if row and row.get("username"):
                    return row["username"]

            # Try sub claim
            sub = payload.get("sub")
            if sub:
                cursor.execute(
                    "SELECT username FROM users_secure WHERE id = (SELECT id FROM users WHERE uuid = UNHEX(REPLACE(%s, '-', '')) LIMIT 1) LIMIT 1",
                    (sub,)
                )
                row = cursor.fetchone()
                if row and row.get("username"):
                    return row["username"]

    except Exception as e:
        logger.warning(f"Could not resolve username from token: {e}")

    return "admin"

# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(
        request: SummarizeRequest,
        authorization: str = Header(None)
):
    """
    Generate SOAP note from transcript + FHIR data using RAG.

    Headers:
        Authorization: Bearer <SMART_access_token> (REQUIRED)

    Request body:
        transcript_text: Doctor-patient conversation (REQUIRED)
        openemr_text: FHIR data from OpenEMR (optional)
        patient_name: Patient name for note header (optional)

    Flow:
    1. Validate token and get user identity
    2. Retrieve user's Groq API key from OpenEMR (server-side)
    3. Extract disease from transcript (Groq)
    4. Retrieve relevant schemas (ChromaDB)
    5. Generate SOAP note (Groq with schemas)
    """
    # Authenticate user
    user_info = await validate_token_and_get_user(authorization)
    user_id = user_info["user_id"]

    logger.info(f"📥 Summarization request from user {user_id}")
    logger.info(f"   Transcript length: {len(request.transcript_text)} chars")
    logger.info(f"   OpenEMR data length: {len(request.openemr_text)} chars")

    start_total = time.time()

    # Validate inputs
    if not request.transcript_text or len(request.transcript_text.strip()) < 10:
        logger.warning("Transcript too short")
        return SummarizeResponse(
            success=False,
            error="Transcription too short (minimum 10 characters required)",
            user_id=user_id
        )

    try:
        # Get user's Groq API key (server-side, never exposed to client)
        groq_api_key = await get_user_groq_key(user_info["token"])
        groq_model = await get_user_groq_model(user_info["token"])


        # Step 1: Extract disease
        start_retrieval = time.time()
        detected_disease = extract_disease(request.transcript_text, groq_api_key, groq_model)

        # Step 2: Retrieve schemas
        schemas = retrieve_schemas(detected_disease, n_results=2)
        retrieval_time = time.time() - start_retrieval

        # Step 3: Generate SOAP note
        start_generation = time.time()
        result = generate_soap_note(
            request.transcript_text,
            request.openemr_text,
            request.patient_name,
            schemas,
            groq_api_key,
            groq_model
        )
        generation_time = time.time() - start_generation

        total_time = time.time() - start_total

        logger.info(f"✅ Complete for user {user_id}! Total time: {total_time:.2f}s")

        return SummarizeResponse(
            success=result.get("success", False),
            soap_note=result.get("soap_note"),
            detected_disease=detected_disease,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens"),
            model=result.get("model"),
            user_id=user_id,
            error=result.get("error")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error for user {user_id}: {str(e)}")
        return SummarizeResponse(
            success=False,
            error=f"Summarization failed: {str(e)}",
            user_id=user_id
        )


@app.post("/save-soap-note", response_model=SaveSoapNoteResponse)
async def save_soap_note(
        request: SaveSoapNoteRequest,
        authorization: str = Header(None)
):
    """
    Save a SOAP note to an OpenEMR encounter.

    This endpoint proxies the save request to OpenEMR's REST API server-side,
    bypassing SMART scope limitations on the frontend.

    Headers:
        Authorization: Bearer <SMART_access_token> (REQUIRED)

    Request body:
        patient_uuid: Patient UUID (REQUIRED)
        encounter_uuid: Encounter UUID/ID (REQUIRED)
        subjective: Subjective section text
        objective: Objective section text
        assessment: Assessment section text
        plan: Plan section text
    """
    # Authenticate user
    user_info = await validate_token_and_get_user(authorization)
    user_id = user_info["user_id"]

    logger.info(f"💾 Save SOAP note request from user {user_id}")
    logger.info(f"   Patient: {request.patient_uuid}")
    logger.info(f"   Encounter: {request.encounter_uuid}")

    # Validate inputs
    if not request.patient_uuid or not request.encounter_uuid:
        raise HTTPException(status_code=400, detail="patient_uuid and encounter_uuid are required")

    # Save to OpenEMR using the user's token
    result = save_soap_note_to_openemr(
        patient_uuid=request.patient_uuid,
        encounter_uuid=request.encounter_uuid,
        subjective=request.subjective,
        objective=request.objective,
        assessment=request.assessment,
        plan=request.plan,
        user_token=user_info["token"],
    )

    if not result["success"]:
        logger.error(f"❌ Save failed for user {user_id}: {result.get('error')}")
        return SaveSoapNoteResponse(
            success=False,
            error=result.get("error", "Unknown error saving SOAP note")
        )

    logger.info(f"✅ SOAP note saved for user {user_id}")
    return SaveSoapNoteResponse(
        success=True,
        message=result.get("message", "SOAP note saved successfully"),
        soap_note_id=result.get("soap_note_id")
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    chroma_status = "connected" if collection else "disconnected"
    chroma_count = collection.count() if collection else 0

    return {
        "status": "healthy",
        "service": "soap-summarization-authenticated",
        "version": "6.1.0",
        "authentication": "token-based",
        "chromadb_status": chroma_status,
        "chromadb_documents": chroma_count,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "SOAP Summarization Service",
        "description": "Authenticated RAG-based SOAP note generation using ChromaDB + Groq",
        "version": "6.1.0",
        "architecture": "ChromaDB (local) + Groq API (external)",
        "authentication": "SMART OAuth2 token required",
        "security": "Server-side API key retrieval",
        "endpoints": {
            "/summarize": "POST - Generate SOAP note (requires Authorization header)",
            "/save-soap-note": "POST - Save SOAP note to encounter (requires Authorization header)",
            "/health": "GET - Health check",
        }
    }

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting SOAP summarization service on 127.0.0.1:{PORT}")
    logger.info(f"ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")
    logger.info(f"Model: {GROQ_MODEL}")
    logger.info("✅ Token-based authentication enabled")
    logger.info("✅ Server-side API key retrieval enabled")
    logger.info("✅ Save SOAP note endpoint enabled")
    uvicorn.run(app, host="127.0.0.1", port=PORT)  # ✅ Bind to localhost only