"""
Automated CPT and ICD-10 Coding Service v4.0
Stage 1: keyword match + BioBERT similarity shortlist via ChromaDB
Stage 2: Modal Gemma4Coder.extract_entities, then match_codes (separate GPU calls)
Run: python3 coding_service.py
"""
import base64
import hashlib
import json, logging, os, time, datetime
from pathlib import Path
from typing import Optional, List

import chromadb
import modal
import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.retrieval import load_codes, build_or_load_collection, retrieve_candidates
from pipeline.coder_base import TruncatedResponseError

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

OPENEMR_SERVER_URL = os.getenv("OPENEMR_SERVER_URL", "https://localhost:9300")
OPENEMR_FHIR_BASE = f"{OPENEMR_SERVER_URL}/apis/default/fhir"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
SHORTLIST_TOP_K = 5
KEYWORD_MATCH_LIMIT = 20

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CPT_COLLECTION = "cpt-codes"
ICD10_COLLECTION = "icd10-codes"

MODAL_APP_NAME = "automated-cpt-icd-coding"
MODAL_CLASS_NAME = "Gemma4Coder"

BASE_DIR = Path(__file__).parent
CPT_JSON = BASE_DIR / "cpt_codes_final.json"
ICD10_JSON = BASE_DIR / "icd10_codes_final.json"








app = FastAPI(title="CPT and ICD-10 Coding Service", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class CodingRequest(BaseModel):
    note: str
    patient_id: str = ""
    encounter_id: str = ""


class CodingResponse(BaseModel):
    success: bool
    cpt_codes: List[str] = []
    icd10_codes: List[str] = []
    cpt_descriptions: dict = {}
    icd10_descriptions: dict = {}
    time_seconds: Optional[float] = None
    error: Optional[str] = None


class ValidationRequest(BaseModel):
    note: str
    code: str
    description: str


class ValidationResponse(BaseModel):
    success: bool
    code: str = ""
    supported: bool = False
    evidence: str = ""
    reason: str = ""
    error: Optional[str] = None






embed_model = None
cpt_col = None
icd10_col = None
gemma_coder = None


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
    """
    Validate SMART access token by calling OpenEMR FHIR metadata.
    If token is JWT, derive user id from claims. Else hash token.
    Mirrors the pattern used in rag-text-summarization/summarize.py.
    """
    token = extract_token_from_header(authorization)

    try:
        resp = requests.get(
            f"{OPENEMR_FHIR_BASE}/metadata",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
            verify=False,
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
        logger.info(f"Authenticated token. Derived user id from JWT: {user_id}")
        return {"user_id": user_id, "username": None, "token": token}

    token_fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    user_id = f"token-{token_fingerprint}"
    logger.info(f"Authenticated token. Derived user id from token hash: {user_id}")
    return {"user_id": user_id, "username": None, "token": token}


@app.on_event("startup")
async def startup():
    global embed_model, cpt_col, icd10_col, gemma_coder
    logger.info("Starting CPT and ICD-10 Coding Service v4.0")
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    cpt_codes = load_codes(CPT_JSON)
    icd10_codes = load_codes(ICD10_JSON)
    cpt_col = build_or_load_collection(client, embed_model, CPT_COLLECTION, cpt_codes)
    icd10_col = build_or_load_collection(client, embed_model, ICD10_COLLECTION, icd10_codes)
    gemma_coder_cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    gemma_coder = gemma_coder_cls()
    logger.info("Coding service ready")


@app.post("/code", response_model=CodingResponse)
async def code_note(request: CodingRequest, authorization: str = Header(None)):
    await validate_token_and_get_user(authorization)
    if not request.note.strip():
        raise HTTPException(status_code=400, detail="note is required")
    start = time.time()
    try:
        extraction_result = gemma_coder.extract_entities.remote(note_text=request.note)
        entities = extraction_result.get("entities", {})
        all_terms = entities.get("diagnoses", []) + entities.get("procedures", [])

        cpt_candidates = retrieve_candidates(embed_model, cpt_col, all_terms)
        icd10_candidates = retrieve_candidates(embed_model, icd10_col, all_terms)
        combined_candidates = {**cpt_candidates, **icd10_candidates}

        result = gemma_coder.match_codes.remote(
            note_text=request.note, entities=entities, candidates=combined_candidates
        )

        cpt_codes, icd10_codes = [], []
        cpt_descriptions, icd10_descriptions = {}, {}
        for m in result.get("matches", []):
            code = m.get("code")
            code_type = m.get("code_type", "")
            desc = m.get("description", "")
            if code_type == "CPT4":
                cpt_codes.append(code)
                cpt_descriptions[code] = desc
            elif code_type == "ICD10":
                icd10_codes.append(code)
                icd10_descriptions[code] = desc

        logger.info(f"CPT: {cpt_codes}, ICD-10: {icd10_codes}")
        return CodingResponse(
            success=True,
            cpt_codes=cpt_codes,
            icd10_codes=icd10_codes,
            cpt_descriptions=cpt_descriptions,
            icd10_descriptions=icd10_descriptions,
            time_seconds=time.time() - start,
        )
    except TruncatedResponseError as e:
        logger.error(f"Truncated response: {e}")
        return CodingResponse(success=False, error=f"Response truncated, please retry: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return CodingResponse(success=False, error=str(e))




@app.post("/validate", response_model=ValidationResponse)
async def validate_code(request: ValidationRequest, authorization: str = Header(None)):
    await validate_token_and_get_user(authorization)
    if not request.note.strip():
        raise HTTPException(status_code=400, detail="note is required")
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="code is required")
    try:
        result = gemma_coder.validate_code.remote(
            note_text=request.note, code=request.code, description=request.description
        )
        return ValidationResponse(
            success=True,
            code=result.get("code", request.code),
            supported=result.get("supported", False),
            evidence=result.get("evidence", ""),
            reason=result.get("reason", ""),
        )
    except TruncatedResponseError as e:
        logger.error(f"Truncated response: {e}")
        return ValidationResponse(success=False, error=f"Response truncated, please retry: {e}")
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return ValidationResponse(success=False, error=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cpt_codes": cpt_col.count() if cpt_col else 0,
        "icd10_codes": icd10_col.count() if icd10_col else 0,
        "llm_backend": "Modal Gemma4 26B",
        "embedding_model": EMBED_MODEL,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
