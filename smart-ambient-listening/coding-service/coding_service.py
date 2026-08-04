"""
Automated CPT and ICD-10 Coding Service v4.0
Stage 1: keyword match + BioBERT similarity shortlist via ChromaDB
Stage 2: Modal Gemma4Coder.generate_codes (extraction + matching)
Run: python3 coding_service.py
"""
import json, logging, os, time, datetime
from pathlib import Path
from typing import Optional, List

import chromadb
import modal
import pymysql
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

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


def load_codes(path):
    with open(path) as f:
        return json.load(f)


def build_or_load_collection(client, model, name, codes):
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        col = client.get_collection(name)
        if col.count() == len(codes):
            logger.info(f"Loaded '{name}' from ChromaDB ({col.count()} codes)")
            return col
        logger.info(f"Rebuilding '{name}'...")
        client.delete_collection(name)
    logger.info(f"Building '{name}' for {len(codes)} codes...")
    col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    code_list = list(codes.keys())
    descs = [codes[c] for c in code_list]
    BATCH = 256
    for i in range(0, len(code_list), BATCH):
        bc = code_list[i:i + BATCH]
        bd = descs[i:i + BATCH]
        embs = model.encode(bd, batch_size=BATCH, normalize_embeddings=True, show_progress_bar=False).tolist()
        col.upsert(ids=bc, embeddings=embs, metadatas=[{"description": d, "code": c} for c, d in zip(bc, bd)], documents=bd)
        logger.info(f"  {min(i + BATCH, len(code_list))}/{len(code_list)}")
    logger.info(f"Built '{name}'")
    return col


def retrieve_candidates(model, col, terms: List[str], k: int = SHORTLIST_TOP_K) -> dict:
    """
    Two-pass retrieval:
      1. Keyword substring match (ChromaDB-side $contains filter, not a full
         table fetch -- avoids "too many SQL variables" on large collections).
      2. Embedding similarity search -- fills in candidates for terms that
         don't literally appear in any description (e.g. "headache" vs "R51").
    """
    if not terms:
        return {}

    candidates: dict = {}

    for term in terms:
        try:
            keyword_res = col.get(
                where_document={"$contains": term.lower()},
                include=["metadatas"],
                limit=KEYWORD_MATCH_LIMIT,
            )
        except Exception as e:
            logger.warning(f"Keyword match failed for '{term}': {e}")
            keyword_res = {"metadatas": []}
        for meta in keyword_res["metadatas"]:
            code = meta.get("code")
            if code and code not in candidates:
                candidates[code] = meta.get("description", "")

    for term in terms:
        query_emb = model.encode([term], normalize_embeddings=True).tolist()
        res = col.query(query_embeddings=query_emb, n_results=k, include=["metadatas"])
        for meta in res["metadatas"][0]:
            code = meta.get("code")
            if code and code not in candidates:
                candidates[code] = meta.get("description", "")

    return candidates


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


def resolve_patient_and_encounter(cursor, patient_uuid: str, encounter_uuid: str):
    """
    Resolve external UUIDs to OpenEMR's internal pid/encounter integers,
    and fetch the encounter date for the same-day-visit check.
    Returns (pid, encounter, encounter_date) or (None, None, None) if not found.

    MariaDB has no UUID_TO_BIN/BIN_TO_UUID (those are MySQL 8.0+ only), so we
    convert manually with UNHEX() against the dash-stripped UUID string.
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


class CodeGemmaRequest(BaseModel):
    note: str
    patient_uuid: str
    encounter_uuid: str


class CodeGemmaResponse(BaseModel):
    success: bool
    cpt_codes: List[str] = []
    icd10_codes: List[str] = []
    cpt_descriptions: dict = {}
    icd10_descriptions: dict = {}
    inserted: int = 0
    billing_ids: List[int] = []
    error: Optional[str] = None


embed_model = None
cpt_col = None
icd10_col = None
gemma_coder = None


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
async def code_note(request: CodingRequest):
    if not request.note.strip():
        raise HTTPException(status_code=400, detail="note is required")
    start = time.time()
    try:
        result = gemma_coder.generate_codes.remote(note_text=request.note, candidates={})
        entities = result.get("entities", {})
        all_terms = entities.get("diagnoses", []) + entities.get("procedures", [])

        cpt_candidates = retrieve_candidates(embed_model, cpt_col, all_terms)
        icd10_candidates = retrieve_candidates(embed_model, icd10_col, all_terms)
        combined_candidates = {**cpt_candidates, **icd10_candidates}

        result = gemma_coder.generate_codes.remote(note_text=request.note, candidates=combined_candidates)

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
    except Exception as e:
        logger.error(f"Error: {e}")
        return CodingResponse(success=False, error=str(e))


@app.post("/code-gemma", response_model=CodeGemmaResponse)
async def code_gemma(request: CodeGemmaRequest):
    """
    Full pipeline: extract -> retrieve -> match -> insert into the Fee
    Sheet. Codes are only inserted for TODAY's encounter -- if the
    encounter date is not today, the pipeline refuses to insert and
    returns an explanatory error, since AI-suggested codes only make
    sense for the visit that's actually happening right now.
    """
    if not request.note.strip():
        raise HTTPException(status_code=400, detail="note is required")

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            pid, encounter, encounter_date = resolve_patient_and_encounter(
                cursor, request.patient_uuid, request.encounter_uuid
            )
            if pid is None:
                return CodeGemmaResponse(success=False, error="Patient not found")
            if encounter is None:
                return CodeGemmaResponse(success=False, error="Encounter not found")

            today = datetime.date.today()
            encounter_day = encounter_date.date() if hasattr(encounter_date, "date") else encounter_date
            if encounter_day != today:
                return CodeGemmaResponse(
                    success=False,
                    error=(
                        f"Encounter date ({encounter_day}) is not today ({today}). "
                        "AI-suggested codes are only inserted for same-day office visits."
                    ),
                )

        result = gemma_coder.generate_codes.remote(note_text=request.note, candidates={})
        entities = result.get("entities", {})
        all_terms = entities.get("diagnoses", []) + entities.get("procedures", [])

        cpt_candidates = retrieve_candidates(embed_model, cpt_col, all_terms)
        icd10_candidates = retrieve_candidates(embed_model, icd10_col, all_terms)
        combined_candidates = {**cpt_candidates, **icd10_candidates}

        result = gemma_coder.generate_codes.remote(note_text=request.note, candidates=combined_candidates)

        cpt_codes, icd10_codes = [], []
        cpt_descriptions, icd10_descriptions = {}, {}
        billing_ids = []

        seen_pairs = set()
        deduped_matches = []
        for m in result.get("matches", []):
            pair = (m.get("code_type", ""), m.get("code", ""))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                deduped_matches.append(m)

        with conn.cursor() as cursor:
            for m in deduped_matches:
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

                if code_type == "CPT4":
                    cpt_codes.append(code)
                    cpt_descriptions[code] = desc
                else:
                    icd10_codes.append(code)
                    icd10_descriptions[code] = desc

            conn.commit()

        return CodeGemmaResponse(
            success=True,
            cpt_codes=cpt_codes,
            icd10_codes=icd10_codes,
            cpt_descriptions=cpt_descriptions,
            icd10_descriptions=icd10_descriptions,
            inserted=len(billing_ids),
            billing_ids=billing_ids,
        )
    except Exception as e:
        logger.error(f"code-gemma error: {e}")
        return CodeGemmaResponse(success=False, error=str(e))
    finally:
        conn.close()


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
