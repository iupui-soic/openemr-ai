"""
Automated CPT and ICD-10 Coding Service v3.0
Stage 1: BGE embed_match shortlist via ChromaDB
Stage 2: Modal Gemma4 26B LLM selection
Run: python3 coding_service.py
"""
import json, logging, os, re, time
from pathlib import Path
from typing import Optional, List
import chromadb
import requests
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
QUERY_PREFIX = "Represent this procedure description for retrieval: "
SHORTLIST_TOP_K = 50
CHUNK_WORDS = 32
CHUNK_STRIDE = 16
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CPT_COLLECTION = "cpt-codes"
ICD10_COLLECTION = "icd10-codes"
MODAL_CODE_URL = os.getenv("MODAL_CODE_URL", "https://chaitrasree-20--gemma4-cpt-coder-gemma4coder-code.modal.run")
BASE_DIR = Path(__file__).parent
CPT_JSON = BASE_DIR / "cpt_codes_final.json"
ICD10_JSON = BASE_DIR / "icd10_codes_final.json"

def load_codes(path):
    with open(path) as f:
        return json.load(f)

def chunk_note(text, window=CHUNK_WORDS, stride=CHUNK_STRIDE):
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return []
    chunks, i = [], 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+window])
        if chunk:
            chunks.append(chunk)
        if i + window >= len(tokens):
            break
        i += stride
    return chunks[:1024]

def build_or_load_collection(client, model, name, codes):
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        col = client.get_collection(name)
        if col.count() == len(codes):
            logger.info(f"✅ Loaded '{name}' from ChromaDB ({col.count()} codes)")
            return col
        logger.info(f"Rebuilding '{name}'...")
        client.delete_collection(name)
    logger.info(f"Building '{name}' for {len(codes)} codes...")
    col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    code_list = list(codes.keys())
    descs = [codes[c] for c in code_list]
    BATCH = 512
    for i in range(0, len(code_list), BATCH):
        bc = code_list[i:i+BATCH]
        bd = descs[i:i+BATCH]
        embs = model.encode([QUERY_PREFIX+d for d in bd], batch_size=256, normalize_embeddings=True, show_progress_bar=False).tolist()
        col.upsert(ids=bc, embeddings=embs, metadatas=[{"description": d} for d in bd], documents=bd)
        logger.info(f"  {min(i+BATCH, len(code_list))}/{len(code_list)}")
    logger.info(f"✅ Built '{name}'")
    return col

def shortlist(model, col, note, top_k=SHORTLIST_TOP_K):
    chunks = chunk_note(note)
    if not chunks:
        return {}
    embs = model.encode(chunks, batch_size=64, normalize_embeddings=True).tolist()
    scores = {}
    for emb in embs:
        res = col.query(query_embeddings=[emb], n_results=min(top_k, col.count()), include=["metadatas","distances"])
        for code, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
            sim = 1 - dist
            if code not in scores or sim > scores[code][0]:
                scores[code] = (sim, meta["description"])
    top = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
    return {c: scores[c][1] for c, _ in top}

def parse_codes(raw, label_space):
    label_set = set(label_space)
    m = re.search(r'\{[^{}]*"codes"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return [str(c).strip() for c in obj.get("codes",[]) if str(c).strip() in label_set]
        except:
            pass
    return []

def select_via_modal(note, candidates, system=""):
    try:
        resp = requests.post(MODAL_CODE_URL, json={"text": note, "descriptions": candidates}, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        codes = data.get("codes", [])
        raw = data.get("raw", "")
        label_space = list(candidates.keys())
        valid = [c for c in codes if c in set(label_space)]
        if not valid and raw:
            valid = parse_codes(raw, label_space)
        return valid
    except Exception as e:
        logger.error(f"Modal error: {e}")
        return []


def select_via_groq(note, candidates, system=''):
    from groq import Groq
    import os
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        return []
    try:
        client = Groq(api_key=groq_api_key)
        code_block = chr(10).join(f'- {c}: {d}' for c, d in candidates.items())
        user_msg = f'Candidate codes:{chr(10)}{code_block}{chr(10)}{chr(10)}Clinical note:{chr(10)}<<<{chr(10)}{note}{chr(10)}>>>{chr(10)}{chr(10)}Return JSON only.'
        resp = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user_msg}],
            max_tokens=512, temperature=0.0
        )
        raw = resp.choices[0].message.content or ''
        import re, json
        m = re.search(r'\{[^{}]*"codes"[^{}]*\}', raw, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            label_set = set(candidates.keys())
            return [str(c).strip() for c in obj.get('codes',[]) if str(c).strip() in label_set]
        return []
    except Exception as e:
        print(f'Groq error: {e}')
        return []
app = FastAPI(title="CPT and ICD-10 Coding Service", version="3.0.0")
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

embed_model = None
cpt_col = None
icd10_col = None

@app.on_event("startup")
async def startup():
    global embed_model, cpt_col, icd10_col
    logger.info("Starting CPT and ICD-10 Coding Service v3.0")
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    cpt_codes = load_codes(CPT_JSON)
    icd10_codes = load_codes(ICD10_JSON)
    cpt_col = build_or_load_collection(client, embed_model, CPT_COLLECTION, cpt_codes)
    icd10_col = build_or_load_collection(client, embed_model, ICD10_COLLECTION, icd10_codes)
    logger.info("✅ Coding service ready")

@app.post("/code", response_model=CodingResponse)
async def code_note(request: CodingRequest):
    if not request.note.strip():
        raise HTTPException(status_code=400, detail="note is required")
    start = time.time()
    try:
        cpt_candidates = shortlist(embed_model, cpt_col, request.note)
        icd10_candidates = shortlist(embed_model, icd10_col, request.note)
        logger.info(f"Shortlisted {len(cpt_candidates)} CPT, {len(icd10_candidates)} ICD-10")
        cpt_codes = select_via_groq(request.note, cpt_candidates, "You are a certified medical coder. Given a clinical note and candidate CPT codes, identify which apply. Respond ONLY with JSON: {\"codes\": [\"<code>\", ...]}. Empty list if none apply.")
        icd10_codes = select_via_groq(request.note, icd10_candidates, "You are a certified medical coder. Given a clinical note and candidate ICD-10 codes, identify which apply. Respond ONLY with JSON: {\"codes\": [\"<code>\", ...]}. Empty list if none apply.")
        logger.info(f"CPT: {cpt_codes}, ICD-10: {icd10_codes}")
        return CodingResponse(success=True, cpt_codes=cpt_codes, icd10_codes=icd10_codes,
            cpt_descriptions={c: cpt_candidates[c] for c in cpt_codes if c in cpt_candidates},
            icd10_descriptions={c: icd10_candidates[c] for c in icd10_codes if c in icd10_candidates},
            time_seconds=time.time()-start)
    except Exception as e:
        logger.error(f"Error: {e}")
        return CodingResponse(success=False, error=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "cpt_codes": cpt_col.count() if cpt_col else 0,
            "icd10_codes": icd10_col.count() if icd10_col else 0,
            "llm_backend": "Modal Gemma4 26B"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
# Groq fallback

def select_via_groq(note, candidates, system=""):
    from groq import Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY not set")
        return []
    try:
        client = Groq(api_key=groq_api_key)
        code_block = "\n".join(f"- {c}: {d}" for c, d in candidates.items())
        user_msg = f"Candidate codes:\n{code_block}\n\nClinical note:\n<<<\n{note}\n>>>\n\nReturn JSON only."
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            max_tokens=512, temperature=0.0
        )
        raw = resp.choices[0].message.content or ""
        return parse_codes(raw, list(candidates.keys()))
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return []
