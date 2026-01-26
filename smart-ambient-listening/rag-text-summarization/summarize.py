"""
SOAP Summarization Service
Uses ChromaDB (local) + Groq API (external) for RAG-based summarization
No Modal needed!

Run: nohup python3 summarize.py > summarize.log 2>&1 &
"""

import os
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
GROQ_MODEL = "openai/gpt-oss-20b"

# Initialize FastAPI
app = FastAPI(
    title="SOAP Summarization Service",
    description="RAG-based SOAP note generation using ChromaDB + Groq",
    version="5.0.0"
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
    groq_api_key: str  # REQUIRED - from OpenEMR user settings


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
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def extract_disease(transcript: str, groq_api_key: str) -> str:
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


def generate_soap_note(transcript: str, openemr_text: str, schemas: str, groq_api_key: str) -> dict:
    """Generate SOAP note using Groq with retrieved schemas."""
    try:
        client = Groq(api_key=groq_api_key)

        summary_messages = [
            {
                "role": "system",
                "content": "You are an expert medical scribe specialized in clinical documentation. Generate comprehensive SOAP-format medical summaries."
            },
            {
                "role": "user",
                "content": f"""Generate a comprehensive medical summary in SOAP format from the following data:

TRANSCRIPT (Doctor-patient conversation):
{transcript}

OPENEMR EXTRACT (Electronic health record):
{openemr_text if openemr_text else "No OpenEMR data available."}

SCHEMA GUIDE (Reference clinical note structure):
{schemas}

OUTPUT FORMAT REQUIREMENTS:
- Generate a NARRATIVE TEXT document, NOT JSON or structured data
- Use clear section headers (e.g., "Patient Information", "Chief Complaint", "History of Present Illness")
- Write in complete sentences and paragraphs
- Use professional medical documentation prose style
- Format similar to a hospital discharge summary
- Do NOT use markdown formatting (no #, ##, **, *, -, ```, etc.)

INSTRUCTIONS:
1. Use the SCHEMA GUIDE as a reference for which sections to include
2. Extract relevant information from the TRANSCRIPT and OPENEMR EXTRACT
3. Write in narrative prose with proper paragraphs
4. If information for a section is missing, write "No information available."
5. If TRANSCRIPT and OPENEMR conflict, trust the TRANSCRIPT for current status
6. Do NOT include any meta-commentary, explanations, or references to this prompt
7. Do NOT output JSON, XML, or any structured data format
8. Do NOT hallucinate or invent information not present in the inputs
9. Do NOT use any markdown syntax - plain text only

Generate the medical summary now in narrative prose format:"""
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


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Generate SOAP note from transcript + FHIR data using RAG.

    Flow:
    1. Extract disease from transcript (Groq)
    2. Retrieve relevant schemas (ChromaDB)
    3. Generate SOAP note (Groq with schemas)

    Requires:
        - transcript_text: Doctor-patient conversation
        - openemr_text: FHIR data from OpenEMR (optional)
        - groq_api_key: User's Groq API key
    """
    logger.info("📥 Summarization request received")
    logger.info(f"   Transcript length: {len(request.transcript_text)} chars")
    logger.info(f"   OpenEMR data length: {len(request.openemr_text)} chars")

    start_total = time.time()

    # Validate inputs
    if not request.transcript_text or len(request.transcript_text.strip()) < 10:
        logger.warning("Transcript too short")
        return SummarizeResponse(
            success=False,
            error="Transcription too short (minimum 10 characters required)"
        )

    if not request.groq_api_key:
        logger.warning("Missing Groq API key")
        return SummarizeResponse(
            success=False,
            error="Groq API key is required. Please configure it in OpenEMR user settings."
        )

    try:
        # Step 1: Extract disease
        start_retrieval = time.time()
        detected_disease = extract_disease(request.transcript_text, request.groq_api_key)

        # Step 2: Retrieve schemas
        schemas = retrieve_schemas(detected_disease, n_results=2)
        retrieval_time = time.time() - start_retrieval

        # Step 3: Generate SOAP note
        start_generation = time.time()
        result = generate_soap_note(
            request.transcript_text,
            request.openemr_text,
            schemas,
            request.groq_api_key
        )
        generation_time = time.time() - start_generation

        total_time = time.time() - start_total

        logger.info(f"✅ Complete! Total time: {total_time:.2f}s")

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
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return SummarizeResponse(
            success=False,
            error=f"Summarization failed: {str(e)}"
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    chroma_status = "connected" if collection else "disconnected"
    chroma_count = collection.count() if collection else 0

    return {
        "status": "healthy",
        "service": "soap-summarization-chromadb-groq",
        "chromadb_status": chroma_status,
        "chromadb_documents": chroma_count,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "SOAP Summarization Service",
        "description": "RAG-based SOAP note generation using ChromaDB + Groq (no Modal)",
        "version": "5.0.0",
        "architecture": "ChromaDB (local) + Groq API (external)",
        "endpoints": {
            "/summarize": "POST - Generate SOAP note (requires transcript + groq_api_key)",
            "/health": "GET - Health check",
        }
    }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting SOAP summarization service on {HOST}:{PORT}")
    logger.info(f"ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")
    logger.info(f"Model: {GROQ_MODEL}")
    uvicorn.run(app, host=HOST, port=PORT)