"""
Transcription Service API

FastAPI service that receives audio from the frontend and calls
the Modal-hosted Parakeet ASR model for transcription.

Run with: python transcribe.py
"""

import os
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import modal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "parakeet-asr")

app = FastAPI(
    title="Transcription Service",
    description="API gateway for Modal-hosted Parakeet ASR",
    version="1.0.0"
)

# CORS - allow frontend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modal transcriber reference (lazy loaded)
_transcriber = None


def get_transcriber():
    """Get or create Modal transcriber reference."""
    global _transcriber
    if _transcriber is None:
        logger.info(f"Connecting to Modal app: {MODAL_APP_NAME}")
        ParakeetTranscriber = modal.Cls.from_name(MODAL_APP_NAME, "ParakeetTranscriber")
        _transcriber = ParakeetTranscriber()
    return _transcriber
# --- ADD THIS NEW ENDPOINT ---
@app.get("/warmup")
async def warmup_model():
    logger.info("Warmup request received, pinging Modal container.")
    try:
        transcriber = get_transcriber()
        # Instantiate the class
        transcriber.wakeup.remote()      # Call the dummy wakeup method
        return {"message": "Warmup signal sent."}
    except Exception as e:
        logger.error(f"Warmup signal failed: {str(e)}")
        return {"message": "Warmup signal failed to send."}


@app.post("/transcribe")
async def transcribe_audio(
        audio: UploadFile = File(...),
        patient_id: str = Form(None)
):
    """
    Transcribe audio file using Modal-hosted Parakeet ASR.

    Args:
        audio: Audio file (webm, wav, mp3, etc.)
        patient_id: Optional patient ID for logging

    Returns:
        Transcription result with text and metadata
    """
    logger.info(f"Transcription request - file: {audio.filename}, patient: {patient_id}")

    # Validate content type
    allowed_types = [
        "audio/webm", "audio/wav", "audio/mp3", "audio/mpeg",
        "audio/ogg", "audio/x-wav", "audio/wave", "audio/m4a",
        "audio/x-m4a", "audio/mp4"
    ]

    content_type = audio.content_type or ""
    # Be lenient with content type checking for browser recordings
    if content_type and not any(t in content_type for t in ["audio", "video", "octet"]):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {content_type}"
        )

    try:
        # Read audio bytes
        audio_bytes = await audio.read()
        logger.info(f"Audio size: {len(audio_bytes)} bytes")

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Call Modal transcriber
        transcriber = get_transcriber()
        result = transcriber.transcribe.remote(audio_bytes)

        # Check for errors
        if not result.get("success", False):
            error_msg = result.get("error", "Transcription failed")
            logger.error(f"Modal transcription error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        # Add metadata
        result["metadata"] = {
            "patient_id": patient_id,
            "timestamp": datetime.utcnow().isoformat(),
            "filename": audio.filename,
            "size_bytes": len(audio_bytes),
        }

        logger.info(f"Transcription complete: {len(result.get('text', ''))} chars")
        return JSONResponse(content=result)

    except modal.exception.NotFoundError:
        logger.error(f"Modal app '{MODAL_APP_NAME}' not found. Deploy with: modal deploy modal_asr.py")
        raise HTTPException(
            status_code=503,
            detail="Transcription service not deployed. Run 'modal deploy modal_asr.py' first."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check if Modal app is accessible
    modal_status = "unknown"
    try:
        transcriber = get_transcriber()
        modal_status = "connected"
    except Exception as e:
        modal_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "service": "transcription-gateway",
        "modal_app": MODAL_APP_NAME,
        "modal_status": modal_status,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Transcription Service",
        "description": "API gateway for Modal-hosted Parakeet ASR",
        "modal_app": MODAL_APP_NAME,
        "endpoints": {
            "POST /transcribe": "Transcribe audio file",
            "GET /health": "Health check"
        },
        "setup": {
            "1": "Deploy Modal app: modal deploy modal_asr.py",
            "2": "Start this service: python transcribe.py",
            "3": "POST audio to /transcribe"
        }
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting transcription service on {HOST}:{PORT}")
    logger.info(f"Modal app: {MODAL_APP_NAME}")
    uvicorn.run(app, host=HOST, port=PORT)