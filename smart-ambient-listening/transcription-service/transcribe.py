"""
Transcription Service API

FastAPI service that receives audio from the frontend and calls
the Modal-hosted Parakeet ASR model for transcription.

Endpoints:
- POST /deploy-and-warmup: Deploy Modal app (first time) and warm up container
- POST /warmup: Keep-alive ping to prevent container scale-down
- POST /transcribe: Transcribe audio using warm container
- GET /health: Health check

Flow:
1. First "Start Recording" → deploys app + warms container (model loads)
2. During recording → warmup pings every 60s keep container alive
3. "Stop Recording" → transcribes using warm container (fast!)
4. After recording stops → no more pings → container scales down after 2 min
5. Next "Start Recording" → app already deployed, just warms container

Run with: python transcribe.py
"""

import os
import logging
import subprocess
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
load_dotenv()

import modal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
MODAL_APP_NAME = "parakeet-asr"
MODAL_CLASS_NAME = "ParakeetTranscriber"

app = FastAPI(
    title="Transcription Service",
    description="API gateway for Modal-deployed Parakeet ASR with container reuse",
    version="2.0.0"
)

# CORS - allow frontend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
is_deployed = False


def get_modal_env():
    """Get environment with Modal credentials."""
    modal_token_id = os.getenv("MODAL_TOKEN_ID")
    modal_token_secret = os.getenv("MODAL_TOKEN_SECRET")

    if not modal_token_id or not modal_token_secret:
        raise HTTPException(
            status_code=500,
            detail="Modal API credentials (MODAL_TOKEN_ID, MODAL_TOKEN_SECRET) not set in environment."
        )

    modal_env = os.environ.copy()
    modal_env["MODAL_TOKEN_ID"] = modal_token_id
    modal_env["MODAL_TOKEN_SECRET"] = modal_token_secret
    return modal_env


def deploy_modal_app():
    """Deploy the Modal app using subprocess."""
    logger.info("🚀 Deploying Modal app...")

    modal_env = get_modal_env()
    script_path = os.path.join(os.path.dirname(__file__), "modal_asr.py")
    command = ["modal", "deploy", script_path]

    logger.info(f"Running command: {' '.join(command)}")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Script exists: {os.path.exists(script_path)}")

    process = subprocess.run(
        command,
        env=modal_env,
        capture_output=True,
        text=True,
        timeout=300  # 5 min timeout for deployment
    )

    logger.info(f"Deploy stdout: {process.stdout}")
    logger.info(f"Deploy stderr: {process.stderr}")
    logger.info(f"Deploy return code: {process.returncode}")

    if process.returncode != 0:
        logger.error(f"Modal deploy failed!")
        logger.error(f"stdout: {process.stdout}")
        logger.error(f"stderr: {process.stderr}")
        raise HTTPException(status_code=500, detail=f"Modal deploy failed: {process.stderr or process.stdout}")

    logger.info("✅ Modal app deployed successfully!")
    return True


def get_transcriber():
    """Get reference to the deployed Modal class."""
    try:
        ParakeetTranscriber = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
        return ParakeetTranscriber()
    except Exception as e:
        logger.error(f"Failed to get Modal class reference: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Modal app not available: {e}"
        )


@app.post("/deploy-and-warmup")
async def deploy_and_warmup():
    """
    Deploy Modal app (if needed) and warm up the container.
    Call this when user starts recording so model is ready when they stop.
    """
    global is_deployed

    logger.info("📥 Deploy and warmup request received")

    try:
        # Always deploy first (Modal handles "already deployed" gracefully)
        # This ensures the app exists before we try to look it up
        logger.info("🚀 Deploying Modal app...")
        deploy_modal_app()
        is_deployed = True

        # Now warm up the container (this loads the model via @modal.enter)
        logger.info("🔥 Warming up container...")
        transcriber = get_transcriber()
        result = transcriber.wakeup.remote()

        logger.info(f"✅ Container warmed up: {result}")
        return JSONResponse(content={
            "status": "ready",
            "deployed": True,
            "warmed_up": True,
            "message": "Modal app deployed and container warm"
        })

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        logger.error("Deploy timed out")
        raise HTTPException(status_code=504, detail="Deploy timed out")
    except Exception as e:
        logger.error(f"Deploy/warmup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/warmup")
async def warmup_only():
    """
    Send a keep-alive ping to the container.
    Called periodically during recording to prevent scale-down.
    """
    global is_deployed

    logger.info("🔥 Warmup ping received")

    try:
        transcriber = get_transcriber()
        result = transcriber.wakeup.remote()
        is_deployed = True
        logger.info(f"✅ Warmup ping successful: {result}")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Warmup ping failed: {e}")
        # Don't raise error - just return failure status
        # The container might have scaled down, next deploy-and-warmup will fix it
        return JSONResponse(
            status_code=200,
            content={"status": "failed", "message": str(e)}
        )


@app.post("/transcribe")
async def transcribe_audio(
        audio: UploadFile = File(...),
        patient_id: str = Form(None)
):
    """
    Transcribe audio using the deployed Modal container.
    Container should already be warm from /deploy-and-warmup call.
    """
    global is_deployed

    logger.info(f"🎤 Transcription request - file: {audio.filename}, patient: {patient_id}")

    try:
        audio_bytes = await audio.read()
        logger.info(f"📊 Audio size: {len(audio_bytes)} bytes")

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Try to get transcriber - this will work if app is deployed
        try:
            transcriber = get_transcriber()
        except HTTPException:
            # App might not be deployed, try to deploy first
            logger.warning("App not available, attempting to deploy...")
            deploy_modal_app()
            is_deployed = True
            transcriber = get_transcriber()

        # Call the deployed Modal container
        result = transcriber.transcribe.remote(audio_bytes)

        if not result.get("success", False):
            error_msg = result.get("error", "Transcription failed")
            logger.error(f"❌ Transcription error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        result["metadata"] = {
            "patient_id": patient_id,
            "timestamp": datetime.utcnow().isoformat(),
            "filename": audio.filename,
            "size_bytes": len(audio_bytes),
        }

        logger.info(f"✅ Transcription complete: {len(result.get('text', ''))} chars")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "transcription-gateway-deployed",
        "modal_deployed": is_deployed,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Transcription Service",
        "description": "API gateway for Modal-deployed Parakeet ASR with container reuse",
        "version": "2.0.0",
        "modal_deployed": is_deployed,
        "endpoints": {
            "/deploy-and-warmup": "POST - Deploy Modal app (first time) and warm up container",
            "/warmup": "POST - Keep-alive ping during recording",
            "/transcribe": "POST - Transcribe audio (requires deploy first)",
            "/health": "GET - Health check",
        }
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting transcription service on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)