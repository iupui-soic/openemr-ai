"""
CPT Coding Service API

Each student uses THEIR OWN Modal account and free tier credits.
Modal credentials are retrieved from OpenEMR user settings per-request.

SECURITY: Token-based authentication with server-side credential retrieval.
ISOLATION: Per-user Modal apps deployed to each student's own Modal account.

Endpoints:
- POST /deploy-and-warmup: Deploy Modal app to student's account (requires token)
- POST /warmup: Keep-alive ping to student's container (requires token)
- POST /code: CPT-code a clinical note using student's Modal app (requires token)
- GET /health: Health check

Run with: python coding.py
"""

import os
import base64
import json
import logging
import subprocess
from datetime import datetime
import hashlib
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

import modal
from modal.exception import NotFoundError
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))
MODAL_CLASS_NAME = "Gemma4Coder"

OPENEMR_SERVER_URL = os.getenv("OPENEMR_SERVER_URL", "https://localhost:9300")
OPENEMR_FHIR_BASE = f"{OPENEMR_SERVER_URL}/apis/default/fhir"
OPENEMR_REST_BASE = f"{OPENEMR_SERVER_URL}/apis/default/api"


app = FastAPI(
    title="CPT Coding Service - TRUE BYOK",
    description="Students use their own Modal accounts. Credentials retrieved from OpenEMR per-request.",
    version="1.0.0-BYOK",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request / Response Models
# ============================================================================

class CodeRequest(BaseModel):
    text: str
    descriptions: Optional[dict] = None


# ============================================================================
# Authentication and Authorization
# ============================================================================

def extract_token_from_header(authorization: str) -> str:
    """Extract bearer token from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    return parts[1]


def try_decode_jwt_subject(token: str):
    """
    If access token is a JWT, extract a stable identifier.
    Returns None if token is not JWT or cannot be decoded.
    """
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
    except requests.exceptions.RequestException:
        logger.exception("Failed to validate token against FHIR")
        raise HTTPException(status_code=503, detail="Failed to validate token with OpenEMR server")

    extracted = try_decode_jwt_subject(token)
    if extracted:
        user_id = str(extracted).replace("/", "-")
        logger.info(f"✅ Authenticated token. Derived user id from JWT: {user_id}")
        return {"user_id": user_id, "username": None, "token": token}

    token_fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    user_id = f"token-{token_fingerprint}"
    logger.info(f"✅ Authenticated token. Derived user id from token hash: {user_id}")
    return {"user_id": user_id, "username": None, "token": token}


async def get_student_modal_credentials(user_token: str) -> tuple:
    """
    Retrieve student's Modal API credentials from OpenEMR server-side.
    """
    try:
        response = requests.get(
            f"{OPENEMR_REST_BASE}/user/settings/custom",
            headers={"Authorization": f"Bearer {user_token}"},
            timeout=10,
            verify=False,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve user settings from OpenEMR",
            )

        settings_data = response.json()
        settings = {}

        if 'data' in settings_data and isinstance(settings_data['data'], list):
            for setting in settings_data['data']:
                settings[setting['field_id']] = setting['field_value']

        modal_token_id = (
            settings.get('modal_api_token') or
            settings.get('MODAL API Token') or
            settings.get('modal_token_id') or
            settings.get('modalApiToken') or
            settings.get('2')
        )

        modal_token_secret = (
            settings.get('modal_api_secret') or
            settings.get('MODAL API secret') or
            settings.get('modal_token_secret') or
            settings.get('modalApiSecret') or
            settings.get('3')
        )

        if not modal_token_id or not modal_token_secret:
            logger.error(f"Modal credentials not found in user settings: {list(settings.keys())}")
            raise HTTPException(
                status_code=400,
                detail=(
                    "Modal API credentials not configured. Please add them in OpenEMR:\n"
                    "1. Go to User Settings → Custom\n"
                    "2. Add 'Modal API Token ID' (field 2)\n"
                    "3. Add 'Modal API Token Secret' (field 3)\n"
                    "4. Get credentials from https://modal.com → Settings → Tokens"
                ),
            )

        if len(modal_token_id) < 10 or len(modal_token_secret) < 10:
            raise HTTPException(
                status_code=400,
                detail="Modal credentials appear invalid (too short). Please check your settings.",
            )

        logger.info(f"✅ Retrieved Modal credentials for user (token_id starts with: {modal_token_id[:10]}...)")
        return (modal_token_id, modal_token_secret)

    except requests.exceptions.RequestException:
        logger.exception("Failed to get user settings from OpenEMR")
        raise HTTPException(
            status_code=503,
            detail="Failed to retrieve user settings from OpenEMR",
        )


def get_student_app_name(user_id: str) -> str:
    """
    Generate unique Modal app name per student.
    Each student gets isolated deployment IN THEIR OWN Modal account.
    """
    safe_id = ''.join(c for c in user_id if c.isalnum() or c == '-')
    if len(safe_id) > 30:
        safe_id = hashlib.sha256(user_id.encode()).hexdigest()[:12]
    return f"gemma4-cpt-coder-{safe_id}"


# ============================================================================
# Helper Functions
# ============================================================================

def get_modal_env(modal_token_id: str, modal_token_secret: str):
    """Get environment with student's Modal credentials."""
    modal_env = os.environ.copy()
    modal_env["MODAL_TOKEN_ID"] = modal_token_id
    modal_env["MODAL_TOKEN_SECRET"] = modal_token_secret
    return modal_env


def deploy_modal_app(app_name: str, modal_token_id: str, modal_token_secret: str):
    """
    Deploy the Modal app to STUDENT's Modal account using subprocess.
    """
    logger.info(f"🚀 Deploying Modal app: {app_name} to student's account")

    modal_env = get_modal_env(modal_token_id, modal_token_secret)
    modal_env["MODAL_APP_NAME"] = app_name

    script_path = os.path.join(os.path.dirname(__file__), "modal_coding.py")
    MODAL_BIN = os.getenv("MODAL_BIN", os.path.expanduser("~/.local/bin/modal"))
    logger.info(f"Using MODAL_BIN={MODAL_BIN}")
    command = [MODAL_BIN, "deploy", script_path]

    logger.info(f"Running command: {' '.join(command)}")
    logger.info(f"Using student's Modal credentials (token_id: {modal_token_id[:10]}...)")

    try:
        process = subprocess.run(
            command,
            env=modal_env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout — image build for transformers/torch is heavy
        )

        logger.info(f"Deploy stdout: {process.stdout}")
        if process.stderr:
            logger.info(f"Deploy stderr: {process.stderr}")
        logger.info(f"Deploy return code: {process.returncode}")

        if process.returncode != 0:
            logger.error(f"Modal deploy failed for {app_name}!")
            logger.error(f"stdout: {process.stdout}")
            logger.error(f"stderr: {process.stderr}")

            error_detail = process.stderr or process.stdout
            if "authentication" in error_detail.lower() or "credentials" in error_detail.lower():
                raise HTTPException(
                    status_code=401,
                    detail="Modal authentication failed. Please check your Modal API credentials in OpenEMR user settings.",
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Modal deploy failed: {error_detail}",
                )

        logger.info(f"✅ Modal app {app_name} deployed successfully to student's account!")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"Deploy timed out for {app_name}")
        raise HTTPException(
            status_code=504,
            detail="Modal deployment timed out (>10 minutes). Please try again.",
        )


def get_coder(app_name: str, modal_token_id: str, modal_token_secret: str):
    """
    Return an instance of the deployed Gemma4Coder class from the student's Modal account.
    """
    try:
        modal_client = modal.Client.from_credentials(modal_token_id, modal_token_secret)
        CoderCls = modal.Cls.from_name(app_name, MODAL_CLASS_NAME, client=modal_client)
        return CoderCls()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Modal app/class not found or not accessible: {e}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/deploy-and-warmup")
async def deploy_and_warmup(authorization: str = Header(None)):
    """
    Deploy Modal app to STUDENT's account and warm up container.
    Uses student's own Modal credentials and free tier quota.
    """
    user_info = await validate_token_and_get_user(authorization)
    user_id = user_info["user_id"]
    app_name = get_student_app_name(user_id)

    logger.info(f"📥 Deploy request from user {user_id}, app: {app_name}")

    try:
        modal_token_id, modal_token_secret = await get_student_modal_credentials(user_info["token"])

        deploy_modal_app(app_name, modal_token_id, modal_token_secret)

        logger.info(f"🔥 Warming up container in student's account: {app_name}")
        coder = get_coder(app_name, modal_token_id, modal_token_secret)
        result = coder.wakeup.remote()

        logger.info(f"✅ Container warmed: {result}")
        return JSONResponse(content={
            "status": "ready",
            "deployed": True,
            "warmed_up": True,
            "app_name": app_name,
            "user_id": user_id,
            "message": "Modal app deployed to YOUR account and container warm",
            "account": "student_account",
        })

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        logger.error(f"Deploy timed out for {app_name}")
        raise HTTPException(status_code=504, detail="Deploy timed out")
    except Exception as e:
        logger.exception(f"Deploy/warmup error for {app_name}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/warmup")
async def warmup_only(authorization: str = Header(None)):
    """Keep-alive ping for student's container in THEIR Modal account."""
    user_info = await validate_token_and_get_user(authorization)
    user_id = user_info["user_id"]
    app_name = get_student_app_name(user_id)

    logger.info(f"🔥 Warmup ping from user {user_id}, app: {app_name}")

    try:
        modal_token_id, modal_token_secret = await get_student_modal_credentials(user_info["token"])

        coder = get_coder(app_name, modal_token_id, modal_token_secret)
        result = coder.wakeup.remote()

        logger.info(f"✅ Warmup successful for {app_name}: {result}")
        return JSONResponse(content={
            "status": "warm",
            "result": result,
            "app_name": app_name,
            "user_id": user_id,
            "account": "student_account",
            "timestamp": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.exception(f"Warmup failed for {app_name}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "failed",
                "message": str(e),
                "app_name": app_name,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@app.post("/code")
async def code_clinical_note(
    request: CodeRequest,
    authorization: str = Header(None),
):
    """
    CPT-code a clinical note using student's Modal app in THEIR account.

    Headers:
        Authorization: Bearer <SMART_access_token> (REQUIRED)

    Body:
        text: Clinical note text (REQUIRED)
        descriptions: Optional dict of {cpt_code: description}.
                      If omitted, the Modal app uses DEFAULT_DESCRIPTIONS.
    """
    user_info = await validate_token_and_get_user(authorization)
    user_id = user_info["user_id"]
    app_name = get_student_app_name(user_id)

    logger.info(f"🩺 Coding request from user {user_id}, app: {app_name}, "
                f"note length: {len(request.text)} chars")

    if not request.text or len(request.text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Clinical note text is required")

    try:
        modal_token_id, modal_token_secret = await get_student_modal_credentials(user_info["token"])

        try:
            coder = get_coder(app_name, modal_token_id, modal_token_secret)
        except HTTPException:
            logger.warning(f"App {app_name} not found. Deploying then retrying...")
            deploy_modal_app(app_name, modal_token_id, modal_token_secret)
            coder = get_coder(app_name, modal_token_id, modal_token_secret)

        result = coder.code.remote(request.text, request.descriptions)
        logger.info(f"Modal code raw result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")

        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail=f"Unexpected Modal response type: {type(result)}")

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "CPT coding failed"))

        result["metadata"] = {
            "user_id": user_id,
            "app_name": app_name,
            "account": "student_account",
            "timestamp": datetime.utcnow().isoformat(),
            "note_length": len(request.text),
        }

        logger.info(f"✅ Coding complete for {user_id}: {len(result.get('codes', []))} codes")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Coding error for {app_name}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "cpt-coding-gateway-TRUE-BYOK",
        "version": "1.0.0-BYOK",
        "authentication": "token-based",
        "isolation": "per-user",
        "byok": True,
        "account_model": "Each student uses their own Modal account",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "CPT Coding Service - TRUE BYOK",
        "description": "Students use their own Modal accounts. Credentials retrieved from OpenEMR per-request.",
        "version": "1.0.0-BYOK",
        "authentication": "SMART OAuth2 token required",
        "byok": True,
        "account_model": "Each student uses their own Modal free tier",
        "setup_instructions": [
            "1. Sign up at https://modal.com (free account)",
            "2. Go to Settings → Tokens → Create new token",
            "3. Copy Token ID and Token Secret",
            "4. In OpenEMR: User Settings → Custom",
            "5. Add Modal API Token ID (field 2)",
            "6. Add Modal API Token Secret (field 3)",
            "7. Call /deploy-and-warmup once to deploy Gemma 4 to YOUR account",
        ],
        "endpoints": {
            "/deploy-and-warmup": "POST - Deploy to YOUR Modal account (requires Authorization header)",
            "/warmup": "POST - Keep-alive ping to YOUR container (requires Authorization header)",
            "/code": "POST - CPT-code a clinical note using YOUR Modal app (requires Authorization header)",
            "/health": "GET - Health check",
        },
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting CPT coding service on 127.0.0.1:{PORT}")
    logger.info("=" * 60)
    logger.info("TRUE BYOK MODE ENABLED")
    logger.info("Students use their OWN Modal accounts")
    logger.info("Credentials retrieved from OpenEMR per-request")
    logger.info("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=PORT)
