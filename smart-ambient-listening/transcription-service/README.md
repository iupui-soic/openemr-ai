# Transcription Service

A two-part transcription service:
1. **Modal App** (`modal_asr.py`): Runs Parakeet TDT 1.1B on Modal's GPU infrastructure
2. **API Gateway** (`transcribe.py`): FastAPI service that receives audio and calls Modal

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│    Frontend     │────▶│  transcribe.py   │────▶│  Modal (GPU Cloud)  │
│  (browser audio)│     │  (API Gateway)   │     │  modal_asr.py       │
│                 │     │  localhost:8001  │     │  Parakeet TDT 1.1B  │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
```

## Setup

### 1. Install Dependencies

```bash
cd transcription-service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Modal

```bash
# Login to Modal (one-time setup)
modal token new

# Or set environment variables
export MODAL_TOKEN_ID=your_token_id
export MODAL_TOKEN_SECRET=your_token_secret
```

### 3. Deploy the Modal App

```bash
modal deploy modal_asr.py
```

This deploys the Parakeet ASR model to Modal's GPU infrastructure. The model loads on first request and stays warm for 2 minutes.

### 4. Start the API Gateway

```bash
python transcribe.py
```

The gateway runs on `http://localhost:8001` and forwards requests to Modal.

## Files

| File | Description |
|------|-------------|
| `modal_asr.py` | Modal app with Parakeet TDT 1.1B model |
| `transcribe.py` | FastAPI gateway that calls Modal |
| `requirements.txt` | Python dependencies for the gateway |

## API Endpoints

### POST /transcribe

Transcribe audio file.

```bash
curl -X POST http://localhost:8001/transcribe \
  -F "audio=@recording.webm" \
  -F "patient_id=12345"
```

**Response:**
```json
{
  "text": "The patient presents with...",
  "duration": 12.5,
  "model": "nvidia/parakeet-tdt-1.1b",
  "success": true,
  "metadata": {
    "patient_id": "12345",
    "timestamp": "2024-01-15T10:30:00Z",
    "filename": "recording.webm",
    "size_bytes": 45678
  }
}
```

### GET /health

Check service status including Modal connectivity.

```json
{
  "status": "healthy",
  "service": "transcription-gateway",
  "modal_app": "parakeet-asr",
  "modal_status": "connected",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Modal App Details

The `modal_asr.py` defines:

- **ParakeetTranscriber**: Class that loads the model on container start
- **GPU**: A10G (24GB VRAM)
- **Timeout**: 300 seconds per request
- **Idle Timeout**: 120 seconds (keeps warm between requests)

### Testing Modal Directly

```bash
# Test with a local file
modal run modal_asr.py -- test_audio.wav

# View logs
modal app logs parakeet-asr
```

### Updating the Modal App

After making changes to `modal_asr.py`:

```bash
modal deploy modal_asr.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | API gateway host |
| `PORT` | `8001` | API gateway port |
| `MODAL_APP_NAME` | `parakeet-asr` | Name of deployed Modal app |
| `MODAL_TOKEN_ID` | - | Modal authentication |
| `MODAL_TOKEN_SECRET` | - | Modal authentication |

## Cost Considerations

- Modal charges for GPU time (~$0.80/hour for A10G)
- Container stays warm for 2 minutes after each request
- Cold starts take ~30 seconds (model loading)
- Subsequent requests: ~1-2 seconds per 10s of audio

## Troubleshooting

### "Modal app not found"

Deploy the Modal app first:
```bash
modal deploy modal_asr.py
```

### "Authentication failed"

Login to Modal:
```bash
modal token new
```

### Slow first request

First request loads the model (~30s). Subsequent requests are fast if within the 2-minute idle timeout.