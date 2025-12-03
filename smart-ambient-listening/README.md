# SMART Ambient Listening

A SMART-on-FHIR application for ambient clinical listening, with ASR powered by NVIDIA Parakeet on Modal.

## Components

```
smart-ambient-listening/
├── frontend/                    # SMART-on-FHIR app (static HTML/JS)
│   ├── launch.html             # SMART launch handler
│   ├── index.html              # Main application
│   └── js/, css/               # Application assets
│
└── transcription-service/       # ASR service
    ├── modal_asr.py            # Modal app (Parakeet on GPU)
    └── transcribe.py           # API gateway (calls Modal)
```

## Architecture

```
┌──────────────┐      ┌────────────────┐      ┌──────────────────────┐
│   OpenEMR    │      │    Frontend    │      │   Transcription      │
│   Patient    │─────▶│  SMART on FHIR │─────▶│   Service            │
│   Dashboard  │      │  (static HTML) │      │   (API Gateway)      │
└──────────────┘      └────────────────┘      └──────────┬───────────┘
                              │                          │
                              │ MediaStream              │ modal.Cls.lookup()
                              │ Recording API            │
                              ▼                          ▼
                      ┌────────────────┐      ┌──────────────────────┐
                      │  Audio Capture │      │   Modal (GPU Cloud)  │
                      │  in Browser    │      │   Parakeet TDT 1.1B  │
                      └────────────────┘      └──────────────────────┘
```

## Quick Start

### 1. Deploy the Modal ASR App

```bash
cd transcription-service

# Login to Modal (first time only)
modal token new

# Deploy Parakeet model to Modal
modal deploy modal_asr.py
```

### 2. Start the API Gateway

```bash
pip install -r requirements.txt
python transcribe.py
```

Gateway runs on `http://localhost:8001`

### 3. Setup the Frontend

```bash
cd frontend/js

# Download FHIR client library
curl -O https://cdn.jsdelivr.net/npm/fhirclient@2/build/fhir-client.min.js
mv fhir-client.min.js fhirclient.min.js
```

Configure transcription URL in `index.html`:
```javascript
window.SMART_CONFIG = {
    transcriptionServiceUrl: 'http://localhost:8001'
};
```

### 4. Serve the Frontend

```bash
cd frontend
python -m http.server 8080
```

### 5. Register in OpenEMR

Add as SMART app:
- **Client ID**: `smart-ambient-listening`
- **Launch URI**: `http://localhost:8080/launch.html`
- **Redirect URI**: `http://localhost:8080/index.html`
- **Scopes**: `launch patient/*.read openid fhirUser`

## How It Works

1. **Launch**: User clicks app in OpenEMR patient dashboard
2. **Authorize**: SMART-on-FHIR OAuth2 flow with patient context
3. **Record**: Browser captures audio via MediaStream Recording API
4. **Transcribe**: Audio sent to API gateway → Modal → Parakeet ASR
5. **Display**: Transcription shown with patient context

## Deployment

### Frontend
- Any static file server (Nginx, S3, CloudFront, Vercel)
- Requires HTTPS for microphone access in production

### Transcription Service
- API Gateway: Any Python hosting (Heroku, Railway, EC2)
- Modal App: Automatically managed by Modal (GPU on-demand)

## Configuration

### Frontend (`index.html`)
```javascript
window.SMART_CONFIG = {
    clientId: 'smart-ambient-listening',
    scope: 'launch patient/*.read openid fhirUser',
    transcriptionServiceUrl: 'https://your-api-gateway.com'
};
```

### Transcription Service (environment)
```bash
MODAL_TOKEN_ID=...       # Modal authentication
MODAL_TOKEN_SECRET=...   # Modal authentication
MODAL_APP_NAME=parakeet-asr  # Deployed Modal app name
```

## Cost

- **Modal GPU**: ~$0.80/hour for A10G
- Container stays warm 2 minutes between requests
- Cold start: ~30 seconds
- Warm requests: ~1-2 seconds per 10s of audio

## See Also

- [frontend/README.md](frontend/README.md) - Frontend setup details
- [transcription-service/README.md](transcription-service/README.md) - Service setup details