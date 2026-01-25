# SMART Ambient Listening - Frontend

A SMART-on-FHIR application for ambient clinical listening that launches from OpenEMR's patient dashboard.

## Overview

This is a pure HTML/JavaScript frontend application that:
- Launches via SMART-on-FHIR from OpenEMR
- Receives patient context through the authorization flow
- Captures audio using the browser's MediaStream Recording API
- Sends audio to a separate transcription service

## Files

```
frontend/
├── index.html          # Main application page
├── launch.html         # SMART-on-FHIR launch handler
├── css/
│   └── styles.css      # Application styles
└── js/
    ├── fhirclient.min.js   # SMART-on-FHIR client library (download required)
    ├── openemr-api.js      # OpenEMR REST API client (custom user settings, etc.)
    ├── ambient-recorder.js  # MediaStream Recording wrapper
    └── app.js              # Main application logic
```

## Setup

### 1. Download FHIR Client Library

Download the official SMART-on-FHIR JavaScript client:

```bash
cd js
curl -O https://cdn.jsdelivr.net/npm/fhirclient@2/build/fhir-client.min.js
mv fhir-client.min.js fhirclient.min.js
```

Or use npm:
```bash
npm install fhirclient
cp node_modules/fhirclient/build/fhir-client.min.js js/fhirclient.min.js
```

### 2. Configure the Application

Edit `index.html` to set your transcription service URL:

```javascript
window.SMART_CONFIG = {
    clientId: 'smart-ambient-listening',
    scope: 'launch patient/*.read openid fhirUser',
    transcriptionServiceUrl: 'http://your-transcription-service:8001'
};
```

### 3. Register in OpenEMR

1. Go to **Administration > System > API Clients** in OpenEMR
2. Add a new client:
   - **Client ID**: `smart-ambient-listening`
   - **Name**: `Ambient Listening`
   - **Redirect URI**: `https://your-app-url/index.html`
   - **Launch URI**: `https://your-app-url/launch.html`
   - **Scopes**: `launch patient/*.read openid fhirUser`
   - **Is Confidential**: No (public client)

## Deployment

The frontend is static HTML/JS and can be served by any web server:

### Using Python
```bash
cd frontend
python -m http.server 8080
```

### Using Node.js
```bash
npx serve frontend -l 8080
```

### Using Nginx
```nginx
server {
    listen 80;
    server_name ambient-listening.example.com;
    root /path/to/frontend;
    index index.html;
}
```

## SMART-on-FHIR Flow

1. User clicks app link in OpenEMR patient dashboard
2. OpenEMR redirects to `launch.html?iss=...&launch=...`
3. `launch.html` uses fhirclient.js to initiate OAuth2 authorization
4. User authorizes (if needed)
5. Redirect back to `index.html` with authorization code
6. `app.js` completes authorization and loads patient context

## Browser Requirements

- Chrome 49+, Firefox 52+, Safari 14+, Edge 79+
- Microphone access permission
- JavaScript enabled

## Standalone Testing

The app can run without SMART context for testing:
1. Serve the frontend on localhost
2. Open `index.html` directly (not via `launch.html`)
3. The app will show "Not connected to EHR" but recording will work

## OpenEMR REST API Client

The app includes `openemr-api.js` which provides access to OpenEMR's standard REST API
endpoints using the SMART on FHIR access token. This enables features like custom user
settings (for app preferences).

### Custom User Settings

Custom user settings are stored per-user in OpenEMR and can be used for app preferences:

```javascript
// The API client is automatically initialized when connecting to OpenEMR
// Access it via the global openemrApi variable in app.js

// Get all custom settings
const settings = await openemrApi.getCustomUserSettings();

// Get available field definitions (from Admin > Layouts > USR)
const fields = await openemrApi.getCustomUserSettingsFields();

// Get a specific setting
const setting = await openemrApi.getCustomUserSetting('preferred_language');

// Update a setting
await openemrApi.updateCustomUserSetting('preferred_language', 'es');

// Delete/reset a setting to default
await openemrApi.deleteCustomUserSetting('preferred_language');
```

### Setting Up Custom Fields

To use custom user settings, configure USR layout fields in OpenEMR:

1. Go to **Administration > Layouts**
2. Select **USR (User Settings)** form
3. Add custom fields like:
   - `transcription_service_url` - Override transcription service URL
   - `preferred_language` - Preferred transcription language
   - `auto_summarize` - Auto-generate SOAP notes after recording

### Required Scope

The `api:oemr` scope is required for REST API access. This is already included in the
app's scope configuration:

```javascript
scope: 'launch patient/*.read openid fhirUser api:oemr'
```