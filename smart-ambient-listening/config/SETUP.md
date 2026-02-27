# Smart Ambient Listening — Instance Setup Guide

## Prerequisites
- OpenEMR running in Docker (ports 8080 for web, 3306 for MySQL)
- Apache2 with SSL and mod_proxy enabled
- Python 3.10+
- pip packages: `pip3 install pymysql chromadb groq --break-system-packages`

## Step 1: Configure Environment Files

Copy `.env.example` to both service directories and update the server URL:
```bash
cp config/.env.example rag-text-summarization/.env
cp config/.env.example transcription-service/.env
```

Edit both `.env` files — change `OPENEMR_SERVER_URL` to your instance:
```
OPENEMR_SERVER_URL=https://<YOUR-SERVER-HOSTNAME>
```

If your OpenEMR database credentials differ from defaults, also update:
```
OPENEMR_DB_HOST=localhost
OPENEMR_DB_PORT=3306
OPENEMR_DB_USER=openemr
OPENEMR_DB_PASS=openemr
OPENEMR_DB_NAME=openemr
```

## Step 2: Deploy Frontend
```bash
sudo cp -r frontend/ /var/www/html/app/
sudo chown -R www-data:www-data /var/www/html/app
```

Then update the server URL in two frontend files:
- `/var/www/html/app/launch.html` — change `iss` URL to your server
- `/var/www/html/app/index.html` — change `iss` URL to your server

## Step 3: Configure Apache Proxy

Add the contents of `config/apache-proxy.conf` to your
`/etc/apache2/sites-enabled/default-ssl.conf` inside the `<VirtualHost *:443>` block.

Then reload Apache:
```bash
sudo a2enmod proxy proxy_http
sudo systemctl reload apache2
```

## Step 4: Install Systemd Services

For each service file in `config/`, replace placeholders and install:

### What to replace:
- `<YOUR_USER>` → your Linux username (e.g., `shfnu`)
- `<PATH_TO>` → full path to the repo (e.g., `/home/shfnu/openemr-ai`)

### Install commands:
```bash
sudo cp config/summarize.service /etc/systemd/system/
sudo cp config/chromadb.service /etc/systemd/system/
sudo cp config/transcription.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable summarize.service chromadb.service transcription.service
sudo systemctl start chromadb.service
sudo systemctl start summarize.service
sudo systemctl start transcription.service
```

### Verify all services are running:
```bash
sudo systemctl status chromadb.service
sudo systemctl status summarize.service
sudo systemctl status transcription.service
```

## Step 5: Register SMART App in OpenEMR

1. Log into OpenEMR as admin
2. Go to Admin → System → API Clients
3. Register a new SMART app with:
   - **App Name:** smart-ambient-listening
   - **Redirect URI:** `https://<YOUR-SERVER>/app/index.html`
   - **Launch URI:** `https://<YOUR-SERVER>/app/launch.html`
   - **Scopes:** patient/Patient.read, patient/Condition.read, patient/Procedure.read, patient/Observation.read, patient/MedicationRequest.read, patient/AllergyIntolerance.read, patient/Encounter.read, openid, fhirUser, api:oemr, user/custom.read, user/custom.write, user/soap_note.write

## Service Ports Summary

| Service | Port | Description |
|---------|------|-------------|
| ChromaDB | 8000 | RAG vector database (medical schemas) |
| Transcription | 8001 | Audio transcription gateway |
| Summarization | 8002 | SOAP note generation + save to OpenEMR |
| OpenEMR (Docker) | 8080 | OpenEMR web interface |
| MySQL (Docker) | 3306 | OpenEMR database |

## Troubleshooting

### Check service logs:
```bash
sudo journalctl -u summarize.service -f
sudo journalctl -u chromadb.service -f
sudo journalctl -u transcription.service -f
```

### Verify ChromaDB has data:
```bash
curl -s http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database/collections
```

### Verify RAG is connected:
```bash
sudo journalctl -u summarize.service --since "5 min ago" | grep -i "chroma\|schema\|retrieved"
```
Should show: "Connected to ChromaDB: 1000 documents available"

### Common issues:
- **Port already in use:** `sudo systemctl restart <service-name>`
- **ChromaDB not connected:** Ensure chromadb.service is running before summarize.service
- **401 on SOAP save:** Ensure `user/soap_note.write` scope is in the SMART app registration
- **SOAP note not visible in UI:** This is expected — notes appear in encounter Summary view, not under Clinical → SOAP dropdown
