#!/usr/bin/env bash
# One-time setup for revision experiments: install deps + transcode institutional m4a.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== revision_setup ==="
echo "Repo: $REPO_ROOT"

# Activate the project venv if present
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "Activated .venv ($(which python))"
fi

echo
echo "--- pip install (DER + ablation + ASR deps) ---"
pip install --quiet \
    pyannote.audio \
    whisperx \
    textgrid \
    sentence-transformers \
    soundfile \
    scipy \
    pandas \
    tqdm \
    nemo_toolkit[asr] \
    "transformers>=4.55" \
    mistral-common \
    librosa \
    || { echo "pip install failed"; exit 1; }
# torchcodec ships with whisperx but breaks pyannote here; remove it
pip uninstall -y torchcodec >/dev/null 2>&1 || true
echo "deps installed"

echo
echo "--- transcode institutional m4a -> 16 kHz mono wav ---"
INST_DIR="openemr_whisper_wer/data/institutional"
WAV_DIR="$INST_DIR/wav"
mkdir -p "$WAV_DIR"
for m4a in "$INST_DIR"/*.m4a; do
    [ -e "$m4a" ] || { echo "no m4a files in $INST_DIR"; break; }
    base="$(basename "${m4a%.m4a}")"
    wav="$WAV_DIR/${base}.wav"
    if [ -f "$wav" ]; then
        echo "  exists: $wav"
        continue
    fi
    ffmpeg -nostdin -loglevel error -y -i "$m4a" -ac 1 -ar 16000 "$wav"
    echo "  wrote:  $wav"
done

echo
echo "--- HF token check (for pyannote 3.1 gating) ---"
if [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "  HF token cached at ~/.cache/huggingface/token"
elif [ -n "${HF_TOKEN:-}" ]; then
    echo "  HF_TOKEN env var set"
else
    echo "  WARN: no HF token found. Run: huggingface-cli login"
fi

echo
echo "=== setup complete ==="
