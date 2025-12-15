"""
Modal App for Parakeet ASR

Runs NVIDIA Parakeet TDT 1.1B on Modal's GPU infrastructure.
The model is loaded once when the container starts and reused for all requests.

Deployment:
    modal deploy modal_asr.py

Usage (from Python):
    from modal import Cls
    transcriber = Cls.lookup("parakeet-asr", "ParakeetTranscriber")
    result = transcriber.transcribe.remote(audio_bytes)
"""

import modal

app = modal.App("parakeet-asr")

MODEL_ID = "nvidia/parakeet-tdt-1.1b"

# Container image with all dependencies
parakeet_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libsndfile1", "ffmpeg", "build-essential")
    .pip_install(
        "cython",
        "packaging",
        "torch",
        "torchaudio",
        "nemo_toolkit[asr]",
        "scipy",
        "fastapi"
    )
)


@app.cls(
    image=parakeet_image,
    gpu="A10G",
    timeout=300,
    container_idle_timeout=120,  # Keep warm for 2 minutes
)
class ParakeetTranscriber:
    """
    Parakeet TDT 1.1B Transcriber running on Modal GPU.

    The model is loaded once when the container starts (@modal.enter)
    and reused for subsequent requests until the container is recycled.
    """

    @modal.enter()
    def load_model(self):
        """Load the Parakeet model when container starts."""
        import nemo.collections.asr as nemo_asr

        print(f"Loading {MODEL_ID}...")
        self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=MODEL_ID,
            map_location="cuda"
        )
        self.model.eval()
        print("Model loaded successfully!")
    # --- ADD THIS NEW METHOD ---
    @modal.method()
    def wakeup(self):
        """A dummy method to warm up the container and trigger the
        load_model() method.
        """
        print("Warm-up signal received. Container is active.")
        pass
    # --- END OF ADDED METHOD ---

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file bytes (webm, m4a, mp3, wav, etc.)

        Returns:
            Dictionary with transcription results:
            - text: Transcribed text
            - duration: Audio duration in seconds
            - success: Whether transcription succeeded
        """
        import tempfile
        import os
        import subprocess
        import soundfile as sf

        # Detect format from magic bytes
        suffix = self._detect_format(audio_bytes)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            input_path = f.name

        output_path = input_path.rsplit('.', 1)[0] + ".wav"

        try:
            # Convert to 16kHz mono WAV (required for Parakeet)
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path
            ], check=True, capture_output=True)

            # Get audio duration
            audio_data, sample_rate = sf.read(output_path)
            duration = len(audio_data) / sample_rate

            # Transcribe with Parakeet TDT
            result = self.model.transcribe([output_path])

            # Handle return format
            text = ""
            if result and len(result) > 0:
                text = result[0]
                if hasattr(text, 'text'):
                    text = text.text
                text = str(text).strip()

            return {
                "text": text,
                "duration": duration,
                "model": MODEL_ID,
                "success": True
            }

        except subprocess.CalledProcessError as e:
            return {
                "text": "",
                "error": f"Audio conversion failed: {e.stderr.decode() if e.stderr else str(e)}",
                "success": False
            }
        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
        finally:
            # Cleanup temp files
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def _detect_format(self, audio_bytes: bytes) -> str:
        """Detect audio format from magic bytes."""
        if len(audio_bytes) < 12:
            return ".wav"

        # WebM
        if audio_bytes[:4] == b'\x1a\x45\xdf\xa3':
            return ".webm"
        # M4A/MP4
        if audio_bytes[4:8] == b'ftyp' or audio_bytes[:12].find(b'ftyp') >= 0:
            return ".m4a"
        # MP3
        if audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
            return ".mp3"
        # WAV
        if audio_bytes[:4] == b'RIFF':
            return ".wav"
        # FLAC
        if audio_bytes[:4] == b'fLaC':
            return ".flac"
        # OGG
        if audio_bytes[:4] == b'OggS':
            return ".ogg"

        return ".webm"  # Default for browser recordings


# Web endpoint for direct HTTP access (optional)
@app.function(image=parakeet_image, gpu="A10G", timeout=300)
@modal.web_endpoint(method="POST")
async def transcribe_endpoint(request: dict) -> dict:
    """
    HTTP endpoint for transcription.

    Expects JSON with base64-encoded audio:
    {
        "audio": "<base64-encoded-audio>",
        "filename": "recording.webm"
    }
    """
    import base64

    audio_b64 = request.get("audio")
    if not audio_b64:
        return {"error": "Missing 'audio' field", "success": False}

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        return {"error": f"Invalid base64: {e}", "success": False}

    transcriber = ParakeetTranscriber()
    return transcriber.transcribe.local(audio_bytes)


# For local testing
@app.local_entrypoint()
def main():
    """Test the transcription with a sample file."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: modal run modal_asr.py -- <audio_file>")
        return

    audio_path = sys.argv[1]
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    transcriber = ParakeetTranscriber()
    result = transcriber.transcribe.remote(audio_bytes)

    print(f"Transcription: {result.get('text', '')}")
    print(f"Duration: {result.get('duration', 0):.2f}s")
    print(f"Success: {result.get('success', False)}")