"""
Modal App for Parakeet ASR

Runs NVIDIA Parakeet TDT 1.1B on Modal's GPU infrastructure.
The model is loaded once when the container starts and reused for all requests.
"""

import modal
import sys

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
    scaledown_window=120,  # Scale down after 2 min idle (warmup pings keep it alive)
)
class ParakeetTranscriber:
    """
    Parakeet TDT 1.1B Transcriber running on Modal GPU.
    """

    @modal.enter()
    def load_model(self):
        """Load the Parakeet model when container starts."""
        import nemo.collections.asr as nemo_asr

        print(f"Loading {MODEL_ID}...", file=sys.stderr)
        self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=MODEL_ID,
            map_location="cuda"
        )
        self.model.eval()
        print("Model loaded successfully!", file=sys.stderr)

    @modal.method()
    def wakeup(self):
        """A dummy method to warm up the container and reset the idle timer."""
        print("✅ Keep-alive signal received. Container is warm.", file=sys.stderr)
        return {"status": "warm"}

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Transcribe audio bytes to text.
        """
        import tempfile
        import os
        import subprocess
        import soundfile as sf

        suffix = self._detect_format(audio_bytes)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            input_path = f.name

        output_path = input_path.rsplit('.', 1)[0] + ".wav"

        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path
            ], check=True, capture_output=True)

            audio_data, sample_rate = sf.read(output_path)
            duration = len(audio_data) / sample_rate

            result = self.model.transcribe([output_path])

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
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def _detect_format(self, audio_bytes: bytes) -> str:
        """Detect audio format from magic bytes."""
        if len(audio_bytes) < 12:
            return ".wav"

        if audio_bytes[:4] == b'\x1a\x45\xdf\xa3':
            return ".webm"
        if audio_bytes[4:8] == b'ftyp' or audio_bytes[:12].find(b'ftyp') >= 0:
            return ".m4a"
        if audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
            return ".mp3"
        if audio_bytes[:4] == b'RIFF':
            return ".wav"
        if audio_bytes[:4] == b'fLaC':
            return ".flac"
        if audio_bytes[:4] == b'OggS':
            return ".ogg"

        return ".webm"


@app.local_entrypoint()
def main(audio_file: str = None):
    """
    Entrypoint for ephemeral, on-demand transcription.
    """
    import base64
    import json

    transcriber = ParakeetTranscriber()

    if not audio_file:
        print(json.dumps({"error": "No audio file provided.", "success": False}))
        return

    try:
        with open(audio_file, 'r') as f:
            audio_b64 = f.read()
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        print(json.dumps({"error": f"Failed to read audio file: {e}", "success": False}))
        return

    # .remote() on a @modal.method() returns the result directly (not a future)
    result_dict = transcriber.transcribe.remote(audio_bytes)
    print(json.dumps(result_dict))