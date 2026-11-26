"""
Speech LLM WER Calculator (Qwen3-ASR, Granite Speech, Ishiki ASR)

Transcribes with an open-weight speech LLM on Modal (ephemeral) and calculates
Word Error Rate, in the same three modes as the other scripts in this folder:
the Notion recordings, a local dataset, or the Kaggle medical speech dataset.

Models (--model):
    qwen3-asr       Qwen/Qwen3-ASR-1.7B-hf             Alibaba, 2026-01, Apache 2.0
    granite-speech  ibm-granite/granite-speech-5.0-2b  IBM, 2026-10, Apache 2.0
    ishiki-asr      ishiki-labs/ishiki-asr-1.0         Ishiki Labs, 2026-11, CC BY 4.0

Every result file has two scores per recording:
    wer       the score of the other scripts in this folder (lower case, punctuation removed)
    wer_norm  WER after the Whisper English text normalizer on both texts, which also
              writes numbers as digits, expands contractions and drops fillers such as "um"
Ishiki ASR returns text that is already in that normalized form ("i have" for
"I've", "3" for "three"), so its `wer` counts those rewrites as errors and
`wer_norm` is the score to compare it on.

Usage:
    python speech_llm_wer.py --model qwen3-asr --output results.csv
    python speech_llm_wer.py --model granite-speech --kaggle --output-dir ./results
    python speech_llm_wer.py --model qwen3-asr --kaggle --limit 20   # quick check
    python speech_llm_wer.py --model qwen3-asr --local-dataset primock57 --output results/primock57-qwen3-asr.csv
    python speech_llm_wer.py --model qwen3-asr --local-dataset fareez --workers 2 --output results/fareez-qwen3-asr.csv

Requirements:
    pip install modal jiwer pandas requests notion-client httpx whisper-normalizer textgrid
    (the local datasets also need ffmpeg on the path)
"""

import os
import re

import modal

# Local-only imports (not needed in Modal container)
if not os.environ.get("MODAL_IS_REMOTE"):
    from wer_utils import (
        NotionFetcher,
        WERCalculator,
        calculate_and_save_results,
    )

# ============================================================================
# Models
# ============================================================================

# max_chunk_s is the longest audio given to the model as one piece. A longer
# recording is cut at quiet points and the pieces are transcribed in order.
#   qwen3-asr: 1200 s is the single-call limit of the model's own toolkit
#     (MAX_ASR_INPUT_SECONDS in QwenLM/Qwen3-ASR).
#   granite-speech: the model card asks for segments of up to about 2 minutes.
# Pieces are transcribed in batches. A batch is padded to its longest piece and
# holds at most batch_max pieces and batch_audio_s seconds of padded audio.
MODELS = {
    "qwen3-asr": {
        "model_id": "Qwen/Qwen3-ASR-1.7B-hf",
        "max_chunk_s": 1200,
        "batch_audio_s": 3000,
        "batch_max": 16,
    },
    "granite-speech": {
        "model_id": "ibm-granite/granite-speech-5.0-2b",
        "max_chunk_s": 100,
        "batch_audio_s": 1200,
        "batch_max": 12,
    },
    # Seven speech models in one checkpoint (13.3 B parameters) with a router that
    # merges their outputs. It cuts long audio at silences itself. The model runs
    # Python code from its repository, so the revision is fixed, and its container
    # has no network access (see IshikiASR). A container without network access
    # cannot receive a large call (64 MiB worked, 96 MiB did not), hence
    # max_call_bytes; local recordings are sent to it as FLAC, which is lossless
    # and about half the size. batch_size is the number of its pieces of audio
    # (about 22 s each) transcribed at once. Its GPU memory also grows with the
    # audio it is given in one generate call (about 30 GiB for 10 minutes and
    # 42 GiB for 40 minutes on a 44 GiB card), so recordings are given to it in
    # sets of at most generate_audio_s seconds.
    "ishiki-asr": {
        "model_id": "ishiki-labs/ishiki-asr-1.0",
        "revision": "902d5c85ad46066c214895c0b8f2200c1800661a",
        "batch_size": 64,
        "generate_audio_s": 1300,
        "max_call_bytes": 48 * 1024 * 1024,
    },
}

GRANITE_PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."

# ============================================================================
# Modal App - Speech LLM Transcription (Ephemeral)
# ============================================================================

app = modal.App("speech-llm-transcription")

# Reference the Kaggle dataset volume (for --kaggle mode)
kaggle_volume = modal.Volume.from_name("medical-speech-dataset")

speech_llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers>=5.13.0",  # Qwen3-ASR needs 5.13 or newer
        "accelerate",
        "peft",
        "soundfile",
        "librosa",
        "sentencepiece",
        "protobuf",
        "numpy",
        "scipy",
        "jiwer",
        "pandas",
    )
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "wer_utils.py"),
        "/root/wer_utils.py",
    )
)

# The Ishiki weights (24.8 GiB) are downloaded while the image is built, because
# the container that runs the model has no network access.
ishiki_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers>=5.24.1",  # the model's code imports the 5.24 model classes
        "accelerate",
        "peft",
        "soundfile",
        "librosa",
        "sentencepiece",
        "protobuf",
        "numpy",
        "scipy",
        "jiwer",
        "pandas",
        "rapidfuzz",
        "huggingface_hub",
    )
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODELS['ishiki-asr']['model_id']}', "
        f"revision='{MODELS['ishiki-asr']['revision']}', local_dir='/model')\""
    )
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "wer_utils.py"),
        "/root/wer_utils.py",
    )
)


def load_mono_16k(audio_path: str):
    """Read an audio file as mono 16 kHz float32."""
    import numpy as np
    import soundfile as sf
    from scipy import signal

    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = signal.resample_poly(audio, 16000, sr)
    return np.ascontiguousarray(audio, dtype=np.float32)


def decode_audio_bytes(audio_bytes: bytes):
    """Decode audio file bytes (m4a, mp3, wav, ...) to mono 16 kHz float32."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
        f.write(audio_bytes)
        input_path = f.name
    output_path = input_path + "_converted.wav"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path
        ], check=True, capture_output=True)
        return load_mono_16k(output_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")
    finally:
        for path in (input_path, output_path):
            if os.path.exists(path):
                os.unlink(path)


def evaluate_kaggle(transcribe_audio, split: str = "validate", limit: int = 0) -> list[dict]:
    """
    Evaluate on the Kaggle dataset, inside a container that has the dataset volume.

    transcribe_audio takes mono 16 kHz audio and returns its text. limit > 0
    takes the first files only.
    """
    import sys
    sys.path.insert(0, "/root")
    from wer_utils import load_kaggle_dataset, calculate_wer_metrics

    entries = load_kaggle_dataset(split)
    if limit > 0:
        entries = entries[:limit]
    results = []

    for i, entry in enumerate(entries):
        print(f"[{i+1}/{len(entries)}] {entry['file_name']}")
        try:
            transcript = transcribe_audio(load_mono_16k(entry["path"]))

            metrics = calculate_wer_metrics(entry["transcript"], transcript)
            print(f"  WER: {metrics['wer']:.2%}")

            results.append({
                "name": entry["file_name"],
                "ground_truth": entry["transcript"],
                "transcript": transcript,
                **metrics
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "name": entry["file_name"],
                "ground_truth": entry["transcript"],
                "transcript": "",
                "wer": 1.0,
                "error": str(e)
            })

    return results


def split_at_quiet_points(audio, sample_rate: int, max_chunk_s: float, search_s: float = 15.0, frame_s: float = 0.3):
    """
    Cut audio into pieces no longer than max_chunk_s.

    Each cut is placed in the quietest frame of the last search_s seconds of the
    piece, so that it falls in a pause and not inside a word. The pieces, joined,
    are the original audio.
    """
    max_len = int(max_chunk_s * sample_rate)
    frame = int(frame_s * sample_rate)
    chunks = []
    start = 0
    while len(audio) - start > max_len:
        end = start + max_len
        lo = max(start + frame, end - int(search_s * sample_rate))
        window = audio[lo:end]
        n_frames = len(window) // frame
        energy = (window[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1)
        cut = lo + int(energy.argmin()) * frame + frame // 2
        chunks.append(audio[start:cut])
        start = cut
    chunks.append(audio[start:])
    return chunks


def plan_batches(lengths: list[int], sample_rate: int, batch_audio_s: float, batch_max: int, min_fill: float = 0.8):
    """
    Group pieces of audio into batches, longest first. Returns lists of indices.

    A batch is padded to its longest piece. It takes no piece shorter than
    min_fill of that length, so little of a batch is padding.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, current = [], []
    for i in order:
        if current:
            longest = lengths[current[0]]
            full = len(current) >= batch_max or (len(current) + 1) * longest > batch_audio_s * sample_rate
            if full or lengths[i] < min_fill * longest:
                batches.append(current)
                current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


@app.cls(
    image=speech_llm_image,
    gpu="A10G",
    timeout=3600,
    volumes={"/data": kaggle_volume},
)
class SpeechLLM:
    """One of the models in MODELS, chosen by model_key."""

    model_key: str = modal.parameter()

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoProcessor

        self.config = MODELS[self.model_key]
        model_id = self.config["model_id"]
        print(f"Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        if self.model_key == "qwen3-asr":
            from transformers import AutoModelForMultimodalLM

            self.model = AutoModelForMultimodalLM.from_pretrained(
                model_id, device_map="cuda", dtype=torch.bfloat16
            )
        else:
            from transformers import AutoModelForSpeechSeq2Seq

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, device_map="cuda", dtype=torch.bfloat16
            )
        self.model.eval()
        # A batch of prompts is padded on the left, so that every row ends where
        # generation starts.
        self.processor.tokenizer.padding_side = "left"
        # A first matrix product sets up the GPU's linear algebra library. Without
        # it, a first batch that takes most of the memory leaves no room for
        # that setup and fails (CUBLAS_STATUS_ALLOC_FAILED).
        warm = torch.ones(8, 8, device="cuda", dtype=torch.bfloat16)
        (warm @ warm).sum().item()
        print("Model loaded!")

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _max_new_tokens(self, n_samples: int) -> int:
        # About 8 tokens per second of audio is well above fast speech, so the
        # limit does not cut a transcript short, and a model that starts to
        # repeat itself still stops.
        return min(8192, 128 + int(n_samples / 16000 * 8))

    def _transcribe_pieces(self, pieces: list) -> list[str]:
        """Transcribe pieces of mono 16 kHz audio, each no longer than max_chunk_s, as one batch."""
        import torch

        max_new_tokens = self._max_new_tokens(max(len(p) for p in pieces))

        if self.model_key == "qwen3-asr":
            import os
            import tempfile

            import soundfile as sf

            wav_paths = []
            try:
                for piece in pieces:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        wav_paths.append(f.name)
                    sf.write(wav_paths[-1], piece, 16000)
                inputs = self.processor.apply_transcription_request(
                    audio=wav_paths, language="English",
                ).to(self.model.device, self.model.dtype)
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False
                    )
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                texts = self.processor.decode(generated, return_format="transcription_only")
                return [t.strip() for t in texts]
            finally:
                for wav_path in wav_paths:
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)

        # granite-speech
        tokenizer = self.processor.tokenizer
        chat = [{"role": "user", "content": GRANITE_PROMPT}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if len(pieces) == 1:
            wavs = torch.from_numpy(pieces[0]).unsqueeze(0)
        else:
            wavs = [torch.from_numpy(p) for p in pieces]
        inputs = self.processor([prompt] * len(pieces), wavs, device="cuda", return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
        texts = tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        return [t.strip() for t in texts]

    def _transcribe_many(self, audios: list) -> list[str]:
        """Transcribe recordings of mono 16 kHz audio of any length."""
        import time

        import torch

        started = time.time()
        pieces, owners = [], []
        for i, audio in enumerate(audios):
            for piece in split_at_quiet_points(audio, 16000, self.config["max_chunk_s"]):
                pieces.append(piece)
                owners.append(i)

        texts = [""] * len(pieces)
        batches = plan_batches(
            [len(p) for p in pieces], 16000, self.config["batch_audio_s"], self.config["batch_max"]
        )
        for batch in batches:
            try:
                out = self._transcribe_pieces([pieces[i] for i in batch])
            except RuntimeError as e:
                # torch.cuda.OutOfMemoryError is a RuntimeError; a failed
                # allocation inside a CUDA library is reported as a plain one.
                if len(batch) == 1 or not ("out of memory" in str(e) or "ALLOC_FAILED" in str(e)):
                    raise
                # Too much for one batch: transcribe its pieces one at a time.
                print(f"  A batch of {len(batch)} pieces did not fit in GPU memory, transcribing them one at a time")
                torch.cuda.empty_cache()
                out = []
                for i in batch:
                    out.append(self._transcribe_pieces([pieces[i]])[0])
                    torch.cuda.empty_cache()
            for i, text in zip(batch, out):
                texts[i] = text

        seconds = sum(len(a) for a in audios) / 16000
        if seconds > 60:
            print(f"  {seconds:.0f} s of audio in {len(audios)} recording(s), {len(pieces)} piece(s), "
                  f"{len(batches)} batch(es), transcribed in {time.time() - started:.0f} s, "
                  f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        return [
            " ".join(t for t, owner in zip(texts, owners) if owner == i and t)
            for i in range(len(audios))
        ]

    def _transcribe_audio(self, audio) -> str:
        """Transcribe mono 16 kHz audio of any length."""
        return self._transcribe_many([audio])[0]

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file bytes (m4a, mp3, wav, etc.)

        Returns:
            Transcribed text
        """
        return self._transcribe_audio(decode_audio_bytes(audio_bytes))

    @modal.method()
    def transcribe_batch(self, audio_files: list[bytes]) -> list[str]:
        """Transcribe several audio files in one call. Returns one text per file, in order."""
        return self._transcribe_many([decode_audio_bytes(b) for b in audio_files])

    @modal.method()
    def evaluate_dataset(self, split: str = "validate", limit: int = 0) -> list[dict]:
        """Evaluate the model on the Kaggle dataset. limit > 0 takes the first files only."""
        return evaluate_kaggle(self._transcribe_audio, split, limit)


@app.cls(
    image=ishiki_image,
    gpu="L40S",
    timeout=3600,
    volumes={"/data": kaggle_volume},
    block_network=True,
)
class IshikiASR:
    """
    ishiki-labs/ishiki-asr-1.0.

    Loading the model runs Python code from its repository (trust_remote_code).
    That code is fixed to one revision, the weights are already in the image, and
    the container has no network access and no secrets, so audio cannot leave it.
    """

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.config = MODELS["ishiki-asr"]
        print(f"Loading {self.config['model_id']} at {self.config['revision'][:7]}...")
        self.processor = AutoProcessor.from_pretrained("/model", trust_remote_code=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "/model", dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda:0").eval()
        print("Model loaded!")

    def _generate(self, audios: list) -> list[str]:
        """One generate call for a set of recordings."""
        import torch

        inputs = self.processor(
            audios if len(audios) > 1 else audios[0], sampling_rate=16000, return_tensors="pt",
            padding="longest", return_attention_mask=True,
        )
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, batch_size=self.config["batch_size"])
        return [t.strip() for t in self.processor.batch_decode(output_ids, skip_special_tokens=True)]

    def _transcribe_many(self, audios: list) -> list[str]:
        """Transcribe recordings of mono 16 kHz audio of any length."""
        import time

        import torch

        started = time.time()
        limit = self.config["generate_audio_s"] * 16000
        sets = []
        for i, audio in enumerate(audios):
            if sets and sum(len(audios[j]) for j in sets[-1]) + len(audio) <= limit:
                sets[-1].append(i)
            else:
                sets.append([i])

        texts = [""] * len(audios)
        for indices in sets:
            for i, text in zip(indices, self._generate([audios[i] for i in indices])):
                texts[i] = text
            torch.cuda.empty_cache()

        seconds = sum(len(a) for a in audios) / 16000
        if seconds > 60:
            print(f"  {seconds:.0f} s of audio in {len(audios)} recording(s), {len(sets)} generate call(s), transcribed in "
                  f"{time.time() - started:.0f} s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        return texts

    def _transcribe_audio(self, audio) -> str:
        """Transcribe mono 16 kHz audio of any length."""
        return self._transcribe_many([audio])[0]

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes (m4a, mp3, wav, etc.) to text."""
        return self._transcribe_audio(decode_audio_bytes(audio_bytes))

    @modal.method()
    def transcribe_batch(self, audio_files: list[bytes]) -> list[str]:
        """Transcribe several audio files in one call. Returns one text per file, in order."""
        return self._transcribe_many([decode_audio_bytes(b) for b in audio_files])

    @modal.method()
    def evaluate_dataset(self, split: str = "validate", limit: int = 0) -> list[dict]:
        """Evaluate the model on the Kaggle dataset. limit > 0 takes the first files only."""
        return evaluate_kaggle(self._transcribe_audio, split, limit)


def make_transcriber(model_key: str):
    """The Modal class instance that runs the model. Call inside app.run()."""
    if model_key == "ishiki-asr":
        return IshikiASR()
    return SpeechLLM(model_key=model_key)


_english_normalizer = None


def normalized_wer(reference: str, hypothesis: str) -> float:
    """
    WER after the Whisper English text normalizer is applied to both texts.

    The normalizer lower-cases, removes punctuation, writes numbers as digits,
    expands contractions, unifies British and American spellings and drops
    fillers such as "um". It is the usual normalization for comparing ASR
    models whose outputs differ in formatting.
    """
    global _english_normalizer
    import jiwer
    if _english_normalizer is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _english_normalizer = EnglishTextNormalizer()
    ref = _english_normalizer(reference)
    hyp = _english_normalizer(hypothesis)
    if not ref:
        return float("nan")
    return jiwer.wer(ref, hyp)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(
        model_key: str,
        database_id: str,
        output_csv: str = "results.csv",
        error_report_path: str = "error_analysis.txt",
):
    """
    Run the complete WER pipeline on the Notion recordings.

    Args:
        model_key: A key of MODELS
        database_id: Notion database ID
        output_csv: Path for results CSV
        error_report_path: Path for detailed error analysis report
    """
    model_id = MODELS[model_key]["model_id"]

    print("=" * 60)
    print("Speech LLM WER Pipeline")
    print("=" * 60)

    print(f"Model: {model_id}")

    # Initialize
    print("\n[1/5] Fetching entries from Notion...")
    fetcher = NotionFetcher()
    entries = fetcher.get_entries(database_id)
    print(f"  Found {len(entries)} entries")

    if not entries:
        print("No entries found!")
        return

    wer_calc = WERCalculator()
    results = []
    wer_norm = {}

    # Run Modal transcription (ephemeral)
    print("\n[2/5] Starting transcription...")

    with app.run():
        transcriber = make_transcriber(model_key)

        for i, entry in enumerate(entries):
            name = entry["name"]
            print(f"\n  [{i+1}/{len(entries)}] {name}")

            try:
                # Download
                print(f"    Downloading audio...")
                audio_bytes = fetcher.download_audio(entry["audio_url"])
                print(f"    Downloaded {len(audio_bytes):,} bytes")

                # Transcribe
                print(f"    Transcribing...")
                transcript = transcriber.transcribe.remote(audio_bytes)

                # Calculate WER
                metrics = wer_calc.calculate(entry["ground_truth"], transcript)
                wer_norm[name] = normalized_wer(entry["ground_truth"], transcript)
                print(f"    WER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%), "
                      f"normalized {wer_norm[name]*100:.2f}%")

                results.append({
                    "name": name,
                    "ground_truth": entry["ground_truth"],
                    "transcript": transcript,
                    **metrics
                })

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({
                    "name": name,
                    "ground_truth": entry.get("ground_truth", ""),
                    "transcript": "",
                    "wer": 1.0,
                    "error": str(e)
                })

    # Calculate and save results
    summary = calculate_and_save_results(
        results=results,
        wer_calc=wer_calc,
        model_name=model_id,
        output_csv=output_csv,
        error_report_path=error_report_path,
    )

    # The shared writer keeps its own columns, so the second score is added here.
    import pandas as pd
    df = pd.read_csv(output_csv)
    df["wer_norm"] = df["name"].map(wer_norm)
    df.to_csv(output_csv, index=False)
    if wer_norm:
        print(f"Average normalized WER: {100 * sum(wer_norm.values()) / len(wer_norm):.2f}%")
    return summary


# ============================================================================
# Local Dataset Pipeline (PriMock57 / Fareez OSCE)
# ============================================================================

# Fareez references begin every turn with "D:" or "P:". The labels are not spoken,
# so they are removed before scoring (the same pattern as
# scripts/revision_asr_fareez_wer.py, which the README's Fareez column uses).
FAREEZ_SPEAKER_LABEL = re.compile(r"(?:(?<=\s)|^)[DP]:(?=\s|$)")


def load_local_entries(dataset_name: str) -> list[dict]:
    """Load PriMock57 or Fareez OSCE as entries with file_name, path and transcript."""
    if dataset_name == "primock57":
        from primock57_utils import load_primock57_dataset
        return load_primock57_dataset("data/primock57")

    from fareez_utils import load_fareez_dataset
    entries = []
    for entry in load_fareez_dataset("data/fareez_osce"):
        if "\x00" in entry["transcript"]:
            # RES0002 and RES0054 are UTF-16 files that the loader does not read
            # correctly. They are left out, as in the README's Fareez column.
            print(f"  Skipping {entry['file_name']}: the reference transcript is not readable")
            continue
        entry["transcript"] = " ".join(FAREEZ_SPEAKER_LABEL.sub(" ", entry["transcript"]).split())
        entries.append(entry)
    return entries


def run_local_pipeline(
        model_key: str,
        dataset_name: str,
        output_csv: str = "results.csv",
        workers: int = 1,
        group_size: int = 4,
        limit: int = 0,
):
    """
    Run WER evaluation on a local dataset (PriMock57 or Fareez OSCE).

    The result file is written after every call to the model. If it already
    exists, the recordings it holds a transcript for are not transcribed again,
    so an interrupted run can be started again with the same command.

    Args:
        model_key: A key of MODELS
        dataset_name: 'primock57' or 'fareez'
        output_csv: Path for results CSV
        workers: Groups transcribed at the same time, one GPU container each
        group_size: Recordings sent to the model in one call, which batches them
        limit: If > 0, evaluate the first recordings only
    """
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import pandas as pd

    model_id = MODELS[model_key]["model_id"]

    entries = load_local_entries(dataset_name)
    if limit > 0:
        entries = entries[:limit]

    print("=" * 60)
    print(f"Speech LLM Local Dataset WER Evaluation ({dataset_name})")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Entries: {len(entries)}")

    # Results of an earlier, interrupted run
    results = {}
    if Path(output_csv).exists():
        for row in pd.read_csv(output_csv).to_dict("records"):
            has_text = isinstance(row.get("transcript"), str) and row["transcript"].strip()
            failed = isinstance(row.get("error"), str) and row["error"]
            if has_text and not failed:
                row = {k: v for k, v in row.items() if k != "error"}
                if not isinstance(row.get("wer_norm"), float) or row["wer_norm"] != row["wer_norm"]:
                    row["wer_norm"] = normalized_wer(row["ground_truth"], row["transcript"])
                results[row["name"]] = row
    todo = [e for e in entries if e["file_name"] not in results]
    if results:
        print(f"Already in {output_csv}: {len(entries) - len(todo)}, to transcribe: {len(todo)}")
    print()

    wer_calc = WERCalculator()
    lock = threading.Lock()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    def save():
        # In dataset order, then any rows of the file that this run did not load
        # (a run with --limit must not shorten an existing file).
        names = [e["file_name"] for e in entries]
        names += [n for n in results if n not in set(names)]
        pd.DataFrame([results[n] for n in names if n in results]).to_csv(output_csv, index=False)

    max_bytes = MODELS[model_key].get("max_call_bytes")

    def payload(entry) -> bytes:
        if max_bytes is None:
            return Path(entry["path"]).read_bytes()
        # Lossless and about half the size of the WAV file
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-i", entry["path"], "-c:a", "flac", "-f", "flac", "pipe:1"],
            check=True, capture_output=True,
        ).stdout

    def transcribe_files(files: list[bytes]) -> list[str]:
        if max_bytes is not None and len(files) > 1 and sum(len(f) for f in files) > max_bytes:
            half = len(files) // 2
            return transcribe_files(files[:half]) + transcribe_files(files[half:])
        return transcriber.transcribe_batch.remote(files)

    def transcribe_group(group):
        try:
            texts = transcribe_files([payload(e) for e in group])
            error = None
        except Exception as e:
            texts, error = [""] * len(group), str(e)
        with lock:
            for entry, transcript in zip(group, texts):
                name = entry["file_name"]
                if error is None:
                    metrics = wer_calc.calculate(entry["transcript"], transcript)
                    results[name] = {
                        "name": name,
                        "ground_truth": entry["transcript"],
                        "transcript": transcript,
                        **metrics,
                        "wer_norm": normalized_wer(entry["transcript"], transcript),
                    }
                    message = (f"WER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%), "
                               f"normalized {results[name]['wer_norm']*100:.2f}%")
                else:
                    results[name] = {
                        "name": name,
                        "ground_truth": entry["transcript"],
                        "transcript": "",
                        "wer": 1.0,
                        "error": error
                    }
                    message = f"ERROR: {error}"
                print(f"  [{len(results)}/{len(entries)}] {name}  {message}")
            save()

    if todo:
        # Recordings of similar length go to the model together, so that little
        # of a batch is padding.
        todo.sort(key=lambda e: -os.path.getsize(e["path"]))
        groups = []
        for entry in todo:
            # A FLAC file is about half the size of the WAV file.
            fits = groups and len(groups[-1]) < max(1, group_size) and (
                max_bytes is None
                or sum(os.path.getsize(e["path"]) for e in groups[-1] + [entry]) * 0.5 <= max_bytes
            )
            if fits:
                groups[-1].append(entry)
            else:
                groups.append([entry])
        with app.run():
            transcriber = make_transcriber(model_key)
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                list(pool.map(transcribe_group, groups))

    save()
    rows = [results[e["file_name"]] for e in entries if e["file_name"] in results]
    valid = [r for r in rows if not r.get("error")]
    avg_wer = sum(r["wer"] for r in valid) / len(valid) if valid else 1.0
    avg_wer_norm = sum(r["wer_norm"] for r in valid) / len(valid) if valid else 1.0

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY - {model_key} on {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Entries: {len(rows)} total, {len(valid)} successful")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Average normalized WER: {avg_wer_norm*100:.2f}%")
    print(f"Results saved to: {output_csv}")

    return {
        "model": model_key,
        "model_id": model_id,
        "dataset": dataset_name,
        "avg_wer": avg_wer,
        "samples": len(valid),
        "total": len(rows),
        "output_csv": output_csv,
    }


# ============================================================================
# Kaggle Pipeline
# ============================================================================

def run_kaggle_pipeline(
        model_key: str,
        split: str = "validate",
        output_dir: str = ".",
        limit: int = 0,
):
    """
    Run WER evaluation on the Kaggle medical speech dataset.

    Args:
        model_key: A key of MODELS
        split: Dataset split to use ('validate' or 'train')
        output_dir: Directory for output CSV files
        limit: If > 0, evaluate the first files only
    """
    import pandas as pd

    model_id = MODELS[model_key]["model_id"]

    print("=" * 60)
    print("Speech LLM Kaggle Dataset WER Evaluation")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Split: {split}")
    print()

    with app.run():
        evaluator = make_transcriber(model_key)
        results = evaluator.evaluate_dataset.remote(split, limit)

    for r in results:
        if not r.get("error"):
            r["wer_norm"] = normalized_wer(r["ground_truth"], r["transcript"])

    # Save results
    output_csv = os.path.join(output_dir, f"kaggle-results-{model_key}.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Calculate summary
    valid = [r for r in results if "error" not in r or not r.get("error")]
    if valid:
        avg_wer = sum(r["wer"] for r in valid) / len(valid)
    else:
        avg_wer = 1.0

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY - {model_key}")
    print(f"{'=' * 60}")
    print(f"Entries: {len(results)} total, {len(valid)} successful")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    if valid:
        print(f"Average normalized WER: {100 * sum(r['wer_norm'] for r in valid) / len(valid):.2f}%")
    print(f"Results saved to: {output_csv}")

    return {
        "model": model_key,
        "model_id": model_id,
        "avg_wer": avg_wer,
        "samples": len(valid),
        "total": len(results),
        "output_csv": output_csv,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate WER using an open-weight speech LLM with detailed error analysis"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS),
        help="Model to evaluate",
    )
    parser.add_argument(
        "--database-id",
        default="294a6166c4978050930fea2073e66dc2",
        help="Notion database ID (for Notion mode)",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV path (for Notion mode)",
    )
    parser.add_argument(
        "--error-report",
        default="error_analysis.txt",
        help="Error analysis report path (for Notion mode)",
    )
    # Kaggle mode arguments
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Evaluate on Kaggle medical speech dataset instead of Notion",
    )
    parser.add_argument(
        "--split",
        default="validate",
        choices=["validate", "train"],
        help="Kaggle dataset split to use (default: validate)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for Kaggle results (default: current directory)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Kaggle and local dataset modes: evaluate the first N files only (default: all)",
    )
    # Local dataset mode
    parser.add_argument(
        "--local-dataset",
        choices=["primock57", "fareez"],
        help="Evaluate on a local dataset (primock57 or fareez)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Local dataset mode: groups transcribed at the same time, one GPU container each (default: 1)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Local dataset mode: recordings sent to the model in one call (default: 4)",
    )

    args = parser.parse_args()

    if args.local_dataset:
        run_local_pipeline(
            model_key=args.model,
            dataset_name=args.local_dataset,
            output_csv=args.output,
            workers=args.workers,
            group_size=args.group_size,
            limit=args.limit,
        )
    elif args.kaggle:
        run_kaggle_pipeline(
            model_key=args.model,
            split=args.split,
            output_dir=args.output_dir,
            limit=args.limit,
        )
    else:
        run_pipeline(
            model_key=args.model,
            database_id=args.database_id,
            output_csv=args.output,
            error_report_path=args.error_report,
        )


if __name__ == "__main__":
    main()
