"""
Utilities for loading Fareez OSCE dataset for ASR evaluation.
272 simulated patient-physician conversations (~55 hours).
Audio: MP3. Transcripts: TXT (manually corrected).

Dataset: https://springernature.figshare.com/collections/5545842
Paper: Fareez et al. (2022) "A dataset of simulated patient-physician
       medical interviews with a focus on respiratory cases"
"""
import subprocess
from pathlib import Path


def convert_mp3_to_wav(mp3_path: str, wav_path: str):
    """Convert MP3 to 16kHz mono WAV using FFmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-i", mp3_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        wav_path
    ], capture_output=True, check=True)


def load_fareez_dataset(data_dir: str = "data/fareez_osce") -> list[dict]:
    """
    Load Fareez OSCE dataset as a list of conversation entries for WER evaluation.

    Returns list of dicts with keys:
        - file_name: conversation ID (e.g., "RES0001")
        - path: path to WAV file (converted from MP3)
        - transcript: reference transcript text
        - category: medical specialty (RES, CAR, GAS, MSK, DER)
    """
    data_path = Path(data_dir)
    wav_dir = data_path / "wav_16khz"
    wav_dir.mkdir(exist_ok=True)

    mp3_files = sorted(data_path.rglob("*.mp3"))
    if not mp3_files:
        raise FileNotFoundError(f"No MP3 files found in {data_dir}. Check extraction path.")

    # Build a lookup of all .txt files by stem (transcripts may be in a separate dir)
    txt_lookup = {}
    for txt_file in data_path.rglob("*.txt"):
        txt_lookup[txt_file.stem] = txt_file

    entries = []
    skipped = 0
    for mp3_path in mp3_files:
        base = mp3_path.stem

        # Find matching transcript from lookup
        txt_path = txt_lookup.get(base)
        if not txt_path:
            print(f"  Skipping {base}: no transcript found")
            skipped += 1
            continue

        # Convert MP3 to WAV
        wav_path = wav_dir / f"{base}.wav"
        if not wav_path.exists():
            try:
                convert_mp3_to_wav(str(mp3_path), str(wav_path))
            except subprocess.CalledProcessError:
                print(f"  Skipping {base}: FFmpeg conversion failed")
                skipped += 1
                continue

        # Some files have BOM or non-UTF-8 encoding
        try:
            transcript = txt_path.read_text(encoding='utf-8-sig').strip()
        except UnicodeDecodeError:
            transcript = txt_path.read_text(encoding='latin-1').strip()
        category = base[:3] if base[:3] in ('RES', 'CAR', 'GAS', 'MSK', 'DER') else 'UNK'

        entries.append({
            "file_name": base,
            "path": str(wav_path),
            "transcript": transcript,
            "category": category,
        })

    print(f"Loaded {len(entries)} Fareez OSCE conversations ({skipped} skipped)")
    categories = {}
    for e in entries:
        categories[e['category']] = categories.get(e['category'], 0) + 1
    print(f"  Categories: {categories}")
    return entries
