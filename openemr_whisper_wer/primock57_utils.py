"""
Utilities for loading PriMock57 dataset for ASR evaluation.
Merges doctor and patient channels into single conversations
with concatenated reference transcripts.

Dataset: https://github.com/babylonhealth/primock57
Paper: Tseng et al. (2022) "PriMock57: A Dataset of Primary Care Mock Consultations"
"""
import re
import wave
import struct
from pathlib import Path

try:
    import textgrid
except ImportError:
    raise ImportError("pip install textgrid")


def strip_tags(text: str) -> str:
    """Remove annotation tags from PriMock57 transcripts."""
    text = re.sub(r'<UNSURE>(.*?)</UNSURE>', r'\1', text)
    text = re.sub(r'<UNIN/>', '', text)
    text = re.sub(r'<INAUDIBLE_SPEECH/>', '', text)
    return text.strip()


def merge_channels(doctor_tg_path: str, patient_tg_path: str) -> str:
    """
    Merge doctor and patient TextGrid transcripts chronologically.
    Returns a single reference transcript string.
    """
    intervals = []
    for tg_path in [doctor_tg_path, patient_tg_path]:
        tg = textgrid.TextGrid.fromFile(tg_path)
        for tier in tg:
            for interval in tier:
                text = interval.mark.strip()
                if text:
                    intervals.append((interval.minTime, strip_tags(text)))
    intervals.sort(key=lambda x: x[0])
    return ' '.join(text for _, text in intervals)


def mix_audio_channels(doctor_wav: str, patient_wav: str, output_wav: str):
    """
    Mix doctor and patient WAV channels into a single mono WAV file.
    Both inputs must be 16kHz, 16-bit, mono.
    """
    with wave.open(doctor_wav, 'rb') as d, wave.open(patient_wav, 'rb') as p:
        assert d.getframerate() == p.getframerate() == 16000, \
            f"Expected 16kHz, got doctor={d.getframerate()}, patient={p.getframerate()}"
        assert d.getsampwidth() == p.getsampwidth() == 2

        d_frames = d.readframes(d.getnframes())
        p_frames = p.readframes(p.getnframes())

    d_samples = list(struct.unpack(f'<{len(d_frames)//2}h', d_frames))
    p_samples = list(struct.unpack(f'<{len(p_frames)//2}h', p_frames))

    # Pad shorter to match
    max_len = max(len(d_samples), len(p_samples))
    d_samples.extend([0] * (max_len - len(d_samples)))
    p_samples.extend([0] * (max_len - len(p_samples)))

    # Mix (average, clamp to 16-bit range)
    mixed = [max(-32768, min(32767, (d + p) // 2)) for d, p in zip(d_samples, p_samples)]

    with wave.open(output_wav, 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(struct.pack(f'<{len(mixed)}h', *mixed))


def load_primock57_dataset(data_dir: str = "data/primock57") -> list[dict]:
    """
    Load PriMock57 as a list of conversation entries for WER evaluation.

    Returns list of dicts with keys:
        - file_name: consultation identifier (e.g., "day1_consultation01")
        - path: path to mixed WAV file
        - transcript: merged reference transcript (doctor + patient, chronological)
    """
    audio_dir = Path(data_dir) / "audio"
    transcript_dir = Path(data_dir) / "transcripts"
    mixed_dir = Path(data_dir) / "mixed_audio"
    mixed_dir.mkdir(exist_ok=True)

    if not audio_dir.exists():
        raise FileNotFoundError(f"PriMock57 audio directory not found: {audio_dir}")

    doctor_wavs = sorted(audio_dir.glob("*_doctor.wav"))
    if not doctor_wavs:
        raise FileNotFoundError(f"No doctor WAV files found in {audio_dir}")

    entries = []
    skipped = 0

    for doc_wav in doctor_wavs:
        base = doc_wav.stem.replace("_doctor", "")
        pat_wav = audio_dir / f"{base}_patient.wav"
        doc_tg = transcript_dir / f"{base}_doctor.TextGrid"
        pat_tg = transcript_dir / f"{base}_patient.TextGrid"

        if not all(p.exists() for p in [pat_wav, doc_tg, pat_tg]):
            print(f"  Skipping {base}: missing files")
            skipped += 1
            continue

        mixed_wav = mixed_dir / f"{base}_mixed.wav"
        if not mixed_wav.exists():
            mix_audio_channels(str(doc_wav), str(pat_wav), str(mixed_wav))

        transcript = merge_channels(str(doc_tg), str(pat_tg))

        entries.append({
            "file_name": base,
            "path": str(mixed_wav),
            "transcript": transcript,
        })

    print(f"Loaded {len(entries)} PriMock57 consultations ({skipped} skipped)")
    return entries
