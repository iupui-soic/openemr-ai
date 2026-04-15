#!/usr/bin/env python
"""
Modal-hosted Voxtral Mini 3B evaluation on the Kaggle medical speech dataset.

Mirrors the ParakeetKaggleEvaluator flow in parakeet_wer.py — reads audio from
the `medical-speech-dataset` Modal volume, transcribes on an L40S, writes a
per-file WER CSV.

Usage:
    python voxtral_mini_modal.py --kaggle --output-dir results
"""
from __future__ import annotations
import os
import argparse
import modal

app = modal.App("voxtral-kaggle-transcription")
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
SHORT = "voxtral-mini"

kaggle_volume = modal.Volume.from_name("medical-speech-dataset")

voxtral_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "transformers>=4.55",
        "torch",
        "torchaudio",
        "mistral-common",
        "librosa",
        "soundfile",
        "jiwer",
        "pandas",
        "accelerate",
    )
)


@app.cls(
    image=voxtral_image,
    gpu="L40S",
    timeout=3600,
    volumes={"/data": kaggle_volume},
    secrets=[modal.Secret.from_dict({
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
    })],
)
class VoxtralKaggleEvaluator:

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        print(f"Loading {MODEL_ID}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        )
        self.device = "cuda"
        print("Loaded")

    @modal.method()
    def evaluate_dataset(self, split: str = "validate") -> list[dict]:
        import sys, time
        import pandas as pd
        from pathlib import Path
        import jiwer

        data_dir = Path(f"/data/Medical Speech, Transcription, and Intent")
        recordings_dir = data_dir / "recordings" / split
        csv_path = data_dir / "overview-of-recordings.csv"
        df = pd.read_csv(csv_path)

        files = sorted(recordings_dir.rglob("*.wav"))
        print(f"Found {len(files)} files in split '{split}'")

        transform = jiwer.Compose([
            jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
            jiwer.RemovePunctuation(), jiwer.ToLowerCase(),
        ])

        results = []
        for i, audio in enumerate(files, 1):
            row = df[df["file_name"] == audio.name]
            if row.empty:
                continue
            ref = row.iloc[0]["phrase"]
            t0 = time.time()
            try:
                inputs = self.processor.apply_transcription_request(
                    language="en", audio=str(audio), model_id=MODEL_ID,
                ).to(self.device, dtype=self.model.dtype)
                out = self.model.generate(**inputs, max_new_tokens=512)
                decoded = self.processor.batch_decode(
                    out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                )
                hyp = (decoded[0] if decoded else "").strip()
                o = jiwer.process_words(transform(ref), transform(hyp))
                results.append({
                    "file_name": audio.name, "ground_truth": ref,
                    "transcript": hyp, "wer": o.wer, "mer": o.mer, "wil": o.wil,
                    "insertions": o.insertions, "deletions": o.deletions,
                    "substitutions": o.substitutions, "hits": o.hits,
                    "compute_s": round(time.time() - t0, 2),
                })
                if i % 20 == 0 or i <= 3:
                    print(f"  [{i}/{len(files)}] {audio.name}  WER={o.wer:.3f}")
            except Exception as e:
                print(f"  [{i}] {audio.name}: ERROR {str(e)[:100]}")
                results.append({"file_name": audio.name, "ground_truth": ref,
                                "transcript": "", "wer": None,
                                "error": str(e)[:200]})
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", action="store_true", required=True)
    ap.add_argument("--split", default="validate")
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args()

    import pandas as pd
    from pathlib import Path
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with app.run():
        ev = VoxtralKaggleEvaluator()
        results = ev.evaluate_dataset.remote(args.split)

    df = pd.DataFrame(results)
    out = Path(args.output_dir) / f"kaggle-{SHORT}.csv"
    df.to_csv(out, index=False)
    valid = df.dropna(subset=["wer"])
    if len(valid):
        print(f"\nMean WER: {valid['wer'].mean()*100:.2f}%  (n={len(valid)}/{len(df)})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
