import os
import modal

app = modal.App("medical-dataset-loader")

dataset_volume = modal.Volume.from_name(
    "medical-speech-dataset",
    create_if_missing=True
)

dataset_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "kaggle",
    "pandas",
)

# Check environment variables before creating secret
_kaggle_username = os.environ.get("KAGGLE_USERNAME")
_kaggle_key = os.environ.get("KAGGLE_KEY")

if not _kaggle_username or not _kaggle_key:
    raise ValueError(
        "KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set!\n"
        "Run with: KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx modal run modal_dataset_volume.py"
    )

# Create secret from environment variables (passed from GitHub Actions)
kaggle_secret = modal.Secret.from_dict({
    "KAGGLE_USERNAME": _kaggle_username,
    "KAGGLE_KEY": _kaggle_key,
})


@app.function(
    image=dataset_image,
    volumes={"/data": dataset_volume},
    secrets=[kaggle_secret],
    timeout=1800,
)
def download_dataset():
    """Downloads the Kaggle medical speech dataset to the persistent volume."""
    import os
    import subprocess
    import zipfile
    from pathlib import Path

    data_dir = Path("/data")
    dataset_marker = data_dir / ".dataset_downloaded"

    if dataset_marker.exists():
        print("Dataset already downloaded to volume!")
        return {"status": "already_exists", "path": str(data_dir)}

    print("=== Downloading Medical Speech Dataset to Modal Volume ===\n")

    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    print(f"✓ Kaggle credentials found for user: {kaggle_username}")

    print("\nDownloading dataset from Kaggle...")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "paultimothymooney/medical-speech-transcription-and-intent",
            "-p", str(data_dir)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")

    print(result.stdout)
    print("✓ Download complete!")

    zip_file = data_dir / "medical-speech-transcription-and-intent.zip"
    if zip_file.exists():
        print("\n📦 Extracting dataset...")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(data_dir)
        print("✓ Extraction complete!")
        zip_file.unlink()
        print("✓ Cleaned up zip file")

    dataset_marker.touch()
    dataset_volume.commit()

    print("\n=== Dataset Ready! ===")
    return {"status": "downloaded", "path": str(data_dir)}


@app.function(
    image=dataset_image,
    volumes={"/data": dataset_volume},
)
def list_volume_contents():
    """Lists the contents of the persistent volume."""
    import os
    from pathlib import Path

    data_dir = Path("/data")
    print(f"=== Volume Contents: {data_dir} ===\n")

    if not data_dir.exists():
        print("Volume is empty")
        return {"status": "empty"}

    total_files = 0
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(str(data_dir), '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        for file in files[:3]:
            print(f"{'  ' * (level + 1)}📄 {file}")
            total_files += 1
        if len(files) > 3:
            print(f"{'  ' * (level + 1)}... and {len(files) - 3} more files")
            total_files += len(files) - 3

    print(f"\nTotal files: {total_files}")
    return {"status": "success", "total_files": total_files}


@app.function(
    image=dataset_image,
    volumes={"/data": dataset_volume},
)
def get_audio_files(split: str = "validate"):
    """
    Returns a list of audio file paths from the dataset.

    Args:
        split: Which split to get files from ('validate', 'train', etc.)

    Returns:
        List of audio file paths within the volume
    """
    import pandas as pd
    from pathlib import Path

    data_dir = Path("/data/Medical Speech, Transcription, and Intent")
    recordings_dir = data_dir / "recordings" / split
    csv_path = data_dir / "overview-of-recordings.csv"

    if not recordings_dir.exists():
        return {"error": f"Split '{split}' not found", "available": []}

    # Get audio files
    audio_files = list(recordings_dir.rglob("*.wav"))

    # Load CSV for transcripts
    df = pd.read_csv(csv_path)

    # Match audio files with transcripts
    results = []
    for audio_path in audio_files:
        file_name = audio_path.name
        transcript_row = df[df['file_name'] == file_name]

        if not transcript_row.empty:
            results.append({
                "file_name": file_name,
                "path": str(audio_path),
                "transcript": transcript_row['phrase'].iloc[0],
                "prompt": transcript_row['prompt'].iloc[0] if 'prompt' in transcript_row.columns else None
            })

    print(f"Found {len(results)} audio files with transcripts in '{split}' split")

    return {
        "split": split,
        "count": len(results),
        "files": results[:10],  # Return first 10 as sample
        "total_available": len(results)
    }


@app.local_entrypoint()
def main():
    """Main entrypoint - downloads dataset and verifies."""
    print("=== Starting Dataset Download ===\n")
    download_dataset.remote()
    print("\n=== Verifying Volume Contents ===\n")
    list_volume_contents.remote()