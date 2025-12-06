"""
Modal Persistent Volume Setup for Vector Database

This script uploads a local vector database to a Modal persistent volume.

Usage:
    modal run vectordb_volume.py --local-path /path/to/vectorDB
"""

import modal

app = modal.App("vectordb-loader")

# Create persistent volume for vector database
vectordb_volume = modal.Volume.from_name(
    "medical-vectordb",
    create_if_missing=True
)

base_image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=base_image,
    volumes={"/vectordb": vectordb_volume},
    timeout=1800,
)
def upload_vectordb(file_data: dict):
    """Receives file data and writes to the volume."""
    from pathlib import Path

    file_path = Path("/vectordb") / file_data["relative_path"]
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write file content
    with open(file_path, "wb") as f:
        f.write(file_data["content"])

    print(f"✓ Uploaded: {file_data['relative_path']}")
    return {"status": "success", "path": str(file_path)}


@app.function(
    image=base_image,
    volumes={"/vectordb": vectordb_volume},
)
def finalize_upload():
    """Commits the volume after all uploads."""
    vectordb_volume.commit()
    print("✓ Volume committed!")
    return {"status": "committed"}


@app.function(
    image=base_image,
    volumes={"/vectordb": vectordb_volume},
)
def list_vectordb_contents():
    """Lists the contents of the vector database volume."""
    import os
    from pathlib import Path

    data_dir = Path("/vectordb")
    print(f"=== Vector DB Volume Contents: {data_dir} ===\n")

    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("Volume is empty")
        return {"status": "empty"}

    total_files = 0
    total_size = 0

    for root, dirs, files in os.walk(data_dir):
        level = root.replace(str(data_dir), '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}📁 {os.path.basename(root)}/")

        for file in files:
            file_path = Path(root) / file
            size = file_path.stat().st_size
            total_size += size
            total_files += 1

            if total_files <= 20:  # Show first 20 files
                size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024*1024):.1f} MB"
                print(f"{'  ' * (level + 1)}📄 {file} ({size_str})")

        if len(files) > 20:
            print(f"{'  ' * (level + 1)}... and {len(files) - 20} more files")

    total_size_str = f"{total_size / (1024*1024):.1f} MB" if total_size > 1024*1024 else f"{total_size / 1024:.1f} KB"
    print(f"\nTotal: {total_files} files, {total_size_str}")

    return {"status": "success", "total_files": total_files, "total_size_bytes": total_size}


@app.function(
    image=base_image,
    volumes={"/vectordb": vectordb_volume},
)
def clear_volume():
    """Clears all contents from the volume (use with caution!)."""
    import shutil
    from pathlib import Path

    data_dir = Path("/vectordb")

    for item in data_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    vectordb_volume.commit()
    print("✓ Volume cleared!")
    return {"status": "cleared"}


@app.local_entrypoint()
def main(local_path: str = "rag_models/RAG_To_See_MedGemma_Performance/vectorDB", clear: bool = False):
    """
    Upload local vector database to Modal volume.

    Args:
        local_path: Path to local vectorDB directory
        clear: If True, clear the volume before uploading
    """
    from pathlib import Path

    local_dir = Path(local_path)

    if not local_dir.exists():
        print(f"❌ Directory not found: {local_dir}")
        print(f"   Current working directory: {Path.cwd()}")
        return

    if clear:
        print("=== Clearing existing volume contents ===\n")
        clear_volume.remote()

    print(f"=== Uploading Vector Database from {local_dir} ===\n")

    # Collect all files
    files_to_upload = []
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_dir)
            files_to_upload.append({
                "relative_path": str(relative_path),
                "content": file_path.read_bytes(),
                "size": file_path.stat().st_size
            })

    if not files_to_upload:
        print("No files found in directory!")
        return

    total_size = sum(f["size"] for f in files_to_upload)
    print(f"Found {len(files_to_upload)} files ({total_size / (1024*1024):.1f} MB)")
    print()

    # Upload files
    for file_data in files_to_upload:
        upload_vectordb.remote(file_data)

    # Commit the volume
    print("\n=== Finalizing Upload ===")
    finalize_upload.remote()

    # Verify
    print("\n=== Verifying Upload ===\n")
    list_vectordb_contents.remote()

    print("\n Vector database uploaded to Modal volume 'medical-vectordb'")
    print("   Access it in other Modal functions with:")
    print('   volumes={"/vectordb": modal.Volume.from_name("medical-vectordb")}')