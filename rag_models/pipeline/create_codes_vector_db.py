"""
Build a dedicated CPT/ICD-10 code vector database.

Adapted from rag_models/pipeline/create_vector_db.py, which built the
"medical-vectordb" volume from disease->SOAP-schema data. That volume ONLY
contains SOAP note schemas -- it has no CPT/ICD code data, so it cannot be
reused for coding (confirmed directly, not assumed).

This script builds a SEPARATE Chroma collection from your existing CPT/ICD
JSON files (cpt_codes_final.json, icd10_codes_final.json) and persists it
to its own Modal Volume, following the identical pattern so it can be
consumed by a Modal-hosted retrieval step the same way medical-vectordb is
consumed in pipeline/run_fareez_summaries.py.

Usage (matches the CI/CD pattern in .github/workflows/rag-summarization.yml):
    modal run create_codes_vector_db.py \
        --cpt-json path/to/cpt_codes_final.json \
        --icd10-json path/to/icd10_codes_final.json
"""

import json
import os
import shutil
import time
from pathlib import Path

import modal

app = modal.App("codes-vector-db-builder")

codes_volume = modal.Volume.from_name("medical-codes-vectordb", create_if_missing=True)
CHROMA_PATH = "/vectordb/chroma_codes"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "langchain-chroma>=0.1.0",
        "langchain-huggingface>=0.0.1",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.22",
    )
)


@app.function(
    image=image,
    volumes={"/vectordb": codes_volume},
    timeout=3600,
)
def build_codes_db(cpt_data: dict, icd10_data: dict):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    print("Building CPT/ICD-10 code vector database:", CHROMA_PATH)
    start = time.time()

    if os.path.exists(CHROMA_PATH):
        print("Removing existing database...")
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)

    print("Loading BioBERT embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    docs, metadatas, ids = [], [], []

    for code, description in cpt_data.items():
        docs.append(description)
        metadatas.append({"code": code, "code_type": "CPT4", "description": description})
        ids.append(f"cpt-{code}")

    for code, description in icd10_data.items():
        docs.append(description)
        metadatas.append({"code": code, "code_type": "ICD10", "description": description})
        ids.append(f"icd10-{code}")

    print(f"Indexing {len(docs)} codes ({len(cpt_data)} CPT + {len(icd10_data)} ICD-10)...")

    batch_size = 512
    for i in range(0, len(docs), batch_size):
        db.add_texts(
            docs[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )
        print(f"  {min(i + batch_size, len(docs))}/{len(docs)}")

    codes_volume.commit()

    elapsed = time.time() - start
    print("=" * 80)
    print(f"SUCCESS! Indexed {len(docs)} codes in {elapsed:.1f}s")
    print(f"Volume: medical-codes-vectordb, path: {CHROMA_PATH}")
    print("=" * 80)

    return {"total_codes": len(docs), "cpt_count": len(cpt_data), "icd10_count": len(icd10_data)}


@app.local_entrypoint()
def main(cpt_json: str, icd10_json: str):
    with open(cpt_json) as f:
        cpt_data = json.load(f)
    with open(icd10_json) as f:
        icd10_data = json.load(f)

    print(f"Loaded {len(cpt_data)} CPT codes from {cpt_json}")
    print(f"Loaded {len(icd10_data)} ICD-10 codes from {icd10_json}")

    result = build_codes_db.remote(cpt_data, icd10_data)
    print(json.dumps(result, indent=2))
