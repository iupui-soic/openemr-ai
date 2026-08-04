import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"  # current production choice; pass --model to override
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
BATCH_SIZE = 256

BASE_DIR = Path(__file__).parent
CPT_JSON = BASE_DIR / "cpt_codes_final.json"
ICD10_JSON = BASE_DIR / "icd10_codes_final.json"


def load_codes(path):
    with open(path) as f:
        return json.load(f)


def build_collection(client, model, name, codes):
    existing_names = [c.name for c in client.list_collections()]

    if name in existing_names:
        col = client.get_collection(name)
        if col.count() == len(codes):
            print(f"'{name}' already exists with {col.count()} codes - skipping")
            return
        print(f"'{name}' exists but count mismatch - rebuilding")
        client.delete_collection(name)

    print(f"Building '{name}' for {len(codes)} codes using BioBERT...")
    col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    code_list = list(codes.keys())
    descriptions = [codes[c] for c in code_list]

    for i in range(0, len(code_list), BATCH_SIZE):
        batch_codes = code_list[i:i + BATCH_SIZE]
        batch_descs = descriptions[i:i + BATCH_SIZE]

        embeddings = model.encode(
            batch_descs,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        col.upsert(
            ids=batch_codes,
            embeddings=embeddings,
            metadatas=[{"description": d, "code": c} for c, d in zip(batch_codes, batch_descs)],
            documents=batch_descs,
        )

        done = min(i + BATCH_SIZE, len(code_list))
        print(f"  {done}/{len(code_list)}")

    print(f"Built '{name}' - {col.count()} codes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--new-suffix", default="")
    args = parser.parse_args()

    suffix = f"-{args.new_suffix}" if args.new_suffix else ""
    cpt_name = f"cpt-codes{suffix}"
    icd10_name = f"icd10-codes{suffix}"

    print(f"Loading embeddings: {args.model}")
    model = SentenceTransformer(args.model)

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    cpt_codes = load_codes(CPT_JSON)
    icd10_codes = load_codes(ICD10_JSON)

    build_collection(client, model, cpt_name, cpt_codes)
    build_collection(client, model, icd10_name, icd10_codes)

    print("Done. Collections now in ChromaDB:")
    for c in client.list_collections():
        print(f"  - {c.name}")


if __name__ == "__main__":
    main()
