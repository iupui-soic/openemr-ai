import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from automated_cpt_and_icd_coding.pipeline.retrieval import load_codes, build_or_load_collection

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"  # current production choice; pass --model to override
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

BASE_DIR = Path(__file__).parent
CPT_JSON = BASE_DIR / "cpt_codes_final.json"
ICD10_JSON = BASE_DIR / "icd10_codes_final.json"


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

    build_or_load_collection(client, model, cpt_name, cpt_codes)
    build_or_load_collection(client, model, icd10_name, icd10_codes)

    print("Done. Collections now in ChromaDB:")
    for c in client.list_collections():
        print(f"  - {c.name}")


if __name__ == "__main__":
    main()
