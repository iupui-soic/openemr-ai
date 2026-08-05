"""
Benchmark a Modal-deployed coder (Gemma4 or Qwen) against the MDACE CPT
benchmark, reusing automated_coding's data loader and metrics so results
are directly comparable to RESULTS.md.

Retrieval (ChromaDB + embeddings) runs locally, same as coding_service.py.
Only extraction and matching run remotely on the already-deployed Modal app --
this script does not deploy anything itself.

Usage:
    modal run benchmark_deployed.py --model gemma4 --limit 10
    modal run benchmark_deployed.py --model qwen
"""
import csv
import os
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from automated_coding.data import load_notes, get_label_space
from automated_cpt_and_icd_coding.pipeline.retrieval import load_codes, build_or_load_collection, retrieve_candidates
from automated_coding.metrics import summarize

app = modal.App("benchmark-runner")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CPT_COLLECTION = "cpt-codes"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
CPT_JSON = REPO_ROOT / "automated_cpt_and_icd_coding" / "coding_service" / "cpt_codes_final.json"

MODEL_REGISTRY = {
    "gemma4": ("automated-cpt-icd-coding", "Gemma4Coder"),
    "qwen": ("qwen3-6-35b-coder", "Qwen3Coder"),
}


@app.local_entrypoint()
def main(model: str = "gemma4", limit: int = 0, output_dir: str = "results"):
    if model not in MODEL_REGISTRY:
        print(f"Unknown model '{model}'. Choose from: {list(MODEL_REGISTRY)}")
        return

    app_name, class_name = MODEL_REGISTRY[model]

    print("=" * 70)
    print(f"CPT CODING BENCHMARK -- {model} (deployed on Modal: {app_name}/{class_name})")
    print("=" * 70)

    print("\n[1/4] Loading MDACE notes...")
    notes = load_notes()
    if limit:
        notes = notes[:limit]
    label_space = get_label_space(notes)
    print(f"   {len(notes)} notes, {len(label_space)} unique codes in label space")

    print("\n[2/4] Setting up local retrieval (ChromaDB)...")
    import chromadb
    from sentence_transformers import SentenceTransformer

    cpt_codes = load_codes(CPT_JSON)
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    cpt_col = build_or_load_collection(client, embed_model, CPT_COLLECTION, cpt_codes)

    print(f"\n[3/4] Connecting to deployed model: {app_name}/{class_name}...")
    try:
        ModelCls = modal.Cls.from_name(app_name, class_name)
        coder = ModelCls()
    except Exception as e:
        print(f"Could not connect: {e}")
        print(f"Make sure it's deployed first: modal deploy {class_name.lower()}...")
        return

    print(f"\n[4/4] Running {len(notes)} notes through the pipeline...")
    gold, pred = [], []
    start = time.time()
    for i, note in enumerate(notes):
        print(f"  [{i+1}/{len(notes)}] {note.note_id}")
        try:
            extraction = coder.extract_entities.remote(note_text=note.text)
            entities = extraction.get("entities", {})
            all_terms = entities.get("diagnoses", []) + entities.get("procedures", [])
            candidates = retrieve_candidates(embed_model, cpt_col, all_terms)
            result = coder.match_codes.remote(note_text=note.text, entities=entities, candidates=candidates)
            pred_codes = {m["code"] for m in result.get("matches", []) if m.get("code_type") == "CPT4"}
        except Exception as e:
            print(f"    error: {e}")
            pred_codes = set()
        gold.append(note.gold_codes)
        pred.append(pred_codes)

    elapsed = time.time() - start

    metrics = summarize(gold, pred, label_space)
    metrics["model"] = model
    metrics["notes"] = len(notes)
    metrics["total_time_seconds"] = round(elapsed, 2)
    metrics["avg_time_per_note"] = round(elapsed / len(notes), 2) if notes else 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_dir = REPO_ROOT / "automated_cpt_and_icd_coding" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"benchmark_{model}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"\nSaved: {csv_path}")
