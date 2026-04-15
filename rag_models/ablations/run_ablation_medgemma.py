#!/usr/bin/env python
"""
MedGemma 4B ablation driver — Modal-backed, runs all configured ablation cells
sequentially in one container so the model loads once.

Why one driver instead of one Modal app per cell: the MedGemma container takes
~30s to load and we want to avoid paying that cost 10+ times.

Usage:
    # Run all ablation cells that involve MedGemma (default)
    modal run rag_models/ablations/run_ablation_medgemma.py

    # Run only the no-RAG cell
    modal run rag_models/ablations/run_ablation_medgemma.py --cells norag

    # Run only k-sweep + embedding-substitution
    modal run rag_models/ablations/run_ablation_medgemma.py --cells k1,k3,k5,embed_clinicalbert,embed_pubmedbert

Cells covered (MedGemma-4B is one of the 4 clinician-evaluated models):
    norag, k1, k3, k5, embed_clinicalbert, embed_pubmedbert

Temperature/prompt sweeps are TOP-2 only (GPT-OSS-120B and GPT-OSS-20B), so
MedGemma is intentionally excluded from those.

Outputs: rag_models/results/fareez/ablations/<cell>/medgemma-4b/<patient>.txt
         rag_models/results/fareez/ablations/<cell>/medgemma-4b/evaluation_results_medgemma-4b.csv
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import modal

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App("fareez-ablation-medgemma")
vectordb_volume = modal.Volume.from_name("medical-vectordb")
CHROMA_PATH = "/vectordb/chroma_schema_improved"

medgemma_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-huggingface>=0.0.1",
        "langchain-chroma>=0.1.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.22",
        "transformers>=4.45.0",
        "torch>=2.1.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "huggingface-hub>=0.20.0",
        "tiktoken>=0.5.0",
    )
)

# Production retrieval defaults
PROD_K = 2
PROD_EMBEDDING = "all-MiniLM-L6-v2"
PROD_PROMPT = "current"

EMBEDDING_MAP = {
    "minilm":        "all-MiniLM-L6-v2",
    "clinicalbert":  "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert":    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}

# ---------------------------------------------------------------------------
# Prompt template — same shape as ablations/_ablation_core.PROMPT_TEMPLATES,
# inlined so the Modal container doesn't need to import the local module.
# ---------------------------------------------------------------------------

_BASE_BODY = """Generate a comprehensive medical summary in SOAP format from the following data:

TRANSCRIPT (Doctor-patient conversation):
{transcript}

OPENEMR EXTRACT (Electronic health record):
{openemr}

SCHEMA GUIDE (Reference sections to include):
{schema}
"""

PROMPT_TEMPLATES = {
    "current": _BASE_BODY + """
OUTPUT FORMAT REQUIREMENTS:
- Generate a NARRATIVE TEXT document, NOT JSON or structured data
- Use clear section headers (e.g., "Patient Information", "Chief Complaint", "History of Present Illness")
- Write in complete sentences and paragraphs
- Use professional medical documentation prose style
- Format similar to a hospital discharge summary
- Do NOT use markdown formatting (no #, ##, **, *, -, ```, etc.)

INSTRUCTIONS:
1. Use the SCHEMA GUIDE as a reference for which sections to include
2. Extract relevant information from the TRANSCRIPT and OPENEMR EXTRACT
3. Write in narrative prose with proper paragraphs
4. If information for a section is missing, write "No information available."
5. If TRANSCRIPT and OPENEMR conflict, trust the TRANSCRIPT for current status
6. Do NOT include any meta-commentary, explanations, or references to this prompt
7. Do NOT output JSON, XML, or any structured data format
8. Do NOT hallucinate or invent information not present in the inputs
9. Do NOT use any markdown syntax - plain text only

Generate the medical summary now in narrative prose format, beginning with "Patient Information":""",
}

# ---------------------------------------------------------------------------
# Cell -> config mapping
# ---------------------------------------------------------------------------

ALL_CELLS = ["norag", "k1", "k3", "k5", "embed_clinicalbert", "embed_pubmedbert"]


def parse_cell(cell: str):
    """
    Map a cell name to (rag_enabled, embedder_key, k, prompt_variant).
    Production defaults except for the one variable being ablated.
    """
    rag_on, emb, k, prompt = True, "minilm", PROD_K, PROD_PROMPT
    if cell == "norag":
        rag_on = False
    elif cell.startswith("k"):
        k = int(cell[1:])
    elif cell.startswith("embed_"):
        emb = cell.split("_", 1)[1]
    else:
        raise ValueError(f"unknown cell: {cell}")
    return rag_on, emb, k, prompt


# ---------------------------------------------------------------------------
# Modal class: warm MedGemma + retrieval components, run any cell
# ---------------------------------------------------------------------------

def _resolve_hf_token() -> str:
    """Return the HF token, preferring the verified ~/.cache file over .env."""
    from pathlib import Path
    cache = Path.home() / ".cache" / "huggingface" / "token"
    if cache.exists():
        tok = cache.read_text().strip()
        if tok:
            return tok
    return os.environ.get("HF_TOKEN", "")


_HF_TOKEN = _resolve_hf_token()


@app.cls(
    image=medgemma_image,
    gpu="A10G",
    timeout=7200,
    volumes={"/vectordb": vectordb_volume},
    secrets=[modal.Secret.from_dict({
        "HF_TOKEN": _HF_TOKEN,
        "HUGGING_FACE_HUB_TOKEN": _HF_TOKEN,   # transformers / huggingface_hub both read this
    })],
)
class MedGemmaAblation:

    @modal.enter()
    def load(self):
        import torch
        from transformers import pipeline as hf_pipeline
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from sentence_transformers import SentenceTransformer
        import tiktoken

        print("Loading MedGemma 4B-IT...")
        t0 = time.time()
        self.pipe = hf_pipeline(
            "text-generation",
            model="google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"MedGemma loaded in {time.time() - t0:.1f}s")

        # Load Chroma metadata once (BioBERT embedder is just for Chroma init)
        embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        coll = vector_store.get(include=["metadatas", "documents"])
        self.all_docs = coll["documents"]
        self.metadata_diseases = [m.get("diseases", "Unspecified") for m in coll["metadatas"]]
        print(f"Loaded {len(self.all_docs)} schemas from {CHROMA_PATH}")

        # Cache SBERT models per embedder so we don't reload across cells
        self._sbert_cache = {}
        # Pre-warm the production embedder
        self._sbert_cache["minilm"] = SentenceTransformer(EMBEDDING_MAP["minilm"])
        self._candidate_cache = {
            "minilm": self._sbert_cache["minilm"].encode(
                self.metadata_diseases, convert_to_tensor=True
            )
        }
        self._SentenceTransformer = SentenceTransformer

        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.encoding = None
        print("All components loaded.")

    def _get_embedder(self, key: str):
        if key not in self._sbert_cache:
            print(f"  Loading embedder: {EMBEDDING_MAP[key]}")
            t0 = time.time()
            m = self._SentenceTransformer(EMBEDDING_MAP[key])
            self._sbert_cache[key] = m
            self._candidate_cache[key] = m.encode(self.metadata_diseases, convert_to_tensor=True)
            print(f"  Encoded {len(self.metadata_diseases)} candidates in {time.time() - t0:.1f}s")
        return self._sbert_cache[key], self._candidate_cache[key]

    def _retrieve(self, condition: str, rag_on: bool, embedder_key: str, k: int) -> str:
        if not rag_on:
            return ""
        from sentence_transformers import util
        sbert, cands = self._get_embedder(embedder_key)
        target_emb = sbert.encode(condition, convert_to_tensor=True)
        cosine = util.cos_sim(target_emb, cands)[0]
        kk = min(k, len(cosine))
        idx = cosine.topk(kk).indices.tolist()
        out = ""
        for rank, i in enumerate(idx):
            out += f"\n\n=== SCHEMA {rank+1} ({self.metadata_diseases[i]}) ===\n{self.all_docs[i]}"
        return out

    def _count(self, s: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(s))
        return int(len(s.split()) * 1.3)

    @modal.method()
    def run_cell(
        self,
        cell: str,
        patients: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run all 40 cases under the given cell config; return per-case results."""
        rag_on, emb, k, prompt_variant = parse_cell(cell)
        template = PROMPT_TEMPLATES.get(prompt_variant, PROMPT_TEMPLATES["current"])
        print(f"\n>>> Cell={cell}  RAG={rag_on}  emb={emb}  k={k}  prompt={prompt_variant}")

        results = []
        for i, p in enumerate(patients, 1):
            name = p["patient_name"]
            cond = p.get("detected_condition", "General")
            t0 = time.time()
            try:
                schema = self._retrieve(cond, rag_on, emb, k)
                body = template.format(
                    transcript=p["transcript"],
                    openemr=p.get("openemr_data") or "No OpenEMR data available.",
                    schema=schema or "No schema retrieved.",
                )
                messages = [
                    {"role": "system", "content": (
                        "You are an expert medical scribe specialized in clinical "
                        "documentation. Generate comprehensive SOAP-format medical "
                        "summaries.")},
                    {"role": "user", "content": body},
                ]
                in_tok = self._count(messages[0]["content"] + messages[1]["content"])
                out = self.pipe(messages, max_new_tokens=2048, temperature=0.3)
                text = out[0]["generated_text"][-1]["content"]
                out_tok = self._count(text)
                elapsed = time.time() - t0
                results.append({
                    "patient_name": name,
                    "model": "medgemma-4b",
                    "detected_disease": cond,
                    "summary": text,
                    "total_time_s": elapsed,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "summary_chars": len(text),
                })
                print(f"  [{i:2d}/{len(patients)}] {name} {elapsed:5.1f}s  {out_tok} tok")
            except Exception as e:
                elapsed = time.time() - t0
                msg = str(e)[:120]
                print(f"  [{i:2d}/{len(patients)}] {name} ERROR: {msg}")
                results.append({
                    "patient_name": name,
                    "model": "medgemma-4b",
                    "detected_disease": cond,
                    "summary": "",
                    "total_time_s": elapsed,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "summary_chars": 0,
                })
        return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(cells: str = ",".join(ALL_CELLS),
         output_root: str = "results/fareez/ablations"):
    """
    Run the requested cells (comma-separated). Default: all 6 cells.

    All cells share one warm MedGemma container.
    """
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
    from fareez_rag_loader import FareezLoader

    cells_list = [c.strip() for c in cells.split(",") if c.strip()]
    print(f"Cells to run: {cells_list}")
    for c in cells_list:
        parse_cell(c)  # validate

    rag_root = Path(__file__).resolve().parent.parent
    out_root = (rag_root / output_root).resolve()

    loader = FareezLoader()
    patients = loader.get_entries()
    print(f"Loaded {len(patients)} Fareez entries")

    runner = MedGemmaAblation()
    overall_start = time.time()

    for cell in cells_list:
        cell_dir = out_root / cell / "medgemma-4b"
        cell_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Cell: {cell} -> {cell_dir} ===")
        cell_start = time.time()

        results = runner.run_cell.remote(cell, patients)

        # Persist per-case summaries + CSV
        import pandas as pd
        rows_for_csv = []
        for r in results:
            patient = r["patient_name"]
            with open(cell_dir / f"{patient}.txt", "w", encoding="utf-8") as f:
                f.write(f"Model: google/medgemma-4b-it\n")
                f.write(f"Patient: {patient}\n")
                f.write(f"Detected Disease: {r['detected_disease']}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Time: {r['total_time_s']:.2f}s\n")
                f.write(f"Tokens: {r['input_tokens']:,} in / {r['output_tokens']:,} out\n")
                f.write(f"Cell: {cell}\n")
                f.write("=" * 60 + "\n\n")
                f.write(r["summary"] or "[No summary generated]")
            rows_for_csv.append({k: r[k] for k in (
                "patient_name", "model", "detected_disease",
                "total_time_s", "input_tokens", "output_tokens", "summary_chars",
            )})
        df = pd.DataFrame(rows_for_csv)
        if len(df):
            avg = {c: (df[c].mean() if df[c].dtype.kind in "fi" else "") for c in df.columns}
            avg["patient_name"] = "AVERAGE"
            avg["model"] = "medgemma-4b"
            df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
        df.to_csv(cell_dir / "evaluation_results_medgemma-4b.csv", index=False)

        successful = sum(1 for r in results if r["summary_chars"] > 0)
        cell_time = time.time() - cell_start
        print(f"=== Cell {cell}: {successful}/{len(results)} in {cell_time/60:.1f} min ===")

    print(f"\nALL CELLS DONE in {(time.time()-overall_start)/60:.1f} min")
