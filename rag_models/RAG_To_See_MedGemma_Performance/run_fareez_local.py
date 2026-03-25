"""
Local runner for Fareez RAG summarization (no Modal required).

Runs Groq-based models locally using the local ChromaDB vector store.
MedGemma requires Modal and is excluded from this runner.

Models:
    1. GPT-OSS 120B (Groq, reasoning API)
    2. GPT-OSS 20B (Groq, reasoning API)
    3. Qwen3 32B (Groq, standard API)

Usage:
    python run_fareez_local.py                          # Run all 3 Groq models
    python run_fareez_local.py --model gpt-oss-120b     # Run single model
    python run_fareez_local.py --model qwen3-32b --output-dir results/fareez
"""

import os
import re
import sys
import time
import json
import argparse
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(__file__)
CHROMA_PATH = os.path.join(BASE_DIR, "vectorDB", "chroma_schema_improved")

MODELS = {
    "gpt-oss-120b": {
        "name": "openai/gpt-oss-120b",
        "short_name": "gpt-oss-120b",
        "api_type": "reasoning",
    },
    "gpt-oss-20b": {
        "name": "openai/gpt-oss-20b",
        "short_name": "gpt-oss-20b",
        "api_type": "reasoning",
    },
    "qwen3-32b": {
        "name": "qwen/qwen3-32b",
        "short_name": "qwen3-32b",
        "api_type": "standard",
    },
}

SUMMARY_PROMPT_TEMPLATE = """Generate a comprehensive medical summary in SOAP format from the following data:

TRANSCRIPT (Doctor-patient conversation):
{transcript}

OPENEMR EXTRACT (Electronic health record):
{openemr}

SCHEMA GUIDE (Reference sections to include):
{schema}

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

Generate the medical summary now in narrative prose format, beginning with "Patient Information":"""


# ============================================================================
# RAG Components (loaded once)
# ============================================================================

class LocalRAG:
    """Local RAG components using ChromaDB + SBERT for schema retrieval."""

    def __init__(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from sentence_transformers import SentenceTransformer

        print("Loading local RAG components...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )
        self.vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings,
        )

        collection_data = self.vector_store.get(include=["metadatas", "documents"])
        self.all_metadatas = collection_data["metadatas"]
        self.all_docs = collection_data["documents"]
        self.metadata_diseases = [m.get("diseases", "Unspecified") for m in self.all_metadatas]

        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.candidate_embs = self.sbert_model.encode(self.metadata_diseases, convert_to_tensor=True)

        print(f"Loaded {len(self.all_docs)} schemas from {CHROMA_PATH}")

    def retrieve_schemas(self, detected_disease: str) -> str:
        from sentence_transformers import util

        target_emb = self.sbert_model.encode(detected_disease, convert_to_tensor=True)
        cosine_scores = util.cos_sim(target_emb, self.candidate_embs)[0]
        k = min(2, len(cosine_scores))
        top_indices = cosine_scores.topk(k).indices.tolist()

        schema_context = ""
        for rank, idx in enumerate(top_indices):
            doc_content = self.all_docs[idx]
            disease_meta = self.metadata_diseases[idx]
            schema_context += f"\n\n=== SCHEMA {rank+1} ({disease_meta}) ===\n{doc_content}"
        return schema_context


# ============================================================================
# Summarizers
# ============================================================================

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def generate_summary_standard(
    client, model_name: str, transcript: str, openemr: str,
    schema_context: str, is_qwen: bool = False,
) -> tuple:
    """Generate summary using standard Groq chat API (Qwen3)."""
    system_suffix = " /no_think" if is_qwen else ""

    prompt_content = SUMMARY_PROMPT_TEMPLATE.format(
        transcript=transcript,
        openemr=openemr or "No OpenEMR data available.",
        schema=schema_context,
    )

    messages = [
        {
            "role": "system",
            "content": "You are an expert medical scribe specialized in clinical documentation. Generate comprehensive SOAP-format medical summaries." + system_suffix,
        },
        {"role": "user", "content": prompt_content},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )
    text = response.choices[0].message.content or ""
    text = strip_think_tags(text).strip()
    return text, messages[0]["content"] + messages[1]["content"]


def generate_summary_reasoning(
    client, model_name: str, transcript: str, openemr: str,
    schema_context: str,
) -> tuple:
    """Generate summary using Groq reasoning API (GPT-OSS)."""
    prompt_content = SUMMARY_PROMPT_TEMPLATE.format(
        transcript=transcript,
        openemr=openemr or "No OpenEMR data available.",
        schema=schema_context,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "You are an expert medical scribe specialized in clinical documentation. "
                "Generate comprehensive SOAP-format medical summaries.\n\n"
                + prompt_content
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=1,
        max_completion_tokens=4096,
        top_p=1,
        reasoning_effort="medium",
    )
    text = (response.choices[0].message.content or "").strip()
    return text, messages[0]["content"]


# ============================================================================
# Results Saver
# ============================================================================

def save_results(results: List[Dict[str, Any]], model_short: str, output_dir: str):
    import pandas as pd

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CSV
    table_data = []
    for r in results:
        table_data.append({
            "patient_name": r.get("patient_name", "Unknown"),
            "model": model_short,
            "detected_disease": r.get("detected_disease", ""),
            "total_time_s": r.get("total_time", 0.0),
            "input_tokens": r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
            "summary_chars": len(r.get("summary", "")),
        })

    df = pd.DataFrame(table_data)
    avg_row = {k: df[k].mean() if k not in ("patient_name", "model", "detected_disease") else ""
               for k in df.columns}
    avg_row["patient_name"] = "AVERAGE"
    avg_row["model"] = model_short
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    csv_path = output_path / f"evaluation_results_{model_short}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    # Individual summaries
    summaries_dir = output_path / "fareez_summaries" / model_short
    summaries_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        patient_name = r.get("patient_name", "unknown")
        summary_file = summaries_dir / f"{patient_name}.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Model: {r.get('model', model_short)}\n")
            f.write(f"Patient: {patient_name}\n")
            f.write(f"Detected Disease: {r.get('detected_disease', 'N/A')}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Time: {r.get('total_time', 0):.2f}s\n")
            f.write(f"Tokens: {r.get('input_tokens', 0):,} in / {r.get('output_tokens', 0):,} out\n")
            f.write("=" * 60 + "\n\n")
            f.write(r.get("summary", "[No summary generated]"))

    print(f"  Saved {len(results)} summaries to {summaries_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Fareez RAG summarization locally")
    parser.add_argument("--model", default="all", choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--output-dir", default="results/fareez")
    args = parser.parse_args()

    from groq import Groq
    from fareez_rag_loader import FareezLoader

    print("=" * 80)
    print("FAREEZ RAG SUMMARIZATION PIPELINE (LOCAL)")
    print("=" * 80)

    # Load data
    loader = FareezLoader()
    patients = loader.get_entries()
    print(f"Loaded {len(patients)} Fareez entries")

    # Load RAG
    rag = LocalRAG()

    # Initialize Groq
    client = Groq()

    # Token counting
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        encoding = None

    def count_tokens(text):
        if encoding:
            return len(encoding.encode(text))
        return int(len(text.split()) * 1.3)

    # Determine models
    if args.model == "all":
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = [args.model]

    for model_key in models_to_run:
        config = MODELS[model_key]
        model_name = config["name"]
        model_short = config["short_name"]
        api_type = config["api_type"]
        is_qwen = "qwen" in model_name.lower()

        print(f"\n{'='*80}")
        print(f"MODEL: {model_name} (api: {api_type})")
        print(f"{'='*80}")

        model_start = time.time()
        results = []

        for i, patient in enumerate(patients):
            patient_name = patient["patient_name"]
            condition = patient.get("detected_condition", "General")
            print(f"  [{i+1:2d}/40] {patient_name} ({patient['category']}, {condition})", end="", flush=True)

            start = time.time()

            try:
                schema_context = rag.retrieve_schemas(condition)
                transcript = patient["transcript"]
                openemr = patient.get("openemr_data", "")

                if api_type == "reasoning":
                    generated_text, prompt_text = generate_summary_reasoning(
                        client, model_name, transcript, openemr, schema_context
                    )
                else:
                    generated_text, prompt_text = generate_summary_standard(
                        client, model_name, transcript, openemr, schema_context, is_qwen
                    )

                elapsed = time.time() - start
                input_tokens = count_tokens(prompt_text)
                output_tokens = count_tokens(generated_text)

                results.append({
                    "patient_name": patient_name,
                    "model": model_name,
                    "detected_disease": condition,
                    "summary": generated_text,
                    "total_time": elapsed,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                })

                print(f" -> {elapsed:.1f}s, {output_tokens} tokens, {len(generated_text)} chars")

            except Exception as e:
                elapsed = time.time() - start
                print(f" -> ERROR: {str(e)[:80]}")
                results.append({
                    "patient_name": patient_name,
                    "model": model_name,
                    "detected_disease": condition,
                    "summary": f"Error: {str(e)}",
                    "total_time": elapsed,
                    "input_tokens": 0,
                    "output_tokens": 0,
                })

                # Rate limit handling
                if "rate" in str(e).lower() or "429" in str(e):
                    print("    Waiting 30s for rate limit...")
                    time.sleep(30)

        save_results(results, model_short, args.output_dir)

        model_time = time.time() - model_start
        successful = sum(1 for r in results if not r.get("summary", "").startswith("Error"))
        print(f"\n  {model_name}: {successful}/{len(results)} summaries in {model_time:.1f}s ({model_time/60:.1f} min)")

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
