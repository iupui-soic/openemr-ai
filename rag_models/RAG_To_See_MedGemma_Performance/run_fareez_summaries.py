"""
Step 4.9.5-4.9.7: Unified orchestrator for Fareez RAG summarization experiment.

Runs 4 models x 40 transcripts = 160 summaries using the Fareez OSCE dataset
paired with OpenEMR patient data. Uses Modal for GPU compute (MedGemma) and
Groq API for hosted inference (GPT-OSS 120B, GPT-OSS 20B, Qwen3 32B).

Design decision: All models use the SAME pre-detected conditions (from step 4.9.2)
for schema retrieval. This isolates summary generation quality as the variable.

Models:
    1. GPT-OSS 120B (Groq, reasoning API) — largest, reasoning model
    2. GPT-OSS 20B (Groq, reasoning API) — 100% ELM accuracy, reasoning model
    3. Qwen3 32B (Groq, standard API) — 95.5% ELM, general-purpose
    4. MedGemma 4B-IT (Modal A10G) — medical-domain specialist

Usage:
    # Run all 4 models:
    modal run run_fareez_summaries.py

    # Run specific model:
    modal run run_fareez_summaries.py --model gpt-oss-120b

Prerequisites:
    modal deploy shared_evaluator_service.py
"""

import modal
import os
import re
from typing import Dict, List, Any

# ============================================================================
# Modal App Configuration
# ============================================================================

app = modal.App("fareez-rag-summarization")

vectordb_volume = modal.Volume.from_name("medical-vectordb")

CHROMA_PATH = "/vectordb/chroma_schema_improved"

# Model registry
MODELS = {
    "gpt-oss-120b": {
        "name": "openai/gpt-oss-120b",
        "short_name": "gpt-oss-120b",
        "backend": "groq-reasoning",
    },
    "gpt-oss-20b": {
        "name": "openai/gpt-oss-20b",
        "short_name": "gpt-oss-20b",
        "backend": "groq-reasoning",
    },
    "qwen3-32b": {
        "name": "qwen/qwen3-32b",
        "short_name": "qwen3-32b",
        "backend": "groq",
    },
    "medgemma-4b": {
        "name": "google/medgemma-4b-it",
        "short_name": "medgemma-4b",
        "backend": "modal",
    },
}

# ============================================================================
# Modal Images
# ============================================================================

groq_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "groq>=0.4.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-huggingface>=0.0.1",
        "langchain-chroma>=0.1.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.22",
        "tiktoken>=0.5.0",
    )
)

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

# Shared summary prompt (used by all models for consistency)
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
# Groq Standard Summarizer (Qwen3)
# ============================================================================

@app.cls(
    image=groq_image,
    timeout=3600,
    volumes={"/vectordb": vectordb_volume},
    secrets=[modal.Secret.from_dict({"GROQ_API_KEY": os.environ.get("GROQ_API_KEY", "")})],
)
class GroqSummarizer:

    @modal.enter()
    def load_models(self):
        from groq import Groq
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from sentence_transformers import SentenceTransformer
        import tiktoken

        print("Loading RAG components...")
        self.client = Groq()

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

        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.encoding = None

        print("RAG components loaded.")

    def _retrieve_schemas(self, detected_disease):
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

    @staticmethod
    def _strip_think_tags(text):
        """Remove <think>...</think> blocks from Qwen3 output."""
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    @modal.method()
    def generate_summary(
        self,
        model_name: str,
        transcript_text: str,
        openemr_text: str = "",
        patient_name: str = "Patient",
        detected_condition: str = "",
    ) -> Dict[str, Any]:
        import time

        print(f"\nGenerating: {patient_name} | Model: {model_name}")
        start_total = time.time()

        is_qwen = "qwen" in model_name.lower()
        system_suffix = " /no_think" if is_qwen else ""

        # Use pre-detected condition for schema retrieval (fair comparison)
        detected_disease = detected_condition or "General"
        schema_context = self._retrieve_schemas(detected_disease)

        # Generate summary
        start_gen = time.time()
        prompt_content = SUMMARY_PROMPT_TEMPLATE.format(
            transcript=transcript_text,
            openemr=openemr_text if openemr_text else "No OpenEMR data available.",
            schema=schema_context,
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert medical scribe specialized in clinical documentation. Generate comprehensive SOAP-format medical summaries." + system_suffix,
            },
            {"role": "user", "content": prompt_content},
        ]

        if self.encoding:
            input_tokens = len(self.encoding.encode(messages[0]["content"] + messages[1]["content"]))
        else:
            input_tokens = int(len((messages[0]["content"] + messages[1]["content"]).split()) * 1.3)

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
            generated_text = response.choices[0].message.content or ""
            generated_text = self._strip_think_tags(generated_text).strip()

            if self.encoding:
                output_tokens = len(self.encoding.encode(generated_text))
            else:
                output_tokens = int(len(generated_text.split()) * 1.3)
        except Exception as e:
            print(f"Generation failed: {e}")
            generated_text = f"Error generating summary: {str(e)}"
            output_tokens = 0

        generation_time = time.time() - start_gen
        total_time = time.time() - start_total
        print(f"  Tokens: {input_tokens:,} in / {output_tokens:,} out | Time: {total_time:.2f}s")

        return {
            "summary": generated_text,
            "patient_name": patient_name,
            "detected_disease": detected_disease,
            "retrieval_time": 0,
            "generation_time": generation_time,
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model_name,
        }


# ============================================================================
# Groq Reasoning Summarizer (GPT-OSS 120B, GPT-OSS 20B)
# ============================================================================

@app.cls(
    image=groq_image,
    timeout=3600,
    volumes={"/vectordb": vectordb_volume},
    secrets=[modal.Secret.from_dict({"GROQ_API_KEY": os.environ.get("GROQ_API_KEY", "")})],
)
class GroqReasoningSummarizer:
    """Summarizer for Groq reasoning models (GPT-OSS) which require different API params."""

    @modal.enter()
    def load_models(self):
        from groq import Groq
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from sentence_transformers import SentenceTransformer
        import tiktoken

        print("Loading RAG components (reasoning mode)...")
        self.client = Groq()

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

        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.encoding = None

        print("RAG components loaded (reasoning mode).")

    def _retrieve_schemas(self, detected_disease):
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

    @modal.method()
    def generate_summary(
        self,
        model_name: str,
        transcript_text: str,
        openemr_text: str = "",
        patient_name: str = "Patient",
        detected_condition: str = "",
    ) -> Dict[str, Any]:
        import time

        print(f"\nGenerating: {patient_name} | Model: {model_name} (reasoning)")
        start_total = time.time()

        detected_disease = detected_condition or "General"
        schema_context = self._retrieve_schemas(detected_disease)

        start_gen = time.time()
        prompt_content = SUMMARY_PROMPT_TEMPLATE.format(
            transcript=transcript_text,
            openemr=openemr_text if openemr_text else "No OpenEMR data available.",
            schema=schema_context,
        )

        # GPT-OSS reasoning models: user-only messages, max_completion_tokens, reasoning_effort
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

        if self.encoding:
            input_tokens = len(self.encoding.encode(messages[0]["content"]))
        else:
            input_tokens = int(len(messages[0]["content"].split()) * 1.3)

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1,
                max_completion_tokens=4096,
                top_p=1,
                reasoning_effort="medium",
            )
            generated_text = (response.choices[0].message.content or "").strip()

            if self.encoding:
                output_tokens = len(self.encoding.encode(generated_text))
            else:
                output_tokens = int(len(generated_text.split()) * 1.3)
        except Exception as e:
            print(f"Reasoning generation failed: {e}")
            generated_text = f"Error generating summary: {str(e)}"
            output_tokens = 0

        generation_time = time.time() - start_gen
        total_time = time.time() - start_total
        print(f"  Tokens: {input_tokens:,} in / {output_tokens:,} out | Time: {total_time:.2f}s")

        return {
            "summary": generated_text,
            "patient_name": patient_name,
            "detected_disease": detected_disease,
            "retrieval_time": 0,
            "generation_time": generation_time,
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model_name,
        }


# ============================================================================
# MedGemma Summarizer (Modal GPU)
# ============================================================================

@app.cls(
    image=medgemma_image,
    gpu="A10G",
    timeout=3600,
    volumes={"/vectordb": vectordb_volume},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
class MedGemmaSummarizer:

    @modal.enter()
    def load_models(self):
        import torch
        from transformers import pipeline
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from sentence_transformers import SentenceTransformer
        import tiktoken
        import time

        model_name = "google/medgemma-4b-it"
        print(f"Loading MedGemma 4B-IT...")
        start = time.time()

        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"MedGemma loaded in {time.time() - start:.2f}s")

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

        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.encoding = None

        print("All models loaded.")

    def _generate_text(self, messages, max_new_tokens=500, temperature=0.3):
        output = self.pipe(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        return output[0]["generated_text"][-1]["content"]

    def _retrieve_schemas(self, detected_disease):
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

    @modal.method()
    def generate_summary(
        self,
        transcript_text: str,
        openemr_text: str = "",
        patient_name: str = "Patient",
        detected_condition: str = "",
    ) -> Dict[str, Any]:
        import time

        model_name = "google/medgemma-4b-it"
        print(f"\nGenerating: {patient_name} | Model: MedGemma 4B-IT")
        start_total = time.time()

        detected_disease = detected_condition or "General"
        schema_context = self._retrieve_schemas(detected_disease)

        start_gen = time.time()
        prompt_content = SUMMARY_PROMPT_TEMPLATE.format(
            transcript=transcript_text,
            openemr=openemr_text if openemr_text else "No OpenEMR data available.",
            schema=schema_context,
        )

        messages = [
            {"role": "system", "content": "You are an expert medical scribe specialized in clinical documentation. Generate comprehensive SOAP-format medical summaries."},
            {"role": "user", "content": prompt_content},
        ]

        full_text = messages[0]["content"] + messages[1]["content"]
        if self.encoding:
            input_tokens = len(self.encoding.encode(full_text))
        else:
            input_tokens = int(len(full_text.split()) * 1.3)

        try:
            generated_text = self._generate_text(messages, max_new_tokens=2048)
            if self.encoding:
                output_tokens = len(self.encoding.encode(generated_text))
            else:
                output_tokens = int(len(generated_text.split()) * 1.3)
        except Exception as e:
            print(f"MedGemma generation failed: {e}")
            generated_text = f"Error generating summary: {str(e)}"
            output_tokens = 0

        generation_time = time.time() - start_gen
        total_time = time.time() - start_total
        print(f"  Tokens: {input_tokens:,} in / {output_tokens:,} out | Time: {total_time:.2f}s")

        return {
            "summary": generated_text,
            "patient_name": patient_name,
            "detected_disease": detected_disease,
            "retrieval_time": 0,
            "generation_time": generation_time,
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model_name,
        }


# ============================================================================
# Results Saver
# ============================================================================

def save_results(results: List[Dict[str, Any]], model_short_name: str, output_dir: str = "results/fareez"):
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CSV
    table_data = []
    for r in results:
        row = {
            "patient_name": r.get("patient_name", "Unknown"),
            "model": model_short_name,
            "detected_disease": r.get("detected_disease", ""),
            "total_time_s": r.get("total_time", 0.0),
            "input_tokens": r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    avg_row = {
        "patient_name": "AVERAGE",
        "model": model_short_name,
        "detected_disease": "",
        "total_time_s": df["total_time_s"].mean(),
        "input_tokens": df["input_tokens"].mean(),
        "output_tokens": df["output_tokens"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    csv_path = output_path / f"evaluation_results_{model_short_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Individual summaries
    summaries_dir = output_path / "fareez_summaries" / model_short_name
    summaries_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        patient_name = r.get("patient_name", "unknown")
        summary = r.get("summary", "")
        summary_file = summaries_dir / f"{patient_name}.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Model: {r.get('model', model_short_name)}\n")
            f.write(f"Patient: {patient_name}\n")
            f.write(f"Detected Disease: {r.get('detected_disease', 'N/A')}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Time: {r.get('total_time', 0):.2f}s\n")
            f.write(f"Tokens: {r.get('input_tokens', 0):,} in / {r.get('output_tokens', 0):,} out\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary if summary else "[No summary generated]")

    print(f"Saved {len(results)} summaries to {summaries_dir}/")


# ============================================================================
# Main Pipeline
# ============================================================================

@app.local_entrypoint()
def main(output_dir: str = "results/fareez"):
    """
    Run MedGemma 4B-IT summarization on Modal (GPU required).
    Groq models should be run via run_fareez_local.py instead.

    Args:
        output_dir: Output directory for results
    """
    import time

    from fareez_rag_loader import FareezLoader

    print("=" * 80)
    print("FAREEZ RAG SUMMARIZATION - MedGemma 4B-IT (Modal GPU)")
    print("=" * 80)

    loader = FareezLoader()
    patients = loader.get_entries()
    print(f"Loaded {len(patients)} Fareez entries\n")

    model_name = "google/medgemma-4b-it"
    model_short = "medgemma-4b"

    print(f"MODEL: {model_name}")
    print(f"{'='*80}")

    model_start = time.time()
    summarizer = MedGemmaSummarizer()
    results = []

    for i, patient in enumerate(patients):
        patient_name = patient["patient_name"]
        condition = patient.get("detected_condition", "")
        print(f"\n[{i+1}/{len(patients)}] {patient_name} ({patient['category']}, {condition})")

        try:
            result = summarizer.generate_summary.remote(
                transcript_text=patient["transcript"],
                openemr_text=patient.get("openemr_data", ""),
                patient_name=patient_name,
                detected_condition=condition,
            )
            results.append(result)
            print(f"  Done: {result.get('total_time', 0):.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "patient_name": patient_name,
                "summary": f"Error: {str(e)}",
                "error": str(e),
                "detected_disease": condition,
                "total_time": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "model": model_name,
            })

    save_results(results, model_short, output_dir)

    model_time = time.time() - model_start
    print(f"\n{model_name}: {len(results)} summaries in {model_time:.1f}s ({model_time/60:.1f} min)")
    print(f"Results saved to {output_dir}/")
