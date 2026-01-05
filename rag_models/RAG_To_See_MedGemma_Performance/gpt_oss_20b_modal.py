"""
Modal-based RAG system for medical transcript summarization
Uses Groq API with GPT-OSS-20B for inference, Modal for vector database storage

Complete pipeline that:
1. Fetches all patients from Notion database (via summary_utils.NotionFetcher)
2. Generates summaries for each patient using RAG + GPT-OSS-20B (Groq)
3. Evaluates summaries against manual references (via shared evaluator service)
4. Outputs: evaluation_results.csv + individual summary files

Usage:
    modal run rag_gpt_oss_20b_pipeline.py
    modal run rag_gpt_oss_20b_pipeline.py --output-dir results/gpt-oss-20b

Prerequisites:
    Deploy shared evaluator first: modal deploy shared_evaluator_service.py

Requirements (local):
    pip install notion-client httpx pandas python-dotenv
"""

import modal
import os
from typing import Dict, List, Any

# ============================================================================
# Modal App Configuration
# ============================================================================

app = modal.App("medical-summarization-rag-gpt-oss-20b")

# Persistent volume for vector database
vectordb_volume = modal.Volume.from_name("medical-vectordb")

# Model configuration
MODEL_NAME = "openai/gpt-oss-20b"
MODEL_SHORT_NAME = "gpt-oss-20b"
CHROMA_PATH = "/vectordb/chroma_schema_improved"

# ============================================================================
# Modal Images
# ============================================================================

# Image for summarization (Groq + RAG)
summarizer_image = (
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


# ============================================================================
# Medical Summarizer Class (with persistent model loading)
# ============================================================================

@app.cls(
    image=summarizer_image,
    timeout=3600,
    volumes={"/vectordb": vectordb_volume},
    secrets=[modal.Secret.from_dict({"GROQ_API_KEY": os.environ.get("GROQ_API_KEY", "")})],
)
class MedicalSummarizer:
    """
    RAG-based medical summarizer using Groq API with GPT-OSS-20B.

    Models are loaded once in @modal.enter() and reused across all
    generate_summary() calls for efficient batch processing.
    """

    @modal.enter()
    def load_models(self):
        """Load all models once when container starts."""
        from groq import Groq
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from sentence_transformers import SentenceTransformer
        import tiktoken

        print("🔄 Loading models (one-time initialization)...")
        print(f"   Model: {MODEL_NAME}")

        # Initialize Groq client
        print("  → Initializing Groq client...")
        self.client = Groq()

        # Load BioBERT embeddings for ChromaDB
        print("  → Loading BioBERT embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )

        # Load vector store
        print(f"  → Loading Vector Store from: {CHROMA_PATH}")
        self.vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )

        # Pre-fetch collection data (avoids repeated DB queries)
        print("  → Pre-fetching collection data...")
        collection_data = self.vector_store.get(include=["metadatas", "documents"])
        self.all_metadatas = collection_data["metadatas"]
        self.all_docs = collection_data["documents"]
        self.metadata_diseases = [m.get("diseases", "Unspecified") for m in self.all_metadatas]

        # Load SBERT for semantic matching
        print("  → Loading SBERT model...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Pre-encode all disease metadata for faster retrieval
        print("  → Pre-encoding disease embeddings...")
        self.candidate_embs = self.sbert_model.encode(self.metadata_diseases, convert_to_tensor=True)

        # Initialize tokenizer for token counting
        print("  → Initializing tokenizer...")
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoding = None

        print("✅ All models loaded successfully!")

    @modal.method()
    def generate_summary(
            self,
            transcript_text: str,
            openemr_text: str = "",
            patient_name: str = "Patient",
    ) -> Dict[str, Any]:
        """
        Generate SOAP-format medical summary from transcript using RAG + Groq.

        Args:
            transcript_text: Doctor-patient conversation transcript
            openemr_text: OpenEMR extract (optional)
            patient_name: Patient name for logging

        Returns:
            dict with summary, detected_disease, timing metrics, token counts
        """
        import time
        from sentence_transformers import util

        print(f"\n{'='*60}")
        print(f"🔹 Generating summary for: {patient_name}")
        print(f"🔹 Model: {MODEL_NAME}")
        print(f"{'='*60}")
        start_total = time.time()

        # ==============================
        # 1. EXTRACT DISEASE USING GROQ
        # ==============================
        print("🔹 Extracting disease from transcript...")
        start_retrieval = time.time()

        disease_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a medical expert. Identify the primary disease or medical "
                    "condition discussed in a clinical conversation. "
                    "Respond with ONLY the disease name."
                ),
            },
            {
                "role": "user",
                "content": f"""
Read the following doctor-patient conversation and identify the PRIMARY medical condition.

Rules:
- Return ONLY the disease name
- If multiple conditions are mentioned, choose the most prominent
- If no disease is clearly stated, return "General"

Transcript:
{transcript_text[:2000]}

Primary Disease:
""",
            },
        ]

        disease_response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=disease_prompt,
            temperature=0.3,
            max_tokens=20,
        )

        raw_disease = disease_response.choices[0].message.content
        detected_disease = raw_disease.strip() if raw_disease else "General"
        print(f"✅ Detected Disease: {detected_disease}")

        # ==============================
        # 2. RETRIEVE SCHEMAS FROM VECTOR DB
        # ==============================
        print("🔹 Retrieving relevant schemas from vector DB...")

        # Encode query and find top matches (using pre-computed candidate embeddings)
        target_emb = self.sbert_model.encode(detected_disease, convert_to_tensor=True)
        cosine_scores = util.cos_sim(target_emb, self.candidate_embs)[0]

        # Get top 2 schemas
        k = min(2, len(cosine_scores))
        top_k_result = cosine_scores.topk(k)
        top_indices = top_k_result.indices.tolist()

        schema_context = ""
        for rank, idx in enumerate(top_indices):
            doc_content = self.all_docs[idx]
            disease_meta = self.metadata_diseases[idx]
            schema_context += f"\n\n=== SCHEMA {rank+1} ({disease_meta}) ===\n{doc_content}"

        retrieval_time = time.time() - start_retrieval
        print(f"⏱️ Disease extraction + retrieval: {retrieval_time:.2f}s")

        # ==============================
        # 3. GENERATE SUMMARY WITH GROQ
        # ==============================
        print("🔹 Generating summary with Groq (GPT-OSS-20B)...")
        start_gen = time.time()

        summary_messages = [
            {
                "role": "system",
                "content": "You are an expert medical scribe specialized in clinical documentation. Generate comprehensive SOAP-format medical summaries."
            },
            {
                "role": "user",
                "content": f"""Generate a comprehensive medical summary in SOAP format from the following data:

TRANSCRIPT (Doctor-patient conversation):
{transcript_text}

OPENEMR EXTRACT (Electronic health record):
{openemr_text if openemr_text else "No OpenEMR data available."}

SCHEMA GUIDE (Reference sections to include):
{schema_context}

OUTPUT FORMAT REQUIREMENTS:
- Generate a NARRATIVE TEXT document, NOT JSON or structured data
- Use clear section headers (e.g., "Patient Information", "Chief Complaint", "History of Present Illness")
- Write in complete sentences and paragraphs
- Use professional medical documentation prose style
- Format similar to a hospital discharge summary

INSTRUCTIONS:
1. Use the SCHEMA GUIDE as a reference for which sections to include
2. Extract relevant information from the TRANSCRIPT and OPENEMR EXTRACT
3. Write in narrative prose with proper paragraphs
4. If information for a section is missing, write "No information available."
5. If TRANSCRIPT and OPENEMR conflict, trust the TRANSCRIPT for current status
6. Do NOT include any meta-commentary, explanations, or references to this prompt
7. Do NOT output JSON, XML, or any structured data format
8. Do NOT hallucinate or invent information not present in the inputs

Generate the medical summary now in narrative prose format, beginning with "Patient Information":"""
            }
        ]

        # Calculate input tokens
        prompt_text = summary_messages[0]["content"] + summary_messages[1]["content"]
        if self.encoding:
            input_tokens = len(self.encoding.encode(prompt_text))
        else:
            input_tokens = int(len(prompt_text.split()) * 1.3)

        print(f"📊 Input tokens: {input_tokens:,}")

        # Generate with Groq
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=summary_messages,
                temperature=0.3,
                max_tokens=2048,
            )
            generated_text = response.choices[0].message.content.strip()

            if self.encoding:
                output_tokens = len(self.encoding.encode(generated_text))
            else:
                output_tokens = int(len(generated_text.split()) * 1.3)

        except Exception as e:
            print(f"❌ Groq generation failed: {e}")
            generated_text = f"Error generating summary: {str(e)}"
            output_tokens = 0

        generation_time = time.time() - start_gen
        total_time = time.time() - start_total

        print(f"📊 Tokens: {input_tokens:,} in / {output_tokens:,} out / {input_tokens + output_tokens:,} total")
        print(f"⏱️ Generation: {generation_time:.2f}s | Total: {total_time:.2f}s")

        return {
            "summary": generated_text,
            "patient_name": patient_name,
            "detected_disease": detected_disease,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": MODEL_NAME,
        }


# ============================================================================
# Results Saver (runs locally)
# ============================================================================

def save_results(
        results: List[Dict[str, Any]],
        output_dir: str = "results",
) -> None:
    """
    Save evaluation results table and individual summaries.

    Args:
        results: List of result dicts from pipeline
        output_dir: Directory to save outputs
    """
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ==============================
    # 1. SAVE EVALUATION RESULTS TABLE (CSV)
    # ==============================
    print("\n📊 Saving evaluation results table...")

    table_data = []
    for r in results:
        row = {
            "patient_name": r.get("patient_name", "Unknown"),
            "model": MODEL_SHORT_NAME,
            "detected_disease": r.get("detected_disease", ""),
            "bleu": r.get("bleu", 0.0),
            "rouge_l": r.get("rouge_l", 0.0),
            "sbert_coherence": r.get("sbert_coherence", 0.0),
            "bert_f1": r.get("bert_f1", 0.0),
            "scispacy_entity_recall": r.get("scispacy_entity_recall", 0.0),
            "medcat_entity_recall": r.get("medcat_entity_recall", 0.0),
            "total_time_s": r.get("total_time", 0.0),
            "input_tokens": r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Calculate averages
    avg_row = {
        "patient_name": "AVERAGE",
        "model": MODEL_SHORT_NAME,
        "detected_disease": "",
        "bleu": df["bleu"].mean(),
        "rouge_l": df["rouge_l"].mean(),
        "sbert_coherence": df["sbert_coherence"].mean(),
        "bert_f1": df["bert_f1"].mean(),
        "scispacy_entity_recall": df["scispacy_entity_recall"].mean(),
        "medcat_entity_recall": df["medcat_entity_recall"].mean(),
        "total_time_s": df["total_time_s"].mean(),
        "input_tokens": df["input_tokens"].mean(),
        "output_tokens": df["output_tokens"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save CSV
    csv_path = output_path / f"evaluation_results_{MODEL_SHORT_NAME}.csv"
    df.to_csv(csv_path, index=False)
    print(f"   ✅ Saved: {csv_path}")

    # Print table to console
    print("\n" + "=" * 120)
    print(f"EVALUATION RESULTS - {MODEL_NAME}")
    print("=" * 120)
    print(df.to_string(index=False, float_format="%.4f"))
    print("=" * 120)

    # ==============================
    # 2. SAVE INDIVIDUAL SUMMARIES
    # ==============================
    print("\n📝 Saving individual summaries...")

    summaries_dir = output_path / "summaries"
    summaries_dir.mkdir(exist_ok=True)

    for r in results:
        patient_name = r.get("patient_name", "unknown")
        summary = r.get("summary", "")

        summary_file = summaries_dir / f"summary_{patient_name}_{MODEL_SHORT_NAME}.txt"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"MEDICAL SUMMARY - {patient_name.upper()}\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")

            f.write(f"Detected Disease: {r.get('detected_disease', 'N/A')}\n")
            f.write(f"Generation Time: {r.get('total_time', 0):.2f}s\n")
            f.write(f"Tokens: {r.get('input_tokens', 0):,} in / {r.get('output_tokens', 0):,} out\n\n")

            f.write(f"{'='*60}\n")
            f.write("EVALUATION METRICS\n")
            f.write(f"{'='*60}\n")
            f.write(f"BLEU:                   {r.get('bleu', 0):.4f}\n")
            f.write(f"ROUGE-L:                {r.get('rouge_l', 0):.4f}\n")
            f.write(f"SBERT Coherence:        {r.get('sbert_coherence', 0):.4f}\n")
            f.write(f"BERTScore F1:           {r.get('bert_f1', 0):.4f}\n")
            f.write(f"scispaCy Entity Recall: {r.get('scispacy_entity_recall', 0):.4f}\n")
            f.write(f"MedCAT Entity Recall:   {r.get('medcat_entity_recall', 0):.4f}\n\n")

            f.write(f"{'='*60}\n")
            f.write("GENERATED SUMMARY\n")
            f.write(f"{'='*60}\n\n")
            f.write(summary if summary else "[No summary generated]")

        print(f"   ✅ {summary_file.name}")

    # ==============================
    # 3. SAVE CONSOLIDATED REPORT
    # ==============================
    report_path = output_path / f"full_report_{MODEL_SHORT_NAME}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"MEDICAL SUMMARIZATION EVALUATION REPORT\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("EVALUATION METRICS SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(df.to_string(index=False, float_format="%.4f"))
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("INDIVIDUAL SUMMARIES\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            patient = r.get("patient_name", "Unknown")
            f.write(f"\n{'#'*80}\n")
            f.write(f"# {patient.upper()}\n")
            f.write(f"{'#'*80}\n\n")
            f.write(r.get("summary", "No summary generated."))
            f.write("\n\n")

    print(f"   ✅ Full report: {report_path}")
    print(f"\n✅ All results saved to: {output_path}/")


# ============================================================================
# Main Pipeline (Local Entrypoint)
# ============================================================================

@app.local_entrypoint()
def main(output_dir: str = "results"):
    """
    Main pipeline: Fetch patients → Generate summaries → Evaluate → Save results.

    Args:
        output_dir: Directory for output files (CSV, summaries, report)
    """
    import time

    # Import here - this runs LOCALLY only, not on Modal containers
    from summary_utils import NotionFetcher

    print("=" * 80)
    print("🏥 MEDICAL TRANSCRIPT SUMMARIZATION PIPELINE")
    print(f"   Model: {MODEL_NAME}")
    print("=" * 80)

    pipeline_start = time.time()

    # ==============================
    # STEP 1: FETCH PATIENTS FROM NOTION
    # ==============================
    print("\n[1/3] FETCHING PATIENT DATA FROM NOTION")
    print("-" * 40)

    try:
        fetcher = NotionFetcher()
        patients = fetcher.get_entries()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("   Make sure VISHNU_NOTION and VISHNU_NOTION_DB_ID are set in your .env file")
        return

    if not patients:
        print("❌ No patients found in database!")
        return

    print(f"   Found {len(patients)} patients to process")

    # ==============================
    # STEP 2: GENERATE SUMMARIES & EVALUATE (on Modal)
    # ==============================
    print("\n[2/3] GENERATING SUMMARIES & EVALUATING")
    print("-" * 40)

    # Initialize summarizer (local to this app)
    summarizer = MedicalSummarizer()

    # Lookup shared evaluator service (must be deployed first)
    try:
        evaluator_app = modal.App.lookup("shared-evaluator-service")
        #SummaryEvaluator = evaluator_app.cls("SummaryEvaluator")
        SummaryEvaluator = evaluator_app.SummaryEvaluator
        evaluator = SummaryEvaluator()
        print("✅ Connected to shared evaluator service")
    except Exception as e:
        print(f"❌ Error: Could not connect to shared evaluator service: {e}")
        print("   Make sure to deploy it first: modal deploy shared_evaluator_service.py")
        return

    results = []

    for i, patient in enumerate(patients):
        patient_name = patient["patient_name"]
        print(f"\n[{i+1}/{len(patients)}] Processing: {patient_name}")

        try:
            # Generate summary (runs on Modal - Groq API call)
            summary_result = summarizer.generate_summary.remote(
                transcript_text=patient["transcript"],
                openemr_text=patient.get("openemr_data", ""),
                patient_name=patient_name,
            )

            # Evaluate against reference (runs on shared evaluator service)
            reference = patient.get("manual_reference_summary", "")
            if reference:
                eval_metrics = evaluator.evaluate.remote(
                    generated=summary_result["summary"],
                    reference=reference,
                )
            else:
                print(f"   ⚠️ No reference summary for {patient_name}, skipping evaluation")
                eval_metrics = {
                    "bleu": 0.0,
                    "rouge_l": 0.0,
                    "sbert_coherence": 0.0,
                    "bert_f1": 0.0,
                    "scispacy_entity_recall": 0.0,
                    "medcat_entity_recall": 0.0,
                }

            # Combine results
            combined = {**summary_result, **eval_metrics}
            results.append(combined)

            print(f"   ✅ Completed: BLEU={eval_metrics['bleu']:.4f}, BERTScore={eval_metrics['bert_f1']:.4f}, scispaCy={eval_metrics['scispacy_entity_recall']:.4f}, MedCAT={eval_metrics['medcat_entity_recall']:.4f}")

        except Exception as e:
            print(f"   ❌ Error processing {patient_name}: {e}")
            results.append({
                "patient_name": patient_name,
                "summary": f"Error: {str(e)}",
                "error": str(e),
                "bleu": 0.0,
                "rouge_l": 0.0,
                "sbert_coherence": 0.0,
                "bert_f1": 0.0,
                "scispacy_entity_recall": 0.0,
                "medcat_entity_recall": 0.0,
            })

    # ==============================
    # STEP 3: SAVE RESULTS
    # ==============================
    print("\n[3/3] SAVING RESULTS")
    print("-" * 40)

    save_results(results, output_dir)

    pipeline_time = time.time() - pipeline_start

    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE")
    print(f"   Patients processed: {len(results)}")
    print(f"   Total time: {pipeline_time:.1f}s ({pipeline_time/60:.1f} min)")
    print(f"   Output directory: {output_dir}/")
    print("=" * 80)