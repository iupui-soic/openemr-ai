"""
Modal-based RAG system for medical transcript summarization
Uses MedGemma 27B-Text-IT (Unsloth 4-bit pre-quantized) for inference

Complete pipeline that:
1. Fetches all patients from Notion database (via summary_utils.NotionFetcher)
2. Generates summaries for each patient using RAG + MedGemma 27B (4-bit)
3. Evaluates summaries against manual references
4. Outputs: evaluation_results.csv + individual summary files

Usage:
    modal run rag_medgemma_27b_4bit_pipeline.py
    modal run rag_medgemma_27b_4bit_pipeline.py --output-dir results/medgemma-27b-4bit

Requirements (local):
    pip install notion-client httpx pandas python-dotenv
"""

import modal
import os
from typing import Dict, List, Any

# Local import for Notion fetching (runs on local machine, not Modal)
from summary_utils import NotionFetcher

# ============================================================================
# Modal App Configuration
# ============================================================================

app = modal.App("medical-summarization-rag-medgemma-27b-4bit")

# Persistent volume for vector database
vectordb_volume = modal.Volume.from_name("medical-vectordb")

# Model configuration - Using Unsloth pre-quantized 4-bit version
MODEL_NAME = "unsloth/medgemma-27b-text-it-unsloth-bnb-4bit"
MODEL_SHORT_NAME = "medgemma-27b-4bit"
CHROMA_PATH = "/vectordb/chroma_schema_improved"

# ============================================================================
# Modal Images
# ============================================================================

# Image for summarization (MedGemma + RAG)
summarizer_image = (
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

# Image for evaluation
evaluator_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "nltk>=3.8.1",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.13",
        "sentence-transformers>=2.2.2",
        "pandas>=2.0.0",
    )
)


# ============================================================================
# Medical Summarizer Class (with persistent model loading)
# ============================================================================

@app.cls(
    image=summarizer_image,
    gpu="A100",
    timeout=3600,
    volumes={"/vectordb": vectordb_volume},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
class MedicalSummarizer:
    """
    RAG-based medical summarizer using MedGemma 27B-Text-IT (4-bit).

    Models are loaded once in @modal.enter() and reused across all
    generate_summary() calls for efficient batch processing.

    This significantly reduces costs since MedGemma 27B takes ~60-90s to load.
    """

    @modal.enter()
    def load_models(self):
        """Load all models once when container starts."""
        import torch
        from transformers import pipeline
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from sentence_transformers import SentenceTransformer
        import tiktoken

        print("🔄 Loading models (one-time initialization)...")
        print(f"   Model: {MODEL_NAME}")

        # ==============================
        # LOAD MEDGEMMA 27B (4-BIT) PIPELINE
        # ==============================
        print("  → Loading MedGemma 27B-Text-IT (4-bit) pipeline...")
        import time
        start_load = time.time()

        self.pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        load_time = time.time() - start_load
        print(f"  → MedGemma loaded in {load_time:.2f}s")

        # ==============================
        # LOAD RAG COMPONENTS
        # ==============================
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

    def _generate_text(self, messages: list, max_new_tokens: int = 500, temperature: float = 0.3) -> str:
        """Generate text using the pipeline."""
        output = self.pipe(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        return output[0]["generated_text"][-1]["content"]

    @modal.method()
    def generate_summary(
            self,
            transcript_text: str,
            openemr_text: str = "",
            patient_name: str = "Patient",
    ) -> Dict[str, Any]:
        """
        Generate SOAP-format medical summary from transcript using RAG + MedGemma.

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
        # 1. EXTRACT DISEASE USING MEDGEMMA
        # ==============================
        print("🔹 Extracting disease from transcript...")
        start_retrieval = time.time()

        disease_messages = [
            {
                "role": "system",
                "content": "You are a medical expert. Identify the primary medical condition from clinical conversations. Respond with ONLY the disease name."
            },
            {
                "role": "user",
                "content": f"""Read this transcript and identify the PRIMARY medical condition being discussed.

Return ONLY the disease name (e.g., "COPD", "Diabetes", "Hypertension", "Asthma"). 
If no specific disease is mentioned, return "General".

Transcript:
{transcript_text[:2000]}

Primary Disease:"""
            }
        ]

        detected_disease = self._generate_text(disease_messages, max_new_tokens=20).strip()

        # Clean up
        if not detected_disease:
            detected_disease = "General"
        detected_disease = detected_disease.split('\n')[0].split(',')[0].strip()

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
        # 3. GENERATE SUMMARY WITH MEDGEMMA
        # ==============================
        print("🔹 Generating summary with MedGemma 27B...")
        start_gen = time.time()

        summary_messages = [
            {
                "role": "system",
                "content": "You are an expert medical scribe specialized in clinical documentation. Generate comprehensive SOAP-format medical summaries."
            },
            {
                "role": "user",
                "content": f"""Generate a comprehensive medical summary in SOAP format from the following data:

### TRANSCRIPT (Doctor-patient conversation):
{transcript_text}

### OPENEMR EXTRACT (Electronic health record):
{openemr_text if openemr_text else "No OpenEMR data available."}

### SCHEMA GUIDE (Required sections and structure):
{schema_context}

### OUTPUT FORMAT REQUIREMENTS
- Generate a NARRATIVE TEXT document, NOT JSON or structured data
- Use clear section headers (e.g., "Patient Information", "Chief Complaint", "History of Present Illness")
- Write in complete sentences and paragraphs
- Use professional medical documentation prose style
- Format similar to a hospital discharge summary

### INSTRUCTIONS
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

        # Generate with MedGemma
        try:
            generated_text = self._generate_text(summary_messages, max_new_tokens=2048)

            if self.encoding:
                output_tokens = len(self.encoding.encode(generated_text))
            else:
                output_tokens = int(len(generated_text.split()) * 1.3)

        except Exception as e:
            print(f"❌ MedGemma generation failed: {e}")
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
# Summary Evaluator Class (with persistent model loading)
# ============================================================================

@app.cls(
    image=evaluator_image,
    timeout=1800,
    cpu=2,
    memory=4096,
)
class SummaryEvaluator:
    """
    Evaluator for medical summaries using multiple metrics.

    SBERT model is loaded once and reused across evaluations.
    """

    @modal.enter()
    def load_models(self):
        """Load evaluation models once."""
        from sentence_transformers import SentenceTransformer
        import nltk

        print("🔄 Loading evaluation models...")

        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

        # Load SBERT for coherence scoring
        print("  → Loading SBERT model...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        print("✅ Evaluation models loaded!")

    @modal.method()
    def evaluate(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Evaluate generated summary against reference using multiple metrics.

        Args:
            generated: Generated summary text
            reference: Reference summary text

        Returns:
            dict with BLEU, ROUGE-L, SBERT coherence, and BERTScore F1
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        from sentence_transformers import util
        from bert_score import score

        print("🔹 Computing evaluation metrics...")

        # Handle empty inputs
        if not generated or not generated.strip() or not reference:
            print("⚠️ Warning: Empty text, returning zero scores")
            return {
                "bleu": 0.0,
                "rouge_l": 0.0,
                "sbert_coherence": 0.0,
                "bert_f1": 0.0,
            }

        # BLEU Score (with smoothing for short texts)
        smoother = SmoothingFunction()
        bleu = sentence_bleu(
            [reference.split()],
            generated.split(),
            smoothing_function=smoother.method1
        )

        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = scorer.score(reference, generated)["rougeL"].fmeasure

        # SBERT Coherence (using pre-loaded model)
        ref_emb = self.sbert_model.encode(reference, convert_to_tensor=True)
        gen_emb = self.sbert_model.encode(generated, convert_to_tensor=True)
        sbert_coherence = util.cos_sim(ref_emb, gen_emb).item()

        # BERTScore
        P, R, F1 = score([generated], [reference], lang="en", verbose=False)
        bert_f1 = F1.mean().item()

        results = {
            "bleu": bleu,
            "rouge_l": rouge_l,
            "sbert_coherence": sbert_coherence,
            "bert_f1": bert_f1,
        }

        print(f"  BLEU: {bleu:.4f} | ROUGE-L: {rouge_l:.4f} | SBERT: {sbert_coherence:.4f} | BERTScore: {bert_f1:.4f}")

        return results


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
    print("\n" + "=" * 100)
    print(f"EVALUATION RESULTS - {MODEL_NAME}")
    print("=" * 100)
    print(df.to_string(index=False, float_format="%.4f"))
    print("=" * 100)

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
            f.write(f"BLEU:            {r.get('bleu', 0):.4f}\n")
            f.write(f"ROUGE-L:         {r.get('rouge_l', 0):.4f}\n")
            f.write(f"SBERT Coherence: {r.get('sbert_coherence', 0):.4f}\n")
            f.write(f"BERTScore F1:    {r.get('bert_f1', 0):.4f}\n\n")

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

    # Initialize Modal classes (models load once via @modal.enter())
    summarizer = MedicalSummarizer()
    evaluator = SummaryEvaluator()

    results = []

    for i, patient in enumerate(patients):
        patient_name = patient["patient_name"]
        print(f"\n[{i+1}/{len(patients)}] Processing: {patient_name}")

        try:
            # Generate summary (runs on Modal A100 GPU)
            summary_result = summarizer.generate_summary.remote(
                transcript_text=patient["transcript"],
                openemr_text=patient.get("openemr_data", ""),
                patient_name=patient_name,
            )

            # Evaluate against reference (runs on Modal CPU)
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
                }

            # Combine results
            combined = {**summary_result, **eval_metrics}
            results.append(combined)

            print(f"   ✅ Completed: BLEU={eval_metrics['bleu']:.4f}, BERTScore={eval_metrics['bert_f1']:.4f}")

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