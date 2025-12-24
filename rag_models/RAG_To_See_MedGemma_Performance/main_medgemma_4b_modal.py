"""
Modal-based RAG system for medical transcript summarization
Uses MedGemma 4B-IT to generate SOAP-format summaries with schema guidance from vector DB
"""

import modal
import os
from pathlib import Path

# Define Modal app
app = modal.App("medical-summarization-rag-medgemma-4b")

# Create persistent volume reference for vector database
vectordb_volume = modal.Volume.from_name("medical-vectordb")

# Define the image with all dependencies
image = (
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
        "nltk>=3.8.1",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.13",
    )
)

# Model configuration
MODEL_NAME = "google/medgemma-4b-it"
CHROMA_PATH = "/vectordb/chroma_schema_improved"


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/vectordb": vectordb_volume},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def generate_summary(
        transcript_text: str,
        openemr_text: str = "",
        patient_name: str = "Patient",
) -> dict:
    """
    Generate SOAP-format medical summary from transcript using RAG.
    """
    import time
    import torch
    from transformers import pipeline
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from sentence_transformers import SentenceTransformer, util
    import tiktoken

    print(f"🔹 Starting summarization for {patient_name}")
    start_total = time.time()

    # ==============================
    # 1. LOAD VECTOR STORE
    # ==============================
    print(f"🔹 Loading Vector Store from: {CHROMA_PATH}")
    embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # ==============================
    # 2. LOAD MEDGEMMA 4B-IT PIPELINE
    # ==============================
    print(f"🔹 Loading MedGemma 4B-IT pipeline...")
    start_load = time.time()

    # Use pipeline approach (matches working Colab code)
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    load_time = time.time() - start_load
    print(f"⏱️ Model loading took {load_time:.2f}s")

    # Helper function using pipeline
    def generate_text(messages: list, max_new_tokens: int = 500) -> str:
        """Generate text using the pipeline."""
        output = pipe(messages, max_new_tokens=max_new_tokens)
        return output[0]["generated_text"][-1]["content"]

    # ==============================
    # 3. EXTRACT DISEASE USING LLM
    # ==============================
    print("🔹 Extracting disease from transcript using MedGemma 4B-IT...")
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

    detected_disease = generate_text(disease_messages, max_new_tokens=20).strip()

    # Clean up
    if not detected_disease:
        detected_disease = "General"
    detected_disease = detected_disease.split('\n')[0].split(',')[0].strip()

    print(f"✅ Detected Disease: {detected_disease}")

    # ==============================
    # 4. RETRIEVE SCHEMAS FROM VECTOR DB
    # ==============================
    print("🔹 Retrieving relevant schemas from vector DB...")

    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    collection_data = vector_store.get(include=["metadatas", "documents"])
    all_metadatas = collection_data["metadatas"]
    all_docs = collection_data["documents"]

    metadata_diseases = [m.get("diseases", "Unspecified") for m in all_metadatas]
    target_emb = sbert_model.encode(detected_disease, convert_to_tensor=True)
    candidate_embs = sbert_model.encode(metadata_diseases, convert_to_tensor=True)
    cosine_scores = util.cos_sim(target_emb, candidate_embs)[0]

    k = min(2, len(cosine_scores))
    top_k_result = cosine_scores.topk(k)
    top_indices = top_k_result.indices.tolist()

    schema_context = ""
    for rank, idx in enumerate(top_indices):
        doc_content = all_docs[idx]
        disease_meta = metadata_diseases[idx]
        schema_context += f"\n\n=== SCHEMA {rank+1} ({disease_meta}) ===\n{doc_content}"

    retrieval_time = time.time() - start_retrieval
    print(f"⏱️ Disease extraction and retrieval took {retrieval_time:.2f}s")

    # ==============================
    # 5. GENERATE SUMMARY
    # ==============================
    print("🔹 Generating summary...")
    start_gen = time.time()

    # Truncate inputs to prevent context overflow
    max_transcript_len = 5000
    max_openemr_len = 1500

    transcript_for_prompt = transcript_text[:max_transcript_len]
    openemr_for_prompt = openemr_text[:max_openemr_len] if openemr_text else ""

    if len(transcript_text) > max_transcript_len:
        print(f"⚠️ Truncated transcript from {len(transcript_text)} to {max_transcript_len} chars")

    summary_messages = [
        {
            "role": "system",
            "content": "You are an expert medical scribe specialized in clinical documentation. Generate comprehensive SOAP-format medical summaries."
        },
        {
            "role": "user",
            "content": f"""Generate a comprehensive medical summary in SOAP format from the following data:

### TRANSCRIPT (Doctor-patient conversation):
{transcript_for_prompt}

### OPENEMR EXTRACT (Electronic health record):
{openemr_for_prompt if openemr_for_prompt else "No OpenEMR data available."}

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

    # Calculate input tokens (approximate)
    prompt_text = summary_messages[0]["content"] + summary_messages[1]["content"]
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        input_tokens = len(encoding.encode(prompt_text))
    except:
        input_tokens = int(len(prompt_text.split()) * 1.3)

    print(f"📊 Input tokens: {input_tokens:,}")

    # Generate using pipeline
    generated_text = generate_text(summary_messages, max_new_tokens=2048)

    # Calculate output tokens
    try:
        output_tokens = len(encoding.encode(generated_text))
    except:
        output_tokens = int(len(generated_text.split()) * 1.3)

    generation_time = time.time() - start_gen
    total_time = time.time() - start_total

    print(f"📊 Output tokens: {output_tokens:,}")
    print(f"📊 Total tokens: {input_tokens + output_tokens:,}")
    print(f"⏱️ Generation took {generation_time:.2f}s")
    print(f"⏱️ Total time: {total_time:.2f}s")

    return {
        "summary": generated_text,
        "detected_disease": detected_disease,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


@app.function(
    image=image,
    volumes={"/vectordb": vectordb_volume},
)
def evaluate_summary(generated: str, reference: str) -> dict:
    """Evaluate generated summary against reference using multiple metrics."""
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer, util
    from bert_score import score
    import nltk

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    print("🔹 Computing evaluation metrics...")

    # Handle empty generated text
    if not generated or not generated.strip():
        print("⚠️ Warning: Generated text is empty, returning zero scores")
        return {
            "bleu": 0.0,
            "rouge_l": 0.0,
            "sbert_coherence": 0.0,
            "bert_f1": 0.0,
        }

    bleu = sentence_bleu([reference.split()], generated.split())

    scorer_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = scorer_rouge.score(reference, generated)["rougeL"].fmeasure

    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    ref_emb = sbert_model.encode(reference, convert_to_tensor=True)
    gen_emb = sbert_model.encode(generated, convert_to_tensor=True)
    sbert_coherence = util.cos_sim(ref_emb, gen_emb).item()

    P, R, F1 = score([generated], [reference], lang="en", verbose=False)
    bert_f1 = F1.mean().item()

    results = {
        "bleu": bleu,
        "rouge_l": rouge_l,
        "sbert_coherence": sbert_coherence,
        "bert_f1": bert_f1,
    }

    print(f"✅ BLEU: {bleu:.4f}")
    print(f"✅ ROUGE-L: {rouge_l:.4f}")
    print(f"✅ SBERT: {sbert_coherence:.4f}")
    print(f"✅ BERTScore F1: {bert_f1:.4f}")

    return results


@app.local_entrypoint()
def main(
        transcript_path: str = "data/transcription_rakesh.txt",
        openemr_path: str = "data/openemr_rakesh.txt",
        reference_path: str = "data/reference_rakesh.txt",
        patient_name: str = "Rakesh",
        output_dir: str = "results",
):
    """Main function to run summarization and evaluation."""
    from pathlib import Path

    print("=" * 60)
    print(f"Medical Transcript Summarization (MedGemma 4B-IT) - {patient_name}")
    print("=" * 60)

    transcript_text = Path(transcript_path).read_text(encoding="utf-8")
    openemr_text = Path(openemr_path).read_text(encoding="utf-8") if Path(openemr_path).exists() else ""
    reference_text = Path(reference_path).read_text(encoding="utf-8") if Path(reference_path).exists() else ""

    print("\n🚀 Generating summary...")
    result = generate_summary.remote(
        transcript_text=transcript_text,
        openemr_text=openemr_text,
        patient_name=patient_name,
    )

    eval_results = {}
    if reference_text:
        print("\n🔍 Evaluating summary...")
        eval_results = evaluate_summary.remote(
            generated=result["summary"],
            reference=reference_text,
        )

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    result_file = output_path / f"evaluation_{patient_name}.txt"

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Medical Summary Evaluation (MedGemma 4B-IT) - {patient_name}\n")
        f.write("=" * 60 + "\n\n")

        f.write("### GENERATION METRICS ###\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Detected Disease: {result['detected_disease']}\n")
        f.write(f"Retrieval Time: {result['retrieval_time']:.2f}s\n")
        f.write(f"Generation Time: {result['generation_time']:.2f}s\n")
        f.write(f"Total Time: {result['total_time']:.2f}s\n")
        f.write(f"Input Tokens: {result['input_tokens']:,}\n")
        f.write(f"Output Tokens: {result['output_tokens']:,}\n")
        f.write(f"Total Tokens: {result['total_tokens']:,}\n\n")

        if eval_results:
            f.write("### EVALUATION METRICS ###\n")
            f.write(f"BLEU Score: {eval_results['bleu']:.4f}\n")
            f.write(f"ROUGE-L Score: {eval_results['rouge_l']:.4f}\n")
            f.write(f"SBERT Coherence: {eval_results['sbert_coherence']:.4f}\n")
            f.write(f"BERTScore F1: {eval_results['bert_f1']:.4f}\n\n")

        f.write("### GENERATED SUMMARY ###\n")
        f.write(result["summary"] if result["summary"] else "[No summary generated]")

    print(f"\n✅ Results saved to: {result_file}")
    print("\n" + "=" * 60)
    print("Summary Preview:")
    print("=" * 60)
    if result["summary"]:
        print(result["summary"][:500] + "..." if len(result["summary"]) > 500 else result["summary"])
    else:
        print("[No summary generated]")

    if eval_results:
        print("\n" + "=" * 60)
        print("Evaluation Metrics:")
        print("=" * 60)
        for metric, value in eval_results.items():
            print(f"{metric.upper()}: {value:.4f}")