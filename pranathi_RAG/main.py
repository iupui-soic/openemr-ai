"""
Modal-based RAG system for medical transcript summarization
Uses Llama 3.1 8B to generate SOAP-format summaries with schema guidance from vector DB
"""

import modal
import os
from pathlib import Path

# Define Modal app
app = modal.App("medical-summarization-rag")

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
        "transformers>=4.37.0",
        "torch>=2.1.0",
        "huggingface-hub>=0.20.0",
        "vllm>=0.3.0",
        "tiktoken>=0.5.0",
    )
)

# Model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CHROMA_PATH = "/vectordb/chroma_schema_improved"


@app.function(
    image=image,
    gpu="A10G",  # A10G is good for Llama 3.1 8B
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

    Args:
        transcript_text: Doctor-patient conversation transcript
        openemr_text: OpenEMR extract (optional)
        patient_name: Patient name for logging

    Returns:
        dict with summary, retrieval info, and metrics
    """
    import time
    from vllm import LLM, SamplingParams
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
    # 2. DISEASE EXTRACTION & RETRIEVAL
    # ==============================
    print("🔹 Extracting disease from transcript...")
    start_retrieval = time.time()

    # Simple keyword-based disease extraction (you can enhance this)
    diseases_keywords = {
        "COPD": ["copd", "chronic obstructive", "emphysema", "chronic bronchitis"],
        "Diabetes": ["diabetes", "diabetic", "blood sugar", "glucose"],
        "Hypertension": ["hypertension", "high blood pressure", "htn"],
        "Asthma": ["asthma", "wheezing", "bronchospasm"],
    }

    transcript_lower = transcript_text.lower()
    detected_disease = "General"
    for disease, keywords in diseases_keywords.items():
        if any(kw in transcript_lower for kw in keywords):
            detected_disease = disease
            break

    print(f"✅ Detected Disease: {detected_disease}")

    # Semantic retrieval from vector DB
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    collection_data = vector_store.get(include=["metadatas", "documents"])
    all_metadatas = collection_data["metadatas"]
    all_ids = collection_data["ids"]
    all_docs = collection_data["documents"]

    # Encode and find top matches
    metadata_diseases = [m.get("diseases", "Unspecified") for m in all_metadatas]
    target_emb = sbert_model.encode(detected_disease, convert_to_tensor=True)
    candidate_embs = sbert_model.encode(metadata_diseases, convert_to_tensor=True)
    cosine_scores = util.cos_sim(target_emb, candidate_embs)[0]

    # Get top 2 schemas
    k = min(2, len(cosine_scores))
    top_k_result = cosine_scores.topk(k)
    top_indices = top_k_result.indices.tolist()

    schema_context = ""
    for rank, idx in enumerate(top_indices):
        doc_content = all_docs[idx]
        disease_meta = metadata_diseases[idx]
        schema_context += f"\n\n=== SCHEMA {rank+1} ({disease_meta}) ===\n{doc_content}"

    retrieval_time = time.time() - start_retrieval
    print(f"⏱️ Retrieval took {retrieval_time:.2f}s")

    # ==============================
    # 3. LOAD LLAMA MODEL
    # ==============================
    print("🔹 Loading Llama 3.1 8B model...")
    start_load = time.time()

    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=8192,
    )

    load_time = time.time() - start_load
    print(f"⏱️ Model loading took {load_time:.2f}s")

    # ==============================
    # 4. GENERATE SUMMARY
    # ==============================
    print("🔹 Generating summary...")
    start_gen = time.time()

    # Build prompt
    prompt = f"""You are an expert medical scribe. Generate a comprehensive medical summary in SOAP format.

### INPUT DATA
1. **TRANSCRIPT**: Doctor-patient conversation
2. **OPENEMR EXTRACT**: Patient's electronic health record
3. **SCHEMA GUIDE**: Required sections and structure

### INSTRUCTIONS
- **Structure**: Follow the sections in the SCHEMA GUIDE exactly
- **Content**: Extract information from TRANSCRIPT and OPENEMR EXTRACT
- **Missing Info**: Write "Information not available" if data is missing
- **Style**: Professional medical documentation, use bullet points for lists
- **Conflict**: Trust TRANSCRIPT over OPENEMR for current status

### CRITICAL REQUIREMENTS
1. Use EXACT structure from SCHEMA GUIDE
2. Do NOT omit sections even if empty
3. Do NOT hallucinate information
4. Include Patient Information section at top

--- START OF DATA ---

**TRANSCRIPT**:
{transcript_text}

**OPENEMR EXTRACT**:
{openemr_text if openemr_text else "No OpenEMR data available."}

**SCHEMA GUIDE**:
{schema_context}

--- END OF DATA ---

**OUTPUT**:
Generate the full medical summary below in clean, formatted text (NOT JSON).
"""

    # Calculate input tokens
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        input_tokens = len(encoding.encode(prompt))
    except:
        input_tokens = int(len(prompt.split()) * 1.3)

    print(f"📊 Input tokens: {input_tokens:,}")

    # Generate
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=4096,
        stop=["</s>", "[/INST]"],
    )

    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()

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
    """
    Evaluate generated summary against reference using multiple metrics.

    Args:
        generated: Generated summary text
        reference: Reference summary text

    Returns:
        dict with BLEU, ROUGE-L, SBERT coherence, and BERTScore
    """
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer, util
    from bert_score import score
    import nltk

    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    print("🔹 Computing evaluation metrics...")

    # BLEU Score
    bleu = sentence_bleu([reference.split()], generated.split())

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = scorer.score(reference, generated)["rougeL"].fmeasure

    # SBERT Coherence
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    ref_emb = sbert_model.encode(reference, convert_to_tensor=True)
    gen_emb = sbert_model.encode(generated, convert_to_tensor=True)
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

    print(f"✅ BLEU: {bleu:.4f}")
    print(f"✅ ROUGE-L: {rouge_l:.4f}")
    print(f"✅ SBERT: {sbert_coherence:.4f}")
    print(f"✅ BERTScore F1: {bert_f1:.4f}")

    return results


@app.local_entrypoint()
def main(
    transcript_path: str = "../rag_models/RAG_To_See_MedGemma_Performance/data/transcription_rakesh.txt",
    openemr_path: str = "../rag_models/RAG_To_See_MedGemma_Performance/data/openemr_rakesh.txt",
    reference_path: str = "../rag_models/RAG_To_See_MedGemma_Performance/data/reference_rakesh.txt",
    patient_name: str = "Rakesh",
    output_dir: str = "results",
):
    """
    Main function to run summarization and evaluation.

    Args:
        transcript_path: Path to transcript file
        openemr_path: Path to OpenEMR file
        reference_path: Path to reference summary
        patient_name: Patient name
        output_dir: Output directory for results
    """
    from pathlib import Path

    print("=" * 60)
    print(f"Medical Transcript Summarization - {patient_name}")
    print("=" * 60)

    # Read input files
    transcript_text = Path(transcript_path).read_text(encoding="utf-8")
    openemr_text = Path(openemr_path).read_text(encoding="utf-8") if Path(openemr_path).exists() else ""
    reference_text = Path(reference_path).read_text(encoding="utf-8") if Path(reference_path).exists() else ""

    # Generate summary
    print("\n🚀 Generating summary...")
    result = generate_summary.remote(
        transcript_text=transcript_text,
        openemr_text=openemr_text,
        patient_name=patient_name,
    )

    # Evaluate if reference exists
    eval_results = {}
    if reference_text:
        print("\n🔍 Evaluating summary...")
        eval_results = evaluate_summary.remote(
            generated=result["summary"],
            reference=reference_text,
        )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    result_file = output_path / f"evaluation_{patient_name}.txt"

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Medical Summary Evaluation - {patient_name}\n")
        f.write("=" * 60 + "\n\n")

        f.write("### GENERATION METRICS ###\n")
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
        f.write(result["summary"])

    print(f"\n✅ Results saved to: {result_file}")
    print("\n" + "=" * 60)
    print("Summary Preview:")
    print("=" * 60)
    print(result["summary"][:500] + "..." if len(result["summary"]) > 500 else result["summary"])

    if eval_results:
        print("\n" + "=" * 60)
        print("Evaluation Metrics:")
        print("=" * 60)
        for metric, value in eval_results.items():
            print(f"{metric.upper()}: {value:.4f}")