"""
Modal-based SOAP summarization using GPT-OSS-20B with RAG
Accepts Groq API key from request (no environment secrets)
Uses disease-specific schemas from ChromaDB to guide extraction
"""

import modal
import json
import time
from typing import Dict, Any

from pydantic import BaseModel

# ============================================================================
# Request Model
# ============================================================================

class SummarizeRequest(BaseModel):
    text: str
    openemr_text: str = ""
    groq_api_key: str

# ============================================================================
# Modal App Configuration
# ============================================================================

app = modal.App("soap-summarization-gpt-oss-20b")

# Persistent volume for vector database
vectordb_volume = modal.Volume.from_name("medical-vectordb")

# Model configuration
MODEL_NAME = "openai/gpt-oss-20b"
CHROMA_PATH = "/vectordb/chroma_schema_improved"

# ============================================================================
# Modal Image
# ============================================================================

summarizer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]>=0.104.0",
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
# SOAP Summarizer Class
# ============================================================================

@app.cls(
    image=summarizer_image,
    timeout=3600,
    # Vector database volume is optional - commented out for testing
    # volumes={"/vectordb": vectordb_volume},
)
class SOAPSummarizer:
    """
    RAG-based medical summarizer using Groq API with GPT-OSS-20B.
    Models are loaded once in @modal.enter() and reused across all calls.
    Groq API key is passed per-request from user settings.
    """

    @modal.enter()
    def load_models(self):
        """Load all models once when container starts."""
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from sentence_transformers import SentenceTransformer
        import tiktoken

        print("🔄 Loading models (one-time initialization)...")
        print(f"   Model: {MODEL_NAME}")

        # Note: Groq client will be initialized per-request with user's API key

        # Try to load vector database (optional for testing)
        self.vector_store = None
        self.all_metadatas = []
        self.all_docs = []
        self.metadata_diseases = []
        self.candidate_embs = None
        self.sbert_model = None  # Initialize to None

        try:
            print("  → Attempting to load BioBERT embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
            )

            print(f"  → Attempting to load Vector Store from: {CHROMA_PATH}")
            self.vector_store = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=self.embeddings
            )

            # Pre-fetch collection data
            print("  → Pre-fetching collection data...")
            collection_data = self.vector_store.get(include=["metadatas", "documents"])
            self.all_metadatas = collection_data["metadatas"]
            self.all_docs = collection_data["documents"]
            self.metadata_diseases = [m.get("diseases", "Unspecified") for m in self.all_metadatas]

            # Check if database is empty
            if not self.metadata_diseases or len(self.metadata_diseases) == 0:
                print("⚠️  Vector database is empty - no documents found")
                print("   Continuing WITHOUT RAG - will use direct summarization")
                self.vector_store = None
                self.candidate_embs = None
                self.sbert_model = None
            else:
                # Load SBERT for semantic matching
                print("  → Loading SBERT model...")
                self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

                # Pre-encode all disease metadata
                print("  → Pre-encoding disease embeddings...")
                self.candidate_embs = self.sbert_model.encode(self.metadata_diseases, convert_to_tensor=True)

                print(f"✅ Vector database loaded successfully with {len(self.metadata_diseases)} documents!")

        except Exception as e:
            print(f"⚠️  Vector database not available: {e}")
            print("   Continuing WITHOUT RAG - will use direct summarization")
            self.vector_store = None
            self.candidate_embs = None
            self.sbert_model = None

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
            openemr_text: str,
            groq_api_key: str,
    ) -> Dict[str, Any]:
        """
        Generate SOAP-format medical summary from transcript using RAG + Groq.

        Args:
            transcript_text: Doctor-patient conversation transcript
            openemr_text: OpenEMR FHIR data (formatted text)
            groq_api_key: Groq API key from OpenEMR user settings

        Returns:
            dict with soap_note, detected_disease, timing metrics, token counts
        """
        from groq import Groq
        from sentence_transformers import util

        print(f"\n{'='*60}")
        print(f"🔹 Generating SOAP summary")
        print(f"🔹 Model: {MODEL_NAME}")
        print(f"{'='*60}")
        start_total = time.time()

        # Initialize Groq client with user's API key
        client = Groq(api_key=groq_api_key)

        # ==============================
        # 1. EXTRACT DISEASE USING GROQ
        # ==============================
        print("🔹 Extracting disease from transcript...")
        start_retrieval = time.time()

        disease_prompt = [
            {
                "role": "system",
                "content": "You are a medical expert. Identify the primary disease from clinical conversations. Respond with ONLY the disease name."
            },
            {
                "role": "user",
                "content": f"""Read this transcript and identify the PRIMARY medical condition.

Return ONLY the disease name (e.g., "COPD", "Diabetes", "Hypertension").
If no specific disease is mentioned, return "General".

Transcript:
{transcript_text[:2000]}

Primary Disease:"""
            }
        ]

        disease_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=disease_prompt,
            temperature=0.3,
            max_tokens=20,
        )

        raw_disease = disease_response.choices[0].message.content
        detected_disease = raw_disease.strip() if raw_disease else "General"
        print(f"✅ Detected Disease: {detected_disease}")

        # ==============================
        # 2. RETRIEVE SCHEMAS FROM VECTOR DB (Optional - skip if not available)
        # ==============================
        schema_context = ""

        if self.vector_store and self.candidate_embs is not None and self.sbert_model is not None:
            print("🔹 Retrieving relevant schemas from vector DB...")

            target_emb = self.sbert_model.encode(detected_disease, convert_to_tensor=True)
            cosine_scores = util.cos_sim(target_emb, self.candidate_embs)[0]

            # Get top 2 schemas
            k = min(2, len(cosine_scores))
            top_k_result = cosine_scores.topk(k)
            top_indices = top_k_result.indices.tolist()

            for rank, idx in enumerate(top_indices):
                doc_content = self.all_docs[idx]
                disease_meta = self.metadata_diseases[idx]
                schema_context += f"\n\n=== SCHEMA {rank+1} ({disease_meta}) ===\n{doc_content}"
        else:
            print("⚠️  Vector database not available - skipping RAG retrieval")
            schema_context = "\n\n(No disease-specific schemas available - generating summary without RAG guidance)"

        retrieval_time = time.time() - start_retrieval
        print(f"⏱️ Disease extraction + retrieval: {retrieval_time:.2f}s")

        # ==============================
        # 3. GENERATE SUMMARY WITH GROQ
        # ==============================
        print("🔹 Generating SOAP summary with Groq (GPT-OSS-20B)...")
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
- Use clear section headers (e.g., "Subjective", "Objective", "Assessment", "Plan")
- Write in complete sentences and paragraphs
- Use professional medical documentation prose style
- Do NOT use markdown formatting (no #, ##, **, *, -, ```, etc.)

INSTRUCTIONS:
1. Use the SCHEMA GUIDE as a reference for which sections to include
2. Extract relevant information from the TRANSCRIPT and OPENEMR EXTRACT
3. Write in narrative prose with proper paragraphs
4. If information for a section is missing, write "No information available."
5. If TRANSCRIPT and OPENEMR conflict, trust the TRANSCRIPT for current status
6. Do NOT output JSON, XML, or any structured data format
7. Do NOT hallucinate or invent information not present in the inputs
8. Do NOT use any markdown syntax - plain text only

Generate the medical summary now in narrative prose format:"""
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
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=summary_messages,
                temperature=0.3,
                max_tokens=4098,
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

        print(f"📊 Tokens: {input_tokens:,} in / {output_tokens:,} out")
        print(f"⏱️ Generation: {generation_time:.2f}s | Total: {total_time:.2f}s")
        print(f"📝 Output preview: {generated_text[:200]}...")

        return {
            "success": True,
            "soap_note": generated_text,
            "detected_disease": detected_disease,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": MODEL_NAME,
        }

    @modal.web_endpoint(method="POST")
    def summarize(self, request: SummarizeRequest) -> Dict[str, Any]:
        """
        HTTP endpoint for SOAP note generation.
        Called by the SMART app's frontend when user clicks "Summarize"
        """
        return self.generate_summary(
            transcript_text=request.text,
            openemr_text=request.openemr_text,
            groq_api_key=request.groq_api_key,
        )