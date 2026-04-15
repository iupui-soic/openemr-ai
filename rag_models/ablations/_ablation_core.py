"""
Shared utilities for RAG ablation experiments on Fareez OSCE n=40.

All ablation drivers use these helpers so behaviour is consistent across the
no-RAG, k-sweep, embedding-substitution, temperature-sweep, and prompt-variant
experiments.

Design choices:
- Retrieval lives in `AblationRAG`, configurable via:
    * `embedding_model_name` (default all-MiniLM-L6-v2; matches production)
    * `k` (default 2)
    * `enabled` (False -> empty schema context; the no-RAG baseline)
- Prompt templates live in `PROMPT_TEMPLATES`; selectable by string key.
- `generate_summary_groq()` wraps the Groq client. Temperature is a parameter,
  so the temp-sweep ablation just iterates over values.

Each driver keeps the AblationRAG warm and iterates the 40 Fareez cases under
one configuration, writing summaries to a per-config subdirectory matching the
existing `fareez_summaries/<model>/<patient>.txt` layout.
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make the existing fareez loader importable
RAG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_ROOT / "pipeline"))

# CHROMA_PATH points to the existing on-disk index. We never query it via Chroma
# similarity (BioBERT is unused legacy); we use it only to pull metadata + docs.
CHROMA_PATH = str(RAG_ROOT / "vectorDB" / "chroma_schema_improved")

# ---------------------------------------------------------------------------
# Prompt templates
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
    # Production prompt — kept verbatim from rag_models/pipeline/run_fareez_summaries.py
    # so the `current` ablation cell is comparable to existing main results.
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

    # Minimal prompt — strip almost all guardrails to test how much the
    # production prompt is doing.
    "minimal": _BASE_BODY + """
Write a SOAP-format clinical summary in narrative prose. Begin with "Patient Information":""",

    # Hallucination-guarded prompt — explicit anti-EHR-overreach + anti-fabrication
    # instructions. Targets the 32% EHR-contamination error class identified in
    # the Fareez clinician-evaluation error analysis (critique #3).
    "hallucination_guarded": _BASE_BODY + """
OUTPUT FORMAT REQUIREMENTS:
- Generate a NARRATIVE TEXT document, NOT JSON or structured data
- Use clear section headers (e.g., "Patient Information", "Chief Complaint", "History of Present Illness")
- Write in complete sentences and paragraphs
- Use professional medical documentation prose style
- Do NOT use markdown formatting

GROUNDING REQUIREMENTS (these override everything else):
- Include ONLY information that is stated in the TRANSCRIPT or in the OPENEMR EXTRACT.
- If the OPENEMR EXTRACT contains demographics, comorbidities, medications, allergies,
  or labs that are NOT discussed in the TRANSCRIPT, do NOT include them in the note —
  they are background chart context the clinician did not raise during the encounter.
- Do NOT invent physical-examination findings, vital signs, lab results, or
  diagnostic-imaging interpretations. If any of these are not stated in the inputs,
  write "Not assessed during this encounter."
- Do NOT propose specific medication doses, frequencies, or new prescriptions
  unless the clinician explicitly stated them in the TRANSCRIPT.
- If TRANSCRIPT and OPENEMR conflict, trust the TRANSCRIPT for current status.
- If a SOAP section has no supporting content in the inputs, write
  "No information available." Do not fill it with inferred or template content.

Generate the medical summary now in narrative prose format, beginning with "Patient Information":""",
}


# ---------------------------------------------------------------------------
# RAG retriever (configurable: embedding model, k, on/off)
# ---------------------------------------------------------------------------

@dataclass
class AblationRAG:
    """
    Disease-string retrieval over the existing Chroma metadata, configurable for
    embedding model and top-k. Set `enabled=False` for the no-RAG baseline —
    `retrieve()` then returns an empty string and the model sees only the
    transcript + OpenEMR extract.
    """
    embedding_model_name: str = "all-MiniLM-L6-v2"
    k: int = 2
    enabled: bool = True

    # Initialised lazily in `load()`
    sbert_model: object = field(default=None, init=False, repr=False)
    candidate_embs: object = field(default=None, init=False, repr=False)
    all_docs: list = field(default=None, init=False, repr=False)
    metadata_diseases: list = field(default=None, init=False, repr=False)

    def load(self):
        """Heavy initialisation: load Chroma metadata + the SBERT encoder."""
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from sentence_transformers import SentenceTransformer

        # We still hand BioBERT to Chroma because that's what the on-disk index
        # was built with — Chroma needs the embedder to load. We never call
        # Chroma .similarity_search; we just .get() the metadata.
        embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
        )
        collection_data = vector_store.get(include=["metadatas", "documents"])
        self.all_docs = collection_data["documents"]
        self.metadata_diseases = [
            m.get("diseases", "Unspecified") for m in collection_data["metadatas"]
        ]

        if self.enabled:
            self.sbert_model = SentenceTransformer(self.embedding_model_name)
            self.candidate_embs = self.sbert_model.encode(
                self.metadata_diseases, convert_to_tensor=True
            )
            print(
                f"  RAG ON  | embedder={self.embedding_model_name} | k={self.k} | "
                f"corpus={len(self.metadata_diseases)} schemas"
            )
        else:
            print("  RAG OFF (no-RAG baseline)")

    def retrieve(self, detected_disease: str) -> str:
        if not self.enabled:
            return ""
        from sentence_transformers import util

        target_emb = self.sbert_model.encode(detected_disease, convert_to_tensor=True)
        cosine_scores = util.cos_sim(target_emb, self.candidate_embs)[0]
        k = min(self.k, len(cosine_scores))
        top_indices = cosine_scores.topk(k).indices.tolist()

        out = ""
        for rank, idx in enumerate(top_indices):
            out += f"\n\n=== SCHEMA {rank+1} ({self.metadata_diseases[idx]}) ===\n{self.all_docs[idx]}"
        return out


# ---------------------------------------------------------------------------
# Groq summarisation
# ---------------------------------------------------------------------------

def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def generate_summary_groq(
    client,
    *,
    model_name: str,
    api_type: str,                       # "reasoning" or "standard"
    prompt_variant: str,
    transcript: str,
    openemr: str,
    schema_context: str,
    temperature: float,
):
    """
    One Groq call. Returns (text, prompt_text).

    For reasoning models (GPT-OSS) we follow the existing pipeline's API shape:
    user-only message, max_completion_tokens, reasoning_effort. We DO honour the
    requested `temperature` — the existing production code hardcoded 1.0 here,
    so the temp ablation is a meaningful behavioural change.
    """
    template = PROMPT_TEMPLATES[prompt_variant]
    body = template.format(
        transcript=transcript,
        openemr=openemr or "No OpenEMR data available.",
        schema=schema_context if schema_context else "No schema retrieved.",
    )
    is_qwen = "qwen" in model_name.lower()
    system = (
        "You are an expert medical scribe specialized in clinical documentation. "
        "Generate comprehensive SOAP-format medical summaries."
    )

    if api_type == "reasoning":
        # GPT-OSS reasoning API: user-only message
        messages = [{"role": "user", "content": system + "\n\n" + body}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=4096,
            top_p=1,
            reasoning_effort="medium",
        )
        text = (response.choices[0].message.content or "").strip()
        prompt_text = messages[0]["content"]
    else:
        suffix = " /no_think" if is_qwen else ""
        messages = [
            {"role": "system", "content": system + suffix},
            {"role": "user", "content": body},
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
        )
        text = _strip_think_tags(response.choices[0].message.content or "").strip()
        prompt_text = messages[0]["content"] + messages[1]["content"]

    return text, prompt_text


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def make_token_counter():
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4")
        return lambda s: len(encoding.encode(s))
    except Exception:
        return lambda s: int(len(s.split()) * 1.3)


# ---------------------------------------------------------------------------
# Output writing — keeps the same on-disk layout as the existing Fareez results
# ---------------------------------------------------------------------------

def write_summary_file(out_dir: Path, patient_name: str, model: str,
                       detected_disease: str, summary: str, total_time: float,
                       input_tokens: int, output_tokens: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{patient_name}.txt"
    with open(fp, "w", encoding="utf-8") as f:
        f.write(f"Model: {model}\n")
        f.write(f"Patient: {patient_name}\n")
        f.write(f"Detected Disease: {detected_disease}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Time: {total_time:.2f}s\n")
        f.write(f"Tokens: {input_tokens:,} in / {output_tokens:,} out\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary if summary else "[No summary generated]")


def write_eval_csv(out_dir: Path, model_short: str, rows: list):
    """Write the per-case CSV in the same shape as `evaluation_results_*.csv`."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if len(df):
        avg = {c: (df[c].mean() if df[c].dtype.kind in "fi" else "") for c in df.columns}
        avg["patient_name"] = "AVERAGE"
        avg["model"] = model_short
        df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"evaluation_results_{model_short}.csv"
    df.to_csv(fp, index=False)
    return fp
