"""
Automated CPT/ICD-10 coding pipeline.

Usage:
    modal run automated_coding_modal.py --note-file path/to/note.txt
"""

import modal
import os
import json
import re
from typing import Dict, List, Any

app = modal.App("automated-cpt-icd-coding")

codes_volume = modal.Volume.from_name("medical-codes-vectordb")
CHROMA_PATH = "/vectordb/chroma_codes"

MODEL_NAME = "google/gemma-4-26B-A4B-it"
TOP_K = 5

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "accelerate>=0.29.0",
        "huggingface_hub>=0.22.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-huggingface>=0.0.1",
        "langchain-chroma>=0.1.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.22",
    )
)

EXTRACTION_PROMPT_TEMPLATE = """You are a certified professional medical coder. Read the following clinical
note and extract the diseases, diagnoses, and procedures mentioned -- as
plain medical terms, not codes.

CLINICAL NOTE:
{note_text}

INSTRUCTIONS:
1. List each distinct disease/diagnosis and each distinct procedure
   mentioned or clearly implied by the note.
2. Use standard medical terminology (e.g., "essential hypertension", not
   "high blood pressure"), since these terms will be used for a similarity
   search against a code database.
3. Do NOT invent conditions or procedures not supported by the note.
4. Output JSON only, no prose:
   {{"diagnoses": ["<term>", ...], "procedures": ["<term>", ...]}}

Return the JSON now:"""

MATCHING_PROMPT_TEMPLATE = """You are a certified professional medical coder. Given a clinical summary,
the clinical terms extracted from it, and a list of candidate codes
retrieved for those terms, select the codes that actually apply.

CLINICAL SUMMARY (for context/disambiguation only -- do not extract new
terms from this that aren't already listed below):
{note_text}

EXTRACTED TERMS:
{extracted_terms}

CANDIDATE CODES (code: description):
{candidate_codes}

INSTRUCTIONS:
1. For each extracted term, select the single best-matching code from the
   candidates, if one genuinely applies.
2. Use the clinical summary to disambiguate between close candidates (e.g.
   laterality, with/without contrast, severity) -- the summary is context,
   not a source of new terms to code.
3. Do NOT invent codes not in the candidate list.
4. Do NOT force a match if none of the candidates are a good fit for a term.
5. Output JSON only, no prose:
   {{"matches": [{{"entity": "<term>", "code": "<code>", "code_type": "CPT4|ICD10"}}, ...]}}
6. Output an empty list if nothing matches well.

Return the JSON now:"""


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


@app.cls(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/vectordb": codes_volume},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
class Gemma4Coder:

    @modal.enter()
    def load_models(self):
        import torch
        from transformers import pipeline
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        print(f"Loading {MODEL_NAME}...")
        self.pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )
        self.vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings,
        )

        print("Coding pipeline loaded.")

    def _extract_entities(self, note_text: str) -> Dict[str, List[str]]:
        prompt_content = EXTRACTION_PROMPT_TEMPLATE.format(note_text=note_text)
        messages = [
            {"role": "system", "content": "You are a certified professional medical coder."},
            {"role": "user", "content": prompt_content},
        ]
        output = self.pipe(messages, max_new_tokens=256, temperature=0.0)
        raw = output[0]["generated_text"][-1]["content"]
        obj = _extract_json(raw)
        return {
            "diagnoses": [str(x).strip() for x in obj.get("diagnoses", []) if str(x).strip()],
            "procedures": [str(x).strip() for x in obj.get("procedures", []) if str(x).strip()],
        }

    def _retrieve_candidates(self, entities: Dict[str, List[str]], k: int = TOP_K) -> Dict[str, str]:
        all_terms = entities.get("diagnoses", []) + entities.get("procedures", [])
        if not all_terms:
            return {}

        candidates: Dict[str, str] = {}

        collection_data = self.vector_store.get(include=["metadatas", "documents"])
        all_metadatas = collection_data["metadatas"]
        for term in all_terms:
            term_lower = term.lower()
            for meta in all_metadatas:
                code = meta.get("code")
                description = meta.get("description", "")
                if code and code not in candidates and term_lower in description.lower():
                    candidates[code] = description

        for term in all_terms:
            results = self.vector_store.similarity_search(term, k=k)
            for doc in results:
                code = doc.metadata.get("code")
                if code and code not in candidates:
                    candidates[code] = doc.metadata.get("description", doc.page_content)

        return candidates

    def _match_codes(
        self, note_text: str, entities: Dict[str, List[str]], candidates: Dict[str, str]
    ) -> List[Dict[str, str]]:
        if not candidates:
            return []

        all_terms = entities.get("diagnoses", []) + entities.get("procedures", [])
        terms_block = "\n".join(f"- {t}" for t in all_terms)
        candidates_block = "\n".join(f"- {c}: {d}" for c, d in candidates.items())

        prompt_content = MATCHING_PROMPT_TEMPLATE.format(
            note_text=note_text,
            extracted_terms=terms_block,
            candidate_codes=candidates_block,
        )
        messages = [
            {"role": "system", "content": "You are a certified professional medical coder."},
            {"role": "user", "content": prompt_content},
        ]
        output = self.pipe(messages, max_new_tokens=512, temperature=0.0)
        raw = output[0]["generated_text"][-1]["content"]
        obj = _extract_json(raw)

        matches = []
        for m in obj.get("matches", []):
            code = str(m.get("code", "")).strip()
            if code in candidates:
                matches.append({
                    "entity": str(m.get("entity", "")).strip(),
                    "code": code,
                    "code_type": str(m.get("code_type", "")).strip(),
                    "description": candidates[code],
                })
        return matches

    @modal.method()
    def generate_codes(self, note_text: str) -> Dict[str, Any]:
        import time

        start_total = time.time()

        t0 = time.time()
        entities = self._extract_entities(note_text)
        extraction_time = time.time() - t0

        t0 = time.time()
        candidates = self._retrieve_candidates(entities)
        retrieval_time = time.time() - t0

        t0 = time.time()
        matches = self._match_codes(note_text, entities, candidates)
        matching_time = time.time() - t0

        return {
            "entities": entities,
            "candidates_considered": len(candidates),
            "matches": matches,
            "extraction_time": extraction_time,
            "retrieval_time": retrieval_time,
            "matching_time": matching_time,
            "total_time": time.time() - start_total,
            "model": MODEL_NAME,
        }


@app.local_entrypoint()
def main(note_file: str = ""):
    if not note_file:
        print("Usage: modal run automated_coding_modal.py --note-file path/to/note.txt")
        return

    with open(note_file) as f:
        note_text = f.read()

    coder = Gemma4Coder()
    result = coder.generate_codes.remote(note_text=note_text)
    print(json.dumps(result, indent=2))
