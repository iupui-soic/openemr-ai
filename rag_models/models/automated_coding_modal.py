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
        "bitsandbytes>=0.43.0",
    )
)

EXTRACTION_PROMPT_TEMPLATE = """You are a certified professional medical coder. Read the following clinical
note and extract the diseases, diagnoses, and procedures mentioned -- as
plain medical terms, not codes.

CLINICAL NOTE:
{note_text}

CRITICAL RULES (FOLLOW EXACTLY):
- DO NOT HALLUCINATE ANY FACT. Only extract diagnoses or procedures that
  explicitly appear in the note text above.
- DO NOT infer, guess, or add anything based on what commonly co-occurs
  with what's mentioned.
- If in doubt about whether something is stated, LEAVE IT OUT.

INSTRUCTIONS:
1. Extract EVERY distinct billable clinical action or fact explicitly
   stated in the note, checking EACH of these categories in turn -- do not
   skip a category just because it feels less obvious than diagnoses:
   - Diagnoses/conditions actually being addressed at this visit
   - Tests, labs, imaging, or procedures performed
   - Medications started, changed, refilled, or continued -- extract the
     drug class or therapy type as its own procedure-like term (whatever
     class is actually named in the note), separate from the diagnosis
     it treats
   - Vaccinations or injections given
   - The visit/encounter type itself, if nothing else procedural happened
     (e.g. "established patient office visit", "follow-up visit") --
     include this as a procedure-like term so a plain visit still gets
     an E/M code
   List EVERY distinct disease/diagnosis and EVERY distinct procedure that
   is explicitly stated in the note -- not just the primary complaint.
   A note that mentions two separate conditions (e.g. a sore throat AND
   high blood pressure) must produce items for BOTH, not just the first
   one discussed.
2. Convert vital-sign statements into a diagnosis when they describe an
   abnormal reading. If the note states an elevated blood pressure
   reading (e.g. "blood pressure of 148 over 86", "high blood pressure"),
   include "essential hypertension" as a diagnosis, even if the clinician
   never used the word "hypertension" directly.
3. Before including any item, check: can you point to the exact word,
   phrase, or vital sign in the note that supports it? If not, do NOT
   include it.
4. Do NOT include related conditions, common comorbidities, or anything
   "usually associated with" what's mentioned, unless it is itself
   explicitly stated or directly derivable from a stated vital sign (as
   in instruction 2). For example, if the note only mentions a headache,
   do not add diabetes, arthropathy, or unrelated injuries just because
   they sometimes co-occur with other conditions.
5. Do NOT include inpatient/hospital-care concepts (e.g. hospital
   admission, observation care, discharge, subsequent hospital care)
   unless the note explicitly describes an inpatient hospital stay. An
   office visit is not a hospital encounter.
6. Use standard medical terminology (e.g., "essential hypertension", not
   "high blood pressure"), since these terms will be used for a similarity
   search against a code database.
7. Return one item per distinct category identified in instruction 1 --
   do not artificially shorten the list by omitting a category, but also
   do not pad it with anything not explicitly stated.
8. If two or more items describe the SAME underlying clinical fact using
   different phrasing (e.g. "lisinopril refill" and "ACE inhibitor
   therapy" both describing one ongoing medication), consolidate them
   into a SINGLE item using the most specific phrasing. Never emit two
   near-duplicate items for one fact -- this causes downstream code
   matching to arbitrarily pick between them and lose the correct code.
9. Output JSON only, no prose:
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
5. Do NOT select inpatient/hospital-care codes (e.g. observation care,
   hospital discharge, subsequent hospital care) unless the summary
   explicitly describes an inpatient hospital stay. An office visit is
   not a hospital encounter, even if it involves prescribing medication.
6. Do NOT select "adverse effect", "adverse reaction", or drug-toxicity
   codes for a medication unless the summary explicitly describes a
   negative reaction, side effect, or complication from that medication.
   Starting or prescribing a drug is NOT itself an adverse effect --
   e.g. "started on lisinopril for hypertension" describes a normal
   prescription, not an adverse effect of an ACE inhibitor.
7. Some candidate codes describe the SAME underlying service or encounter
   at different complexity/severity levels (e.g. an office-visit E/M code
   family like 99201-99205, or a hospital-care family). When you see
   several such codes for the same entity, select only the SINGLE
   best-fitting level based on the summary -- never select multiple
   levels of the same code family for one entity.
8. Output JSON only, no prose:
   {{"matches": [{{"entity": "<term>", "code": "<code>", "code_type": "CPT4|ICD10"}}, ...]}}
9. Output an empty list if nothing matches well.

Return the JSON now:"""

VALIDATION_PROMPT_TEMPLATE = """You are a certified professional medical coder performing claims validation.
You are given a clinical note and ONE specific code that has already been
entered (by a human coder or an automated system). Your job is NOT to find
codes -- it is to judge whether THIS SPECIFIC code is actually supported by
the note.

CLINICAL NOTE:
{note_text}

CODE TO VALIDATE:
{code}: {description}

CRITICAL RULES (FOLLOW EXACTLY):
- Judge ONLY this one code. Do not comment on what other codes might apply.
- The code is supported ONLY if the note contains explicit textual evidence
  for it -- an exact or clearly equivalent statement of the diagnosis,
  procedure, medication action, or encounter type this code represents.
- Mentioning a condition in passing (e.g. as unrelated past medical history
  that isn't being addressed today) does NOT support a code for that
  condition, unless the note explicitly shows it was addressed at this
  visit.
- A code for a DIFFERENT but related concept (e.g. a different body part,
  a different severity, an unrelated complication, an inpatient/hospital
  code for what is clearly an office visit) is NOT supported, even if
  something superficially similar is mentioned.
- When supported, quote the exact phrase from the note that supports it.
- When NOT supported, leave evidence empty and explain briefly why in
  "reason" (e.g. "note describes an office visit, not an inpatient stay"
  or "this condition is listed as history only, not addressed today").

Output JSON only, no prose:
{{"supported": true or false, "evidence": "<exact quoted phrase, or empty string>", "reason": "<brief explanation>"}}

Return the JSON now:"""

def _extract_json(raw: str) -> dict:
    """Best-effort JSON extraction: try a brace-matched span first, then the whole string."""
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
    gpu="A100-80GB",
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

    @modal.method()
    def validate_code(
        self, note_text: str, code: str, description: str
    ) -> Dict[str, Any]:
        """
        Given a note and a SINGLE already-entered code (from a human coder
        or from our own generate_codes()), judge whether the note actually
        supports it. Narrower and more checkable than generation: a closed
        yes/no judgment with cited evidence, not an open-ended search.
        """
        prompt_content = VALIDATION_PROMPT_TEMPLATE.format(
            note_text=note_text, code=code, description=description
        )
        messages = [
            {"role": "system", "content": "You are a certified professional medical coder performing claims validation."},
            {"role": "user", "content": prompt_content},
        ]
        output = self.pipe(messages, max_new_tokens=256, do_sample=False)
        raw = output[0]["generated_text"][-1]["content"]
        print(f"[validate_code] raw LLM output: {raw}")
        obj = _extract_json(raw)
        result = {
            "code": code,
            "supported": bool(obj.get("supported", False)),
            "evidence": str(obj.get("evidence", "")).strip(),
            "reason": str(obj.get("reason", "")).strip(),
        }
        print(f"[validate_code] parsed: {result}")
        return result

    @modal.method()
    def wakeup(self):
        """Trivial method to force container/model load and reset the idle timer."""
        return {"status": "warm"}

    def _extract_entities(self, note_text: str) -> Dict[str, List[str]]:
        prompt_content = EXTRACTION_PROMPT_TEMPLATE.format(note_text=note_text)
        messages = [
            {"role": "system", "content": "You are a certified professional medical coder."},
            {"role": "user", "content": prompt_content},
        ]
        output = self.pipe(messages, max_new_tokens=256, do_sample=False)
        raw = output[0]["generated_text"][-1]["content"]
        print(f"[extract_entities] raw LLM output: {raw}")
        obj = _extract_json(raw)
        result = {
            "diagnoses": [str(x).strip() for x in obj.get("diagnoses", []) if str(x).strip()],
            "procedures": [str(x).strip() for x in obj.get("procedures", []) if str(x).strip()],
        }
        print(f"[extract_entities] parsed: {result}")
        return result

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

        print(f"[retrieve_candidates] terms searched: {all_terms}")
        print(f"[retrieve_candidates] found {len(candidates)} candidates: {candidates}")
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
        output = self.pipe(messages, max_new_tokens=512, do_sample=False)
        raw = output[0]["generated_text"][-1]["content"]
        print(f"[match_codes] raw LLM output: {raw}")
        obj = _extract_json(raw)
        print(f"[match_codes] parsed obj: {obj}")

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
    def benchmark_predict(
        self, note_text: str, candidates: Dict[str, str]
    ) -> List[str]:
        """
        Benchmark-only entrypoint: skips our own retrieval stage and uses the
        externally-provided candidate set instead, so this can be compared
        fairly against the other approaches in automated_coding/, which are
        all given the same fixed candidate list rather than doing their own
        retrieval.
        """
        entities = self._extract_entities(note_text)
        matches = self._match_codes(note_text, entities, candidates)
        return [m["code"] for m in matches if m.get("code") in candidates]

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
