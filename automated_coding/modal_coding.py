"""Modal app for automated CPT medical coding using Gemma 4 26B-A4B-it."""
from __future__ import annotations
import json
import re
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "huggingface_hub>=0.22.0",
        "fastapi",
        "pydantic",
        "sentencepiece",
        "protobuf",
    )
)

app = modal.App("gemma4-cpt-coder", image=image)

SYSTEM = (
    "You are a certified professional medical coder. Given a clinical note, "
    "identify which CPT procedure codes from the provided candidate list apply. "
    'Respond with JSON only: {"codes": ["<code>", ...]}. '
    "Output an empty list if none apply. Do not invent codes."
)

USER_TEMPLATE = (
    "Candidate CPT codes (code: description):\n"
    "{code_block}\n\n"
    "Clinical note:\n"
    "<<<\n{text}\n>>>\n\n"
    "Return only the subset of candidate codes whose procedures are documented "
    "in the note. Output JSON only, no prose."
)

DEFAULT_DESCRIPTIONS = {
    "99201": "Office or other outpatient visit, new patient, straightforward",
    "99202": "Office or other outpatient visit, new patient, low complexity",
    "99203": "Office or other outpatient visit, new patient, moderate complexity",
    "99204": "Office or other outpatient visit, new patient, moderate-high complexity",
    "99205": "Office or other outpatient visit, new patient, high complexity",
    "99211": "Office or other outpatient visit, established patient, minimal",
    "99212": "Office or other outpatient visit, established patient, straightforward",
    "99213": "Office or other outpatient visit, established patient, low complexity",
    "99214": "Office or other outpatient visit, established patient, moderate complexity",
    "99215": "Office or other outpatient visit, established patient, high complexity",
    "99221": "Initial hospital care, low complexity",
    "99222": "Initial hospital care, moderate complexity",
    "99223": "Initial hospital care, high complexity",
    "99231": "Subsequent hospital care, straightforward",
    "99232": "Subsequent hospital care, moderate complexity",
    "99233": "Subsequent hospital care, high complexity",
    "99238": "Hospital discharge day management, 30 minutes or less",
    "99239": "Hospital discharge day management, more than 30 minutes",
    "99281": "Emergency department visit, minimal severity",
    "99282": "Emergency department visit, low complexity",
    "99283": "Emergency department visit, moderate complexity",
    "99284": "Emergency department visit, high complexity",
    "99285": "Emergency department visit, high complexity with threat to life",
    "93000": "Electrocardiogram, routine ECG with interpretation and report",
    "93005": "Electrocardiogram, routine ECG, tracing only",
    "93010": "Electrocardiogram, routine ECG, interpretation and report only",
    "71046": "Radiologic examination, chest, 2 views",
    "71045": "Radiologic examination, chest, single view",
    "80053": "Comprehensive metabolic panel",
    "80048": "Basic metabolic panel",
    "85025": "Blood count; complete (CBC), automated",
    "85027": "Blood count; complete (CBC), automated, and automated differential",
    "84443": "Thyroid stimulating hormone (TSH)",
    "82565": "Creatinine; blood",
    "82947": "Glucose; quantitative, blood",
    "83036": "Hemoglobin A1C",
    "84100": "Phosphorus inorganic (phosphate)",
    "84132": "Potassium; serum, plasma or whole blood",
    "84295": "Sodium; serum, plasma or whole blood",
    "84520": "Urea nitrogen; quantitative",
    "85610": "Prothrombin time",
    "85730": "Thromboplastin time, partial (PTT)",
    "86900": "Blood typing, serologic; ABO",
    "87086": "Culture, bacterial; quantitative colony count, urine",
    "87880": "Streptococcus, group A; direct optical observation",
    "36415": "Collection of venous blood by venipuncture",
    "36416": "Collection of capillary blood specimen",
    "90460": "Immunization administration through 18 years via injection",
    "90461": "Immunization administration through 18 years, each additional vaccine",
    "90471": "Immunization administration; one vaccine (includes percutaneous)",
    "90472": "Immunization administration; each additional vaccine",
    "96372": "Therapeutic, prophylactic, or diagnostic injection; subcutaneous or intramuscular",
    "96374": "Therapeutic, prophylactic, or diagnostic injection; intravenous push",
    "97110": "Therapeutic procedure, therapeutic exercises",
    "97530": "Therapeutic activities, direct patient contact",
    "97803": "Medical nutrition therapy; reassessment and intervention",
    "99406": "Smoking and tobacco use cessation counseling visit; 3-10 minutes",
    "99407": "Smoking and tobacco use cessation counseling visit; greater than 10 minutes",
    "99495": "Transitional care management, moderate complexity, 14-day contact",
    "99496": "Transitional care management, high complexity, 7-day contact",
    "99487": "Complex chronic care management, 60 minutes",
    "99489": "Complex chronic care management, each additional 30 minutes",
    "99490": "Chronic care management, 20 minutes",
}

_JSON_SPAN_RE = re.compile(r"\{[^{}]*\"codes\"[^{}]*\}", re.DOTALL)
MODEL_ID = "google/gemma-4-26B-A4B-it"
HF_MAX_INPUT_TOKENS = 14000
MAX_NEW_TOKENS = 512


def _build_code_block(descriptions):
    return "\n".join(f"- {code}: {desc}" for code, desc in descriptions.items())


def _parse_codes(raw, label_space):
    label_set = set(label_space)
    candidates = []
    match = _JSON_SPAN_RE.search(raw)
    if match:
        try:
            obj = json.loads(match.group(0))
            codes = obj.get("codes", [])
            if isinstance(codes, list):
                candidates = [str(c).strip() for c in codes]
        except json.JSONDecodeError:
            pass
    if not candidates:
        try:
            obj = json.loads(raw)
            codes = obj.get("codes", []) if isinstance(obj, dict) else []
            if isinstance(codes, list):
                candidates = [str(c).strip() for c in codes]
        except json.JSONDecodeError:
            pass
    return [c for c in candidates if c in label_set]


@app.cls(
    gpu="A100-80GB",
    timeout=300,
    scaledown_window=120,
)
class Gemma4Coder:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {MODEL_ID} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        print("Model loaded.")

    def _generate(self, text, descriptions):
        import torch
        user = USER_TEMPLATE.format(
            code_block=_build_code_block(descriptions), text=text
        )
        chat = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"{SYSTEM}\n\n{user}\n\nAssistant:"
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=HF_MAX_INPUT_TOKENS,
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    @modal.fastapi_endpoint(method="POST")
    def code(self, request: dict) -> dict:
        text = request.get("text", "").strip()
        if not text:
            return {"error": "text is required", "codes": [], "raw": ""}
        descriptions = request.get("descriptions") or DEFAULT_DESCRIPTIONS
        label_space = list(descriptions.keys())
        raw = self._generate(text, descriptions)
        codes = _parse_codes(raw, label_space)
        return {"codes": codes, "raw": raw}

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        return {"status": "ok", "model": MODEL_ID}


@app.local_entrypoint()
def main():
    sample_note = """
    Patient is a 65-year-old male with type 2 diabetes presenting for follow-up.
    HbA1c checked today. Blood pressure 138/88. EKG performed showing normal sinus rhythm.
    CBC and comprehensive metabolic panel ordered. Venipuncture performed for lab collection.
    Patient counseled on smoking cessation for 5 minutes.
    """
    coder = Gemma4Coder()
    result = coder.code.remote({"text": sample_note})
    print("Predicted CPT codes:", result["codes"])
    print("Raw model output   :", result["raw"])
