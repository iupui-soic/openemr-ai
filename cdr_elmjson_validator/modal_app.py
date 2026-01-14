"""
ELM Validator using Multiple LLMs on Modal

Consolidated batch processing - each model loads once and processes all files.
"""

import modal
import json
import time

app = modal.App("elm-validator")

# Shared image with ML dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers", "torch", "accelerate", "sentencepiece"
)

# Groq image for API-based inference (no GPU/volume needed)
groq_image = modal.Image.debian_slim().pip_install("groq>=0.4.0")

# Cache volume for model weights
volume = modal.Volume.from_name("elm-model-cache", create_if_missing=True)


# ============================================
# Model Configuration Registry
# ============================================

MODEL_CONFIGS = {
    "llama-3.2-1b": {
        "hf_name": "meta-llama/Llama-3.2-1B-Instruct",
        "gpu": "T4",
        "timeout": 1800,
        "needs_hf_token": True,
        "provider": "huggingface",
    },
    "llama-3.2-3b": {
        "hf_name": "meta-llama/Llama-3.2-3B-Instruct",
        "gpu": "T4",
        "timeout": 1800,
        "needs_hf_token": True,
        "provider": "huggingface",
    },
    "qwen-2.5-1.5b": {
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "gpu": "T4",
        "timeout": 1800,
        "needs_hf_token": False,
        "provider": "huggingface",
    },
    "qwen-2.5-3b": {
        "hf_name": "Qwen/Qwen2.5-3B-Instruct",
        "gpu": "T4",
        "timeout": 1800,
        "needs_hf_token": False,
        "provider": "huggingface",
    },
    "phi-3-mini": {
        "hf_name": "microsoft/Phi-3-mini-4k-instruct",
        "gpu": "T4",
        "timeout": 1800,
        "needs_hf_token": False,
        "provider": "huggingface",
    },
    "gemma-3-4b": {
        "hf_name": "google/gemma-3-4b-it",
        "gpu": "T4",
        "timeout": 1800,
        "needs_hf_token": True,
        "provider": "huggingface",
    },
    "medgemma-4b": {
        "hf_name": "google/medgemma-4b-it",
        "gpu": "T4",
        "timeout": 1800,
        "needs_hf_token": True,
        "provider": "huggingface",
    },
    "medgemma-1.5-4b": {
        "hf_name": "google/medgemma-1.5-4b-it",
        "gpu": "T4",
        "timeout": 1800,
        "needs_hf_token": True,
        "provider": "huggingface",
    },
    "llama-3.1-8b": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu": "T4",
        "timeout": 3600,  # Longer timeout for larger model
        "needs_hf_token": True,
        "provider": "huggingface",
    },
    "gemma-3-270m": {
        "hf_name": "google/gemma-3-270m-it",
        "gpu": "T4",
        "timeout": 1800,
        "needs_hf_token": True,
        "provider": "huggingface",
    },
    "gpt-oss-120b": {
        "hf_name": "openai/gpt-oss-120b",
        "gpu": None,  # API-based, no GPU needed
        "timeout": 1800,
        "needs_hf_token": False,
        "provider": "groq",
    },
    "gpt-oss-20b": {
        "hf_name": "openai/gpt-oss-20b",
        "gpu": None,
        "timeout": 1800,
        "needs_hf_token": False,
        "provider": "groq",
    },
    "llama-3.3-70b": {
        "hf_name": "llama-3.3-70b-versatile",
        "gpu": None,
        "timeout": 1800,
        "needs_hf_token": False,
        "provider": "groq",
    },
}


def extract_embedded_errors(elm_json):
    """Extract errors from ELM annotations."""
    errors = []
    annotations = elm_json.get("library", {}).get("annotation", [])

    for ann in annotations:
        if ann.get("type") == "CqlToElmError" and ann.get("errorSeverity") == "error":
            msg = ann.get("message", "Unknown error")
            line = ann.get("startLine", "?")
            errors.append(f"{msg} (Line {line})")

    return errors


def build_prompt(elm_json, library_name, cpg_content=None):
    """Build validation prompt for LLM."""
    library = elm_json.get("library", {})

    elm_content = {"library": library}

    if cpg_content:
        prompt = f"""You are a Clinical Decision Support (CDS) validation expert. Your task is to validate whether the ELM (Expression Logical Model) implementation correctly implements the Clinical Practice Guideline (CPG).

Clinical Practice Guideline (user-written, any format):
{cpg_content}

ELM Implementation:
{json.dumps(elm_content, indent=2)}

Validation Process:
1. Understand the CPG: Read and understand what clinical criteria the CPG describes (age ranges, lab values, time periods, procedures, conditions, etc.) - regardless of how it's written
2. Find in ELM: Locate where those criteria are implemented in the ELM statements
3. Compare: Check if the ELM implementation matches what the CPG describes
4. Report: Note any discrepancies between the CPG's intent and the ELM's implementation

The CPG is the source of truth. If the ELM does something different from what the CPG describes, it's an error.

Response Format (respond EXACTLY in this format):
VALID: YES or NO
ERRORS: List specific issues where ELM doesn't match CPG, or "None"
WARNINGS: List potential issues or suggestions, or "None"

Your response:"""
    else:
        prompt = f"""You are a Clinical Quality Language (CQL) expert. Analyze this ELM clinical logic.

ELM Library (FULL):
{json.dumps(elm_content, indent=2)}

Check for LOGICAL ISSUES only:
- Contradictory conditions?
- Missing logic steps?
- Inappropriate recommendations?

DO NOT check syntax - already validated.

Respond EXACTLY:
VALID: YES or NO
ERRORS: List or "None"
WARNINGS: List or "None"

Your response:"""

    return prompt


def parse_response(response):
    """Parse LLM's structured response."""
    lines = [l.strip() for l in response.split('\n') if l.strip()]

    valid = True
    errors = []
    warnings = []
    section = None

    for line in lines:
        upper = line.upper()

        if upper.startswith('VALID:'):
            valid = 'YES' in upper
        elif upper.startswith('ERRORS:'):
            section = 'errors'
            content = line.split(':', 1)[1].strip()
            if content.lower() != 'none':
                errors.append(content)
        elif upper.startswith('WARNINGS:'):
            section = 'warnings'
            content = line.split(':', 1)[1].strip()
            if content.lower() != 'none':
                warnings.append(content)
        elif line.startswith('-') or line.startswith('*'):
            item = line[1:].strip()
            if item.lower() != 'none':
                if section == 'warnings':
                    warnings.append(item)
                elif section == 'errors':
                    errors.append(item)

    return {
        "valid": valid and len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def run_single_inference(elm_json, library_name, cpg_content, model, tokenizer, model_id, model_name):
    """Run inference for a single ELM file using pre-loaded model."""
    start_time = time.time()

    # Check for embedded CQL-to-ELM errors
    embedded_errors = extract_embedded_errors(elm_json)
    if embedded_errors:
        return {
            "valid": False,
            "errors": embedded_errors,
            "warnings": [],
            "source": "embedded",
            "model": model_id,
            "model_name": model_name,
            "library_name": library_name,
            "time_seconds": time.time() - start_time
        }

    # Build validation prompt
    prompt = build_prompt(elm_json, library_name, cpg_content)

    if cpg_content:
        print(f"  Validating {library_name} against CPG...")

    # Generate response
    inference_start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Gemma models have numerical stability issues with low-temperature sampling
    is_gemma = "gemma" in model_name.lower()

    if is_gemma:
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response_text[len(prompt):].strip()
    inference_time = time.time() - inference_start

    # Parse response
    result = parse_response(answer)
    total_time = time.time() - start_time

    result.update({
        "source": "llm",
        "model": model_id,
        "model_name": model_name,
        "library_name": library_name,
        "has_cpg": cpg_content is not None,
        "time_seconds": total_time,
        "inference_time_seconds": inference_time,
        "raw_response": answer[:500]
    })

    print(f"  {library_name}: valid={result['valid']} in {total_time:.2f}s")

    return result


def run_batch_validation(items: list, model_name: str, model_id: str) -> list:
    """
    Validate multiple ELM files with a single model load.

    Args:
        items: List of dicts with keys: elm_json, library_name, cpg_content, file_name
        model_name: HuggingFace model name
        model_id: Short model identifier

    Returns:
        List of validation results
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if not items:
        return []

    # Load model once
    print(f"Loading {model_name} for batch of {len(items)} files...")
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/cache")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/cache"
    )

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Process all items with the loaded model
    results = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] Processing {item.get('file_name', 'unknown')}...")

        result = run_single_inference(
            elm_json=item.get("elm_json"),
            library_name=item.get("library_name", "Unknown"),
            cpg_content=item.get("cpg_content"),
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            model_name=model_name
        )
        result["file_name"] = item.get("file_name")
        result["load_time_seconds"] = load_time if i == 1 else 0
        results.append(result)

    total_inference_time = sum(r.get("inference_time_seconds", 0) for r in results)
    print(f"\nBatch complete: {len(results)} files in {load_time + total_inference_time:.2f}s "
          f"(load: {load_time:.2f}s, inference: {total_inference_time:.2f}s)")

    return results


def run_batch_groq_validation(items: list, model_name: str, model_id: str) -> list:
    """Groq API-based batch validation - no model loading needed."""
    from groq import Groq

    if not items:
        return []

    print(f"Running batch validation with {model_name} via Groq API for {len(items)} files...")

    client = Groq()

    results = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] Processing {item.get('file_name', 'unknown')}...")

        start_time = time.time()

        elm_json = item.get("elm_json")
        library_name = item.get("library_name", "Unknown")
        cpg_content = item.get("cpg_content")

        embedded_errors = extract_embedded_errors(elm_json)
        if embedded_errors:
            result = {
                "valid": False,
                "errors": embedded_errors,
                "warnings": [],
                "source": "embedded",
                "model": model_id,
                "model_name": model_name,
                "library_name": library_name,
                "file_name": item.get("file_name"),
                "time_seconds": time.time() - start_time,
                "load_time_seconds": 0
            }
            results.append(result)
            continue

        prompt = build_prompt(elm_json, library_name, cpg_content)

        inference_start = time.time()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Groq API error: {e}")
            result = {
                "valid": False,
                "errors": [f"Groq API error: {str(e)}"],
                "warnings": [],
                "source": "error",
                "model": model_id,
                "model_name": model_name,
                "library_name": library_name,
                "file_name": item.get("file_name"),
                "time_seconds": time.time() - start_time,
                "load_time_seconds": 0
            }
            results.append(result)
            continue

        inference_time = time.time() - inference_start

        result = parse_response(answer)
        result.update({
            "source": "groq",
            "model": model_id,
            "model_name": model_name,
            "library_name": library_name,
            "file_name": item.get("file_name"),
            "has_cpg": cpg_content is not None,
            "time_seconds": time.time() - start_time,
            "inference_time_seconds": inference_time,
            "load_time_seconds": 0,
            "raw_response": answer[:500]
        })

        results.append(result)

    total_time = sum(r.get("time_seconds", 0) for r in results)
    print(f"\nBatch complete: {len(results)} files in {total_time:.2f}s (API calls only)")

    return results


# ============================================
# Modal Functions - One per model for proper caching
# ============================================

@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_llama_1b(items: list) -> list:
    """Batch validate with Llama 3.2 1B."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    return run_batch_validation(items, "meta-llama/Llama-3.2-1B-Instruct", "llama-3.2-1b")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_llama_3b(items: list) -> list:
    """Batch validate with Llama 3.2 3B."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    return run_batch_validation(items, "meta-llama/Llama-3.2-3B-Instruct", "llama-3.2-3b")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume}
)
def validate_qwen_1_5b(items: list) -> list:
    """Batch validate with Qwen 2.5 1.5B."""
    return run_batch_validation(items, "Qwen/Qwen2.5-1.5B-Instruct", "qwen-2.5-1.5b")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume}
)
def validate_qwen_3b(items: list) -> list:
    """Batch validate with Qwen 2.5 3B."""
    return run_batch_validation(items, "Qwen/Qwen2.5-3B-Instruct", "qwen-2.5-3b")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume}
)
def validate_phi3(items: list) -> list:
    """Batch validate with Phi-3 Mini."""
    return run_batch_validation(items, "microsoft/Phi-3-mini-4k-instruct", "phi-3-mini")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_gemma(items: list) -> list:
    """Batch validate with Gemma 3 4B."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    return run_batch_validation(items, "google/gemma-3-4b-it", "gemma-3-4b")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_medgemma(items: list) -> list:
    """Batch validate with MedGemma 4B."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    return run_batch_validation(items, "google/medgemma-4b-it", "medgemma-4b")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_medgemma_1_5(items: list) -> list:
    """Batch validate with MedGemma 1.5 4B."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    return run_batch_validation(items, "google/medgemma-1.5-4b-it", "medgemma-1.5-4b")


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_llama_3_1_8b(items: list) -> list:
    """Batch validate with Llama 3.1 8B."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    return run_batch_validation(items, "meta-llama/Llama-3.1-8B-Instruct", "llama-3.1-8b")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_gemma_270m(items: list) -> list:
    """Batch validate with Gemma 3 270M."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    return run_batch_validation(items, "google/gemma-3-270m-it", "gemma-3-270m")


@app.function(
    image=groq_image,
    timeout=1800,
    secrets=[modal.Secret.from_name("groq-api")]
)
def validate_gpt_oss_120b(items: list) -> list:
    """Batch validate with GPT OSS 120B via Groq API."""
    return run_batch_groq_validation(items, "openai/gpt-oss-120b", "gpt-oss-120b")


@app.function(
    image=groq_image,
    timeout=1800,
    secrets=[modal.Secret.from_name("groq-api")]
)
def validate_gpt_oss_20b(items: list) -> list:
    """Batch validate with GPT OSS 20B via Groq API."""
    return run_batch_groq_validation(items, "openai/gpt-oss-20b", "gpt-oss-20b")


@app.function(
    image=groq_image,
    timeout=1800,
    secrets=[modal.Secret.from_name("groq-api")]
)
def validate_llama_3_3_70b(items: list) -> list:
    """Batch validate with Llama 3.3 70B via Groq API."""
    return run_batch_groq_validation(items, "llama-3.3-70b-versatile", "llama-3.3-70b")


# Single unified registry
MODEL_FUNCTIONS = {
    "llama-3.2-1b": validate_llama_1b,
    "llama-3.2-3b": validate_llama_3b,
    "qwen-2.5-1.5b": validate_qwen_1_5b,
    "qwen-2.5-3b": validate_qwen_3b,
    "phi-3-mini": validate_phi3,
    "gemma-3-4b": validate_gemma,
    "medgemma-4b": validate_medgemma,
    "medgemma-1.5-4b": validate_medgemma_1_5,
    "llama-3.1-8b": validate_llama_3_1_8b,
    "gemma-3-270m": validate_gemma_270m,
    "gpt-oss-120b": validate_gpt_oss_120b,
    "gpt-oss-20b": validate_gpt_oss_20b,
    "llama-3.3-70b": validate_llama_3_3_70b,
}


def get_validator(model_id: str):
    """Get the validation function for a model."""
    if model_id not in MODEL_FUNCTIONS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(MODEL_FUNCTIONS.keys())}")
    return MODEL_FUNCTIONS[model_id]


def get_model_config(model_id: str) -> dict:
    """Get configuration for a model."""
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_id]


@app.local_entrypoint()
def main(
    elm_file: str = None,
    model: str = "llama-3.2-1b",
    cpg_file: str = None,
    all_models: bool = False
):
    """
    CLI entrypoint for validation.

    Usage:
        modal run modal_app.py --elm-file path/to/elm.json
        modal run modal_app.py --elm-file path/to/elm.json --model qwen-2.5-1.5b
        modal run modal_app.py --elm-file path/to/elm.json --cpg-file path/to/cpg.md
        modal run modal_app.py --elm-file path/to/elm.json --all-models
    """
    from pathlib import Path

    if not elm_file:
        print("Usage: modal run modal_app.py --elm-file <path>")
        print("\nAvailable models:")
        for model_id in MODEL_FUNCTIONS.keys():
            print(f"  {model_id}")
        return

    elm_path = Path(elm_file)
    if not elm_path.exists():
        print(f"Error: File not found: {elm_file}")
        return

    with open(elm_path, 'r') as f:
        elm_json = json.load(f)

    library_name = elm_json.get("library", {}).get("identifier", {}).get("id", elm_path.stem)

    # Load CPG if provided
    cpg_content = None
    if cpg_file:
        cpg_path = Path(cpg_file)
        if cpg_path.exists():
            with open(cpg_path, 'r') as f:
                cpg_content = f.read()
            print(f"Loaded CPG: {cpg_file}")

    # Create batch item (single file)
    items = [{
        "elm_json": elm_json,
        "library_name": library_name,
        "cpg_content": cpg_content,
        "file_name": elm_path.name
    }]

    models_to_run = list(MODEL_FUNCTIONS.keys()) if all_models else [model]

    print(f"\n{'='*70}")
    print(f"ELM Validation: {library_name}")
    if cpg_content:
        print(f"CPG: {cpg_file}")
    print(f"{'='*70}\n")

    all_results = []
    for model_id in models_to_run:
        print(f"\nValidating with {model_id}...")
        validator = get_validator(model_id)
        results = validator.remote(items)

        for result in results:
            all_results.append(result)
            status = "VALID" if result.get("valid") else "INVALID"
            print(f"  Result: {status}")
            print(f"  Time: {result['time_seconds']:.2f}s")
            if result["errors"]:
                print(f"  Errors: {result['errors']}")
            if result["warnings"]:
                print(f"  Warnings: {result['warnings']}")

    if all_models:
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'Status':<10} {'Time (s)':<10}")
        print("-" * 40)
        for r in sorted(all_results, key=lambda x: x['time_seconds']):
            status = "VALID" if r["valid"] else "INVALID"
            print(f"{r['model']:<20} {status:<10} {r['time_seconds']:<10.2f}")