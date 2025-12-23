"""
ELM Validator using Multiple LLMs on Modal

Each model has its own dedicated function for better caching and parallel execution.
"""

import modal
import json
import time

app = modal.App("elm-validator")

# Shared image with ML dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers", "torch", "accelerate", "sentencepiece"
)

# Cache volume for model weights
volume = modal.Volume.from_name("elm-model-cache", create_if_missing=True)


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
    statements = library.get("statements", {}).get("def", [])

    # Extract statement details for better analysis
    stmt_details = []
    for s in statements[:15]:
        if isinstance(s, dict):
            stmt_info = {
                "name": s.get("name"),
                "context": s.get("context"),
                "accessLevel": s.get("accessLevel")
            }
            expr = s.get("expression", {})
            if isinstance(expr, dict):
                stmt_info["expressionType"] = expr.get("type")
            stmt_details.append(stmt_info)

    summary = {
        "libraryName": library.get("identifier", {}).get("id", library_name),
        "version": library.get("identifier", {}).get("version", "unknown"),
        "statementCount": len(statements),
        "statements": stmt_details
    }

    if cpg_content:
        prompt = f"""You are a Clinical Decision Support (CDS) validation expert. Your task is to validate whether the ELM (Expression Logical Model) implementation correctly implements the Clinical Practice Guideline (CPG).

## Clinical Practice Guideline:
{cpg_content}

## ELM Implementation Summary:
{json.dumps(summary, indent=2)}

## Validation Task:
Analyze if the ELM logic correctly implements the CPG by checking:

1. **Population Criteria**: Does the ELM correctly identify the target population (age, conditions)?
2. **Risk Factors**: Are all required risk factors from the CPG checked?
3. **Thresholds**: Are the correct thresholds (e.g., risk percentages, lab values) used?
4. **Exclusions**: Are the exclusion criteria properly implemented?
5. **Recommendation Logic**: Does the logic flow correctly produce the right recommendation?

## Response Format (respond EXACTLY in this format):
VALID: YES or NO
ERRORS: List specific issues where ELM doesn't match CPG, or "None"
WARNINGS: List potential issues or suggestions, or "None"

Your response:"""
    else:
        prompt = f"""You are a Clinical Quality Language (CQL) expert. Analyze this ELM clinical logic.

ELM Library:
{json.dumps(summary, indent=2)}

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
    # Use greedy decoding for Gemma, sampling for others
    is_gemma = "gemma" in model_name.lower()

    if is_gemma:
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,  # Greedy decoding for Gemma (avoids NaN/inf in probability tensor)
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


def run_validation(elm_json, library_name, cpg_content, model_name, model_id):
    """Common validation logic for all models (single file)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load model
    print(f"Loading {model_name}...")
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

    result = run_single_inference(elm_json, library_name, cpg_content, model, tokenizer, model_id, model_name)
    result["load_time_seconds"] = load_time

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
        result["load_time_seconds"] = load_time if i == 1 else 0  # Only count load time once
        results.append(result)

    total_inference_time = sum(r.get("inference_time_seconds", 0) for r in results)
    print(f"\nBatch complete: {len(results)} files in {load_time + total_inference_time:.2f}s (load: {load_time:.2f}s, inference: {total_inference_time:.2f}s)")

    return results


# ============================================
# Model-specific validation functions
# ============================================

@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_llama_1b(data: dict) -> dict:
    """Validate ELM JSON with Llama 3.2 1B."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

    return run_validation(
        elm_json=data.get("elm_json"),
        library_name=data.get("library_name", "Unknown"),
        cpg_content=data.get("cpg_content"),
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        model_id="llama-3.2-1b"
    )


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_llama_3b(data: dict) -> dict:
    """Validate ELM JSON with Llama 3.2 3B."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

    return run_validation(
        elm_json=data.get("elm_json"),
        library_name=data.get("library_name", "Unknown"),
        cpg_content=data.get("cpg_content"),
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        model_id="llama-3.2-3b"
    )


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/cache": volume}
)
def validate_qwen_1_5b(data: dict) -> dict:
    """Validate ELM JSON with Qwen 2.5 1.5B."""
    return run_validation(
        elm_json=data.get("elm_json"),
        library_name=data.get("library_name", "Unknown"),
        cpg_content=data.get("cpg_content"),
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        model_id="qwen-2.5-1.5b"
    )


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/cache": volume}
)
def validate_qwen_3b(data: dict) -> dict:
    """Validate ELM JSON with Qwen 2.5 3B."""
    return run_validation(
        elm_json=data.get("elm_json"),
        library_name=data.get("library_name", "Unknown"),
        cpg_content=data.get("cpg_content"),
        model_name="Qwen/Qwen2.5-3B-Instruct",
        model_id="qwen-2.5-3b"
    )


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/cache": volume}
)
def validate_phi3(data: dict) -> dict:
    """Validate ELM JSON with Phi-3 Mini."""
    return run_validation(
        elm_json=data.get("elm_json"),
        library_name=data.get("library_name", "Unknown"),
        cpg_content=data.get("cpg_content"),
        model_name="microsoft/Phi-3-mini-4k-instruct",
        model_id="phi-3-mini"
    )


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_gemma(data: dict) -> dict:
    """Validate ELM JSON with Gemma 3 4B."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

    return run_validation(
        elm_json=data.get("elm_json"),
        library_name=data.get("library_name", "Unknown"),
        cpg_content=data.get("cpg_content"),
        model_name="google/gemma-3-4b-it",
        model_id="gemma-3-4b"
    )


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_medgemma(data: dict) -> dict:
    """Validate ELM JSON with MedGemma 4B (healthcare-specialized)."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

    return run_validation(
        elm_json=data.get("elm_json"),
        library_name=data.get("library_name", "Unknown"),
        cpg_content=data.get("cpg_content"),
        model_name="google/medgemma-4b-it",
        model_id="medgemma-4b"
    )


# ============================================
# Batch validation functions (process multiple files with single model load)
# ============================================

@app.function(
    image=image,
    gpu="T4",
    timeout=1800,  # 30 min for batch processing
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_batch_llama_1b(items: list) -> list:
    """Batch validate ELM JSON files with Llama 3.2 1B."""
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
def validate_batch_llama_3b(items: list) -> list:
    """Batch validate ELM JSON files with Llama 3.2 3B."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    return run_batch_validation(items, "meta-llama/Llama-3.2-3B-Instruct", "llama-3.2-3b")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume}
)
def validate_batch_qwen_1_5b(items: list) -> list:
    """Batch validate ELM JSON files with Qwen 2.5 1.5B."""
    return run_batch_validation(items, "Qwen/Qwen2.5-1.5B-Instruct", "qwen-2.5-1.5b")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume}
)
def validate_batch_qwen_3b(items: list) -> list:
    """Batch validate ELM JSON files with Qwen 2.5 3B."""
    return run_batch_validation(items, "Qwen/Qwen2.5-3B-Instruct", "qwen-2.5-3b")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume}
)
def validate_batch_phi3(items: list) -> list:
    """Batch validate ELM JSON files with Phi-3 Mini."""
    return run_batch_validation(items, "microsoft/Phi-3-mini-4k-instruct", "phi-3-mini")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def validate_batch_gemma(items: list) -> list:
    """Batch validate ELM JSON files with Gemma 3 4B."""
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
def validate_batch_medgemma(items: list) -> list:
    """Batch validate ELM JSON files with MedGemma 4B."""
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    return run_batch_validation(items, "google/medgemma-4b-it", "medgemma-4b")


# Model function registry (single file)
MODEL_FUNCTIONS = {
    "llama-3.2-1b": validate_llama_1b,
    "llama-3.2-3b": validate_llama_3b,
    "qwen-2.5-1.5b": validate_qwen_1_5b,
    "qwen-2.5-3b": validate_qwen_3b,
    "phi-3-mini": validate_phi3,
    "gemma-3-4b": validate_gemma,
    "medgemma-4b": validate_medgemma,
}

# Model function registry (batch processing)
BATCH_MODEL_FUNCTIONS = {
    "llama-3.2-1b": validate_batch_llama_1b,
    "llama-3.2-3b": validate_batch_llama_3b,
    "qwen-2.5-1.5b": validate_batch_qwen_1_5b,
    "qwen-2.5-3b": validate_batch_qwen_3b,
    "phi-3-mini": validate_batch_phi3,
    "gemma-3-4b": validate_batch_gemma,
    "medgemma-4b": validate_batch_medgemma,
}


def get_validator(model_id: str):
    """Get the validation function for a model."""
    if model_id not in MODEL_FUNCTIONS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(MODEL_FUNCTIONS.keys())}")
    return MODEL_FUNCTIONS[model_id]


def get_batch_validator(model_id: str):
    """Get the batch validation function for a model."""
    if model_id not in BATCH_MODEL_FUNCTIONS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(BATCH_MODEL_FUNCTIONS.keys())}")
    return BATCH_MODEL_FUNCTIONS[model_id]


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

    data = {
        "elm_json": elm_json,
        "library_name": library_name,
        "cpg_content": cpg_content
    }

    models_to_run = list(MODEL_FUNCTIONS.keys()) if all_models else [model]

    print(f"\n{'='*70}")
    print(f"ELM Validation: {library_name}")
    if cpg_content:
        print(f"CPG: {cpg_file}")
    print(f"{'='*70}\n")

    results = []
    for model_id in models_to_run:
        print(f"\nValidating with {model_id}...")
        validator = get_validator(model_id)
        result = validator.remote(data)
        results.append(result)

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
        for r in sorted(results, key=lambda x: x['time_seconds']):
            status = "VALID" if r["valid"] else "INVALID"
            print(f"{r['model']:<20} {status:<10} {r['time_seconds']:<10.2f}")