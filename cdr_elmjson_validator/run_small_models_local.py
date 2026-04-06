#!/usr/bin/env python3
"""
Run small models (1-3B) locally on GPU for genuine inference results.

These models previously failed on Modal but can run locally on RTX 6000.
"""

import os
import sys
import json
import csv
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from modal_app import build_prompt, parse_response, extract_embedded_errors, MAX_PROMPT_CHARS_LOCAL

BASE_DIR = Path(__file__).parent
TEST_DATA_DIR = BASE_DIR / "test_data"
GROUND_TRUTH_FILE = TEST_DATA_DIR / "ground_truth.json"
RESULTS_DIR = BASE_DIR / "results"

SMALL_MODELS = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
}


def load_test_data():
    with open(GROUND_TRUTH_FILE) as f:
        gt = json.load(f)

    items = []
    for fname, tc in gt["test_cases"].items():
        elm_path = TEST_DATA_DIR / fname
        if not elm_path.exists():
            continue
        with open(elm_path) as f:
            elm_json = json.load(f)

        cpg_content = None
        cpg_file = tc.get("cpg_file")
        if cpg_file:
            cpg_path = TEST_DATA_DIR / cpg_file
            if cpg_path.exists():
                with open(cpg_path) as f:
                    cpg_content = f.read()

        library = elm_json.get("library", {}).get("identifier", {}).get("id", fname)
        items.append({
            "file_name": fname, "elm_json": elm_json, "library_name": library,
            "cpg_content": cpg_content, "cpg_file": cpg_file,
            "expected_valid": tc["valid"],
        })
    return items


def run_model(model_id, hf_name, items):
    print(f"\n{'='*70}")
    print(f"Loading {model_id} ({hf_name})...")

    output_path = RESULTS_DIR / f"results-{model_id}.csv"

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=torch.float16, device_map="auto"
        )
    except Exception as e:
        print(f"  Failed to load: {e}")
        return None

    load_time = time.time()
    print(f"  Model loaded on {next(model.parameters()).device}")

    results = []
    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item['file_name']}", end="")
        start = time.time()

        # Check embedded errors
        embedded_errors = extract_embedded_errors(item["elm_json"])
        if embedded_errors:
            result = {
                "file": item["file_name"], "library": item["library_name"],
                "model": model_id, "valid": False, "errors": embedded_errors,
                "warnings": [], "source": "embedded", "time_seconds": 0,
            }
        else:
            prompt = build_prompt(item["elm_json"], item["library_name"],
                                  item["cpg_content"], max_chars=MAX_PROMPT_CHARS_LOCAL)
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=4096).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=500,
                        temperature=0.1, do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = response_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
                parsed = parse_response(answer)
                elapsed = time.time() - start
                result = {
                    "file": item["file_name"], "library": item["library_name"],
                    "model": model_id, "valid": parsed["valid"],
                    "errors": parsed["errors"], "warnings": parsed["warnings"],
                    "source": "llm", "time_seconds": elapsed,
                }
            except Exception as e:
                elapsed = time.time() - start
                print(f" ERROR: {str(e)[:80]}", end="")
                torch.cuda.empty_cache()
                result = {
                    "file": item["file_name"], "library": item["library_name"],
                    "model": model_id, "valid": False,
                    "errors": [f"Inference error: {str(e)[:200]}"],
                    "warnings": [], "source": "error", "time_seconds": elapsed,
                }

        # Add ground truth
        result["cpg_file"] = item.get("cpg_file")
        result["has_cpg"] = item["cpg_content"] is not None
        result["expected_valid"] = item["expected_valid"]
        result["correct"] = result["valid"] == item["expected_valid"]
        result["has_ground_truth"] = True
        result["error_match"] = ""
        result["warning_match"] = ""
        results.append(result)

        status = "correct" if result["correct"] else "WRONG"
        print(f"  -> {status} ({result['time_seconds']:.1f}s)")

    correct = sum(1 for r in results if r["correct"])
    print(f"\n  Accuracy: {correct}/{len(results)} ({correct/len(results):.1%})")

    # Save
    fieldnames = [
        "file", "library", "model", "valid", "time_seconds",
        "errors", "warnings", "source", "cpg_file", "has_cpg",
        "has_ground_truth", "correct", "expected_valid",
        "error_match", "warning_match"
    ]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = r.copy()
            row["errors"] = "; ".join(r.get("errors", []))
            row["warnings"] = "; ".join(r.get("warnings", []))
            writer.writerow(row)
    print(f"  Saved: {output_path}")

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(SMALL_MODELS.keys()),
                       help="Run specific model (default: all)")
    args = parser.parse_args()

    items = load_test_data()
    print(f"Loaded {len(items)} test cases")
    print(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")

    models = {args.model: SMALL_MODELS[args.model]} if args.model else SMALL_MODELS

    for model_id, hf_name in models.items():
        # Skip if results already exist with valid source
        output_path = RESULTS_DIR / f"results-{model_id}.csv"
        if output_path.exists():
            with open(output_path) as f:
                rows = list(csv.DictReader(f))
            if rows and rows[0].get("source") != "error":
                print(f"\nSkipping {model_id} — valid results already exist")
                continue
            print(f"\nRe-running {model_id} — previous results were errors")

        run_model(model_id, hf_name, items)

    print("\nAll small model experiments complete.")


if __name__ == "__main__":
    main()
