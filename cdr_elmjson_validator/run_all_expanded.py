#!/usr/bin/env python3
"""Run all 12 models on expanded 31-case benchmark."""

import os
import sys
import json
import csv
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from modal_app import (build_prompt, parse_response, extract_embedded_errors,
                       MAX_PROMPT_CHARS_GROQ, MAX_PROMPT_CHARS_LOCAL)

BASE_DIR = Path(__file__).parent
TEST_DATA_DIR = BASE_DIR / "test_data"
GROUND_TRUTH_FILE = TEST_DATA_DIR / "ground_truth.json"
RESULTS_DIR = BASE_DIR / "results"

GROQ_MODELS = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-32b": "qwen/qwen3-32b",
    "llama-3.3-70b": "llama-3.3-70b-versatile",
}

LOCAL_MODELS = {
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "medgemma-4b": "google/medgemma-4b-it",
    "medgemma-1.5-4b": "google/medgemma-1.5-4b-it",
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
            print(f"  WARNING: {fname} not found, skipping")
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


def save_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "library", "model", "valid", "time_seconds",
                  "errors", "warnings", "source", "cpg_file", "has_cpg",
                  "has_ground_truth", "correct", "expected_valid",
                  "error_match", "warning_match"]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = r.copy()
            row["errors"] = "; ".join(r.get("errors", []))
            row["warnings"] = "; ".join(r.get("warnings", []))
            writer.writerow(row)


def run_groq_model(client, model_id, groq_model, items):
    print(f"\n{'='*60}")
    print(f"  {model_id} via Groq API")
    print(f"{'='*60}")
    results = []
    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item['file_name']}", end="", flush=True)
        embedded = extract_embedded_errors(item["elm_json"])
        if embedded:
            r = {"file": item["file_name"], "library": item["library_name"],
                 "model": model_id, "valid": False, "errors": embedded,
                 "warnings": [], "source": "embedded", "time_seconds": 0}
        else:
            prompt = build_prompt(item["elm_json"], item["library_name"],
                                  item["cpg_content"], max_chars=MAX_PROMPT_CHARS_GROQ)
            start = time.time()
            try:
                resp = client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1, max_tokens=500)
                answer = resp.choices[0].message.content.strip()
                parsed = parse_response(answer)
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "valid": parsed["valid"],
                     "errors": parsed["errors"], "warnings": parsed["warnings"],
                     "source": "groq", "time_seconds": time.time() - start}
            except Exception as e:
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "valid": False,
                     "errors": [f"API error: {str(e)[:200]}"],
                     "warnings": [], "source": "error",
                     "time_seconds": time.time() - start}
        r["cpg_file"] = item.get("cpg_file")
        r["has_cpg"] = item["cpg_content"] is not None
        r["expected_valid"] = item["expected_valid"]
        r["correct"] = r["valid"] == item["expected_valid"]
        r["has_ground_truth"] = True
        r["error_match"] = ""
        r["warning_match"] = ""
        results.append(r)
        status = "OK" if r["correct"] else "WRONG"
        print(f"  {status} ({r['time_seconds']:.1f}s)")
    correct = sum(1 for r in results if r["correct"])
    print(f"  => {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    return results


def run_local_model(model_id, hf_name, items):
    print(f"\n{'='*60}")
    print(f"  {model_id} (local GPU)")
    print(f"{'='*60}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=torch.float16, device_map="auto")
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None
    is_gemma = "gemma" in hf_name.lower()
    results = []
    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item['file_name']}", end="", flush=True)
        embedded = extract_embedded_errors(item["elm_json"])
        if embedded:
            r = {"file": item["file_name"], "library": item["library_name"],
                 "model": model_id, "valid": False, "errors": embedded,
                 "warnings": [], "source": "embedded", "time_seconds": 0}
        else:
            prompt = build_prompt(item["elm_json"], item["library_name"],
                                  item["cpg_content"], max_chars=MAX_PROMPT_CHARS_LOCAL)
            start = time.time()
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=4096).to(model.device)
                with torch.no_grad():
                    if is_gemma:
                        outputs = model.generate(**inputs, max_new_tokens=500,
                                                 do_sample=False,
                                                 pad_token_id=tokenizer.eos_token_id)
                    else:
                        outputs = model.generate(**inputs, max_new_tokens=500,
                                                 temperature=0.1, do_sample=True,
                                                 pad_token_id=tokenizer.eos_token_id)
                resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = resp[len(tokenizer.decode(inputs["input_ids"][0],
                                                    skip_special_tokens=True)):].strip()
                parsed = parse_response(answer)
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "valid": parsed["valid"],
                     "errors": parsed["errors"], "warnings": parsed["warnings"],
                     "source": "llm", "time_seconds": time.time() - start}
            except Exception as e:
                torch.cuda.empty_cache()
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "valid": False,
                     "errors": [f"Inference error: {str(e)[:200]}"],
                     "warnings": [], "source": "error",
                     "time_seconds": time.time() - start}
        r["cpg_file"] = item.get("cpg_file")
        r["has_cpg"] = item["cpg_content"] is not None
        r["expected_valid"] = item["expected_valid"]
        r["correct"] = r["valid"] == item["expected_valid"]
        r["has_ground_truth"] = True
        r["error_match"] = ""
        r["warning_match"] = ""
        results.append(r)
        status = "OK" if r["correct"] else "WRONG"
        print(f"  {status} ({r['time_seconds']:.1f}s)")
    correct = sum(1 for r in results if r["correct"])
    print(f"  => {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    del model, tokenizer
    torch.cuda.empty_cache()
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--groq-only", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--model", help="Run specific model only")
    args = parser.parse_args()

    items = load_test_data()
    print(f"Loaded {len(items)} test cases")

    # Groq models
    if not args.local_only:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("WARNING: No GROQ_API_KEY, skipping Groq models")
        else:
            client = Groq(api_key=api_key)
            for mid, gm in GROQ_MODELS.items():
                if args.model and args.model != mid:
                    continue
                out = RESULTS_DIR / f"results-{mid}.csv"
                if out.exists():
                    print(f"\nSkipping {mid} — exists")
                    continue
                results = run_groq_model(client, mid, gm, items)
                save_results(results, out)
                print(f"  Saved: {out}")

    # Local models
    if not args.groq_only:
        for mid, hf in LOCAL_MODELS.items():
            if args.model and args.model != mid:
                continue
            out = RESULTS_DIR / f"results-{mid}.csv"
            if out.exists():
                print(f"\nSkipping {mid} — exists")
                continue
            results = run_local_model(mid, hf, items)
            if results:
                save_results(results, out)
                print(f"  Saved: {out}")

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
