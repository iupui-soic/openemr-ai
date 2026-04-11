#!/usr/bin/env python3
"""
Run 5-trial evaluation for new models added to the benchmark:
  API:   gemini-3-flash (Google AI Studio), gpt-5-nano (OpenAI)
  Local: gemma-4-26b-a4b, gemma-4-31b (4-bit quantization, needs GPU)

Each model gets 5 trials at temperature=0.1 to match the existing
reproducibility protocol for frontier models.

Usage:
    # Run API models only (no GPU needed)
    GOOGLE_API_KEY=xxx OPENAI_API_KEY=xxx python run_new_models_multi_trial.py --api-only

    # Run a specific model
    GOOGLE_API_KEY=xxx python run_new_models_multi_trial.py --model gemini-3-flash
    OPENAI_API_KEY=xxx python run_new_models_multi_trial.py --model gpt-5-nano
    HF_TOKEN=xxx python run_new_models_multi_trial.py --model gemma-4-26b-a4b

    # Run local models only (needs GPU + HF_TOKEN)
    HF_TOKEN=xxx python run_new_models_multi_trial.py --local-only

    # Run everything
    GOOGLE_API_KEY=xxx OPENAI_API_KEY=xxx HF_TOKEN=xxx python run_new_models_multi_trial.py

Requirements:
    API:   pip install google-genai openai
    Local: pip install transformers torch accelerate bitsandbytes sentencepiece
"""

import os
import sys
import json
import csv
import time
import argparse
import collections
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from modal_app import (build_prompt, parse_response, extract_embedded_errors,
                       MAX_PROMPT_CHARS_GROQ)

BASE_DIR = Path(__file__).parent
TEST_DATA_DIR = BASE_DIR / "test_data"
GROUND_TRUTH_FILE = TEST_DATA_DIR / "ground_truth.json"
RESULTS_DIR = BASE_DIR / "results" / "multi_trial"

N_TRIALS = 5
TEMPERATURE = 0.1

# Use the same prompt size as existing Groq multi-trial experiments for fairness
MAX_PROMPT_CHARS = MAX_PROMPT_CHARS_GROQ

MODELS = {
    # Google AI Studio API
    "gemini-3-flash": {
        "provider": "google",
        "api_model": "gemini-3-flash-preview",
        "env_key": "GOOGLE_API_KEY",
        "rate_limit_sleep": 4,  # free tier: 15 req/min
    },
    # OpenAI API
    "gpt-5-nano": {
        "provider": "openai",
        "api_model": "gpt-5-nano",
        "env_key": "OPENAI_API_KEY",
        "rate_limit_sleep": 0,
        "reasoning_effort": "minimal",
    },
    "gpt-5.4-mini": {
        "provider": "openai",
        "api_model": "gpt-5.4-mini",
        "env_key": "OPENAI_API_KEY",
        "rate_limit_sleep": 0,
        "reasoning_effort": "low",
    },
    # Gemma 4 via OpenRouter (bf16) with pinned providers
    "gemma-4-26b-a4b": {
        "provider": "openrouter",
        "api_model": "google/gemma-4-26b-a4b-it",
        "env_key": "OPENROUTER_API_KEY",
        "openrouter_provider": "Parasail",
    },
    "gemma-4-31b": {
        "provider": "openrouter",
        "api_model": "google/gemma-4-31b-it",
        "env_key": "OPENROUTER_API_KEY",
        "openrouter_provider": "Novita",
    },
    "qwen3.5-35b-a3b": {
        "provider": "openrouter",
        "api_model": "qwen/qwen3.5-35b-a3b",
        "env_key": "OPENROUTER_API_KEY",
    },
    # OpenRouter-hosted small/mid models
    "gemma-3-4b": {
        "provider": "openrouter",
        "api_model": "google/gemma-3-4b-it",
        "env_key": "OPENROUTER_API_KEY",
    },
    "medgemma-4b": {
        "provider": "local",
        "hf_name": "google/medgemma-4b-it",
        "env_key": "HF_TOKEN",
    },
    "medgemma-1.5-4b": {
        "provider": "local",
        "hf_name": "google/medgemma-1.5-4b-it",
        "env_key": "HF_TOKEN",
    },
    # Local small models that ran clean (keep local)
    "llama-3.2-1b": {
        "provider": "local",
        "hf_name": "meta-llama/Llama-3.2-1B-Instruct",
        "env_key": "HF_TOKEN",
    },
    "llama-3.2-3b": {
        "provider": "local",
        "hf_name": "meta-llama/Llama-3.2-3B-Instruct",
        "env_key": "HF_TOKEN",
    },
    "qwen-2.5-1.5b": {
        "provider": "local",
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "qwen-2.5-3b": {
        "provider": "local",
        "hf_name": "Qwen/Qwen2.5-3B-Instruct",
    },
    "phi-3-mini": {
        "provider": "local",
        "hf_name": "microsoft/Phi-3-mini-4k-instruct",
    },
}

API_MODELS = [k for k, v in MODELS.items() if v["provider"] != "local"]
LOCAL_MODELS = [k for k, v in MODELS.items() if v["provider"] == "local"]


# ── Data loading (reuses existing ground truth) ─────────────────────────────

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


# ── Provider-specific inference ──────────────────────────────────────────────

def infer_google(client, genai_types, api_model, prompt):
    response = client.models.generate_content(
        model=api_model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=4096,
        )
    )
    return response.text.strip()


def infer_openrouter(client, api_model, prompt, provider_name=None):
    import re, time
    extra = {"reasoning": {"enabled": False}}
    if provider_name:
        extra["provider"] = {"only": [provider_name]}
    # Retry with exponential backoff on rate limits
    last_err = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=4096,
                extra_body=extra,
            )
            content = response.choices[0].message.content or ""
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            return content.strip()
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "rate" in msg.lower() or "temporarily" in msg.lower():
                wait = 5 * (2 ** attempt)  # 5, 10, 20, 40, 80 seconds
                time.sleep(wait)
                continue
            raise
    raise last_err


def infer_openai(client, api_model, prompt, reasoning_effort="minimal"):
    response = client.chat.completions.create(
        model=api_model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4096,
        reasoning_effort=reasoning_effort,
    )
    return response.choices[0].message.content.strip()


def infer_local(model, tokenizer, prompt):
    import torch
    messages = [{"role": "user", "content": prompt}]
    # Disable thinking mode for models that support it (Qwen3.5, etc.)
    # This prevents the model from consuming all tokens on a reasoning chain
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                       max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=4096,
            temperature=TEMPERATURE, do_sample=True, top_p=0.95,
        )
    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip()


# ── Trial runner ─────────────────────────────────────────────────────────────

def run_single_trial(infer_fn, items, model_id, trial_num):
    """Run one trial of a single model on all test cases."""
    results = []
    for i, item in enumerate(items, 1):
        print(f"    [{i}/{len(items)}] {item['file_name']}", end="", flush=True)

        # Check for embedded CQL-to-ELM compiler errors first
        embedded = extract_embedded_errors(item["elm_json"])
        if embedded:
            r = {"file": item["file_name"], "library": item["library_name"],
                 "model": model_id, "trial": trial_num,
                 "valid": False, "errors": embedded,
                 "warnings": [], "source": "embedded", "time_seconds": 0}
        else:
            prompt = build_prompt(item["elm_json"], item["library_name"],
                                  item["cpg_content"], max_chars=MAX_PROMPT_CHARS)
            start = time.time()
            try:
                answer = infer_fn(prompt)
                parsed = parse_response(answer)
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "trial": trial_num,
                     "valid": parsed["valid"],
                     "errors": parsed["errors"], "warnings": parsed["warnings"],
                     "source": MODELS[model_id]["provider"],
                     "time_seconds": time.time() - start}
            except Exception as e:
                print(f"  ERROR: {e}")
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "trial": trial_num,
                     "valid": False,
                     "errors": [f"API error: {str(e)[:200]}"],
                     "warnings": [], "source": "error",
                     "time_seconds": time.time() - start}

        r["cpg_file"] = item.get("cpg_file")
        r["has_cpg"] = item["cpg_content"] is not None
        r["expected_valid"] = item["expected_valid"]
        r["correct"] = r["valid"] == item["expected_valid"]
        r["has_ground_truth"] = True
        results.append(r)

        status = "OK" if r["correct"] else "WRONG"
        print(f"  {status} ({r['time_seconds']:.1f}s)")

    correct = sum(1 for r in results if r["correct"])
    print(f"    => Trial {trial_num}: {correct}/{len(results)} "
          f"({correct / len(results) * 100:.1f}%)")
    return results


# ── Results I/O ──────────────────────────────────────────────────────────────

def save_trial_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "library", "model", "trial", "valid", "time_seconds",
                  "errors", "warnings", "source", "cpg_file", "has_cpg",
                  "has_ground_truth", "correct", "expected_valid"]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = r.copy()
            row["errors"] = "; ".join(r.get("errors", []))
            row["warnings"] = "; ".join(r.get("warnings", []))
            writer.writerow(row)


def analyze_results(all_results, models_run):
    """Print per-model and per-case stability statistics."""
    by_model = collections.defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    print("\n" + "=" * 70)
    print("MULTI-TRIAL REPRODUCIBILITY ANALYSIS (NEW MODELS)")
    print(f"Trials: {N_TRIALS}, Temperature: {TEMPERATURE}")
    print("=" * 70)

    for model_id in models_run:
        model_results = by_model[model_id]
        if not model_results:
            continue

        by_trial = collections.defaultdict(list)
        for r in model_results:
            by_trial[r["trial"]].append(r)

        trial_accuracies = []
        for t in sorted(by_trial.keys()):
            acc = sum(1 for r in by_trial[t] if r["correct"]) / len(by_trial[t])
            trial_accuracies.append(acc)

        accs = np.array(trial_accuracies)
        print(f"\n{model_id}:")
        print(f"  Per-trial accuracy: {[f'{a*100:.1f}%' for a in accs]}")
        print(f"  Mean: {accs.mean()*100:.1f}%  SD: {accs.std()*100:.1f}%  "
              f"Min: {accs.min()*100:.1f}%  Max: {accs.max()*100:.1f}%")

        # Per-case stability
        by_case = collections.defaultdict(list)
        for r in model_results:
            by_case[r["file"]].append(r["correct"])

        unstable = [(c, sum(v), len(v))
                    for c, v in sorted(by_case.items()) if len(set(v)) > 1]
        if unstable:
            print(f"  Unstable cases ({len(unstable)}):")
            for case, nc, nt in unstable:
                print(f"    {case}: {nc}/{nt} correct")
        else:
            print(f"  All cases stable across {N_TRIALS} trials")


# ── Client/model setup ──────────────────────────────────────────────────────

def create_google_infer(config):
    from google import genai
    api_key = os.environ.get(config["env_key"])
    if not api_key:
        print(f"ERROR: {config['env_key']} not set")
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    api_model = config["api_model"]
    sleep_s = config.get("rate_limit_sleep", 0)

    def infer(prompt):
        result = infer_google(client, genai.types, api_model, prompt)
        if sleep_s > 0:
            time.sleep(sleep_s)
        return result
    return infer


def create_openrouter_infer(config):
    from openai import OpenAI
    api_key = os.environ.get(config["env_key"])
    if not api_key:
        print(f"ERROR: {config['env_key']} not set")
        sys.exit(1)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    api_model = config["api_model"]
    provider_name = config.get("openrouter_provider")

    def infer(prompt):
        return infer_openrouter(client, api_model, prompt, provider_name=provider_name)
    return infer


def create_openai_infer(config):
    from openai import OpenAI
    api_key = os.environ.get(config["env_key"])
    if not api_key:
        print(f"ERROR: {config['env_key']} not set")
        sys.exit(1)
    client = OpenAI(api_key=api_key)
    api_model = config["api_model"]
    effort = config.get("reasoning_effort", "minimal")

    def infer(prompt):
        return infer_openai(client, api_model, prompt, reasoning_effort=effort)
    return infer


def create_local_infer(config):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    hf_name = config["hf_name"]
    token = os.environ.get(config.get("env_key", ""), None)
    use_4bit = config.get("use_4bit", False)

    if use_4bit:
        from transformers import BitsAndBytesConfig
        print(f"  Loading {hf_name} with 4-bit quantization...")
        load_start = time.time()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            quantization_config=quantization_config,
            device_map="auto",
            token=token,
        )
    else:
        print(f"  Loading {hf_name} (FP16)...")
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(hf_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )
    print(f"  Loaded in {time.time() - load_start:.0f}s")

    def infer(prompt):
        return infer_local(model, tokenizer, prompt)
    return infer


def create_infer_fn(model_id):
    config = MODELS[model_id]
    provider = config["provider"]
    if provider == "google":
        return create_google_infer(config)
    elif provider == "openai":
        return create_openai_infer(config)
    elif provider == "openrouter":
        return create_openrouter_infer(config)
    elif provider == "local":
        return create_local_infer(config)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-trial evaluation for new models")
    parser.add_argument("--model", help="Run a specific model only")
    parser.add_argument("--api-only", action="store_true",
                        help="Run API models only (gemini-3-flash, gpt-5-nano)")
    parser.add_argument("--local-only", action="store_true",
                        help="Run local models only (gemma-4-26b-a4b, gemma-4-31b)")
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help=f"Number of trials (default: {N_TRIALS})")
    args = parser.parse_args()

    # Determine which models to run
    if args.model:
        if args.model not in MODELS:
            print(f"ERROR: Unknown model '{args.model}'. "
                  f"Available: {list(MODELS.keys())}")
            sys.exit(1)
        models_to_run = [args.model]
    elif args.api_only:
        models_to_run = API_MODELS
    elif args.local_only:
        models_to_run = LOCAL_MODELS
    else:
        models_to_run = list(MODELS.keys())

    items = load_test_data()
    print(f"Loaded {len(items)} test cases")
    print(f"Running {args.trials} trials per model at temperature={TEMPERATURE}")
    print(f"Models: {models_to_run}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_id in models_to_run:
        print(f"\n{'=' * 60}")
        print(f"  {model_id} ({MODELS[model_id]['provider']}) — {args.trials} trials")
        print(f"{'=' * 60}")

        infer_fn = create_infer_fn(model_id)

        for trial in range(1, args.trials + 1):
            print(f"\n  --- Trial {trial}/{args.trials} ---")
            results = run_single_trial(infer_fn, items, model_id, trial)

            trial_path = RESULTS_DIR / f"results-{model_id}-trial{trial}.csv"
            save_trial_results(results, trial_path)
            all_results.extend(results)

            if trial < args.trials:
                time.sleep(2)

    analyze_results(all_results, models_to_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
