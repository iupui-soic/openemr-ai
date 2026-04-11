#!/usr/bin/env python3
"""
Run ablation study (4 conditions x 5 trials) for new models:
  API:   gemini-3-flash (Google), gpt-5.4-mini (OpenAI)
  Local: gemma-4-26b-a4b, gemma-4-31b (4-bit quantization)

Conditions:
  full            – Simplified ELM + CPG
  no_cpg          – Simplified ELM only
  no_simplify     – Raw truncated JSON + CPG
  no_cpg_no_simplify – Raw truncated JSON only

Usage:
    # Run a specific model
    GOOGLE_API_KEY=xxx python run_new_ablation_multi_trial.py --model gemini-3-flash
    HF_TOKEN=xxx python run_new_ablation_multi_trial.py --model gemma-4-26b-a4b

    # Run all API models
    GOOGLE_API_KEY=xxx OPENAI_API_KEY=xxx python run_new_ablation_multi_trial.py --api-only
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
RESULTS_DIR = BASE_DIR / "results" / "ablation_multi_trial"

N_TRIALS = 5
TEMPERATURE = 0.1
MAX_PROMPT_CHARS = MAX_PROMPT_CHARS_GROQ

ABLATION_CONDITIONS = ["full", "no_cpg", "no_simplify", "no_cpg_no_simplify"]
CONDITION_LABELS = {
    "full": "Full (simpl.+CPG)",
    "no_cpg": "No CPG",
    "no_simplify": "No simplification",
    "no_cpg_no_simplify": "Neither",
}

MODELS = {
    "gemini-3-flash": {
        "provider": "google",
        "api_model": "gemini-3-flash-preview",
        "env_key": "GOOGLE_API_KEY",
        "rate_limit_sleep": 4,
    },
    "gpt-5.4-mini": {
        "provider": "openai",
        "api_model": "gpt-5.4-mini",
        "env_key": "OPENAI_API_KEY",
        "rate_limit_sleep": 0,
        "reasoning_effort": "low",
    },
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
}

API_MODELS = [k for k, v in MODELS.items() if v["provider"] != "local"]
LOCAL_MODELS = [k for k, v in MODELS.items() if v["provider"] == "local"]


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


# ── Provider-specific inference (same as run_new_models_multi_trial.py) ──────

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
                wait = 5 * (2 ** attempt)
                time.sleep(wait)
                continue
            raise
    raise last_err


def infer_openai(client, api_model, prompt, reasoning_effort="none"):
    response = client.chat.completions.create(
        model=api_model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4096,
        reasoning_effort=reasoning_effort,
    )
    return response.choices[0].message.content.strip()


def infer_local(model, tokenizer, prompt):
    import torch, gc
    messages = [{"role": "user", "content": prompt}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    # Larger context for raw JSON ablation (up to 16k input tokens)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                       max_length=16384).to(model.device)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512,
                temperature=TEMPERATURE, do_sample=True, top_p=0.95,
                use_cache=True,
            )
        answer = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        # Aggressively clear cache to prevent OOM across cases
        del inputs
        if 'outputs' in dir():
            del outputs
        gc.collect()
        torch.cuda.empty_cache()
    return answer.strip()


def create_google_infer(config):
    from google import genai
    client = genai.Client(api_key=os.environ[config["env_key"]])
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
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ[config["env_key"]],
    )
    api_model = config["api_model"]
    provider_name = config.get("openrouter_provider")

    def infer(prompt):
        return infer_openrouter(client, api_model, prompt, provider_name=provider_name)
    return infer


def create_openai_infer(config):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ[config["env_key"]])
    api_model = config["api_model"]
    effort = config.get("reasoning_effort", "none")

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
        # Use specific GPU if CUDA_VISIBLE_DEVICES is set, else auto
        dev_map = {"": int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])} \
                  if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, quantization_config=quantization_config,
            device_map=dev_map, token=token,
        )
    else:
        print(f"  Loading {hf_name} (FP16)...")
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(hf_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, dtype=torch.bfloat16,
            device_map="auto", token=token,
        )
    print(f"  Loaded in {time.time() - load_start:.0f}s")

    def infer(prompt):
        return infer_local(model, tokenizer, prompt)
    return infer


def create_infer_fn(model_id):
    config = MODELS[model_id]
    if config["provider"] == "google":
        return create_google_infer(config)
    elif config["provider"] == "openai":
        return create_openai_infer(config)
    elif config["provider"] == "openrouter":
        return create_openrouter_infer(config)
    elif config["provider"] == "local":
        return create_local_infer(config)
    raise ValueError(f"Unknown provider: {config['provider']}")


# ── Trial runner ─────────────────────────────────────────────────────────────

def run_single_trial(infer_fn, items, model_id, ablation_mode, trial_num):
    results = []
    for i, item in enumerate(items, 1):
        print(f"      [{i}/{len(items)}] {item['file_name'][:40]}", end="", flush=True)
        embedded = extract_embedded_errors(item["elm_json"])
        if embedded:
            r = {"file": item["file_name"], "library": item["library_name"],
                 "model": model_id, "trial": trial_num,
                 "ablation": ablation_mode,
                 "valid": False, "errors": embedded,
                 "warnings": [], "source": "embedded", "time_seconds": 0}
        else:
            prompt = build_prompt(item["elm_json"], item["library_name"],
                                  item["cpg_content"], max_chars=MAX_PROMPT_CHARS,
                                  ablation_mode=ablation_mode)
            start = time.time()
            try:
                answer = infer_fn(prompt)
                parsed = parse_response(answer)
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "trial": trial_num,
                     "ablation": ablation_mode,
                     "valid": parsed["valid"],
                     "errors": parsed["errors"], "warnings": parsed["warnings"],
                     "source": MODELS[model_id]["provider"],
                     "time_seconds": time.time() - start}
            except Exception as e:
                print(f"  ERROR: {e}")
                r = {"file": item["file_name"], "library": item["library_name"],
                     "model": model_id, "trial": trial_num,
                     "ablation": ablation_mode,
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
        status = "OK" if r["correct"] else "XX"
        print(f"  {status} ({r['time_seconds']:.1f}s)")

    correct = sum(1 for r in results if r["correct"])
    print(f"      => {correct}/{len(results)} ({correct / len(results) * 100:.1f}%)")
    return results


def save_trial_csv(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "library", "model", "trial", "ablation", "valid",
              "time_seconds", "errors", "warnings", "source", "cpg_file",
              "has_cpg", "has_ground_truth", "correct", "expected_valid"]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in results:
            row = r.copy()
            row["errors"] = "; ".join(r.get("errors", []))
            row["warnings"] = "; ".join(r.get("warnings", []))
            w.writerow(row)


def analyze(all_results, models_run, n_trials):
    by_model_cond = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in all_results:
        by_model_cond[r["model"]][r["ablation"]].append(r)

    print("\n" + "=" * 78)
    print("ABLATION MULTI-TRIAL RESULTS (NEW MODELS)")
    print(f"Trials: {n_trials}, Temperature: {TEMPERATURE}")
    print("=" * 78)

    summary_rows = []
    for model_id in models_run:
        print(f"\n  {model_id}:")
        for cond in ABLATION_CONDITIONS:
            results = by_model_cond[model_id][cond]
            if not results:
                continue
            by_trial = collections.defaultdict(list)
            for r in results:
                by_trial[r["trial"]].append(r)
            accs = np.array([
                sum(1 for r in by_trial[t] if r["correct"]) / len(by_trial[t])
                for t in sorted(by_trial.keys())
            ])
            correct_per_trial = [
                sum(1 for r in by_trial[t] if r["correct"])
                for t in sorted(by_trial.keys())
            ]
            print(f"    {CONDITION_LABELS[cond]:25s}  "
                  f"{accs.mean()*100:.1f}% ± {accs.std()*100:.1f}%  "
                  f"({correct_per_trial})")
            summary_rows.append({
                "model": model_id, "condition": cond,
                "mean_accuracy": round(accs.mean(), 4),
                "std_accuracy": round(accs.std(), 4),
                "min_accuracy": round(accs.min(), 4),
                "max_accuracy": round(accs.max(), 4),
                "per_trial_correct": correct_per_trial,
            })

    # Append to existing summary CSV
    summary_path = RESULTS_DIR / "ablation_multi_trial_summary.csv"
    write_header = not summary_path.exists()
    with open(summary_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "condition", "mean_accuracy", "std_accuracy",
            "min_accuracy", "max_accuracy", "per_trial_correct"])
        if write_header:
            w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary appended: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation multi-trial for new models")
    parser.add_argument("--model", help="Run a specific model only")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

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
    print(f"{args.trials} trials x {len(ABLATION_CONDITIONS)} conditions per model")
    print(f"Models: {models_to_run}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_id in models_to_run:
        print(f"\n{'=' * 70}")
        print(f"  {model_id} ({MODELS[model_id]['provider']})")
        print(f"{'=' * 70}")

        infer_fn = create_infer_fn(model_id)

        for cond in ABLATION_CONDITIONS:
            print(f"\n    --- {CONDITION_LABELS[cond]} ---")
            for trial in range(1, args.trials + 1):
                print(f"\n    Trial {trial}/{args.trials}:")
                results = run_single_trial(
                    infer_fn, items, model_id, cond, trial)

                path = RESULTS_DIR / f"results-{model_id}-{cond}-trial{trial}.csv"
                save_trial_csv(results, path)
                all_results.extend(results)

                if trial < args.trials:
                    time.sleep(2)

    analyze(all_results, models_to_run, args.trials)
    print("\nDone.")


if __name__ == "__main__":
    main()
