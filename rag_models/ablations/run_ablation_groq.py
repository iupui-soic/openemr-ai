#!/usr/bin/env python
"""
Groq ablation driver — 3 models on Fareez n=40, configurable ablation cell.

Usage:
    # No-RAG baseline (one cell at a time)
    python run_ablation_groq.py --mode norag --model gpt-oss-120b
    python run_ablation_groq.py --mode norag --model gpt-oss-20b
    python run_ablation_groq.py --mode norag --model qwen3-32b

    # k-sweep
    python run_ablation_groq.py --mode k --k 1 --model gpt-oss-120b
    python run_ablation_groq.py --mode k --k 3 --model gpt-oss-120b
    python run_ablation_groq.py --mode k --k 5 --model gpt-oss-120b

    # Embedding substitution
    python run_ablation_groq.py --mode embed --embedding clinicalbert --model gpt-oss-120b
    python run_ablation_groq.py --mode embed --embedding pubmedbert  --model gpt-oss-120b

    # Temperature sweep (top-2 models)
    python run_ablation_groq.py --mode temp --temperature 0.1 --model gpt-oss-120b
    python run_ablation_groq.py --mode temp --temperature 0.5 --model gpt-oss-120b
    python run_ablation_groq.py --mode temp --temperature 0.7 --model gpt-oss-120b

    # Prompt variants (top-2 models)
    python run_ablation_groq.py --mode prompt --prompt-variant minimal              --model gpt-oss-120b
    python run_ablation_groq.py --mode prompt --prompt-variant hallucination_guarded --model gpt-oss-120b

Outputs (always written under rag_models/results/fareez/ablations/<cell>/<model>/):
    <patient>.txt          - full summary
    evaluation_results_<model>.csv  - per-case timings + token counts
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make pipeline importable for FareezLoader
RAG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_ROOT))
sys.path.insert(0, str(RAG_ROOT / "pipeline"))

from ablations._ablation_core import (   # noqa: E402
    AblationRAG,
    PROMPT_TEMPLATES,
    generate_summary_groq,
    make_token_counter,
    write_eval_csv,
    write_summary_file,
)
from fareez_rag_loader import FareezLoader  # noqa: E402

# Load .env if present (mirrors run_fareez_local.py)
try:
    from dotenv import load_dotenv
    load_dotenv(str(RAG_ROOT.parent / ".env"))
except Exception:
    pass

MODELS = {
    "gpt-oss-120b": {"name": "openai/gpt-oss-120b", "api_type": "reasoning"},
    "gpt-oss-20b":  {"name": "openai/gpt-oss-20b",  "api_type": "reasoning"},
    "qwen3-32b":    {"name": "qwen/qwen3-32b",      "api_type": "standard"},
}

EMBEDDING_MAP = {
    "minilm":        "all-MiniLM-L6-v2",                                 # production
    "clinicalbert":  "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert":    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}

# Production reference (used as defaults / sanity checks)
PROD_K = 2
PROD_TEMP_REASONING = 1.0
PROD_TEMP_STANDARD = 0.3
PROD_PROMPT = "current"
PROD_EMBEDDING = "minilm"


def cell_subdir(args) -> str:
    """Return the per-config subdirectory name under fareez/ablations/."""
    if args.mode == "norag":
        return "norag"
    if args.mode == "k":
        return f"k{args.k}"
    if args.mode == "embed":
        return f"embed_{args.embedding}"
    if args.mode == "temp":
        return f"temp_{args.temperature:g}"
    if args.mode == "prompt":
        return f"prompt_{args.prompt_variant}"
    raise SystemExit(f"unknown mode {args.mode}")


def resolve_config(args, api_type: str):
    """
    Resolve (rag_enabled, embedding_key, k, temperature, prompt_variant) for the run,
    starting from production defaults and overriding the one variable being ablated.
    """
    rag_enabled = True
    embedding_key = PROD_EMBEDDING
    k = PROD_K
    temperature = PROD_TEMP_REASONING if api_type == "reasoning" else PROD_TEMP_STANDARD
    prompt_variant = PROD_PROMPT

    if args.mode == "norag":
        rag_enabled = False
    elif args.mode == "k":
        k = args.k
    elif args.mode == "embed":
        embedding_key = args.embedding
    elif args.mode == "temp":
        temperature = args.temperature
    elif args.mode == "prompt":
        prompt_variant = args.prompt_variant

    return rag_enabled, embedding_key, k, temperature, prompt_variant


def main():
    p = argparse.ArgumentParser(description="Groq ablation driver for Fareez n=40")
    p.add_argument("--mode", required=True,
                   choices=["norag", "k", "embed", "temp", "prompt"])
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--k", type=int, default=PROD_K)
    p.add_argument("--embedding", choices=list(EMBEDDING_MAP), default=PROD_EMBEDDING)
    p.add_argument("--temperature", type=float, default=None,
                   help="Override temperature (default: production value per api_type)")
    p.add_argument("--prompt-variant", choices=list(PROMPT_TEMPLATES), default=PROD_PROMPT)
    p.add_argument("--output-root", default="results/fareez/ablations")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N cases (smoke test)")
    args = p.parse_args()

    # Resolve config
    cfg = MODELS[args.model]
    rag_enabled, embedding_key, k, temperature, prompt_variant = resolve_config(
        args, cfg["api_type"]
    )
    if args.temperature is not None:
        temperature = args.temperature

    cell = cell_subdir(args)
    out_root = (RAG_ROOT / args.output_root / cell).resolve()
    out_dir = out_root / args.model
    summaries_dir = out_dir   # we don't add a fareez_summaries/ layer here — the cell name disambiguates

    print("=" * 72)
    print(f"Groq ablation: mode={args.mode} cell={cell} model={args.model}")
    print(f"  RAG={'on' if rag_enabled else 'off'}  embedder={embedding_key}  k={k}"
          f"  temp={temperature}  prompt={prompt_variant}")
    print(f"  output: {out_dir}")
    print("=" * 72)

    # Load data
    loader = FareezLoader()
    patients = loader.get_entries()
    if args.limit:
        patients = patients[:args.limit]
    print(f"Loaded {len(patients)} Fareez entries")

    # Load RAG (heavy)
    rag = AblationRAG(
        embedding_model_name=EMBEDDING_MAP[embedding_key],
        k=k,
        enabled=rag_enabled,
    )
    rag.load()

    # Initialise Groq client
    from groq import Groq
    client = Groq()
    count_tokens = make_token_counter()

    # Iterate cases
    rows = []
    model_start = time.time()
    for i, patient in enumerate(patients, 1):
        patient_name = patient["patient_name"]
        condition = patient.get("detected_condition", "General")

        case_start = time.time()
        try:
            schema_context = rag.retrieve(condition)
            text, prompt_text = generate_summary_groq(
                client,
                model_name=cfg["name"],
                api_type=cfg["api_type"],
                prompt_variant=prompt_variant,
                transcript=patient["transcript"],
                openemr=patient.get("openemr_data", ""),
                schema_context=schema_context,
                temperature=temperature,
            )
            elapsed = time.time() - case_start
            input_tokens = count_tokens(prompt_text)
            output_tokens = count_tokens(text)

            write_summary_file(
                summaries_dir, patient_name, cfg["name"], condition, text,
                elapsed, input_tokens, output_tokens,
            )
            rows.append({
                "patient_name": patient_name,
                "model": args.model,
                "detected_disease": condition,
                "total_time_s": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "summary_chars": len(text),
            })
            print(f"  [{i:2d}/{len(patients)}] {patient_name:>10s} "
                  f"({patient['category']}, {condition[:24]:>24s}) "
                  f"-> {elapsed:5.1f}s  {output_tokens:>5d} tok  {len(text):>5d} chars")
        except Exception as e:
            elapsed = time.time() - case_start
            msg = str(e)[:120]
            print(f"  [{i:2d}/{len(patients)}] {patient_name:>10s} -> ERROR: {msg}")
            rows.append({
                "patient_name": patient_name,
                "model": args.model,
                "detected_disease": condition,
                "total_time_s": elapsed,
                "input_tokens": 0,
                "output_tokens": 0,
                "summary_chars": 0,
            })
            # Crude rate-limit handling, mirrors run_fareez_local.py
            if "rate" in msg.lower() or "429" in msg:
                print("    (sleeping 30s for rate limit)")
                time.sleep(30)

    # Write per-case CSV
    csv_path = write_eval_csv(out_dir, args.model, rows)
    elapsed = time.time() - model_start
    successful = sum(1 for r in rows if r["summary_chars"] > 0)
    print(f"\nDone: {successful}/{len(rows)} summaries in {elapsed/60:.1f} min "
          f"-> {csv_path}")


if __name__ == "__main__":
    main()
