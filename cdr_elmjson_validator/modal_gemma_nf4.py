"""
Modal deployment of Gemma 4 26B A4B and Gemma 4 31B at NF4 (4-bit) on H100.

Purpose: deployment-precision evaluation for the ELM Validator paper.
The original Gemma 4 runs at bf16 via OpenRouter weakened the compute
equalizer argument because bf16 is not the edge deployment target for
these models. This Modal app runs them at NF4 — the standard edge
quantization — on an H100 80GB GPU to avoid the OOM issues that hit
the local RTX 6000 Ada (48GB) on the USPSTFStatin case.

Quantization config (pinned for reproducibility):
    load_in_4bit=True
    bnb_4bit_quant_type="nf4"
    bnb_4bit_use_double_quant=True
    bnb_4bit_compute_dtype=bfloat16

Usage (driven by run_gemma_nf4_modal.py):
    modal run modal_gemma_nf4.py::run_batch \\
        --hf-name google/gemma-4-26b-a4b-it --prompts-json /tmp/prompts.json
"""

import modal

app = modal.App("gemma-nf4-elm-validator")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.56.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        "sentencepiece",
        "protobuf",
    )
)

# Persistent cache for HF model weights (so we don't re-download on every run)
volume = modal.Volume.from_name("gemma-nf4-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=10800,  # 3 hours max per invocation
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_batch(hf_name: str, prompts: list) -> list:
    """Load Gemma 4 at NF4 once and run a batch of prompts sequentially.

    Args:
        hf_name: e.g. "google/gemma-4-26b-a4b-it" or "google/gemma-4-31b-it"
        prompts: list of dicts with keys "id" (str) and "prompt" (str)

    Returns:
        list of dicts with keys "id", "answer", "time_seconds", "error"
    """
    import os
    import time
    import torch

    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # avoid hf_transfer if not installed

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    token = os.getenv("HF_TOKEN", "")
    if not token:
        print("WARNING: HF_TOKEN not set — gated models will fail")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 1: Pre-download all weights with explicit retry ────────────────
    print(f"Downloading {hf_name} weights to /cache (with retry)...")
    download_start = time.time()
    local_dir = None
    last_err = None
    for attempt in range(1, 6):
        try:
            local_dir = snapshot_download(
                repo_id=hf_name,
                token=token,
                cache_dir="/cache",
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.model",
                    "tokenizer*",
                    "*.py",
                ],
                max_workers=4,
            )
            break
        except (HfHubHTTPError, ConnectionError, OSError) as e:
            last_err = e
            wait = 5 * (2 ** (attempt - 1))  # 5, 10, 20, 40, 80
            print(f"  Download attempt {attempt}/5 failed: {str(e)[:150]}")
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)
    if local_dir is None:
        raise RuntimeError(f"snapshot_download failed after 5 attempts: {last_err}")
    print(f"Weights downloaded to {local_dir} in {time.time()-download_start:.0f}s")
    volume.commit()  # persist cache for next run

    # ── Step 2: Load tokenizer + model from local cache ─────────────────────
    print(f"Loading {hf_name} at NF4 from local cache...")
    load_start = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        local_dir, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    load_time = time.time() - load_start
    print(f"Loaded in {load_time:.1f}s")
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Gemma 4 has documented numerical-stability issues with low-temperature
    # sampling at 4-bit — outputs collapse to token repetition (e.g. the same
    # line over and over, or Unicode garbage). Force greedy decoding for Gemma
    # to match the convention in modal_app.py::run_single_inference. Greedy
    # decoding is also what the OpenRouter bf16 runs effectively produced
    # (±0.0% SD across trials), so it's consistent with the prior results.
    use_greedy = "gemma" in hf_name.lower()
    print(f"Decoding strategy: {'greedy (do_sample=False)' if use_greedy else 'sampling T=0.1'}")

    results = []
    total = len(prompts)
    oom_count = 0

    for i, item in enumerate(prompts, 1):
        pid = item["id"]
        prompt = item["prompt"]
        start = time.time()
        error = None
        answer = ""

        try:
            messages = [{"role": "user", "content": prompt}]
            try:
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                input_text = prompt

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=16384,
            ).to(model.device)

            with torch.no_grad():
                if use_greedy:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.1,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            answer = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()

            del inputs, outputs
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            oom_count += 1
            error = f"OOM: {str(e)[:200]}"
            torch.cuda.empty_cache()
            print(f"  [{i}/{total}] OOM on id={pid}")
        except Exception as e:
            error = str(e)[:300]
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            print(f"  [{i}/{total}] error: {error[:100]}")

        elapsed = time.time() - start
        results.append({
            "id": pid,
            "answer": answer,
            "time_seconds": elapsed,
            "error": error,
        })

        if i % 20 == 0 or i == total or i == 1:
            done_time = sum(r["time_seconds"] for r in results)
            print(
                f"  [{i}/{total}] last={elapsed:.1f}s, "
                f"total={done_time/60:.1f}m, oom={oom_count}"
            )

    print(f"\nBatch complete. Model load: {load_time:.0f}s, "
          f"total inference: {sum(r['time_seconds'] for r in results)/60:.1f}m, "
          f"OOM errors: {oom_count}")

    return results


@app.function(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def inspect_config(hf_name: str) -> dict:
    """Download and return the raw config.json for a HF model (no GPU needed)."""
    import os
    import json
    from huggingface_hub import hf_hub_download

    token = os.getenv("HF_TOKEN", "")
    path = hf_hub_download(repo_id=hf_name, filename="config.json", token=token)
    with open(path) as f:
        cfg = json.load(f)
    print(f"\n=== {hf_name} config.json ===")
    print(f"model_type: {cfg.get('model_type')}")
    print(f"architectures: {cfg.get('architectures')}")
    print(f"torch_dtype: {cfg.get('torch_dtype')}")
    print(f"transformers_version: {cfg.get('transformers_version')}")
    print(f"hidden_size: {cfg.get('hidden_size')}")
    print(f"num_hidden_layers: {cfg.get('num_hidden_layers')}")
    print(f"num_attention_heads: {cfg.get('num_attention_heads')}")
    print(f"all keys: {sorted(cfg.keys())}")
    return cfg


@app.local_entrypoint()
def inspect(hf_name: str = "google/gemma-4-31b-it"):
    """Print config of a Gemma 4 HF repo to diagnose architecture mismatches."""
    inspect_config.remote(hf_name)


@app.local_entrypoint()
def sanity(hf_name: str = "google/gemma-4-26b-a4b-it"):
    """Smoke test using real ELM test cases.

    Includes USPSTFStatin (the case that OOM'd the RTX 6000 Ada original run)
    plus one small valid case, so we exercise the real build_prompt pipeline
    with the full ELM JSON + CPG payload. A successful run should produce
    parseable VALID:/ERRORS: output for both cases.
    """
    import sys, json
    from pathlib import Path
    base = Path(__file__).parent
    sys.path.insert(0, str(base))
    from modal_app import build_prompt

    test_data = base / "test_data"
    with open(test_data / "ground_truth.json") as f:
        gt = json.load(f)["test_cases"]

    targets = [
        "USPSTFStatinUseForPrimaryPreventionOfCVDInAdultsSharedLogicFHIRv401.json",  # prior OOM case
        "Adult-Weight-Screening-and-Follow-Up-OpenEMR.json",  # small valid case
    ]

    test_prompts = []
    expected = {}
    for fname in targets:
        if fname not in gt:
            print(f"SKIP: {fname} not in ground truth")
            continue
        tc = gt[fname]
        elm_path = test_data / fname
        with open(elm_path) as f:
            elm_json = json.load(f)
        cpg_content = None
        if tc.get("cpg_file"):
            cpg_path = test_data / tc["cpg_file"]
            if cpg_path.exists():
                with open(cpg_path) as f:
                    cpg_content = f.read()
        library = elm_json.get("library", {}).get("identifier", {}).get("id", fname)
        prompt = build_prompt(
            elm_json, library, cpg_content,
            max_chars=24000, ablation_mode="full",
        )
        test_prompts.append({"id": fname, "prompt": prompt})
        expected[fname] = tc["valid"]
        print(f"Built prompt for {fname}: {len(prompt)} chars, expected_valid={tc['valid']}")

    print(f"\nDispatching {len(test_prompts)} prompts to {hf_name}...")
    results = run_batch.remote(hf_name, test_prompts)

    for r in results:
        fname = r["id"]
        exp = expected.get(fname)
        print(f"\n=== {fname} ({r['time_seconds']:.1f}s, expected_valid={exp}) ===")
        if r["error"]:
            print(f"ERROR: {r['error']}")
        else:
            print(r["answer"][:800])
