"""
Modal deployment of Gemma 4 26B A4B and Gemma 4 31B via Unsloth GGUF Q4_K_M
using llama-cpp-python on A10G GPU.

Why this path:
  Earlier attempts to run Gemma 4 at 4-bit via bitsandbytes NF4 (on Modal)
  and vLLM MXFP4 (on Cloud Run Blackwell) both hit blockers specifically
  for Gemma 4 26B A4B: bnb NF4 produces garbage output on the 31B dense
  model, while vLLM MXFP4's MARLIN MoE backend has a weight-loader bug
  (IndexError at fused_moe/layer.py:1041) that prevents loading the 26B
  A4B MoE checkpoint.

  Unsloth has published pre-calibrated GGUF Q4_K_M weights for both
  Gemma 4 variants that work out of the box with llama.cpp / llama-cpp-
  python. GGUF Q4_K_M is the community-standard 4-bit deployment format
  for edge LLMs — it's what users actually run on consumer GPUs, Ollama,
  LM Studio, etc. Storage is ~4 bits per weight; compute is fp16 (GGUF
  dequantizes to fp16 for matmul). This is NOT the same as native
  Blackwell FP4 tensor-core math, but it IS the dominant real-world
  deployment format for these models.

Paper framing:
  We measure each model at its ecosystem-standard deployment precision:
  GPT-OSS at MXFP4 (native release via Groq), Qwen3/Llama/Qwen3.5 at
  fp16 (Groq/OpenRouter default), Gemma 4 at GGUF Q4_K_M (Unsloth, the
  dominant HF-published 4-bit variant for this model family), mid/small
  models at bf16 (local). This heterogeneous-but-realistic precision
  policy reflects actual deployment practice; we document the per-model
  precision in the methods section.

Usage:
  modal run modal_gemma_gguf.py::sanity --model 26b
  modal run modal_gemma_gguf.py::sanity --model 31b

  (Full experiment is driven by run_gemma_gguf.py which imports run_batch.)
"""

import modal

app = modal.App("gemma-gguf-elm-validator")

# CUDA 12.4 runtime base image — llama-cpp-python's CUDA wheel dynamically
# links against libcudart.so.12 which needs the full CUDA runtime, not just
# the devel headers. The nvidia/cuda:*-runtime image is the minimal ship
# that has libcudart.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget", "curl", "git", "libgomp1")
    .pip_install(
        "huggingface-hub>=0.26.0",
        # Use latest llama-cpp-python which has Gemma 4 architecture support
        # (upstream llama.cpp added it shortly after the Feb 2026 model release).
        # Earlier versions (e.g. 0.3.16) cannot load Gemma 4 GGUFs at all.
        "llama-cpp-python>=0.3.20",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124",
    )
)

# Persistent cache for GGUF weights so cold starts don't re-download
volume = modal.Volume.from_name("gemma-gguf-cache", create_if_missing=True)


# ── Model registry ───────────────────────────────────────────────────────────

GGUF_MODELS = {
    "gemma-4-26b-a4b": {
        "repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "pattern": "*Q4_K_M*.gguf",  # Q4_K_M is the recommended 4-bit quant
    },
    "gemma-4-31b": {
        "repo": "unsloth/gemma-4-31B-it-GGUF",
        "pattern": "*Q4_K_M*.gguf",
    },
}


@app.function(
    image=image,
    # A100-80GB chosen after discovering that L40S (48 GB) cannot fit
    # Gemma 4 31B Q4_K_M (18.3 GB weights) plus n_ctx=32768 KV cache.
    # Gemma 4's heterogeneous V head dimensions (per-layer padding to
    # 2048-4096 V-dim by llama.cpp) inflate the KV cache well beyond the
    # naive calculation, causing `Failed to create llama_context` OOM on
    # L40S. A100 80GB has ample headroom (~40 GB free after loading the
    # model + 32K KV cache) and supports both Gemma 4 variants on the
    # same hardware, preserving the paper's "identical backend" internal
    # consistency for the 26B vs 31B compute equalizer comparison.
    gpu="A100-80GB",
    timeout=10800,  # 3 hours
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_batch(model_id: str, prompts: list) -> list:
    """Load a Gemma 4 GGUF model once and run a batch of prompts.

    Args:
        model_id: "gemma-4-26b-a4b" or "gemma-4-31b"
        prompts: list of dicts with "id" (str) and "prompt" (str)

    Returns:
        list of dicts with "id", "answer", "time_seconds", "error"
    """
    import os
    import time
    import glob
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
    from llama_cpp import Llama

    os.environ["HF_HOME"] = "/cache"
    token = os.getenv("HF_TOKEN", "")

    if model_id not in GGUF_MODELS:
        raise ValueError(f"Unknown model_id: {model_id}. Available: {list(GGUF_MODELS.keys())}")
    cfg = GGUF_MODELS[model_id]

    # ── Download GGUF weights (cached in volume after first run) ────────────
    print(f"Downloading {cfg['repo']} ({cfg['pattern']}) to /cache...")
    download_start = time.time()
    last_err = None
    local_dir = None
    for attempt in range(1, 6):
        try:
            local_dir = snapshot_download(
                repo_id=cfg["repo"],
                token=token,
                cache_dir="/cache",
                allow_patterns=[cfg["pattern"]],
                max_workers=4,
            )
            break
        except (HfHubHTTPError, ConnectionError, OSError) as e:
            last_err = e
            wait = 5 * (2 ** (attempt - 1))
            print(f"  Download attempt {attempt}/5 failed: {str(e)[:150]}")
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)
    if local_dir is None:
        raise RuntimeError(f"snapshot_download failed after 5 attempts: {last_err}")
    print(f"Downloaded to {local_dir} in {time.time()-download_start:.0f}s")
    volume.commit()

    # Find the actual GGUF file path (pattern may have matched multiple files;
    # pick the single Q4_K_M file).
    gguf_candidates = sorted(glob.glob(os.path.join(local_dir, "**", "*.gguf"), recursive=True))
    q4km_files = [p for p in gguf_candidates if "Q4_K_M" in p]
    if not q4km_files:
        raise RuntimeError(f"No Q4_K_M GGUF found in {local_dir}. Found: {gguf_candidates}")
    # Prefer non-UD (standard Q4_K_M) over Unsloth Dynamic (UD-Q4_K_M), since
    # the standard variant is guaranteed to be a plain llama.cpp K-quant
    # whereas UD uses Unsloth's dynamic precision scheme which may require
    # newer llama.cpp versions or specific UD-aware loaders.
    standard_q4km = [p for p in q4km_files if "UD-Q4_K_M" not in p]
    if standard_q4km:
        gguf_path = standard_q4km[0]
        print(f"Picked standard Q4_K_M (not UD): {os.path.basename(gguf_path)}")
    else:
        gguf_path = q4km_files[0]
        print(f"Only UD-Q4_K_M available: {os.path.basename(gguf_path)}")
    size_gb = os.path.getsize(gguf_path) / 1e9
    print(f"Using GGUF: {gguf_path} ({size_gb:.1f} GB)")
    print(f"All Q4_K_M files found: {[os.path.basename(p) for p in q4km_files]}")

    # ── Load model with llama-cpp-python ────────────────────────────────────
    print(f"Loading GGUF with n_gpu_layers=-1 (full offload)...")
    load_start = time.time()
    # n_ctx = 32768: bumped from 16384 after observing raw-JSON prompts
    # tokenize to 16-21K tokens in the no_simplify ablation condition
    # (Gemma 4's tokenizer is aggressive on JSON structural chars — the
    # effective ratio for structured content is closer to 1.5 chars/token
    # than the usual 4 chars/token for English prose). 32K safely covers
    # the worst observed prompt size with margin. Memory cost on L40S:
    # weights (~18 GB 31B, ~17 GB 26B A4B) + KV cache at 32K (~15-18 GB
    # accounting for Gemma 4's heterogeneous V head dims which force
    # llama.cpp to pad V cache) + activations (~4 GB) ≈ 37-40 GB. Fits
    # within L40S 48 GB with 8-10 GB headroom.
    llm = Llama(
        model_path=gguf_path,
        n_ctx=32768,
        n_gpu_layers=-1,         # offload all layers to GPU
        n_batch=512,             # prompt processing batch size
        verbose=False,
        chat_format="gemma",     # matches Gemma's chat template; fallback to default if unknown
    )
    print(f"Loaded in {time.time()-load_start:.0f}s")

    # ── Run batch ───────────────────────────────────────────────────────────
    results = []
    total = len(prompts)
    for i, item in enumerate(prompts, 1):
        pid = item["id"]
        prompt = item["prompt"]
        start = time.time()
        error = None
        answer = ""
        try:
            resp = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                top_p=0.95,
                max_tokens=1024,
            )
            answer = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            error = str(e)[:300]
            print(f"  [{i}/{total}] ERROR: {error[:150]}")

        elapsed = time.time() - start
        results.append({
            "id": pid,
            "answer": answer,
            "time_seconds": elapsed,
            "error": error,
        })

        if i % 20 == 0 or i == total or i == 1:
            total_time = sum(r["time_seconds"] for r in results)
            err_count = sum(1 for r in results if r["error"])
            print(f"  [{i}/{total}] last={elapsed:.1f}s, total={total_time/60:.1f}m, errors={err_count}")

    print(f"\nBatch complete. Total inference: "
          f"{sum(r['time_seconds'] for r in results)/60:.1f} minutes, "
          f"errors: {sum(1 for r in results if r['error'])}")
    return results


@app.local_entrypoint()
def sanity(model: str = "26b"):
    """Smoke test using real ELM cases (USPSTFStatin + small valid case)."""
    import sys
    import json
    from pathlib import Path

    base = Path(__file__).parent
    sys.path.insert(0, str(base))
    from modal_app import build_prompt

    model_id = f"gemma-4-{model}-a4b" if model == "26b" else f"gemma-4-{model}"

    test_data = base / "test_data"
    with open(test_data / "ground_truth.json") as f:
        gt = json.load(f)["test_cases"]

    targets = [
        "USPSTFStatinUseForPrimaryPreventionOfCVDInAdultsSharedLogicFHIRv401.json",
        "Adult-Weight-Screening-and-Follow-Up-OpenEMR.json",
    ]

    test_prompts = []
    expected = {}
    for fname in targets:
        if fname not in gt:
            continue
        tc = gt[fname]
        with open(test_data / fname) as f:
            elm_json = json.load(f)
        cpg_content = None
        if tc.get("cpg_file"):
            cpg_path = test_data / tc["cpg_file"]
            if cpg_path.exists():
                cpg_content = cpg_path.read_text()
        library = elm_json.get("library", {}).get("identifier", {}).get("id", fname)
        prompt = build_prompt(
            elm_json, library, cpg_content,
            max_chars=24000, ablation_mode="full",
        )
        test_prompts.append({"id": fname, "prompt": prompt})
        expected[fname] = tc["valid"]
        print(f"Built prompt for {fname}: {len(prompt)} chars, expected_valid={tc['valid']}")

    print(f"\nDispatching to Modal: {model_id}...")
    results = run_batch.remote(model_id, test_prompts)

    for r in results:
        fname = r["id"]
        exp = expected.get(fname)
        print(f"\n=== {fname} ({r['time_seconds']:.1f}s, expected_valid={exp}) ===")
        if r["error"]:
            print(f"ERROR: {r['error']}")
        else:
            print(r["answer"][:800])
