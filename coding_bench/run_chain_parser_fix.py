"""Re-run the notes that came back empty, on every model, after the parser fix.

`approaches/llm.py::salvage_codes` recovers codes from an answer whose JSON does
not parse. Models are asked to quote the note verbatim beside every code,
clinical notes contain quote marks, and an unescaped one inside a JSON string
flips the parser's in-string state so that a complete answer is discarded as
silence. That is not specific to a model or a provider, it depends only on
whether the note being quoted happens to contain a quote mark, so every run here
is affected and every run here is repeated.

    modal deploy coding_bench/adapters/modal_muse_gguf.py     # unchanged, but
    modal deploy coding_bench/adapters/modal_gemma4_gguf.py   # deploy to be sure
    modal deploy coding_bench/run_chain_parser_fix.py
    modal run coding_bench/run_chain_parser_fix.py::launch

## Why this needs `recompute_empty` rather than an adapter equivalence

The other two fixes were in adapters, and the adapter hash is in the cache key,
so declaring an equivalence was enough to say which notes could keep their
answers. This fix is in the parser, which no key covers at all: change it and
every note still hits, including the ones it was written to rescue.

The deeper reason is that the cache stores the parse and not the response. A
discarded answer was never written down, so there is nothing to reinterpret and
the note has to be generated again. `recompute_empty` treats a cached empty
prediction as a miss and leaves every other note alone, which is the smallest
set that can possibly change.

## The bill

Between 11 and 155 notes per run rather than 150 to 578, and the Groq half is
pennies. The parameters below are the ones each run's manifest records, in
particular gpt-oss at 4,096 and 8,192 rather than the 16,384 default. Getting
one of them wrong does not fail, it silently misses every note in that run and
pays full price, which is why they are written out per step rather than
defaulted.

## What it actually bought, which is not what it looked like it would

The salvage fired on **5 notes across all 11 runs**: 3 in qwen icd10 full, 2 in
qwen icd10 gold, none anywhere else. Every other point of movement came from
`recompute_empty` asking again and getting an answer that parsed on the second
attempt, which means those notes were transient rather than structurally broken.

So the parser bug was real, and notes 716852 and 717377 are proof of it, but it
was far rarer than the silence counts implied. Most of what looked like the same
problem was not the same problem. Worth remembering before reading a future
count of empty predictions as evidence of anything in particular.

The reruns also show how much noise a single run carries. Nothing about these
models changed, and re-asking their silent notes moved gpt-oss cpt full from
0.747 to 0.778 and Muse CPT from 0.488 to 0.551. Gemma's failure rate went the
other way, 5.1% to 5.5%, because the notes it was asked again all ran long.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from coding_bench.eval_remote import image as eval_image

RESULTS_MOUNT = "/results"
GOLD_MOUNT = "/gold"

app = modal.App("coding-bench-chain-parser-fix")

results_volume = modal.Volume.from_name("coding-benchmark-results", create_if_missing=True)
gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

# Groq first. Those steps need nothing but HTTP, so they finish while the GPU
# servers are still cold, and an interruption loses the least.
CHAIN: list[dict] = [
    {"task": "cpt", "model": "qwen3.6-27b", "candidate_space": "gold",
     "concurrency": 2, "max_tokens": 16384},
    {"task": "cpt", "model": "qwen3.6-27b", "candidate_space": "full",
     "concurrency": 2, "max_tokens": 16384},
    {"task": "cpt", "model": "gpt-oss-120b", "candidate_space": "gold",
     "concurrency": 3, "max_tokens": 16384},
    {"task": "cpt", "model": "gpt-oss-120b", "candidate_space": "full",
     "concurrency": 3, "max_tokens": 16384},
    {"task": "icd10", "model": "qwen3.6-27b", "candidate_space": "gold",
     "concurrency": 2, "max_tokens": 16384},
    {"task": "icd10", "model": "qwen3.6-27b", "candidate_space": "full",
     "concurrency": 2, "max_tokens": 16384},
    # Both of these ran at a smaller budget than everything else. Matching it is
    # what makes the other 500 notes free.
    {"task": "icd10", "model": "gpt-oss-120b", "candidate_space": "gold",
     "concurrency": 3, "max_tokens": 4096},
    {"task": "icd10", "model": "gpt-oss-120b", "candidate_space": "full",
     "concurrency": 3, "max_tokens": 8192},
    # Then the GPUs, cheapest first.
    {"task": "cpt", "model": "gemma4-26b-a4b-gguf", "candidate_space": "gold",
     "concurrency": 3, "max_tokens": 32768},
    {"task": "cpt", "model": "muse-glimmer-30b-gguf", "candidate_space": "gold",
     "concurrency": 4, "max_tokens": 16384, "limit": 150},
    {"task": "icd10", "model": "muse-glimmer-30b-gguf", "candidate_space": "gold",
     "concurrency": 4, "max_tokens": 16384, "limit": 150},
]

DEFAULTS = {
    "approach": "llm",
    "max_tokens": 16384,
    "reasoning_strength": "medium",
    "cache": "auto",
    "recompute_empty": True,
}


@app.function(
    image=eval_image,
    volumes={RESULTS_MOUNT: results_volume, GOLD_MOUNT: gold_volume},
    secrets=[modal.Secret.from_name("groq-api-key")],
    timeout=86400,
    cpu=4.0,
    memory=8192,
)
def run_parser_chain(chain: list[dict] | None = None, git_sha: str | None = None) -> list[dict]:
    """Run the queue inside Modal, committing each record as it lands."""
    from coding_bench.run_chain import _evaluate_inline

    queue = chain if chain is not None else CHAIN
    summaries = []

    for index, step in enumerate(queue, start=1):
        config = {**DEFAULTS, **step, "git_sha": git_sha}
        label = f"{config['task']}/{config['model']}/{config['candidate_space']}"
        print(f"\n[{index}/{len(queue)}] starting {label} "
              f"(max_tokens {config['max_tokens']})", flush=True)

        try:
            record = _evaluate_inline(config)
        except Exception as exc:  # noqa: BLE001
            print(f"[{index}/{len(queue)}] FAILED {label}: {type(exc).__name__}: {exc}", flush=True)
            summaries.append({"step": label, "status": "failed", "error": str(exc)[:300]})
            continue

        manifest = record["manifest"]
        metrics = record["metrics"]
        ops = metrics["operational"]
        error_rate = ops.get("error_rate", 0.0)

        out_path = Path(RESULTS_MOUNT) / f"{manifest['run_id']}.json"
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf8")
        results_volume.commit()

        summaries.append(
            {
                "step": label,
                "status": "ok",
                "run_id": manifest["run_id"],
                "n_notes": manifest["n_notes"],
                "micro_f1": round(metrics["core"]["micro_f1"], 4),
                "error_rate": round(error_rate, 4),
                "valid": error_rate <= 0.05,
                "cache_hits": ops.get("cache_hits", 0),
                # The point of the whole exercise: notes whose codes were read
                # out of JSON that does not parse. Zero here means the parser
                # fix bought this run nothing.
                "salvaged": ops.get("salvaged", 0),
            }
        )
        print(f"[{index}/{len(queue)}] done {label}: {json.dumps(summaries[-1])}", flush=True)

    print("\n=== parser fix chain complete ===", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


@app.local_entrypoint()
def launch():
    """Spawn the queue server side, so nothing on this machine is holding it up."""
    from coding_bench.eval_remote import local_git_state

    sha, dirty = local_git_state()
    if dirty:
        print("Working tree is dirty; these runs will not be exactly reproducible.")

    print(f"Queueing {len(CHAIN)} reruns, empty notes only:")
    for step in CHAIN:
        print(f"  {step['task']:6} {step['model']:22} {step['candidate_space']:5} "
              f"max_tokens={step['max_tokens']}")

    fn = modal.Function.from_name(app.name, "run_parser_chain")
    call = fn.spawn(git_sha=sha)

    print(f"\nSpawned as call {call.object_id}")
    print("Watch or collect with:")
    print(f"  modal app logs {app.name}")
    print("  modal run coding_bench/run_chain.py::fetch")
