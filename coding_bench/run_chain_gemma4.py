"""The Gemma 4 queue, reconstructed from the two run records it produced.

Like `adapters/modal_gemma4_gguf.py`, this is a reconstruction rather than the
original file. Both runs it describes are already banked in `results/runs/`, and
the parameters below are read straight out of their manifests rather than
chosen here:

    cpt / gold / 312 notes / max_tokens 16384 / concurrency 4   ->  022922Z
    cpt / gold / 312 notes / max_tokens 32768 / concurrency 3   ->  035127Z

Re-running this queue will not reproduce those records byte for byte. The cache
keys on the adapter hash, so every note will miss and be regenerated, and the
new record will name this commit rather than the two that are missing from this
history.

    modal deploy coding_bench/adapters/modal_gemma4_gguf.py   # the GPU server
    modal deploy coding_bench/run_chain_gemma4.py             # this queue
    modal run coding_bench/run_chain_gemma4.py::launch

Its own app, for the reason `run_chain_muse.py` has one: deploying an app while
one of its functions is mid run is not something to do to a queue that is an
hour into a long evaluation. A second app with a second name cannot touch the
first.

Records land on the same `coding-benchmark-results` volume as every other
queue, so the existing collector picks them up unchanged:

    modal run coding_bench/run_chain.py::fetch

## Why the second step exists, and why it did not work

The first run failed 10.9% of its notes on truncation, against a 5% ceiling that
quarantines a result. The obvious reading is that the generation budget was too
small, so the second run doubled it to 32,768 and lowered concurrency to 3 to
pay for the extra KV cache.

It bought one point. Truncation went to 9.9%, mean latency went from 26.7s to
36.4s, and p95 from 148s to 266s. Both runs are quarantined and neither is a
result.

A third run at a bigger number is the one thing not worth queueing. On a task
whose gold answer averages 1.01 codes, a model exhausting 32k tokens is not
short of room, and the adapter's smoke test now shows why: the generated text is
arriving in a channel the adapter does not read, so a longer budget only buys
more unread thinking. The adapter docstring has the detail. Requeue this after
that is fixed, not before.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from coding_bench.eval_remote import image as eval_image

RESULTS_MOUNT = "/results"
GOLD_MOUNT = "/gold"

app = modal.App("coding-bench-chain-gemma4")

results_volume = modal.Volume.from_name("coding-benchmark-results", create_if_missing=True)
gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

# Matches PARALLEL_SLOTS on the GGUF server. Asking for more than the server has
# slots does not go faster, it just queues the surplus at Modal's door where it
# is harder to see.
GEMMA_SLOTS = 4

CHAIN: list[dict] = [
    # As first run. 16,384 tokens is the package default and 4 notes in flight
    # fills the server exactly.
    {"task": "cpt", "model": "gemma4-26b-a4b-gguf", "candidate_space": "gold",
     "concurrency": GEMMA_SLOTS, "max_tokens": 16384},
    # The retry. Doubling the budget doubles what a slot must hold, so one slot
    # is given up to keep the KV cache within what the card already fits.
    {"task": "cpt", "model": "gemma4-26b-a4b-gguf", "candidate_space": "gold",
     "concurrency": GEMMA_SLOTS - 1, "max_tokens": 32768},
]

DEFAULTS = {
    "approach": "llm",
    "max_tokens": 16384,
    "reasoning_strength": "medium",
    "cache": "auto",
}


@app.function(
    image=eval_image,
    volumes={RESULTS_MOUNT: results_volume, GOLD_MOUNT: gold_volume},
    secrets=[modal.Secret.from_name("groq-api-key")],
    # Nothing here touches Groq, but the image expects the secret to resolve.
    timeout=86400,
    cpu=4.0,
    memory=8192,
)
def run_gemma4_chain(chain: list[dict] | None = None, git_sha: str | None = None) -> list[dict]:
    """Run the queue inside Modal, committing each record as it lands."""
    # Imported here rather than at module scope. `run_chain` defines its own
    # modal.App, and pulling that into this namespace makes `modal deploy`
    # ambiguous about which app it was asked to deploy.
    from coding_bench.run_chain import _evaluate_inline

    queue = chain if chain is not None else CHAIN
    summaries = []

    for index, step in enumerate(queue, start=1):
        config = {**DEFAULTS, **step, "git_sha": git_sha}
        label = f"{config['task']}/{config['model']}/{config['candidate_space']}"
        print(f"\n[{index}/{len(queue)}] starting {label} "
              f"(max_tokens {config['max_tokens']}, concurrency {config['concurrency']})",
              flush=True)

        try:
            record = _evaluate_inline(config)
        except Exception as exc:  # noqa: BLE001
            # One failing step must not abandon the rest of the queue.
            print(f"[{index}/{len(queue)}] FAILED {label}: {type(exc).__name__}: {exc}", flush=True)
            summaries.append({"step": label, "status": "failed", "error": str(exc)[:300]})
            continue

        manifest = record["manifest"]
        metrics = record["metrics"]
        error_rate = metrics["operational"].get("error_rate", 0.0)

        # Commit immediately. If the next step dies, this one is still banked.
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
            }
        )
        print(f"[{index}/{len(queue)}] done {label}: {json.dumps(summaries[-1])}", flush=True)

    print("\n=== gemma4 chain complete ===", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


@app.local_entrypoint()
def launch():
    """Spawn the queue server side, so nothing on this machine is holding it up.

    See the note in `run_chain.py::launch`: a spawn against the deployed
    function belongs to the deployed app, so there is no client whose death can
    cancel it. `modal run --detach` is not equivalent and was measured losing a
    step ninety seconds after the launcher was killed.
    """
    from coding_bench.eval_remote import local_git_state

    sha, dirty = local_git_state()
    if dirty:
        print("Working tree is dirty; these runs will not be exactly reproducible.")

    print(f"Queueing {len(CHAIN)} Gemma 4 runs:")
    for step in CHAIN:
        print(f"  {step['task']:6} {step['candidate_space']:5} "
              f"max_tokens={step['max_tokens']} concurrency={step['concurrency']}")

    fn = modal.Function.from_name(app.name, "run_gemma4_chain")
    call = fn.spawn(git_sha=sha)

    print(f"\nSpawned as call {call.object_id}")
    print("Watch or collect with:")
    print(f"  modal app logs {app.name}")
    print("  modal run coding_bench/run_chain.py::fetch")
