"""Re-run only the notes the channel bug lost, on both llama.cpp models.

`approaches/base.py::answer_text` now reads the reasoning channel when `content`
is empty, which is where these two models were putting their answers. Every run
that went through `modal_muse_gguf.py` or `modal_gemma4_gguf.py` scored some
notes as empty predictions when the model had in fact answered.

    modal deploy coding_bench/adapters/modal_muse_gguf.py     # both servers,
    modal deploy coding_bench/adapters/modal_gemma4_gguf.py   # they changed
    modal deploy coding_bench/run_chain_channel_fix.py
    modal run coding_bench/run_chain_channel_fix.py::launch

Deploy both servers first. Their `complete` methods now return every channel
rather than a single resolved string, so a queue running against a stale
deployment will fail on the first note.

## What this costs, and why it is not a full rerun

Nothing that already worked is generated again. `bench/cache.py` declares the
four adapter revisions these runs used as equivalent to the two fixed ones, for
notes that came back with codes and without truncation. `answer_text` returns
`content` byte for byte whenever `content` holds anything, so those notes cannot
have changed, and their committed predictions are reused as cache hits:

    cpt   / gemma4 / gold / 312   157 reused, 155 re-run
    cpt   / muse   / gold / 150    30 reused, 120 re-run
    icd10 / muse   / gold / 150   105 reused,  45 re-run

The parameters below are the ones those records name, because the cache key
covers max_tokens and reasoning strength. Change either and every note misses
and is paid for again, which is the whole saving gone.

Gemma is queued at 32,768 rather than 16,384. Both budgets were tried and both
are banked, the larger one truncated slightly less, and re-running only one of
them halves the bill for a comparison that has already been made.

## What this will not fix

The truncated notes, 31 of Gemma's 312. A model cut off mid thought has no
answer in any channel, so those come back truncated again and Gemma's failure
rate stays above the 5% ceiling. Expect that run to stay quarantined, with a
better score inside the quarantine table. Muse truncated nothing, so its two
runs should come back clean.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from coding_bench.eval_remote import image as eval_image

RESULTS_MOUNT = "/results"
GOLD_MOUNT = "/gold"

app = modal.App("coding-bench-chain-channel-fix")

results_volume = modal.Volume.from_name("coding-benchmark-results", create_if_missing=True)
gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

# Both GGUF servers pin themselves to one container and four slots, so this is
# the concurrency that fills one without queueing work at Modal's door. The
# queue is sequential, so the two servers are never busy at the same time.
SLOTS = 4

CHAIN: list[dict] = [
    # Cheapest first. Gemma generates at roughly 220 tok/s, so even its
    # truncating notes are minutes rather than the hour Muse takes.
    {"task": "cpt", "model": "gemma4-26b-a4b-gguf", "candidate_space": "gold",
     "concurrency": SLOTS - 1, "max_tokens": 32768},
    # Muse at the parameters its records name. 120 of its 150 CPT notes were
    # lost to the bug, so this run is almost entirely new work.
    {"task": "cpt", "model": "muse-glimmer-30b-gguf", "candidate_space": "gold",
     "concurrency": SLOTS, "max_tokens": 16384, "limit": 150},
    # Last, because it is the slowest. This one was never quarantined: it
    # scored 0.701 with 45 of 150 notes silently thrown away, so it is the run
    # whose published number is most likely to move.
    {"task": "icd10", "model": "muse-glimmer-30b-gguf", "candidate_space": "gold",
     "concurrency": SLOTS, "max_tokens": 16384, "limit": 150},
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
    timeout=86400,
    cpu=4.0,
    memory=8192,
)
def run_fix_chain(chain: list[dict] | None = None, git_sha: str | None = None) -> list[dict]:
    """Run the queue inside Modal, committing each record as it lands."""
    from coding_bench.run_chain import _evaluate_inline

    queue = chain if chain is not None else CHAIN
    summaries = []

    for index, step in enumerate(queue, start=1):
        config = {**DEFAULTS, **step, "git_sha": git_sha}
        label = f"{config['task']}/{config['model']}/{config['candidate_space']}"
        print(f"\n[{index}/{len(queue)}] starting {label}", flush=True)

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
                # The saving, stated per step rather than assumed. If `carried`
                # is far below what the queue predicted, the equivalence did not
                # match and this run is quietly costing full price.
                "cache_hits": ops.get("cache_hits", 0),
                "carried": ops.get("cache_carried_forward", 0),
            }
        )
        print(f"[{index}/{len(queue)}] done {label}: {json.dumps(summaries[-1])}", flush=True)

    print("\n=== channel fix chain complete ===", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


@app.local_entrypoint()
def launch():
    """Spawn the queue server side, so nothing on this machine is holding it up."""
    from coding_bench.eval_remote import local_git_state

    sha, dirty = local_git_state()
    if dirty:
        print("Working tree is dirty; these runs will not be exactly reproducible.")

    print(f"Queueing {len(CHAIN)} reruns:")
    for step in CHAIN:
        print(f"  {step['task']:6} {step['model']:22} n={step.get('limit', 'all')} "
              f"max_tokens={step['max_tokens']}")

    fn = modal.Function.from_name(app.name, "run_fix_chain")
    call = fn.spawn(git_sha=sha)

    print(f"\nSpawned as call {call.object_id}")
    print("Watch or collect with:")
    print(f"  modal app logs {app.name}")
    print("  modal run coding_bench/run_chain.py::fetch")
