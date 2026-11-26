"""The Muse Glimmer half of the queue, on its own app so it can be deployed safely.

Everything here could have been appended to `run_chain.py`'s CHAIN. It is a
separate file for one reason: deploying an app while one of its functions is
mid run is not something to do to a queue that is fifty minutes into a 578 note
evaluation. A second app with a second name cannot touch the first, so this can
be written, deployed and launched while the Groq queue is still going.

    modal deploy coding_bench/adapters/modal_muse_gguf.py   # the GPU server
    modal deploy coding_bench/run_chain_muse.py             # this queue
    modal run coding_bench/run_chain_muse.py::launch --wait-for fc-XXXX

Records land on the same `coding-benchmark-results` volume as the Groq queue,
so the existing collector picks them up with no change:

    modal run coding_bench/run_chain.py::fetch

Why `--wait-for`. The GGUF server is pinned to `max_containers=1`, because the
alternative is every caller getting its own GPU and its own 20GB copy of the
weights. That pin makes the server a single resource, and two queues driving it
at once would interleave their notes through the same four slots and finish
neither any sooner. Passing the other queue's call id makes this one sit and
wait for it to finish first. It is a courtesy, not a lock: if the id is wrong or
already finished, this queue starts anyway and says so.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from coding_bench.eval_remote import image as eval_image

RESULTS_MOUNT = "/results"
GOLD_MOUNT = "/gold"

app = modal.App("coding-bench-chain-muse")

results_volume = modal.Volume.from_name("coding-benchmark-results", create_if_missing=True)
gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

# Concurrency matches PARALLEL_SLOTS in the GGUF adapter. Asking for more than
# the server has slots does not make it faster, it just queues the surplus at
# Modal's door where it is harder to see.
MUSE_SLOTS = 4

# Ordered as asked: the full catalogue first, because that is the number the
# scaling result is missing, then CPT, then the gold extension. Ordering costs
# less than it looks like it should. Every note is cached individually as it
# lands, so a step that dies half way is not repaid on the retry, only resumed.
CHAIN: list[dict] = [
    # The headline gap. gpt-oss has 578 at full catalogue and qwen is getting
    # its 578 in the Groq queue right now; Muse has never been run at full
    # catalogue at all. This is also the step the slot resize was for: at
    # 20,480 per slot the longest prompts here would have overflowed.
    {"task": "icd10", "model": "muse-glimmer-30b-gguf", "candidate_space": "full",
     "concurrency": MUSE_SLOTS},
    # Completes CPT across all three models at the full 312 notes.
    {"task": "cpt", "model": "muse-glimmer-30b-gguf", "candidate_space": "gold",
     "concurrency": MUSE_SLOTS},
    # Muse is the only model whose ICD-10 gold number rests on 150 notes rather
    # than 578, which is why it is the one row in the table that cannot be
    # compared against the others without a caveat. This removes the caveat.
    {"task": "icd10", "model": "muse-glimmer-30b-gguf", "candidate_space": "gold",
     "concurrency": MUSE_SLOTS},
]

DEFAULTS = {
    "approach": "llm",
    "max_tokens": 16384,
    "reasoning_strength": "medium",
    "cache": "auto",
}


def _wait_for_call(call_id: str, poll_s: int = 60, max_wait_s: int = 43200) -> None:
    """Block until another queue's call finishes, then return.

    Deliberately forgiving. Every failure mode here resolves to "start anyway",
    because the cost of starting early is some contention on the GPU server,
    while the cost of refusing to start is a queue that never runs at all.
    """
    try:
        call = modal.FunctionCall.from_id(call_id)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot resolve {call_id} ({type(exc).__name__}), starting now", flush=True)
        return

    print(f"waiting for {call_id} to finish before touching the GPU", flush=True)
    waited = 0
    while waited < max_wait_s:
        try:
            call.get(timeout=0)
            print(f"{call_id} finished, starting", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            # Modal signals "not ready" with a timeout, and anything else means
            # that call is over, whether it succeeded or raised. Either way this
            # queue's turn has come.
            if "timeout" not in type(exc).__name__.lower():
                print(
                    f"{call_id} ended with {type(exc).__name__}, starting", flush=True
                )
                return
        time.sleep(poll_s)
        waited += poll_s

    print(f"gave up waiting for {call_id} after {max_wait_s}s, starting", flush=True)


@app.function(
    image=eval_image,
    volumes={RESULTS_MOUNT: results_volume, GOLD_MOUNT: gold_volume},
    secrets=[modal.Secret.from_name("groq-api-key")],
    # Muse is slow. The full catalogue step alone is measured in hours, so this
    # ceiling is the 24h maximum rather than anything tuned.
    timeout=86400,
    cpu=4.0,
    memory=8192,
)
def run_muse_chain(
    chain: list[dict] | None = None,
    git_sha: str | None = None,
    wait_for: str | None = None,
) -> list[dict]:
    """Run the Muse queue, optionally behind another queue, persisting as we go."""
    # Imported here rather than at module scope on purpose. `run_chain` defines
    # its own modal.App, and pulling that into this file's namespace makes
    # `modal deploy` ambiguous about which app it was asked to deploy.
    from coding_bench.run_chain import _evaluate_inline

    if wait_for:
        _wait_for_call(wait_for)

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
        error_rate = metrics["operational"].get("error_rate", 0.0)

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

    print("\n=== muse chain complete ===", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


@app.local_entrypoint()
def launch(wait_for: str = ""):
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

    print(f"Queueing {len(CHAIN)} Muse runs:")
    for step in CHAIN:
        limit = step.get("limit", "all")
        print(f"  {step['task']:6} {step['candidate_space']:5} n={limit}")
    if wait_for:
        print(f"Holding until {wait_for} finishes.")

    fn = modal.Function.from_name(app.name, "run_muse_chain")
    call = fn.spawn(git_sha=sha, wait_for=wait_for or None)

    print(f"\nSpawned as call {call.object_id}")
    print("Watch or collect with:")
    print(f"  modal app logs {app.name}")
    print("  modal run coding_bench/run_chain.py::fetch")
