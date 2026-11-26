"""Muse Glimmer at the full ICD-10 catalogue, the one run this benchmark lacks.

Every other model has both conditions. Muse has only ever been measured at gold
candidates, so the gold to full collapse that this benchmark exists to quantify
rests on two models rather than three. This is the run that closes that gap, and
it has been queued twice before without ever completing.

    modal deploy coding_bench/adapters/modal_muse_gguf.py
    modal deploy coding_bench/run_chain_muse_full.py
    modal run coding_bench/run_chain_muse_full.py::launch

## Expect hours, not minutes

578 notes at roughly 11k tokens of prompt each, against a model that measured 96
seconds per note at gold candidates where the prompt was a tenth of the size.
Four slots on one GPU. Four to six hours is the realistic range, and the queue is
detached precisely so that nothing on a laptop has to survive it.

Every note is cached as it lands, so an interruption costs the notes in flight
rather than the run. Relaunching resumes.

## The slot arithmetic, which is why this fits at all

The GGUF server offers 4 slots of 40,960 tokens. A slot has to hold the whole
conversation: the longest full catalogue prompt in this dataset is 19,130 tokens
and the generation budget is 16,384, so the worst case is 35,514 and it fits
with room to spare. That sizing is the reason `PARALLEL_SLOTS` was halved from
eight when the full catalogue was first attempted. At the old 20,480 per slot a
long note would have overflowed, come back truncated, scored as an empty answer
and still been charged for in full.

`recompute_empty` is off. There is nothing cached for this configuration to
reconsider, and on a resume it would pay to regenerate notes the model had
already genuinely declined.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from coding_bench.eval_remote import image as eval_image

RESULTS_MOUNT = "/results"
GOLD_MOUNT = "/gold"

app = modal.App("coding-bench-chain-muse-full")

results_volume = modal.Volume.from_name("coding-benchmark-results", create_if_missing=True)
gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

CHAIN: list[dict] = [
    {"task": "icd10", "model": "muse-glimmer-30b-gguf", "candidate_space": "full",
     "concurrency": 4, "max_tokens": 16384, "recompute_empty": False},
]

DEFAULTS = {
    "approach": "llm",
    "max_tokens": 16384,
    "reasoning_strength": "medium",
    "cache": "auto",
    "recompute_empty": False,
}


@app.function(
    image=eval_image,
    volumes={RESULTS_MOUNT: results_volume, GOLD_MOUNT: gold_volume},
    secrets=[modal.Secret.from_name("groq-api-key")],
    # A day, because this step is measured in hours and the ceiling should never
    # be the thing that ends it.
    timeout=86400,
    cpu=4.0,
    memory=8192,
)
def run_muse_full(chain: list[dict] | None = None, git_sha: str | None = None) -> list[dict]:
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
                "truncation_rate": round(ops.get("truncation_rate", 0.0), 4),
                "cache_hits": ops.get("cache_hits", 0),
                "salvaged": ops.get("salvaged", 0),
            }
        )
        print(f"[{index}/{len(queue)}] done {label}: {json.dumps(summaries[-1])}", flush=True)

    print("\n=== muse full chain complete ===", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


@app.local_entrypoint()
def launch():
    """Spawn the run server side, so nothing on this machine is holding it up."""
    from coding_bench.eval_remote import local_git_state

    sha, dirty = local_git_state()
    if dirty:
        print("Working tree is dirty; this run will not be exactly reproducible.")

    fn = modal.Function.from_name(app.name, "run_muse_full")
    call = fn.spawn(git_sha=sha)

    print(f"Spawned as call {call.object_id}")
    print("Expect four to six hours. Watch or collect with:")
    print(f"  modal app logs {app.name}")
    print("  modal run coding_bench/run_chain.py::fetch")
