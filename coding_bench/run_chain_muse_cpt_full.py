"""Muse Glimmer at the full CPT catalogue, the last condition it lacks.

With the ICD-10 full record collected on 2026-08-14, Muse sits at three of the
four conditions: both candidate spaces on ICD-10, gold only on CPT. This run
closes the square, so that the gold to full collapse is measured on all three
models for both code systems rather than for ICD-10 alone.

    modal deploy coding_bench/adapters/modal_muse_gguf.py
    modal deploy coding_bench/run_chain_muse_cpt_full.py
    modal run coding_bench/run_chain_muse_cpt_full.py::launch

## Expect minutes, not the hours ICD-10 full took

The two runs are not comparable in cost. CPT offers 61 codes against ICD-10's
673, which adds roughly 900 tokens of code list to a prompt rather than the
10,920 token average the full ICD-10 catalogue produces. Muse measured 19.2
seconds per note on CPT at gold candidates, and the extra code list should not
move that far. 312 notes over four slots is well under an hour, against the
nineteen the ICD-10 full run spent.

## Why 312 notes and not the 150 the gold run used

Every other model reports CPT full over the whole 312 note set, and a row that
cannot be read against the rows above it is not worth the GPU time. This is the
same choice the ICD-10 full run made: it ran all 578 while Muse's gold row
stayed at 150. The consequence is that Muse's CPT gold and CPT full numbers come
from different samples, so the collapse between them is indicative rather than
paired. The head to head table is unaffected, since it pairs across models
within a condition and never across conditions.

## Slots are left at four

`PARALLEL_SLOTS` is 4 in the server, halved from eight when the full ICD-10
catalogue arrived and a slot had to hold a 19,130 token prompt. CPT prompts are
smaller than even the ICD-10 gold ones that eight slots were originally sized
for, so eight would be safe here and would roughly halve the wall clock. It is
not worth doing. The saving is about fifteen minutes of GPU, and the failure
mode it risks is the one that has already cost this benchmark four runs: a
prompt that overflows its slot comes back truncated, scores as an empty answer,
and is charged for in full. The committed four slot configuration is the one
with a completed run behind it.

`recompute_empty` is off, for the reason it is off everywhere: nothing is cached
for this configuration to reconsider, and on a resume it would pay to regenerate
notes the model had already genuinely declined.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from coding_bench.eval_remote import image as eval_image

RESULTS_MOUNT = "/results"
GOLD_MOUNT = "/gold"

app = modal.App("coding-bench-chain-muse-cpt-full")

results_volume = modal.Volume.from_name("coding-benchmark-results", create_if_missing=True)
gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

CHAIN: list[dict] = [
    # Concurrency 4 because that is what the server offers. Raising it past
    # PARALLEL_SLOTS does not add throughput, it just queues at the door.
    {"task": "cpt", "model": "muse-glimmer-30b-gguf", "candidate_space": "full",
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
    # A day, matching the other chains. This run should need a fraction of it,
    # but the ceiling should never be the thing that ends a run.
    timeout=86400,
    cpu=4.0,
    memory=8192,
)
def run_muse_cpt_full(chain: list[dict] | None = None, git_sha: str | None = None) -> list[dict]:
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

    print("\n=== muse cpt full chain complete ===", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


@app.local_entrypoint()
def launch():
    """Spawn the run server side, so nothing on this machine is holding it up."""
    from coding_bench.eval_remote import local_git_state

    sha, dirty = local_git_state()
    if dirty:
        print("Working tree is dirty; this run will not be exactly reproducible.")

    fn = modal.Function.from_name(app.name, "run_muse_cpt_full")
    call = fn.spawn(git_sha=sha)

    print(f"Spawned as call {call.object_id}")
    print("Expect well under an hour. Watch or collect with:")
    print(f"  modal app logs {app.name}")
    print("  modal run coding_bench/run_chain.py::fetch")
