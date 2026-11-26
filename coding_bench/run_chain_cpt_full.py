"""CPT at the full catalogue, for the two Groq models.

CPT has only ever been measured at gold candidates, and on this dataset that
condition is close to degenerate: notes carry 1.04 gold codes on average, so the
model is handed a single candidate and asked whether to emit it. That is what
the 0.92 and 0.93 micro F1 actually measure, and it is why CPT looks easier than
ICD-10 rather than merely different. It also means CPT contributes nothing to
the gold to full collapse this benchmark exists to quantify.

The full catalogue is 61 codes against ICD-10's 673, so the prompts stay small
and these two runs are minutes and pennies rather than hours and dollars.

    modal deploy coding_bench/run_chain_cpt_full.py
    modal run coding_bench/run_chain_cpt_full.py::launch

Why this waits, and why it does not wait for the whole Groq queue. Groq's real
ceiling is tokens per minute, somewhere near 250k, and concurrency past two or
three at full catalogue prompt sizes has already produced a 74% failure rate
once. The qwen icd10/full step running now is 578 notes of 12k token prompts and
its 200 note predecessor already scored a 4.0% error rate against a 5%
quarantine ceiling, so stealing throughput from it could invalidate forty
minutes of work. But that queue ends with a GPU step that touches Groq not at
all, and waiting through it would idle these runs for half an hour for nothing.
So the gate is the arrival of the qwen full record on the results volume, which
is precisely the moment Groq goes quiet.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from coding_bench.eval_remote import image as eval_image

RESULTS_MOUNT = "/results"
GOLD_MOUNT = "/gold"

app = modal.App("coding-bench-chain-cptfull")

results_volume = modal.Volume.from_name("coding-benchmark-results", create_if_missing=True)
gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

# The record whose appearance means the Groq half of the other queue is done.
GATE_PREFIX = "icd10__llm__qwen-qwen3.6-27b__candfull__"

CHAIN: list[dict] = [
    # Concurrency 3 matches the gold CPT run that scored 0% errors. The full
    # catalogue adds about 900 tokens of code list per prompt, which is small
    # enough not to change the arithmetic.
    {"task": "cpt", "model": "gpt-oss-120b", "candidate_space": "full", "concurrency": 3},
    # qwen generates noticeably more per note than gpt-oss, 4.4k against 1.9k at
    # ICD-10 full, so it gets the same 2 its other runs use.
    {"task": "cpt", "model": "qwen3.6-27b", "candidate_space": "full", "concurrency": 2},
]

DEFAULTS = {
    "approach": "llm",
    "max_tokens": 16384,
    "reasoning_strength": "medium",
    "cache": "auto",
}


def _wait_for_gate(prefix: str, poll_s: int = 60, max_wait_s: int = 21600) -> None:
    """Block until a record with this prefix lands on the results volume.

    `reload()` on every pass is the whole trick. A mounted volume shows the
    container the state it had when it was mounted, so without the reload this
    would poll a frozen snapshot and wait out the full six hours no matter what
    the other queue did.
    """
    deadline = time.time() + max_wait_s
    print(f"waiting for a {prefix}* record before using Groq", flush=True)

    while time.time() < deadline:
        results_volume.reload()
        hits = sorted(p.name for p in Path(RESULTS_MOUNT).glob(f"{prefix}*"))
        if hits:
            print(f"gate opened by {hits[-1]}", flush=True)
            return
        time.sleep(poll_s)

    print(f"gate never opened after {max_wait_s}s, starting anyway", flush=True)


@app.function(
    image=eval_image,
    volumes={RESULTS_MOUNT: results_volume, GOLD_MOUNT: gold_volume},
    secrets=[modal.Secret.from_name("groq-api-key")],
    timeout=86400,
    cpu=4.0,
    memory=8192,
)
def run_cpt_full_chain(
    chain: list[dict] | None = None,
    git_sha: str | None = None,
    gate_prefix: str | None = GATE_PREFIX,
) -> list[dict]:
    """Run the CPT full catalogue queue once Groq is free."""
    from coding_bench.run_chain import _evaluate_inline

    if gate_prefix:
        _wait_for_gate(gate_prefix)

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

    print("\n=== cpt full chain complete ===", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


@app.local_entrypoint()
def launch(gate: str = GATE_PREFIX):
    """Spawn the queue server side. See `run_chain.py::launch` for why not --detach."""
    from coding_bench.eval_remote import local_git_state

    sha, dirty = local_git_state()
    if dirty:
        print("Working tree is dirty; these runs will not be exactly reproducible.")

    print(f"Queueing {len(CHAIN)} CPT full catalogue runs:")
    for step in CHAIN:
        print(f"  {step['task']:6} {step['model']:16} {step['candidate_space']:5} conc={step['concurrency']}")
    if gate:
        print(f"Holding until a {gate}* record appears.")

    fn = modal.Function.from_name(app.name, "run_cpt_full_chain")
    call = fn.spawn(git_sha=sha, gate_prefix=gate or None)

    print(f"\nSpawned as call {call.object_id}")
    print("Watch or collect with:")
    print(f"  modal app logs {app.name}")
    print("  modal run coding_bench/run_chain.py::fetch")
