"""Run a queue of evaluations entirely inside Modal, detached from this machine.

The point is survivability. A normal `modal run` is driven by the local process:
close the laptop, lose the network, or cut the power, and the ephemeral app dies
with it, and the run record, which the local entrypoint writes, is never
written at all. That happened once already and cost a nearly complete run.

Here the orchestrator is itself a Modal function. Launched detached it keeps
running in Modal's cloud with nothing depending on this machine, and each step
writes its record to a volume the moment it finishes, so an interruption costs
at most the step in flight rather than the whole queue.

    modal deploy coding_bench/adapters/modal_muse_gguf.py    # only if Muse is queued
    modal deploy coding_bench/run_chain.py                   # register the chain itself
    modal run coding_bench/run_chain.py::launch              # spawn it, then walk away

    # later, from anywhere, pull the finished records into the repo
    modal run coding_bench/run_chain.py::fetch

Watch it with `modal app logs coding-bench-chain`. Because every step is cache
seeded, relaunching the same queue after a failure re-runs only what is missing.

Two caveats, both learned by testing rather than by reading docs.

Use `launch`, not `modal run --detach ...::main`. Detach keeps the app alive
after the client exits, but it does not stop the client's death from cancelling
the input already running: killing the launcher logged "Received a cancellation
signal" about ninety seconds later and lost that step at note 100 of 312. A
spawn against the deployed function is created server side and belongs to the
deployed app, so there is no client whose death can cancel anything.

Scoring happens inline here rather than by calling the deployed evaluator,
because a cross app .remote() needs a live client and raises ClientClosed once
the launching session is gone. The one step that still needs a cross app call is
Muse, which talks to the GGUF server, and it is queued last for that reason.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

from coding_bench.eval_remote import image as eval_image

# The evaluator lives on its own app and is reached by name at call time, not
# imported. Importing the function object gives an unhydrated handle unless that
# app happens to be running, which for a detached chain it is not; looking it up
# by name binds to the deployed version instead. It also keeps eval_remote's
# entrypoints out of this file's namespace, which otherwise makes `modal run`
# ambiguous about what to launch.
EVAL_APP = "coding-bench-eval"

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_MOUNT = "/results"
GOLD_MOUNT = "/gold"

app = modal.App("coding-bench-chain")

# Records land here rather than on the caller's disk, so they survive the
# caller going away. This is the whole reason the chain exists.
results_volume = modal.Volume.from_name("coding-benchmark-results", create_if_missing=True)
gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

# The queue. Each entry is one call to evaluate(). Ordered cheapest and most
# valuable first, so an interruption loses the least.
# Ordered by durability, not by interest. Everything reachable over plain HTTP
# runs first, because those steps need nothing from Modal once the container is
# up and so survive the launching client disappearing. The Muse step is last
# because it talks to the GGUF server on another Modal app, and that cross app
# call needs a live client: if the session that launched this queue is gone, the
# Groq steps will already have banked their records and only Muse will fail.
CHAIN: list[dict] = [
    # CPT has never been measured in this package, and its prompts are small:
    # 61 codes against ICD-10's 673, so these are quick and cheap.
    {"task": "cpt", "model": "gpt-oss-120b", "candidate_space": "gold", "concurrency": 3},
    {"task": "cpt", "model": "qwen3.6-27b", "candidate_space": "gold", "concurrency": 2},
    # Completes the scaling stress: qwen currently has only 200 notes at full
    # catalogue against gpt-oss's 578.
    {"task": "icd10", "model": "qwen3.6-27b", "candidate_space": "full", "concurrency": 2},
    # Least durable, so last. Needs the GGUF server deployed and a live client.
    {"task": "cpt", "model": "muse-glimmer-30b-gguf", "candidate_space": "gold",
     "concurrency": 8, "limit": 150},
]

DEFAULTS = {
    "approach": "llm",
    "max_tokens": 16384,
    "reasoning_strength": "medium",
    "cache": "auto",
}


def _evaluate_inline(config: dict) -> dict:
    """Score one configuration inside this container, with no Modal RPC.

    A near copy of eval_remote.evaluate's body. Duplicated deliberately: the
    alternative is calling that function remotely, which is precisely the
    dependency that made detached runs fail.
    """
    from pathlib import Path as _Path

    from coding_bench.eval_remote import (
        ADAPTER_FILES,
        EXTERNAL_PROVIDERS,
        SINGLE_THREADED_APPROACHES,
        build_predictor,
    )
    from coding_bench.bench import loaders, runner

    task = config["task"]
    provider = EXTERNAL_PROVIDERS.get(config["model"])

    if task in ("cpt", "icd10"):
        stamp = loaders.verify_volume(_Path(GOLD_MOUNT))
        print(f"  volume verified: {stamp}", flush=True)

    dataset = loaders.load(task)
    print(f"  loaded {len(dataset)} notes for {task} ({dataset.code_system})", flush=True)

    raw_space = config["candidate_space"]
    space: int | str | None = raw_space
    if raw_space == "full":
        space = None
    elif str(raw_space).isdigit():
        space = int(raw_space)

    predictor = build_predictor(
        config["approach"], config["model"], dataset.code_system,
        config["max_tokens"], config["reasoning_strength"],
    )
    adapter_path = _Path("/root/coding_bench/adapters") / ADAPTER_FILES.get(
        config["model"], "api_groq.py"
    )
    prompt_text = None
    if hasattr(predictor, "build_prompt"):
        prompt_text = predictor.build_prompt(dataset.notes[0], [])[0]

    record = runner.run(
        predictor,
        dataset,
        candidate_space=space,
        limit=config.get("limit"),
        parameters={
            "max_tokens": config["max_tokens"],
            "approach": config["approach"],
            "concurrency": config.get("concurrency", 1),
            "reasoning_strength": config["reasoning_strength"],
            "recompute_empty": config.get("recompute_empty", False),
        },
        adapter_path=adapter_path,
        prompt_text=prompt_text,
        external_provider=provider,
        concurrency=1 if config["approach"] in SINGLE_THREADED_APPROACHES else config.get("concurrency", 1),
        cache=config["cache"],
        git_sha=config.get("git_sha"),
        recompute_empty=config.get("recompute_empty", False),
    )
    print(runner.summarise(record), flush=True)
    return record


@app.function(
    image=eval_image,
    # The chain now does the scoring itself, so it needs the gold volume and the
    # provider secret that the evaluator used to hold.
    volumes={RESULTS_MOUNT: results_volume, GOLD_MOUNT: gold_volume},
    secrets=[modal.Secret.from_name("groq-api-key")],
    # Long enough for a queue of full length runs. Each step commits its own
    # record, so hitting this ceiling loses one step, not the queue.
    timeout=86400,
    cpu=4.0,
    memory=8192,
)
def run_chain(chain: list[dict] | None = None, git_sha: str | None = None) -> list[dict]:
    """Execute each configuration in turn, in this container, persisting as we go.

    Evaluations run inline rather than through evaluate.remote(). That is the
    whole point: a cross app call needs a live Modal client, and the client a
    detached run inherits dies with the session that launched it. The first
    version of this file delegated, survived exactly as long as the step already
    in flight, and then failed every remaining step with ClientClosed. Running
    the work here means the container needs no outbound Modal calls at all.
    """
    queue = chain if chain is not None else CHAIN
    summaries = []

    for index, step in enumerate(queue, start=1):
        config = {**DEFAULTS, **step, "git_sha": git_sha}
        label = f"{config['task']}/{config['model']}/{config['candidate_space']}"
        print(f"\n[{index}/{len(queue)}] starting {label}", flush=True)

        try:
            record = _evaluate_inline(config)
        except Exception as exc:  # noqa: BLE001
            # One failing configuration must not abandon the rest of the queue.
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

    print("\n=== chain complete ===", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


@app.function(image=eval_image, volumes={RESULTS_MOUNT: results_volume}, timeout=900)
def list_results() -> list[str]:
    return sorted(p.name for p in Path(RESULTS_MOUNT).glob("*.json"))


@app.function(image=eval_image, volumes={RESULTS_MOUNT: results_volume}, timeout=900)
def read_result(name: str) -> dict:
    return json.loads((Path(RESULTS_MOUNT) / name).read_text(encoding="utf8"))


@app.local_entrypoint()
def main():
    """Launch the queue. Use `modal run --detach` so it outlives this shell."""
    from coding_bench.eval_remote import local_git_state

    sha, dirty = local_git_state()
    if dirty:
        print("Working tree is dirty; these runs will not be exactly reproducible.")

    print(f"Queueing {len(CHAIN)} runs:")
    for step in CHAIN:
        limit = step.get("limit", "all")
        print(f"  {step['task']:6} {step['model']:24} {step['candidate_space']:5} n={limit}")
    print("\nDetached runs survive this machine. Watch with:")
    print("  modal app logs coding-bench-chain")

    summaries = run_chain.remote(git_sha=sha)
    print(json.dumps(summaries, indent=2))


@app.local_entrypoint()
def launch():
    """Spawn the queue with nothing at all attached to this machine.

    Prefer this over `modal run --detach`. Detach keeps the *app* alive when the
    client goes away, but it does not stop the client's death from cancelling
    the input already in flight: killing the launcher produced "Received a
    cancellation signal" and lost the step mid way through note 100 of 312.

    A spawn against the deployed function is a different thing entirely. The
    call is created server side and belongs to the deployed app, so there is no
    client to lose. Deploy first, then spawn:

        modal deploy coding_bench/run_chain.py
        modal run coding_bench/run_chain.py::launch
    """
    from coding_bench.eval_remote import local_git_state

    sha, dirty = local_git_state()
    if dirty:
        print("Working tree is dirty; these runs will not be exactly reproducible.")

    # Bind to the deployed function by name. Calling the local `run_chain`
    # object would spawn inside this ephemeral `modal run` app, which is torn
    # down when this entrypoint returns, defeating the whole purpose.
    fn = modal.Function.from_name(app.name, "run_chain")
    call = fn.spawn(git_sha=sha)

    print(f"Spawned {len(CHAIN)} runs as call {call.object_id}")
    print("Nothing on this machine is holding it up. Watch or collect with:")
    print(f"  modal app logs {app.name}")
    print("  modal run coding_bench/run_chain.py::fetch")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT)


def _sha_is_reachable(sha: str | None) -> bool:
    """True if this commit still exists and is an ancestor of HEAD.

    Existence alone is not enough. A rewritten commit stays in the object
    database as long as some other ref is holding it, so a record can look fine
    right up until the backup branch is deleted and the id evaporates. Ancestry
    is the question actually worth asking: is this commit part of the history
    this checkout is on.
    """
    if not sha or sha == "unknown":
        return True
    if _git("git", "cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        return False
    return _git("git", "merge-base", "--is-ancestor", sha, "HEAD").returncode == 0


@app.local_entrypoint()
def fetch(repoint: bool = False):
    """Copy finished records from the volume into the repo.

    A queue is spawned with the commit id that was checked out at launch, and it
    stamps that id on every record it produces, for hours afterwards. Rewrite the
    branch in the meantime, by squashing or by stripping something out of
    history, and every one of those ids becomes a reference to a commit that no
    longer exists. The runs are still perfectly valid, but the one field that
    says which code produced them now points at nothing.

    So this checks, every time, and says so. Pass `--repoint` to move the dead
    ids onto the current HEAD. That is a real loss of precision and worth
    understanding before reaching for it: several distinct code states can
    collapse onto one commit, and git_sha stops distinguishing them. What does
    survive any rewrite is the content hashes, since adapter_sha256,
    prompt_sha256 and dataset_manifest_sha256 digest the bytes rather than name
    a commit. A record whose adapter_sha256 disagrees with the adapter at the
    commit it names is telling you the code moved after the run, which is the
    question git_sha was there to answer in the first place.
    """
    runs_dir = REPO_ROOT / "coding_bench" / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    names = list_results.remote()
    if not names:
        print("No records on the volume yet.")
        return

    for name in names:
        target = runs_dir / name
        if target.exists():
            print(f"  skip {name} (already here)")
            continue
        record = read_result.remote(name)
        target.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf8")
        print(f"  wrote {name}")

    # Every record on disk, not just the ones just written. A rewrite invalidates
    # what was already sitting in the repo just as thoroughly as what arrived now.
    stale = []
    for path in sorted(runs_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf8"))
        sha = record["manifest"].get("git_sha")
        if not _sha_is_reachable(sha):
            stale.append((path, record, sha))

    if not stale:
        print(f"\n{len(names)} record(s) on the volume, all naming a reachable commit.")
    elif repoint:
        head = _git("git", "rev-parse", "HEAD").stdout.strip()
        for path, record, sha in stale:
            record["manifest"]["git_sha"] = head
            path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf8")
            print(f"  repointed {path.name}: {sha[:12]} -> {head[:12]}")
        print(f"\n{len(stale)} record(s) repointed at HEAD.")
    else:
        print(f"\n{len(stale)} record(s) name a commit that is not in this history:")
        for path, _record, sha in stale:
            print(f"  {sha[:12]}  {path.name}")
        print("Rerun with --repoint to move them onto HEAD, having read why that costs something:")
        print("  modal run coding_bench/run_chain.py::fetch --repoint")

    print("\nRegenerate the leaderboard with:")
    print("  python -m coding_bench.bench.reporting")
