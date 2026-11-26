"""Run the benchmark inside Modal, where the restricted data already is.

Tier 2 note text is never copied to a runner, a laptop, or a log. The gold
volume is mounted read only, its checksum is checked against the committed
manifest before anything is scored, and what comes back is aggregate metrics
plus predictions that carry code lists and note ids only.

    modal run coding_bench/eval_remote.py --task icd10 --model qwen3.6-27b --limit 25
    modal run coding_bench/eval_remote.py --task cpt --model muse-glimmer-30b --candidate-space full

A note on external APIs. What the code still does is record
which provider saw the text, in the run manifest, so any run can be audited
later without anyone having to remember.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "coding_bench"

GOLD_MOUNT = "/gold"

app = modal.App("coding-bench-eval")

gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow>=15.0.0",
        "numpy>=1.26",
        "groq>=0.11.0",
        "anthropic>=0.116.0",
        "modal>=1.0.0",
        "sentence-transformers>=3.0.0",
    )
    .env({"CODING_BENCH_GOLD_ROOT": GOLD_MOUNT, "PYTHONPATH": "/root"})
    .add_local_dir(
        PACKAGE_DIR,
        remote_path="/root/coding_bench",
        # The run records are deliberately NOT excluded. They are the cache
        # seed: every past prediction, keyed by everything the cache key needs,
        # already paid for. Leaving them out made `seed_from_records` dead code
        # inside Modal, so a container could only reuse work through the live
        # Dict and a fresh workspace started from nothing. They are 3MB.
        ignore=modal.FilePatternMatcher(
            "**/gold/**",
            "**/*.parquet",
            "**/__pycache__/**",
        ),
    )
)

# Which models send note text to an outside inference provider.
# Recorded per run rather than blocked, so
# that the provenance of any number is answerable after the fact.
EXTERNAL_PROVIDERS = {
    "qwen3.6-27b": "groq",
    # Was self hosted on vLLM, where the note text never left Modal. On Groq it
    # does, which is why it moves into this list rather than staying silent.
    "qwen3.8-27b": "groq",
    "gpt-oss-120b": "groq",
    # A second provider, and a second clearance question. See the governance
    # note at the top of adapters/api_anthropic.py before running this one.
    "sonnet-5": "anthropic",
}

# Which adapter file backs each model. Its hash goes into the run manifest and
# into the prediction cache key, so pointing at the wrong file would both
# misattribute a result and let an edited adapter serve stale cached answers.
ADAPTER_FILES = {
    "qwen3.6-27b": "api_groq.py",
    "gpt-oss-120b": "api_groq.py",
    "muse-glimmer-30b-gguf": "modal_muse_gguf.py",
    "muse-glimmer-30b": "modal_muse.py",
    "gemma4-26b-a4b-gguf": "modal_gemma4_gguf.py",
    "sonnet-5": "api_anthropic.py",
    # Its own file rather than a third line in api_groq.py, so that adding it
    # does not invalidate the cached notes of the two models already there. The
    # reasoning, and what that costs, is at the top of the adapter.
    "qwen3.8-27b": "api_groq_qwen38.py",
}

# Everything with an adapter that is not reached over someone else's API, which
# is to say the models whose weights run on our own GPU. Derived rather than
# listed again, so a model can never appear in one list and be forgotten in the
# other.
LOCAL_MODELS = set(ADAPTER_FILES) - set(EXTERNAL_PROVIDERS)

# Approaches that cache encoded candidates on the instance and are therefore not
# safe to share across threads. Named in one place rather than repeated at each
# call site, because the failure mode of forgetting one is a corrupted cache
# rather than an exception.
SINGLE_THREADED_APPROACHES = {"retrieval", "retr_llm", "embed_match", "entity_match"}


def local_git_state() -> tuple[str, bool]:
    """The launching machine's commit, resolved without importing the package.

    The local entrypoint runs in whatever environment the modal CLI lives in,
    which is not the benchmark's environment and generally has neither numpy nor
    pyarrow. Importing `runner` here to reuse its git helper looks tidy and
    breaks every run with a ModuleNotFoundError.
    """
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def build_predictor(
    approach: str,
    model: str,
    code_system: str,
    max_tokens: int = 16384,
    reasoning_strength: str = "medium",
):
    """Assemble an approach from its parts."""
    from coding_bench.adapters.api_anthropic import MODELS as ANTHROPIC_MODELS
    from coding_bench.adapters.api_groq import MODELS as GROQ_MODELS, GroqClient
    from coding_bench.approaches.llm import LLMPredictor
    from coding_bench.approaches.retrieval import RetrievalPredictor
    from coding_bench.approaches.retr_llm import RetrievalThenLLM

    if approach == "retrieval":
        return RetrievalPredictor(top_k=10)
    if approach == "embed_match":
        from coding_bench.approaches.embed_match import EmbedMatchPredictor

        return EmbedMatchPredictor()
    if approach == "entity_match":
        from coding_bench.approaches.entity_match import EntityMatchPredictor

        return EntityMatchPredictor()

    if model in GROQ_MODELS:
        client = GroqClient(model_id=GROQ_MODELS[model])
    elif model in ANTHROPIC_MODELS:
        from coding_bench.adapters.api_anthropic import AnthropicClient

        # No temperature: Claude Sonnet 5 rejects one. Reasoning strength is
        # passed through because it maps onto Anthropic's effort, so the run
        # parameter keeps steering what it claims to steer.
        client = AnthropicClient(
            model_id=ANTHROPIC_MODELS[model], reasoning_strength=reasoning_strength
        )
    elif model == "muse-glimmer-30b-gguf":
        from coding_bench.adapters.modal_muse_gguf import MuseGGUFClient

        # Quantised weights, named as such in the run manifest so this can never
        # be tabulated next to a bf16 number as though they were one model.
        client = MuseGGUFClient(reasoning_strength=reasoning_strength)
    elif model == "muse-glimmer-30b":
        from coding_bench.adapters.modal_muse import MuseClient

        # Reasoning strength drives generation length, and generation length is
        # the entire wall clock of a self hosted 30B on one GPU. At "high" this
        # model runs its trace out to the token cap on clinical notes, which is
        # minutes per note. It is a recorded run parameter, not a hidden default.
        client = MuseClient(reasoning_strength=reasoning_strength)
    elif model == "gemma4-26b-a4b-gguf":
        from coding_bench.adapters.modal_gemma4_gguf import Gemma4GGUFClient

        # Quantised, and named as such in the run manifest, for the same reason
        # the Muse GGUF is: a QAT 4 bit build is not the released model.
        client = Gemma4GGUFClient(reasoning_strength=reasoning_strength)
    elif model == "qwen3.8-27b":
        from coding_bench.adapters.api_groq_qwen38 import qwen38_client

        # reasoning_strength has to be passed, and that is the whole lesson of
        # the discarded 2026-09-07 runs. Unset, this endpoint does not think,
        # while the qwen3.6 endpoint it is compared against does, so leaving it
        # out measures an instruct model against a reasoning one and calls the
        # difference a model difference. The adapter maps the strength onto
        # Groq's reasoning_effort; see its header for why `high` is not it.
        client = qwen38_client(reasoning_strength=reasoning_strength)
    else:
        raise ValueError(
            f"Unknown model {model!r}. Known: "
            # ADAPTER_FILES rather than the three lists it is derived from: a
            # model reached over HTTP from its own adapter file, as qwen3.8-27b
            # is, belongs to none of them and would be missing from this list at
            # exactly the moment someone is reading it to find the right name.
            f"{sorted(ADAPTER_FILES)}"
        )

    llm = LLMPredictor(client=client, code_system=code_system, max_tokens=max_tokens)
    if approach == "llm":
        return llm
    if approach == "retr_llm":
        return RetrievalThenLLM(retriever=RetrievalPredictor(), llm=llm, shortlist=50)
    raise ValueError(f"Unknown approach {approach!r}")


@app.function(
    image=image,
    volumes={GOLD_MOUNT: gold_volume},
    # Both provider secrets. Modal resolves every secret in an app at startup,
    # so `anthropic-api` must exist before this is deployed or every run breaks,
    # including the ones that never touch Anthropic:
    #     modal secret create anthropic-api ANTHROPIC_API_KEY=<key>
    secrets=[
        modal.Secret.from_name("groq-api-key"),
        modal.Secret.from_name("anthropic-api"),
    ],
    timeout=14400,
    memory=8192,
    cpu=4.0,
)
def evaluate(
    task: str = "icd10",
    approach: str = "llm",
    model: str = "qwen3.6-27b",
    candidate_space: str = "gold",
    limit: int | None = None,
    max_tokens: int = 16384,
    concurrency: int = 1,
    reasoning_strength: str = "medium",
    cache: str = "auto",
    git_sha: str | None = None,
    recompute_empty: bool = False,
) -> dict:
    """Score one approach against one task. Returns a text free run record."""
    from coding_bench.bench import loaders, runner

    provider = EXTERNAL_PROVIDERS.get(model)
    if provider and task in ("cpt", "icd10"):
        # States what is happening, not that it is allowed. The old wording
        # asserted the provider was "cleared for MIMIC derived data" for every
        # entry in the table, which was true of the one provider in it when it
        # was written and became a claim the code cannot check the moment a
        # second was added. Clearance is a fact about an agreement; a run can
        # record which provider saw the text and nothing more.
        print(
            f"Note text for {task} is being sent to {provider}. Recorded in the run "
            f"manifest as external_provider so this stays answerable later.",
            flush=True,
        )

    if task in ("cpt", "icd10"):
        stamp = loaders.verify_volume(Path(GOLD_MOUNT))
        print(f"Volume verified against the committed manifests: {stamp}", flush=True)

    dataset = loaders.load(task)
    print(f"Loaded {len(dataset)} notes for {task} ({dataset.code_system})", flush=True)

    space: int | str | None = candidate_space
    if candidate_space == "full":
        space = None
    elif candidate_space.isdigit():
        space = int(candidate_space)

    predictor = build_predictor(
        approach, model, dataset.code_system, max_tokens, reasoning_strength
    )
    adapter_path = Path("/root/coding_bench/adapters") / ADAPTER_FILES.get(model, "api_groq.py")
    prompt_text = None
    if hasattr(predictor, "build_prompt"):
        prompt_text = predictor.build_prompt(dataset.notes[0], [])[0]

    record = runner.run(
        predictor,
        dataset,
        candidate_space=space,
        limit=limit,
        parameters={"max_tokens": max_tokens, "approach": approach, "concurrency": concurrency,
                    "reasoning_strength": reasoning_strength, "recompute_empty": recompute_empty},
        adapter_path=adapter_path,
        prompt_text=prompt_text,
        external_provider=provider,
        # The retriever caches encoded candidates on the instance, so it is not
        # safe to share across threads.
        concurrency=1 if approach in SINGLE_THREADED_APPROACHES else concurrency,
        cache=cache,
        git_sha=git_sha,
        recompute_empty=recompute_empty,
    )
    print(runner.summarise(record), flush=True)
    return record


@app.local_entrypoint()
def main(
    task: str = "icd10",
    approach: str = "llm",
    model: str = "qwen3.6-27b",
    candidate_space: str = "gold",
    limit: int = 0,
    max_tokens: int = 16384,
    concurrency: int = 1,
    reasoning_strength: str = "medium",
    cache: str = "auto",
    recompute_empty: bool = False,
):
    sha, dirty = local_git_state()
    if dirty:
        print("Working tree is dirty; this run will not be exactly reproducible.")
    record = evaluate.remote(
        task=task,
        approach=approach,
        model=model,
        candidate_space=candidate_space,
        limit=limit or None,
        max_tokens=max_tokens,
        concurrency=concurrency,
        reasoning_strength=reasoning_strength,
        cache=cache,
        git_sha=sha,
        recompute_empty=recompute_empty,
    )

    runs_dir = PACKAGE_DIR / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record['manifest']['run_id']}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf8")
    print(f"\nRun record written to {path.relative_to(REPO_ROOT)}")
