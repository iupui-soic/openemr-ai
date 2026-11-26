"""Show what Muse actually says on a CPT note, and where the answer is lost.

Muse scored 0.333 on CPT gold while gpt-oss and qwen scored 0.92 on the same
notes, and it returned nothing at all on 120 of 150. Those empties are not
refusals and not timeouts: no error, no truncation, and 624 completion tokens on
average, so the model wrote most of a page and the harness extracted no code
from it. Either it is not emitting the JSON the parser looks for, or it is
emitting something the parser rejects. Only the raw text can say which.

    modal run coding_bench/scripts/muse_cpt_probe.py

Prints, per note, the raw completion and then each stage of parsing, so the
exact point of loss is visible rather than inferred.
"""

from __future__ import annotations

import modal

from coding_bench.eval_remote import image as eval_image

GOLD_MOUNT = "/gold"

app = modal.App("muse-cpt-probe")

gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

# Three that came back empty, and one that worked, so the difference between
# them is visible side by side rather than only the failures.
EMPTY_NOTES = ["35990", "60755", "81467"]
WORKING_NOTE = "76207"


@app.function(
    image=eval_image,
    volumes={GOLD_MOUNT: gold_volume},
    secrets=[modal.Secret.from_name("groq-api-key")],
    timeout=3600,
)
def probe(note_ids: list[str]) -> None:
    from pathlib import Path

    from coding_bench.approaches.llm import extract_json, parse_codes
    from coding_bench.bench import loaders, runner
    from coding_bench.eval_remote import build_predictor

    loaders.verify_volume(Path(GOLD_MOUNT))
    dataset = loaders.load("cpt")
    by_id = {note.note_id: note for note in dataset.notes}

    # Built the same way the run built it, rather than assembled by hand here,
    # so this reproduces the failing call and not a friendlier version of it.
    predictor = build_predictor("llm", "muse-glimmer-30b-gguf", dataset.code_system, 16384, "medium")
    client = predictor.client

    for note_id in note_ids:
        note = by_id[note_id]
        candidates = runner.build_candidates(note, dataset.label_space, "gold")
        system, user = predictor.build_prompt(note, candidates)
        offered = {c.code for c in candidates}

        print("=" * 78, flush=True)
        print(f"note {note_id}  gold={note.gold_codes}  offered={sorted(offered)}", flush=True)
        print("=" * 78, flush=True)

        try:
            completion = client.complete(system, user, 16384)
        except Exception as exc:  # noqa: BLE001
            print(f"  RAISED {type(exc).__name__}: {exc}", flush=True)
            continue

        text = completion.text
        print(f"--- raw completion ({len(text)} chars, "
              f"{completion.usage.get('completion_tokens')} tokens, "
              f"stop={completion.stop_reason}) ---", flush=True)
        print(text, flush=True)

        print("--- parsing ---", flush=True)
        payload = extract_json(text)
        print(f"  extract_json -> {payload!r}", flush=True)
        print(f"  parse_codes(restricted) -> {parse_codes(text, note.text, offered)}", flush=True)
        print(f"  parse_codes(unrestricted) -> {parse_codes(text, note.text, None)}", flush=True)
        print(flush=True)


@app.local_entrypoint()
def main(notes: str = ""):
    ids = notes.split(",") if notes else EMPTY_NOTES + [WORKING_NOTE]
    probe.remote(ids)
