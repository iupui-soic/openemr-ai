"""Show what Groq actually returns on the notes qwen came back silent on.

The channel bug was found on the two llama.cpp models and fixed there. The
question this answers is whether the Groq adapter has it too, because it reads
`choice.message.content` the same way and its runs are not obviously clean:

    qwen  icd10 full  note 716852   11,468 completion tokens, 7 gold codes, no prediction
    qwen  icd10 full  note 717377    9,384 completion tokens, 5 gold codes, no prediction
    qwen  cpt   full  note 1197953   8,656 completion tokens, 1 gold code,  no prediction

A model does not spend eleven thousand tokens deciding that nothing applies.
Against that, gpt-oss's silent notes average 193 to 269 tokens, which is what an
honest empty answer looks like, so the two models may not be doing the same
thing and the record cannot tell them apart. Only the raw response can.

    modal run coding_bench/scripts/groq_silence_probe.py
    modal run coding_bench/scripts/groq_silence_probe.py --task cpt --candidate-space full --notes 1197953

This sends note text to Groq, exactly as the runs being investigated did, and
for the same reason. Nothing restricted is printed: the probe reports field
names, lengths and parse results, never the note or the response body.

## What it found

**Not the channel bug.** Groq returns one string channel, `content`, for both
models. There is no `reasoning_content` to lose an answer in, so the fix applied
to the llama.cpp adapters was never the problem here.

**Two different things are being called silence.** Note 250188 emitted
`{"codes": []}` nine times over 4,462 tokens. That is the model genuinely
declining to code, it parses correctly, and scoring it as an empty prediction is
right. Notes 716852 and 717377 are not that:

    note 716852  31,478 chars  57 braces open, 57 close, final depth 0
                 1,399 quote characters, 0 of them escaped
                 balanced_objects finds 0 objects, extract_json returns None
                 34 of the offered codes appear in the text, including the gold ones

The braces balance perfectly, so nothing was cut off. The quote count is odd and
none are escaped, which is the whole story: the prompt asks for a verbatim quote
from the note beside every code, clinical notes contain quote marks, and qwen
copies them in without escaping. One unescaped quote inside a JSON string flips
the parser's in-string state, every brace after it is counted in the wrong
state, no `{...}` span ever closes at depth 0, and a complete answer naming the
right codes is discarded as if the model had said nothing.

This is a parser problem, not an adapter problem. It is fixed in
`approaches/llm.py::salvage_codes`, and re-running every empty note across all
eleven leaderboard runs recovered 5 of them, 3 here and 2 in qwen icd10 gold.
Both notes above are among them.

Five, not the sixteen this run's silence count suggested. The rest of the notes
that came back empty either answered normally when asked again, or were the
model genuinely declining. Counting empty predictions tells you where to look,
and nothing more than that.
"""

from __future__ import annotations

import modal

from coding_bench.eval_remote import image as eval_image

GOLD_MOUNT = "/gold"

app = modal.App("groq-silence-probe")

gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

# Silent in the 578 note icd10 full run, ordered by how much the model wrote
# before saying nothing. The last is a note it answered normally, kept as a
# control so the difference is visible side by side rather than inferred.
SILENT_ICD10 = ["716852", "717377", "36738", "250188"]
# Answered normally in the same run, 5 gold codes and 6 predicted.
CONTROL_ICD10 = "1237"


@app.function(
    image=eval_image,
    volumes={GOLD_MOUNT: gold_volume},
    secrets=[modal.Secret.from_name("groq-api-key")],
    timeout=3600,
)
def probe(
    note_ids: list[str],
    task: str = "icd10",
    candidate_space: str = "full",
    model: str = "qwen3.6-27b",
    max_tokens: int = 16384,
) -> None:
    import time
    from pathlib import Path

    from coding_bench.approaches.llm import extract_json, parse_codes
    from coding_bench.bench import loaders, runner
    from coding_bench.eval_remote import build_predictor

    def _create_with_retry(client, **kwargs):
        from groq import RateLimitError

        for attempt in range(8):
            try:
                return client._client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                # Groq states how long to wait. Guessing takes longer.
                wait = 2**attempt
                header = getattr(getattr(exc, "response", None), "headers", {}) or {}
                try:
                    wait = max(wait, float(header.get("retry-after", 0)))
                except (TypeError, ValueError):
                    pass
                print(f"    rate limited, waiting {min(wait, 60):.0f}s", flush=True)
                time.sleep(min(wait, 60))
        raise RuntimeError("still rate limited after 8 attempts")

    loaders.verify_volume(Path(GOLD_MOUNT))
    dataset = loaders.load(task)
    by_id = {note.note_id: note for note in dataset.notes}

    space: int | str | None = None if candidate_space == "full" else candidate_space

    # Built the same way the run built it, so this reproduces the failing call
    # rather than a friendlier version of it.
    predictor = build_predictor("llm", model, dataset.code_system, max_tokens, "medium")
    client = predictor.client

    for note_id in note_ids:
        note = by_id.get(note_id)
        if note is None:
            print(f"note {note_id} is not in this dataset", flush=True)
            continue

        candidates = runner.build_candidates(note, dataset.label_space, space)
        system, user = predictor.build_prompt(note, candidates)
        offered = {c.code for c in candidates}

        print("=" * 78, flush=True)
        print(f"note {note_id}  gold={sorted(note.gold_codes)}  offered={len(offered)}", flush=True)

        # The raw SDK call, not client.complete, because the whole question is
        # what the adapter is dropping on its way to a Completion. Going around
        # the adapter also goes around its retry loop, and Groq bills the
        # *requested* max_tokens against the output ceiling, so a 16k request
        # exhausts a 32k per minute allowance in two notes. Hence both the
        # retry and the wait between notes.
        response = _create_with_retry(
            client,
            model=client.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=client.temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        message = choice.message

        fields = {}
        for name in ("content", "reasoning", "reasoning_content"):
            value = getattr(message, name, None)
            if value is not None:
                fields[name] = len(value)
        extra = getattr(message, "model_extra", None) or {}
        for name, value in extra.items():
            if isinstance(value, str) and name not in fields:
                fields[name] = len(value)

        print(f"  finish_reason={choice.finish_reason} "
              f"completion_tokens={getattr(response.usage, 'completion_tokens', None)}", flush=True)
        print(f"  string channels (name: chars) = {fields}", flush=True)

        for name in fields:
            text = getattr(message, name, None) or extra.get(name) or ""
            payload = extract_json(text)
            codes = parse_codes(text, note.text, offered)
            print(f"  {name}: extract_json -> "
                  f"{'None' if payload is None else sorted(payload)[:4]}, "
                  f"parse_codes -> {sorted(codes)}", flush=True)
            if codes:
                continue

            # Why nothing came out. Everything reported here is a count, a
            # length, a parser error or a code string, so no note text and no
            # response body can reach the log.
            import json as _json

            from coding_bench.approaches.llm import _FENCE, balanced_objects

            fenced = _FENCE.findall(text)
            objects = balanced_objects(text)
            print(f"    {len(fenced)} fenced block(s), {len(objects)} balanced object(s), "
                  f"'\"codes\"' appears {text.count('\"codes\"')} time(s)", flush=True)
            for index, chunk in enumerate(objects[-3:], start=max(1, len(objects) - 2)):
                try:
                    _json.loads(chunk)
                    verdict = "parses"
                except ValueError as exc:
                    verdict = f"{type(exc).__name__}: {exc}"
                print(f"    object {index} of {len(objects)}, {len(chunk)} chars: {verdict}",
                      flush=True)
            mentioned = sorted(code for code in offered if code in text)
            print(f"    {len(mentioned)} offered code(s) appear in the text: {mentioned[:8]}",
                  flush=True)

            # Zero balanced objects while '"codes"' is present means the brace
            # scan never came back to depth 0. Either the braces really are
            # unbalanced, or a quote inside a string desynchronised the string
            # tracking and every brace after it was counted in the wrong state.
            depth = 0
            lowest = 0
            in_string = False
            escaped = False
            unescaped_quotes_in_string = 0
            for char in text:
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    lowest = min(lowest, depth)
            print(f"    braces: {text.count('{')} open, {text.count('}')} close, "
                  f"final depth {depth}, lowest depth {lowest}, "
                  f"ends inside a string: {in_string}", flush=True)
            print(f"    quotes: {text.count(chr(34))} total, "
                  f"{text.count(chr(92) + chr(34))} escaped", flush=True)
        print(flush=True)
        # Stay under the output tokens per minute ceiling rather than spending
        # the next note's first attempt discovering it.
        time.sleep(35)


@app.local_entrypoint()
def main(
    notes: str = "",
    task: str = "icd10",
    candidate_space: str = "full",
    model: str = "qwen3.6-27b",
):
    ids = notes.split(",") if notes else SILENT_ICD10 + [CONTROL_ICD10]
    probe.remote(ids, task=task, candidate_space=candidate_space, model=model)
