"""
Second rating round: generate Fareez notes with a Groq-hosted model, using prompts
rebuilt by models/build_fareez_prompts.py (identical inputs to round one).

Runs the Groq calls inside Modal so that the stored Groq secret is used.
Settings: temperature 0.3 and thinking disabled (reasoning_effort="none"), as for Qwen3-32B in
round one. The output limit is 4,096 tokens (2,048 in round one), because two of 40 notes were
cut off at 2,048 in a first run, which is kept under fareez_summaries/qwen3.8-27b_max2048.

Usage:
    modal run models/run_fareez_round2_groq.py --prompts /path/prompts.json \
        --model qwen/qwen3.8-27b --short-name qwen3.8-27b [--limit 1]
"""
import json, os, re, time
from datetime import datetime
import modal

app = modal.App("fareez-round2-groq")
image = modal.Image.debian_slim(python_version="3.11").pip_install("groq>=0.30.0")


@app.function(image=image, secrets=[modal.Secret.from_name("groq-api")], timeout=900, max_containers=4, retries=2)
def generate(item: dict, model: str, max_tokens: int = 4096) -> dict:
    from groq import Groq
    key = next(v for k, v in os.environ.items() if "GROQ" in k.upper() and v)
    client = Groq(api_key=key)
    messages = [{"role": "system", "content": item["system"]}, {"role": "user", "content": item["user"]}]
    t0 = time.time()
    last = None
    for attempt in range(6):
        try:
            r = client.chat.completions.create(model=model, messages=messages, temperature=0.3,
                                               max_tokens=max_tokens, reasoning_effort="none")
            break
        except Exception as e:  # rate limits on preview models
            last = e
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(20 * (attempt + 1)); continue
            raise
    else:
        raise last
    text = r.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    return {"conversation": item["conversation"], "detected_condition": item["detected_condition"],
            "summary": text, "total_time": round(time.time() - t0, 2),
            "input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens,
            "finish_reason": r.choices[0].finish_reason, "model_returned": r.model}


@app.local_entrypoint()
def main(prompts: str, model: str = "qwen/qwen3.8-27b", short_name: str = "qwen3.8-27b",
         limit: int = 0, output_dir: str = "", dry: bool = False, max_tokens: int = 4096):
    items = json.load(open(os.path.expanduser(prompts)))
    if dry:  # check the model and parameters with a prompt that holds no study data
        items = [{"conversation": "DRY", "detected_condition": "", "system": "You are concise.",
                  "user": "Reply with the single word: ready"}]
    elif limit:
        items = items[:limit]
    results = list(generate.map(items, kwargs={"model": model, "max_tokens": max_tokens}, return_exceptions=True, wrap_returned_exceptions=False))
    ok = [r for r in results if isinstance(r, dict)]
    for r in results:
        if not isinstance(r, dict):
            print("FAILED:", repr(r)[:300])
    if dry:
        print(ok); return
    rag_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = output_dir or os.path.join(rag_root, "results", "fareez")
    sdir = os.path.join(out, "fareez_summaries", short_name); os.makedirs(sdir, exist_ok=True)
    for r in ok:
        with open(os.path.join(sdir, f"{r['conversation']}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Model: {model}\nPatient: {r['conversation']}\nDetected Disease: {r['detected_condition']}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTime: {r['total_time']:.2f}s\n")
            f.write(f"Tokens: {r['input_tokens']:,} in / {r['output_tokens']:,} out\n" + "=" * 60 + "\n\n")
            f.write(r["summary"])
    import csv
    with open(os.path.join(out, f"evaluation_results_{short_name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_name", "model", "detected_disease", "total_time_s", "input_tokens", "output_tokens",
                    "summary_chars", "summary_words", "finish_reason"])
        for r in ok:
            w.writerow([r["conversation"], short_name, r["detected_condition"], r["total_time"], r["input_tokens"],
                        r["output_tokens"], len(r["summary"]), len(r["summary"].split()), r["finish_reason"]])
    print(f"saved {len(ok)} of {len(items)} notes to {sdir}")
