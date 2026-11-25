"""
Second rating round: generate Fareez notes with a model on Amazon Bedrock (Converse API),
using prompts rebuilt by models/build_fareez_prompts.py (identical inputs to round one).

Defaults target LLaMA-4-Scout through the US inference profile. Settings: temperature 0.3,
at most 4,096 output tokens. Needs AWS credentials with bedrock:InvokeModel in a US region
and model access enabled. Confirm that model invocation logging is off before sending charts.

Usage:
    python models/run_fareez_round2_bedrock.py --prompts /path/prompts.json [--dry] [--limit 1]
"""
import argparse, csv, json, os, time
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", default="us.meta.llama4-scout-17b-instruct-v1:0")
    ap.add_argument("--short-name", default="llama-4-scout")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="one call with no study data, to check access")
    a = ap.parse_args()
    import boto3
    from botocore.config import Config
    br = boto3.client("bedrock-runtime", region_name=a.region,
                      config=Config(read_timeout=600, retries={"max_attempts": 8, "mode": "adaptive"}))
    try:
        log = boto3.client("bedrock", region_name=a.region).get_model_invocation_logging_configuration()
        print("invocation logging configured:", bool(log.get("loggingConfig")))
    except Exception as e:
        print("could not read invocation logging configuration:", str(e)[:120])
    items = json.load(open(os.path.expanduser(a.prompts)))
    if a.dry:
        items = [{"conversation": "DRY", "detected_condition": "", "system": "You are concise.",
                  "user": "Reply with the single word: ready"}]
    elif a.limit:
        items = items[:a.limit]
    rag_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(rag_root, "results", "fareez")
    sdir = os.path.join(out, "fareez_summaries", a.short_name)
    cache = os.path.join(out, f"round2_{a.short_name}_raw.jsonl")  # each note is saved as soon as it returns
    done = {}
    if os.path.exists(cache) and not a.dry:
        done = {json.loads(l)["conversation"]: json.loads(l) for l in open(cache)}
    results = []
    for it in items:
        if it["conversation"] in done:
            results.append(done[it["conversation"]]); continue
        t0 = time.time()
        # The US inference profile routes across regions. A region where this Legacy model has not
        # been used in the last 30 days refuses the call, so a refused call is retried.
        for attempt in range(12):
            try:
                r = br.converse(modelId=a.model, system=[{"text": it["system"]}],
                                messages=[{"role": "user", "content": [{"text": it["user"]}]}],
                                inferenceConfig={"temperature": 0.3, "maxTokens": a.max_tokens})
                break
            except br.exceptions.ResourceNotFoundException as e:
                print(f"{it['conversation']} refused (attempt {attempt + 1}): {str(e)[:90]}", flush=True)
                time.sleep(3)
        else:
            raise RuntimeError(f"{it['conversation']}: refused 12 times")
        text = "".join(b.get("text", "") for b in r["output"]["message"]["content"]).strip()
        results.append(dict(conversation=it["conversation"], detected_condition=it["detected_condition"], summary=text,
                            total_time=round(time.time() - t0, 2), input_tokens=r["usage"]["inputTokens"],
                            output_tokens=r["usage"]["outputTokens"], finish_reason=r["stopReason"]))
        print(it["conversation"], results[-1]["output_tokens"], results[-1]["finish_reason"], flush=True)
        if not a.dry:
            with open(cache, "a") as f:
                f.write(json.dumps(results[-1]) + "\n")
    if a.dry:
        print(results); return
    os.makedirs(sdir, exist_ok=True)
    for r in results:
        with open(os.path.join(sdir, f"{r['conversation']}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Model: {a.model}\nPatient: {r['conversation']}\nDetected Disease: {r['detected_condition']}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTime: {r['total_time']:.2f}s\n")
            f.write(f"Tokens: {r['input_tokens']:,} in / {r['output_tokens']:,} out\n" + "=" * 60 + "\n\n")
            f.write(r["summary"])
    with open(os.path.join(out, f"evaluation_results_{a.short_name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_name", "model", "detected_disease", "total_time_s", "input_tokens", "output_tokens",
                    "summary_chars", "summary_words", "finish_reason"])
        for r in results:
            w.writerow([r["conversation"], a.short_name, r["detected_condition"], r["total_time"], r["input_tokens"],
                        r["output_tokens"], len(r["summary"]), len(r["summary"].split()), r["finish_reason"]])
    print(f"saved {len(results)} notes to {sdir}")


if __name__ == "__main__":
    main()
