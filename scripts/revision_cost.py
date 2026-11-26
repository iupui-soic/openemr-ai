"""
Cost model: marginal compute cost per encounter and self-hosted total cost of ownership.

Part A. Marginal compute cost per encounter for the evaluated cloud configuration:
        ASR GPU seconds x Modal per-second GPU price, plus Groq input and output token prices x mean
        tokens from the generation logs of the 40 Fareez conversations.
Part B. Total cost of ownership per encounter for a self-hosted deployment:
        (GPU server capital / amortization years + power + hosting + IT staff FTE x salary)
        / annual encounters, at three volumes, with a low and high sensitivity case.

Every input is a named parameter below. The measured inputs are read from files in the repo.
The list prices were read from the vendor pages on PRICE_DATE. The Part B inputs are assumptions,
not measurements; replace them with local values.

Run: python scripts/revision_cost.py --repo .
Output: rag_models/results/revision/cost_model.csv (printed too).
"""
import argparse
import pandas as pd

# ----------------------------------------------------------------------------------------------
# List prices (US$), retrieved PRICE_DATE
# ----------------------------------------------------------------------------------------------
PRICE_DATE = "2026-11-11"
MODAL_URL = "https://modal.com/pricing"
GROQ_URL = "https://console.groq.com/docs/models"
MODAL_GPU_PER_S = {"A10": 0.000306, "L40S": 0.000542, "A100-40GB": 0.000583, "A100-80GB": 0.000694}
GROQ_PER_M_TOKENS = {  # (input, output)
    "gpt-oss-20b": (0.075, 0.30),
    "gpt-oss-120b": (0.15, 0.60),
}
# Not priced: llama-3.3-70b-versatile (the Fareez condition extractor) shows "Contact Sales" on
# PRICE_DATE, and qwen/qwen3-32b is no longer listed. Condition extraction is therefore priced
# with gpt-oss-20b, which is what the deployed app uses.

# ----------------------------------------------------------------------------------------------
# Part A assumptions
# ----------------------------------------------------------------------------------------------
ASR_LOG = "openemr_whisper_wer/results/fareez-voxtral-mini.csv"   # compute_s per file, local RTX 6000 Ada
ASR_LOG_GPU = "L40S"                                              # Modal GPU whose price the logged seconds are charged at
ASR_LOG_MODEL = "Voxtral Mini 3B"
DURATION_LOG = "openemr_whisper_wer/results/fareez_der.csv"       # duration_s per file
# The deployed service runs Parakeet TDT 1.1B on a Modal A10G, but its Modal runs did not log
# per-file processing time. This scenario prices it with the Voxtral GPU seconds as a stand-in.
DEPLOYED_ASR_GPU = "A10"
# Condition extraction call: first 2,000 characters of the transcript plus a short instruction,
# at most 20 output tokens (deployed app settings). About 4 characters per token.
EXTRACT_INPUT_TOKENS = 2000 / 4 + 100
EXTRACT_OUTPUT_TOKENS = 20

# ----------------------------------------------------------------------------------------------
# Part B assumptions (base, low, high). Assumptions, not measurements.
# ----------------------------------------------------------------------------------------------
TCO = dict(
    gpu_server_capital=dict(base=25000, low=12000, high=45000,
                            note="one server with one 48 GB GPU (L40S class), enough for ASR plus a 20B LLM"),
    amortization_years=dict(base=4, low=5, high=3, note="straight line, no residual value"),
    power_kw=dict(base=0.6, low=0.4, high=0.9, note="average draw, always on"),
    electricity_per_kwh=dict(base=0.13, low=0.10, high=0.18, note="US commercial rate"),
    pue=dict(base=1.5, low=1.3, high=1.8, note="cooling and distribution overhead"),
    hosting_per_year=dict(base=3000, low=1500, high=6000, note="rack space, network, backup"),
    it_staff_fte=dict(base=0.10, low=0.05, high=0.25, note="administration, updates, monitoring"),
    it_staff_salary=dict(base=120000, low=100000, high=150000, note="salary plus benefits"),
)
VOLUMES = [5000, 20000, 50000]


def tco_per_year(case):
    v = {k: d[case] for k, d in TCO.items()}
    capital = v["gpu_server_capital"] / v["amortization_years"]
    power = v["power_kw"] * 8760 * v["electricity_per_kwh"] * v["pue"]
    staff = v["it_staff_fte"] * v["it_staff_salary"]
    return dict(capital=capital, power=power, hosting=v["hosting_per_year"], staff=staff,
                total=capital + power + v["hosting_per_year"] + staff)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); a = ap.parse_args()
    repo = a.repo.rstrip("/"); R = f"{repo}/rag_models"

    asr = pd.read_csv(f"{repo}/{ASR_LOG}"); dur = pd.read_csv(f"{repo}/{DURATION_LOG}")
    gpu_s = asr.compute_s.mean(); audio_s = dur.duration_s.mean()
    tok = {}
    for m in GROQ_PER_M_TOKENS:
        d = pd.read_csv(f"{R}/results/fareez/evaluation_results_{m}.csv"); d = d[d.patient_name != "AVERAGE"]
        tok[m] = (d.input_tokens.mean(), d.output_tokens.mean(), len(d))

    print("=" * 100); print("PROVENANCE"); print("=" * 100)
    print(f"Prices retrieved {PRICE_DATE}. Modal: {MODAL_URL}. Groq: {GROQ_URL}.")
    print(f"Modal GPU, US$ per second: {MODAL_GPU_PER_S}")
    print(f"Groq, US$ per million tokens (input, output): {GROQ_PER_M_TOKENS}")
    print(f"Audio duration per encounter: {audio_s:.0f} s ({audio_s/60:.1f} min), mean of {len(dur)} Fareez recordings "
          f"({dur.duration_s.sum()/3600:.1f} h in total), from {DURATION_LOG}")
    print(f"ASR GPU seconds per encounter: {gpu_s:.1f} s (median {asr.compute_s.median():.1f}, max {asr.compute_s.max():.1f}), "
          f"{ASR_LOG_MODEL} on a local RTX 6000 Ada, priced at the Modal {ASR_LOG_GPU} rate, {len(asr)} files, from {ASR_LOG}. "
          f"Processing time only; container start-up and idle time before scale-down are not included.")
    for m, (i, o, n) in tok.items():
        print(f"Generation tokens, {m}: input {i:.0f}, output {o:.0f} (mean of {n} Fareez conversations)")
    print(f"Condition extraction: {EXTRACT_INPUT_TOKENS:.0f} input and {EXTRACT_OUTPUT_TOKENS} output tokens (assumed from the deployed settings)")

    rows = []
    def llm_cost(m):
        pi, po = GROQ_PER_M_TOKENS[m]; i, o, _ = tok[m]
        gen = (i * pi + o * po) / 1e6
        e_in, e_out = GROQ_PER_M_TOKENS["gpt-oss-20b"]
        ext = (EXTRACT_INPUT_TOKENS * e_in + EXTRACT_OUTPUT_TOKENS * e_out) / 1e6
        return gen, ext
    scenarios = [
        ("Logged ASR run: Voxtral Mini 3B at the L40S price; gpt-oss-20b", ASR_LOG_GPU, "gpt-oss-20b"),
        ("Logged ASR run: Voxtral Mini 3B at the L40S price; gpt-oss-120b", ASR_LOG_GPU, "gpt-oss-120b"),
        ("Deployed GPU type: A10 at the logged GPU seconds; gpt-oss-20b", DEPLOYED_ASR_GPU, "gpt-oss-20b"),
    ]
    for name, gpu, m in scenarios:
        c_asr = gpu_s * MODAL_GPU_PER_S[gpu]; gen, ext = llm_cost(m)
        rows.append(dict(part="A. Marginal compute cost per encounter", item=name,
                         asr_usd=round(c_asr, 4), llm_generation_usd=round(gen, 4), llm_extraction_usd=round(ext, 5),
                         total_usd=round(c_asr + gen + ext, 4),
                         basis=f"{gpu_s:.1f} GPU s x ${MODAL_GPU_PER_S[gpu]}/s; {tok[m][0]:.0f} in x ${GROQ_PER_M_TOKENS[m][0]}/M + "
                               f"{tok[m][1]:.0f} out x ${GROQ_PER_M_TOKENS[m][1]}/M; prices {PRICE_DATE}"))

    print("\n" + "=" * 100); print("TOTAL COST OF OWNERSHIP, SELF-HOSTED (all inputs are assumptions)"); print("=" * 100)
    for k, d in TCO.items():
        print(f"  {k:22s} base {d['base']:>9,}  low {d['low']:>9,}  high {d['high']:>9,}   {d['note']}")
    yr = {c: tco_per_year(c) for c in ("base", "low", "high")}
    for c in ("base", "low", "high"):
        y = yr[c]
        rows.append(dict(part="B. Self-hosted annual cost", item=f"{c} case", capital_usd=round(y["capital"]), power_usd=round(y["power"]),
                         hosting_usd=round(y["hosting"]), staff_usd=round(y["staff"]), total_usd=round(y["total"]),
                         basis="; ".join(f"{k}={d[c]}" for k, d in TCO.items())))
    for n in VOLUMES:
        rows.append(dict(part="B. Self-hosted cost per encounter", item=f"{n:,} encounters per year",
                         total_usd=round(yr["base"]["total"] / n, 2), low_usd=round(yr["low"]["total"] / n, 2),
                         high_usd=round(yr["high"]["total"] / n, 2),
                         staff_share_pct=round(100 * yr["base"]["staff"] / yr["base"]["total"]),
                         basis=f"base annual total ${yr['base']['total']:,.0f} / {n:,}; excludes clinician editing time, integration, and compliance work"))
    out = pd.DataFrame(rows)
    cols = ["part", "item", "asr_usd", "llm_generation_usd", "llm_extraction_usd", "capital_usd", "power_usd", "hosting_usd",
            "staff_usd", "total_usd", "low_usd", "high_usd", "staff_share_pct", "basis"]
    out = out[cols]
    path = f"{R}/results/revision/cost_model.csv"; out.to_csv(path, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print("\n" + out.drop(columns="basis").to_string(index=False, na_rep=""))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
