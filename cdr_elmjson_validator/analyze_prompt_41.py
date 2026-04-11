#!/usr/bin/env python3
"""Recompute prompt-strategy comparison table from 41-case CSVs."""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
PROMPTS = BASE / "results" / "prompts"
ANALYSIS = BASE / "results" / "analysis"

MODELS = ["qwen3-32b", "gpt-oss-120b", "gpt-oss-20b", "llama-3.3-70b"]
MODES = ["standard", "few-shot", "cot", "structured", "minimal"]


def main():
    rows = []
    for model in MODELS:
        for mode in MODES:
            path = PROMPTS / f"results-{model}-{mode}.csv"
            if not path.exists():
                print(f"MISSING: {path.name}")
                continue
            df = pd.read_csv(path)
            n = len(df)
            correct = int(df["correct"].sum())
            acc = correct / n
            TP = int(((df["valid"] == True) & (df["expected_valid"] == True)).sum())
            FP = int(((df["valid"] == True) & (df["expected_valid"] == False)).sum())
            FN = int(((df["valid"] == False) & (df["expected_valid"] == True)).sum())
            TN = int(((df["valid"] == False) & (df["expected_valid"] == False)).sum())
            sens = TP / (TP + FN) if (TP + FN) else 0
            spec = TN / (TN + FP) if (TN + FP) else 0
            rows.append({
                "model": model, "prompt_mode": mode, "correct": correct,
                "total": n, "accuracy": acc, "sens": sens, "spec": spec,
            })

    df = pd.DataFrame(rows)
    print("\n=== Accuracy (%) by model and prompt mode ===")
    pivot = df.pivot(index="prompt_mode", columns="model", values="accuracy") * 100
    pivot = pivot.reindex(MODES)[MODELS]
    print(pivot.round(1).to_string())

    print("\n=== Correct/Total ===")
    piv2 = df.pivot(index="prompt_mode", columns="model",
                    values="correct").reindex(MODES)[MODELS]
    tot = df.pivot(index="prompt_mode", columns="model",
                   values="total").reindex(MODES)[MODELS]
    for mode in MODES:
        line = f"{mode:<20}"
        for m in MODELS:
            line += f" {piv2.loc[mode, m]}/{tot.loc[mode, m]}"
        print(line)

    # Save a fresh analysis CSV
    out = ANALYSIS / "prompt_accuracy_41.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
