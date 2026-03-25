"""
Generate all paper tables from saved experiment results.
Run after completing Parts 1-3 of the experiment improvement plan.

Usage:
    python scripts/generate_tables.py
"""
import pandas as pd
from pathlib import Path


def table1_asr():
    """Table 1: ASR model WER (%) across all 4 datasets."""
    print("\n" + "=" * 110)
    print("TABLE 1: ASR Model Word Error Rate (%)")
    print("=" * 110)

    wer_dir = Path("openemr_whisper_wer")

    # Collect results from all datasets
    dataset_patterns = {
        "Notion": "results/*notion*.csv",
        "Kaggle": "results/*kaggle*.csv",
        "PriMock57": "results/primock57-*.csv",
        "Fareez": "results/fareez-*.csv",
    }

    results = {}
    for ds_name, pattern in dataset_patterns.items():
        for csv_file in sorted(wer_dir.glob(pattern)):
            df = pd.read_csv(csv_file)
            if "wer" not in df.columns:
                continue
            # Extract model name from filename
            stem = csv_file.stem
            parts = stem.split("-", 1)
            model = parts[1] if len(parts) > 1 else stem
            # Exclude error rows (WER=1.0 from failed transcriptions)
            valid = df[~df["error"].notna()] if "error" in df.columns else df
            if len(valid) == 0:
                continue
            avg_wer = valid["wer"].mean() * 100
            n_ok = len(valid)
            n_total = len(df)

            if model not in results:
                results[model] = {}
            results[model][ds_name] = avg_wer
            results[model][f"{ds_name}_n"] = f"{n_ok}/{n_total}"

    if not results:
        print("  No ASR results found. Run Part 1 first.")
        return

    datasets = list(dataset_patterns.keys())
    header = f"{'Model':<25}"
    for ds in datasets:
        header += f"  {ds:<15}"
    header += f"  {'Avg':<10}"
    print(header)
    print("-" * 110)

    for model in sorted(results.keys()):
        row = f"{model:<25}"
        vals = []
        for ds in datasets:
            val = results[model].get(ds)
            if val is not None:
                row += f"  {val:>6.2f}{'':>9}"
                vals.append(val)
            else:
                row += f"  {'--':>6}{'':>9}"
        avg = sum(vals) / len(vals) if vals else 0
        row += f"  {avg:>6.2f}"
        print(row)


def table2_rag():
    """Table 2: RAG summarization -- delegates to aggregated results."""
    rag_dir = Path("rag_models/RAG_To_See_MedGemma_Performance/results")
    agg_file = rag_dir / "aggregated_results.csv"

    if agg_file.exists():
        print("\n" + "=" * 110)
        print("TABLE 2: RAG Summarization (mean +/- SD across 3 runs)")
        print("=" * 110)
        df = pd.read_csv(agg_file, index_col=0)
        print(df.to_string())
    else:
        print("\n  TABLE 2: Run aggregate_results.py first")
        print(f"    cd rag_models/RAG_To_See_MedGemma_Performance && python aggregate_results.py")


def table3_elm():
    """Table 3: ELM validation results across all models."""
    print("\n" + "=" * 110)
    print("TABLE 3: ELM JSON Validation Results (22 test cases: 15 valid + 7 invalid)")
    print("=" * 110)

    elm_dir = Path("cdr_elmjson_validator/results")
    if not elm_dir.exists():
        print("  No ELM results found. Run Part 3 first.")
        return

    csv_files = sorted(elm_dir.glob("elm-*.csv"))
    if not csv_files:
        csv_files = sorted(elm_dir.glob("results-*.csv"))
    if not csv_files:
        csv_files = sorted(elm_dir.glob("results_*.csv"))
    if not csv_files:
        print("  No ELM result CSV files found.")
        return

    header = f"{'Model':<25} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Time (s)':<14}"
    print(header)
    print("-" * 70)

    rows = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        model = csv_file.stem.replace("elm-", "").replace("results-", "").replace("results_", "")
        total = len(df)
        correct = df["correct"].sum() if "correct" in df.columns else 0
        accuracy = correct / total * 100 if total > 0 else 0
        avg_time = df["time_seconds"].mean() if "time_seconds" in df.columns else 0
        rows.append((accuracy, model, correct, total, avg_time))

    # Sort by accuracy descending
    for accuracy, model, correct, total, avg_time in sorted(rows, reverse=True):
        print(f"{model:<25} {accuracy:>6.1f}%{'':>5} {int(correct)}/{total}{'':>9} {avg_time:>8.2f}")


if __name__ == "__main__":
    table1_asr()
    table2_rag()
    table3_elm()
