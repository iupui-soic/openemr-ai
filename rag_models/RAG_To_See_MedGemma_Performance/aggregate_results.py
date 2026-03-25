"""
Aggregate RAG evaluation results across 3 independent runs.
Computes mean +/- SD for each model and metric.
Outputs Table 2 for the paper.

Usage:
    python aggregate_results.py [results_dir]
"""
import pandas as pd
from pathlib import Path
import sys


def aggregate(results_dir: str = "results"):
    runs = []
    for run_dir in sorted(Path(results_dir).glob("run*")):
        for csv_file in run_dir.glob("evaluation_results_*.csv"):
            df = pd.read_csv(csv_file)
            # Exclude the AVERAGE row appended by save_results()
            df = df[df["patient_name"] != "AVERAGE"]
            df["run"] = run_dir.name
            runs.append(df)

    if not runs:
        print(f"No results found in {results_dir}/run*/")
        sys.exit(1)

    all_data = pd.concat(runs, ignore_index=True)

    metrics = ["bleu", "rouge_l", "sbert_coherence", "bert_f1",
               "scispacy_entity_recall", "medcat_entity_recall", "total_time_s"]
    available = [m for m in metrics if m in all_data.columns]

    # Group by model, compute mean and std across all patients and runs
    summary = all_data.groupby("model")[available].agg(["mean", "std"])

    # Flatten column names
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]

    # Print formatted table
    print("\n" + "=" * 130)
    print("TABLE 2: RAG Summarization Performance (mean +/- SD across 3 runs)")
    print("=" * 130)

    header = f"{'Model':<20}"
    for m in available:
        header += f"  {m:<22}"
    print(header)
    print("-" * 130)

    for model in sorted(summary.index):
        row = f"{model:<20}"
        for m in available:
            mean_val = summary.loc[model, f"{m}_mean"]
            std_val = summary.loc[model, f"{m}_std"]
            if m == "total_time_s":
                row += f"  {mean_val:>6.1f} +/- {std_val:<5.1f}{'':>4}"
            else:
                row += f"  {mean_val:.4f} +/- {std_val:.4f}"
        print(row)

    # Save to CSV
    out_path = f"{results_dir}/aggregated_results.csv"
    summary.to_csv(out_path)
    print(f"\nSaved to {out_path}")

    # Also save a clean version for the paper
    paper_df = pd.DataFrame()
    paper_df["model"] = summary.index
    for m in available:
        paper_df[m] = [
            f"{summary.loc[model, f'{m}_mean']:.4f} +/- {summary.loc[model, f'{m}_std']:.4f}"
            for model in summary.index
        ]
    paper_path = f"{results_dir}/table2_paper.csv"
    paper_df.to_csv(paper_path, index=False)
    print(f"Paper-ready table saved to {paper_path}")


if __name__ == "__main__":
    aggregate(sys.argv[1] if len(sys.argv) > 1 else "results")
