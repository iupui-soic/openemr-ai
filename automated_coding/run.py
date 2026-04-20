"""CLI orchestrator for the CPT medical coding benchmark.

Examples:
    # Regex baseline, all 312 notes
    python -m automated_coding.run --approach regex

    # scispaCy + SapBERT
    python -m automated_coding.run --approach scispacy_sapbert

    # MedCAT (requires MEDCAT_MODEL_PACK env var)
    python -m automated_coding.run --approach medcat

    # LLM via HuggingFace
    python -m automated_coding.run --approach llm \
        --backend hf --model-id google/gemma-4-26B-A4B-it

    # LLM via Anthropic
    python -m automated_coding.run --approach llm \
        --backend anthropic --model-id claude-sonnet-4-6

    # Produce the summary.md table across all approach metrics JSON on disk
    python -m automated_coding.run --summarize
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from . import data, metrics
from .approaches.base import Predictor


logger = logging.getLogger(__name__)


def _build_predictor(
    approach: str,
    backend: str | None,
    model_id: str | None,
    threshold: float | None,
) -> Predictor:
    if approach == "regex":
        from .approaches.regex import RegexPredictor

        return RegexPredictor()
    if approach == "scispacy_sapbert":
        from .approaches.scispacy_sapbert import (
            SIMILARITY_THRESHOLD,
            ScispacySapbertPredictor,
        )

        return ScispacySapbertPredictor(
            similarity_threshold=threshold
            if threshold is not None
            else SIMILARITY_THRESHOLD
        )
    if approach == "medcat":
        from .approaches.medcat import MEDCAT_SCORE_THRESHOLD, MedCATPredictor

        return MedCATPredictor(
            score_threshold=threshold
            if threshold is not None
            else MEDCAT_SCORE_THRESHOLD
        )
    if approach == "medcat_tight":
        from .approaches.medcat_tight import SCORE_THRESHOLD, MedCATTightPredictor

        return MedCATTightPredictor(
            score_threshold=threshold if threshold is not None else SCORE_THRESHOLD
        )
    if approach == "embed_match":
        from .approaches.embed_match import DEFAULT_THRESHOLD, EmbedMatchPredictor

        return EmbedMatchPredictor(
            similarity_threshold=threshold
            if threshold is not None
            else DEFAULT_THRESHOLD
        )
    if approach == "entity_match":
        from .approaches.entity_match import EntityMatchPredictor

        return EntityMatchPredictor()
    if approach == "hybrid_match":
        from .approaches.hybrid_match import HybridMatchPredictor

        return HybridMatchPredictor()
    if approach == "rerank_match":
        from .approaches.rerank_match import RerankMatchPredictor

        return RerankMatchPredictor()
    if approach == "llm":
        from .approaches.llm import LLMPredictor

        if not backend or not model_id:
            raise SystemExit(
                "--approach llm requires --backend {hf|anthropic|groq} and --model-id"
            )
        return LLMPredictor(model_id=model_id, backend=backend)
    raise SystemExit(f"Unknown approach: {approach}")


_RUN_ID_FORBIDDEN = re.compile(r"[^A-Za-z0-9._-]+")


def _run_id(approach: str, backend: str | None, model_id: str | None) -> str:
    if approach == "llm":
        slug = f"llm-{backend}-{model_id}"
    else:
        slug = approach
    return _RUN_ID_FORBIDDEN.sub("_", slug).strip("_")


def _run(
    approach: str,
    backend: str | None,
    model_id: str | None,
    threshold: float | None,
    limit: int | None,
    out_dir: Path,
    dataset_path: Path,
) -> None:
    import os
    notes = data.load_notes(dataset_path)
    if limit is not None:
        notes = notes[:limit]
    gold_space = data.get_label_space(data.load_notes(dataset_path))
    expanded_path = os.environ.get("EXPANDED_CODES_JSON")
    if expanded_path:
        descriptions = json.loads(Path(expanded_path).read_text(encoding="utf8"))
        label_space = sorted(descriptions.keys())
        logger.info(
            "label-space override: %d codes (%d gold, %d distractors)",
            len(label_space),
            sum(1 for c in label_space if c in gold_space),
            sum(1 for c in label_space if c not in gold_space),
        )
    else:
        label_space = gold_space
        descriptions = data.load_cpt_descriptions(label_space=label_space)

    codes_path = out_dir / "codes.json"
    if not codes_path.exists():
        data.write_codes_json(descriptions, codes_path)

    predictor = _build_predictor(approach, backend, model_id, threshold)
    logger.info("Preparing predictor %s", predictor.name)
    predictor.prepare(label_space, descriptions)

    preds: list[set[str]] = []
    latencies: list[float] = []
    per_note_records = []
    start = time.perf_counter()
    for idx, note in enumerate(notes):
        t0 = time.perf_counter()
        pred = set(predictor.predict(note.text))
        dt = time.perf_counter() - t0
        preds.append(pred)
        latencies.append(dt)
        per_note_records.append(
            {
                "note_id": note.note_id,
                "gold": sorted(note.gold_codes),
                "pred": sorted(pred),
                "latency_s": dt,
            }
        )
        if (idx + 1) % 25 == 0:
            logger.info(
                "  [%s] %d/%d notes (avg latency %.2fs)",
                predictor.name,
                idx + 1,
                len(notes),
                sum(latencies) / len(latencies),
            )
    total_wall = time.perf_counter() - start
    predictor.close()

    gold = [note.gold_codes for note in notes]
    summary_metrics = metrics.summarize(gold, preds, label_space)
    per_label = metrics.per_label_f1(gold, preds, label_space)

    run_id = _run_id(approach, backend, model_id)
    preds_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    preds_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (preds_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "approach": approach,
                "backend": backend,
                "model_id": model_id,
                "n_notes": len(notes),
                "per_note": per_note_records,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    (metrics_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "approach": approach,
                "backend": backend,
                "model_id": model_id,
                "n_notes": len(notes),
                "total_wall_s": total_wall,
                "mean_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
                "metrics": summary_metrics,
                "per_label_f1": per_label,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    logger.info(
        "[%s] micro-F1=%.3f  macro-F1=%.3f  EMR=%.3f  Jaccard=%.3f  LCR=%.2f  mean_lat=%.2fs",
        run_id,
        summary_metrics["micro_f1"],
        summary_metrics["macro_f1"],
        summary_metrics["exact_match_ratio"],
        summary_metrics["jaccard_mean"],
        summary_metrics["label_cardinality_ratio"],
        sum(latencies) / len(latencies) if latencies else 0.0,
    )


def _summarize(out_dir: Path) -> None:
    metrics_dir = out_dir / "metrics"
    rows = []
    for fp in sorted(metrics_dir.glob("*.json")):
        blob = json.loads(fp.read_text(encoding="utf8"))
        m = blob["metrics"]
        rows.append(
            {
                "run_id": blob["run_id"],
                "micro_f1": m["micro_f1"],
                "macro_f1": m["macro_f1"],
                "exact_match_ratio": m["exact_match_ratio"],
                "jaccard_mean": m["jaccard_mean"],
                "label_cardinality_ratio": m["label_cardinality_ratio"],
                "mean_latency_s": blob.get("mean_latency_s", 0.0),
                "n_notes": blob.get("n_notes", 0),
            }
        )

    lines = [
        "# CPT Medical Coding Benchmark — Summary\n",
        "Dataset: `data/processed/mdace_cpt/all.parquet` (312 notes, 61 unique CPT codes).\n",
        "| Approach | micro-F1 | macro-F1 | Exact-Match | Jaccard | Label-Card. Ratio | Mean latency (s) | n notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['run_id']}` | {r['micro_f1']:.3f} | {r['macro_f1']:.3f} | "
            f"{r['exact_match_ratio']:.3f} | {r['jaccard_mean']:.3f} | "
            f"{r['label_cardinality_ratio']:.2f} | {r['mean_latency_s']:.2f} | "
            f"{r['n_notes']} |"
        )
    lines.append("")
    lines.append(
        "Notes: LLM predictions use greedy decoding (`temperature=0`, single seed)."
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf8")
    logger.info("Wrote %s with %d rows", out_dir / "summary.md", len(rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approach",
        choices=[
            "regex",
            "scispacy_sapbert",
            "medcat",
            "medcat_tight",
            "embed_match",
            "entity_match",
            "hybrid_match",
            "rerank_match",
            "llm",
        ],
    )
    parser.add_argument(
        "--backend", choices=["hf", "anthropic", "groq"], default=None
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reports/benchmark/cpt")
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=data.default_parquet(),
    )
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        _summarize(args.out_dir)
        return 0
    if not args.approach:
        parser.error("--approach is required unless --summarize is given")
    _run(
        approach=args.approach,
        backend=args.backend,
        model_id=args.model_id,
        threshold=args.threshold,
        limit=args.limit,
        out_dir=args.out_dir,
        dataset_path=args.dataset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
