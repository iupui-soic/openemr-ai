"""Set-based multi-label classification metrics for the CPT benchmark.

All functions take `gold` and `pred` as parallel `list[set[str]]` (one entry per
note). `labels` is the full label space (61 CPT codes). Zero denominators are
handled by returning 0.0 so no NaN propagates into the summary table.
"""
from __future__ import annotations

from collections.abc import Iterable


def _confusion_counts(
    gold: list[set[str]], pred: list[set[str]], label: str
) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        g_has = label in g
        p_has = label in p
        if g_has and p_has:
            tp += 1
        elif p_has and not g_has:
            fp += 1
        elif g_has and not p_has:
            fn += 1
    return tp, fp, fn


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def micro_f1(gold: list[set[str]], pred: list[set[str]], labels: Iterable[str]) -> float:
    tp_sum = fp_sum = fn_sum = 0
    for label in labels:
        tp, fp, fn = _confusion_counts(gold, pred, label)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
    return _f1_from_counts(tp_sum, fp_sum, fn_sum)


def macro_f1(gold: list[set[str]], pred: list[set[str]], labels: Iterable[str]) -> float:
    scores = [
        _f1_from_counts(*_confusion_counts(gold, pred, label)) for label in labels
    ]
    return sum(scores) / len(scores) if scores else 0.0


def per_label_f1(
    gold: list[set[str]], pred: list[set[str]], labels: Iterable[str]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        tp, fp, fn = _confusion_counts(gold, pred, label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": _f1_from_counts(tp, fp, fn),
            "support": tp + fn,
        }
    return out


def exact_match_ratio(gold: list[set[str]], pred: list[set[str]]) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for g, p in zip(gold, pred) if g == p)
    return hits / len(gold)


def jaccard_mean(gold: list[set[str]], pred: list[set[str]]) -> float:
    if not gold:
        return 0.0
    total = 0.0
    for g, p in zip(gold, pred):
        union = g | p
        if not union:
            total += 1.0
            continue
        total += len(g & p) / len(union)
    return total / len(gold)


def label_cardinality_ratio(
    gold: list[set[str]], pred: list[set[str]]
) -> float:
    if not gold:
        return 0.0
    mean_gold = sum(len(g) for g in gold) / len(gold)
    mean_pred = sum(len(p) for p in pred) / len(pred)
    if mean_gold == 0:
        return 0.0
    return mean_pred / mean_gold


def summarize(
    gold: list[set[str]], pred: list[set[str]], labels: Iterable[str]
) -> dict[str, float]:
    labels_list = list(labels)
    return {
        "micro_f1": micro_f1(gold, pred, labels_list),
        "macro_f1": macro_f1(gold, pred, labels_list),
        "exact_match_ratio": exact_match_ratio(gold, pred),
        "jaccard_mean": jaccard_mean(gold, pred),
        "label_cardinality_ratio": label_cardinality_ratio(gold, pred),
    }
