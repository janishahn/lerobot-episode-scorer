import math

import numpy as np


def compute_binary_metrics(
    probabilities: list[float],
    labels: list[int],
    threshold: float = 0.5,
) -> dict[str, float | None]:
    if len(probabilities) != len(labels):
        raise ValueError("Probabilities and labels must have the same length")
    if not labels:
        raise ValueError("At least one label is required to compute binary metrics")

    probs = np.asarray(probabilities, dtype=float)
    truths = np.asarray(labels, dtype=int)
    predictions = (probs >= threshold).astype(int)

    positives = int(truths.sum())
    negatives = int(len(truths) - positives)
    tp = int(np.sum((predictions == 1) & (truths == 1)))
    tn = int(np.sum((predictions == 0) & (truths == 0)))
    fp = int(np.sum((predictions == 1) & (truths == 0)))
    fn = int(np.sum((predictions == 0) & (truths == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    true_positive_rate = tp / positives if positives else None
    true_negative_rate = tn / negatives if negatives else None
    if true_positive_rate is None or true_negative_rate is None:
        balanced_accuracy = None
    else:
        balanced_accuracy = (true_positive_rate + true_negative_rate) / 2.0

    clipped_probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    log_loss = float(
        -np.mean(truths * np.log(clipped_probs) + (1 - truths) * np.log(1.0 - clipped_probs))
    )

    return {
        "count": float(len(truths)),
        "prevalence": positives / len(truths),
        "accuracy": float(np.mean(predictions == truths)),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": binary_auroc(probs, truths),
        "auprc": binary_auprc(probs, truths),
        "brier_score": float(np.mean(np.square(probs - truths))),
        "log_loss": log_loss,
        "threshold": threshold,
    }


def binary_auroc(probabilities: np.ndarray, labels: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_true = np.cumsum(sorted_labels)
    cumulative_false = np.cumsum(1 - sorted_labels)

    tpr = np.concatenate(([0.0], cumulative_true / positives, [1.0]))
    fpr = np.concatenate(([0.0], cumulative_false / negatives, [1.0]))
    return float(np.trapezoid(tpr, fpr))


def binary_auprc(probabilities: np.ndarray, labels: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_true = np.cumsum(sorted_labels)
    cumulative_total = np.arange(1, len(sorted_labels) + 1)

    precision = cumulative_true / cumulative_total
    recall = cumulative_true / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapezoid(precision, recall))


def sanitized_metrics(metrics: dict[str, float | None]) -> dict[str, float | None]:
    clean: dict[str, float | None] = {}
    for key, value in metrics.items():
        if value is None:
            clean[key] = None
            continue
        if math.isnan(value) or math.isinf(value):
            clean[key] = None
            continue
        clean[key] = float(value)
    return clean
