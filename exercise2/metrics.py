from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> dict:
    matrix = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    accuracy = correct / total if total > 0 else 0.0

    precisions = []
    recalls = []
    f1s = []
    per_class = []

    for class_idx in range(num_classes):
        tp = int(matrix[class_idx, class_idx])
        fp = int(matrix[:, class_idx].sum() - tp)
        fn = int(matrix[class_idx, :].sum() - tp)
        tn = total - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class.append(
            {
                "class": class_idx,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tpr": tpr,
                "fpr": fpr,
                "support": int(matrix[class_idx, :].sum()),
            }
        )

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(0.5 * np.mean(np.sum((y_true - y_pred) ** 2, axis=1)))


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def save_metrics_json(path: Path, payload: dict) -> None:
    serializable = _to_serializable(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))
