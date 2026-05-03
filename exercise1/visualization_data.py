from __future__ import annotations

import json
from pathlib import Path

import numpy as np


VISUALIZATIONS_DIR = Path(__file__).resolve().parent / "visualizations"
VISUALIZATION_DATA_DIR = VISUALIZATIONS_DIR / "data"


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def _metrics_summary(metrics: dict) -> dict:
    return {
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def save_final_model_visualization_data(
    filename: str,
    feature_names: list[str],
    weights: np.ndarray,
    bias: float,
    activation: str,
    hyperparameters: dict,
    split_sizes: dict,
    threshold: float,
    train_metrics: dict,
    validation_metrics: dict,
    test_metrics: dict,
    test_roc_auc: float,
    test_pr_auc: float,
) -> Path:
    VISUALIZATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = VISUALIZATION_DATA_DIR / filename

    payload = {
        "title": "Exercise 1 — Final Non-Linear Perceptron",
        "model": {
            "type": "NonLinearPerceptron",
            "activation": activation,
            "feature_names": feature_names,
            "weights": np.asarray(weights, dtype=np.float64),
            "bias": float(bias),
        },
        "hyperparameters": hyperparameters,
        "splits": split_sizes,
        "selection": {
            "threshold_source": "validation_max_f1",
            "threshold": float(threshold),
            "test_roc_auc": float(test_roc_auc),
            "test_pr_auc": float(test_pr_auc),
        },
        "metrics": {
            "train": _metrics_summary(train_metrics),
            "validation": _metrics_summary(validation_metrics),
            "test": _metrics_summary(test_metrics),
        },
        "notes": [
            "Weights operate on z-scored input features.",
            "The perceptron is trained to regress big_model_fraud_probability.",
            "Fraud decisions are evaluated against flagged_fraud using the validation-selected threshold.",
        ],
    }

    path.write_text(json.dumps(_to_serializable(payload), indent=2))
    return path
