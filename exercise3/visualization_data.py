from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
VIS_DATA_DIR = Path(__file__).resolve().parent / "visualizations" / "data"

_EXPERIMENT_SLUGS = [
    ("1_baseline",           "1 — Baseline"),
    ("2_weighted_loss",      "2 — Weighted Loss"),
    ("3_weighted_sampling",  "3 — Weighted Sampling"),
    ("4_synthetic_balancing","4 — Synthetic Balancing"),
    ("5_smote",              "5 — SMOTE"),
]


def _load(slug: str) -> dict:
    path = RESULTS_DIR / f"{slug}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_visualization_data(filename: str = "nn_overview.json") -> Path:
    VIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = VIS_DATA_DIR / filename

    baseline = _load("1_baseline")
    config = baseline.get("config", {})

    experiments = []
    for slug, display_name in _EXPERIMENT_SLUGS:
        raw = _load(slug)
        if not raw:
            continue
        test = raw.get("test", {})
        per_class = test.get("per_class", [])
        experiments.append({
            "name": display_name,
            "mode": raw.get("mode", ""),
            "test_accuracy": test.get("accuracy", 0.0),
            "test_f1_macro": test.get("f1_macro", 0.0),
            "best_epoch": raw.get("best_epoch", 0),
            "elapsed_seconds": raw.get("elapsed_seconds", 0.0),
            "per_class_recall": [round(c["recall"], 6) for c in per_class],
            "per_class_support": [c["support"] for c in per_class],
        })

    best_idx = max(range(len(experiments)), key=lambda i: experiments[i]["test_accuracy"]) if experiments else 0
    best = experiments[best_idx] if experiments else {}

    train_support = sum(c["support"] for c in baseline.get("train", {}).get("per_class", []))
    val_support = sum(c["support"] for c in baseline.get("validation", {}).get("per_class", []))
    test_support = sum(c["support"] for c in baseline.get("test", {}).get("per_class", []))

    payload = {
        "title": "Exercise 3 — Digit Classification MLP",
        "architecture": config.get("architecture", [784, 256, 128, 10]),
        "config": {
            "hidden_activation": config.get("hidden_activation", "leaky_relu"),
            "output_activation": config.get("output_activation", "logistic"),
            "optimizer": config.get("optimizer", "adam"),
            "learning_rate": config.get("learning_rate", 0.001),
            "batch_size": config.get("batch_size", 128),
            "l2_lambda": config.get("l2_lambda", 1e-4),
            "epochs_max": config.get("epochs", 180),
            "patience": config.get("patience", 24),
            "normalization": config.get("normalization", "minmax"),
            "augment_shift_max": config.get("augment_shift_max", 2),
            "augment_noise_std": config.get("augment_noise_std", 0.03),
        },
        "dataset": {
            "train_total": train_support,
            "val_total": val_support,
            "test_total": test_support,
        },
        "experiments": experiments,
        "best_experiment_index": best_idx,
        "best": {
            "name": best.get("name", ""),
            "test_accuracy": best.get("test_accuracy", 0.0),
            "test_f1_macro": best.get("test_f1_macro", 0.0),
            "best_epoch": best.get("best_epoch", 0),
            "per_class_recall": best.get("per_class_recall", []),
        },
        "notes": [
            "Images are 28×28 pixels (784 features), normalized with min-max scaler.",
            "Training: digits.csv + more_digits.csv, augmented with pixel shifts and Gaussian noise.",
            "Best strategy: weighted sampling — oversamples rare digit classes each epoch.",
            "Test set: digits_test.csv, held out and never used for tuning.",
        ],
    }

    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved visualization data → {path}")
    return path


if __name__ == "__main__":
    save_visualization_data()
