"""
Exercise 3 — Digit classification with more data + class imbalance handling.

Three experiments run on the combined dataset (digits.csv + more_digits.csv):
  1. Baseline     — best config from Ex2, no weighting
  2. Weighted loss — same config, class weights scale the backprop gradient
  3. Weighted sampling — same config, rare classes oversampled each epoch

All three are evaluated on digits_test.csv (held-out, never touched during training).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

# exercise3/data.py (local)
from data import (  # noqa: E402
    augment_images,
    build_scaler,
    compute_class_weights,
    load_combined,
    load_test,
    print_class_weights,
    print_combined_eda,
    print_dataset_overview,
    stratified_train_validation_split,
)

# exercise2 modules loaded via explicit path to avoid name collisions
_EX2 = Path(__file__).resolve().parent.parent / "exercise2"
sys.path.insert(0, str(_EX2))

from metrics import classification_metrics, save_metrics_json  # noqa: E402
from mlp import MultilayerPerceptron  # noqa: E402
import plots as _ex2_plots  # noqa: E402
from plots import save_confusion_matrix, save_training_curves  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path(__file__).resolve().parent / "models"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Best hyperparameters found in Exercise 2
ARCHITECTURE = [784, 256, 128, 10]
LEARNING_RATE = 0.001
OPTIMIZER = "adam"
BATCH_SIZE = 128
HIDDEN_ACTIVATION = "leaky_relu"
OUTPUT_ACTIVATION = "logistic"
LEAKY_RELU_SLOPE = 0.01
L2_LAMBDA = 1e-4
EPOCHS = 180
PATIENCE = 24
VALIDATION_RATIO = 0.15
NORMALIZATION = "minmax"
AUGMENT_REPEATS = 1
AUGMENT_SHIFT_MAX = 2
AUGMENT_NOISE_STD = 0.03


def _section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def _history_to_dicts(history) -> list[dict]:
    return [
        {
            "epoch": p.epoch,
            "train_loss": p.train_loss,
            "train_accuracy": p.train_accuracy,
            "train_f1": p.train_f1,
            "val_loss": p.val_loss,
            "val_accuracy": p.val_accuracy,
            "val_f1": p.val_f1,
        }
        for p in history
    ]


def run_experiment(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_metadata: dict | None,
    class_weights: np.ndarray | None = None,
    use_weighted_sampling: bool = False,
) -> dict:
    _section(name)
    mode = (
        "weighted loss" if class_weights is not None
        else "weighted sampling" if use_weighted_sampling
        else "baseline (no weighting)"
    )
    print(f"  mode            : {mode}")
    print(f"  architecture    : {ARCHITECTURE}")
    print(f"  activation      : hidden={HIDDEN_ACTIVATION} output={OUTPUT_ACTIVATION}")
    print(f"  optimizer/lr    : {OPTIMIZER} / {LEARNING_RATE}")
    print(f"  l2 lambda       : {L2_LAMBDA}")
    print(f"  epochs (max)    : {EPOCHS}  patience={PATIENCE}")
    print(f"  train / val     : {len(X_train)} / {len(X_val)}\n")

    model = MultilayerPerceptron(
        layer_sizes=ARCHITECTURE,
        learning_rate=LEARNING_RATE,
        activation=HIDDEN_ACTIVATION,
        output_activation=OUTPUT_ACTIVATION,
        optimizer=OPTIMIZER,
        batch_size=BATCH_SIZE,
        l2_lambda=L2_LAMBDA,
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        seed=42,
    )

    t0 = perf_counter()
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=EPOCHS,
        verbose=True,
        early_stopping=True,
        patience=PATIENCE,
        class_weights=class_weights,
        use_weighted_sampling=use_weighted_sampling,
    )
    elapsed = perf_counter() - t0

    train_eval = model.evaluate(X_train, y_train)
    val_eval = model.evaluate(X_val, y_val)
    test_eval = model.evaluate(X_test, y_test)

    print(f"\n  elapsed         : {elapsed:.1f}s")
    print(f"  best epoch      : {model.best_epoch}")
    print(f"  train  acc/F1   : {train_eval['accuracy']:.2%} / {train_eval['f1_macro']:.4f}")
    print(f"  val    acc/F1   : {val_eval['accuracy']:.2%} / {val_eval['f1_macro']:.4f}")
    print(f"  test   acc/F1   : {test_eval['accuracy']:.2%} / {test_eval['f1_macro']:.4f}")

    print(f"\n  Per-class accuracy on test:")
    cm = test_eval["confusion_matrix"]
    for cls in range(10):
        row_total = cm[cls].sum()
        acc = cm[cls, cls] / row_total if row_total > 0 else 0.0
        bar = "#" * int(acc * 20)
        print(f"    {cls}: {acc:.3f}  {bar}")

    history = _history_to_dicts(model.history)
    slug = name.lower().replace(" ", "_").replace("-", "_")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(
        MODELS_DIR / slug,
        metadata={
            "config": {
                "architecture": ARCHITECTURE,
                "learning_rate": LEARNING_RATE,
                "optimizer": OPTIMIZER,
                "batch_size": BATCH_SIZE,
                "hidden_activation": HIDDEN_ACTIVATION,
                "output_activation": OUTPUT_ACTIVATION,
                "leaky_relu_slope": LEAKY_RELU_SLOPE,
                "l2_lambda": L2_LAMBDA,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "normalization": NORMALIZATION,
                "augment_repeats": AUGMENT_REPEATS,
                "augment_shift_max": AUGMENT_SHIFT_MAX,
                "augment_noise_std": AUGMENT_NOISE_STD,
            },
            "scaler": scaler_metadata,
            "best_epoch": model.best_epoch,
            "mode": mode,
        },
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_json(
        RESULTS_DIR / f"{slug}.json",
        {
            "name": name,
            "mode": mode,
            "config": {
                "architecture": ARCHITECTURE,
                "learning_rate": LEARNING_RATE,
                "optimizer": OPTIMIZER,
                "batch_size": BATCH_SIZE,
                "hidden_activation": HIDDEN_ACTIVATION,
                "output_activation": OUTPUT_ACTIVATION,
                "leaky_relu_slope": LEAKY_RELU_SLOPE,
                "l2_lambda": L2_LAMBDA,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "normalization": NORMALIZATION,
                "augment_repeats": AUGMENT_REPEATS,
                "augment_shift_max": AUGMENT_SHIFT_MAX,
                "augment_noise_std": AUGMENT_NOISE_STD,
            },
            "elapsed_seconds": elapsed,
            "best_epoch": model.best_epoch,
            "train": {k: v for k, v in train_eval.items() if k != "confusion_matrix"},
            "validation": {k: v for k, v in val_eval.items() if k != "confusion_matrix"},
            "test": {k: v for k, v in test_eval.items() if k != "confusion_matrix"},
            "history": history,
        },
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_training_curves(
        f"{slug}_curves.png",
        history,
        f"Ex3 — {name}",
    )
    save_confusion_matrix(
        f"{slug}_confusion.png",
        test_eval["confusion_matrix"],
        f"Ex3 — {name} — Test Confusion Matrix",
    )

    return {"name": name, "model": model, "test_eval": test_eval, "history": history}


def print_comparison(results: list[dict]) -> None:
    _section("Comparison — digits_test.csv")
    header = f"  {'Experiment':<25} {'Acc':>8} {'F1 macro':>10} {'Best epoch':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        te = r["test_eval"]
        best = r["model"].best_epoch or len(r["history"])
        print(
            f"  {r['name']:<25} {te['accuracy']:>8.2%} {te['f1_macro']:>10.4f} {best:>11}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise 3 - Digit classification with more data, augmentation, and imbalance handling"
    )
    parser.add_argument(
        "--experiment",
        choices=["all", "baseline", "weighted_loss", "weighted_sampling"],
        default="all",
        help="Run the full comparison or a single experiment for faster iteration.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    _section("Exercise 3 — Digit Classification with More Data")

    print("\nLoading combined dataset...")
    X_all, y_all = load_combined()
    X_test, y_test = load_test()

    print_combined_eda(X_all, y_all)
    print_dataset_overview("digits_test.csv", X_test, y_test)

    # Train / validation split (stratified)
    X_train_raw, y_train, X_val_raw, y_val = stratified_train_validation_split(
        X_all, y_all, validation_ratio=VALIDATION_RATIO
    )
    class_w = compute_class_weights(y_train)
    print_class_weights(class_w)
    print(f"\nTrain: {len(X_train_raw)}  Val: {len(X_val_raw)}  Test: {len(X_test)}")
    print(
        f"Augmentation: repeats={AUGMENT_REPEATS}  shift_max={AUGMENT_SHIFT_MAX}  "
        f"noise_std={AUGMENT_NOISE_STD}"
    )
    print("Tip: use --experiment weighted_sampling to rerun only the best variant while iterating.")

    X_train_aug, y_train_aug = augment_images(
        X_train_raw,
        y_train,
        repeats=AUGMENT_REPEATS,
        shift_max=AUGMENT_SHIFT_MAX,
        noise_std=AUGMENT_NOISE_STD,
        seed=42,
    )
    print(f"Augmented train samples: {len(X_train_aug)}")

    # Redirect exercise2 plots module to save into exercise3/plots/
    _ex2_plots.PLOTS_DIR = PLOTS_DIR
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Normalization (fit on train, apply everywhere)
    scaler = build_scaler(NORMALIZATION)
    X_train = scaler.fit_transform(X_train_aug)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test)
    scaler_metadata = scaler.to_metadata() if hasattr(scaler, "to_metadata") else None

    results = []

    # -----------------------------------------------------------------------
    # Experiment 1 — Baseline
    # -----------------------------------------------------------------------
    if args.experiment in ("all", "baseline"):
        results.append(run_experiment(
            "1-Baseline",
            X_train, y_train_aug, X_val, y_val, X_test, y_test,
            scaler_metadata=scaler_metadata,
        ))

    # -----------------------------------------------------------------------
    # Experiment 2 — Weighted Loss
    # -----------------------------------------------------------------------
    if args.experiment in ("all", "weighted_loss"):
        results.append(run_experiment(
            "2-Weighted-Loss",
            X_train, y_train_aug, X_val, y_val, X_test, y_test,
            scaler_metadata=scaler_metadata,
            class_weights=class_w,
        ))

    # -----------------------------------------------------------------------
    # Experiment 3 — Weighted Sampling
    # -----------------------------------------------------------------------
    if args.experiment in ("all", "weighted_sampling"):
        results.append(run_experiment(
            "3-Weighted-Sampling",
            X_train, y_train_aug, X_val, y_val, X_test, y_test,
            scaler_metadata=scaler_metadata,
            use_weighted_sampling=True,
        ))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_comparison(results)
