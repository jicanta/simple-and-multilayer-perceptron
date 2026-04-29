from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Load exercise2/data.py under an explicit alias to avoid circular import
# (both files are named data.py)
_ex2_data_path = Path(__file__).resolve().parent.parent / "exercise2" / "data.py"
_spec = importlib.util.spec_from_file_location("ex2_data", _ex2_data_path)
_ex2_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ex2_data)

build_scaler = _ex2_data.build_scaler
load_digits_dataset = _ex2_data.load_digits_dataset
print_data_quality = _ex2_data.print_data_quality
print_dataset_overview = _ex2_data.print_dataset_overview
train_validation_split = _ex2_data.train_validation_split

N_CLASSES = 10


def load_combined() -> tuple[np.ndarray, np.ndarray]:
    """digits.csv + more_digits.csv concatenated."""
    X1, y1 = load_digits_dataset("digits.csv")
    X2, y2 = load_digits_dataset("more_digits.csv")
    return np.vstack([X1, X2]), np.concatenate([y1, y2])


def load_test() -> tuple[np.ndarray, np.ndarray]:
    return load_digits_dataset("digits_test.csv")


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """
    Inverse-frequency weights, one per class.
    Formula: w_c = n_total / (n_classes * count_c)
    Rare classes get higher weights so their errors matter more.
    """
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = len(y) / (N_CLASSES * counts)
    return weights.astype(np.float32)


def print_class_weights(weights: np.ndarray) -> None:
    print("\n  Class weights (inverse frequency):")
    for cls, w in enumerate(weights):
        bar = "#" * int(w * 5)
        print(f"    class {cls}: {w:.3f}  {bar}")


def print_combined_eda(X: np.ndarray, y: np.ndarray) -> None:
    print("\nCombined dataset (digits.csv + more_digits.csv)")
    print("-" * 47)
    print(f"Samples  : {len(X)}")
    counts = np.bincount(y, minlength=N_CLASSES)
    print("Class distribution:")
    for cls, count in enumerate(counts):
        pct = 100.0 * count / len(y)
        print(f"  {cls}: {int(count):>5} ({pct:5.1f}%)")
    missing = [str(c) for c, n in enumerate(counts) if n == 0]
    scarce = [str(c) for c, n in enumerate(counts) if 0 < n < counts.max() * 0.3]
    if missing:
        print(f"\n  WARNING: missing classes: {missing}")
    if scarce:
        print(f"  WARNING: underrepresented classes: {scarce}")
