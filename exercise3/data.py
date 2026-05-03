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
IMAGE_SIDE = 28


def load_combined() -> tuple[np.ndarray, np.ndarray]:
    """digits.csv + more_digits.csv concatenated."""
    X1, y1 = load_digits_dataset("digits.csv")
    X2, y2 = load_digits_dataset("more_digits.csv")
    return np.vstack([X1, X2]), np.concatenate([y1, y2])


def load_test() -> tuple[np.ndarray, np.ndarray]:
    return load_digits_dataset("digits_test.csv")


def stratified_train_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for class_idx in range(N_CLASSES):
        class_indices = np.flatnonzero(y == class_idx)
        class_indices = rng.permutation(class_indices)
        split = int(len(class_indices) * (1.0 - validation_ratio))
        train_parts.append(class_indices[:split])
        val_parts.append(class_indices[split:])

    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _shift_image(flat_image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    image = flat_image.reshape(IMAGE_SIDE, IMAGE_SIDE)
    shifted = np.roll(image, shift=(dy, dx), axis=(0, 1))

    if dy > 0:
        shifted[:dy, :] = 0.0
    elif dy < 0:
        shifted[dy:, :] = 0.0

    if dx > 0:
        shifted[:, :dx] = 0.0
    elif dx < 0:
        shifted[:, dx:] = 0.0

    return shifted.reshape(-1)


def augment_images(
    X: np.ndarray,
    y: np.ndarray,
    repeats: int = 1,
    shift_max: int = 2,
    noise_std: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if repeats <= 0:
        return X.astype(np.float32, copy=True), y.astype(np.int64, copy=True)

    rng = np.random.default_rng(seed)
    augmented_batches = [X.astype(np.float32, copy=True)]
    augmented_labels = [y.astype(np.int64, copy=True)]

    for _ in range(repeats):
        X_aug = np.empty_like(X, dtype=np.float32)
        for idx, sample in enumerate(X):
            dx = int(rng.integers(-shift_max, shift_max + 1))
            dy = int(rng.integers(-shift_max, shift_max + 1))
            aug = _shift_image(sample, dx=dx, dy=dy)
            if noise_std > 0.0:
                aug = np.clip(aug + rng.normal(0.0, noise_std, size=aug.shape), 0.0, 1.0)
            X_aug[idx] = aug.astype(np.float32)

        augmented_batches.append(X_aug)
        augmented_labels.append(y.astype(np.int64, copy=True))

    return np.vstack(augmented_batches), np.concatenate(augmented_labels)


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
