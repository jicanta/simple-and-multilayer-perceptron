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


def synthetic_balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    target_count: int | None = None,
    shift_max: int = 1,
    noise_std: float = 0.02,
    mix_min: float = 0.35,
    mix_max: float = 0.65,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance minority classes with SMOTE-like synthetic samples.

    For each underrepresented class, interpolate between two examples of the
    same class and add a small shift/noise perturbation so the generated digit
    is not just an exact duplicate.
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if mix_min > mix_max:
        raise ValueError("mix_min must be <= mix_max.")

    counts = np.bincount(y, minlength=N_CLASSES)
    goal = int(counts.max()) if target_count is None else int(target_count)
    if goal <= 0:
        return X.astype(np.float32, copy=True), y.astype(np.int64, copy=True)

    rng = np.random.default_rng(seed)
    synthetic_batches: list[np.ndarray] = []
    synthetic_labels: list[np.ndarray] = []

    for class_idx, count in enumerate(counts):
        needed = max(0, goal - int(count))
        if needed == 0:
            continue

        class_samples = X[y == class_idx].astype(np.float32, copy=False)
        if len(class_samples) == 0:
            continue

        X_syn = np.empty((needed, X.shape[1]), dtype=np.float32)
        for idx in range(needed):
            anchor_idx = int(rng.integers(0, len(class_samples)))
            anchor = class_samples[anchor_idx]

            if len(class_samples) == 1:
                synthetic = anchor.copy()
            else:
                neighbor_idx = int(rng.integers(0, len(class_samples) - 1))
                if neighbor_idx >= anchor_idx:
                    neighbor_idx += 1
                neighbor = class_samples[neighbor_idx]
                alpha = float(rng.uniform(mix_min, mix_max))
                synthetic = anchor + alpha * (neighbor - anchor)

            if shift_max > 0:
                dx = int(rng.integers(-shift_max, shift_max + 1))
                dy = int(rng.integers(-shift_max, shift_max + 1))
                synthetic = _shift_image(synthetic, dx=dx, dy=dy)
            if noise_std > 0.0:
                synthetic = synthetic + rng.normal(0.0, noise_std, size=synthetic.shape)

            X_syn[idx] = np.clip(synthetic, 0.0, 1.0).astype(np.float32)

        synthetic_batches.append(X_syn)
        synthetic_labels.append(np.full(needed, class_idx, dtype=np.int64))

    if not synthetic_batches:
        return X.astype(np.float32, copy=True), y.astype(np.int64, copy=True)

    X_balanced = np.vstack([X.astype(np.float32, copy=True), *synthetic_batches])
    y_balanced = np.concatenate([y.astype(np.int64, copy=True), *synthetic_labels])
    indices = rng.permutation(len(X_balanced))
    return X_balanced[indices], y_balanced[indices]


def _smote_neighbor_indices(class_samples: np.ndarray, k_neighbors: int) -> np.ndarray:
    if len(class_samples) < 2:
        raise ValueError("SMOTE requires at least 2 samples in the class.")

    n_neighbors = min(k_neighbors, len(class_samples) - 1)
    if n_neighbors <= 0:
        raise ValueError("SMOTE requires at least one same-class neighbor.")

    samples = class_samples.astype(np.float32, copy=False)
    dot_products = samples @ samples.T
    squared_norms = np.sum(samples * samples, axis=1, dtype=np.float32)
    distances = (-2.0 * dot_products).astype(np.float32, copy=False)
    distances += squared_norms[:, np.newaxis]
    distances += squared_norms[np.newaxis, :]
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, np.inf)

    return np.argpartition(distances, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]


def smote_balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    target_count: int | None = None,
    k_neighbors: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance minority classes with standard SMOTE interpolation.

    New samples are created as:
        x_new = x_i + gap * (x_nn - x_i)
    where x_nn is one of the k nearest same-class neighbors of x_i and
    gap ~ U(0, 1).
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1.")

    counts = np.bincount(y, minlength=N_CLASSES)
    goal = int(counts.max()) if target_count is None else int(target_count)
    if goal <= 0:
        return X.astype(np.float32, copy=True), y.astype(np.int64, copy=True)

    rng = np.random.default_rng(seed)
    synthetic_batches: list[np.ndarray] = []
    synthetic_labels: list[np.ndarray] = []

    for class_idx, count in enumerate(counts):
        needed = max(0, goal - int(count))
        if needed == 0:
            continue

        class_samples = X[y == class_idx].astype(np.float32, copy=False)
        if len(class_samples) == 0:
            continue
        if len(class_samples) < 2:
            raise ValueError(
                f"Cannot apply SMOTE to class {class_idx}: need at least 2 samples, got {len(class_samples)}."
            )

        neighbors = _smote_neighbor_indices(class_samples, k_neighbors=k_neighbors)
        anchor_indices = rng.integers(0, len(class_samples), size=needed)
        neighbor_choices = rng.integers(0, neighbors.shape[1], size=needed)
        neighbor_indices = neighbors[anchor_indices, neighbor_choices]
        gaps = rng.random(needed, dtype=np.float32)[:, np.newaxis]

        anchors = class_samples[anchor_indices]
        selected_neighbors = class_samples[neighbor_indices]
        X_syn = anchors + gaps * (selected_neighbors - anchors)

        synthetic_batches.append(X_syn.astype(np.float32, copy=False))
        synthetic_labels.append(np.full(needed, class_idx, dtype=np.int64))

    if not synthetic_batches:
        return X.astype(np.float32, copy=True), y.astype(np.int64, copy=True)

    X_balanced = np.vstack([X.astype(np.float32, copy=True), *synthetic_batches])
    y_balanced = np.concatenate([y.astype(np.int64, copy=True), *synthetic_labels])
    indices = rng.permutation(len(X_balanced))
    return X_balanced[indices], y_balanced[indices]


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
