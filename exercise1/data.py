from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


def _candidate_data_dirs() -> list[Path]:
    env_dir = os.environ.get("TP3_DATA_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            repo_root / "data and documentation",
            repo_root / "data_and_documentation",
            Path.cwd() / "data and documentation",
            Path.cwd() / "data_and_documentation",
        ]
    )
    return candidates


def resolve_data_path(filename: str) -> Path:
    for directory in _candidate_data_dirs():
        candidate = directory / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename!r}. Checked: "
        + ", ".join(str(directory / filename) for directory in _candidate_data_dirs())
        + ". Place the extracted dataset folder in the repository root "
        + "or set TP3_DATA_DIR=/path/to/data_and_documentation."
    )


FEATURE_COLUMNS = [
    "timestamp",
    "amount_usd",
    "quantity_purchased",
    "session_duration_seconds",
    "days_since_last_purchase",
    "account_age_days",
    "device_screen_resolution",
    "time_since_last_login_s",
    "items_viewed_before_purchase",
]

TARGET_COLUMN = "big_model_fraud_probability"
GROUND_TRUTH_COLUMN = "flagged_fraud"


def load_dataset():
    data_path = resolve_data_path("fraud_dataset.csv")
    rows = []
    with open(data_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    X = np.array([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows])
    y = np.array([float(row[TARGET_COLUMN]) for row in rows])
    ground_truth = np.array([int(row[GROUND_TRUTH_COLUMN]) for row in rows])

    return X, y, ground_truth


def print_eda(X, y, ground_truth):
    title = "Dataset Exploration"
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(
        f"Target (big_model_fraud_probability): "
        f"min={y.min():.4f}  max={y.max():.4f}  mean={y.mean():.4f}  std={y.std():.4f}"
    )
    print(
        f"Fraud rate (flagged_fraud): "
        f"{ground_truth.mean():.2%} ({ground_truth.sum()} / {len(ground_truth)})"
    )
    print()
    print(f"{'Feature':<35} {'Min':>14} {'Max':>14} {'Mean':>14} {'Std':>14}")
    print("-" * 93)
    for i, col in enumerate(FEATURE_COLUMNS):
        col_data = X[:, i]
        print(
            f"{col:<35} {col_data.min():>14.2f} {col_data.max():>14.2f} "
            f"{col_data.mean():>14.2f} {col_data.std():>14.2f}"
        )
    print()


def print_data_quality(X: np.ndarray, y: np.ndarray, ground_truth: np.ndarray) -> None:
    title = "Data Quality Check"
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")

    # NaN check
    nan_X = np.isnan(X).sum(axis=0)
    nan_y = int(np.isnan(y).sum())
    nan_gt = int(np.isnan(ground_truth.astype(float)).sum())
    if nan_X.sum() == 0 and nan_y == 0 and nan_gt == 0:
        print("NaN values: none found")
    else:
        print("NaN values per column:")
        for i, col in enumerate(FEATURE_COLUMNS):
            if nan_X[i] > 0:
                print(f"  {col}: {int(nan_X[i])}")
        if nan_y > 0:
            print(f"  {TARGET_COLUMN}: {nan_y}")
        if nan_gt > 0:
            print(f"  {GROUND_TRUTH_COLUMN}: {nan_gt}")

    # Duplicate rows (features + target)
    combined = np.hstack([X, y.reshape(-1, 1)])
    n_unique = len(np.unique(combined, axis=0))
    n_duplicates = len(X) - n_unique
    print(f"Duplicate rows: {n_duplicates}")

    # Outlier detection per feature (IQR method, factor=1.5)
    print(f"\nOutliers per feature (IQR method, factor=1.5):")
    print(f"  {'Feature':<35} {'Count':>7} {'%':>7} {'Lower bound':>14} {'Upper bound':>14}")
    print("  " + "-" * 81)
    for i, col in enumerate(FEATURE_COLUMNS):
        col_data = X[:, i]
        q1 = np.percentile(col_data, 25)
        q3 = np.percentile(col_data, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(np.sum((col_data < lower) | (col_data > upper)))
        pct = 100.0 * n_out / len(col_data)
        print(f"  {col:<35} {n_out:>7} {pct:>6.2f}%  {lower:>14.2f} {upper:>14.2f}")


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    ground_truth: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split = int(len(X) * train_ratio)
    train_idx, test_idx = indices[:split], indices[split:]
    return (
        X[train_idx], y[train_idx], ground_truth[train_idx],
        X[test_idx], y[test_idx], ground_truth[test_idx],
    )


def kfold_split(
    X: np.ndarray,
    y: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 5,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    folds = np.array_split(indices, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield (
            X[train_idx], y[train_idx], ground_truth[train_idx],
            X[val_idx], y[val_idx], ground_truth[val_idx],
        )
