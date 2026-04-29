from __future__ import annotations

import ast
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
            Path(__file__).resolve().parent / "data",
        ]
    )
    return candidates


def resolve_data_path(filename: str) -> Path:
    directories = _candidate_data_dirs()
    for directory in directories:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename!r}. Checked: "
        + ", ".join(str(directory / filename) for directory in directories)
        + ". Place the extracted dataset folder in the repository root "
        + "or set TP3_DATA_DIR=/path/to/data_and_documentation."
    )


def load_digits_dataset(filename: str) -> tuple[np.ndarray, np.ndarray]:
    path = resolve_data_path(filename)
    labels: list[int] = []
    images: list[np.ndarray] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.append(int(row["label"]))
            images.append(np.array(ast.literal_eval(row["image"]), dtype=np.float32))

    X = np.vstack(images).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y


def print_dataset_overview(name: str, X: np.ndarray, y: np.ndarray) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Pixel range: min={float(X.min()):.4f}  max={float(X.max()):.4f}")
    print(f"Pixel mean/std: mean={float(X.mean()):.4f}  std={float(X.std()):.4f}")

    counts = np.bincount(y, minlength=10)
    print("Class distribution:")
    for label, count in enumerate(counts):
        pct = 100.0 * count / len(y)
        print(f"  {label}: {int(count):>5} ({pct:>6.2f}%)")


def print_data_quality(name: str, X: np.ndarray, y: np.ndarray) -> None:
    print(f"\n{name} - Data Quality")
    print("-" * (len(name) + 15))

    nan_features = int(np.isnan(X).sum())
    nan_labels = int(np.isnan(y.astype(np.float32)).sum())
    print(f"NaN values in features: {nan_features}")
    print(f"NaN values in labels:   {nan_labels}")

    invalid_labels = y[(y < 0) | (y > 9)]
    print(f"Invalid labels:         {len(invalid_labels)}")

    zero_pixels = int(np.sum(np.all(X == 0.0, axis=1)))
    print(f"All-zero images:        {zero_pixels}")

    duplicates = len(X) - len(np.unique(np.hstack([X, y.reshape(-1, 1)]), axis=0))
    print(f"Duplicate rows:         {duplicates}")

    near_constant_pixels = int(np.sum(X.std(axis=0) < 1e-8))
    print(f"Constant pixel columns: {near_constant_pixels}")


def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    encoded = np.zeros((len(y), num_classes), dtype=np.float32)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def train_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx = train_validation_indices(len(X), validation_ratio=validation_ratio, seed=seed)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def train_validation_indices(
    n_samples: int,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    split = int(n_samples * (1.0 - validation_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return train_idx, val_idx


class MinMaxScaler:
    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a = a
        self.b = b
        self.x_min_: np.ndarray | None = None
        self.x_max_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.x_min_ = X.min(axis=0)
        self.x_max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        denom = self.x_max_ - self.x_min_
        denom = np.where(denom == 0.0, 1.0, denom)
        scaled = (X - self.x_min_) / denom
        return scaled * (self.b - self.a) + self.a

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_metadata(self) -> dict:
        return {
            "type": "minmax",
            "a": self.a,
            "b": self.b,
            "x_min": self.x_min_.tolist(),
            "x_max": self.x_max_.tolist(),
        }


class StandardScaler:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_metadata(self) -> dict:
        return {
            "type": "zscore",
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }


class UnitLengthScaler:
    def fit(self, X: np.ndarray) -> "UnitLengthScaler":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return X / norms

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def to_metadata(self) -> dict:
        return {"type": "unit"}


def build_scaler(normalization: str):
    if normalization == "none":
        return None
    if normalization == "minmax":
        return MinMaxScaler()
    if normalization == "zscore":
        return StandardScaler()
    if normalization == "unit":
        return UnitLengthScaler()
    raise ValueError(f"Unsupported normalization: {normalization}")


def scaler_from_metadata(metadata: dict | None):
    if metadata is None:
        return None
    scaler_type = metadata.get("type")
    if scaler_type == "minmax":
        scaler = MinMaxScaler(a=float(metadata["a"]), b=float(metadata["b"]))
        scaler.x_min_ = np.array(metadata["x_min"], dtype=np.float32)
        scaler.x_max_ = np.array(metadata["x_max"], dtype=np.float32)
        return scaler
    if scaler_type == "zscore":
        scaler = StandardScaler()
        scaler.mean_ = np.array(metadata["mean"], dtype=np.float32)
        scaler.std_ = np.array(metadata["std"], dtype=np.float32)
        return scaler
    if scaler_type == "unit":
        return UnitLengthScaler()
    return None
