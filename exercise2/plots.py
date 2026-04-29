from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import numpy as np


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def _plt():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _ensure_dir() -> Path:
    PLOTS_DIR.mkdir(exist_ok=True)
    return PLOTS_DIR


def _wrapped_title(title: str, width: int = 48) -> str:
    return "\n".join(textwrap.wrap(title, width=width))


def save_training_curves(filename: str, history: list[dict], title: str) -> Path | None:
    plt = _plt()
    if plt is None or not history:
        return None

    path = _ensure_dir() / filename
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    val_loss = [row.get("val_loss") for row in history]
    val_acc = [row.get("val_accuracy") for row in history]
    train_f1 = [row.get("train_f1") for row in history]
    val_f1 = [row.get("val_f1") for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(epochs, train_loss, label="train loss", color="tab:blue")
    if any(v is not None for v in val_loss):
        axes[0].plot(epochs, val_loss, label="val loss", color="tab:orange")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train acc", color="tab:green")
    if any(v is not None for v in val_acc):
        axes[1].plot(epochs, val_acc, label="val acc", color="tab:red")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if any(v is not None for v in train_f1):
        axes[2].plot(epochs, train_f1, label="train F1", color="tab:purple")
    if any(v is not None for v in val_f1):
        axes[2].plot(epochs, val_f1, label="val F1", color="tab:brown")
    axes[2].set_title("F1 Macro")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(_wrapped_title(title), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)
    return path


def save_confusion_matrix(filename: str, matrix: np.ndarray, title: str) -> Path | None:
    plt = _plt()
    if plt is None:
        return None

    path = _ensure_dir() / filename
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(_wrapped_title(title), fontsize=12, pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    return path
