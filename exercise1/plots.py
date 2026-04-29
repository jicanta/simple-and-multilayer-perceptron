from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def _plt():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _dir() -> Path:
    PLOTS_DIR.mkdir(exist_ok=True)
    return PLOTS_DIR


def save_loss_comparison(
    filename: str, linear_losses: list[float], nonlinear_losses: list[float]
) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    path = _dir() / filename
    fig, ax = plt.subplots()
    ax.plot(linear_losses, label="linear", color="tab:blue")
    ax.plot(nonlinear_losses, label="non-linear (sigmoid)", color="tab:orange")
    ax.set_title("Learning Curve Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_loss_plot(filename: str, losses: list[float], title: str) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    path = _dir() / filename
    fig, ax = plt.subplots()
    ax.plot(losses, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_roc_curve(
    filename: str, y_true: np.ndarray, y_scores: np.ndarray, title: str
) -> tuple[Path | None, float]:
    plt = _plt()

    thresholds = np.linspace(0, 1, 300)
    tprs, fprs = [], []
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    order = np.argsort(fprs_arr)
    auc = float(np.trapz(tprs_arr[order], fprs_arr[order]))

    if plt is None:
        return None, auc

    path = _dir() / filename
    fig, ax = plt.subplots()
    ax.plot(fprs, tprs, color="tab:blue", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path, auc


def save_feature_engineering_plot(
    filename: str,
    labels: list[str],
    f1_means: list[float],
    f1_stds: list[float],
    baseline_f1: float,
) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    path = _dir() / filename

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [
        "tab:gray" if lbl == "Baseline (9)" else
        "tab:blue" if lbl == "All engineered (+6)" else
        "tab:orange"
        for lbl in labels
    ]
    bars = ax.barh(labels, f1_means, xerr=f1_stds, color=colors,
                   alpha=0.85, capsize=4, error_kw={"elinewidth": 1.5})
    ax.axvline(baseline_f1, color="tab:gray", linestyle="--", linewidth=1.2,
               label=f"Baseline F1 = {baseline_f1:.4f}")
    for bar, mean in zip(bars, f1_means):
        ax.text(mean + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{mean:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Mean F1 (K-Fold)")
    ax.set_title("Ex1 — Feature Engineering Impact on F1")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_activation_comparison(
    filename: str,
    sigmoid_losses: list[float],
    relu_losses: list[float],
    sigmoid_outputs: np.ndarray,
    relu_outputs: np.ndarray,
) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    path = _dir() / filename
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(sigmoid_losses, label="sigmoid", color="tab:blue")
    axes[0].plot(relu_losses, label="relu", color="tab:orange")
    axes[0].set_title("Training Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    all_vals = np.concatenate([sigmoid_outputs, relu_outputs])
    bins = np.linspace(all_vals.min(), max(all_vals.max(), 1.05), 50)
    axes[1].hist(sigmoid_outputs, bins=bins, alpha=0.6, label="sigmoid", color="tab:blue")
    axes[1].hist(relu_outputs, bins=bins, alpha=0.6, label="relu", color="tab:orange")
    axes[1].axvline(0.5, color="red", linestyle="--", alpha=0.6, label="threshold = 0.5")
    axes[1].set_title("Output Score Distribution (full dataset)")
    axes[1].set_xlabel("Predicted score")
    axes[1].set_ylabel("Sample count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Ex1 — Sigmoid vs ReLU Activation Comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_threshold_analysis(
    filename: str, y_true: np.ndarray, y_scores: np.ndarray, title: str
) -> tuple[Path | None, float]:
    plt = _plt()

    thresholds = np.linspace(0.01, 0.99, 300)
    precisions, recalls, f1s = [], [], []
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    best_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])

    if plt is None:
        return None, best_threshold

    path = _dir() / filename
    fig, ax = plt.subplots()
    ax.plot(thresholds, precisions, label="precision", color="tab:blue")
    ax.plot(thresholds, recalls, label="recall", color="tab:orange")
    ax.plot(thresholds, f1s, label="F1", color="tab:green")
    ax.axvline(
        best_threshold,
        color="tab:red",
        linestyle="--",
        label=f"best F1 @ {best_threshold:.2f}",
    )
    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path, best_threshold
