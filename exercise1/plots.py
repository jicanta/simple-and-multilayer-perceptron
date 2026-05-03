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


def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


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
    auc = _trapezoid_area(tprs_arr[order], fprs_arr[order])

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


def save_precision_recall_curve(
    filename: str, y_true: np.ndarray, y_scores: np.ndarray, title: str
) -> tuple[Path | None, float]:
    plt = _plt()

    thresholds = np.linspace(0, 1, 300)
    precisions, recalls = [], []
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    recalls_arr = np.array(recalls)
    precisions_arr = np.array(precisions)
    order = np.argsort(recalls_arr)
    pr_auc = _trapezoid_area(precisions_arr[order], recalls_arr[order])

    if plt is None:
        return None, pr_auc

    path = _dir() / filename
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions, color="tab:purple", label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(float(np.mean(y_true)), color="tab:gray", linestyle="--", alpha=0.6, label="class prevalence")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path, pr_auc


def save_confusion_matrix(
    filename: str, matrix: np.ndarray, title: str
) -> Path | None:
    plt = _plt()
    if plt is None:
        return None

    path = _dir() / filename
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not fraud", "Fraud"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not fraud", "Fraud"])

    total = matrix.sum()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            pct = 100.0 * value / total if total > 0 else 0.0
            ax.text(j, i, f"{value}\n({pct:.1f}%)", ha="center", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_calibration_plot(
    filename: str,
    scores_raw: np.ndarray,
    scores_cal: np.ndarray,
    labels: np.ndarray,
    ece_raw: float,
    ece_cal: float,
    n_bins: int = 10,
) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    path = _dir() / filename

    def _reliability_curve(scores, labels, n_bins):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_means, bin_fracs, bin_counts = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (scores >= lo) & (scores < hi)
            if mask.sum() > 0:
                bin_means.append(scores[mask].mean())
                bin_fracs.append(labels[mask].mean())
                bin_counts.append(mask.sum())
        return np.array(bin_means), np.array(bin_fracs), np.array(bin_counts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, scores, title, ece in zip(
        axes,
        [scores_raw, scores_cal],
        [f"Before calibration  (ECE={ece_raw:.4f})",
         f"After Platt scaling (ECE={ece_cal:.4f})"],
        [ece_raw, ece_cal],
    ):
        bm, bf, bc = _reliability_curve(scores, labels, n_bins)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
        ax.scatter(bm, bf, s=bc / bc.max() * 200, zorder=3)
        ax.plot(bm, bf, color="tab:blue", label="model")
        ax.fill_between(bm, bm, bf, alpha=0.15, color="tab:red", label="calibration gap")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives (actual fraud rate)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ex1 — Probability Calibration (reliability diagram)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


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
