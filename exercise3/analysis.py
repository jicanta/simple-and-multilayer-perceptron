"""
Exercise 3 — Optional analysis

A. Noise robustness
   For each Gaussian noise level σ, add noise to digits_test.csv and evaluate
   all three trained models. Plots accuracy and F1-macro vs σ, plus a per-class
   breakdown for the best model.

B. Attribution / Interpretability
   1. Saliency maps  — mean |∂output_k / ∂pixel_i| over correctly classified
      test samples, one map per digit class. Shows which pixels the model
      "looks at" to decide each digit.
   2. First-layer weight visualization — each of the 128 hidden neurons' weight
      vectors reshaped to 28×28. Shows the primitive patterns the hidden layer
      has learned to detect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import local exercise3/data.py BEFORE inserting exercise2 into sys.path,
# otherwise Python resolves "data" to exercise2/data.py.
from data import build_scaler, load_combined, load_test  # noqa: E402

_EX2 = Path(__file__).resolve().parent.parent / "exercise2"
sys.path.insert(0, str(_EX2))

from mlp import MultilayerPerceptron  # noqa: E402
from metrics import classification_metrics  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent / "models"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

_MODEL_SLUGS = [
    ("1_baseline", "1-Baseline"),
    ("2_weighted_loss", "2-Weighted-Loss"),
    ("3_weighted_sampling", "3-Weighted-Sampling"),
]

N_CLASSES = 10
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def _load_models() -> list[tuple[str, MultilayerPerceptron]]:
    return [
        (label, MultilayerPerceptron.load(MODELS_DIR / f"{slug}.npz"))
        for slug, label in _MODEL_SLUGS
    ]


def _plt():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# A. Noise robustness
# ---------------------------------------------------------------------------

def run_noise_robustness(
    models: list[tuple[str, MultilayerPerceptron]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Evaluate each model at each noise level. Returns nested results dict."""
    results: dict[str, dict] = {
        label: {"sigma": NOISE_LEVELS, "accuracy": [], "f1": []}
        for _, label in _MODEL_SLUGS
    }

    print(f"\n  {'sigma':>6} | " + " | ".join(f"{label:>22}" for _, label in _MODEL_SLUGS))
    print("  " + "-" * (10 + 27 * len(_MODEL_SLUGS)))

    for sigma in NOISE_LEVELS:
        noise = rng.normal(0.0, sigma, X_test.shape).astype(np.float32)
        X_noisy = np.clip(X_test + noise, 0.0, 1.0)

        row = f"  {sigma:>6.2f} |"
        for label, model in models:
            preds = model.predict(X_noisy)
            m = classification_metrics(y_test, preds, num_classes=N_CLASSES)
            results[label]["accuracy"].append(m["accuracy"])
            results[label]["f1"].append(m["f1_macro"])
            row += f" {label}: {m['accuracy']:>6.2%} F1={m['f1_macro']:.4f} |"
        print(row)

    return results


def plot_noise_curves(results: dict) -> None:
    plt = _plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for (_, label), color in zip(_MODEL_SLUGS, colors):
        r = results[label]
        axes[0].plot(r["sigma"], [a * 100 for a in r["accuracy"]],
                     marker="o", label=label, color=color)
        axes[1].plot(r["sigma"], r["f1"],
                     marker="o", label=label, color=color)

    for ax, ylabel, title in zip(
        axes,
        ["Accuracy (%)", "F1 Macro"],
        ["Accuracy vs Gaussian noise", "F1 Macro vs Gaussian noise"],
    ):
        ax.set_title(title)
        ax.set_xlabel("Noise σ (added to MinMax-scaled pixels)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ex3 — Noise Robustness (digits_test.csv)", fontsize=12)
    fig.tight_layout()
    path = PLOTS_DIR / "noise_robustness.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path.relative_to(Path(__file__).parent)}")


def plot_noise_per_class(
    model: MultilayerPerceptron,
    label: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rng: np.random.Generator,
    sigmas: tuple[float, ...] = (0.0, 0.2, 0.5, 1.0),
) -> None:
    plt = _plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    per_class: dict[float, list[float]] = {}
    for sigma in sigmas:
        noise = rng.normal(0.0, sigma, X_test.shape).astype(np.float32)
        X_noisy = np.clip(X_test + noise, 0.0, 1.0)
        preds = model.predict(X_noisy)
        accs = []
        for cls in range(N_CLASSES):
            mask = y_test == cls
            accs.append((preds[mask] == cls).mean() if mask.sum() > 0 else 0.0)
        per_class[sigma] = accs

    x = np.arange(N_CLASSES)
    width = 0.18
    colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (sigma, color) in enumerate(zip(sigmas, colors)):
        ax.bar(x + i * width, per_class[sigma], width,
               label=f"σ={sigma}", color=color, alpha=0.85)

    ax.set_xticks(x + width * (len(sigmas) - 1) / 2)
    ax.set_xticklabels([str(c) for c in range(N_CLASSES)])
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Per-class accuracy")
    ax.set_title(f"Ex3 — {label} — Per-class accuracy under noise")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    slug = label.lower().replace(" ", "_").replace("-", "_")
    path = PLOTS_DIR / f"noise_per_class_{slug}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.relative_to(Path(__file__).parent)}")


# ---------------------------------------------------------------------------
# B. Attribution / Interpretability
# ---------------------------------------------------------------------------

def compute_saliency(
    model: MultilayerPerceptron,
    X: np.ndarray,
    class_idx: int,
) -> np.ndarray:
    """
    Input-space gradient of output[class_idx] w.r.t. each input pixel.

    Backpropagates a unit signal at output neuron `class_idx` through the
    activation derivatives and weight matrices down to the input layer.
    Works for any network depth.

    Returns array of shape (n_samples, n_pixels).
    """
    activations = model.forward(X)

    # Seed the delta at the output: f'(a_out) at class_idx, zeros elsewhere
    a_out = activations[-1]                      # (batch, 10)
    delta = np.zeros_like(a_out)
    delta[:, class_idx] = model._activate_derivative(a_out)[:, class_idx]

    # Propagate backward through all hidden layers (stop before input)
    n_layers = len(model.weights)
    for layer_idx in range(n_layers - 1, 0, -1):
        W = model.weights[layer_idx][:-1, :]     # drop bias row: (in, out)
        delta_prev = delta @ W.T                  # (batch, in)
        delta = delta_prev * model._activate_derivative(activations[layer_idx])

    # Project from first hidden layer back to input pixels
    W0 = model.weights[0][:-1, :]               # (784, hidden)
    saliency = delta @ W0.T                      # (batch, 784)
    return saliency


def plot_saliency_maps(
    models: list[tuple[str, MultilayerPerceptron]],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    plt = _plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for label, model in models:
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))
        axes_flat = axes.flatten()

        for cls in range(N_CLASSES):
            # Only correctly classified samples
            preds = model.predict(X_test)
            mask = (y_test == cls) & (preds == cls)
            X_cls = X_test[mask]

            ax = axes_flat[cls]
            if len(X_cls) == 0:
                ax.set_title(f"Digit {cls}\n(none correct)")
                ax.axis("off")
                continue

            sal = compute_saliency(model, X_cls, cls)   # (n, 784)
            mean_sal = np.abs(sal).mean(axis=0).reshape(28, 28)

            im = ax.imshow(mean_sal, cmap="hot", interpolation="nearest")
            ax.set_title(f"Digit {cls}  (n={mask.sum()})", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        slug = label.lower().replace(" ", "_").replace("-", "_")
        fig.suptitle(
            f"Ex3 — {label} — Saliency Maps\n"
            f"mean |∂output_k / ∂pixel| over correctly classified test samples",
            fontsize=11,
        )
        fig.tight_layout()
        path = PLOTS_DIR / f"saliency_{slug}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  Saved: {path.relative_to(Path(__file__).parent)}")


def plot_first_layer_weights(
    models: list[tuple[str, MultilayerPerceptron]],
) -> None:
    plt = _plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for label, model in models:
        W = model.weights[0][:-1, :]   # (784, 128) — drop bias row
        n_hidden = W.shape[1]
        ncols = 16
        nrows = (n_hidden + ncols - 1) // ncols
        vmax = np.abs(W).max()

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.1, nrows * 1.1))
        axes_flat = axes.flatten()

        for i in range(n_hidden):
            axes_flat[i].imshow(
                W[:, i].reshape(28, 28),
                cmap="RdBu_r",
                interpolation="nearest",
                vmin=-vmax,
                vmax=vmax,
            )
            axes_flat[i].axis("off")

        for i in range(n_hidden, len(axes_flat)):
            axes_flat[i].axis("off")

        slug = label.lower().replace(" ", "_").replace("-", "_")
        fig.suptitle(
            f"Ex3 — {label} — First-layer weights (128 neurons, 28×28 receptive fields)\n"
            "Red = positive weight, Blue = negative weight",
            fontsize=9,
        )
        fig.tight_layout()
        path = PLOTS_DIR / f"weights_layer1_{slug}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  Saved: {path.relative_to(Path(__file__).parent)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("Loading data...")
    X_all, _ = load_combined()      # training data — used to fit the scaler
    X_test_raw, y_test = load_test()

    scaler = build_scaler("minmax")
    scaler.fit(X_all)
    X_test = scaler.transform(X_test_raw)
    print(f"  test: {X_test.shape}  classes: {np.bincount(y_test).tolist()}")

    print("Loading models...")
    models = _load_models()
    for label, m in models:
        print(f"  {label}: {m.layer_sizes}  best_epoch={m.best_epoch}")

    # -------------------------------------------------------------------
    # A. Noise robustness
    # -------------------------------------------------------------------
    _section("A — Noise Robustness")
    noise_results = run_noise_robustness(models, X_test, y_test, rng)
    plot_noise_curves(noise_results)

    # Per-class breakdown for the best model (weighted sampling)
    best_label, best_model = models[-1]
    print(f"\n  Per-class breakdown for best model ({best_label}):")
    plot_noise_per_class(best_model, best_label, X_test, y_test, rng)

    # -------------------------------------------------------------------
    # B. Attribution / Interpretability
    # -------------------------------------------------------------------
    _section("B — Attribution / Interpretability")

    print("\n  Saliency maps (all 3 models)...")
    plot_saliency_maps(models, X_test, y_test)

    print("\n  First-layer weight visualizations (all 3 models)...")
    plot_first_layer_weights(models)

    _section("Done")
    print("  All plots saved to exercise3/plots/\n")
