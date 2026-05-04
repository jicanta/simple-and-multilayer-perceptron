"""
Exercise 3 — Optional analysis

A. Noise robustness
   For each Gaussian noise level σ, add noise to digits_test.csv and evaluate
   all trained models. Plots accuracy and F1-macro vs σ, plus a per-class
   breakdown for the best model.

B. Attribution / Interpretability
   1. Saliency maps  — mean |∂output_k / ∂pixel_i| over correctly classified
      test samples, one map per digit class. Shows which pixels the model
      "looks at" to decide each digit.
   2. First-layer weight visualization — each hidden neurons' weight
      vector reshaped to 28×28. Shows the primitive patterns the hidden layer
      has learned to detect.

Usage examples
--------------
    python3 exercise3/analysis.py
    python3 exercise3/analysis.py --noise-levels 0,0.1,0.3,0.5
    python3 exercise3/analysis.py --skip-noise
    python3 exercise3/analysis.py --skip-attribution
    python3 exercise3/analysis.py --noise-repeats 5 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Import local exercise3/data.py BEFORE inserting exercise2 into sys.path,
# otherwise Python resolves "data" to exercise2/data.py.
from data import build_scaler, load_combined, load_test  # noqa: E402

_EX2 = Path(__file__).resolve().parent.parent / "exercise2"
sys.path.insert(0, str(_EX2))

from mlp import MultilayerPerceptron  # noqa: E402
from metrics import classification_metrics, save_metrics_json  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent / "models"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_MODEL_SLUGS = [
    ("1_baseline", "1-Baseline"),
    ("2_weighted_loss", "2-Weighted-Loss"),
    ("3_weighted_sampling", "3-Weighted-Sampling"),
    ("4_synthetic_balancing", "4-Synthetic-Balancing"),
    ("5_smote", "5-SMOTE"),
]

N_CLASSES = 10
DEFAULT_NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
DEFAULT_PER_CLASS_SIGMAS = [0.0, 0.2, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def _parse_float_list(raw: str) -> list[float]:
    try:
        return [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of floats, e.g. 0,0.05,0.1,0.2."
        ) from exc


def _load_models() -> list[tuple[str, MultilayerPerceptron]]:
    missing = [slug for slug, _ in _MODEL_SLUGS
               if not (MODELS_DIR / f"{slug}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"Could not find the following Exercise 3 models in {MODELS_DIR}:\n"
            + "\n".join(f"  {slug}.npz" for slug in missing)
            + "\nRun `python3 exercise3/train.py` first."
        )
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
    noise_levels: list[float],
    repeats: int,
    rng: np.random.Generator,
) -> dict:
    """Evaluate each model at each noise level. Returns nested results dict."""
    results: dict[str, dict] = {
        label: {"sigma": noise_levels, "accuracy": [], "f1": [], "accuracy_std": [], "f1_std": []}
        for _, label in _MODEL_SLUGS
    }

    print(f"\n  {'sigma':>6} | " + " | ".join(f"{label:>22}" for _, label in _MODEL_SLUGS))
    print("  " + "-" * (10 + 27 * len(_MODEL_SLUGS)))

    for sigma in noise_levels:
        row = f"  {sigma:>6.2f} |"
        for label, model in models:
            acc_runs, f1_runs = [], []
            for _ in range(repeats):
                noise = rng.normal(0.0, sigma, X_test.shape).astype(np.float32)
                X_noisy = np.clip(X_test + noise, 0.0, 1.0)
                preds = model.predict(X_noisy)
                m = classification_metrics(y_test, preds, num_classes=N_CLASSES)
                acc_runs.append(m["accuracy"])
                f1_runs.append(m["f1_macro"])
            results[label]["accuracy"].append(float(np.mean(acc_runs)))
            results[label]["accuracy_std"].append(float(np.std(acc_runs)))
            results[label]["f1"].append(float(np.mean(f1_runs)))
            results[label]["f1_std"].append(float(np.std(f1_runs)))
            row += f" {label}: {np.mean(acc_runs):>6.2%} F1={np.mean(f1_runs):.4f} |"
        print(row)

    return results


def plot_noise_curves(results: dict, noise_levels: list[float]) -> None:
    plt = _plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for (_, label), color in zip(_MODEL_SLUGS, colors):
        r = results[label]
        axes[0].plot(noise_levels, [a * 100 for a in r["accuracy"]],
                     marker="o", label=label, color=color)
        axes[1].plot(noise_levels, r["f1"],
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
    sigmas: list[float],
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
    width = 0.8 / max(1, len(sigmas))
    colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:green"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (sigma, color) in enumerate(zip(sigmas, colors)):
        ax.bar(x + i * width, per_class[sigma], width,
               label=f"σ={sigma}", color=color, alpha=0.85)

    center = width * (len(sigmas) - 1) / 2
    ax.set_xticks(x + center)
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
    return model.input_gradients(X, class_idx)


def plot_saliency_maps(
    models: list[tuple[str, MultilayerPerceptron]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_per_class: int,
) -> None:
    plt = _plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for label, model in models:
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))
        axes_flat = axes.flatten()
        preds = model.predict(X_test)

        for cls in range(N_CLASSES):
            mask = (y_test == cls) & (preds == cls)
            X_cls = X_test[mask][:max_per_class]

            ax = axes_flat[cls]
            if len(X_cls) == 0:
                ax.set_title(f"Digit {cls}\n(none correct)")
                ax.axis("off")
                continue

            sal = compute_saliency(model, X_cls, cls)
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
        W = model.weights[0][:-1, :]   # (784, hidden) — drop bias row
        n_hidden = W.shape[1]
        ncols = 16
        nrows = (n_hidden + ncols - 1) // ncols
        vmax = np.abs(W).max()

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.1, nrows * 1.1))
        axes_flat = np.atleast_1d(axes).flatten()

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
            f"Ex3 — {label} — First-layer weights ({n_hidden} neurons, 28×28 receptive fields)\n"
            "Red = positive weight, Blue = negative weight",
            fontsize=9,
        )
        fig.tight_layout()
        path = PLOTS_DIR / f"weights_layer1_{slug}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  Saved: {path.relative_to(Path(__file__).parent)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise 3 — Optional analysis (noise robustness + interpretability)"
    )
    parser.add_argument(
        "--noise-levels",
        type=_parse_float_list,
        default=DEFAULT_NOISE_LEVELS,
        help=(
            "Comma-separated Gaussian noise sigmas to evaluate. "
            f"Default: {','.join(str(s) for s in DEFAULT_NOISE_LEVELS)}"
        ),
    )
    parser.add_argument(
        "--noise-repeats",
        type=int,
        default=3,
        help="Noisy realizations to average per sigma (default: 3).",
    )
    parser.add_argument(
        "--per-class-sigmas",
        type=_parse_float_list,
        default=DEFAULT_PER_CLASS_SIGMAS,
        help=(
            "Subset of sigmas shown in the per-class breakdown plot. "
            f"Default: {','.join(str(s) for s in DEFAULT_PER_CLASS_SIGMAS)}"
        ),
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=100,
        help="Max correctly-classified test samples per class used for saliency maps (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise generation (default: 42).",
    )
    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Skip the noise-robustness analysis.",
    )
    parser.add_argument(
        "--skip-attribution",
        action="store_true",
        help="Skip saliency maps and first-layer weight visualizations.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.noise_repeats <= 0:
        raise ValueError("--noise-repeats must be greater than 0.")

    rng = np.random.default_rng(args.seed)

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
    if not args.skip_noise:
        _section("A — Noise Robustness")
        print(f"  noise levels : {args.noise_levels}")
        print(f"  repeats/sigma: {args.noise_repeats}")

        noise_results = run_noise_robustness(
            models, X_test, y_test,
            noise_levels=args.noise_levels,
            repeats=args.noise_repeats,
            rng=rng,
        )
        plot_noise_curves(noise_results, args.noise_levels)

        # Per-class breakdown for the best model (weighted sampling)
        best_label, best_model = models[-1]
        print(f"\n  Per-class breakdown for best model ({best_label}):")
        plot_noise_per_class(
            best_model, best_label, X_test, y_test, rng,
            sigmas=args.per_class_sigmas,
        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_metrics_json(
            RESULTS_DIR / "noise_robustness.json",
            {"noise_levels": args.noise_levels, "repeats": args.noise_repeats, "results": noise_results},
        )
    else:
        print("\n  [skipped] noise robustness (--skip-noise)")

    # -------------------------------------------------------------------
    # B. Attribution / Interpretability
    # -------------------------------------------------------------------
    if not args.skip_attribution:
        _section("B — Attribution / Interpretability")

        print("\n  Saliency maps (all models)...")
        plot_saliency_maps(models, X_test, y_test, max_per_class=args.max_samples_per_class)

        print("\n  First-layer weight visualizations (all models)...")
        plot_first_layer_weights(models)
    else:
        print("\n  [skipped] attribution / interpretability (--skip-attribution)")

    _section("Done")
    print("  All plots saved to exercise3/plots/\n")
