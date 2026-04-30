"""
Exercise 2 — Optional analysis

A. Noise robustness
   Evaluates the chosen Exercise 2 model on digits_test.csv after injecting
   Gaussian noise into the raw images. The goal is to inspect whether the
   generalization performance degrades gracefully as the perturbation grows.

B. Interpretability with attribution methods
   Generates per-class attribution heatmaps using gradient-based methods:
     - Gradient·Input
     - Integrated Gradients
   Both are computed over correctly classified test samples and then averaged
   to highlight the pixels that contribute most to each predicted digit.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from data import load_digits_dataset, scaler_from_metadata
from metrics import classification_metrics, save_metrics_json
from mlp import MultilayerPerceptron


RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
DEFAULT_NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
DEFAULT_PER_CLASS_SIGMAS = [0.0, 0.1, 0.2, 0.5]
DEFAULT_ATTRIBUTION_METHODS = ("gradient_input", "integrated_gradients")
VALID_ATTRIBUTION_METHODS = {"gradients", "gradient_input", "integrated_gradients"}


def _section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def _plt():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    return plt


def _parse_float_list(raw: str) -> list[float]:
    try:
        return [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of floats, e.g. 0,0.05,0.1,0.2."
        ) from exc


def _resolve_model_path(model_arg: Path | None) -> Path:
    if model_arg is not None:
        candidate = model_arg
        if candidate.suffix != ".npz":
            candidate = candidate.with_suffix(".npz")
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find model file: {candidate}")
        return candidate

    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        best_name = summary.get("best_model")
        for row in summary.get("results", []):
            if row.get("name") == best_name:
                return Path(row["model_path"]).resolve()

    evaluated_runs = sorted(
        RESULTS_DIR.glob("*-test.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for result_path in evaluated_runs:
        payload = json.loads(result_path.read_text())
        model_path = payload.get("model_path")
        if model_path:
            return Path(model_path).resolve()

    models = sorted((Path(__file__).resolve().parent / "models").glob("*.npz"))
    if len(models) == 1:
        return models[0].resolve()

    raise FileNotFoundError(
        "Could not infer which Exercise 2 model to analyze. "
        "Use --model /path/to/model.npz."
    )


def _load_model_bundle(model_path: Path) -> tuple[MultilayerPerceptron, dict, str]:
    model = MultilayerPerceptron.load(model_path)
    metadata = MultilayerPerceptron.load_metadata(model_path)
    label = metadata.get("config", {}).get("name", model_path.stem)
    return model, metadata, label


def _prepare_test_data(metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_test_raw, y_test = load_digits_dataset("digits_test.csv")
    scaler = scaler_from_metadata(metadata.get("scaler"))
    X_test = scaler.transform(X_test_raw) if scaler is not None else X_test_raw.copy()
    return X_test_raw, X_test, y_test


def _infer_image_shape(n_features: int) -> tuple[int, int]:
    side = int(np.sqrt(n_features))
    if side * side != n_features:
        raise ValueError(
            f"Cannot reshape {n_features} features into a square image for visualization."
        )
    return side, side


def _sample_correct_examples(
    model: MultilayerPerceptron,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    preds = model.predict(X_test)
    selected: dict[int, np.ndarray] = {}

    for class_idx in range(model.layer_sizes[-1]):
        valid = np.flatnonzero((y_test == class_idx) & (preds == class_idx))
        if len(valid) == 0:
            selected[class_idx] = np.empty((0,), dtype=np.int64)
            continue
        if len(valid) > max_per_class:
            valid = np.sort(rng.choice(valid, size=max_per_class, replace=False))
        selected[class_idx] = valid
    return selected


def evaluate_noise_robustness(
    model: MultilayerPerceptron,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    metadata: dict,
    noise_levels: list[float],
    repeats: int,
    rng: np.random.Generator,
) -> tuple[list[dict], dict[float, np.ndarray]]:
    scaler = scaler_from_metadata(metadata.get("scaler"))
    raw_min = float(X_test_raw.min())
    raw_max = float(X_test_raw.max())
    per_sigma_rows: list[dict] = []
    per_class_by_sigma: dict[float, np.ndarray] = {}

    print(f"\n  {'sigma':>6} {'acc_mean':>10} {'acc_std':>9} {'f1_mean':>10} {'f1_std':>9}")
    print("  " + "-" * 48)

    for sigma in noise_levels:
        acc_runs: list[float] = []
        f1_runs: list[float] = []
        per_class_runs: list[np.ndarray] = []

        for _ in range(repeats):
            noise = rng.normal(0.0, sigma, size=X_test_raw.shape).astype(np.float32)
            X_noisy_raw = np.clip(X_test_raw + noise, raw_min, raw_max)
            X_noisy = scaler.transform(X_noisy_raw) if scaler is not None else X_noisy_raw

            preds = model.predict(X_noisy)
            metrics = classification_metrics(y_test, preds, num_classes=model.layer_sizes[-1])
            acc_runs.append(metrics["accuracy"])
            f1_runs.append(metrics["f1_macro"])
            per_class_runs.append(
                np.array([row["recall"] for row in metrics["per_class"]], dtype=np.float32)
            )

        per_class_mean = np.mean(per_class_runs, axis=0)
        per_class_by_sigma[float(sigma)] = per_class_mean
        row = {
            "sigma": float(sigma),
            "accuracy_mean": float(np.mean(acc_runs)),
            "accuracy_std": float(np.std(acc_runs)),
            "f1_mean": float(np.mean(f1_runs)),
            "f1_std": float(np.std(f1_runs)),
            "per_class_recall": per_class_mean.tolist(),
        }
        per_sigma_rows.append(row)
        print(
            f"  {sigma:>6.2f} {row['accuracy_mean']:>10.2%} {row['accuracy_std']:>9.2%} "
            f"{row['f1_mean']:>10.4f} {row['f1_std']:>9.4f}"
        )

    return per_sigma_rows, per_class_by_sigma


def plot_noise_curves(
    label: str,
    noise_rows: list[dict],
    output_prefix: str,
) -> Path:
    plt = _plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sigma = [row["sigma"] for row in noise_rows]
    accuracy = np.array([row["accuracy_mean"] for row in noise_rows], dtype=np.float32)
    accuracy_std = np.array([row["accuracy_std"] for row in noise_rows], dtype=np.float32)
    f1 = np.array([row["f1_mean"] for row in noise_rows], dtype=np.float32)
    f1_std = np.array([row["f1_std"] for row in noise_rows], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(sigma, accuracy * 100.0, marker="o", color="tab:blue")
    axes[0].fill_between(
        sigma,
        (accuracy - accuracy_std) * 100.0,
        (accuracy + accuracy_std) * 100.0,
        color="tab:blue",
        alpha=0.18,
    )
    axes[0].set_title("Accuracy vs Gaussian noise")
    axes[0].set_xlabel("Noise sigma")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sigma, f1, marker="o", color="tab:orange")
    axes[1].fill_between(
        sigma,
        f1 - f1_std,
        f1 + f1_std,
        color="tab:orange",
        alpha=0.18,
    )
    axes[1].set_title("F1 macro vs Gaussian noise")
    axes[1].set_xlabel("Noise sigma")
    axes[1].set_ylabel("F1 macro")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Exercise 2 — Noise robustness — {label}", fontsize=12)
    fig.tight_layout()

    path = PLOTS_DIR / f"{output_prefix}_noise_robustness.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_noise_per_class(
    label: str,
    per_class_by_sigma: dict[float, np.ndarray],
    sigmas: list[float],
    output_prefix: str,
) -> Path:
    plt = _plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    available_sigmas = [float(sigma) for sigma in sigmas if float(sigma) in per_class_by_sigma]
    if not available_sigmas:
        available_sigmas = sorted(per_class_by_sigma)[: min(4, len(per_class_by_sigma))]

    x = np.arange(len(next(iter(per_class_by_sigma.values()))))
    width = 0.8 / max(1, len(available_sigmas))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, sigma in enumerate(available_sigmas):
        ax.bar(
            x + idx * width,
            per_class_by_sigma[float(sigma)],
            width=width,
            label=f"sigma={sigma:g}",
            color=colors[idx % len(colors)],
            alpha=0.85,
        )

    center_offset = width * (len(available_sigmas) - 1) / 2
    ax.set_xticks(x + center_offset)
    ax.set_xticklabels([str(class_idx) for class_idx in x])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Per-class recall")
    ax.set_title(f"Exercise 2 — Per-class robustness — {label}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()
    path = PLOTS_DIR / f"{output_prefix}_noise_per_class.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_attribution_maps(
    model: MultilayerPerceptron,
    label: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str,
    max_per_class: int,
    integrated_steps: int,
    rng: np.random.Generator,
    output_prefix: str,
) -> tuple[Path, dict[int, int]]:
    plt = _plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sample_indices = _sample_correct_examples(model, X_test, y_test, max_per_class, rng)
    image_shape = _infer_image_shape(X_test.shape[1])
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes_flat = axes.flatten()
    counts: dict[int, int] = {}

    for class_idx, ax in enumerate(axes_flat):
        indices = sample_indices[class_idx]
        counts[class_idx] = int(len(indices))

        if len(indices) == 0:
            ax.set_title(f"Digit {class_idx}\n(no correct samples)", fontsize=9)
            ax.axis("off")
            continue

        X_class = X_test[indices]
        kwargs = {"steps": integrated_steps} if method == "integrated_gradients" else {}
        attribution = model.attribution(X_class, class_idx, method=method, **kwargs)
        heatmap = np.abs(attribution).mean(axis=0).reshape(image_shape)

        image = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
        ax.set_title(f"Digit {class_idx} (n={len(indices)})", fontsize=9)
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    title_method = {
        "gradients": "Input gradients",
        "gradient_input": "Gradient·Input",
        "integrated_gradients": f"Integrated Gradients ({integrated_steps} steps)",
    }.get(method, method)

    fig.suptitle(
        f"Exercise 2 — {label} — {title_method}\n"
        "Mean absolute attribution over correctly classified test samples",
        fontsize=11,
    )
    fig.tight_layout()

    path = PLOTS_DIR / f"{output_prefix}_{method}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path, counts


def plot_first_layer_weights(
    model: MultilayerPerceptron,
    label: str,
    output_prefix: str,
) -> Path:
    plt = _plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    image_shape = _infer_image_shape(model.layer_sizes[0])
    weights = model.weights[0][:-1, :]
    n_hidden = weights.shape[1]
    ncols = min(16, n_hidden)
    nrows = (n_hidden + ncols - 1) // ncols
    vmax = float(np.abs(weights).max())

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.1, nrows * 1.1))
    axes_flat = np.atleast_1d(axes).flatten()

    for neuron_idx in range(n_hidden):
        axes_flat[neuron_idx].imshow(
            weights[:, neuron_idx].reshape(image_shape),
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        axes_flat[neuron_idx].axis("off")

    for neuron_idx in range(n_hidden, len(axes_flat)):
        axes_flat[neuron_idx].axis("off")

    fig.suptitle(
        f"Exercise 2 — {label} — First-layer receptive fields\n"
        "Red = positive contribution, Blue = negative contribution",
        fontsize=10,
    )
    fig.tight_layout()

    path = PLOTS_DIR / f"{output_prefix}_weights_layer1.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 2 optional analysis")
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to the saved Exercise 2 model (.npz). If omitted, the script tries to infer the best run.",
    )
    parser.add_argument(
        "--noise-levels",
        type=_parse_float_list,
        default=DEFAULT_NOISE_LEVELS,
        help="Comma-separated Gaussian noise sigmas to evaluate.",
    )
    parser.add_argument(
        "--noise-repeats",
        type=int,
        default=5,
        help="How many noisy realizations to average for each sigma.",
    )
    parser.add_argument(
        "--per-class-sigmas",
        type=_parse_float_list,
        default=DEFAULT_PER_CLASS_SIGMAS,
        help="Subset of sigmas to include in the per-class robustness plot.",
    )
    parser.add_argument(
        "--attribution-methods",
        default=",".join(DEFAULT_ATTRIBUTION_METHODS),
        help="Comma-separated list of attribution methods: gradients, gradient_input, integrated_gradients.",
    )
    parser.add_argument(
        "--integrated-steps",
        type=int,
        default=32,
        help="Interpolation steps for integrated gradients.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=50,
        help="Maximum number of correctly classified test samples used per class in attribution maps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise generation and attribution sampling.",
    )
    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Skip the robustness-to-noise analysis.",
    )
    parser.add_argument(
        "--skip-attribution",
        action="store_true",
        help="Skip attribution heatmaps and first-layer visualizations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.noise_repeats <= 0:
        raise ValueError("--noise-repeats must be greater than 0.")

    rng = np.random.default_rng(args.seed)

    model_path = _resolve_model_path(args.model)
    model, metadata, label = _load_model_bundle(model_path)
    output_prefix = (
        label.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )
    methods = [part.strip().lower() for part in args.attribution_methods.split(",") if part.strip()]
    invalid_methods = [method for method in methods if method not in VALID_ATTRIBUTION_METHODS]
    if invalid_methods:
        raise ValueError(
            "Unsupported attribution methods: "
            + ", ".join(invalid_methods)
            + ". Valid choices are: gradients, gradient_input, integrated_gradients."
        )

    _section("Exercise 2 — Optional Analysis")
    print(f"Model: {label}")
    print(f"Path:  {model_path}")
    print(f"Arch:  {model.layer_sizes}")

    X_test_raw, X_test, y_test = _prepare_test_data(metadata)
    clean_eval = classification_metrics(y_test, model.predict(X_test), num_classes=model.layer_sizes[-1])
    print(f"Clean test accuracy: {clean_eval['accuracy']:.2%}")
    print(f"Clean test F1:       {clean_eval['f1_macro']:.4f}")

    artifacts: dict[str, object] = {
        "model_path": str(model_path),
        "model_name": label,
        "clean_test": {
            "accuracy": clean_eval["accuracy"],
            "f1_macro": clean_eval["f1_macro"],
            "precision_macro": clean_eval["precision_macro"],
            "recall_macro": clean_eval["recall_macro"],
        },
    }

    if not args.skip_noise:
        _section("A — Noise Robustness")
        noise_rows, per_class_by_sigma = evaluate_noise_robustness(
            model=model,
            X_test_raw=X_test_raw,
            y_test=y_test,
            metadata=metadata,
            noise_levels=args.noise_levels,
            repeats=args.noise_repeats,
            rng=rng,
        )
        curve_path = plot_noise_curves(label, noise_rows, output_prefix)
        class_path = plot_noise_per_class(label, per_class_by_sigma, args.per_class_sigmas, output_prefix)

        artifacts["noise_robustness"] = {
            "noise_levels": noise_rows,
            "repeats": args.noise_repeats,
            "plots": {
                "curves": str(curve_path),
                "per_class": str(class_path),
            },
        }
        print(f"\nSaved: {curve_path.relative_to(Path(__file__).parent)}")
        print(f"Saved: {class_path.relative_to(Path(__file__).parent)}")

    if not args.skip_attribution:
        _section("B — Interpretability")
        attribution_artifacts: dict[str, object] = {}
        for method in methods:
            path, counts = plot_attribution_maps(
                model=model,
                label=label,
                X_test=X_test,
                y_test=y_test,
                method=method,
                max_per_class=args.max_samples_per_class,
                integrated_steps=args.integrated_steps,
                rng=rng,
                output_prefix=output_prefix,
            )
            attribution_artifacts[method] = {
                "plot": str(path),
                "samples_per_class": counts,
            }
            print(f"Saved: {path.relative_to(Path(__file__).parent)}")

        weights_path = plot_first_layer_weights(model, label, output_prefix)
        attribution_artifacts["first_layer_weights"] = str(weights_path)
        artifacts["interpretability"] = attribution_artifacts
        print(f"Saved: {weights_path.relative_to(Path(__file__).parent)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{output_prefix}_analysis.json"
    save_metrics_json(results_path, artifacts)

    _section("Done")
    print(f"Analysis summary: {results_path.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
