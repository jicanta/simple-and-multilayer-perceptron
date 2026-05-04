from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EXERCISE3_DIR = ROOT / "exercise3"
RESULTS_DIR = EXERCISE3_DIR / "results"
PLOTS_DIR = EXERCISE3_DIR / "plots"
SLIDES_DIR = ROOT / "slides"
OVERVIEW_DIR = SLIDES_DIR / "00_overview"


def load_run_results() -> list[dict]:
    run_paths = sorted(
        [
            path for path in RESULTS_DIR.glob("*.json")
            if path.stem[:1].isdigit() and not path.stem.endswith("_analysis")
        ],
        key=lambda path: int(path.stem.split("_", 1)[0]),
    )
    runs = []
    for path in run_paths:
        with path.open() as handle:
            payload = json.load(handle)
        payload["slug"] = path.stem
        payload["result_path"] = path
        runs.append(payload)
    return runs


def copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def summarize_per_class(per_class: list[dict], top_k: int = 3) -> tuple[list[dict], list[dict]]:
    ranked = sorted(per_class, key=lambda item: item["f1"], reverse=True)
    best = ranked[:top_k]
    worst = list(reversed(ranked[-top_k:]))
    return best, worst


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def build_run_folder(run: dict) -> None:
    slug = run["slug"]
    run_dir = SLIDES_DIR / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    test_best, test_worst = summarize_per_class(run["test"]["per_class"])
    metrics_summary = {
        "name": run["name"],
        "mode": run["mode"],
        "best_epoch": run["best_epoch"],
        "elapsed_seconds": run["elapsed_seconds"],
        "train": {
            "accuracy": run["train"]["accuracy"],
            "f1_macro": run["train"]["f1_macro"],
            "loss": run["train"]["loss"],
        },
        "validation": {
            "accuracy": run["validation"]["accuracy"],
            "f1_macro": run["validation"]["f1_macro"],
            "loss": run["validation"]["loss"],
        },
        "test": {
            "accuracy": run["test"]["accuracy"],
            "f1_macro": run["test"]["f1_macro"],
            "precision_macro": run["test"]["precision_macro"],
            "recall_macro": run["test"]["recall_macro"],
        },
        "test_best_classes": test_best,
        "test_worst_classes": test_worst,
    }
    write_json(run_dir / "hyperparameters.json", run["config"])
    write_json(run_dir / "metrics_summary.json", metrics_summary)
    copy_if_exists(run["result_path"], run_dir / "full_result.json")

    copied_artifacts = []
    artifact_map = {
        "training_curves.png": PLOTS_DIR / f"{slug}_curves.png",
        "confusion_matrix.png": PLOTS_DIR / f"{slug}_confusion.png",
        "saliency_map.png": PLOTS_DIR / f"saliency_{slug}.png",
        "first_layer_weights.png": PLOTS_DIR / f"weights_layer1_{slug}.png",
    }
    for dest_name, src in artifact_map.items():
        if copy_if_exists(src, run_dir / dest_name):
            copied_artifacts.append(dest_name)

    readme_lines = [
        f"# {run['name']}",
        "",
        f"- Mode: `{run['mode']}`",
        f"- Best epoch: `{run['best_epoch']}`",
        f"- Elapsed: `{run['elapsed_seconds']:.1f}s`",
        f"- Test accuracy: `{run['test']['accuracy']:.2%}`",
        f"- Test F1 macro: `{run['test']['f1_macro']:.4f}`",
        f"- Validation accuracy: `{run['validation']['accuracy']:.2%}`",
        f"- Validation F1 macro: `{run['validation']['f1_macro']:.4f}`",
        "",
        "## Hyperparameters",
        "",
    ]
    for key, value in run["config"].items():
        readme_lines.append(f"- `{key}`: `{value}`")

    readme_lines.extend([
        "",
        "## Test Highlights",
        "",
        "Top classes by test F1:",
    ])
    for item in test_best:
        readme_lines.append(
            f"- class `{item['class']}`: F1 `{item['f1']:.4f}`, acc/recall `{item['recall']:.2%}`"
        )

    readme_lines.append("")
    readme_lines.append("Weakest classes by test F1:")
    for item in test_worst:
        readme_lines.append(
            f"- class `{item['class']}`: F1 `{item['f1']:.4f}`, acc/recall `{item['recall']:.2%}`"
        )

    readme_lines.extend([
        "",
        "## Included Files",
        "",
        "- `hyperparameters.json`",
        "- `metrics_summary.json`",
        "- `full_result.json`",
    ])
    for artifact in copied_artifacts:
        readme_lines.append(f"- `{artifact}`")

    (run_dir / "README.md").write_text("\n".join(readme_lines) + "\n")


def build_overview_plots(runs: list[dict]) -> None:
    labels = [run["name"] for run in runs]
    acc = [run["test"]["accuracy"] * 100 for run in runs]
    f1 = [run["test"]["f1_macro"] for run in runs]
    elapsed = [run["elapsed_seconds"] for run in runs]
    best_epochs = [run["best_epoch"] for run in runs]
    x = np.arange(len(runs))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x - width / 2, acc, width, label="Test accuracy (%)", color="tab:blue")
    axes[0].bar(x + width / 2, [score * 100 for score in f1], width, label="Test F1 macro (%)", color="tab:orange")
    axes[0].set_title("Exercise 3 Benchmark — Test Metrics")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - width / 2, elapsed, width, label="Elapsed seconds", color="tab:green")
    axes[1].bar(x + width / 2, best_epochs, width, label="Best epoch", color="tab:red")
    axes[1].set_title("Exercise 3 Benchmark — Runtime / Convergence")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OVERVIEW_DIR / "benchmark_comparison.png", dpi=120)
    plt.close(fig)


def build_overview_files(runs: list[dict]) -> None:
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for stale_path in OVERVIEW_DIR.glob("noise_per_class_*.png"):
        stale_path.unlink()
    build_overview_plots(runs)
    latest_noise_per_class = None
    noise_per_class_paths = sorted(
        PLOTS_DIR.glob("noise_per_class_*.png"),
        key=lambda path: path.stat().st_mtime,
    )
    if noise_per_class_paths:
        latest_noise_per_class = noise_per_class_paths[-1]

    csv_lines = [
        "run,mode,test_accuracy,test_f1_macro,val_accuracy,val_f1_macro,best_epoch,elapsed_seconds"
    ]
    for run in runs:
        csv_lines.append(
            ",".join([
                run["name"],
                run["mode"],
                f"{run['test']['accuracy']:.6f}",
                f"{run['test']['f1_macro']:.6f}",
                f"{run['validation']['accuracy']:.6f}",
                f"{run['validation']['f1_macro']:.6f}",
                str(run["best_epoch"]),
                f"{run['elapsed_seconds']:.6f}",
            ])
        )
    (OVERVIEW_DIR / "benchmark_table.csv").write_text("\n".join(csv_lines) + "\n")

    ranked = sorted(runs, key=lambda run: run["test"]["f1_macro"], reverse=True)
    best_clean = ranked[0]
    readme_lines = [
        "# Exercise 3 Benchmark Overview",
        "",
        f"- Best clean-test run by macro F1: `{best_clean['name']}` (`{best_clean['test']['f1_macro']:.4f}`)",
        "- Included runs: `1-Baseline`, `2-Weighted-Loss`, `3-Weighted-Sampling`, `4-Synthetic-Balancing`, `5-SMOTE`",
        "",
        "## Ranking by Test F1",
        "",
        "| Run | Mode | Test Acc | Test F1 | Val Acc | Val F1 | Best Epoch | Time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in ranked:
        readme_lines.append(
            f"| {run['name']} | {run['mode']} | {run['test']['accuracy']:.2%} | {run['test']['f1_macro']:.4f} | {run['validation']['accuracy']:.2%} | {run['validation']['f1_macro']:.4f} | {run['best_epoch']} | {run['elapsed_seconds']:.1f} |"
        )

    readme_lines.extend([
        "",
        "## Files",
        "",
        "- `benchmark_comparison.png`: grouped comparison of test metrics and runtime/convergence.",
        "- `benchmark_table.csv`: spreadsheet-friendly summary table.",
        "- `noise_robustness.png`: robustness under Gaussian noise for all runs.",
    ])
    if latest_noise_per_class is not None:
        readme_lines.append(
            f"- `{latest_noise_per_class.name}`: per-class noise breakdown for the latest best-noise plot generated by the analysis script."
        )
    (OVERVIEW_DIR / "README.md").write_text("\n".join(readme_lines) + "\n")

    copy_if_exists(PLOTS_DIR / "noise_robustness.png", OVERVIEW_DIR / "noise_robustness.png")
    if latest_noise_per_class is not None:
        copy_if_exists(latest_noise_per_class, OVERVIEW_DIR / latest_noise_per_class.name)


def build_root_readme(runs: list[dict]) -> None:
    lines = [
        "# Slides Package",
        "",
        "Carpeta armada para presentar los benchmarks del Ejercicio 3.",
        "",
        "## Contenido",
        "",
        "- `00_overview/`: comparativas globales entre runs.",
    ]
    for run in runs:
        lines.append(f"- `{run['slug']}/`: evidencia completa del run `{run['name']}`.")

    lines.extend([
        "",
        "## Qué tiene cada run",
        "",
        "- hiperparámetros usados",
        "- resumen de métricas train / validation / test",
        "- JSON completo del resultado",
        "- curva de entrenamiento",
        "- matriz de confusión",
        "- saliency map y visualización de pesos si existen",
    ])
    (SLIDES_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    runs = load_run_results()
    if not runs:
        raise FileNotFoundError(
            f"No run result JSON files found in {RESULTS_DIR}. Run exercise3/train.py first."
        )

    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    build_overview_files(runs)
    for run in runs:
        build_run_folder(run)
    build_root_readme(runs)
    print(f"Slides package generated at: {SLIDES_DIR}")


if __name__ == "__main__":
    main()
