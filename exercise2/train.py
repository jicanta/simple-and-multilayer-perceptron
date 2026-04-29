from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from data import (
    build_scaler,
    load_digits_dataset,
    print_data_quality,
    print_dataset_overview,
    scaler_from_metadata,
    train_validation_indices,
    train_validation_split,
)
from metrics import classification_metrics, save_metrics_json
from mlp import MultilayerPerceptron
from plots import save_confusion_matrix, save_training_curves


RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass
class ExperimentConfig:
    name: str
    architecture: list[int]
    learning_rate: float
    optimizer: str
    study_axis: str = "custom"
    epochs: int = 60
    batch_size: int = 128
    activation: str = "logistic"
    beta: float = 1.0
    normalization: str = "minmax"
    l2_lambda: float = 0.0
    early_stopping: bool = True
    patience: int = 12
    validation_ratio: float = 0.15
    seed: int = 42


def _section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def _history_to_dicts(history) -> list[dict]:
    return [
        {
            "epoch": row.epoch,
            "train_loss": row.train_loss,
            "train_accuracy": row.train_accuracy,
            "train_precision": row.train_precision,
            "train_recall": row.train_recall,
            "train_f1": row.train_f1,
            "val_loss": row.val_loss,
            "val_accuracy": row.val_accuracy,
            "val_precision": row.val_precision,
            "val_recall": row.val_recall,
            "val_f1": row.val_f1,
        }
        for row in history
    ]


def _plot_label(config: ExperimentConfig) -> str:
    hidden = "-".join(str(size) for size in config.architecture[1:-1]) or "none"
    return (
        f"Exercise 2 - {config.optimizer.upper()} "
        f"(lr={config.learning_rate}, hidden={hidden})"
    )


def infer_input_dimension() -> int:
    X_full, _ = load_digits_dataset("digits.csv")
    return int(X_full.shape[1])


def _prepare_data(config: ExperimentConfig):
    X_full, y_full = load_digits_dataset("digits.csv")
    train_idx, val_idx = train_validation_indices(
        len(X_full), validation_ratio=config.validation_ratio, seed=config.seed
    )
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]

    scaler = build_scaler(config.normalization)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val, scaler, train_idx, val_idx


def _prepare_data_from_metadata(config: ExperimentConfig, metadata: dict):
    if "train_indices" not in metadata or "validation_indices" not in metadata:
        return _prepare_data(config)
    X_full, y_full = load_digits_dataset("digits.csv")
    train_idx = np.array(metadata["train_indices"], dtype=np.int64)
    val_idx = np.array(metadata["validation_indices"], dtype=np.int64)
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]
    scaler = scaler_from_metadata(metadata.get("scaler"))
    if scaler is not None:
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
    return X_train, y_train, X_val, y_val, scaler, train_idx, val_idx


def _prepare_test_data(scaler):
    X_test, y_test = load_digits_dataset("digits_test.csv")
    if scaler is not None:
        X_test = scaler.transform(X_test)
    return X_test, y_test


def print_dataset_diagnostics() -> None:
    X_train_full, y_train_full = load_digits_dataset("digits.csv")
    X_test, y_test = load_digits_dataset("digits_test.csv")
    _section("Dataset Exploration")
    print_dataset_overview("digits.csv", X_train_full, y_train_full)
    print_data_quality("digits.csv", X_train_full, y_train_full)
    print_dataset_overview("digits_test.csv", X_test, y_test)
    print_data_quality("digits_test.csv", X_test, y_test)


def run_experiment(config: ExperimentConfig) -> dict:
    X_train, y_train, X_val, y_val, scaler, train_idx, val_idx = _prepare_data(config)

    model = MultilayerPerceptron(
        layer_sizes=config.architecture,
        learning_rate=config.learning_rate,
        activation=config.activation,
        beta=config.beta,
        batch_size=config.batch_size,
        optimizer=config.optimizer,
        l2_lambda=config.l2_lambda,
        seed=config.seed,
    )

    start = perf_counter()
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=config.epochs,
        verbose=True,
        early_stopping=config.early_stopping,
        patience=config.patience,
    )
    elapsed = perf_counter() - start

    train_eval = model.evaluate(X_train, y_train)
    val_eval = model.evaluate(X_val, y_val)

    history = _history_to_dicts(model.history)
    model_path = MODELS_DIR / f"{config.name}.npz"
    model.save(
        model_path,
        metadata={
            "config": asdict(config),
            "best_epoch": model.best_epoch,
            "train_indices": train_idx.tolist(),
            "validation_indices": val_idx.tolist(),
            "scaler": None if scaler is None else scaler.to_metadata(),
            "history": history,
        },
    )

    save_metrics_json(
        RESULTS_DIR / f"{config.name}.json",
        {
            "config": asdict(config),
            "elapsed_seconds": elapsed,
            "best_epoch": model.best_epoch,
            "train": train_eval,
            "validation": val_eval,
            "model_path": str(model_path),
            "history": history,
        },
    )

    save_training_curves(
        f"{config.name}-curves.png",
        history,
        _plot_label(config),
    )
    save_confusion_matrix(
        f"{config.name}-confusion.png",
        val_eval["confusion_matrix"],
        f"{_plot_label(config)} - Validation Confusion Matrix",
    )

    return {
        "config": config,
        "elapsed_seconds": elapsed,
        "best_epoch": model.best_epoch,
        "train": train_eval,
        "validation": val_eval,
        "history": history,
        "model_path": model_path,
        "model": model,
        "scaler": scaler,
    }


def run_resumed_experiment(
    model_path: Path,
    extra_epochs: int,
    patience: int | None = None,
) -> dict:
    metadata = MultilayerPerceptron.load_metadata(model_path)
    if "config" not in metadata:
        raise ValueError(
            f"Model {model_path} does not include saved config metadata, so it cannot be resumed safely."
        )

    config = ExperimentConfig(**metadata["config"])
    X_train, y_train, X_val, y_val, scaler, train_idx, val_idx = _prepare_data_from_metadata(
        config, metadata
    )
    model = MultilayerPerceptron.load(model_path)
    saved_history = metadata.get("history", [])
    if saved_history:
        from mlp import HistoryPoint

        model.history = [HistoryPoint(**row) for row in saved_history]

    start = perf_counter()
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=extra_epochs,
        verbose=True,
        early_stopping=config.early_stopping,
        patience=config.patience if patience is None else patience,
    )
    elapsed = perf_counter() - start

    train_eval = model.evaluate(X_train, y_train)
    val_eval = model.evaluate(X_val, y_val)
    history = _history_to_dicts(model.history)

    resumed_name = f"{model_path.stem}-resume-{extra_epochs}e"
    resumed_model_path = MODELS_DIR / f"{resumed_name}.npz"
    model.save(
        resumed_model_path,
        metadata={
            "config": asdict(config),
            "best_epoch": model.best_epoch,
            "resumed_from": str(model_path),
            "extra_epochs": extra_epochs,
            "train_indices": train_idx.tolist(),
            "validation_indices": val_idx.tolist(),
            "scaler": None if scaler is None else scaler.to_metadata(),
            "history": history,
        },
    )

    save_metrics_json(
        RESULTS_DIR / f"{resumed_name}.json",
        {
            "config": asdict(config),
            "elapsed_seconds": elapsed,
            "best_epoch": model.best_epoch,
            "train": train_eval,
            "validation": val_eval,
            "model_path": str(resumed_model_path),
            "history": history,
            "resumed_from": str(model_path),
            "extra_epochs": extra_epochs,
        },
    )
    save_training_curves(
        f"{resumed_name}-curves.png",
        history,
        f"{_plot_label(config)} - Resumed",
    )
    save_confusion_matrix(
        f"{resumed_name}-confusion.png",
        val_eval["confusion_matrix"],
        f"{_plot_label(config)} - Resumed Validation Confusion Matrix",
    )

    return {
        "config": config,
        "elapsed_seconds": elapsed,
        "best_epoch": model.best_epoch,
        "train": train_eval,
        "validation": val_eval,
        "history": history,
        "model_path": resumed_model_path,
        "model": model,
        "scaler": scaler,
        "resumed_from": model_path,
        "extra_epochs": extra_epochs,
    }


def print_experiment_summary(result: dict) -> None:
    config = result["config"]
    print(
        f"\n[{config.name}] arch={config.architecture} "
        f"optimizer={config.optimizer} lr={config.learning_rate} "
        f"axis={config.study_axis} "
        f"activation={config.activation}"
    )
    print(f"  elapsed:      {result['elapsed_seconds']:.2f}s")
    print(f"  best epoch:   {result['best_epoch']}")
    print(f"  train acc:    {result['train']['accuracy']:.2%}")
    print(f"  val acc:      {result['validation']['accuracy']:.2%}")
    print(f"  val macro P:  {result['validation']['precision_macro']:.4f}")
    print(f"  val macro R:  {result['validation']['recall_macro']:.4f}")
    print(f"  val macro F1: {result['validation']['f1_macro']:.4f}")


def choose_best(results: list[dict]) -> dict:
    return max(
        results,
        key=lambda result: (
            result["validation"]["accuracy"],
            -result["validation"]["loss"],
        ),
    )


def summarize_by_axis(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        grouped.setdefault(result["config"].study_axis, []).append(result)

    summary: dict[str, dict] = {}
    for axis, axis_results in grouped.items():
        best = choose_best(axis_results)
        summary[axis] = {
            "best_model": best["config"].name,
            "best_validation_accuracy": best["validation"]["accuracy"],
            "best_validation_f1_macro": best["validation"]["f1_macro"],
            "results": [
                {
                    "name": result["config"].name,
                    "architecture": result["config"].architecture,
                    "optimizer": result["config"].optimizer,
                    "learning_rate": result["config"].learning_rate,
                    "activation": result["config"].activation,
                    "validation_accuracy": result["validation"]["accuracy"],
                    "validation_precision_macro": result["validation"]["precision_macro"],
                    "validation_recall_macro": result["validation"]["recall_macro"],
                    "validation_f1_macro": result["validation"]["f1_macro"],
                    "best_epoch": result["best_epoch"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "model_path": str(result["model_path"]),
                }
                for result in axis_results
            ],
        }
    return summary


def evaluate_best_on_test(result: dict) -> dict:
    X_test, y_test = _prepare_test_data(result["scaler"])
    test_eval = result["model"].evaluate(X_test, y_test)
    save_metrics_json(
        RESULTS_DIR / f"{result['config'].name}-test.json",
        {
            "config": asdict(result["config"]),
            "best_epoch": result["best_epoch"],
            "test": test_eval,
            "model_path": str(result["model_path"]),
        },
    )
    save_confusion_matrix(
        f"{result['config'].name}-test-confusion.png",
        test_eval["confusion_matrix"],
        f"{_plot_label(result['config'])} - Test Confusion Matrix",
    )
    return test_eval


def default_experiments(input_dim: int | None = None) -> list[ExperimentConfig]:
    if input_dim is None:
        input_dim = infer_input_dimension()

    baseline_architecture = [input_dim, 128, 10]
    baseline_lr = 0.05
    baseline_optimizer = "adam"
    l2_for_adam_family = 1e-4

    experiments = [
        ExperimentConfig(
            name=f"axis_lr_mlp_adam_lr_0_001_arch_{input_dim}_128_10",
            architecture=baseline_architecture,
            learning_rate=0.001,
            optimizer=baseline_optimizer,
            study_axis="learning_rate",
            l2_lambda=l2_for_adam_family,
        ),
        ExperimentConfig(
            name=f"axis_lr_mlp_adam_lr_0_01_arch_{input_dim}_128_10",
            architecture=baseline_architecture,
            learning_rate=0.01,
            optimizer=baseline_optimizer,
            study_axis="learning_rate",
            l2_lambda=l2_for_adam_family,
        ),
        ExperimentConfig(
            name=f"axis_lr_mlp_adam_lr_0_05_arch_{input_dim}_128_10",
            architecture=baseline_architecture,
            learning_rate=0.05,
            optimizer=baseline_optimizer,
            study_axis="learning_rate",
            l2_lambda=l2_for_adam_family,
        ),
        ExperimentConfig(
            name=f"axis_lr_mlp_adam_lr_0_10_arch_{input_dim}_128_10",
            architecture=baseline_architecture,
            learning_rate=0.10,
            optimizer=baseline_optimizer,
            study_axis="learning_rate",
            l2_lambda=l2_for_adam_family,
        ),
        ExperimentConfig(
            name=f"axis_arch_mlp_adam_lr_0_05_arch_{input_dim}_64_10",
            architecture=[input_dim, 64, 10],
            learning_rate=baseline_lr,
            optimizer=baseline_optimizer,
            study_axis="architecture",
            l2_lambda=l2_for_adam_family,
        ),
        ExperimentConfig(
            name=f"axis_arch_mlp_adam_lr_0_05_arch_{input_dim}_128_10",
            architecture=[input_dim, 128, 10],
            learning_rate=baseline_lr,
            optimizer=baseline_optimizer,
            study_axis="architecture",
            l2_lambda=l2_for_adam_family,
        ),
        ExperimentConfig(
            name=f"axis_arch_mlp_adam_lr_0_05_arch_{input_dim}_128_64_10",
            architecture=[input_dim, 128, 64, 10],
            learning_rate=baseline_lr,
            optimizer=baseline_optimizer,
            study_axis="architecture",
            l2_lambda=l2_for_adam_family,
        ),
        ExperimentConfig(
            name=f"axis_arch_mlp_adam_lr_0_05_arch_{input_dim}_256_128_10",
            architecture=[input_dim, 256, 128, 10],
            learning_rate=baseline_lr,
            optimizer=baseline_optimizer,
            study_axis="architecture",
            l2_lambda=l2_for_adam_family,
        ),
        ExperimentConfig(
            name=f"axis_opt_mlp_sgd_lr_0_05_arch_{input_dim}_128_10",
            architecture=baseline_architecture,
            learning_rate=baseline_lr,
            optimizer="sgd",
            study_axis="optimizer",
        ),
        ExperimentConfig(
            name=f"axis_opt_mlp_momentum_lr_0_05_arch_{input_dim}_128_10",
            architecture=baseline_architecture,
            learning_rate=baseline_lr,
            optimizer="momentum",
            study_axis="optimizer",
        ),
        ExperimentConfig(
            name=f"axis_opt_mlp_adaptive_eta_lr_0_05_arch_{input_dim}_128_10",
            architecture=baseline_architecture,
            learning_rate=baseline_lr,
            optimizer="adaptive_eta",
            study_axis="optimizer",
        ),
        ExperimentConfig(
            name=f"axis_opt_mlp_rmsprop_lr_0_05_arch_{input_dim}_128_10",
            architecture=baseline_architecture,
            learning_rate=baseline_lr,
            optimizer="rmsprop",
            study_axis="optimizer",
        ),
        ExperimentConfig(
            name=f"axis_opt_mlp_adam_lr_0_05_arch_{input_dim}_128_10",
            architecture=baseline_architecture,
            learning_rate=baseline_lr,
            optimizer="adam",
            study_axis="optimizer",
            l2_lambda=l2_for_adam_family,
        ),
    ]

    return experiments


def baseline_config(input_dim: int | None = None) -> ExperimentConfig:
    if input_dim is None:
        input_dim = infer_input_dimension()
    return ExperimentConfig(
        name=f"mlp_adam_lr_0_001_arch_{input_dim}_128_10",
        architecture=[input_dim, 128, 10],
        learning_rate=0.001,
        optimizer="adam",
        study_axis="custom",
        l2_lambda=1e-4,
    )


def normalize_optimizer_name(name: str) -> str:
    aliases = {
        "gd": "sgd",
        "mini_batch_gd": "sgd",
    }
    return aliases.get(name.lower(), name.lower())


def parse_architecture(raw: str) -> list[int]:
    try:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Architecture must be a comma-separated list of integers, e.g. 784,128,10."
        ) from exc


def build_cli_experiment(args: argparse.Namespace) -> ExperimentConfig:
    baseline = baseline_config()
    requested_optimizer = normalize_optimizer_name(args.optimizer or "adam")
    architecture = args.architecture if args.architecture is not None else baseline.architecture
    lr = args.learning_rate if args.learning_rate is not None else baseline.learning_rate
    activation = args.activation if args.activation is not None else baseline.activation
    epochs = args.epochs if args.epochs is not None else baseline.epochs
    batch_size = args.batch_size if args.batch_size is not None else baseline.batch_size
    l2_lambda = args.l2_lambda if args.l2_lambda is not None else baseline.l2_lambda
    normalization = args.normalization if args.normalization is not None else baseline.normalization

    name = args.name
    if name is None:
        arch_label = "_".join(str(size) for size in architecture)
        lr_label = str(lr).replace(".", "_")
        name = f"mlp_{requested_optimizer}_lr_{lr_label}_arch_{arch_label}"
        if activation != "logistic":
            name = f"{name}_{activation}"

    return ExperimentConfig(
        name=name,
        architecture=architecture,
        learning_rate=lr,
        optimizer=requested_optimizer,
        study_axis="custom",
        epochs=epochs,
        batch_size=batch_size,
        activation=activation,
        beta=baseline.beta,
        normalization=normalization,
        l2_lambda=l2_lambda,
        early_stopping=baseline.early_stopping,
        patience=args.patience if args.patience is not None else baseline.patience,
        validation_ratio=baseline.validation_ratio,
        seed=baseline.seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 2 - Digit Classification with MLP")
    parser.add_argument(
        "--resume-model",
        type=Path,
        help="Path to a saved .npz model to continue training from.",
    )
    parser.add_argument(
        "--extra-epochs",
        type=int,
        default=20,
        help="Extra epochs to run when using --resume-model.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Optional patience override when resuming training.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "momentum", "adaptive_eta", "rmsprop", "adam"],
        help="Run a single experiment with the selected optimizer.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Optional learning rate override for a single custom run.",
    )
    parser.add_argument(
        "--architecture",
        type=parse_architecture,
        help="Optional architecture override for a single custom run, e.g. 784,128,10.",
    )
    parser.add_argument(
        "--activation",
        choices=["logistic", "tanh"],
        help="Optional activation override for a single custom run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Optional epoch count override for a single custom run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional batch size override for a single custom run.",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        help="Optional L2 regularization override for a single custom run.",
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "minmax", "zscore", "unit"],
        help="Optional normalization override for a single custom run.",
    )
    parser.add_argument(
        "--name",
        help="Optional explicit name for a single custom run.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate on digits_test.csv. Use this only for a final chosen model, not while tuning hyperparameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _section("Exercise 2 - Digit Classification with MLP")
    print("Training set: digits.csv")
    print("Generalization set: digits_test.csv")
    print("Experimental axes: learning rate, architecture, optimizer")
    print_dataset_diagnostics()

    if args.resume_model is not None:
        _section(f"Resuming {args.resume_model.name}")
        result = run_resumed_experiment(
            model_path=args.resume_model,
            extra_epochs=args.extra_epochs,
            patience=args.patience,
        )
        print_experiment_summary(result)
        if args.evaluate_test:
            test_eval = evaluate_best_on_test(result)
            print(f"  test acc:     {test_eval['accuracy']:.2%}")
            print(f"  test macro P: {test_eval['precision_macro']:.4f}")
            print(f"  test macro R: {test_eval['recall_macro']:.4f}")
            print(f"  test macro F1:{test_eval['f1_macro']:.4f}")
        else:
            print("  test eval:    skipped by default to preserve the hold-out protocol")
        print(f"  saved model:  {result['model_path']}")
        return

    if any(
        value is not None
        for value in (
            args.optimizer,
            args.learning_rate,
            args.architecture,
            args.activation,
            args.epochs,
            args.batch_size,
            args.l2_lambda,
            args.normalization,
            args.name,
        )
    ):
        config = build_cli_experiment(args)
        _section(f"Running {config.name}")
        result = run_experiment(config)
        print_experiment_summary(result)
        if args.evaluate_test:
            test_eval = evaluate_best_on_test(result)
            print(f"  test acc:     {test_eval['accuracy']:.2%}")
            print(f"  test macro P: {test_eval['precision_macro']:.4f}")
            print(f"  test macro R: {test_eval['recall_macro']:.4f}")
            print(f"  test macro F1:{test_eval['f1_macro']:.4f}")
        else:
            print("  test eval:    skipped by default to preserve the hold-out protocol")
        print(f"  saved model:  {result['model_path']}")
        return

    results = []
    for config in default_experiments():
        _section(f"Running {config.name}")
        result = run_experiment(config)
        results.append(result)
        print_experiment_summary(result)

    best = choose_best(results)
    best_test = evaluate_best_on_test(best)
    _section("Best Model")
    print_experiment_summary(best)
    print(f"  test acc:     {best_test['accuracy']:.2%}")
    print(f"  test macro P: {best_test['precision_macro']:.4f}")
    print(f"  test macro R: {best_test['recall_macro']:.4f}")
    print(f"  test macro F1:{best_test['f1_macro']:.4f}")
    print(f"  saved model:  {best['model_path']}")

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "best_model": best["config"].name,
        "by_axis": summarize_by_axis(results),
        "results": [
            {
                "name": result["config"].name,
                "architecture": result["config"].architecture,
                "optimizer": result["config"].optimizer,
                "study_axis": result["config"].study_axis,
                "learning_rate": result["config"].learning_rate,
                "activation": result["config"].activation,
                "train_accuracy": result["train"]["accuracy"],
                "validation_accuracy": result["validation"]["accuracy"],
                "validation_precision_macro": result["validation"]["precision_macro"],
                "validation_recall_macro": result["validation"]["recall_macro"],
                "validation_f1_macro": result["validation"]["f1_macro"],
                "best_epoch": result["best_epoch"],
                "elapsed_seconds": result["elapsed_seconds"],
                "model_path": str(result["model_path"]),
            }
            for result in results
        ],
        "final_test_evaluation": {
            "name": best["config"].name,
            "test_accuracy": best_test["accuracy"],
            "test_precision_macro": best_test["precision_macro"],
            "test_recall_macro": best_test["recall_macro"],
            "test_f1_macro": best_test["f1_macro"],
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
