from pathlib import Path
import warnings

import numpy as np


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def get_plt():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unable to import Axes3D.*",
                category=UserWarning,
            )
            import matplotlib.pyplot as plt
    except ImportError:
        return None

    return plt


def print_section(title):
    line = "=" * len(title)
    print(f"\n{line}")
    print(title)
    print(line)


def print_named_values(title, values):
    print_section(title)
    for name, value in values.items():
        if isinstance(value, np.ndarray):
            print(f"{name}:\n{value}")
        else:
            print(f"{name}: {value}")


def print_table(title, headers, rows):
    print_section(title)

    widths = [len(header) for header in headers]
    formatted_rows = []

    for row in rows:
        formatted_row = [str(cell) for cell in row]
        formatted_rows.append(formatted_row)
        widths = [max(width, len(cell)) for width, cell in zip(widths, formatted_row)]

    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    separator = "-+-".join("-" * width for width in widths)

    print(header_line)
    print(separator)

    for row in formatted_rows:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def ensure_plots_dir():
    PLOTS_DIR.mkdir(exist_ok=True)
    return PLOTS_DIR


def can_plot():
    return get_plt() is not None


def save_loss_plot(filename, losses, title):
    plt = get_plt()
    if plt is None or not losses:
        return None

    path = ensure_plots_dir() / filename

    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses) + 1), losses, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path


def save_regression_plot(filename, x, y_true, y_pred, title):
    plt = get_plt()
    if plt is None:
        return None

    path = ensure_plots_dir() / filename

    fig, ax = plt.subplots()
    ax.plot(x, y_true, label="expected", color="tab:green")
    ax.plot(x, y_pred, label="predicted", color="tab:orange")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path


def save_residual_plot(filename, x, residuals, title):
    plt = get_plt()
    if plt is None:
        return None

    path = ensure_plots_dir() / filename

    fig, ax = plt.subplots()
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.scatter(x, residuals, color="tab:purple")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("residual")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path


def save_prediction_scatter_plot(filename, y_true, y_pred, title):
    plt = get_plt()
    if plt is None:
        return None

    path = ensure_plots_dir() / filename
    min_value = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_value = max(float(np.max(y_true)), float(np.max(y_pred)))

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, color="tab:blue")
    ax.plot([min_value, max_value], [min_value, max_value], color="tab:red", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("expected")
    ax.set_ylabel("predicted")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path


def save_classification_plot(filename, x, y_true, y_pred, title):
    plt = get_plt()
    if plt is None:
        return None

    path = ensure_plots_dir() / filename

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    true_colors = ["tab:green" if value == 1 else "tab:red" for value in y_true]
    pred_colors = ["tab:green" if value == 1 else "tab:red" for value in y_pred]

    axes[0].scatter(x[:, 0], x[:, 1], c=true_colors, s=100)
    axes[0].set_title("Expected")
    axes[1].scatter(x[:, 0], x[:, 1], c=pred_colors, s=100)
    axes[1].set_title("Predicted")

    for ax in axes:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path
