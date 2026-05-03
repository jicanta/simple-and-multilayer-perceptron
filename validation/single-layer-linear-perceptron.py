import numpy as np

from common import (
    print_named_values,
    print_table,
    save_loss_plot,
    save_prediction_scatter_plot,
    save_regression_plot,
    save_residual_plot,
)


EPOCHS = 50
LEARNING_RATE = 0.1
N = 50
TRAIN_RATIO = 0.8

TARGET_FUNCTIONS = {
    "identity": {"slope": 1.0, "bias": 0.0, "label": "y = x"},
    "scaled": {"slope": 2.0, "bias": 0.0, "label": "y = 2x"},
    "shifted": {"slope": 1.0, "bias": 1.0, "label": "y = x + 1"},
    "descending": {"slope": -1.5, "bias": 0.5, "label": "y = -1.5x + 0.5"},
}


def get_target_function(target_name):
    if target_name not in TARGET_FUNCTIONS:
        raise ValueError(f"Unsupported target function: {target_name}")

    return TARGET_FUNCTIONS[target_name]


def f(x, w, b):
    return w * x + b


def update_weights(x, error, w, b):
    # technically there is a theta' term hidden here
    # in this as the activation function is the identity, this term equals 1
    b = b + LEARNING_RATE * error
    w = w + LEARNING_RATE * error * x 
    return w, b


def build_dataset(target_name="identity"):
    target = get_target_function(target_name)
    x = np.linspace(-1, 1, N)
    y = target["slope"] * x + target["bias"]
    return x, y


def split_dataset(x, y):
    split_index = int(len(x) * TRAIN_RATIO)
    return x[:split_index], y[:split_index], x[split_index:], y[split_index:]


def train_model(x, y):
    w = 0.0
    b = 0.0
    losses = []

    for _ in range(EPOCHS):
        squared_errors = []

        for xi, yi in zip(x, y):
            y_pred = f(xi, w, b)
            error = yi - y_pred
            w, b = update_weights(xi, error, w, b)
            squared_errors.append(error ** 2)

        losses.append(float(np.mean(squared_errors)))

    return w, b, losses


def evaluate_model(x, y, w, b):
    predictions = f(x, w, b)
    errors = y - predictions
    mse = float(np.mean(errors ** 2))
    max_abs_error = float(np.max(np.abs(errors)))
    rows = []

    for xi, yi, y_pred, error in zip(x[:10], y[:10], predictions[:10], errors[:10]):
        rows.append([f"{xi:.2f}", f"{yi:.4f}", f"{y_pred:.4f}", f"{error:.4f}"])

    return {
        "predictions": predictions,
        "errors": errors,
        "residuals": errors,
        "mse": mse,
        "max_abs_error": max_abs_error,
        "rows": rows,
    }


def run_validation(target_name="identity"):
    x, y = build_dataset(target_name)
    train_x, train_y, test_x, test_y = split_dataset(x, y)
    w, b, losses = train_model(train_x, train_y)
    train_evaluation = evaluate_model(train_x, train_y, w, b)
    test_evaluation = evaluate_model(test_x, test_y, w, b)
    full_evaluation = evaluate_model(x, y, w, b)
    target = get_target_function(target_name)

    print_named_values(
        "Linear Perceptron",
        {
            "target": target["label"],
            "weight": f"{w:.6f}",
            "bias": f"{b:.6f}",
            "epochs": len(losses),
            "train_samples": len(train_x),
            "test_samples": len(test_x),
            "train_mse": f"{train_evaluation['mse']:.8f}",
            "test_mse": f"{test_evaluation['mse']:.8f}",
            "test_max_abs_error": f"{test_evaluation['max_abs_error']:.8f}",
        },
    )

    print_table(
        "Held Out Validation Sample",
        ["x", "expected", "predicted", "error"],
        test_evaluation["rows"],
    )

    plot_prefix = f"single-layer-linear-perceptron-{target_name}"
    loss_plot_path = save_loss_plot(
        f"{plot_prefix}-loss.png",
        losses,
        f"Linear Perceptron - Loss - {target['label']}",
    )
    fit_plot_path = save_regression_plot(
        f"{plot_prefix}-fit.png",
        x,
        y,
        full_evaluation["predictions"],
        f"Linear Perceptron - {target['label']}",
    )
    residual_plot_path = save_residual_plot(
        f"{plot_prefix}-residuals.png",
        test_x,
        test_evaluation["residuals"],
        f"Linear Perceptron - Residuals - {target['label']}",
    )
    scatter_plot_path = save_prediction_scatter_plot(
        f"{plot_prefix}-scatter.png",
        test_y,
        test_evaluation["predictions"],
        f"Linear Perceptron - Expected vs Predicted - {target['label']}",
    )

    for path in [loss_plot_path, fit_plot_path, residual_plot_path, scatter_plot_path]:
        if path is not None:
            print(f"plot: {path}")

    return {
        "target": target,
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "weight": w,
        "bias": b,
        "losses": losses,
        "train": train_evaluation,
        "test": test_evaluation,
        "full": full_evaluation,
    }


if __name__ == "__main__":
    for target_name in TARGET_FUNCTIONS:
        run_validation(target_name)
