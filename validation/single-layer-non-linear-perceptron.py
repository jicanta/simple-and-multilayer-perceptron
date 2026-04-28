import numpy as np

from common import (
    print_named_values,
    print_table,
    save_loss_plot,
    save_prediction_scatter_plot,
    save_regression_plot,
    save_residual_plot,
)


EPOCHS = 200
LEARNING_RATE = 0.01
N = 50
TRAIN_RATIO = 0.8


def activation(z):
    return np.tanh(z)


def activation_derivative(z):
    return 1 - np.tanh(z) ** 2


def forward(x, w, b):
    z = w * x + b
    y_pred = activation(z)
    return y_pred, z


def update(x, error, z, w, b):
    delta = error * activation_derivative(z)
    w = w + LEARNING_RATE * delta * x
    b = b + LEARNING_RATE * delta
    return w, b


def build_dataset():
    x = np.linspace(-2, 2, N)
    y = np.tanh(x)
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
            y_pred, z = forward(xi, w, b)
            error = yi - y_pred
            w, b = update(xi, error, z, w, b)
            squared_errors.append(error ** 2)

        losses.append(float(np.mean(squared_errors)))

    return w, b, losses


def evaluate_model(x, y, w, b):
    predictions = np.array([forward(xi, w, b)[0] for xi in x])
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


def run_validation():
    x, y = build_dataset()
    train_x, train_y, test_x, test_y = split_dataset(x, y)
    w, b, losses = train_model(train_x, train_y)
    train_evaluation = evaluate_model(train_x, train_y, w, b)
    test_evaluation = evaluate_model(test_x, test_y, w, b)
    full_evaluation = evaluate_model(x, y, w, b)

    print_named_values(
        "Non Linear Perceptron",
        {
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

    loss_plot_path = save_loss_plot(
        "single-layer-non-linear-perceptron-loss.png",
        losses,
        "Non Linear Perceptron - Loss",
    )
    fit_plot_path = save_regression_plot(
        "single-layer-non-linear-perceptron-fit.png",
        x,
        y,
        full_evaluation["predictions"],
        "Non Linear Perceptron - tanh(x)",
    )
    residual_plot_path = save_residual_plot(
        "single-layer-non-linear-perceptron-residuals.png",
        test_x,
        test_evaluation["residuals"],
        "Non Linear Perceptron - Residuals",
    )
    scatter_plot_path = save_prediction_scatter_plot(
        "single-layer-non-linear-perceptron-scatter.png",
        test_y,
        test_evaluation["predictions"],
        "Non Linear Perceptron - Expected vs Predicted",
    )

    for path in [loss_plot_path, fit_plot_path, residual_plot_path, scatter_plot_path]:
        if path is not None:
            print(f"plot: {path}")

    return {
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
    run_validation()
