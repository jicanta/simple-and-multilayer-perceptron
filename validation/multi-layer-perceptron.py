import numpy as np

from common import print_named_values, print_table, save_classification_plot


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def build_model(architecture):
    if architecture == [2, 2, 1]:
        weights = [
            np.array([[-10, 10], [10, -10]]),
            np.array([[10, 10]]),
        ]

        biases = [
            np.array([-10, -10]),
            np.array([-5]),
        ]

        return weights, biases

    if architecture == [2, 3, 2, 1]:
        weights = [
            np.array([[-10, 0], [0, -10], [10, 10]]),
            np.array([[10, -10, 10], [-10, 10, 10]]),
            np.array([[10, 10]]),
        ]

        biases = [
            np.array([0, 0, 10]),
            np.array([-15, -15]),
            np.array([-5]),
        ]

        return weights, biases

    raise ValueError(f"Unsupported architecture: {architecture}")


def build_dataset():
    x = np.array([
        [-1, 1],
        [1, -1],
        [-1, -1],
        [1, 1],
    ])
    y = np.array([1, 1, -1, -1])
    return x, y


def forward(x, weights, biases):
    activations = [x]
    a = x

    for weight_matrix, bias_vector in zip(weights, biases):
        a = sigmoid(np.dot(a, weight_matrix.T) + bias_vector)
        activations.append(a)

    return activations


def predict(x, weights, biases):
    activations = forward(x, weights, biases)
    y = activations[-1]
    return 1 if y[0] > 0.5 else -1


def evaluate_model(architecture):
    weights, biases = build_model(architecture)
    x, y = build_dataset()
    predictions = np.array([predict(xi, weights, biases) for xi in x])
    rows = []

    for xi, yi, y_pred in zip(x, y, predictions):
        rows.append([f"[{int(xi[0]):2d}, {int(xi[1]):2d}]", yi, y_pred, "yes" if yi == y_pred else "no"])

    return {
        "weights": weights,
        "biases": biases,
        "x": x,
        "y": y,
        "predictions": predictions,
        "accuracy": float(np.mean(predictions == y)),
        "rows": rows,
    }


def run_validation(architecture):
    evaluation = evaluate_model(architecture)

    named_values = {
        "architecture": architecture,
        "accuracy": f"{evaluation['accuracy']:.2%}",
    }

    for index, (weights, biases) in enumerate(zip(evaluation["weights"], evaluation["biases"]), start=1):
        named_values[f"weights_layer_{index}"] = weights
        named_values[f"bias_layer_{index}"] = biases

    print_named_values("Multi Layer Perceptron", named_values)
    print_table(
        "Validation",
        ["x", "expected", "predicted", "correct"],
        evaluation["rows"],
    )

    architecture_name = "-".join(str(size) for size in architecture)
    plot_path = save_classification_plot(
        f"multi-layer-perceptron-{architecture_name}.png",
        evaluation["x"],
        evaluation["y"],
        evaluation["predictions"],
        f"Multi Layer Perceptron - XOR - {architecture}",
    )

    if plot_path is not None:
        print(f"\nplot: {plot_path}")

    return evaluation


if __name__ == "__main__":
    run_validation([2, 2, 1])
    run_validation([2, 3, 2, 1])
