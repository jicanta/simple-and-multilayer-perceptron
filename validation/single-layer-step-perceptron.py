import numpy as np

from common import print_named_values, print_table, save_classification_plot


EPOCHS = 20
LEARNING_RATE = 1


def f(x, w, b):
    return 1 if np.dot(x, w) + b >= 0 else -1


def update_weights(x, y, w, b):
    b = b + LEARNING_RATE * y
    for i in range(len(w)):
        w[i] = w[i] + LEARNING_RATE * x[i] * y
    return w, b


def build_dataset(logical_gate="and"):
    x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    if logical_gate == "and":
        y = np.array([-1, -1, -1, 1])
    elif logical_gate == "xor":
        y = np.array([-1, 1, 1, -1])
    else:
        raise ValueError(f"Unsupported logical gate: {logical_gate}")
    return x, y


def train_model(x, y):
    w = np.zeros(x.shape[1])
    b = 0
    errors_by_epoch = []

    for _ in range(EPOCHS):
        errors = 0
        for xi, yi in zip(x, y):
            y_pred = f(xi, w, b)

            if y_pred != yi:
                w, b = update_weights(xi, yi, w, b)
                errors += 1

        errors_by_epoch.append(errors)

        if errors == 0:
            break

    return w, b, errors_by_epoch


def evaluate_model(x, y, w, b):
    predictions = np.array([f(xi, w, b) for xi in x])
    accuracy = float(np.mean(predictions == y))
    rows = []

    for xi, yi, y_pred in zip(x, y, predictions):
        rows.append([f"[{xi[0]:2d}, {xi[1]:2d}]", yi, y_pred, "yes" if yi == y_pred else "no"])

    return {
        "predictions": predictions,
        "accuracy": accuracy,
        "rows": rows,
    }


def run_validation():
    x, y = build_dataset()
    w, b, errors_by_epoch = train_model(x, y)
    evaluation = evaluate_model(x, y, w, b)

    print_named_values(
        "Step Perceptron (AND)",
        {
            "weights": w,
            "bias": b,
            "epochs": len(errors_by_epoch),
            "final_accuracy": f"{evaluation['accuracy']:.2%}",
        },
    )

    print_table(
        "Validation (AND)",
        ["x", "expected", "predicted", "correct"],
        evaluation["rows"],
    )

    plot_path = save_classification_plot(
        "single-layer-step-perceptron.png",
        x,
        y,
        evaluation["predictions"],
        "Step Perceptron - AND",
    )

    if plot_path is not None:
        print(f"\nplot: {plot_path}")

    return {
        "weights": w,
        "bias": b,
        "errors_by_epoch": errors_by_epoch,
        **evaluation,
    }


def run_xor_comparison():
    print("\n" + "=" * 50)
    print("Testing Step Perceptron on XOR")
    print("=" * 50)
    x, y = build_dataset("xor")
    w, b, errors_by_epoch = train_model(x, y)
    evaluation = evaluate_model(x, y, w, b)

    print_named_values(
        "Step Perceptron (XOR)",
        {
            "weights": w,
            "bias": b,
            "epochs": len(errors_by_epoch),
            "final_accuracy": f"{evaluation['accuracy']:.2%}",
        },
    )

    print_table(
        "Validation (XOR)",
        ["x", "expected", "predicted", "correct"],
        evaluation["rows"],
    )

    print("\nNote: The Step Perceptron cannot solve the XOR problem because it is not linearly separable.")
    print("It reaches the maximum number of epochs without converging (errors > 0 in the last epoch).")
    print("Compare this with the Multi Layer Perceptron, which successfully solves XOR.")

    plot_path = save_classification_plot(
        "single-layer-step-perceptron-xor.png",
        x,
        y,
        evaluation["predictions"],
        "Step Perceptron - XOR (Fails)",
    )

    if plot_path is not None:
        print(f"\nplot: {plot_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step Perceptron Validation")
    parser.add_argument("--xor", action="store_true", help="Run the XOR comparison instead of AND")
    args = parser.parse_args()

    if args.xor:
        run_xor_comparison()
    else:
        run_validation()
