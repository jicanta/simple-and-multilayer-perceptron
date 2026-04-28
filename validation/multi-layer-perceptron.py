# arquitectura [2, 2, 1]:
# a xor b = (a and (not b)) or ((not a) and b)
# vamos a hacer que el primer nodo de la hidden layer
# "haga" la operacion a and (not b) y el segundo (not a) and b
# asi el output node finalmente "hace" h1 or h2.
# arquitectura [2, 3, 2, 1]:
# h1 detecta x1 = -1
# h2 detecta x2 = -1
# h3 detecta que no sean ambos -1
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def build_model(architecture):
    if architecture == [2, 2, 1]:
        weights = [
            np.array([[-10,  10],
                      [ 10, -10]]),
            np.array([[10, 10]])
        ]

        biases = [
            np.array([-10, -10]),
            np.array([-5])
        ]

    elif architecture == [2, 3, 2, 1]:
        weights = [
            np.array([
                [-10,   0],
                [  0, -10],
                [ 10,  10]
            ]),
            np.array([
                [ 10, -10, 10],
                [-10,  10, 10]
            ]),
            np.array([[10, 10]])
        ]

        biases = [
            np.array([0, 0, 10]),
            np.array([-15, -15]),
            np.array([-5])
        ]

    return weights, biases

def forward(x, weights, biases):
    activations = [x]

    a = x
    for W, b in zip(weights, biases):
        a = sigmoid(np.dot(a, W.T) + b)
        activations.append(a)

    return activations

def predict(x, weights, biases):
    activations = forward(x, weights, biases)
    y = activations[-1]
    return 1 if y[0] > 0.5 else -1

def validate(architecture):
    weights, biases = build_model(architecture)

    X = np.array([
        [-1,  1],
        [ 1, -1],
        [-1, -1],
        [ 1,  1]
    ])

    Y = np.array([1, 1, -1, -1])

    print("\nArchitecture:", architecture)

    for i, (W, b) in enumerate(zip(weights, biases), start=1):
        print(f"\nweights layer {i}:\n", W)
        print(f"bias layer {i}:", b)

    print("\n=== VALIDATION ===")
    print(f"{'x':>12} | {'expected':>10} | {'predicted':>10} | {'error':>10}")
    print("-" * 60)

    for xi, yi in zip(X, Y):
        y_pred = predict(xi, weights, biases)
        err = yi - y_pred

        print(f"{str(xi):>12} | {yi:>10} | {y_pred:>10} | {err:>10}")

validate([2, 2, 1])
validate([2, 3, 2, 1])