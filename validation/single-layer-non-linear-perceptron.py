import numpy as np

EPOCHS = 200
LEARNING_RATE = 0.01
N = 50

def activation(z):
    return np.tanh(z)

def activation_derivative(z):
    return 1 - np.tanh(z)**2

def forward(x, w, b):
    z = w * x + b
    y_pred = activation(z)
    return y_pred, z

def update(x, error, z, w, b):
    delta = error * activation_derivative(z)
    w = w + LEARNING_RATE * delta * x
    b = b + LEARNING_RATE * delta
    return w, b

X = np.linspace(-2, 2, N)
Y = np.tanh(X)

w = 0.0
b = 0.0

for epoch in range(EPOCHS):
    for xi, yi in zip(X, Y):
        y_pred, z = forward(xi, w, b)

        error = yi - y_pred
        w, b = update(xi, error, z, w, b)

print("w:", w)
print("b:", b)

print("\n=== VALIDATION ===")
print(f"{'x':>6} | {'expected':>10} | {'predicted':>10} | {'error':>10}")
print("-" * 45)

for xi, yi in zip(X[:10], Y[:10]):
    y_pred, _ = forward(xi, w, b)
    err = yi - y_pred

    print(f"{xi:>6.2f} | {yi:>10.2f} | {y_pred:>10.2f} | {err:>10.2f}")
