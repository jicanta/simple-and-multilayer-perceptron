import numpy as np

EPOCHS = 100
LEARNING_RATE = 0.1
QUAN_POINTS = 50

def f(x, w, b):
  return np.tanh(w * x + b)

def update_weights(x, delta, w, b):
  b = b + LEARNING_RATE * delta
  w = w + LEARNING_RATE * delta * x
  return w, b

w = 0.0
b = 0.0
X = np.linspace(-2, 2, QUAN_POINTS)
Y = np.tanh(X)

for epoch in range(EPOCHS):
  for xi, yi in zip(X, Y):
    y_pred = f(xi, w, b)

    error = yi - y_pred
    delta = error * (1 - y_pred ** 2)
    w, b = update_weights(xi, delta, w, b)

print("weights: ", w)
print("bias: ", b)

print("\n=== VALIDATION ===")
print(f"{'x':>6} | {'expected':>10} | {'predicted':>10} | {'error':>10}")
print("-" * 45)

for xi, yi in zip(X[:10], Y[:10]):
    y_pred = f(xi, w, b)
    err = yi - y_pred

    print(f"{xi:>6.2f} | {yi:>10.2f} | {y_pred:>10.2f} | {err:>10.2f}")
