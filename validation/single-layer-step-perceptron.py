# w = w + η * y * x
# b = b + η * y
import numpy as np

EPOCHS = 20
LEARNING_RATE = 1

def f(x, w, b):
  return 1 if np.dot(x, w) + b >= 0 else -1

def update_weights(x, y, w, b):
  b = b + LEARNING_RATE * y
  for i in range(len(w)):
    w[i] = w[i] + LEARNING_RATE * x[i] * y
  return w, b

w = np.zeros(2)
b = 0
X = np.array([[-1, -1], [-1, +1], [+1, -1], [+1, +1]])
Y = np.array([ -1,       -1,       -1,       +1])

for epoch in range(EPOCHS):
  errors = 0
  for i in range(len(X)):
    xi = X[i]
    yi = Y[i]
    y_pred = f(xi, w, b)

    if y_pred != yi:
      w, b = update_weights(xi, yi, w, b)
      errors += 1

  if errors == 0:
    break

print("weights: ", w)
print("bias: ", b)

print("\n=== VALIDATION ===")
print(f"{'x1':>3} {'x2':>3} | {'expected':>8} | {'predicted':>9} | {'ok?':>4}")
print("-" * 40)

for xi, yi in zip(X, Y):
    y_pred = f(xi, w, b)
    ok = "✔" if y_pred == yi else "✘"

    print(f"{xi[0]:>3} {xi[1]:>3} | {yi:>8} | {y_pred:>9} | {ok:>4}")