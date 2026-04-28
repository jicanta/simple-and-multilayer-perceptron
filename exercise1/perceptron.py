from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LinearPerceptron:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100, batch_size: int = 32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.losses: list[float] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return X @ self.w + self.b

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
        patience: int = 0,
        min_delta: float = 1e-6,
    ) -> "LinearPerceptron":
        n_samples, n_features = X.shape
        rng = np.random.default_rng(42)
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.losses = []
        report_every = max(1, self.epochs // 10)
        best_loss = float("inf")
        no_improve = 0

        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            X_sh, y_sh = X[indices], y[indices]
            batch_losses = []

            for start in range(0, n_samples, self.batch_size):
                Xb = X_sh[start : start + self.batch_size]
                yb = y_sh[start : start + self.batch_size]
                error = yb - (Xb @ self.w + self.b)
                self.w += self.learning_rate * (Xb.T @ error) / len(Xb)
                self.b += self.learning_rate * error.mean()
                batch_losses.append(float(np.mean(error**2)))

            self.losses.append(float(np.mean(batch_losses)))

            if verbose and (epoch + 1) % report_every == 0:
                print(f"  [linear]  epoch {epoch + 1:>4}/{self.epochs}  loss={self.losses[-1]:.6f}")

            if patience > 0:
                if best_loss - self.losses[-1] > min_delta:
                    best_loss = self.losses[-1]
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        if verbose:
                            print(
                                f"  [linear]  early stop at epoch {epoch + 1}  "
                                f"loss={self.losses[-1]:.6f}"
                            )
                        break

        return self


class NonLinearPerceptron:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100, batch_size: int = 32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.losses: list[float] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return sigmoid(X @ self.w + self.b)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
        patience: int = 0,
        min_delta: float = 1e-6,
    ) -> "NonLinearPerceptron":
        n_samples, n_features = X.shape
        rng = np.random.default_rng(42)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.losses = []
        report_every = max(1, self.epochs // 10)
        best_loss = float("inf")
        no_improve = 0

        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            X_sh, y_sh = X[indices], y[indices]
            batch_losses = []

            for start in range(0, n_samples, self.batch_size):
                Xb = X_sh[start : start + self.batch_size]
                yb = y_sh[start : start + self.batch_size]
                y_pred = sigmoid(Xb @ self.w + self.b)
                error = yb - y_pred
                delta = error * y_pred * (1.0 - y_pred)
                self.w += self.learning_rate * (Xb.T @ delta) / len(Xb)
                self.b += self.learning_rate * delta.mean()
                batch_losses.append(float(np.mean(error**2)))

            self.losses.append(float(np.mean(batch_losses)))

            if verbose and (epoch + 1) % report_every == 0:
                print(f"  [sigmoid] epoch {epoch + 1:>4}/{self.epochs}  loss={self.losses[-1]:.6f}")

            if patience > 0:
                if best_loss - self.losses[-1] > min_delta:
                    best_loss = self.losses[-1]
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        if verbose:
                            print(
                                f"  [sigmoid] early stop at epoch {epoch + 1}  "
                                f"loss={self.losses[-1]:.6f}"
                            )
                        break

        return self
