from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metrics import classification_metrics, mse_loss


def logistic(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    z = np.clip(-2.0 * beta * x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))


def logistic_derivative(output: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return 2.0 * beta * output * (1.0 - output)


def tanh_activation(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return np.tanh(beta * x)


def tanh_derivative(output: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return beta * (1.0 - output ** 2)


@dataclass
class HistoryPoint:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float | None
    val_accuracy: float | None
    train_precision: float = 0.0
    train_recall: float = 0.0
    train_f1: float = 0.0
    val_precision: float | None = None
    val_recall: float | None = None
    val_f1: float | None = None


class MultilayerPerceptron:
    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.01,
        activation: str = "logistic",
        beta: float = 1.0,
        batch_size: int = 64,
        optimizer: str = "sgd",
        momentum_alpha: float = 0.9,
        rmsprop_gamma: float = 0.9,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        epsilon: float = 1e-8,
        l2_lambda: float = 0.0,
        adaptive_k: int = 5,
        adaptive_increase: float = 1.05,
        adaptive_decrease: float = 0.5,
        seed: int = 42,
    ):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.beta = beta
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.momentum_alpha = momentum_alpha
        self.rmsprop_gamma = rmsprop_gamma
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.epsilon = epsilon
        self.l2_lambda = l2_lambda
        self.adaptive_k = adaptive_k
        self.adaptive_increase = adaptive_increase
        self.adaptive_decrease = adaptive_decrease
        self.seed = seed

        self.weights: list[np.ndarray] = []
        self._velocity: list[np.ndarray] = []
        self._rms: list[np.ndarray] = []
        self._adam_m: list[np.ndarray] = []
        self._adam_v: list[np.ndarray] = []
        self._step = 0
        self.history: list[HistoryPoint] = []
        self.best_epoch: int | None = None

        self._initialize()

    def _initialize(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weights = []
        for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            weight = rng.normal(0.0, 0.05, size=(in_size + 1, out_size)).astype(np.float32)
            self.weights.append(weight)

        self._velocity = [np.zeros_like(weight) for weight in self.weights]
        self._rms = [np.zeros_like(weight) for weight in self.weights]
        self._adam_m = [np.zeros_like(weight) for weight in self.weights]
        self._adam_v = [np.zeros_like(weight) for weight in self.weights]
        self._step = 0
        self.history = []
        self.best_epoch = None

    def _append_bias(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([X, ones])

    def _activate(self, h: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return tanh_activation(h, beta=self.beta)
        return logistic(h, beta=self.beta)

    def _activate_derivative(self, output: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return tanh_derivative(output, beta=self.beta)
        return logistic_derivative(output, beta=self.beta)

    def forward(self, X: np.ndarray) -> list[np.ndarray]:
        activations = [X]
        current = X
        for weight in self.weights:
            h = self._append_bias(current) @ weight
            current = self._activate(h)
            activations.append(current)
        return activations

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def _backward(
        self,
        activations: list[np.ndarray],
        y_true: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        deltas: list[np.ndarray] = [np.empty((0, 0), dtype=np.float32) for _ in self.weights]
        output = activations[-1]
        delta_out = (y_true - output) * self._activate_derivative(output)
        if sample_weights is not None:
            # Scale each sample's error by its class weight — (batch,1) broadcasts over (batch, n_out)
            delta_out = delta_out * sample_weights[:, np.newaxis]
        deltas[-1] = delta_out

        for layer_idx in range(len(self.weights) - 2, -1, -1):
            propagated = deltas[layer_idx + 1] @ self.weights[layer_idx + 1][:-1, :].T
            deltas[layer_idx] = propagated * self._activate_derivative(
                activations[layer_idx + 1]
            )

        gradients = []
        for layer_idx, delta in enumerate(deltas):
            prev_activation = self._append_bias(activations[layer_idx])
            gradient = (prev_activation.T @ delta) / len(prev_activation)
            gradients.append(gradient)
        return gradients

    def _apply_optimizer(self, gradients: list[np.ndarray]) -> None:
        self._step += 1
        for idx, gradient in enumerate(gradients):
            if self.l2_lambda > 0.0:
                penalty = self.weights[idx].copy()
                penalty[-1, :] = 0.0
                gradient = gradient - self.l2_lambda * penalty

            if self.optimizer in ("sgd", "gd", "adaptive_eta"):
                update = self.learning_rate * gradient
            elif self.optimizer == "momentum":
                self._velocity[idx] = (
                    self.momentum_alpha * self._velocity[idx] + self.learning_rate * gradient
                )
                update = self._velocity[idx]
            elif self.optimizer == "rmsprop":
                self._rms[idx] = (
                    self.rmsprop_gamma * self._rms[idx]
                    + (1.0 - self.rmsprop_gamma) * (gradient**2)
                )
                update = self.learning_rate * gradient / np.sqrt(self._rms[idx] + self.epsilon)
            elif self.optimizer == "adam":
                self._adam_m[idx] = (
                    self.adam_beta1 * self._adam_m[idx] + (1.0 - self.adam_beta1) * gradient
                )
                self._adam_v[idx] = (
                    self.adam_beta2 * self._adam_v[idx] + (1.0 - self.adam_beta2) * (gradient**2)
                )
                m_hat = self._adam_m[idx] / (1.0 - self.adam_beta1**self._step)
                v_hat = self._adam_v[idx] / (1.0 - self.adam_beta2**self._step)
                update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            else:
                raise ValueError(f"Unsupported optimizer: {self.optimizer}")

            self.weights[idx] = self.weights[idx] + update

    def _adapt_lr(self) -> None:
        """
        Increase η when error decreases consistently for adaptive_k epochs;
        decrease η when error increases in the latest epoch.
        """
        if len(self.history) < 2:
            return
        current_loss = self.history[-1].train_loss
        prev_loss = self.history[-2].train_loss
        if current_loss < prev_loss:
            if len(self.history) >= self.adaptive_k:
                recent = [h.train_loss for h in self.history[-self.adaptive_k :]]
                if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                    self.learning_rate = min(
                        self.learning_rate * self.adaptive_increase, 1.0
                    )
        else:
            self.learning_rate *= self.adaptive_decrease

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        metrics = classification_metrics(y_true, predictions, num_classes=self.layer_sizes[-1])
        metrics["loss"] = mse_loss(np.eye(self.layer_sizes[-1])[y_true], probabilities)
        return metrics

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 100,
        verbose: bool = False,
        early_stopping: bool = False,
        patience: int = 20,
        min_delta: float = 1e-5,
        class_weights: np.ndarray | None = None,
        use_weighted_sampling: bool = False,
    ) -> "MultilayerPerceptron":
        rng = np.random.default_rng(self.seed)
        n = len(X_train)
        n_classes = self.layer_sizes[-1]
        y_train_oh = np.eye(n_classes, dtype=np.float32)[y_train]
        best_val_loss = float("inf")
        best_weights = [weight.copy() for weight in self.weights]
        epochs_without_improvement = 0
        should_stop = False
        epoch_offset = self.history[-1].epoch if self.history else 0

        # Pre-compute sampling probabilities once (for weighted sampling)
        sample_probs: np.ndarray | None = None
        if use_weighted_sampling:
            counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
            counts = np.where(counts == 0, 1.0, counts)
            w = (1.0 / counts)[y_train]
            sample_probs = w / w.sum()

        for local_epoch in range(1, epochs + 1):
            epoch = epoch_offset + local_epoch

            if use_weighted_sampling and sample_probs is not None:
                indices = rng.choice(n, size=n, replace=True, p=sample_probs)
            else:
                indices = rng.permutation(n)

            X_epoch = X_train[indices]
            y_epoch = y_train_oh[indices]
            y_labels_epoch = y_train[indices]  # original labels, needed for per-sample weights

            for start in range(0, n, self.batch_size):
                X_batch = X_epoch[start : start + self.batch_size]
                y_batch = y_epoch[start : start + self.batch_size]
                activations = self.forward(X_batch)

                batch_sample_weights: np.ndarray | None = None
                if class_weights is not None:
                    batch_sample_weights = class_weights[y_labels_epoch[start : start + self.batch_size]]

                gradients = self._backward(activations, y_batch, batch_sample_weights)
                self._apply_optimizer(gradients)

            train_proba = self.predict_proba(X_train)
            train_loss = mse_loss(y_train_oh, train_proba)
            train_metrics = classification_metrics(
                y_train, np.argmax(train_proba, axis=1), num_classes=self.layer_sizes[-1]
            )

            val_loss = None
            val_accuracy = None
            val_precision = None
            val_recall = None
            val_f1 = None
            if X_val is not None and y_val is not None:
                y_val_oh = np.eye(self.layer_sizes[-1], dtype=np.float32)[y_val]
                val_proba = self.predict_proba(X_val)
                val_loss = mse_loss(y_val_oh, val_proba)
                val_metrics = classification_metrics(
                    y_val, np.argmax(val_proba, axis=1), num_classes=self.layer_sizes[-1]
                )
                val_accuracy = val_metrics["accuracy"]
                val_precision = val_metrics["precision_macro"]
                val_recall = val_metrics["recall_macro"]
                val_f1 = val_metrics["f1_macro"]

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_weights = [weight.copy() for weight in self.weights]
                    self.best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if early_stopping and epochs_without_improvement >= patience:
                        should_stop = True
                        if verbose:
                            print(
                                f"  early stopping at epoch {epoch} "
                                f"(best epoch={self.best_epoch}, val_loss={best_val_loss:.6f})"
                            )

            self.history.append(
                HistoryPoint(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_metrics["accuracy"],
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    train_precision=train_metrics["precision_macro"],
                    train_recall=train_metrics["recall_macro"],
                    train_f1=train_metrics["f1_macro"],
                    val_precision=val_precision,
                    val_recall=val_recall,
                    val_f1=val_f1,
                )
            )

            if self.optimizer == "adaptive_eta":
                self._adapt_lr()

            if verbose and (local_epoch == 1 or epoch % max(1, epochs // 10) == 0):
                msg = (
                    f"  epoch {epoch:>4}/{epoch_offset + epochs} "
                    f"train_loss={train_loss:.6f} train_acc={train_metrics['accuracy']:.2%}"
                )
                if val_loss is not None and val_accuracy is not None:
                    msg += f" val_loss={val_loss:.6f} val_acc={val_accuracy:.2%}"
                print(msg)

            if should_stop:
                break

        if X_val is not None and y_val is not None and self.best_epoch is not None:
            self.weights = best_weights
        return self

    def save(self, path: Path, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"weight_{idx}": weight for idx, weight in enumerate(self.weights)}
        payload["layer_sizes"] = np.array(self.layer_sizes, dtype=np.int64)
        payload["learning_rate"] = np.array([self.learning_rate], dtype=np.float32)
        payload["activation"] = np.array([self.activation])
        payload["beta"] = np.array([self.beta], dtype=np.float32)
        payload["batch_size"] = np.array([self.batch_size], dtype=np.int64)
        payload["optimizer"] = np.array([self.optimizer])
        payload["momentum_alpha"] = np.array([self.momentum_alpha], dtype=np.float32)
        payload["rmsprop_gamma"] = np.array([self.rmsprop_gamma], dtype=np.float32)
        payload["adaptive_k"] = np.array([self.adaptive_k], dtype=np.int64)
        payload["adaptive_increase"] = np.array([self.adaptive_increase], dtype=np.float32)
        payload["adaptive_decrease"] = np.array([self.adaptive_decrease], dtype=np.float32)
        payload["adam_beta1"] = np.array([self.adam_beta1], dtype=np.float32)
        payload["adam_beta2"] = np.array([self.adam_beta2], dtype=np.float32)
        payload["epsilon"] = np.array([self.epsilon], dtype=np.float32)
        payload["l2_lambda"] = np.array([self.l2_lambda], dtype=np.float32)
        payload["seed"] = np.array([self.seed], dtype=np.int64)
        payload["step"] = np.array([self._step], dtype=np.int64)
        payload["best_epoch"] = np.array(
            [-1 if self.best_epoch is None else self.best_epoch], dtype=np.int64
        )
        for idx, velocity in enumerate(self._velocity):
            payload[f"velocity_{idx}"] = velocity
        for idx, rms in enumerate(self._rms):
            payload[f"rms_{idx}"] = rms
        for idx, adam_m in enumerate(self._adam_m):
            payload[f"adam_m_{idx}"] = adam_m
        for idx, adam_v in enumerate(self._adam_v):
            payload[f"adam_v_{idx}"] = adam_v
        if metadata is not None:
            payload["metadata_json"] = np.array([json.dumps(metadata)])
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "MultilayerPerceptron":
        data = np.load(path, allow_pickle=True)
        layer_sizes = data["layer_sizes"].astype(int).tolist()
        model = cls(
            layer_sizes=layer_sizes,
            learning_rate=float(data["learning_rate"][0]),
            activation=str(data["activation"][0]) if "activation" in data else "logistic",
            beta=float(data["beta"][0]),
            batch_size=int(data["batch_size"][0]),
            optimizer=str(data["optimizer"][0]),
            momentum_alpha=float(data["momentum_alpha"][0]),
            rmsprop_gamma=float(data["rmsprop_gamma"][0]),
            adam_beta1=float(data["adam_beta1"][0]),
            adam_beta2=float(data["adam_beta2"][0]),
            epsilon=float(data["epsilon"][0]),
            l2_lambda=float(data["l2_lambda"][0]),
            adaptive_k=int(data["adaptive_k"][0]) if "adaptive_k" in data else 5,
            adaptive_increase=float(data["adaptive_increase"][0]) if "adaptive_increase" in data else 1.05,
            adaptive_decrease=float(data["adaptive_decrease"][0]) if "adaptive_decrease" in data else 0.5,
            seed=int(data["seed"][0]),
        )
        model.weights = [data[f"weight_{idx}"] for idx in range(len(layer_sizes) - 1)]
        model._velocity = [data[f"velocity_{idx}"] for idx in range(len(layer_sizes) - 1)]
        model._rms = [data[f"rms_{idx}"] for idx in range(len(layer_sizes) - 1)]
        model._adam_m = [data[f"adam_m_{idx}"] for idx in range(len(layer_sizes) - 1)]
        model._adam_v = [data[f"adam_v_{idx}"] for idx in range(len(layer_sizes) - 1)]
        model._step = int(data["step"][0])
        best_epoch = int(data["best_epoch"][0])
        model.best_epoch = None if best_epoch < 0 else best_epoch
        return model

    @staticmethod
    def load_metadata(path: Path) -> dict:
        data = np.load(path, allow_pickle=True)
        if "metadata_json" not in data:
            return {}
        return json.loads(str(data["metadata_json"][0]))
