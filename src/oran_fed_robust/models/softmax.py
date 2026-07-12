"""Multinomial logistic regression (softmax) classifier in NumPy.

Chosen as the xApp task model because it is convex, fast, CPU-only, and exposes
a flat parameter vector -- ideal for studying aggregation behaviour without GPU
or framework overhead. A PyTorch model can be substituted behind the same
get_params/set_params/local_train interface (see docs/architecture.md).
"""
from __future__ import annotations

import numpy as np


class SoftmaxClassifier:
    def __init__(self, n_features: int, n_classes: int, seed: int = 0):
        self.n_features = n_features
        self.n_classes = n_classes
        rng = np.random.default_rng(seed)
        # weights [d, C] and bias [C], stored together as a flat vector externally
        self.W = rng.normal(0.0, 0.01, size=(n_features, n_classes))
        self.b = np.zeros(n_classes)

    # --- parameter (de)serialization -------------------------------------
    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W.ravel(), self.b.ravel()])

    def set_params(self, vec: np.ndarray) -> None:
        d, c = self.n_features, self.n_classes
        self.W = vec[: d * c].reshape(d, c).copy()
        self.b = vec[d * c :].reshape(c).copy()

    @property
    def dim(self) -> int:
        return self.n_features * self.n_classes + self.n_classes

    # --- forward / loss ---------------------------------------------------
    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._softmax(x @ self.W + self.b).argmax(axis=1)

    def _grad(self, x: np.ndarray, y: np.ndarray):
        n = len(y)
        probs = self._softmax(x @ self.W + self.b)
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), y] = 1.0
        diff = (probs - onehot) / n
        gW = x.T @ diff
        gb = diff.sum(axis=0)
        return gW, gb

    # --- local training ---------------------------------------------------
    def local_train(
        self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Run local SGD; return the resulting flat parameter vector."""
        n = len(y)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                gW, gb = self._grad(x[idx], y[idx])
                self.W -= lr * gW
                self.b -= lr * gb
        return self.get_params()
