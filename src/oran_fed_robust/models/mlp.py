"""Two-layer ReLU MLP classifier in NumPy (non-convex control model).

Provided as a drop-in alternative to SoftmaxClassifier behind the identical
get_params/set_params/dim/predict/local_train interface, so the exact same
federated pipeline, attacks, and aggregation rules can be re-run on a
*non-convex* model. Used for the non-convex robustness confirmation
(Appendix): it verifies that the IPM failure of the distance/order-statistic
rules is not an artifact of the convex softmax model.
"""
from __future__ import annotations

import numpy as np


class MLPClassifier:
    def __init__(self, n_features: int, n_classes: int, hidden: int = 32, seed: int = 0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden = hidden
        rng = np.random.default_rng(seed)
        # He initialization for the ReLU hidden layer.
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / n_features), size=(n_features, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, n_classes))
        self.b2 = np.zeros(n_classes)

    # --- parameter (de)serialization -------------------------------------
    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1.ravel(),
                               self.W2.ravel(), self.b2.ravel()])

    def set_params(self, vec: np.ndarray) -> None:
        d, h, c = self.n_features, self.hidden, self.n_classes
        i = 0
        self.W1 = vec[i:i + d * h].reshape(d, h).copy(); i += d * h
        self.b1 = vec[i:i + h].reshape(h).copy(); i += h
        self.W2 = vec[i:i + h * c].reshape(h, c).copy(); i += h * c
        self.b2 = vec[i:i + c].reshape(c).copy()

    @property
    def dim(self) -> int:
        d, h, c = self.n_features, self.hidden, self.n_classes
        return d * h + h + h * c + c

    # --- forward / loss ---------------------------------------------------
    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _forward(self, x: np.ndarray):
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, z2

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, _, z2 = self._forward(x)
        return self._softmax(z2).argmax(axis=1)

    def _grads(self, x: np.ndarray, y: np.ndarray):
        n = len(y)
        z1, a1, z2 = self._forward(x)
        probs = self._softmax(z2)
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), y] = 1.0
        dz2 = (probs - onehot) / n
        gW2 = a1.T @ dz2
        gb2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        gW1 = x.T @ dz1
        gb1 = dz1.sum(axis=0)
        return gW1, gb1, gW2, gb2

    # --- local training ---------------------------------------------------
    def local_train(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float,
                    batch_size: int, rng: np.random.Generator) -> np.ndarray:
        n = len(y)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                gW1, gb1, gW2, gb2 = self._grads(x[idx], y[idx])
                self.W1 -= lr * gW1
                self.b1 -= lr * gb1
                self.W2 -= lr * gW2
                self.b2 -= lr * gb2
        return self.get_params()
