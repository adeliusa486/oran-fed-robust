"""Evaluation metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def evaluate_model(model, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(x)
    return {"accuracy": accuracy(y, y_pred), "macro_f1": macro_f1(y, y_pred)}
