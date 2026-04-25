from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RidgeRegressor:
    feature_names: list[str]
    alpha: float = 1.0
    weights: np.ndarray | None = None
    bias: float = 0.0
    means: np.ndarray | None = None
    stds: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        self.means = x.mean(axis=0)
        self.stds = x.std(axis=0)
        self.stds[self.stds == 0] = 1.0
        x_scaled = (x - self.means) / self.stds

        ones = np.ones((x_scaled.shape[0], 1))
        design = np.hstack([ones, x_scaled])
        identity = np.eye(design.shape[1])
        identity[0, 0] = 0.0

        solution = np.linalg.solve(design.T @ design + self.alpha * identity, design.T @ y)
        self.bias = float(solution[0])
        self.weights = solution[1:]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None or self.means is None or self.stds is None:
            raise ValueError("Model must be fitted before prediction.")
        x_scaled = (x - self.means) / self.stds
        return self.bias + x_scaled @ self.weights

    def save(self, path: str | Path, metrics: dict[str, float] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": self.feature_names,
            "alpha": self.alpha,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": self.bias,
            "means": self.means.tolist() if self.means is not None else None,
            "stds": self.stds.tolist() if self.stds is not None else None,
            "metrics": metrics or {},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RidgeRegressor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(feature_names=payload["feature_names"], alpha=payload["alpha"])
        model.weights = np.asarray(payload["weights"], dtype=float)
        model.bias = float(payload["bias"])
        model.means = np.asarray(payload["means"], dtype=float)
        model.stds = np.asarray(payload["stds"], dtype=float)
        return model


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    split_idx = int(len(indices) * (1.0 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    total_var = float(np.sum((y_true - y_true.mean()) ** 2))
    residual_var = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - (residual_var / total_var if total_var else 0.0)
    return {"rmse": rmse, "mae": mae, "r2": r2}
