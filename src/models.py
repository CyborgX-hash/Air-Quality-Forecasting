# src/models.py
"""Machine Learning model definitions and utilities for Phase 1.

This module provides:
- Persistence baseline (last observed PM2.5).
- Linear regression baseline.
- Gradient Boosting Regressor with a lightweight hyper‑parameter tuner.
- Evaluation helpers (MAE, RMSE, R²).
- Feature‑importance extraction for the Gradient Boosting model.

All functions expect pre‑engineered feature matrices (`X`) and target vectors (`y`).
Pre‑processing (scaling, lag creation) is handled elsewhere (see ``src/features.py`` and ``src/preprocessing.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

class PersistenceBaseline(BaseEstimator):
    """Predict the next hour's PM2.5 using the latest available observation.

    If a feature column (e.g. 'PM2.5') is present in X, the model returns
    that contemporaneous observation (standard 1-step time-series persistence: y_{t+1} = y_t).
    Otherwise, or if feature_col is None, it memorises the last target value seen during
    ``fit`` and returns that constant for every row in ``predict``.
    """

    def __init__(self, feature_col: str | None = "PM2.5") -> None:
        self.feature_col = feature_col
        self._last_value: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceBaseline":
        if len(y) == 0:
            raise ValueError("Target series `y` is empty.")
        self._last_value = float(y.iloc[-1])
        logger.debug("PersistenceBaseline fitted – last target: %s", self._last_value)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._last_value is None:
            raise RuntimeError("Model has not been fitted.")
        if (
            self.feature_col is not None
            and isinstance(X, pd.DataFrame)
            and self.feature_col in X.columns
        ):
            return X[self.feature_col].to_numpy(dtype=float)
        return np.full(shape=(len(X),), fill_value=self._last_value, dtype=float)


class LinearRegressionBaseline(BaseEstimator):
    """Linear regression wrapper to keep a consistent API.
    """

    def __init__(self) -> None:
        self._model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionBaseline":
        self._model.fit(X, y)
        logger.debug("LinearRegressionBaseline fitted – coefficients: %s", self._model.coef_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# Gradient Boosting utilities
# ---------------------------------------------------------------------------

def _default_gb_params() -> Dict[str, Any]:
    """Return a sensible default hyper‑parameter set.

    Values are chosen to be inexpensive while still giving a reasonable model.
    They can be overridden via the JSON file at ``experiments/configs/gradient_boosting.json``.
    """
    return {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "subsample": 0.8,
        "random_state": 42,
    }


def load_gb_config(config_path: Path | str = "experiments/configs/gradient_boosting.json") -> Dict[str, Any]:
    """Load a JSON configuration for Gradient Boosting.

    If the file is missing, defaults are returned.
    """
    path = Path(config_path)
    if not path.is_file():
        logger.info("GB config %s not found – using defaults.", path)
        return _default_gb_params()
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    logger.debug("Loaded GB config: %s", cfg)
    return cfg


def build_gradient_boosting(**overrides: Any) -> GradientBoostingRegressor:
    """Create a ``GradientBoostingRegressor`` with defaults that can be overridden.
    """
    params = _default_gb_params()
    params.update(overrides)
    if "random_state" not in params:
        params["random_state"] = 42
    model = GradientBoostingRegressor(**params)
    logger.debug("GradientBoostingRegressor created with params: %s", params)
    return model


def tune_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_grid: Dict[str, list],
    random_state: int = 42,
) -> Tuple[GradientBoostingRegressor, Dict[str, Any]]:
    """Lightweight manual grid‑search using MAE on the validation set.
    """
    from itertools import product

    best_score = float("inf")
    best_params: Dict[str, Any] = {}
    best_model: GradientBoostingRegressor | None = None

    keys, values = zip(*param_grid.items())
    for combo in product(*values):
        params = dict(zip(keys, combo))
        params["random_state"] = random_state
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_val, model.predict(X_val))
        logger.debug("GB combo %s – MAE: %.4f", params, mae)
        if mae < best_score:
            best_score = mae
            best_params = params
            best_model = model
    if best_model is None:
        raise RuntimeError("No model was trained during hyper‑parameter search.")
    logger.info("Best GB params: %s with MAE %.4f", best_params, best_score)
    return best_model, best_params


# ---------------------------------------------------------------------------
# Training orchestration helpers
# ---------------------------------------------------------------------------

def build_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, BaseEstimator]:
    """Fit persistence and linear‑regression baselines.
    """
    models: Dict[str, BaseEstimator] = {}
    models["persistence"] = PersistenceBaseline(feature_col="PM2.5").fit(X_train, y_train)
    models["constant_persistence"] = PersistenceBaseline(feature_col=None).fit(X_train, y_train)
    models["linear_regression"] = LinearRegressionBaseline().fit(X_train, y_train)
    return models


def _calc_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    from src.evaluate import calculate_metrics
    m = calculate_metrics(y_true, y_pred)
    return {"mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]}


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Tuple[BaseEstimator, Dict[str, float]]]:
    """Train all Phase 1 models and compute validation metrics.

    Returns a mapping ``model_name -> (model, metrics)``.
    """
    results: Dict[str, Tuple[BaseEstimator, Dict[str, float]]] = {}
    # Baselines
    baselines = build_baseline_models(X_train, y_train)
    for name, model in baselines.items():
        preds = model.predict(X_val)
        results[name] = (model, _calc_metrics(y_val, preds))
    # Gradient Boosting (default params)
    gb = build_gradient_boosting()
    gb.fit(X_train, y_train)
    results["gradient_boosting_default"] = (gb, _calc_metrics(y_val, gb.predict(X_val)))
    return results


def get_feature_importance(
    model: GradientBoostingRegressor,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model lacks feature_importances_ attribute.")
    importances = model.feature_importances_
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)

__all__ = [
    "PersistenceBaseline",
    "LinearRegressionBaseline",
    "build_gradient_boosting",
    "load_gb_config",
    "tune_gradient_boosting",
    "build_baseline_models",
    "train_models",
    "get_feature_importance",
]
