"""
Prediction, Inference, and Multi-Horizon Evaluation Module for Phase 2 LSTM.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Harsh Kumar (Deep Learning Engineer – LSTM Architecture and Training)

Responsibilities:
1. Load saved Keras model checkpoints (.keras).
2. Generate multi-horizon predictions for validation and test sequence tensors.
3. Automatically apply target inverse-transformation back to physical PM2.5 units (µg/m³).
4. Compute standard regression evaluation metrics (MAE, RMSE, R², MBE) per forecast horizon.
5. Provide structured evaluation tables consumable by downstream reporting and visualization.
"""

from __future__ import annotations

import os
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from src.evaluate import calculate_metrics, validate_metric_inputs
from src.phase2.data_pipeline import StationDataSplits
from src.phase2.lstm_model import DEFAULT_HORIZONS, HORIZON_NAMES
from src.phase2.scaling import StationScaler

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSummary:
    """Encapsulates multi-horizon prediction arrays and metric tables."""
    station_name: str
    partition: str
    window_size: int
    horizons: List[int]
    y_pred: np.ndarray  # Shape (N, len(horizons)) in µg/m³
    y_true: np.ndarray  # Shape (N, len(horizons)) in µg/m³
    timestamps: pd.DatetimeIndex
    metrics_by_horizon: Dict[str, Dict[str, float]]
    metrics_table: pd.DataFrame
    macro_mae: float
    macro_rmse: float
    macro_r2: float


def load_trained_model(model_path: Union[str, Path]) -> keras.Model:
    """
    Load a compiled Keras LSTM model checkpoint from disk.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to .keras model checkpoint file.

    Returns
    -------
    keras.Model
        Loaded and ready Keras model.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")

    # Compile with custom loss objects if needed
    model = keras.models.load_model(path, compile=True)
    logger.debug("Successfully loaded LSTM model from %s", path)
    return model


def evaluate_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Calculate MAE, RMSE, R2, and MBE across each forecast horizon.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth unscaled PM2.5 concentrations of shape (N, H) or (N,).
    y_pred : np.ndarray
        Predicted unscaled PM2.5 concentrations of shape (N, H) or (N,).
    horizons : Optional[Sequence[int]], default [1, 3, 6, 12, 24]
        Forecast horizon offsets in hours.

    Returns
    -------
    Tuple[Dict[str, Dict[str, float]], pd.DataFrame]
        - Dictionary mapping horizon label (e.g. '1h') to metrics dict.
        - Clean tabular summary DataFrame.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if y_t.ndim == 1:
        y_t = y_t.reshape(-1, 1)
    if y_p.ndim == 1:
        y_p = y_p.reshape(-1, 1)

    num_horizons = y_t.shape[1]
    active_horizons = list(horizons) if horizons is not None else DEFAULT_HORIZONS[:num_horizons]

    metrics_by_horizon: Dict[str, Dict[str, float]] = {}
    table_rows: List[Dict[str, Any]] = []

    for idx, h in enumerate(active_horizons):
        h_label = f"{h}h"
        yt_col = y_t[:, idx]
        yp_col = y_p[:, idx]

        # Use project single-source-of-truth metric calculator
        m = calculate_metrics(y_true=yt_col, y_pred=yp_col)
        metrics_by_horizon[h_label] = m

        table_rows.append({
            "horizon": h_label,
            "horizon_hours": h,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "r2": m["r2"],
            "mbe": m["mbe"],
            "n_obs": m.get("n_obs", int(len(yt_col))),
        })

    summary_df = pd.DataFrame(table_rows)
    return metrics_by_horizon, summary_df


def predict_and_evaluate(
    model: keras.Model,
    splits: StationDataSplits,
    partition: str = "val",
    window_size: int = 24,
    horizons: Optional[Sequence[int]] = None,
    use_context: bool = True,
    batch_size: int = 64,
) -> EvaluationSummary:
    """
    Generate predictions, invert to physical units (µg/m³), and evaluate metrics.

    Parameters
    ----------
    model : keras.Model
        Trained Keras LSTM model.
    splits : StationDataSplits
        Data container for the target monitoring station.
    partition : str, default 'val'
        Partition to evaluate: 'train', 'val', or 'test'.
    window_size : int, default 24
        Historical lookback window W.
    horizons : Optional[Sequence[int]], default [1, 3, 6, 12, 24]
        Forecast horizons to evaluate.
    use_context : bool, default True
        Whether to use warm-up context slices at partition boundaries.
    batch_size : int, default 64
        Batch size for model inference.

    Returns
    -------
    EvaluationSummary
        Comprehensive container with unscaled arrays and evaluation tables.
    """
    active_horizons = list(horizons) if horizons is not None else DEFAULT_HORIZONS

    # Extract sequences for partition
    if partition.lower() == "train":
        X_scaled, y_scaled, ts = splits.get_train_sequences(
            window_size=window_size,
            horizons=active_horizons,
            scaled=True,
            scale_target=True,
        )
    elif partition.lower() == "val":
        X_scaled, y_scaled, ts = splits.get_val_sequences(
            window_size=window_size,
            horizons=active_horizons,
            use_context=use_context,
            scaled=True,
            scale_target=True,
        )
    elif partition.lower() == "test":
        X_scaled, y_scaled, ts = splits.get_test_sequences(
            window_size=window_size,
            horizons=active_horizons,
            use_context=use_context,
            scaled=True,
            scale_target=True,
        )
    else:
        raise ValueError(f"Unknown partition '{partition}'. Must be 'train', 'val', or 'test'.")

    if len(X_scaled) == 0:
        raise ValueError(f"No valid sequences generated for partition '{partition}' with window_size={window_size}")

    # Forward pass through trained model
    preds_scaled = model.predict(X_scaled, batch_size=batch_size, verbose=0)

    # Invert predictions and targets back to µg/m³
    y_pred_actual = splits.inverse_transform_target(preds_scaled)
    y_true_actual = splits.inverse_transform_target(y_scaled)

    # Ensure 2D shape (N, H)
    if y_pred_actual.ndim == 1:
        y_pred_actual = y_pred_actual.reshape(-1, 1)
    if y_true_actual.ndim == 1:
        y_true_actual = y_true_actual.reshape(-1, 1)

    # Compute metrics per horizon
    metrics_by_h, table_df = evaluate_horizon_metrics(
        y_true=y_true_actual,
        y_pred=y_pred_actual,
        horizons=active_horizons,
    )

    macro_mae = float(table_df["mae"].mean())
    macro_rmse = float(table_df["rmse"].mean())
    macro_r2 = float(table_df["r2"].mean())

    return EvaluationSummary(
        station_name=splits.station_name,
        partition=partition,
        window_size=window_size,
        horizons=active_horizons,
        y_pred=y_pred_actual,
        y_true=y_true_actual,
        timestamps=ts,
        metrics_by_horizon=metrics_by_h,
        metrics_table=table_df,
        macro_mae=macro_mae,
        macro_rmse=macro_rmse,
        macro_r2=macro_r2,
    )


__all__ = [
    "EvaluationSummary",
    "load_trained_model",
    "evaluate_horizon_metrics",
    "predict_and_evaluate",
]
