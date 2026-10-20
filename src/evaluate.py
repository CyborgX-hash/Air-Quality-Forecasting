"""
Evaluation metrics and model comparison utilities for air quality forecasting.

This module is the single source of truth for evaluation metrics across the project.
It implements and wraps standard, verified regression metrics:
- MAE  — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R²   — Coefficient of Determination
- MBE  — Mean Bias Error (systematic over/under-prediction)
- MAPE / sMAPE — Percentage error metrics

Key Design Principles:
1. Standard Implementations: Metrics leverage well-tested scikit-learn routines,
   verified against mathematical definitions.
2. Robust Input Handling: Automatically handles pandas Series, numpy arrays,
   shape validation, and NaN diagnostics.
3. Comparative Benchmarking: Computes relative improvements over baselines
   (% reduction in MAE/RMSE, delta in R²).
4. Error Stratification: Provides slicing helpers to evaluate error by pollution
   concentration tiers (e.g. severe pollution > 150 µg/m³) and temporal bins.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error as skl_mae,
    mean_squared_error as skl_mse,
    r2_score as skl_r2,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input Validation Helpers
# ---------------------------------------------------------------------------

def validate_metric_inputs(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
    allow_nan: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate, align, and cast target and prediction vectors.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted values.
    allow_nan : bool, default False
        If False, drops paired (y_true, y_pred) records where either is NaN.
        If True, preserves arrays as-is.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        1-D float64 numpy arrays of (y_true, y_pred).

    Raises
    ------
    ValueError
        If inputs are empty, have mismatched lengths, or contain non-numeric types.
    """
    # Convert to 1-D numpy arrays
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    if yt.size == 0 or yp.size == 0:
        raise ValueError("Inputs cannot be empty.")

    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true has length {yt.shape[0]}, but y_pred has length {yp.shape[0]}."
        )

    if not allow_nan:
        valid_mask = (~np.isnan(yt)) & (~np.isnan(yp)) & (~np.isinf(yt)) & (~np.isinf(yp))
        dropped_count = int(np.sum(~valid_mask))
        if dropped_count > 0:
            logger.warning(
                "Dropped %d records containing NaN or Inf before metric evaluation.",
                dropped_count,
            )
        yt = yt[valid_mask]
        yp = yp[valid_mask]

        if yt.size == 0:
            raise ValueError("All records contained NaN or Inf; no valid pairs remain for evaluation.")

    return yt, yp


# ---------------------------------------------------------------------------
# Core Metric Implementations
# ---------------------------------------------------------------------------

def mean_absolute_error(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    MAE = (1 / N) * sum(|y_true - y_pred|)

    Units: Same as target (µg/m³ for PM2.5).
    Interpretation: Average magnitude of forecasting errors. Linear penalty;
    does not disproportionately punish large outliers compared to RMSE.
    """
    yt, yp = validate_metric_inputs(y_true, y_pred)
    return float(skl_mae(yt, yp))


def root_mean_squared_error(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    RMSE = sqrt( (1 / N) * sum((y_true - y_pred)^2) )

    Units: Same as target (µg/m³ for PM2.5).
    Interpretation: Root of average squared errors. Because deviations are squared,
    RMSE penalizes larger errors more heavily than MAE. RMSE >= MAE always.
    A large spread between RMSE and MAE indicates high variance in prediction errors
    or the presence of large episodic forecasting misses.
    """
    yt, yp = validate_metric_inputs(y_true, y_pred)
    return float(np.sqrt(skl_mse(yt, yp)))


def r2_score(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
) -> float:
    """
    Calculate Coefficient of Determination (R²).

    R² = 1 - ( sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2) )

    Interpretation: Proportion of target variance explained by the model relative
    to an uninformative baseline that always predicts the sample mean of y_true.
    - R² = 1.0: Perfect forecast.
    - R² = 0.0: Model performs no better than predicting the mean of y_true.
    - R² < 0.0: Model performs worse than the mean-prediction reference. This commonly
      occurs on out-of-sample test/validation splits when predictions have large bias
      or high variance.
    """
    yt, yp = validate_metric_inputs(y_true, y_pred)
    if len(yt) < 2 or np.all(yt == yt[0]):
        logger.warning("Target vector has zero variance; R² is undefined (returning 0.0).")
        return 0.0
    return float(skl_r2(yt, yp))


def mean_bias_error(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
) -> float:
    """
    Calculate Mean Bias Error (MBE).

    MBE = (1 / N) * sum(y_pred - y_true)

    Units: Same as target (µg/m³ for PM2.5).
    Interpretation:
    - MBE > 0: Systematic overprediction on average.
    - MBE < 0: Systematic underprediction on average.
    - MBE = 0: Unbiased on average (errors cancel out).
    """
    yt, yp = validate_metric_inputs(y_true, y_pred)
    return float(np.mean(yp - yt))


def compute_residuals(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
) -> np.ndarray:
    """
    Calculate prediction residuals.

    Residual convention: e_i = y_true - y_pred
    - Positive residual (e > 0): Model underpredicted ground truth.
    - Negative residual (e < 0): Model overpredicted ground truth.

    Returns
    -------
    np.ndarray
        1-D array of residuals.
    """
    yt, yp = validate_metric_inputs(y_true, y_pred)
    return yt - yp


# ---------------------------------------------------------------------------
# Composite Evaluation Functions
# ---------------------------------------------------------------------------

def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute full dictionary of quantitative evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth observations.
    y_pred : array-like
        Predicted values.
    prefix : str, default ""
        Optional prefix for returned dictionary keys (e.g. 'val_' or 'test_').

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: [mae, rmse, r2, mbe, n_obs].
    """
    yt, yp = validate_metric_inputs(y_true, y_pred)
    mae_val = mean_absolute_error(yt, yp)
    rmse_val = root_mean_squared_error(yt, yp)
    r2_val = r2_score(yt, yp)
    mbe_val = mean_bias_error(yt, yp)

    pfx = f"{prefix}_" if prefix and not prefix.endswith("_") else prefix
    return {
        f"{pfx}mae": round(mae_val, 4),
        f"{pfx}rmse": round(rmse_val, 4),
        f"{pfx}r2": round(r2_val, 4),
        f"{pfx}mbe": round(mbe_val, 4),
        f"{pfx}n_obs": int(len(yt)),
    }


def compare_models(
    predictions_dict: Dict[str, Union[np.ndarray, pd.Series, Sequence[float]]],
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    baseline_name: Optional[str] = "persistence",
    training_times: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build a structured model comparison table.

    Parameters
    ----------
    predictions_dict : Dict[str, array-like]
        Mapping from model name to its prediction vector.
    y_true : array-like
        Ground truth observations.
    baseline_name : Optional[str], default 'persistence'
        Model name to compute relative percentage improvements against.
        If None or not present, improvement columns are omitted.
    training_times : Optional[Dict[str, float]], default None
        Optional mapping from model name to training execution time in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by ascending MAE containing:
        - model: Model identifier
        - mae: Mean Absolute Error (µg/m³)
        - rmse: Root Mean Squared Error (µg/m³)
        - r2: Coefficient of Determination
        - mbe: Mean Bias Error (µg/m³)
        - n_obs: Number of evaluated observations
        - training_time_s: Training time in seconds (if provided)
        - mae_improvement_pct: Relative MAE reduction over baseline (%)
        - rmse_improvement_pct: Relative RMSE reduction over baseline (%)
        - r2_delta: Absolute improvement in R² over baseline
    """
    records = []
    base_mae: Optional[float] = None
    base_rmse: Optional[float] = None
    base_r2: Optional[float] = None

    for name, y_pred in predictions_dict.items():
        m = calculate_metrics(y_true, y_pred)
        rec = {
            "model": name,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "r2": m["r2"],
            "mbe": m["mbe"],
            "n_obs": m["n_obs"],
        }
        if training_times and name in training_times:
            rec["training_time_s"] = round(training_times[name], 2)
        records.append(rec)

        if baseline_name and name.lower() == baseline_name.lower():
            base_mae = m["mae"]
            base_rmse = m["rmse"]
            base_r2 = m["r2"]

    df_cmp = pd.DataFrame(records)

    # Compute improvement metrics over baseline
    if base_mae is not None and base_rmse is not None:
        df_cmp["mae_improvement_pct"] = np.round(
            (base_mae - df_cmp["mae"]) / base_mae * 100.0, 2
        )
        df_cmp["rmse_improvement_pct"] = np.round(
            (base_rmse - df_cmp["rmse"]) / base_rmse * 100.0, 2
        )
        if base_r2 is not None:
            df_cmp["r2_delta"] = np.round(df_cmp["r2"] - base_r2, 4)

    # Sort by primary model selection criterion: lowest MAE
    df_cmp = df_cmp.sort_values(by="mae", ascending=True).reset_index(drop=True)
    return df_cmp


# ---------------------------------------------------------------------------
# Stratified Error Analysis Helpers
# ---------------------------------------------------------------------------

# Standard China / WHO Ambient Air Quality Standards for PM2.5 (24h breakpoints)
DEFAULT_POLLUTION_BINS: List[float] = [0.0, 35.0, 75.0, 115.0, 150.0, 250.0, float("inf")]
DEFAULT_POLLUTION_LABELS: List[str] = [
    "Excellent (0-35)",
    "Good (35-75)",
    "Light Pollution (75-115)",
    "Moderate Pollution (115-150)",
    "Heavy Pollution (150-250)",
    "Severe Smog (>250)",
]


def evaluate_by_pollution_tier(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
    bins: Sequence[float] = DEFAULT_POLLUTION_BINS,
    labels: Sequence[str] = DEFAULT_POLLUTION_LABELS,
) -> pd.DataFrame:
    """
    Stratify prediction errors across air quality pollution concentration tiers.

    Investigates whether high pollution episodes suffer from systematic underprediction
    or elevated variance.

    Parameters
    ----------
    y_true : array-like
        Ground truth observations.
    y_pred : array-like
        Predicted values.
    bins : Sequence[float]
        Breakpoints defining the concentration tiers.
    labels : Sequence[str]
        Category names for each bin.

    Returns
    -------
    pd.DataFrame
        Table summarizing MAE, RMSE, MBE, count, and percentage of observations per tier.
    """
    yt, yp = validate_metric_inputs(y_true, y_pred)
    tier_series = pd.cut(yt, bins=bins, labels=labels, right=True, include_lowest=True)

    records = []
    total_n = len(yt)

    for label in labels:
        mask = tier_series == label
        sub_yt = yt[mask]
        sub_yp = yp[mask]
        count = int(np.sum(mask))

        if count > 0:
            mae = skl_mae(sub_yt, sub_yp)
            rmse = np.sqrt(skl_mse(sub_yt, sub_yp))
            mbe = np.mean(sub_yp - sub_yt)
            mean_actual = np.mean(sub_yt)
            mean_predicted = np.mean(sub_yp)
        else:
            mae = np.nan
            rmse = np.nan
            mbe = np.nan
            mean_actual = np.nan
            mean_predicted = np.nan

        records.append({
            "tier": label,
            "count": count,
            "pct_of_total": round(count / total_n * 100.0, 2),
            "mean_actual": round(mean_actual, 2) if not np.isnan(mean_actual) else np.nan,
            "mean_predicted": round(mean_predicted, 2) if not np.isnan(mean_predicted) else np.nan,
            "mae": round(mae, 2) if not np.isnan(mae) else np.nan,
            "rmse": round(rmse, 2) if not np.isnan(rmse) else np.nan,
            "mbe": round(mbe, 2) if not np.isnan(mbe) else np.nan,
        })

    return pd.DataFrame(records)


def evaluate_by_temporal_group(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
    timestamps: Union[pd.DatetimeIndex, pd.Series, Sequence[pd.Timestamp]],
    group_by: str = "hour",
) -> pd.DataFrame:
    """
    Compute error metrics grouped by temporal cycles (hour of day, month, season).

    Parameters
    ----------
    y_true : array-like
        Ground truth observations.
    y_pred : array-like
        Predicted values.
    timestamps : array-like of Timestamps
        Aligned timestamps for each observation.
    group_by : str, default 'hour'
        Grouping dimension: 'hour', 'month', 'season', or 'day_of_week'.

    Returns
    -------
    pd.DataFrame
        Table containing grouped MAE, RMSE, MBE, and count.
    """
    yt, yp = validate_metric_inputs(y_true, y_pred)
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))

    if len(ts) != len(yt):
        raise ValueError(
            f"Length mismatch between timestamps ({len(ts)}) and observations ({len(yt)})."
        )

    df_temp = pd.DataFrame({
        "actual": yt,
        "predicted": yp,
        "datetime": ts,
    })

    if group_by == "hour":
        df_temp["group"] = df_temp["datetime"].dt.hour
    elif group_by == "month":
        df_temp["group"] = df_temp["datetime"].dt.month
    elif group_by == "day_of_week":
        df_temp["group"] = df_temp["datetime"].dt.dayofweek
    elif group_by == "season":
        # Meteorological seasons: Spring(3-5), Summer(6-8), Autumn(9-11), Winter(12,1,2)
        month = df_temp["datetime"].dt.month
        df_temp["group"] = np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
            ["Winter", "Spring", "Summer", "Autumn"],
            default="Unknown",
        )
    else:
        raise ValueError(f"Unknown group_by option: {group_by}. Choose 'hour', 'month', 'season', or 'day_of_week'.")

    records = []
    for grp_val, grp_df in df_temp.groupby("group", sort=True):
        m = calculate_metrics(grp_df["actual"], grp_df["predicted"])
        records.append({
            group_by: grp_val,
            "count": m["n_obs"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "mbe": m["mbe"],
            "r2": m["r2"],
        })

    return pd.DataFrame(records)


def analyze_prediction_lag(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Investigate whether predictions systematically lag behind rapid ground-truth transitions.

    Calculates cross-correlation between y_pred(t) and y_true(t + k) for k in [-max_lag, max_lag].
    If cross-correlation peaks at k < 0 (i.e. with y_true(t - 1)), the model is largely
    mimicking the previous hour's observation rather than anticipating shifts.

    Parameters
    ----------
    y_true : array-like
        Ground truth observations.
    y_pred : array-like
        Predicted values.
    max_lag : int, default 5
        Maximum lag offset in hours to test.

    Returns
    -------
    pd.DataFrame
        DataFrame with [lag_hours, correlation, interpretation].
    """
    yt, yp = validate_metric_inputs(y_true, y_pred)
    s_true = pd.Series(yt)
    s_pred = pd.Series(yp)

    records = []
    for k in range(-max_lag, max_lag + 1):
        corr = float(s_pred.corr(s_true.shift(-k)))
        if k < 0:
            interp = f"Pred(t) vs Actual(t - {-k}h) [Lag-{-k} memory]"
        elif k > 0:
            interp = f"Pred(t) vs Actual(t + {k}h) [Lead-{k} anticipation]"
        else:
            interp = "Pred(t) vs Actual(t) [Target synchronous]"

        records.append({
            "lag_hours": k,
            "pearson_correlation": round(corr, 4),
            "interpretation": interp,
        })

    return pd.DataFrame(records)
