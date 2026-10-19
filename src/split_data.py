"""
Chronological Splitting and Leakage Validation Module.

Key Design Principles:
1. Strict Chronological Partitioning: Training, validation, and test sets are partitioned
   in strictly forward time order. Random shuffling is strictly prohibited.
2. Synchronized Multi-Station Boundaries: Exact timestamp boundaries are established across
   all stations simultaneously using sorted unique timestamps.
3. Cold-Start / Overlap Boundary Handling: Features should be engineered on contiguous
   time series prior to splitting so historical lags at the start of validation/test
   have valid context.
4. Embargo / Purging Support: Configurable embargo parameter to drop boundary transition
   records where the training target timestamp touches the validation start timestamp.
5. Learned Transformation Isolation: Scalers and encoders are fitted strictly on the
   training set, then applied downstream to validation and test sets.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

logger = logging.getLogger(__name__)


def get_split_boundaries(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = "datetime",
) -> Dict[str, Any]:
    """
    Calculate exact chronological timestamp cutoffs and expected sample proportions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing timestamp column.
    train_ratio : float, default 0.70
        Proportion of time series allocated to training.
    val_ratio : float, default 0.15
        Proportion of time series allocated to validation.
    test_ratio : float, default 0.15
        Proportion of time series allocated to final testing.
    timestamp_col : str, default 'datetime'
        Name of timestamp column.

    Returns
    -------
    Dict[str, Any]
        Dictionary with cutoff timestamps, counts, and percentages.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-5):
        raise ValueError(
            f"Split ratios must sum to 1.0. Given: train={train_ratio}, "
            f"val={val_ratio}, test={test_ratio} (sum={total_ratio})"
        )

    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    unique_ts = np.sort(df[timestamp_col].unique())
    n_ts = len(unique_ts)

    if n_ts < 3:
        raise ValueError(f"Insufficient distinct timestamps for 3-way split: found {n_ts}.")

    train_end_idx = int(np.floor(n_ts * train_ratio))
    val_end_idx = int(np.floor(n_ts * (train_ratio + val_ratio)))

    # Prevent empty partitions
    train_end_idx = max(1, min(train_end_idx, n_ts - 2))
    val_end_idx = max(train_end_idx + 1, min(val_end_idx, n_ts - 1))

    train_start_ts = pd.Timestamp(unique_ts[0])
    train_end_ts = pd.Timestamp(unique_ts[train_end_idx - 1])

    val_start_ts = pd.Timestamp(unique_ts[train_end_idx])
    val_end_ts = pd.Timestamp(unique_ts[val_end_idx - 1])

    test_start_ts = pd.Timestamp(unique_ts[val_end_idx])
    test_end_ts = pd.Timestamp(unique_ts[-1])

    return {
        "num_unique_timestamps": n_ts,
        "train_timestamps_count": train_end_idx,
        "val_timestamps_count": val_end_idx - train_end_idx,
        "test_timestamps_count": n_ts - val_end_idx,
        "train_start": train_start_ts,
        "train_end": train_end_ts,
        "val_start": val_start_ts,
        "val_end": val_end_ts,
        "test_start": test_start_ts,
        "test_end": test_end_ts,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = "datetime",
    station_col: str = "station",
    embargo_hours: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split time-series DataFrame into strictly chronological train, val, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    train_ratio : float, default 0.70
        Fraction of temporal period for training.
    val_ratio : float, default 0.15
        Fraction of temporal period for validation (hyperparameter tuning).
    test_ratio : float, default 0.15
        Fraction of temporal period for final model test evaluation.
    timestamp_col : str, default 'datetime'
        Name of timestamp column.
    station_col : str, default 'station'
        Name of station column.
    embargo_hours : int, default 0
        Number of hours before the validation and test start boundaries to drop
        from training and validation respectively, ensuring zero target overlap
        across splits.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        - train_df: Training set DataFrame.
        - val_df: Validation set DataFrame.
        - test_df: Test set DataFrame.
        - report: Detailed split report dictionary.
    """
    boundaries = get_split_boundaries(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        timestamp_col=timestamp_col,
    )

    train_end = boundaries["train_end"]
    val_start = boundaries["val_start"]
    val_end = boundaries["val_end"]
    test_start = boundaries["test_start"]

    ts_series = pd.to_datetime(df[timestamp_col])

    # Apply embargo if requested
    effective_train_end = train_end
    effective_val_end = val_end
    if embargo_hours > 0:
        embargo_delta = pd.Timedelta(hours=embargo_hours)
        effective_train_end = train_end - embargo_delta
        effective_val_end = val_end - embargo_delta

    train_mask = ts_series <= effective_train_end
    val_mask = (ts_series >= val_start) & (ts_series <= effective_val_end)
    test_mask = ts_series >= test_start

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    total_records = len(df)
    train_records = len(train_df)
    val_records = len(val_df)
    test_records = len(test_df)
    embargoed_records = total_records - (train_records + val_records + test_records)

    stations = sorted(df[station_col].unique().tolist()) if station_col in df.columns else ["all"]

    report: Dict[str, Any] = {
        "total_records": total_records,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "embargoed_records": embargoed_records,
        "train_percentage": round(train_records / total_records * 100, 2),
        "val_percentage": round(val_records / total_records * 100, 2),
        "test_percentage": round(test_records / total_records * 100, 2),
        "train_start": str(boundaries["train_start"]),
        "train_end": str(effective_train_end),
        "val_start": str(val_start),
        "val_end": str(effective_val_end),
        "test_start": str(test_start),
        "test_end": str(boundaries["test_end"]),
        "embargo_hours": embargo_hours,
        "stations": stations,
        "records_per_station": {
            st: {
                "train": int((train_df[station_col] == st).sum()) if station_col in train_df.columns else train_records,
                "val": int((val_df[station_col] == st).sum()) if station_col in val_df.columns else val_records,
                "test": int((test_df[station_col] == st).sum()) if station_col in test_df.columns else test_records,
            }
            for st in stations
        },
    }

    return train_df, val_df, test_df, report


def validate_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = "datetime",
    target_col: str = "target_pm25",
    horizon_hours: int = 1,
) -> Dict[str, Any]:
    """
    Perform rigorous anti-leakage audit on chronological dataset partitions.

    Checks:
    1. Strict timestamp monotonicity (max(train) < min(val) < min(test)).
    2. Zero set overlap among timestamps across partitions.
    3. Target lookahead audit: Verify that future targets in the train set
       do not cross into validation evaluation, and val targets do not cross
       into test evaluation.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training partition.
    val_df : pd.DataFrame
        Validation partition.
    test_df : pd.DataFrame
        Testing partition.
    timestamp_col : str, default 'datetime'
        Name of timestamp column.
    target_col : str, default 'target_pm25'
        Name of target column.
    horizon_hours : int, default 1
        Forecasting horizon in hours.

    Returns
    -------
    Dict[str, Any]
        Audit dictionary with pass/fail boolean flags and diagnostic details.
    """
    ts_train = pd.to_datetime(train_df[timestamp_col])
    ts_val = pd.to_datetime(val_df[timestamp_col])
    ts_test = pd.to_datetime(test_df[timestamp_col])

    # 1. Monotonic partition boundaries
    train_max = ts_train.max()
    val_min = ts_val.min()
    val_max = ts_val.max()
    test_min = ts_test.min()

    is_chronological = (train_max < val_min) and (val_max < test_min)

    # 2. Overlap checks
    train_set = set(ts_train.unique())
    val_set = set(ts_val.unique())
    test_set = set(ts_test.unique())

    overlap_train_val = len(train_set.intersection(val_set))
    overlap_val_test = len(val_set.intersection(test_set))
    overlap_train_test = len(train_set.intersection(test_set))
    zero_timestamp_overlap = (
        overlap_train_val == 0 and overlap_val_test == 0 and overlap_train_test == 0
    )

    # 3. Target future timestamp analysis
    # For a row at timestamp t, its target belongs to timestamp t + horizon
    train_last_target_ts = train_max + pd.Timedelta(hours=horizon_hours)
    val_last_target_ts = val_max + pd.Timedelta(hours=horizon_hours)

    # If unembargoed: train_last_target_ts == val_min (the boundary transition)
    # If embargoed: train_last_target_ts < val_min
    train_target_leaks_into_val = bool(train_last_target_ts > val_min)
    val_target_leaks_into_test = bool(val_last_target_ts > test_min)

    passed_all = (
        is_chronological
        and zero_timestamp_overlap
        and not train_target_leaks_into_val
        and not val_target_leaks_into_test
    )

    return {
        "passed_all_checks": passed_all,
        "is_chronological": is_chronological,
        "zero_timestamp_overlap": zero_timestamp_overlap,
        "train_max_timestamp": str(train_max),
        "val_min_timestamp": str(val_min),
        "val_max_timestamp": str(val_max),
        "test_min_timestamp": str(test_min),
        "train_last_target_timestamp": str(train_last_target_ts),
        "val_last_target_timestamp": str(val_last_target_ts),
        "train_target_leaks_into_val": train_target_leaks_into_val,
        "val_target_leaks_into_test": val_target_leaks_into_test,
        "overlap_counts": {
            "train_val": overlap_train_val,
            "val_test": overlap_val_test,
            "train_test": overlap_train_test,
        },
    }


def fit_transform_scaler(
    train_X: pd.DataFrame,
    val_X: pd.DataFrame,
    test_X: pd.DataFrame,
    scaler_type: str = "standard",
    numeric_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """
    Fit a learned scaler strictly on the training set and transform validation and test sets.

    Demonstrates and enforces the core ML rule:
    Never fit statistics (mean, std, min, max, quantiles) on validation or test sets.

    Parameters
    ----------
    train_X : pd.DataFrame
        Training feature matrix.
    val_X : pd.DataFrame
        Validation feature matrix.
    test_X : pd.DataFrame
        Testing feature matrix.
    scaler_type : str, default 'standard'
        Type of scaler: 'standard' (StandardScaler), 'robust' (RobustScaler),
        or 'minmax' (MinMaxScaler).
    numeric_cols : Optional[Sequence[str]], default None
        Specific numeric columns to scale. If None, all numeric columns are scaled.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]
        Scaled train_X, val_X, test_X, and the fitted scaler object.
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}. Choose 'standard', 'robust', or 'minmax'.")

    cols_to_scale = list(numeric_cols) if numeric_cols else train_X.select_dtypes(include=[np.number]).columns.tolist()

    train_scaled = train_X.copy()
    val_scaled = val_X.copy()
    test_scaled = test_X.copy()

    if not cols_to_scale:
        return train_scaled, val_scaled, test_scaled, scaler

    # Fit strictly on train
    scaler.fit(train_scaled[cols_to_scale])

    # Transform all three partitions using fitted parameters
    train_scaled[cols_to_scale] = scaler.transform(train_scaled[cols_to_scale])
    val_scaled[cols_to_scale] = scaler.transform(val_scaled[cols_to_scale])
    test_scaled[cols_to_scale] = scaler.transform(test_scaled[cols_to_scale])

    return train_scaled, val_scaled, test_scaled, scaler
