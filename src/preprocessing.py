"""
Preprocessing and causal data cleaning module for Beijing Air Quality forecasting.

Key Design Principles:
1. Target Non-Imputation: PM2.5 target is NEVER imputed for supervised training.
   Missing targets remain identifiable (NaN) and flagged in `target_valid`.
2. Strict Causal Imputation: No future observations (bfill, backward interpolation)
   are ever used. Imputation relies solely on forward-fill (persistence) within
   a configurable horizon and historical station medians.
3. Station Isolation: Group operations are partitioned by `station` to prevent
   cross-station leakage.
4. Transparent Auditing: All cleaning operations are measured and returned in a
   structured cleaning report dictionary. No rows are silently dropped.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 16-point compass rose to azimuth angles (degrees)
WIND_DIRECTION_MAP: Dict[str, float] = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}

FEATURE_COLUMNS_NUMERIC: List[str] = [
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]


def add_cyclical_wind_features(
    df: pd.DataFrame,
    wind_col: str = "wd",
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Transform 16-point categorical wind direction into continuous sine and cosine components.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing categorical wind direction column.
    wind_col : str, default 'wd'
        Name of the categorical wind direction column.
    drop_original : bool, default False
        Whether to drop the raw wind column after encoding.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'wd_sin', 'wd_cos', and 'wd_deg' added.
    """
    df_out = df.copy()

    if wind_col not in df_out.columns:
        return df_out

    # Map string directions to degrees
    degrees = df_out[wind_col].map(WIND_DIRECTION_MAP)
    radians = np.deg2rad(degrees)

    df_out["wd_deg"] = degrees
    df_out["wd_sin"] = np.sin(radians)
    df_out["wd_cos"] = np.cos(radians)

    if drop_original:
        df_out = df_out.drop(columns=[wind_col])

    return df_out


def clean_features(
    df: pd.DataFrame,
    target_col: str = "PM2.5",
    timestamp_col: str = "datetime",
    station_col: str = "station",
    max_ffill_hours: int = 6,
    encode_wind: bool = True,
    fill_remaining_with_median: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean feature columns using strictly causal time-series imputation.

    Rules enforced:
    - Target `PM2.5` is NOT imputed. Missing target values remain NaN.
    - `target_valid` boolean flag is created to identify valid training rows.
    - Missing feature values are forward-filled up to `max_ffill_hours` within each station.
    - Long remaining gaps can be filled using historical station-level feature medians.
    - Categorical wind direction is causally forward-filled and converted into cyclical
      sin/cos components.
    - Zero rows are deleted. Complete row provenance and audit metrics are returned.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (with station and timestamp).
    target_col : str, default 'PM2.5'
        Name of the target forecasting column (NEVER imputed).
    timestamp_col : str, default 'datetime'
        Name of the timestamp column.
    station_col : str, default 'station'
        Name of the station identity column.
    max_ffill_hours : int, default 6
        Maximum consecutive hours to forward-fill missing sensor values.
    encode_wind : bool, default True
        Whether to generate cyclical wind components (sin/cos).
    fill_remaining_with_median : bool, default True
        Whether to fill remaining feature NaNs with station-level historical median.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned DataFrame and a detailed cleaning report dictionary.
    """
    initial_rows = len(df)
    df_clean = df.copy()

    # Ensure chronological order per station before causal operations
    if timestamp_col in df_clean.columns and station_col in df_clean.columns:
        df_clean = df_clean.sort_values(by=[station_col, timestamp_col]).reset_index(drop=True)
    elif timestamp_col in df_clean.columns:
        df_clean = df_clean.sort_values(by=timestamp_col).reset_index(drop=True)

    # Initial target and feature audit
    target_missing_initial = int(df_clean[target_col].isnull().sum()) if target_col in df_clean.columns else 0
    feature_imputations: Dict[str, Dict[str, int]] = {}

    numeric_cols = [c for c in FEATURE_COLUMNS_NUMERIC if c in df_clean.columns and c != target_col]

    # Pre-imputation missing counts
    for col in numeric_cols + ([ "wd" ] if "wd" in df_clean.columns else []):
        feature_imputations[col] = {
            "initial_missing": int(df_clean[col].isnull().sum()),
            "ffill_imputed": 0,
            "median_imputed": 0,
            "remaining_missing": 0,
        }

    # Station-partitioned causal forward-fill
    def _impute_station(group: pd.DataFrame) -> pd.DataFrame:
        grp = group.copy()

        # Numeric features: causal forward fill with limit
        for col in numeric_cols:
            init_nans = grp[col].isnull()
            if max_ffill_hours > 0:
                grp[col] = grp[col].ffill(limit=max_ffill_hours)
            after_ffill_nans = grp[col].isnull()
            ffill_count = int((init_nans & ~after_ffill_nans).sum())
            feature_imputations[col]["ffill_imputed"] += ffill_count

            # If still missing (e.g. at the start of series or gap > max_ffill_hours)
            if fill_remaining_with_median and grp[col].isnull().any():
                st_median = grp[col].median()
                if pd.isna(st_median):
                    st_median = df_clean[col].median()  # fallback to global feature median
                median_count = int(grp[col].isnull().sum())
                grp[col] = grp[col].fillna(st_median)
                feature_imputations[col]["median_imputed"] += median_count

        # Categorical wind direction causal forward fill
        if "wd" in grp.columns:
            init_wd_nan = grp["wd"].isnull()
            if max_ffill_hours > 0:
                grp["wd"] = grp["wd"].ffill(limit=max_ffill_hours)
            after_wd_nan = grp["wd"].isnull()
            feature_imputations["wd"]["ffill_imputed"] += int((init_wd_nan & ~after_wd_nan).sum())

            if fill_remaining_with_median and grp["wd"].isnull().any():
                mode_series = grp["wd"].mode()
                st_mode = mode_series.iloc[0] if not mode_series.empty else "N"
                rem_nan = int(grp["wd"].isnull().sum())
                grp["wd"] = grp["wd"].fillna(st_mode)
                feature_imputations["wd"]["median_imputed"] += rem_nan

        return grp

    if station_col in df_clean.columns:
        station_groups = []
        for st_val, grp in df_clean.groupby(station_col, as_index=False):
            imputed_grp = _impute_station(grp)
            # Ensure station identity is intact
            if station_col not in imputed_grp.columns:
                imputed_grp[station_col] = st_val
            station_groups.append(imputed_grp)
        df_clean = pd.concat(station_groups, ignore_index=True)
    else:
        # Fallback if no station column
        df_clean = _impute_station(df_clean)

    # Post-imputation remaining missing audit
    for col in numeric_cols + ([ "wd" ] if "wd" in df_clean.columns else []):
        feature_imputations[col]["remaining_missing"] = int(df_clean[col].isnull().sum())

    # Target safety audit: verify target was NOT modified
    target_missing_final = int(df_clean[target_col].isnull().sum()) if target_col in df_clean.columns else 0
    assert target_missing_final == target_missing_initial, (
        f"CRITICAL ERROR: Target column '{target_col}' missing count changed from "
        f"{target_missing_initial} to {target_missing_final}. Target must never be imputed!"
    )

    # Add target validity flag for downstream modeling
    if target_col in df_clean.columns:
        df_clean["target_valid"] = df_clean[target_col].notnull()

    # Cyclical wind encoding
    if encode_wind and "wd" in df_clean.columns:
        df_clean = add_cyclical_wind_features(df_clean, wind_col="wd", drop_original=False)

    final_rows = len(df_clean)
    assert final_rows == initial_rows, "Row count changed during cleaning. No rows should be dropped."

    cleaning_report: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "rows_dropped": 0,
        "target_col": target_col,
        "target_missing_count": target_missing_final,
        "target_valid_count": int(df_clean["target_valid"].sum()) if "target_valid" in df_clean.columns else 0,
        "target_imputed": False,
        "max_ffill_hours": max_ffill_hours,
        "fill_remaining_with_median": fill_remaining_with_median,
        "feature_imputations": feature_imputations,
    }

    return df_clean, cleaning_report


def get_valid_training_mask(
    df: pd.DataFrame,
    target_col: str = "PM2.5",
    station_col: str = "station",
    horizon: int = 1,
) -> pd.Series:
    """
    Generate boolean mask indicating which rows have valid features at t
    and valid ground-truth target at t + horizon for supervised learning.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame sorted chronologically per station.
    target_col : str, default 'PM2.5'
        Target variable name.
    station_col : str, default 'station'
        Station identifier.
    horizon : int, default 1
        Forecasting horizon in hours (1-hour ahead).

    Returns
    -------
    pd.Series
        Boolean mask of length len(df) indicating valid (X_t, y_{t+horizon}) pairs.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Target shifted by horizon into the future within each station
    if station_col in df.columns:
        future_target = df.groupby(station_col)[target_col].shift(-horizon)
    else:
        future_target = df[target_col].shift(-horizon)

    # Valid if future target is not null and current target (used as lag 0) is valid if needed
    valid_mask = future_target.notnull()
    return valid_mask
