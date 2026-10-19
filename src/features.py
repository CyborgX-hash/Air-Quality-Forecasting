"""
Time-Series Feature Engineering Module for Air Quality Forecasting.

Key Design Principles:
1. Forecasting Definition: Predict PM2.5 at time t + 1 hour using strictly information
   available at time t and earlier (X_t -> y_{t+1}).
2. Strict Station Isolation: All lags, rolling statistics, and targets are computed
   strictly within individual station partitions. Cross-station leakage is impossible.
3. Timestamp-Aware Continuity: Targets and lags are calculated on a continuous hourly
   grid. Missing timestamps produce NaNs rather than erroneously shifting across gaps.
4. Future Target Exclusion: Target y_{t+1} is strictly excluded from all feature sets,
   lag features, and rolling calculations.
5. Explainable Physics & Domain Features: Clear names, physical meanings (e.g. dew point
   depression, wind stagnation), and clean separation of feature matrix X, target y, and metadata.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import logging
import numpy as np
import pandas as pd

from src.preprocessing import WIND_DIRECTION_MAP, add_cyclical_wind_features

logger = logging.getLogger(__name__)

# Standard candidate lag configurations
DEFAULT_PM25_LAGS: List[int] = [1, 2, 3, 6, 12, 24]
DEFAULT_POLLUTANT_LAGS: Dict[str, List[int]] = {
    "PM10": [1, 3, 6, 24],
    "NO2": [1, 3, 6, 24],
    "CO": [1, 3, 6, 24],
}
DEFAULT_ROLLING_WINDOWS: List[int] = [3, 6, 12, 24]
DEFAULT_ROLLING_STATS: List[str] = ["mean", "std"]


def validate_feature_inputs(
    df: pd.DataFrame,
    target_col: str = "PM2.5",
    timestamp_col: str = "datetime",
    station_col: str = "station",
) -> None:
    """
    Validate input DataFrame structure before feature engineering.

    Parameters
    ----------
    df : pd.DataFrame
        Input air quality DataFrame.
    target_col : str, default 'PM2.5'
        Column name of the target variable.
    timestamp_col : str, default 'datetime'
        Column name of the timestamp.
    station_col : str, default 'station'
        Column name of the station identifier.
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"Required timestamp column '{timestamp_col}' not found.")
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        raise TypeError(f"Timestamp column '{timestamp_col}' must be a datetime64 dtype.")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")


def create_target(
    df: pd.DataFrame,
    target_col: str = "PM2.5",
    horizon: int = 1,
    timestamp_col: str = "datetime",
    station_col: str = "station",
    new_target_col: str = "target_pm25",
) -> pd.DataFrame:
    """
    Construct strictly verified (t + horizon) target column per station.

    A row at timestamp t receives the target observation strictly at (t + horizon hours)
    within the same station. If the observation at (t + horizon hours) is missing or
    non-continuous, NaN is assigned to prevent blind row-shift leakage across gaps.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sorted chronologically per station.
    target_col : str, default 'PM2.5'
        Source target variable at time t.
    horizon : int, default 1
        Forecasting horizon in hours (1-hour ahead).
    timestamp_col : str, default 'datetime'
        Name of datetime column.
    station_col : str, default 'station'
        Name of station column.
    new_target_col : str, default 'target_pm25'
        Name of generated target column.

    Returns
    -------
    pd.DataFrame
        DataFrame with new_target_col added.
    """
    validate_feature_inputs(df, target_col=target_col, timestamp_col=timestamp_col)
    df_out = df.copy()

    def _compute_station_target(grp: pd.DataFrame) -> pd.DataFrame:
        grp_sorted = grp.sort_values(by=timestamp_col)
        min_dt = grp_sorted[timestamp_col].min()
        max_dt = grp_sorted[timestamp_col].max()

        # Build complete hourly index to handle gaps accurately
        full_idx = pd.date_range(min_dt, max_dt, freq="1h")
        indexed = grp_sorted.set_index(timestamp_col).reindex(full_idx)

        # Shift backward in time: row t gets value at t + horizon
        indexed[new_target_col] = indexed[target_col].shift(-horizon)

        # Re-align back to original group timestamps
        target_series = indexed.loc[grp_sorted[timestamp_col], new_target_col].values
        grp_sorted[new_target_col] = target_series
        return grp_sorted

    if station_col in df_out.columns:
        station_groups = [
            _compute_station_target(grp)
            for _, grp in df_out.groupby(station_col, as_index=False)
        ]
        df_out = pd.concat(station_groups, ignore_index=True)
    else:
        df_out = _compute_station_target(df_out)

    return df_out


def create_lag_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int],
    timestamp_col: str = "datetime",
    station_col: str = "station",
    prefix: str = "",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate timestamp-aware lag features strictly within station partitions.

    For each specified column and lag k, row at timestamp t receives the value at
    (t - k hours). If the record at (t - k hours) is missing, NaN is assigned.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : Sequence[str]
        Columns to generate lags for.
    lags : Sequence[int]
        Lag steps in hours (e.g. [1, 2, 3, 6, 12, 24]).
    timestamp_col : str, default 'datetime'
        Timestamp column name.
    station_col : str, default 'station'
        Station identifier column name.
    prefix : str, default ''
        Optional column name prefix.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with lag columns added, and list of newly created feature names.
    """
    df_out = df.copy()
    valid_cols = [c for c in columns if c in df_out.columns]
    if not valid_cols or not lags:
        return df_out, []

    lag_feature_names: List[str] = []
    for col in valid_cols:
        col_clean = col.lower().replace(".", "")
        for k in lags:
            name = f"{prefix}{col_clean}_lag_{k}h"
            lag_feature_names.append(name)

    def _compute_station_lags(grp: pd.DataFrame) -> pd.DataFrame:
        grp_sorted = grp.sort_values(by=timestamp_col)
        min_dt = grp_sorted[timestamp_col].min()
        max_dt = grp_sorted[timestamp_col].max()

        full_idx = pd.date_range(min_dt, max_dt, freq="1h")
        indexed = grp_sorted.set_index(timestamp_col).reindex(full_idx)

        for col in valid_cols:
            col_clean = col.lower().replace(".", "")
            for k in lags:
                name = f"{prefix}{col_clean}_lag_{k}h"
                indexed[name] = indexed[col].shift(k)

        for name in lag_feature_names:
            grp_sorted[name] = indexed.loc[grp_sorted[timestamp_col], name].values
        return grp_sorted

    if station_col in df_out.columns:
        station_groups = [
            _compute_station_lags(grp)
            for _, grp in df_out.groupby(station_col, as_index=False)
        ]
        df_out = pd.concat(station_groups, ignore_index=True)
    else:
        df_out = _compute_station_lags(df_out)

    return df_out, lag_feature_names


def create_rolling_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    stats: Sequence[str] = ("mean", "std"),
    min_periods: Optional[int] = 1,
    timestamp_col: str = "datetime",
    station_col: str = "station",
    prefix: str = "",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate causal rolling statistics over historical windows up to time t.

    The rolling calculation over window W at time t includes historical records
    in [t - W + 1, t]. It NEVER includes the future target at (t + 1).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : Sequence[str]
        Columns to compute rolling statistics on.
    windows : Sequence[int], default [3, 6, 12, 24]
        Window sizes in hours.
    stats : Sequence[str], default ('mean', 'std')
        Aggregation statistics ('mean', 'std', 'min', 'max').
    min_periods : Optional[int], default 1
        Minimum number of valid observations required in window.
    timestamp_col : str, default 'datetime'
        Timestamp column name.
    station_col : str, default 'station'
        Station column name.
    prefix : str, default ''
        Optional column name prefix.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with rolling columns added, and list of newly created feature names.
    """
    df_out = df.copy()
    valid_cols = [c for c in columns if c in df_out.columns]
    if not valid_cols or not windows:
        return df_out, []

    rolling_feature_names: List[str] = []
    for col in valid_cols:
        col_clean = col.lower().replace(".", "")
        for w in windows:
            for stat in stats:
                name = f"{prefix}{col_clean}_roll_{stat}_{w}h"
                rolling_feature_names.append(name)

    def _compute_station_rolling(grp: pd.DataFrame) -> pd.DataFrame:
        grp_sorted = grp.sort_values(by=timestamp_col)
        min_dt = grp_sorted[timestamp_col].min()
        max_dt = grp_sorted[timestamp_col].max()

        full_idx = pd.date_range(min_dt, max_dt, freq="1h")
        indexed = grp_sorted.set_index(timestamp_col).reindex(full_idx)

        for col in valid_cols:
            col_clean = col.lower().replace(".", "")
            for w in windows:
                roller = indexed[col].rolling(window=w, min_periods=min_periods)
                for stat in stats:
                    name = f"{prefix}{col_clean}_roll_{stat}_{w}h"
                    if stat == "mean":
                        indexed[name] = roller.mean()
                    elif stat == "std":
                        indexed[name] = roller.std()
                    elif stat == "min":
                        indexed[name] = roller.min()
                    elif stat == "max":
                        indexed[name] = roller.max()
                    else:
                        raise ValueError(f"Unsupported rolling statistic: {stat}")

        for name in rolling_feature_names:
            grp_sorted[name] = indexed.loc[grp_sorted[timestamp_col], name].values
        return grp_sorted

    if station_col in df_out.columns:
        station_groups = [
            _compute_station_rolling(grp)
            for _, grp in df_out.groupby(station_col, as_index=False)
        ]
        df_out = pd.concat(station_groups, ignore_index=True)
    else:
        df_out = _compute_station_rolling(df_out)

    return df_out, rolling_feature_names


def create_calendar_features(
    df: pd.DataFrame,
    timestamp_col: str = "datetime",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract discrete calendar components from datetime column.

    Features created:
    - hour (0-23): Diurnal cycle
    - day_of_week (0-6, Mon=0, Sun=6)
    - day (1-31): Day of month
    - month (1-12): Seasonal cycle
    - is_weekend (0 or 1): Anthropogenic activity pattern flag

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with timestamp column.
    timestamp_col : str, default 'datetime'
        Name of datetime column.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with calendar features, and list of added feature names.
    """
    df_out = df.copy()
    ts = pd.to_datetime(df_out[timestamp_col])

    df_out["hour"] = ts.dt.hour
    df_out["day_of_week"] = ts.dt.dayofweek
    df_out["day"] = ts.dt.day
    df_out["month"] = ts.dt.month
    df_out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    feature_names = ["hour", "day_of_week", "day", "month", "is_weekend"]
    return df_out, feature_names


def create_cyclical_features(
    df: pd.DataFrame,
    timestamp_col: str = "datetime",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Encode periodic calendar variables into continuous sine and cosine components.

    Resolves artificial boundary discontinuities (e.g. hour 23 -> hour 0).

    Features created:
    - sin_hour, cos_hour (period = 24)
    - sin_month, cos_month (period = 12)
    - sin_dow, cos_dow (period = 7)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    timestamp_col : str, default 'datetime'
        Name of datetime column.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with cyclical features, and list of added feature names.
    """
    df_out = df.copy()
    ts = pd.to_datetime(df_out[timestamp_col])

    # Diurnal cycle (24 hours)
    df_out["sin_hour"] = np.sin(2.0 * np.pi * ts.dt.hour / 24.0)
    df_out["cos_hour"] = np.cos(2.0 * np.pi * ts.dt.hour / 24.0)

    # Seasonal cycle (12 months, offset by 1 for zero-indexed phase)
    df_out["sin_month"] = np.sin(2.0 * np.pi * (ts.dt.month - 1) / 12.0)
    df_out["cos_month"] = np.cos(2.0 * np.pi * (ts.dt.month - 1) / 12.0)

    # Day-of-week cycle (7 days)
    df_out["sin_dow"] = np.sin(2.0 * np.pi * ts.dt.dayofweek / 7.0)
    df_out["cos_dow"] = np.cos(2.0 * np.pi * ts.dt.dayofweek / 7.0)

    feature_names = ["sin_hour", "cos_hour", "sin_month", "cos_month", "sin_dow", "cos_dow"]
    return df_out, feature_names


def create_domain_weather_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct physically grounded meteorological interaction features.

    Features created:
    - temp_dewp_spread: (TEMP - DEWP). Dew point depression. When close to 0,
      relative humidity approaches 100%, accelerating hygroscopic aerosol growth
      and stabilizing nocturnal smog formation.
    - wind_stagnation: Binary indicator (WSPM < 1.5 m/s). Low ventilation traps
      particulate emissions in the surface boundary layer.
    - rain_clearing: Binary indicator (RAIN > 0.0 mm/h). Wet deposition wash-out.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing weather variables.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with domain features, and list of added feature names.
    """
    df_out = df.copy()
    added_cols: List[str] = []

    # Dew point depression
    if "TEMP" in df_out.columns and "DEWP" in df_out.columns:
        df_out["temp_dewp_spread"] = df_out["TEMP"] - df_out["DEWP"]
        added_cols.append("temp_dewp_spread")

    # Wind stagnation threshold
    if "WSPM" in df_out.columns:
        df_out["wind_stagnation"] = (df_out["WSPM"] < 1.5).astype(int)
        added_cols.append("wind_stagnation")

    # Precipitation indicator
    if "RAIN" in df_out.columns:
        df_out["rain_clearing"] = (df_out["RAIN"] > 0.0).astype(int)
        added_cols.append("rain_clearing")

    return df_out, added_cols


def generate_features(
    df: pd.DataFrame,
    target_col: str = "PM2.5",
    horizon: int = 1,
    pm25_lags: Sequence[int] = DEFAULT_PM25_LAGS,
    pollutant_lags: Optional[Dict[str, List[int]]] = None,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    rolling_stats: Sequence[str] = ("mean", "std"),
    rolling_cols: Optional[Sequence[str]] = ("PM2.5", "PM10", "TEMP"),
    include_calendar: bool = True,
    include_cyclical: bool = True,
    include_domain_weather: bool = True,
    include_wind_encoding: bool = True,
    timestamp_col: str = "datetime",
    station_col: str = "station",
    new_target_col: str = "target_pm25",
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Comprehensive pipeline orchestrator to generate a leakage-safe supervised forecasting dataset.

    Pipeline Steps:
    1. Chronological sorting per station.
    2. Construction of verified (t + horizon) target per station.
    3. Autoregressive PM2.5 lag features (e.g. 1h, 2h, 3h, 6h, 12h, 24h).
    4. Multi-pollutant lag features (PM10, NO2, CO).
    5. Causal rolling statistics (mean, std over 3h, 6h, 12h, 24h) excluding future target.
    6. Calendar and cyclical time representations.
    7. Continuous wind vector decomposition (wd_sin, wd_cos).
    8. Domain physical interactions (dew point depression, stagnation flag).
    9. Target validity audit and feature metadata dictionary generation.

    Parameters
    ----------
    df : pd.DataFrame
        Input cleaned or raw DataFrame.
    target_col : str, default 'PM2.5'
        Column name of the target variable at time t.
    horizon : int, default 1
        Forecasting horizon in hours.
    pm25_lags : Sequence[int], default [1, 2, 3, 6, 12, 24]
        Autoregressive lag steps for PM2.5.
    pollutant_lags : Optional[Dict[str, List[int]]], default None
        Dictionary mapping pollutant names to lag step lists.
    rolling_windows : Sequence[int], default [3, 6, 12, 24]
        Window sizes for historical rolling statistics.
    rolling_stats : Sequence[str], default ('mean', 'std')
        Statistics to calculate over rolling windows.
    rolling_cols : Optional[Sequence[str]], default ('PM2.5', 'PM10', 'TEMP')
        Columns to calculate rolling statistics for.
    include_calendar : bool, default True
        Whether to generate discrete calendar features.
    include_cyclical : bool, default True
        Whether to generate cyclical sine/cosine features.
    include_domain_weather : bool, default True
        Whether to generate physical weather interactions.
    include_wind_encoding : bool, default True
        Whether to encode categorical wind direction as cyclical vectors.
    timestamp_col : str, default 'datetime'
        Name of timestamp column.
    station_col : str, default 'station'
        Name of station identifier column.
    new_target_col : str, default 'target_pm25'
        Name of the created future target column.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], Dict[str, Any]]
        - Transformed DataFrame with all engineered features and target.
        - List of candidate feature column names (excluding target and metadata).
        - Metadata dictionary detailing feature counts, types, and schema.
    """
    validate_feature_inputs(df, target_col=target_col, timestamp_col=timestamp_col)

    # Ensure chronological order per station before any transformation
    df_feat = df.copy()
    if station_col in df_feat.columns:
        df_feat = df_feat.sort_values(by=[station_col, timestamp_col]).reset_index(drop=True)
    else:
        df_feat = df_feat.sort_values(by=timestamp_col).reset_index(drop=True)

    feature_cols: List[str] = []

    # 1. Base contemporaneous features at time t
    base_numeric = [
        col for col in [
            "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
            "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
        ] if col in df_feat.columns
    ]
    feature_cols.extend(base_numeric)

    # 2. Wind direction encoding
    if include_wind_encoding and "wd" in df_feat.columns:
        df_feat = add_cyclical_wind_features(df_feat, wind_col="wd", drop_original=False)
        for wind_f in ["wd_sin", "wd_cos", "wd_deg"]:
            if wind_f in df_feat.columns and wind_f not in feature_cols:
                feature_cols.append(wind_f)

    # 3. Create supervised forecast target at t + horizon
    df_feat = create_target(
        df_feat,
        target_col=target_col,
        horizon=horizon,
        timestamp_col=timestamp_col,
        station_col=station_col,
        new_target_col=new_target_col,
    )

    # 4. PM2.5 autoregressive lags
    if pm25_lags:
        df_feat, lags_pm25 = create_lag_features(
            df_feat,
            columns=[target_col],
            lags=pm25_lags,
            timestamp_col=timestamp_col,
            station_col=station_col,
        )
        feature_cols.extend(lags_pm25)

    # 5. Multi-pollutant lags
    active_pollutant_lags = pollutant_lags or DEFAULT_POLLUTANT_LAGS
    for pol_col, p_lags in active_pollutant_lags.items():
        if pol_col in df_feat.columns and p_lags:
            df_feat, lags_pol = create_lag_features(
                df_feat,
                columns=[pol_col],
                lags=p_lags,
                timestamp_col=timestamp_col,
                station_col=station_col,
            )
            feature_cols.extend(lags_pol)

    # 6. Causal historical rolling features
    if rolling_cols and rolling_windows:
        active_roll_cols = [c for c in rolling_cols if c in df_feat.columns]
        if active_roll_cols:
            df_feat, rolls = create_rolling_features(
                df_feat,
                columns=active_roll_cols,
                windows=rolling_windows,
                stats=rolling_stats,
                timestamp_col=timestamp_col,
                station_col=station_col,
            )
            feature_cols.extend(rolls)

    # 7. Calendar features
    if include_calendar:
        df_feat, cal_cols = create_calendar_features(df_feat, timestamp_col=timestamp_col)
        feature_cols.extend(cal_cols)

    # 8. Cyclical features
    if include_cyclical:
        df_feat, cyc_cols = create_cyclical_features(df_feat, timestamp_col=timestamp_col)
        feature_cols.extend(cyc_cols)

    # 9. Domain weather interactions
    if include_domain_weather:
        df_feat, dom_cols = create_domain_weather_features(df_feat)
        feature_cols.extend(dom_cols)

    # Ensure feature list is unique and strictly excludes target and non-feature columns
    excluded_from_features = {
        new_target_col,
        "No",
        "station",
        "datetime",
        "year",
        "wd",
        "target_valid",
    }
    feature_cols = [col for col in dict.fromkeys(feature_cols) if col not in excluded_from_features]

    # Add target validity flag
    df_feat["target_valid"] = df_feat[new_target_col].notnull()

    metadata: Dict[str, Any] = {
        "target_col": new_target_col,
        "horizon_hours": horizon,
        "num_features": len(feature_cols),
        "total_records": len(df_feat),
        "valid_target_records": int(df_feat["target_valid"].sum()),
        "missing_target_records": int((~df_feat["target_valid"]).sum()),
        "feature_columns": feature_cols,
    }

    return df_feat, feature_cols, metadata


def extract_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "target_pm25",
    station_col: str = "station",
    timestamp_col: str = "datetime",
    require_valid_target: bool = True,
    drop_na_features: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extract pure feature matrix X, target series y, and provenance metadata.

    Guarantees that target column and non-feature columns never enter X.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with engineered features and target.
    feature_cols : Sequence[str]
        List of feature column names.
    target_col : str, default 'target_pm25'
        Target column name.
    station_col : str, default 'station'
        Station identifier column name.
    timestamp_col : str, default 'datetime'
        Datetime timestamp column name.
    require_valid_target : bool, default True
        Whether to filter out rows where target is NaN.
    drop_na_features : bool, default False
        Whether to drop rows with any NaNs in the feature matrix.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        - X: Feature matrix DataFrame of shape (N, D).
        - y: Target Series of shape (N,).
        - metadata: DataFrame containing [station, datetime] of shape (N, 2).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Strict anti-leakage audit: ensure target is not in feature_cols
    cleaned_features = [c for c in feature_cols if c != target_col]
    if len(cleaned_features) != len(feature_cols):
        logger.warning(f"Target column '{target_col}' was detected in feature_cols and removed.")

    sub_df = df.copy()
    if require_valid_target:
        sub_df = sub_df[sub_df[target_col].notnull()]

    if drop_na_features:
        sub_df = sub_df.dropna(subset=cleaned_features)

    X = sub_df[cleaned_features].copy()
    y = sub_df[target_col].copy()

    meta_cols = [c for c in [station_col, timestamp_col] if c in sub_df.columns]
    metadata = sub_df[meta_cols].copy()

    return X, y, metadata


def get_feature_dictionary() -> pd.DataFrame:
    """
    Return a structured feature dictionary documenting every candidate feature.

    Returns
    -------
    pd.DataFrame
        DataFrame containing [feature_group, feature_name, formula, physical_meaning, role].
    """
    entries = [
        # Target
        ("Target", "target_pm25", "PM2.5_{t+1}", "Ground truth PM2.5 concentration 1 hour ahead", "Supervised Target Label"),
        # Base Pollutants & Weather
        ("Base Measurements", "PM2.5", "PM2.5_t", "Contemporaneous PM2.5 at time t (1h before target)", "Primary Autoregressive Feature"),
        ("Base Measurements", "PM10", "PM10_t", "Contemporaneous coarse particulate matter", "Predictive Feature (r=0.88 with PM2.5)"),
        ("Base Measurements", "SO2", "SO2_t", "Sulfur dioxide; coal combustion indicator", "Emissions Feature (r=0.48 with PM2.5)"),
        ("Base Measurements", "NO2", "NO2_t", "Nitrogen dioxide; traffic and combustion tracer", "Emissions Feature (r=0.66 with PM2.5)"),
        ("Base Measurements", "CO", "CO_t", "Carbon monoxide; incomplete combustion tracer", "Emissions Feature (r=0.78 with PM2.5)"),
        ("Base Measurements", "O3", "O3_t", "Ground-level ozone; photochemical oxidant", "Secondary Chemistry Feature"),
        ("Base Meteorology", "TEMP", "TEMP_t", "Surface dry-bulb temperature (°C)", "Atmospheric boundary layer proxy"),
        ("Base Meteorology", "PRES", "PRES_t", "Surface barometric pressure (hPa)", "Synoptic weather front tracker"),
        ("Base Meteorology", "DEWP", "DEWP_t", "Dew point temperature (°C)", "Absolute atmospheric moisture proxy"),
        ("Base Meteorology", "RAIN", "RAIN_t", "Hourly accumulated precipitation (mm/h)", "Aerosol wet deposition wash-out indicator"),
        ("Base Meteorology", "WSPM", "WSPM_t", "Surface horizontal wind speed (m/s)", "Key dispersion vs stagnation threshold"),
        # Autoregressive Lags
        ("Autoregressive Lags", "pm25_lag_1h", "PM2.5_{t-1}", "PM2.5 concentration 1 hour prior to observation (2h before target)", "Short-term persistence"),
        ("Autoregressive Lags", "pm25_lag_2h", "PM2.5_{t-2}", "PM2.5 concentration 2 hours prior to observation", "Short-term trajectory"),
        ("Autoregressive Lags", "pm25_lag_3h", "PM2.5_{t-3}", "PM2.5 concentration 3 hours prior to observation", "Multi-hour trend"),
        ("Autoregressive Lags", "pm25_lag_6h", "PM2.5_{t-6}", "PM2.5 concentration 6 hours prior to observation", "Synoptic shift tracking"),
        ("Autoregressive Lags", "pm25_lag_12h", "PM2.5_{t-12}", "PM2.5 concentration 12 hours prior to observation", "Semi-diurnal cycle baseline"),
        ("Autoregressive Lags", "pm25_lag_24h", "PM2.5_{t-24}", "PM2.5 concentration same hour previous day", "Diurnal seasonality anchor"),
        # Multi-pollutant Lags
        ("Multi-Pollutant Lags", "pm10_lag_1h", "PM10_{t-1}", "PM10 concentration 1 hour prior to observation", "Coarse particle momentum"),
        ("Multi-Pollutant Lags", "pm10_lag_3h", "PM10_{t-3}", "PM10 concentration 3 hours prior to observation", "Coarse particle trend"),
        ("Multi-Pollutant Lags", "pm10_lag_6h", "PM10_{t-6}", "PM10 concentration 6 hours prior to observation", "Dust/smog event propagation"),
        ("Multi-Pollutant Lags", "pm10_lag_24h", "PM10_{t-24}", "PM10 concentration 24 hours prior to observation", "Daily coarse baseline"),
        ("Multi-Pollutant Lags", "no2_lag_1h", "NO2_{t-1}", "NO2 concentration 1 hour prior to observation", "Traffic rush hour lag"),
        ("Multi-Pollutant Lags", "no2_lag_3h", "NO2_{t-3}", "NO2 concentration 3 hours prior to observation", "Traffic dispersion momentum"),
        ("Multi-Pollutant Lags", "no2_lag_6h", "NO2_{t-6}", "NO2 concentration 6 hours prior to observation", "Nocturnal boundary layer buildup"),
        ("Multi-Pollutant Lags", "no2_lag_24h", "NO2_{t-24}", "NO2 concentration 24 hours prior to observation", "Daily commute cycle baseline"),
        ("Multi-Pollutant Lags", "co_lag_1h", "CO_{t-1}", "CO concentration 1 hour prior to observation", "Combustion plume tracking"),
        ("Multi-Pollutant Lags", "co_lag_3h", "CO_{t-3}", "CO concentration 3 hours prior to observation", "Combustion persistence"),
        ("Multi-Pollutant Lags", "co_lag_6h", "CO_{t-6}", "CO concentration 6 hours prior to observation", "Industrial plume trajectory"),
        ("Multi-Pollutant Lags", "co_lag_24h", "CO_{t-24}", "CO concentration 24 hours prior to observation", "Daily industrial baseline"),
        # Rolling Statistics
        ("Rolling Statistics", "pm25_roll_mean_3h", "mean(PM2.5_{t-2:t})", "3-hour moving average of historical PM2.5", "Short-term smoothed level"),
        ("Rolling Statistics", "pm25_roll_std_3h", "std(PM2.5_{t-2:t})", "3-hour moving standard deviation of PM2.5", "Short-term volatility / spike onset"),
        ("Rolling Statistics", "pm25_roll_mean_6h", "mean(PM2.5_{t-5:t})", "6-hour moving average of historical PM2.5", "Sub-diurnal moving level"),
        ("Rolling Statistics", "pm25_roll_std_6h", "std(PM2.5_{t-5:t})", "6-hour moving standard deviation of PM2.5", "Sub-diurnal dispersion turbulence"),
        ("Rolling Statistics", "pm25_roll_mean_12h", "mean(PM2.5_{t-11:t})", "12-hour moving average of historical PM2.5", "Half-day trend level"),
        ("Rolling Statistics", "pm25_roll_std_12h", "std(PM2.5_{t-11:t})", "12-hour moving standard deviation of PM2.5", "Half-day volatility"),
        ("Rolling Statistics", "pm25_roll_mean_24h", "mean(PM2.5_{t-23:t})", "24-hour moving average of historical PM2.5", "Daily baseline pollution level"),
        ("Rolling Statistics", "pm25_roll_std_24h", "std(PM2.5_{t-23:t})", "24-hour moving standard deviation of PM2.5", "Full-day episode variability"),
        ("Rolling Statistics", "pm10_roll_mean_6h", "mean(PM10_{t-5:t})", "6-hour moving average of historical PM10", "Coarse particle rolling level"),
        ("Rolling Statistics", "pm10_roll_std_6h", "std(PM10_{t-5:t})", "6-hour moving standard deviation of PM10", "Coarse particle rolling volatility"),
        ("Rolling Statistics", "pm10_roll_mean_24h", "mean(PM10_{t-23:t})", "24-hour moving average of historical PM10", "Daily coarse particle baseline"),
        ("Rolling Statistics", "temp_roll_mean_24h", "mean(TEMP_{t-23:t})", "24-hour moving average temperature", "Daily thermal inertia / cold wave tracer"),
        # Calendar & Cyclical
        ("Temporal Features", "hour", "datetime.hour", "Hour of day (0-23)", "Diurnal rush-hour & planetary boundary layer height"),
        ("Temporal Features", "day_of_week", "datetime.dayofweek", "Day of week (0=Mon, 6=Sun)", "Weekly commercial/industrial traffic pattern"),
        ("Temporal Features", "day", "datetime.day", "Day of month (1-31)", "Calendar indexing"),
        ("Temporal Features", "month", "datetime.month", "Month of year (1-12)", "Strong winter heating season vs summer contrast"),
        ("Temporal Features", "is_weekend", "day_of_week >= 5", "Binary indicator (1=Weekend, 0=Weekday)", "Weekend effect on urban emissions"),
        ("Cyclical Encodings", "sin_hour", "sin(2π · hour / 24)", "Cyclical sine component of hour", "Continuous circular diurnal encoding"),
        ("Cyclical Encodings", "cos_hour", "cos(2π · hour / 24)", "Cyclical cosine component of hour", "Continuous circular diurnal encoding"),
        ("Cyclical Encodings", "sin_month", "sin(2π · (month - 1) / 12)", "Cyclical sine component of month", "Continuous circular seasonal encoding"),
        ("Cyclical Encodings", "cos_month", "cos(2π · (month - 1) / 12)", "Cyclical cosine component of month", "Continuous circular seasonal encoding"),
        ("Cyclical Encodings", "sin_dow", "sin(2π · day_of_week / 7)", "Cyclical sine component of day of week", "Continuous circular weekly encoding"),
        ("Cyclical Encodings", "cos_dow", "cos(2π · day_of_week / 7)", "Cyclical cosine component of day of week", "Continuous circular weekly encoding"),
        # Wind & Meteorological Domain
        ("Wind Direction", "wd_deg", "Azimuth angle θ ∈ [0, 337.5°]", "Compass direction in degrees", "Meteorological wind angle"),
        ("Wind Direction", "wd_sin", "sin(θ_rad)", "East-West wind vector component (+East, -West)", "Transport from industrial south/east"),
        ("Wind Direction", "wd_cos", "cos(θ_rad)", "North-South wind vector component (+North, -South)", "Clean Siberian wind from north (+1)"),
        ("Domain Weather", "temp_dewp_spread", "TEMP - DEWP", "Dew point depression (°C)", "High humidity / fog formation (low spread -> haze growth)"),
        ("Domain Weather", "wind_stagnation", "WSPM < 1.5 m/s", "Binary stagnation flag", "Critical ventilation threshold preventing dispersion"),
        ("Domain Weather", "rain_clearing", "RAIN > 0.0 mm/h", "Binary wet deposition flag", "Atmospheric scavenging wash-out"),
    ]
    columns = ["feature_group", "feature_name", "formula", "physical_meaning", "role"]
    return pd.DataFrame(entries, columns=columns)
