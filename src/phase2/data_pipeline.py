"""
End-to-End Data Pipeline for Phase 2 LSTM Forecasting.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Saksham Sontakke (Data Pipeline & Preprocessing Engineer)

Key Responsibilities & Guarantees:
1. Strict Phase 1 Compatibility: Reuses validated Phase 1 loaders, causal forward-fill
   cleaning, and feature transformations without logic duplication.
2. Station Isolation: Every station is processed independently; no cross-station mixing.
3. Target Invariance: Missing ground-truth PM2.5 target labels are NEVER imputed. Sequences
   whose target observation is missing are filtered out cleanly.
4. Chronological Integrity: Strict 70% Train / 15% Val / 15% Test partition ordering
   with synchronized global timestamp cutoffs across all 12 stations.
5. Zero Leakage Scaling: Scalers are fitted EXCLUSIVELY on training data.
6. Context-Preserving Boundary Slices: Provides warm-up context (e.g. up to 72 hours)
   for validation and test partitions so sequence models don't lose the initial hours.
7. Modular Sequence Generation: Converts station data directly into 3D tensors (N, W, D)
   and 1D targets (N,) for direct consumption by TensorFlow/Keras LSTM architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.data.load_data import (
    REQUIRED_COLUMNS,
    build_timestamp,
    load_all_stations,
    load_station_data,
    validate_data,
)
from src.features import (
    create_cyclical_features,
    create_domain_weather_features,
)
from src.preprocessing import (
    add_cyclical_wind_features,
    clean_features,
)
from src.split_data import get_split_boundaries
from src.phase2.config import Phase2Config
from src.phase2.scaling import MultiStationScalerManager, StationScaler

logger = logging.getLogger(__name__)


@dataclass
class StationDataSplits:
    """
    Complete data container for a single monitoring station.

    Holds unscaled and scaled chronological partitions, context slices,
    fitted station scaler, feature column registry, and sequence generators.
    """
    station_name: str
    feature_cols: List[str]
    target_col: str
    boundaries: Dict[str, Any]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    val_with_context_df: pd.DataFrame
    test_with_context_df: pd.DataFrame
    train_scaled: pd.DataFrame
    val_scaled: pd.DataFrame
    test_scaled: pd.DataFrame
    val_scaled_with_context: pd.DataFrame
    test_scaled_with_context: pd.DataFrame
    scaler: StationScaler
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_train_sequences(
        self,
        window_size: int = 24,
        horizon: int = 1,
        scaled: bool = True,
        scale_target: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Generate (X, y, timestamps) sequences for the training set.

        Parameters
        ----------
        window_size : int, default 24
            Historical lookback window in hours (e.g. 24, 48, 72).
        horizon : int, default 1
            Forecast horizon in hours ahead (e.g. 1, 3, 6, 12, 24).
        scaled : bool, default True
            Whether input features X should be scaled.
        scale_target : bool, default True
            Whether target values y should be scaled.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]
            - X: 3D array of shape (N, window_size, len(feature_cols))
            - y: 1D array of shape (N,)
            - target_timestamps: DatetimeIndex of length N
        """
        df_feat = self.train_scaled if scaled else self.train_df[self.feature_cols]
        raw_target = self.train_df[self.target_col]
        target_series = raw_target
        if scale_target and self.scaler.is_fitted:
            target_series = pd.Series(
                self.scaler.transform_target(raw_target.values),
                index=self.train_df.index,
            )

        return create_lstm_sequences(
            df_features=df_feat,
            target_series=target_series,
            timestamps=self.train_df["datetime"],
            feature_cols=self.feature_cols,
            window_size=window_size,
            horizon=horizon,
            require_valid_target=True,
        )

    def get_val_sequences(
        self,
        window_size: int = 24,
        horizon: int = 1,
        use_context: bool = True,
        scaled: bool = True,
        scale_target: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Generate (X, y, timestamps) sequences for the validation set.

        When `use_context=True`, historical lookback observations at the start
        of validation are drawn from the tail of the training set, ensuring no
        loss of validation predictions.
        """
        if use_context:
            df_source = self.val_with_context_df
            df_feat = self.val_scaled_with_context if scaled else df_source[self.feature_cols]
        else:
            df_source = self.val_df
            df_feat = self.val_scaled if scaled else df_source[self.feature_cols]

        raw_target = df_source[self.target_col]
        target_series = raw_target
        if scale_target and self.scaler.is_fitted:
            target_series = pd.Series(
                self.scaler.transform_target(raw_target.values),
                index=df_source.index,
            )

        X, y, ts = create_lstm_sequences(
            df_features=df_feat,
            target_series=target_series,
            timestamps=df_source["datetime"],
            feature_cols=self.feature_cols,
            window_size=window_size,
            horizon=horizon,
            require_valid_target=True,
        )

        # If context was used, filter predictions strictly to the validation period
        if use_context and len(ts) > 0:
            val_start = self.boundaries["val_start"]
            val_end = self.boundaries["val_end"]
            mask = (ts >= val_start) & (ts <= val_end)
            return X[mask], y[mask], ts[mask]

        return X, y, ts

    def get_test_sequences(
        self,
        window_size: int = 24,
        horizon: int = 1,
        use_context: bool = True,
        scaled: bool = True,
        scale_target: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Generate (X, y, timestamps) sequences for the test set.

        When `use_context=True`, historical lookback observations at the start
        of testing are drawn from the tail of the validation set.
        """
        if use_context:
            df_source = self.test_with_context_df
            df_feat = self.test_scaled_with_context if scaled else df_source[self.feature_cols]
        else:
            df_source = self.test_df
            df_feat = self.test_scaled if scaled else df_source[self.feature_cols]

        raw_target = df_source[self.target_col]
        target_series = raw_target
        if scale_target and self.scaler.is_fitted:
            target_series = pd.Series(
                self.scaler.transform_target(raw_target.values),
                index=df_source.index,
            )

        X, y, ts = create_lstm_sequences(
            df_features=df_feat,
            target_series=target_series,
            timestamps=df_source["datetime"],
            feature_cols=self.feature_cols,
            window_size=window_size,
            horizon=horizon,
            require_valid_target=True,
        )

        # If context was used, filter predictions strictly to the test period
        if use_context and len(ts) > 0:
            test_start = self.boundaries["test_start"]
            mask = ts >= test_start
            return X[mask], y[mask], ts[mask]

        return X, y, ts

    def inverse_transform_target(
        self,
        y_pred: Union[pd.Series, np.ndarray, float, List[float]],
    ) -> Union[np.ndarray, float]:
        """Invert scaled predictions back to PM2.5 concentration in µg/m³."""
        return self.scaler.inverse_transform_target(y_pred)


def prepare_station_features(
    df_clean: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    station_col: str = "station",
    timestamp_col: str = "datetime",
    target_col: str = "PM2.5",
    causal_ffill_input_pm25: bool = True,
    max_ffill_hours: int = 6,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare clean contemporaneous features for LSTM sequence modeling.

    Pipeline Steps:
    1. Ensure chronological order.
    2. Preserve the strictly un-imputed ground truth target column (`target_pm25`).
    3. Causally forward-fill input feature PM2.5 (<= max_ffill_hours) so that
       short sensor dropouts in past observations do not invalidate entire
       24-72h sequence windows. The prediction TARGET remains strictly un-imputed.
    4. Ensure cyclical wind components (wd_sin, wd_cos) are present.
    5. Generate meteorological interaction features (temp_dewp_spread, wind_stagnation, rain_clearing).
    6. Generate cyclical calendar features (sin_hour, cos_hour, sin_month, cos_month, sin_dow, cos_dow).

    Parameters
    ----------
    df_clean : pd.DataFrame
        DataFrame that has passed causal cleaning.
    feature_cols : Optional[Sequence[str]], default None
        Specific feature columns to select.
    station_col : str, default 'station'
        Station identifier column name.
    timestamp_col : str, default 'datetime'
        Continuous timestamp column name.
    target_col : str, default 'PM2.5'
        Target variable name.
    causal_ffill_input_pm25 : bool, default True
        Whether to causally forward-fill the input feature PM2.5 for sequence continuity.
    max_ffill_hours : int, default 6
        Maximum hours to forward-fill input feature PM2.5.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Enriched DataFrame and the ordered list of feature column names.
    """
    df_out = df_clean.copy()

    # Sort chronologically
    sort_cols = [station_col, timestamp_col] if station_col in df_out.columns else [timestamp_col]
    df_out = df_out.sort_values(by=sort_cols).reset_index(drop=True)

    # 1. Preserve strictly UN-IMPUTED target column
    if "target_pm25" not in df_out.columns and target_col in df_out.columns:
        df_out["target_pm25"] = df_out[target_col].copy()

    # Initial target missing count check
    initial_target_missing = int(df_out["target_pm25"].isnull().sum())

    # 2. Causally forward-fill contemporaneous input feature PM2.5
    if causal_ffill_input_pm25 and target_col in df_out.columns:
        df_out[target_col] = df_out[target_col].ffill(limit=max_ffill_hours)
        if df_out[target_col].isnull().any():
            med = df_out[target_col].median()
            df_out[target_col] = df_out[target_col].fillna(med)

    # 3. Cyclical wind direction encoding if not already present
    if "wd_sin" not in df_out.columns and "wd" in df_out.columns:
        df_out = add_cyclical_wind_features(df_out, wind_col="wd", drop_original=False)

    # 4. Domain meteorological interaction features
    df_out, _ = create_domain_weather_features(df_out)

    # 5. Cyclical temporal components
    df_out, _ = create_cyclical_features(df_out, timestamp_col=timestamp_col)

    # Determine feature column list
    candidate_features = [
        # Pollutants
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        # Meteorology
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
        # Wind vectors
        "wd_sin", "wd_cos",
        # Domain features
        "temp_dewp_spread", "wind_stagnation", "rain_clearing",
        # Temporal features
        "sin_hour", "cos_hour", "sin_month", "cos_month", "sin_dow", "cos_dow",
    ]

    if feature_cols is not None:
        selected_features = [c for c in feature_cols if c in df_out.columns]
    else:
        selected_features = [c for c in candidate_features if c in df_out.columns]

    # Target safety audit: target_pm25 must NEVER be imputed
    final_target_missing = int(df_out["target_pm25"].isnull().sum())
    assert final_target_missing == initial_target_missing, (
        f"Target invariance violated: target missing count changed from "
        f"{initial_target_missing} to {final_target_missing}."
    )

    return df_out, selected_features


def split_station_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = "datetime",
    max_context_window: int = 72,
    embargo_hours: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split time series into chronological partitions and context-preserving slices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing continuous timestamp column.
    train_ratio : float, default 0.70
        Training proportion.
    val_ratio : float, default 0.15
        Validation proportion.
    test_ratio : float, default 0.15
        Test proportion.
    timestamp_col : str, default 'datetime'
        Timestamp column name.
    max_context_window : int, default 72
        Maximum lookback hours to prepend to validation and test partitions.
    embargo_hours : int, default 0
        Embargo hours to purge at boundaries.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        - train_df: Pure training partition.
        - val_df: Pure validation partition.
        - test_df: Pure test partition.
        - val_with_context_df: Validation set with last `max_context_window` hours of train prepended.
        - test_with_context_df: Test set with last `max_context_window` hours of val prepended.
        - boundaries: Split cutoffs and timestamp metadata.
    """
    boundaries = get_split_boundaries(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        timestamp_col=timestamp_col,
    )

    ts_series = pd.to_datetime(df[timestamp_col])

    train_end = boundaries["train_end"]
    val_start = boundaries["val_start"]
    val_end = boundaries["val_end"]
    test_start = boundaries["test_start"]

    if embargo_hours > 0:
        embargo_delta = pd.Timedelta(hours=embargo_hours)
        train_end = train_end - embargo_delta
        val_end = val_end - embargo_delta

    train_mask = ts_series <= train_end
    val_mask = (ts_series >= val_start) & (ts_series <= val_end)
    test_mask = ts_series >= test_start

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    # Context slices: prepend prior window to prevent sequence clipping at partition start
    context_train_len = min(max_context_window, len(train_df))
    context_train_tail = train_df.tail(context_train_len)
    val_with_context_df = pd.concat([context_train_tail, val_df], ignore_index=True)

    context_val_len = min(max_context_window, len(val_df))
    context_val_tail = val_df.tail(context_val_len)
    test_with_context_df = pd.concat([context_val_tail, test_df], ignore_index=True)

    return train_df, val_df, test_df, val_with_context_df, test_with_context_df, boundaries


def create_lstm_sequences(
    df_features: pd.DataFrame,
    target_series: pd.Series,
    timestamps: Sequence[Any],
    feature_cols: Sequence[str],
    window_size: int = 24,
    horizon: int = 1,
    require_valid_target: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Construct sliding sequence windows for LSTM training and evaluation.

    Given a continuous hourly sequence:
    At index i (corresponding to timestamp t):
      Input window: Features from [i - window_size + 1] up to [i] (length window_size).
      Target: PM2.5 observation at [i + horizon].

    Parameters
    ----------
    df_features : pd.DataFrame
        Clean, ordered feature matrix of shape (L, D).
    target_series : pd.Series
        Target PM2.5 series aligned with df_features (unimputed ground truth).
    timestamps : Sequence[Any]
        Timestamps corresponding to rows.
    feature_cols : Sequence[str]
        Ordered column names for input features.
    window_size : int, default 24
        Historical lookback steps W.
    horizon : int, default 1
        Forecast horizon H in hours ahead.
    require_valid_target : bool, default True
        If True, drops sequences where the target at (t + horizon) is NaN.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]
        - X: 3D numpy array of shape (N, window_size, len(feature_cols))
        - y: 1D numpy array of shape (N,)
        - target_timestamps: DatetimeIndex of length N indicating the exact target observation time
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    X_mat = df_features[list(feature_cols)].values
    y_vec = np.asarray(target_series, dtype=float)
    ts_vec = pd.to_datetime(timestamps).values

    num_records = len(df_features)
    first_idx = window_size - 1
    last_idx = num_records - horizon

    if last_idx < first_idx:
        return (
            np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DatetimeIndex([]),
        )

    num_candidates = last_idx - first_idx
    X_candidates = np.empty((num_candidates, window_size, len(feature_cols)), dtype=np.float32)
    y_candidates = np.empty(num_candidates, dtype=np.float32)
    ts_candidates = np.empty(num_candidates, dtype="datetime64[ns]")
    valid_mask = np.ones(num_candidates, dtype=bool)

    for k, i in enumerate(range(first_idx, last_idx)):
        # Input window: [i - window_size + 1 : i + 1]
        X_candidates[k] = X_mat[i - window_size + 1 : i + 1]

        target_idx = i + horizon
        target_val = y_vec[target_idx]
        y_candidates[k] = target_val
        ts_candidates[k] = ts_vec[target_idx]

        # Check target validity: NEVER train or evaluate on an imputed target
        if require_valid_target and np.isnan(target_val):
            valid_mask[k] = False

        # Check feature window validity (no unexpected NaNs)
        if np.isnan(X_candidates[k]).any():
            valid_mask[k] = False

    X_out = X_candidates[valid_mask]
    y_out = y_candidates[valid_mask]
    ts_out = pd.DatetimeIndex(ts_candidates[valid_mask])

    return X_out, y_out, ts_out


def process_station(
    station_name: str,
    config: Optional[Phase2Config] = None,
    df_raw: Optional[pd.DataFrame] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> StationDataSplits:
    """
    Execute complete Phase 2 data preparation pipeline for a single station.

    Steps:
    1. Load raw data (or use df_raw if supplied).
    2. Causal cleaning (clean_features) with forward-fill <= 6h.
    3. Feature generation (meteorological domain, cyclical calendar, wind vectors).
    4. Chronological 70% Train / 15% Val / 15% Test splitting with context slices.
    5. Fit StationScaler STRICTLY on training features and PM2.5 target.
    6. Transform train, val, and test partitions (plus context slices).
    7. Return populated StationDataSplits container.

    Parameters
    ----------
    station_name : str
        Monitoring station name (e.g. 'Aotizhongxin').
    config : Optional[Phase2Config], default None
        Configuration object. If None, default configuration is loaded.
    df_raw : Optional[pd.DataFrame], default None
        In-memory raw DataFrame (bypasses file loading, useful for unit tests).
    data_dir : Optional[Union[str, Path]], default None
        Override directory path where station CSV files are stored.

    Returns
    -------
    StationDataSplits
        Container with unscaled, scaled, and sequence-ready datasets for the station.
    """
    cfg = config or Phase2Config.get_default_config()
    target_col = "target_pm25"
    raw_pm25_col = cfg.data.target_column  # "PM2.5"
    timestamp_col = cfg.data.timestamp_column
    station_col = cfg.data.station_column
    feature_list = cfg.get_feature_list()

    logger.info("Processing station: %s", station_name)

    # 1. Load data
    if df_raw is not None:
        df_station = df_raw.copy()
        if timestamp_col not in df_station.columns:
            df_station = build_timestamp(df_station, drop_source_cols=False)
        if station_col not in df_station.columns:
            df_station[station_col] = station_name
        df_station = df_station.sort_values(by=timestamp_col).reset_index(drop=True)
    else:
        raw_dir = Path(data_dir or cfg.data.raw_dir)
        matches = sorted(raw_dir.glob(f"PRSA_Data_{station_name}_*.csv"))
        if not matches:
            for f in raw_dir.glob("PRSA_Data_*.csv"):
                if station_name.lower() in f.name.lower():
                    matches = [f]
                    break
        if not matches:
            raise FileNotFoundError(
                f"No CSV file found for station '{station_name}' in {raw_dir}"
            )
        df_station = load_station_data(matches[0])

    # 2. Causal cleaning of features
    df_clean, clean_report = clean_features(
        df_station,
        target_col=raw_pm25_col,
        timestamp_col=timestamp_col,
        station_col=station_col,
        max_ffill_hours=cfg.cleaning.max_ffill_hours,
        encode_wind=cfg.cleaning.encode_wind,
        fill_remaining_with_median=cfg.cleaning.fill_remaining_with_median,
    )

    # 3. Feature engineering
    df_feat, active_features = prepare_station_features(
        df_clean,
        feature_cols=feature_list,
        station_col=station_col,
        timestamp_col=timestamp_col,
        target_col=raw_pm25_col,
        causal_ffill_input_pm25=True,
        max_ffill_hours=cfg.cleaning.max_ffill_hours,
    )

    # 4. Chronological splitting
    max_w = max(cfg.lstm_windows.input_sequence_lengths)
    (
        train_df,
        val_df,
        test_df,
        val_with_context_df,
        test_with_context_df,
        boundaries,
    ) = split_station_data(
        df=df_feat,
        train_ratio=cfg.split.train_ratio,
        val_ratio=cfg.split.val_ratio,
        test_ratio=cfg.split.test_ratio,
        timestamp_col=timestamp_col,
        max_context_window=max_w,
        embargo_hours=cfg.split.embargo_hours,
    )

    # 5. Scaling: Fit strictly on training data
    scaler = StationScaler(
        scaler_type=cfg.scaling.scaler_type,
        target_col=target_col,
        station_name=station_name,
    )
    # Fit target scaler on the un-imputed ground truth target column
    scaler.fit(X=train_df[active_features], y=train_df[target_col])

    # Optionally persist scaler
    if cfg.scaling.save_scalers:
        scalers_dir = Path(cfg.scaling.scalers_dir)
        scaler.save(scalers_dir / f"scaler_{station_name}.joblib")

    # 6. Transform partitions
    train_scaled = scaler.transform_features(train_df[active_features])
    val_scaled = scaler.transform_features(val_df[active_features])
    test_scaled = scaler.transform_features(test_df[active_features])
    val_scaled_with_context = scaler.transform_features(val_with_context_df[active_features])
    test_scaled_with_context = scaler.transform_features(test_with_context_df[active_features])

    metadata: Dict[str, Any] = {
        "station_name": station_name,
        "clean_report": clean_report,
        "num_features": len(active_features),
        "feature_cols": active_features,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "max_context_window": max_w,
        "scaler_summary": scaler.get_params_summary(),
    }

    return StationDataSplits(
        station_name=station_name,
        feature_cols=active_features,
        target_col=target_col,
        boundaries=boundaries,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        val_with_context_df=val_with_context_df,
        test_with_context_df=test_with_context_df,
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        test_scaled=test_scaled,
        val_scaled_with_context=val_scaled_with_context,
        test_scaled_with_context=test_scaled_with_context,
        scaler=scaler,
        metadata=metadata,
    )


def process_all_stations(
    config: Optional[Phase2Config] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, StationDataSplits]:
    """
    Process all 12 Beijing monitoring stations independently.

    Parameters
    ----------
    config : Optional[Phase2Config], default None
        Configuration object.
    data_dir : Optional[Union[str, Path]], default None
        Directory where CSV files are stored.

    Returns
    -------
    Dict[str, StationDataSplits]
        Dictionary mapping station name to its StationDataSplits container.
    """
    cfg = config or Phase2Config.get_default_config()
    stations = cfg.stations.all
    results: Dict[str, StationDataSplits] = {}

    for station in stations:
        logger.info("Starting processing for station %s...", station)
        splits = process_station(
            station_name=station,
            config=cfg,
            data_dir=data_dir,
        )
        results[station] = splits

    return results


__all__ = [
    "StationDataSplits",
    "prepare_station_features",
    "split_station_data",
    "create_lstm_sequences",
    "process_station",
    "process_all_stations",
]
