"""
Phase 2 Time-Series Sequence Generation Module for LSTM Forecasting.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Ajit Kumar Prasad (Time-Series Sequence Engineer)

Responsibilities & Architectural Guarantees:
1. Multi-Horizon Sliding Windows:
   Constructs 3D tensors of shape (N, W, D) for historical input sequences of
   lengths W in {24, 48, 72} hours, aligned with multi-horizon ground truth
   target arrays of shape (N, 5) for future PM2.5 at [t+1, t+3, t+6, t+12, t+24].
2. Target Alignment:
   If the last input observation is at time t, targets strictly represent PM2.5
   at t+1h, t+3h, t+6h, t+12h, and t+24h in that exact immutable order.
   The current hour t is never predicted as a future horizon.
3. Station Separation:
   Sequences are generated in strict isolation per monitoring station.
   Never combines or windows data across different stations.
4. Target Invariance & Strict Missing Value Handling:
   Targets are never imputed. A candidate window is retained if and only if
   ALL five horizon target values are valid and finite. Missing target exclusions
   are rigorously audited and reported without silent sample loss.
5. Chronological Boundary Integrity:
   Strictly adheres to 70% Train / 15% Val / 15% Test chronological splits.
   Context slices (up to 72h) are leveraged for validation and test inputs so
   no boundary predictions are lost, while guaranteeing:
   - No target from the future enters an input window.
   - No training target crosses into validation (max train target <= train_end).
   - No validation target crosses into test (max val target <= val_end).
6. TensorFlow/Keras Compatibility:
   Provides memory-efficient NumPy arrays, batch iterators, and direct
   tf.data.Dataset converters for seamless ingestion by LSTM model architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.phase2.config import Phase2Config
from src.phase2.scaling import StationScaler

logger = logging.getLogger(__name__)

# Standard Default Conventions
DEFAULT_SEQUENCE_LENGTHS: Tuple[int, ...] = (24, 48, 72)
DEFAULT_HORIZONS: Tuple[int, ...] = (1, 3, 6, 12, 24)
DEFAULT_TARGET_COL: str = "PM2.5"
UNIMPUTED_TARGET_COL: str = "target_pm25"
DEFAULT_TIMESTAMP_COL: str = "datetime"
DEFAULT_STATION_COL: str = "station"


@dataclass
class SequenceExclusionReport:
    """
    Detailed audit accounting of candidate sequence exclusions.

    Guarantees full transparency on why candidate sliding windows were
    retained or discarded across stations and chronological partitions.
    """
    station_name: str
    split_name: str
    window_size: int
    horizons: Tuple[int, ...]
    total_candidates_evaluated: int
    valid_sequences_retained: int
    excluded_missing_target: int
    excluded_nan_features: int
    excluded_timestamp_gaps: int
    excluded_boundary_violations: int

    @property
    def total_excluded(self) -> int:
        """Total candidate sequences discarded."""
        return self.total_candidates_evaluated - self.valid_sequences_retained

    @property
    def exclusion_rate_pct(self) -> float:
        """Percentage of candidate windows excluded."""
        if self.total_candidates_evaluated == 0:
            return 0.0
        return 100.0 * self.total_excluded / self.total_candidates_evaluated

    @property
    def retention_rate_pct(self) -> float:
        """Percentage of candidate windows retained."""
        if self.total_candidates_evaluated == 0:
            return 0.0
        return 100.0 * self.valid_sequences_retained / self.total_candidates_evaluated

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "station_name": self.station_name,
            "split_name": self.split_name,
            "window_size": self.window_size,
            "horizons": list(self.horizons),
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "valid_sequences_retained": self.valid_sequences_retained,
            "total_excluded": self.total_excluded,
            "excluded_missing_target": self.excluded_missing_target,
            "excluded_nan_features": self.excluded_nan_features,
            "excluded_timestamp_gaps": self.excluded_timestamp_gaps,
            "excluded_boundary_violations": self.excluded_boundary_violations,
            "exclusion_rate_pct": round(self.exclusion_rate_pct, 4),
            "retention_rate_pct": round(self.retention_rate_pct, 4),
        }

    def summary(self) -> str:
        """Human-readable formatted summary string."""
        return (
            f"[{self.station_name} | {self.split_name} | W={self.window_size}h | H={list(self.horizons)}]\n"
            f"  Total candidates: {self.total_candidates_evaluated:,}\n"
            f"  Valid retained  : {self.valid_sequences_retained:,} ({self.retention_rate_pct:.2f}%)\n"
            f"  Total excluded  : {self.total_excluded:,} ({self.exclusion_rate_pct:.2f}%)\n"
            f"    - Missing target(s)   : {self.excluded_missing_target:,}\n"
            f"    - Non-finite features : {self.excluded_nan_features:,}\n"
            f"    - Timestamp gaps      : {self.excluded_timestamp_gaps:,}\n"
            f"    - Boundary violations : {self.excluded_boundary_violations:,}"
        )


@dataclass
class SequenceDataset:
    """
    Structured container for time-series LSTM sequence datasets.

    Holds the 3D feature tensor X, 2D multi-horizon target matrix y,
    input window endpoint timestamps, multi-horizon future timestamps,
    and auditing metadata.
    """
    X: np.ndarray
    y: np.ndarray
    input_end_timestamps: pd.DatetimeIndex
    target_timestamps: np.ndarray
    station_name: str
    split_name: str
    feature_names: List[str]
    horizons: Tuple[int, ...]
    window_size: int
    is_scaled: bool
    target_scaled: bool
    scaler: Optional[StationScaler] = None
    exclusion_report: Optional[SequenceExclusionReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure standard types and array shapes."""
        if not isinstance(self.input_end_timestamps, pd.DatetimeIndex):
            self.input_end_timestamps = pd.DatetimeIndex(self.input_end_timestamps)

    def __len__(self) -> int:
        """Number of valid sequence samples."""
        return len(self.X)

    @property
    def num_samples(self) -> int:
        """Number of valid sequence samples."""
        return len(self.X)

    @property
    def num_features(self) -> int:
        """Number of contemporaneous input features per timestep."""
        return len(self.feature_names)

    @property
    def num_horizons(self) -> int:
        """Number of forecast horizons."""
        return len(self.horizons)

    def shape_summary(self) -> Dict[str, Any]:
        """Summary of tensor dimensions."""
        return {
            "X_shape": self.X.shape,
            "y_shape": self.y.shape,
            "num_samples": len(self.X),
            "window_size": self.window_size,
            "num_features": self.num_features,
            "num_horizons": self.num_horizons,
            "station": self.station_name,
            "split": self.split_name,
            "is_scaled": self.is_scaled,
            "target_scaled": self.target_scaled,
        }

    def inverse_transform_targets(
        self,
        y_pred: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
    ) -> np.ndarray:
        """
        Invert scaled targets or model predictions back to physical concentration (µg/m³).

        Parameters
        ----------
        y_pred : Optional[Union[np.ndarray, pd.Series, List[float]]], default None
            Predictions to invert. If None, inverts self.y.

        Returns
        -------
        np.ndarray
            Inverted targets/predictions of shape (N, H).
        """
        target_to_invert = self.y if y_pred is None else np.asarray(y_pred, dtype=float)

        if not self.target_scaled or self.scaler is None:
            return np.asarray(target_to_invert, dtype=np.float32)

        inverted = self.scaler.inverse_transform_target(target_to_invert)
        return np.asarray(inverted, dtype=np.float32)

    def iter_batches(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_remainder: bool = False,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Memory-efficient pure NumPy/Python batch generator.

        Compatible with training loops without requiring TensorFlow.
        """
        n = len(self.X)
        if n == 0:
            return

        indices = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n and drop_remainder:
                break
            batch_idx = indices[start:end]
            yield self.X[batch_idx], self.y[batch_idx]

    def to_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        buffer_size: int = 1000,
        seed: int = 42,
        drop_remainder: bool = False,
    ) -> Any:
        """
        Convert sequence arrays into a high-performance TensorFlow tf.data.Dataset.

        Parameters
        ----------
        batch_size : int, default 32
            Batch size for model training or inference.
        shuffle : bool, default False
            Whether to shuffle training sequences. Note: shuffling sequence order
            during SGD training is standard practice; chronological splitting has
            already guaranteed strict temporal isolation between train, val, and test.
        buffer_size : int, default 1000
            Shuffle buffer size.
        seed : int, default 42
            Random seed for shuffling.
        drop_remainder : bool, default False
            Whether to discard the final partial batch.

        Returns
        -------
        tf.data.Dataset
            Batched, prefetched TensorFlow dataset yielding (X_batch, y_batch).
        """
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "TensorFlow is required to create a tf.data.Dataset. "
                "Install it via `pip install tensorflow` or consume NumPy arrays `X` and `y` directly."
            ) from e

        ds = tf.data.Dataset.from_tensor_slices((self.X, self.y))
        if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def validate(self) -> None:
        """
        Execute strict data integrity and alignment verification checks.

        Raises
        ------
        ValueError, AssertionError
            If any structural, dimension, numerical, or alignment check fails.
        """
        n = len(self.X)

        # 1. Dimension checks
        if self.X.ndim != 3:
            raise ValueError(f"X must be 3-dimensional (N, W, D), got ndim={self.X.ndim}, shape={self.X.shape}")
        if self.y.ndim != 2:
            raise ValueError(f"y must be 2-dimensional (N, H), got ndim={self.y.ndim}, shape={self.y.shape}")

        if self.X.shape[0] != n or self.y.shape[0] != n:
            raise ValueError(f"Sample count mismatch: X has {len(self.X)}, y has {len(self.y)}")
        if len(self.input_end_timestamps) != n:
            raise ValueError(
                f"Timestamp length mismatch: expected {n}, got {len(self.input_end_timestamps)}"
            )
        if len(self.target_timestamps) != n:
            raise ValueError(
                f"Target timestamp length mismatch: expected {n}, got {len(self.target_timestamps)}"
            )

        if self.X.shape[1] != self.window_size:
            raise ValueError(f"X window size mismatch: expected {self.window_size}, got {self.X.shape[1]}")
        if self.X.shape[2] != len(self.feature_names):
            raise ValueError(
                f"X feature dimension mismatch: expected {len(self.feature_names)}, got {self.X.shape[2]}"
            )
        if self.y.shape[1] != len(self.horizons):
            raise ValueError(
                f"y horizon dimension mismatch: expected {len(self.horizons)}, got {self.y.shape[1]}"
            )

        # If empty, nothing further to validate
        if n == 0:
            return

        # 2. NaN and Inf checks: strictly NO invalid values permitted
        if np.isnan(self.X).any():
            raise AssertionError("Invariant violated: X array contains NaN values.")
        if np.isnan(self.y).any():
            raise AssertionError("Invariant violated: y target array contains NaN values.")
        if np.isinf(self.X).any():
            raise AssertionError("Invariant violated: X array contains Inf values.")
        if np.isinf(self.y).any():
            raise AssertionError("Invariant violated: y target array contains Inf values.")

        # 3. Target alignment verification
        # For each sample k and horizon h_j:
        # target_timestamp[k, j] must equal input_end_timestamp[k] + h_j hours
        for j, h in enumerate(self.horizons):
            expected_target_ts = self.input_end_timestamps + pd.Timedelta(hours=h)
            actual_target_ts = pd.DatetimeIndex(self.target_timestamps[:, j])
            mismatches = (expected_target_ts != actual_target_ts).sum()
            if mismatches > 0:
                raise AssertionError(
                    f"Target alignment violated for horizon {h}h: {mismatches} timestamps do not "
                    f"equal input_end_timestamp + {h} hours."
                )

        # 4. Input timestamp monotonicity check
        if not self.input_end_timestamps.is_monotonic_increasing:
            raise AssertionError("Input end timestamps are not strictly monotonically increasing.")

    def to_dict(self) -> Dict[str, Any]:
        """Export dataset metadata and summary statistics."""
        return {
            "station_name": self.station_name,
            "split_name": self.split_name,
            "window_size": self.window_size,
            "horizons": list(self.horizons),
            "num_samples": len(self.X),
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "is_scaled": self.is_scaled,
            "target_scaled": self.target_scaled,
            "first_input_end": str(self.input_end_timestamps[0]) if len(self) > 0 else None,
            "last_input_end": str(self.input_end_timestamps[-1]) if len(self) > 0 else None,
            "first_target": str(pd.to_datetime(self.target_timestamps[0, 0])) if len(self) > 0 else None,
            "last_target": str(pd.to_datetime(self.target_timestamps[-1, -1])) if len(self) > 0 else None,
            "exclusion_report": self.exclusion_report.to_dict() if self.exclusion_report else None,
        }


@dataclass
class StationSequenceBundle:
    """
    Standard integration contract container for a single monitoring station.

    Holds the complete Train, Validation, and Test SequenceDataset objects,
    scaler reference, and validation utilities. Directly provides X_train,
    y_train, X_val, y_val, X_test, y_test properties for model training.
    """
    station_name: str
    window_size: int
    horizons: Tuple[int, ...]
    feature_names: List[str]
    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    scaler: Optional[StationScaler] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Clean integration properties for Harsh (LSTM Model Development)
    @property
    def X_train(self) -> np.ndarray:
        """Training feature sequences of shape (N_train, window_size, num_features)."""
        return self.train.X

    @property
    def y_train(self) -> np.ndarray:
        """Training multi-horizon targets of shape (N_train, 5)."""
        return self.train.y

    @property
    def X_val(self) -> np.ndarray:
        """Validation feature sequences of shape (N_val, window_size, num_features)."""
        return self.val.X

    @property
    def y_val(self) -> np.ndarray:
        """Validation multi-horizon targets of shape (N_val, 5)."""
        return self.val.y

    @property
    def X_test(self) -> np.ndarray:
        """Test feature sequences of shape (N_test, window_size, num_features)."""
        return self.test.X

    @property
    def y_test(self) -> np.ndarray:
        """Test multi-horizon targets of shape (N_test, 5)."""
        return self.test.y

    def validate(self) -> bool:
        """
        Validate all datasets and cross-split chronological boundary separation.

        Guarantees:
        1. All three datasets pass internal dimension and target alignment audits.
        2. No training target crosses into validation:
           max(train target timestamps) < min(val target timestamps).
        3. No validation target crosses into test:
           max(val target timestamps) < min(test target timestamps).
        """
        self.train.validate()
        self.val.validate()
        self.test.validate()

        # Cross-split temporal leakage audit
        if len(self.train) > 0 and len(self.val) > 0:
            max_train_target = pd.to_datetime(self.train.target_timestamps.max())
            min_val_target = pd.to_datetime(self.val.target_timestamps.min())
            if max_train_target >= min_val_target:
                raise AssertionError(
                    f"Temporal leakage detected: max training target ({max_train_target}) "
                    f"is not strictly before min validation target ({min_val_target})."
                )

        if len(self.val) > 0 and len(self.test) > 0:
            max_val_target = pd.to_datetime(self.val.target_timestamps.max())
            min_test_target = pd.to_datetime(self.test.target_timestamps.min())
            if max_val_target >= min_test_target:
                raise AssertionError(
                    f"Temporal leakage detected: max validation target ({max_val_target}) "
                    f"is not strictly before min test target ({min_test_target})."
                )

        return True

    def get_summary_table(self) -> str:
        """Generate human-readable Markdown summary table."""
        headers = ["Split", "Samples (N)", "Input Shape (X)", "Target Shape (y)", "Excluded Missing", "Exclusion %"]
        rows = []
        for name, ds in [("Train", self.train), ("Val", self.val), ("Test", self.test)]:
            rep = ds.exclusion_report
            ex_missing = rep.excluded_missing_target if rep else 0
            ex_pct = f"{rep.exclusion_rate_pct:.2f}%" if rep else "0.00%"
            rows.append([
                name,
                f"{len(ds):,}",
                str(ds.X.shape),
                str(ds.y.shape),
                f"{ex_missing:,}",
                ex_pct,
            ])

        # Simple markdown table formatting
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        row_lines = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header_line, sep_line] + row_lines)


def create_multistep_sequences(
    df_features: pd.DataFrame,
    target_series: Union[pd.Series, np.ndarray],
    timestamps: Union[pd.Series, pd.DatetimeIndex, Sequence[Any]],
    feature_cols: Sequence[str],
    window_size: int = 24,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    station_name: str = "Unknown",
    split_name: str = "custom",
    target_start_bound: Optional[pd.Timestamp] = None,
    target_end_bound: Optional[pd.Timestamp] = None,
    max_timestamp_gap_hours: int = 1,
    require_valid_target: bool = True,
    require_valid_features: bool = True,
    scaler: Optional[StationScaler] = None,
    is_scaled: bool = False,
    target_scaled: bool = False,
) -> SequenceDataset:
    """
    Construct multi-horizon sliding sequence windows for LSTM forecasting.

    Mathematical Indexing Convention:
    For an input window ending at index i (observation timestamp t = T[i]):
      Input window: Timesteps [i - window_size + 1, ..., i] (length = window_size).
      Targets: Ground truth PM2.5 at future indices [i + h_1, i + h_2, ..., i + h_m],
               corresponding exactly to timestamps [t + h_1, t + h_2, ..., t + h_m].
      Current observation at timestamp t is strictly NEVER predicted as a target.

    Parameters
    ----------
    df_features : pd.DataFrame
        Contemporaneous feature DataFrame.
    target_series : Union[pd.Series, np.ndarray]
        Un-imputed ground truth PM2.5 target series (or pre-scaled target series).
    timestamps : Union[pd.Series, pd.DatetimeIndex, Sequence[Any]]
        Continuous chronological timestamps corresponding to df_features rows.
    feature_cols : Sequence[str]
        Ordered sequence of input feature column names.
    window_size : int, default 24
        Historical lookback window W in hours (e.g. 24, 48, 72).
    horizons : Sequence[int], default (1, 3, 6, 12, 24)
        Forecast horizons in hours ahead in strict ascending order.
    station_name : str, default 'Unknown'
        Station identifier.
    split_name : str, default 'custom'
        Partition label ('train', 'val', 'test', 'custom').
    target_start_bound : Optional[pd.Timestamp], default None
        If provided, discards candidates whose earliest target is before this timestamp.
    target_end_bound : Optional[pd.Timestamp], default None
        If provided, discards candidates whose latest target exceeds this timestamp.
    max_timestamp_gap_hours : int, default 1
        Maximum allowable difference in hours between adjacent observations.
    require_valid_target : bool, default True
        If True, discards sequences where ANY of the multi-horizon target values is non-finite.
    require_valid_features : bool, default True
        If True, discards sequences where ANY input feature value in the window is non-finite.
    scaler : Optional[StationScaler], default None
        Fitted station scaler reference.
    is_scaled : bool, default False
        Whether df_features are scaled.
    target_scaled : bool, default False
        Whether target_series is scaled.

    Returns
    -------
    SequenceDataset
        Container with 3D input tensor X, 2D target matrix y, timestamps, and audit report.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    horizons_tuple = tuple(sorted(set(horizons)))
    if len(horizons_tuple) == 0:
        raise ValueError("horizons must contain at least one positive integer")
    if any(h < 1 for h in horizons_tuple):
        raise ValueError(f"All horizons must be >= 1, got {horizons_tuple}")

    # Station isolation enforcement: check for cross-station pollution
    if "station" in df_features.columns:
        unique_stations = df_features["station"].unique()
        if len(unique_stations) > 1:
            raise ValueError(
                f"Cross-station sequence generation detected! Expected single station, "
                f"found {len(unique_stations)} stations: {list(unique_stations)}. "
                f"Sequences must be generated in strict isolation per station."
            )
        if station_name == "Unknown" and len(unique_stations) == 1:
            station_name = str(unique_stations[0])

    # Convert to contiguous arrays
    feature_list = list(feature_cols)
    X_mat = df_features[feature_list].to_numpy(dtype=np.float32)
    y_vec = np.asarray(target_series, dtype=np.float32)

    if not isinstance(timestamps, pd.DatetimeIndex):
        ts_series = pd.to_datetime(timestamps)
    else:
        ts_series = timestamps

    ts_values = ts_series.to_numpy(dtype="datetime64[ns]")
    num_records = len(df_features)

    if len(y_vec) != num_records or len(ts_values) != num_records:
        raise ValueError(
            f"Length mismatch: df_features ({num_records}), target_series ({len(y_vec)}), "
            f"timestamps ({len(ts_values)}) must have identical lengths."
        )

    max_horizon = horizons_tuple[-1]
    first_idx = window_size - 1
    last_idx = num_records - max_horizon

    # Early exit if dataset is too small to construct a single window
    if last_idx < first_idx:
        empty_report = SequenceExclusionReport(
            station_name=station_name,
            split_name=split_name,
            window_size=window_size,
            horizons=horizons_tuple,
            total_candidates_evaluated=0,
            valid_sequences_retained=0,
            excluded_missing_target=0,
            excluded_nan_features=0,
            excluded_timestamp_gaps=0,
            excluded_boundary_violations=0,
        )
        return SequenceDataset(
            X=np.empty((0, window_size, len(feature_list)), dtype=np.float32),
            y=np.empty((0, len(horizons_tuple)), dtype=np.float32),
            input_end_timestamps=pd.DatetimeIndex([]),
            target_timestamps=np.empty((0, len(horizons_tuple)), dtype="datetime64[ns]"),
            station_name=station_name,
            split_name=split_name,
            feature_names=feature_list,
            horizons=horizons_tuple,
            window_size=window_size,
            is_scaled=is_scaled,
            target_scaled=target_scaled,
            scaler=scaler,
            exclusion_report=empty_report,
        )

    # Convert bounds to numpy datetime64
    np_start_bound = np.datetime64(target_start_bound) if target_start_bound is not None else None
    np_end_bound = np.datetime64(target_end_bound) if target_end_bound is not None else None

    # Step duration in nanoseconds for gap checks
    expected_step_ns = np.timedelta64(max_timestamp_gap_hours, "h").astype("timedelta64[ns]")
    window_span_ns = np.timedelta64(window_size - 1, "h").astype("timedelta64[ns]")

    candidates_evaluated = 0
    excluded_missing_target = 0
    excluded_nan_features = 0
    excluded_timestamp_gaps = 0
    excluded_boundary_violations = 0

    valid_indices: List[int] = []

    # Horizon index offsets
    h_offsets = np.array(horizons_tuple, dtype=int)
    min_h = horizons_tuple[0]

    # Evaluate each candidate window ending at index i
    for i in range(first_idx, last_idx):
        candidates_evaluated += 1

        # 1. Boundary filter: ensure multi-horizon targets respect partition boundaries
        earliest_target_ts = ts_values[i + min_h]
        latest_target_ts = ts_values[i + max_horizon]

        if np_start_bound is not None and earliest_target_ts < np_start_bound:
            excluded_boundary_violations += 1
            continue
        if np_end_bound is not None and latest_target_ts > np_end_bound:
            excluded_boundary_violations += 1
            continue

        # 2. Timestamp gap check: ensure continuity within lookback window
        # Check window endpoint minus startpoint matches window_size - 1 hours
        window_start_ts = ts_values[i - window_size + 1]
        input_end_ts = ts_values[i]
        if (input_end_ts - window_start_ts) != window_span_ns:
            excluded_timestamp_gaps += 1
            continue

        # Check target horizon alignments: each target timestamp must be exactly i + h
        has_ts_gap = False
        for h in horizons_tuple:
            expected_h_ts = input_end_ts + np.timedelta64(h, "h").astype("timedelta64[ns]")
            if ts_values[i + h] != expected_h_ts:
                has_ts_gap = True
                break
        if has_ts_gap:
            excluded_timestamp_gaps += 1
            continue

        # 3. Input feature check: verify window contains no NaNs or Infs
        if require_valid_features:
            input_window = X_mat[i - window_size + 1 : i + 1]
            if not np.isfinite(input_window).all():
                excluded_nan_features += 1
                continue

        # 4. Target validity check: retain ONLY if ALL target horizons are finite
        if require_valid_target:
            targets = y_vec[i + h_offsets]
            if not np.isfinite(targets).all():
                excluded_missing_target += 1
                continue

        valid_indices.append(i)

    num_valid = len(valid_indices)

    if num_valid > 0:
        idx_arr = np.array(valid_indices, dtype=int)

        # Construct 3D feature array (N, window_size, num_features)
        # Using preallocated array for optimal memory layout and performance
        X_out = np.empty((num_valid, window_size, len(feature_list)), dtype=np.float32)
        for k, idx in enumerate(idx_arr):
            X_out[k] = X_mat[idx - window_size + 1 : idx + 1]

        # Construct 2D target matrix (N, num_horizons)
        y_out = np.column_stack([y_vec[idx_arr + h] for h in horizons_tuple]).astype(np.float32)

        # Timestamps
        input_end_ts_out = pd.DatetimeIndex(ts_values[idx_arr])
        target_ts_out = np.column_stack([ts_values[idx_arr + h] for h in horizons_tuple])
    else:
        X_out = np.empty((0, window_size, len(feature_list)), dtype=np.float32)
        y_out = np.empty((0, len(horizons_tuple)), dtype=np.float32)
        input_end_ts_out = pd.DatetimeIndex([])
        target_ts_out = np.empty((0, len(horizons_tuple)), dtype="datetime64[ns]")

    report = SequenceExclusionReport(
        station_name=station_name,
        split_name=split_name,
        window_size=window_size,
        horizons=horizons_tuple,
        total_candidates_evaluated=candidates_evaluated,
        valid_sequences_retained=num_valid,
        excluded_missing_target=excluded_missing_target,
        excluded_nan_features=excluded_nan_features,
        excluded_timestamp_gaps=excluded_timestamp_gaps,
        excluded_boundary_violations=excluded_boundary_violations,
    )

    logger.debug(
        "Station %s [%s] W=%dh: %d/%d valid sequences retained (missing targets: %d)",
        station_name,
        split_name,
        window_size,
        num_valid,
        candidates_evaluated,
        excluded_missing_target,
    )

    dataset = SequenceDataset(
        X=X_out,
        y=y_out,
        input_end_timestamps=input_end_ts_out,
        target_timestamps=target_ts_out,
        station_name=station_name,
        split_name=split_name,
        feature_names=feature_list,
        horizons=horizons_tuple,
        window_size=window_size,
        is_scaled=is_scaled,
        target_scaled=target_scaled,
        scaler=scaler,
        exclusion_report=report,
    )

    # Built-in validation check
    dataset.validate()
    return dataset


def generate_station_sequence_bundle(
    splits: Any,  # StationDataSplits from src.phase2.data_pipeline
    window_size: int = 24,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    scaled: bool = True,
    scale_target: bool = True,
    require_valid_target: bool = True,
    require_valid_features: bool = True,
) -> StationSequenceBundle:
    """
    Generate complete Train, Val, and Test sequence bundle for a single station.

    Leverages StationDataSplits container produced by Saksham's data pipeline:
    - Training sequences drawn from train partition.
    - Validation sequences drawn from val_with_context (historical context preserved).
    - Test sequences drawn from test_with_context (historical context preserved).
    - Strict boundary bounds applied to prevent any cross-partition target leakage.

    Parameters
    ----------
    splits : StationDataSplits
        Data container from process_station().
    window_size : int, default 24
        Input sequence lookback in hours.
    horizons : Sequence[int], default (1, 3, 6, 12, 24)
        Forecast horizons in hours ahead.
    scaled : bool, default True
        Whether input features should be scaled.
    scale_target : bool, default True
        Whether target PM2.5 should be scaled using the training-fitted scaler.
    require_valid_target : bool, default True
        Strictly discard samples with any non-finite target value.
    require_valid_features : bool, default True
        Strictly discard samples with any non-finite feature value.

    Returns
    -------
    StationSequenceBundle
        Bundle with train, val, and test SequenceDataset objects and scaler reference.
    """
    horizons_tuple = tuple(sorted(set(horizons)))
    station_name = splits.station_name
    feature_cols = splits.feature_cols
    scaler = splits.scaler
    boundaries = splits.boundaries

    # 1. Prepare Training Data
    train_feat = splits.train_scaled if scaled else splits.train_df[feature_cols]
    raw_train_target = splits.train_df[splits.target_col]
    if scale_target and scaler.is_fitted:
        train_target = pd.Series(
            scaler.transform_target(raw_train_target.values),
            index=splits.train_df.index,
        )
    else:
        train_target = raw_train_target

    train_ds = create_multistep_sequences(
        df_features=train_feat,
        target_series=train_target,
        timestamps=splits.train_df["datetime"],
        feature_cols=feature_cols,
        window_size=window_size,
        horizons=horizons_tuple,
        station_name=station_name,
        split_name="train",
        target_start_bound=None,
        target_end_bound=boundaries["train_end"],  # No training target crosses into val
        require_valid_target=require_valid_target,
        require_valid_features=require_valid_features,
        scaler=scaler,
        is_scaled=scaled,
        target_scaled=scale_target,
    )

    # 2. Prepare Validation Data (using context slice to preserve initial predictions)
    val_feat = splits.val_scaled_with_context if scaled else splits.val_with_context_df[feature_cols]
    raw_val_target = splits.val_with_context_df[splits.target_col]
    if scale_target and scaler.is_fitted:
        val_target = pd.Series(
            scaler.transform_target(raw_val_target.values),
            index=splits.val_with_context_df.index,
        )
    else:
        val_target = raw_val_target

    val_ds = create_multistep_sequences(
        df_features=val_feat,
        target_series=val_target,
        timestamps=splits.val_with_context_df["datetime"],
        feature_cols=feature_cols,
        window_size=window_size,
        horizons=horizons_tuple,
        station_name=station_name,
        split_name="val",
        target_start_bound=boundaries["val_start"],  # Target strictly >= val_start
        target_end_bound=boundaries["val_end"],      # Target strictly <= val_end (no crossing into test)
        require_valid_target=require_valid_target,
        require_valid_features=require_valid_features,
        scaler=scaler,
        is_scaled=scaled,
        target_scaled=scale_target,
    )

    # 3. Prepare Test Data (using context slice)
    test_feat = splits.test_scaled_with_context if scaled else splits.test_with_context_df[feature_cols]
    raw_test_target = splits.test_with_context_df[splits.target_col]
    if scale_target and scaler.is_fitted:
        test_target = pd.Series(
            scaler.transform_target(raw_test_target.values),
            index=splits.test_with_context_df.index,
        )
    else:
        test_target = raw_test_target

    test_ds = create_multistep_sequences(
        df_features=test_feat,
        target_series=test_target,
        timestamps=splits.test_with_context_df["datetime"],
        feature_cols=feature_cols,
        window_size=window_size,
        horizons=horizons_tuple,
        station_name=station_name,
        split_name="test",
        target_start_bound=boundaries["test_start"],  # Target strictly >= test_start
        target_end_bound=boundaries["test_end"],      # Target strictly <= test_end
        require_valid_target=require_valid_target,
        require_valid_features=require_valid_features,
        scaler=scaler,
        is_scaled=scaled,
        target_scaled=scale_target,
    )

    bundle = StationSequenceBundle(
        station_name=station_name,
        window_size=window_size,
        horizons=horizons_tuple,
        feature_names=feature_cols,
        train=train_ds,
        val=val_ds,
        test=test_ds,
        scaler=scaler,
        metadata={
            "boundaries": boundaries,
            "scaled": scaled,
            "target_scaled": scale_target,
        },
    )

    # Validate bundle
    bundle.validate()
    return bundle


def generate_all_stations_sequences(
    all_splits: Optional[Dict[str, Any]] = None,
    config: Optional[Phase2Config] = None,
    window_sizes: Sequence[int] = DEFAULT_SEQUENCE_LENGTHS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    scaled: bool = True,
    scale_target: bool = True,
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[int, StationSequenceBundle]]:
    """
    Generate multi-horizon sequence bundles for all 12 monitoring stations.

    Parameters
    ----------
    all_splits : Optional[Dict[str, StationDataSplits]], default None
        Precomputed station splits. If None, process_all_stations() is called.
    config : Optional[Phase2Config], default None
        Configuration object.
    window_sizes : Sequence[int], default (24, 48, 72)
        Input sequence lookback lengths to generate.
    horizons : Sequence[int], default (1, 3, 6, 12, 24)
        Forecast horizons in hours ahead.
    scaled : bool, default True
        Whether features should be scaled.
    scale_target : bool, default True
        Whether target should be scaled.
    data_dir : Optional[Union[str, Path]], default None
        Directory where raw data files are stored.

    Returns
    -------
    Dict[str, Dict[int, StationSequenceBundle]]
        Nested dictionary mapping station_name -> window_size -> StationSequenceBundle.
    """
    from src.phase2.data_pipeline import process_all_stations, process_station

    cfg = config or Phase2Config.get_default_config()
    station_names = cfg.stations.all

    if all_splits is None:
        logger.info("Processing splits for all %d stations...", len(station_names))
        splits_dict = process_all_stations(config=cfg, data_dir=data_dir)
    else:
        splits_dict = all_splits

    results: Dict[str, Dict[int, StationSequenceBundle]] = {}

    for station in station_names:
        logger.info("Generating multi-horizon sequences for station %s...", station)
        splits = splits_dict[station]
        results[station] = {}
        for w in window_sizes:
            bundle = generate_station_sequence_bundle(
                splits=splits,
                window_size=w,
                horizons=horizons,
                scaled=scaled,
                scale_target=scale_target,
            )
            results[station][w] = bundle

    return results


def validate_sequence_dataset(dataset: SequenceDataset) -> bool:
    """Standalone validator function for a SequenceDataset."""
    dataset.validate()
    return True


def validate_sequence_bundle(bundle: StationSequenceBundle) -> bool:
    """Standalone validator function for a StationSequenceBundle."""
    return bundle.validate()


__all__ = [
    "DEFAULT_SEQUENCE_LENGTHS",
    "DEFAULT_HORIZONS",
    "DEFAULT_TARGET_COL",
    "UNIMPUTED_TARGET_COL",
    "DEFAULT_TIMESTAMP_COL",
    "DEFAULT_STATION_COL",
    "SequenceExclusionReport",
    "SequenceDataset",
    "StationSequenceBundle",
    "create_multistep_sequences",
    "generate_station_sequence_bundle",
    "generate_all_stations_sequences",
    "validate_sequence_dataset",
    "validate_sequence_bundle",
]

