"""
Unit Test Suite for Phase 2 Time-Series Sequence Generator.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Ajit Kumar Prasad (Time-Series Sequence Engineer)

Tests Requirements:
- Multi-horizon sequence shapes: W in {24, 48, 72}, targets in (N, 5).
- Precise target alignment using synthetic monotonic values (y_t = t).
- Strict station isolation (cross-station rejection).
- Missing target exclusions (samples discarded if ANY of 5 horizons is NaN).
- Non-finite feature window exclusions.
- Timestamp continuity & internal gap detection.
- Chronological split boundary isolation (train targets <= train_end, val targets <= val_end).
- TensorFlow dataset conversion & pure-NumPy batch iterators.
- End-to-end integration with real station datasets.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase2.config import Phase2Config
from src.phase2.data_pipeline import process_station
from src.phase2.sequence_generator import (
    DEFAULT_HORIZONS,
    DEFAULT_SEQUENCE_LENGTHS,
    SequenceDataset,
    SequenceExclusionReport,
    StationSequenceBundle,
    create_multistep_sequences,
    generate_station_sequence_bundle,
    validate_sequence_bundle,
    validate_sequence_dataset,
)


def make_synthetic_hourly_df(
    num_hours: int = 300,
    station: str = "SyntheticStation",
    start_time: str = "2020-01-01 00:00:00",
) -> pd.DataFrame:
    """
    Construct synthetic hourly DataFrame with predictable monotonic values.

    For hour offset t:
      datetime = start_time + t hours
      PM2.5 = t
      feat_1 = t * 10
      feat_2 = t * 0.5
    """
    dates = pd.date_range(start=start_time, periods=num_hours, freq="1h")
    t_vals = np.arange(num_hours, dtype=float)
    return pd.DataFrame({
        "datetime": dates,
        "station": station,
        "PM2.5": t_vals,
        "feat_1": t_vals * 10.0,
        "feat_2": t_vals * 0.5,
    })


# ==============================================================================
# 1. Synthetic Increasing Value Target Alignment Test
# ==============================================================================

@pytest.mark.parametrize("window_size", [24, 48, 72])
def test_synthetic_increasing_target_alignment(window_size: int):
    """
    Prove mathematically that a 1-hour target is actually 1h after the input endpoint,
    3h target is 3h after, etc., using monotonic synthetic hourly targets y_t = t.
    """
    num_hours = 250
    df = make_synthetic_hourly_df(num_hours=num_hours)
    feature_cols = ["feat_1", "feat_2"]
    horizons = (1, 3, 6, 12, 24)

    dataset = create_multistep_sequences(
        df_features=df,
        target_series=df["PM2.5"],
        timestamps=df["datetime"],
        feature_cols=feature_cols,
        window_size=window_size,
        horizons=horizons,
        station_name="SyntheticStation",
    )

    assert len(dataset) > 0
    # Expected number of candidates: (num_hours - max(horizons)) - (window_size - 1)
    expected_samples = (num_hours - 24) - (window_size - 1)
    assert len(dataset) == expected_samples

    # Check alignment for every sample
    for sample_idx in range(len(dataset)):
        # Input window features
        input_window_feat = dataset.X[sample_idx]  # shape: (window_size, 2)
        input_end_ts = dataset.input_end_timestamps[sample_idx]

        # The last observation in the input window should correspond to the same time t
        # In our synthetic data: feat_1 = t * 10.0
        last_feat_val = input_window_feat[-1, 0]
        inferred_t = last_feat_val / 10.0

        # Target vector y must be exactly [t+1, t+3, t+6, t+12, t+24]
        target_vector = dataset.y[sample_idx]
        expected_targets = np.array([inferred_t + 1, inferred_t + 3, inferred_t + 6, inferred_t + 12, inferred_t + 24])
        np.testing.assert_allclose(target_vector, expected_targets, err_msg=f"Target mismatch at sample {sample_idx}")

        # Target timestamps must be exactly t + h hours
        sample_target_ts = pd.DatetimeIndex(dataset.target_timestamps[sample_idx])
        for j, h in enumerate(horizons):
            expected_h_ts = input_end_ts + pd.Timedelta(hours=h)
            assert sample_target_ts[j] == expected_h_ts, (
                f"Timestamp mismatch at sample {sample_idx}, horizon {h}h: "
                f"expected {expected_h_ts}, got {sample_target_ts[j]}"
            )


# ==============================================================================
# 2. Sequence Lengths & Multi-Horizon Shapes Test
# ==============================================================================

@pytest.mark.parametrize("window_size", [24, 48, 72])
def test_multi_horizon_shapes(window_size: int):
    """Verify input shape (N, W, D) and target shape (N, 5) across lookbacks 24, 48, 72."""
    df = make_synthetic_hourly_df(num_hours=200)
    features = ["feat_1", "feat_2"]
    horizons = (1, 3, 6, 12, 24)

    dataset = create_multistep_sequences(
        df_features=df,
        target_series=df["PM2.5"],
        timestamps=df["datetime"],
        feature_cols=features,
        window_size=window_size,
        horizons=horizons,
    )

    n_samples = len(dataset)
    assert dataset.X.shape == (n_samples, window_size, len(features))
    assert dataset.y.shape == (n_samples, 5)
    assert dataset.X.dtype == np.float32
    assert dataset.y.dtype == np.float32
    assert dataset.target_timestamps.shape == (n_samples, 5)
    assert len(dataset.input_end_timestamps) == n_samples


# ==============================================================================
# 3. Missing Target Exclusions & Accounting Test
# ==============================================================================

def test_missing_target_exclusions():
    """
    Verify that candidate sequences are excluded when ANY of the 5 targets is missing,
    and exclusion statistics are accurately tracked.
    """
    df = make_synthetic_hourly_df(num_hours=200)
    target_series = df["PM2.5"].copy()

    # Total possible candidates for W=24, H_max=24:
    # first_idx = 23, last_idx = 200 - 24 = 176 => 176 - 23 = 153 candidates.

    # 1. Inject NaN at index 50 (which acts as t+1 for candidate i=49)
    target_series.iloc[50] = np.nan

    # 2. Inject NaN at index 80 (acts as t+3 for candidate i=77, t+6 for candidate i=74, etc.)
    target_series.iloc[80] = np.nan

    # 3. Inject NaN at index 120 (acts as t+24 for candidate i=96)
    target_series.iloc[120] = np.nan

    dataset = create_multistep_sequences(
        df_features=df,
        target_series=target_series,
        timestamps=df["datetime"],
        feature_cols=["feat_1", "feat_2"],
        window_size=24,
        horizons=(1, 3, 6, 12, 24),
        require_valid_target=True,
    )

    # Invariant 1: Resulting y array must NOT contain any NaNs
    assert not np.isnan(dataset.y).any(), "Resulting y array contains NaNs!"

    # Invariant 2: Candidate exclusion accounting
    report = dataset.exclusion_report
    assert report is not None
    assert report.total_candidates_evaluated == 153
    assert report.excluded_missing_target > 0
    assert report.valid_sequences_retained == len(dataset)
    assert report.total_candidates_evaluated == report.valid_sequences_retained + report.total_excluded

    # Verify that candidate i=49 (whose t+1 is index 50) was discarded:
    assert df["datetime"].iloc[49] not in dataset.input_end_timestamps

    # Verify that candidate i=96 (whose t+24 is index 120) was discarded:
    assert df["datetime"].iloc[96] not in dataset.input_end_timestamps


# ==============================================================================
# 4. Station Isolation Guarantee Test
# ==============================================================================

def test_station_isolation_guarantee():
    """Verify that multi-station contamination raises an immediate error."""
    df1 = make_synthetic_hourly_df(num_hours=100, station="Station_A")
    df2 = make_synthetic_hourly_df(num_hours=100, station="Station_B")
    combined_df = pd.concat([df1, df2], ignore_index=True)

    with pytest.raises(ValueError, match="Cross-station sequence generation detected"):
        create_multistep_sequences(
            df_features=combined_df,
            target_series=combined_df["PM2.5"],
            timestamps=combined_df["datetime"],
            feature_cols=["feat_1", "feat_2"],
            window_size=24,
            horizons=(1, 3, 6, 12, 24),
        )


# ==============================================================================
# 5. Timestamp Continuity & Gap Detection Test
# ==============================================================================

def test_timestamp_continuity_and_gap_exclusion():
    """Verify that candidate windows spanning discontinuous hours are cleanly excluded."""
    # Create hourly data, then drop hours from 04:00 to 08:00
    dates = pd.date_range("2020-01-01 00:00", periods=150, freq="1h")
    df = pd.DataFrame({
        "datetime": dates,
        "station": "TestStation",
        "PM2.5": np.arange(150, dtype=float),
        "feat_1": np.arange(150, dtype=float),
    })

    # Drop 5 hours: rows 30 through 34
    df_with_gap = df.drop(index=list(range(30, 35))).reset_index(drop=True)

    dataset = create_multistep_sequences(
        df_features=df_with_gap,
        target_series=df_with_gap["PM2.5"],
        timestamps=df_with_gap["datetime"],
        feature_cols=["feat_1"],
        window_size=24,
        horizons=(1, 3, 6, 12, 24),
    )

    report = dataset.exclusion_report
    assert report is not None
    assert report.excluded_timestamp_gaps > 0
    # Every retained sample must pass validation
    dataset.validate()


# ==============================================================================
# 6. Non-Finite Feature Window Exclusion Test
# ==============================================================================

def test_non_finite_feature_exclusion():
    """Verify that candidate windows containing NaN or Inf in features are rejected."""
    df = make_synthetic_hourly_df(num_hours=150)
    df_feat = df[["feat_1", "feat_2"]].copy()

    # Inject NaN into feature at row 40
    df_feat.iloc[40, 0] = np.nan

    dataset = create_multistep_sequences(
        df_features=df_feat,
        target_series=df["PM2.5"],
        timestamps=df["datetime"],
        feature_cols=["feat_1", "feat_2"],
        window_size=24,
        horizons=(1, 3, 6, 12, 24),
        require_valid_features=True,
    )

    report = dataset.exclusion_report
    assert report is not None
    # All windows that cover row 40 (indices 40 through 40+23 = 63) must be excluded
    assert report.excluded_nan_features == 24
    assert not np.isnan(dataset.X).any()


# ==============================================================================
# 7. Chronological Split Boundary Isolation Test
# ==============================================================================

def test_chronological_split_boundary_isolation():
    """
    Verify chronological split boundary constraints:
    - Train targets strictly <= train_end
    - Val targets strictly >= val_start and <= val_end
    - Test targets strictly >= test_start
    - max(train_targets) < min(val_targets)
    - max(val_targets) < min(test_targets)
    """
    # Create 350 hours of synthetic data
    df = make_synthetic_hourly_df(num_hours=350)

    # Fake boundaries
    train_end = pd.Timestamp("2020-01-08 00:00:00")  # hour 168
    val_start = pd.Timestamp("2020-01-08 01:00:00")  # hour 169
    val_end = pd.Timestamp("2020-01-11 00:00:00")    # hour 240
    test_start = pd.Timestamp("2020-01-11 01:00:00") # hour 241
    test_end = pd.Timestamp("2020-01-15 13:00:00")   # hour 349

    # Train sequence dataset
    train_df = df[df["datetime"] <= train_end]
    train_ds = create_multistep_sequences(
        df_features=train_df,
        target_series=train_df["PM2.5"],
        timestamps=train_df["datetime"],
        feature_cols=["feat_1", "feat_2"],
        window_size=24,
        horizons=(1, 3, 6, 12, 24),
        split_name="train",
        target_end_bound=train_end,
    )

    # Val sequence dataset (with prior context)
    val_df_with_ctx = df[(df["datetime"] >= train_end - pd.Timedelta(hours=48)) & (df["datetime"] <= val_end)]
    val_ds = create_multistep_sequences(
        df_features=val_df_with_ctx,
        target_series=val_df_with_ctx["PM2.5"],
        timestamps=val_df_with_ctx["datetime"],
        feature_cols=["feat_1", "feat_2"],
        window_size=24,
        horizons=(1, 3, 6, 12, 24),
        split_name="val",
        target_start_bound=val_start,
        target_end_bound=val_end,
    )

    # Test sequence dataset (with prior context)
    test_df_with_ctx = df[(df["datetime"] >= val_end - pd.Timedelta(hours=48)) & (df["datetime"] <= test_end)]
    test_ds = create_multistep_sequences(
        df_features=test_df_with_ctx,
        target_series=test_df_with_ctx["PM2.5"],
        timestamps=test_df_with_ctx["datetime"],
        feature_cols=["feat_1", "feat_2"],
        window_size=24,
        horizons=(1, 3, 6, 12, 24),
        split_name="test",
        target_start_bound=test_start,
        target_end_bound=test_end,
    )

    # Verify zero leakage across boundaries
    assert pd.to_datetime(train_ds.target_timestamps.max()) <= train_end
    assert pd.to_datetime(val_ds.target_timestamps.min()) >= val_start
    assert pd.to_datetime(val_ds.target_timestamps.max()) <= val_end
    assert pd.to_datetime(test_ds.target_timestamps.min()) >= test_start

    assert pd.to_datetime(train_ds.target_timestamps.max()) < pd.to_datetime(val_ds.target_timestamps.min())
    assert pd.to_datetime(val_ds.target_timestamps.max()) < pd.to_datetime(test_ds.target_timestamps.min())


# ==============================================================================
# 8. Batch Iterator & Empty Dataset Handling Test
# ==============================================================================

def test_batch_iterator_and_empty_dataset():
    """Test pure-NumPy iter_batches and graceful handling of empty datasets."""
    df = make_synthetic_hourly_df(num_hours=80)
    dataset = create_multistep_sequences(
        df_features=df,
        target_series=df["PM2.5"],
        timestamps=df["datetime"],
        feature_cols=["feat_1"],
        window_size=24,
        horizons=(1, 3, 6, 12, 24),
    )

    batches = list(dataset.iter_batches(batch_size=16, shuffle=False))
    assert len(batches) > 0
    total_iter_samples = sum(len(X_b) for X_b, y_b in batches)
    assert total_iter_samples == len(dataset)
    for X_b, y_b in batches:
        assert X_b.shape[1] == 24
        assert X_b.shape[2] == 1
        assert y_b.shape[1] == 5

    # Test empty dataset: window larger than data length
    empty_ds = create_multistep_sequences(
        df_features=df.head(10),
        target_series=df["PM2.5"].head(10),
        timestamps=df["datetime"].head(10),
        feature_cols=["feat_1"],
        window_size=24,
    )
    assert len(empty_ds) == 0
    assert empty_ds.X.shape == (0, 24, 1)
    assert empty_ds.y.shape == (0, 5)
    empty_batches = list(empty_ds.iter_batches(batch_size=16))
    assert len(empty_batches) == 0


# ==============================================================================
# 9. Real Station Data Integration Test
# ==============================================================================

def test_real_station_sequence_bundle():
    """
    Test end-to-end multi-horizon sequence generation using real station data (Aotizhongxin).
    Verifies shapes, target scaling, inverse transformation, and boundary isolation.
    """
    splits = process_station("Aotizhongxin")

    for w in [24, 48, 72]:
        bundle = generate_station_sequence_bundle(
            splits=splits,
            window_size=w,
            horizons=(1, 3, 6, 12, 24),
            scaled=True,
            scale_target=True,
        )

        # Validate bundle integrity
        assert bundle.validate() is True

        # Shape checks
        num_feat = len(splits.feature_cols)
        assert bundle.X_train.shape[1] == w
        assert bundle.X_train.shape[2] == num_feat
        assert bundle.y_train.shape[1] == 5

        assert bundle.X_val.shape[1] == w
        assert bundle.X_val.shape[2] == num_feat
        assert bundle.y_val.shape[1] == 5

        assert bundle.X_test.shape[1] == w
        assert bundle.X_test.shape[2] == num_feat
        assert bundle.y_test.shape[1] == 5

        # Check that no NaNs exist in any array
        assert not np.isnan(bundle.X_train).any()
        assert not np.isnan(bundle.y_train).any()
        assert not np.isnan(bundle.X_val).any()
        assert not np.isnan(bundle.y_val).any()
        assert not np.isnan(bundle.X_test).any()
        assert not np.isnan(bundle.y_test).any()

        # Check inverse transformation
        y_train_inv = bundle.train.inverse_transform_targets()
        assert y_train_inv.shape == bundle.y_train.shape
        # Physical PM2.5 in Beijing should generally fall in [0, 1000] µg/m³
        assert (y_train_inv >= 0.0).all()
        assert (y_train_inv < 1500.0).all()

        # Summary table generation check
        table_str = bundle.get_summary_table()
        assert "Train" in table_str
        assert "Val" in table_str
        assert "Test" in table_str

