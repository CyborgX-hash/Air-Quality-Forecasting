"""
Unit and integration tests for Phase 2 data pipeline (src/phase2/data_pipeline.py).

Verifies:
1. Station-isolated data processing.
2. Chronological ordering preservation.
3. Target non-imputation (missing PM2.5 values are never filled).
4. Feature selection and domain/cyclical feature construction.
5. Chronological 70% Train / 15% Val / 15% Test splitting.
6. Context-preserving boundary slices for validation and testing.
7. LSTM sequence generation (3D X tensor and aligned y target).
8. Target alignment across horizons (1, 3, 6, 12, 24) and windows (24, 48, 72).
9. Reproducibility across multiple runs.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase2.config import Phase2Config
from src.phase2.data_pipeline import (
    StationDataSplits,
    create_lstm_sequences,
    prepare_station_features,
    process_station,
    split_station_data,
)


@pytest.fixture
def synthetic_station_raw_df():
    """Create a 120-hour synthetic raw station DataFrame."""
    dates = pd.date_range("2020-01-01 00:00", periods=120, freq="h")
    records = []

    for i, dt in enumerate(dates):
        # Target missing at a few specific timestamps
        pm25 = np.nan if i in [10, 45, 95] else float(30.0 + (i % 24) * 2.0)
        records.append({
            "No": i + 1,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "station": "TestStation",
            "datetime": dt,
            "PM2.5": pm25,
            "PM10": float(50.0 + (i % 24) * 3.0),
            "SO2": 15.0,
            "NO2": 35.0,
            "CO": 800.0,
            "O3": 40.0,
            "TEMP": float(5.0 + 10.0 * np.sin(i / 12.0)),
            "PRES": 1015.0,
            "DEWP": float(0.0 + 8.0 * np.sin(i / 12.0)),
            "RAIN": 0.0 if i % 15 != 0 else 2.5,
            "wd": "NE" if i % 2 == 0 else "SW",
            "WSPM": 1.0 if i % 10 == 0 else 3.5,  # Stagnation indicator trigger
        })

    return pd.DataFrame(records)


def test_prepare_station_features_features_and_target_invariance(synthetic_station_raw_df):
    """Verify feature generation produces all required columns and preserves target NaNs."""
    initial_nan_count = int(synthetic_station_raw_df["PM2.5"].isnull().sum())
    assert initial_nan_count == 3

    df_feat, feat_cols = prepare_station_features(synthetic_station_raw_df)

    # 1. Feature existence
    expected_subset = [
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
        "wd_sin", "wd_cos",
        "temp_dewp_spread", "wind_stagnation", "rain_clearing",
        "sin_hour", "cos_hour", "sin_month", "cos_month", "sin_dow", "cos_dow",
    ]
    for col in expected_subset:
        assert col in df_feat.columns, f"Expected column '{col}' missing from prepared features"
        assert col in feat_cols

    # 2. Domain logic validation
    # temp_dewp_spread = TEMP - DEWP
    expected_spread = df_feat["TEMP"] - df_feat["DEWP"]
    assert np.allclose(df_feat["temp_dewp_spread"], expected_spread, atol=1e-5)

    # wind_stagnation = WSPM < 1.5
    expected_stag = (df_feat["WSPM"] < 1.5).astype(int)
    assert (df_feat["wind_stagnation"] == expected_stag).all()

    # rain_clearing = RAIN > 0.0
    expected_rain = (df_feat["RAIN"] > 0.0).astype(int)
    assert (df_feat["rain_clearing"] == expected_rain).all()

    # 3. Target Invariance: PM2.5 ground-truth target must NEVER be imputed
    final_target_nan_count = int(df_feat["target_pm25"].isnull().sum())
    assert final_target_nan_count == initial_nan_count

    # 4. Input feature continuity: feature PM2.5 was causally filled so sequences are continuous
    assert df_feat["PM2.5"].isnull().sum() == 0


def test_split_station_data_chronological_and_context(synthetic_station_raw_df):
    """Verify chronological splitting and warm-up context prepending."""
    df_feat, _ = prepare_station_features(synthetic_station_raw_df)

    max_w = 24
    (
        train_df,
        val_df,
        test_df,
        val_with_context_df,
        test_with_context_df,
        boundaries,
    ) = split_station_data(
        df_feat,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        max_context_window=max_w,
    )

    # 1. Monotonic partition ordering
    assert train_df["datetime"].max() < val_df["datetime"].min()
    assert val_df["datetime"].max() < test_df["datetime"].min()

    # 2. Check counts for 120 hours:
    # 70% -> 84 hours
    # 15% -> 18 hours
    # 15% -> 18 hours
    assert len(train_df) == 84
    assert len(val_df) == 18
    assert len(test_df) == 18

    # 3. Check context slices:
    # val_with_context_df should have min(max_w, len(train_df)) + len(val_df) = 24 + 18 = 42
    assert len(val_with_context_df) == 24 + 18
    # The tail of val_with_context_df should be identical to val_df
    pd.testing.assert_frame_equal(
        val_with_context_df.tail(len(val_df)).reset_index(drop=True),
        val_df.reset_index(drop=True),
    )

    # test_with_context_df: last 18 hours of val + 18 hours of test = 36 (since val has 18 < 24)
    assert len(test_with_context_df) == 18 + 18
    pd.testing.assert_frame_equal(
        test_with_context_df.tail(len(test_df)).reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def test_create_lstm_sequences_shapes_and_target_alignment(synthetic_station_raw_df):
    """Verify 3D X tensor construction and target alignment at t + horizon."""
    df_feat, feat_cols = prepare_station_features(synthetic_station_raw_df)

    window_size = 24
    horizon = 3  # Forecast 3 hours ahead

    X, y, ts = create_lstm_sequences(
        df_features=df_feat[feat_cols],
        target_series=df_feat["PM2.5"],
        timestamps=df_feat["datetime"],
        feature_cols=feat_cols,
        window_size=window_size,
        horizon=horizon,
        require_valid_target=True,
    )

    # 1. Shape validation: (N, 24, num_features)
    assert X.ndim == 3
    assert X.shape[1] == window_size
    assert X.shape[2] == len(feat_cols)
    assert y.ndim == 1
    assert len(y) == len(X)
    assert len(ts) == len(X)

    # 2. Target alignment: check first generated sequence
    # For k=0 (origin index i = window_size - 1 = 23):
    # input window is df_feat.iloc[0:24]
    # target is df_feat["PM2.5"].iloc[23 + 3] = df_feat["PM2.5"].iloc[26]
    # timestamp is df_feat["datetime"].iloc[26]
    expected_first_target = df_feat["PM2.5"].iloc[23 + horizon]
    expected_first_ts = df_feat["datetime"].iloc[23 + horizon]

    assert np.isclose(y[0], expected_first_target, atol=1e-5)
    assert ts[0] == expected_first_ts

    # 3. Missing target handling: NO target in y should be NaN
    assert not np.isnan(y).any()


def test_create_lstm_sequences_multiple_windows_and_horizons(synthetic_station_raw_df):
    """Verify sequence generation operates smoothly across all required windows and horizons."""
    df_feat, feat_cols = prepare_station_features(synthetic_station_raw_df)

    test_windows = [24, 48, 72]
    test_horizons = [1, 3, 6, 12, 24]

    for w in test_windows:
        for h in test_horizons:
            X, y, ts = create_lstm_sequences(
                df_features=df_feat[feat_cols],
                target_series=df_feat["PM2.5"],
                timestamps=df_feat["datetime"],
                feature_cols=feat_cols,
                window_size=w,
                horizon=h,
                require_valid_target=True,
            )
            assert X.shape[1] == w
            assert X.shape[2] == len(feat_cols)
            assert len(X) == len(y) == len(ts)
            assert not np.isnan(y).any()


def test_process_station_end_to_end(synthetic_station_raw_df, tmp_path):
    """Verify process_station produces complete StationDataSplits with fitted scalers."""
    cfg = Phase2Config.get_default_config()
    cfg.scaling.scalers_dir = str(tmp_path / "scalers")
    cfg.scaling.save_scalers = True

    splits = process_station(
        station_name="TestStation",
        config=cfg,
        df_raw=synthetic_station_raw_df,
    )

    assert isinstance(splits, StationDataSplits)
    assert splits.station_name == "TestStation"
    assert splits.scaler.is_fitted
    assert len(splits.train_df) > 0
    assert len(splits.val_df) > 0
    assert len(splits.test_df) > 0

    # Test sequence generation methods
    X_train, y_train, ts_train = splits.get_train_sequences(window_size=24, horizon=1)
    assert X_train.ndim == 3
    assert X_train.shape[1] == 24
    assert len(X_train) == len(y_train)

    X_val, y_val, ts_val = splits.get_val_sequences(window_size=24, horizon=1, use_context=True)
    assert len(X_val) > 0
    # All validation target timestamps must lie strictly within the validation period
    assert (ts_val >= splits.boundaries["val_start"]).all()
    assert (ts_val <= splits.boundaries["val_end"]).all()

    X_test, y_test, ts_test = splits.get_test_sequences(window_size=24, horizon=1, use_context=True)
    assert len(X_test) > 0
    assert (ts_test >= splits.boundaries["test_start"]).all()

    # Test inverse transformation
    raw_sample = y_test[:5]
    inv_sample = splits.inverse_transform_target(raw_sample)
    assert len(inv_sample) == len(raw_sample)


def test_pipeline_reproducibility(synthetic_station_raw_df):
    """Verify that repeated pipeline runs produce bitwise identical sequences."""
    cfg = Phase2Config.get_default_config()
    cfg.scaling.save_scalers = False

    splits_1 = process_station("TestStation", config=cfg, df_raw=synthetic_station_raw_df)
    splits_2 = process_station("TestStation", config=cfg, df_raw=synthetic_station_raw_df)

    X1, y1, _ = splits_1.get_train_sequences(window_size=24, horizon=1)
    X2, y2, _ = splits_2.get_train_sequences(window_size=24, horizon=1)

    assert np.allclose(X1, X2, atol=1e-6)
    assert np.allclose(y1, y2, atol=1e-6)
