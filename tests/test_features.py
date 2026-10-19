"""
Unit tests for src/features.py.

Verifies:
1. Target is strictly one hour ahead within the same station.
2. Lag features use only historical data (no future leakage).
3. Rolling features exclude future target values (proven via perturbation).
4. Features are calculated independently by station (strict isolation).
5. Missing hourly timestamps are handled correctly (no blind shifting across gaps).
6. Feature matrix X strictly excludes target and metadata columns.
7. Cyclical and domain-specific weather features are mathematically sound.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    create_target,
    create_lag_features,
    create_rolling_features,
    create_calendar_features,
    create_cyclical_features,
    create_domain_weather_features,
    generate_features,
    extract_feature_matrix,
    get_feature_dictionary,
)


@pytest.fixture
def synthetic_twostation_df():
    """Create a controlled multi-station dataset for time-series feature testing."""
    dates = pd.date_range("2020-01-01 00:00", periods=6, freq="h")
    records = []

    # Station A: values 10, 20, 30, 40, 50, 60
    for i, dt in enumerate(dates):
        records.append({
            "station": "StationA",
            "datetime": dt,
            "PM2.5": 10.0 + i * 10.0,
            "PM10": 20.0 + i * 10.0,
            "NO2": 15.0 + i * 5.0,
            "CO": 500.0 + i * 50.0,
            "SO2": 5.0,
            "O3": 25.0,
            "TEMP": 5.0 + i,
            "PRES": 1010.0,
            "DEWP": 2.0,
            "RAIN": 0.0 if i < 4 else 1.5,
            "wd": "N" if i % 2 == 0 else "E",
            "WSPM": 1.0 if i < 3 else 3.5,
        })

    # Station B: values 100, 110, 120, 130, 140, 150
    for i, dt in enumerate(dates):
        records.append({
            "station": "StationB",
            "datetime": dt,
            "PM2.5": 100.0 + i * 10.0,
            "PM10": 200.0 + i * 10.0,
            "NO2": 50.0 + i * 5.0,
            "CO": 1000.0 + i * 50.0,
            "SO2": 10.0,
            "O3": 35.0,
            "TEMP": 10.0 + i,
            "PRES": 1015.0,
            "DEWP": 4.0,
            "RAIN": 0.0,
            "wd": "S",
            "WSPM": 0.8,
        })

    return pd.DataFrame(records)


def test_target_is_strictly_one_hour_ahead(synthetic_twostation_df):
    """Verify target row t is PM2.5 at t + 1 hour within the same station."""
    df_with_target = create_target(synthetic_twostation_df, horizon=1)

    st_a = df_with_target[df_with_target["station"] == "StationA"].reset_index(drop=True)
    # PM2.5 for Station A is [10, 20, 30, 40, 50, 60]
    # Target should be [20, 30, 40, 50, 60, NaN]
    assert st_a["target_pm25"].iloc[0] == 20.0
    assert st_a["target_pm25"].iloc[1] == 30.0
    assert st_a["target_pm25"].iloc[2] == 40.0
    assert st_a["target_pm25"].iloc[3] == 50.0
    assert st_a["target_pm25"].iloc[4] == 60.0
    assert pd.isna(st_a["target_pm25"].iloc[5])

    st_b = df_with_target[df_with_target["station"] == "StationB"].reset_index(drop=True)
    # PM2.5 for Station B is [100, 110, 120, 130, 140, 150]
    # Target should be [110, 120, 130, 140, 150, NaN]
    assert st_b["target_pm25"].iloc[0] == 110.0
    assert st_b["target_pm25"].iloc[4] == 150.0
    assert pd.isna(st_b["target_pm25"].iloc[5])


def test_lag_features_use_only_historical_data(synthetic_twostation_df):
    """Verify lag features reflect past observations and contain NaNs at the start."""
    df_lags, names = create_lag_features(
        synthetic_twostation_df,
        columns=["PM2.5"],
        lags=[1, 2, 3],
    )

    st_a = df_lags[df_lags["station"] == "StationA"].reset_index(drop=True)

    # At row 0 (00:00), past lags must be NaN
    assert pd.isna(st_a["pm25_lag_1h"].iloc[0])
    assert pd.isna(st_a["pm25_lag_2h"].iloc[0])
    assert pd.isna(st_a["pm25_lag_3h"].iloc[0])

    # At row 2 (02:00, PM2.5=30):
    # lag 1h should be PM2.5 at 01:00 (20.0)
    # lag 2h should be PM2.5 at 00:00 (10.0)
    # lag 3h should be NaN
    assert st_a["pm25_lag_1h"].iloc[2] == 20.0
    assert st_a["pm25_lag_2h"].iloc[2] == 10.0
    assert pd.isna(st_a["pm25_lag_3h"].iloc[2])

    # At row 3 (03:00, PM2.5=40): lag 3h is 10.0
    assert st_a["pm25_lag_3h"].iloc[3] == 10.0


def test_rolling_features_exclude_future_target_values(synthetic_twostation_df):
    """
    Perturbation test proving rolling features exclude future target values.

    If we modify a future row's PM2.5, past rolling statistics must remain invariant.
    """
    df_clean = synthetic_twostation_df.copy()
    df_rolled_orig, _ = create_rolling_features(
        df_clean,
        columns=["PM2.5"],
        windows=[3],
        stats=["mean", "std"],
        min_periods=1,
    )

    st_a_orig = df_rolled_orig[df_rolled_orig["station"] == "StationA"].reset_index(drop=True)
    # At row 2 (02:00), window is [10, 20, 30], mean = 20.0
    assert np.isclose(st_a_orig["pm25_roll_mean_3h"].iloc[2], 20.0)

    # Perturb row 3 (which is in the future relative to row 2)
    df_perturbed = df_clean.copy()
    idx_row3 = df_perturbed[(df_perturbed["station"] == "StationA") & (df_perturbed["datetime"] == "2020-01-01 03:00")].index
    df_perturbed.loc[idx_row3, "PM2.5"] = 99999.0

    df_rolled_pert, _ = create_rolling_features(
        df_perturbed,
        columns=["PM2.5"],
        windows=[3],
        stats=["mean", "std"],
        min_periods=1,
    )
    st_a_pert = df_rolled_pert[df_rolled_pert["station"] == "StationA"].reset_index(drop=True)

    # Row 2 rolling mean MUST NOT CHANGE despite massive future perturbation at row 3
    assert np.isclose(st_a_pert["pm25_roll_mean_3h"].iloc[2], 20.0)
    assert np.isclose(st_a_pert["pm25_roll_std_3h"].iloc[2], st_a_orig["pm25_roll_std_3h"].iloc[2])


def test_station_isolation_prevents_cross_boundary_leakage(synthetic_twostation_df):
    """Verify lags and targets never cross station boundaries."""
    df_feat, names, _ = generate_features(
        synthetic_twostation_df,
        pm25_lags=[1, 2],
        rolling_windows=[3],
        rolling_stats=["mean"],
    )

    st_b = df_feat[df_feat["station"] == "StationB"].reset_index(drop=True)
    st_a = df_feat[df_feat["station"] == "StationA"].reset_index(drop=True)

    # First row of Station B (00:00) must NOT pick up Station A's last row (60.0)
    assert pd.isna(st_b["pm25_lag_1h"].iloc[0])
    # Station B's rolling mean at 00:00 must be 100.0 (not including Station A)
    assert st_b["pm25_roll_mean_3h"].iloc[0] == 100.0

    # Last row of Station A (05:00) must NOT pick up Station B's first row (100.0) as target
    assert pd.isna(st_a["target_pm25"].iloc[-1])


def test_missing_timestamps_handled_correctly():
    """
    Verify that gaps in hourly timestamps do not create false 1-hour targets or lags.

    If 02:00 is missing, row 01:00 must NOT receive row 03:00 as its 1-hour target.
    Row 03:00 must NOT receive row 01:00 as its 1-hour lag.
    """
    dates = [
        pd.Timestamp("2020-01-01 00:00:00"),
        pd.Timestamp("2020-01-01 01:00:00"),
        # 02:00 is missing!
        pd.Timestamp("2020-01-01 03:00:00"),
        pd.Timestamp("2020-01-01 04:00:00"),
    ]
    df_gap = pd.DataFrame({
        "station": ["StationA"] * 4,
        "datetime": dates,
        "PM2.5": [10.0, 20.0, 40.0, 50.0],
    })

    # 1. Target check
    df_with_target = create_target(df_gap, horizon=1)
    # Row 0 (00:00) -> 01:00 (20.0)
    assert df_with_target["target_pm25"].iloc[0] == 20.0
    # Row 1 (01:00) -> 02:00 (missing) -> MUST BE NaN
    assert pd.isna(df_with_target["target_pm25"].iloc[1])
    # Row 2 (03:00) -> 04:00 (50.0)
    assert df_with_target["target_pm25"].iloc[2] == 50.0
    # Row 3 (04:00) -> 05:00 (missing/end) -> MUST BE NaN
    assert pd.isna(df_with_target["target_pm25"].iloc[3])

    # 2. Lag check
    df_with_lags, _ = create_lag_features(df_gap, columns=["PM2.5"], lags=[1, 2])
    # Row 2 (03:00): lag 1h (02:00) is missing -> MUST BE NaN
    assert pd.isna(df_with_lags["pm25_lag_1h"].iloc[2])
    # Row 2 (03:00): lag 2h (01:00) is present -> MUST BE 20.0
    assert df_with_lags["pm25_lag_2h"].iloc[2] == 20.0


def test_extract_feature_matrix_prevents_target_leakage(synthetic_twostation_df):
    """Verify extracted feature matrix X never contains target_pm25 or future-only columns."""
    df_feat, feature_cols, meta = generate_features(synthetic_twostation_df)

    X, y, metadata = extract_feature_matrix(
        df_feat,
        feature_cols=feature_cols,
        target_col="target_pm25",
        require_valid_target=True,
    )

    # 1. Target must NOT be in X
    assert "target_pm25" not in X.columns
    assert "PM2.5_target" not in X.columns
    assert "target_valid" not in X.columns

    # 2. Metadata columns not in X
    assert "station" not in X.columns
    assert "datetime" not in X.columns
    assert "No" not in X.columns

    # 3. y matches target and valid length
    assert len(X) == len(y)
    assert y.name == "target_pm25"
    assert not y.isnull().any()

    # 4. Metadata preserved
    assert "station" in metadata.columns
    assert "datetime" in metadata.columns


def test_calendar_and_cyclical_features(synthetic_twostation_df):
    """Verify mathematical properties of cyclical trigonometric transformations."""
    df_cyc, cyc_cols = create_cyclical_features(synthetic_twostation_df)

    for col in cyc_cols:
        assert col in df_cyc.columns
        assert df_cyc[col].between(-1.0001, 1.0001).all()

    # Verify sin^2 + cos^2 == 1 for hour
    hour_norm = df_cyc["sin_hour"]**2 + df_cyc["cos_hour"]**2
    assert np.allclose(hour_norm, 1.0, atol=1e-5)

    # Verify sin^2 + cos^2 == 1 for month
    month_norm = df_cyc["sin_month"]**2 + df_cyc["cos_month"]**2
    assert np.allclose(month_norm, 1.0, atol=1e-5)


def test_domain_weather_features():
    """Verify dew point depression, wind stagnation, and rain indicators."""
    df = pd.DataFrame({
        "TEMP": [25.0, 10.0, 0.0],
        "DEWP": [20.0, 10.0, -5.0],
        "WSPM": [1.0, 1.5, 4.0],
        "RAIN": [0.0, 0.5, 0.0],
    })
    df_dom, dom_cols = create_domain_weather_features(df)

    # Dew point depression: [5.0, 0.0, 5.0]
    assert list(df_dom["temp_dewp_spread"]) == [5.0, 0.0, 5.0]

    # Stagnation (< 1.5 m/s): [1, 0, 0]
    assert list(df_dom["wind_stagnation"]) == [1, 0, 0]

    # Rain clearing (> 0): [0, 1, 0]
    assert list(df_dom["rain_clearing"]) == [0, 1, 0]


def test_feature_dictionary():
    """Verify the feature dictionary returns structured, complete metadata."""
    dict_df = get_feature_dictionary()
    assert isinstance(dict_df, pd.DataFrame)
    assert set(["feature_group", "feature_name", "formula", "physical_meaning", "role"]).issubset(dict_df.columns)
    assert "target_pm25" in dict_df["feature_name"].values
    assert "pm25_lag_1h" in dict_df["feature_name"].values
    assert "temp_dewp_spread" in dict_df["feature_name"].values
    assert len(dict_df) > 30
