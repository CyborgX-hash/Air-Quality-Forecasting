"""
Unit tests for src/preprocessing.py.
"""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing import (
    clean_features,
    add_cyclical_wind_features,
    get_valid_training_mask,
    WIND_DIRECTION_MAP,
)


@pytest.fixture
def synthetic_multistation_df():
    """Create a controlled multi-station time series dataset with missing values."""
    dates = pd.date_range("2016-01-01 00:00", periods=6, freq="h")
    records = []

    # Station A
    for i, dt in enumerate(dates):
        records.append({
            "station": "StationA",
            "datetime": dt,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            # PM2.5 missing at index 2
            "PM2.5": np.nan if i == 2 else 50.0 + i * 5,
            # PM10 missing at index 0 and 3
            "PM10": np.nan if i in [0, 3] else 80.0 + i * 2,
            "SO2": 10.0,
            "NO2": 30.0,
            "CO": 800.0,
            "O3": 40.0,
            "TEMP": 5.0,
            "PRES": 1015.0,
            "DEWP": -5.0,
            "RAIN": 0.0,
            "wd": np.nan if i == 1 else "N",
            "WSPM": 2.0,
        })

    # Station B
    for i, dt in enumerate(dates):
        records.append({
            "station": "StationB",
            "datetime": dt,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            # PM2.5 missing at index 0 and 5
            "PM2.5": np.nan if i in [0, 5] else 30.0 + i * 4,
            "PM10": 70.0,
            "SO2": 8.0,
            "NO2": 25.0,
            "CO": 600.0,
            "O3": 35.0,
            "TEMP": 4.0,
            "PRES": 1016.0,
            "DEWP": -6.0,
            "RAIN": 0.0,
            "wd": "E",
            "WSPM": 1.5,
        })

    return pd.DataFrame(records)


def test_clean_features_preserves_target_nans(synthetic_multistation_df):
    """Verify target PM2.5 is strictly never imputed."""
    initial_target_nans = synthetic_multistation_df["PM2.5"].isnull().sum()
    assert initial_target_nans == 3  # 1 in StationA, 2 in StationB

    df_clean, report = clean_features(synthetic_multistation_df, target_col="PM2.5")

    # Cleaned target missing count must remain exactly the same
    assert df_clean["PM2.5"].isnull().sum() == initial_target_nans
    assert report["target_imputed"] is False
    assert report["target_missing_count"] == initial_target_nans
    assert "target_valid" in df_clean.columns
    assert (~df_clean["target_valid"]).sum() == initial_target_nans


def test_clean_features_zero_rows_dropped(synthetic_multistation_df):
    """Verify no rows are silently deleted."""
    initial_len = len(synthetic_multistation_df)
    df_clean, report = clean_features(synthetic_multistation_df)
    assert len(df_clean) == initial_len
    assert report["rows_dropped"] == 0
    assert report["initial_rows"] == initial_len
    assert report["final_rows"] == initial_len


def test_clean_features_causal_no_future_leakage():
    """Verify forward-fill does not back-propagate future values (preventing lookahead bias)."""
    dates = pd.date_range("2016-01-01 00:00", periods=4, freq="h")
    df = pd.DataFrame({
        "station": ["StationA"] * 4,
        "datetime": dates,
        "PM2.5": [10.0, 20.0, 30.0, 40.0],
        "PM10": [np.nan, np.nan, 80.0, 90.0],  # first two are missing
        "SO2": [10.0, 10.0, 10.0, 10.0],
        "NO2": [20.0, 20.0, 20.0, 20.0],
        "CO": [500.0, 500.0, 500.0, 500.0],
        "O3": [30.0, 30.0, 30.0, 30.0],
        "TEMP": [1.0, 2.0, 3.0, 4.0],
        "PRES": [1010.0, 1010.0, 1010.0, 1010.0],
        "DEWP": [0.0, 0.0, 0.0, 0.0],
        "RAIN": [0.0, 0.0, 0.0, 0.0],
        "wd": ["N", "N", "N", "N"],
        "WSPM": [1.0, 1.0, 1.0, 1.0],
    })

    # Run clean with fill_remaining_with_median=False to isolate ffill
    df_clean, report = clean_features(df, max_ffill_hours=2, fill_remaining_with_median=False)

    # First two rows cannot be forward-filled from future row 2 (which is 80.0)
    assert pd.isna(df_clean["PM10"].iloc[0])
    assert pd.isna(df_clean["PM10"].iloc[1])
    assert df_clean["PM10"].iloc[2] == 80.0
    assert df_clean["PM10"].iloc[3] == 90.0


def test_clean_features_station_isolation():
    """Verify station A values do not forward-fill into station B."""
    dates = pd.date_range("2016-01-01 00:00", periods=2, freq="h")
    df = pd.DataFrame({
        "station": ["StationA", "StationA", "StationB", "StationB"],
        "datetime": list(dates) + list(dates),
        "PM2.5": [10.0, 20.0, 30.0, 40.0],
        "PM10": [100.0, 100.0, np.nan, np.nan],  # StationB has NaNs at start
        "SO2": [1.0, 1.0, 1.0, 1.0],
        "NO2": [2.0, 2.0, 2.0, 2.0],
        "CO": [300.0, 300.0, 300.0, 300.0],
        "O3": [4.0, 4.0, 4.0, 4.0],
        "TEMP": [5.0, 5.0, 5.0, 5.0],
        "PRES": [1000.0, 1000.0, 1000.0, 1000.0],
        "DEWP": [0.0, 0.0, 0.0, 0.0],
        "RAIN": [0.0, 0.0, 0.0, 0.0],
        "wd": ["N", "N", "E", "E"],
        "WSPM": [1.0, 1.0, 1.0, 1.0],
    })

    df_clean, report = clean_features(df, fill_remaining_with_median=False)
    # StationB index 2 and 3 must not take StationA's 100.0
    st_b = df_clean[df_clean["station"] == "StationB"]
    assert pd.isna(st_b["PM10"].iloc[0])
    assert pd.isna(st_b["PM10"].iloc[1])


def test_cyclical_wind_encoding():
    """Verify sine and cosine cyclical transformations for cardinal directions."""
    df_wind = pd.DataFrame({"wd": ["N", "E", "S", "W"]})
    df_encoded = add_cyclical_wind_features(df_wind)

    # N: 0 deg -> sin=0, cos=1
    assert np.isclose(df_encoded.loc[df_encoded["wd"] == "N", "wd_sin"].iloc[0], 0.0, atol=1e-5)
    assert np.isclose(df_encoded.loc[df_encoded["wd"] == "N", "wd_cos"].iloc[0], 1.0, atol=1e-5)

    # E: 90 deg -> sin=1, cos=0
    assert np.isclose(df_encoded.loc[df_encoded["wd"] == "E", "wd_sin"].iloc[0], 1.0, atol=1e-5)
    assert np.isclose(df_encoded.loc[df_encoded["wd"] == "E", "wd_cos"].iloc[0], 0.0, atol=1e-5)

    # S: 180 deg -> sin=0, cos=-1
    assert np.isclose(df_encoded.loc[df_encoded["wd"] == "S", "wd_sin"].iloc[0], 0.0, atol=1e-5)
    assert np.isclose(df_encoded.loc[df_encoded["wd"] == "S", "wd_cos"].iloc[0], -1.0, atol=1e-5)

    # W: 270 deg -> sin=-1, cos=0
    assert np.isclose(df_encoded.loc[df_encoded["wd"] == "W", "wd_sin"].iloc[0], -1.0, atol=1e-5)
    assert np.isclose(df_encoded.loc[df_encoded["wd"] == "W", "wd_cos"].iloc[0], 0.0, atol=1e-5)


def test_get_valid_training_mask(synthetic_multistation_df):
    """Verify supervised training mask computation for 1-hour ahead forecast."""
    # Station A has 6 rows: PM2.5 missing at idx 2
    # For horizon=1:
    # row 0 targets row 1 (valid)
    # row 1 targets row 2 (missing -> False)
    # row 2 targets row 3 (valid)
    # row 3 targets row 4 (valid)
    # row 4 targets row 5 (valid)
    # row 5 targets row 6 (out of bounds -> False)
    mask = get_valid_training_mask(synthetic_multistation_df, horizon=1)
    st_a_mask = mask.iloc[:6].tolist()
    assert st_a_mask == [True, False, True, True, True, False]
