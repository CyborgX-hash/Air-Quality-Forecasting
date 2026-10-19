"""
Unit tests for src/data/load_data.py.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.load_data import (
    build_timestamp,
    validate_data,
    load_station_data,
    load_all_stations,
    REQUIRED_COLUMNS,
)

SAMPLE_RAW_FILE = Path("data/raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv")


@pytest.fixture
def sample_toy_df():
    """Create a minimal synthetic valid air quality DataFrame."""
    dates = pd.date_range("2015-01-01 00:00", periods=5, freq="h")
    data = {
        "No": [1, 2, 3, 4, 5],
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "hour": dates.hour,
        "PM2.5": [10.0, 20.0, np.nan, 40.0, 50.0],
        "PM10": [15.0, 25.0, 35.0, 45.0, 55.0],
        "SO2": [2.0, 3.0, 4.0, 5.0, 6.0],
        "NO2": [20.0, 22.0, 24.0, 26.0, 28.0],
        "CO": [500.0, 600.0, 700.0, 800.0, 900.0],
        "O3": [30.0, 35.0, 40.0, 45.0, 50.0],
        "TEMP": [1.0, 2.0, 3.0, 4.0, 5.0],
        "PRES": [1020.0, 1019.0, 1018.0, 1017.0, 1016.0],
        "DEWP": [-10.0, -9.0, -8.0, -7.0, -6.0],
        "RAIN": [0.0, 0.0, 0.0, 0.0, 0.0],
        "wd": ["NNW", "N", "NW", "NNE", "ENE"],
        "WSPM": [2.0, 2.5, 3.0, 1.5, 2.2],
        "station": ["Aotizhongxin"] * 5,
    }
    return pd.DataFrame(data)


def test_build_timestamp(sample_toy_df):
    df_with_ts = build_timestamp(sample_toy_df)
    assert "datetime" in df_with_ts.columns
    assert pd.api.types.is_datetime64_any_dtype(df_with_ts["datetime"])
    assert df_with_ts["datetime"].iloc[0] == pd.Timestamp("2015-01-01 00:00:00")
    assert df_with_ts["datetime"].iloc[-1] == pd.Timestamp("2015-01-01 04:00:00")


def test_build_timestamp_missing_col():
    df_invalid = pd.DataFrame({"year": [2015], "month": [1], "day": [1]})
    with pytest.raises(KeyError, match="Missing required date component column: 'hour'"):
        build_timestamp(df_invalid)


def test_build_timestamp_drop_source(sample_toy_df):
    df_dropped = build_timestamp(sample_toy_df, drop_source_cols=True)
    assert "datetime" in df_dropped.columns
    for col in ["year", "month", "day", "hour"]:
        assert col not in df_dropped.columns


def test_validate_data_valid(sample_toy_df):
    df_with_ts = build_timestamp(sample_toy_df)
    report = validate_data(df_with_ts)
    assert report["is_valid"] is True
    assert report["duplicate_records"] == 0
    assert report["is_chronological"] is True
    assert report["is_hourly_continuous"] is True
    assert report["target_stats"]["missing"] == 1
    assert report["target_stats"]["valid"] == 4


def test_validate_data_detects_duplicates(sample_toy_df):
    df_with_ts = build_timestamp(sample_toy_df)
    # Duplicate row 0
    df_dup = pd.concat([df_with_ts, df_with_ts.iloc[[0]]], ignore_index=True)
    report = validate_data(df_dup)
    assert report["duplicate_records"] == 1
    assert report["is_valid"] is False


def test_validate_data_detects_non_monotonic(sample_toy_df):
    df_with_ts = build_timestamp(sample_toy_df)
    # Reverse rows to disrupt monotonic order
    df_reversed = df_with_ts.iloc[::-1].reset_index(drop=True)
    report = validate_data(df_reversed)
    assert report["is_chronological"] is False


def test_validate_data_detects_hourly_gap(sample_toy_df):
    df_with_ts = build_timestamp(sample_toy_df)
    # Drop row 2 to create a 2-hour gap between row 1 and row 3
    df_gap = df_with_ts.drop(index=2).reset_index(drop=True)
    report = validate_data(df_gap)
    assert report["is_hourly_continuous"] is False
    assert report["continuity_gaps_per_station"].get("Aotizhongxin") == 1


@pytest.mark.skipif(not SAMPLE_RAW_FILE.exists(), reason="Raw dataset not present on disk")
def test_load_station_data_real_file():
    df = load_station_data(SAMPLE_RAW_FILE)
    assert len(df) == 35064
    assert "datetime" in df.columns
    assert "station" in df.columns
    assert df["station"].iloc[0] == "Aotizhongxin"
    assert df["datetime"].is_monotonic_increasing


def test_load_station_data_nonexistent():
    with pytest.raises(FileNotFoundError, match="Station data file not found"):
        load_station_data("data/raw/non_existent_station.csv")


@pytest.mark.skipif(not SAMPLE_RAW_FILE.exists(), reason="Raw dataset not present on disk")
def test_load_all_stations_real_files():
    df_all = load_all_stations("data/raw")
    assert len(df_all) == 420768
    assert len(df_all["station"].unique()) == 12
