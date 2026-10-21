"""
Unit and integration tests for multi-station air-quality forecasting.

Covers:
1. All 12 station CSV files validation and schema conformity.
2. Station isolation in causal cleaning.
3. Station isolation in time-series feature engineering and target creation.
4. Chronological splitting synchronization across multiple stations.
5. Station-wise metric accounting and aggregate benchmark calculations.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import VALID_STATIONS
from src.data.load_data import validate_all_stations_schema, REQUIRED_COLUMNS
from src.preprocessing import clean_features
from src.features import generate_features, create_target
from src.split_data import chronological_split
from experiments.run_experiments import compute_aggregate_benchmarks


DATA_DIR = Path("data/raw")


@pytest.mark.skipif(not DATA_DIR.exists() or len(list(DATA_DIR.glob("PRSA_Data_*.csv"))) < 12,
                    reason="Raw 12-station dataset not present in data/raw")
def test_all_12_station_files_exist_and_validate():
    """Verify all 12 stations exist, follow the required schema, and have 35,064 records."""
    audit = validate_all_stations_schema(data_dir=DATA_DIR, expected_stations=VALID_STATIONS)
    assert audit["all_valid"], f"Validation failed for some stations: {audit['stations_missing']}"
    assert len(audit["stations_found"]) == 12
    assert len(audit["stations_missing"]) == 0

    for st in VALID_STATIONS:
        st_rep = audit["station_reports"][st]
        assert st_rep["valid"] is True
        assert st_rep["record_count"] == 35064
        assert len(st_rep["missing_columns"]) == 0
        assert st_rep["duplicate_records"] == 0
        assert st_rep["is_chronological"] is True
        assert st_rep["is_hourly_continuous"] is True


def test_station_isolation_in_target_creation():
    """Verify that create_target never carries target values across station boundaries."""
    # Create two stations where station A ends at 2015-01-02 23:00 and station B starts at 2015-01-01 00:00
    dates_a = pd.date_range("2015-01-01", periods=10, freq="1h")
    dates_b = pd.date_range("2015-01-01", periods=10, freq="1h")

    df_a = pd.DataFrame({"station": "StationA", "datetime": dates_a, "PM2.5": [10.0 + i for i in range(10)]})
    df_b = pd.DataFrame({"station": "StationB", "datetime": dates_b, "PM2.5": [500.0 + i for i in range(10)]})
    df_combined = pd.concat([df_a, df_b], ignore_index=True)

    df_res = create_target(df_combined, target_col="PM2.5", horizon=1, timestamp_col="datetime", station_col="station")

    # The last row of StationA MUST have NaN as target because horizon=1 has no t+1 within StationA
    station_a_last = df_res[df_res["station"] == "StationA"].iloc[-1]
    assert pd.isna(station_a_last["target_pm25"]), "Target bled across station boundaries!"

    # The second-to-last row of StationA must equal the last PM2.5 of StationA
    station_a_penultimate = df_res[df_res["station"] == "StationA"].iloc[-2]
    assert station_a_penultimate["target_pm25"] == station_a_last["PM2.5"]


def test_station_isolation_in_feature_engineering():
    """Verify that perturbing Station B does not alter any engineered feature of Station A."""
    dates = pd.date_range("2015-01-01", periods=50, freq="1h")
    rng = np.random.RandomState(42)

    df_a = pd.DataFrame({
        "station": "StationA",
        "datetime": dates,
        "PM2.5": rng.uniform(10, 100, size=len(dates)),
        "PM10": rng.uniform(20, 120, size=len(dates)),
        "SO2": rng.uniform(2, 20, size=len(dates)),
        "NO2": rng.uniform(10, 50, size=len(dates)),
        "CO": rng.uniform(500, 1500, size=len(dates)),
        "O3": rng.uniform(10, 80, size=len(dates)),
        "TEMP": rng.uniform(-10, 30, size=len(dates)),
        "PRES": rng.uniform(1000, 1025, size=len(dates)),
        "DEWP": rng.uniform(-15, 20, size=len(dates)),
        "RAIN": np.zeros(len(dates)),
        "wd": ["N"] * len(dates),
        "WSPM": rng.uniform(0.5, 5.0, size=len(dates)),
    })
    df_b1 = df_a.copy()
    df_b1["station"] = "StationB"

    # Combined dataset 1
    df_comb1 = pd.concat([df_a, df_b1], ignore_index=True)
    df_feat1, cols1, _ = generate_features(df_comb1)
    feat_a1 = df_feat1[df_feat1["station"] == "StationA"][cols1].copy()

    # Combined dataset 2: Perturb Station B drastically
    df_b2 = df_b1.copy()
    df_b2["PM2.5"] = df_b2["PM2.5"] * 1000.0
    df_b2["TEMP"] = -99.0
    df_comb2 = pd.concat([df_a, df_b2], ignore_index=True)
    df_feat2, cols2, _ = generate_features(df_comb2)
    feat_a2 = df_feat2[df_feat2["station"] == "StationA"][cols2].copy()

    # Features of Station A must be bit-for-bit identical regardless of Station B
    pd.testing.assert_frame_equal(feat_a1, feat_a2)


def test_station_isolation_in_causal_cleaning():
    """Verify that causal cleaning on Station B does not cross into Station A."""
    dates = pd.date_range("2015-01-01", periods=20, freq="1h")

    df_a = pd.DataFrame({
        "station": "StationA",
        "datetime": dates,
        "PM2.5": [10.0] * 20,
        "PM10": [20.0] * 20,
        "TEMP": [5.0] * 20,
        "wd": ["N"] * 20,
    })
    # Station B has missing values
    df_b = pd.DataFrame({
        "station": "StationB",
        "datetime": dates,
        "PM2.5": [50.0] * 20,
        "PM10": [np.nan] * 5 + [80.0] * 15,
        "TEMP": [np.nan] * 20,  # All missing
        "wd": [None] * 20,
    })

    df_comb = pd.concat([df_a, df_b], ignore_index=True)
    df_clean, rep = clean_features(df_comb, max_ffill_hours=6, fill_remaining_with_median=True)

    # Station A should have unchanged values
    st_a_clean = df_clean[df_clean["station"] == "StationA"]
    assert (st_a_clean["PM10"] == 20.0).all()
    assert (st_a_clean["TEMP"] == 5.0).all()


def test_multi_station_synchronized_chronological_split():
    """Verify chronological splitting synchronizes timestamps across multiple stations."""
    dates = pd.date_range("2015-01-01", periods=100, freq="1h")

    df_a = pd.DataFrame({"station": "StationA", "datetime": dates, "val": range(100)})
    df_b = pd.DataFrame({"station": "StationB", "datetime": dates, "val": range(100)})
    df_comb = pd.concat([df_a, df_b], ignore_index=True)

    train_df, val_df, test_df, rep = chronological_split(
        df_comb, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    # Each partition should have equal counts from each station
    assert (train_df["station"] == "StationA").sum() == (train_df["station"] == "StationB").sum() == 70
    assert (val_df["station"] == "StationA").sum() == (val_df["station"] == "StationB").sum() == 15
    assert (test_df["station"] == "StationA").sum() == (test_df["station"] == "StationB").sum() == 15

    # Check temporal boundaries are strictly non-overlapping
    assert train_df["datetime"].max() < val_df["datetime"].min()
    assert val_df["datetime"].max() < test_df["datetime"].min()


def test_station_wise_metric_accounting():
    """Verify macro-average and micro-average calculations."""
    mock_data = pd.DataFrame([
        {"station": "StA", "model": "ModelX", "test_mae": 10.0, "test_rmse": 20.0, "test_r2": 0.90, "test_mbe": -0.5, "n_obs": 1000, "mae_improvement_pct": 10.0},
        {"station": "StB", "model": "ModelX", "test_mae": 20.0, "test_rmse": 30.0, "test_r2": 0.80, "test_mbe": 0.5, "n_obs": 3000, "mae_improvement_pct": 5.0},
    ])

    agg = compute_aggregate_benchmarks(mock_data)
    row = agg[agg["model"] == "ModelX"].iloc[0]

    # Macro MAE: (10 + 20) / 2 = 15.0
    assert np.isclose(row["macro_mae"], 15.0)

    # Micro MAE: (10*1000 + 20*3000) / 4000 = 70000 / 4000 = 17.5
    assert np.isclose(row["micro_mae"], 17.5)

    # Macro R2: (0.90 + 0.80) / 2 = 0.85
    assert np.isclose(row["macro_r2"], 0.85)

    # Micro R2: (0.90*1000 + 0.80*3000) / 4000 = 3300 / 4000 = 0.825
    assert np.isclose(row["micro_r2"], 0.825)

    assert row["total_observations"] == 4000
    assert row["station_count"] == 2
