"""
Unit tests for src/split_data.py.

Verifies:
1. Split boundaries remain strictly chronological (train < val < test).
2. Split proportions match requested configuration (e.g. 70% / 15% / 15%).
3. Multi-station timestamp alignment (all stations share identical cutoff timestamps).
4. No target timestamp leaks across split boundaries.
5. Embargo gap effectively prevents target transition overlap.
6. Learned transformation isolation (scalers fitted strictly on train).
7. Invalid configurations (ratios != 1.0, missing columns) raise descriptive errors.
"""

import numpy as np
import pandas as pd
import pytest

from src.split_data import (
    get_split_boundaries,
    chronological_split,
    validate_split_leakage,
    fit_transform_scaler,
)


@pytest.fixture
def synthetic_split_df():
    """Create a 100-hour chronological multi-station dataset for split verification."""
    dates = pd.date_range("2020-01-01 00:00", periods=100, freq="h")
    records = []

    for st in ["StationA", "StationB"]:
        for i, dt in enumerate(dates):
            records.append({
                "station": st,
                "datetime": dt,
                "PM2.5": float(i + (100 if st == "StationB" else 0)),
                "target_pm25": float(i + 1 + (100 if st == "StationB" else 0)),
                "feature_1": float(i * 2),
                "feature_2": float(i * 0.5),
            })

    return pd.DataFrame(records)


def test_chronological_order_and_proportions(synthetic_split_df):
    """Verify train, val, test partitions are ordered strictly forward in time."""
    train_df, val_df, test_df, report = chronological_split(
        synthetic_split_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    # 1. Monotonic partition boundaries
    assert train_df["datetime"].max() < val_df["datetime"].min()
    assert val_df["datetime"].max() < test_df["datetime"].min()

    # 2. Sample counts (100 hours * 2 stations = 200 total records)
    # 70% -> 70 hours * 2 = 140 rows
    # 15% -> 15 hours * 2 = 30 rows
    # 15% -> 15 hours * 2 = 30 rows
    assert len(train_df) == 140
    assert len(val_df) == 30
    assert len(test_df) == 30
    assert report["train_percentage"] == 70.0
    assert report["val_percentage"] == 15.0
    assert report["test_percentage"] == 15.0


def test_zero_timestamp_overlap(synthetic_split_df):
    """Verify no timestamp appears in more than one partition."""
    train_df, val_df, test_df, _ = chronological_split(synthetic_split_df)

    train_ts = set(train_df["datetime"].unique())
    val_ts = set(val_df["datetime"].unique())
    test_ts = set(test_df["datetime"].unique())

    assert len(train_ts.intersection(val_ts)) == 0
    assert len(val_ts.intersection(test_ts)) == 0
    assert len(train_ts.intersection(test_ts)) == 0


def test_multi_station_synchronized_boundaries(synthetic_split_df):
    """Verify all stations share identical timestamp cutoffs."""
    train_df, val_df, test_df, report = chronological_split(synthetic_split_df)

    st_a_train = train_df[train_df["station"] == "StationA"]
    st_b_train = train_df[train_df["station"] == "StationB"]

    assert st_a_train["datetime"].min() == st_b_train["datetime"].min()
    assert st_a_train["datetime"].max() == st_b_train["datetime"].max()

    assert report["records_per_station"]["StationA"]["train"] == 70
    assert report["records_per_station"]["StationB"]["train"] == 70


def test_embargo_prevents_target_overlap(synthetic_split_df):
    """Verify embargo hours prevent the last training target from touching validation start."""
    # Without embargo: last train row at hour 69 has target at hour 70 (which is val start)
    train_raw, val_raw, test_raw, _ = chronological_split(synthetic_split_df, embargo_hours=0)
    audit_raw = validate_split_leakage(train_raw, val_raw, test_raw, horizon_hours=1)
    assert audit_raw["train_last_target_timestamp"] == audit_raw["val_min_timestamp"]

    # With 1-hour embargo: last train row at hour 68 has target at hour 69 < val start (hour 70)
    train_emb, val_emb, test_emb, rep_emb = chronological_split(synthetic_split_df, embargo_hours=1)
    audit_emb = validate_split_leakage(train_emb, val_emb, test_emb, horizon_hours=1)

    assert audit_emb["passed_all_checks"] is True
    assert pd.to_datetime(audit_emb["train_last_target_timestamp"]) < pd.to_datetime(audit_emb["val_min_timestamp"])
    assert rep_emb["embargoed_records"] == 4  # 1 hour * 2 stations at train boundary + 1 hour * 2 at val boundary


def test_invalid_ratios_raise_error(synthetic_split_df):
    """Verify invalid ratio configurations raise ValueError."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        chronological_split(synthetic_split_df, train_ratio=0.80, val_ratio=0.15, test_ratio=0.15)


def test_learned_transformation_isolation():
    """Verify scalers are fitted strictly on training data."""
    train_X = pd.DataFrame({"feat": [10.0, 20.0, 30.0]})  # mean=20, std=8.165
    val_X = pd.DataFrame({"feat": [100.0, 200.0, 300.0]}) # vastly different distribution
    test_X = pd.DataFrame({"feat": [1000.0, 2000.0, 3000.0]})

    train_scaled, val_scaled, test_scaled, scaler = fit_transform_scaler(
        train_X, val_X, test_X, scaler_type="standard"
    )

    # Scaler mean must equal train mean (20.0), NOT pooled mean
    assert np.isclose(scaler.mean_[0], 20.0)
    assert np.isclose(train_scaled["feat"].mean(), 0.0, atol=1e-5)
    # Val scaled using train mean: (100 - 20) / std != 0
    assert not np.isclose(val_scaled["feat"].mean(), 0.0)
