"""Integration tests: data pipeline + scaling (Saksham's modules) on synthetic CSVs."""

import numpy as np
import pandas as pd
import pytest

from src.phase2 import integrity
from tests.phase2.conftest import STATIONS
from tests.phase2.synthetic import make_station_frame


def test_target_never_imputed_features_are_filled(splits_by_station, raw_dir):
    """Missing PM2.5 stays NaN in the target column; model inputs have no NaN."""
    for st, s in splits_by_station.items():
        raw = pd.read_csv(next(raw_dir.glob(f"PRSA_Data_{st}_*.csv")))
        n_missing_raw = int(raw["PM2.5"].isna().sum())
        n_missing_target = int(sum(d[s.target_col].isna().sum() for d in (s.train_df, s.val_df, s.test_df)))
        assert n_missing_raw > 0, "fixture must contain missing targets"
        assert n_missing_target == n_missing_raw, "target must be neither imputed nor dropped"
        for d in (s.train_scaled, s.val_scaled, s.test_scaled):
            assert not d[s.feature_cols].isna().any().any()


def test_station_isolation(splits_by_station):
    a, b = (splits_by_station[st] for st in STATIONS)
    assert a.station_name != b.station_name
    assert a.scaler is not b.scaler
    assert not np.isclose(a.scaler.target_mean_, b.scaler.target_mean_)
    for s in (a, b):
        stations_in_data = set(s.train_df.get("station", [s.station_name])) | \
                           set(s.test_df.get("station", [s.station_name]))
        assert stations_in_data == {s.station_name}


def test_timestamp_continuity(splits_by_station):
    for s in splits_by_station.values():
        for part in (s.train_df, s.val_df, s.test_df):
            deltas = part["datetime"].diff().dropna().unique()
            assert list(deltas) == [pd.Timedelta(hours=1)]


def test_chronological_split_and_ratios(splits_by_station, cfg):
    res = integrity.check_chronological_split(splits_by_station)
    assert res["status"] == integrity.PASS, res
    for s in splits_by_station.values():
        n = len(s.train_df) + len(s.val_df) + len(s.test_df)
        assert len(s.train_df) / n == pytest.approx(cfg.split.train_ratio, abs=0.01)
        assert len(s.val_df) / n == pytest.approx(cfg.split.val_ratio, abs=0.01)


def test_random_split_would_be_detected(splits_by_station):
    """Negative control: shuffling the train partition must fail the chronology check."""
    s = splits_by_station[STATIONS[0]]
    original = s.train_df
    try:
        s.train_df = original.sample(frac=1.0, random_state=0)
        res = integrity.check_chronological_split({STATIONS[0]: s})
    finally:
        s.train_df = original
    assert res["status"] == integrity.FAIL


def test_scaler_fitted_on_training_only(splits_by_station):
    assert integrity.check_scaler_train_only(splits_by_station)["status"] == integrity.PASS
    s = splits_by_station[STATIONS[0]]
    full = pd.concat([s.train_df, s.val_df, s.test_df])[s.target_col].dropna()
    assert not np.isclose(s.scaler.target_mean_, full.mean(), rtol=1e-9), \
        "fixture must make leaked and train-only statistics distinguishable"


def test_leaky_scaler_is_detected(splits_by_station):
    s = splits_by_station[STATIONS[0]]
    saved = s.scaler.target_mean_
    try:
        s.scaler.target_mean_ = float(pd.concat([s.train_df, s.test_df])[s.target_col].mean())
        res = integrity.check_scaler_train_only({STATIONS[0]: s})
    finally:
        s.scaler.target_mean_ = saved
    assert res["status"] == integrity.FAIL


def test_inverse_transform_roundtrip(splits_by_station):
    assert integrity.check_inverse_transform_roundtrip(splits_by_station)["status"] == integrity.PASS


def test_feature_order_identical_across_stations(splits_by_station, cfg):
    res = integrity.check_feature_order_consistent(splits_by_station, cfg.get_feature_list())
    assert res["status"] == integrity.PASS, res
