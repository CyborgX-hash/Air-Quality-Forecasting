"""Integration tests: Saksham's splits -> Ajit's sequence bundle (+ cross-generator contract)."""

import numpy as np
import pandas as pd
import pytest

from src.phase2 import integrity
from src.phase2.sequence_generator import generate_station_sequence_bundle
from tests.phase2.conftest import STATIONS

HORIZONS = [1, 3, 6, 12, 24]
W = 24


@pytest.fixture(scope="module")
def bundles(splits_by_station):
    return {st: generate_station_sequence_bundle(sp, W, HORIZONS) for st, sp in splits_by_station.items()}


def test_shapes(bundles, splits_by_station):
    for st, b in bundles.items():
        d = len(splits_by_station[st].feature_cols)
        for part in (b.train, b.val, b.test):
            assert part.X.shape[1:] == (W, d)
            assert part.y.shape == (len(part), len(HORIZONS))
            assert len(part) > 0


def test_target_and_horizon_alignment(bundles, splits_by_station):
    """y[i, j] must equal the raw PM2.5 observed exactly horizon[j] hours after the window end."""
    for st, b in bundles.items():
        s = splits_by_station[st]
        raw = pd.concat([s.train_df, s.val_df, s.test_df]).set_index("datetime")[s.target_col]
        for ds in (b.train, b.val, b.test):
            for i in np.linspace(0, len(ds) - 1, 25, dtype=int):
                origin = ds.input_end_timestamps[i]
                expected = [raw[origin + pd.Timedelta(hours=h)] for h in HORIZONS]
                got = s.scaler.inverse_transform_target(ds.y[i])
                np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


def test_samples_with_missing_target_are_excluded(bundles):
    for b in bundles.values():
        for ds in (b.train, b.val, b.test):
            assert np.isfinite(ds.y).all()
            assert np.isfinite(ds.X).all()
            assert ds.exclusion_report.excluded_missing_target > 0  # fixture has missing PM2.5


def test_no_cross_station_and_no_boundary_crossing(bundles, splits_by_station):
    assert integrity.check_no_cross_station(bundles)["status"] == integrity.PASS
    assert integrity.check_sequence_target_bounds(bundles, splits_by_station)["status"] == integrity.PASS


def test_train_val_test_targets_do_not_overlap(bundles):
    for b in bundles.values():
        t = lambda ds: pd.DatetimeIndex(np.asarray(ds.target_timestamps).ravel())
        assert t(b.train).max() < t(b.val).min()
        assert t(b.val).max() < t(b.test).min()


def test_generators_agree_on_samples_with_one_hour_label_offset(bundles, splits_by_station):
    """
    Contract between Ajit's generator (used for training) and Saksham's
    ``get_*_sequences`` (used by predict/evaluate): identical samples and targets,
    but Saksham labels a sample with the first *target* time (origin + 1 h) while
    Ajit labels it with the window-end time. If this changes, diurnal/monthly
    analysis in evaluate.py silently shifts.
    """
    for st, b in bundles.items():
        s = splits_by_station[st]
        X, y, ts = s.get_test_sequences(window_size=W, horizons=HORIZONS, use_context=True)
        np.testing.assert_allclose(X, b.test.X, atol=1e-5)
        np.testing.assert_allclose(y, b.test.y, atol=1e-5)
        assert (ts == b.test.input_end_timestamps + pd.Timedelta(hours=1)).all()
