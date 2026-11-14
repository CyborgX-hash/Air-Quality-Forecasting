"""Evaluation integration: station-wise, macro and pooled metrics on the pipeline's outputs."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from src.evaluate import calculate_metrics
from src.phase2 import integrity
from src.phase2.evaluate import calculate_pooled_metrics
from tests.phase2.conftest import STATIONS

HORIZONS = [1, 3, 6, 12, 24]


def test_pooled_rmse_is_not_mean_of_station_rmse():
    rng = np.random.default_rng(0)
    yt = [rng.uniform(10, 100, (50, 5)), rng.uniform(10, 100, (500, 5))]
    yp = [yt[0] + rng.normal(0, 2, (50, 5)), yt[1] + rng.normal(0, 20, (500, 5))]
    pooled = calculate_pooled_metrics(yt, yp, HORIZONS).set_index("horizon_hours")
    station_rmse = [np.sqrt(np.mean((p[:, 0] - t[:, 0]) ** 2)) for t, p in zip(yt, yp)]
    exact = np.sqrt(np.mean(np.concatenate([(p[:, 0] - t[:, 0]) ** 2 for t, p in zip(yt, yp)])))
    assert pooled.loc[1, "pooled_rmse"] == pytest.approx(exact, abs=1e-3)
    assert abs(pooled.loc[1, "pooled_rmse"] - np.mean(station_rmse)) > 1.0


def test_metric_conventions_match_phase1():
    y = np.array([10.0, 20.0, 30.0]); p = np.array([12.0, 18.0, 36.0])
    m = calculate_metrics(y, p)
    assert m["mae"] == pytest.approx(10 / 3, abs=1e-3)
    assert m["mbe"] == pytest.approx(2.0, abs=1e-3)  # errors +2,-2,+6; mean(pred - true), + = over-prediction


def test_pipeline_succeeded_with_all_integrity_checks_green(pipeline_run):
    assert pipeline_run["rc"] == 0
    s = json.loads((pipeline_run["dir"] / "integrity_checks.json").read_text())
    failed = [c for c in s["checks"] if c["status"] == "FAIL"]
    assert not failed, failed
    assert s["passed"] >= 12


def test_station_wise_results_cover_every_station_and_horizon(pipeline_run):
    d = pd.read_csv(pipeline_run["dir"] / "evaluation" / "all_stations_multi_horizon_comparison.csv")
    assert set(d["station"]) == set(STATIONS)
    for st in STATIONS:
        assert sorted(d.loc[d["station"] == st, "horizon_hours"]) == HORIZONS
    assert (d["n_obs"] > 0).all()


def test_macro_and_pooled_tables_follow_definitions(pipeline_run):
    ev = pipeline_run["dir"] / "evaluation"
    det = pd.read_csv(ev / "all_stations_multi_horizon_comparison.csv")
    macro = pd.read_csv(ev / "macro_average_performance_by_horizon.csv").set_index("horizon_hours")
    pooled = pd.read_csv(ev / "pooled_performance_by_horizon.csv").set_index("horizon_hours")
    for h in HORIZONS:
        sub = det[det["horizon_hours"] == h]
        assert macro.loc[h, "lstm_macro_mae"] == pytest.approx(sub["lstm_mae"].mean(), abs=1e-3)
        assert pooled.loc[h, "total_observations"] == sub["n_obs"].sum()
        # pooled MAE is the observation-weighted mean of station MAEs
        w = np.average(sub["lstm_mae"], weights=sub["n_obs"])
        assert pooled.loc[h, "pooled_mae_lstm"] == pytest.approx(w, abs=1e-2)


def test_tampered_ground_truth_is_detected(pipeline_run, raw_dir, cfg, splits_by_station):
    from src.phase2.run_pipeline import Phase2Pipeline, evaluator_bound_to_run  # noqa: F401
    from src.phase2.experiment_manager import ExperimentManager, load_phase2_config
    c, s, raw = load_phase2_config("configs/phase2_config.yaml")
    s.window_size = 24
    mgr = ExperimentManager(c, s, raw, run_id="e2e", output_root=pipeline_run["root"])
    pipe = Phase2Pipeline(mgr, stations=STATIONS, data_dir=str(raw_dir))
    pipe._splits = splits_by_station
    suite = pipe.suite()
    assert integrity.check_ground_truth_observed(suite, splits_by_station)["status"] == integrity.PASS
    assert integrity.check_metrics_follow_phase1(suite)["status"] == integrity.PASS
    rec = suite.station_records[STATIONS[0]]
    saved = rec.y_true.copy()
    try:
        rec.y_true[0, 0] += 123.456          # a value that was never observed
        assert integrity.check_ground_truth_observed(suite, splits_by_station)["status"] == integrity.FAIL
        assert integrity.check_metrics_follow_phase1(suite)["status"] == integrity.FAIL
    finally:
        rec.y_true[:] = saved


def test_evaluator_patch_is_fully_restored(splits_by_station):
    import src.phase2.evaluate as ev
    from src.phase2.run_pipeline import evaluator_bound_to_run
    a, b = ev.evaluate_single_station, ev.process_station
    with evaluator_bound_to_run(splits_by_station, {}):
        assert ev.evaluate_single_station is not a
    assert ev.evaluate_single_station is a and ev.process_station is b
