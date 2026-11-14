"""End-to-end: synthetic data -> ... -> report, run through the real CLI."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from src.phase2.run_pipeline import STAGES, main
from tests.phase2.conftest import STATIONS, run_cli


def test_full_run_creates_every_artifact(pipeline_run):
    d = pipeline_run["dir"]
    for st in STATIONS:
        assert (d / "scalers" / f"scaler_{st}.joblib").is_file()
        assert (d / "checkpoints" / f"lstm_{st}_w24_u64_l1_best.keras").is_file()
        assert (d / "history" / f"lstm_{st}_w24_u64_l1_history.json").is_file()
        assert (d / "history" / f"lstm_{st}_w24_u64_l1_history.csv").is_file()
        assert (d / "predictions" / f"{st}_w24_test.npz").is_file()
    for name in ("all_stations_multi_horizon_comparison.csv", "macro_average_performance_by_horizon.csv",
                 "pooled_performance_by_horizon.csv", "pollution_tier_error_analysis.csv"):
        assert (d / "evaluation" / name).is_file()
    assert len(list((d / "figures").glob("0*.png"))) == 8
    for name in ("report.md", "manifest.json", "config.yaml", "integrity_checks.json"):
        assert (d / name).is_file()
    m = json.loads((d / "manifest.json").read_text())
    assert [s for s in STAGES if m["stages"].get(s, {}).get("status") == "ok"] == list(STAGES)


def test_predictions_are_inverse_scaled_physical_units(pipeline_run, splits_by_station):
    """Saved predictions are in ug/m3 (same scale as raw PM2.5), not standardized values."""
    for st in STATIONS:
        z = np.load(pipeline_run["dir"] / "predictions" / f"{st}_w24_test.npz")
        raw = splits_by_station[st].test_df["target_pm25"].dropna()
        assert z["y_pred"].shape == z["y_true"].shape and z["y_pred"].shape[1] == 5
        assert abs(np.median(z["y_pred"]) - raw.median()) < 3 * raw.std()
        assert np.median(np.abs(z["y_pred"])) > 5       # standardized values would sit near 0
        assert np.isfinite(z["y_pred"]).all() and np.isfinite(z["y_true"]).all()


def test_report_is_generated_from_actual_outputs(pipeline_run):
    d = pipeline_run["dir"]
    report = (d / "report.md").read_text()
    macro = pd.read_csv(d / "evaluation" / "macro_average_performance_by_horizon.csv")
    for v in macro["lstm_macro_mae"]:
        assert f"{v:g}" in report or f"{v:.4f}" in report or f"{v:.3f}" in report
    assert "0 failed" in report and "none are hand-written" in report


def test_stages_can_run_separately_and_resume(raw_dir, tmp_path):
    for stages in ("prepare,sequences", "train", "predict", "evaluate,visualize,report"):
        assert run_cli(raw_dir, tmp_path, "staged", stages=stages) == 0
    m = json.loads((tmp_path / "staged" / "manifest.json").read_text())
    assert all(m["stages"][s]["status"] == "ok" for s in STAGES)
    assert (tmp_path / "staged" / "report.md").is_file()


def test_same_seed_reproduces_predictions(raw_dir, tmp_path, pipeline_run):
    assert run_cli(raw_dir, tmp_path, "repeat") == 0
    for st in STATIONS:
        a = np.load(pipeline_run["dir"] / "predictions" / f"{st}_w24_test.npz")["y_pred"]
        b = np.load(tmp_path / "repeat" / "predictions" / f"{st}_w24_test.npz")["y_pred"]
        np.testing.assert_allclose(a, b, atol=1e-4)
    ma = json.loads((pipeline_run["dir"] / "manifest.json").read_text())
    mb = json.loads((tmp_path / "repeat" / "manifest.json").read_text())
    assert ma["config_hash"] == mb["config_hash"]


def test_evaluate_without_checkpoints_fails_clearly(raw_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        run_cli(raw_dir, tmp_path, "nockpt", stages="evaluate")


def test_cli_rejects_bad_arguments(raw_dir, tmp_path):
    assert main(["--stages", "nonsense"]) == 2
    with pytest.raises(ValueError, match="Unknown stations"):
        main(["--stations", "Atlantis", "--stages", "prepare", "--output-root", str(tmp_path)])
