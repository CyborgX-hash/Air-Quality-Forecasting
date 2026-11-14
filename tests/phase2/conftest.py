"""Shared fixtures for the Phase 2 integration tests (synthetic data only)."""

import os

os.environ.setdefault("KERAS_HOME", "/tmp/.keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")

import pytest

from src.phase2.config import Phase2Config
from src.phase2.data_pipeline import process_station
from tests.phase2.synthetic import write_synthetic_raw_dir

STATIONS = ["Aotizhongxin", "Changping"]


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    """Two synthetic stations in the exact UCI CSV schema (2400 hourly rows each)."""
    return write_synthetic_raw_dir(tmp_path_factory.mktemp("raw"), STATIONS, n_hours=2400)


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    c = Phase2Config.from_yaml("configs/phase2_config.yaml")
    c.scaling.save_scalers = False
    c.scaling.scalers_dir = str(tmp_path_factory.mktemp("scalers"))
    return c


@pytest.fixture(scope="session")
def splits_by_station(raw_dir, cfg):
    return {st: process_station(st, config=cfg, data_dir=raw_dir) for st in STATIONS}


def run_cli(raw_dir, out_root, run_id, stages="all", extra=()):
    """Invoke the real CLI entry point on the synthetic data (fast settings)."""
    from src.phase2.run_pipeline import main
    argv = ["--config", "configs/phase2_config.yaml", "--stages", stages,
            "--stations", ",".join(STATIONS), "--raw-dir", str(raw_dir),
            "--output-root", str(out_root), "--run-id", run_id,
            "--epochs", "2", "--window-size", "24", *extra]
    return main(argv)


@pytest.fixture(scope="session")
def pipeline_run(raw_dir, tmp_path_factory):
    """One complete synthetic pipeline run shared by evaluation / e2e tests."""
    out = tmp_path_factory.mktemp("runs")
    rc = run_cli(raw_dir, out, "e2e")
    return {"rc": rc, "dir": out / "e2e", "root": out}
