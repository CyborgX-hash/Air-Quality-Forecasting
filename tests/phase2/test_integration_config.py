"""Shared Phase 2 config + experiment manager (reproducibility) tests."""

import json
import random

import numpy as np
import pytest
import yaml

from src.config import VALID_STATIONS
from src.phase2.experiment_manager import (
    ExperimentManager,
    ExperimentSettings,
    config_hash,
    load_phase2_config,
    seed_everything,
)

CONFIG = "configs/phase2_config.yaml"


def test_shared_config_contains_every_required_section():
    raw = yaml.safe_load(open(CONFIG))
    for key in ("stations", "data", "features", "lstm_windows", "forecast_horizons", "split",
                "model", "training", "scaling", "project", "experiment", "evaluation",
                "reproducibility"):
        assert key in raw, f"missing config section '{key}'"
    assert raw["project"]["random_seed"] is not None
    assert raw["data"]["target_column"] == "PM2.5"


def test_config_is_consistent_with_phase1_and_itself():
    cfg, settings, _ = load_phase2_config(CONFIG)
    assert cfg.stations.all == list(VALID_STATIONS) and len(cfg.stations.all) == 12
    assert cfg.forecast_horizons.horizons == [1, 3, 6, 12, 24]
    assert cfg.model.output_units == len(cfg.forecast_horizons.horizons)
    assert cfg.split.chronological is True
    assert cfg.cleaning.impute_target is False
    assert len(cfg.get_feature_list()) == len(set(cfg.get_feature_list()))
    assert settings.evaluation_partition == "test"


def test_raw_dir_matches_data_readme_casing():
    cfg, _, _ = load_phase2_config(CONFIG)
    assert cfg.data.raw_dir == "data/raw"  # Linux is case sensitive


def test_scoped_config_redirects_all_outputs_and_leaves_original(tmp_path):
    cfg, settings, raw = load_phase2_config(CONFIG)
    before = cfg.to_dict()
    mgr = ExperimentManager(cfg, settings, raw, run_id="t1", output_root=tmp_path)
    sc = mgr.scoped_config()
    for d in (sc.scaling.scalers_dir, sc.training.checkpoint_dir,
              sc.training.history_dir, sc.training.figures_dir):
        assert str(mgr.run_dir) in d
    assert cfg.to_dict() == before


def test_run_id_and_config_hash(tmp_path):
    cfg, settings, raw = load_phase2_config(CONFIG)
    h1 = config_hash(cfg, settings)
    assert h1 == config_hash(cfg, settings)
    cfg.model.lstm_units = 128
    assert config_hash(cfg, settings) != h1
    mgr = ExperimentManager(cfg, settings, raw, output_root=tmp_path)
    assert mgr.run_id.startswith(settings.run_name) and mgr.run_id.endswith(mgr.config_hash)
    assert mgr.run_dir.is_dir()


def test_manifest_records_environment_and_stages(tmp_path):
    cfg, settings, raw = load_phase2_config(CONFIG)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "PRSA_Data_X_1.csv").write_text("a,b\n1,2\n")
    mgr = ExperimentManager(cfg, settings, raw, run_id="m1", output_root=tmp_path)
    mgr.record_environment(seed_everything(7), raw_data_dir=raw_dir)
    mgr.record_stage("prepare", "ok", stations=1)
    m = json.loads(mgr.manifest_path.read_text())
    assert m["environment"]["python"] and "numpy" in m["environment"]["libraries"]
    assert "tensorflow" in m["environment"]["libraries"]
    assert len(m["raw_data_sha256"]["PRSA_Data_X_1.csv"]) == 64
    assert m["seeding"]["seed"] == 7 and m["stages"]["prepare"]["status"] == "ok"
    # effective config is snapshotted and re-loadable
    snap_cfg, snap_settings, _ = load_phase2_config(mgr.run_dir / "config.yaml")
    assert snap_cfg.to_dict() == cfg.to_dict()


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = (random.random(), np.random.rand())
    seed_everything(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_settings_defaults_when_sections_absent():
    s = ExperimentSettings.from_dict({})
    assert s.evaluation_partition == "test" and s.window_size is None
