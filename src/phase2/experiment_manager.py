"""
Experiment management and reproducibility for the Phase 2 pipeline.

Author: Priyanshu Agrahari (Integration Engineer, Testing and Reproducibility)

Responsibilities
----------------
* Parse the *integration-level* settings (``experiment``, ``evaluation`` and
  ``reproducibility`` sections) from the single shared Phase 2 YAML config.
  The per-module configuration (``Phase2Config``) is left untouched; unknown
  sections are ignored by it, so both views read the same file.
* Create a run-scoped directory tree with a clear experiment identifier.
* Redirect every module's output directory (scalers, checkpoints, history,
  figures) into that run directory without modifying any other member's code.
* Seed every random number generator and (optionally) request deterministic
  TensorFlow kernels.
* Record the exact software / hardware environment, the config hash, the git
  commit and the SHA-256 of the raw data files in ``manifest.json``.

Nothing in this module imports TensorFlow at import time, so it can be used
(and tested) on machines without TensorFlow.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from src.phase2.config import Phase2Config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/phase2_config.yaml"
RUN_SUBDIRS = (
    "scalers",
    "checkpoints",
    "history",
    "predictions",
    "evaluation",
    "figures",
    "audit",
)


# ---------------------------------------------------------------------------
# Integration-level settings
# ---------------------------------------------------------------------------
@dataclass
class ExperimentSettings:
    """Integration settings that live beside (not inside) ``Phase2Config``."""

    # experiment
    output_root: str = "experiments/phase2_runs"
    run_name: str = "phase2"
    window_size: Optional[int] = None  # None -> lstm_windows.default_sequence_length
    primary_station: Optional[str] = None  # None -> stations.default
    # evaluation
    evaluation_partition: str = "test"
    evaluation_batch_size: int = 64
    prediction_atol: float = 1e-4  # saved vs. re-computed predictions
    # reproducibility
    deterministic_ops: bool = False
    hash_raw_data: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExperimentSettings":
        exp = raw.get("experiment", {}) or {}
        ev = raw.get("evaluation", {}) or {}
        rep = raw.get("reproducibility", {}) or {}
        return cls(
            output_root=exp.get("output_root", cls.output_root),
            run_name=exp.get("run_name", cls.run_name),
            window_size=exp.get("window_size", None),
            primary_station=exp.get("primary_station", None),
            evaluation_partition=ev.get("partition", cls.evaluation_partition),
            evaluation_batch_size=int(ev.get("batch_size", cls.evaluation_batch_size)),
            prediction_atol=float(ev.get("prediction_atol", cls.prediction_atol)),
            deterministic_ops=bool(rep.get("deterministic_ops", cls.deterministic_ops)),
            hash_raw_data=bool(rep.get("hash_raw_data", cls.hash_raw_data)),
        )


def load_phase2_config(
    path: Union[str, Path] = DEFAULT_CONFIG_PATH,
) -> "tuple[Phase2Config, ExperimentSettings, Dict[str, Any]]":
    """Load the shared YAML once and return (module config, settings, raw dict)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Phase2Config.from_dict(raw), ExperimentSettings.from_dict(raw), raw


def config_hash(cfg: Phase2Config, settings: Optional[ExperimentSettings] = None) -> str:
    """Stable 8-char hash of everything that influences results."""
    payload: Dict[str, Any] = {"phase2": cfg.to_dict()}
    if settings is not None:
        payload["settings"] = asdict(settings)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_everything(seed: int, deterministic_ops: bool = False) -> Dict[str, Any]:
    """
    Seed python, numpy and (if installed) TensorFlow/Keras.

    ``PYTHONHASHSEED`` only takes effect for *new* interpreters; it is exported
    so that subprocesses inherit it, and recorded for documentation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    info: Dict[str, Any] = {"seed": seed, "tensorflow_seeded": False,
                            "deterministic_ops": False}
    try:
        import tensorflow as tf  # noqa: WPS433 (lazy import by design)

        tf.keras.utils.set_random_seed(seed)  # python + numpy + tf
        info["tensorflow_seeded"] = True
        if deterministic_ops:
            tf.config.experimental.enable_op_determinism()
            info["deterministic_ops"] = True
    except ImportError:  # pragma: no cover - TF is a hard requirement elsewhere
        logger.warning("TensorFlow not installed; only python/numpy seeded.")
    return info


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------
def _git_info() -> Dict[str, Any]:
    def _run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10, check=True
            )
            return out.stdout.strip()
        except Exception:  # noqa: BLE001 - git may be absent / not a repo
            return None

    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def capture_environment() -> Dict[str, Any]:
    """Versions of Python and every library that influences numerical results."""
    versions: Dict[str, Optional[str]] = {}
    for mod in ("tensorflow", "keras", "numpy", "pandas", "sklearn", "scipy",
                "matplotlib", "joblib", "yaml", "tabulate"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            versions[mod] = None

    hardware: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "tensorflow_devices": [],
    }
    try:
        import tensorflow as tf

        hardware["tensorflow_devices"] = [
            f"{d.device_type}:{d.name}" for d in tf.config.list_physical_devices()
        ]
    except Exception:  # noqa: BLE001
        pass

    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "libraries": versions,
        "hardware": hardware,
        "git": _git_info(),
    }


def hash_files(paths: List[Path]) -> Dict[str, str]:
    """SHA-256 of each file (streamed)."""
    out: Dict[str, str] = {}
    for p in paths:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[p.name] = h.hexdigest()
    return out


# ---------------------------------------------------------------------------
# Run directory + manifest
# ---------------------------------------------------------------------------
def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def write_json(obj: Any, path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return p


class ExperimentManager:
    """
    Owns one pipeline run: identifier, directories, manifest and config snapshot.

    Parameters
    ----------
    cfg, settings, raw_config
        Result of :func:`load_phase2_config` (after any CLI overrides).
    run_id
        Explicit identifier. When omitted one is derived as
        ``<run_name>_<UTC timestamp>_<config hash>``. Re-using an existing
        ``run_id`` resumes that run directory (stages can be executed one by one).
    """

    def __init__(
        self,
        cfg: Phase2Config,
        settings: ExperimentSettings,
        raw_config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        output_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.cfg = cfg
        self.settings = settings
        self.raw_config = raw_config if raw_config is not None else cfg.to_dict()
        self.config_hash = config_hash(cfg, settings)
        self.run_id = run_id or self._new_run_id()
        root = Path(output_root or settings.output_root)
        self.run_dir = root / self.run_id
        for sub in RUN_SUBDIRS:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"
        self._manifest: Dict[str, Any] = self._load_manifest()

    # -- identifiers -------------------------------------------------------
    def _new_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{self.settings.run_name}_{stamp}_{self.config_hash}"

    # -- directories -------------------------------------------------------
    def path(self, sub: str) -> Path:
        if sub not in RUN_SUBDIRS:
            raise KeyError(f"Unknown run sub-directory '{sub}'. Valid: {RUN_SUBDIRS}")
        return self.run_dir / sub

    def scoped_config(self) -> Phase2Config:
        """
        Deep copy of the config whose output directories point inside the run.

        This is how Saksham's scaler persistence and Harsh's checkpoint / history /
        figure writers are redirected without touching their code.
        """
        scoped = Phase2Config.from_dict(copy.deepcopy(self.cfg.to_dict()))
        scoped.scaling.scalers_dir = str(self.path("scalers"))
        scoped.training.checkpoint_dir = str(self.path("checkpoints"))
        scoped.training.history_dir = str(self.path("history"))
        scoped.training.figures_dir = str(self.path("figures"))
        scoped.tuning.results_dir = str(self.path("evaluation"))
        return scoped

    # -- manifest ----------------------------------------------------------
    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.is_file():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_hash": self.config_hash,
            "stages": {},
        }

    def record_environment(self, seed_info: Dict[str, Any],
                           raw_data_dir: Optional[Union[str, Path]] = None) -> None:
        self._manifest["environment"] = capture_environment()
        self._manifest["seeding"] = seed_info
        if self.settings.hash_raw_data and raw_data_dir is not None:
            files = sorted(Path(raw_data_dir).glob("PRSA_Data_*.csv"))
            self._manifest["raw_data_sha256"] = hash_files(files)
        self.save_config_snapshot()
        self.save_manifest()

    def record_stage(self, stage: str, status: str, **info: Any) -> None:
        self._manifest["stages"][stage] = {
            "status": status,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            **info,
        }
        self.save_manifest()

    def stage_status(self, stage: str) -> Optional[str]:
        return self._manifest["stages"].get(stage, {}).get("status")

    @property
    def manifest(self) -> Dict[str, Any]:
        return self._manifest

    def save_manifest(self) -> Path:
        return write_json(self._manifest, self.manifest_path)

    def save_config_snapshot(self) -> Path:
        """Write the *effective* config (after CLI overrides) next to the run."""
        snap = copy.deepcopy(self.raw_config)
        snap.update(self.cfg.to_dict())
        snap["experiment"] = {
            "output_root": self.settings.output_root,
            "run_name": self.settings.run_name,
            "window_size": self.settings.window_size,
            "primary_station": self.settings.primary_station,
        }
        snap["evaluation"] = {
            "partition": self.settings.evaluation_partition,
            "batch_size": self.settings.evaluation_batch_size,
            "prediction_atol": self.settings.prediction_atol,
        }
        snap["reproducibility"] = {
            "deterministic_ops": self.settings.deterministic_ops,
            "hash_raw_data": self.settings.hash_raw_data,
        }
        p = self.run_dir / "config.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(snap, f, sort_keys=False, default_flow_style=False)
        return p


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "RUN_SUBDIRS",
    "ExperimentSettings",
    "ExperimentManager",
    "load_phase2_config",
    "config_hash",
    "seed_everything",
    "capture_environment",
    "hash_files",
    "write_json",
]
