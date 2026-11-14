"""
Phase 2 pipeline orchestrator (command-line entry point).

Author: Priyanshu Agrahari (Integration Engineer, Testing and Reproducibility)

Usage
-----
    python -m src.phase2.run_pipeline --config configs/phase2_config.yaml
    python -m src.phase2.run_pipeline --stages prepare,sequences --run-id demo
    python -m src.phase2.run_pipeline --stages evaluate,visualize,report --run-id demo

Stages (executed in this order; any subset may be selected)::

    prepare    load -> clean -> features -> chronological split -> train-only scaling
    sequences  multi-horizon sliding windows + sequence audit
    train      per-station LSTM training (validation-based checkpointing only)
    predict    test-set prediction + inverse scaling -> predictions/*.npz
    evaluate   Mihika's multi-station evaluator + integrity checks
    visualize  Mihika's figure suite (+ loss curve)
    report     Markdown report generated from the files above

Every module is called through its existing public interface. The only
"adaptation" is :func:`evaluator_bound_to_run`, which feeds the run's own
checkpoints and station splits into ``evaluate_all_stations`` (whose helper
hard-codes default paths) *without* modifying ``src/phase2/evaluate.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.phase2 import integrity
from src.phase2.experiment_manager import (
    DEFAULT_CONFIG_PATH,
    ExperimentManager,
    ExperimentSettings,
    load_phase2_config,
    seed_everything,
    write_json,
)

logger = logging.getLogger("phase2.pipeline")

STAGES: Sequence[str] = (
    "prepare", "sequences", "train", "predict", "evaluate", "visualize", "report",
)


# ---------------------------------------------------------------------------
# Adapter around the evaluator's hard-coded defaults
# ---------------------------------------------------------------------------
@contextmanager
def evaluator_bound_to_run(splits_by_station: Dict[str, Any],
                           model_paths: Dict[str, Path]) -> Iterator[None]:
    """
    Temporarily bind ``src.phase2.evaluate`` to this run's splits and checkpoints.

    ``evaluate_single_station`` resolves its checkpoint from a fixed ``u64_l1``
    path and re-runs ``process_station(station)`` with the default config.
    Wrapping those two names lets us reuse the evaluator unchanged (identical
    metrics, aggregation and error analysis) for any run configuration.
    """
    import src.phase2.evaluate as ev

    orig_single, orig_process = ev.evaluate_single_station, ev.process_station

    def _single(station_name: str, model_path: Optional[Path] = None, **kwargs: Any):
        return orig_single(station_name, model_path=model_paths[station_name], **kwargs)

    ev.evaluate_single_station = _single
    ev.process_station = lambda station, *a, **k: splits_by_station[station]
    try:
        yield
    finally:
        ev.evaluate_single_station, ev.process_station = orig_single, orig_process


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class Phase2Pipeline:
    """Runs the stages against one :class:`ExperimentManager` run directory."""

    def __init__(
        self,
        manager: ExperimentManager,
        stations: Optional[Sequence[str]] = None,
        data_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        epochs: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        self.mgr = manager
        self.cfg = manager.scoped_config()  # outputs redirected into the run dir
        self.settings: ExperimentSettings = manager.settings
        all_stations = list(self.cfg.stations.all)
        self.stations = list(stations) if stations else all_stations
        unknown = sorted(set(self.stations) - set(all_stations))
        if unknown:
            raise ValueError(f"Unknown stations {unknown}; valid: {all_stations}")
        self.subset_run = set(self.stations) != set(all_stations)
        self.data_dir = data_dir or self.cfg.data.raw_dir
        self.window = int(self.settings.window_size or self.cfg.lstm_windows.default_sequence_length)
        self.horizons: List[int] = list(self.cfg.forecast_horizons.horizons)
        self.epochs = int(epochs) if epochs is not None else self.cfg.training.epochs
        self.verbose = verbose
        self.primary = self.settings.primary_station or self.cfg.stations.default
        if self.primary not in self.stations:
            self.primary = self.stations[0]
        self.model_tag_suffix = f"u{self.cfg.model.lstm_units}_l{self.cfg.model.lstm_layers}"
        self.ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else self.mgr.path("checkpoints")
        # in-process caches (stages can also be run in separate invocations)
        self._splits: Optional[Dict[str, Any]] = None
        self._bundles: Optional[Dict[str, Any]] = None
        self._suite: Optional[Any] = None
        self._training_rows: List[Dict[str, Any]] = []

    # -- helpers -----------------------------------------------------------
    def checkpoint_path(self, station: str) -> Path:
        return self.ckpt_dir / f"lstm_{station}_w{self.window}_{self.model_tag_suffix}_best.keras"

    def splits(self) -> Dict[str, Any]:
        if self._splits is None:
            from src.phase2.data_pipeline import process_station
            self._splits = {}
            for st in self.stations:
                self._splits[st] = process_station(st, config=self.cfg, data_dir=self.data_dir)
        return self._splits

    def bundles(self) -> Dict[str, Any]:
        if self._bundles is None:
            from src.phase2.sequence_generator import generate_station_sequence_bundle
            self._bundles = {
                st: generate_station_sequence_bundle(
                    splits=sp, window_size=self.window, horizons=self.horizons,
                    scaled=True, scale_target=self.cfg.scaling.scale_target)
                for st, sp in self.splits().items()
            }
        return self._bundles

    def _read_training_rows(self) -> List[Dict[str, Any]]:
        if self._training_rows:
            return self._training_rows
        csv = self.mgr.path("history") / "training_summary.csv"
        if csv.is_file():
            self._training_rows = pd.read_csv(csv).to_dict(orient="records")
        return self._training_rows

    # -- stages ------------------------------------------------------------
    def stage_prepare(self) -> Dict[str, Any]:
        splits = self.splits()
        rows = []
        for st, s in splits.items():
            rows.append({
                "station": st,
                "train_rows": len(s.train_df), "val_rows": len(s.val_df), "test_rows": len(s.test_df),
                "train_start": s.boundaries["train_start"], "train_end": s.boundaries["train_end"],
                "val_start": s.boundaries["val_start"], "val_end": s.boundaries["val_end"],
                "test_start": s.boundaries["test_start"], "test_end": s.boundaries["test_end"],
                "n_features": len(s.feature_cols),
            })
        pd.DataFrame(rows).to_csv(self.mgr.path("audit") / "data_split_audit.csv", index=False)
        checks = [
            integrity.check_stations_accounted(self.cfg.stations.all, splits, self.subset_run),
            integrity.check_chronological_split(splits),
            integrity.check_scaler_train_only(splits),
            integrity.check_inverse_transform_roundtrip(splits),
            integrity.check_feature_order_consistent(splits, self.cfg.get_feature_list()),
        ]
        write_json(integrity.summarize(checks), self.mgr.path("audit") / "integrity_data.json")
        return {"stations": len(splits), "integrity": integrity.summarize(checks)["all_ok"]}

    def stage_sequences(self) -> Dict[str, Any]:
        bundles = self.bundles()
        rows = []
        for st, b in bundles.items():
            b.validate()
            for part in ("train", "val", "test"):
                ds = getattr(b, part)
                row = {"station": st, "partition": part, "window": self.window,
                       "X_shape": list(ds.X.shape), "y_shape": list(ds.y.shape)}
                if ds.exclusion_report is not None:
                    row.update(ds.exclusion_report.to_dict())
                rows.append(row)
        write_json(rows, self.mgr.path("audit") / "sequence_audit.json")
        checks = [
            integrity.check_no_cross_station(bundles),
            integrity.check_sequence_target_bounds(bundles, self.splits()),
        ]
        write_json(integrity.summarize(checks), self.mgr.path("audit") / "integrity_sequences.json")
        return {"stations": len(bundles), "integrity": integrity.summarize(checks)["all_ok"]}

    def stage_train(self) -> Dict[str, Any]:
        from src.phase2.train import train_station_model

        seed = self.cfg.project.random_seed
        c = self.cfg
        rows: List[Dict[str, Any]] = []
        for st in self.stations:
            res = train_station_model(
                station_name=st, config=c, splits=self.splits()[st],
                window_size=self.window, horizons=self.horizons,
                lstm_units=c.model.lstm_units, lstm_layers=c.model.lstm_layers,
                dropout_rate=c.model.dropout_rate, dense_units=c.model.dense_units,
                learning_rate=c.model.learning_rate, loss_function=c.model.loss_function,
                batch_size=c.training.batch_size, epochs=self.epochs,
                patience_early_stopping=c.training.early_stopping_patience,
                patience_reduce_lr=c.training.reduce_lr_patience, seed=seed,
                checkpoint_dir=c.training.checkpoint_dir, history_dir=c.training.history_dir,
                figures_dir=c.training.figures_dir, save_artifacts=c.training.save_artifacts,
                verbose=self.verbose,
            )
            rows.append({
                "station": st, "window": self.window, "best_epoch": res.best_epoch,
                "best_val_loss": res.best_val_loss, "macro_val_mae": res.val_summary.macro_mae,
                "macro_val_rmse": res.val_summary.macro_rmse, "macro_val_r2": res.val_summary.macro_r2,
                "selection_partition": res.val_summary.partition,
                "train_time_s": res.training_time_seconds, "seed": seed,
                "checkpoint": res.saved_artifacts.get("model_checkpoint", ""),
            })
        pd.DataFrame(rows).to_csv(self.mgr.path("history") / "training_summary.csv", index=False)
        self._training_rows = rows
        return {"stations_trained": len(rows), "epochs_max": self.epochs}

    def stage_predict(self) -> Dict[str, Any]:
        from src.phase2.predict import load_trained_model, predict_and_evaluate

        part = self.settings.evaluation_partition
        out = self.mgr.path("predictions")
        written = []
        for st in self.stations:
            model = load_trained_model(self.checkpoint_path(st))
            summ = predict_and_evaluate(
                model=model, splits=self.splits()[st], partition=part,
                window_size=self.window, horizons=self.horizons, use_context=True,
                batch_size=self.settings.evaluation_batch_size,
            )
            f = out / f"{st}_w{self.window}_{part}.npz"
            np.savez_compressed(
                f, y_true=summ.y_true, y_pred=summ.y_pred,
                timestamps=np.asarray(summ.timestamps.values, dtype="datetime64[ns]").astype("int64"),
                horizons=np.asarray(self.horizons),
            )
            summ.metrics_table.assign(station=st).to_csv(out / f"{st}_w{self.window}_{part}_metrics.csv", index=False)
            written.append(str(f))
        return {"prediction_files": len(written), "partition": part}

    def _load_saved_predictions(self) -> Dict[str, Dict[str, np.ndarray]]:
        part = self.settings.evaluation_partition
        saved = {}
        for st in self.stations:
            f = self.mgr.path("predictions") / f"{st}_w{self.window}_{part}.npz"
            if f.is_file():
                with np.load(f) as z:
                    saved[st] = {"y_true": z["y_true"], "y_pred": z["y_pred"]}
        return saved

    def suite(self) -> Any:
        if self._suite is None:
            from src.phase2.evaluate import evaluate_all_stations

            paths = {st: self.checkpoint_path(st) for st in self.stations}
            missing = [str(p) for p in paths.values() if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing model checkpoints (run the 'train' stage): {missing}")
            with evaluator_bound_to_run(self.splits(), paths):
                self._suite = evaluate_all_stations(
                    stations=self.stations, window_size=self.window,
                    horizons=self.horizons, partition=self.settings.evaluation_partition,
                )
        return self._suite

    def stage_evaluate(self) -> Dict[str, Any]:
        from src.phase2.evaluate import save_evaluation_artifacts

        suite = self.suite()
        files = save_evaluation_artifacts(suite, output_dir=self.mgr.path("evaluation"))
        splits = self.splits()
        checks = [
            integrity.check_stations_accounted(self.cfg.stations.all, suite.station_records, self.subset_run),
            integrity.check_horizons_evaluated(suite, self.horizons),
            integrity.check_chronological_split(splits),
            integrity.check_scaler_train_only(splits),
            integrity.check_inverse_transform_roundtrip(splits),
            integrity.check_feature_order_consistent(splits, self.cfg.get_feature_list()),
            integrity.check_no_cross_station(self.bundles()),
            integrity.check_sequence_target_bounds(self.bundles(), splits),
            integrity.check_ground_truth_observed(suite, splits),
            integrity.check_metrics_follow_phase1(suite),
            integrity.check_macro_pooled_consistent(suite),
            integrity.check_selection_on_validation_only(self._read_training_rows()),
            integrity.check_predictions_match_saved(
                suite, self._load_saved_predictions(), self.settings.prediction_atol),
        ]
        summary = integrity.summarize(checks)
        write_json(summary, self.mgr.run_dir / "integrity_checks.json")
        return {"artifacts": len(files), "integrity_all_ok": summary["all_ok"],
                "integrity_failed": summary["failed"]}

    def stage_visualize(self) -> Dict[str, Any]:
        from src.phase2.visualize import (
            generate_all_evaluation_figures,
            plot_training_and_validation_loss,
        )
        import matplotlib.pyplot as plt

        fig_dir = self.mgr.path("figures")
        figs = generate_all_evaluation_figures(
            self.suite(), figures_dir=fig_dir, primary_station=self.primary)
        hist = self.mgr.path("history") / (
            f"lstm_{self.primary}_w{self.window}_{self.model_tag_suffix}_history.json")
        if hist.is_file():
            p = fig_dir / "03_training_vs_validation_loss.png"
            plot_training_and_validation_loss(hist, station_name=self.primary, save_path=p)
            figs["loss_curve"] = str(p)
        plt.close("all")
        return {"figures": len(figs)}

    def stage_report(self) -> Dict[str, Any]:
        path = write_report(self.mgr, self.window, self.horizons, self.stations)
        return {"report": str(path)}

    # -- driver ------------------------------------------------------------
    def run(self, stages: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for stage in [s for s in STAGES if s in stages]:
            logger.info("=== stage: %s ===", stage)
            t0 = time.time()
            try:
                info = getattr(self, f"stage_{stage}")()
            except Exception as exc:
                self.mgr.record_stage(stage, "failed", error=repr(exc))
                logger.exception("Stage '%s' failed", stage)
                raise
            info["seconds"] = round(time.time() - t0, 2)
            self.mgr.record_stage(stage, "ok", **info)
            results[stage] = info
        return results


# ---------------------------------------------------------------------------
# Report (generated only from files that earlier stages wrote)
# ---------------------------------------------------------------------------
def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_report(mgr: ExperimentManager, window: int, horizons: Sequence[int],
                 stations: Sequence[str]) -> Path:
    ev = mgr.path("evaluation")
    man = mgr.manifest
    lines = [f"# Phase 2 Run Report: `{mgr.run_id}`", ""]
    env = man.get("environment", {})
    lines += [
        "## Run identity",
        f"- Config hash: `{man.get('config_hash')}`  |  git commit: `{(env.get('git') or {}).get('commit')}`"
        f" (dirty: {(env.get('git') or {}).get('dirty')})",
        f"- Window: {window} h  |  horizons: {list(horizons)}  |  stations: {len(stations)}",
        f"- Seed: {(man.get('seeding') or {}).get('seed')}  |  Python {env.get('python')}  |  "
        f"TensorFlow {(env.get('libraries') or {}).get('tensorflow')}",
        "",
    ]
    stage_rows = [{"stage": k, "status": v.get("status"), "seconds": v.get("seconds")}
                  for k, v in man.get("stages", {}).items()]
    if stage_rows:
        lines += ["## Stages", _md(pd.DataFrame(stage_rows)), ""]

    tr = mgr.path("history") / "training_summary.csv"
    if tr.is_file():
        lines += ["## Training (validation-based selection)", _md(pd.read_csv(tr).drop(columns=["checkpoint"], errors="ignore")), ""]

    for title, name in (("Macro-average by horizon (mean of station metrics)", "macro_average_performance_by_horizon.csv"),
                        ("Pooled by horizon (pooled observations)", "pooled_performance_by_horizon.csv")):
        f = ev / name
        if f.is_file():
            lines += [f"## {title}", _md(pd.read_csv(f)), ""]
    f = ev / "all_stations_multi_horizon_comparison.csv"
    if f.is_file():
        d = pd.read_csv(f)
        cols = ["station", "horizon", "lstm_mae", "lstm_rmse", "lstm_r2", "lstm_mbe",
                "pers_mae", "mae_improvement_pct", "n_obs"]
        lines += ["## Station-wise results", _md(d[cols]), ""]

    ic = mgr.run_dir / "integrity_checks.json"
    if ic.is_file():
        import json
        s = json.loads(ic.read_text())
        lines += [f"## Integrity checks ({s['passed']} passed, {s['failed']} failed, {s['skipped']} skipped)",
                  _md(pd.DataFrame(s["checks"])), ""]
    figs = sorted(p.name for p in mgr.path("figures").glob("*.png"))
    if figs:
        lines += ["## Figures"] + [f"- `figures/{n}`" for n in figs] + [""]
    lines += ["_All numbers above were read from the run's output files; none are hand-written._", ""]
    out = mgr.run_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.phase2.run_pipeline",
        description="Run the integrated Phase 2 LSTM forecasting pipeline.")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Phase 2 YAML config")
    p.add_argument("--stages", default="all",
                   help=f"comma list from {','.join(STAGES)} or 'all' (default)")
    p.add_argument("--stations", default=None, help="comma list; default = all stations in config")
    p.add_argument("--window-size", type=int, default=None, help="override experiment window (hours)")
    p.add_argument("--epochs", type=int, default=None, help="override max epochs (smoke tests)")
    p.add_argument("--run-id", default=None, help="reuse/resume a run directory")
    p.add_argument("--output-root", default=None)
    p.add_argument("--raw-dir", default=None, help="override data.raw_dir")
    p.add_argument("--checkpoint-dir", default=None, help="evaluate pre-trained checkpoints from here")
    p.add_argument("--seed", type=int, default=None, help="override project.random_seed")
    p.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2], help="Keras verbosity")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg, settings, raw = load_phase2_config(args.config)
    if args.seed is not None:
        cfg.project.random_seed = args.seed
    if args.window_size is not None:
        settings.window_size = args.window_size
    if args.raw_dir:
        cfg.data.raw_dir = args.raw_dir

    stages = list(STAGES) if args.stages == "all" else [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"Unknown stage(s) {bad}. Valid: {list(STAGES)}", file=sys.stderr)
        return 2

    mgr = ExperimentManager(cfg, settings, raw, run_id=args.run_id, output_root=args.output_root)
    seed_info = seed_everything(cfg.project.random_seed, settings.deterministic_ops)
    mgr.record_environment(seed_info, raw_data_dir=cfg.data.raw_dir)
    logger.info("Run id: %s  (dir: %s)", mgr.run_id, mgr.run_dir)

    pipe = Phase2Pipeline(
        mgr, stations=args.stations.split(",") if args.stations else None,
        data_dir=cfg.data.raw_dir, checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs, verbose=args.verbose)
    pipe.run(stages)

    ic = mgr.run_dir / "integrity_checks.json"
    if ic.is_file():
        import json
        if not json.loads(ic.read_text())["all_ok"]:
            logger.error("Integrity checks FAILED; see %s", ic)
            return 1
    logger.info("Done. Outputs in %s", mgr.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
