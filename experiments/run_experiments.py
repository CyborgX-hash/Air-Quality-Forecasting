"""
Experiment Runner CLI for Phase 1 Air Quality Forecasting.

Allows researchers and engineers to execute configured experiments, compare
models across individual monitoring stations, or run an aggregated multi-station
benchmark sweep across all 12 Beijing monitoring stations.

Usage Examples:
    # 1. Run standard benchmark on primary station (Aotizhongxin):
    python experiments/run_experiments.py --station Aotizhongxin

    # 2. Run with hyperparameter grid search enabled:
    python experiments/run_experiments.py --station Aotizhongxin --tune-gb

    # 3. Sweep across all 12 stations and compile comparison summary:
    python experiments/run_experiments.py --all-stations

    # 4. Run from a custom configuration JSON file:
    python experiments/run_experiments.py --config experiments/configs/custom_experiment.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig, VALID_STATIONS
from src.pipeline import run_pipeline, export_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("experiments.run_experiments")


def run_single_station_experiment(
    station: str,
    config: Optional[PipelineConfig] = None,
    data_dir: str = "data/raw",
    output_dir: str = "reports/results",
    figures_dir: str = "reports/figures",
    generate_figures: bool = True,
    tune_gb: bool = False,
    random_seed: int = 42,
) -> None:
    """Run pipeline on a single specified station."""
    cfg = config or PipelineConfig.get_default_config()
    cfg.data.station = station
    cfg.data.data_dir = data_dir
    cfg.output.results_dir = output_dir
    cfg.output.figures_dir = figures_dir
    cfg.output.metadata_dir = output_dir
    cfg.output.generate_figures = generate_figures
    cfg.model.tune_gb_grid = tune_gb
    cfg.model.random_seed = random_seed
    cfg.experiment_name = f"phase1_experiment_{station}"

    logger.info("Starting experiment for station: %s", station)
    result = run_pipeline(config=cfg, save_artifacts=True)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT SUMMARY: {station}")
    print("=" * 70)
    print(f"Champion Model: {result.selected_model_name}")
    print("\nFinal Test Set Results:")
    print(result.test_comparison_df.to_string(index=False))
    print("=" * 70)


def run_all_stations_sweep(
    data_dir: str = "data/raw",
    output_dir: str = "reports/results/multi_station_sweep",
    figures_dir: str = "reports/figures/multi_station_sweep",
    tune_gb: bool = False,
    random_seed: int = 42,
) -> None:
    """Iterate through all 12 monitoring stations, execute pipeline, and synthesize benchmark."""
    logger.info("=" * 70)
    logger.info("STARTING 12-STATION BENCHMARK SWEEP")
    logger.info("Stations: %s", VALID_STATIONS)
    logger.info("=" * 70)

    sweep_results: List[Dict[str, Any]] = []
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for idx, station in enumerate(VALID_STATIONS, 1):
        logger.info("\n[%d/%d] Processing station: %s", idx, len(VALID_STATIONS), station)
        cfg = PipelineConfig.get_default_config()
        cfg.data.station = station
        cfg.data.data_dir = data_dir
        cfg.output.results_dir = str(out_path / station)
        cfg.output.figures_dir = str(Path(figures_dir) / station)
        cfg.output.metadata_dir = str(out_path / station)
        cfg.output.generate_figures = False  # Faster batch execution
        cfg.model.tune_gb_grid = tune_gb
        cfg.model.random_seed = random_seed
        cfg.experiment_name = f"sweep_{station}"

        try:
            res = run_pipeline(config=cfg, save_artifacts=True)
            for _, row in res.test_comparison_df.iterrows():
                sweep_results.append({
                    "station": station,
                    "model": row["model"],
                    "test_mae": row["mae"],
                    "test_rmse": row["rmse"],
                    "test_r2": row["r2"],
                    "test_mbe": row["mbe"],
                    "n_obs": row["n_obs"],
                    "mae_improvement_pct": row.get("mae_improvement_pct", 0.0),
                })
        except Exception as e:
            logger.error("Station %s failed: %s", station, e)

    if sweep_results:
        df_sweep = pd.DataFrame(sweep_results)
        sweep_csv, sweep_md = export_dataframe(df_sweep, "all_stations_model_comparison", out_path)
        logger.info("Saved 12-station benchmark table to %s and %s", sweep_csv, sweep_md)

        # Average across stations
        df_avg = df_sweep.groupby("model")[["test_mae", "test_rmse", "test_r2", "test_mbe", "mae_improvement_pct"]].mean().reset_index()
        df_avg = df_avg.sort_values(by="test_mae", ascending=True).reset_index(drop=True)
        avg_csv, avg_md = export_dataframe(df_avg, "all_stations_average_benchmark", out_path)
        logger.info("Saved 12-station average benchmark to %s and %s", avg_csv, avg_md)

        print("\n" + "=" * 70)
        print("ALL-STATION AVERAGE TEST PERFORMANCE:")
        print("=" * 70)
        print(df_avg.to_string(index=False))
        print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Air Quality Forecasting Phase 1 Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--station",
        type=str,
        default="Aotizhongxin",
        help=f"Target station name. Valid: {VALID_STATIONS}",
    )
    parser.add_argument(
        "--all-stations",
        action="store_true",
        help="Run batch benchmark sweep sequentially across all 12 stations",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory where station CSV files are stored",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file overriding default settings",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="reports/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure plotting",
    )
    parser.add_argument(
        "--tune-gb",
        action="store_true",
        help="Execute hyperparameter grid search on validation set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser


def main() -> None:
    """CLI entry point for running experiments."""
    parser = build_parser()
    args = parser.parse_args()

    cfg = None
    if args.config:
        cfg = PipelineConfig.from_json(args.config)

    if args.all_stations:
        run_all_stations_sweep(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
            tune_gb=args.tune_gb,
            random_seed=args.seed,
        )
    else:
        run_single_station_experiment(
            station=args.station,
            config=cfg,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
            generate_figures=not args.no_figures,
            tune_gb=args.tune_gb,
            random_seed=args.seed,
        )


if __name__ == "__main__":
    main()
