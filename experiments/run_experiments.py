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

import concurrent.futures
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig, VALID_STATIONS
from src.data.load_data import validate_all_stations_schema
from src.pipeline import run_pipeline, export_dataframe
from src.visualization import (
    plot_multi_station_mae_comparison,
    plot_multi_station_r2_comparison,
    plot_multi_station_metric_heatmap,
    plot_multi_station_improvement_pct,
)

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


def _run_single_station_worker(args_tuple: tuple) -> Dict[str, Any]:
    """Worker function for executing pipeline for a single station."""
    station, data_dir, out_path_str, figures_path_str, tune_gb, random_seed = args_tuple
    cfg = PipelineConfig.get_default_config()
    cfg.data.station = station
    cfg.data.data_dir = data_dir
    cfg.output.results_dir = str(Path(out_path_str) / station)
    cfg.output.figures_dir = str(Path(figures_path_str) / station)
    cfg.output.metadata_dir = str(Path(out_path_str) / station)
    cfg.output.generate_figures = False  # Faster station-level run; aggregate figures created at sweep level
    cfg.model.tune_gb_grid = tune_gb
    cfg.model.random_seed = random_seed
    cfg.experiment_name = f"sweep_{station}"

    res = run_pipeline(config=cfg, save_artifacts=True)

    rows = []
    for _, row in res.test_comparison_df.iterrows():
        rows.append({
            "station": station,
            "model": row["model"],
            "test_mae": float(row["mae"]),
            "test_rmse": float(row["rmse"]),
            "test_r2": float(row["r2"]),
            "test_mbe": float(row["mbe"]),
            "n_obs": int(row["n_obs"]),
            "mae_improvement_pct": float(row.get("mae_improvement_pct", 0.0)),
            "rmse_improvement_pct": float(row.get("rmse_improvement_pct", 0.0)),
            "r2_delta": float(row.get("r2_delta", 0.0)),
        })

    station_meta = {
        "station": station,
        "run_id": res.run_id,
        "champion_model": res.selected_model_name,
        "validation_mae": float(res.val_comparison_df.iloc[0]["mae"]),
        "train_observations": int(res.sample_counts.get("train", 0)),
        "val_observations": int(res.sample_counts.get("val", 0)),
        "test_observations": int(res.sample_counts.get("test", 0)),
        "total_records": int(res.df_raw_shape[0]),
        "missing_target_count": int(res.clean_report.get("target_missing_count", 0)),
        "train_start": str(res.split_report.get("train_start", "")),
        "train_end": str(res.split_report.get("train_end", "")),
        "val_start": str(res.split_report.get("val_start", "")),
        "val_end": str(res.split_report.get("val_end", "")),
        "test_start": str(res.split_report.get("test_start", "")),
        "test_end": str(res.split_report.get("test_end", "")),
    }

    return {
        "station": station,
        "rows": rows,
        "station_meta": station_meta,
        "champion_model": res.selected_model_name,
    }


def compute_aggregate_benchmarks(df_sweep: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mathematically well-defined aggregate metrics across stations.

    Calculates:
    - Macro-average: Unweighted arithmetic mean across stations
    - Micro-average: Observation-weighted mean across all evaluated samples
    - Inter-station standard deviation
    """
    records = []
    for model_name, grp in df_sweep.groupby("model"):
        n_stations = len(grp)
        total_obs = grp["n_obs"].sum()

        # Macro averages
        macro_mae = grp["test_mae"].mean()
        macro_rmse = grp["test_rmse"].mean()
        macro_r2 = grp["test_r2"].mean()
        macro_mbe = grp["test_mbe"].mean()
        macro_imprv = grp["mae_improvement_pct"].mean()

        # Micro (weighted) averages
        micro_mae = (grp["test_mae"] * grp["n_obs"]).sum() / total_obs
        micro_rmse = np.sqrt((grp["test_rmse"] ** 2 * grp["n_obs"]).sum() / total_obs)
        micro_r2 = (grp["test_r2"] * grp["n_obs"]).sum() / total_obs
        micro_mbe = (grp["test_mbe"] * grp["n_obs"]).sum() / total_obs

        # Standard deviations
        std_mae = grp["test_mae"].std()
        std_rmse = grp["test_rmse"].std()
        std_r2 = grp["test_r2"].std()

        records.append({
            "model": model_name,
            "macro_mae": round(macro_mae, 4),
            "macro_rmse": round(macro_rmse, 4),
            "macro_r2": round(macro_r2, 4),
            "macro_mbe": round(macro_mbe, 4),
            "macro_mae_improvement_pct": round(macro_imprv, 2),
            "micro_mae": round(micro_mae, 4),
            "pooled_rmse": round(micro_rmse, 4),
            "micro_rmse": round(micro_rmse, 4),
            "micro_r2": round(micro_r2, 4),
            "micro_mbe": round(micro_mbe, 4),
            "std_mae": round(std_mae, 4),
            "std_rmse": round(std_rmse, 4),
            "std_r2": round(std_r2, 4),
            "total_observations": int(total_obs),
            "station_count": int(n_stations),
        })

    df_agg = pd.DataFrame(records).sort_values(by="macro_mae", ascending=True).reset_index(drop=True)
    return df_agg


def run_all_stations_sweep(
    data_dir: str = "data/raw",
    output_dir: str = "reports/results/multi_station_sweep",
    figures_dir: str = "reports/figures/multi_station_sweep",
    tune_gb: bool = False,
    random_seed: int = 42,
    n_jobs: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterate through all 12 monitoring stations, execute pipeline, and synthesize benchmark.

    Validates schema, preserves station identity and isolation, computes station-wise
    and aggregate metrics, and exports comparison reports and figures.
    """
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("STARTING 12-STATION BENCHMARK SWEEP")
    logger.info("Stations: %s", VALID_STATIONS)
    logger.info("=" * 70)

    # 1. Validate all station files up front
    logger.info("Step 1/4: Validating all 12 station CSV files and schemas in '%s'...", data_dir)
    audit = validate_all_stations_schema(data_dir=data_dir, expected_stations=VALID_STATIONS)
    if not audit["all_valid"]:
        missing = audit["stations_missing"]
        logger.error("Dataset validation failed! Missing stations: %s", missing)
        raise RuntimeError(f"Missing required station CSVs: {missing}")
    logger.info("Dataset validation PASSED: all %d stations verified.", len(audit["stations_found"]))

    out_path = Path(output_dir)
    fig_path = Path(figures_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    # 2. Execute pipeline per station
    logger.info("Step 2/4: Executing Phase 1 pipeline across %d stations (n_jobs=%d)...", len(VALID_STATIONS), n_jobs)
    worker_args = [
        (st, data_dir, str(out_path), str(fig_path), tune_gb, random_seed)
        for st in VALID_STATIONS
    ]

    sweep_results: List[Dict[str, Any]] = []
    station_meta_list: List[Dict[str, Any]] = []
    champion_records: List[Dict[str, Any]] = []

    if n_jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_station = {
                executor.submit(_run_single_station_worker, arg): arg[0]
                for arg in worker_args
            }
            for future in concurrent.futures.as_completed(future_to_station):
                st_name = future_to_station[future]
                try:
                    res_dict = future.result()
                    sweep_results.extend(res_dict["rows"])
                    station_meta_list.append(res_dict["station_meta"])
                    champ_name = res_dict["champion_model"]
                    for r in res_dict["rows"]:
                        if r["model"] == champ_name:
                            champion_records.append(r)
                    logger.info("Completed station: %s (Champion: %s)", st_name, champ_name)
                except Exception as exc:
                    logger.exception("Station '%s' execution failed: %s", st_name, exc)
                    raise
    else:
        for arg in worker_args:
            st_name = arg[0]
            logger.info("Processing station sequentially: %s", st_name)
            res_dict = _run_single_station_worker(arg)
            sweep_results.extend(res_dict["rows"])
            station_meta_list.append(res_dict["station_meta"])
            champ_name = res_dict["champion_model"]
            for r in res_dict["rows"]:
                if r["model"] == champ_name:
                    champion_records.append(r)
            logger.info("Completed station: %s (Champion: %s)", st_name, champ_name)

    # 3. Export Reports
    logger.info("Step 3/4: Generating structured station-wise comparison reports...")
    df_sweep = pd.DataFrame(sweep_results)
    # Ensure consistent station and model ordering
    df_sweep["station_order"] = df_sweep["station"].map(lambda s: VALID_STATIONS.index(s) if s in VALID_STATIONS else 99)
    df_sweep = df_sweep.sort_values(by=["station_order", "test_mae"]).drop(columns=["station_order"]).reset_index(drop=True)

    export_dataframe(df_sweep, "all_stations_model_comparison", out_path)

    # Station Metadata Summary
    df_station_meta = pd.DataFrame(station_meta_list)
    df_station_meta["station_order"] = df_station_meta["station"].map(lambda s: VALID_STATIONS.index(s) if s in VALID_STATIONS else 99)
    df_station_meta = df_station_meta.sort_values(by="station_order").drop(columns=["station_order"]).reset_index(drop=True)
    export_dataframe(df_station_meta, "station_split_and_data_summary", out_path)

    # Champion Models by Station
    df_champions = pd.DataFrame(champion_records)
    df_champions["station_order"] = df_champions["station"].map(lambda s: VALID_STATIONS.index(s) if s in VALID_STATIONS else 99)
    df_champions = df_champions.sort_values(by="station_order").drop(columns=["station_order"]).reset_index(drop=True)
    export_dataframe(df_champions, "champion_models_by_station", out_path)

    # Aggregate Model Performance
    df_summary = compute_aggregate_benchmarks(df_sweep)
    export_dataframe(df_summary, "all_stations_summary_by_model", out_path)

    # 4. Generate Aggregate Visualizations
    logger.info("Step 4/4: Generating multi-station publication figures...")
    plot_multi_station_mae_comparison(df_sweep, save_path=fig_path / "01_station_mae_comparison.png")
    plot_multi_station_r2_comparison(df_sweep, save_path=fig_path / "02_station_r2_comparison.png")
    plot_multi_station_improvement_pct(df_sweep, save_path=fig_path / "03_station_improvement_pct.png")
    plot_multi_station_metric_heatmap(df_sweep, metric="test_mae", save_path=fig_path / "04_station_performance_heatmap.png")

    total_time = round(time.time() - t_start, 2)
    logger.info("12-station benchmark sweep completed successfully in %.2fs.", total_time)

    # Save comprehensive metadata
    sweep_metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_seconds": total_time,
        "stations_evaluated": VALID_STATIONS,
        "validation_audit": {
            "all_valid": audit["all_valid"],
            "num_stations": audit["num_stations"],
        },
        "station_summaries": station_meta_list,
        "aggregate_model_performance": df_summary.to_dict(orient="records"),
    }
    with open(out_path / "multi_station_sweep_metadata.json", "w", encoding="utf-8") as f:
        json.dump(sweep_metadata, f, indent=2)

    print("\n" + "=" * 75)
    print("ALL-STATION AGGREGATE TEST PERFORMANCE (12 BEIJING MONITORING SITES):")
    print("=" * 75)
    display_cols = ["model", "macro_mae", "micro_mae", "macro_rmse", "macro_r2", "macro_mae_improvement_pct"]
    print(df_summary[display_cols].to_string(index=False))
    print("=" * 75)

    return df_sweep, df_summary


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
        help="Run batch benchmark sweep sequentially or in parallel across all 12 stations",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of concurrent workers for multi-station sweep",
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
        sweep_out = Path(args.output_dir)
        sweep_fig = Path(args.figures_dir)
        # If default reports/results is passed, route to multi_station_sweep subdirectory
        if sweep_out == Path("reports/results"):
            sweep_out = sweep_out / "multi_station_sweep"
        if sweep_fig == Path("reports/figures"):
            sweep_fig = sweep_fig / "multi_station_sweep"

        run_all_stations_sweep(
            data_dir=args.data_dir,
            output_dir=str(sweep_out),
            figures_dir=str(sweep_fig),
            tune_gb=args.tune_gb,
            random_seed=args.seed,
            n_jobs=args.n_jobs,
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
