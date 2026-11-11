"""
Controlled Hyperparameter Tuning & Experimentation Suite for Phase 2 LSTM.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Harsh Kumar (Deep Learning Engineer – LSTM Architecture and Training)

Key Design Principles:
1. Systematic, Non-Blind Search: Rather than running a combinatorial Cartesian grid,
   we implement a controlled ablation plan starting from a baseline and systematically
   varying one dimension at a time:
   - Sequence length: 24, 48, 72 hours.
   - LSTM units: 32, 64, 128.
   - LSTM layers: 1 and 2.
   - Dropout: 0.1, 0.2, 0.3.
   - Learning rate: 0.001 and 0.0005.
   - Batch size: 32 and 64.
   - Loss function: MAE vs Huber loss.
2. Strict Validation-Only Selection: Champion configurations are selected EXCLUSIVELY
   using Validation set MAE. Test data is never touched during hyperparameter tuning.
3. Automated Experiment Logging: Generates structured Markdown and CSV comparison tables
   and persists the winning configuration to configs/best_hyperparameters.json.
"""

from __future__ import annotations

import os
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from src.phase2.config import Phase2Config
from src.phase2.data_pipeline import StationDataSplits, process_station
from src.phase2.lstm_model import DEFAULT_HORIZONS
from src.phase2.train import TrainingResult, train_station_model

logger = logging.getLogger(__name__)


def get_default_controlled_experiment_plan() -> List[Dict[str, Any]]:
    """
    Return controlled, structured hyperparameter experiments across 5 systematic phases.

    Returns
    -------
    List[Dict[str, Any]]
        List of experiment parameter dictionaries.
    """
    return [
        # 1. Baseline Anchor
        {
            "experiment_id": "EXP01_Baseline_W24_U64_L1",
            "phase": "Baseline",
            "description": "Baseline anchor: W=24h, 1 layer, 64 units, dropout=0.2, lr=0.001, bs=32, MAE",
            "window_size": 24,
            "lstm_units": 64,
            "lstm_layers": 1,
            "dropout_rate": 0.2,
            "dense_units": 32,
            "learning_rate": 0.001,
            "batch_size": 32,
            "loss_function": "mae",
        },
        # 2. Sequence Length Context Sensitivity (24 vs 48 vs 72)
        {
            "experiment_id": "EXP02_Window_W48",
            "phase": "Lookback Window",
            "description": "Extend temporal context to 48 hours",
            "window_size": 48,
            "lstm_units": 64,
            "lstm_layers": 1,
            "dropout_rate": 0.2,
            "dense_units": 32,
            "learning_rate": 0.001,
            "batch_size": 32,
            "loss_function": "mae",
        },
        {
            "experiment_id": "EXP03_Window_W72",
            "phase": "Lookback Window",
            "description": "Extend temporal context to 72 hours (3 full diurnal cycles)",
            "window_size": 72,
            "lstm_units": 64,
            "lstm_layers": 1,
            "dropout_rate": 0.2,
            "dense_units": 32,
            "learning_rate": 0.001,
            "batch_size": 32,
            "loss_function": "mae",
        },
        # 3. Recurrent Layer Capacity & Stacking (32 vs 64 vs 128 units, 1 vs 2 layers)
        {
            "experiment_id": "EXP04_Units_U32",
            "phase": "Model Capacity",
            "description": "Reduced capacity: 32 hidden units",
            "window_size": 24,
            "lstm_units": 32,
            "lstm_layers": 1,
            "dropout_rate": 0.2,
            "dense_units": 16,
            "learning_rate": 0.001,
            "batch_size": 32,
            "loss_function": "mae",
        },
        {
            "experiment_id": "EXP05_Units_U128",
            "phase": "Model Capacity",
            "description": "Increased capacity: 128 hidden units",
            "window_size": 24,
            "lstm_units": 128,
            "lstm_layers": 1,
            "dropout_rate": 0.2,
            "dense_units": 32,
            "learning_rate": 0.001,
            "batch_size": 32,
            "loss_function": "mae",
        },
        {
            "experiment_id": "EXP06_Stacked_2Layers",
            "phase": "Network Depth",
            "description": "Stacked deep recurrent architecture: 2 LSTM layers (64 units each)",
            "window_size": 24,
            "lstm_units": 64,
            "lstm_layers": 2,
            "dropout_rate": 0.2,
            "dense_units": 32,
            "learning_rate": 0.001,
            "batch_size": 32,
            "loss_function": "mae",
        },
        # 4. Regularization & Optimization Dynamics
        {
            "experiment_id": "EXP07_Dropout_0.1",
            "phase": "Regularization",
            "description": "Lower dropout rate (0.1)",
            "window_size": 24,
            "lstm_units": 64,
            "lstm_layers": 1,
            "dropout_rate": 0.1,
            "dense_units": 32,
            "learning_rate": 0.001,
            "batch_size": 32,
            "loss_function": "mae",
        },
        {
            "experiment_id": "EXP08_Dropout_0.3",
            "phase": "Regularization",
            "description": "Higher dropout rate (0.3)",
            "window_size": 24,
            "lstm_units": 64,
            "lstm_layers": 1,
            "dropout_rate": 0.3,
            "dense_units": 32,
            "learning_rate": 0.001,
            "batch_size": 32,
            "loss_function": "mae",
        },
        {
            "experiment_id": "EXP09_LR_0.0005",
            "phase": "Learning Rate",
            "description": "Conservative learning rate (0.0005)",
            "window_size": 24,
            "lstm_units": 64,
            "lstm_layers": 1,
            "dropout_rate": 0.2,
            "dense_units": 32,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "loss_function": "mae",
        },
        {
            "experiment_id": "EXP10_BatchSize_64",
            "phase": "Batch Size",
            "description": "Larger mini-batch size (64)",
            "window_size": 24,
            "lstm_units": 64,
            "lstm_layers": 1,
            "dropout_rate": 0.2,
            "dense_units": 32,
            "learning_rate": 0.001,
            "batch_size": 64,
            "loss_function": "mae",
        },
        # 5. Loss Function Formulation (MAE vs Huber Loss)
        {
            "experiment_id": "EXP11_Loss_Huber",
            "phase": "Loss Formulation",
            "description": "Robust Huber loss objective (delta=1.0) against acute pollution spikes",
            "window_size": 24,
            "lstm_units": 64,
            "lstm_layers": 1,
            "dropout_rate": 0.2,
            "dense_units": 32,
            "learning_rate": 0.001,
            "batch_size": 32,
            "loss_function": "huber",
        },
    ]


def run_hyperparameter_experiments(
    station_name: str = "Aotizhongxin",
    experiment_plan: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Phase2Config] = None,
    epochs: int = 25,
    patience: int = 6,
    horizons: Optional[Sequence[int]] = None,
    results_dir: Optional[Union[str, Path]] = "reports/results/phase2",
    seed: int = 42,
    verbose: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute controlled hyperparameter search on the Validation partition.

    Parameters
    ----------
    station_name : str, default 'Aotizhongxin'
        Benchmark monitoring station to tune.
    experiment_plan : Optional[List[Dict[str, Any]]], default None
        List of experiment parameter configs.
    config : Optional[Phase2Config], default None
        Master configuration.
    epochs : int, default 25
        Epochs per experiment.
    patience : int, default 6
        Early stopping patience.
    horizons : Optional[Sequence[int]], default [1, 3, 6, 12, 24]
        Forecast horizons.
    results_dir : Optional[Union[str, Path]], default 'reports/results/phase2'
        Directory to export comparison tables.
    seed : int, default 42
        Random seed.
    verbose : int, default 1
        Verbosity.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - Ranked comparison DataFrame sorted by Validation Macro MAE.
        - Best hyperparameter configuration dictionary.
    """
    cfg = config or Phase2Config.get_default_config()
    active_horizons = list(horizons) if horizons is not None else DEFAULT_HORIZONS
    plan = experiment_plan or get_default_controlled_experiment_plan()
    out_dir = Path(results_dir or "reports/results/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 75)
    logger.info("STARTING PHASE 2 CONTROLLED HYPERPARAMETER SEARCH")
    logger.info("Station: %s | Total Experiments: %d | Target: %s", station_name, len(plan), active_horizons)
    logger.info("=" * 75)

    # Process station once so datasets are ready
    splits = process_station(station_name, config=cfg)

    comparison_rows: List[Dict[str, Any]] = []
    best_result: Optional[TrainingResult] = None
    best_config: Optional[Dict[str, Any]] = None
    min_macro_val_mae = float("inf")

    total_tuning_start = time.time()

    for idx, exp in enumerate(plan, 1):
        exp_id = exp["experiment_id"]
        logger.info("\n[%d/%d] Running Experiment: %s (%s)", idx, len(plan), exp_id, exp.get("description", ""))

        t_exp = time.time()
        result = train_station_model(
            station_name=station_name,
            config=cfg,
            splits=splits,
            window_size=exp["window_size"],
            horizons=active_horizons,
            lstm_units=exp["lstm_units"],
            lstm_layers=exp["lstm_layers"],
            dropout_rate=exp["dropout_rate"],
            dense_units=exp.get("dense_units", 32),
            learning_rate=exp["learning_rate"],
            loss_function=exp.get("loss_function", "mae"),
            batch_size=exp["batch_size"],
            epochs=epochs,
            patience_early_stopping=patience,
            patience_reduce_lr=max(2, patience // 2),
            seed=seed,
            checkpoint_dir=out_dir / "tuning_checkpoints",
            history_dir=out_dir / "tuning_history",
            figures_dir=out_dir / "tuning_figures",
            save_artifacts=True,
            verbose=verbose,
        )
        exp_duration = round(time.time() - t_exp, 2)

        val_summary = result.val_summary
        macro_mae = val_summary.macro_mae
        macro_rmse = val_summary.macro_rmse
        macro_r2 = val_summary.macro_r2

        # Extract per-horizon MAEs
        row = {
            "rank": 0,  # Populated after sorting
            "experiment_id": exp_id,
            "phase": exp.get("phase", "Exploration"),
            "window_size": exp["window_size"],
            "lstm_units": exp["lstm_units"],
            "lstm_layers": exp["lstm_layers"],
            "dropout_rate": exp["dropout_rate"],
            "learning_rate": exp["learning_rate"],
            "batch_size": exp["batch_size"],
            "loss_function": exp.get("loss_function", "mae"),
            "best_epoch": result.best_epoch,
            "val_loss": round(result.best_val_loss, 4),
            "macro_val_mae": round(macro_mae, 2),
            "macro_val_rmse": round(macro_rmse, 2),
            "macro_val_r2": round(macro_r2, 4),
            "duration_s": exp_duration,
        }

        for _, h_row in val_summary.metrics_table.iterrows():
            row[f"mae_{h_row['horizon']}"] = round(h_row["mae"], 2)

        comparison_rows.append(row)

        if macro_mae < min_macro_val_mae:
            min_macro_val_mae = macro_mae
            best_result = result
            best_config = exp.copy()
            logger.info(">>> NEW CHAMPION! %s achieves Macro Val MAE: %.2f µg/m³", exp_id, macro_mae)

    # Sort results strictly by Macro Validation MAE
    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="macro_val_mae").reset_index(drop=True)
    comparison_df["rank"] = range(1, len(comparison_df) + 1)

    total_time = round(time.time() - total_tuning_start, 2)
    logger.info("\n" + "=" * 75)
    logger.info("HYPERPARAMETER SEARCH COMPLETED IN %.2fs", total_time)
    logger.info("Champion Configuration: %s (Val MAE: %.2f µg/m³)", best_config["experiment_id"], min_macro_val_mae)
    logger.info("=" * 75)
    logger.info("\n%s\n", comparison_df.to_string(index=False))

    # Export comparison artifacts
    csv_path = out_dir / "hyperparameter_tuning_comparison.csv"
    md_path = out_dir / "hyperparameter_tuning_comparison.md"
    comparison_df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2 LSTM Hyperparameter Tuning & Ablation Study\n\n")
        f.write(f"**Station:** {station_name} | **Horizons:** {active_horizons} | **Total Runtime:** {total_time}s\n\n")
        f.write(f"**Selection Criterion:** Lowest Macro Validation MAE (Strict Validation-Only Selection)\n\n")
        f.write(comparison_df.to_markdown(index=False) + "\n")

    # Export winning configuration JSON
    best_config_path = out_dir / "best_hyperparameters.json"
    winning_meta = {
        "station_name": station_name,
        "selected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "champion_experiment": best_config["experiment_id"],
        "selection_metric": "macro_val_mae",
        "macro_val_mae": min_macro_val_mae,
        "parameters": best_config,
    }
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(winning_meta, f, indent=2)

    logger.info("Exported tuning tables to %s and %s", csv_path, md_path)
    logger.info("Saved champion hyperparameter configuration to %s", best_config_path)

    return comparison_df, winning_meta


__all__ = [
    "get_default_controlled_experiment_plan",
    "run_hyperparameter_experiments",
]

