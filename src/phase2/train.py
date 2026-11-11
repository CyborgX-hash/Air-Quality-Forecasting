"""
Model Training and Checkpointing Pipeline for Phase 2 LSTM Forecasting.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Harsh Kumar (Deep Learning Engineer – LSTM Architecture and Training)

Responsibilities:
1. Orchestrate end-to-end station-specific LSTM training.
2. Monitor validation loss with EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.
3. Enforce deterministic random seeds for reproducible weight initialization.
4. Persist best model checkpoints (.keras), training history (JSON/CSV), and loss plots.
5. Record comprehensive experiment metadata for reproducibility and Mihika's evaluation suite.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

from src.phase2.config import Phase2Config
from src.phase2.data_pipeline import StationDataSplits, process_station
from src.phase2.sequence_generator import (
    DEFAULT_HORIZONS,
    StationSequenceBundle,
    generate_station_sequence_bundle,
)
from src.phase2.lstm_model import (
    LSTMArchitectureConfig,
    build_lstm_model,
)
from src.phase2.predict import EvaluationSummary, predict_and_evaluate

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Encapsulates the outputs, model, history, and metadata of a completed training run."""
    station_name: str
    model: keras.Model
    history: Dict[str, List[float]]
    best_epoch: int
    best_val_loss: float
    val_summary: EvaluationSummary
    saved_artifacts: Dict[str, str]
    training_time_seconds: float
    config: Dict[str, Any]
    metadata: Dict[str, Any]


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    station_name: str = "Aotizhongxin",
    loss_name: str = "MAE",
) -> plt.Figure:
    """
    Plot publication-quality training and validation loss curves.

    Parameters
    ----------
    history : Dict[str, List[float]]
        Keras training history dictionary.
    save_path : Optional[Union[str, Path]], default None
        File path to save the generated figure.
    station_name : str, default 'Aotizhongxin'
        Name of the station for plot title.
    loss_name : str, default 'MAE'
        Name of the loss objective.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # 1. Loss Curve
    ax1.plot(epochs, history["loss"], label=f"Train {loss_name}", color="#1f77b4", lw=2)
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], label=f"Val {loss_name}", color="#ff7f0e", lw=2, linestyle="--")
        best_epoch = int(np.argmin(history["val_loss"])) + 1
        best_val = history["val_loss"][best_epoch - 1]
        ax1.scatter([best_epoch], [best_val], color="red", s=80, zorder=5, label=f"Best Epoch ({best_epoch})")

    ax1.set_title(f"Loss Convergence — {station_name}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel(f"Loss ({loss_name})", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Metric Curve (MAE or MSE)
    metric_key = "mae" if "mae" in history else "loss"
    val_metric_key = f"val_{metric_key}"
    ax2.plot(epochs, history[metric_key], label="Train MAE", color="#2ca02c", lw=2)
    if val_metric_key in history:
        ax2.plot(epochs, history[val_metric_key], label="Val MAE", color="#d62728", lw=2, linestyle="--")

    ax2.set_title(f"Validation MAE Trajectory — {station_name}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("MAE (Standardized)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        logger.debug("Saved training history plot to %s", p)

    return fig


def train_station_model(
    station_name: str,
    config: Optional[Phase2Config] = None,
    splits: Optional[StationDataSplits] = None,
    window_size: int = 24,
    horizons: Optional[Sequence[int]] = None,
    lstm_units: int = 64,
    lstm_layers: int = 1,
    dropout_rate: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 0.001,
    loss_function: str = "mae",
    batch_size: int = 32,
    epochs: int = 50,
    patience_early_stopping: int = 10,
    patience_reduce_lr: int = 5,
    seed: int = 42,
    checkpoint_dir: Optional[Union[str, Path]] = "models/phase2/checkpoints",
    history_dir: Optional[Union[str, Path]] = "reports/results/phase2/history",
    figures_dir: Optional[Union[str, Path]] = "reports/figures/phase2",
    save_artifacts: bool = True,
    verbose: int = 1,
) -> TrainingResult:
    """
    Train, validate, and checkpoint a station-isolated LSTM model.

    Parameters
    ----------
    station_name : str
        Monitoring station identifier.
    config : Optional[Phase2Config], default None
        Master Phase 2 configuration.
    splits : Optional[StationDataSplits], default None
        Preprocessed station splits container. If None, process_station() is called.
    window_size : int, default 24
        Input lookback window W.
    horizons : Optional[Sequence[int]], default [1, 3, 6, 12, 24]
        Forecast horizons to predict simultaneously.
    lstm_units : int, default 64
        Number of hidden units in LSTM layers.
    lstm_layers : int, default 1
        Number of LSTM recurrent layers (1 or 2).
    dropout_rate : float, default 0.2
        Dropout probability.
    dense_units : int, default 32
        Dense projection layer units.
    learning_rate : float, default 0.001
        Adam optimizer learning rate.
    loss_function : str, default 'mae'
        Loss objective: 'mae' or 'huber'.
    batch_size : int, default 32
        Training batch size.
    epochs : int, default 50
        Maximum training epochs.
    patience_early_stopping : int, default 10
        Epochs without validation improvement before stopping.
    patience_reduce_lr : int, default 5
        Epochs before halving the learning rate on plateau.
    seed : int, default 42
        Reproducible random seed.
    checkpoint_dir : Optional[Union[str, Path]], default 'models/phase2/checkpoints'
        Directory for saving best .keras model checkpoints.
    history_dir : Optional[Union[str, Path]], default 'reports/results/phase2/history'
        Directory for saving training history JSON and metrics tables.
    figures_dir : Optional[Union[str, Path]], default 'reports/figures/phase2'
        Directory for saving loss curves.
    save_artifacts : bool, default True
        Whether to export artifacts to disk.
    verbose : int, default 1
        Keras training verbosity (0=silent, 1=progress, 2=one line per epoch).

    Returns
    -------
    TrainingResult
        Container holding trained model, history, validation metrics, and paths.
    """
    t_start = time.time()
    cfg = config or Phase2Config.get_default_config()
    active_horizons = list(horizons) if horizons is not None else DEFAULT_HORIZONS
    output_units = len(active_horizons)

    # 1. Enforce deterministic seeds
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    # 2. Acquire station data splits
    station_splits = splits or process_station(station_name, config=cfg)
    num_features = len(station_splits.feature_cols)

    logger.info("=" * 70)
    logger.info("TRAINING LSTM MODEL: Station '%s' | W=%dh | H=%s", station_name, window_size, active_horizons)
    logger.info("Topology: %dL, %dU, Dropout=%.1f | Loss: %s | LR: %.4f", lstm_layers, lstm_units, dropout_rate, loss_function, learning_rate)
    logger.info("=" * 70)

    # 3. Extract training and validation sequences using validated sequence generator
    bundle = generate_station_sequence_bundle(
        splits=station_splits,
        window_size=window_size,
        horizons=active_horizons,
        scaled=True,
        scale_target=True,
    )
    X_train, y_train = bundle.X_train, bundle.y_train
    X_val, y_val = bundle.X_val, bundle.y_val

    logger.info("Extracted sequences via StationSequenceBundle: Train=%s, Val=%s", X_train.shape, X_val.shape)

    # 4. Build and compile model
    arch_cfg = LSTMArchitectureConfig(
        sequence_length=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        lstm_layers=lstm_layers,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        output_units=output_units,
        loss_function=loss_function,
        learning_rate=learning_rate,
        horizons=active_horizons,
    )

    model = build_lstm_model(config=arch_cfg, seed=seed)

    # 5. Configure callbacks
    saved_artifacts: Dict[str, str] = {}
    callback_list: List[callbacks.Callback] = []

    # Prepare directories
    ckpt_dir_path = Path(checkpoint_dir or "models/phase2/checkpoints")
    hist_dir_path = Path(history_dir or "reports/results/phase2/history")
    fig_dir_path = Path(figures_dir or "reports/figures/phase2")

    if save_artifacts:
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)
        hist_dir_path.mkdir(parents=True, exist_ok=True)
        fig_dir_path.mkdir(parents=True, exist_ok=True)

    model_tag = f"lstm_{station_name}_w{window_size}_u{lstm_units}_l{lstm_layers}"
    best_checkpoint_file = ckpt_dir_path / f"{model_tag}_best.keras"

    # Early stopping (restoring best weights)
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience_early_stopping,
        restore_best_weights=True,
        verbose=1 if verbose else 0,
    )
    callback_list.append(es)

    # Checkpoint callback
    if save_artifacts:
        mc = callbacks.ModelCheckpoint(
            filepath=str(best_checkpoint_file),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        )
        callback_list.append(mc)

    # Learning rate reduction on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=patience_reduce_lr,
        min_lr=1e-5,
        verbose=1 if verbose else 0,
    )
    callback_list.append(reduce_lr)

    # 6. Model Training
    history_obj = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=verbose,
    )

    raw_history: Dict[str, List[float]] = {
        k: [float(val) for val in v]
        for k, v in history_obj.history.items()
    }

    best_epoch = int(np.argmin(raw_history["val_loss"])) + 1
    best_val_loss = float(raw_history["val_loss"][best_epoch - 1])
    training_time = round(time.time() - t_start, 2)

    logger.info("Training completed in %.2fs. Best Epoch: %d (Val Loss: %.4f)", training_time, best_epoch, best_val_loss)

    # If checkpointing was enabled, reload the saved best checkpoint to guarantee exact model weights
    if save_artifacts and best_checkpoint_file.is_file():
        model = keras.models.load_model(str(best_checkpoint_file))
        saved_artifacts["model_checkpoint"] = str(best_checkpoint_file)

    # 7. Evaluate on validation set
    val_summary = predict_and_evaluate(
        model=model,
        splits=station_splits,
        partition="val",
        window_size=window_size,
        horizons=active_horizons,
        use_context=True,
        batch_size=batch_size,
    )

    logger.info("Validation Performance (µg/m³):\n%s", val_summary.metrics_table.to_string(index=False))
    logger.info("Macro Validation MAE across horizons: %.2f µg/m³", val_summary.macro_mae)

    # 8. Save artifacts and metadata
    metadata = {
        "station_name": station_name,
        "model_tag": model_tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_time_seconds": training_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "macro_val_mae": val_summary.macro_mae,
        "macro_val_rmse": val_summary.macro_rmse,
        "macro_val_r2": val_summary.macro_r2,
        "random_seed": seed,
        "architecture": arch_cfg.to_dict(),
        "training_parameters": {
            "batch_size": batch_size,
            "max_epochs": epochs,
            "early_stopping_patience": patience_early_stopping,
            "reduce_lr_patience": patience_reduce_lr,
        },
        "horizons": active_horizons,
        "feature_columns": station_splits.feature_cols,
        "scaler_summary": station_splits.scaler.get_params_summary(),
        "val_metrics_table": val_summary.metrics_table.to_dict(orient="records"),
    }

    if save_artifacts:
        # History JSON
        hist_json_path = hist_dir_path / f"{model_tag}_history.json"
        with open(hist_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_history, f, indent=2)
        saved_artifacts["history_json"] = str(hist_json_path)

        # History CSV
        hist_df = pd.DataFrame(raw_history)
        hist_df.insert(0, "epoch", range(1, len(hist_df) + 1))
        hist_csv_path = hist_dir_path / f"{model_tag}_history.csv"
        hist_df.to_csv(hist_csv_path, index=False)
        saved_artifacts["history_csv"] = str(hist_csv_path)

        # Metadata JSON
        meta_json_path = hist_dir_path / f"{model_tag}_metadata.json"
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        saved_artifacts["metadata_json"] = str(meta_json_path)

        # Validation Metrics CSV & Markdown
        val_csv_path = hist_dir_path / f"{model_tag}_val_metrics.csv"
        val_summary.metrics_table.to_csv(val_csv_path, index=False)
        saved_artifacts["val_metrics_csv"] = str(val_csv_path)

        # Loss curve plot
        fig_path = fig_dir_path / f"{model_tag}_loss_curve.png"
        plot_training_history(
            history=raw_history,
            save_path=fig_path,
            station_name=station_name,
            loss_name=loss_function.upper(),
        )
        plt.close("all")
        saved_artifacts["loss_plot"] = str(fig_path)

    return TrainingResult(
        station_name=station_name,
        model=model,
        history=raw_history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        val_summary=val_summary,
        saved_artifacts=saved_artifacts,
        training_time_seconds=training_time,
        config=arch_cfg.to_dict(),
        metadata=metadata,
    )


def train_all_stations(
    stations: Optional[Sequence[str]] = None,
    config: Optional[Phase2Config] = None,
    window_size: int = 24,
    horizons: Optional[Sequence[int]] = None,
    lstm_units: int = 64,
    lstm_layers: int = 1,
    dropout_rate: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 1,
) -> Dict[str, TrainingResult]:
    """
    Train separate station-isolated LSTM models across all monitoring stations.

    Parameters
    ----------
    stations : Optional[Sequence[str]], default None
        List of station names. If None, trains across all 12 stations.
    config : Optional[Phase2Config], default None
        Master configuration.
    window_size : int, default 24
        Input lookback window W.
    horizons : Optional[Sequence[int]], default [1, 3, 6, 12, 24]
        Forecast horizons.
    lstm_units : int, default 64
        Units in LSTM layers.
    lstm_layers : int, default 1
        Number of LSTM layers.
    dropout_rate : float, default 0.2
        Dropout probability.
    epochs : int, default 50
        Maximum training epochs.
    batch_size : int, default 32
        Batch size.
    verbose : int, default 1
        Training verbosity.

    Returns
    -------
    Dict[str, TrainingResult]
        Dictionary mapping station name to its TrainingResult.
    """
    cfg = config or Phase2Config.get_default_config()
    target_stations = list(stations) if stations is not None else cfg.stations.all
    results: Dict[str, TrainingResult] = {}

    logger.info("Initiating LSTM training across %d stations: %s", len(target_stations), target_stations)

    for st in target_stations:
        res = train_station_model(
            station_name=st,
            config=cfg,
            window_size=window_size,
            horizons=horizons,
            lstm_units=lstm_units,
            lstm_layers=lstm_layers,
            dropout_rate=dropout_rate,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        results[st] = res

    # Generate summary benchmark table
    rows = []
    for st, res in results.items():
        row = {"station": st, "macro_val_mae": res.val_summary.macro_mae}
        for _, h_row in res.val_summary.metrics_table.iterrows():
            row[f"mae_{h_row['horizon']}"] = h_row["mae"]
        row["best_epoch"] = res.best_epoch
        row["train_time_s"] = res.training_time_seconds
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    logger.info("\n=== MULTI-STATION VALIDATION BENCHMARK ===\n%s", summary_df.to_string(index=False))

    return results


__all__ = [
    "TrainingResult",
    "plot_training_history",
    "train_station_model",
    "train_all_stations",
]
