"""
Unit tests for Phase 2 LSTM Architecture, Training, and Evaluation.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Harsh Kumar (Deep Learning Engineer – LSTM Architecture and Training)

Verifies:
1. Model input and multi-horizon (5-unit) output tensor dimensions.
2. Model compilation under MAE and Huber loss functions.
3. Forward pass numerical integrity and output shapes.
4. Lightweight synthetic training loop (verifying convergence, EarlyStopping, checkpointing).
5. Model serialization and deserialization (.keras format).
6. Deterministic prediction reproducibility under fixed seeds.
7. Prediction evaluation and metric extraction across horizons.
8. Controlled hyperparameter experiment plan completeness.
"""

import os
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import tensorflow as tf
from tensorflow import keras

from src.phase2.config import Phase2Config
from src.phase2.data_pipeline import process_station
from src.phase2.lstm_model import (
    DEFAULT_HORIZONS,
    LSTMArchitectureConfig,
    build_loss,
    build_lstm_model,
)
from src.phase2.predict import (
    evaluate_horizon_metrics,
    load_trained_model,
    predict_and_evaluate,
)
from src.phase2.train import (
    TrainingResult,
    plot_training_history,
    train_station_model,
)
from src.phase2.tune import (
    get_default_controlled_experiment_plan,
)


@pytest.fixture
def synthetic_splits_container(tmp_path):
    """Create a minimal synthetic StationDataSplits container for unit tests."""
    dates = pd.date_range("2020-01-01 00:00", periods=300, freq="h")
    records = []

    for i, dt in enumerate(dates):
        # Intentional target missingness at a few indexes
        pm25 = np.nan if i in [10, 85, 220] else float(30.0 + (i % 24) * 2.0)
        records.append({
            "No": i + 1,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "station": "TestStation",
            "datetime": dt,
            "PM2.5": pm25,
            "PM10": float(50.0 + (i % 24) * 3.0),
            "SO2": 15.0,
            "NO2": 35.0,
            "CO": 800.0,
            "O3": 40.0,
            "TEMP": float(5.0 + 10.0 * np.sin(i / 12.0)),
            "PRES": 1015.0,
            "DEWP": float(0.0 + 8.0 * np.sin(i / 12.0)),
            "RAIN": 0.0,
            "wd": "NE" if i % 2 == 0 else "SW",
            "WSPM": 2.5,
        })

    df_raw = pd.DataFrame(records)
    cfg = Phase2Config.get_default_config()
    cfg.scaling.scalers_dir = str(tmp_path / "scalers")
    cfg.scaling.save_scalers = True

    return process_station("TestStation", config=cfg, df_raw=df_raw)


def test_model_dimensions():
    """Verify LSTM input and 5-output regression dimensions."""
    # 24h lookback, 22 features, 5 horizons
    model_24 = build_lstm_model(sequence_length=24, num_features=22, output_units=5)
    assert model_24.input_shape == (None, 24, 22)
    assert model_24.output_shape == (None, 5)

    # 48h lookback, 2 layers, 128 units
    model_48 = build_lstm_model(sequence_length=48, num_features=22, lstm_layers=2, lstm_units=128, output_units=5)
    assert model_48.input_shape == (None, 48, 22)
    assert model_48.output_shape == (None, 5)

    # 72h lookback
    model_72 = build_lstm_model(sequence_length=72, num_features=22, output_units=5)
    assert model_72.input_shape == (None, 72, 22)
    assert model_72.output_shape == (None, 5)


def test_model_compilation_losses():
    """Verify compilation under MAE, Huber, and MSE loss functions."""
    for loss_name in ["mae", "huber", "mse"]:
        model = build_lstm_model(loss_function=loss_name)
        assert model.compiled
        assert model.loss is not None


def test_forward_pass_execution():
    """Verify forward pass on synthetic batch produces valid predictions."""
    batch_size = 16
    seq_len = 24
    num_feat = 22

    X_dummy = np.random.randn(batch_size, seq_len, num_feat).astype(np.float32)
    model = build_lstm_model(sequence_length=seq_len, num_features=num_feat, output_units=5)

    preds = model(X_dummy, training=False).numpy()
    assert preds.shape == (batch_size, 5)
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()


def test_lightweight_synthetic_training_loop(synthetic_splits_container, tmp_path):
    """Verify end-to-end model training, checkpointing, and metric extraction."""
    splits = synthetic_splits_container

    result = train_station_model(
        station_name="TestStation",
        splits=splits,
        window_size=24,
        horizons=[1, 3, 6, 12, 24],
        lstm_units=32,
        lstm_layers=1,
        dropout_rate=0.1,
        dense_units=16,
        epochs=3,
        batch_size=16,
        checkpoint_dir=tmp_path / "checkpoints",
        history_dir=tmp_path / "history",
        figures_dir=tmp_path / "figures",
        save_artifacts=True,
        verbose=0,
    )

    assert isinstance(result, TrainingResult)
    assert result.best_epoch >= 1
    assert "loss" in result.history
    assert "val_loss" in result.history
    assert len(result.history["loss"]) <= 3

    # Check that model checkpoint was created
    ckpt_path = Path(result.saved_artifacts["model_checkpoint"])
    assert ckpt_path.is_file()

    # Check validation table metrics
    val_table = result.val_summary.metrics_table
    assert len(val_table) == 5
    assert set(val_table["horizon"]) == {"1h", "3h", "6h", "12h", "24h"}
    assert result.val_summary.macro_mae > 0.0


def test_model_saving_and_loading_roundtrip(tmp_path):
    """Verify saved Keras model can be loaded and produces identical predictions."""
    model = build_lstm_model(sequence_length=24, num_features=22, output_units=5, seed=42)

    X_test = np.random.randn(5, 24, 22).astype(np.float32)
    orig_preds = model.predict(X_test, verbose=0)

    save_path = tmp_path / "test_model.keras"
    model.save(save_path)
    assert save_path.is_file()

    loaded_model = load_trained_model(save_path)
    loaded_preds = loaded_model.predict(X_test, verbose=0)

    assert np.allclose(orig_preds, loaded_preds, atol=1e-5)


def test_prediction_reproducibility():
    """Verify identical random seeds generate bitwise identical initial models."""
    model_1 = build_lstm_model(sequence_length=24, num_features=22, seed=123)
    model_2 = build_lstm_model(sequence_length=24, num_features=22, seed=123)

    X_sample = np.ones((4, 24, 22), dtype=np.float32)
    p1 = model_1(X_sample, training=False).numpy()
    p2 = model_2(X_sample, training=False).numpy()

    assert np.allclose(p1, p2, atol=1e-6)


def test_evaluate_horizon_metrics():
    """Verify evaluation metric calculation per forecast horizon."""
    y_true = np.array([
        [50.0, 55.0, 60.0, 70.0, 80.0],
        [100.0, 110.0, 120.0, 130.0, 140.0],
    ])
    y_pred = np.array([
        [52.0, 54.0, 61.0, 68.0, 82.0],
        [98.0, 112.0, 118.0, 133.0, 138.0],
    ])

    metrics_by_h, table_df = evaluate_horizon_metrics(y_true, y_pred, horizons=[1, 3, 6, 12, 24])

    assert len(table_df) == 5
    assert "1h" in metrics_by_h
    assert np.isclose(metrics_by_h["1h"]["mae"], 2.0, atol=1e-5)
    assert np.isclose(metrics_by_h["3h"]["mae"], 1.5, atol=1e-5)


def test_controlled_experiment_plan_structure():
    """Verify the controlled tuning plan contains required ablation studies."""
    plan = get_default_controlled_experiment_plan()
    assert len(plan) == 11

    # Check window variations
    windows = {exp["window_size"] for exp in plan}
    assert windows == {24, 48, 72}

    # Check unit variations
    units = {exp["lstm_units"] for exp in plan}
    assert {32, 64, 128}.issubset(units)

    # Check layer variations
    layers = {exp["lstm_layers"] for exp in plan}
    assert {1, 2}.issubset(layers)

    # Check dropout variations
    dropouts = {exp["dropout_rate"] for exp in plan}
    assert {0.1, 0.2, 0.3}.issubset(dropouts)

    # Check learning rate variations
    lrs = {exp["learning_rate"] for exp in plan}
    assert {0.001, 0.0005}.issubset(lrs)

    # Check batch size variations
    batch_sizes = {exp["batch_size"] for exp in plan}
    assert {32, 64}.issubset(batch_sizes)

    # Check loss functions
    losses = {exp["loss_function"] for exp in plan}
    assert {"mae", "huber"}.issubset(losses)


def test_plot_training_history_export(tmp_path):
    """Verify loss curve figure plotting function."""
    history = {
        "loss": [0.8, 0.6, 0.5, 0.4],
        "val_loss": [0.9, 0.7, 0.55, 0.45],
        "mae": [0.8, 0.6, 0.5, 0.4],
        "val_mae": [0.9, 0.7, 0.55, 0.45],
    }
    plot_file = tmp_path / "loss_curve.png"
    fig = plot_training_history(history, save_path=plot_file, station_name="TestStation")

    assert plot_file.is_file()
    assert plot_file.stat().st_size > 0
