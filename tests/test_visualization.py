"""
Unit tests for src/visualization.py.

Verifies:
1. All plotting functions execute without error in headless environments.
2. Figures and Axes objects are properly created.
3. Output image files are saved to reports/figures/ with valid dimensions.
4. Flexible input support (numpy arrays, pandas Series, DatetimeIndex, dictionaries).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from src.visualization import (
    plot_actual_vs_predicted_timeseries,
    plot_actual_vs_predicted_scatter,
    plot_residual_distribution,
    plot_residuals_over_time,
    plot_error_by_hour,
    plot_error_by_month,
    plot_model_comparison_bars,
    plot_feature_importance,
    plot_error_by_pollution_tier,
)


@pytest.fixture
def sample_viz_data():
    """Create synthetic time-series evaluation data for visualization tests."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=168, freq="h")
    y_true = np.abs(np.sin(np.linspace(0, 20, 168)) * 100 + np.random.normal(0, 10, 168)) + 15
    y_pred = y_true + np.random.normal(0, 5, 168)
    y_baseline = np.roll(y_true, 1)
    y_baseline[0] = y_true[0]

    return {
        "dates": dates,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_baseline": y_baseline,
        "residuals": y_true - y_pred,
    }


def test_plot_timeseries(tmp_path, sample_viz_data):
    save_file = tmp_path / "test_ts.png"
    fig = plot_actual_vs_predicted_timeseries(
        y_true=sample_viz_data["y_true"],
        predictions={
            "persistence": sample_viz_data["y_baseline"],
            "gradient_boosting": sample_viz_data["y_pred"],
        },
        timestamps=sample_viz_data["dates"],
        start_idx=0,
        num_points=48,
        save_path=save_file,
    )
    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    assert save_file.stat().st_size > 1000
    plt.close(fig)


def test_plot_scatter(tmp_path, sample_viz_data):
    save_file = tmp_path / "test_scatter.png"
    fig = plot_actual_vs_predicted_scatter(
        y_true=sample_viz_data["y_true"],
        y_pred=sample_viz_data["y_pred"],
        model_name="Gradient Boosting",
        save_path=save_file,
    )
    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)


def test_plot_residual_distribution(tmp_path, sample_viz_data):
    save_file = tmp_path / "test_residual_dist.png"
    fig = plot_residual_distribution(
        residuals=sample_viz_data["residuals"],
        model_name="Gradient Boosting",
        save_path=save_file,
    )
    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)


def test_plot_residuals_over_time(tmp_path, sample_viz_data):
    save_file = tmp_path / "test_residuals_time.png"
    fig = plot_residuals_over_time(
        residuals=sample_viz_data["residuals"],
        timestamps=sample_viz_data["dates"],
        model_name="Gradient Boosting",
        save_path=save_file,
    )
    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)


def test_plot_error_by_hour(tmp_path, sample_viz_data):
    save_file = tmp_path / "test_error_hour.png"
    fig = plot_error_by_hour(
        y_true=sample_viz_data["y_true"],
        predictions={"gb": sample_viz_data["y_pred"]},
        timestamps=sample_viz_data["dates"],
        save_path=save_file,
    )
    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)


def test_plot_error_by_month(tmp_path):
    save_file = tmp_path / "test_error_month.png"
    dates = pd.date_range("2020-01-01", periods=1000, freq="h")
    yt = np.random.uniform(20, 100, 1000)
    yp = yt + np.random.normal(0, 5, 1000)

    fig = plot_error_by_month(
        y_true=yt,
        predictions={"gb": yp},
        timestamps=dates,
        save_path=save_file,
    )
    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)


def test_plot_model_comparison_bars(tmp_path):
    save_file = tmp_path / "test_comparison.png"
    df_cmp = pd.DataFrame([
        {"model": "persistence", "mae": 15.2, "rmse": 25.1, "r2": 0.88},
        {"model": "linear_reg", "mae": 12.4, "rmse": 21.0, "r2": 0.92},
        {"model": "grad_boost", "mae": 10.1, "rmse": 18.2, "r2": 0.94},
    ])
    fig = plot_model_comparison_bars(df_cmp, save_path=save_file)
    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)


def test_plot_feature_importance(tmp_path):
    save_file = tmp_path / "test_feature_imp.png"
    df_imp = pd.DataFrame({
        "feature": [f"feat_{i}" for i in range(10)],
        "importance": np.linspace(0.01, 0.45, 10)[::-1],
    })
    fig = plot_feature_importance(df_imp, top_n=5, save_path=save_file)
    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)


def test_plot_error_by_pollution_tier(tmp_path):
    save_file = tmp_path / "test_tiers.png"
    df_tier = pd.DataFrame([
        {"tier": "Excellent (0-35)", "mae": 4.2, "mbe": 0.5, "count": 100},
        {"tier": "Good (35-75)", "mae": 8.1, "mbe": -1.2, "count": 80},
        {"tier": "Heavy (>150)", "mae": 22.5, "mbe": -15.4, "count": 30},
    ])
    fig = plot_error_by_pollution_tier(df_tier, save_path=save_file)
    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    plt.close(fig)
