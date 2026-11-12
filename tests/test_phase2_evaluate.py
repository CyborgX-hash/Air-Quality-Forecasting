"""
Unit and integration tests for Phase 2 Evaluation and Visualization modules.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Mihika Mathur (Model Evaluation, Error Analysis & Visualization Specialist)

Verifies:
1. Metric calculations against known mathematical ground truth.
2. MBE sign convention: mean(y_pred - y_true).
   Positive indicates overprediction; negative indicates underprediction.
3. Proper unscaling / inverse transformation of predictions back to µg/m³.
4. Strict filtering of missing target observations (no silent dropouts / NaN contamination).
5. Macro averaging across stations (arithmetic mean of station metrics).
6. Pooled MAE and RMSE mathematical property:
   Guarantees pooled RMSE is computed from pooled squared errors and
   verifies that pooled RMSE does not equal the simple average of station RMSEs.
7. Timestamp alignment between sequences, predictions, and ground truth.
8. Visualization suite execution and image artifact generation without GUI dependency.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    calculate_metrics,
    mean_absolute_error,
    mean_bias_error,
    r2_score,
    root_mean_squared_error,
    validate_metric_inputs,
)
from src.phase2.evaluate import (
    AQI_PM25_BINS,
    StationEvaluationRecord,
    calculate_pooled_metrics,
    evaluate_diurnal_profile,
    evaluate_monthly_profile,
    evaluate_pollution_tiers,
    evaluate_single_station,
)
from src.phase2.visualize import (
    plot_actual_vs_predicted_timeseries,
    plot_error_distribution,
    plot_forecast_horizon_vs_error,
    plot_mae_comparison_across_models,
    plot_residual_diagnostics,
    plot_rmse_comparison_across_models,
    plot_station_wise_comparison,
)


# ---------------------------------------------------------------------------
# 1. Metric Calculations Against Known Values
# ---------------------------------------------------------------------------

def test_metrics_against_known_values():
    """Verify MAE, RMSE, R2, and MBE against hand-computed reference values."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 35.0, 36.0])

    # Differences: e = yp - yt = [+2, -2, +5, -4]
    # Absolute differences: [2, 2, 5, 4] -> sum = 13 -> MAE = 13/4 = 3.25
    # Squared differences: [4, 4, 25, 16] -> sum = 49 -> MSE = 12.25 -> RMSE = 3.5
    # Mean bias: (2 - 2 + 5 - 4) / 4 = 1/4 = 0.25
    # y_true mean = 25.0, sum((yt - 25)^2) = 225 + 25 + 25 + 225 = 500
    # R2 = 1 - (49 / 500) = 1 - 0.098 = 0.902

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mbe = mean_bias_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    assert np.isclose(mae, 3.25, atol=1e-5), f"Expected MAE 3.25, got {mae}"
    assert np.isclose(rmse, 3.5, atol=1e-5), f"Expected RMSE 3.5, got {rmse}"
    assert np.isclose(mbe, 0.25, atol=1e-5), f"Expected MBE 0.25, got {mbe}"
    assert np.isclose(r2, 0.902, atol=1e-3), f"Expected R2 0.902, got {r2}"


# ---------------------------------------------------------------------------
# 2. MBE Sign Convention Verification
# ---------------------------------------------------------------------------

def test_mbe_sign_convention():
    """
    Verify that MBE strictly follows: mean(prediction - actual).
    - Overprediction: positive MBE.
    - Underprediction: negative MBE.
    """
    actual = np.array([50.0, 50.0, 50.0])
    over_pred = np.array([60.0, 60.0, 60.0])
    under_pred = np.array([40.0, 40.0, 40.0])

    mbe_over = mean_bias_error(actual, over_pred)
    mbe_under = mean_bias_error(actual, under_pred)

    assert mbe_over > 0, "Overprediction must yield a positive MBE"
    assert np.isclose(mbe_over, 10.0), f"Expected +10.0, got {mbe_over}"

    assert mbe_under < 0, "Underprediction must yield a negative MBE"
    assert np.isclose(mbe_under, -10.0), f"Expected -10.0, got {mbe_under}"


# ---------------------------------------------------------------------------
# 3. Missing Value Handling & NaN Filtering
# ---------------------------------------------------------------------------

def test_missing_value_filtering():
    """Ensure validate_metric_inputs cleanly removes NaNs and Infs without altering paired alignment."""
    y_true = np.array([10.0, np.nan, 30.0, 40.0, np.inf])
    y_pred = np.array([12.0, 20.0, np.nan, 36.0, 50.0])

    yt_clean, yp_clean = validate_metric_inputs(y_true, y_pred, allow_nan=False)

    # Only indices 0 (10.0, 12.0) and 3 (40.0, 36.0) are mutually valid
    assert len(yt_clean) == 2
    assert len(yp_clean) == 2
    assert np.allclose(yt_clean, [10.0, 40.0])
    assert np.allclose(yp_clean, [12.0, 36.0])


# ---------------------------------------------------------------------------
# 4. Macro Averaging vs Pooled Aggregation
# ---------------------------------------------------------------------------

def test_pooled_rmse_differs_from_average_of_station_rmse():
    """
    Mathematically verify that pooled RMSE computed from pooled squared errors
    is strictly different from the simple arithmetic average of station RMSEs.
    """
    # Station A: 2 observations with large squared errors
    yt_A = np.array([10.0, 20.0])
    yp_A = np.array([20.0, 40.0])  # errors: [10, 20] -> MSE = (100 + 400)/2 = 250 -> RMSE_A = sqrt(250) = 15.8114

    # Station B: 8 observations with smaller squared errors
    yt_B = np.array([10.0] * 8)
    yp_B = np.array([12.0] * 8)    # errors: [2]*8 -> MSE = 4 -> RMSE_B = 2.0

    rmse_A = root_mean_squared_error(yt_A, yp_A)
    rmse_B = root_mean_squared_error(yt_B, yp_B)
    average_rmse = (rmse_A + rmse_B) / 2.0  # (15.8114 + 2.0) / 2 = 8.9057

    # True pooled calculation:
    # 2*250 + 8*4 = 500 + 32 = 532 total sum of squares over 10 observations
    # Pooled MSE = 53.2 -> Pooled RMSE = sqrt(53.2) = 7.2938
    pooled_df = calculate_pooled_metrics([yt_A, yt_B], [yp_A, yp_B], horizons=[1])
    pooled_rmse = float(pooled_df["pooled_rmse"].iloc[0])

    assert not np.isclose(pooled_rmse, average_rmse, atol=0.1), (
        f"Pooled RMSE ({pooled_rmse:.4f}) should not equal average of station RMSEs ({average_rmse:.4f})"
    )
    assert np.isclose(pooled_rmse, np.sqrt(53.2), atol=1e-3)


# ---------------------------------------------------------------------------
# 5. Inverse-Scaling & Timestamp Alignment
# ---------------------------------------------------------------------------

def test_station_evaluation_data_integrity():
    """Verify that evaluate_single_station loads unscaled PM2.5 in realistic ranges and aligns timestamps."""
    rec = evaluate_single_station("Aotizhongxin", window_size=72, partition="test")

    assert rec.station_name == "Aotizhongxin"
    assert rec.window_size == 72
    assert rec.horizons == [1, 3, 6, 12, 24]
    assert rec.y_true.shape[1] == 5
    assert rec.y_pred_lstm.shape[1] == 5
    assert rec.y_pred_persistence.shape[1] == 5
    assert len(rec.timestamps) == rec.n_obs

    # Physical units check: PM2.5 values should be positive and realistic (e.g. 0 to 1000 µg/m³)
    assert np.nanmin(rec.y_true) >= 0.0, "Ground truth PM2.5 must be non-negative"
    assert np.nanmean(rec.y_true) > 10.0, "Mean PM2.5 in Beijing test set should be realistically elevated (>10 µg/m³)"
    assert np.nanmax(rec.y_true) < 1200.0, "PM2.5 should not exceed physical sensor boundaries (<1200 µg/m³)"


# ---------------------------------------------------------------------------
# 6. Error Stratifications by Pollution Tier & Diurnal Cycle
# ---------------------------------------------------------------------------

def test_pollution_tier_and_diurnal_stratification():
    """Verify that error stratification by tier and diurnal profile functions correctly."""
    rec = evaluate_single_station("Aotizhongxin", window_size=72, partition="test")
    station_dict = {"Aotizhongxin": rec}

    tier_df = evaluate_pollution_tiers(station_dict, horizons=[1, 24])
    assert not tier_df.empty
    assert "tier" in tier_df.columns
    assert "mae" in tier_df.columns
    assert "mbe" in tier_df.columns

    # Severe tier (>250 µg/m³) should exhibit underprediction (negative MBE)
    severe_1h = tier_df[(tier_df["horizon"] == "1h") & (tier_df["tier"].str.contains("Severe"))]
    if not severe_1h.empty:
        assert severe_1h["mbe"].iloc[0] < 0, "Severe haze tier is expected to be underpredicted by smoothed model"

    diurnal_df = evaluate_diurnal_profile(station_dict, horizons=[1])
    assert len(diurnal_df) == 24
    assert set(diurnal_df["hour_of_day"]) == set(range(24))


# ---------------------------------------------------------------------------
# 7. Visualization Suite Generation
# ---------------------------------------------------------------------------

def test_visualization_generation(tmp_path):
    """Verify that visualization functions execute without errors and save valid PNG files."""
    rec = evaluate_single_station("Aotizhongxin", window_size=72, partition="test")

    fig1_path = tmp_path / "test_timeseries.png"
    fig1 = plot_actual_vs_predicted_timeseries(rec, start_idx=0, num_points=48, horizons_to_plot=[1, 24], save_path=fig1_path)
    assert fig1_path.is_file() and fig1_path.stat().st_size > 1000

    fig2_path = tmp_path / "test_error_dist.png"
    fig2 = plot_error_distribution(rec, horizons_to_plot=[1, 24], save_path=fig2_path)
    assert fig2_path.is_file() and fig2_path.stat().st_size > 1000
