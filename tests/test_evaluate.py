"""
Unit tests for src/evaluate.py.

Verifies:
1. Exact metric correctness against known hand-calculated small arrays.
2. Perfect parity against scikit-learn standard reference metrics.
3. Shape mismatches, empty arrays, and invalid input handling.
4. Missing value (NaN/Inf) diagnostic handling.
5. Perfect prediction edge cases (MAE=0, RMSE=0, R²=1.0).
6. Negative R² detection for poor out-of-sample predictors.
7. Model comparison ranking, baseline improvement calculations, and error stratification.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    mean_absolute_error as skl_mae,
    mean_squared_error as skl_mse,
    r2_score as skl_r2,
)

from src.evaluate import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_bias_error,
    compute_residuals,
    calculate_metrics,
    compare_models,
    evaluate_by_pollution_tier,
    evaluate_by_temporal_group,
    analyze_prediction_lag,
    validate_metric_inputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def known_small_arrays():
    """Hand-calculated toy arrays."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # Deviations (yt - yp): [-2.0, +2.0, -3.0]
    # Abs deviations: [2.0, 2.0, 3.0] -> MAE = 7.0 / 3 = 2.3333333...
    # Squared deviations: [4.0, 4.0, 9.0] -> MSE = 17.0 / 3 = 5.6666666... -> RMSE = 2.380476...
    # y_true mean = 20.0, SS_tot = (-10)^2 + 0^2 + 10^2 = 200.0, SS_res = 17.0 -> R² = 1 - 17/200 = 0.915
    # Bias (yp - yt): [+2.0, -2.0, +3.0] -> MBE = 3.0 / 3 = 1.0
    return y_true, y_pred


# ---------------------------------------------------------------------------
# 1. Correctness Tests
# ---------------------------------------------------------------------------

def test_mae_known_values(known_small_arrays):
    yt, yp = known_small_arrays
    expected_mae = 7.0 / 3.0
    assert np.isclose(mean_absolute_error(yt, yp), expected_mae, atol=1e-6)
    assert np.isclose(mean_absolute_error(yt, yp), skl_mae(yt, yp), atol=1e-6)


def test_rmse_known_values(known_small_arrays):
    yt, yp = known_small_arrays
    expected_rmse = np.sqrt(17.0 / 3.0)
    assert np.isclose(root_mean_squared_error(yt, yp), expected_rmse, atol=1e-6)
    assert np.isclose(root_mean_squared_error(yt, yp), np.sqrt(skl_mse(yt, yp)), atol=1e-6)


def test_r2_known_values(known_small_arrays):
    yt, yp = known_small_arrays
    expected_r2 = 1.0 - (17.0 / 200.0)
    assert np.isclose(r2_score(yt, yp), expected_r2, atol=1e-6)
    assert np.isclose(r2_score(yt, yp), skl_r2(yt, yp), atol=1e-6)


def test_mbe_known_values(known_small_arrays):
    yt, yp = known_small_arrays
    expected_mbe = 1.0  # (2.0 - 2.0 + 3.0) / 3.0
    assert np.isclose(mean_bias_error(yt, yp), expected_mbe, atol=1e-6)


def test_compute_residuals(known_small_arrays):
    yt, yp = known_small_arrays
    residuals = compute_residuals(yt, yp)
    # Convention: yt - yp
    expected_res = np.array([-2.0, 2.0, -3.0])
    np.testing.assert_allclose(residuals, expected_res)


def test_perfect_predictions():
    yt = np.array([5.0, 15.0, 25.0, 35.0])
    yp = np.array([5.0, 15.0, 25.0, 35.0])
    assert mean_absolute_error(yt, yp) == 0.0
    assert root_mean_squared_error(yt, yp) == 0.0
    assert r2_score(yt, yp) == 1.0
    assert mean_bias_error(yt, yp) == 0.0


def test_negative_r2():
    # If predictions are wildly inaccurate, R² is negative
    yt = np.array([10.0, 20.0, 30.0, 40.0])
    yp = np.array([100.0, 200.0, 300.0, 400.0])
    score = r2_score(yt, yp)
    assert score < 0.0
    assert np.isclose(score, skl_r2(yt, yp))


# ---------------------------------------------------------------------------
# 2. Input Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        mean_absolute_error([1.0, 2.0], [1.0, 2.0, 3.0])


def test_empty_inputs_raises():
    with pytest.raises(ValueError, match="Inputs cannot be empty"):
        mean_absolute_error([], [])


def test_pandas_series_support():
    s_true = pd.Series([10.0, 20.0, 30.0], name="actual")
    s_pred = pd.Series([11.0, 19.0, 31.0], name="pred")
    mae = mean_absolute_error(s_true, s_pred)
    assert np.isclose(mae, 1.0)


def test_nan_handling():
    # Paired NaN should be dropped gracefully
    yt = [10.0, np.nan, 30.0, 40.0]
    yp = [12.0, 20.0, np.nan, 38.0]
    # Rows 1 and 2 dropped, leaves [10, 40] and [12, 38] -> |10-12|=2, |40-38|=2 -> MAE = 2.0
    mae = mean_absolute_error(yt, yp)
    assert np.isclose(mae, 2.0)


def test_all_nan_raises():
    with pytest.raises(ValueError, match="no valid pairs remain"):
        mean_absolute_error([np.nan, np.nan], [1.0, 2.0])


# ---------------------------------------------------------------------------
# 3. Model Comparison & Summary
# ---------------------------------------------------------------------------

def test_calculate_metrics():
    yt = np.array([10.0, 20.0, 30.0])
    yp = np.array([12.0, 18.0, 33.0])
    m = calculate_metrics(yt, yp, prefix="val")
    assert "val_mae" in m
    assert "val_rmse" in m
    assert "val_r2" in m
    assert "val_mbe" in m
    assert m["val_n_obs"] == 3


def test_compare_models():
    yt = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    preds = {
        "persistence": np.array([8.0, 16.0, 25.0, 32.0, 40.0]),   # MAE = 4.2
        "linear_regression": np.array([9.5, 19.5, 29.5, 39.5, 49.5]), # MAE = 0.5
        "gradient_boosting": np.array([10.2, 19.8, 30.1, 40.0, 49.9]), # MAE = 0.12
    }
    times = {"persistence": 0.01, "linear_regression": 0.05, "gradient_boosting": 1.20}

    df_cmp = compare_models(preds, yt, baseline_name="persistence", training_times=times)

    assert len(df_cmp) == 3
    # Sorted by MAE ascending: best first
    assert df_cmp.iloc[0]["model"] == "gradient_boosting"
    assert "mae_improvement_pct" in df_cmp.columns
    assert df_cmp.loc[df_cmp["model"] == "gradient_boosting", "mae_improvement_pct"].iloc[0] > 0
    assert "training_time_s" in df_cmp.columns


# ---------------------------------------------------------------------------
# 4. Stratified & Temporal Error Analysis
# ---------------------------------------------------------------------------

def test_evaluate_by_pollution_tier():
    yt = np.array([20.0, 50.0, 100.0, 200.0])
    yp = np.array([22.0, 45.0, 95.0, 180.0])
    df_tiers = evaluate_by_pollution_tier(yt, yp)
    assert "tier" in df_tiers.columns
    assert "mae" in df_tiers.columns
    assert "mbe" in df_tiers.columns
    # Check that tier counts sum to total observations
    assert df_tiers["count"].sum() == 4


def test_evaluate_by_temporal_group():
    dates = pd.date_range("2020-01-01 00:00", periods=24, freq="h")
    yt = np.linspace(10, 50, 24)
    yp = yt + 2.0
    df_hour = evaluate_by_temporal_group(yt, yp, dates, group_by="hour")
    assert len(df_hour) == 24
    assert np.all(df_hour["mae"] == 2.0)


def test_analyze_prediction_lag():
    yt = np.sin(np.linspace(0, 10, 100))
    yp = np.sin(np.linspace(0, 10, 100))
    df_lag = analyze_prediction_lag(yt, yp, max_lag=2)
    assert len(df_lag) == 5  # -2, -1, 0, 1, 2
    # Lag 0 should have correlation near 1.0
    r0 = df_lag.loc[df_lag["lag_hours"] == 0, "pearson_correlation"].iloc[0]
    assert np.isclose(r0, 1.0, atol=1e-3)
