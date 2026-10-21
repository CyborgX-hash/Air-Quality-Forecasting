"""
Integration and Unit Tests for End-to-End Pipeline (src/pipeline.py).

Verifies:
1. Pipeline executes end-to-end on synthetic time-series data without errors.
2. Each pipeline stage produces the expected schema, columns, and data types.
3. The workflow runs autonomously without depending on notebook execution state.
4. Non-existent input paths fail fast with a helpful, descriptive FileNotFoundError.
5. Result tables, markdown reports, and figures are created in the configured output directory.
6. The test period is strictly isolated and untouched during training and model selection.
7. PipelineConfig JSON serialization and deserialization maintain full fidelity.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig, DataConfig, SplitConfig, ModelConfig, OutputConfig
from src.pipeline import run_pipeline, PipelineRunResult, export_dataframe
from src.data.load_data import build_timestamp, validate_data
from src.preprocessing import clean_features
from src.features import generate_features, extract_feature_matrix
from src.split_data import chronological_split, validate_split_leakage


@pytest.fixture
def synthetic_pipeline_df():
    """
    Generate 500 consecutive hourly records (~21 days) with realistic Beijing pollutant
    and weather patterns, complete with missing values to test causal cleaning.
    """
    np.random.seed(42)
    n_hours = 500
    dates = pd.date_range("2016-01-01 00:00", periods=n_hours, freq="h")

    # Diurnal & trend baseline
    diurnal = np.sin(2 * np.pi * dates.hour.to_numpy() / 24.0) * 20.0
    trend = np.linspace(20, 80, n_hours)
    noise = np.random.normal(0, 5, n_hours)
    pm25 = np.clip(trend + diurnal + noise + 30.0, 5.0, 450.0).astype(float)

    # Introduce realistic 3-hour missing gap
    pm25[50:53] = np.nan

    pm10 = pm25 * 1.3 + np.random.normal(0, 8, n_hours)
    pm10[60:63] = np.nan

    df = pd.DataFrame({
        "No": range(1, n_hours + 1),
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "hour": dates.hour,
        "PM2.5": pm25,
        "PM10": np.clip(pm10, 5.0, 600.0),
        "SO2": np.random.uniform(2.0, 50.0, n_hours),
        "NO2": np.random.uniform(10.0, 90.0, n_hours),
        "CO": np.random.uniform(300.0, 2500.0, n_hours),
        "O3": np.random.uniform(5.0, 120.0, n_hours),
        "TEMP": np.random.uniform(-10.0, 15.0, n_hours),
        "PRES": np.random.uniform(1005.0, 1030.0, n_hours),
        "DEWP": np.random.uniform(-20.0, 5.0, n_hours),
        "RAIN": np.where(np.random.random(n_hours) > 0.95, np.random.uniform(0.5, 5.0, n_hours), 0.0),
        "wd": np.random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], n_hours),
        "WSPM": np.random.uniform(0.5, 6.0, n_hours),
        "station": ["Aotizhongxin"] * n_hours,
    })
    return df


def test_pipeline_end_to_end_synthetic(tmp_path, synthetic_pipeline_df):
    """Verify that the pipeline runs end-to-end on synthetic data and saves all artifacts."""
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"

    config = PipelineConfig.get_default_config()
    config.output.results_dir = str(results_dir)
    config.output.figures_dir = str(figures_dir)
    config.output.metadata_dir = str(results_dir)
    config.output.generate_figures = True
    config.model.gb_default_params = {
        "n_estimators": 20,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
    }
    config.model.gb_tuned_params = {
        "n_estimators": 25,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
    }

    result = run_pipeline(config=config, df_input=synthetic_pipeline_df, save_artifacts=True)

    assert isinstance(result, PipelineRunResult)
    assert result.df_raw_shape == (500, 19)  # 18 raw columns + datetime
    assert result.df_clean_shape == (500, 23)  # 19 raw + target_valid + wd_deg + wd_sin + wd_cos
    assert result.clean_report["rows_dropped"] == 0

    # Split assertions
    assert result.sample_counts["train"] > 0
    assert result.sample_counts["val"] > 0
    assert result.sample_counts["test"] > 0
    assert result.leakage_audit["passed_all_checks"] is True

    # Model evaluation assertions
    assert not result.val_comparison_df.empty
    assert not result.test_comparison_df.empty
    assert "mae" in result.val_comparison_df.columns
    assert "rmse" in result.val_comparison_df.columns
    assert "r2" in result.val_comparison_df.columns

    # Champion model selection is based on validation MAE
    best_val_model = result.val_comparison_df.iloc[0]["model"]
    assert result.selected_model_name == best_val_model

    # Check exported files
    expected_files = [
        results_dir / "validation_model_comparison.csv",
        results_dir / "validation_model_comparison.md",
        results_dir / "final_test_model_comparison.csv",
        results_dir / "final_test_model_comparison.md",
        results_dir / "pollution_tier_error_analysis.csv",
        results_dir / "pollution_tier_error_analysis.md",
        results_dir / "diurnal_error_analysis.csv",
        results_dir / "monthly_error_analysis.csv",
        results_dir / "lag_cross_correlation.csv",
        results_dir / "experiment_metadata.json",
        figures_dir / "10_actual_vs_predicted_timeseries.png",
        figures_dir / "11_actual_vs_predicted_scatter.png",
        figures_dir / "12_residual_distribution.png",
        figures_dir / "13_residuals_over_time.png",
        figures_dir / "14_error_by_hour_of_day.png",
        figures_dir / "15_error_by_month_season.png",
        figures_dir / "16_model_comparison_benchmark.png",
        figures_dir / "17_feature_importance_gradient_boosting.png",
        figures_dir / "18_pollution_tier_error_analysis.png",
    ]

    for p in expected_files:
        assert p.exists(), f"Expected artifact {p.name} was not generated."


def test_pipeline_stage_schemas(synthetic_pipeline_df):
    """Verify schema integrity and non-leakage at each intermediate stage."""
    # 1. Loading / Timestamping
    df_ts = build_timestamp(synthetic_pipeline_df)
    assert "datetime" in df_ts.columns
    assert pd.api.types.is_datetime64_any_dtype(df_ts["datetime"])

    # 2. Cleaning
    df_clean, clean_report = clean_features(df_ts)
    assert len(df_clean) == len(df_ts)
    assert "target_valid" in df_clean.columns
    assert df_clean["target_valid"].dtype == bool
    # Target NaNs must NOT be imputed
    assert clean_report["target_imputed"] is False

    # 3. Feature Generation
    df_feat, feature_cols, feat_meta = generate_features(df_clean, horizon=1)
    assert "target_pm25" in df_feat.columns
    assert "target_pm25" not in feature_cols
    assert "station" not in feature_cols
    assert "datetime" not in feature_cols
    assert len(feature_cols) > 20

    # 4. Splitting
    train_df, val_df, test_df, split_report = chronological_split(
        df_feat, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    max_train_ts = train_df["datetime"].max()
    min_val_ts = val_df["datetime"].min()
    max_val_ts = val_df["datetime"].max()
    min_test_ts = test_df["datetime"].min()

    assert max_train_ts < min_val_ts
    assert max_val_ts < min_test_ts

    # 5. Feature extraction
    X_train, y_train, meta_train = extract_feature_matrix(train_df, feature_cols, drop_na_features=True)
    assert not y_train.isnull().any()
    assert len(X_train) == len(y_train)
    assert len(X_train) == len(meta_train)


def test_pipeline_invalid_input_path_raises_helpful_error():
    """Verify that a missing CSV path fails fast with an explanatory FileNotFoundError."""
    config = PipelineConfig.get_default_config()
    config.data.data_dir = "non_existent_directory_for_testing"
    config.data.station = "Aotizhongxin"

    with pytest.raises(FileNotFoundError, match="Dataset not found at"):
        run_pipeline(config=config, df_input=None)


def test_pipeline_test_period_isolation(synthetic_pipeline_df):
    """Verify that the test set remains untouched during training and validation model selection."""
    config = PipelineConfig.get_default_config()
    config.output.generate_figures = False
    config.model.gb_default_params = {"n_estimators": 10, "max_depth": 2, "random_state": 42}
    config.model.gb_tuned_params = {"n_estimators": 15, "max_depth": 2, "random_state": 42}

    res = run_pipeline(config=config, df_input=synthetic_pipeline_df, save_artifacts=False)

    train_end = pd.Timestamp(res.split_report["train_end"])
    val_start = pd.Timestamp(res.split_report["val_start"])
    val_end = pd.Timestamp(res.split_report["val_end"])
    test_start = pd.Timestamp(res.split_report["test_start"])

    # Strict chronological ordering: train < val < test
    assert train_end < val_start
    assert val_end < test_start

    # Check model selection was performed using validation metrics, not test metrics
    best_val_model = res.val_comparison_df.iloc[0]["model"]
    assert res.selected_model_name == best_val_model


def test_pipeline_config_serialization(tmp_path):
    """Verify full fidelity when saving and loading PipelineConfig to and from JSON."""
    cfg = PipelineConfig.get_default_config()
    cfg.data.station = "Tiantan"
    cfg.split.train_ratio = 0.60
    cfg.split.val_ratio = 0.20
    cfg.split.test_ratio = 0.20
    cfg.model.random_seed = 999

    cfg_file = tmp_path / "test_config.json"
    cfg.to_json(cfg_file)

    loaded_cfg = PipelineConfig.from_json(cfg_file)
    assert loaded_cfg.data.station == "Tiantan"
    assert loaded_cfg.split.train_ratio == 0.60
    assert loaded_cfg.split.val_ratio == 0.20
    assert loaded_cfg.split.test_ratio == 0.20
    assert loaded_cfg.model.random_seed == 999
    assert loaded_cfg.to_dict() == cfg.to_dict()
