"""
End-to-End Machine Learning Pipeline for Air Quality Forecasting (Phase 1).

Orchestrates the complete Phase 1 workflow:
1. Dataset loading & schema validation
2. Causal data cleaning & forward-fill imputation
3. Time-series feature engineering (AR lags, multi-pollutant lags, rolling stats, cyclical/domain features)
4. Chronological train/validation/test splitting & anti-leakage audit
5. Model training & validation-based model selection (Persistence, Linear Regression, Gradient Boosting)
6. Final generalization evaluation on the untouched test period
7. Publication-quality figure generation and structured results table export
8. Comprehensive reproducibility metadata recording

Command-line entry point:
    python -m src.pipeline --station Aotizhongxin
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PipelineConfig, VALID_STATIONS
from src.data.load_data import (
    REQUIRED_COLUMNS,
    build_timestamp,
    load_all_stations,
    load_station_data,
    validate_data,
)
from src.preprocessing import clean_features
from src.features import extract_feature_matrix, generate_features
from src.split_data import (
    chronological_split,
    fit_transform_scaler,
    validate_split_leakage,
)
from src.models import (
    LinearRegressionBaseline,
    PersistenceBaseline,
    build_gradient_boosting,
    get_feature_importance,
    tune_gradient_boosting,
)
from src.evaluate import (
    analyze_prediction_lag,
    calculate_metrics,
    compare_models,
    compute_residuals,
    evaluate_by_pollution_tier,
    evaluate_by_temporal_group,
)
from src.visualization import (
    plot_actual_vs_predicted_scatter,
    plot_actual_vs_predicted_timeseries,
    plot_error_by_hour,
    plot_error_by_month,
    plot_error_by_pollution_tier,
    plot_feature_importance,
    plot_model_comparison_bars,
    plot_residual_distribution,
    plot_residuals_over_time,
)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("src.pipeline")


@dataclass
class PipelineRunResult:
    """Encapsulates the outputs, models, tables, and metadata of a completed pipeline run."""
    run_id: str
    timestamp: str
    config: PipelineConfig
    df_raw_shape: Tuple[int, int]
    df_clean_shape: Tuple[int, int]
    clean_report: Dict[str, Any]
    feature_cols: List[str]
    feature_meta: Dict[str, Any]
    split_report: Dict[str, Any]
    leakage_audit: Dict[str, Any]
    sample_counts: Dict[str, int]
    trained_models: Dict[str, Any]
    training_times: Dict[str, float]
    val_predictions: Dict[str, np.ndarray]
    val_comparison_df: pd.DataFrame
    selected_model_name: str
    selected_model: Any
    test_predictions: Dict[str, np.ndarray]
    test_comparison_df: pd.DataFrame
    tier_analysis_df: pd.DataFrame
    diurnal_analysis_df: pd.DataFrame
    monthly_analysis_df: pd.DataFrame
    lag_analysis_df: pd.DataFrame
    feature_importance_df: Optional[pd.DataFrame]
    saved_files: Dict[str, str]
    metadata: Dict[str, Any]


def export_dataframe(df: pd.DataFrame, base_filename: str, output_dir: Path) -> Tuple[Path, Path]:
    """Export a DataFrame as both CSV and GitHub Flavored Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{base_filename}.csv"
    md_path = output_dir / f"{base_filename}.md"

    df.to_csv(csv_path, index=False)
    try:
        md_content = df.to_markdown(index=False)
    except Exception:
        md_content = df.to_string(index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content + "\n")

    logger.debug("Exported table to %s and %s", csv_path, md_path)
    return csv_path, md_path


def get_environment_metadata() -> Dict[str, Any]:
    """Gather runtime environment, OS, git commit, and dependency versions for reproducibility."""
    meta: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "processor": platform.processor(),
    }

    # Git commit hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        meta["git_commit"] = git_hash
        meta["git_branch"] = git_branch
    except Exception:
        meta["git_commit"] = "unknown"
        meta["git_branch"] = "unknown"

    # Core package versions
    packages = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "pytest"]
    pkg_versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg.replace("-", "_"))
            pkg_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkg_versions[pkg] = "not installed"
    meta["packages"] = pkg_versions

    return meta


def run_pipeline(
    config: Optional[PipelineConfig] = None,
    df_input: Optional[pd.DataFrame] = None,
    save_artifacts: bool = True,
) -> PipelineRunResult:
    """
    Execute the full end-to-end Phase 1 machine learning pipeline.

    Parameters
    ----------
    config : Optional[PipelineConfig]
        Master pipeline configuration. If None, default configuration is used.
    df_input : Optional[pd.DataFrame]
        In-memory input DataFrame. If provided, disk loading is skipped (ideal for testing).
    save_artifacts : bool, default True
        Whether to export result tables, figures, and metadata JSON files to disk.

    Returns
    -------
    PipelineRunResult
        Container holding all datasets, trained models, predictions, metric tables,
        and reproducibility metadata.
    """
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_timestamp = datetime.now().isoformat()
    config = config or PipelineConfig.get_default_config()

    logger.info("=" * 70)
    logger.info("STARTING PHASE 1 AIR QUALITY FORECASTING PIPELINE")
    logger.info("Run ID: %s | Station: %s", run_id, config.data.station)
    logger.info("=" * 70)

    results_dir = Path(config.output.results_dir)
    figures_dir = Path(config.output.figures_dir)
    metadata_dir = Path(config.output.metadata_dir)
    saved_files: Dict[str, str] = {}

    # -------------------------------------------------------------------------
    # STAGE 1: Dataset Loading & Validation
    # -------------------------------------------------------------------------
    t_start = time.time()
    logger.info("[1/8] Loading and validating dataset...")

    if df_input is not None:
        df_raw = df_input.copy()
        if "datetime" not in df_raw.columns:
            df_raw = build_timestamp(df_raw, drop_source_cols=False)
        if "station" not in df_raw.columns:
            df_raw["station"] = config.data.station
        df_raw = df_raw.sort_values(by=["station", "datetime"]).reset_index(drop=True)
        data_source_desc = "in_memory_dataframe"
    else:
        resolved_path = config.data.resolve_data_path()
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{resolved_path}'. "
                f"Please ensure station CSV files are placed in '{config.data.data_dir}' "
                "or specify a valid --data-file. See data/README.md for instructions."
            )

        if config.data.station.lower() == "all" and resolved_path.is_dir():
            logger.info("Loading all 12 stations from %s...", resolved_path)
            df_raw = load_all_stations(resolved_path)
            data_source_desc = f"all_stations_dir:{resolved_path}"
        else:
            logger.info("Loading station data from %s...", resolved_path)
            df_raw = load_station_data(resolved_path)
            data_source_desc = str(resolved_path)

    # Perform structural data audit
    val_audit = validate_data(df_raw)
    logger.info(
        "Data validation: %d records, %d duplicate timestamps, is_chronological=%s, is_continuous=%s",
        len(df_raw),
        val_audit["duplicate_records"],
        val_audit["is_chronological"],
        val_audit["is_hourly_continuous"],
    )

    # -------------------------------------------------------------------------
    # STAGE 2: Data Cleaning & Causal Imputation
    # -------------------------------------------------------------------------
    logger.info("[2/8] Executing causal data cleaning & forward-fill imputation...")
    df_clean, clean_report = clean_features(
        df_raw,
        target_col=config.data.target_column,
        timestamp_col=config.data.timestamp_column,
        station_col=config.data.station_column,
        max_ffill_hours=config.cleaning.max_ffill_hours,
        encode_wind=config.cleaning.encode_wind,
        fill_remaining_with_median=config.cleaning.fill_remaining_with_median,
    )
    logger.info(
        "Cleaned %d records. Target missing: %d (never imputed). Valid target rows: %d.",
        clean_report["final_rows"],
        clean_report["target_missing_count"],
        clean_report["target_valid_count"],
    )

    # -------------------------------------------------------------------------
    # STAGE 3: Feature Engineering & Target Creation
    # -------------------------------------------------------------------------
    logger.info("[3/8] Engineering time-series features (horizon=%dh)...", config.feature.horizon)
    df_feat, feature_cols, feat_meta = generate_features(
        df_clean,
        target_col=config.feature.target_col,
        horizon=config.feature.horizon,
        pm25_lags=config.feature.pm25_lags,
        pollutant_lags=config.feature.pollutant_lags,
        rolling_windows=config.feature.rolling_windows,
        rolling_stats=config.feature.rolling_stats,
        rolling_cols=config.feature.rolling_cols,
        include_calendar=config.feature.include_calendar,
        include_cyclical=config.feature.include_cyclical,
        include_domain_weather=config.feature.include_domain_weather,
        include_wind_encoding=config.feature.include_wind_encoding,
        timestamp_col=config.data.timestamp_column,
        station_col=config.data.station_column,
        new_target_col=config.feature.new_target_col,
    )
    logger.info("Engineered %d candidate features across %d rows.", len(feature_cols), len(df_feat))

    # -------------------------------------------------------------------------
    # STAGE 4: Chronological Splitting & Anti-Leakage Audit
    # -------------------------------------------------------------------------
    logger.info(
        "[4/8] Partitioning data chronologically: %.0f%% Train / %.0f%% Val / %.0f%% Test...",
        config.split.train_ratio * 100,
        config.split.val_ratio * 100,
        config.split.test_ratio * 100,
    )
    train_df, val_df, test_df, split_report = chronological_split(
        df_feat,
        train_ratio=config.split.train_ratio,
        val_ratio=config.split.val_ratio,
        test_ratio=config.split.test_ratio,
        timestamp_col=config.data.timestamp_column,
        station_col=config.data.station_column,
        embargo_hours=config.split.embargo_hours,
    )

    # Anti-leakage audit
    leakage_audit = validate_split_leakage(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        timestamp_col=config.data.timestamp_column,
        target_col=config.feature.new_target_col,
        horizon_hours=config.feature.horizon,
    )
    if not leakage_audit["passed_all_checks"]:
        logger.warning("Split leakage warning: %s", leakage_audit)
    else:
        logger.info("Anti-leakage audit PASSED: strict chronological monotonicity & zero overlap.")

    # Extract feature matrices (X, y, metadata)
    X_train, y_train, meta_train = extract_feature_matrix(
        train_df,
        feature_cols=feature_cols,
        target_col=config.feature.new_target_col,
        station_col=config.data.station_column,
        timestamp_col=config.data.timestamp_column,
        require_valid_target=True,
        drop_na_features=True,
    )
    X_val, y_val, meta_val = extract_feature_matrix(
        val_df,
        feature_cols=feature_cols,
        target_col=config.feature.new_target_col,
        station_col=config.data.station_column,
        timestamp_col=config.data.timestamp_column,
        require_valid_target=True,
        drop_na_features=True,
    )
    X_test, y_test, meta_test = extract_feature_matrix(
        test_df,
        feature_cols=feature_cols,
        target_col=config.feature.new_target_col,
        station_col=config.data.station_column,
        timestamp_col=config.data.timestamp_column,
        require_valid_target=True,
        drop_na_features=True,
    )

    # Optional learned scaling fitted strictly on training data
    fitted_scaler = None
    if config.model.scaler_type:
        logger.info("Fitting '%s' scaler on train set and transforming val/test...", config.model.scaler_type)
        X_train, X_val, X_test, fitted_scaler = fit_transform_scaler(
            X_train, X_val, X_test, scaler_type=config.model.scaler_type
        )

    sample_counts = {
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test),
        "total_supervised": len(X_train) + len(X_val) + len(X_test),
    }
    logger.info(
        "Supervised observations: Train=%d (%s to %s), Val=%d, Test=%d",
        sample_counts["train"],
        split_report["train_start"][:10],
        split_report["train_end"][:10],
        sample_counts["val"],
        sample_counts["test"],
    )

    # -------------------------------------------------------------------------
    # STAGE 5: Model Training & Validation-Based Selection
    # -------------------------------------------------------------------------
    logger.info("[5/8] Training Phase 1 models and benchmarking on Validation set...")
    trained_models: Dict[str, Any] = {}
    training_times: Dict[str, float] = {}

    # 5a. Naive Persistence Baseline (y_{t+1} = PM2.5_t)
    t0 = time.time()
    persist_naive = PersistenceBaseline(feature_col=config.data.target_column).fit(X_train, y_train)
    training_times["Persistence (Naive Lag-1)"] = time.time() - t0
    trained_models["Persistence (Naive Lag-1)"] = persist_naive

    # 5b. Constant Persistence Baseline
    t0 = time.time()
    persist_const = PersistenceBaseline(feature_col=None).fit(X_train, y_train)
    training_times["Persistence (Constant Mean)"] = time.time() - t0
    trained_models["Persistence (Constant Mean)"] = persist_const

    # 5c. Linear Regression Baseline
    t0 = time.time()
    lin_reg = LinearRegressionBaseline().fit(X_train, y_train)
    training_times["Linear Regression"] = time.time() - t0
    trained_models["Linear Regression"] = lin_reg

    # 5d. Gradient Boosting Regressor (Default parameters)
    t0 = time.time()
    gb_params = config.model.gb_default_params.copy()
    gb_params["random_state"] = config.model.random_seed
    gb_default = build_gradient_boosting(**gb_params)
    gb_default.fit(X_train, y_train)
    training_times["Gradient Boosting (Default)"] = time.time() - t0
    trained_models["Gradient Boosting (Default)"] = gb_default

    # 5e. Gradient Boosting Regressor (Tuned / Grid-Searched)
    t0 = time.time()
    if config.model.tune_gb_grid:
        logger.info("Executing lightweight hyperparameter grid search for Gradient Boosting...")
        gb_tuned, best_params = tune_gradient_boosting(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_grid=config.model.gb_tune_param_grid,
            random_state=config.model.random_seed,
        )
        logger.info("Best GB grid search parameters: %s", best_params)
    else:
        gb_tuned_params = config.model.gb_tuned_params.copy()
        gb_tuned_params["random_state"] = config.model.random_seed
        gb_tuned = build_gradient_boosting(**gb_tuned_params)
        gb_tuned.fit(X_train, y_train)
    training_times["Gradient Boosting (Tuned)"] = time.time() - t0
    trained_models["Gradient Boosting (Tuned)"] = gb_tuned

    # Compute validation predictions and benchmark table
    val_predictions = {name: model.predict(X_val) for name, model in trained_models.items()}
    val_comparison_df = compare_models(
        predictions_dict=val_predictions,
        y_true=y_val,
        baseline_name=config.model.baseline_name,
        training_times=training_times,
    )
    logger.info("\n--- VALIDATION MODEL COMPARISON (Model Selection) ---\n%s", val_comparison_df.to_string(index=False))

    # Select best model strictly using Validation MAE
    selected_model_name = str(val_comparison_df.iloc[0]["model"])
    selected_model = trained_models[selected_model_name]
    logger.info("Selected Champion Model: '%s' (Validation MAE: %.2f µg/m³)", selected_model_name, val_comparison_df.iloc[0]["mae"])

    # -------------------------------------------------------------------------
    # STAGE 6: Final Evaluation on Untouched Test Period
    # -------------------------------------------------------------------------
    logger.info("[6/8] Evaluating models on untouched Test Set (Generalization Assessment)...")
    test_predictions = {name: model.predict(X_test) for name, model in trained_models.items()}
    test_comparison_df = compare_models(
        predictions_dict=test_predictions,
        y_true=y_test,
        baseline_name=config.model.baseline_name,
        training_times=training_times,
    )
    logger.info("\n--- FINAL TEST MODEL COMPARISON (Untouched Test Evaluation) ---\n%s", test_comparison_df.to_string(index=False))

    # Error analysis for selected model
    selected_test_preds = test_predictions[selected_model_name]
    selected_residuals = compute_residuals(y_test, selected_test_preds)

    # 6a. Error by Pollution Tier
    tier_analysis_df = evaluate_by_pollution_tier(y_test, selected_test_preds)

    # 6b. Temporal Diurnal and Monthly Error Profiles
    diurnal_analysis_df = evaluate_by_temporal_group(y_test, selected_test_preds, meta_test["datetime"], group_by="hour")
    monthly_analysis_df = evaluate_by_temporal_group(y_test, selected_test_preds, meta_test["datetime"], group_by="month")

    # 6c. Prediction Lag Cross-Correlation
    lag_analysis_df = analyze_prediction_lag(y_test, selected_test_preds, max_lag=5)

    # 6d. Feature Importance (for Gradient Boosting)
    feature_importance_df: Optional[pd.DataFrame] = None
    gb_model_for_importance = None
    if hasattr(selected_model, "feature_importances_"):
        gb_model_for_importance = selected_model
    elif "Gradient Boosting (Tuned)" in trained_models and hasattr(trained_models["Gradient Boosting (Tuned)"], "feature_importances_"):
        gb_model_for_importance = trained_models["Gradient Boosting (Tuned)"]
    elif "Gradient Boosting (Default)" in trained_models and hasattr(trained_models["Gradient Boosting (Default)"], "feature_importances_"):
        gb_model_for_importance = trained_models["Gradient Boosting (Default)"]

    if gb_model_for_importance is not None:
        feature_importance_df = get_feature_importance(gb_model_for_importance, feature_names=list(X_train.columns))

    # -------------------------------------------------------------------------
    # STAGE 7: Results Table and Figure Generation
    # -------------------------------------------------------------------------
    if save_artifacts:
        logger.info("[7/8] Exporting result tables and publication-quality figures...")
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Export result tables
        csv_p, md_p = export_dataframe(val_comparison_df, "validation_model_comparison", results_dir)
        saved_files["val_comparison_csv"] = str(csv_p)
        saved_files["val_comparison_md"] = str(md_p)

        csv_p, md_p = export_dataframe(test_comparison_df, "final_test_model_comparison", results_dir)
        saved_files["test_comparison_csv"] = str(csv_p)
        saved_files["test_comparison_md"] = str(md_p)

        csv_p, md_p = export_dataframe(tier_analysis_df, "pollution_tier_error_analysis", results_dir)
        saved_files["tier_analysis_csv"] = str(csv_p)
        saved_files["tier_analysis_md"] = str(md_p)

        csv_p, md_p = export_dataframe(diurnal_analysis_df, "diurnal_error_analysis", results_dir)
        saved_files["diurnal_analysis_csv"] = str(csv_p)
        saved_files["diurnal_analysis_md"] = str(md_p)

        csv_p, md_p = export_dataframe(monthly_analysis_df, "monthly_error_analysis", results_dir)
        saved_files["monthly_analysis_csv"] = str(csv_p)
        saved_files["monthly_analysis_md"] = str(md_p)

        csv_p, md_p = export_dataframe(lag_analysis_df, "lag_cross_correlation", results_dir)
        saved_files["lag_analysis_csv"] = str(csv_p)
        saved_files["lag_analysis_md"] = str(md_p)

        if feature_importance_df is not None:
            csv_p, md_p = export_dataframe(feature_importance_df.head(20), "feature_importance_top20", results_dir)
            saved_files["feature_importance_csv"] = str(csv_p)
            saved_files["feature_importance_md"] = str(md_p)

        # Generate figures if enabled
        if config.output.generate_figures:
            logger.info("Generating publication figures in %s...", figures_dir)

            # Fig 10: Actual vs Predicted Time-Series (Representative 168h episode)
            slice_len = min(168, len(y_test))
            fig_path = figures_dir / "10_actual_vs_predicted_timeseries.png"
            plot_actual_vs_predicted_timeseries(
                y_true=y_test,
                predictions={
                    "Ground Truth": y_test,
                    "Persistence": test_predictions["Persistence (Naive Lag-1)"],
                    "Linear Regression": test_predictions["Linear Regression"],
                    "Gradient Boosting": selected_test_preds,
                },
                timestamps=meta_test["datetime"],
                start_idx=0,
                num_points=slice_len,
                title="Actual vs. Predicted PM2.5 Concentration (Test Set Representative Interval)",
                save_path=fig_path,
            )
            plt.close("all")
            saved_files["fig_timeseries"] = str(fig_path)

            # Fig 11: Parity Scatter Plot
            fig_path = figures_dir / "11_actual_vs_predicted_scatter.png"
            plot_actual_vs_predicted_scatter(
                y_true=y_test,
                y_pred=selected_test_preds,
                model_name=f"{selected_model_name} — Test Set",
                save_path=fig_path,
            )
            plt.close("all")
            saved_files["fig_scatter"] = str(fig_path)

            # Fig 12: Residual Distribution
            fig_path = figures_dir / "12_residual_distribution.png"
            plot_residual_distribution(
                residuals=selected_residuals,
                model_name=f"{selected_model_name} — Test Set",
                save_path=fig_path,
            )
            plt.close("all")
            saved_files["fig_residuals_dist"] = str(fig_path)

            # Fig 13: Residuals Over Time
            fig_path = figures_dir / "13_residuals_over_time.png"
            plot_residuals_over_time(
                residuals=selected_residuals,
                timestamps=meta_test["datetime"],
                model_name=selected_model_name,
                save_path=fig_path,
            )
            plt.close("all")
            saved_files["fig_residuals_time"] = str(fig_path)

            # Fig 14: Error by Hour of Day
            fig_path = figures_dir / "14_error_by_hour_of_day.png"
            plot_error_by_hour(
                y_true=y_test,
                predictions={
                    "Persistence": test_predictions["Persistence (Naive Lag-1)"],
                    "Linear Regression": test_predictions["Linear Regression"],
                    "Gradient Boosting": selected_test_preds,
                },
                timestamps=meta_test["datetime"],
                title="Diurnal Error Profile (MAE by Hour of Day, Test Set)",
                save_path=fig_path,
            )
            plt.close("all")
            saved_files["fig_error_hour"] = str(fig_path)

            # Fig 15: Error by Month / Season
            fig_path = figures_dir / "15_error_by_month_season.png"
            plot_error_by_month(
                y_true=y_test,
                predictions={
                    "Persistence": test_predictions["Persistence (Naive Lag-1)"],
                    "Linear Regression": test_predictions["Linear Regression"],
                    "Gradient Boosting": selected_test_preds,
                },
                timestamps=meta_test["datetime"],
                title="Monthly Error Profile (MAE across Calendar Months, Test Set)",
                save_path=fig_path,
            )
            plt.close("all")
            saved_files["fig_error_month"] = str(fig_path)

            # Fig 16: Model Comparison Benchmark Bar Chart
            fig_path = figures_dir / "16_model_comparison_benchmark.png"
            plot_model_comparison_bars(
                test_comparison_df,
                title="Phase 1 Model Generalization Benchmark (Final Test Partition)",
                save_path=fig_path,
            )
            plt.close("all")
            saved_files["fig_model_bars"] = str(fig_path)

            # Fig 17: Feature Importance Bar Chart
            if feature_importance_df is not None:
                fig_path = figures_dir / "17_feature_importance_gradient_boosting.png"
                plot_feature_importance(
                    feature_importance_df,
                    top_n=20,
                    title=f"Top 20 Predictive Features — {selected_model_name}",
                    save_path=fig_path,
                )
                plt.close("all")
                saved_files["fig_feature_importance"] = str(fig_path)

            # Fig 18: Error Across Pollution Tiers
            fig_path = figures_dir / "18_pollution_tier_error_analysis.png"
            plot_error_by_pollution_tier(
                tier_analysis_df,
                title="Forecasting Error (MAE & Bias) Across Air Quality Pollution Tiers",
                save_path=fig_path,
            )
            plt.close("all")
            saved_files["fig_pollution_tier"] = str(fig_path)

    # -------------------------------------------------------------------------
    # STAGE 8: Save Experiment Metadata
    # -------------------------------------------------------------------------
    logger.info("[8/8] Recording comprehensive experiment reproducibility metadata...")
    env_meta = get_environment_metadata()
    total_elapsed = round(time.time() - t_start, 2)

    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": run_timestamp,
        "total_elapsed_seconds": total_elapsed,
        "experiment_name": config.experiment_name,
        "environment": env_meta,
        "config": config.to_dict(),
        "dataset": {
            "source": data_source_desc,
            "station": config.data.station,
            "raw_shape": list(df_raw.shape),
            "clean_shape": list(df_clean.shape),
            "target_column": config.data.target_column,
            "future_target_column": config.feature.new_target_col,
            "horizon_hours": config.feature.horizon,
        },
        "cleaning": clean_report,
        "features": {
            "count": len(feature_cols),
            "list": feature_cols,
            "metadata": feat_meta,
        },
        "split": split_report,
        "leakage_audit": leakage_audit,
        "sample_counts": sample_counts,
        "models": {
            "trained": list(trained_models.keys()),
            "training_times_s": training_times,
            "selected_champion_model": selected_model_name,
        },
        "validation_metrics": val_comparison_df.to_dict(orient="records"),
        "final_test_metrics": test_comparison_df.to_dict(orient="records"),
    }

    if save_artifacts:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        meta_run_path = metadata_dir / f"experiment_metadata_{run_id}.json"
        meta_latest_path = metadata_dir / "experiment_metadata.json"

        with open(meta_run_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        with open(meta_latest_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        saved_files["metadata_run_json"] = str(meta_run_path)
        saved_files["metadata_latest_json"] = str(meta_latest_path)
        logger.info("Saved metadata to %s and %s", meta_run_path, meta_latest_path)

    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN %.2fs", total_elapsed)
    logger.info("Selected Champion: %s (Test MAE: %.2f µg/m³, Test R²: %.4f)",
                selected_model_name,
                test_comparison_df[test_comparison_df['model'] == selected_model_name]['mae'].values[0],
                test_comparison_df[test_comparison_df['model'] == selected_model_name]['r2'].values[0])
    logger.info("=" * 70)

    return PipelineRunResult(
        run_id=run_id,
        timestamp=run_timestamp,
        config=config,
        df_raw_shape=df_raw.shape,
        df_clean_shape=df_clean.shape,
        clean_report=clean_report,
        feature_cols=feature_cols,
        feature_meta=feat_meta,
        split_report=split_report,
        leakage_audit=leakage_audit,
        sample_counts=sample_counts,
        trained_models=trained_models,
        training_times=training_times,
        val_predictions=val_predictions,
        val_comparison_df=val_comparison_df,
        selected_model_name=selected_model_name,
        selected_model=selected_model,
        test_predictions=test_predictions,
        test_comparison_df=test_comparison_df,
        tier_analysis_df=tier_analysis_df,
        diurnal_analysis_df=diurnal_analysis_df,
        monthly_analysis_df=monthly_analysis_df,
        lag_analysis_df=lag_analysis_df,
        feature_importance_df=feature_importance_df,
        saved_files=saved_files,
        metadata=metadata,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Air Quality Forecasting Phase 1 End-to-End Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--station",
        type=str,
        default="Aotizhongxin",
        help=f"Station name to forecast, or 'all' for multi-station dataset. Valid: {VALID_STATIONS}",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to directory containing station CSV files",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Explicit path to single station CSV file (overrides --data-dir and --station)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file overriding defaults",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save result tables (defaults to reports/results)",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Directory to save figures (defaults to reports/figures)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation for faster headless execution or automated testing",
    )
    parser.add_argument(
        "--tune-gb",
        action="store_true",
        help="Run grid-search hyperparameter tuning on validation set for Gradient Boosting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model reproducibility",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecasting horizon in hours (default: 1-hour ahead)",
    )
    return parser


def main() -> None:
    """CLI entry point for running the pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load or construct configuration
    if args.config:
        config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig.get_default_config()

    # CLI Overrides
    if args.station:
        config.data.station = args.station
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.data_file:
        config.data.data_file = args.data_file
    if args.output_dir:
        config.output.results_dir = args.output_dir
        config.output.metadata_dir = args.output_dir
    if args.figures_dir:
        config.output.figures_dir = args.figures_dir
    if args.no_figures:
        config.output.generate_figures = False
    if args.tune_gb:
        config.model.tune_gb_grid = True
    if args.seed:
        config.model.random_seed = args.seed
    if args.horizon:
        config.feature.horizon = args.horizon

    try:
        run_pipeline(config=config, save_artifacts=True)
    except FileNotFoundError as e:
        logger.error("Configuration / Data error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Pipeline failed with unexpected error: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
