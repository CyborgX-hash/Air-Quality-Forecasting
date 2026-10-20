"""
Full Model Evaluation, Benchmarking, and Error Analysis Pipeline.

Executes:
1. Data loading and causal cleaning (UCI Beijing Aotizhongxin Benchmark Station).
2. Leakage-safe feature engineering (70 candidate features).
3. Chronological 70% Train / 15% Val / 15% Test splitting.
4. Model training: Persistence (naive & constant), Linear Regression, Gradient Boosting.
5. Quantitative validation evaluation & model selection.
6. Quantitative final test evaluation.
7. Publication-quality figure generation saved to reports/figures/.
8. Structured results tables exported to reports/results/.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.load_data import load_station_data
from src.preprocessing import clean_features
from src.features import generate_features, extract_feature_matrix
from src.split_data import chronological_split
from src.models import (
    PersistenceBaseline,
    LinearRegressionBaseline,
    build_gradient_boosting,
    get_feature_importance,
)
from src.evaluate import (
    calculate_metrics,
    compare_models,
    compute_residuals,
    evaluate_by_pollution_tier,
    evaluate_by_temporal_group,
    analyze_prediction_lag,
)
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

RESULTS_DIR = Path("reports/results")
FIGURES_DIR = Path("reports/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def export_dataframe(df: pd.DataFrame, base_filename: str) -> None:
    """Save dataframe as CSV and clean Markdown table."""
    csv_path = RESULTS_DIR / f"{base_filename}.csv"
    md_path = RESULTS_DIR / f"{base_filename}.md"
    df.to_csv(csv_path, index=False)
    try:
        md_content = df.to_markdown(index=False)
    except Exception:
        md_content = df.to_string(index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Exported: {csv_path} and {md_path}")


def main():
    print("=" * 70)
    print("PHASE 1 MODEL EVALUATION & ERROR ANALYSIS PIPELINE")
    print("Role: Model Evaluation Engineer (Mihika)")
    print("=" * 70)

    # 1. Load Data
    data_path = Path("data/raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv")
    print(f"\n[1/7] Loading dataset: {data_path}...")
    df_raw = load_station_data(data_path)
    print(f"Loaded {len(df_raw):,} records from {df_raw['datetime'].min()} to {df_raw['datetime'].max()}.")

    # 2. Preprocess & Clean
    print("\n[2/7] Running causal imputation and wind encoding...")
    df_clean, clean_report = clean_features(df_raw)

    # 3. Feature Engineering
    print("\n[3/7] Generating 70 candidate time-series features...")
    df_feat, feature_cols, feat_meta = generate_features(df_clean)
    print(f"Engineered {len(feature_cols)} features. Total dataset rows: {len(df_feat):,}.")

    # 4. Chronological Splitting
    print("\n[4/7] Splitting data chronologically: 70% Train / 15% Val / 15% Test...")
    train_df, val_df, test_df, split_report = chronological_split(
        df_feat,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    X_train, y_train, meta_train = extract_feature_matrix(train_df, feature_cols, drop_na_features=True)
    X_val, y_val, meta_val = extract_feature_matrix(val_df, feature_cols, drop_na_features=True)
    X_test, y_test, meta_test = extract_feature_matrix(test_df, feature_cols, drop_na_features=True)

    print(f"Partition sample counts:")
    print(f"  Training set:   {len(X_train):,} observations ({split_report['train_start'][:10]} to {split_report['train_end'][:10]})")
    print(f"  Validation set: {len(X_val):,} observations ({split_report['val_start'][:10]} to {split_report['val_end'][:10]})")
    print(f"  Final Test set: {len(X_test):,} observations ({split_report['test_start'][:10]} to {split_report['test_end'][:10]})")

    # 5. Fit Models & Evaluate on Validation Set
    print("\n[5/7] Fitting Phase 1 models and benchmarking on Validation Set...")
    models = {}
    training_times = {}

    # 5a. Naive Persistence Baseline (PM2.5_t)
    t0 = time.time()
    persist_naive = PersistenceBaseline(feature_col="PM2.5").fit(X_train, y_train)
    training_times["Persistence (Naive Lag-1)"] = time.time() - t0
    models["Persistence (Naive Lag-1)"] = persist_naive

    # 5b. Constant Persistence Baseline
    t0 = time.time()
    persist_const = PersistenceBaseline(feature_col=None).fit(X_train, y_train)
    training_times["Persistence (Constant Mean)"] = time.time() - t0
    models["Persistence (Constant Mean)"] = persist_const

    # 5c. Linear Regression Baseline
    t0 = time.time()
    lin_reg = LinearRegressionBaseline().fit(X_train, y_train)
    training_times["Linear Regression"] = time.time() - t0
    models["Linear Regression"] = lin_reg

    # 5d. Gradient Boosting (Default)
    t0 = time.time()
    gb_default = build_gradient_boosting(n_estimators=200, learning_rate=0.05, max_depth=3)
    gb_default.fit(X_train, y_train)
    training_times["Gradient Boosting (Default)"] = time.time() - t0
    models["Gradient Boosting (Default)"] = gb_default

    # 5e. Gradient Boosting (Optimized / Tuned: deeper trees, min samples leaf)
    t0 = time.time()
    gb_tuned = build_gradient_boosting(n_estimators=250, learning_rate=0.04, max_depth=5, min_samples_leaf=8)
    gb_tuned.fit(X_train, y_train)
    training_times["Gradient Boosting (Tuned)"] = time.time() - t0
    models["Gradient Boosting (Tuned)"] = gb_tuned

    # Compute validation predictions
    val_preds = {name: model.predict(X_val) for name, model in models.items()}

    # Validation comparison table
    df_val_comparison = compare_models(
        val_preds,
        y_val,
        baseline_name="Persistence (Naive Lag-1)",
        training_times=training_times,
    )
    print("\n--- VALIDATION MODEL COMPARISON (Model Selection) ---")
    print(df_val_comparison.to_string(index=False))
    export_dataframe(df_val_comparison, "validation_model_comparison")

    # 6. Evaluate Selected Models on Final Test Set
    print("\n[6/7] Evaluating selected models on Final Test Set (No hyperparameter tuning)...")
    test_preds = {name: model.predict(X_test) for name, model in models.items()}

    df_test_comparison = compare_models(
        test_preds,
        y_test,
        baseline_name="Persistence (Naive Lag-1)",
        training_times=training_times,
    )
    print("\n--- FINAL TEST MODEL COMPARISON (Generalization Assessment) ---")
    print(df_test_comparison.to_string(index=False))
    export_dataframe(df_test_comparison, "final_test_model_comparison")

    # Selected Best Model for Deep Error Analysis
    selected_model_name = "Gradient Boosting (Tuned)"
    selected_test_preds = test_preds[selected_model_name]
    selected_residuals = compute_residuals(y_test, selected_test_preds)

    # 7. In-Depth Error Analysis & Stratification
    print("\n[7/7] Executing Stratified Error Analysis and Visualizations...")

    # 7a. Error by Pollution Tier
    df_tier_analysis = evaluate_by_pollution_tier(y_test, selected_test_preds)
    print("\n--- ERROR BY POLLUTION CONCENTRATION TIER ---")
    print(df_tier_analysis.to_string(index=False))
    export_dataframe(df_tier_analysis, "pollution_tier_error_analysis")

    # 7b. Diurnal Error Profile (by Hour of Day)
    df_diurnal = evaluate_by_temporal_group(y_test, selected_test_preds, meta_test["datetime"], group_by="hour")
    export_dataframe(df_diurnal, "diurnal_error_analysis")

    # 7c. Monthly Error Profile
    df_monthly = evaluate_by_temporal_group(y_test, selected_test_preds, meta_test["datetime"], group_by="month")
    export_dataframe(df_monthly, "monthly_error_analysis")

    # 7d. Lag Cross-Correlation
    df_lag = analyze_prediction_lag(y_test, selected_test_preds, max_lag=5)
    print("\n--- PREDICTION LAG CROSS-CORRELATION AUDIT ---")
    print(df_lag.to_string(index=False))
    export_dataframe(df_lag, "lag_cross_correlation")

    # 7e. Feature Importances
    df_imp = get_feature_importance(gb_tuned, feature_names=list(X_train.columns))
    export_dataframe(df_imp.head(20), "feature_importance_top20")
    print("\nTop 5 Most Important Features in Gradient Boosting:")
    for _, row in df_imp.head(5).iterrows():
        print(f"  {row['feature']:<20}: {row['importance']:.4f}")

    # 8. Generate Publication-Quality Figures
    print("\nGenerating and saving publication-quality figures under reports/figures/...")

    # Figure 10: Actual vs Predicted Time-Series (1 representative winter week: 168 hours with high pollution episode)
    # Find a winter window with high variability
    winter_mask = (meta_test["datetime"] >= "2016-12-15") & (meta_test["datetime"] <= "2016-12-22")
    start_winter_idx = int(np.where(winter_mask)[0][0]) if np.any(winter_mask) else 0

    plot_actual_vs_predicted_timeseries(
        y_true=y_test,
        predictions={
            "Ground Truth": y_test,
            "Persistence": test_preds["Persistence (Naive Lag-1)"],
            "Linear Regression": test_preds["Linear Regression"],
            "Gradient Boosting": test_preds["Gradient Boosting (Tuned)"],
        },
        timestamps=meta_test["datetime"],
        start_idx=start_winter_idx,
        num_points=168,
        title="Actual vs. Predicted PM2.5 Concentration (Winter Severe Haze Episode, Test Set)",
        save_path=FIGURES_DIR / "10_actual_vs_predicted_timeseries.png",
    )

    # Figure 11: Parity Scatter Plot
    plot_actual_vs_predicted_scatter(
        y_true=y_test,
        y_pred=selected_test_preds,
        model_name="Gradient Boosting (Tuned) — Test Set",
        save_path=FIGURES_DIR / "11_actual_vs_predicted_scatter.png",
    )

    # Figure 12: Residual Distribution
    plot_residual_distribution(
        residuals=selected_residuals,
        model_name="Gradient Boosting (Tuned) — Test Set",
        save_path=FIGURES_DIR / "12_residual_distribution.png",
    )

    # Figure 13: Residuals Over Time
    plot_residuals_over_time(
        residuals=selected_residuals,
        timestamps=meta_test["datetime"],
        model_name="Gradient Boosting (Tuned)",
        save_path=FIGURES_DIR / "13_residuals_over_time.png",
    )

    # Figure 14: Error by Hour of Day
    plot_error_by_hour(
        y_true=y_test,
        predictions={
            "Persistence": test_preds["Persistence (Naive Lag-1)"],
            "Linear Regression": test_preds["Linear Regression"],
            "Gradient Boosting": test_preds["Gradient Boosting (Tuned)"],
        },
        timestamps=meta_test["datetime"],
        title="Diurnal Error Profile (MAE by Hour of Day, Test Set)",
        save_path=FIGURES_DIR / "14_error_by_hour_of_day.png",
    )

    # Figure 15: Error by Month / Season
    plot_error_by_month(
        y_true=y_test,
        predictions={
            "Persistence": test_preds["Persistence (Naive Lag-1)"],
            "Linear Regression": test_preds["Linear Regression"],
            "Gradient Boosting": test_preds["Gradient Boosting (Tuned)"],
        },
        timestamps=meta_test["datetime"],
        title="Monthly Error Profile (MAE across Calendar Months, Test Set)",
        save_path=FIGURES_DIR / "15_error_by_month_season.png",
    )

    # Figure 16: Model Comparison Benchmark Bar Chart
    # Filter to main benchmark models for visual clarity
    plot_models_df = df_test_comparison[
        df_test_comparison["model"].isin([
            "Persistence (Naive Lag-1)",
            "Linear Regression",
            "Gradient Boosting (Default)",
            "Gradient Boosting (Tuned)",
        ])
    ].copy()
    plot_model_comparison_bars(
        plot_models_df,
        title="Phase 1 Model Generalization Benchmark (Final Test Partition, N=4,920)",
        save_path=FIGURES_DIR / "16_model_comparison_benchmark.png",
    )

    # Figure 17: Feature Importance
    plot_feature_importance(
        df_imp,
        top_n=20,
        title="Top 20 Predictive Features — Gradient Boosting Regressor",
        save_path=FIGURES_DIR / "17_feature_importance_gradient_boosting.png",
    )

    # Figure 18: Error across Pollution Concentration Tiers
    plot_error_by_pollution_tier(
        df_tier_analysis,
        title="Forecasting Error (MAE & Bias) Across Air Quality Pollution Tiers",
        save_path=FIGURES_DIR / "18_pollution_tier_error_analysis.png",
    )

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS, TABLES, AND FIGURES SUCCESSFULLY GENERATED!")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
