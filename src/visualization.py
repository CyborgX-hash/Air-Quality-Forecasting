"""
Visualization module for air quality forecasting model evaluation and error analysis.

Provides publication-quality plots with clear labeling, units, annotations,
and legends suitable for academic and technical reporting:
1. Actual vs. Predicted time-series trace over representative intervals
2. Actual vs. Predicted scatter plot with 1:1 parity line
3. Residual distribution (histogram + KDE with statistical annotations)
4. Residuals over time with standard deviation error envelopes
5. Diurnal error profile by hour of day (0-23)
6. Monthly / Seasonal error profile
7. Model comparison bar chart across MAE, RMSE, and R²
8. Feature importance ranking for Gradient Boosting models
9. Stratified error by pollution concentration tiers
10. Cross-correlation lag analysis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluate import (
    calculate_metrics,
    compute_residuals,
    validate_metric_inputs,
    evaluate_by_pollution_tier,
    evaluate_by_temporal_group,
)

logger = logging.getLogger(__name__)

# Configure global publication aesthetics
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DEFAULT_FIGURES_DIR = Path("reports/figures")


def _ensure_dir(save_path: Optional[Union[str, Path]]) -> Optional[Path]:
    if save_path is None:
        return None
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. Time-Series Trace: Actual vs Predicted
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted_timeseries(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    predictions: Union[np.ndarray, pd.Series, Dict[str, Union[np.ndarray, pd.Series]]],
    timestamps: Optional[Union[pd.DatetimeIndex, pd.Series, Sequence[pd.Timestamp]]] = None,
    start_idx: int = 0,
    num_points: int = 168,  # 1 week of hourly data
    title: str = "One-Hour-Ahead PM2.5 Forecasting: Actual vs. Predicted",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot actual versus predicted PM2.5 concentrations over a representative time window.

    Parameters
    ----------
    y_true : array-like
        Ground truth PM2.5 observations.
    predictions : array-like or Dict[str, array-like]
        Predicted values for a single model or a mapping of model names to predictions.
    timestamps : array-like of Timestamps, optional
        Temporal timestamps. If omitted, integer steps are used.
    start_idx : int, default 0
        Starting index for the slice.
    num_points : int, default 168
        Number of consecutive hours to plot (168h = 1 week).
    title : str
        Figure title.
    save_path : Path or str, optional
        Destination path to save the high-resolution figure.
    """
    end_idx = min(start_idx + num_points, len(y_true))
    idx_slice = slice(start_idx, end_idx)

    y_t_sub = np.asarray(y_true)[idx_slice]

    if timestamps is not None:
        ts_sub = pd.to_datetime(pd.Series(timestamps).iloc[idx_slice])
        x_axis = ts_sub
        use_dates = True
    else:
        x_axis = np.arange(len(y_t_sub))
        use_dates = False

    fig, ax = plt.subplots(figsize=(14, 5.5))

    # Plot Ground Truth
    ax.plot(
        x_axis,
        y_t_sub,
        label="Actual PM2.5 (Ground Truth)",
        color="#2c3e50",
        linewidth=2.2,
        alpha=0.9,
        zorder=5,
    )

    palette = {
        "persistence": "#7f8c8d",
        "linear_regression": "#2980b9",
        "gradient_boosting": "#27ae60",
        "gradient_boosting_default": "#27ae60",
        "gradient_boosting_tuned": "#e67e22",
    }
    fallback_colors = ["#8e44ad", "#d35400", "#16a085", "#c0392b"]

    if isinstance(predictions, dict):
        for i, (name, preds) in enumerate(predictions.items()):
            p_sub = np.asarray(preds)[idx_slice]
            c = palette.get(name.lower(), fallback_colors[i % len(fallback_colors)])
            ls = "--" if "persistence" in name.lower() else "-"
            ax.plot(
                x_axis,
                p_sub,
                label=f"Predicted ({name})",
                color=c,
                linewidth=1.8,
                linestyle=ls,
                alpha=0.85,
            )
    else:
        p_sub = np.asarray(predictions)[idx_slice]
        ax.plot(
            x_axis,
            p_sub,
            label="Predicted PM2.5",
            color="#27ae60",
            linewidth=1.8,
            linestyle="-",
            alpha=0.85,
        )

    ax.set_ylabel("PM2.5 Concentration (µg/m³)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5)

    if use_dates:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:00"))
        plt.xticks(rotation=25, ha="right")
        ax.set_xlabel("Observation Timestamp (UTC+8)", fontweight="bold")
    else:
        ax.set_xlabel("Time Elapsed (Hours)", fontweight="bold")

    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved time-series plot to %s", out_file)

    return fig


# ---------------------------------------------------------------------------
# 2. Scatter Plot: Actual vs Predicted with 1:1 Parity Line
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted_scatter(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
    model_name: str = "Gradient Boosting",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate parity scatter plot with 1:1 line and metric text annotations.
    """
    yt, yp = validate_metric_inputs(y_true, y_pred)
    metrics = calculate_metrics(yt, yp)

    fig, ax = plt.subplots(figsize=(7, 7))

    max_val = max(np.percentile(yt, 99.9), np.percentile(yp, 99.9)) * 1.05
    min_val = 0.0

    # Scatter with density alpha
    ax.scatter(
        yt,
        yp,
        alpha=0.25,
        color="#2980b9",
        edgecolors="none",
        s=16,
        label="Predictions",
    )

    # 1:1 Parity line
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="#e74c3c",
        linestyle="--",
        linewidth=2.0,
        label="1:1 Perfect Parity",
        zorder=4,
    )

    # Linear regression fit line through scatter
    fit_coef = np.polyfit(yt, yp, 1)
    fit_fn = np.poly1d(fit_coef)
    ax.plot(
        [min_val, max_val],
        fit_fn([min_val, max_val]),
        color="#2c3e50",
        linestyle=":",
        linewidth=2.0,
        label=f"Fit Line (slope: {fit_coef[0]:.2f})",
        zorder=5,
    )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Actual PM2.5 (µg/m³)", fontweight="bold")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)", fontweight="bold")

    plot_title = title or f"Actual vs. Predicted PM2.5 ({model_name})"
    ax.set_title(plot_title, fontweight="bold", pad=12)

    # Metric annotations box
    stats_text = (
        f"MAE:  {metrics['mae']:.2f} µg/m³\n"
        f"RMSE: {metrics['rmse']:.2f} µg/m³\n"
        f"R²:   {metrics['r2']:.4f}\n"
        f"MBE:  {metrics['mbe']:+.2f} µg/m³\n"
        f"N:    {metrics['n_obs']:,}"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
    )

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved scatter plot to %s", out_file)

    return fig


# ---------------------------------------------------------------------------
# 3. Residual Distribution: Histogram + KDE
# ---------------------------------------------------------------------------

def plot_residual_distribution(
    residuals: Union[np.ndarray, pd.Series, Sequence[float]],
    model_name: str = "Gradient Boosting",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot histogram and kernel density estimation (KDE) of residuals (y_true - y_pred).
    """
    res = np.asarray(residuals, dtype=float)
    res = res[~np.isnan(res) & ~np.isinf(res)]

    mean_res = np.mean(res)
    median_res = np.median(res)
    std_res = np.std(res)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Clip extreme display range to 1st and 99th percentiles for clear distribution visibility
    low_lim, high_lim = np.percentile(res, 0.5), np.percentile(res, 99.5)
    display_res = res[(res >= low_lim) & (res <= high_lim)]

    sns.histplot(
        display_res,
        bins=60,
        kde=True,
        color="#34495e",
        edgecolor="#2c3e50",
        stat="density",
        ax=ax,
        label="Residual Density",
    )

    # Reference vertical lines
    ax.axvline(0.0, color="#7f8c8d", linestyle="-", linewidth=1.5, label="Zero Bias (e=0)")
    ax.axvline(
        mean_res,
        color="#e74c3c",
        linestyle="--",
        linewidth=2.0,
        label=f"Mean: {mean_res:+.2f} µg/m³",
    )
    ax.axvline(
        median_res,
        color="#27ae60",
        linestyle="-.",
        linewidth=2.0,
        label=f"Median: {median_res:+.2f} µg/m³",
    )

    ax.set_xlabel("Residual: Actual - Predicted (µg/m³)", fontweight="bold")
    ax.set_ylabel("Probability Density", fontweight="bold")

    plot_title = title or f"Residual Distribution: Actual − Predicted ({model_name})"
    ax.set_title(plot_title, fontweight="bold", pad=12)

    stats_box = (
        f"Mean:   {mean_res:+.2f} µg/m³\n"
        f"Median: {median_res:+.2f} µg/m³\n"
        f"Std:    {std_res:.2f} µg/m³\n"
        f"IQR:    {np.percentile(res, 75) - np.percentile(res, 25):.2f} µg/m³"
    )
    ax.text(
        0.05,
        0.95,
        stats_box,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
    )

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved residual distribution plot to %s", out_file)

    return fig


# ---------------------------------------------------------------------------
# 4. Residuals Over Time
# ---------------------------------------------------------------------------

def plot_residuals_over_time(
    residuals: Union[np.ndarray, pd.Series, Sequence[float]],
    timestamps: Optional[Union[pd.DatetimeIndex, pd.Series, Sequence[pd.Timestamp]]] = None,
    model_name: str = "Gradient Boosting",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot residuals across time with zero reference and +/- 2 standard deviation bands.
    """
    res = np.asarray(residuals, dtype=float)

    if timestamps is not None:
        x_axis = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
        use_dates = True
    else:
        x_axis = np.arange(len(res))
        use_dates = False

    std_res = np.nanstd(res)

    fig, ax = plt.subplots(figsize=(14, 5.0))

    ax.scatter(
        x_axis,
        res,
        color="#2980b9",
        alpha=0.35,
        s=12,
        edgecolors="none",
        label="Residual (y - ŷ)",
    )
    ax.axhline(0.0, color="#e74c3c", linestyle="-", linewidth=1.8, label="Zero Error")
    ax.axhline(+2 * std_res, color="#e67e22", linestyle="--", linewidth=1.4, label=f"+2σ ({+2*std_res:.1f})")
    ax.axhline(-2 * std_res, color="#e67e22", linestyle="--", linewidth=1.4, label=f"-2σ ({-2*std_res:.1f})")

    ax.set_ylabel("Residual (µg/m³)", fontweight="bold")
    plot_title = title or f"Residuals Across Time ({model_name})"
    ax.set_title(plot_title, fontweight="bold", pad=12)

    if use_dates:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=20, ha="right")
        ax.set_xlabel("Timestamp", fontweight="bold")
    else:
        ax.set_xlabel("Test Index", fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved residuals over time plot to %s", out_file)

    return fig


# ---------------------------------------------------------------------------
# 5. Diurnal Error Profile (by Hour of Day)
# ---------------------------------------------------------------------------

def plot_error_by_hour(
    y_true: Union[np.ndarray, pd.Series],
    predictions: Union[np.ndarray, pd.Series, Dict[str, Union[np.ndarray, pd.Series]]],
    timestamps: Union[pd.DatetimeIndex, pd.Series],
    title: str = "Forecast Error (MAE) by Hour of Day (Diurnal Cycle)",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot diurnal error distribution across hours (0 to 23).
    """
    fig, ax = plt.subplots(figsize=(11, 5.0))

    palette = {
        "persistence": "#7f8c8d",
        "linear_regression": "#2980b9",
        "gradient_boosting": "#27ae60",
        "gradient_boosting_default": "#27ae60",
    }
    fallback = ["#e67e22", "#8e44ad"]

    if isinstance(predictions, dict):
        for i, (name, yp) in enumerate(predictions.items()):
            df_hour = evaluate_by_temporal_group(y_true, yp, timestamps, group_by="hour")
            c = palette.get(name.lower(), fallback[i % len(fallback)])
            ax.plot(
                df_hour["hour"],
                df_hour["mae"],
                marker="o",
                linewidth=2.0,
                color=c,
                label=f"{name} (Mean MAE: {df_hour['mae'].mean():.2f})",
            )
    else:
        df_hour = evaluate_by_temporal_group(y_true, predictions, timestamps, group_by="hour")
        ax.plot(
            df_hour["hour"],
            df_hour["mae"],
            marker="o",
            linewidth=2.2,
            color="#27ae60",
            label="Gradient Boosting",
        )

    ax.set_xlabel("Hour of Day (0–23, Beijing Local Time)", fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (µg/m³)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(range(0, 24))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved error by hour plot to %s", out_file)

    return fig


# ---------------------------------------------------------------------------
# 6. Seasonal / Monthly Error Profile
# ---------------------------------------------------------------------------

def plot_error_by_month(
    y_true: Union[np.ndarray, pd.Series],
    predictions: Union[np.ndarray, pd.Series, Dict[str, Union[np.ndarray, pd.Series]]],
    timestamps: Union[pd.DatetimeIndex, pd.Series],
    title: str = "Forecast Error (MAE) by Month / Season",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot seasonal error distribution across calendar months.
    """
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }

    fig, ax = plt.subplots(figsize=(11, 5.0))

    if isinstance(predictions, dict):
        bar_data = []
        for name, yp in predictions.items():
            df_m = evaluate_by_temporal_group(y_true, yp, timestamps, group_by="month")
            for _, r in df_m.iterrows():
                bar_data.append({
                    "model": name,
                    "month": month_names.get(int(r["month"]), str(int(r["month"]))),
                    "month_num": int(r["month"]),
                    "mae": r["mae"],
                })
        df_all = pd.DataFrame(bar_data).sort_values("month_num")
        sns.barplot(
            data=df_all,
            x="month",
            y="mae",
            hue="model",
            palette="Blues_r",
            ax=ax,
        )
    else:
        df_m = evaluate_by_temporal_group(y_true, predictions, timestamps, group_by="month")
        df_m["month_name"] = df_m["month"].map(month_names)
        sns.barplot(
            data=df_m,
            x="month_name",
            y="mae",
            color="#3498db",
            ax=ax,
        )

    ax.set_xlabel("Calendar Month", fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (µg/m³)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved error by month plot to %s", out_file)

    return fig


# ---------------------------------------------------------------------------
# 7. Model Comparison Chart (MAE, RMSE, R²)
# ---------------------------------------------------------------------------

def plot_model_comparison_bars(
    comparison_df: pd.DataFrame,
    title: str = "Phase 1 Model Benchmark Comparison",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate side-by-side comparative bar charts for MAE, RMSE, and R².
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

    colors = ["#7f8c8d", "#2980b9", "#27ae60", "#e67e22"]

    models = comparison_df["model"].tolist()
    palette = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    # Subplot 1: MAE
    bars1 = axes[0].bar(
        comparison_df["model"],
        comparison_df["mae"],
        color=[palette[m] for m in models],
        width=0.55,
        edgecolor="#2c3e50",
    )
    axes[0].set_ylabel("MAE (µg/m³) — Lower is better", fontweight="bold")
    axes[0].set_title("Mean Absolute Error", fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    for b in bars1:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 0.3, f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot 2: RMSE
    bars2 = axes[1].bar(
        comparison_df["model"],
        comparison_df["rmse"],
        color=[palette[m] for m in models],
        width=0.55,
        edgecolor="#2c3e50",
    )
    axes[1].set_ylabel("RMSE (µg/m³) — Lower is better", fontweight="bold")
    axes[1].set_title("Root Mean Squared Error", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    for b in bars2:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h + 0.4, f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot 3: R²
    bars3 = axes[2].bar(
        comparison_df["model"],
        comparison_df["r2"],
        color=[palette[m] for m in models],
        width=0.55,
        edgecolor="#2c3e50",
    )
    axes[2].set_ylabel("R² Score — Higher is better (Max 1.0)", fontweight="bold")
    axes[2].set_title("Coefficient of Determination (R²)", fontweight="bold")
    axes[2].axhline(0.0, color="#e74c3c", linestyle="--", linewidth=1.2)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    for b in bars3:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width() / 2, h + 0.01 if h >= 0 else h - 0.05, f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for ax in axes:
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right")

    fig.suptitle(title, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved model comparison plot to %s", out_file)

    return fig


# ---------------------------------------------------------------------------
# 8. Feature Importance Ranking
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top Predictive Features (Gradient Boosting)",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot horizontal bar chart of the top feature importances.
    """
    df_top = importance_df.head(top_n).copy().sort_values(by="importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7.5))

    y_pos = np.arange(len(df_top))
    bars = ax.barh(
        y_pos,
        df_top["importance"],
        color="#2980b9",
        edgecolor="#1f618d",
        height=0.65,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top["feature"], fontsize=10)
    ax.set_xlabel("Relative Feature Importance (Normalized)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Add numeric value tags
    for b in bars:
        w = b.get_width()
        ax.text(w + 0.003, b.get_y() + b.get_height() / 2, f"{w:.3f}", ha="left", va="center", fontsize=9)

    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved feature importance plot to %s", out_file)

    return fig


# ---------------------------------------------------------------------------
# 9. Error by Pollution Concentration Tier
# ---------------------------------------------------------------------------

def plot_error_by_pollution_tier(
    tier_df: pd.DataFrame,
    title: str = "Forecasting Error (MAE & MBE) across Pollution Tiers",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot error metrics stratified by AQI pollution categories.
    Reveals underprediction during severe smog episodes.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5.5))

    x = np.arange(len(tier_df))
    width = 0.35

    bars_mae = ax1.bar(
        x - width / 2,
        tier_df["mae"],
        width,
        label="MAE (Error Magnitude)",
        color="#e74c3c",
        edgecolor="#c0392b",
    )
    bars_mbe = ax1.bar(
        x + width / 2,
        tier_df["mbe"],
        width,
        label="MBE (Bias: +Over / -Under)",
        color="#3498db",
        edgecolor="#2980b9",
    )

    ax1.axhline(0.0, color="#7f8c8d", linestyle="-", linewidth=1.2)
    ax1.set_ylabel("Concentration Error (µg/m³)", fontweight="bold")
    ax1.set_title(title, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_df["tier"], rotation=18, ha="right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    for b in bars_mae:
        h = b.get_height()
        if not np.isnan(h):
            ax1.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for b in bars_mbe:
        h = b.get_height()
        if not np.isnan(h):
            pos_y = h - 2.5 if h < 0 else h + 0.8
            ax1.text(b.get_x() + b.get_width() / 2, pos_y, f"{h:+.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved pollution tier error plot to %s", out_file)

    return fig


# ---------------------------------------------------------------------------
# Multi-Station Aggregate Visualization Suite
# ---------------------------------------------------------------------------

def plot_multi_station_mae_comparison(
    df_sweep: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Test MAE across 12 Beijing Monitoring Stations by Model",
) -> plt.Figure:
    """
    Generate grouped bar chart of Test MAE across all 12 stations and candidate models.

    Parameters
    ----------
    df_sweep : pd.DataFrame
        DataFrame with columns: ['station', 'model', 'test_mae'].
    save_path : Optional[Union[str, Path]]
        Destination path for saving PNG.
    title : str
        Figure title.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    models_order = [
        "Gradient Boosting (Tuned)",
        "Gradient Boosting (Default)",
        "Linear Regression",
        "Persistence (Naive Lag-1)",
    ]
    # Filter to known models if available
    available_models = [m for m in models_order if m in df_sweep["model"].unique()]
    if not available_models:
        available_models = sorted(df_sweep["model"].unique())

    df_filtered = df_sweep[df_sweep["model"].isin(available_models)].copy()
    stations = sorted(df_filtered["station"].unique())

    palette = {
        "Gradient Boosting (Tuned)": "#16a085",
        "Gradient Boosting (Default)": "#2980b9",
        "Linear Regression": "#f39c12",
        "Persistence (Naive Lag-1)": "#c0392b",
        "Persistence (Constant Mean)": "#7f8c8d",
    }

    sns.barplot(
        data=df_filtered,
        x="station",
        y="test_mae",
        hue="model",
        hue_order=available_models,
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
    )

    # Plot overall benchmark mean line for tuned model
    tuned_rows = df_filtered[df_filtered["model"] == "Gradient Boosting (Tuned)"]
    if not tuned_rows.empty:
        mean_tuned_mae = tuned_rows["test_mae"].mean()
        ax.axhline(
            mean_tuned_mae,
            color="#16a085",
            linestyle="--",
            linewidth=1.8,
            label=f"12-Station Tuned GB Mean ({mean_tuned_mae:.2f} µg/m³)",
        )

    ax.set_ylabel("Test MAE (µg/m³)", fontweight="bold")
    ax.set_xlabel("Monitoring Station", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xticklabels(stations, rotation=25, ha="right", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(title="Model Architecture", frameon=True, framealpha=0.9, loc="upper right")
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved multi-station MAE plot to %s", out_file)

    return fig


def plot_multi_station_r2_comparison(
    df_sweep: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Test R² Performance across 12 Beijing Monitoring Stations",
) -> plt.Figure:
    """
    Generate comparison of Test R² across all 12 stations for champion vs baseline models.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    models_order = [
        "Gradient Boosting (Tuned)",
        "Linear Regression",
        "Persistence (Naive Lag-1)",
    ]
    available_models = [m for m in models_order if m in df_sweep["model"].unique()]
    if not available_models:
        available_models = sorted(df_sweep["model"].unique())

    df_filtered = df_sweep[df_sweep["model"].isin(available_models)].copy()
    stations = sorted(df_filtered["station"].unique())

    palette = {
        "Gradient Boosting (Tuned)": "#16a085",
        "Linear Regression": "#f39c12",
        "Persistence (Naive Lag-1)": "#c0392b",
    }

    sns.barplot(
        data=df_filtered,
        x="station",
        y="test_r2",
        hue="model",
        hue_order=available_models,
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_ylabel("Test R²", fontweight="bold")
    ax.set_xlabel("Monitoring Station", fontweight="bold")
    ax.set_ylim(0.90, 0.98)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xticklabels(stations, rotation=25, ha="right", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(title="Model Architecture", frameon=True, framealpha=0.9, loc="lower right")
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved multi-station R² plot to %s", out_file)

    return fig


def plot_multi_station_metric_heatmap(
    df_sweep: pd.DataFrame,
    metric: str = "test_mae",
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a heatmap of stations (rows) vs models (columns) for a specific metric.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    models_order = [
        "Gradient Boosting (Tuned)",
        "Gradient Boosting (Default)",
        "Linear Regression",
        "Persistence (Naive Lag-1)",
    ]
    df_filtered = df_sweep[df_sweep["model"].isin(models_order)].copy()
    pivot_df = df_filtered.pivot(index="station", columns="model", values=metric)
    # Reorder columns
    cols = [m for m in models_order if m in pivot_df.columns]
    pivot_df = pivot_df[cols]

    cmap = "viridis_r" if "mae" in metric or "rmse" in metric else "viridis"
    fmt = ".2f"

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={"label": metric.replace("_", " ").upper()},
        ax=ax,
    )

    plot_title = title or f"12-Station Comparison Matrix ({metric.replace('_', ' ').title()})"
    ax.set_title(plot_title, fontweight="bold", pad=14)
    ax.set_xlabel("Model Architecture", fontweight="bold")
    ax.set_ylabel("Monitoring Station", fontweight="bold")
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved multi-station heatmap to %s", out_file)

    return fig


def plot_multi_station_improvement_pct(
    df_sweep: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "MAE Improvement over Naive Persistence (%) across 12 Stations",
) -> plt.Figure:
    """
    Plot relative MAE reduction (%) over persistence baseline across all stations.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    tuned_df = df_sweep[df_sweep["model"] == "Gradient Boosting (Tuned)"].sort_values(
        by="mae_improvement_pct", ascending=False
    )

    bars = ax.bar(
        tuned_df["station"],
        tuned_df["mae_improvement_pct"],
        color="#27ae60",
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
    )

    mean_imprv = tuned_df["mae_improvement_pct"].mean()
    ax.axhline(
        mean_imprv,
        color="#c0392b",
        linestyle="--",
        linewidth=1.8,
        label=f"12-Station Average Improvement ({mean_imprv:.2f}%)",
    )

    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.2,
            f"+{h:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_ylabel("MAE Improvement (%)", fontweight="bold")
    ax.set_xlabel("Monitoring Station", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xticklabels(tuned_df["station"], rotation=25, ha="right", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")
    plt.tight_layout()

    out_file = _ensure_dir(save_path)
    if out_file:
        fig.savefig(out_file)
        logger.info("Saved multi-station improvement plot to %s", out_file)

    return fig
