"""
Phase 2 Visualization Suite for Model Evaluation and Error Analysis.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Mihika Mathur (Model Evaluation, Error Analysis & Visualization Specialist)

Publication-Quality Plotting Suite implementing the 8 core evaluation visual requirements:
1. Actual vs Predicted PM2.5 Time Series
2. Prediction Error Distribution (Histogram + KDE + Gaussian Fit)
3. Training vs Validation Loss Curves
4. MAE Comparison Across Models and Horizons
5. RMSE Comparison Across Models and Horizons
6. Forecast Horizon vs Error Degradation Curve
7. Station-Wise Performance Comparison (Heatmap & Spatial Breakdown)
8. Residual Diagnostics (Residuals vs Predictions, Residuals vs Time, Diurnal Boxplots)
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src.phase2.evaluate import (
    MultiStationEvaluationSuite,
    StationEvaluationRecord,
)

logger = logging.getLogger(__name__)

# Configure global publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
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

DEFAULT_FIGURES_DIR = Path("reports/figures/phase2")


def _ensure_dir(save_path: Optional[Union[str, Path]]) -> Optional[Path]:
    if save_path is None:
        return None
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. Actual vs. Predicted PM2.5 Time-Series Trace
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted_timeseries(
    record: StationEvaluationRecord,
    start_idx: int = 100,
    num_points: int = 168,  # 1 week (168 hours)
    horizons_to_plot: Sequence[int] = (1, 6, 24),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot actual vs predicted PM2.5 traces over a representative time window.
    """
    save_path = _ensure_dir(save_path)
    end_idx = min(start_idx + num_points, len(record.timestamps))
    ts_slice = record.timestamps[start_idx:end_idx]

    fig, axes = plt.subplots(len(horizons_to_plot), 1, figsize=(14, 3.8 * len(horizons_to_plot)), sharex=True)
    if len(horizons_to_plot) == 1:
        axes = [axes]

    colors = {"1h": "#1f77b4", "6h": "#ff7f0e", "24h": "#2ca02c"}

    for ax, h in zip(axes, horizons_to_plot):
        h_idx = record.horizons.index(h)
        yt = record.y_true[start_idx:end_idx, h_idx]
        yp = record.y_pred_lstm[start_idx:end_idx, h_idx]
        y_pers = record.y_pred_persistence[start_idx:end_idx, h_idx]

        h_col = colors.get(f"{h}h", "#9467bd")

        ax.plot(ts_slice, yt, label="Actual Ground Truth", color="black", lw=1.8, alpha=0.85)
        ax.plot(ts_slice, yp, label=f"LSTM Forecast (t+{h}h)", color=h_col, lw=2.0, linestyle="-")
        ax.plot(ts_slice, y_pers, label=f"Naive Persistence (t+{h}h)", color="#7f7f7f", lw=1.2, linestyle="--", alpha=0.7)

        # Highlight severe pollution zone
        ax.axhline(150, color="#d62728", linestyle=":", alpha=0.5, label="Severe Pollution (>150 µg/m³)")

        ax.set_ylabel(f"t+{h}h PM2.5 (µg/m³)")
        ax.set_title(f"Station: {record.station_name} — Horizon: t+{h}h (W={record.window_size}h)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.9)

    axes[-1].set_xlabel("Date and Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %H:00"))
    fig.autofmt_xdate(rotation=20)
    fig.suptitle(f"Multi-Horizon PM2.5 Forecast Trace — {record.station_name}", fontsize=14, y=0.99)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Prediction Error Distribution (Residuals Histogram + KDE)
# ---------------------------------------------------------------------------

def plot_error_distribution(
    record: StationEvaluationRecord,
    horizons_to_plot: Sequence[int] = (1, 6, 24),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot the distribution of forecast residuals (y_pred - y_true) across horizons.
    """
    save_path = _ensure_dir(save_path)
    fig, axes = plt.subplots(1, len(horizons_to_plot), figsize=(5.2 * len(horizons_to_plot), 4.5), sharey=True)
    if len(horizons_to_plot) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, h, col in zip(axes, horizons_to_plot, colors):
        h_idx = record.horizons.index(h)
        residuals = record.y_pred_lstm[:, h_idx] - record.y_true[:, h_idx]

        mbe = float(np.mean(residuals))
        std = float(np.std(residuals))
        mae = float(np.mean(np.abs(residuals)))

        # Plot histogram & KDE
        sns.histplot(residuals, bins=60, kde=True, ax=ax, color=col, stat="density", alpha=0.4, edgecolor="none")

        # Zero reference & Mean error line
        ax.axvline(0, color="black", linestyle="--", lw=1.5, label="Zero Error (Unbiased)")
        ax.axvline(mbe, color="red", linestyle="-", lw=1.5, label=f"MBE = {mbe:+.2f} µg/m³")

        ax.set_title(f"Horizon t+{h}h\nMAE: {mae:.2f} | Std: {std:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Prediction Error: (y_pred - y_true) [µg/m³]")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(-150, 150)
        ax.legend(loc="upper left", fontsize=9)

    axes[0].set_ylabel("Probability Density")
    fig.suptitle(f"Forecast Error Distribution by Horizon — {record.station_name}", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Training vs Validation Loss Curves
# ---------------------------------------------------------------------------

def plot_training_and_validation_loss(
    history_file: Union[str, Path],
    station_name: str = "Aotizhongxin",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot training loss and validation loss across epochs.
    """
    save_path = _ensure_dir(save_path)
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val_loss = min(history["val_loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    ax1.plot(epochs, history["loss"], label="Training Loss (MAE)", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss (MAE)", color="#d62728", lw=2)
    ax1.scatter(best_epoch, best_val_loss, color="black", s=70, zorder=5,
                label=f"Best Checkpoint (Epoch {best_epoch}: {best_val_loss:.4f})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Standardized MAE Loss")
    ax1.set_title("Training & Validation Loss Progression", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Learning Rate curve if present
    if "lr" in history or "learning_rate" in history:
        lr_key = "lr" if "lr" in history else "learning_rate"
        ax2.plot(epochs, history[lr_key], label="Learning Rate", color="#2ca02c", lw=2, marker="o", markersize=4)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate (Log Scale)")
        ax2.set_yscale("log")
        ax2.set_title("Adam Learning Rate Schedule (ReduceLROnPlateau)", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right")
    else:
        # Plot relative loss reduction
        init_loss = history["loss"][0]
        loss_reduction = [100.0 * (init_loss - l) / init_loss for l in history["loss"]]
        ax2.plot(epochs, loss_reduction, label="Training Loss Reduction (%)", color="#2ca02c", lw=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Relative Loss Reduction (%)")
        ax2.set_title("Optimization Convergence Profile", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right")

    fig.suptitle(f"LSTM Model Training History — Station: {station_name}", fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. MAE Comparison Across Models and Horizons
# ---------------------------------------------------------------------------

def plot_mae_comparison_across_models(
    macro_df: pd.DataFrame,
    phase1_df: Optional[pd.DataFrame] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot MAE across forecast horizons comparing LSTM against Multi-Horizon Baselines.
    Also plots 1-hour comparison against all Phase 1 candidate models.
    """
    save_path = _ensure_dir(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Multi-Horizon MAE (LSTM vs Persistence)
    horizons = macro_df["horizon_hours"].values
    lstm_mae = macro_df["lstm_macro_mae"].values
    pers_mae = macro_df["pers_macro_mae"].values

    x = np.arange(len(horizons))
    width = 0.35

    rects1 = ax1.bar(x - width/2, lstm_mae, width, label="LSTM (Champion W72)", color="#1f77b4", edgecolor="black", alpha=0.85)
    rects2 = ax1.bar(x + width/2, pers_mae, width, label="Naive Persistence", color="#aec7e8", edgecolor="black", alpha=0.85)

    ax1.set_ylabel("Macro Mean Absolute Error (µg/m³)")
    ax1.set_xlabel("Forecast Horizon")
    ax1.set_title("Macro MAE Comparison Across Horizons (12 Stations)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"t+{h}h" for h in horizons])
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(loc="upper left")

    # Annotate bar values
    for r in rects1:
        ax1.annotate(f"{r.get_height():.1f}", xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for r in rects2:
        ax1.annotate(f"{r.get_height():.1f}", xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    # Panel 2: Phase 1 vs Phase 2 at 1-Hour Horizon
    if phase1_df is not None and not phase1_df.empty:
        p1_summary = phase1_df.groupby("model")[["test_mae", "test_rmse"]].mean().reset_index()
        p1_summary = p1_summary.sort_values(by="test_mae", ascending=True)

        palette = ["#2ca02c" if "Gradient" in m else "#1f77b4" if "LSTM" in m else "#ff7f0e" if "Linear" in m else "#7f7f7f"
                   for m in p1_summary["model"]]

        bars = ax2.barh(p1_summary["model"], p1_summary["test_mae"], color=palette, edgecolor="black", alpha=0.85)
        ax2.set_xlabel("Macro 1-Hour Test MAE (µg/m³)")
        ax2.set_title("1-Hour Ahead Benchmark: Phase 1 vs. Phase 2 LSTM", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4, axis="x")

        for b in bars:
            ax2.annotate(f"{b.get_width():.2f} µg/m³", xy=(b.get_width(), b.get_y() + b.get_height()/2),
                         xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")
        ax2.set_xlim(0, max(p1_summary["test_mae"]) * 1.18)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. RMSE Comparison Across Models and Horizons
# ---------------------------------------------------------------------------

def plot_rmse_comparison_across_models(
    macro_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot RMSE comparison across models and horizons (Macro vs Pooled).
    """
    save_path = _ensure_dir(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    horizons = macro_df["horizon_hours"].values
    x = np.arange(len(horizons))
    width = 0.35

    # Panel 1: Macro RMSE
    ax1.bar(x - width/2, macro_df["lstm_macro_rmse"], width, label="LSTM (Champion W72)", color="#2ca02c", edgecolor="black", alpha=0.85)
    ax1.bar(x + width/2, macro_df["pers_macro_rmse"], width, label="Naive Persistence", color="#98df8a", edgecolor="black", alpha=0.85)

    ax1.set_ylabel("Macro RMSE (µg/m³)")
    ax1.set_xlabel("Forecast Horizon")
    ax1.set_title("Macro RMSE Across Horizons (Station Average)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"t+{h}h" for h in horizons])
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(loc="upper left")

    for i, h in enumerate(horizons):
        l_val = macro_df["lstm_macro_rmse"].iloc[i]
        p_val = macro_df["pers_macro_rmse"].iloc[i]
        ax1.annotate(f"{l_val:.1f}", (x[i] - width/2, l_val + 1), ha="center", fontsize=8.5)
        ax1.annotate(f"{p_val:.1f}", (x[i] + width/2, p_val + 1), ha="center", fontsize=8.5)

    # Panel 2: Pooled Observation RMSE
    ax2.bar(x - width/2, pooled_df["pooled_rmse_lstm"], width, label="LSTM Pooled RMSE", color="#d62728", edgecolor="black", alpha=0.85)
    ax2.bar(x + width/2, pooled_df["pooled_rmse_pers"], width, label="Persistence Pooled RMSE", color="#ff9896", edgecolor="black", alpha=0.85)

    ax2.set_ylabel("Pooled RMSE (µg/m³)")
    ax2.set_xlabel("Forecast Horizon")
    ax2.set_title("Pooled RMSE Across All 58,815 Observations", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"t+{h}h" for h in horizons])
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(loc="upper left")

    for i, h in enumerate(horizons):
        l_val = pooled_df["pooled_rmse_lstm"].iloc[i]
        p_val = pooled_df["pooled_rmse_pers"].iloc[i]
        ax2.annotate(f"{l_val:.1f}", (x[i] - width/2, l_val + 1), ha="center", fontsize=8.5)
        ax2.annotate(f"{p_val:.1f}", (x[i] + width/2, p_val + 1), ha="center", fontsize=8.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Forecast Horizon vs Error Degradation Curve
# ---------------------------------------------------------------------------

def plot_forecast_horizon_vs_error(
    macro_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot the continuous degradation of MAE and RMSE as forecast horizon increases from 1h to 24h.
    """
    save_path = _ensure_dir(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    horizons = macro_df["horizon_hours"].values

    # Panel 1: Absolute Error Growth
    ax1.plot(horizons, macro_df["lstm_macro_mae"], marker="o", lw=2.5, color="#1f77b4", label="LSTM MAE (µg/m³)")
    ax1.plot(horizons, macro_df["pers_macro_mae"], marker="s", lw=2.0, linestyle="--", color="#7f7f7f", label="Persistence MAE (µg/m³)")

    ax1.plot(horizons, macro_df["lstm_macro_rmse"], marker="^", lw=2.5, color="#d62728", label="LSTM RMSE (µg/m³)")
    ax1.plot(horizons, macro_df["pers_macro_rmse"], marker="v", lw=2.0, linestyle="--", color="#ff7f0e", label="Persistence RMSE (µg/m³)")

    ax1.set_xlabel("Forecast Horizon H (Hours Ahead)")
    ax1.set_ylabel("Error Metric (µg/m³)")
    ax1.set_title("Forecast Accuracy Degradation Over Time", fontsize=12, fontweight="bold")
    ax1.set_xticks(horizons)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Panel 2: Relative Advantage / Improvement of LSTM over Persistence
    pct_mae_impr = macro_df["macro_mae_improvement_pct"].values
    pct_rmse_impr = macro_df["macro_rmse_improvement_pct"].values

    ax2.plot(horizons, pct_mae_impr, marker="o", lw=2.5, color="#2ca02c", label="% MAE Reduction over Persistence")
    ax2.plot(horizons, pct_rmse_impr, marker="s", lw=2.2, color="#17becf", label="% RMSE Reduction over Persistence")
    ax2.axhline(0, color="black", linestyle="--", alpha=0.6, label="Parity Baseline (0%)")

    # Shade outperformance region
    ax2.fill_between(horizons, 0, pct_mae_impr, where=(pct_mae_impr >= 0), color="#2ca02c", alpha=0.15)
    ax2.fill_between(horizons, 0, pct_mae_impr, where=(pct_mae_impr < 0), color="#d62728", alpha=0.15)

    ax2.set_xlabel("Forecast Horizon H (Hours Ahead)")
    ax2.set_ylabel("Relative Error Reduction (%)")
    ax2.set_title("LSTM Outperformance vs. Naive Persistence Horizon", fontsize=12, fontweight="bold")
    ax2.set_xticks(horizons)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    fig.suptitle("Temporal Error Scaling Across Forecast Horizons", fontsize=14, y=0.99)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Station-Wise Performance Comparison Heatmap
# ---------------------------------------------------------------------------

def plot_station_wise_comparison(
    detailed_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot a matrix heatmap of LSTM test MAE across all 12 stations and 5 horizons.
    """
    save_path = _ensure_dir(save_path)
    pivot_mae = detailed_df.pivot(index="station", columns="horizon", values="lstm_mae")
    # Sort horizons correctly
    col_order = ["1h", "3h", "6h", "12h", "24h"]
    pivot_mae = pivot_mae[[c for c in col_order if c in pivot_mae.columns]]

    # Sort stations by 1h MAE
    pivot_mae = pivot_mae.sort_values(by="1h", ascending=True)

    pivot_r2 = detailed_df.pivot(index="station", columns="horizon", values="lstm_r2")[pivot_mae.columns]
    pivot_r2 = pivot_r2.loc[pivot_mae.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Heatmap 1: MAE
    sns.heatmap(pivot_mae, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "MAE (µg/m³)"}, ax=ax1, linewidths=0.5)
    ax1.set_title("LSTM Forecast MAE by Station & Horizon (µg/m³)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Monitoring Station")
    ax1.set_xlabel("Forecast Horizon")

    # Heatmap 2: R2
    sns.heatmap(pivot_r2, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "Coefficient of Determination (R²)"}, ax=ax2, linewidths=0.5)
    ax2.set_title("LSTM Explained Variance (R²) by Station & Horizon", fontsize=12, fontweight="bold")
    ax2.set_ylabel("")
    ax2.set_xlabel("Forecast Horizon")

    fig.suptitle("Station-Wise Multi-Horizon Performance Across All 12 Beijing Sites", fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 8. Comprehensive Residual Diagnostics
# ---------------------------------------------------------------------------

def plot_residual_diagnostics(
    record: StationEvaluationRecord,
    diurnal_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Comprehensive 4-panel residual diagnostic suite:
    1. Residuals vs Predicted PM2.5 (heteroskedasticity check)
    2. Residuals time series trace with ±1 and ±2 standard deviation bounds
    3. Diurnal error profile by hour of day (0-23)
    4. MBE bias by concentration tier
    """
    save_path = _ensure_dir(save_path)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: Residuals vs Predicted (t+1h)
    yt_1h = record.y_true[:, 0]
    yp_1h = record.y_pred_lstm[:, 0]
    res_1h = yp_1h - yt_1h

    axes[0, 0].scatter(yp_1h, res_1h, alpha=0.25, color="#1f77b4", s=15, edgecolors="none")
    axes[0, 0].axhline(0, color="red", linestyle="--", lw=1.5, label="Zero Error")
    axes[0, 0].set_xlabel("Predicted PM2.5 (µg/m³)")
    axes[0, 0].set_ylabel("Residual: (y_pred - y_true) [µg/m³]")
    axes[0, 0].set_title(f"Residuals vs. Predicted PM2.5 (t+1h) — {record.station_name}", fontsize=11, fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)
    axes[0, 0].legend(loc="upper right")

    # Panel 2: Residuals over time (last 300 test hours)
    n_pts = min(300, len(record.timestamps))
    ts_sub = record.timestamps[-n_pts:]
    res_sub = res_1h[-n_pts:]
    std_res = float(np.std(res_1h))

    axes[0, 1].plot(ts_sub, res_sub, color="#1f77b4", lw=1.5, label="Residuals (t+1h)")
    axes[0, 1].axhline(0, color="black", linestyle="-", lw=1.0)
    axes[0, 1].axhline(std_res, color="orange", linestyle="--", label=f"+1σ ({std_res:.1f})")
    axes[0, 1].axhline(-std_res, color="orange", linestyle="--", label=f"-1σ (-{std_res:.1f})")
    axes[0, 1].fill_between(ts_sub, -2 * std_res, 2 * std_res, color="orange", alpha=0.1, label="±2σ Error Envelope")

    axes[0, 1].set_xlabel("Timestamp")
    axes[0, 1].set_ylabel("Residual (µg/m³)")
    axes[0, 1].set_title("Residual Temporal Stability (Recent Test Period)", fontsize=11, fontweight="bold")
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend(loc="lower left", fontsize=8.5)

    # Panel 3: Diurnal MAE Profile (Hour of day 0-23 across horizons)
    for h in [1, 6, 24]:
        sub_d = diurnal_df[diurnal_df["horizon"] == f"{h}h"]
        if not sub_d.empty:
            axes[1, 0].plot(sub_d["hour_of_day"], sub_d["mae"], marker="o", lw=1.8, label=f"t+{h}h MAE")

    axes[1, 0].set_xlabel("Hour of Day (0–23)")
    axes[1, 0].set_ylabel("MAE (µg/m³)")
    axes[1, 0].set_title("Diurnal Error Profile Across All Stations", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(range(0, 24, 2))
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)
    axes[1, 0].legend(loc="upper left")

    # Panel 4: Diurnal Bias Profile (MBE by Hour)
    for h in [1, 6, 24]:
        sub_d = diurnal_df[diurnal_df["horizon"] == f"{h}h"]
        if not sub_d.empty:
            axes[1, 1].plot(sub_d["hour_of_day"], sub_d["mbe"], marker="s", lw=1.8, label=f"t+{h}h MBE")

    axes[1, 1].axhline(0, color="black", linestyle="--", alpha=0.6)
    axes[1, 1].set_xlabel("Hour of Day (0–23)")
    axes[1, 1].set_ylabel("Mean Bias Error (µg/m³)")
    axes[1, 1].set_title("Diurnal Mean Bias Error Profile (Over/Under-Prediction)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks(range(0, 24, 2))
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)
    axes[1, 1].legend(loc="upper left")

    fig.suptitle("Phase 2 LSTM Residual Diagnostics and Error Stratifications", fontsize=14, y=0.99)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Master Visualization Pipeline Runner
# ---------------------------------------------------------------------------

def generate_all_evaluation_figures(
    suite: MultiStationEvaluationSuite,
    figures_dir: Union[str, Path] = "reports/figures/phase2",
    primary_station: str = "Aotizhongxin",
) -> Dict[str, str]:
    """
    Generate all 8 required evaluation and error analysis figures.
    """
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    created_figures: Dict[str, str] = {}

    rep_record = suite.station_records.get(primary_station, next(iter(suite.station_records.values())))

    # 1. Actual vs Predicted Timeseries
    p1 = fig_dir / "01_actual_vs_predicted_timeseries.png"
    plot_actual_vs_predicted_timeseries(rep_record, start_idx=200, num_points=168, save_path=p1)
    created_figures["timeseries"] = str(p1)

    # 2. Prediction Error Distribution
    p2 = fig_dir / "02_prediction_error_distribution.png"
    plot_error_distribution(rep_record, save_path=p2)
    created_figures["error_distribution"] = str(p2)

    # 3. Training vs Validation Loss Curve
    hist_file = Path("reports/results/phase2/history") / f"lstm_{primary_station}_w{suite.window_size}_u64_l1_history.json"
    if not hist_file.is_file():
        # Fallback to tuning history
        hist_file = Path("reports/results/phase2/tuning_history") / f"lstm_{primary_station}_w{suite.window_size}_u64_l1_history.json"

    p3 = fig_dir / "03_training_vs_validation_loss.png"
    if hist_file.is_file():
        plot_training_and_validation_loss(hist_file, station_name=primary_station, save_path=p3)
        created_figures["loss_curve"] = str(p3)

    # 4. MAE Comparison Across Models & Horizons
    p4 = fig_dir / "04_mae_comparison_across_models.png"
    plot_mae_comparison_across_models(suite.macro_summary_df, suite.phase1_comparison_1h_df, save_path=p4)
    created_figures["mae_comparison"] = str(p4)

    # 5. RMSE Comparison Across Models & Horizons
    p5 = fig_dir / "05_rmse_comparison_across_models.png"
    plot_rmse_comparison_across_models(suite.macro_summary_df, suite.pooled_summary_df, save_path=p5)
    created_figures["rmse_comparison"] = str(p5)

    # 6. Forecast Horizon vs Error Growth
    p6 = fig_dir / "06_forecast_horizon_vs_mae.png"
    plot_forecast_horizon_vs_error(suite.macro_summary_df, save_path=p6)
    created_figures["horizon_degradation"] = str(p6)

    # 7. Station-Wise Comparison
    p7 = fig_dir / "07_station_wise_performance_comparison.png"
    plot_station_wise_comparison(suite.detailed_metrics_df, save_path=p7)
    created_figures["station_comparison"] = str(p7)

    # 8. Residual Diagnostics
    p8 = fig_dir / "08_residual_analysis.png"
    plot_residual_diagnostics(rep_record, suite.diurnal_error_df, save_path=p8)
    created_figures["residual_diagnostics"] = str(p8)

    plt.close("all")
    logger.info("Generated %d publication-quality figures in %s", len(created_figures), fig_dir)
    return created_figures


__all__ = [
    "plot_actual_vs_predicted_timeseries",
    "plot_error_distribution",
    "plot_training_and_validation_loss",
    "plot_mae_comparison_across_models",
    "plot_rmse_comparison_across_models",
    "plot_forecast_horizon_vs_error",
    "plot_station_wise_comparison",
    "plot_residual_diagnostics",
    "generate_all_evaluation_figures",
]
