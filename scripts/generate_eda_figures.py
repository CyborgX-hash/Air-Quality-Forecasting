"""
Generate and save all high-resolution figures for Exploratory Data Analysis (EDA).
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data.load_data import load_all_stations, load_station_data

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)


def generate_all_figures():
    print("Loading all 12 stations dataset...")
    df_all = load_all_stations("data/raw")
    df_aoti = df_all[df_all["station"] == "Aotizhongxin"].copy()

    # --- FIGURE 1: Missing values ---
    print("Generating Figure 1: Missing values...")
    null_summary = df_all.isnull().sum() / len(df_all) * 100
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cols_with_nulls = null_summary[null_summary > 0].sort_values(ascending=False)
    bars = ax.bar(cols_with_nulls.index, cols_with_nulls.values, color="#3498db", edgecolor="#2980b9", width=0.6)
    ax.set_ylabel("Missing Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("UCI Beijing Air Quality Dataset: Missing Value Percentages (All 12 Stations, N=420,768)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(cols_with_nulls.values) * 1.25)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.12, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "01_missing_values_by_station.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

    # --- FIGURE 2: PM2.5 Distribution and Outliers ---
    print("Generating Figure 2: PM2.5 Distribution and Outliers...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pm25_mean = df_aoti["PM2.5"].mean()
    pm25_median = df_aoti["PM2.5"].median()
    sns.histplot(df_aoti["PM2.5"].dropna(), bins=60, kde=True, ax=axes[0], color="#2c3e50", stat="density")
    axes[0].axvline(pm25_mean, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Mean: {pm25_mean:.1f} µg/m³")
    axes[0].axvline(pm25_median, color="#27ae60", linestyle="-", linewidth=1.8, label=f"Median: {pm25_median:.1f} µg/m³")
    axes[0].axvline(75, color="#f39c12", linestyle=":", linewidth=1.5, label="China 24h Standard: 75 µg/m³")
    axes[0].set_title("PM2.5 Concentration Density (Aotizhongxin Station)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("PM2.5 (µg/m³)", fontsize=10)
    axes[0].legend(loc="upper right", frameon=True)

    sns.boxplot(data=df_all, x="station", y="PM2.5", ax=axes[1], palette="tab20", flierprops=dict(marker=".", markersize=2, alpha=0.3))
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_title("PM2.5 Distribution Across All 12 Stations", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Station", fontsize=10)
    axes[1].set_ylabel("PM2.5 (µg/m³)", fontsize=10)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "02_pm25_distribution_and_boxplots.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")

    # --- FIGURE 3: Longitudinal PM2.5 Time Series ---
    print("Generating Figure 3: Time Series Trend...")
    fig, ax = plt.subplots(figsize=(14, 5))
    df_aoti_sorted = df_aoti.sort_values(by="datetime").set_index("datetime")
    df_aoti_sorted["PM2.5_30d"] = df_aoti_sorted["PM2.5"].rolling(window=24 * 30, min_periods=24 * 7).mean()
    ax.plot(df_aoti_sorted.index, df_aoti_sorted["PM2.5"], alpha=0.2, color="#7f8c8d", label="Hourly Raw Measurement")
    ax.plot(df_aoti_sorted.index, df_aoti_sorted["PM2.5_30d"], color="#c0392b", linewidth=2.2, label="30-Day Moving Average Trend")
    ax.set_title("Longitudinal PM2.5 Time Series: 2013-03 to 2017-02 (Aotizhongxin)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=10)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "03_pm25_time_series_trend.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved: {fig3_path}")

    # --- FIGURE 4: Diurnal Daily Pattern ---
    print("Generating Figure 4: Diurnal Daily Pattern...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_all["season"] = df_all["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    })

    diurnal_overall = df_all.groupby("hour")["PM2.5"].agg(["mean", "median", "std"]).reset_index()
    axes[0].plot(diurnal_overall["hour"], diurnal_overall["mean"], marker="o", color="#2980b9", linewidth=2, label="Mean")
    axes[0].plot(diurnal_overall["hour"], diurnal_overall["median"], marker="s", color="#27ae60", linewidth=2, label="Median")
    axes[0].fill_between(
        diurnal_overall["hour"],
        diurnal_overall["mean"] - 0.3 * diurnal_overall["std"],
        diurnal_overall["mean"] + 0.3 * diurnal_overall["std"],
        color="#2980b9", alpha=0.15, label="±0.3 Std Dev",
    )
    axes[0].set_title("Diurnal Profile Across All Stations", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Hour of Day (0–23)", fontsize=10)
    axes[0].set_ylabel("PM2.5 (µg/m³)", fontsize=10)
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend(loc="lower right")

    sns.lineplot(data=df_all, x="hour", y="PM2.5", hue="season", ax=axes[1], errorbar=None, palette=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"])
    axes[1].set_title("Diurnal PM2.5 Profile Stratified by Season", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Hour of Day (0–23)", fontsize=10)
    axes[1].set_ylabel("Mean PM2.5 (µg/m³)", fontsize=10)
    axes[1].set_xticks(range(0, 24, 2))
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "04_pm25_diurnal_daily_pattern.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"Saved: {fig4_path}")

    # --- FIGURE 5: Monthly and Seasonal Pattern ---
    print("Generating Figure 5: Monthly and Seasonal Pattern...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df_all, x="month", y="PM2.5", ax=axes[0], palette="coolwarm", capsize=0.1, err_kws={"linewidth": 1.2})
    axes[0].set_title("Monthly Mean PM2.5 Concentration", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Month of Year", fontsize=10)
    axes[0].set_ylabel("Mean PM2.5 (µg/m³)", fontsize=10)
    axes[0].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    sns.boxplot(data=df_all, x="season", y="PM2.5", order=season_order, ax=axes[1], palette="Set2", showfliers=False)
    axes[1].set_title("PM2.5 Distribution by Season (Outliers Suppressed)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Season", fontsize=10)
    axes[1].set_ylabel("PM2.5 (µg/m³)", fontsize=10)
    plt.tight_layout()
    fig5_path = FIGURES_DIR / "05_pm25_monthly_seasonal_pattern.png"
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f"Saved: {fig5_path}")

    # --- FIGURE 6: Correlation Heatmap ---
    print("Generating Figure 6: Correlation Heatmap...")
    corr_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    corr_mat = df_all[corr_cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.6, vmax=1.0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Pearson Correlation Matrix: Air Pollutants & Meteorological Drivers", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig6_path = FIGURES_DIR / "06_correlation_heatmap.png"
    plt.savefig(fig6_path, dpi=200)
    plt.close()
    print(f"Saved: {fig6_path}")

    # --- FIGURE 7: PM2.5 vs Weather Drivers ---
    print("Generating Figure 7: PM2.5 vs Weather Drivers...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    df_all["wspm_bin"] = pd.cut(df_all["WSPM"], bins=[0, 1, 2, 3, 5, 8, 15], labels=["0-1", "1-2", "2-3", "3-5", "5-8", ">8"])
    sns.barplot(data=df_all, x="wspm_bin", y="PM2.5", ax=axes[0, 0], palette="Blues_r")
    axes[0, 0].set_title("Wind Speed Dispersion Effect on PM2.5", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Wind Speed (m/s)", fontsize=10)
    axes[0, 0].set_ylabel("Mean PM2.5 (µg/m³)", fontsize=10)

    df_all["temp_bin"] = pd.cut(df_all["TEMP"], bins=[-30, -10, 0, 10, 20, 30, 45], labels=["<-10°C", "-10 to 0", "0 to 10", "10 to 20", "20 to 30", ">30°C"])
    sns.barplot(data=df_all, x="temp_bin", y="PM2.5", ax=axes[0, 1], palette="viridis")
    axes[0, 1].set_title("Temperature Regime vs PM2.5 Concentration", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Temperature (°C)", fontsize=10)
    axes[0, 1].set_ylabel("Mean PM2.5 (µg/m³)", fontsize=10)

    df_all["rain_flag"] = (df_all["RAIN"] > 0).map({True: "Rain > 0 mm", False: "Dry (No Rain)"})
    sns.boxplot(data=df_all, x="rain_flag", y="PM2.5", ax=axes[1, 0], palette=["#e67e22", "#3498db"], showfliers=False)
    axes[1, 0].set_title("Precipitation Wash-out Effect (Wet Deposition)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Weather Condition", fontsize=10)
    axes[1, 0].set_ylabel("PM2.5 (µg/m³)", fontsize=10)

    df_all["dewp_bin"] = pd.qcut(df_all["DEWP"].dropna(), q=5, labels=["Very Low", "Low", "Moderate", "High", "Very High"])
    sns.barplot(data=df_all, x="dewp_bin", y="PM2.5", ax=axes[1, 1], palette="mako")
    axes[1, 1].set_title("Dew Point (Atmospheric Moisture) vs PM2.5", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Dew Point Quintile", fontsize=10)
    axes[1, 1].set_ylabel("Mean PM2.5 (µg/m³)", fontsize=10)
    plt.tight_layout()
    fig7_path = FIGURES_DIR / "07_pm25_vs_weather.png"
    plt.savefig(fig7_path, dpi=200)
    plt.close()
    print(f"Saved: {fig7_path}")

    # --- FIGURE 8: PM2.5 by Wind Direction ---
    print("Generating Figure 8: PM2.5 by Wind Direction...")
    wind_dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    wind_agg = df_all.groupby("wd")["PM2.5"].mean().reindex(wind_dirs)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if ("N" in w and "E" not in w and "S" not in w) else "#e74c3c" if ("S" in w or "E" in w) else "#f39c12" for w in wind_dirs]
    bars = ax.bar(wind_dirs, wind_agg.values, color=colors, edgecolor="black", alpha=0.85)
    global_mean = df_all["PM2.5"].mean()
    ax.axhline(global_mean, color="black", linestyle="--", linewidth=1.5, label=f"Global Mean ({global_mean:.1f} µg/m³)")
    ax.set_title("Mean PM2.5 Concentration by Wind Direction (Transport Corridor)", fontsize=12, fontweight="bold")
    ax.set_xlabel("16-Point Wind Direction", fontsize=10)
    ax.set_ylabel("Mean PM2.5 (µg/m³)", fontsize=10)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig8_path = FIGURES_DIR / "08_pm25_by_wind_direction.png"
    plt.savefig(fig8_path, dpi=200)
    plt.close()
    print(f"Saved: {fig8_path}")

    # --- FIGURE 9: PM2.5 vs Other Pollutants ---
    print("Generating Figure 9: PM2.5 vs Other Pollutants...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sample_df = df_all.dropna(subset=["PM2.5", "CO", "NO2", "O3"]).sample(n=5000, random_state=42)

    sns.regplot(data=sample_df, x="CO", y="PM2.5", ax=axes[0], scatter_kws={"alpha": 0.25, "color": "#8e44ad", "s": 10}, line_kws={"color": "#2c3e50", "linewidth": 2})
    r_co = df_all["PM2.5"].corr(df_all["CO"])
    axes[0].set_title(f"PM2.5 vs CO (r = {r_co:.2f})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("CO (µg/m³)", fontsize=10)
    axes[0].set_ylabel("PM2.5 (µg/m³)", fontsize=10)

    sns.regplot(data=sample_df, x="NO2", y="PM2.5", ax=axes[1], scatter_kws={"alpha": 0.25, "color": "#d35400", "s": 10}, line_kws={"color": "#2c3e50", "linewidth": 2})
    r_no2 = df_all["PM2.5"].corr(df_all["NO2"])
    axes[1].set_title(f"PM2.5 vs NO2 (r = {r_no2:.2f})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("NO2 (µg/m³)", fontsize=10)
    axes[1].set_ylabel("PM2.5 (µg/m³)", fontsize=10)

    sns.regplot(data=sample_df, x="O3", y="PM2.5", ax=axes[2], scatter_kws={"alpha": 0.25, "color": "#16a085", "s": 10}, line_kws={"color": "#2c3e50", "linewidth": 2})
    r_o3 = df_all["PM2.5"].corr(df_all["O3"])
    axes[2].set_title(f"PM2.5 vs O3 (r = {r_o3:.2f})", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("O3 (µg/m³)", fontsize=10)
    axes[2].set_ylabel("PM2.5 (µg/m³)", fontsize=10)

    plt.tight_layout()
    fig9_path = FIGURES_DIR / "09_pm25_vs_pollutants.png"
    plt.savefig(fig9_path, dpi=200)
    plt.close()
    print(f"Saved: {fig9_path}")

    print("All 9 figures generated successfully!")


if __name__ == "__main__":
    generate_all_figures()
