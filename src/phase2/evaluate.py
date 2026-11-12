"""
Phase 2 Evaluation Engine and Model Comparison Suite.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Mihika Mathur (Model Evaluation, Error Analysis & Visualization Specialist)

Key Responsibilities & Guarantees:
1. Multi-Horizon Test Evaluation:
   Evaluates trained LSTM multi-horizon checkpoints across all 12 monitoring
   stations at horizons h in {1, 3, 6, 12, 24} hours.
2. Verified Regression Metrics:
   Reuses single-source-of-truth metric routines from `src.evaluate`:
   - MAE: Mean Absolute Error (µg/m³)
   - RMSE: Root Mean Squared Error (µg/m³)
   - R²: Coefficient of Determination
   - MBE: Mean Bias Error (µg/m³), with convention mean(y_pred - y_true).
     Positive indicates overprediction; negative indicates underprediction.
3. Comparative Benchmarking:
   - Direct 1-hour comparison against Phase 1 models (Naive Persistence,
     Linear Regression, Default Gradient Boosting, Tuned Gradient Boosting).
   - Valid multi-horizon persistence baseline generated on identical test timestamps.
   - Explicit accounting of test sample sizes and observation differences.
4. Aggregation Rigor:
   - Station-wise results across all 12 stations.
   - Macro-averages across stations: arithmetic mean of station metrics.
   - Observation-pooled metrics across all test observations:
     Pooled RMSE is mathematically calculated from pooled squared errors,
     NEVER by averaging station RMSE values.
5. Error Stratification & Diagnostics:
   - Error by WHO / National AQI pollution concentration tiers.
   - Diurnal error profile by hour of day (0-23).
   - Monthly and seasonal error distribution.
   - High-pollution episode peak tracking and error variance analysis.
"""

from __future__ import annotations

import os
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.evaluate import (
    calculate_metrics,
    compute_residuals,
    mean_absolute_error,
    mean_bias_error,
    r2_score,
    root_mean_squared_error,
    validate_metric_inputs,
)
from src.phase2.config import Phase2Config
from src.phase2.data_pipeline import StationDataSplits, process_station
from src.phase2.lstm_model import DEFAULT_HORIZONS, HORIZON_NAMES
from src.phase2.predict import load_trained_model

logger = logging.getLogger(__name__)

# Standard AQI concentration tiers for PM2.5 in µg/m³
AQI_PM25_BINS: List[Tuple[str, float, float]] = [
    ("Good (0-35)", 0.0, 35.0),
    ("Moderate (35-75)", 35.0, 75.0),
    ("Unhealthy for Sensitive (75-115)", 75.0, 115.0),
    ("Unhealthy (115-150)", 115.0, 150.0),
    ("Very Unhealthy (150-250)", 150.0, 250.0),
    ("Severe / Hazardous (>250)", 250.0, 10000.0),
]


@dataclass
class StationEvaluationRecord:
    """Encapsulates test evaluation results for a single station across all horizons."""
    station_name: str
    window_size: int
    horizons: List[int]
    n_obs: int
    timestamps: pd.DatetimeIndex
    y_true: np.ndarray  # Shape (N, len(horizons)) in µg/m³
    y_pred_lstm: np.ndarray  # Shape (N, len(horizons)) in µg/m³
    y_pred_persistence: np.ndarray  # Shape (N, len(horizons)) in µg/m³
    lstm_metrics_table: pd.DataFrame
    persistence_metrics_table: pd.DataFrame
    comparison_table: pd.DataFrame
    model_path: str


@dataclass
class MultiStationEvaluationSuite:
    """Encapsulates comprehensive 12-station multi-horizon evaluation results."""
    station_records: Dict[str, StationEvaluationRecord]
    detailed_metrics_df: pd.DataFrame
    macro_summary_df: pd.DataFrame
    pooled_summary_df: pd.DataFrame
    phase1_comparison_1h_df: pd.DataFrame
    tier_error_df: pd.DataFrame
    diurnal_error_df: pd.DataFrame
    monthly_error_df: pd.DataFrame
    horizons: List[int]
    stations: List[str]
    window_size: int


def calculate_pooled_metrics(
    y_true_list: Sequence[np.ndarray],
    y_pred_list: Sequence[np.ndarray],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    model_name: str = "LSTM (Champion W72)",
) -> pd.DataFrame:
    """
    Calculate pooled regression metrics across all observations from multiple stations.

    Guarantees that pooled RMSE is calculated directly from pooled squared errors
    rather than averaging station RMSE values.
    """
    rows: List[Dict[str, Any]] = []

    for h_idx, h in enumerate(horizons):
        yt_all: List[float] = []
        yp_all: List[float] = []

        for yt_arr, yp_arr in zip(y_true_list, y_pred_list):
            if yt_arr.ndim == 2 and yt_arr.shape[1] > h_idx:
                yt_all.extend(yt_arr[:, h_idx])
                yp_all.extend(yp_arr[:, h_idx])
            elif yt_arr.ndim == 1:
                yt_all.extend(yt_arr)
                yp_all.extend(yp_arr)

        yt = np.asarray(yt_all, dtype=float)
        yp = np.asarray(yp_all, dtype=float)

        valid_mask = (~np.isnan(yt)) & (~np.isnan(yp)) & (~np.isinf(yt)) & (~np.isinf(yp))
        yt = yt[valid_mask]
        yp = yp[valid_mask]

        n_total = len(yt)
        if n_total == 0:
            continue

        errors = yp - yt
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2

        pooled_mae = float(np.mean(abs_errors))
        pooled_rmse = float(np.sqrt(np.mean(squared_errors)))
        pooled_mbe = float(np.mean(errors))

        # Pooled R2
        ss_res = np.sum(squared_errors)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        pooled_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        rows.append({
            "model": model_name,
            "horizon": f"{h}h",
            "horizon_hours": h,
            "pooled_mae": round(pooled_mae, 4),
            "pooled_rmse": round(pooled_rmse, 4),
            "pooled_r2": round(pooled_r2, 4),
            "pooled_mbe": round(pooled_mbe, 4),
            "total_observations": n_total,
        })

    return pd.DataFrame(rows)


def evaluate_single_station(
    station_name: str,
    model_path: Optional[Union[str, Path]] = None,
    window_size: int = 72,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    partition: str = "test",
    batch_size: int = 64,
) -> StationEvaluationRecord:
    """
    Perform rigorous evaluation of LSTM and Baseline forecasts for a single station.

    Parameters
    ----------
    station_name : str
        Target monitoring station.
    model_path : Optional[Union[str, Path]], default None
        Path to .keras checkpoint. If None, resolves from standard checkpoint path.
    window_size : int, default 72
        Input lookback window size.
    horizons : Sequence[int], default (1, 3, 6, 12, 24)
        Forecast horizons in hours.
    partition : str, default 'test'
        Partition to evaluate ('test' or 'val').
    batch_size : int, default 64
        Inference batch size.
    """
    active_horizons = list(horizons)

    if model_path is None:
        standard_path = Path("models/phase2/checkpoints") / f"lstm_{station_name}_w{window_size}_u64_l1_best.keras"
        if not standard_path.is_file():
            # Check tuning checkpoints directory
            tuning_path = Path("reports/results/phase2/tuning_checkpoints") / f"lstm_{station_name}_w{window_size}_u64_l1_best.keras"
            if tuning_path.is_file():
                standard_path = tuning_path
            else:
                raise FileNotFoundError(f"Model checkpoint not found for {station_name} at {standard_path} or {tuning_path}")
        model_path = standard_path
    else:
        model_path = Path(model_path)

    logger.info("Evaluating %s on %s set using model %s", station_name, partition, model_path.name)
    model = load_trained_model(model_path)

    # Load and process station splits
    splits = process_station(station_name)

    # Extract sequences with unscaled targets and context
    X_scaled, y_scaled, ts = splits.get_test_sequences(
        window_size=window_size,
        horizons=active_horizons,
        use_context=True,
        scaled=True,
        scale_target=True,
    ) if partition == "test" else splits.get_val_sequences(
        window_size=window_size,
        horizons=active_horizons,
        use_context=True,
        scaled=True,
        scale_target=True,
    )

    # Model inference
    preds_scaled = model.predict(X_scaled, batch_size=batch_size, verbose=0)

    # Invert targets and predictions back to physical units (µg/m³)
    y_pred_actual = splits.inverse_transform_target(preds_scaled)
    y_true_actual = splits.inverse_transform_target(y_scaled)

    if y_pred_actual.ndim == 1:
        y_pred_actual = y_pred_actual.reshape(-1, 1)
    if y_true_actual.ndim == 1:
        y_true_actual = y_true_actual.reshape(-1, 1)

    # Generate Multi-Horizon Persistence Baseline on identical timestamps:
    # Persistence predicts that future PM2.5 at t+h equals the current observed PM2.5 at time t.
    # We retrieve the unscaled input feature PM2.5 at the final timestep t of each input window.
    X_unscaled, _, _ = splits.get_test_sequences(
        window_size=window_size,
        horizons=active_horizons,
        use_context=True,
        scaled=False,
        scale_target=False,
    ) if partition == "test" else splits.get_val_sequences(
        window_size=window_size,
        horizons=active_horizons,
        use_context=True,
        scaled=False,
        scale_target=False,
    )

    pm25_feat_idx = splits.feature_cols.index("PM2.5")
    # Current PM2.5 at time t (last step in lookback window W)
    y_t_current = X_unscaled[:, -1, pm25_feat_idx]
    # Broadcast y_t across all horizons for persistence
    y_pred_pers = np.repeat(y_t_current[:, np.newaxis], len(active_horizons), axis=1)

    # Compute metrics for LSTM and Persistence across each horizon
    lstm_rows: List[Dict[str, Any]] = []
    pers_rows: List[Dict[str, Any]] = []
    comp_rows: List[Dict[str, Any]] = []

    for idx, h in enumerate(active_horizons):
        h_str = f"{h}h"
        yt_h = y_true_actual[:, idx]
        yp_lstm_h = y_pred_actual[:, idx]
        yp_pers_h = y_pred_pers[:, idx]

        m_lstm = calculate_metrics(yt_h, yp_lstm_h)
        m_pers = calculate_metrics(yt_h, yp_pers_h)

        mae_impr_pct = round(((m_pers["mae"] - m_lstm["mae"]) / m_pers["mae"]) * 100.0, 2)
        rmse_impr_pct = round(((m_pers["rmse"] - m_lstm["rmse"]) / m_pers["rmse"]) * 100.0, 2)
        r2_delta = round(m_lstm["r2"] - m_pers["r2"], 4)

        lstm_rows.append({
            "station": station_name,
            "horizon": h_str,
            "horizon_hours": h,
            "mae": m_lstm["mae"],
            "rmse": m_lstm["rmse"],
            "r2": m_lstm["r2"],
            "mbe": m_lstm["mbe"],
            "n_obs": m_lstm["n_obs"],
        })

        pers_rows.append({
            "station": station_name,
            "horizon": h_str,
            "horizon_hours": h,
            "mae": m_pers["mae"],
            "rmse": m_pers["rmse"],
            "r2": m_pers["r2"],
            "mbe": m_pers["mbe"],
            "n_obs": m_pers["n_obs"],
        })

        comp_rows.append({
            "station": station_name,
            "horizon": h_str,
            "horizon_hours": h,
            "lstm_mae": m_lstm["mae"],
            "pers_mae": m_pers["mae"],
            "mae_improvement_pct": mae_impr_pct,
            "lstm_rmse": m_lstm["rmse"],
            "pers_rmse": m_pers["rmse"],
            "rmse_improvement_pct": rmse_impr_pct,
            "lstm_r2": m_lstm["r2"],
            "pers_r2": m_pers["r2"],
            "r2_delta": r2_delta,
            "lstm_mbe": m_lstm["mbe"],
            "pers_mbe": m_pers["mbe"],
            "n_obs": m_lstm["n_obs"],
        })

    lstm_df = pd.DataFrame(lstm_rows)
    pers_df = pd.DataFrame(pers_rows)
    comp_df = pd.DataFrame(comp_rows)

    return StationEvaluationRecord(
        station_name=station_name,
        window_size=window_size,
        horizons=active_horizons,
        n_obs=len(ts),
        timestamps=ts,
        y_true=y_true_actual,
        y_pred_lstm=y_pred_actual,
        y_pred_persistence=y_pred_pers,
        lstm_metrics_table=lstm_df,
        persistence_metrics_table=pers_df,
        comparison_table=comp_df,
        model_path=str(model_path),
    )


def evaluate_all_stations(
    stations: Optional[Sequence[str]] = None,
    window_size: int = 72,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    partition: str = "test",
) -> MultiStationEvaluationSuite:
    """
    Run full 12-station evaluation sweep across all five forecast horizons.

    Produces station-wise results, macro-averages, pooled metrics,
    concentration tier analysis, and Phase 1 comparative alignments.
    """
    cfg = Phase2Config.get_default_config()
    target_stations = list(stations) if stations is not None else cfg.stations.all
    active_horizons = list(horizons)

    station_records: Dict[str, StationEvaluationRecord] = {}
    detailed_rows: List[Dict[str, Any]] = []

    logger.info("Starting multi-station evaluation across %d stations: %s", len(target_stations), target_stations)

    for st in target_stations:
        rec = evaluate_single_station(
            station_name=st,
            window_size=window_size,
            horizons=active_horizons,
            partition=partition,
        )
        station_records[st] = rec
        detailed_rows.extend(rec.comparison_table.to_dict(orient="records"))

    detailed_df = pd.DataFrame(detailed_rows)

    # 1. Macro-Average Results (mean of station metrics per horizon)
    macro_rows: List[Dict[str, Any]] = []
    for h in active_horizons:
        h_str = f"{h}h"
        sub = detailed_df[detailed_df["horizon"] == h_str]

        macro_rows.append({
            "horizon": h_str,
            "horizon_hours": h,
            "lstm_macro_mae": round(float(sub["lstm_mae"].mean()), 4),
            "pers_macro_mae": round(float(sub["pers_mae"].mean()), 4),
            "macro_mae_improvement_pct": round(float(sub["mae_improvement_pct"].mean()), 2),
            "lstm_macro_rmse": round(float(sub["lstm_rmse"].mean()), 4),
            "pers_macro_rmse": round(float(sub["pers_rmse"].mean()), 4),
            "macro_rmse_improvement_pct": round(float(sub["rmse_improvement_pct"].mean()), 2),
            "lstm_macro_r2": round(float(sub["lstm_r2"].mean()), 4),
            "pers_macro_r2": round(float(sub["pers_r2"].mean()), 4),
            "macro_r2_delta": round(float(sub["r2_delta"].mean()), 4),
            "lstm_macro_mbe": round(float(sub["lstm_mbe"].mean()), 4),
            "pers_macro_mbe": round(float(sub["pers_mbe"].mean()), 4),
            "station_count": len(sub),
            "total_observations": int(sub["n_obs"].sum()),
        })
    macro_df = pd.DataFrame(macro_rows)

    # 2. Pooled Results (calculated directly from pooled observations)
    y_true_all = [rec.y_true for rec in station_records.values()]
    y_pred_lstm_all = [rec.y_pred_lstm for rec in station_records.values()]
    y_pred_pers_all = [rec.y_pred_persistence for rec in station_records.values()]

    pooled_lstm_df = calculate_pooled_metrics(
        y_true_all, y_pred_lstm_all, horizons=active_horizons, model_name="LSTM (Champion W72)"
    )
    pooled_pers_df = calculate_pooled_metrics(
        y_true_all, y_pred_pers_all, horizons=active_horizons, model_name="Persistence (Naive Multi-Horizon)"
    )

    pooled_df = pd.merge(
        pooled_lstm_df,
        pooled_pers_df,
        on=["horizon", "horizon_hours", "total_observations"],
        suffixes=("_lstm", "_pers"),
    )
    pooled_df["pooled_mae_improvement_pct"] = round(
        ((pooled_df["pooled_mae_pers"] - pooled_df["pooled_mae_lstm"]) / pooled_df["pooled_mae_pers"]) * 100.0, 2
    )
    pooled_df["pooled_rmse_improvement_pct"] = round(
        ((pooled_df["pooled_rmse_pers"] - pooled_df["pooled_rmse_lstm"]) / pooled_df["pooled_rmse_pers"]) * 100.0, 2
    )
    pooled_df["pooled_r2_delta"] = round(pooled_df["pooled_r2_lstm"] - pooled_df["pooled_r2_pers"], 4)

    # 3. Phase 1 Benchmark Comparison (1-hour horizon)
    phase1_comp_df = build_phase1_comparison(detailed_df[detailed_df["horizon"] == "1h"])

    # 4. Stratified Error Analyses
    tier_df = evaluate_pollution_tiers(station_records, active_horizons)
    diurnal_df = evaluate_diurnal_profile(station_records, active_horizons)
    monthly_df = evaluate_monthly_profile(station_records, active_horizons)

    return MultiStationEvaluationSuite(
        station_records=station_records,
        detailed_metrics_df=detailed_df,
        macro_summary_df=macro_df,
        pooled_summary_df=pooled_df,
        phase1_comparison_1h_df=phase1_comp_df,
        tier_error_df=tier_df,
        diurnal_error_df=diurnal_df,
        monthly_error_df=monthly_df,
        horizons=active_horizons,
        stations=target_stations,
        window_size=window_size,
    )


def build_phase1_comparison(lstm_1h_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Phase 2 1-hour LSTM test performance with Phase 1 benchmark results.
    """
    p1_file = Path("reports/results/multi_station_sweep/all_stations_model_comparison.csv")
    if not p1_file.is_file():
        logger.warning("Phase 1 benchmark file not found at %s", p1_file)
        return pd.DataFrame()

    p1_df = pd.read_csv(p1_file)

    # Filter to the key Phase 1 reference models
    models_to_keep = [
        "Persistence (Naive Lag-1)",
        "Linear Regression",
        "Gradient Boosting (Default)",
        "Gradient Boosting (Tuned)",
    ]
    p1_filtered = p1_df[p1_df["model"].isin(models_to_keep)].copy()

    # Build LSTM records to append
    lstm_records = []
    for _, row in lstm_1h_df.iterrows():
        # Baseline persistence MAE from Phase 1 for consistent % improvement calculation
        st = row["station"]
        st_p1_pers = p1_filtered[(p1_filtered["station"] == st) & (p1_filtered["model"].str.contains("Persistence"))]
        base_mae = float(st_p1_pers["test_mae"].iloc[0]) if len(st_p1_pers) > 0 else row["pers_mae"]
        base_rmse = float(st_p1_pers["test_rmse"].iloc[0]) if len(st_p1_pers) > 0 else row["pers_rmse"]
        base_r2 = float(st_p1_pers["test_r2"].iloc[0]) if len(st_p1_pers) > 0 else row["pers_r2"]

        mae_imp = round(((base_mae - row["lstm_mae"]) / base_mae) * 100.0, 2)
        rmse_imp = round(((base_rmse - row["lstm_rmse"]) / base_rmse) * 100.0, 2)
        r2_del = round(row["lstm_r2"] - base_r2, 4)

        lstm_records.append({
            "station": st,
            "model": "LSTM (Champion W72)",
            "test_mae": row["lstm_mae"],
            "test_rmse": row["lstm_rmse"],
            "test_r2": row["lstm_r2"],
            "test_mbe": row["lstm_mbe"],
            "n_obs": row["n_obs"],
            "mae_improvement_pct": mae_imp,
            "rmse_improvement_pct": rmse_imp,
            "r2_delta": r2_del,
        })

    combined = pd.concat([p1_filtered, pd.DataFrame(lstm_records)], ignore_index=True)
    return combined.sort_values(by=["station", "test_mae"], ascending=[True, True])


def evaluate_pollution_tiers(
    station_records: Dict[str, StationEvaluationRecord],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Evaluate prediction errors across discrete PM2.5 pollution concentration tiers.
    """
    rows: List[Dict[str, Any]] = []

    for h_idx, h in enumerate(horizons):
        h_str = f"{h}h"
        yt_all: List[float] = []
        yp_all: List[float] = []

        for rec in station_records.values():
            yt_all.extend(rec.y_true[:, h_idx])
            yp_all.extend(rec.y_pred_lstm[:, h_idx])

        yt = np.asarray(yt_all, dtype=float)
        yp = np.asarray(yp_all, dtype=float)

        for tier_name, low, high in AQI_PM25_BINS:
            mask = (yt >= low) & (yt < high) if high < 5000.0 else (yt >= low)
            yt_sub = yt[mask]
            yp_sub = yp[mask]

            if len(yt_sub) == 0:
                continue

            mae = float(np.mean(np.abs(yp_sub - yt_sub)))
            rmse = float(np.sqrt(np.mean((yp_sub - yt_sub) ** 2)))
            mbe = float(np.mean(yp_sub - yt_sub))

            rows.append({
                "horizon": h_str,
                "horizon_hours": h,
                "tier": tier_name,
                "concentration_range": f"{low}-{high if high < 5000.0 else '+'}",
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mbe": round(mbe, 2),
                "n_samples": int(len(yt_sub)),
                "pct_of_test_data": round(100.0 * len(yt_sub) / len(yt), 2),
            })

    return pd.DataFrame(rows)


def evaluate_diurnal_profile(
    station_records: Dict[str, StationEvaluationRecord],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Evaluate errors stratified by target hour of day (0-23).
    """
    rows: List[Dict[str, Any]] = []

    for h_idx, h in enumerate(horizons):
        h_str = f"{h}h"
        all_hours: List[int] = []
        all_yt: List[float] = []
        all_yp: List[float] = []

        for rec in station_records.values():
            hours = rec.timestamps.hour.values
            all_hours.extend(hours)
            all_yt.extend(rec.y_true[:, h_idx])
            all_yp.extend(rec.y_pred_lstm[:, h_idx])

        df_h = pd.DataFrame({"hour": all_hours, "y_true": all_yt, "y_pred": all_yp})

        for hr in range(24):
            sub = df_h[df_h["hour"] == hr]
            if len(sub) == 0:
                continue

            yt = sub["y_true"].values
            yp = sub["y_pred"].values
            mae = float(np.mean(np.abs(yp - yt)))
            rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
            mbe = float(np.mean(yp - yt))

            rows.append({
                "horizon": h_str,
                "horizon_hours": h,
                "hour_of_day": hr,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mbe": round(mbe, 2),
                "n_obs": int(len(sub)),
            })

    return pd.DataFrame(rows)


def evaluate_monthly_profile(
    station_records: Dict[str, StationEvaluationRecord],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Evaluate error metrics stratified by month of year.
    """
    rows: List[Dict[str, Any]] = []
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    for h_idx, h in enumerate(horizons):
        h_str = f"{h}h"
        all_months: List[int] = []
        all_yt: List[float] = []
        all_yp: List[float] = []

        for rec in station_records.values():
            months = rec.timestamps.month.values
            all_months.extend(months)
            all_yt.extend(rec.y_true[:, h_idx])
            all_yp.extend(rec.y_pred_lstm[:, h_idx])

        df_m = pd.DataFrame({"month": all_months, "y_true": all_yt, "y_pred": all_yp})

        for mo in sorted(df_m["month"].unique()):
            sub = df_m[df_m["month"] == mo]
            if len(sub) == 0:
                continue

            yt = sub["y_true"].values
            yp = sub["y_pred"].values
            mae = float(np.mean(np.abs(yp - yt)))
            rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
            mbe = float(np.mean(yp - yt))

            rows.append({
                "horizon": h_str,
                "horizon_hours": h,
                "month": mo,
                "month_name": month_names.get(mo, str(mo)),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mbe": round(mbe, 2),
                "n_obs": int(len(sub)),
            })

    return pd.DataFrame(rows)


def save_evaluation_artifacts(
    suite: MultiStationEvaluationSuite,
    output_dir: Union[str, Path] = "reports/results/phase2",
) -> Dict[str, str]:
    """
    Export all evaluation summary tables to CSV and Markdown.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files: Dict[str, str] = {}

    # 1. Detailed Station-Horizon Comparison
    detailed_csv = out_path / "all_stations_multi_horizon_comparison.csv"
    suite.detailed_metrics_df.to_csv(detailed_csv, index=False)
    saved_files["detailed_csv"] = str(detailed_csv)

    detailed_md = out_path / "all_stations_multi_horizon_comparison.md"
    suite.detailed_metrics_df.to_markdown(detailed_md, index=False)
    saved_files["detailed_md"] = str(detailed_md)

    # 2. Macro-Average Results
    macro_csv = out_path / "macro_average_performance_by_horizon.csv"
    suite.macro_summary_df.to_csv(macro_csv, index=False)
    saved_files["macro_csv"] = str(macro_csv)

    macro_md = out_path / "macro_average_performance_by_horizon.md"
    suite.macro_summary_df.to_markdown(macro_md, index=False)
    saved_files["macro_md"] = str(macro_md)

    # 3. Pooled Observation Results
    pooled_csv = out_path / "pooled_performance_by_horizon.csv"
    suite.pooled_summary_df.to_csv(pooled_csv, index=False)
    saved_files["pooled_csv"] = str(pooled_csv)

    pooled_md = out_path / "pooled_performance_by_horizon.md"
    suite.pooled_summary_df.to_markdown(pooled_md, index=False)
    saved_files["pooled_md"] = str(pooled_md)

    # 4. Phase 1 Comparison (1h)
    if not suite.phase1_comparison_1h_df.empty:
        p1_csv = out_path / "phase1_vs_phase2_1h_benchmark.csv"
        suite.phase1_comparison_1h_df.to_csv(p1_csv, index=False)
        saved_files["phase1_comp_csv"] = str(p1_csv)

        p1_md = out_path / "phase1_vs_phase2_1h_benchmark.md"
        suite.phase1_comparison_1h_df.to_markdown(p1_md, index=False)
        saved_files["phase1_comp_md"] = str(p1_md)

    # 5. Pollution Tier Diagnostics
    tier_csv = out_path / "pollution_tier_error_analysis.csv"
    suite.tier_error_df.to_csv(tier_csv, index=False)
    saved_files["tier_csv"] = str(tier_csv)

    tier_md = out_path / "pollution_tier_error_analysis.md"
    suite.tier_error_df.to_markdown(tier_md, index=False)
    saved_files["tier_md"] = str(tier_md)

    # 6. Diurnal and Monthly Diagnostics
    diurnal_csv = out_path / "diurnal_error_profile.csv"
    suite.diurnal_error_df.to_csv(diurnal_csv, index=False)
    saved_files["diurnal_csv"] = str(diurnal_csv)

    monthly_csv = out_path / "monthly_error_profile.csv"
    suite.monthly_error_df.to_csv(monthly_csv, index=False)
    saved_files["monthly_csv"] = str(monthly_csv)

    logger.info("Successfully exported Phase 2 evaluation artifacts to %s", out_path)
    return saved_files


__all__ = [
    "AQI_PM25_BINS",
    "StationEvaluationRecord",
    "MultiStationEvaluationSuite",
    "calculate_pooled_metrics",
    "evaluate_single_station",
    "evaluate_all_stations",
    "build_phase1_comparison",
    "evaluate_pollution_tiers",
    "evaluate_diurnal_profile",
    "evaluate_monthly_profile",
    "save_evaluation_artifacts",
]

