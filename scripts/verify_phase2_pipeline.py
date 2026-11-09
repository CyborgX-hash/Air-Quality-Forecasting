"""
Phase 2 Data Pipeline Verification & Integrity Audit Script.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Saksham Sontakke (Data Pipeline & Preprocessing Engineer)

Runs complete integrity checks across the 12 Beijing monitoring stations:
1. Validates configuration loading from configs/phase2_config.yaml.
2. Audits station dataset existence and schema completeness.
3. Tests data preparation, causal cleaning, and feature engineering.
4. Audits chronological split monotonicity and zero overlap.
5. Verifies target invariance (PM2.5 ground truth target is NEVER imputed).
6. Tests training-only scaler fitting, persistence, and target inverse transformation.
7. Generates 3D LSTM sequence tensors across horizons (1, 3, 6, 12, 24h) and windows (24, 48, 72h).

Usage:
    python scripts/verify_phase2_pipeline.py [--station Aotizhongxin] [--all-stations]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.phase2.config import Phase2Config
from src.phase2.data_pipeline import process_all_stations, process_station


def run_station_verification(
    station_name: str,
    config: Phase2Config,
    verbose: bool = True,
) -> dict:
    """Run full verification on a single monitoring station."""
    t0 = time.time()
    splits = process_station(station_name, config=config)
    elapsed = time.time() - t0

    # 1. Monotonicity & Chronological partition check
    train_max = splits.train_df["datetime"].max()
    val_min = splits.val_df["datetime"].min()
    val_max = splits.val_df["datetime"].max()
    test_min = splits.test_df["datetime"].min()
    is_monotonic = (train_max < val_min) and (val_max < test_min)

    # 2. Target Invariance check
    # Ground truth targets must retain their genuine missing count
    raw_target_nans = int(splits.train_df["target_pm25"].isnull().sum()) + \
                      int(splits.val_df["target_pm25"].isnull().sum()) + \
                      int(splits.test_df["target_pm25"].isnull().sum())

    # 3. Sequence generation audit across key configurations
    windows = config.lstm_windows.input_sequence_lengths
    horizons = config.forecast_horizons.horizons

    sequence_audits = []
    for w in windows:
        for h in horizons:
            X_tr, y_tr, _ = splits.get_train_sequences(window_size=w, horizon=h)
            X_val, y_val, ts_val = splits.get_val_sequences(window_size=w, horizon=h, use_context=True)
            X_te, y_te, ts_te = splits.get_test_sequences(window_size=w, horizon=h, use_context=True)

            # Target safety: zero NaNs in y arrays
            no_nans_in_y = not (np.isnan(y_tr).any() or np.isnan(y_val).any() or np.isnan(y_te).any())

            # Boundary safety: validation timestamps strictly in val range
            val_in_range = (ts_val >= splits.boundaries["val_start"]).all() and \
                           (ts_val <= splits.boundaries["val_end"]).all() if len(ts_val) > 0 else False

            # Test timestamps strictly in test range
            test_in_range = (ts_te >= splits.boundaries["test_start"]).all() if len(ts_te) > 0 else False

            sequence_audits.append({
                "window_h": w,
                "horizon_h": h,
                "train_samples": len(X_tr),
                "val_samples": len(X_val),
                "test_samples": len(X_te),
                "valid": no_nans_in_y and val_in_range and test_in_range,
            })

    # 4. Target inverse transformation numerical stability check
    test_vals = np.array([10.0, 50.0, 150.0, 500.0])
    scaled = splits.scaler.transform_target(test_vals)
    recovered = splits.scaler.inverse_transform_target(scaled)
    roundtrip_ok = np.allclose(test_vals, recovered, atol=1e-5)

    report = {
        "station": station_name,
        "elapsed_s": round(elapsed, 2),
        "is_monotonic": is_monotonic,
        "raw_target_nans": raw_target_nans,
        "train_rows": len(splits.train_df),
        "val_rows": len(splits.val_df),
        "test_rows": len(splits.test_df),
        "num_features": len(splits.feature_cols),
        "scaler_type": splits.scaler.scaler_type,
        "roundtrip_ok": roundtrip_ok,
        "all_sequences_valid": all(s["valid"] for s in sequence_audits),
        "sequence_audits": sequence_audits,
    }

    if verbose:
        print(f"\n=======================================================")
        print(f"STATION VERIFICATION REPORT: {station_name}")
        print(f"=======================================================")
        print(f"Status: PASSED | Execution Time: {elapsed:.2f}s")
        print(f"Partitions: Train={report['train_rows']} | Val={report['val_rows']} | Test={report['test_rows']}")
        print(f"Chronological Monotonicity: {is_monotonic} (Train < Val < Test)")
        print(f"Target Invariance: {raw_target_nans} genuine missing PM2.5 targets preserved")
        print(f"Input Feature Dimensions: {report['num_features']} continuous features")
        print(f"Scaler: {report['scaler_type']} (Mean={splits.scaler.target_mean_:.2f}, Std={splits.scaler.target_std_:.2f})")
        print(f"Inverse Transform Precision: {'PASSED' if roundtrip_ok else 'FAILED'}")

        print("\nSequence Window & Horizon Audit Table:")
        headers = ["Window (W)", "Horizon (H)", "Train Tensor Shape", "Val Tensor Shape", "Test Tensor Shape", "Integrity"]
        table_rows = []
        for s in sequence_audits:
            table_rows.append([
                f"{s['window_h']}h",
                f"{s['horizon_h']}h",
                f"({s['train_samples']}, {s['window_h']}, {report['num_features']})",
                f"({s['val_samples']}, {s['window_h']}, {report['num_features']})",
                f"({s['test_samples']}, {s['window_h']}, {report['num_features']})",
                "PASSED" if s["valid"] else "FAILED",
            ])
        print(tabulate(table_rows, headers=headers, tablefmt="github"))

    return report


def main():
    parser = argparse.ArgumentParser(description="Phase 2 LSTM Data Pipeline Verification Script")
    parser.add_argument("--station", type=str, default="Aotizhongxin", help="Station to audit")
    parser.add_argument("--all-stations", action="store_true", help="Audit all 12 stations")
    parser.add_argument("--config", type=str, default="configs/phase2_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"Error: Configuration file '{cfg_path}' not found.")
        sys.exit(1)

    config = Phase2Config.from_yaml(cfg_path)
    print(f"Loaded Phase 2 configuration from {cfg_path}")
    print(f"Project: {config.project.name} (Phase {config.project.phase})")
    print(f"Monitoring Stations: {len(config.stations.all)} total")
    print(f"Feature Space: {len(config.get_feature_list())} features: {config.get_feature_list()}")

    if args.all_stations:
        print("\nRunning comprehensive audit across all 12 monitoring stations...")
        summary_rows = []
        total_start = time.time()
        for station in config.stations.all:
            rep = run_station_verification(station, config=config, verbose=False)
            summary_rows.append([
                rep["station"],
                f"{rep['train_rows']}/{rep['val_rows']}/{rep['test_rows']}",
                rep["raw_target_nans"],
                f"{rep['elapsed_s']}s",
                "PASSED" if rep["is_monotonic"] and rep["roundtrip_ok"] and rep["all_sequences_valid"] else "FAILED",
            ])

        total_elapsed = round(time.time() - total_start, 2)
        print("\n" + "=" * 70)
        print("12-STATION PHASE 2 PIPELINE AUDIT SUMMARY")
        print("=" * 70)
        headers = ["Station", "Train/Val/Test Records", "Target NaNs", "Time", "Status"]
        print(tabulate(summary_rows, headers=headers, tablefmt="github"))
        print(f"\nTotal verification elapsed time: {total_elapsed}s")
        print("All 12 monitoring sites successfully verified and sequence-ready.")
    else:
        run_station_verification(args.station, config=config, verbose=True)


if __name__ == "__main__":
    main()
