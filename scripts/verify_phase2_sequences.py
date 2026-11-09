"""
Phase 2 Multi-Horizon Sequence Generation Verification & Audit Script.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Ajit Kumar Prasad (Time-Series Sequence Engineer)

Audits sequence generation across the 12 Beijing monitoring stations:
1. Multi-horizon targets [t+1, t+3, t+6, t+12, t+24].
2. Input windows W in {24, 48, 72}.
3. Station isolation and zero cross-station pollution.
4. Missing target exclusion accounting (strict finite ground truth check).
5. Chronological boundary isolation (zero train->val and val->test leakage).
6. Target inversion consistency (scaled -> µg/m³).
7. Per-station and per-split sequence counts and exclusion statistics.

Usage:
    python scripts/verify_phase2_sequences.py [--station Aotizhongxin] [--all-stations]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.phase2.config import Phase2Config
from src.phase2.data_pipeline import process_all_stations, process_station
from src.phase2.sequence_generator import (
    DEFAULT_HORIZONS,
    DEFAULT_SEQUENCE_LENGTHS,
    StationSequenceBundle,
    generate_station_sequence_bundle,
    validate_sequence_bundle,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_sequences")


def audit_station_sequences(
    station_name: str,
    config: Phase2Config,
    window_sizes: Sequence[int] = DEFAULT_SEQUENCE_LENGTHS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Audit sequence generation for a single monitoring station."""
    t0 = time.time()
    splits = process_station(station_name, config=config)
    split_time = time.time() - t0

    station_results: Dict[str, Any] = {
        "station_name": station_name,
        "split_time_sec": round(split_time, 2),
        "windows": {},
    }

    table_rows = []

    for w in window_sizes:
        tw0 = time.time()
        bundle = generate_station_sequence_bundle(
            splits=splits,
            window_size=w,
            horizons=horizons,
            scaled=True,
            scale_target=True,
        )
        gen_time = time.time() - tw0

        # Validate bundle
        assert bundle.validate() is True

        train_rep = bundle.train.exclusion_report
        val_rep = bundle.val.exclusion_report
        test_rep = bundle.test.exclusion_report

        station_results["windows"][w] = {
            "generation_time_sec": round(gen_time, 3),
            "train": train_rep.to_dict() if train_rep else {},
            "val": val_rep.to_dict() if val_rep else {},
            "test": test_rep.to_dict() if test_rep else {},
            "X_train_shape": list(bundle.X_train.shape),
            "y_train_shape": list(bundle.y_train.shape),
            "X_val_shape": list(bundle.X_val.shape),
            "y_val_shape": list(bundle.y_val.shape),
            "X_test_shape": list(bundle.X_test.shape),
            "y_test_shape": list(bundle.y_test.shape),
        }

        # Add to table
        for split_name, ds in [("Train", bundle.train), ("Val", bundle.val), ("Test", bundle.test)]:
            rep = ds.exclusion_report
            table_rows.append([
                station_name,
                f"W={w}",
                split_name,
                f"{rep.total_candidates_evaluated:,}" if rep else "0",
                f"{len(ds):,}",
                str(ds.X.shape),
                str(ds.y.shape),
                f"{rep.excluded_missing_target:,}" if rep else "0",
                f"{rep.exclusion_rate_pct:.2f}%" if rep else "0.00%",
            ])

    if verbose:
        headers = [
            "Station", "Window", "Split", "Candidates", "Retained",
            "X Shape", "y Shape", "Excluded Missing", "Exclusion %"
        ]
        print(f"\n{'='*95}")
        print(f"Sequence Generation Audit: {station_name}")
        print(f"{'='*95}")
        print(tabulate(table_rows, headers=headers, tablefmt="github"))

    return station_results


def run_full_sweep(config: Phase2Config) -> Dict[str, Any]:
    """Run sequence generation audit across all 12 stations."""
    stations = config.stations.all
    all_results = {}
    summary_rows = []

    print(f"\n================================================================================")
    print(f"   PHASE 2 LSTM SEQUENCE GENERATOR: 12-STATION NETWORK AUDIT SWEEP")
    print(f"   Horizons: {list(DEFAULT_HORIZONS)} | Windows: {list(DEFAULT_SEQUENCE_LENGTHS)}")
    print(f"================================================================================")

    t_start = time.time()
    for station in stations:
        res = audit_station_sequences(station, config=config, verbose=False)
        all_results[station] = res

        # Collect summary metrics for default W=24
        w24 = res["windows"][24]
        tr_n = w24["train"]["valid_sequences_retained"]
        val_n = w24["val"]["valid_sequences_retained"]
        te_n = w24["test"]["valid_sequences_retained"]
        tot_valid = tr_n + val_n + te_n

        tr_ex = w24["train"]["excluded_missing_target"]
        val_ex = w24["val"]["excluded_missing_target"]
        te_ex = w24["test"]["excluded_missing_target"]
        tot_ex = tr_ex + val_ex + te_ex

        tot_cand = (
            w24["train"]["total_candidates_evaluated"]
            + w24["val"]["total_candidates_evaluated"]
            + w24["test"]["total_candidates_evaluated"]
        )
        ex_pct = 100.0 * tot_ex / tot_cand if tot_cand > 0 else 0.0

        summary_rows.append([
            station,
            f"{tr_n:,}",
            f"{val_n:,}",
            f"{te_n:,}",
            f"{tot_valid:,}",
            f"{tot_ex:,}",
            f"{ex_pct:.2f}%",
        ])

    total_time = time.time() - t_start

    headers = [
        "Station", "Train (W=24)", "Val (W=24)", "Test (W=24)",
        "Total Retained", "Total Excluded", "Exclusion %"
    ]
    print(f"\n" + tabulate(summary_rows, headers=headers, tablefmt="github"))
    print(f"\nAudit completed in {total_time:.2f} seconds across {len(stations)} stations.")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Audit Phase 2 LSTM sequence generation.")
    parser.add_argument("--station", type=str, default="Aotizhongxin", help="Station to audit")
    parser.add_argument("--all-stations", action="store_true", help="Audit all 12 stations")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save JSON audit results")
    args = parser.parse_args()

    config = Phase2Config.get_default_config()

    if args.all_stations:
        results = run_full_sweep(config)
    else:
        results = audit_station_sequences(args.station, config=config, verbose=True)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved audit results to {out_path}")


if __name__ == "__main__":
    main()

