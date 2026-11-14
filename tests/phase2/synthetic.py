"""Synthetic PRSA-format station data for fast, offline Phase 2 tests."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def make_station_frame(station: str, n_hours: int = 2400, seed: int = 0,
                       pm25_missing_frac: float = 0.03) -> pd.DataFrame:
    """Raw frame with the exact schema of the UCI Beijing CSVs."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2015-01-01 00:00", periods=n_hours, freq="h")
    t = np.arange(n_hours)
    base = 60 + 35 * np.sin(2 * np.pi * t / 24) + 25 * np.sin(2 * np.pi * t / (24 * 7))
    pm25 = np.clip(base + np.cumsum(rng.normal(0, 3, n_hours)) * 0.2 + rng.normal(0, 8, n_hours), 2, None)
    df = pd.DataFrame({
        "No": np.arange(1, n_hours + 1),
        "year": ts.year, "month": ts.month, "day": ts.day, "hour": ts.hour,
        "PM2.5": pm25,
        "PM10": pm25 * 1.3 + rng.normal(0, 5, n_hours),
        "SO2": np.clip(10 + rng.normal(0, 3, n_hours), 1, None),
        "NO2": np.clip(40 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 5, n_hours), 1, None),
        "CO": np.clip(900 + pm25 * 6 + rng.normal(0, 80, n_hours), 100, None),
        "O3": np.clip(50 - 20 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 8, n_hours), 1, None),
        "TEMP": 5 + 10 * np.sin(2 * np.pi * t / (24 * 365)) + rng.normal(0, 1, n_hours),
        "PRES": 1015 + rng.normal(0, 3, n_hours),
        "DEWP": -5 + rng.normal(0, 2, n_hours),
        "RAIN": np.where(rng.random(n_hours) < 0.03, rng.random(n_hours), 0.0),
        "wd": rng.choice(WIND_DIRS, n_hours),
        "WSPM": np.clip(2 + rng.normal(0, 1, n_hours), 0, None),
        "station": station,
    })
    miss = rng.random(n_hours) < pm25_missing_frac
    df.loc[miss, "PM2.5"] = np.nan                       # missing TARGET values
    df.loc[rng.random(n_hours) < 0.02, "NO2"] = np.nan   # missing feature values
    return df


def write_synthetic_raw_dir(directory: Path, stations: Sequence[str], n_hours: int = 2400) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for i, st in enumerate(stations):
        make_station_frame(st, n_hours=n_hours, seed=100 + i).to_csv(
            directory / f"PRSA_Data_{st}_20150101-20150411.csv", index=False)
    return directory
