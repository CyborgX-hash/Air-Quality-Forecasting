"""
Data loading and validation module for Beijing Air Quality dataset.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import glob
import pandas as pd
import numpy as np

REQUIRED_COLUMNS: List[str] = [
    "No",
    "year",
    "month",
    "day",
    "hour",
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",
    "WSPM",
    "station",
]

DATE_COMPONENTS: List[str] = ["year", "month", "day", "hour"]


def build_timestamp(
    df: pd.DataFrame,
    drop_source_cols: bool = False,
) -> pd.DataFrame:
    """
    Construct continuous datetime column from year, month, day, hour.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing date component columns.
    drop_source_cols : bool, default False
        Whether to drop the component columns after creating 'datetime'.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime64 column added.
    """
    df_out = df.copy()
    for col in DATE_COMPONENTS:
        if col not in df_out.columns:
            raise KeyError(f"Missing required date component column: '{col}'")

    df_out["datetime"] = pd.to_datetime(df_out[DATE_COMPONENTS])

    if drop_source_cols:
        df_out = df_out.drop(columns=DATE_COMPONENTS)

    return df_out


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data integrity, monotonicity, duplicate records, and continuity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'datetime' and optionally 'station'.

    Returns
    -------
    Dict[str, Any]
        Audit dictionary with validation metrics.
    """
    if "datetime" not in df.columns:
        raise KeyError("Column 'datetime' is required for validation.")

    station_col = "station" if "station" in df.columns else None

    # Check duplicates
    subset_cols = [station_col, "datetime"] if station_col else ["datetime"]
    duplicates_count = int(df.duplicated(subset=subset_cols).sum())

    # Check chronological monotonicity
    if station_col:
        is_chronological = True
        for _, grp in df.groupby(station_col):
            if not grp["datetime"].is_monotonic_increasing:
                is_chronological = False
                break
    else:
        is_chronological = bool(df["datetime"].is_monotonic_increasing)

    # Check hourly continuity
    continuity_gaps: Dict[str, int] = {}
    is_hourly_continuous = True

    if station_col:
        for st_name, grp in df.groupby(station_col):
            grp_sorted = grp.sort_values(by="datetime")
            diffs = grp_sorted["datetime"].diff().dropna()
            expected_diff = pd.Timedelta(hours=1)
            gaps = int((diffs != expected_diff).sum())
            if gaps > 0:
                continuity_gaps[str(st_name)] = gaps
                is_hourly_continuous = False
    else:
        diffs = df.sort_values(by="datetime")["datetime"].diff().dropna()
        gaps = int((diffs != pd.Timedelta(hours=1)).sum())
        if gaps > 0:
            continuity_gaps["global"] = gaps
            is_hourly_continuous = False

    # Target statistics
    target_missing = int(df["PM2.5"].isnull().sum()) if "PM2.5" in df.columns else 0
    target_valid = int(df["PM2.5"].notnull().sum()) if "PM2.5" in df.columns else 0

    is_valid = (
        duplicates_count == 0
        and is_chronological
        and is_hourly_continuous
    )

    return {
        "is_valid": is_valid,
        "duplicate_records": duplicates_count,
        "is_chronological": is_chronological,
        "is_hourly_continuous": is_hourly_continuous,
        "continuity_gaps_per_station": continuity_gaps,
        "target_stats": {
            "missing": target_missing,
            "valid": target_valid,
        },
    }


def load_station_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load single station CSV, validate schema, build timestamp, and sort chronologically.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the station CSV.

    Returns
    -------
    pd.DataFrame
        Loaded and timestamped DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Station data file not found: {path}")

    df = pd.read_csv(path)
    df = build_timestamp(df, drop_source_cols=False)
    df = df.sort_values(by="datetime").reset_index(drop=True)
    return df


def load_all_stations(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load all station CSV files in the data directory and concatenate them.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Path to directory containing station CSVs.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame for all stations.
    """
    dir_path = Path(data_dir)
    pattern = str(dir_path / "PRSA_Data_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No station CSV files found matching {pattern}")

    station_dfs = [load_station_data(f) for f in files]
    df_all = pd.concat(station_dfs, ignore_index=True)
    df_all = df_all.sort_values(by=["station", "datetime"]).reset_index(drop=True)
    return df_all
