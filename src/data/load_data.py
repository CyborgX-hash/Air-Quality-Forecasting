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


def validate_all_stations_schema(
    data_dir: Union[str, Path],
    expected_stations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate dataset existence, schema conformity, and temporal continuity for all stations.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory path containing the station CSV files.
    expected_stations : Optional[List[str]], default None
        List of expected station names. If None, defaults to the standard 12 stations.

    Returns
    -------
    Dict[str, Any]
        Audit dictionary containing:
        - all_valid: bool indicating if all checks passed
        - num_stations: count of validated stations
        - stations_found: list of station names found and verified
        - stations_missing: list of missing station names
        - station_reports: detailed dictionary per station
    """
    if expected_stations is None:
        expected_stations = [
            "Aotizhongxin",
            "Changping",
            "Dingling",
            "Dongsi",
            "Guanyuan",
            "Gucheng",
            "Huairou",
            "Nongzhanguan",
            "Shunyi",
            "Tiantan",
            "Wanliu",
            "Wanshouxigong",
        ]

    dir_path = Path(data_dir)
    reports: Dict[str, Any] = {}
    missing_stations: List[str] = []
    found_stations: List[str] = []
    all_valid = True

    for station in expected_stations:
        # Match pattern: PRSA_Data_{station}_*.csv
        matches = sorted(dir_path.glob(f"PRSA_Data_{station}_*.csv"))
        if not matches:
            # Case-insensitive fallback
            for f in dir_path.glob("PRSA_Data_*.csv"):
                if station.lower() in f.name.lower():
                    matches = [f]
                    break

        if not matches:
            missing_stations.append(station)
            all_valid = False
            reports[station] = {
                "valid": False,
                "error": f"File not found for station '{station}' in {dir_path}",
            }
            continue

        file_path = matches[0]
        found_stations.append(station)

        try:
            df = load_station_data(file_path)
            # Check schema
            missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            audit = validate_data(df)
            has_valid_schema = len(missing_cols) == 0
            station_valid = has_valid_schema and audit["is_valid"]
            if not station_valid:
                all_valid = False

            reports[station] = {
                "valid": station_valid,
                "file_path": str(file_path),
                "record_count": len(df),
                "start_timestamp": str(df["datetime"].min()),
                "end_timestamp": str(df["datetime"].max()),
                "missing_columns": missing_cols,
                "duplicate_records": audit["duplicate_records"],
                "is_chronological": audit["is_chronological"],
                "is_hourly_continuous": audit["is_hourly_continuous"],
                "target_missing": audit["target_stats"]["missing"],
                "target_valid": audit["target_stats"]["valid"],
            }
        except Exception as exc:
            all_valid = False
            reports[station] = {
                "valid": False,
                "file_path": str(file_path),
                "error": str(exc),
            }

    return {
        "all_valid": all_valid and len(missing_stations) == 0,
        "num_stations": len(found_stations),
        "expected_count": len(expected_stations),
        "stations_found": sorted(found_stations),
        "stations_missing": sorted(missing_stations),
        "station_reports": reports,
    }
