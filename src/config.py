"""
Pipeline Configuration Module for Air Quality Forecasting.

Defines typed dataclasses for every stage of the Phase 1 machine learning workflow:
- Data loading and station selection
- Causal data cleaning & imputation
- Time-series feature engineering & forecasting horizon
- Chronological train/validation/test splitting
- Model hyperparameters, baselines, and tuning
- Result and visualization export paths
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


# 12 UCI Beijing Monitoring Stations
VALID_STATIONS: List[str] = [
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


@dataclass
class DataConfig:
    """Dataset loading and station selection settings."""
    data_dir: str = "data/raw"
    station: str = "Aotizhongxin"  # Station name or "all" for all 12 stations
    data_file: Optional[str] = None  # Direct explicit file path override
    target_column: str = "PM2.5"
    timestamp_column: str = "datetime"
    station_column: str = "station"

    def resolve_data_path(self, project_root: Optional[Path] = None) -> Path:
        """Resolve the path to the CSV file or data directory."""
        root = project_root or Path.cwd()
        if self.data_file:
            path = Path(self.data_file)
            return path if path.is_absolute() else root / path

        d_dir = Path(self.data_dir)
        d_dir = d_dir if d_dir.is_absolute() else root / d_dir

        if self.station.lower() == "all":
            return d_dir

        # Match station filename pattern: PRSA_Data_{station}_*.csv
        matches = sorted(d_dir.glob(f"PRSA_Data_{self.station}_*.csv"))
        if matches:
            return matches[0]

        # Case-insensitive fallback
        for candidate in d_dir.glob("PRSA_Data_*.csv"):
            if self.station.lower() in candidate.name.lower():
                return candidate

        return d_dir / f"PRSA_Data_{self.station}_20130301-20170228.csv"


@dataclass
class CleaningConfig:
    """Causal data cleaning and imputation settings."""
    max_ffill_hours: int = 6
    encode_wind: bool = True
    fill_remaining_with_median: bool = True


@dataclass
class FeatureConfig:
    """Time-series feature generation settings."""
    target_col: str = "PM2.5"
    horizon: int = 1
    new_target_col: str = "target_pm25"
    pm25_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    pollutant_lags: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "PM10": [1, 3, 6, 24],
            "NO2": [1, 3, 6, 24],
            "CO": [1, 3, 6, 24],
        }
    )
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std"])
    rolling_cols: List[str] = field(default_factory=lambda: ["PM2.5", "PM10", "TEMP"])
    include_calendar: bool = True
    include_cyclical: bool = True
    include_domain_weather: bool = True
    include_wind_encoding: bool = True


@dataclass
class SplitConfig:
    """Chronological dataset partitioning settings."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    embargo_hours: int = 0  # Hours to purge at split boundaries to eliminate overlap

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Split ratios must sum to 1.0 (train={self.train_ratio}, "
                f"val={self.val_ratio}, test={self.test_ratio}, sum={total})"
            )


@dataclass
class ModelConfig:
    """Model training and evaluation parameters."""
    random_seed: int = 42
    baseline_name: str = "Persistence (Naive Lag-1)"
    models_to_train: List[str] = field(
        default_factory=lambda: [
            "persistence_naive",
            "persistence_constant",
            "linear_regression",
            "gradient_boosting_default",
            "gradient_boosting_tuned",
        ]
    )
    gb_default_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 5,
            "subsample": 0.8,
            "random_state": 42,
        }
    )
    gb_tuned_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 250,
            "learning_rate": 0.04,
            "max_depth": 5,
            "min_samples_leaf": 8,
            "subsample": 0.8,
            "random_state": 42,
        }
    )
    tune_gb_grid: bool = False
    gb_tune_param_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "n_estimators": [150, 250],
            "learning_rate": [0.03, 0.05],
            "max_depth": [3, 5],
            "min_samples_leaf": [5, 8],
        }
    )
    scaler_type: Optional[str] = None  # None, "standard", "robust", or "minmax"


@dataclass
class OutputConfig:
    """Output directories and reporting flags."""
    results_dir: str = "reports/results"
    figures_dir: str = "reports/figures"
    metadata_dir: str = "reports/results"
    generate_figures: bool = True


@dataclass
class PipelineConfig:
    """Comprehensive master configuration for the ML pipeline."""
    experiment_name: str = "phase1_air_quality_benchmark"
    data: DataConfig = field(default_factory=DataConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a plain dictionary."""
        return asdict(self)

    def to_json(self, file_path: Union[str, Path], indent: int = 2) -> None:
        """Export configuration to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        """Construct PipelineConfig from a dictionary."""
        config_data = data.copy()
        experiment_name = config_data.pop("experiment_name", "phase1_air_quality_benchmark")
        data_cfg = DataConfig(**config_data.get("data", {}))
        clean_cfg = CleaningConfig(**config_data.get("cleaning", {}))
        feat_cfg = FeatureConfig(**config_data.get("feature", {}))
        split_cfg = SplitConfig(**config_data.get("split", {}))
        model_cfg = ModelConfig(**config_data.get("model", {}))
        out_cfg = OutputConfig(**config_data.get("output", {}))

        return cls(
            experiment_name=experiment_name,
            data=data_cfg,
            cleaning=clean_cfg,
            feature=feat_cfg,
            split=split_cfg,
            model=model_cfg,
            output=out_cfg,
        )

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> PipelineConfig:
        """Load PipelineConfig from a JSON file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        return cls.from_dict(cfg_dict)

    @classmethod
    def get_default_config(cls) -> PipelineConfig:
        """Return fresh instance of default pipeline configuration."""
        return cls()


__all__ = [
    "VALID_STATIONS",
    "DataConfig",
    "CleaningConfig",
    "FeatureConfig",
    "SplitConfig",
    "ModelConfig",
    "OutputConfig",
    "PipelineConfig",
]
