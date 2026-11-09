"""
Configuration Module for Phase 2 LSTM Data Pipeline.

Defines typed dataclasses and YAML loading/saving utilities for:
- 12 Beijing monitoring stations
- Input feature selection (pollutants, meteorology, cyclical wind, domain, temporal)
- PM2.5 forecasting target
- Forecast horizons (1, 3, 6, 12, 24 hours)
- Historical input sequence windows (24, 48, 72 hours)
- Chronological split ratios (70% Train / 15% Val / 15% Test)
- Feature and target scaling configuration
- Random seeds and reproducibility metadata
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from src.config import VALID_STATIONS


@dataclass
class ProjectMetadataConfig:
    """General project and phase metadata."""
    name: str = "Air Quality Forecasting via Gradient Boosting and LSTM Models"
    phase: int = 2
    target: str = "PM2.5"
    random_seed: int = 42


@dataclass
class StationConfig:
    """Station registry and default station."""
    all: List[str] = field(default_factory=lambda: list(VALID_STATIONS))
    default: str = "Aotizhongxin"


@dataclass
class DataPathConfig:
    """Dataset location and schema column names."""
    raw_dir: str = "Data/Raw"
    timestamp_column: str = "datetime"
    station_column: str = "station"
    target_column: str = "PM2.5"


@dataclass
class CleaningConfig:
    """Causal cleaning and imputation constraints."""
    max_ffill_hours: int = 6
    encode_wind: bool = True
    fill_remaining_with_median: bool = True
    impute_target: bool = False  # NEVER impute target PM2.5


@dataclass
class FeatureGroupConfig:
    """Grouped feature definitions for Phase 2 sequence modeling."""
    pollutants: List[str] = field(
        default_factory=lambda: ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    )
    meteorology: List[str] = field(
        default_factory=lambda: ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    )
    wind: List[str] = field(
        default_factory=lambda: ["wd_sin", "wd_cos"]
    )
    domain: List[str] = field(
        default_factory=lambda: ["temp_dewp_spread", "wind_stagnation", "rain_clearing"]
    )
    temporal: List[str] = field(
        default_factory=lambda: [
            "sin_hour", "cos_hour", "sin_month", "cos_month", "sin_dow", "cos_dow"
        ]
    )

    def get_all_features(self) -> List[str]:
        """Return full ordered list of all feature column names."""
        ordered: List[str] = []
        for group in [self.pollutants, self.meteorology, self.wind, self.domain, self.temporal]:
            for feat in group:
                if feat not in ordered:
                    ordered.append(feat)
        return ordered


@dataclass
class WindowConfig:
    """LSTM sequence window configurations."""
    input_sequence_lengths: List[int] = field(default_factory=lambda: [24, 48, 72])
    default_sequence_length: int = 24


@dataclass
class HorizonConfig:
    """Forecasting horizon configurations."""
    horizons: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24])
    default_horizon: int = 1


@dataclass
class SplitConfig:
    """Chronological dataset partitioning settings."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    chronological: bool = True
    embargo_hours: int = 0

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Split ratios must sum to 1.0 (train={self.train_ratio}, "
                f"val={self.val_ratio}, test={self.test_ratio}, sum={total})"
            )


@dataclass
class ScalingConfig:
    """Scaler selection, directory, and target scaling rules."""
    scaler_type: str = "standard"  # "standard" or "minmax"
    scalers_dir: str = "models/phase2/scalers"
    save_scalers: bool = True
    scale_target: bool = True

    def __post_init__(self):
        allowed = {"standard", "minmax"}
        if self.scaler_type.lower() not in allowed:
            raise ValueError(
                f"Invalid scaler_type: '{self.scaler_type}'. Allowed: {allowed}"
            )
        self.scaler_type = self.scaler_type.lower()


@dataclass
class Phase2Config:
    """Master configuration container for Phase 2 LSTM Data Pipeline."""
    project: ProjectMetadataConfig = field(default_factory=ProjectMetadataConfig)
    stations: StationConfig = field(default_factory=StationConfig)
    data: DataPathConfig = field(default_factory=DataPathConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    features: FeatureGroupConfig = field(default_factory=FeatureGroupConfig)
    lstm_windows: WindowConfig = field(default_factory=WindowConfig)
    forecast_horizons: HorizonConfig = field(default_factory=HorizonConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)

    def get_feature_list(self) -> List[str]:
        """Return flat list of all active feature columns in deterministic order."""
        return self.features.get_all_features()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to nested dictionary."""
        return asdict(self)

    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """Export configuration to YAML file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Phase2Config:
        """Construct Phase2Config from nested dictionary."""
        d = data.copy()
        return cls(
            project=ProjectMetadataConfig(**d.get("project", {})),
            stations=StationConfig(**d.get("stations", {})),
            data=DataPathConfig(**d.get("data", {})),
            cleaning=CleaningConfig(**d.get("cleaning", {})),
            features=FeatureGroupConfig(**d.get("features", {})),
            lstm_windows=WindowConfig(**d.get("lstm_windows", {})),
            forecast_horizons=HorizonConfig(**d.get("forecast_horizons", {})),
            split=SplitConfig(**d.get("split", {})),
            scaling=ScalingConfig(**d.get("scaling", {})),
        )

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> Phase2Config:
        """Load Phase2Config from a YAML file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f)
        return cls.from_dict(raw_dict or {})

    @classmethod
    def get_default_config(cls) -> Phase2Config:
        """Return default configuration instance."""
        default_yaml = Path("configs/phase2_config.yaml")
        if default_yaml.is_file():
            return cls.from_yaml(default_yaml)
        return cls()


__all__ = [
    "ProjectMetadataConfig",
    "StationConfig",
    "DataPathConfig",
    "CleaningConfig",
    "FeatureGroupConfig",
    "WindowConfig",
    "HorizonConfig",
    "SplitConfig",
    "ScalingConfig",
    "Phase2Config",
]

