"""
Phase 2 Data Pipeline & Preprocessing Package.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Saksham Sontakke (Data Pipeline & Preprocessing Engineer)

Provides modular, reproducible, station-isolated data preparation
for Phase 2 sequence-based LSTM forecasting in TensorFlow/Keras.
"""

from src.phase2.config import Phase2Config
from src.phase2.scaling import MultiStationScalerManager, StationScaler
from src.phase2.data_pipeline import (
    StationDataSplits,
    create_lstm_sequences,
    prepare_station_features,
    process_all_stations,
    process_station,
    split_station_data,
)
from src.phase2.sequence_generator import (
    DEFAULT_HORIZONS,
    DEFAULT_SEQUENCE_LENGTHS,
    DEFAULT_TARGET_COL,
    UNIMPUTED_TARGET_COL,
    SequenceDataset,
    SequenceExclusionReport,
    StationSequenceBundle,
    create_multistep_sequences,
    generate_all_stations_sequences,
    generate_station_sequence_bundle,
    validate_sequence_bundle,
    validate_sequence_dataset,
)

__all__ = [
    "Phase2Config",
    "StationScaler",
    "MultiStationScalerManager",
    "StationDataSplits",
    "prepare_station_features",
    "split_station_data",
    "process_station",
    "process_all_stations",
    "create_lstm_sequences",
    "DEFAULT_SEQUENCE_LENGTHS",
    "DEFAULT_HORIZONS",
    "DEFAULT_TARGET_COL",
    "UNIMPUTED_TARGET_COL",
    "SequenceExclusionReport",
    "SequenceDataset",
    "StationSequenceBundle",
    "create_multistep_sequences",
    "generate_station_sequence_bundle",
    "generate_all_stations_sequences",
    "validate_sequence_dataset",
    "validate_sequence_bundle",
]

