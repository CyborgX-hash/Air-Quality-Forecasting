"""
Reusable Feature and Target Scaling Module for Phase 2 LSTM Forecasting.

Design Principles:
1. Strict Training-Only Fitting: Scaler parameters (mean, std, min, max) are
   computed EXCLUSIVELY on training data partitions. Validation and test sets
   are transformed using frozen parameters to guarantee zero leakage.
2. Station Isolation: Separate scaler instances are fitted and maintained for each
   monitoring station to account for local microclimate and elevation differences.
3. Dedicated Target Inversion: Provides seamless `inverse_transform_target()`
   for 1D, 2D, or multi-horizon predicted PM2.5 values without requiring dummy
   feature columns.
4. Flexible Scaler Algorithms: Supports both `StandardScaler` (zero-mean, unit-variance,
   recommended for deep learning with extreme haze spikes) and `MinMaxScaler`
   (bounded [0, 1] range).
5. Robust Serialization: Full persistence support via joblib to save and load
   station-specific scalers for reproducible model inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class StationScaler:
    """
    Station-isolated feature and target scaler for time-series forecasting.

    Maintains separate scaling transformers for multivariate input features X
    and the scalar target variable y (PM2.5), allowing direct inverse transformation
    of model predictions back to original concentration units (µg/m³).

    Parameters
    ----------
    scaler_type : str, default 'standard'
        Scaling algorithm: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler).
    target_col : str, default 'PM2.5'
        Name of the target column.
    station_name : Optional[str], default None
        Name of the monitoring station.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        target_col: str = "PM2.5",
        station_name: Optional[str] = None,
    ) -> None:
        self.scaler_type = scaler_type.lower()
        if self.scaler_type not in {"standard", "minmax"}:
            raise ValueError(
                f"Unsupported scaler_type '{scaler_type}'. Choose 'standard' or 'minmax'."
            )
        self.target_col = target_col
        self.station_name = station_name or "unknown_station"

        self.feature_scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.target_scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

        # Target scaling parameters for transparent inspection and fast inversion
        self.target_mean_: Optional[float] = None
        self.target_std_: Optional[float] = None
        self.target_min_: Optional[float] = None
        self.target_max_: Optional[float] = None
        self.target_scale_: Optional[float] = None

    def _create_scaler_instance(self) -> Union[StandardScaler, MinMaxScaler]:
        """Instantiate underlying scikit-learn scaler."""
        if self.scaler_type == "standard":
            return StandardScaler()
        return MinMaxScaler(feature_range=(0.0, 1.0))

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> StationScaler:
        """
        Fit feature and target scalers strictly on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : Optional[Union[pd.Series, np.ndarray]], default None
            Training target observations. If None and `target_col` is present in X,
            target scaler is fitted on X[target_col] (ignoring missing target values).

        Returns
        -------
        StationScaler
            Self (fitted scaler).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

        self.feature_names_ = list(X.columns)
        self.feature_scaler = self._create_scaler_instance()
        self.target_scaler = self._create_scaler_instance()

        # Fit feature scaler
        self.feature_scaler.fit(X.values)

        # Fit target scaler
        target_values: Optional[np.ndarray] = None
        if y is not None:
            if isinstance(y, pd.Series):
                clean_y = y.dropna().values
            else:
                arr = np.asarray(y)
                clean_y = arr[~np.isnan(arr)]
            if len(clean_y) == 0:
                raise ValueError("Cannot fit target scaler: all target values in y are NaN.")
            target_values = clean_y.reshape(-1, 1)
        elif self.target_col in X.columns:
            clean_series = X[self.target_col].dropna()
            if len(clean_series) == 0:
                raise ValueError(
                    f"Cannot fit target scaler: column '{self.target_col}' has no valid values."
                )
            target_values = clean_series.values.reshape(-1, 1)

        if target_values is not None:
            self.target_scaler.fit(target_values)
            if self.scaler_type == "standard":
                self.target_mean_ = float(self.target_scaler.mean_[0])
                self.target_std_ = float(self.target_scaler.scale_[0])
            else:
                self.target_min_ = float(self.target_scaler.data_min_[0])
                self.target_max_ = float(self.target_scaler.data_max_[0])
                self.target_scale_ = float(self.target_scaler.scale_[0])
        else:
            logger.warning(
                "Target scaler was not fitted because no target y or '%s' column was found.",
                self.target_col,
            )

        self.is_fitted = True
        return self

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale feature DataFrame using fitted training parameters.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with identical index and column names.
        """
        if not self.is_fitted or self.feature_scaler is None:
            raise RuntimeError("StationScaler must be fitted before transforming features.")

        if list(X.columns) != self.feature_names_:
            missing = set(self.feature_names_) - set(X.columns)
            extra = set(X.columns) - set(self.feature_names_)
            if missing or extra:
                raise ValueError(
                    f"Feature mismatch in transform. Missing: {missing}, Extra: {extra}"
                )
            # Reorder columns to match fit order
            X = X[self.feature_names_]

        transformed_vals = self.feature_scaler.transform(X.values)
        return pd.DataFrame(transformed_vals, index=X.index, columns=self.feature_names_)

    def transform_target(
        self,
        y: Union[pd.Series, np.ndarray, float],
    ) -> Union[np.ndarray, float]:
        """
        Scale ground-truth target values using fitted target parameters.

        Preserves NaN locations.

        Parameters
        ----------
        y : Union[pd.Series, np.ndarray, float]
            Target values in physical concentration units (µg/m³).

        Returns
        -------
        Union[np.ndarray, float]
            Scaled target array or float.
        """
        if not self.is_fitted or self.target_scaler is None:
            raise RuntimeError("StationScaler must be fitted before transforming target.")

        is_scalar = np.isscalar(y)
        arr = np.asarray(y, dtype=float)
        orig_shape = arr.shape

        reshaped = arr.reshape(-1, 1)
        valid_mask = ~np.isnan(reshaped)

        scaled = np.full_like(reshaped, fill_value=np.nan)
        if valid_mask.any():
            scaled[valid_mask] = self.target_scaler.transform(reshaped[valid_mask].reshape(-1, 1)).flatten()

        out = scaled.reshape(orig_shape)
        return float(out) if is_scalar else out

    def inverse_transform_target(
        self,
        y_scaled: Union[pd.Series, np.ndarray, float, List[float]],
    ) -> Union[np.ndarray, float]:
        """
        Invert scaled target predictions back to physical concentration units (µg/m³).

        Supports:
        - 1D predictions of shape (N,)
        - 2D predictions of shape (N, 1) or (N, H) for multi-horizon forecasts
        - Single float scalar predictions
        - List of predictions

        Parameters
        ----------
        y_scaled : Union[pd.Series, np.ndarray, float, List[float]]
            Scaled model predictions.

        Returns
        -------
        Union[np.ndarray, float]
            Inverted predictions in physical units (µg/m³).
        """
        if not self.is_fitted or self.target_scaler is None:
            raise RuntimeError("StationScaler must be fitted before inverse-transforming target.")

        is_scalar = np.isscalar(y_scaled)
        if isinstance(y_scaled, pd.Series):
            arr = y_scaled.values.astype(float)
        else:
            arr = np.asarray(y_scaled, dtype=float)

        orig_shape = arr.shape
        reshaped = arr.reshape(-1, 1)
        valid_mask = ~np.isnan(reshaped)

        inverted = np.full_like(reshaped, fill_value=np.nan)
        if valid_mask.any():
            inverted[valid_mask] = self.target_scaler.inverse_transform(
                reshaped[valid_mask].reshape(-1, 1)
            ).flatten()

        out = inverted.reshape(orig_shape)
        return float(out) if is_scalar else out

    def inverse_transform_features(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        """
        Invert scaled feature matrix back to physical units.

        Parameters
        ----------
        X_scaled : pd.DataFrame
            Scaled feature DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame in original feature units.
        """
        if not self.is_fitted or self.feature_scaler is None:
            raise RuntimeError("StationScaler must be fitted before inverse-transforming features.")

        inverted_vals = self.feature_scaler.inverse_transform(X_scaled.values)
        return pd.DataFrame(inverted_vals, index=X_scaled.index, columns=self.feature_names_)

    def get_params_summary(self) -> Dict[str, Any]:
        """Return audit summary of fitted scaling parameters."""
        summary: Dict[str, Any] = {
            "station_name": self.station_name,
            "scaler_type": self.scaler_type,
            "target_col": self.target_col,
            "is_fitted": self.is_fitted,
            "num_features": len(self.feature_names_),
            "feature_names": self.feature_names_,
        }
        if self.is_fitted and self.target_scaler is not None:
            if self.scaler_type == "standard":
                summary["target_mean"] = self.target_mean_
                summary["target_std"] = self.target_std_
            else:
                summary["target_min"] = self.target_min_
                summary["target_max"] = self.target_max_
        return summary

    def save(self, file_path: Union[str, Path]) -> Path:
        """
        Serialize fitted scaler to disk.

        Parameters
        ----------
        file_path : Union[str, Path]
            Target path (e.g. models/scalers/scaler_Aotizhongxin.joblib).

        Returns
        -------
        Path
            Resolved path of the saved file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved StationScaler for '%s' to %s", self.station_name, path)
        return path

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> StationScaler:
        """
        Load serialized StationScaler from disk.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to saved joblib file.

        Returns
        -------
        StationScaler
            Loaded and fitted StationScaler instance.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Scaler file not found: {path}")
        scaler = joblib.load(path)
        if not isinstance(scaler, cls):
            raise TypeError(f"Loaded object is not a StationScaler: {type(scaler)}")
        return scaler


class MultiStationScalerManager:
    """
    Manages and coordinates station-isolated scalers across all monitoring sites.

    Parameters
    ----------
    scaler_type : str, default 'standard'
        Algorithm to use across stations ('standard' or 'minmax').
    target_col : str, default 'PM2.5'
        Prediction target column name.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        target_col: str = "PM2.5",
    ) -> None:
        self.scaler_type = scaler_type
        self.target_col = target_col
        self.scalers: Dict[str, StationScaler] = {}

    def get_or_create(self, station_name: str) -> StationScaler:
        """Retrieve existing scaler or instantiate a new one for station."""
        if station_name not in self.scalers:
            self.scalers[station_name] = StationScaler(
                scaler_type=self.scaler_type,
                target_col=self.target_col,
                station_name=station_name,
            )
        return self.scalers[station_name]

    def fit_station(
        self,
        station_name: str,
        X_train: pd.DataFrame,
        y_train: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> StationScaler:
        """
        Fit a dedicated scaler for the specified station strictly on training data.

        Parameters
        ----------
        station_name : str
            Monitoring station identifier.
        X_train : pd.DataFrame
            Station training feature DataFrame.
        y_train : Optional[Union[pd.Series, np.ndarray]], default None
            Station training target values.

        Returns
        -------
        StationScaler
            Fitted scaler for the station.
        """
        scaler = self.get_or_create(station_name)
        scaler.fit(X_train, y=y_train)
        return scaler

    def get_scaler(self, station_name: str) -> StationScaler:
        """Get fitted scaler for station, raising KeyError if not found."""
        if station_name not in self.scalers:
            raise KeyError(
                f"No scaler found for station '{station_name}'. Available: {list(self.scalers.keys())}"
            )
        return self.scalers[station_name]

    def save_all(self, directory: Union[str, Path]) -> Dict[str, str]:
        """
        Save all station scalers into designated directory.

        Parameters
        ----------
        directory : Union[str, Path]
            Target directory.

        Returns
        -------
        Dict[str, str]
            Map of station name to saved file path.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        saved_paths: Dict[str, str] = {}
        for station_name, scaler in self.scalers.items():
            fname = dir_path / f"scaler_{station_name}.joblib"
            scaler.save(fname)
            saved_paths[station_name] = str(fname)
        return saved_paths

    @classmethod
    def load_all(
        cls,
        directory: Union[str, Path],
        scaler_type: str = "standard",
        target_col: str = "PM2.5",
    ) -> MultiStationScalerManager:
        """
        Load all station scalers from directory.

        Parameters
        ----------
        directory : Union[str, Path]
            Source directory containing scaler_*.joblib files.
        scaler_type : str, default 'standard'
            Expected scaler type.
        target_col : str, default 'PM2.5'
            Target column.

        Returns
        -------
        MultiStationScalerManager
            Manager populated with loaded scalers.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Scaler directory not found: {dir_path}")

        manager = cls(scaler_type=scaler_type, target_col=target_col)
        for fpath in dir_path.glob("scaler_*.joblib"):
            scaler = StationScaler.load(fpath)
            manager.scalers[scaler.station_name] = scaler
            logger.info("Loaded scaler for '%s' from %s", scaler.station_name, fpath)
        return manager


__all__ = ["StationScaler", "MultiStationScalerManager"]

