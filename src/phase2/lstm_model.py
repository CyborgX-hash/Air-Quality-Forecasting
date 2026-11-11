"""
Configurable LSTM Neural Network Architecture for PM2.5 Air Quality Forecasting.

Air Quality Forecasting via Gradient Boosting and LSTM Models.
Author: Harsh Kumar (Deep Learning Engineer – LSTM Architecture and Training)

Key Design Principles:
1. Multi-Horizon Forecasting: Default 5-unit output layer predicting PM2.5 concentration
   across five key forecast horizons (t+1h, t+3h, t+6h, t+12h, and t+24h).
2. Highly Configurable Topology: Supports 1 or 2 LSTM recurrent layers, flexible unit
   counts (32, 64, 128), dropout rates (0.1, 0.2, 0.3), and optional intermediate Dense projections.
3. Loss Function Adaptability: Configurable between standard MAE loss (aligned with Phase 1
   primary metric) and robust Huber loss (for resilience against acute episodic haze outliers).
4. Full Reproducibility: Deterministic weight initialization and seed pinning.
"""

from __future__ import annotations

import os
# Ensure safe temporary cache directories inside sandbox
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")

from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics

logger = logging.getLogger(__name__)

# Standard multi-horizon prediction targets in hours
DEFAULT_HORIZONS: List[int] = [1, 3, 6, 12, 24]
HORIZON_NAMES: List[str] = ["1h", "3h", "6h", "12h", "24h"]


@dataclass
class LSTMArchitectureConfig:
    """
    Configuration specification for LSTM network architecture.

    Parameters
    ----------
    sequence_length : int, default 24
        Number of historical lookback time steps W (e.g. 24, 48, 72 hours).
    num_features : int, default 22
        Dimensionality of input feature vectors D at each time step.
    lstm_units : int, default 64
        Hidden state dimensionality of LSTM layer(s) (e.g. 32, 64, 128).
    lstm_layers : int, default 1
        Number of stacked recurrent layers (1 or 2).
    dropout_rate : float, default 0.2
        Dropout probability applied after recurrent and dense layers.
    dense_units : int, default 32
        Number of units in the penultimate Dense projection layer (0 to omit).
    dense_activation : str, default 'relu'
        Activation function for intermediate Dense layer.
    output_units : int, default 5
        Dimensionality of the final output vector (5 for horizons [1, 3, 6, 12, 24]).
    loss_function : str, default 'mae'
        Training loss objective: 'mae' (Mean Absolute Error) or 'huber' (Huber loss).
    huber_delta : float, default 1.0
        Threshold delta parameter for Huber loss.
    learning_rate : float, default 0.001
        Initial Adam optimizer learning rate.
    horizons : List[int] = [1, 3, 6, 12, 24]
        Forecast horizons associated with the output units.
    """
    sequence_length: int = 24
    num_features: int = 22
    lstm_units: int = 64
    lstm_layers: int = 1
    dropout_rate: float = 0.2
    dense_units: int = 32
    dense_activation: str = "relu"
    output_units: int = 5
    loss_function: str = "mae"
    huber_delta: float = 1.0
    learning_rate: float = 0.001
    horizons: List[int] = field(default_factory=lambda: list(DEFAULT_HORIZONS))

    def __post_init__(self):
        if self.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {self.sequence_length}")
        if self.num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {self.num_features}")
        if self.lstm_units < 1:
            raise ValueError(f"lstm_units must be >= 1, got {self.lstm_units}")
        if self.lstm_layers not in {1, 2}:
            raise ValueError(f"lstm_layers must be 1 or 2, got {self.lstm_layers}")
        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {self.dropout_rate}")
        if self.output_units < 1:
            raise ValueError(f"output_units must be >= 1, got {self.output_units}")
        allowed_losses = {"mae", "mean_absolute_error", "huber", "mse", "mean_squared_error"}
        if self.loss_function.lower() not in allowed_losses:
            raise ValueError(f"Unsupported loss_function '{self.loss_function}'. Choose from {allowed_losses}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a standard dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LSTMArchitectureConfig:
        """Construct configuration from dictionary."""
        return cls(**data)


def build_loss(loss_name: str = "mae", delta: float = 1.0) -> keras.losses.Loss:
    """Instantiate compiled Keras loss object."""
    clean_name = loss_name.lower()
    if clean_name in {"mae", "mean_absolute_error"}:
        return losses.MeanAbsoluteError()
    elif clean_name == "huber":
        return losses.Huber(delta=delta)
    elif clean_name in {"mse", "mean_squared_error"}:
        return losses.MeanSquaredError()
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")


def build_lstm_model(
    config: Optional[LSTMArchitectureConfig] = None,
    sequence_length: Optional[int] = None,
    num_features: Optional[int] = None,
    lstm_units: Optional[int] = None,
    lstm_layers: Optional[int] = None,
    dropout_rate: Optional[float] = None,
    dense_units: Optional[int] = None,
    output_units: Optional[int] = None,
    loss_function: Optional[str] = None,
    learning_rate: Optional[float] = None,
    seed: Optional[int] = 42,
) -> keras.Model:
    """
    Construct, assemble, and compile a configurable LSTM regression model.

    Architecture Overview:
    1. Input layer: shape (sequence_length, num_features).
    2. Recurrent stage:
       - If lstm_layers == 2:
         LSTM_1(units, return_sequences=True) -> Dropout(rate) ->
         LSTM_2(units, return_sequences=False) -> Dropout(rate)
       - If lstm_layers == 1:
         LSTM_1(units, return_sequences=False) -> Dropout(rate)
    3. Intermediate projection:
       - Dense(dense_units, activation='relu') -> Dropout(rate / 2)
    4. Multi-Horizon Output layer:
       - Dense(output_units, activation='linear')

    Parameters
    ----------
    config : Optional[LSTMArchitectureConfig], default None
        Configuration specification. If None, default settings are used,
        optionally overridden by explicit arguments.
    sequence_length : Optional[int], default None
        Input sequence lookback W (e.g. 24, 48, 72).
    num_features : Optional[int], default None
        Number of feature channels D (e.g. 22).
    lstm_units : Optional[int], default None
        Number of hidden units in LSTM cells (e.g. 32, 64, 128).
    lstm_layers : Optional[int], default None
        Number of LSTM layers (1 or 2).
    dropout_rate : Optional[float], default None
        Dropout rate.
    dense_units : Optional[int], default None
        Units in dense layer.
    output_units : Optional[int], default None
        Output dimension (default: 5 for horizons [1, 3, 6, 12, 24]).
    loss_function : Optional[str], default None
        Loss function: 'mae' or 'huber'.
    learning_rate : Optional[float], default None
        Adam learning rate.
    seed : Optional[int], default 42
        Random seed for reproducible weight initialization.

    Returns
    -------
    keras.Model
        Compiled Keras sequential model ready for training.
    """
    if seed is not None:
        keras.utils.set_random_seed(seed)

    base_cfg = config or LSTMArchitectureConfig()
    cfg_dict = base_cfg.to_dict()

    # Apply explicit overrides if supplied
    if sequence_length is not None:
        cfg_dict["sequence_length"] = sequence_length
    if num_features is not None:
        cfg_dict["num_features"] = num_features
    if lstm_units is not None:
        cfg_dict["lstm_units"] = lstm_units
    if lstm_layers is not None:
        cfg_dict["lstm_layers"] = lstm_layers
    if dropout_rate is not None:
        cfg_dict["dropout_rate"] = dropout_rate
    if dense_units is not None:
        cfg_dict["dense_units"] = dense_units
    if output_units is not None:
        cfg_dict["output_units"] = output_units
    if loss_function is not None:
        cfg_dict["loss_function"] = loss_function
    if learning_rate is not None:
        cfg_dict["learning_rate"] = learning_rate

    resolved_cfg = LSTMArchitectureConfig.from_dict(cfg_dict)

    # Build model using Keras Functional API
    inputs = keras.Input(
        shape=(resolved_cfg.sequence_length, resolved_cfg.num_features),
        name="input_sequence",
    )

    x = inputs
    if resolved_cfg.lstm_layers == 2:
        x = layers.LSTM(
            units=resolved_cfg.lstm_units,
            return_sequences=True,
            name="lstm_layer_1",
        )(x)
        if resolved_cfg.dropout_rate > 0.0:
            x = layers.Dropout(rate=resolved_cfg.dropout_rate, name="dropout_1")(x)

        x = layers.LSTM(
            units=resolved_cfg.lstm_units,
            return_sequences=False,
            name="lstm_layer_2",
        )(x)
        if resolved_cfg.dropout_rate > 0.0:
            x = layers.Dropout(rate=resolved_cfg.dropout_rate, name="dropout_2")(x)
    else:
        x = layers.LSTM(
            units=resolved_cfg.lstm_units,
            return_sequences=False,
            name="lstm_layer_1",
        )(x)
        if resolved_cfg.dropout_rate > 0.0:
            x = layers.Dropout(rate=resolved_cfg.dropout_rate, name="dropout_1")(x)

    # Optional intermediate projection
    if resolved_cfg.dense_units > 0:
        x = layers.Dense(
            units=resolved_cfg.dense_units,
            activation=resolved_cfg.dense_activation,
            name="dense_projection",
        )(x)
        if resolved_cfg.dropout_rate > 0.0:
            x = layers.Dropout(rate=resolved_cfg.dropout_rate / 2.0, name="dropout_dense")(x)

    # Final regression output layer (linear activation)
    outputs = layers.Dense(
        units=resolved_cfg.output_units,
        activation="linear",
        name="multi_horizon_output",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"LSTM_{resolved_cfg.lstm_layers}L_{resolved_cfg.lstm_units}U")

    # Compile model with Adam optimizer and configured loss
    opt = optimizers.Adam(learning_rate=resolved_cfg.learning_rate)
    loss_obj = build_loss(resolved_cfg.loss_function, delta=resolved_cfg.huber_delta)

    model.compile(
        optimizer=opt,
        loss=loss_obj,
        metrics=[metrics.MeanAbsoluteError(name="mae"), metrics.MeanSquaredError(name="mse")],
    )

    # Attach architecture config directly to model instance for downstream audit
    setattr(model, "arch_config", resolved_cfg)

    logger.debug(
        "Built LSTM model with input shape (%d, %d), %d layers, %d units, %d outputs.",
        resolved_cfg.sequence_length,
        resolved_cfg.num_features,
        resolved_cfg.lstm_layers,
        resolved_cfg.lstm_units,
        resolved_cfg.output_units,
    )

    return model


__all__ = [
    "LSTMArchitectureConfig",
    "DEFAULT_HORIZONS",
    "HORIZON_NAMES",
    "build_loss",
    "build_lstm_model",
]

