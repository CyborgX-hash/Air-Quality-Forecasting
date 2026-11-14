"""LSTM contract tests against the *pipeline's* real tensors (Harsh's model x Ajit's sequences)."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from src.phase2.lstm_model import LSTMArchitectureConfig, build_lstm_model
from src.phase2.predict import load_trained_model
from src.phase2.sequence_generator import generate_station_sequence_bundle
from tests.phase2.conftest import STATIONS

HORIZONS = [1, 3, 6, 12, 24]


@pytest.fixture(scope="module")
def bundle(splits_by_station):
    return generate_station_sequence_bundle(splits_by_station[STATIONS[0]], 24, HORIZONS)


@pytest.fixture(scope="module")
def model(bundle, cfg):
    arch = LSTMArchitectureConfig(
        sequence_length=24, num_features=bundle.X_train.shape[2],
        lstm_units=cfg.model.lstm_units, lstm_layers=cfg.model.lstm_layers,
        dropout_rate=cfg.model.dropout_rate, dense_units=cfg.model.dense_units,
        output_units=cfg.model.output_units, loss_function=cfg.model.loss_function,
        learning_rate=cfg.model.learning_rate, horizons=HORIZONS)
    return build_lstm_model(config=arch, seed=cfg.project.random_seed)


def test_input_and_output_dimensions_match_pipeline_tensors(model, bundle):
    assert model.input_shape == (None,) + bundle.X_train.shape[1:]
    assert model.output_shape == (None, len(HORIZONS))
    assert bundle.y_train.shape[1] == model.output_shape[1]


def test_forward_pass_is_finite(model, bundle):
    out = model.predict(bundle.X_val[:16], verbose=0)
    assert out.shape == (16, len(HORIZONS)) and np.isfinite(out).all()


def test_save_load_roundtrip_preserves_predictions(model, bundle, tmp_path):
    p = tmp_path / "m.keras"
    model.save(p)
    loaded = load_trained_model(p)
    x = bundle.X_test[:16]
    np.testing.assert_allclose(model.predict(x, verbose=0), loaded.predict(x, verbose=0), atol=1e-6)
