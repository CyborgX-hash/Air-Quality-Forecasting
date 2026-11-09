"""
Unit tests for Phase 2 scaling module (src/phase2/scaling.py).

Verifies:
1. Training-only scaler fitting (no data leakage from validation or test).
2. Proper behavior of StandardScaler and MinMaxScaler.
3. Dedicated target scaling and exact inverse transformation (1D, 2D, scalar).
4. Tolerance to missing values in target during fitting.
5. Serialization and deserialization via joblib (.save() and .load()).
6. MultiStationScalerManager coordination.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.phase2.scaling import MultiStationScalerManager, StationScaler


@pytest.fixture
def synthetic_train_val_dfs():
    """Create distinct train and val DataFrames with different distributions."""
    np.random.seed(42)
    n_train, n_val = 100, 30

    train_df = pd.DataFrame({
        "PM2.5": np.random.uniform(20.0, 80.0, size=n_train),
        "PM10": np.random.uniform(40.0, 150.0, size=n_train),
        "TEMP": np.random.uniform(-5.0, 25.0, size=n_train),
        "wd_sin": np.sin(np.linspace(0, 2 * np.pi, n_train)),
    })
    # Introduce intentional NaNs in target to test resilience
    train_df.loc[5, "PM2.5"] = np.nan
    train_df.loc[12, "PM2.5"] = np.nan

    val_df = pd.DataFrame({
        "PM2.5": np.random.uniform(100.0, 200.0, size=n_val),  # Different regime
        "PM10": np.random.uniform(150.0, 300.0, size=n_val),
        "TEMP": np.random.uniform(20.0, 35.0, size=n_val),
        "wd_sin": np.sin(np.linspace(0, 2 * np.pi, n_val)),
    })

    return train_df, val_df


def test_scaler_initialization():
    """Verify scaler rejects unknown algorithms."""
    s = StationScaler(scaler_type="standard", target_col="PM2.5", station_name="TestStation")
    assert s.scaler_type == "standard"
    assert s.station_name == "TestStation"
    assert not s.is_fitted

    with pytest.raises(ValueError, match="Unsupported scaler_type"):
        StationScaler(scaler_type="invalid_algo")


def test_standard_scaler_train_only_fit(synthetic_train_val_dfs):
    """Verify StandardScaler fits strictly on training statistics."""
    train_df, val_df = synthetic_train_val_dfs
    features = ["PM10", "TEMP", "wd_sin"]

    scaler = StationScaler(scaler_type="standard", target_col="PM2.5")
    scaler.fit(X=train_df[features], y=train_df["PM2.5"])

    assert scaler.is_fitted
    # Scaler mean must match training PM10 mean exactly
    expected_pm10_mean = train_df["PM10"].mean()
    idx = features.index("PM10")
    assert np.isclose(scaler.feature_scaler.mean_[idx], expected_pm10_mean, atol=1e-5)

    # Transform training features -> mean should be 0, std should be 1
    train_scaled = scaler.transform_features(train_df[features])
    assert np.isclose(train_scaled["PM10"].mean(), 0.0, atol=1e-5)
    assert np.isclose(train_scaled["PM10"].std(ddof=0), 1.0, atol=1e-5)

    # Validation features transformed with train params -> mean should NOT be 0
    val_scaled = scaler.transform_features(val_df[features])
    assert not np.isclose(val_scaled["PM10"].mean(), 0.0, atol=1e-2)


def test_minmax_scaler(synthetic_train_val_dfs):
    """Verify MinMaxScaler maps training data into [0, 1]."""
    train_df, _ = synthetic_train_val_dfs
    features = ["PM10", "TEMP"]

    scaler = StationScaler(scaler_type="minmax", target_col="PM2.5")
    scaler.fit(X=train_df[features], y=train_df["PM2.5"])

    train_scaled = scaler.transform_features(train_df[features])
    assert np.isclose(train_scaled["PM10"].min(), 0.0, atol=1e-5)
    assert np.isclose(train_scaled["PM10"].max(), 1.0, atol=1e-5)


def test_target_inverse_transform_roundtrip(synthetic_train_val_dfs):
    """Verify target transform followed by inverse_transform recovers original values."""
    train_df, _ = synthetic_train_val_dfs
    scaler = StationScaler(scaler_type="standard", target_col="PM2.5")
    scaler.fit(X=train_df[["PM10", "TEMP"]], y=train_df["PM2.5"])

    raw_y = train_df["PM2.5"].values
    scaled_y = scaler.transform_target(raw_y)
    recovered_y = scaler.inverse_transform_target(scaled_y)

    # Valid non-nan entries should match original values
    mask = ~np.isnan(raw_y)
    assert np.allclose(recovered_y[mask], raw_y[mask], atol=1e-5)
    # NaN locations should remain NaN
    assert np.isnan(recovered_y[~mask]).all()


def test_target_inverse_transform_shapes():
    """Verify inverse_transform_target works on 1D, 2D, scalar, and multi-step predictions."""
    X = pd.DataFrame({"feat": [10.0, 20.0, 30.0, 40.0]})
    y = np.array([50.0, 100.0, 150.0, 200.0])

    scaler = StationScaler(scaler_type="standard", target_col="PM2.5")
    scaler.fit(X, y)

    # 1. Scalar
    scaled_scalar = scaler.transform_target(100.0)
    inv_scalar = scaler.inverse_transform_target(scaled_scalar)
    assert np.isclose(inv_scalar, 100.0, atol=1e-5)

    # 2. 1D Array (N,)
    preds_1d = np.array([0.0, 1.0, -1.0])
    inv_1d = scaler.inverse_transform_target(preds_1d)
    assert inv_1d.shape == (3,)

    # 3. 2D Array (N, 1)
    preds_2d_col = np.array([[0.0], [1.0]])
    inv_2d_col = scaler.inverse_transform_target(preds_2d_col)
    assert inv_2d_col.shape == (2, 1)

    # 4. Multi-horizon 2D Array (N, H) e.g. 5 horizons
    preds_multi_h = np.zeros((10, 5))
    inv_multi_h = scaler.inverse_transform_target(preds_multi_h)
    assert inv_multi_h.shape == (10, 5)
    assert np.allclose(inv_multi_h, scaler.target_mean_, atol=1e-5)


def test_scaler_save_and_load(tmp_path, synthetic_train_val_dfs):
    """Verify serialized scaler produces identical transformations upon loading."""
    train_df, val_df = synthetic_train_val_dfs
    features = ["PM10", "TEMP"]

    scaler = StationScaler(scaler_type="standard", target_col="PM2.5", station_name="Dongsi")
    scaler.fit(X=train_df[features], y=train_df["PM2.5"])

    save_file = tmp_path / "scaler_Dongsi.joblib"
    scaler.save(save_file)

    assert save_file.is_file()

    loaded_scaler = StationScaler.load(save_file)
    assert loaded_scaler.station_name == "Dongsi"
    assert loaded_scaler.is_fitted
    assert loaded_scaler.feature_names_ == features

    # Compare transformations on validation set
    orig_val_scaled = scaler.transform_features(val_df[features])
    loaded_val_scaled = loaded_scaler.transform_features(val_df[features])
    pd.testing.assert_frame_equal(orig_val_scaled, loaded_val_scaled)

    # Compare target inverse transformations
    test_preds = np.array([0.5, -0.2, 1.8])
    assert np.allclose(
        scaler.inverse_transform_target(test_preds),
        loaded_scaler.inverse_transform_target(test_preds),
    )


def test_multi_station_scaler_manager(tmp_path):
    """Verify MultiStationScalerManager manages station scalers independently."""
    manager = MultiStationScalerManager(scaler_type="standard", target_col="PM2.5")

    df_a = pd.DataFrame({"PM10": [10.0, 20.0, 30.0], "TEMP": [5.0, 10.0, 15.0]})
    df_b = pd.DataFrame({"PM10": [100.0, 200.0, 300.0], "TEMP": [25.0, 30.0, 35.0]})

    manager.fit_station("StationA", df_a, y_train=np.array([10.0, 20.0, 30.0]))
    manager.fit_station("StationB", df_b, y_train=np.array([100.0, 200.0, 300.0]))

    scaler_a = manager.get_scaler("StationA")
    scaler_b = manager.get_scaler("StationB")

    assert np.isclose(scaler_a.target_mean_, 20.0)
    assert np.isclose(scaler_b.target_mean_, 200.0)

    # Save and reload all
    save_dir = tmp_path / "scalers"
    manager.save_all(save_dir)

    reloaded_manager = MultiStationScalerManager.load_all(save_dir)
    assert "StationA" in reloaded_manager.scalers
    assert "StationB" in reloaded_manager.scalers
    assert np.isclose(reloaded_manager.get_scaler("StationA").target_mean_, 20.0)

