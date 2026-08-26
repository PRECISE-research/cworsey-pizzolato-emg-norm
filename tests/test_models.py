from pathlib import Path

import numpy as np

from emg_normalisation.models import (
    PublishedModelBundle,
    SklearnEstimatorAdapter,
    StandardisedLinearModel,
)


def test_standardised_linear_round_trip(tmp_path: Path):
    x = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y = 0.1 + 0.2 * x[:, 0] + 0.3 * x[:, 1]
    model = StandardisedLinearModel(("variance_mV2", "kde_fwhm_mV"), np.zeros(2), np.ones(2), 0, np.zeros(2))
    model.fit(x, y)
    path = model.save(tmp_path / "model.json")
    loaded = StandardisedLinearModel.load(path)
    assert np.allclose(model.predict(x), loaded.predict(x))


def test_side_specific_channels_resolve_same_model():
    model = StandardisedLinearModel(("a", "b"), np.zeros(2), np.ones(2), 0, np.ones(2), muscle="vaslat")
    bundle = PublishedModelBundle({"vaslat": model}, {})
    assert bundle.model_for_channel("l_vaslat") is model
    assert bundle.model_for_channel("r_vaslat") is model


def test_sklearn_adapter_contract(tmp_path: Path):
    from sklearn.linear_model import LinearRegression

    adapter = SklearnEstimatorAdapter(LinearRegression()).fit(
        np.array([[0.0], [1.0], [2.0]]),
        np.array([0.0, 1.0, 2.0]),
    )
    assert np.allclose(adapter.predict([[3.0]]), [3.0])
    loaded = SklearnEstimatorAdapter.load(adapter.save(tmp_path / "model.pkl"))
    assert np.allclose(loaded.predict([[3.0]]), [3.0])
