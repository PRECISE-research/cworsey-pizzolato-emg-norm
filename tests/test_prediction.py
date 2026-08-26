import numpy as np
import pandas as pd

from emg_normalisation.models import PublishedModelBundle, StandardisedLinearModel
from emg_normalisation.prediction import interpolate_window_predictions, predict_trial


def _bundle() -> PublishedModelBundle:
    model = StandardisedLinearModel(
        ("variance_mV2", "kde_fwhm_mV"),
        feature_mean=np.array([0.0, 0.0]),
        feature_sd=np.array([1.0, 1.0]),
        intercept=0.3,
        coefficients=np.array([0.0, 0.0]),
        muscle="vaslat",
    )
    return PublishedModelBundle({"vaslat": model}, {})


def test_interpolation_is_bounded():
    result = interpolate_window_predictions(
        np.array([10, 20, 30, 40]),
        np.array([-0.1, 0.3, 1.2, 0.4]),
        sample_count=50,
    )
    assert result.min() >= 0
    assert result.max() <= 1


def test_prediction_modes_share_estimated_peak():
    time = np.arange(0, 1.0, 0.001)
    signal = 0.1 * np.sin(2 * np.pi * 70 * time)
    frame = pd.DataFrame({"time": time, "r_vaslat": signal})
    scaled = predict_trial(frame, _bundle(), output_mode="scaled_envelope")
    direct = predict_trial(frame, _bundle(), output_mode="model_trace")
    assert np.isclose(scaled.output["r_vaslat"].max(), 0.3, atol=1e-6)
    assert np.allclose(direct.output["r_vaslat"], 0.3)
    assert not np.allclose(scaled.output["r_vaslat"], direct.output["r_vaslat"])
