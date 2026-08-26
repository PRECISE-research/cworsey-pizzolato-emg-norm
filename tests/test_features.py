import numpy as np

from emg_normalisation.features import extract_window_features, kde_fwhm


def test_kde_fwhm_increases_with_signal_scale():
    rng = np.random.default_rng(42)
    low = rng.normal(0, 0.02, 2000)
    high = 4.0 * low
    assert kde_fwhm(high) > 3.5 * kde_fwhm(low)


def test_extract_window_features_uses_seconds():
    signal = np.sin(np.linspace(0, 20, 1000))
    features = extract_window_features(
        signal,
        sampling_rate_hz=1000,
        window_s=0.1,
        step_s=0.025,
        min_valid_s=0.03,
    )
    assert features.centre_indices[0] == 50
    assert np.all(np.diff(features.centre_indices) == 25)
    assert features.as_matrix().shape[1] == 2
