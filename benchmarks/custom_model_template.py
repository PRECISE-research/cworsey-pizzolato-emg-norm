"""Minimal model factory used by the benchmark CLI."""

from sklearn.linear_model import HuberRegressor

from emg_normalisation.models import SklearnEstimatorAdapter


def create_model() -> SklearnEstimatorAdapter:
    """Return an unfitted robust linear estimator.

    Replace the estimator below with any scikit-learn-compatible implementation.
    """

    return SklearnEstimatorAdapter(HuberRegressor())
