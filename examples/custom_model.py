"""Minimal custom-estimator factory used by the CLI documentation."""

from sklearn.ensemble import HistGradientBoostingRegressor

from emg_normalisation.models import SklearnEstimatorAdapter


def create_model() -> SklearnEstimatorAdapter:
    """Return a fresh estimator for one LOSO fold and anatomical muscle."""

    return SklearnEstimatorAdapter(
        HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            random_state=1,
        )
    )
