"""Distribution-informed EMG normalisation.

The public API exposes the complete signal-to-envelope workflow used in the
Carrall-Worsey and Pizzolato manuscript, together with extension points for
alternative regression models.
"""

from .calibration import remove_gain, removeGain
from .features import WindowFeatures, extract_window_features
from .io import read_emg_mot, write_emg_mot
from .models import (
    EMGAmplitudeModel,
    PublishedModelBundle,
    SklearnEstimatorAdapter,
    StandardisedLinearModel,
    load_published_model,
)
from .prediction import PredictionResult, predict_trial
from .preprocessing import conventional_linear_envelope, reference_normalise

__all__ = [
    "EMGAmplitudeModel",
    "PredictionResult",
    "PublishedModelBundle",
    "SklearnEstimatorAdapter",
    "StandardisedLinearModel",
    "WindowFeatures",
    "conventional_linear_envelope",
    "extract_window_features",
    "load_published_model",
    "predict_trial",
    "read_emg_mot",
    "reference_normalise",
    "removeGain",
    "remove_gain",
    "write_emg_mot",
]

__version__ = "0.1.2"
