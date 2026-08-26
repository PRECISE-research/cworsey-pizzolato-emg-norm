"""Window-centre prediction and the two supported output reconstructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from .features import extract_window_features
from .models import PublishedModelBundle
from .preprocessing import conventional_linear_envelope, infer_sampling_rate

OutputMode = Literal["scaled_envelope", "model_trace"]


@dataclass(frozen=True)
class PredictionResult:
    """Outputs from one bandpass EMG trial."""

    output: pd.DataFrame
    model_trace: pd.DataFrame
    conventional_envelope_mV: pd.DataFrame
    estimated_peak_by_channel: dict[str, float]
    output_mode: OutputMode


def interpolate_window_predictions(
    centre_indices: np.ndarray,
    predictions: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Interpolate bounded predictions to every sample.

    Cubic spline interpolation is used with at least four unique centres;
    otherwise linear interpolation is used. Endpoint values are extended to
    cover samples outside the first and last window centres.
    """

    centres = np.asarray(centre_indices, dtype=float)
    values = np.asarray(predictions, dtype=float)
    valid = np.isfinite(centres) & np.isfinite(values)
    centres = centres[valid]
    values = values[valid]
    unique, indices = np.unique(centres, return_index=True)
    values = values[indices]
    if unique.size < 2:
        return np.full(sample_count, values[0] if values.size else np.nan)
    x = np.arange(sample_count, dtype=float)
    if unique.size >= 4:
        result = CubicSpline(unique, values, extrapolate=False)(x)
        result[x < unique[0]] = values[0]
        result[x > unique[-1]] = values[-1]
    else:
        result = np.interp(x, unique, values)
    return np.clip(result, 0.0, 1.0)


def predict_trial(
    bandpass_emg_mV: pd.DataFrame,
    model_bundle: PublishedModelBundle,
    output_mode: OutputMode = "scaled_envelope",
    window_s: float = 0.100,
    step_s: float = 0.025,
    min_valid_s: float = 0.030,
    kde_grid_points: int = 120,
) -> PredictionResult:
    """Predict normalised excitation from one gain-corrected EMG trial.

    Parameters
    ----------
    bandpass_emg_mV:
        Time and side-specific bandpass EMG channels in input-referred mV.
    model_bundle:
        Published or user-trained side-pooled anatomical-muscle models.
    output_mode:
        ``"scaled_envelope"`` reproduces the manuscript. It scales the
        conventional envelope to the maximum model estimate. ``"model_trace"``
        returns the bounded interpolated window-centre predictions directly.

    Returns
    -------
    PredictionResult
        Selected output, direct model trace, conventional envelope, and trial
        peak estimates.
    """

    if output_mode not in ("scaled_envelope", "model_trace"):
        raise ValueError("output_mode must be 'scaled_envelope' or 'model_trace'")
    if "time" not in bandpass_emg_mV.columns:
        raise ValueError("bandpass_emg_mV must contain 'time'")
    fs = infer_sampling_rate(bandpass_emg_mV["time"])
    envelope = conventional_linear_envelope(bandpass_emg_mV)
    trace = pd.DataFrame({"time": bandpass_emg_mV["time"].to_numpy(dtype=float)})
    scaled = pd.DataFrame({"time": trace["time"]})
    peaks: dict[str, float] = {}
    for channel in bandpass_emg_mV.columns:
        if channel == "time":
            continue
        try:
            model = model_bundle.model_for_channel(channel)
        except KeyError:
            continue
        feature_set = extract_window_features(
            bandpass_emg_mV[channel].to_numpy(dtype=float),
            sampling_rate_hz=fs,
            window_s=window_s,
            step_s=step_s,
            min_valid_s=min_valid_s,
            grid_points=kde_grid_points,
        )
        centre_predictions = np.clip(model.predict(feature_set.as_matrix()), 0.0, 1.0)
        reconstructed = interpolate_window_predictions(
            feature_set.centre_indices,
            centre_predictions,
            len(bandpass_emg_mV),
        )
        trace[channel] = reconstructed
        estimated_peak = float(np.nanmax(reconstructed))
        peaks[channel] = estimated_peak
        conventional = envelope[channel].to_numpy(dtype=float)
        conventional_peak = float(np.nanmax(conventional))
        scaled[channel] = (
            conventional * estimated_peak / conventional_peak
            if conventional_peak > 0
            else np.zeros_like(conventional)
        )
    selected = scaled if output_mode == "scaled_envelope" else trace
    return PredictionResult(
        output=selected,
        model_trace=trace,
        conventional_envelope_mV=envelope,
        estimated_peak_by_channel=peaks,
        output_mode=output_mode,
    )
