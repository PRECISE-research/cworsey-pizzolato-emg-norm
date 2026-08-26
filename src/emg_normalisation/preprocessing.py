"""Conventional linear-envelope and reference-normalisation operations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


def infer_sampling_rate(time_s: np.ndarray | pd.Series) -> float:
    """Infer sampling frequency in Hz from a strictly increasing time vector."""

    time = np.asarray(time_s, dtype=float)
    differences = np.diff(time)
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if differences.size == 0:
        raise ValueError("time must contain at least two increasing finite samples")
    return float(1.0 / np.median(differences))


def conventional_linear_envelope(
    bandpass_emg_mV: pd.DataFrame,
    cutoff_hz: float = 6.0,
    order: int = 2,
) -> pd.DataFrame:
    """Construct the conventional non-normalised EMG linear envelope.

    EMG is full-wave rectified and low-pass filtered using a zero-phase
    Butterworth filter. No residual-floor subtraction is applied.

    Parameters
    ----------
    bandpass_emg_mV:
        Time plus bandpass-filtered EMG channels in input-referred mV.
    cutoff_hz:
        Low-pass cutoff frequency in Hz.
    order:
        Butterworth order. The manuscript used order 2.

    Returns
    -------
    pandas.DataFrame
        Time and non-negative envelope channels in mV.
    """

    if "time" not in bandpass_emg_mV.columns:
        raise ValueError("bandpass_emg_mV must contain 'time'")
    fs = infer_sampling_rate(bandpass_emg_mV["time"])
    if not 0 < cutoff_hz < fs / 2:
        raise ValueError("cutoff_hz must be between zero and Nyquist")
    sos = butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    output = pd.DataFrame({"time": bandpass_emg_mV["time"].to_numpy(dtype=float)})
    for channel in bandpass_emg_mV.columns:
        if channel == "time":
            continue
        values = pd.to_numeric(bandpass_emg_mV[channel], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
        output[channel] = np.clip(sosfiltfilt(sos, np.abs(values)), 0.0, None)
    return output


def reference_normalise(
    envelope_mV: pd.DataFrame,
    maxima_mV: Mapping[str, float],
) -> pd.DataFrame:
    """Normalise an envelope using participant-muscle reference maxima.

    Parameters
    ----------
    envelope_mV:
        Time and conventional envelope channels in mV.
    maxima_mV:
        Channel-to-maximum mapping from the participant reference audit.

    Returns
    -------
    pandas.DataFrame
        Time and dimensionless reference-normalised excitation channels.
    """

    output = envelope_mV.copy()
    for channel in output.columns:
        if channel == "time":
            continue
        maximum = float(maxima_mV.get(channel, np.nan))
        if not np.isfinite(maximum) or maximum <= 0:
            output[channel] = np.nan
        else:
            output[channel] = output[channel].to_numpy(dtype=float) / maximum
    return output
