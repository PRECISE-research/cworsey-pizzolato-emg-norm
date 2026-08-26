"""Sliding-window distribution feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowFeatures:
    """Features and locations extracted from one EMG channel.

    Attributes
    ----------
    centre_indices:
        Sample indices corresponding to each valid window centre.
    variance_mV2:
        Population variance of input-referred EMG in mV squared.
    kde_fwhm_mV:
        Full width at half maximum of the Gaussian KDE in mV.
    """

    centre_indices: np.ndarray
    variance_mV2: np.ndarray
    kde_fwhm_mV: np.ndarray

    def as_matrix(self) -> np.ndarray:
        """Return features ordered as variance then KDE FWHM."""

        return np.column_stack((self.variance_mV2, self.kde_fwhm_mV))


def kde_fwhm(values_mV: np.ndarray, grid_points: int = 120) -> float:
    """Calculate Gaussian-KDE full width at half maximum in mV.

    Scott's bandwidth rule is used by SciPy's ``gaussian_kde`` default.
    The Gaussian kernel smooths the empirical amplitude distribution; it does
    not assume that the underlying EMG distribution is Gaussian.
    """

    values = np.asarray(values_mV, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.ptp(values) <= np.finfo(float).eps:
        return 0.0
    standard_deviation = float(np.std(values, ddof=1))
    bandwidth = standard_deviation * values.size ** (-1.0 / 5.0)
    if not np.isfinite(bandwidth) or bandwidth <= np.finfo(float).eps:
        return 0.0
    grid = np.linspace(float(values.min()), float(values.max()), int(grid_points))
    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).sum(axis=1)
    density /= values.size * bandwidth * np.sqrt(2.0 * np.pi)
    half_maximum = 0.5 * float(np.nanmax(density))
    support = grid[density >= half_maximum]
    return float(support[-1] - support[0]) if support.size >= 2 else 0.0


def extract_window_features(
    signal_mV: np.ndarray,
    sampling_rate_hz: float,
    window_s: float = 0.100,
    step_s: float = 0.025,
    min_valid_s: float = 0.030,
    grid_points: int = 120,
) -> WindowFeatures:
    """Extract variance and KDE FWHM from overlapping EMG windows.

    Parameters
    ----------
    signal_mV:
        One-dimensional bandpass-filtered EMG in input-referred mV.
    sampling_rate_hz:
        Signal sampling frequency in Hz.
    window_s, step_s, min_valid_s:
        Window duration, window-centre advance, and minimum finite duration in
        seconds. Defaults reproduce the 100-ms/25-ms/30-ms manuscript setup.
    grid_points:
        Number of amplitude locations used to evaluate each KDE.

    Returns
    -------
    WindowFeatures
        Valid centre indices and two feature vectors.
    """

    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    values = np.asarray(signal_mV, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal_mV must be one-dimensional")
    window = max(2, round(window_s * sampling_rate_hz))
    step = max(1, round(step_s * sampling_rate_hz))
    minimum = max(2, round(min_valid_s * sampling_rate_hz))
    centres: list[int] = []
    variances: list[float] = []
    widths: list[float] = []
    for start in range(0, max(0, len(values) - window + 1), step):
        stop = start + window
        local = values[start:stop]
        finite = local[np.isfinite(local)]
        if finite.size < minimum:
            continue
        centres.append(start + window // 2)
        variances.append(float(np.var(finite, ddof=0)))
        widths.append(kde_fwhm(finite, grid_points=grid_points))
    return WindowFeatures(
        centre_indices=np.asarray(centres, dtype=int),
        variance_mV2=np.asarray(variances, dtype=float),
        kde_fwhm_mV=np.asarray(widths, dtype=float),
    )
