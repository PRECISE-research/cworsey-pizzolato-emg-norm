"""Task-cycle segmentation helpers used by manuscript evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def time_normalise(values: np.ndarray, points: int) -> np.ndarray:
    """Resample one finite movement segment to a fixed number of points."""

    signal = np.asarray(values, dtype=float)
    valid = np.isfinite(signal)
    if valid.sum() < 2:
        return np.full(points, np.nan)
    source = np.linspace(0.0, 1.0, len(signal))[valid]
    target = np.linspace(0.0, 1.0, points)
    if valid.sum() >= 4:
        return CubicSpline(source, signal[valid])(target)
    return np.interp(target, source, signal[valid])


def crank_cycle_boundaries(angle_deg: np.ndarray, side: str) -> np.ndarray:
    """Find manuscript crank-cycle boundaries for one limb.

    Right-side cycles use upward 180-degree crossings. Left-side cycles use
    positive-to-negative jumps in wrapped angle, aligning both sides to top
    dead centre at 0 degrees.
    """

    angle = np.mod(np.asarray(angle_deg, dtype=float), 360.0)
    if side.lower().startswith("r"):
        shifted = angle - 180.0
        return np.flatnonzero((shifted[:-1] < 0) & (shifted[1:] >= 0)) + 1
    if side.lower().startswith("l"):
        return np.flatnonzero(np.diff(angle) < -180.0) + 1
    raise ValueError("side must start with 'l' or 'r'")


def ground_contact_intervals(
    vertical_force_n: np.ndarray,
    threshold_n: float = 20.0,
    minimum_samples: int = 2,
) -> np.ndarray:
    """Find gait contacts from a vertical ground-reaction-force threshold.

    Parameters
    ----------
    vertical_force_n:
        One-dimensional vertical force in newtons.
    threshold_n:
        Contact threshold. The manuscript used 20 N together with the right
        calcaneus-marker event audit.
    minimum_samples:
        Minimum number of consecutive above-threshold samples.

    Returns
    -------
    numpy.ndarray
        Integer ``(n_contacts, 2)`` array of inclusive start and exclusive end
        indices.
    """

    force = np.asarray(vertical_force_n, dtype=float)
    active = np.isfinite(force) & (force >= threshold_n)
    transitions = np.diff(np.pad(active.astype(int), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    intervals = np.column_stack((starts, stops))
    return intervals[(stops - starts) >= int(minimum_samples)]


def normalise_segments(
    values: np.ndarray,
    intervals: np.ndarray,
    points: int,
) -> np.ndarray:
    """Time-normalise multiple signal segments to a common length.

    Parameters
    ----------
    values:
        One-dimensional signal.
    intervals:
        Integer start/stop pairs, with stop exclusive.
    points:
        Number of output points per segment.

    Returns
    -------
    numpy.ndarray
        Array shaped ``(n_segments, points)``.
    """

    signal = np.asarray(values, dtype=float)
    bounds = np.asarray(intervals, dtype=int)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("intervals must have shape (n_segments, 2)")
    output = []
    for start, stop in bounds:
        if start < 0 or stop > len(signal) or stop <= start:
            raise ValueError(f"invalid segment [{start}, {stop}) for length {len(signal)}")
        output.append(time_normalise(signal[start:stop], points))
    return np.vstack(output) if output else np.empty((0, points))


def select_analysis_window(
    frame: pd.DataFrame,
    start_s: float,
    end_s: float,
) -> pd.DataFrame:
    """Select a validated maximal-effort analysis window by time."""

    if "time" not in frame:
        raise ValueError("frame must include a time column")
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")
    selected = frame.loc[frame["time"].between(start_s, end_s)].copy()
    if selected.empty:
        raise ValueError("analysis window does not overlap the supplied frame")
    return selected.reset_index(drop=True)


def cycling_condition(
    mapping: pd.DataFrame,
    participant: str,
    trial: str,
) -> str:
    """Resolve a randomised trial code to its cadence-corrected condition."""

    required = {"participant", "trial", "condition"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"condition mapping is missing columns: {sorted(missing)}")
    match = mapping.loc[
        mapping["participant"].astype(str).eq(str(participant))
        & mapping["trial"].astype(str).eq(str(trial)),
        "condition",
    ]
    if len(match) != 1:
        raise KeyError(f"Expected one condition for {participant}/{trial}, found {len(match)}")
    return str(match.iloc[0])
