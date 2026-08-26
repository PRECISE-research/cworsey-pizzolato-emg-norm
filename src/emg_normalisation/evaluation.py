"""Common metrics for published and user-supplied models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeakError:
    """Signed and absolute peak error in percentage points."""

    reference_peak: float
    predicted_peak: float
    signed_percent: float
    absolute_percent: float


def peak_error(reference: np.ndarray, predicted: np.ndarray) -> PeakError:
    """Calculate manuscript peak-excitation error.

    Positive signed error indicates model underestimation; negative error
    indicates overestimation. Because excitation is on a 0-1 scale, the
    difference multiplied by 100 is percentage of the normalised range.
    """

    ref_peak = float(np.nanmax(np.asarray(reference, dtype=float)))
    pred_peak = float(np.nanmax(np.asarray(predicted, dtype=float)))
    signed = 100.0 * (ref_peak - pred_peak)
    return PeakError(ref_peak, pred_peak, signed, abs(signed))


def window_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and coefficient of determination for aligned values."""

    y = np.asarray(reference, dtype=float).reshape(-1)
    y_hat = np.asarray(predicted, dtype=float).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(y_hat)
    if valid.sum() < 2:
        return {"n": int(valid.sum()), "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    residual = y[valid] - y_hat[valid]
    total = np.sum((y[valid] - np.mean(y[valid])) ** 2)
    return {
        "n": int(valid.sum()),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / total) if total > 0 else np.nan,
    }
