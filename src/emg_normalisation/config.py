"""Final manuscript protocol constants and anonymised identifiers."""

from __future__ import annotations

from dataclasses import dataclass

PUBLIC_PARTICIPANTS = (*(f"C{i:02d}" for i in range(1, 8)), "E01")
CYCLING_PARTICIPANTS = tuple(f"C{i:02d}" for i in range(1, 8))

REFERENCE_OVERRIDES = {
    "E01": {"r_tibant": "trial_4d068ec7e2"},
    "C05": {"r_glmax": "trial_b4af0473c9"},
    "C06": {"r_add": "trial_49b3331954"},
}

CHANNEL_EXCLUSIONS = {
    "C01": {"r_add"},
    "C03": {"r_add"},
    "C04": {"r_add"},
    "C05": {"r_add"},
    "C07": {"l_add"},
}

MUSCLE_LABELS = {
    "vaslat": "VL",
    "vasmed": "VM",
    "recfem": "RF",
    "add": "ADD",
    "tibant": "TA",
    "semiten": "ST",
    "bflh": "BFLH",
    "latgas": "LG",
    "medgas": "MG",
    "soleus": "SOL",
    "glmax": "GMax",
    "glmed": "GMed",
    "tfl": "TFL",
}

MODEL_MUSCLES = tuple(MUSCLE_LABELS)


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window parameters expressed in seconds.

    Values reproduce the manuscript settings at 2000 Hz: 200 samples,
    50-sample advance, and at least 60 finite samples.
    """

    window_s: float = 0.100
    step_s: float = 0.025
    min_valid_s: float = 0.030
    kde_grid_points: int = 120


DEFAULT_WINDOW_CONFIG = WindowConfig()


def anatomical_muscle(channel: str) -> str:
    """Return the side-independent anatomical muscle key.

    Parameters
    ----------
    channel:
        EMG channel such as ``"r_vaslat"`` or ``"l_vaslat"``.

    Returns
    -------
    str
        Side-independent key, for example ``"vaslat"``.
    """

    text = str(channel)
    return text[2:] if text.startswith(("l_", "r_")) else text


def is_channel_excluded(participant: str, channel: str) -> bool:
    """Return whether a channel was excluded by the final manual QC audit."""

    return str(channel) in CHANNEL_EXCLUSIONS.get(str(participant), set())
