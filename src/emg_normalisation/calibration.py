"""Hardware-agnostic amplifier-gain removal."""

from __future__ import annotations

import pandas as pd

MANUSCRIPT_AMPLIFIER_GAIN = 1000.0
INPUT_UNIT = "mV"
_UNIT_TO_MV = {"V": 1000.0, "mV": 1.0, "uV": 0.001, "µV": 0.001}


def removeGain(
    frame: pd.DataFrame,
    amplifier_gain: float,
    *,
    stored_unit: str = "V",
) -> pd.DataFrame:
    """Remove a known amplifier gain and return input-referred EMG in mV.

    The conversion is independent of amplifier brand. For stored volts, it is
    ``input_mV = stored_V * 1000 / amplifier_gain``. The time column is copied
    without modification.

    Parameters
    ----------
    frame:
        DataFrame with time in seconds and amplified EMG channels.
    amplifier_gain:
        Dimensionless voltage gain applied by the acquisition amplifier.
        This value must be supplied explicitly.
    stored_unit:
        Voltage unit used by the stored EMG channels: ``"V"``, ``"mV"``,
        ``"uV"`` or ``"µV"``.

    Returns
    -------
    pandas.DataFrame
        Copy with EMG channels expressed in input-referred mV.

    Raises
    ------
    ValueError
        If ``amplifier_gain`` is not positive or ``stored_unit`` is unsupported.

    Examples
    --------
    >>> import pandas as pd
    >>> stored = pd.DataFrame({"time": [0.0], "r_vaslat": [0.25]})
    >>> removeGain(stored, amplifier_gain=1000)["r_vaslat"].iloc[0]
    0.25
    """

    if amplifier_gain <= 0:
        raise ValueError("amplifier_gain must be positive")
    if stored_unit not in _UNIT_TO_MV:
        raise ValueError(
            f"stored_unit must be one of {sorted(_UNIT_TO_MV)}; got {stored_unit!r}"
        )
    conversion = _UNIT_TO_MV[stored_unit] / amplifier_gain
    output = frame.copy()
    for column in output.columns:
        if column != "time":
            output[column] = pd.to_numeric(output[column], errors="coerce") * conversion
    return output


def remove_gain(
    frame: pd.DataFrame,
    amplifier_gain: float,
    *,
    stored_unit: str = "V",
) -> pd.DataFrame:
    """Snake-case alias for :func:`removeGain`."""

    return removeGain(frame, amplifier_gain=amplifier_gain, stored_unit=stored_unit)


def calibration_metadata() -> dict[str, object]:
    """Return the gain-removal assumptions embedded in model provenance."""

    return {
        "version": "gain_calibrated_v1",
        "input_unit": INPUT_UNIT,
        "sensor": "Cometa MiniWave",
        "sensor_gain": MANUSCRIPT_AMPLIFIER_GAIN,
        "vicon_source_gain_default": 5.0,
        "vicon_source_gain_overrides": {
            "C01": {"r_vasmed": 10.0},
            "C02": {"r_vasmed": 10.0},
            "C03": {"r_vasmed": 10.0},
            "C04": {"r_vasmed": 10.0},
            "C05": {"r_vasmed": 10.0},
            "C07": {"l_vasmed": 10.0},
        },
        "formula": "input_mV = stored_mot_V * 1000 / sensor_gain",
        "assumption": "Vicon Nexus SourceGain was already applied before .mot export",
    }
