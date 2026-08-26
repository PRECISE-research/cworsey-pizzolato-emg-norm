"""Tests for hardware-agnostic amplifier-gain removal."""

import pandas as pd
import pytest

from emg_normalisation.calibration import remove_gain, removeGain


def test_remove_gain_from_stored_volts() -> None:
    stored = pd.DataFrame({"time": [0.0, 0.1], "emg": [0.25, -0.5]})
    result = removeGain(stored, amplifier_gain=1000)
    assert result["time"].tolist() == stored["time"].tolist()
    assert result["emg"].tolist() == pytest.approx([0.25, -0.5])


def test_remove_gain_supports_other_stored_units() -> None:
    stored_mV = pd.DataFrame({"time": [0.0], "emg": [250.0]})
    result = remove_gain(stored_mV, amplifier_gain=1000, stored_unit="mV")
    assert result["emg"].iloc[0] == pytest.approx(0.25)


@pytest.mark.parametrize("gain", [0.0, -1.0])
def test_remove_gain_rejects_non_positive_gain(gain: float) -> None:
    with pytest.raises(ValueError, match="amplifier_gain"):
        removeGain(pd.DataFrame({"time": [0.0], "emg": [1.0]}), gain)
