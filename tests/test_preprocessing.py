import numpy as np
import pandas as pd

from emg_normalisation.preprocessing import conventional_linear_envelope, reference_normalise


def test_conventional_envelope_has_no_floor_subtraction():
    time = np.arange(0, 2, 0.001)
    signal = 0.02 + 0.1 * np.sin(2 * np.pi * 50 * time)
    frame = pd.DataFrame({"time": time, "r_vaslat": signal})
    envelope = conventional_linear_envelope(frame)
    assert envelope["r_vaslat"].min() > 0


def test_reference_normalisation_uses_supplied_maximum():
    frame = pd.DataFrame({"time": [0.0, 0.1], "r_vaslat": [0.2, 0.4]})
    result = reference_normalise(frame, {"r_vaslat": 0.8})
    assert np.allclose(result["r_vaslat"], [0.25, 0.5])
