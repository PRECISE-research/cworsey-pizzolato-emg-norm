import numpy as np

from emg_normalisation.evaluation import peak_error
from emg_normalisation.segmentation import (
    crank_cycle_boundaries,
    ground_contact_intervals,
    normalise_segments,
    time_normalise,
)


def test_peak_error_direction():
    result = peak_error([0, 0.8], [0, 0.7])
    assert np.isclose(result.signed_percent, 10.0)
    assert np.isclose(result.absolute_percent, 10.0)


def test_crank_boundary_conventions():
    angle = np.array([350, 355, 0, 5, 175, 185, 350, 355, 2, 10])
    assert crank_cycle_boundaries(angle, "left").tolist() == [2, 8]
    assert crank_cycle_boundaries(angle, "right").tolist() == [5]


def test_time_normalise_length():
    assert len(time_normalise(np.arange(20.0), 361)) == 361


def test_ground_contacts_and_segment_normalisation():
    force = np.array([0, 0, 25, 30, 22, 0, 0, 21, 23, 0], dtype=float)
    intervals = ground_contact_intervals(force)
    assert intervals.tolist() == [[2, 5], [7, 9]]
    normalised = normalise_segments(np.arange(10.0), intervals, points=5)
    assert normalised.shape == (2, 5)
