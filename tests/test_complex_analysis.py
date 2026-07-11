from __future__ import annotations

import numpy as np
import pytest

from hornlab_plots.complex_analysis import _patterns_to_complex


def test_complex_patterns_are_reindexed_by_angle_value():
    angles, pressure = _patterns_to_complex(
        [
            [
                [-10.0, 1.0, 10.0],
                [0.0, 2.0, 20.0],
                [10.0, 3.0, 30.0],
            ],
            [
                [10.0, 30.0, 300.0],
                [0.0, 20.0, 200.0],
                [-10.0, 10.0, 100.0],
            ],
        ],
        n_freq=2,
    )

    np.testing.assert_array_equal(angles, [-10.0, 0.0, 10.0])
    np.testing.assert_array_equal(
        pressure,
        [
            [1.0 + 10.0j, 2.0 + 20.0j, 3.0 + 30.0j],
            [10.0 + 100.0j, 20.0 + 200.0j, 30.0 + 300.0j],
        ],
    )


def test_complex_patterns_reject_changed_angle_grid_with_same_count():
    with pytest.raises(RuntimeError, match="theta grid changes"):
        _patterns_to_complex(
            [
                [[-10.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                [[-5.0, 3.0, 0.0], [0.0, 4.0, 0.0]],
            ],
            n_freq=2,
        )


def test_complex_patterns_keep_malformed_points_as_nan():
    angles, pressure = _patterns_to_complex(
        [
            [[-10.0, 1.0, 0.0], [0.0, 2.0, 0.0], [10.0, 3.0, 0.0]],
            [None, [0.0, 20.0, 2.0], []],
        ],
        n_freq=2,
    )

    np.testing.assert_array_equal(angles, [-10.0, 0.0, 10.0])
    assert np.isnan(pressure[1, 0])
    assert pressure[1, 1] == 20.0 + 2.0j
    assert np.isnan(pressure[1, 2])
