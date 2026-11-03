"""Phase polars mirror only one-sided angular grids."""

import numpy as np
import pytest

from hornlab_plots import complex_analysis as analysis


@pytest.mark.parametrize(
    "theta, phases, expected_theta, expected_phases",
    [
        ([-90, 0, 90], [10, 0, 40], [-90, 0, 90], [10, 0, 40]),
        ([90, 0, -90], [40, 0, 10], [90, 0, -90], [40, 0, 10]),
        ([0, 30, 90], [0, 10, 40], [-90, -30, 0, 30, 90], [40, 10, 0, 10, 40]),
        ([-90, -30, 0], [40, 10, 0], [-90, -30, 0, 30, 90], [40, 10, 0, 10, 40]),
        ([10, 90], [0, 40], [-90, -10, 10, 90], [40, 0, 0, 40]),
        ([90, 30, 0], [40, 10, 0], [-90, -30, 0, 30, 90], [40, 10, 0, 10, 40]),
        ([0, -30, -90], [0, 10, 40], [-90, -30, 0, 30, 90], [40, 10, 0, 10, 40]),
    ],
)
def test_phase_polar_uses_each_signed_direction_once(
    monkeypatch, tmp_path, theta, phases, expected_theta, expected_phases,
):
    captured = []

    def capture(fig, out, **kwargs):
        for line in fig.axes[0].lines:
            captured.append((line.get_xdata().copy(), line.get_ydata().copy()))
        analysis.plt.close(fig)
        return out

    monkeypatch.setattr(analysis, "_finish_fig", capture)
    pressure = np.exp(1j * np.deg2rad([phases]))
    source = analysis.ComplexDirectivity(
        np.array([1000.0]), np.array(theta, dtype=float), pressure, pressure.copy(),
    )
    analysis.plot_phase_polar(source, tmp_path / "phase.png", freqs_to_show=(1000.0,))
    assert len(captured) == 2
    for angles, values in captured:
        np.testing.assert_array_equal(angles, np.deg2rad(expected_theta))
        np.testing.assert_allclose(values, expected_phases, atol=1e-12)
    np.testing.assert_array_equal(source.p_h, pressure)
