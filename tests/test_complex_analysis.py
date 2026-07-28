from __future__ import annotations

import numpy as np
import pytest

import hornlab_plots.complex_analysis as complex_analysis
from hornlab_plots.complex_analysis import (
    _di_from_magnitude,
    _patterns_to_complex,
    _pressure_at_angle,
    _propagation_phase,
    _resample_frequency_onto,
    _resample_pressure_at_angle,
    _resample_theta_onto,
)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_di_kernel_matches_legacy_expression_without_mutating_input(dtype):
    theta = np.linspace(0.0, 180.0, 5)
    pressure = np.asarray(
        [
            [1.0 + 2.0j, 2.0 + 1.0j, 3.0 - 1.0j, 2.0 - 2.0j, 1.0 - 1.0j],
            [2.0 + 0.0j, 1.0 + 3.0j, 4.0 + 1.0j, 3.0 - 2.0j, 2.0 - 1.0j],
        ],
        dtype=dtype,
    )
    pressure_before = pressure.copy()
    theta_rad = np.deg2rad(theta)
    sin_theta = np.sin(theta_rad)
    pressure_squared = np.abs(pressure) ** 2
    norm = max(
        float(complex_analysis._trapz(sin_theta, theta_rad)),
        1e-30,
    )
    mean = (
        complex_analysis._trapz(
            pressure_squared * sin_theta[None, :],
            theta_rad,
            axis=1,
        )
        / norm
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = 10.0 * np.log10(
            (pressure_squared[:, 0] + 1e-30) / (mean + 1e-30)
        )

    actual = _di_from_magnitude(pressure, theta)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(pressure, pressure_before)


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


def test_frequency_resampling_preserves_log_magnitude_and_residual_phase():
    frequencies = np.geomspace(100.0, 10000.0, 5)
    targets = np.geomspace(150.0, 8000.0, 9)
    log_frequencies = np.log(frequencies)
    log_targets = np.log(targets)
    distance_m = 2.0
    magnitude_db = np.column_stack(
        (
            3.0 + 2.0 * log_frequencies,
            -4.0 + 0.5 * log_frequencies,
        )
    )
    residual_phase = np.column_stack(
        (
            -0.2 + 0.03 * log_frequencies,
            0.4 - 0.02 * log_frequencies,
        )
    )
    pressure = 10.0 ** (magnitude_db / 20.0) * np.exp(
        1j
        * (
            residual_phase
            + _propagation_phase(frequencies, distance_m)[:, None]
        )
    )

    result = _resample_frequency_onto(
        targets, frequencies, pressure, distance_m
    )

    expected_magnitude_db = np.column_stack(
        (
            3.0 + 2.0 * log_targets,
            -4.0 + 0.5 * log_targets,
        )
    )
    expected_residual_phase = np.column_stack(
        (
            -0.2 + 0.03 * log_targets,
            0.4 - 0.02 * log_targets,
        )
    )
    expected = 10.0 ** (expected_magnitude_db / 20.0) * np.exp(
        1j
        * (
            expected_residual_phase
            + _propagation_phase(targets, distance_m)[:, None]
        )
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_exact_angle_resampling_skips_unused_pressure_columns(monkeypatch):
    frequencies = np.array([100.0, 200.0, 400.0])
    theta = np.array([-10.0, 0.0, 10.0])
    pressure = np.arange(9, dtype=float).reshape(3, 3).astype(complex)
    seen_shapes = []

    def record_resample(freqs_target, freqs_src, selected, distance_m):
        seen_shapes.append(selected.shape)
        np.testing.assert_array_equal(freqs_target, frequencies)
        np.testing.assert_array_equal(freqs_src, frequencies)
        assert distance_m == 2.0
        return selected

    monkeypatch.setattr(
        complex_analysis, "_resample_frequency_onto", record_resample
    )

    result = _resample_pressure_at_angle(
        frequencies, frequencies, pressure, theta, 2.0, 0.0
    )

    assert seen_shapes == [(3, 1)]
    np.testing.assert_array_equal(result, pressure[:, 1])


def test_inexact_angle_resampling_keeps_frequency_then_angle_order():
    frequencies = np.geomspace(100.0, 1000.0, 4)
    targets = np.geomspace(125.0, 800.0, 7)
    theta = np.array([-10.0, 10.0])
    pressure = np.column_stack(
        (
            np.exp(1j * frequencies / 1000.0),
            2.0 * np.exp(1j * frequencies / 1200.0),
        )
    )

    expected = _pressure_at_angle(
        _resample_frequency_onto(targets, frequencies, pressure, 2.0),
        theta,
        0.0,
    )
    result = _resample_pressure_at_angle(
        targets, frequencies, pressure, theta, 2.0, 0.0
    )

    np.testing.assert_array_equal(result, expected)


def test_scalar_and_grid_angle_interpolation_match():
    theta = np.array([-10.0, 10.0])
    pressure = np.array(
        [
            [1.0 + 0.0j, 0.0 + 4.0j],
            [complex(np.nan, np.nan), 2.0 + 0.0j],
        ]
    )

    scalar = _pressure_at_angle(pressure, theta, 0.0)
    grid = _resample_theta_onto(pressure, theta, np.array([0.0]))[:, 0]

    np.testing.assert_array_equal(scalar, grid)
    np.testing.assert_allclose(scalar[0], np.sqrt(2.0) * (1.0 + 1.0j))
    assert np.isnan(scalar[1])


def test_impulse_response_is_invariant_to_frequency_order(monkeypatch):
    captured = []

    def capture_response(fig, out, suptitle=""):
        response = next(
            line for line in fig.axes[0].get_lines()
            if line.get_label() == "0°"
        )
        captured.append((response.get_xdata(), response.get_ydata()))
        complex_analysis.plt.close(fig)
        return out

    monkeypatch.setattr(complex_analysis, "_finish_fig", capture_response)

    frequencies = np.array([100.0, 200.0, 400.0, 800.0])
    residual_phase = 0.2 * np.log(frequencies)
    pressure = (
        np.linspace(1.0, 2.0, len(frequencies))
        * np.exp(
            1j
            * (
                residual_phase
                + _propagation_phase(frequencies, 2.0)
            )
        )
    )[:, None]
    sorted_source = complex_analysis.ComplexDirectivity(
        frequencies,
        np.array([0.0]),
        pressure,
        None,
    )
    complex_analysis.plot_impulse_response(
        sorted_source, "sorted.png", angles_deg=(0.0,)
    )

    order = np.array([2, 0, 3, 1])
    permuted_source = complex_analysis.ComplexDirectivity(
        frequencies[order],
        np.array([0.0]),
        pressure[order],
        None,
    )
    complex_analysis.plot_impulse_response(
        permuted_source, "permuted.png", angles_deg=(0.0,)
    )

    np.testing.assert_array_equal(captured[0][0], captured[1][0])
    np.testing.assert_allclose(captured[0][1], captured[1][1])
