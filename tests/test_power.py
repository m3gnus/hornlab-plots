from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import hornlab_plots as hlp
from hornlab_plots.derived import _build_directivity_power_figure


def _axes(theta_max: float = 180.0) -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(0.0, theta_max, 721), np.arange(0.0, 360.0, 5.0)


def test_full_sphere_monopole_power_and_coverage():
    theta, phi = _axes()
    pressure_value = 0.02 + 0.0j
    pressure = np.full((2, theta.size, phi.size), pressure_value, dtype=np.complex128)

    metrics = hlp.sphere_power_metrics(
        pressure,
        theta,
        phi,
        distance_m=2.0,
    )

    np.testing.assert_allclose(metrics["directivity_index_db"], 0.0, atol=1.0e-12)
    expected_power = (
        4.0
        * np.pi
        * 2.0**2
        * abs(pressure_value) ** 2
        / (2.0 * 1.2041 * 343.0)
    )
    np.testing.assert_allclose(metrics["acoustic_power_w"], expected_power, rtol=1.0e-12)
    np.testing.assert_allclose(metrics["solid_angle_sum_sr"], 4.0 * np.pi, rtol=1.0e-12)
    assert metrics["solid_angle_coverage_fraction"] == 1.0
    assert metrics["method"] == hlp.FULL_SPHERE_POWER_NOTE


def test_full_sphere_cosine_pattern_has_dipole_di():
    theta, phi = _axes()
    pattern = np.cos(np.deg2rad(theta))[None, :, None]
    pressure = np.broadcast_to(pattern, (2, theta.size, phi.size)).astype(np.complex128)

    metrics = hlp.sphere_power_metrics(pressure, theta, phi, distance_m=1.0)

    np.testing.assert_allclose(
        metrics["directivity_index_db"],
        10.0 * np.log10(3.0),
        atol=2.0e-5,
    )


def test_hemisphere_treats_rear_as_silent():
    theta, phi = _axes(90.0)
    pressure = np.ones((1, theta.size, phi.size), dtype=np.complex128)

    metrics = hlp.sphere_power_metrics(pressure, theta, phi, distance_m=1.0)

    np.testing.assert_allclose(
        metrics["directivity_index_db"],
        10.0 * np.log10(2.0),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(metrics["solid_angle_sum_sr"], 2.0 * np.pi, rtol=1.0e-12)
    assert metrics["solid_angle_coverage_fraction"] == 0.5
    expected_power = 2.0 * np.pi / (2.0 * 1.2041 * 343.0)
    np.testing.assert_allclose(metrics["acoustic_power_w"], expected_power, rtol=1.0e-12)
    np.testing.assert_allclose(
        metrics["spatial_average_intensity_w_m2"],
        1.0 / (2.0 * 1.2041 * 343.0),
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        metrics["power_response_db"],
        metrics["on_axis_spl_db"],
        atol=1.0e-12,
    )


def test_hemisphere_cosine_pattern_includes_silent_rear():
    theta, phi = _axes(90.0)
    pattern = np.cos(np.deg2rad(theta))[None, :, None]
    pressure = np.broadcast_to(pattern, (1, theta.size, phi.size)).astype(np.complex128)

    metrics = hlp.sphere_power_metrics(pressure, theta, phi, distance_m=1.0)

    np.testing.assert_allclose(
        metrics["directivity_index_db"],
        10.0 * np.log10(6.0),
        atol=2.0e-5,
    )


def test_flat_theta_major_and_gridded_inputs_match():
    theta = np.linspace(0.0, 180.0, 37)
    phi = np.arange(0.0, 360.0, 30.0)
    grid = (
        (1.0 + 0.3 * np.cos(np.deg2rad(theta))[:, None])
        * np.exp(1j * np.deg2rad(phi))[None, :]
    )[None, :, :]
    theta_flat = np.repeat(theta, phi.size)
    phi_flat = np.tile(phi, theta.size)

    gridded = hlp.sphere_power_metrics(grid, theta, phi, distance_m=1.5)
    flat = hlp.sphere_power_metrics(
        grid.reshape(1, -1),
        theta_flat,
        phi_flat,
        distance_m=1.5,
    )

    assert gridded.keys() == flat.keys()
    for key in gridded:
        if isinstance(gridded[key], np.ndarray):
            np.testing.assert_allclose(gridded[key], flat[key], rtol=0.0, atol=0.0)
        else:
            assert gridded[key] == flat[key]


def test_full_sphere_note_can_be_rendered(tmp_path):
    freqs = np.geomspace(100.0, 10000.0, 8)
    out = tmp_path / "full_sphere_di_power.png"
    result = hlp.save_directivity_power_plot(
        out,
        freqs,
        np.linspace(0.0, 6.0, freqs.size),
        np.linspace(90.0, 84.0, freqs.size),
        footnote=hlp.FULL_SPHERE_POWER_NOTE,
    )
    assert result == out
    assert out.stat().st_size > 500

    fig = _build_directivity_power_figure(
        freqs,
        np.zeros(freqs.size),
        np.zeros(freqs.size),
        footnote=hlp.FULL_SPHERE_POWER_NOTE,
    )
    try:
        assert any(
            text.get_text() == hlp.FULL_SPHERE_POWER_NOTE for text in fig.texts
        )
    finally:
        plt.close(fig)
