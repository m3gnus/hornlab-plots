from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Partial spherical caps: refused unless their coverage is declared
# ---------------------------------------------------------------------------


def _cap_axes(theta_max: float, n_theta: int = 7) -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(0.0, theta_max, n_theta), np.arange(0.0, 360.0, 30.0)


@pytest.mark.parametrize("theta_max", [30.0, 60.0, 89.0, 120.0, 150.0, 179.0])
def test_partial_cap_is_refused_without_a_declared_coverage(theta_max):
    theta, phi = _cap_axes(theta_max)
    pressure = np.ones((1, theta.size, phi.size), dtype=np.complex128)

    with pytest.raises(ValueError, match="must end at 90 deg .* or 180 deg"):
        hlp.sphere_power_metrics(pressure, theta, phi, distance_m=1.0)


@pytest.mark.parametrize("theta_max", [90.0, 180.0])
def test_full_cap_endpoints_are_still_accepted(theta_max):
    theta, phi = _cap_axes(theta_max, n_theta=181)
    pressure = np.ones((1, theta.size, phi.size), dtype=np.complex128)

    metrics = hlp.sphere_power_metrics(pressure, theta, phi, distance_m=1.0)

    expected_fraction = 0.5 if theta_max == 90.0 else 1.0
    assert metrics["solid_angle_coverage_fraction"] == expected_fraction
    np.testing.assert_allclose(
        metrics["solid_angle_sum_sr"],
        4.0 * np.pi * expected_fraction,
        rtol=1.0e-12,
    )
    assert metrics["method"] == hlp.FULL_SPHERE_POWER_NOTE


@pytest.mark.parametrize("theta_max", [60.0, 90.0, 120.0, 180.0])
def test_declared_cap_integrates_only_the_sampled_solid_angle(theta_max):
    theta, phi = _cap_axes(theta_max, n_theta=181)
    pressure_value = 0.5 + 0.0j
    pressure = np.full((2, theta.size, phi.size), pressure_value, dtype=np.complex128)

    metrics = hlp.sphere_power_metrics(
        pressure,
        theta,
        phi,
        distance_m=2.0,
        cap_theta_max_deg=theta_max,
    )

    # Solid angle of a cap of half-angle theta_max, not of a hemisphere.
    cap_sr = 2.0 * np.pi * (1.0 - np.cos(np.deg2rad(theta_max)))
    np.testing.assert_allclose(metrics["solid_angle_sum_sr"], cap_sr, rtol=1.0e-12)
    np.testing.assert_allclose(
        metrics["solid_angle_coverage_fraction"],
        cap_sr / (4.0 * np.pi),
        rtol=1.0e-12,
    )
    expected_power = (
        2.0**2 * cap_sr * abs(pressure_value) ** 2 / (2.0 * 1.2041 * 343.0)
    )
    np.testing.assert_allclose(
        metrics["acoustic_power_w"], expected_power, rtol=1.0e-12
    )
    # A uniform cap radiating into 1/n of the sphere has DI = 10*log10(n).
    np.testing.assert_allclose(
        metrics["directivity_index_db"],
        10.0 * np.log10(4.0 * np.pi / cap_sr),
        atol=1.0e-12,
    )


def test_declared_cap_does_not_claim_a_full_sphere_integration():
    theta, phi = _cap_axes(60.0, n_theta=61)
    pressure = np.ones((1, theta.size, phi.size), dtype=np.complex128)

    metrics = hlp.sphere_power_metrics(
        pressure, theta, phi, distance_m=1.0, cap_theta_max_deg=60.0
    )

    assert metrics["method"] == hlp.PARTIAL_CAP_POWER_NOTE
    assert metrics["approximation"] == hlp.PARTIAL_CAP_POWER_NOTE


def test_partial_cap_power_is_not_the_silently_enlarged_hemisphere_power():
    # The defect this guards: the 0..60 deg cap used to be classified as a
    # hemisphere and its final band stretched to 90 deg, doubling the solid
    # angle and overstating radiated power by 3.01 dB.
    theta, phi = _cap_axes(60.0, n_theta=61)
    pressure = np.ones((1, theta.size, phi.size), dtype=np.complex128)

    metrics = hlp.sphere_power_metrics(
        pressure, theta, phi, distance_m=1.0, cap_theta_max_deg=60.0
    )

    np.testing.assert_allclose(metrics["solid_angle_sum_sr"], np.pi, rtol=1.0e-12)
    assert metrics["solid_angle_sum_sr"] < 2.0 * np.pi


def test_declared_cap_must_match_the_sampled_endpoint():
    theta, phi = _cap_axes(60.0)
    pressure = np.ones((1, theta.size, phi.size), dtype=np.complex128)

    with pytest.raises(ValueError, match="must equal the last theta sample"):
        hlp.sphere_power_metrics(
            pressure, theta, phi, distance_m=1.0, cap_theta_max_deg=90.0
        )
    with pytest.raises(ValueError, match="within \\(0, 180\\] deg"):
        hlp.sphere_power_metrics(
            pressure, theta, phi, distance_m=1.0, cap_theta_max_deg=0.0
        )
