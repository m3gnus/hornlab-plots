"""Smoke tests: each public plot function produces a base64 PNG without
crashing on a small synthetic dataset.

These tests prove the package wires up correctly — they don't verify
visual parity with WG's legacy renderer (that's covered separately in
``test_wg_parity.py``).
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

import hornlab_plots as hlp


def _synthetic_legacy_dict(n_freq=12, n_angle=37, on_axis_db=0.0, edge_db=-15.0):
    """Build a fake directivity payload in the legacy WG shape.

    Returns ``(frequencies, directivity_dict)``.
    """
    freqs = np.geomspace(200.0, 16000.0, n_freq).tolist()
    angles = np.linspace(-90.0, 90.0, n_angle)

    # Smooth bell shape per frequency, on-axis = 0 dB, edge ≈ -15 dB.
    def per_freq_pattern(fi):
        # Mild frequency-dependent narrowing: high freqs slightly tighter.
        sigma_deg = 60.0 - fi * 2.0
        sigma_deg = max(sigma_deg, 20.0)
        db = on_axis_db + (edge_db - on_axis_db) * (1 - np.exp(-(angles ** 2) / (2 * sigma_deg ** 2)))
        return [[float(a), float(d)] for a, d in zip(angles, db)]

    directivity = {
        "horizontal": [per_freq_pattern(fi) for fi in range(n_freq)],
        "vertical": [per_freq_pattern(fi) for fi in range(n_freq)],
    }
    return freqs, directivity


def _is_valid_b64_png(value):
    """Best-effort check: decodes as base64 and starts with the PNG magic."""
    assert isinstance(value, str), f"expected str, got {type(value).__name__}"
    raw = base64.b64decode(value)
    return raw.startswith(b"\x89PNG\r\n\x1a\n")


def test_directivity_heatmap_from_legacy_dict_returns_png():
    freqs, directivity = _synthetic_legacy_dict()
    b64 = hlp.directivity_heatmap_from_legacy_dict(freqs, directivity)
    assert _is_valid_b64_png(b64)


def test_directivity_heatmap_from_legacy_dict_h_only():
    freqs, directivity = _synthetic_legacy_dict()
    directivity = {"horizontal": directivity["horizontal"]}
    b64 = hlp.directivity_heatmap_from_legacy_dict(freqs, directivity)
    assert _is_valid_b64_png(b64)


def test_directivity_heatmap_from_legacy_dict_collapses_symmetric():
    """H == V → single-panel composite. Smoke: still produces a PNG."""
    freqs, directivity = _synthetic_legacy_dict()
    # H and V identical → check_symmetry → single-panel composite
    directivity["vertical"] = [list(row) for row in directivity["horizontal"]]
    b64 = hlp.directivity_heatmap_from_legacy_dict(freqs, directivity)
    assert _is_valid_b64_png(b64)


def test_directivity_heatmap_from_legacy_dict_empty_returns_none():
    assert hlp.directivity_heatmap_from_legacy_dict([], {}) is None
    assert hlp.directivity_heatmap_from_legacy_dict([100, 200], {}) is None
    assert hlp.directivity_heatmap_from_legacy_dict([100, 200], {"horizontal": []}) is None


def test_directivity_heatmap_b64_with_title_kwarg():
    freqs = np.geomspace(200.0, 16000.0, 12)
    angles = np.linspace(-90.0, 90.0, 37)
    sigma = 40.0
    spl = (-15.0) * (1 - np.exp(-(angles ** 2) / (2 * sigma ** 2)))
    spl_db = np.tile(spl[:, None], (1, freqs.size))

    b64 = hlp.directivity_heatmap_b64(freqs, angles, spl_db, title="Numpy test")
    assert _is_valid_b64_png(b64)


def test_prepare_polar_line_data_selects_nearest_and_normalizes():
    freqs = np.array([100.0, 200.0, 400.0])
    angles = np.array([0.0, 30.0, 60.0, 90.0])
    db = np.array([
        [10.0, 7.0, 4.0, 1.0],
        [12.0, 8.0, 2.0, -2.0],
        [14.0, 9.0, 1.0, -5.0],
    ])

    out_angles, selected_freqs, rows = hlp.prepare_polar_line_data(
        freqs,
        angles,
        db,
        [180.0, 390.0],
        angle_max_deg=60.0,
    )

    np.testing.assert_allclose(out_angles, [0.0, 30.0, 60.0])
    np.testing.assert_allclose(selected_freqs, [200.0, 400.0])
    np.testing.assert_allclose(rows, [[0.0, -4.0, -10.0], [0.0, -5.0, -13.0]])


def test_polar_line_b64_accepts_complex_pressure():
    freqs = np.array([100.0, 200.0, 400.0])
    angles = np.linspace(0.0, 90.0, 7)
    amp = np.vstack([
        1.0 - 0.002 * angles,
        1.0 - 0.004 * angles,
        1.0 - 0.006 * angles,
    ])
    pressure = amp * np.exp(1j * np.deg2rad(angles))[None, :]

    b64 = hlp.polar_line_b64(
        freqs,
        angles,
        pressure,
        [100.0, 350.0],
        angle_max_deg=90.0,
        title="H fixed-frequency polar",
    )
    assert _is_valid_b64_png(b64)


def test_save_polar_line_plot_to_disk(tmp_path):
    freqs = np.array([100.0, 200.0, 400.0])
    angles = np.linspace(0.0, 90.0, 7)
    db = np.tile(-0.1 * angles[None, :], (freqs.size, 1))

    out = tmp_path / "polar_lines.png"
    result = hlp.save_polar_line_plot(
        out,
        freqs,
        angles,
        db,
        [100.0, 400.0],
        angle_max_deg=90.0,
    )
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 500


def test_frequency_response_b64():
    freqs = np.geomspace(100.0, 20000.0, 30)
    spl = 100.0 + 3.0 * np.sin(np.log10(freqs))
    b64 = hlp.frequency_response_b64(freqs, spl)
    assert _is_valid_b64_png(b64)


def test_directivity_index_b64_flat_list():
    freqs = np.geomspace(100.0, 20000.0, 30)
    di = 6.0 + 3.0 * np.log10(freqs / 1000.0)
    b64 = hlp.directivity_index_b64(freqs, di.tolist())
    assert _is_valid_b64_png(b64)


def test_directivity_index_b64_per_plane_dict():
    freqs = np.geomspace(100.0, 20000.0, 30)
    di_h = (6.0 + 3.0 * np.log10(freqs / 1000.0)).tolist()
    di_v = (5.0 + 3.0 * np.log10(freqs / 1000.0)).tolist()
    b64 = hlp.directivity_index_b64(freqs, {"horizontal": di_h, "vertical": di_v})
    assert _is_valid_b64_png(b64)


def test_impedance_b64():
    freqs = np.geomspace(100.0, 20000.0, 30)
    real = 400.0 + 50.0 * np.sin(np.log10(freqs) * 2)
    imag = 100.0 * np.cos(np.log10(freqs) * 2)
    b64 = hlp.impedance_b64(freqs, real, imag)
    assert _is_valid_b64_png(b64)


def test_save_directivity_plot_to_disk(tmp_path):
    freqs, directivity = _synthetic_legacy_dict()
    out = tmp_path / "directivity.png"
    result = hlp.save_directivity_plot(out, freqs, directivity)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 500  # non-empty PNG


def test_save_impedance_plot_to_disk(tmp_path):
    freqs = np.geomspace(100.0, 20000.0, 30)
    real = 400.0 + 50.0 * np.sin(np.log10(freqs) * 2)
    imag = 100.0 * np.cos(np.log10(freqs) * 2)
    out = tmp_path / "impedance.png"
    result = hlp.save_impedance_plot(out, freqs, real, imag)
    assert result == out
    assert out.exists()


def test_render_all_charts_b64():
    freqs, directivity = _synthetic_legacy_dict()
    payload = {
        "frequencies": list(freqs),
        "spl": [100.0 + 0.5 * i for i in range(len(freqs))],
        "di": [6.0 + 0.1 * i for i in range(len(freqs))],
        "impedance_real": [400.0 + 5.0 * i for i in range(len(freqs))],
        "impedance_imaginary": [10.0 * i for i in range(len(freqs))],
        "directivity": directivity,
    }
    out = hlp.render_all_charts_b64(payload)
    assert _is_valid_b64_png(out["frequency_response"])
    assert _is_valid_b64_png(out["directivity_index"])
    assert _is_valid_b64_png(out["impedance"])
    assert _is_valid_b64_png(out["directivity_map"])


def test_theme_constants_importable():
    """The Arctic Night palette must be importable from hornlab_plots.style."""
    from hornlab_plots.style import (
        AXES_BG,
        CONTOUR_OUTLINE,
        FIGURE_BG,
        HEATMAP_CMAP,
        REFERENCE_CONTOUR_COLOR,
        SPINE_COLOR,
        TEXT_COLOR,
        TICK_COLOR,
    )
    # Sanity check: the colors are hex strings.
    for color in (FIGURE_BG, AXES_BG, TEXT_COLOR, TICK_COLOR,
                  SPINE_COLOR, CONTOUR_OUTLINE, REFERENCE_CONTOUR_COLOR):
        assert isinstance(color, str) and color.startswith("#")
    # And HEATMAP_CMAP is a colormap.
    assert HEATMAP_CMAP.name == "waveguide_arctic"


def test_legacy_grid_helpers_match_wg():
    """`log_grid_lines` and `preferred_frequency_ticks` must produce the
    same values WG's `directivity_plot.py` did. Pin a few known cases."""
    # Decade-mantissa sub-boundaries 1, 2, 3, 5 within [100, 1000]
    lines = hlp.log_grid_lines(100.0, 1000.0)
    assert 100.0 in lines and 200.0 in lines and 300.0 in lines and 500.0 in lines and 1000.0 in lines

    # Preferred ticks at 100 Hz steps within 100-1000:
    ticks = hlp.preferred_frequency_ticks(100.0, 1000.0)
    for expected in (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000):
        assert any(abs(t - expected) < 1e-3 for t in ticks), f"missing tick {expected} in {ticks}"


def test_optimizer_isobar_jet_template(tmp_path):
    """Optimizer-Dashboard jet-style isobar template produces a PNG."""
    from hornlab_plots.templates import optimizer_isobar_jet

    angles = np.linspace(-180.0, 180.0, 73)
    freqs = np.geomspace(200.0, 20000.0, 40)
    sigma = 60.0
    spl = (-25.0) * (1 - np.exp(-(angles ** 2) / (2 * sigma ** 2)))
    spl_matrix = np.tile(spl[:, None], (1, freqs.size))

    out = tmp_path / "isobar_h.png"
    optimizer_isobar_jet(
        out,
        angle_deg=angles,
        freqs_hz=freqs,
        spl_matrix=spl_matrix,
        title="Horizontal Isobar",
        clip_min_db=-30.0,
        clip_max_db=0.0,
    )
    assert out.exists()
    assert out.stat().st_size > 500
