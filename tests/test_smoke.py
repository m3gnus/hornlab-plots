"""Smoke tests: each public plot function produces a base64 PNG without
crashing on a small synthetic dataset.

These tests prove the package wires up correctly. The post-deletion canonical
renderer contract is covered separately in ``test_canonical_renderers.py``.
"""

from __future__ import annotations

import base64
import warnings
from pathlib import Path

import matplotlib.figure
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


def _synthetic_beam_shape():
    freqs = np.geomspace(200.0, 20000.0, 30)
    p_values = np.linspace(1.0, 8.0, freqs.size)
    p_values[[5, 17]] = np.nan
    residuals = np.linspace(0.0, 20.0, freqs.size)
    residuals[[8, 21]] = np.nan
    return {
        "frequencies": freqs.tolist(),
        "shape_exponent": [None if np.isnan(value) else float(value) for value in p_values],
        "fit_residual_percent": [None if np.isnan(value) else float(value) for value in residuals],
        "spherical_di_db": np.linspace(0.0, 15.0, freqs.size).tolist(),
    }


def test_forward_beam_shape_b64():
    assert _is_valid_b64_png(hlp.forward_beam_shape_b64(_synthetic_beam_shape()))


def test_forward_beam_shape_b64_empty_returns_none():
    assert hlp.forward_beam_shape_b64({}) is None
    assert hlp.forward_beam_shape_b64({"frequencies": []}) is None


def test_forward_beam_shape_b64_renders_di_only():
    beam_shape = _synthetic_beam_shape()
    beam_shape["shape_exponent"] = [None] * len(beam_shape["frequencies"])
    assert _is_valid_b64_png(hlp.forward_beam_shape_b64(beam_shape))


def test_forward_beam_shape_b64_reference_and_theme():
    beam_shape = _synthetic_beam_shape()
    assert _is_valid_b64_png(hlp.forward_beam_shape_b64(
        beam_shape,
        reference_beam_shape=beam_shape,
        reference_label="Baseline",
        theme="granite",
    ))


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


def test_frequency_response_multi_b64():
    freqs = np.geomspace(100.0, 20000.0, 30)
    lf = 100.0 - 12.0 * np.log10(freqs / 800.0).clip(min=0.0)
    hf = 100.0 + 12.0 * np.log10(freqs / 800.0).clip(max=0.0)
    combined = np.maximum(lf, hf) + 1.5
    b64 = hlp.frequency_response_multi_b64(
        [
            hlp.FrequencyResponseCurve(freqs, lf, "LF LR4 LP", role="lf", crossover=True),
            hlp.FrequencyResponseCurve(freqs, hf, "HF LR4 HP", role="hf", crossover=True),
            hlp.FrequencyResponseCurve(freqs, combined, "Combined", role="combined"),
        ],
        title="Synthetic crossover",
        crossover_hz=800.0,
    )
    assert _is_valid_b64_png(b64)


def test_save_frequency_response_plot_to_disk(tmp_path):
    freqs = np.geomspace(100.0, 20000.0, 16)
    spl = 100.0 + 2.0 * np.sin(np.log10(freqs))
    out = tmp_path / "response.png"
    result = hlp.save_frequency_response_plot(
        out,
        [hlp.FrequencyResponseCurve(freqs, spl, "Combined", role="combined")],
    )
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 500


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


@pytest.mark.parametrize(
    "di",
    [
        [4.0, 5.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ],
)
def test_directivity_index_b64_clips_mismatched_lengths(di):
    freqs = [100.0, 200.0, 400.0, 800.0]

    assert _is_valid_b64_png(hlp.directivity_index_b64(freqs, di))


def test_directivity_index_b64_accepts_one_valid_point_without_axis_warning():
    with warnings.catch_warnings(record=True) as warnings_seen:
        warnings.simplefilter("always")
        rendered = hlp.directivity_index_b64([100.0], [4.0])

    assert not warnings_seen
    assert _is_valid_b64_png(rendered)


def test_render_all_charts_clips_di_when_di_frequencies_are_omitted():
    out = hlp.render_all_charts_b64(
        {
            "frequencies": [100.0, 200.0, 400.0, 800.0],
            "di": [4.0, 5.0],
        }
    )

    assert _is_valid_b64_png(out["directivity_index"])


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


def test_save_impedance_plot_label_overrides_land_on_rendered_figure(tmp_path, monkeypatch):
    freqs = np.geomspace(100.0, 20000.0, 30)
    real = 4.0 + np.sin(np.log10(freqs) * 2)
    imag = 2.0 * np.cos(np.log10(freqs) * 2)
    out = tmp_path / "electrical_impedance.png"
    captured = {}
    original_savefig = matplotlib.figure.Figure.savefig

    def spy_savefig(self, *args, **kwargs):
        ax = self.axes[0]
        captured["title"] = ax.get_title()
        captured["ylabel"] = ax.get_ylabel()
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy_savefig)

    result = hlp.save_impedance_plot(
        out,
        freqs,
        real,
        imag,
        title="Electrical Input Impedance",
        ylabel="|Z| [ohm] / phase-split real+imag [ohm]",
    )

    assert result == out
    assert out.exists()
    assert captured == {
        "title": "Electrical Input Impedance",
        "ylabel": "|Z| [ohm] / phase-split real+imag [ohm]",
    }


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


def test_render_all_charts_b64_beam_shape_wiring():
    with_beam_shape = hlp.render_all_charts_b64({"beam_shape": _synthetic_beam_shape()})
    assert _is_valid_b64_png(with_beam_shape["beam_shape"])

    without_beam_shape = hlp.render_all_charts_b64({})
    assert not without_beam_shape.get("beam_shape")


def test_render_all_charts_warns_when_directivity_heatmap_fails():
    payload = {
        "frequencies": [100.0],
        "directivity": {
            "horizontal": [[[0.0, 0.0], [0.0, -1.0]]],
        },
    }

    with pytest.warns(RuntimeWarning, match="duplicate angle 0"):
        out = hlp.render_all_charts_b64(payload)

    assert out["directivity_map"] is None


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


def test_theme_api_pins_hornlab_defaults_and_restores_context():
    from hornlab_plots import style

    previous = style.get_theme()
    style.set_theme("hornlab")
    try:
        theme = style.get_theme()
        assert theme.name == "hornlab"
        assert theme.figure_bg == "#080D16"
        assert theme.axes_bg == "#0F1927"
        assert theme.text_color == "#C8D8EC"
        assert theme.tick_color == "#6A90B8"
        assert theme.spine_color == "#1E3050"
        assert theme.grid_color == "white"
        assert theme.heatmap_cmap.name == "waveguide_arctic"
        assert theme.diverging_cmap.name == "arctic_diverging"
        assert theme.cyclic_cmap.name == "arctic_cyclic"
        assert theme.line_colors == (
            "#4FC3F7",
            "#E8A83A",
            "#EF5350",
            "#66BB6A",
            "#AB47BC",
            "#78909C",
        )
        assert dict(theme.response_colors) == {
            "lf": "#4FC3F7",
            "mf": "#E8A83A",
            "hf": "#EF5350",
            "combined": "#EAF2FF",
            "raw": "#78909C",
            "other": "#AB47BC",
        }

        assert style.FIGURE_BG == theme.figure_bg
        assert style.AXES_BG == theme.axes_bg
        assert style.TEXT_COLOR == theme.text_color
        assert style.TICK_COLOR == theme.tick_color
        assert style.SPINE_COLOR == theme.spine_color
        assert style.GRID_COLOR == theme.grid_color
        assert style.HEATMAP_CMAP is theme.heatmap_cmap
        assert style.LINE_COLORS == list(theme.line_colors)
        assert style.RESPONSE_COLORS == dict(theme.response_colors)
        assert style.DPI == theme.dpi

        assert style.apply_theme_overrides(line_colors=[]).line_colors == theme.line_colors

        dark = style.get_theme("dark")
        assert dark.name == "dark"
        assert dark is style.DARK_THEME
        assert dark is not style.HORNLAB_THEME
        assert dark.axes_bg == theme.axes_bg

        with pytest.raises(RuntimeError):
            with style.theme_context("dark"):
                assert style.get_theme() is dark
                raise RuntimeError("boom")
        assert style.get_theme() is theme
    finally:
        style.set_theme(previous)


def test_directivity_default_render_uses_hornlab_theme_values():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from hornlab_plots._heatmap import _build_figure_from_planes, _build_planes_from_legacy
    from hornlab_plots.style import AXES_BG, FIGURE_BG, TEXT_COLOR

    freqs, directivity = _synthetic_legacy_dict(n_freq=5, n_angle=9)
    planes = _build_planes_from_legacy(freqs, directivity)
    fig = _build_figure_from_planes(planes)
    try:
        ax = fig.axes[0]
        assert mcolors.to_hex(fig.get_facecolor(), keep_alpha=False).lower() == FIGURE_BG.lower()
        assert mcolors.to_hex(ax.get_facecolor(), keep_alpha=False).lower() == AXES_BG.lower()
        assert ax.xaxis.label.get_color() == TEXT_COLOR
    finally:
        plt.close(fig)


def test_heatmap_reference_contour_legend_entry_exists():
    import matplotlib.pyplot as plt
    from hornlab_plots._heatmap import _build_figure_from_planes, _build_planes_from_legacy

    freqs, directivity = _synthetic_legacy_dict(n_freq=8, n_angle=21)
    planes = _build_planes_from_legacy(freqs, {"horizontal": directivity["horizontal"]})

    fig = _build_figure_from_planes(planes, reference_level=-6.0)
    try:
        legend = fig.axes[0].get_legend()
        assert legend is not None
        assert "ref @ -6 dB" in [text.get_text() for text in legend.get_texts()]
    finally:
        plt.close(fig)


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


def test_templates_export_no_public_presets():
    import hornlab_plots.templates as templates

    assert templates.__all__ == []
