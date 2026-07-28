"""Derived-output renderer tests (Fusion addin ports).

Same style as the other suites: synthetic data in, assert the files render,
and check that key style values honor the active theme.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hornlab_plots as hlp
from hornlab_plots.derived import (
    BEAMWIDTH_SYMMETRY_NOTE,
    POLAR_POWER_APPROXIMATION_NOTE,
    _build_beamwidth_figure,
    _build_directivity_power_figure,
    _build_excursion_figure,
    _build_group_delay_figure,
    _build_interference_figure,
)

SPEED_OF_SOUND_M_S = 343.0


def _synthetic_pressure_pair(n_freq=12, n_angle=13, planes=2):
    """Two engineering-convention pressure grids with real interference.

    Source B sits 0.1 m behind A along the axis projected per angle, so the
    coherent sum develops angle/frequency comb structure.
    """
    freqs = np.geomspace(200.0, 8000.0, n_freq)
    angles = np.linspace(0.0, 90.0, n_angle)
    k = 2.0 * np.pi * freqs / SPEED_OF_SOUND_M_S
    path_m = 0.1 * np.cos(np.deg2rad(angles))
    phase = np.exp(-1j * k[:, None, None] * path_m[None, None, :])
    a = np.ones((n_freq, planes, n_angle), dtype=complex)
    b = 0.8 * np.ones_like(a) * phase
    return freqs, angles, {"A": a, "B": b}


def test_derived_renderer_exports_are_available():
    assert callable(hlp.interference_ratio_db)
    assert callable(hlp.save_interference_heatmap)
    assert callable(hlp.save_directivity_power_plot)
    assert callable(hlp.save_beamwidth_plot)
    assert callable(hlp.save_group_delay_plot)
    assert callable(hlp.save_excursion_plot)


def test_interference_ratio_db_math():
    """0 dB for fully coherent addition; deeply negative for anti-phase."""
    grid = np.full((4, 1, 5), 1.0 + 0.5j, dtype=complex)
    ratio = hlp.interference_ratio_db({"A": grid, "B": grid.copy()})
    np.testing.assert_allclose(ratio, 0.0, atol=1e-12)

    single = hlp.interference_ratio_db({"A": grid})
    np.testing.assert_allclose(single, 0.0, atol=1e-12)

    anti = hlp.interference_ratio_db({"A": grid, "B": -grid})
    assert np.all(anti < -100.0)

    # Subset selection via members and shape validation.
    partial = hlp.interference_ratio_db({"A": grid, "B": -grid, "C": grid}, members=["A", "C"])
    np.testing.assert_allclose(partial, 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        hlp.interference_ratio_db({"A": grid, "B": grid[:, :, :-1]})
    with pytest.raises(ValueError):
        hlp.interference_ratio_db({})


def test_interference_ratio_db_matches_ordered_sum_without_mutating_inputs():
    rng = np.random.default_rng(260728)
    sources = {
        name: rng.standard_normal((7, 2, 5))
        + 1j * rng.standard_normal((7, 2, 5))
        for name in ("LF", "MF", "HF")
    }
    sources["MF"] = sources["MF"].astype(np.complex64)
    originals = {name: grid.copy() for name, grid in sources.items()}
    grids = [np.asarray(grid, dtype=np.complex128) for grid in sources.values()]
    expected = 20.0 * np.log10(
        np.maximum(np.abs(sum(grids)), 1.0e-30)
        / np.maximum(sum(np.abs(grid) for grid in grids), 1.0e-30)
    )

    actual = hlp.interference_ratio_db(sources)

    np.testing.assert_array_equal(actual, expected)
    for name, grid in sources.items():
        np.testing.assert_array_equal(grid, originals[name])


def test_save_interference_heatmap_to_disk(tmp_path):
    freqs, angles, sources = _synthetic_pressure_pair()
    out = tmp_path / "interference.png"
    result = hlp.save_interference_heatmap(
        out,
        freqs,
        sources,
        angles,
        ("horizontal", "vertical"),
        crossovers_hz=[800.0],
        mesh_valid_hz=3600.0,
        mesh_valid_radiating_hz=6000.0,
    )
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 500


def test_save_interference_heatmap_single_plane(tmp_path):
    freqs, angles, sources = _synthetic_pressure_pair(planes=1)
    out = tmp_path / "interference_h.png"
    result = hlp.save_interference_heatmap(out, freqs, sources, angles, ("horizontal",))
    assert result == out
    assert out.exists()


def test_interference_figure_annotates_crossovers_and_validates_shapes():
    freqs, angles, sources = _synthetic_pressure_pair()
    fig = _build_interference_figure(
        freqs,
        sources,
        angles,
        ("horizontal", "vertical"),
        crossovers_hz=[800.0],
    )
    try:
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any("XO 800 Hz" in text for text in texts)
        titles = [ax.get_title() for ax in fig.axes]
        assert "H Driver Interference (0 dB = coherent sum)" in titles
        assert "V Driver Interference (0 dB = coherent sum)" in titles
    finally:
        plt.close(fig)

    with pytest.raises(ValueError):
        _build_interference_figure(freqs, sources, angles, ("horizontal",))  # plane count mismatch


def test_save_directivity_power_plot_to_disk(tmp_path):
    freqs = np.geomspace(100.0, 20000.0, 30)
    di = 5.0 + 4.0 * np.log10(freqs / 100.0)
    power = 92.0 - 2.0 * np.log10(freqs / 100.0)
    out = tmp_path / "di_power.png"
    result = hlp.save_directivity_power_plot(out, freqs, di, power)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 500


def test_directivity_power_figure_legend_and_footnote():
    freqs = np.geomspace(1000.0, 20000.0, 30)
    di = 5.0 + 4.0 * np.log10(freqs / 1000.0)
    power = 92.0 - 2.0 * np.log10(freqs / 1000.0)
    fig = _build_directivity_power_figure(
        freqs,
        di,
        power,
        mesh_valid_hz=3642.0,
        mesh_valid_radiating_hz=11352.0,
    )
    try:
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "DI" in labels
        assert "Power response" in labels
        assert any("mesh-valid" in label for label in labels)
        assert any("aperture-valid" in label for label in labels)
        # Grid helper lines must not leak into the legend.
        assert not any(label.startswith("_") for label in labels)
        assert any(
            POLAR_POWER_APPROXIMATION_NOTE in text.get_text() for text in fig.texts
        )
    finally:
        plt.close(fig)


def test_save_beamwidth_plot_to_disk(tmp_path):
    freqs = np.geomspace(200.0, 16000.0, 24)
    beamwidths = {
        "horizontal": np.clip(360.0 / (1.0 + freqs / 800.0), 0.0, 360.0),
        "vertical": np.clip(300.0 / (1.0 + freqs / 600.0), 0.0, 360.0),
    }
    out = tmp_path / "beamwidth.png"
    result = hlp.save_beamwidth_plot(
        out,
        freqs,
        beamwidths,
        mesh_valid_hz=8000.0,
    )
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 500


def test_beamwidth_figure_symmetry_footnote_and_ylim():
    freqs = np.geomspace(200.0, 16000.0, 24)
    beamwidths = {"horizontal": np.full(freqs.size, 120.0)}
    fig = _build_beamwidth_figure(
        freqs,
        beamwidths,
        assumed_symmetric_from_one_sided_grid=True,
    )
    try:
        assert fig.axes[0].get_ylim() == (0.0, 360.0)
        assert any(BEAMWIDTH_SYMMETRY_NOTE in text.get_text() for text in fig.texts)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert "horizontal" in labels
    finally:
        plt.close(fig)

    fig = _build_beamwidth_figure(freqs, beamwidths)
    try:
        assert not any(BEAMWIDTH_SYMMETRY_NOTE in text.get_text() for text in fig.texts)
    finally:
        plt.close(fig)


def test_save_group_delay_plot_to_disk(tmp_path):
    freqs = np.geomspace(100.0, 20000.0, 40)
    group_delay_s = 1.0e-3 * (1.0 + 0.2 * np.sin(np.log10(freqs) * 3.0))
    group_delay_s[5] = np.nan  # ambiguous bin: the trace simply breaks
    out = tmp_path / "group_delay.png"
    result = hlp.save_group_delay_plot(
        out,
        freqs,
        group_delay_s,
        mesh_valid_hz=9000.0,
    )
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 500


def test_save_excursion_plot_to_disk_with_xmax(tmp_path):
    freqs = np.geomspace(20.0, 2000.0, 40)
    excursion_m = 2.0e-3 / (1.0 + (freqs / 120.0) ** 2)
    out = tmp_path / "excursion.png"
    result = hlp.save_excursion_plot(
        out,
        freqs,
        excursion_m,
        label="MF",
        xmax_m=1.5e-3,
    )
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 500


def test_excursion_figure_draws_xmax_marker_in_mm():
    freqs = np.geomspace(20.0, 2000.0, 40)
    excursion_m = 2.0e-3 / (1.0 + (freqs / 120.0) ** 2)
    fig = _build_excursion_figure(freqs, excursion_m, label="MF", xmax_m=1.5e-3)
    try:
        ax = fig.axes[0]
        assert ax.get_title() == "MF Cone Excursion"
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "MF" in labels and "Xmax" in labels
        horizontal = [
            line
            for line in ax.get_lines()
            if line.get_label() == "Xmax"
        ]
        assert len(horizontal) == 1
        np.testing.assert_allclose(horizontal[0].get_ydata(), 1.5)  # mm
        assert horizontal[0].get_linestyle() == "--"
    finally:
        plt.close(fig)


def test_derived_charts_honor_theme():
    """Key style values follow the requested theme, including the designed
    themes' rc_params grid style."""
    freqs = np.geomspace(100.0, 20000.0, 30)
    group_delay_s = 1.0e-3 * np.ones(freqs.size)

    granite = hlp.get_theme("granite")
    fig = _build_group_delay_figure(freqs, group_delay_s, theme="granite")
    try:
        ax = fig.axes[0]
        assert (
            mcolors.to_hex(fig.get_facecolor(), keep_alpha=False).lower()
            == granite.figure_bg.lower()
        )
        assert ax.title.get_color() == granite.text_color
        v_grid = [
            line
            for line in ax.get_lines()
            if len(set(np.atleast_1d(line.get_xdata()).tolist())) == 1
        ]
        assert v_grid
        for line in v_grid:
            assert line.get_linestyle() == "--"
            assert line.get_linewidth() == pytest.approx(0.6)
        for gridline in ax.yaxis.get_gridlines():
            assert gridline.get_linestyle() == "--"
            assert gridline.get_linewidth() == pytest.approx(0.6)
    finally:
        plt.close(fig)

    hornlab = hlp.get_theme("hornlab")
    fig = _build_group_delay_figure(freqs, group_delay_s)
    try:
        ax = fig.axes[0]
        assert (
            mcolors.to_hex(fig.get_facecolor(), keep_alpha=False).lower()
            == hornlab.figure_bg.lower()
        )
        v_grid = [
            line
            for line in ax.get_lines()
            if len(set(np.atleast_1d(line.get_xdata()).tolist())) == 1
        ]
        assert v_grid
        assert {round(line.get_linewidth(), 3) for line in v_grid} <= {0.5, 0.7}
        assert all(line.get_linestyle() == "-" for line in v_grid)
    finally:
        plt.close(fig)
