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
)


def test_derived_renderer_exports_are_available():
    assert callable(hlp.save_directivity_power_plot)
    assert callable(hlp.save_beamwidth_plot)


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


def test_derived_charts_honor_theme():
    """Key style values follow the requested theme, including the designed
    themes' rc_params grid style."""
    freqs = np.geomspace(100.0, 20000.0, 30)
    beamwidths = {"horizontal": np.full(freqs.size, 120.0)}

    granite = hlp.get_theme("granite")
    fig = _build_beamwidth_figure(freqs, beamwidths, theme="granite")
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
    fig = _build_beamwidth_figure(freqs, beamwidths)
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
