"""Canonical plot renderer smoke tests after WG legacy deletion."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

import hornlab_plots as hlp


_REPO_ROOT = Path(__file__).resolve().parents[2]
_WG_SOLVER = _REPO_ROOT / "Waveguide-Generator" / "server" / "solver"


def _decode_png(b64: str) -> bytes:
    payload = b64.split(",", 1)[-1]
    data = base64.b64decode(payload)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    return data


def test_wg_legacy_plot_modules_are_deleted():
    if _WG_SOLVER.exists():
        assert not (_WG_SOLVER / "directivity_plot.py").exists()
        assert not (_WG_SOLVER / "charts.py").exists()


def test_directivity_heatmap_from_legacy_dict_returns_png():
    freqs = np.geomspace(200.0, 16000.0, 8).tolist()
    angles = np.linspace(-90.0, 90.0, 25)

    def pattern(freq_index: int) -> list[list[float]]:
        width = max(55.0 - 3.0 * freq_index, 18.0)
        db = -12.0 * (1.0 - np.exp(-(angles ** 2) / (2.0 * width ** 2)))
        return [[float(a), float(v)] for a, v in zip(angles, db)]

    directivity = {
        "horizontal": [pattern(i) for i in range(len(freqs))],
        "vertical": [pattern(i) for i in range(len(freqs))],
    }

    _decode_png(hlp.directivity_heatmap_from_legacy_dict(freqs, directivity))


def test_line_chart_renderers_return_png():
    freqs = np.geomspace(100.0, 20000.0, 16).tolist()
    spl = (100.0 + 3.0 * np.sin(np.log10(freqs))).tolist()
    di = {
        "horizontal": (6.0 + np.log10(freqs)).tolist(),
        "vertical": (5.0 + np.log10(freqs)).tolist(),
    }
    real = (400.0 + 50.0 * np.sin(np.log10(freqs) * 2.0)).tolist()
    imag = (100.0 * np.cos(np.log10(freqs) * 2.0)).tolist()

    _decode_png(hlp.frequency_response_b64(freqs, spl))
    _decode_png(hlp.directivity_index_b64(freqs, di))
    _decode_png(hlp.impedance_b64(freqs, real, imag))


def test_frequency_response_draws_mesh_valid_marker():
    import matplotlib.pyplot as plt
    from hornlab_plots.charts import _build_frequency_response_figure

    freqs = np.geomspace(1000.0, 20000.0, 30)
    spl = -60.0 - 5.0 * np.log10(freqs / 1000.0)
    fig = _build_frequency_response_figure(
        [hlp.FrequencyResponseCurve(freqs, spl, "HF", role="hf")],
        mesh_valid_hz=11352.0,
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("mesh-valid" in label for label in labels)
    vertical_xs = [
        float(line.get_xdata()[0])
        for line in ax.get_lines()
        if len(set(np.atleast_1d(line.get_xdata()).tolist())) == 1
    ]
    assert any(abs(x - 11352.0) < 1.0 for x in vertical_xs)
    plt.close(fig)


def test_directivity_heatmap_draws_mesh_valid_marker():
    import matplotlib.pyplot as plt
    from hornlab_plots._heatmap import _build_figure_from_planes, _build_planes_from_legacy

    freqs = np.geomspace(1000.0, 20000.0, 12).tolist()
    angles = np.linspace(0.0, 90.0, 13)
    pattern = [[[float(a), -float(abs(a)) * 0.1] for a in angles] for _ in freqs]
    planes = _build_planes_from_legacy(freqs, {"horizontal": pattern, "vertical": pattern})
    fig = _build_figure_from_planes(planes, mesh_valid_hz=11352.0)
    has_marker = any(
        ax.get_legend() is not None
        and any("mesh-valid" in t.get_text() for t in ax.get_legend().get_texts())
        for ax in fig.axes
    )
    assert has_marker
    plt.close(fig)
