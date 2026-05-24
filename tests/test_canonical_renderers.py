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
