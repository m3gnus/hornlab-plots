"""WG-legacy byte parity tests.

`hornlab_plots` was extracted from `Waveguide-Generator/server/solver/`
unchanged. These tests prove the extraction preserved byte-for-byte
output. If they ever fail, either:

1. WG legacy was patched without porting the fix here, or
2. `hornlab_plots` was patched without preserving the byte contract.

In both cases, the WG bridge soak gate (see canonical-plots-migration.md)
catches the divergence at the visual-review step. These tests catch it
faster.

Skipped automatically when the WG server tree isn't importable (e.g.
running this package in isolation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import hornlab_plots as hlp


# Discover the WG server dir from the repo layout (sibling at ../Waveguide-Generator/server).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WG_SERVER = _REPO_ROOT / "Waveguide-Generator" / "server"


def _ensure_wg_on_path():
    if not _WG_SERVER.is_dir():
        pytest.skip(f"Waveguide-Generator server tree not found at {_WG_SERVER}")
    if str(_WG_SERVER) not in sys.path:
        sys.path.insert(0, str(_WG_SERVER))


def _synthetic_legacy_dict(n_freq=12, n_angle=37, on_axis_db=0.0, edge_db=-15.0):
    freqs = np.geomspace(200.0, 16000.0, n_freq).tolist()
    angles = np.linspace(-90.0, 90.0, n_angle)

    def per_freq_pattern(fi):
        sigma_deg = max(60.0 - fi * 2.0, 20.0)
        db = on_axis_db + (edge_db - on_axis_db) * (1 - np.exp(-(angles ** 2) / (2 * sigma_deg ** 2)))
        return [[float(a), float(d)] for a, d in zip(angles, db)]

    directivity = {
        "horizontal": [per_freq_pattern(fi) for fi in range(n_freq)],
        "vertical":   [per_freq_pattern(fi) for fi in range(n_freq)],
    }
    return freqs, directivity


def test_directivity_heatmap_matches_wg_legacy_bytes():
    _ensure_wg_on_path()
    try:
        from solver.directivity_plot import render_directivity_plot as wg_render
    except ImportError as exc:
        pytest.skip(f"WG legacy import unavailable: {exc}")

    freqs, directivity = _synthetic_legacy_dict()
    wg_b64 = wg_render(freqs, directivity)
    canon_b64 = hlp.directivity_heatmap_from_legacy_dict(freqs, directivity)

    assert wg_b64 is not None and canon_b64 is not None
    assert wg_b64 == canon_b64, (
        "hornlab-plots heatmap diverged from WG legacy output. "
        "Either the extraction lost a styling tweak, or WG was patched "
        "without porting the fix. See docs/plans/canonical-plots-migration.md."
    )


def test_frequency_response_matches_wg_legacy_bytes():
    _ensure_wg_on_path()
    try:
        from solver.charts import render_frequency_response as wg_render
    except ImportError as exc:
        pytest.skip(f"WG legacy import unavailable: {exc}")

    freqs = np.geomspace(100.0, 20000.0, 30).tolist()
    spl = (100.0 + 3.0 * np.sin(np.log10(freqs))).tolist()

    wg_b64 = wg_render(freqs, spl)
    canon_b64 = hlp.frequency_response_b64(freqs, spl)

    assert wg_b64 == canon_b64


def test_directivity_index_matches_wg_legacy_bytes():
    _ensure_wg_on_path()
    try:
        from solver.charts import render_directivity_index as wg_render
    except ImportError as exc:
        pytest.skip(f"WG legacy import unavailable: {exc}")

    freqs = np.geomspace(100.0, 20000.0, 30).tolist()
    di_h = (6.0 + 3.0 * np.log10(np.asarray(freqs) / 1000.0)).tolist()
    di_v = (5.0 + 3.0 * np.log10(np.asarray(freqs) / 1000.0)).tolist()
    di_per_plane = {"horizontal": di_h, "vertical": di_v}

    wg_b64 = wg_render(freqs, di_per_plane)
    canon_b64 = hlp.directivity_index_b64(freqs, di_per_plane)

    assert wg_b64 == canon_b64


def test_impedance_matches_wg_legacy_bytes():
    _ensure_wg_on_path()
    try:
        from solver.charts import render_impedance as wg_render
    except ImportError as exc:
        pytest.skip(f"WG legacy import unavailable: {exc}")

    freqs = np.geomspace(100.0, 20000.0, 30).tolist()
    real = (400.0 + 50.0 * np.sin(np.log10(np.asarray(freqs)) * 2)).tolist()
    imag = (100.0 * np.cos(np.log10(np.asarray(freqs)) * 2)).tolist()

    wg_b64 = wg_render(freqs, real, imag)
    canon_b64 = hlp.impedance_b64(freqs, real, imag)

    assert wg_b64 == canon_b64
