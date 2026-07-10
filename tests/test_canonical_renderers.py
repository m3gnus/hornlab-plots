"""Canonical plot renderer smoke tests."""

from __future__ import annotations

import base64
import numpy as np
import pytest

import hornlab_plots as hlp


def _decode_png(b64: str) -> bytes:
    payload = b64.split(",", 1)[-1]
    data = base64.b64decode(payload)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    return data


def test_canonical_renderer_exports_are_available():
    assert callable(hlp.directivity_heatmap_from_legacy_dict)
    assert callable(hlp.frequency_response_b64)
    assert callable(hlp.save_directivity_plot)


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


def test_legacy_directivity_points_are_reindexed_by_angle_value():
    from hornlab_plots._heatmap import build_grid_from_legacy

    angles, freqs, values = build_grid_from_legacy(
        [100.0, 200.0],
        [
            [[-10.0, -1.0], [0.0, 0.0], [10.0, -1.0]],
            [[10.0, -2.0], [0.0, 0.0], [-10.0, -20.0]],
        ],
    )

    np.testing.assert_allclose(angles, [-10.0, 0.0, 10.0])
    np.testing.assert_allclose(freqs, [100.0, 200.0])
    np.testing.assert_allclose(values[:, 1], [-20.0, 0.0, -2.0])


def test_legacy_directivity_missing_middle_angle_is_interpolated():
    from hornlab_plots._heatmap import build_grid_from_legacy

    angles, _freqs, values = build_grid_from_legacy(
        [100.0, 200.0],
        [
            [[-20.0, -2.0], [-5.0, -1.0], [10.0, 0.0]],
            [[-20.0, -30.0], [10.0, 0.0]],
        ],
    )

    np.testing.assert_allclose(angles, [-20.0, -5.0, 10.0])
    np.testing.assert_allclose(values[:, 1], [-30.0, -15.0, 0.0])


def test_legacy_directivity_missing_end_angle_uses_nearest_endpoint():
    from hornlab_plots._heatmap import build_grid_from_legacy

    angles, _freqs, values = build_grid_from_legacy(
        [100.0, 200.0],
        [
            [[-20.0, -2.0], [-5.0, -1.0], [10.0, 0.0]],
            [[-20.0, -30.0], [-5.0, -15.0]],
        ],
    )

    np.testing.assert_allclose(angles, [-20.0, -5.0, 10.0])
    np.testing.assert_allclose(values[:, 1], [-30.0, -15.0, -15.0])


def test_legacy_directivity_duplicate_angles_are_rejected():
    from hornlab_plots._heatmap import build_grid_from_legacy

    with pytest.raises(ValueError, match="duplicate angle 0"):
        build_grid_from_legacy(
            [100.0],
            [[[0.0, 0.0], [0.0, -1.0]]],
        )


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


def test_frequency_response_draws_both_mesh_valid_limits():
    import matplotlib.pyplot as plt
    from hornlab_plots.charts import _build_frequency_response_figure

    freqs = np.geomspace(1000.0, 20000.0, 30)
    spl = -60.0 - 5.0 * np.log10(freqs / 1000.0)
    fig = _build_frequency_response_figure(
        [hlp.FrequencyResponseCurve(freqs, spl, "HF", role="hf")],
        mesh_valid_hz=3642.0,
        mesh_valid_radiating_hz=11352.0,
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("mesh-valid" in label for label in labels)
    assert any("aperture-valid" in label for label in labels)
    vertical_xs = [
        round(float(line.get_xdata()[0]))
        for line in ax.get_lines()
        if len(set(np.atleast_1d(line.get_xdata()).tolist())) == 1
    ]
    assert any(abs(x - 3642) < 2 for x in vertical_xs)
    assert any(abs(x - 11352) < 2 for x in vertical_xs)
    plt.close(fig)


def test_directivity_heatmap_draws_both_mesh_valid_limits():
    import matplotlib.pyplot as plt
    from hornlab_plots._heatmap import _build_figure_from_planes, _build_planes_from_legacy

    freqs = np.geomspace(1000.0, 20000.0, 12).tolist()
    angles = np.linspace(0.0, 90.0, 13)
    pattern = [[[float(a), -float(abs(a)) * 0.1] for a in angles] for _ in freqs]
    planes = _build_planes_from_legacy(freqs, {"horizontal": pattern, "vertical": pattern})
    fig = _build_figure_from_planes(planes, mesh_valid_hz=3642.0, mesh_valid_radiating_hz=11352.0)
    labels = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            labels.extend(t.get_text() for t in legend.get_texts())
    assert any("mesh-valid" in label for label in labels)
    assert any("aperture-valid" in label for label in labels)
    plt.close(fig)
