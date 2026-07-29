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


def test_heatmap_missing_values_use_physical_coordinates():
    from hornlab_plots._heatmap import _fill_missing_values

    angles = np.array([0.0, 10.0, 90.0])
    freqs = np.array([100.0, 200.0, 800.0])
    values = np.array([
        [0.0, np.nan, -30.0],
        [np.nan, np.nan, -20.0],
        [90.0, np.nan, 60.0],
    ])

    filled = _fill_missing_values(values, angles, freqs)

    np.testing.assert_allclose(
        filled,
        [
            [0.0, -10.0, -30.0],
            [10.0, 0.0, -20.0],
            [90.0, 80.0, 60.0],
        ],
        atol=1e-12,
    )


def test_prepare_heatmap_data_sorts_coordinates_with_their_values():
    from hornlab_plots._heatmap import prepare_heatmap_data

    angles = np.array([-70.0, -5.0, 40.0])
    freqs = np.array([250.0, 900.0, 7000.0])
    values = -0.1 * np.abs(angles[:, np.newaxis])
    values = values - 2.0 * np.log10(freqs[np.newaxis, :] / freqs[0])

    expected = prepare_heatmap_data(angles, freqs, values)
    angle_order = np.array([2, 0, 1])
    frequency_order = np.array([1, 2, 0])
    actual = prepare_heatmap_data(
        angles[angle_order],
        freqs[frequency_order],
        values[np.ix_(angle_order, frequency_order)],
    )

    for expected_array, actual_array in zip(expected, actual):
        np.testing.assert_allclose(actual_array, expected_array)


def _smoothing_probe_data():
    angles = np.array([-60.0, 0.0, 60.0])
    freqs = np.geomspace(500.0, 20_000.0, 56)
    values = (
        -10.0
        + 3.0 * np.sin(np.linspace(0.0, 9.0, freqs.size))[None, :]
        - np.abs(angles[:, None]) / 20.0
    )
    return angles, freqs, values


def test_heatmap_default_matches_legacy_output_on_realistic_grid():
    from hornlab_plots._heatmap import (
        _fill_missing_values,
        _fractional_octave_smooth,
        _interpolate_heatmap_grid,
        prepare_heatmap_data,
    )
    from hornlab_plots.style import (
        ANGLE_SAMPLES,
        FRACTIONAL_OCTAVE,
        FREQ_SAMPLES,
        MAX_DB,
        MIN_DB,
    )

    angles, freqs, values = _smoothing_probe_data()
    filled = _fill_missing_values(values, angles, freqs)
    legacy_smoothed = _fractional_octave_smooth(
        filled,
        freqs,
        FRACTIONAL_OCTAVE,
    )
    assert np.array_equal(legacy_smoothed, filled)
    legacy_angles, legacy_freqs, legacy_values = _interpolate_heatmap_grid(
        angles,
        freqs,
        legacy_smoothed,
        ANGLE_SAMPLES,
        FREQ_SAMPLES,
    )
    legacy = (
        legacy_angles,
        legacy_freqs,
        np.clip(legacy_values, MIN_DB, MAX_DB),
    )

    actual = prepare_heatmap_data(angles, freqs, values)

    for actual_array, legacy_array in zip(actual, legacy):
        np.testing.assert_array_equal(actual_array, legacy_array)


def test_heatmap_smoothing_option_changes_interpolated_data():
    from hornlab_plots._heatmap import prepare_heatmap_data

    angles, freqs, values = _smoothing_probe_data()
    default_values = prepare_heatmap_data(angles, freqs, values)[2]
    smoothed_values = prepare_heatmap_data(
        angles,
        freqs,
        values,
        smooth=True,
    )[2]

    assert not np.array_equal(smoothed_values, default_values)
    assert np.max(np.abs(smoothed_values - default_values)) > 0.0


def test_heatmap_smoothing_fraction_is_honoured():
    from hornlab_plots._heatmap import (
        _fractional_octave_smooth,
        prepare_heatmap_data,
    )
    from hornlab_plots.style import FRACTIONAL_OCTAVE, MAX_DB, MIN_DB

    angles, freqs, values = _smoothing_probe_data()
    interp_angles, interp_freqs, interp_values = prepare_heatmap_data(
        angles,
        freqs,
        values,
    )
    default_fraction = prepare_heatmap_data(
        angles,
        freqs,
        values,
        smooth=True,
    )
    explicit_default_fraction = prepare_heatmap_data(
        angles,
        freqs,
        values,
        smooth=True,
        smoothing_fraction=FRACTIONAL_OCTAVE,
    )
    twelfth_octave = prepare_heatmap_data(
        angles,
        freqs,
        values,
        smooth=True,
        smoothing_fraction=12.0,
    )
    expected_twelfth_octave = np.clip(
        _fractional_octave_smooth(interp_values, interp_freqs, 12.0),
        MIN_DB,
        MAX_DB,
    )

    np.testing.assert_array_equal(twelfth_octave[0], interp_angles)
    np.testing.assert_array_equal(twelfth_octave[1], interp_freqs)
    np.testing.assert_array_equal(twelfth_octave[2], expected_twelfth_octave)
    for implicit, explicit in zip(
        default_fraction,
        explicit_default_fraction,
    ):
        np.testing.assert_array_equal(implicit, explicit)
    assert not np.array_equal(twelfth_octave[2], default_fraction[2])


def test_directivity_planes_with_different_frequency_grids_do_not_collapse():
    import matplotlib.pyplot as plt
    from hornlab_plots._heatmap import _build_figure_from_planes, _build_planes_from_legacy

    pattern = [[-10.0, -1.0], [0.0, 0.0], [10.0, -1.0]]
    planes = _build_planes_from_legacy(
        [100.0, 200.0, 400.0],
        {
            "horizontal": [pattern, pattern, []],
            "vertical": [[], pattern, pattern],
        },
    )

    fig = _build_figure_from_planes(planes)
    try:
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert titles == ["H Normalized Directivity", "V Normalized Directivity"]
    finally:
        plt.close(fig)


def test_directivity_planes_with_different_frequency_interiors_do_not_collapse():
    import matplotlib.pyplot as plt
    from hornlab_plots._heatmap import _build_figure_from_planes, _build_planes_from_legacy

    pattern = [[-10.0, -1.0], [0.0, 0.0], [10.0, -1.0]]
    planes = _build_planes_from_legacy(
        [100.0, 150.0, 300.0, 400.0],
        {
            "horizontal": [pattern, pattern, [], pattern],
            "vertical": [pattern, [], pattern, pattern],
        },
    )

    np.testing.assert_array_equal(planes[0]["freqs_raw"], [100.0, 150.0, 400.0])
    np.testing.assert_array_equal(planes[1]["freqs_raw"], [100.0, 300.0, 400.0])
    np.testing.assert_array_equal(planes[0]["freqs"], planes[1]["freqs"])

    fig = _build_figure_from_planes(planes)
    try:
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert titles == ["H Normalized Directivity", "V Normalized Directivity"]
    finally:
        plt.close(fig)


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
