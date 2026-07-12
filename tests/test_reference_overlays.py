"""Regression tests for optional current-vs-reference overlays."""

from __future__ import annotations

import base64
import inspect
import warnings

import numpy as np
import pytest

import hornlab_plots as hlp
from hornlab_plots import style


THEME_NAMES = sorted(style.BUILTIN_THEMES)
FREQUENCIES = [100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0]


def _assert_png(value):
    assert isinstance(value, str)
    assert base64.b64decode(value).startswith(b"\x89PNG\r\n\x1a\n")


def _line_payload():
    return {
        "frequencies": FREQUENCIES,
        "spl": [92.0, 94.0, 96.0, 97.0, 96.0, 94.0, 91.0],
        "di_frequencies": FREQUENCIES,
        "di": {
            "horizontal": [2.0, 2.5, 3.5, 5.0, 7.0, 9.0, 11.0],
            "vertical": [1.5, 2.0, 3.0, 4.5, 6.5, 8.5, 10.5],
        },
        "impedance_frequencies": FREQUENCIES,
        "impedance_real": [390.0, 410.0, 430.0, 405.0, 385.0, 400.0, 420.0],
        "impedance_imaginary": [-75.0, -40.0, 10.0, 60.0, 35.0, -15.0, -45.0],
        "impedance_units": "Pa·s/m",
        "impedance_normalization": "rho_c",
    }


def _line_reference():
    return {
        "label": "Baseline",
        "frequencies": FREQUENCIES,
        "spl": [90.0, 92.0, 94.5, 95.0, 94.0, 91.5, 88.0],
        "di_frequencies": FREQUENCIES,
        "di": {
            "horizontal": [1.5, 2.0, 3.0, 4.0, 5.5, 7.0, 8.5],
            "vertical": [1.0, 1.5, 2.5, 3.5, 5.0, 6.5, 8.0],
        },
        "impedance_frequencies": FREQUENCIES,
        "impedance_real": [370.0, 385.0, 405.0, 395.0, 375.0, 380.0, 400.0],
        "impedance_imaginary": [-55.0, -20.0, 25.0, 45.0, 20.0, -5.0, -30.0],
        "impedance_units": "Pa·s/m",
        "impedance_normalization": "rho_c",
    }


def _plane_patterns(*, width, edge_db=-18.0, n_freq=5):
    angles = np.linspace(-90.0, 90.0, 19)
    patterns = []
    for index in range(n_freq):
        sigma = max(width - 2.0 * index, 16.0)
        values = edge_db * (1.0 - np.exp(-(angles ** 2) / (2.0 * sigma ** 2)))
        patterns.append([[float(angle), float(value)] for angle, value in zip(angles, values)])
    return patterns


def _heatmap_data():
    frequencies = np.geomspace(200.0, 6400.0, 5).tolist()
    primary = {
        "horizontal": _plane_patterns(width=52.0),
        "vertical": _plane_patterns(width=38.0),
    }
    reference = {
        "horizontal": _plane_patterns(width=43.0),
        "vertical": _plane_patterns(width=29.0),
    }
    return frequencies, primary, reference


def test_new_reference_parameters_are_keyword_only_and_default_to_none():
    expected = {
        hlp.frequency_response_b64: (
            "reference_frequencies",
            "reference_spl",
            "reference_label",
        ),
        hlp.directivity_index_b64: (
            "reference_frequencies",
            "reference_di",
            "reference_label",
        ),
        hlp.impedance_b64: (
            "reference_frequencies",
            "reference_real",
            "reference_imaginary",
            "reference_label",
            "normalization",
            "reference_normalization",
        ),
        hlp.directivity_heatmap_from_legacy_dict: (
            "reference_frequencies",
            "reference_directivity",
            "reference_label",
        ),
    }

    for renderer, parameter_names in expected.items():
        parameters = inspect.signature(renderer).parameters
        for name in parameter_names:
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is None


def test_explicit_none_reference_is_byte_identical_to_omitted_reference():
    payload = _line_payload()
    dpi = 42

    assert hlp.frequency_response_b64(
        payload["frequencies"], payload["spl"], dpi=dpi
    ) == hlp.frequency_response_b64(
        payload["frequencies"],
        payload["spl"],
        dpi=dpi,
        reference_frequencies=None,
        reference_spl=None,
        reference_label=None,
    )
    assert hlp.directivity_index_b64(
        payload["di_frequencies"], payload["di"], dpi=dpi
    ) == hlp.directivity_index_b64(
        payload["di_frequencies"],
        payload["di"],
        dpi=dpi,
        reference_frequencies=None,
        reference_di=None,
        reference_label=None,
    )
    assert hlp.impedance_b64(
        payload["impedance_frequencies"],
        payload["impedance_real"],
        payload["impedance_imaginary"],
        dpi=dpi,
    ) == hlp.impedance_b64(
        payload["impedance_frequencies"],
        payload["impedance_real"],
        payload["impedance_imaginary"],
        dpi=dpi,
        reference_frequencies=None,
        reference_real=None,
        reference_imaginary=None,
        reference_label=None,
        normalization=None,
        reference_normalization=None,
    )

    heatmap_frequencies, directivity, _reference = _heatmap_data()
    assert hlp.directivity_heatmap_from_legacy_dict(
        heatmap_frequencies, directivity, dpi=dpi
    ) == hlp.directivity_heatmap_from_legacy_dict(
        heatmap_frequencies,
        directivity,
        dpi=dpi,
        reference_frequencies=None,
        reference_directivity=None,
        reference_label=None,
    )


def test_render_all_without_reference_byte_matches_direct_renderers():
    payload = _line_payload()
    frequencies, directivity, _reference = _heatmap_data()
    payload["frequencies"] = frequencies
    payload["directivity"] = directivity
    dpi = 42

    combined = hlp.render_all_charts_b64(
        {**payload, "reference": None},
        dpi=dpi,
    )
    direct = {
        "frequency_response": hlp.frequency_response_b64(
            payload["frequencies"],
            payload["spl"],
            dpi=dpi,
        ),
        "directivity_index": hlp.directivity_index_b64(
            payload["di_frequencies"],
            payload["di"],
            dpi=dpi,
        ),
        "impedance": hlp.impedance_b64(
            payload["impedance_frequencies"],
            payload["impedance_real"],
            payload["impedance_imaginary"],
            dpi=dpi,
        ),
        "directivity_map": hlp.directivity_heatmap_from_legacy_dict(
            payload["frequencies"],
            payload["directivity"],
            dpi=dpi,
        ),
    }

    assert combined == direct


@pytest.mark.parametrize("reference", [None, {}, {"label": "Unused"}])
def test_render_all_empty_reference_is_byte_identical_to_absent(reference):
    payload = _line_payload()
    without_reference = hlp.render_all_charts_b64(payload, dpi=42)

    with_empty_reference = hlp.render_all_charts_b64(
        {**payload, "reference": reference}, dpi=42
    )

    assert with_empty_reference == without_reference


def test_render_all_reference_frequency_fallbacks(monkeypatch):
    from hornlab_plots import charts as charts_module

    captured = {}

    def capture(name):
        def renderer(*_args, **kwargs):
            captured[name] = kwargs["reference_frequencies"]
            return None

        return renderer

    monkeypatch.setattr(charts_module, "frequency_response_b64", capture("spl"))
    monkeypatch.setattr(charts_module, "directivity_index_b64", capture("di"))
    monkeypatch.setattr(charts_module, "impedance_b64", capture("impedance"))

    reference = _line_reference()
    reference["di_frequencies"] = []
    reference["impedance_frequencies"] = []
    charts_module.render_all_charts_b64(
        {**_line_payload(), "reference": reference},
        dpi=42,
    )

    assert captured == {
        "spl": reference["frequencies"],
        "di": reference["frequencies"],
        "impedance": reference["frequencies"],
    }


@pytest.mark.parametrize("theme_name", THEME_NAMES)
def test_reference_overlays_render_distinct_pngs_without_warnings_in_every_theme(theme_name):
    payload = _line_payload()
    reference = _line_reference()
    without_reference = hlp.render_all_charts_b64(payload, dpi=36, theme=theme_name)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with_reference = hlp.render_all_charts_b64(
            {**payload, "reference": reference}, dpi=36, theme=theme_name
        )

    assert seen == []
    for chart_name in ("frequency_response", "directivity_index", "impedance"):
        _assert_png(with_reference[chart_name])
        assert with_reference[chart_name] != without_reference[chart_name]

    heatmap_frequencies, directivity, heatmap_reference = _heatmap_data()
    without_heatmap_reference = hlp.directivity_heatmap_from_legacy_dict(
        heatmap_frequencies,
        {"horizontal": directivity["horizontal"]},
        dpi=36,
        theme=theme_name,
    )
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with_heatmap_reference = hlp.directivity_heatmap_from_legacy_dict(
            heatmap_frequencies,
            {"horizontal": directivity["horizontal"]},
            dpi=36,
            theme=theme_name,
            reference_frequencies=heatmap_frequencies,
            reference_directivity={"horizontal": heatmap_reference["horizontal"]},
            reference_label="Baseline",
        )

    assert seen == []
    _assert_png(with_heatmap_reference)
    assert with_heatmap_reference != without_heatmap_reference


def test_compare_legends_and_reference_line_style_appear_only_when_drawn(monkeypatch):
    import matplotlib.pyplot as plt
    from hornlab_plots import charts as charts_module

    monkeypatch.setattr(charts_module, "_fig_to_base64", lambda fig, _dpi: fig)
    payload = _line_payload()
    reference = _line_reference()
    figures = []
    try:
        response_without = hlp.frequency_response_b64(
            payload["frequencies"],
            payload["spl"],
            dpi=42,
        )
        figures.append(response_without)
        assert "Current" not in response_without.axes[0].get_legend_handles_labels()[1]

        response_with = hlp.frequency_response_b64(
            payload["frequencies"],
            payload["spl"],
            dpi=42,
            reference_frequencies=reference["frequencies"],
            reference_spl=reference["spl"],
        )
        figures.append(response_with)
        assert response_with.axes[0].get_legend_handles_labels()[1] == [
            "Current",
            "Reference",
        ]

        di_with = hlp.directivity_index_b64(
            payload["di_frequencies"],
            payload["di"],
            dpi=42,
            reference_frequencies=reference["di_frequencies"],
            reference_di=reference["di"],
            reference_label="Baseline",
        )
        figures.append(di_with)
        di_labels = di_with.axes[0].get_legend_handles_labels()[1]
        assert any(label.startswith("Current") for label in di_labels)
        assert any(label.startswith("Baseline") for label in di_labels)

        impedance_with = hlp.impedance_b64(
            payload["impedance_frequencies"],
            payload["impedance_real"],
            payload["impedance_imaginary"],
            dpi=42,
            reference_frequencies=reference["impedance_frequencies"],
            reference_real=reference["impedance_real"],
            reference_imaginary=reference["impedance_imaginary"],
            reference_label="Baseline",
            normalization=payload["impedance_normalization"],
            reference_normalization=reference["impedance_normalization"],
        )
        figures.append(impedance_with)
        impedance_labels = impedance_with.axes[0].get_legend_handles_labels()[1]
        assert any(label.startswith("Current") for label in impedance_labels)
        assert any(label.startswith("Baseline") for label in impedance_labels)

        for figure in (response_with, di_with, impedance_with):
            reference_lines = [
                line for line in figure.axes[0].get_lines()
                if line.get_linestyle() == "--" and line.get_alpha() == pytest.approx(0.6)
            ]
            assert reference_lines
            assert all(line.get_linewidth() == pytest.approx(1.2) for line in reference_lines)
    finally:
        for figure in figures:
            plt.close(figure)


@pytest.mark.parametrize(
    ("primary_di", "reference_di"),
    [
        ([2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 2.5, 3.0]),
        (
            {"horizontal": [2.0, 3.0, 4.0, 5.0], "vertical": [1.5, 2.5, 3.5, 4.5]},
            {"horizontal": [1.0, 2.0, 2.5, 3.0], "vertical": [1.0, 1.5, 2.5, 3.5]},
        ),
    ],
    ids=("flat-list", "per-plane-dict"),
)
def test_directivity_index_reference_handles_both_supported_shapes(primary_di, reference_di):
    frequencies = [200.0, 400.0, 800.0, 1600.0]
    baseline = hlp.directivity_index_b64(frequencies, primary_di, dpi=42)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        compared = hlp.directivity_index_b64(
            frequencies,
            primary_di,
            dpi=42,
            reference_frequencies=frequencies,
            reference_di=reference_di,
            reference_label="Baseline",
        )

    assert seen == []
    _assert_png(compared)
    assert compared != baseline


def test_impedance_normalization_mismatch_warns_and_skips_overlay():
    payload = _line_payload()
    payload = {
        key: value
        for key, value in payload.items()
        if key.startswith("impedance_")
    }
    baseline = hlp.render_all_charts_b64(payload, dpi=42)
    reference = _line_reference()
    reference["impedance_normalization"] = "absolute"

    with pytest.warns(RuntimeWarning, match="normaliz"):
        compared = hlp.render_all_charts_b64(
            {**payload, "reference": reference}, dpi=42
        )

    _assert_png(compared["impedance"])
    assert compared["impedance"] == baseline["impedance"]


def test_render_all_reference_does_not_change_directivity_map():
    payload = _line_payload()
    frequencies, directivity, _heatmap_reference = _heatmap_data()
    payload["frequencies"] = frequencies
    payload["directivity"] = directivity

    baseline = hlp.render_all_charts_b64(payload, dpi=42)
    compared = hlp.render_all_charts_b64(
        {**payload, "reference": _line_reference()}, dpi=42
    )

    _assert_png(compared["directivity_map"])
    assert compared["directivity_map"] == baseline["directivity_map"]


def test_heatmap_reference_uses_only_plane_keys_present_in_primary_and_reference():
    frequencies, primary, _reference = _heatmap_data()
    reference_vertical = _plane_patterns(width=27.0)
    unmatched_diagonal = _plane_patterns(width=20.0)
    baseline = hlp.directivity_heatmap_from_legacy_dict(
        frequencies, primary, dpi=42
    )

    unmatched_only = hlp.directivity_heatmap_from_legacy_dict(
        frequencies,
        primary,
        dpi=42,
        reference_frequencies=frequencies,
        reference_directivity={"diagonal": unmatched_diagonal},
        reference_label="Baseline",
    )
    common_only = hlp.directivity_heatmap_from_legacy_dict(
        frequencies,
        primary,
        dpi=42,
        reference_frequencies=frequencies,
        reference_directivity={"vertical": reference_vertical},
        reference_label="Baseline",
    )
    common_plus_unmatched = hlp.directivity_heatmap_from_legacy_dict(
        frequencies,
        primary,
        dpi=42,
        reference_frequencies=frequencies,
        reference_directivity={
            "vertical": reference_vertical,
            "diagonal": unmatched_diagonal,
        },
        reference_label="Baseline",
    )

    for rendered in (baseline, unmatched_only, common_only, common_plus_unmatched):
        _assert_png(rendered)
    assert unmatched_only == baseline
    assert common_only != baseline
    assert common_plus_unmatched == common_only
