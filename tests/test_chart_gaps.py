"""Unavailable line-chart samples remain gaps in exported figures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hornlab_plots import charts


@pytest.mark.parametrize("missing", [None, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("chart", ["spl", "di", "impedance"])
def test_primary_and_reference_curves_keep_missing_samples(monkeypatch, chart, missing):
    frequencies = [100.0, 200.0, 400.0, 0.0, np.nan, -1.0, np.inf]
    values = [1.0, missing, 3.0, 8.0, 8.0, 8.0, 8.0]
    before = np.asarray(values, dtype=float).copy()
    if chart == "spl":
        fig = charts._build_frequency_response_figure(
            [(frequencies, values, "Current")],
            reference_frequencies=frequencies,
            reference_spl=values,
            reference_label="Baseline",
        )
        labels = ["Current", "Baseline"]
    elif chart == "di":
        monkeypatch.setattr(charts, "_fig_to_base64", lambda fig, dpi: fig)
        fig = charts.directivity_index_b64(
            frequencies, values,
            reference_frequencies=frequencies,
            reference_di=values,
            reference_label="Baseline",
        )
        labels = ["Current", "Baseline"]
    else:
        fig = charts._build_impedance_figure(
            frequencies, values, values,
            reference_frequencies=frequencies,
            reference_real=values,
            reference_imaginary=values,
            reference_label="Baseline",
        )
        labels = ["Current Re(Z)", "Current Im(Z)", "Baseline Re(Z)", "Baseline Im(Z)"]
    try:
        lines = {line.get_label(): line for line in fig.axes[0].lines}
        for label in labels:
            np.testing.assert_array_equal(lines[label].get_xdata(), [100.0, 200.0, 400.0])
            np.testing.assert_array_equal(lines[label].get_ydata(), [1.0, np.nan, 3.0])
        np.testing.assert_array_equal(np.asarray(values, dtype=float), before)
    finally:
        plt.close(fig)


def test_impedance_components_keep_their_independent_valid_samples():
    fig = charts._build_impedance_figure(
        [100.0, 200.0, 400.0], [1.0, None, 3.0], [None, 2.0, 4.0],
        reference_frequencies=[100.0, 200.0, 400.0],
        reference_real=[None, 5.0, 6.0],
        reference_imaginary=[7.0, None, 8.0],
    )
    try:
        lines = {line.get_label(): line for line in fig.axes[0].lines}
        for label, expected in {
            "Current Re(Z)": [1.0, np.nan, 3.0],
            "Current Im(Z)": [np.nan, 2.0, 4.0],
            "Reference Re(Z)": [np.nan, 5.0, 6.0],
            "Reference Im(Z)": [7.0, np.nan, 8.0],
        }.items():
            np.testing.assert_array_equal(lines[label].get_ydata(), expected)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("chart", ["spl", "di", "impedance"])
def test_all_missing_curves_have_no_figure(chart):
    frequencies, values = [100.0, 200.0], [None, np.inf]
    if chart == "spl":
        result = charts._build_frequency_response_figure([(frequencies, values, "SPL")])
    elif chart == "di":
        result = charts.directivity_index_b64(frequencies, values)
    else:
        result = charts._build_impedance_figure(frequencies, values, values)
    assert result is None


@pytest.mark.parametrize("chart", ["spl", "di", "impedance"])
def test_missing_endpoints_do_not_expand_frequency_limits(monkeypatch, chart):
    frequencies = [1.0, 100.0, 200.0, 1e9]
    values = [None, 1.0, 2.0, None]
    if chart == "spl":
        fig = charts._build_frequency_response_figure(
            [(frequencies, values, "SPL")],
            reference_frequencies=frequencies, reference_spl=values,
        )
    elif chart == "di":
        monkeypatch.setattr(charts, "_fig_to_base64", lambda fig, dpi: fig)
        fig = charts.directivity_index_b64(
            frequencies, {"horizontal": values},
            reference_frequencies=frequencies, reference_di={"horizontal": values},
        )
    else:
        fig = charts._build_impedance_figure(
            frequencies, values, values,
            reference_frequencies=frequencies, reference_real=values,
            reference_imaginary=values,
        )
    try:
        assert fig.axes[0].get_xlim() == (100.0, 200.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("imaginary", [None, [], [1.0]])
def test_impedance_keeps_real_only_input_compatibility(imaginary):
    fig = charts._build_impedance_figure([100.0, 200.0], [1.0, 2.0], imaginary)
    try:
        labels = [line.get_label() for line in fig.axes[0].lines]
        assert "Re(Z)" in labels
        assert "Im(Z)" not in labels
    finally:
        plt.close(fig)
