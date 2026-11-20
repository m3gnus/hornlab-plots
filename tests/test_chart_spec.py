"""Charts described as data render in the active theme."""

from __future__ import annotations

import base64

import numpy as np
import pytest

import hornlab_plots as hlp
from hornlab_plots.spec import build_chart_spec_figure
from hornlab_plots.style import CLASSIC_THEME


def _png(b64: str) -> bytes:
    data = base64.b64decode(b64)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    return data


def _lines_spec(**overrides):
    freqs = np.geomspace(100.0, 20000.0, 40).tolist()
    spec = {
        "kind": "lines",
        "title": "Group delay",
        "subtitle": "on axis",
        "x": {"label": "Frequency [Hz]", "scale": "log"},
        "y": {"label": "Group delay [ms]"},
        "series": [
            {"label": "Run 1", "x": freqs, "y": [0.1 * i for i in range(40)], "color": 0},
            {"label": "Run 2", "x": freqs, "y": [None] + [0.05 * i for i in range(39)],
             "color": 1, "style": "dashed", "width": 1.2},
        ],
        "bands": [{"x0": 4000.0, "x1": 6000.0, "label": "Unqualified"}],
        "hlines": [{"y": 0}],
    }
    spec.update(overrides)
    return spec


def test_lines_spec_renders_png():
    _png(hlp.chart_spec_b64(_lines_spec(), theme="classic"))


def test_lines_spec_keeps_values_and_theme_colors():
    fig = build_chart_spec_figure(_lines_spec(), theme="classic")
    ax = fig.axes[0]
    lines = [line for line in ax.get_lines() if line.get_label() in {"Run 1", "Run 2"}]
    assert [line.get_label() for line in lines] == ["Run 1", "Run 2"]
    assert lines[0].get_color() == CLASSIC_THEME.line_colors[0]
    assert lines[1].get_color() == CLASSIC_THEME.line_colors[1]
    assert lines[1].get_linestyle() == "--"
    # Values are drawn as given: no smoothing, gaps stay gaps.
    np.testing.assert_allclose(lines[0].get_ydata(), [0.1 * i for i in range(40)])
    assert np.isnan(lines[1].get_ydata()[0])
    assert ax.get_xscale() == "log"
    assert ax.get_ylabel() == "Group delay [ms]"


def test_lines_spec_honours_axis_ranges_and_second_axis():
    spec = _lines_spec(
        x={"label": "Frequency [Hz]", "scale": "log", "min": 200.0, "max": 10000.0},
        y={"label": "Power [W]", "min": -1.0, "max": 5.0},
        y2={"label": "Current [A]", "min": 0.0, "max": 2.0},
    )
    spec["series"][1]["axis"] = 1
    fig = build_chart_spec_figure(spec, theme="classic")
    main, second = fig.axes
    assert main.get_xlim() == pytest.approx((200.0, 10000.0))
    assert main.get_ylim() == pytest.approx((-1.0, 5.0))
    assert second.get_ylim() == pytest.approx((0.0, 2.0))
    assert second.get_ylabel() == "Current [A]"


def test_polar_spec_renders_png():
    angles = list(range(-180, 181, 5))
    spec = {
        "kind": "polar",
        "title": "Polar response",
        "subtitle": "horizontal · 1 kHz",
        "r": {"min": -30, "max": 0, "step": 10, "unit": "dB"},
        "series": [{"label": "Run 1", "angle": angles, "r": [-abs(a) / 10 for a in angles], "color": 0}],
        "shade": [{"from": 90, "to": 180, "label": "Outside half-space"}],
    }
    _png(hlp.chart_spec_b64(spec, theme="classic"))


def test_empty_spec_draws_nothing():
    assert hlp.chart_spec_b64({"series": []}) is None
    assert hlp.chart_spec_b64({"series": [{"x": [100, 200], "y": [None, None]}]}) is None


@pytest.mark.parametrize("bad", [
    {"kind": "pie"},
    {"series": [{"x": [100, 200], "y": [1, 2], "color": "#ff0000"}]},
    {"series": [{"x": [100, 200], "y": [1, 2], "style": "wavy"}]},
])
def test_invalid_spec_is_refused(bad):
    with pytest.raises(ValueError):
        hlp.chart_spec_b64(bad)
