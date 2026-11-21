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


def _polar_spec(**radius):
    angles = list(range(-180, 181, 30))
    return {
        "kind": "polar",
        "r": {"min": -30, "max": 0, **radius},
        "series": [{"label": "Run", "angle": angles, "r": [2.0 if a == 30 else -abs(a) / 10 for a in angles]}],
    }


def test_polar_values_above_the_ring_are_drawn_as_given():
    fig = build_chart_spec_figure(_polar_spec(), theme="classic")
    line = next(line for line in fig.axes[0].get_lines() if line.get_label() == "Run")
    # Not clamped onto the 0 dB ring: the axes' clip path cuts the trace.
    assert 2.0 in list(line.get_ydata())
    assert line.get_clip_on()


def test_a_missing_polar_angle_is_a_gap():
    spec = _polar_spec()
    spec["series"][0]["angle"][3] = None
    fig = build_chart_spec_figure(spec, theme="classic")
    line = next(line for line in fig.axes[0].get_lines() if line.get_label() == "Run")
    assert np.isnan(line.get_xdata()[3])


def test_marked_points_get_hollow_markers():
    spec = _lines_spec()
    spec["series"][0]["markers"] = [2, 5, 99]
    fig = build_chart_spec_figure(spec, theme="classic")
    markers = [line for line in fig.axes[0].get_lines() if line.get_marker() == "o"]
    assert len(markers) == 1
    np.testing.assert_allclose(markers[0].get_ydata(), [0.2, 0.5])


@pytest.mark.parametrize("spec", [
    _polar_spec(step=0.001),
    _polar_spec(min=-2000, step=1),
    {"x": {"scale": "log", "min": 1e-300, "max": 1e300}, "series": [{"x": [100, 200], "y": [1, 2]}]},
    {"series": [{"x": [1e-300, 1e300], "y": [1, 2]}]},
    {"series": ["x"]},
    {"x": 5, "series": [{"x": [100, 200], "y": [1, 2]}]},
    "not a mapping",
])
def test_costly_or_malformed_specs_are_refused_quickly_without_leaking_figures(spec):
    import time

    import matplotlib.pyplot as plt

    open_before = set(plt.get_fignums())
    started = time.perf_counter()
    with pytest.raises(ValueError):
        hlp.chart_spec_b64(spec)
    assert time.perf_counter() - started < 2.0
    assert set(plt.get_fignums()) == open_before


def test_an_empty_chart_leaves_no_figure_open():
    import matplotlib.pyplot as plt

    open_before = set(plt.get_fignums())
    assert hlp.chart_spec_b64({"series": [{"x": [100, 200], "y": [None, None]}]}) is None
    assert set(plt.get_fignums()) == open_before


@pytest.mark.parametrize("spec", [
    {"hlines": [{"y": 0}] * 17, "series": [{"x": [100, 200], "y": [1, 2]}]},
    {"bands": [{"x0": 100, "x1": 200}] * 65, "series": [{"x": [100, 200], "y": [1, 2]}]},
    {"series": [{"x": [100, 200], "y": [1, 2]}] * 65},
    {"series": [{"x": [100.0] * 20_001, "y": [1.0] * 20_001}]},
    {"series": [{"x": [100.0] * 20_000, "y": [1.0] * 20_000}] * 11},
    {"kind": "polar", "series": [], "shade": [{"from": 0, "to": 10}] * 9},
    {"series": [{"x": "abc", "y": [1]}]},
])
def test_the_renderer_bounds_counts_itself(spec):
    import time

    import matplotlib.pyplot as plt

    open_before = set(plt.get_fignums())
    started = time.perf_counter()
    with pytest.raises(ValueError):
        hlp.chart_spec_b64(spec)
    assert time.perf_counter() - started < 1.0
    assert set(plt.get_fignums()) == open_before


def test_the_twin_axis_frame_takes_the_theme_colour():
    from matplotlib.colors import to_hex

    spec = _lines_spec(y2={"label": "Phase"})
    spec["series"][1]["axis"] = 1
    for theme in ("classic", "console"):
        from hornlab_plots.style import resolve_theme

        fig = build_chart_spec_figure(spec, theme=theme)
        second = fig.axes[1]
        expected = to_hex(resolve_theme(theme).spine_color)
        assert {to_hex(spine.get_edgecolor()) for spine in second.spines.values()} == {expected}
