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


def _sphere_spec(kind, **overrides):
    theta = list(range(0, 181, 10))
    phi = list(range(0, 360, 15))
    levels = [[-t / 6.0 - (p % 90) / 30.0 for p in phi] for t in theta]
    levels[3][4] = None
    spec = {"kind": kind, "title": kind, "subtitle": "1 kHz", "theta_deg": theta, "phi_deg": phi, "levels_db": levels}
    spec.update(overrides)
    return spec


@pytest.mark.parametrize("kind", ["beam_map", "balloon"])
def test_spherical_views_render_in_the_theme(kind):
    _png(hlp.chart_spec_b64(_sphere_spec(kind), theme="classic"))
    fig = build_chart_spec_figure(_sphere_spec(kind), theme="classic")
    # A colour bar on the theme's heatmap ramp, spanning -30..0 dB.
    bar = fig.axes[-1]
    assert bar.get_ylim() == pytest.approx((-30.0, 0.0))


def test_beam_map_draws_the_forward_hemisphere_with_phi_counter_clockwise():
    fig = build_chart_spec_figure(_sphere_spec("beam_map"), theme="classic")
    ax = fig.axes[0]
    assert ax.get_rmax() == pytest.approx(90.0)
    assert ax.get_theta_direction() == 1
    mesh = ax.collections[0]
    values = np.asarray(mesh.get_array()).reshape(19, 25)
    # Missing samples read as the floor; the seam repeats the first azimuth.
    assert values[3, 4] == pytest.approx(-30.0)
    np.testing.assert_allclose(values[:, -1], values[:, 0])


@pytest.mark.parametrize("bad", [
    _sphere_spec("balloon", theta_deg=[0]),
    _sphere_spec("balloon", phi_deg=list(range(0, 800))),
    _sphere_spec("beam_map", levels_db=[[0.0]]),
    _sphere_spec("beam_map", levels_db="x"),
    _sphere_spec("beam_map", range_db=0),
])
def test_spherical_specs_are_checked(bad):
    import matplotlib.pyplot as plt

    open_before = set(plt.get_fignums())
    with pytest.raises(ValueError):
        hlp.chart_spec_b64(bad)
    assert set(plt.get_fignums()) == open_before


@pytest.mark.parametrize("span", [30, 35, 45])
def test_level_colour_bar_is_labelled_at_0_db(span):
    fig = build_chart_spec_figure(_sphere_spec("beam_map", range_db=span), theme="classic")
    ticks = list(fig.axes[-1].get_yticks())
    assert ticks[-1] == pytest.approx(0.0)
    assert all(tick >= -span - 1e-9 for tick in ticks)


def test_beam_map_fills_the_disc_when_theta_stops_short_of_90():
    spec = _sphere_spec("beam_map")
    keep = [i for i, t in enumerate(spec["theta_deg"]) if 20 <= t <= 60]
    spec["theta_deg"] = [spec["theta_deg"][i] for i in keep]
    spec["levels_db"] = [spec["levels_db"][i] for i in keep]
    fig = build_chart_spec_figure(spec, theme="classic")
    mesh = fig.axes[0].collections[0]
    radii = np.asarray(mesh.get_coordinates())[..., 1]
    # As on screen: the nearest sampled row is read out to the centre and the rim.
    assert radii.min() == pytest.approx(0.0)
    assert radii.max() == pytest.approx(90.0)


def test_balloon_floor_is_lifted_off_the_page_and_a_dense_grid_stays_quick():
    import time

    from matplotlib.colors import to_rgba

    from hornlab_plots.style import resolve_theme

    fig = build_chart_spec_figure(_sphere_spec("balloon"), theme="classic")
    surface = fig.axes[0].collections[0]
    page = np.asarray(to_rgba(resolve_theme("classic").figure_bg))
    floor = min(surface.get_facecolor(), key=lambda rgba: float(np.sum(rgba[:3])))
    # The quietest face is visibly darker than the white page.
    assert np.sum(page[:3]) - np.sum(np.asarray(floor)[:3]) > 0.15

    theta = np.linspace(0, 180, 361).tolist()
    phi = np.linspace(0, 360, 554, endpoint=False).tolist()
    dense = {"kind": "balloon", "theta_deg": theta, "phi_deg": phi, "levels_db": [[-t / 6 for _ in phi] for t in theta]}
    started = time.perf_counter()
    _png(hlp.chart_spec_b64(dense, theme="classic"))
    # It was 13 s before the face budget; a few seconds allows a loaded machine.
    assert time.perf_counter() - started < 6.0
