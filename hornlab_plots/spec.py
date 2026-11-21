"""Charts described as data, drawn in the active HornLab theme.

A consumer that draws its own interactive charts (Waveguide Generator's
on-screen charts) describes a figure as a plain mapping -- axes, series, shaded
bands -- and gets the same figure back as a themed PNG. This is how one
renderer serves every saved image of a chart the consumer shows, including
charts the canonical renderers in :mod:`hornlab_plots.charts` do not cover
(phase, group delay, power, excursion, polar cuts, ...).

Colours are theme roles, never literal hex: an integer is a slot in the
theme's line cycle, and ``"accent"``, ``"ink"`` and ``"muted"`` name the
theme's accent, text and tick colours. The theme therefore owns every hue, and
a consumer that assigns slots the way its screen does gets the same run in the
same position of the palette in both places.

Spec shape (``kind`` defaults to ``"lines"``)::

    {
      "kind": "lines",
      "title": "Group delay", "subtitle": "optional second line",
      "x": {"label": "Frequency [Hz]", "scale": "log", "min": 100, "max": 20000},
      "y": {"label": "Group delay [ms]", "min": null, "max": null},
      "y2": {"label": "Phase [deg]"},            # optional right-hand axis
      "series": [{"label": "Run 1", "x": [...], "y": [...], "axis": 0,
                  "color": 0, "style": "solid", "width": 2, "opacity": 1,
                  "markers": [3, 7]}],   # optional hollow circles at indices
      "bands": [{"x0": 4000, "x1": 6000, "label": "Unqualified"}],
      "hlines": [{"y": 0, "axis": 0}],
    }

    {
      "kind": "polar",
      "title": "Polar response", "subtitle": "horizontal - 1 kHz",
      "r": {"min": -30, "max": 0, "step": 10, "unit": "dB"},
      "series": [{"label": "Run 1", "angle": [...], "r": [...], "color": 0}],
      "shade": [{"from": 90, "to": 180, "label": "Outside half-space"}],
    }

``None`` values are gaps. Every series point is drawn as given: nothing is
smoothed, resampled or re-referenced here, so the image carries exactly the
numbers the consumer put on its screen.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from ._grid import freq_formatter
from ._png import fig_to_b64
from .charts import _add_log_grid, _apply_canonical_axes, _style_twin_axis, _themed_legend
from .style import apply_theme_overrides, theme_rc_context

LINE_FIGSIZE = (10.0, 4.8)
POLAR_FIGSIZE = (7.0, 7.0)
# Screen widths are CSS pixels; the canonical charts draw a primary trace at
# 1.65 pt against the screen's 2 px.
_WIDTH_SCALE = 0.825
# Bounds on the work a spec can ask for: the rings and log grid lines are
# drawn one by one, so an absurd step or span would take minutes.
MAX_POLAR_RINGS = 60
MAX_LOG_DECADES = 8
# Counts a chart may ask for. Each band, line and sector is drawn as its own
# artist, so they are bounded as well as the points. Waveguide Generator's
# route model applies the same bounds before a request reaches here.
MAX_SERIES = 64
MAX_POINTS_PER_SERIES = 20_000
MAX_POINTS = 200_000
MAX_BANDS = 64
MAX_HLINES = 16
MAX_SHADE = 8
_LOG_SPAN_REFUSED = f"a log frequency axis must start above 0 and span at most {MAX_LOG_DECADES} decades"
_LINESTYLES = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}


def _role_color(theme_obj, role) -> str:
    if isinstance(role, bool):
        raise ValueError("series color must be a palette slot or a role name")
    if isinstance(role, (int, np.integer)):
        cycle = theme_obj.line_colors
        return cycle[int(role) % len(cycle)]
    if role in (None, "accent"):
        return theme_obj.role_colors.get("accent", theme_obj.line_colors[0])
    if role == "ink":
        return theme_obj.text_color
    if role == "muted":
        return theme_obj.tick_color
    raise ValueError(f"unknown series color role: {role!r}")


def _floats(values) -> np.ndarray:
    return np.asarray(
        [math.nan if value is None else float(value) for value in (values or [])],
        dtype=float,
    )


def _line_kwargs(theme_obj, series: Mapping[str, Any]) -> dict[str, Any]:
    style = str(series.get("style") or "solid")
    if style not in _LINESTYLES:
        raise ValueError(f"unknown series line style: {style!r}")
    return {
        "color": _role_color(theme_obj, series.get("color", 0)),
        "linestyle": _LINESTYLES[style],
        "linewidth": float(series.get("width") or 2.0) * _WIDTH_SCALE,
        "alpha": float(series.get("opacity", 1.0)),
        "zorder": float(series.get("z", 2)),
    }


def _marker_mask(indices, keep: np.ndarray) -> np.ndarray | None:
    """Which kept points carry a marker, from indices into the series as sent."""
    if not indices:
        return None
    full = np.zeros(keep.size, dtype=bool)
    for index in indices:
        index = int(index)
        if 0 <= index < keep.size:
            full[index] = True
    return full[keep]


def _title(ax, spec: Mapping[str, Any], theme_obj) -> None:
    subtitle = spec.get("subtitle")
    if subtitle:
        ax.set_title(
            str(spec.get("title") or ""),
            color=theme_obj.text_color,
            fontsize=13,
            fontweight="600",
            pad=22,
        )
        ax.annotate(
            str(subtitle),
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=theme_obj.tick_color,
        )


def _limits(axis: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    axis = axis or {}
    lower, upper = axis.get("min"), axis.get("max")
    return (
        None if lower is None else float(lower),
        None if upper is None else float(upper),
    )


def _mapping(value, what: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be a mapping")
    return value


def _entries(value, what: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{what} must be a list")
    return [_mapping(item, f"each entry of {what}") for item in value]


def _check_counts(spec: Mapping[str, Any]) -> None:
    """Refuse a spec asking for more artists or points than any chart needs."""
    for key, cap in (("series", MAX_SERIES), ("bands", MAX_BANDS), ("hlines", MAX_HLINES), ("shade", MAX_SHADE)):
        entries = _entries(spec.get(key), key)
        if len(entries) > cap:
            raise ValueError(f"a chart may have at most {cap} {key}")
    total = 0
    for series in _entries(spec.get("series"), "series"):
        for key in ("x", "y", "angle", "r", "markers"):
            values = series.get(key)
            if values is None:
                continue
            if not isinstance(values, (list, tuple, np.ndarray)):
                raise ValueError(f"series {key} must be a list")
            if len(values) > MAX_POINTS_PER_SERIES:
                raise ValueError(f"a series may have at most {MAX_POINTS_PER_SERIES} points")
        total += len(series.get("x") or series.get("angle") or [])
    if total > MAX_POINTS:
        raise ValueError(f"a chart may draw at most {MAX_POINTS} points")


def _build_lines_figure(fig, spec: Mapping[str, Any], theme_obj):
    series_specs = _entries(spec.get("series"), "series")
    x_axis = _mapping(spec.get("x"), "x")
    log_x = str(x_axis.get("scale", "log")) == "log"
    x_lo, x_hi = _limits(x_axis)
    if log_x and x_lo is not None and x_lo <= 0.0:
        raise ValueError(_LOG_SPAN_REFUSED)
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor(theme_obj.figure_bg)
    second = None
    drawn_x: list[np.ndarray] = []
    for series in series_specs:
        x = _floats(series.get("x"))
        y = _floats(series.get("y"))
        count = min(x.size, y.size)
        x, y = x[:count], y[:count]
        keep = np.isfinite(x) & ((x > 0.0) if log_x else True)
        x, y = x[keep], y[keep]
        if not np.any(np.isfinite(y)):
            continue
        target = ax
        if int(series.get("axis", 0)) == 1:
            if second is None:
                second = ax.twinx()
                second.set_facecolor("none")
            target = second
        label = series.get("label")
        style = _line_kwargs(theme_obj, series)
        target.plot(x, y, label=str(label) if label else "_nolegend_", **style)
        marked = _marker_mask(series.get("markers"), keep)
        if marked is not None and np.any(marked):
            # Hollow circles on the points the consumer marks (for example
            # reconstructed, not solved, frequencies).
            target.plot(
                x[marked], y[marked], linestyle="none", marker="o", markersize=4,
                markerfacecolor="none", markeredgecolor=style["color"],
                alpha=style["alpha"], zorder=style["zorder"], label="_nolegend_",
            )
        drawn_x.append(x)
    if not drawn_x:
        return None

    if log_x:
        ax.set_xscale("log")
    _apply_canonical_axes(
        ax,
        str(x_axis.get("label") or ""),
        str((spec.get("y") or {}).get("label") or ""),
        str(spec.get("title") or ""),
        theme=theme_obj,
        grid_axis="y" if log_x else "both",
    )
    _title(ax, spec, theme_obj)
    data_lo = min(float(np.min(values)) for values in drawn_x)
    data_hi = max(float(np.max(values)) for values in drawn_x)
    x_lo = data_lo if x_lo is None else x_lo
    x_hi = data_hi if x_hi is None else x_hi
    if log_x and x_hi > x_lo and math.log10(x_hi / x_lo) > MAX_LOG_DECADES:
        raise ValueError(_LOG_SPAN_REFUSED)
    if x_hi > x_lo:
        ax.set_xlim(x_lo, x_hi)
    if log_x:
        ax.xaxis.set_major_formatter(FuncFormatter(freq_formatter))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda _value, _pos: ""))
        lo, hi = ax.get_xlim()
        _add_log_grid(ax, lo, hi, detailed=True, theme=theme_obj)
    y_lo, y_hi = _limits(_mapping(spec.get("y"), "y"))
    if y_lo is not None or y_hi is not None:
        ax.set_ylim(y_lo, y_hi)

    for line in _entries(spec.get("hlines"), "hlines"):
        target = second if int(line.get("axis", 0)) == 1 and second is not None else ax
        target.axhline(float(line["y"]), color=theme_obj.tick_color, linewidth=0.8, alpha=0.45, zorder=1)

    band_handle = None
    for band in _entries(spec.get("bands"), "bands"):
        ax.axvspan(
            float(band["x0"]),
            float(band["x1"]),
            facecolor="none",
            edgecolor=theme_obj.mesh_limit_color,
            hatch="///",
            linewidth=0.0,
            alpha=0.45,
            zorder=0,
        )
        if band_handle is None:
            band_handle = Patch(
                facecolor="none",
                edgecolor=theme_obj.mesh_limit_color,
                hatch="///",
                alpha=0.45,
                label=str(band.get("label") or "Unqualified"),
            )

    if second is not None:
        y2 = _mapping(spec.get("y2"), "y2")
        _style_twin_axis(
            second,
            str(y2.get("label") or ""),
            label_color=theme_obj.text_color,
            tick_color=theme_obj.tick_color,
            spine_color=theme_obj.spine_color,
        )
        # The twin draws its own frame over the main axes': keep all of it in
        # the theme, not only the right-hand side the helper styles.
        for side in ("left", "top", "bottom"):
            second.spines[side].set_color(theme_obj.spine_color)
        y2_lo, y2_hi = _limits(y2)
        if y2_lo is not None or y2_hi is not None:
            second.set_ylim(y2_lo, y2_hi)

    handles, labels = ax.get_legend_handles_labels()
    if second is not None:
        more_handles, more_labels = second.get_legend_handles_labels()
        handles += more_handles
        labels += more_labels
    if band_handle is not None:
        handles.append(band_handle)
        labels.append(band_handle.get_label())
    if handles:
        _themed_legend(ax, handles, labels, theme=theme_obj, loc="best")
    fig.tight_layout(pad=1.5)
    return fig


def _build_polar_figure(fig, spec: Mapping[str, Any], theme_obj):
    radius = _mapping(spec.get("r"), "r")
    r_lo, r_hi = _limits(radius)
    r_lo = -30.0 if r_lo is None else r_lo
    r_hi = 0.0 if r_hi is None else r_hi
    if not r_hi > r_lo:
        raise ValueError("polar radius range is reversed or empty")
    step = float(radius.get("step") or (10.0 if abs(r_hi - r_lo) >= 30 else 5.0))
    if not step > 0.0 or (r_hi - r_lo) / step > MAX_POLAR_RINGS:
        raise ValueError(f"the polar ring step must be positive and give at most {MAX_POLAR_RINGS} rings")
    unit = str(radius.get("unit") or "dB")

    fig.patch.set_facecolor(theme_obj.figure_bg)
    ax = fig.add_subplot(projection="polar")
    ax.set_facecolor(theme_obj.axes_bg)
    # On-axis up, positive angles to the right: a loudspeaker polar plot.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetalim(-math.pi, math.pi)
    ax.set_rlim(r_lo, r_hi)
    drawn = False
    for series in _entries(spec.get("series"), "series"):
        angle = _floats(series.get("angle"))
        level = _floats(series.get("r"))
        count = min(angle.size, level.size)
        angle, level = angle[:count], level[:count]
        if not np.any(np.isfinite(level)):
            continue
        label = series.get("label")
        ax.plot(
            np.radians(angle),
            # Drawn as given: the polar axes' own clip path cuts anything
            # beyond the outer ring, as the screen's polar coordinate system does.
            level,
            label=str(label) if label else "_nolegend_",
            **_line_kwargs(theme_obj, series),
        )
        drawn = True
    if not drawn:
        return None
    for wedge in _entries(spec.get("shade"), "shade"):
        theta = np.radians(np.linspace(float(wedge["from"]), float(wedge["to"]), 37))
        ax.fill_between(
            theta, r_lo, r_hi, color=theme_obj.tick_color, alpha=0.12, linewidth=0,
            label=str(wedge.get("label") or "_nolegend_"), zorder=0,
        )

    ax.set_thetagrids(
        np.arange(-150, 181, 30),
        labels=[f"{value}°" for value in range(-150, 181, 30)],
        color=theme_obj.tick_color,
        fontsize=9,
    )
    rings = np.arange(r_lo, r_hi + step * 0.5, step)
    ax.set_rgrids(
        rings,
        labels=["" if value in (r_lo, r_hi) else f"{value:g} {unit}" for value in rings],
        angle=180 - 22.5,
        color=theme_obj.tick_color,
        fontsize=8,
    )
    ax.grid(True, color=theme_obj.grid_color, linewidth=0.6)
    ax.spines["polar"].set_color(theme_obj.spine_color)
    ax.set_title(str(spec.get("title") or ""), color=theme_obj.text_color, fontsize=13, fontweight="600", pad=44)
    if spec.get("subtitle"):
        ax.annotate(
            str(spec["subtitle"]),
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 24),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=theme_obj.tick_color,
        )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        _themed_legend(
            ax, handles, labels, theme=theme_obj,
            loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=min(3, len(handles)),
        )
    fig.tight_layout(pad=1.5)
    return fig


_BUILDERS = {
    "lines": (_build_lines_figure, LINE_FIGSIZE),
    "polar": (_build_polar_figure, POLAR_FIGSIZE),
}


def build_chart_spec_figure(spec: Mapping[str, Any], *, theme=None):
    """Build the matplotlib figure for ``spec``; ``None`` when nothing is drawable."""
    spec = _mapping(spec, "a chart spec")
    kind = str(spec.get("kind") or "lines")
    if kind not in _BUILDERS:
        raise ValueError(f"chart kind must be one of: {', '.join(_BUILDERS)}")
    builder, figsize = _BUILDERS[kind]
    _check_counts(spec)
    theme_obj = apply_theme_overrides(theme)
    with theme_rc_context(theme_obj):
        fig = plt.figure(figsize=figsize)
        try:
            drawn = builder(fig, spec, theme_obj)
        except BaseException:
            plt.close(fig)
            raise
        if drawn is None:
            plt.close(fig)
        return drawn


def chart_spec_b64(spec: Mapping[str, Any], dpi=150, *, theme=None) -> str | None:
    """Render ``spec`` and return a base64 PNG (no data-URI prefix), or ``None``."""
    fig = build_chart_spec_figure(spec, theme=theme)
    if fig is None:
        return None
    return fig_to_b64(fig, dpi)


__all__ = ["build_chart_spec_figure", "chart_spec_b64"]
