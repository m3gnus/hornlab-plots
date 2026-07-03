"""Line-chart renderers for frequency response, DI, and impedance.

Re-uses the canonical ``_grid`` helpers from this package so the log-grid
styling is shared with the directivity heatmap.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from ._grid import freq_formatter, log_grid_lines, preferred_frequency_ticks
from .style import (
    DPI,
    apply_theme_overrides,
    theme_grid_kwargs,
    theme_rc_context,
)


@dataclass(frozen=True)
class FrequencyResponseCurve:
    """One trace in a canonical frequency-response plot.

    ``role`` controls color. Use ``lf``, ``mf``, ``hf``, ``combined``,
    ``raw``, or ``other``. Set ``crossover=True`` for filtered LP/HP/BP
    traces so they render as dotted component responses.
    """

    frequencies: object
    spl_db: object
    label: str
    role: str = "other"
    crossover: bool = False


def spl_window(
    curves,
    *,
    span_db: float = 40.0,
    top_margin_db: float = 3.0,
    floor_db: float | None = None,
) -> tuple[float, float] | None:
    """Return a compact SPL y-window for one or more plotted curves."""
    finite_values = []
    for curve in curves:
        values = np.asarray(curve, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_values.append(finite)
    if not finite_values:
        return None

    top = float(max(np.max(values) for values in finite_values)) + top_margin_db
    bottom = top - span_db
    if floor_db is not None and bottom < floor_db:
        bottom = floor_db
        top = bottom + span_db
    return bottom, top


def set_spl_window(
    ax,
    curves,
    *,
    span_db: float = 40.0,
    top_margin_db: float = 3.0,
    floor_db: float | None = None,
) -> None:
    """Apply a compact SPL y-window to a Matplotlib axes."""
    window = spl_window(curves, span_db=span_db, top_margin_db=top_margin_db, floor_db=floor_db)
    if window is not None:
        ax.set_ylim(*window)


def _apply_canonical_axes(ax, xlabel, ylabel, title, *, theme=None, colors=None):
    theme_obj = apply_theme_overrides(theme, colors=colors)
    ax.set_facecolor(theme_obj.axes_bg)
    ax.set_xlabel(xlabel, color=theme_obj.text_color, fontsize=11)
    ax.set_ylabel(ylabel, color=theme_obj.text_color, fontsize=11)
    ax.set_title(title, color=theme_obj.text_color, fontsize=13, fontweight="600", pad=8)
    ax.tick_params(colors=theme_obj.tick_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(theme_obj.spine_color)
    ax.grid(
        True,
        alpha=theme_obj.secondary_grid_alpha,
        color=theme_obj.grid_color,
        **theme_grid_kwargs(theme_obj, linewidth=0.5),
    )


def _setup_dark_axes(ax, xlabel, ylabel, title, *, theme=None, colors=None):
    """Apply dark theme styling to axes."""
    _apply_canonical_axes(ax, xlabel, ylabel, title, theme=theme, colors=colors)


def _add_log_grid(ax, freq_min, freq_max, *, detailed=False, theme=None, colors=None):
    """Add log-frequency grid lines matching the directivity-heatmap style."""
    theme_obj = apply_theme_overrides(theme, colors=colors)
    if detailed:
        ticks = preferred_frequency_ticks(freq_min, freq_max)
        if ticks:
            ax.set_xticks(ticks)
        for freq in ticks:
            ax.axvline(
                freq,
                color=theme_obj.grid_color,
                alpha=theme_obj.primary_grid_alpha,
                **theme_grid_kwargs(theme_obj, linewidth=0.7),
            )
        for freq in log_grid_lines(freq_min, freq_max):
            if not any(np.isclose(freq, tick, rtol=1e-6, atol=1e-6) for tick in ticks):
                ax.axvline(
                    freq,
                    color=theme_obj.grid_color,
                    alpha=theme_obj.secondary_grid_alpha,
                    **theme_grid_kwargs(theme_obj, linewidth=0.5),
                )
        return

    for freq in log_grid_lines(freq_min, freq_max):
        ax.axvline(
            freq,
            color=theme_obj.grid_color,
            alpha=theme_obj.secondary_grid_alpha,
            **theme_grid_kwargs(theme_obj, linewidth=0.5),
        )


def _fig_to_base64(fig, dpi=150):
    """Export figure to base64-encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _coerce_response_curve(curve) -> FrequencyResponseCurve:
    if isinstance(curve, FrequencyResponseCurve):
        return curve
    if isinstance(curve, dict):
        return FrequencyResponseCurve(
            frequencies=curve.get("frequencies", curve.get("freqs")),
            spl_db=curve.get("spl_db", curve.get("spl")),
            label=str(curve.get("label", curve.get("role", "response"))),
            role=str(curve.get("role", "other")),
            crossover=bool(curve.get("crossover", False)),
        )
    if isinstance(curve, (list, tuple)) and len(curve) >= 3:
        return FrequencyResponseCurve(
            frequencies=curve[0],
            spl_db=curve[1],
            label=str(curve[2]),
            role=str(curve[3]) if len(curve) > 3 else "other",
            crossover=bool(curve[4]) if len(curve) > 4 else False,
        )
    raise TypeError("frequency response curves must be FrequencyResponseCurve, dict, or tuple")


def _response_curve_style(
    curve: FrequencyResponseCurve,
    *,
    theme=None,
    colors=None,
    response_colors=None,
) -> dict[str, object]:
    theme_obj = apply_theme_overrides(theme, colors=colors, response_colors=response_colors)
    role = curve.role.lower()
    if role in {"sum", "total", "combined"}:
        return {
            "color": theme_obj.response_colors["combined"],
            "linewidth": 2.8,
            "linestyle": "-",
            "alpha": 1.0,
            "zorder": 4,
        }
    linestyle = ":" if curve.crossover else "-"
    linewidth = 2.0 if curve.crossover else 1.65
    return {
        "color": theme_obj.response_colors.get(role, theme_obj.response_colors["other"]),
        "linewidth": linewidth,
        "linestyle": linestyle,
        "alpha": 0.95 if curve.crossover else 0.78,
        "zorder": 3 if curve.crossover else 2,
    }


def _build_frequency_response_figure(
    curves,
    *,
    title: str = "Frequency Response",
    ylabel: str = "SPL [dB]",
    xlabel: str = "Frequency [Hz]",
    crossover_hz: float | None = None,
    crossover_label: str | None = None,
    mesh_valid_hz: float | None = None,
    mesh_valid_radiating_hz: float | None = None,
    xlim: tuple[float, float] | None = None,
    span_db: float = 40.0,
    top_margin_db: float = 3.0,
    floor_db: float | None = None,
    figsize: tuple[float, float] = (10.0, 4.8),
    theme=None,
    colors=None,
    line_colors=None,
    response_colors=None,
) -> object | None:
    theme_obj = apply_theme_overrides(
        theme,
        colors=colors,
        line_colors=line_colors,
        response_colors=response_colors,
    )
    response_curves = [_coerce_response_curve(curve) for curve in curves]
    if not response_curves:
        return None

    with theme_rc_context(theme_obj):
        plotted = []
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.patch.set_facecolor(theme_obj.figure_bg)

        for curve in response_curves:
            freqs = np.asarray(curve.frequencies, dtype=float)
            values = np.asarray(curve.spl_db, dtype=float)
            if freqs.size == 0 or values.size == 0:
                continue
            n = min(freqs.size, values.size)
            freqs = freqs[:n]
            values = values[:n]
            finite = np.isfinite(freqs) & np.isfinite(values) & (freqs > 0)
            if not np.any(finite):
                continue
            freqs = freqs[finite]
            values = values[finite]
            plotted.append((freqs, values))
            ax.semilogx(
                freqs,
                values,
                label=curve.label,
                **_response_curve_style(curve, theme=theme_obj),
            )

        if not plotted:
            plt.close(fig)
            return None

        if crossover_hz is not None:
            label = crossover_label or f"XO {float(crossover_hz):.0f} Hz"
            ax.axvline(
                float(crossover_hz),
                color=theme_obj.text_color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.72,
                label=label,
                zorder=1,
            )

        if mesh_valid_hz is not None or mesh_valid_radiating_hz is not None:
            from ._heatmap import _draw_mesh_valid_markers

            freq_lo = min(float(np.min(freqs)) for freqs, _values in plotted)
            freq_hi = max(float(np.max(freqs)) for freqs, _values in plotted)
            _draw_mesh_valid_markers(
                ax,
                lo=freq_lo,
                hi=freq_hi,
                mesh_valid_hz=mesh_valid_hz,
                mesh_valid_radiating_hz=mesh_valid_radiating_hz,
                span_zorder=0,
                line_zorder=1,
                theme=theme_obj,
            )

        _apply_canonical_axes(ax, xlabel, ylabel, title, theme=theme_obj)
        ax.xaxis.set_major_formatter(FuncFormatter(freq_formatter))
        freq_min = min(float(np.min(freqs)) for freqs, _values in plotted)
        freq_max = max(float(np.max(freqs)) for freqs, _values in plotted)
        ax.set_xlim(*(xlim or (freq_min, freq_max)))
        set_spl_window(
            ax,
            [values for _freqs, values in plotted],
            span_db=span_db,
            top_margin_db=top_margin_db,
            floor_db=floor_db,
        )
        _add_log_grid(ax, ax.get_xlim()[0], ax.get_xlim()[1], detailed=True, theme=theme_obj)
        legend = ax.legend(
            loc="best",
            fontsize=9,
            facecolor=theme_obj.axes_bg,
            edgecolor=theme_obj.spine_color,
            labelcolor=theme_obj.text_color,
        )
        legend.get_frame().set_alpha(0.92)
        fig.tight_layout(pad=1.5)
        return fig


def frequency_response_multi_b64(curves, dpi=DPI, **kwargs):
    """Render a canonical multi-trace frequency-response plot as base64 PNG."""
    fig = _build_frequency_response_figure(curves, **kwargs)
    if fig is None:
        return None
    return _fig_to_base64(fig, dpi)


def save_frequency_response_plot(output_path, curves, dpi=DPI, **kwargs):
    """Save a canonical frequency-response plot to a PNG file."""
    fig = _build_frequency_response_figure(curves, **kwargs)
    if fig is None:
        return None
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(out),
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close(fig)
    return out


def frequency_response_b64(
    frequencies,
    spl,
    dpi=150,
    *,
    theme=None,
    colors=None,
    line_colors=None,
    response_colors=None,
):
    """Render on-axis SPL vs frequency and return base64-encoded PNG."""
    return frequency_response_multi_b64(
        [FrequencyResponseCurve(frequencies, spl, "On-axis", role="combined")],
        title="Frequency Response (On-Axis)",
        dpi=dpi,
        theme=theme,
        colors=colors,
        line_colors=line_colors,
        response_colors=response_colors,
    )


def directivity_index_b64(
    frequencies,
    di,
    dpi=150,
    *,
    theme=None,
    colors=None,
    line_colors=None,
):
    """Render DI vs frequency, per-plane, and return base64-encoded PNG.

    Args:
        frequencies: 1D array.
        di: Either a flat list of DI values (legacy single-plane) or a dict
            mapping plane IDs to DI value lists, e.g.
            ``{"horizontal": [...], "vertical": [...], "diagonal": [...]}``.
    """
    freqs = np.array(frequencies, dtype=float)
    if len(freqs) == 0:
        return None

    theme_obj = apply_theme_overrides(theme, colors=colors, line_colors=line_colors)
    plane_colors = theme_obj.plane_colors
    line_palette = tuple(line_colors) if line_colors is not None else None
    plane_labels = {
        "horizontal": "H",
        "vertical":   "V",
        "diagonal":   "D",
    }

    if isinstance(di, dict):
        planes = {}
        for plane_id in ("horizontal", "vertical", "diagonal"):
            vals = di.get(plane_id)
            if vals and any(v is not None for v in vals):
                arr = np.array([v if v is not None else np.nan for v in vals], dtype=float)
                planes[plane_id] = arr
    elif isinstance(di, list) and len(di) > 0:
        arr = np.array([v if v is not None else np.nan for v in di], dtype=float)
        if not np.all(np.isnan(arr)):
            planes = {"horizontal": arr}
        else:
            return None
    else:
        return None

    if not planes:
        return None

    with theme_rc_context(theme_obj):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        fig.patch.set_facecolor(theme_obj.figure_bg)

        all_vals = []
        for index, (plane_id, di_vals) in enumerate(planes.items()):
            color = (
                line_palette[index % len(line_palette)]
                if line_palette
                else plane_colors.get(plane_id, "#81c784")
            )
            label = plane_labels.get(plane_id, plane_id.capitalize())
            ax.semilogx(freqs, di_vals, color=color, linewidth=1.5, label=label)
            valid = di_vals[~np.isnan(di_vals)]
            if len(valid) > 0:
                all_vals.extend(valid.tolist())

        if not all_vals:
            plt.close(fig)
            return None

        if len(planes) > 1:
            ax.legend(loc="upper left", fontsize=9, facecolor=theme_obj.axes_bg,
                      edgecolor=theme_obj.spine_color, labelcolor=theme_obj.text_color)

        _setup_dark_axes(ax, "Frequency [Hz]", "DI [dB]", "Directivity Index", theme=theme_obj)
        ax.xaxis.set_major_formatter(FuncFormatter(freq_formatter))

        ax.set_xlim(freqs[0], freqs[-1])
        di_min, di_max = np.nanmin(all_vals), np.nanmax(all_vals)
        margin = max(2, (di_max - di_min) * 0.1)
        ax.set_ylim(min(0, di_min - margin), di_max + margin)

        _add_log_grid(ax, freqs[0], freqs[-1], detailed=True, theme=theme_obj)
        ax.tick_params(axis="x", labelsize=8)

        fig.tight_layout(pad=1.5)
    return _fig_to_base64(fig, dpi)


def impedance_b64(
    frequencies,
    real,
    imaginary,
    dpi=150,
    *,
    theme=None,
    colors=None,
    line_colors=None,
):
    """Render acoustic impedance (real + imaginary) and return base64 PNG."""
    fig = _build_impedance_figure(frequencies, real, imaginary, theme=theme, colors=colors, line_colors=line_colors)
    if fig is None:
        return None
    return _fig_to_base64(fig, dpi)


def save_impedance_plot(
    output_path,
    frequencies,
    real,
    imaginary,
    dpi=150,
    *,
    theme=None,
    colors=None,
    line_colors=None,
):
    """Save acoustic impedance chart to a PNG file on disk."""
    fig = _build_impedance_figure(frequencies, real, imaginary, theme=theme, colors=colors, line_colors=line_colors)
    if fig is None:
        return None
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), format="png", dpi=dpi, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return out


def _build_impedance_figure(frequencies, real, imaginary, *, theme=None, colors=None, line_colors=None):
    theme_obj = apply_theme_overrides(theme, colors=colors, line_colors=line_colors)
    line_palette = tuple(line_colors) if line_colors is not None else None
    freqs = np.array(frequencies, dtype=float)
    re_vals = np.array(real, dtype=float)
    im_vals = np.array(imaginary, dtype=float)

    if len(freqs) == 0 or len(re_vals) == 0:
        return None

    with theme_rc_context(theme_obj):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        fig.patch.set_facecolor(theme_obj.figure_bg)

        real_color = line_palette[0] if line_palette else theme_obj.impedance_colors["real"]
        imag_color = (
            line_palette[1 % len(line_palette)]
            if line_palette
            else theme_obj.impedance_colors["imaginary"]
        )
        ax.semilogx(freqs, re_vals, color=real_color, linewidth=1.5, label="Re(Z)")
        if len(im_vals) > 0:
            ax.semilogx(freqs, im_vals, color=imag_color, linewidth=1.5, label="Im(Z)")

        _setup_dark_axes(ax, "Frequency [Hz]", "Z [Pa·s/m]", "Acoustic Impedance", theme=theme_obj)
        ax.xaxis.set_major_formatter(FuncFormatter(freq_formatter))

        ax.set_xlim(freqs[0], freqs[-1])

        all_vals = np.concatenate([re_vals, im_vals]) if len(im_vals) > 0 else re_vals
        z_min, z_max = np.nanmin(all_vals), np.nanmax(all_vals)
        margin = max(50, (z_max - z_min) * 0.1)
        ax.set_ylim(z_min - margin, z_max + margin)

        _add_log_grid(ax, freqs[0], freqs[-1], theme=theme_obj)

        legend = ax.legend(loc="upper right", fontsize=10,
                           facecolor=theme_obj.axes_bg, edgecolor=theme_obj.spine_color,
                           labelcolor=theme_obj.text_color)
        legend.get_frame().set_alpha(0.9)

        fig.tight_layout(pad=1.5)
        return fig


def render_all_charts_b64(
    payload,
    dpi=150,
    *,
    theme=None,
    colors=None,
    line_colors=None,
    response_colors=None,
):
    """Render all charts from a combined results payload.

    Returns a dict with keys ``frequency_response``, ``directivity_index``,
    ``impedance``, ``directivity_map`` — each a base64 PNG or None.
    """
    from ._heatmap import directivity_heatmap_from_legacy_dict

    freqs = payload.get("frequencies", [])
    spl = payload.get("spl", [])
    di_freqs = payload.get("di_frequencies", []) or freqs
    imp_freqs = payload.get("impedance_frequencies", []) or freqs
    imp_real = payload.get("impedance_real", [])
    imp_imag = payload.get("impedance_imaginary", [])
    directivity = payload.get("directivity", {})

    charts = {}

    charts["frequency_response"] = frequency_response_b64(
        freqs,
        spl,
        dpi,
        theme=theme,
        colors=colors,
        line_colors=line_colors,
        response_colors=response_colors,
    ) if spl else None
    di_input = payload.get("di", [])
    charts["directivity_index"] = directivity_index_b64(
        di_freqs,
        di_input,
        dpi,
        theme=theme,
        colors=colors,
        line_colors=line_colors,
    ) if di_input else None
    charts["impedance"] = impedance_b64(
        imp_freqs,
        imp_real,
        imp_imag,
        dpi,
        theme=theme,
        colors=colors,
        line_colors=line_colors,
    ) if imp_real else None

    dir_b64 = None
    if directivity and freqs:
        try:
            dir_b64 = directivity_heatmap_from_legacy_dict(freqs, directivity, dpi, theme=theme, colors=colors)
        except Exception:
            pass
    charts["directivity_map"] = dir_b64

    return charts
