"""Directivity heatmap core renderer.

The public entry points (`directivity_heatmap_b64`, `save_directivity_plot`,
`directivity_heatmap_from_legacy_dict`) are wired in `__init__.py`. This
module owns the actual rendering primitives.

Byte-equivalence note: the heatmap code was extracted unchanged from the
former WG directivity renderer. The legacy ``[[angle, dB], ...]``
per-frequency dict shape is preserved via the
adapter ``directivity_heatmap_from_legacy_dict``; new callers should
prefer the numpy-array signature ``directivity_heatmap_b64``.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from ._grid import (
    contains_frequency,
    freq_formatter,
    log_grid_lines,
    preferred_frequency_ticks,
)
from .style import (
    ANGLE_SAMPLES,
    AXES_BG,
    CONTOUR_OUTLINE,
    FIGURE_BG,
    FRACTIONAL_OCTAVE,
    FREQ_SAMPLES,
    GRID_COLOR,
    HEATMAP_CMAP,
    MAX_DB,
    MESH_LIMIT_COLOR,
    MIN_DB,
    PRIMARY_GRID_ALPHA,
    REFERENCE_CONTOUR_COLOR,
    SECONDARY_GRID_ALPHA,
    SPINE_COLOR,
    TEXT_COLOR,
    TICK_COLOR,
)


# ---------------------------------------------------------------------------
# Legacy-shape adapters
# ---------------------------------------------------------------------------

def build_grid_from_legacy(freqs, patterns):
    """Convert list of ``[[angle, dB], ...]`` per frequency into 2D arrays.

    Returns ``(angles, freqs, values)`` where ``values`` is a 2D
    ``(n_angles, n_freqs)`` array. Returns ``(None, None, None)`` if the
    input is empty or contains no finite values.
    """
    if not patterns:
        return None, None, None

    n_freqs = min(len(patterns), len(freqs))
    if n_freqs == 0:
        return None, None, None

    angles = None
    for pattern in patterns[:n_freqs]:
        candidate = _extract_angles(pattern)
        if candidate is not None and candidate.size > 0:
            angles = candidate
            break
    if angles is None or angles.size == 0:
        return None, None, None

    values = np.full((angles.size, n_freqs), np.nan, dtype=float)
    for fi in range(n_freqs):
        pattern = patterns[fi]
        if not isinstance(pattern, list):
            continue
        for ai, point in enumerate(pattern[: angles.size]):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            db = _safe_float(point[1])
            if db is not None:
                values[ai, fi] = db

    keep_cols = np.any(np.isfinite(values), axis=0)
    if not np.any(keep_cols):
        return None, None, None

    return angles, freqs[:n_freqs][keep_cols], values[:, keep_cols]


def _extract_angles(pattern):
    if not isinstance(pattern, list):
        return None
    out = []
    for point in pattern:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        ang = _safe_float(point[0])
        if ang is not None:
            out.append(ang)
    if not out:
        return None
    return np.array(out, dtype=float)


def _safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


# ---------------------------------------------------------------------------
# Heatmap data prep
# ---------------------------------------------------------------------------

def prepare_heatmap_data(angles, freqs, values):
    """Fill missing values, fractional-octave smooth, interpolate to the
    canonical (361 x 500) display grid, and clip to [MIN_DB, MAX_DB]."""
    values_filled = _fill_missing_values(values)
    values_smooth = _fractional_octave_smooth(values_filled, freqs, FRACTIONAL_OCTAVE)
    interp_angles, interp_freqs, interp_values = _interpolate_heatmap_grid(
        angles, freqs, values_smooth, ANGLE_SAMPLES, FREQ_SAMPLES
    )
    return interp_angles, interp_freqs, np.clip(interp_values, MIN_DB, MAX_DB)


def _fill_missing_values(values):
    filled = np.array(values, dtype=float, copy=True)

    # Fill missing values along angle for each frequency.
    for col in range(filled.shape[1]):
        y = filled[:, col]
        finite = np.isfinite(y)
        if np.all(finite):
            continue
        if np.count_nonzero(finite) == 0:
            continue
        x = np.arange(y.size)
        if np.count_nonzero(finite) == 1:
            filled[:, col] = y[finite][0]
        else:
            filled[:, col] = np.interp(x, x[finite], y[finite])

    # Fill remaining gaps across frequency.
    for row in range(filled.shape[0]):
        y = filled[row, :]
        finite = np.isfinite(y)
        if np.all(finite):
            continue
        if np.count_nonzero(finite) == 0:
            filled[row, :] = MIN_DB
            continue
        x = np.arange(y.size)
        if np.count_nonzero(finite) == 1:
            filled[row, :] = y[finite][0]
        else:
            filled[row, :] = np.interp(x, x[finite], y[finite])

    filled[~np.isfinite(filled)] = MIN_DB
    return filled


def _fractional_octave_smooth(values, freqs, fraction):
    if fraction is None or fraction <= 0 or len(freqs) < 2:
        return values

    log2_freqs = np.log2(freqs)
    half_band = 1.0 / (2.0 * float(fraction))
    smoothed = np.empty_like(values)
    for i in range(freqs.size):
        mask = np.abs(log2_freqs - log2_freqs[i]) <= half_band
        smoothed[:, i] = np.mean(values[:, mask], axis=1)
    return smoothed


def _interpolate_heatmap_grid(angles, freqs, values, angle_samples, freq_samples):
    if len(angles) < 2 or len(freqs) < 2:
        return angles, freqs, values

    target_angles = np.linspace(
        float(angles[0]), float(angles[-1]), max(int(angle_samples), len(angles))
    )
    log_freqs = np.log10(freqs)
    target_log_freqs = np.linspace(
        float(log_freqs[0]),
        float(log_freqs[-1]),
        max(int(freq_samples), len(freqs)),
    )

    angle_interp = np.empty((target_angles.size, freqs.size), dtype=float)
    for i in range(freqs.size):
        angle_interp[:, i] = np.interp(target_angles, angles, values[:, i])

    full_interp = np.empty((target_angles.size, target_log_freqs.size), dtype=float)
    for j in range(target_angles.size):
        full_interp[j, :] = np.interp(target_log_freqs, log_freqs, angle_interp[j, :])

    return target_angles, np.power(10.0, target_log_freqs), full_interp


def check_symmetry(h_values, v_values):
    """Check if H and V patterns are effectively identical (within 1% of max)."""
    if h_values is None or v_values is None:
        return False
    if h_values.shape != v_values.shape:
        return False

    finite = np.isfinite(h_values) & np.isfinite(v_values)
    if not np.any(finite):
        return False

    h_ref = np.nanmax(np.abs(h_values[finite]))
    v_ref = np.nanmax(np.abs(v_values[finite]))
    scale = max(h_ref, v_ref, 1e-9)
    rel_diff = np.nanmax(np.abs(h_values[finite] - v_values[finite])) / scale
    return rel_diff < 0.01


# ---------------------------------------------------------------------------
# Single heatmap renderer
# ---------------------------------------------------------------------------

def render_single_heatmap(
    ax, freqs, angles, values, title, reference_level=-6.0, mesh_valid_hz=None
):
    """Render a single directivity heatmap onto the given matplotlib Axes.

    The freqs / angles / values arrays should already have been passed
    through `prepare_heatmap_data` for canonical look. Callers that want
    to overlay their own contours (e.g. BIGMEH baseline-overlay) call
    this and then add contours to the returned axes themselves.

    ``mesh_valid_hz`` draws a vertical marker (and shades the band above it)
    at the highest frequency the mesh resolves; results to the right are
    under-resolved and increasingly inaccurate.
    """
    ax.set_facecolor(AXES_BG)

    log_freqs = np.log10(freqs)
    if len(log_freqs) > 1:
        d_log = np.diff(log_freqs)
        freq_edges = np.zeros(len(freqs) + 1)
        freq_edges[0] = 10 ** (log_freqs[0] - d_log[0] / 2)
        freq_edges[-1] = 10 ** (log_freqs[-1] + d_log[-1] / 2)
        for i in range(1, len(freqs)):
            freq_edges[i] = 10 ** ((log_freqs[i - 1] + log_freqs[i]) / 2)
    else:
        freq_edges = np.array([freqs[0] * 0.9, freqs[0] * 1.1])

    if len(angles) > 1:
        d_ang = np.diff(angles)
        angle_edges = np.zeros(len(angles) + 1)
        angle_edges[0] = angles[0] - d_ang[0] / 2
        angle_edges[-1] = angles[-1] + d_ang[-1] / 2
        for i in range(1, len(angles)):
            angle_edges[i] = (angles[i - 1] + angles[i]) / 2
    else:
        angle_edges = np.array([angles[0] - 1, angles[0] + 1])

    mesh = ax.pcolormesh(
        freq_edges,
        angle_edges,
        values,
        cmap=HEATMAP_CMAP,
        vmin=MIN_DB,
        vmax=MAX_DB,
        shading="flat",
    )

    X, Y = np.meshgrid(freqs, angles)
    contour_levels = [-24, -18, -12, -9, -6, -3]
    try:
        contour = ax.contour(
            X,
            Y,
            values,
            levels=contour_levels,
            colors=GRID_COLOR,
            linewidths=0.6,
            alpha=0.45,
        )
        for collection in contour.collections:
            collection.set_path_effects([
                pe.Stroke(linewidth=1.2, foreground=CONTOUR_OUTLINE, alpha=0.8),
                pe.Normal(),
            ])
    except Exception:
        pass

    try:
        ref_contour = ax.contour(
            X,
            Y,
            values,
            levels=[reference_level],
            colors=REFERENCE_CONTOUR_COLOR,
            linewidths=1.5,
        )
        for collection in ref_contour.collections:
            collection.set_path_effects([
                pe.Stroke(linewidth=2.6, foreground=CONTOUR_OUTLINE, alpha=0.85),
                pe.Normal(),
            ])
    except Exception:
        ref_contour = None

    ax.set_xscale("log")
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(angles[0], angles[-1])

    detailed_ticks = preferred_frequency_ticks(freqs[0], freqs[-1])
    if detailed_ticks:
        ax.set_xticks(detailed_ticks)
    else:
        detailed_ticks = log_grid_lines(freqs[0], freqs[-1])
        ax.set_xticks(detailed_ticks)

    for freq in detailed_ticks:
        ax.axvline(freq, color=GRID_COLOR, alpha=PRIMARY_GRID_ALPHA, linewidth=0.7)

    for freq in log_grid_lines(freqs[0], freqs[-1]):
        if not contains_frequency(detailed_ticks, freq):
            ax.axvline(freq, color=GRID_COLOR, alpha=SECONDARY_GRID_ALPHA, linewidth=0.5)

    angle_range = angles[-1] - angles[0]
    if angle_range > 120:
        angle_step = 30
    elif angle_range > 60:
        angle_step = 15
    else:
        angle_step = 10
    start = np.ceil(angles[0] / angle_step) * angle_step
    for a in np.arange(start, angles[-1] + angle_step * 0.5, angle_step):
        if angles[0] < a < angles[-1]:
            ax.axhline(a, color=GRID_COLOR, alpha=SECONDARY_GRID_ALPHA, linewidth=0.5)

    mesh_valid_handle = None
    if mesh_valid_hz and freqs[0] < float(mesh_valid_hz) < freqs[-1]:
        ax.axvspan(float(mesh_valid_hz), freqs[-1], color=MESH_LIMIT_COLOR, alpha=0.12, zorder=2)
        mesh_valid_handle = ax.axvline(
            float(mesh_valid_hz),
            color=MESH_LIMIT_COLOR,
            linestyle="--",
            linewidth=2.0,
            label=f"mesh-valid {float(mesh_valid_hz):.0f} Hz",
            zorder=3,
        )

    ax.xaxis.set_major_formatter(FuncFormatter(freq_formatter))
    ax.set_xlabel("Frequency [Hz]", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Angle [deg]", color=TEXT_COLOR, fontsize=11)
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="600", pad=8)
    ax.tick_params(colors=TICK_COLOR, labelsize=8)

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("dB", color=TEXT_COLOR, fontsize=10)
    cbar.ax.tick_params(colors=TICK_COLOR, labelsize=9)
    cbar.outline.set_edgecolor(SPINE_COLOR)

    legend_handles = []
    if ref_contour is not None:
        from matplotlib.lines import Line2D

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=REFERENCE_CONTOUR_COLOR,
                linewidth=1.5,
                label=f"ref @ {reference_level:g} dB",
            )
        )
    if mesh_valid_handle is not None:
        legend_handles.append(mesh_valid_handle)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            facecolor=AXES_BG,
            edgecolor=SPINE_COLOR,
            labelcolor=TEXT_COLOR,
            framealpha=0.85,
        )


# ---------------------------------------------------------------------------
# Multi-plane composite renderer (replaces WG's render_directivity_plot)
# ---------------------------------------------------------------------------

def _plane_title(key):
    if key == "horizontal":
        return "H Normalized Directivity"
    if key == "vertical":
        return "V Normalized Directivity"
    if key == "diagonal":
        return "D Normalized Directivity"
    return "Normalized Directivity"


def _build_planes_from_legacy(frequencies, directivity):
    """Convert WG-shaped ``{"horizontal": [[[angle, db], ...], ...], ...}``
    payload into the list-of-dicts shape used by the composite renderer."""
    freqs = np.array(frequencies, dtype=float)
    if freqs.size == 0:
        return []

    planes = []
    for key in ("horizontal", "vertical", "diagonal"):
        patterns = directivity.get(key, [])
        if not patterns:
            continue
        angles_raw, freqs_raw, values_raw = build_grid_from_legacy(freqs, patterns)
        if values_raw is None:
            continue
        angles, plane_freqs, values = prepare_heatmap_data(angles_raw, freqs_raw, values_raw)
        planes.append({
            "key": key,
            "angles": angles,
            "freqs": plane_freqs,
            "values": values,
            "values_raw": values_raw,
        })
    return planes


def _build_figure_from_planes(planes, reference_level=-6.0, mesh_valid_hz=None):
    """Render planes into a matplotlib figure and return it.

    Collapses to a single H=V panel when both planes match within 1%.
    ``mesh_valid_hz`` overlays the mesh-valid frequency marker on each panel.
    """
    by_key = {entry["key"]: entry for entry in planes}
    has_only_hv = set(by_key.keys()) == {"horizontal", "vertical"}
    symmetric = has_only_hv and check_symmetry(
        by_key["horizontal"]["values_raw"],
        by_key["vertical"]["values_raw"],
    )

    if symmetric:
        fig, axes = plt.subplots(1, 1, figsize=(11, 5))
        axes = [axes]
        titles = ["Directivity (H = V, Symmetric)"]
        datasets = [(
            by_key["horizontal"]["freqs"],
            by_key["horizontal"]["angles"],
            by_key["horizontal"]["values"],
        )]
    else:
        plane_count = len(planes)
        fig_height = 5 if plane_count == 1 else (4 * plane_count)
        fig, axes = plt.subplots(plane_count, 1, figsize=(11, fig_height))
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        else:
            axes = list(np.atleast_1d(axes))
        titles = [_plane_title(entry["key"]) for entry in planes]
        datasets = [(entry["freqs"], entry["angles"], entry["values"]) for entry in planes]

    fig.patch.set_facecolor(FIGURE_BG)
    for ax, title, (plot_freqs, plot_angles, plot_values) in zip(axes, titles, datasets):
        render_single_heatmap(
            ax,
            plot_freqs,
            plot_angles,
            plot_values,
            title,
            reference_level=reference_level,
            mesh_valid_hz=mesh_valid_hz,
        )

    fig.tight_layout(pad=1.5)
    return fig


def directivity_heatmap_from_legacy_dict(
    frequencies,
    directivity,
    dpi=150,
    reference_level=-6.0,
):
    """Render directivity heatmap(s) and return base64-encoded PNG (no prefix).

    Accepts the legacy WG dict shape:
        directivity = {
            "horizontal": [[[angle_deg, dB], ...], ...],  # per frequency
            "vertical":   [...],
            "diagonal":   [...],
        }

    This is the byte-equivalent replacement for WG's former
    ``render_directivity_plot`` entry point.

    Returns None when there are no plottable patterns.
    """
    planes = _build_planes_from_legacy(frequencies, directivity)
    if not planes:
        return None

    fig = _build_figure_from_planes(planes, reference_level=reference_level)

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def save_directivity_plot(
    output_path,
    frequencies,
    directivity,
    dpi=150,
    reference_level=-6.0,
    mesh_valid_hz=None,
):
    """Render directivity heatmap(s) and save to ``output_path`` (PNG).

    ``mesh_valid_hz`` overlays the mesh-valid frequency marker; results above
    it are under-resolved and increasingly inaccurate.
    """
    planes = _build_planes_from_legacy(frequencies, directivity)
    if not planes:
        return None

    fig = _build_figure_from_planes(
        planes, reference_level=reference_level, mesh_valid_hz=mesh_valid_hz
    )

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


# ---------------------------------------------------------------------------
# Numpy-array signature (preferred for new callers)
# ---------------------------------------------------------------------------

def directivity_heatmap_b64(
    freq_hz,
    angles_deg,
    spl_db,
    *,
    title=None,
    reference_level=-6.0,
    dpi=150,
):
    """Render a single directivity heatmap and return base64 PNG (no prefix).

    Args:
        freq_hz: 1D array, ``(n_freq,)``.
        angles_deg: 1D array, ``(n_angle,)``.
        spl_db: 2D array, ``(n_angle, n_freq)``, already normalized.
        title: Plot title. Defaults to "Normalized Directivity".
        reference_level: dB level for prominent contour (default -6).

    Returns:
        Base64-encoded PNG string (without data URI prefix), or None when
        the inputs are empty.
    """
    freqs = np.asarray(freq_hz, dtype=float)
    angles = np.asarray(angles_deg, dtype=float)
    values_raw = np.asarray(spl_db, dtype=float)
    if freqs.size == 0 or angles.size == 0:
        return None

    angles_p, freqs_p, values_p = prepare_heatmap_data(angles, freqs, values_raw)

    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    fig.patch.set_facecolor(FIGURE_BG)
    render_single_heatmap(
        ax,
        freqs_p,
        angles_p,
        values_p,
        title or "Normalized Directivity",
        reference_level=reference_level,
    )
    fig.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
