"""Fixed-frequency polar line plots for HornLab BEM results."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .style import (
    AXES_BG,
    FIGURE_BG,
    GRID_COLOR,
    SPINE_COLOR,
    TEXT_COLOR,
    TICK_COLOR,
)


def _field_to_db(values):
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        return 20.0 * np.log10(np.abs(arr).astype(float) + 1e-30)
    return np.asarray(arr, dtype=float)


def prepare_polar_line_data(
    frequencies_hz,
    angles_deg,
    values,
    target_frequencies_hz,
    *,
    normalize=True,
    reference_angle_deg=0.0,
    angle_min_deg=None,
    angle_max_deg=None,
):
    """Return selected fixed-frequency polar traces as dB-vs-angle rows.

    ``values`` is shaped ``(n_freq, n_angle)`` and may be complex pressure or
    already-dB data. Each requested target frequency resolves to the nearest
    available frequency in ``frequencies_hz``.
    """
    freqs = np.asarray(frequencies_hz, dtype=float)
    angles = np.asarray(angles_deg, dtype=float)
    db = _field_to_db(values)
    targets = np.asarray(target_frequencies_hz, dtype=float)

    if freqs.ndim != 1 or angles.ndim != 1:
        raise ValueError("frequencies_hz and angles_deg must be 1D arrays")
    if db.shape != (freqs.size, angles.size):
        raise ValueError(
            "values must have shape (n_freq, n_angle); "
            f"got {db.shape}, expected {(freqs.size, angles.size)}"
        )
    if targets.size == 0:
        raise ValueError("target_frequencies_hz must contain at least one frequency")

    angle_mask = np.ones(angles.shape, dtype=bool)
    if angle_min_deg is not None:
        angle_mask &= angles >= float(angle_min_deg)
    if angle_max_deg is not None:
        angle_mask &= angles <= float(angle_max_deg)
    if not np.any(angle_mask):
        raise ValueError("angle range contains no samples")

    selected_angles = angles[angle_mask]
    ref_idx = int(np.argmin(np.abs(angles - float(reference_angle_deg))))
    rows = []
    selected_freqs = []
    for target in targets:
        fi = int(np.argmin(np.abs(freqs - target)))
        row = np.array(db[fi], dtype=float, copy=True)
        if normalize:
            row -= row[ref_idx]
        rows.append(row[angle_mask])
        selected_freqs.append(float(freqs[fi]))

    return selected_angles, np.asarray(selected_freqs), np.vstack(rows)


def _render_polar_line_figure(
    angles,
    selected_freqs,
    rows,
    *,
    title=None,
    ylabel=None,
    ylim=None,
    xlim=None,
):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.patch.set_facecolor(FIGURE_BG)
    ax.set_facecolor(AXES_BG)

    cmap = plt.get_cmap("viridis", max(len(selected_freqs), 2))
    for idx, (freq, row) in enumerate(zip(selected_freqs, rows)):
        ax.plot(
            angles,
            row,
            lw=2.0,
            color=cmap(idx),
            label=f"{freq:g} Hz",
        )

    ax.axhline(0.0, color=GRID_COLOR, lw=0.8, alpha=0.55)
    ax.grid(True, which="both", color=GRID_COLOR, alpha=0.28, linewidth=0.7)
    ax.set_xlabel("Angle [deg]", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel(ylabel or "dB relative to reference angle", color=TEXT_COLOR, fontsize=11)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="600", pad=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(float(angles[0]), float(angles[-1]))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(colors=TICK_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.legend(
        loc="best",
        fontsize=9,
        facecolor=AXES_BG,
        edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR,
        framealpha=0.88,
    )
    fig.tight_layout()
    return fig


def polar_line_b64(
    frequencies_hz,
    angles_deg,
    values,
    target_frequencies_hz,
    *,
    normalize=True,
    reference_angle_deg=0.0,
    angle_min_deg=None,
    angle_max_deg=None,
    title=None,
    ylabel=None,
    ylim=None,
    xlim=None,
    dpi=150,
):
    """Render selected fixed-frequency polar traces and return base64 PNG."""
    angles, selected_freqs, rows = prepare_polar_line_data(
        frequencies_hz,
        angles_deg,
        values,
        target_frequencies_hz,
        normalize=normalize,
        reference_angle_deg=reference_angle_deg,
        angle_min_deg=angle_min_deg,
        angle_max_deg=angle_max_deg,
    )
    fig = _render_polar_line_figure(
        angles,
        selected_freqs,
        rows,
        title=title,
        ylabel=ylabel,
        ylim=ylim,
        xlim=xlim,
    )
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


def save_polar_line_plot(
    output_path,
    frequencies_hz,
    angles_deg,
    values,
    target_frequencies_hz,
    *,
    normalize=True,
    reference_angle_deg=0.0,
    angle_min_deg=None,
    angle_max_deg=None,
    title=None,
    ylabel=None,
    ylim=None,
    xlim=None,
    dpi=150,
):
    """Render selected fixed-frequency polar traces and save to PNG."""
    angles, selected_freqs, rows = prepare_polar_line_data(
        frequencies_hz,
        angles_deg,
        values,
        target_frequencies_hz,
        normalize=normalize,
        reference_angle_deg=reference_angle_deg,
        angle_min_deg=angle_min_deg,
        angle_max_deg=angle_max_deg,
    )
    fig = _render_polar_line_figure(
        angles,
        selected_freqs,
        rows,
        title=title,
        ylabel=ylabel,
        ylim=ylim,
        xlim=xlim,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close(fig)
    return out
