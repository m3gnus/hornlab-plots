#!/usr/bin/env python3
"""Canonical analyses unlocked by complex-pressure directivity data.

Reads a result.json produced by `run_rectangular_horn_sim.py` (which now
defaults to saving complex pressure) or an NPZ in the BIGMEH cabinet
format, and renders one or more diagnostic plots.

Usage:
    # Single-source analyses (plots 1, 2, 4, 5, 6, 7, 8, 9):
    python analyze_complex_directivity.py SRC.json --outdir OUT --plot all

    # Pair-of-sources analyses (plots 3, 10):
    python analyze_complex_directivity.py SRC1 --input2 SRC2 \\
        --outdir OUT --plot pair

Plot keys (matches the BIGMEH-introduction post research list):
    1  group_delay         Group delay vs frequency (cycles), per axis.
    2  impulse_response    Time-domain impulse response per axis (IFFT).
    4  phase_polar         Phase-vs-angle sonogram per axis.
    5  acoustic_centre     Acoustic centre estimate vs frequency.
    6  di_coherent         H-plane vs V-plane DI proxy comparison.
    7  phase_sonogram      Phase response vs (freq, angle), normalized to on-axis.
    8  real_imag           Real/Imag heatmaps (diagnostic).
    9  hv_phase_track      H vs V on-axis phase tracking error.
    3  pair_coherent       Two-source coherent vs power-summed on-axis sum.
    10 pair_phase_diff     Two-source real-phase vs flat-phase summation diff.

Inputs accepted:
- WG result.json with `directivity_complex.{horizontal,vertical}` per
  freq → list of [angle_deg, real_Pa, imag_Pa].
- BIGMEH NPZ with `freqs`, `theta_deg`, `field_h`, `field_v` (complex).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

try:
    from ._grid import freq_formatter
    from .style import apply_theme_overrides, theme_grid_kwargs
except ImportError:  # pragma: no cover - direct ``python complex_analysis.py`` use.
    from hornlab_plots._grid import freq_formatter
    from hornlab_plots.style import apply_theme_overrides, theme_grid_kwargs

_trapz = getattr(np, "trapezoid", None) or np.trapz

C_AIR = 343.0


def _theme():
    return apply_theme_overrides()


def _set_figure_bg(fig):
    fig.patch.set_facecolor(_theme().figure_bg)


def _line_color(index: int) -> str:
    line_colors = _theme().line_colors
    return line_colors[index % len(line_colors)]


def _style_ax(ax, xlabel="", ylabel="", title="", grid=True):
    theme_obj = _theme()
    ax.set_facecolor(theme_obj.axes_bg)
    if xlabel:
        ax.set_xlabel(xlabel, color=theme_obj.text_color, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=theme_obj.text_color, fontsize=10)
    if title:
        ax.set_title(title, color=theme_obj.text_color, fontsize=11, fontweight="600", pad=6)
    ax.tick_params(colors=theme_obj.tick_color, labelsize=8, which="both")
    for spine in ax.spines.values():
        spine.set_color(theme_obj.spine_color)
    if grid:
        # Explicit reads of Theme.rc_params (grid.linestyle/linewidth) so the
        # designed themes' grid styles reach these per-plot ax.grid calls;
        # empty rc_params (hornlab/dark) keep the exact legacy call.
        ax.grid(
            True,
            which="major",
            color=theme_obj.grid_color,
            alpha=theme_obj.grid_alpha,
            **theme_grid_kwargs(theme_obj, linewidth=0.5),
        )
        ax.grid(
            True,
            which="minor",
            color=theme_obj.grid_color,
            alpha=theme_obj.grid_alpha_minor,
            **theme_grid_kwargs(theme_obj, linewidth=0.3),
        )


def _finish_fig(fig, out, suptitle=""):
    theme_obj = _theme()
    if suptitle:
        fig.suptitle(suptitle, color=theme_obj.text_color, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=theme_obj.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return out


def _add_cbar(fig, mappable, ax, label=""):
    theme_obj = _theme()
    cbar = fig.colorbar(mappable, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label(label, color=theme_obj.text_color, fontsize=9)
    cbar.ax.tick_params(colors=theme_obj.tick_color, labelsize=8)
    cbar.outline.set_edgecolor(theme_obj.spine_color)
    return cbar


def _add_legend(ax, **kwargs):
    theme_obj = _theme()
    return ax.legend(
        fontsize=8,
        facecolor=theme_obj.axes_bg,
        edgecolor=theme_obj.spine_color,
        labelcolor=theme_obj.text_color,
        framealpha=0.92,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class ComplexDirectivity:
    """Complex-pressure directivity dataset for one source."""

    freqs: np.ndarray  # (n_freq,) Hz
    theta: np.ndarray  # (n_theta,) deg, 0 = on-axis
    p_h: np.ndarray  # (n_freq, n_theta) complex, horizontal axis
    p_v: Optional[np.ndarray]  # (n_freq, n_theta) complex, vertical axis (or None)
    label: str = ""
    distance_m: float = 2.0


def _patterns_to_complex(
    patterns: List[List[List[float]]], n_freq: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert WG complex patterns, aligning every row by angle value."""
    sample = next((pat for pat in patterns[:n_freq] if pat), None)
    if sample is None:
        raise ValueError("empty pattern list")
    n_theta = len(sample)
    angles = np.array([p[0] for p in sample], dtype=float)
    angle_indices: dict[float, int] = {}
    for index, angle in enumerate(angles):
        angle_f = float(angle)
        if angle_f in angle_indices:
            raise ValueError(
                f"duplicate angle {angle_f:g} in directivity_complex pattern"
            )
        angle_indices[angle_f] = index
    p = np.full((n_freq, n_theta), complex(np.nan, np.nan), dtype=complex)
    for fi, freq_pat in enumerate(patterns[:n_freq]):
        if not freq_pat:
            continue
        if len(freq_pat) != n_theta:
            raise RuntimeError("directivity_complex theta count changes across frequencies")
        seen: set[float] = set()
        for point in freq_pat:
            if point is None or len(point) < 1:
                continue
            angle = float(point[0])
            if angle in seen:
                raise ValueError(
                    f"duplicate angle {angle:g} in directivity_complex pattern "
                    f"at frequency index {fi}"
                )
            seen.add(angle)
            if angle not in angle_indices:
                raise RuntimeError(
                    "directivity_complex theta grid changes across frequencies"
                )
            if (
                point is not None
                and len(point) >= 3
                and point[1] is not None
                and point[2] is not None
            ):
                p[fi, angle_indices[angle]] = complex(point[1], point[2])
    return angles, p


def _json_observation_distance_m(result: Dict) -> float:
    """Read the effective observation distance from WG result metadata."""
    observation = result.get("metadata", {}).get("observation", {})
    for key in ("effective_distance_m", "distance_m", "requested_distance_m"):
        try:
            value = float(observation.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            return value
    return 2.0


def load_directivity(path: Path, label: str = "") -> ComplexDirectivity:
    """Load complex directivity from WG result.json or BIGMEH NPZ."""
    path = Path(path)
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        freqs = np.array(d["frequencies"], dtype=float)
        dc = d.get("directivity_complex")
        if dc is None:
            raise RuntimeError(
                f"{path}: no directivity_complex; re-run with --save-complex "
                "(default) or use a result.json from run_rectangular_horn_sim.py"
            )
        theta_h, p_h = _patterns_to_complex(dc.get("horizontal", []), len(freqs))
        if "vertical" in dc and dc["vertical"]:
            theta_v, p_v = _patterns_to_complex(dc["vertical"], len(freqs))
            if not np.allclose(theta_h, theta_v):
                raise RuntimeError("H and V theta grids differ in result.json")
        else:
            p_v = None
        distance_m = _json_observation_distance_m(d)
        return ComplexDirectivity(freqs, theta_h, p_h, p_v, label or path.stem, distance_m)
    elif path.suffix == ".npz":
        d = np.load(path)
        freqs = np.asarray(d["freqs"], dtype=float)
        theta = np.asarray(d["theta_deg"], dtype=float)
        p_h = np.asarray(d["field_h"], dtype=complex)
        p_v = np.asarray(d["field_v"], dtype=complex) if "field_v" in d.files else None
        return ComplexDirectivity(freqs, theta, p_h, p_v, label or path.stem, 2.0)
    else:
        raise ValueError(f"unknown directivity format: {path.suffix}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_complex(values: np.ndarray) -> np.ndarray:
    return np.isfinite(np.real(values)) & np.isfinite(np.imag(values))


def _propagation_phase(freqs: np.ndarray, distance_m: float) -> np.ndarray:
    """Expected HornLab BEM propagation phase for exp(-i*w*t): exp(+i*k*r)."""
    return +2.0 * np.pi * np.asarray(freqs, dtype=float) * float(distance_m) / C_AIR


def _unwrapped_phase(
    values: np.ndarray,
    freqs: np.ndarray,
    distance_m: float,
    *,
    readd_propagation: bool = True,
) -> np.ndarray:
    """Unwrap sparse log-spaced phase after removing the expected bulk path."""
    values = np.asarray(values, dtype=complex)
    freqs = np.asarray(freqs, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = _finite_complex(values) & np.isfinite(freqs)
    if np.sum(valid) == 0:
        return out

    phase_prop = _propagation_phase(freqs[valid], distance_m)
    # HornLab BEM fields propagate as exp(+i*k*r), so multiplying by
    # exp(-i*k*r) leaves a slow residual phase for unwrap.
    residual = values[valid] * np.exp(-1j * phase_prop)
    residual_phase = np.unwrap(np.angle(residual))
    out[valid] = residual_phase + phase_prop if readd_propagation else residual_phase
    return out


def _onaxis_phase(p: np.ndarray, freqs: np.ndarray, distance_m: float,
                   theta: Optional[np.ndarray] = None) -> np.ndarray:
    """Unwrapped on-axis phase vs frequency using propagation de-embedding."""
    ai = int(np.argmin(np.abs(theta))) if theta is not None else 0
    return _unwrapped_phase(p[:, ai], freqs, distance_m, readd_propagation=True)


def _rotate_to_onaxis_phase_zero(p: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotate every angle so the nearest on-axis phase is zero per frequency."""
    onaxis_idx = int(np.argmin(np.abs(theta)))
    rot = np.exp(-1j * np.angle(p[:, onaxis_idx:onaxis_idx + 1]))
    return p * rot


def _band_taper(n: int) -> np.ndarray:
    if n < 4:
        return np.ones(n)
    return np.hanning(n)


def _log_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(freq_formatter))


# ---------------------------------------------------------------------------
# Plot 1: Group delay (in cycles)
# ---------------------------------------------------------------------------

def plot_group_delay(d: ComplexDirectivity, out: Path,
                     angles_deg: Tuple[float, ...] = (0.0, 30.0, 60.0)) -> Path:
    """Group delay vs frequency, in cycles. GD_cycles(f) = f * GD_seconds(f).

    For the HornLab BEM exp(-i*w*t) convention, phase grows as +k*r for outgoing
    waves, so GD_seconds(f) = +dphi/d(omega) = +dphi/(2*pi*df).
    GD_cycles(f) = f * GD_seconds(f).
    """
    fig, axes = plt.subplots(1, 2 if d.p_v is not None else 1,
                             figsize=(12, 5), sharey=True)
    _set_figure_bg(fig)
    if d.p_v is None:
        axes = [axes]
    for ax, p, name in zip(axes,
                            [d.p_h] + ([d.p_v] if d.p_v is not None else []),
                            ["H"] + (["V"] if d.p_v is not None else [])):
        omega = 2 * np.pi * d.freqs
        for i, a in enumerate(angles_deg):
            ai = int(np.argmin(np.abs(d.theta - a)))
            phi = _unwrapped_phase(
                p[:, ai], d.freqs, d.distance_m, readd_propagation=True
            )
            valid = np.isfinite(phi)
            if np.sum(valid) < 2:
                continue
            gd_s = np.full_like(d.freqs, np.nan, dtype=float)
            gd_s[valid] = np.gradient(phi[valid], omega[valid])
            gd_cycles = gd_s * d.freqs
            ax.plot(d.freqs[valid], gd_cycles[valid], lw=1.6,
                    color=_line_color(i),
                    label=f"{a:.0f}°")
        ax.axhline(0, color=_theme().tick_color, lw=0.5, alpha=0.4)
        _log_axis(ax)
        if ax is axes[0]:
            _style_ax(ax, xlabel="Frequency", ylabel="Group delay (cycles)",
                      title=f"{name} axis")
            _add_legend(ax, loc="best")
        else:
            _style_ax(ax, xlabel="Frequency", title=f"{name} axis")
    return _finish_fig(fig, out, suptitle=f"{d.label} — Group Delay")


# ---------------------------------------------------------------------------
# Plot 2: Impulse response per axis (IFFT)
# ---------------------------------------------------------------------------

def plot_impulse_response(d: ComplexDirectivity, out: Path,
                          angles_deg: Tuple[float, ...] = (0.0, 30.0, 60.0, 90.0),
                          window_ms: float = 8.0) -> Path:
    """Time-domain impulse response per axis at a few angles.

    Reinterprets the freq-sweep H(f) as a half-spectrum, builds a Hermitian
    extension (so the IFFT is real), and renders the resulting impulse
    response. Useful for spotting throat ringing and mouth reflections.
    """
    finite_freq_indices = np.flatnonzero(np.isfinite(d.freqs))
    frequency_order = finite_freq_indices[
        np.argsort(d.freqs[finite_freq_indices], kind="stable")
    ]
    finite_freqs = d.freqs[frequency_order]
    f_min = float(finite_freqs[0])
    f_max = float(finite_freqs[-1])
    df = max(f_min / 4.0, 25.0)
    fs = 4 * f_max
    n = 1
    while n * df < fs:
        n *= 2
    n = max(n, 256)
    f_uniform = np.arange(n // 2 + 1) * (fs / n)
    t_ms = np.arange(n) / fs * 1000.0
    time_mask = t_ms <= window_ms

    fig, axes = plt.subplots(1, 2 if d.p_v is not None else 1,
                             figsize=(12, 5), sharey=True)
    _set_figure_bg(fig)
    if d.p_v is None:
        axes = [axes]

    sources = [(d.p_h, "H")] + ([(d.p_v, "V")] if d.p_v is not None else [])
    for ax, (p, name) in zip(axes, sources):
        for i, a in enumerate(angles_deg):
            ai = int(np.argmin(np.abs(d.theta - a)))
            p_sorted = p[frequency_order, ai]
            valid = _finite_complex(p_sorted)
            if np.sum(valid) < 2:
                continue
            f_valid = finite_freqs[valid]
            p_valid = p_sorted[valid]
            mag_db = 20 * np.log10(np.abs(p_valid) + 1e-30)
            phase = _unwrapped_phase(
                p_valid, f_valid, d.distance_m, readd_propagation=False
            )
            in_band = (f_uniform >= f_valid[0]) & (f_uniform <= f_valid[-1])
            if np.sum(in_band) < 2:
                continue
            log_valid = np.log(f_valid)
            log_uniform = np.log(f_uniform[in_band])
            mag_interp = np.interp(log_uniform, log_valid, mag_db)
            phase_interp = np.interp(log_uniform, log_valid, phase)
            spectrum_half = np.zeros(len(f_uniform), dtype=complex)
            # HornLab BEM uses exp(-i*w*t); NumPy irfft assumes exp(+i*w*t),
            # so conjugate (negate phase) to match.
            spectrum_half[in_band] = (
                10.0 ** (mag_interp / 20.0)
                * np.exp(-1j * phase_interp)
                * _band_taper(int(np.sum(in_band)))
            )
            ir = np.fft.irfft(spectrum_half, n=n) * n
            ir = ir / max(np.max(np.abs(ir)), 1e-30)
            ax.plot(t_ms[time_mask], ir[time_mask], lw=1.2,
                    color=_line_color(i),
                    label=f"{a:.0f}°")
        ax.axhline(0, color=_theme().tick_color, lw=0.5, alpha=0.4)
        if ax is axes[0]:
            _style_ax(ax, xlabel="Time (ms)", ylabel="Normalized amplitude",
                      title=f"{name} axis")
            _add_legend(ax, loc="best")
        else:
            _style_ax(ax, xlabel="Time (ms)", title=f"{name} axis")
    return _finish_fig(fig, out, suptitle=f"{d.label} — Impulse Response")


# ---------------------------------------------------------------------------
# Plot 4 & 7: Phase sonogram (phase vs angle and freq)
# Plot 4 is the polar-style version, 7 is the heatmap rectangular version.
# We render both views for completeness.
# ---------------------------------------------------------------------------

def plot_phase_sonogram(d: ComplexDirectivity, out: Path) -> Path:
    """Heatmap (angle x freq) of unwrapped phase relative to on-axis."""
    sources = [(d.p_h, "H")] + ([(d.p_v, "V")] if d.p_v is not None else [])
    fig, axes = plt.subplots(len(sources), 1, figsize=(12, 4.5 * len(sources)),
                             sharex=True)
    _set_figure_bg(fig)
    if len(sources) == 1:
        axes = [axes]
    onaxis_idx = int(np.argmin(np.abs(d.theta)))
    for ax, (p, name) in zip(axes, sources):
        phi_rel = np.unwrap(np.angle(p) - np.angle(p[:, onaxis_idx:onaxis_idx + 1]), axis=1)
        phi_deg = np.rad2deg(phi_rel)
        vlim = 360.0
        im = ax.pcolormesh(d.freqs, d.theta, phi_deg.T,
                           cmap=_theme().cyclic_cmap, vmin=-vlim, vmax=vlim, shading="auto")
        _log_axis(ax)
        _style_ax(ax, ylabel="Angle (°)", title=f"{name} axis", grid=False)
        _add_cbar(fig, im, ax, label="Phase rel. on-axis (°)")
    axes[-1].set_xlabel("Frequency", color=_theme().text_color, fontsize=10)
    return _finish_fig(fig, out, suptitle=f"{d.label} — Phase Sonogram")


def plot_phase_polar(d: ComplexDirectivity, out: Path,
                     freqs_to_show: Optional[Tuple[float, ...]] = None) -> Path:
    """Polar plot of phase relative to on-axis at a few representative freqs."""
    if freqs_to_show is None:
        n = 4
        idxs = np.linspace(0, len(d.freqs) - 1, n).astype(int)
        freqs_to_show = tuple(float(d.freqs[i]) for i in idxs)

    fig, axes = plt.subplots(1, len(freqs_to_show),
                             figsize=(3.5 * len(freqs_to_show), 4.5),
                             subplot_kw={"projection": "polar"})
    _set_figure_bg(fig)
    if len(freqs_to_show) == 1:
        axes = [axes]
    onaxis_idx = int(np.argmin(np.abs(d.theta)))
    for ax, f_target in zip(axes, freqs_to_show):
        theme_obj = _theme()
        ax.set_facecolor(theme_obj.axes_bg)
        idx = int(np.argmin(np.abs(d.freqs - f_target)))
        for p, name, color in (
            [(d.p_h, "H", _line_color(0))] +
            ([(d.p_v, "V", _line_color(1))] if d.p_v is not None else [])
        ):
            phi = np.unwrap(np.angle(p[idx]) - np.angle(p[idx, onaxis_idx]))
            phi_deg = np.rad2deg(phi)
            t_full = np.concatenate([-d.theta[::-1], d.theta])
            phi_full = np.concatenate([phi_deg[::-1], phi_deg])
            ax.plot(np.deg2rad(t_full), phi_full, lw=1.6, color=color, label=name)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetalim(-np.pi, np.pi)
        freq_hz = d.freqs[idx]
        freq_label = freq_formatter(freq_hz, None) if freq_hz >= 1000 else f"{freq_hz:.0f}"
        ax.set_title(f"{freq_label} Hz", color=theme_obj.text_color, fontsize=11,
                     fontweight="bold", pad=12)
        ax.tick_params(colors=theme_obj.tick_color, labelsize=7)
        ax.grid(
            True,
            color=theme_obj.grid_color,
            alpha=theme_obj.grid_alpha,
            **theme_grid_kwargs(theme_obj, linewidth=0.5),
        )
        ax.spines["polar"].set_color(theme_obj.spine_color)
        _add_legend(ax, loc="upper left", bbox_to_anchor=(-0.15, 1.08))
    return _finish_fig(fig, out, suptitle=f"{d.label} — Phase Polar")


# ---------------------------------------------------------------------------
# Plot 5: Acoustic centre estimate vs frequency
# ---------------------------------------------------------------------------

def plot_acoustic_centre(d: ComplexDirectivity, out: Path,
                         window_octaves: float = 1.0 / 3.0) -> Path:
    """Estimate acoustic centre offset (m) along the firing axis from the
    on-axis phase slope.

    HornLab BEM uses exp(-i*w*t) time convention, so outgoing propagation is
    exp(+i*k*r). We remove that expected observation-distance phase before
    unwrap, re-add it for the slope fit, then report path_length - distance_m.
    Positive offset means the apparent source is farther than the measurement
    origin; negative means closer.
    """
    sources = [(d.p_h, "H")] + ([(d.p_v, "V")] if d.p_v is not None else [])
    fig, ax = plt.subplots(figsize=(11, 5))
    _set_figure_bg(fig)
    log_f = np.log2(d.freqs)
    half = window_octaves / 2.0
    min_points = 5
    for si, (p, name) in enumerate(sources):
        phi = _onaxis_phase(p, d.freqs, d.distance_m, d.theta)
        valid_all = np.isfinite(phi) & np.isfinite(d.freqs)
        offsets = np.full_like(d.freqs, np.nan)
        for i, lf in enumerate(log_f):
            mask = valid_all & (log_f >= lf - half) & (log_f <= lf + half)
            if mask.sum() < min_points:
                valid_idx = np.where(valid_all)[0]
                if valid_idx.size < 2:
                    continue
                idx_sorted = valid_idx[np.argsort(np.abs(log_f[valid_idx] - lf))]
                pick = idx_sorted[:min(min_points, valid_idx.size)]
                xs = d.freqs[pick]
                ys = phi[pick]
            else:
                xs = d.freqs[mask]
                ys = phi[mask]
            if len(xs) < 2:
                continue
            slope, _ = np.polyfit(xs, ys, 1)
            path_length_m = C_AIR / (2 * np.pi) * slope
            offsets[i] = path_length_m - d.distance_m
        if np.all(np.isnan(offsets)):
            continue
        ax.plot(d.freqs, offsets * 1000.0, lw=1.6,
                color=_line_color(si),
                label=f"{name} on-axis")
    ax.axhline(0, color=_theme().tick_color, lw=0.5, alpha=0.4)
    _log_axis(ax)
    _style_ax(ax, xlabel="Frequency",
              ylabel="Acoustic-centre offset (mm, +ve = farther)",
              title=f"Offset from {d.distance_m:.2f} m origin "
                    f"({window_octaves:.2f} oct window)")
    _add_legend(ax)
    return _finish_fig(fig, out, suptitle=f"{d.label} — Acoustic Centre")


# ---------------------------------------------------------------------------
# Plot 6: H-plane vs V-plane DI proxy
# ---------------------------------------------------------------------------

def _di_from_magnitude(p: np.ndarray, theta_deg: np.ndarray,
                       hemisphere: bool = True) -> np.ndarray:
    """Plane-derived DI proxy from |p|^2.

    Assumes the selected polar cut can stand in for an axisymmetric sphere:
    mean power = integral(|p(theta)|^2 sin(theta) dtheta) /
    integral(sin(theta) dtheta). Comparing H and V exposes how weak that
    assumption is for asymmetric horns.
    """
    theta = np.deg2rad(theta_deg)
    sin_t = np.sin(theta)
    p2 = np.abs(p)
    np.square(p2, out=p2)
    onaxis_idx = int(np.argmin(np.abs(theta_deg)))
    on_axis = p2[:, onaxis_idx].copy()
    p2 = p2.astype(np.result_type(p2.dtype, sin_t.dtype), copy=False)
    np.multiply(p2, sin_t[None, :], out=p2)
    norm = max(float(_trapz(sin_t, theta)), 1e-30)
    mean = _trapz(p2, theta, axis=1) / norm
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10((on_axis + 1e-30) / (mean + 1e-30))


def plot_di_coherent_vs_mag(d: ComplexDirectivity, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    _set_figure_bg(fig)
    di_h = _di_from_magnitude(d.p_h, d.theta)
    valid_h = np.isfinite(di_h)
    ax.plot(d.freqs[valid_h], di_h[valid_h], lw=2.0, color=_line_color(0),
            label="H-plane DI proxy")

    if d.p_v is not None:
        di_v = _di_from_magnitude(d.p_v, d.theta)
        valid_v = np.isfinite(di_v)
        ax.plot(d.freqs[valid_v], di_v[valid_v], lw=1.8, color=_line_color(1),
                ls="--", label="V-plane DI proxy")
        both = valid_h & valid_v
        if np.any(both):
            ax.fill_between(d.freqs[both], di_h[both], di_v[both],
                            color=_line_color(0), alpha=_theme().fill_alpha,
                            label="H/V mismatch")

    _log_axis(ax)
    _style_ax(ax, xlabel="Frequency", ylabel="Plane-derived DI proxy (dB)",
              title="Power DI from polar cuts")
    _add_legend(ax)
    return _finish_fig(fig, out, suptitle=f"{d.label} — H/V Plane DI")


# ---------------------------------------------------------------------------
# Plot 8: Real / Imag heatmaps (diagnostic)
# ---------------------------------------------------------------------------

def plot_real_imag_heatmaps(d: ComplexDirectivity, out: Path) -> Path:
    sources = [(d.p_h, "H")] + ([(d.p_v, "V")] if d.p_v is not None else [])
    fig, axes = plt.subplots(len(sources), 2, figsize=(14, 4.5 * len(sources)),
                             sharex=True)
    _set_figure_bg(fig)
    if len(sources) == 1:
        axes = axes.reshape(1, 2)
    onaxis_idx = int(np.argmin(np.abs(d.theta)))
    for row, (p, name) in zip(axes, sources):
        on_axis_mag = np.abs(p[:, onaxis_idx:onaxis_idx + 1]) + 1e-30
        re = np.real(p) / on_axis_mag
        im = np.imag(p) / on_axis_mag
        for ax, data, label in zip(row, [re, im],
                                    [f"{name} Re / |on-axis|",
                                     f"{name} Im / |on-axis|"]):
            finite = np.isfinite(data)
            v = (
                max(float(np.nanpercentile(np.abs(data[finite]), 99.0)), 1e-9)
                if np.any(finite)
                else 1.0
            )
            mesh = ax.pcolormesh(d.freqs, d.theta, data.T,
                                 cmap=_theme().diverging_cmap, vmin=-v, vmax=v, shading="auto")
            _log_axis(ax)
            _style_ax(ax, ylabel="Angle (°)", title=label, grid=False)
            _add_cbar(fig, mesh, ax)
    axes[-1, 0].set_xlabel("Frequency", color=_theme().text_color, fontsize=10)
    axes[-1, 1].set_xlabel("Frequency", color=_theme().text_color, fontsize=10)
    return _finish_fig(fig, out, suptitle=f"{d.label} — Real / Imag Heatmaps")


# ---------------------------------------------------------------------------
# Plot 9: H vs V on-axis phase tracking error
# ---------------------------------------------------------------------------

def plot_hv_phase_tracking(d: ComplexDirectivity, out: Path) -> Path:
    if d.p_v is None:
        raise RuntimeError("H-vs-V phase tracking requires both axes")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    _set_figure_bg(fig)

    diff_onaxis = (
        _onaxis_phase(d.p_h, d.freqs, d.distance_m, d.theta)
        - _onaxis_phase(d.p_v, d.freqs, d.distance_m, d.theta)
    )
    valid_onaxis = np.isfinite(diff_onaxis)
    diff_onaxis_deg = np.full_like(diff_onaxis, np.nan, dtype=float)
    if np.any(valid_onaxis):
        diff_onaxis_deg[valid_onaxis] = np.rad2deg(np.unwrap(diff_onaxis[valid_onaxis]))
    axes[0].plot(d.freqs[valid_onaxis], diff_onaxis_deg[valid_onaxis],
                 lw=2.0, color=_line_color(0))
    axes[0].axhline(0, color=_theme().tick_color, lw=0.5, alpha=0.4)
    _log_axis(axes[0])
    _style_ax(axes[0], xlabel="Frequency",
              ylabel="φ_H − φ_V on-axis (°)",
              title="On-axis H–V phase error")

    onaxis_idx = int(np.argmin(np.abs(d.theta)))
    phi_h = np.unwrap(np.angle(d.p_h) - np.angle(d.p_h[:, onaxis_idx:onaxis_idx + 1]), axis=1)
    phi_v = np.unwrap(np.angle(d.p_v) - np.angle(d.p_v[:, onaxis_idx:onaxis_idx + 1]), axis=1)
    diff = np.rad2deg(phi_h - phi_v)
    vlim = 90.0
    mesh = axes[1].pcolormesh(d.freqs, d.theta, diff.T,
                              cmap=_theme().diverging_cmap, vmin=-vlim, vmax=vlim, shading="auto")
    _log_axis(axes[1])
    _style_ax(axes[1], xlabel="Frequency",
              ylabel="Angle (°)",
              title="Off-axis H–V relative-phase difference (°)", grid=False)
    _add_cbar(fig, mesh, axes[1], label="°")

    return _finish_fig(fig, out, suptitle=f"{d.label} — H vs V Phase Tracking")


# ---------------------------------------------------------------------------
# Plot 3: Two-source coherent vs power-summed on-axis sum
# ---------------------------------------------------------------------------

def _common_frequency_grid(
    a: ComplexDirectivity,
    b: ComplexDirectivity,
    samples: int = 60,
) -> np.ndarray:
    a_freqs = a.freqs[np.isfinite(a.freqs)]
    b_freqs = b.freqs[np.isfinite(b.freqs)]
    if a_freqs.size == 0 or b_freqs.size == 0:
        raise RuntimeError("pair plots require finite frequency grids")
    f_min = max(float(np.min(a_freqs)), float(np.min(b_freqs)))
    f_max = min(float(np.max(a_freqs)), float(np.max(b_freqs)))
    if not f_min < f_max:
        raise RuntimeError(
            f"pair plots require overlapping frequency ranges; got "
            f"{a.label} [{np.min(a_freqs):.1f}, {np.max(a_freqs):.1f}] Hz and "
            f"{b.label} [{np.min(b_freqs):.1f}, {np.max(b_freqs):.1f}] Hz"
        )
    if np.sum((a.freqs >= f_min) & (a.freqs <= f_max)) < 2 or np.sum(
        (b.freqs >= f_min) & (b.freqs <= f_max)
    ) < 2:
        raise RuntimeError("pair plots require at least two samples per source in the overlap")
    return np.logspace(np.log10(f_min), np.log10(f_max), samples)


def _resample_frequency_onto(
    freqs_target: np.ndarray,
    freqs_src: np.ndarray,
    p_src: np.ndarray,
    distance_m: float,
) -> np.ndarray:
    log_t = np.log(freqs_target)
    finite_freqs = np.isfinite(freqs_src)
    finite_pressure = _finite_complex(p_src)
    propagation_src = _propagation_phase(freqs_src, distance_m)
    propagation_target = _propagation_phase(freqs_target, distance_m)
    n_theta = p_src.shape[1]
    out = np.full((len(freqs_target), n_theta), complex(np.nan, np.nan), dtype=complex)
    for ai in range(n_theta):
        col = p_src[:, ai]
        valid = finite_pressure[:, ai] & finite_freqs
        if np.sum(valid) < 2:
            continue
        f_valid = freqs_src[valid]
        target_mask = (freqs_target >= f_valid[0]) & (freqs_target <= f_valid[-1])
        if np.sum(target_mask) < 1:
            continue
        log_s = np.log(f_valid)
        mag_db = 20 * np.log10(np.abs(col[valid]) + 1e-30)
        phase_residual = np.unwrap(
            np.angle(col[valid] * np.exp(-1j * propagation_src[valid]))
        )
        mag = np.interp(log_t[target_mask], log_s, mag_db)
        phase_residual = np.interp(
            log_t[target_mask], log_s, phase_residual
        )
        phase_total = phase_residual + propagation_target[target_mask]
        out[target_mask, ai] = 10.0 ** (mag / 20.0) * np.exp(1j * phase_total)
    return out


def _matching_angle_index(theta_src: np.ndarray, angle_deg: float) -> int | None:
    theta_src = np.asarray(theta_src, dtype=float)
    exact = np.where(np.isclose(theta_src, angle_deg, atol=1e-9))[0]
    return int(exact[0]) if exact.size else None


def _pressure_at_angle(p: np.ndarray, theta_src: np.ndarray, angle_deg: float) -> np.ndarray:
    theta_src = np.asarray(theta_src, dtype=float)
    angle_index = _matching_angle_index(theta_src, angle_deg)
    if angle_index is not None:
        return p[:, angle_index]
    if theta_src.size < 2:
        return np.full(p.shape[0], complex(np.nan, np.nan), dtype=complex)
    return _resample_theta_onto(p, theta_src, np.asarray([angle_deg]))[:, 0]


def _resample_pressure_at_angle(
    freqs_target: np.ndarray,
    freqs_src: np.ndarray,
    p_src: np.ndarray,
    theta_src: np.ndarray,
    distance_m: float,
    angle_deg: float,
) -> np.ndarray:
    """Resample one angular trace without eagerly resampling unused traces."""
    angle_index = _matching_angle_index(theta_src, angle_deg)
    if angle_index is not None:
        return _resample_frequency_onto(
            freqs_target,
            freqs_src,
            p_src[:, angle_index:angle_index + 1],
            distance_m,
        )[:, 0]
    return _pressure_at_angle(
        _resample_frequency_onto(freqs_target, freqs_src, p_src, distance_m),
        theta_src,
        angle_deg,
    )


def _common_theta_grid(a: ComplexDirectivity, b: ComplexDirectivity) -> np.ndarray:
    theta_min = max(float(np.nanmin(a.theta)), float(np.nanmin(b.theta)))
    theta_max = min(float(np.nanmax(a.theta)), float(np.nanmax(b.theta)))
    if not theta_min < theta_max:
        raise RuntimeError("pair plots require overlapping theta ranges")
    theta = a.theta[(a.theta >= theta_min - 1e-9) & (a.theta <= theta_max + 1e-9)]
    if theta.size < 2:
        raise RuntimeError("pair plots require at least two common theta samples")
    return theta


def _resample_theta_onto(
    p_src: np.ndarray,
    theta_src: np.ndarray,
    theta_target: np.ndarray,
) -> np.ndarray:
    theta_src = np.asarray(theta_src, dtype=float)
    theta_target = np.asarray(theta_target, dtype=float)
    if theta_src.shape == theta_target.shape and np.allclose(theta_src, theta_target):
        return p_src
    out = np.full((p_src.shape[0], len(theta_target)), complex(np.nan, np.nan), dtype=complex)
    for fi, row in enumerate(p_src):
        valid = _finite_complex(row) & np.isfinite(theta_src)
        if np.sum(valid) < 2:
            continue
        theta_valid = theta_src[valid]
        target_mask = (theta_target >= theta_valid[0]) & (theta_target <= theta_valid[-1])
        if not np.any(target_mask):
            continue
        mag_db = 20 * np.log10(np.abs(row[valid]) + 1e-30)
        phase = np.unwrap(np.angle(row[valid]))
        mag = np.interp(theta_target[target_mask], theta_valid, mag_db)
        ph = np.interp(theta_target[target_mask], theta_valid, phase)
        out[fi, target_mask] = 10.0 ** (mag / 20.0) * np.exp(1j * ph)
    return out


def plot_pair_coherent_vs_power_sum(a: ComplexDirectivity,
                                    b: ComplexDirectivity,
                                    out: Path) -> Path:
    """Compare on-axis coherent and power-summed source addition.

    Coherent: |a + b|, preserving each source's BEM/measured phase.
    Power sum: sqrt(|a|^2 + |b|^2), equivalent to uncorrelated sources.

    Difference shows constructive/destructive interference across the overlap.
    """
    f_common = _common_frequency_grid(a, b)
    a_on = _resample_pressure_at_angle(
        f_common, a.freqs, a.p_h, a.theta, a.distance_m, 0.0
    )
    b_on = _resample_pressure_at_angle(
        f_common, b.freqs, b.p_h, b.theta, b.distance_m, 0.0
    )
    valid = _finite_complex(a_on) & _finite_complex(b_on)
    if np.sum(valid) < 2:
        raise RuntimeError("pair coherent plot has fewer than two valid overlapping samples")
    coh = np.abs(a_on[valid] + b_on[valid])
    inc = np.sqrt(np.abs(a_on[valid]) ** 2 + np.abs(b_on[valid]) ** 2)
    f_plot = f_common[valid]

    fig, ax = plt.subplots(figsize=(11, 5))
    _set_figure_bg(fig)
    ax.plot(f_plot, 20 * np.log10(coh + 1e-30), lw=2.0, color=_line_color(0),
            label="Coherent |a + b|")
    ax.plot(f_plot, 20 * np.log10(inc + 1e-30), lw=1.6, color=_line_color(1),
            ls="--", label="Power sum √(|a|² + |b|²)")
    ax.fill_between(f_plot,
                    20 * np.log10(coh + 1e-30),
                    20 * np.log10(inc + 1e-30),
                    color=_line_color(0), alpha=_theme().fill_alpha,
                    label="Interference delta")
    _log_axis(ax)
    _style_ax(ax, xlabel="Frequency", ylabel="On-axis SPL (dB, arbitrary)",
              title="Coherent vs power-summed on-axis sum")
    _add_legend(ax)
    return _finish_fig(fig, out, suptitle=f"{a.label} + {b.label}")


# ---------------------------------------------------------------------------
# Plot 10: Real-phase vs flat-phase summation difference (off-axis sonogram)
# ---------------------------------------------------------------------------

def plot_pair_phase_diff(a: ComplexDirectivity, b: ComplexDirectivity,
                         out: Path) -> Path:
    """For each (freq, angle), compare |a + b| computed with each source's
    actual angular phase to |a + b| computed assuming the off-axis phase
    matches on-axis (the magnitude-only assumption).

    The on-axis phase is flattened first (linear-phase per source) so the
    plot isolates the off-axis phase contribution.
    """
    f_common = _common_frequency_grid(a, b)
    theta_common = _common_theta_grid(a, b)
    a_h = _resample_frequency_onto(f_common, a.freqs, a.p_h, a.distance_m)
    b_h = _resample_frequency_onto(f_common, b.freqs, b.p_h, b.distance_m)
    a_h = _rotate_to_onaxis_phase_zero(
        _resample_theta_onto(a_h, a.theta, theta_common), theta_common
    )
    b_h = _rotate_to_onaxis_phase_zero(
        _resample_theta_onto(b_h, b.theta, theta_common), theta_common
    )
    real_sum = np.abs(a_h + b_h)
    flat_sum = np.abs(np.abs(a_h) + np.abs(b_h))
    diff_db = 20 * np.log10((real_sum + 1e-30) / (flat_sum + 1e-30))

    fig, ax = plt.subplots(figsize=(11, 5))
    _set_figure_bg(fig)
    finite = np.isfinite(diff_db)
    if not np.any(finite):
        raise RuntimeError("pair phase-diff plot has no valid overlapping samples")
    vlim = max(float(np.nanpercentile(np.abs(diff_db[finite]), 98.0)), 0.5)
    mesh = ax.pcolormesh(f_common, theta_common, diff_db.T,
                         cmap=_theme().diverging_cmap, vmin=-vlim, vmax=vlim, shading="auto")
    _log_axis(ax)
    _style_ax(ax, xlabel="Frequency", ylabel="Angle (°)",
              title="Real-phase minus flat-phase summation (dB)", grid=False)
    _add_cbar(fig, mesh, ax, label="dB")
    return _finish_fig(fig, out, suptitle=f"{a.label} + {b.label}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PLOTS = {
    "group_delay": (plot_group_delay, "single"),
    "impulse_response": (plot_impulse_response, "single"),
    "phase_polar": (plot_phase_polar, "single"),
    "phase_sonogram": (plot_phase_sonogram, "single"),
    "acoustic_centre": (plot_acoustic_centre, "single"),
    "di_coherent": (plot_di_coherent_vs_mag, "single"),
    "real_imag": (plot_real_imag_heatmaps, "single"),
    "hv_phase_track": (plot_hv_phase_tracking, "single"),
    "pair_coherent": (plot_pair_coherent_vs_power_sum, "pair"),
    "pair_phase_diff": (plot_pair_phase_diff, "pair"),
}

ALL_SINGLE = [k for k, (_, kind) in PLOTS.items() if kind == "single"]
ALL_PAIR = [k for k, (_, kind) in PLOTS.items() if kind == "pair"]


def render_all_single_source_plots(
    source: ComplexDirectivity,
    outdir: Path,
    *,
    quiet: bool = False,
) -> Dict[str, Optional[Path]]:
    """Render every single-source analysis to outdir/<name>.png.

    Designed for `run_rectangular_horn_sim.py` to auto-write all analyses
    alongside the directivity heatmap on every solve. Failures per plot are
    swallowed with a warning so one bad analysis does not abort the rest.

    Returns a dict mapping plot name to its output path (or None on failure).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Optional[Path]] = {}
    for name in ALL_SINGLE:
        fn, _ = PLOTS[name]
        out_png = outdir / f"{name}.png"
        try:
            fn(source, out_png)
            results[name] = out_png
            if not quiet:
                print(f"  wrote {out_png}", file=sys.stderr)
        except Exception as exc:
            results[name] = None
            print(f"  WARN {name} failed: {exc}", file=sys.stderr)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path, help="result.json or NPZ")
    p.add_argument("--input2", type=Path, default=None,
                   help="second source (enables pair plots)")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--label", type=str, default="")
    p.add_argument("--label2", type=str, default="")
    p.add_argument("--plot", default="all",
                   help=f"comma-separated names, or 'all', 'single', 'pair'. "
                        f"Names: {','.join(PLOTS.keys())}")
    args = p.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    a = load_directivity(args.input, args.label)
    b = load_directivity(args.input2, args.label2) if args.input2 else None

    requested = []
    raw = args.plot.lower()
    if raw == "all":
        requested = list(PLOTS.keys())
    elif raw == "single":
        requested = ALL_SINGLE
    elif raw == "pair":
        requested = ALL_PAIR
    else:
        requested = [s.strip() for s in raw.split(",") if s.strip()]

    for name in requested:
        if name not in PLOTS:
            print(f"  skip unknown plot: {name}", file=sys.stderr)
            continue
        fn, kind = PLOTS[name]
        out_png = args.outdir / f"{name}.png"
        try:
            if kind == "single":
                fn(a, out_png)
            else:
                if b is None:
                    print(f"  skip {name}: needs --input2", file=sys.stderr)
                    continue
                fn(a, b, out_png)
            print(f"  wrote {out_png}")
        except Exception as exc:
            print(f"  ERROR {name}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
