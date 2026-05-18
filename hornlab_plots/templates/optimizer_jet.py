"""Optimizer-Dashboard "jet" isobar template.

Independent palette from the WG Arctic Night look — Dashboard tunes for
absolute SPL comparison across many candidates in the leaderboard view,
where the warm-to-cool jet gradient makes the dB ceiling visually
unambiguous. Migrated unchanged from
``Optimizer-Dashboard/bem_optimizer/visualization/plots.py`` so existing
candidate PNGs and leaderboard tiles look identical.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


DEFAULT_JET_COLORS: tuple[str, ...] = (
    "#00008F",
    "#0000FF",
    "#006FFF",
    "#00DFFF",
    "#4FFFBF",
    "#BFFF4F",
    "#FFDF00",
    "#FF6F00",
    "#FF0000",
    "#8F0000",
)


def _setup_log_frequency_axis(ax: plt.Axes) -> None:
    tickvals = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ticktext = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]

    ax.set_xscale("log")
    ax.set_xlim(200, 20000)
    ax.set_xticks(tickvals)
    ax.set_xticklabels(ticktext)


def _build_db_tick_values(min_db: float, max_db: float, step_db: float) -> np.ndarray:
    if step_db <= 0:
        raise ValueError("colorbar_tick_step_db must be > 0")
    start = np.ceil(min_db / step_db) * step_db
    end = np.floor(max_db / step_db) * step_db
    if end < start:
        return np.array([min_db, max_db], dtype=float)
    return np.arange(start, end + 0.5 * step_db, step_db, dtype=float)


def _upsample_isobar_grid(
    angle_deg: np.ndarray,
    freqs_hz: np.ndarray,
    spl_matrix: np.ndarray,
    angle_factor: int,
    freq_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if angle_factor < 1 or freq_factor < 1:
        raise ValueError("Interpolation factors must be >= 1")

    if angle_factor == 1 and freq_factor == 1:
        return angle_deg, freqs_hz, spl_matrix

    n_angles = angle_deg.shape[0]
    n_freqs = freqs_hz.shape[0]

    dense_angle_count = (n_angles - 1) * angle_factor + 1
    dense_freq_count = (n_freqs - 1) * freq_factor + 1

    dense_angles = np.linspace(float(angle_deg[0]), float(angle_deg[-1]), dense_angle_count)
    dense_log_freqs = np.linspace(
        np.log10(float(freqs_hz[0])), np.log10(float(freqs_hz[-1])), dense_freq_count
    )
    source_log_freqs = np.log10(freqs_hz)

    freq_upsampled = np.empty((n_angles, dense_freq_count), dtype=float)
    for row_idx in range(n_angles):
        freq_upsampled[row_idx, :] = np.interp(dense_log_freqs, source_log_freqs, spl_matrix[row_idx, :])

    dense_spl = np.empty((dense_angle_count, dense_freq_count), dtype=float)
    for col_idx in range(dense_freq_count):
        dense_spl[:, col_idx] = np.interp(dense_angles, angle_deg, freq_upsampled[:, col_idx])

    return dense_angles, np.power(10.0, dense_log_freqs), dense_spl


def optimizer_isobar_jet(
    output_path,
    angle_deg: np.ndarray,
    freqs_hz: np.ndarray,
    spl_matrix: np.ndarray,
    *,
    title: str,
    clip_min_db: float,
    clip_max_db: float,
    colors: tuple[str, ...] = DEFAULT_JET_COLORS,
    colorbar_tick_step_db: float = 3.0,
    figure_width_in: float = 11.0,
    figure_height_in: float = 6.0,
    figure_dpi: int = 160,
    isobar_interp_angle_factor: int = 2,
    isobar_interp_freq_factor: int = 3,
) -> Path:
    """Render an Optimizer-Dashboard jet-style isobar to ``output_path``.

    Matches the previous behavior of
    ``Optimizer-Dashboard/bem_optimizer/visualization/plots.py::_save_isobar_plot``
    byte-for-byte. Returns the resolved output Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    boundaries = np.linspace(clip_min_db, clip_max_db, len(colors) + 1)
    cmap = ListedColormap(list(colors))
    norm = BoundaryNorm(boundaries, cmap.N)
    cbar_ticks = _build_db_tick_values(clip_min_db, clip_max_db, colorbar_tick_step_db)

    plot_angles, plot_freqs, plot_spl = _upsample_isobar_grid(
        angle_deg=angle_deg,
        freqs_hz=freqs_hz,
        spl_matrix=spl_matrix,
        angle_factor=isobar_interp_angle_factor,
        freq_factor=isobar_interp_freq_factor,
    )
    plot_spl = np.clip(plot_spl, clip_min_db, clip_max_db)

    fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in), dpi=figure_dpi)
    mesh = ax.pcolormesh(plot_freqs, plot_angles, plot_spl, cmap=cmap, norm=norm, shading="gouraud")

    _setup_log_frequency_axis(ax)
    ax.set_ylim(-180, 180)
    ax.set_yticks(np.arange(-180, 181, 30))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title(title)
    ax.grid(which="major", color="#808080", linewidth=0.8)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Normalized SPL (dB)")
    cbar.set_ticks(cbar_ticks)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def optimizer_impedance(
    output_path,
    impedance_freq_hz: np.ndarray,
    impedance_real: np.ndarray,
    impedance_imag: np.ndarray,
    *,
    figure_width_in: float = 11.0,
    figure_height_in: float = 6.0,
    figure_dpi: int = 160,
) -> Path:
    """Optimizer-Dashboard impedance line plot.

    Lighter-weight than ``hornlab_plots.impedance_b64`` — Dashboard uses
    a plain semilog plot with no log-grid overlay (matches the existing
    leaderboard look).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in), dpi=figure_dpi)

    ax.plot(impedance_freq_hz, impedance_real, linewidth=2, label="Z real")
    ax.plot(impedance_freq_hz, impedance_imag, linewidth=2, linestyle="--", label="Z imag")

    _setup_log_frequency_axis(ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Acoustic Impedance (Pa·s/m³)")
    ax.set_title("Acoustic Impedance")
    ax.grid(which="major", color="#808080", linewidth=0.8)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def optimizer_beamwidth(
    output_path,
    freq_hz: np.ndarray,
    bw_measured: np.ndarray,
    bw_target: np.ndarray | None,
    axis_label: str,
    *,
    figure_width_in: float = 11.0,
    figure_height_in: float = 6.0,
    figure_dpi: int = 160,
) -> Path:
    """Beamwidth-vs-frequency plot with optional target overlay."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in), dpi=figure_dpi)

    ax.plot(freq_hz, bw_measured, linewidth=2, color="#2196F3", label="Measured -6 dB BW")
    if bw_target is not None:
        ax.plot(freq_hz, bw_target, linewidth=2, linestyle="--", color="#FF5722", label="Target BW")
        ax.fill_between(freq_hz, bw_measured, bw_target, alpha=0.15, color="#FF5722")

    _setup_log_frequency_axis(ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Beamwidth (deg)")
    ax.set_title(f"{axis_label} Beamwidth vs Frequency")
    ax.grid(which="major", color="#808080", linewidth=0.8)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
