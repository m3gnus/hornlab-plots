"""Derived-output renderers promoted from the Fusion addin solve script.

Faithful ports of the ``solve_fusion_wg_metal.py`` derived renderers — the
driver-interference heatmap, the directivity-index + power-response chart,
the -6 dB beamwidth chart, the on-axis group-delay chart, and the cone
excursion chart with its Xmax marker. Package style: data in, path out,
optional ``theme``/``colors`` overrides, mesh-valid marker options where the
addin originals support them. The math and matplotlib styling match the
addin originals exactly.

Complex-pressure contract: ``save_interference_heatmap`` /
``interference_ratio_db`` accept complex far-field pressure grids in the
HornLab **engineering convention** (``exp(+j*omega*t)`` time dependence,
outgoing waves ``exp(-j*k*r)``) — the convention stored in the Fusion
pressure-basis NPZs. The interference ratio itself only compares magnitudes
of sums, so it is insensitive to global conjugation, but callers should pass
engineering-convention data for consistency with the other derived outputs;
in particular any group delay handed to ``save_group_delay_plot`` must have
been derived from engineering-convention pressure upstream
(``GD = common_delay - d(unwrap(angle(p * exp(+j*omega*common_delay))))/d(omega)``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from ._grid import freq_formatter, log_grid_lines, preferred_frequency_ticks
from ._heatmap import _draw_mesh_valid_markers, prepare_heatmap_data, render_single_heatmap
from .style import (
    FRACTIONAL_OCTAVE,
    apply_theme_overrides,
    theme_grid_kwargs,
    theme_rc_context,
)

# Footnote texts carried verbatim from the addin so rendered wording stays
# identical across the call-site swap.
POLAR_POWER_APPROXIMATION_NOTE = (
    "Polar-cut estimate: intensity is averaged over the stored planes at each "
    "polar angle, solid-angle weighted, and extrapolated to 4*pi; this is not "
    "a full-sphere integration."
)
FULL_SPHERE_POWER_NOTE = (
    "Full-sphere integration of the solved balloon: intensity is solid-angle "
    "weighted over n_theta x n_phi directions."
)
BEAMWIDTH_SYMMETRY_NOTE = (
    "One-sided angle grid: -6 dB beamwidth assumes symmetry about 0 deg."
)


def _sphere_grid_from_pressure(pressure_complex, theta_deg, phi_deg):
    """Return ``(pressure, theta_axis, phi_axis)`` on a sorted sphere grid."""
    pressure = np.asarray(pressure_complex, dtype=np.complex128)
    theta_values = np.asarray(theta_deg, dtype=np.float64)
    phi_values = np.asarray(phi_deg, dtype=np.float64)

    if pressure.ndim == 3:
        if theta_values.ndim != 1 or phi_values.ndim != 1:
            raise ValueError("gridded sphere angles must be 1-D theta and phi axes")
        if pressure.shape[1:] != (theta_values.size, phi_values.size):
            raise ValueError(
                "sphere pressure shape does not match theta/phi axes: "
                f"{pressure.shape} vs "
                f"(n_freq, {theta_values.size}, {phi_values.size})"
            )
        theta_order = np.argsort(theta_values)
        phi_order = np.argsort(phi_values)
        return (
            pressure[:, theta_order, :][:, :, phi_order],
            theta_values[theta_order],
            phi_values[phi_order],
        )

    if pressure.ndim != 2:
        raise ValueError(
            "sphere pressure must be (n_freq, n_direction) or "
            f"(n_freq, n_theta, n_phi), got {pressure.shape}"
        )
    if theta_values.ndim != 1 or phi_values.ndim != 1:
        raise ValueError("flat sphere angles must be 1-D arrays")
    if theta_values.shape != phi_values.shape or theta_values.size != pressure.shape[1]:
        raise ValueError(
            "flat sphere angle arrays must match the pressure direction dimension"
        )

    theta_axis = np.unique(theta_values)
    phi_axis = np.unique(phi_values)
    expected_directions = theta_axis.size * phi_axis.size
    if expected_directions != pressure.shape[1]:
        raise ValueError(
            "flat sphere directions must form a complete theta x phi grid"
        )
    theta_indices = np.searchsorted(theta_axis, theta_values)
    phi_indices = np.searchsorted(phi_axis, phi_values)
    flat_indices = theta_indices * phi_axis.size + phi_indices
    if np.unique(flat_indices).size != expected_directions:
        raise ValueError("flat sphere directions contain duplicate theta/phi pairs")
    gridded = np.empty(
        (pressure.shape[0], theta_axis.size, phi_axis.size),
        dtype=np.complex128,
    )
    gridded[:, theta_indices, phi_indices] = pressure
    return gridded, theta_axis, phi_axis


def _sphere_cell_weights_sr(theta_deg, phi_deg):
    """Return cell solid angles and represented fraction for a sphere grid.

    Counterpart: Waveguide Generator's ``server/solver/directivity_index.py``.
    Both implementations use midpoint theta bands with cosine-edge areas and
    periodic midpoint widths in phi.
    """
    theta = np.asarray(theta_deg, dtype=np.float64)
    phi = np.asarray(phi_deg, dtype=np.float64)
    if (
        theta.ndim != 1
        or theta.size < 2
        or not np.all(np.isfinite(theta))
        or not np.all(np.diff(theta) > 0.0)
        or abs(float(theta[0])) > 1.0e-6
        or float(theta[-1]) > 180.0 + 1.0e-6
    ):
        raise ValueError("theta axis must increase from 0 deg within 0..180 deg")
    if (
        phi.ndim != 1
        or phi.size < 3
        or not np.all(np.isfinite(phi))
        or not np.all(np.diff(phi) > 0.0)
        or float(phi[0]) < -1.0e-6
        or float(phi[-1]) >= 360.0 - 1.0e-6
    ):
        raise ValueError(
            "phi axis must contain at least three increasing periodic samples "
            "in [0, 360) without a wrap duplicate"
        )

    hemisphere = float(theta[-1]) <= 90.0 + 1.0e-6
    theta_upper_deg = 90.0 if hemisphere else 180.0
    theta_edges_deg = np.empty(theta.size + 1, dtype=np.float64)
    theta_edges_deg[0] = 0.0
    theta_edges_deg[-1] = theta_upper_deg
    theta_edges_deg[1:-1] = 0.5 * (theta[:-1] + theta[1:])
    if not np.all(np.diff(theta_edges_deg) > 0.0):
        raise ValueError("theta samples do not define positive midpoint bands")
    theta_edges_rad = np.deg2rad(theta_edges_deg)
    theta_band_weights = (
        np.cos(theta_edges_rad[:-1]) - np.cos(theta_edges_rad[1:])
    )

    previous_phi = np.concatenate(([phi[-1] - 360.0], phi[:-1]))
    following_phi = np.concatenate((phi[1:], [phi[0] + 360.0]))
    phi_widths_rad = np.deg2rad(0.5 * (following_phi - previous_phi))
    if np.any(phi_widths_rad <= 0.0):
        raise ValueError("phi samples do not define positive periodic cells")

    weights_sr = theta_band_weights[:, None] * phi_widths_rad[None, :]
    represented_fraction = 0.5 if hemisphere else 1.0
    return weights_sr, represented_fraction


def sphere_power_metrics(
    pressure_complex,
    theta_deg,
    phi_deg,
    *,
    distance_m,
    rho=1.2041,
    c=343.0,
    p_ref=20.0e-6,
):
    """Integrate DI and acoustic power from a solved spherical pressure grid.

    ``pressure_complex`` accepts metal-bem's flat ``(n_freq, n_direction)``
    balloon with equally flat theta/phi arrays, or a gridded
    ``(n_freq, n_theta, n_phi)`` array with 1-D axes. A grid ending at or
    before 90 degrees represents a hemisphere: its power is integrated over
    2*pi sr and the missing rear hemisphere is treated as silent for DI and
    the full-sphere spatial average.
    """
    distance = float(distance_m)
    density = float(rho)
    sound_speed = float(c)
    reference_pressure = float(p_ref)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_m must be finite and positive")
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("rho must be finite and positive")
    if not np.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("c must be finite and positive")
    if not np.isfinite(reference_pressure) or reference_pressure <= 0.0:
        raise ValueError("p_ref must be finite and positive")

    pressure, theta, phi = _sphere_grid_from_pressure(
        pressure_complex,
        theta_deg,
        phi_deg,
    )
    if pressure.shape[0] == 0 or not np.all(np.isfinite(pressure)):
        raise ValueError("sphere pressure must contain finite frequency rows")
    weights_sr, represented_fraction = _sphere_cell_weights_sr(theta, phi)
    solid_angle_sum = float(np.sum(weights_sr))
    expected_solid_angle = 4.0 * np.pi * represented_fraction
    if not np.isclose(solid_angle_sum, expected_solid_angle, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("sphere solid-angle weights do not match represented coverage")

    intensity = np.abs(pressure) ** 2 / (2.0 * density * sound_speed)
    integrated_intensity = np.sum(intensity * weights_sr[None, :, :], axis=(1, 2))
    spatial_average_intensity = integrated_intensity / solid_angle_sum
    # Only DI expands the represented-domain average over the complete sphere.
    # For a hemisphere, its absent rear half is silent; reported spatial
    # average and power response still describe the sampled hemisphere.
    directivity_average_intensity = (
        spatial_average_intensity * represented_fraction
    )
    on_axis_intensity = np.sum(
        intensity[:, 0, :] * weights_sr[0][None, :], axis=1
    ) / np.sum(weights_sr[0])
    reference_intensity = reference_pressure**2 / (2.0 * density * sound_speed)
    floor = reference_intensity * 1.0e-30
    directivity_index_db = 10.0 * np.log10(
        np.maximum(on_axis_intensity, floor)
        / np.maximum(directivity_average_intensity, floor)
    )
    power_response_db = 10.0 * np.log10(
        np.maximum(spatial_average_intensity, floor) / reference_intensity
    )
    on_axis_spl_db = 10.0 * np.log10(
        np.maximum(on_axis_intensity, floor) / reference_intensity
    )
    acoustic_power_w = distance**2 * integrated_intensity
    polar_weights_sr = np.sum(weights_sr, axis=1)
    return {
        "directivity_index_db": directivity_index_db,
        "power_response_db": power_response_db,
        "on_axis_spl_db": on_axis_spl_db,
        "acoustic_power_w": acoustic_power_w,
        "spatial_average_intensity_w_m2": spatial_average_intensity,
        "solid_angle_sum_sr": solid_angle_sum,
        "solid_angle_coverage_fraction": represented_fraction,
        "polar_angles_deg": theta,
        "polar_weights_sr": polar_weights_sr,
        "sphere_theta_deg": theta,
        "sphere_phi_deg": phi,
        "sphere_weights_sr": weights_sr,
        "approximation": FULL_SPHERE_POWER_NOTE,
        "method": FULL_SPHERE_POWER_NOTE,
    }


# ---------------------------------------------------------------------------
# Shared axis styling (port of the addin's derived-chart line-axes styling)
# ---------------------------------------------------------------------------

def _apply_mesh_valid_markers(
    ax,
    freqs,
    *,
    mesh_valid_hz=None,
    mesh_valid_radiating_hz=None,
    theme=None,
):
    """Overlay the mesh-valid / aperture-valid frequency markers, if any."""
    if mesh_valid_hz is None and mesh_valid_radiating_hz is None:
        return
    _draw_mesh_valid_markers(
        ax,
        lo=float(np.min(freqs)),
        hi=float(np.max(freqs)),
        mesh_valid_hz=mesh_valid_hz,
        mesh_valid_radiating_hz=mesh_valid_radiating_hz,
        span_zorder=0,
        line_zorder=1,
        theme=theme,
    )


def _style_line_axes(ax, *, title, xlabel, ylabel, theme):
    """Canonical derived-chart axes: log-frequency ticks + grid, y-grid."""
    theme_obj = theme
    ax.set_facecolor(theme_obj.axes_bg)
    ax.set_title(title, color=theme_obj.text_color, fontsize=13, fontweight="600", pad=8)
    ax.set_xlabel(xlabel, color=theme_obj.text_color, fontsize=11)
    ax.set_ylabel(ylabel, color=theme_obj.text_color, fontsize=11)
    ax.tick_params(colors=theme_obj.tick_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(theme_obj.spine_color)
    ax.xaxis.set_major_formatter(FuncFormatter(freq_formatter))
    ticks = preferred_frequency_ticks(*ax.get_xlim())
    if ticks:
        ax.set_xticks(ticks)
    for freq in log_grid_lines(*ax.get_xlim()):
        major = any(np.isclose(freq, tick, rtol=1.0e-6, atol=1.0e-6) for tick in ticks)
        ax.axvline(
            freq,
            color=theme_obj.grid_color,
            alpha=theme_obj.primary_grid_alpha if major else theme_obj.secondary_grid_alpha,
            zorder=0,
            **theme_grid_kwargs(theme_obj, linewidth=0.7 if major else 0.5),
        )
    ax.grid(
        True,
        axis="y",
        alpha=theme_obj.secondary_grid_alpha,
        color=theme_obj.grid_color,
        **theme_grid_kwargs(theme_obj, linewidth=0.5),
    )


def _save_figure(fig, output_path, *, dpi, bbox_inches=None):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {} if bbox_inches is None else {"bbox_inches": bbox_inches}
    fig.savefig(
        str(out),
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        **kwargs,
    )
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver-interference heatmap
# ---------------------------------------------------------------------------

def interference_ratio_db(pressure_by_source, members=None):
    """Coherent-vs-incoherent sum ratio, ``20*log10(|sum p| / sum |p|)``.

    Args:
        pressure_by_source: Mapping of source name to a complex pressure grid
            shaped ``(n_freq, n_plane, n_angle)`` in the engineering
            ``exp(+j*omega*t)`` convention (see module docstring). All grids
            must share one shape (already interpolated onto a common
            frequency grid and time/level aligned as intended).
        members: Optional iterable of source names to sum; defaults to every
            key in ``pressure_by_source``.

    Returns:
        Float array shaped like one input grid. 0 dB where the sources add
        fully in phase, dropping toward the clip floor where driver-spacing
        path differences cancel.
    """
    names = list(members) if members is not None else list(pressure_by_source)
    if not names:
        raise ValueError("interference ratio requires at least one source")
    first_name, *remaining_names = names
    first_grid = np.asarray(pressure_by_source[first_name], dtype=np.complex128)
    shape = first_grid.shape
    if not remaining_names:
        coherent = np.abs(first_grid)
        incoherent = coherent.copy()
    else:
        second_name, *remaining_names = remaining_names
        second_grid = np.asarray(
            pressure_by_source[second_name], dtype=np.complex128
        )
        if second_grid.shape != shape:
            raise ValueError(
                f"pressure grid shape mismatch: {second_name} has "
                f"{second_grid.shape}, expected {shape}"
            )
        coherent_sum = np.add(first_grid, second_grid)
        incoherent = np.add(np.abs(first_grid), np.abs(second_grid))
        # The sums own their data; release any dtype-conversion buffers before
        # streaming the remaining sources.
        del first_grid, second_grid

        for name in remaining_names:
            grid = np.asarray(pressure_by_source[name], dtype=np.complex128)
            if grid.shape != shape:
                raise ValueError(
                    f"pressure grid shape mismatch: {name} has {grid.shape}, "
                    f"expected {shape}"
                )
            np.add(coherent_sum, grid, out=coherent_sum)
            np.add(incoherent, np.abs(grid), out=incoherent)
        coherent = np.abs(coherent_sum)

    np.maximum(coherent, 1.0e-30, out=coherent)
    np.maximum(incoherent, 1.0e-30, out=incoherent)
    np.divide(coherent, incoherent, out=coherent)
    np.log10(coherent, out=coherent)
    coherent *= 20.0
    return coherent


def _build_interference_figure(
    frequencies_hz,
    pressure_by_source,
    angles_deg,
    planes,
    *,
    members=None,
    crossovers_hz=None,
    reference_level=-6.0,
    mesh_valid_hz=None,
    mesh_valid_radiating_hz=None,
    theme=None,
    colors=None,
    smooth=False,
    smoothing_fraction=FRACTIONAL_OCTAVE,
):
    theme_obj = apply_theme_overrides(theme, colors=colors)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    angles = np.asarray(angles_deg, dtype=np.float64)
    plane_names = [str(plane) for plane in planes]
    ratio_db = interference_ratio_db(pressure_by_source, members)
    if ratio_db.ndim != 3:
        raise ValueError(
            f"pressure grids must be (n_freq, n_plane, n_angle), got {ratio_db.shape}"
        )
    if ratio_db.shape != (freqs.size, len(plane_names), angles.size):
        raise ValueError(
            "pressure grid shape does not match frequencies/planes/angles: "
            f"{ratio_db.shape} vs {(freqs.size, len(plane_names), angles.size)}"
        )

    with theme_rc_context(theme_obj):
        fig, axes = plt.subplots(
            len(plane_names), 1, figsize=(11.0, 4.6 * len(plane_names))
        )
        if len(plane_names) == 1:
            axes = [axes]
        fig.patch.set_facecolor(theme_obj.figure_bg)
        for plane_index, (ax, plane) in enumerate(zip(axes, plane_names)):
            values = ratio_db[:, plane_index, :].T  # (n_angle, n_freq)
            angles_p, freqs_p, values_p = prepare_heatmap_data(
                angles,
                freqs,
                values,
                smooth=smooth,
                smoothing_fraction=smoothing_fraction,
            )
            render_single_heatmap(
                ax,
                freqs_p,
                angles_p,
                values_p,
                f"{plane[:1].upper()} Driver Interference (0 dB = coherent sum)",
                reference_level=reference_level,
                mesh_valid_hz=mesh_valid_hz,
                mesh_valid_radiating_hz=mesh_valid_radiating_hz,
                theme=theme_obj,
            )
            # Point straight at the bands that matter: cancellation lives where
            # two drivers carry comparable level, around each crossover.
            for xo in crossovers_hz or []:
                ax.axvline(
                    float(xo),
                    color=theme_obj.text_color,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                    zorder=4,
                )
                ax.annotate(
                    f"XO {xo:g} Hz",
                    xy=(float(xo), angles_p[-1]),
                    xytext=(3, -12),
                    textcoords="offset points",
                    color=theme_obj.text_color,
                    fontsize=8,
                    alpha=0.9,
                )
        fig.tight_layout(pad=1.5)
        return fig


def save_interference_heatmap(
    output_path,
    frequencies_hz,
    pressure_by_source,
    angles_deg,
    planes,
    *,
    members=None,
    crossovers_hz=None,
    reference_level=-6.0,
    mesh_valid_hz=None,
    mesh_valid_radiating_hz=None,
    dpi=150,
    theme=None,
    colors=None,
    smooth=False,
    smoothing_fraction=FRACTIONAL_OCTAVE,
):
    """Save the driver-interference heatmap (one panel per plane) to PNG.

    ``20*log10(|sum p| / sum |p|)`` per (frequency, plane, angle): 0 dB means
    the drivers add fully coherently at that angle/frequency; deep negative
    values mark driver-spacing cancellation. The crossover bands, where two
    drivers carry comparable level, are where it matters — pass
    ``crossovers_hz`` to annotate them.

    Args:
        output_path: PNG destination (parents created).
        frequencies_hz: 1D array, ``(n_freq,)`` — the shared frequency grid.
        pressure_by_source: Mapping of source name to a complex pressure grid
            ``(n_freq, n_plane, n_angle)`` in the engineering
            ``exp(+j*omega*t)`` convention (see module docstring), already
            aligned/level-matched as intended for the sum.
        angles_deg: 1D array, ``(n_angle,)``.
        planes: Sequence of plane names, e.g. ``("horizontal", "vertical")``;
            panel titles use the capitalized first letter.
        members: Optional subset/order of ``pressure_by_source`` keys to sum.
        crossovers_hz: Optional crossover frequencies to annotate.
        reference_level: dB level for the prominent contour (default -6).
        mesh_valid_hz: Solid conservative fully-resolved frequency marker.
        mesh_valid_radiating_hz: Dashed radiating-aperture frequency marker.
        smooth: Apply fractional-octave smoothing after canonical interpolation.
        smoothing_fraction: Fractional-octave denominator (default 24).

    Returns:
        The output Path.
    """
    fig = _build_interference_figure(
        frequencies_hz,
        pressure_by_source,
        angles_deg,
        planes,
        members=members,
        crossovers_hz=crossovers_hz,
        reference_level=reference_level,
        mesh_valid_hz=mesh_valid_hz,
        mesh_valid_radiating_hz=mesh_valid_radiating_hz,
        theme=theme,
        colors=colors,
        smooth=smooth,
        smoothing_fraction=smoothing_fraction,
    )
    return _save_figure(fig, output_path, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Directivity index + power response
# ---------------------------------------------------------------------------

def _build_directivity_power_figure(
    frequencies_hz,
    directivity_index_db,
    power_response_db,
    *,
    title="Directivity Index and Power Response",
    footnote=POLAR_POWER_APPROXIMATION_NOTE,
    mesh_valid_hz=None,
    mesh_valid_radiating_hz=None,
    theme=None,
    colors=None,
):
    theme_obj = apply_theme_overrides(theme, colors=colors)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    with theme_rc_context(theme_obj):
        fig, ax = plt.subplots(figsize=(10.0, 4.8))
        fig.patch.set_facecolor(theme_obj.figure_bg)
        ax.semilogx(
            freqs,
            np.asarray(directivity_index_db, dtype=np.float64),
            color=theme_obj.response_colors["combined"],
            linewidth=2.2,
            label="DI",
        )
        ax.set_xlim(float(np.min(freqs)), float(np.max(freqs)))
        _apply_mesh_valid_markers(
            ax,
            freqs,
            mesh_valid_hz=mesh_valid_hz,
            mesh_valid_radiating_hz=mesh_valid_radiating_hz,
            theme=theme_obj,
        )
        ax2 = ax.twinx()
        ax2.semilogx(
            freqs,
            np.asarray(power_response_db, dtype=np.float64),
            color=theme_obj.response_colors["mf"],
            linewidth=1.6,
            linestyle="--",
            alpha=0.82,
            label="Power response",
        )
        ax2.set_ylabel(
            "Power response [dB SPL avg]", color=theme_obj.text_color, fontsize=10
        )
        ax2.tick_params(colors=theme_obj.tick_color, labelsize=8)
        ax2.set_facecolor(theme_obj.axes_bg)
        ax2.spines["right"].set_color(theme_obj.spine_color)
        # Styled after the twin axis: ``ax2.semilogx`` re-applies the shared
        # log x-scale, which resets the canonical tick locator/formatter the
        # addin installed beforehand (its DI chart lost the preferred
        # frequency ticks to that quirk; the styling intent is preserved by
        # running last).
        _style_line_axes(
            ax,
            title=title,
            xlabel="Frequency [Hz]",
            ylabel="Directivity index [dB]",
            theme=theme_obj,
        )
        # Combine the labeled lines from both axes (DI, mesh-valid markers,
        # power response). The addin passed every axes line here; the
        # underscore-labeled grid helper lines are filtered so they cannot
        # leak literal "_child" entries into the legend.
        lines = [
            line
            for line in (*ax.get_lines(), *ax2.get_lines())
            if not str(line.get_label()).startswith("_")
        ]
        labels = [line.get_label() for line in lines]
        legend = ax.legend(
            lines,
            labels,
            loc="best",
            fontsize=9,
            facecolor=theme_obj.axes_bg,
            edgecolor=theme_obj.spine_color,
            labelcolor=theme_obj.text_color,
        )
        legend.get_frame().set_alpha(0.92)
        if footnote:
            fig.text(
                0.5,
                0.01,
                footnote,
                ha="center",
                va="bottom",
                color=theme_obj.text_color,
                fontsize=8,
                alpha=0.78,
            )
            fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0), pad=1.5)
        else:
            fig.tight_layout(pad=1.5)
        return fig


def save_directivity_power_plot(
    output_path,
    frequencies_hz,
    directivity_index_db,
    power_response_db,
    *,
    title="Directivity Index and Power Response",
    footnote=POLAR_POWER_APPROXIMATION_NOTE,
    mesh_valid_hz=None,
    mesh_valid_radiating_hz=None,
    dpi=150,
    theme=None,
    colors=None,
):
    """Save the DI (left axis) + power-response (right axis) chart to PNG.

    Args:
        output_path: PNG destination (parents created).
        frequencies_hz: 1D array, ``(n_freq,)``.
        directivity_index_db: DI values per frequency [dB].
        power_response_db: Spatially averaged power response per frequency
            [dB SPL avg], drawn dashed on a twin right axis.
        title: Chart title (callers usually prefix a device label).
        footnote: Caption under the chart; defaults to the polar-cut
            approximation note the addin renders. Pass ``None`` to omit.
        mesh_valid_hz: Solid conservative fully-resolved frequency marker.
        mesh_valid_radiating_hz: Dashed radiating-aperture frequency marker.

    Returns:
        The output Path.
    """
    fig = _build_directivity_power_figure(
        frequencies_hz,
        directivity_index_db,
        power_response_db,
        title=title,
        footnote=footnote,
        mesh_valid_hz=mesh_valid_hz,
        mesh_valid_radiating_hz=mesh_valid_radiating_hz,
        theme=theme,
        colors=colors,
    )
    return _save_figure(fig, output_path, dpi=dpi)


# ---------------------------------------------------------------------------
# -6 dB beamwidth vs frequency
# ---------------------------------------------------------------------------

def _build_beamwidth_figure(
    frequencies_hz,
    beamwidths_deg,
    *,
    title="-6 dB Beamwidth",
    mesh_valid_hz=None,
    mesh_valid_radiating_hz=None,
    assumed_symmetric_from_one_sided_grid=False,
    theme=None,
    colors=None,
):
    theme_obj = apply_theme_overrides(theme, colors=colors)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    with theme_rc_context(theme_obj):
        fig, ax = plt.subplots(figsize=(10.0, 4.5))
        fig.patch.set_facecolor(theme_obj.figure_bg)
        plane_colors = [
            theme_obj.response_colors["combined"],
            theme_obj.response_colors["mf"],
            theme_obj.response_colors["hf"],
        ]
        for index, (plane, values) in enumerate(beamwidths_deg.items()):
            ax.semilogx(
                freqs,
                np.asarray(values, dtype=np.float64),
                linewidth=2.0,
                color=plane_colors[index % len(plane_colors)],
                label=str(plane),
            )
        ax.set_xlim(float(np.min(freqs)), float(np.max(freqs)))
        ax.set_ylim(0.0, 360.0)
        _apply_mesh_valid_markers(
            ax,
            freqs,
            mesh_valid_hz=mesh_valid_hz,
            mesh_valid_radiating_hz=mesh_valid_radiating_hz,
            theme=theme_obj,
        )
        _style_line_axes(
            ax,
            title=title,
            xlabel="Frequency [Hz]",
            ylabel="-6 dB beamwidth [deg]",
            theme=theme_obj,
        )
        legend = ax.legend(
            loc="best",
            fontsize=9,
            facecolor=theme_obj.axes_bg,
            edgecolor=theme_obj.spine_color,
            labelcolor=theme_obj.text_color,
        )
        legend.get_frame().set_alpha(0.92)
        if assumed_symmetric_from_one_sided_grid:
            fig.text(
                0.5,
                0.01,
                BEAMWIDTH_SYMMETRY_NOTE,
                ha="center",
                va="bottom",
                color=theme_obj.text_color,
                fontsize=8,
                alpha=0.78,
            )
            fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0), pad=1.5)
        else:
            fig.tight_layout(pad=1.5)
        return fig


def save_beamwidth_plot(
    output_path,
    frequencies_hz,
    beamwidths_deg,
    *,
    title="-6 dB Beamwidth",
    mesh_valid_hz=None,
    mesh_valid_radiating_hz=None,
    assumed_symmetric_from_one_sided_grid=False,
    dpi=150,
    theme=None,
    colors=None,
):
    """Save the -6 dB beamwidth-vs-frequency chart (one trace per plane).

    Args:
        output_path: PNG destination (parents created).
        frequencies_hz: 1D array, ``(n_freq,)``.
        beamwidths_deg: Mapping of plane name (e.g. ``"horizontal"``) to a
            per-frequency -6 dB beamwidth array [deg]; the y-axis is fixed to
            0-360 deg like the addin original.
        title: Chart title (callers usually prefix a device label).
        mesh_valid_hz: Solid conservative fully-resolved frequency marker.
        mesh_valid_radiating_hz: Dashed radiating-aperture frequency marker.
        assumed_symmetric_from_one_sided_grid: Render the one-sided-grid
            symmetry footnote (set when the widths were doubled from a
            0..+max angle grid).

    Returns:
        The output Path.
    """
    fig = _build_beamwidth_figure(
        frequencies_hz,
        beamwidths_deg,
        title=title,
        mesh_valid_hz=mesh_valid_hz,
        mesh_valid_radiating_hz=mesh_valid_radiating_hz,
        assumed_symmetric_from_one_sided_grid=assumed_symmetric_from_one_sided_grid,
        theme=theme,
        colors=colors,
    )
    return _save_figure(fig, output_path, dpi=dpi)


# ---------------------------------------------------------------------------
# On-axis group delay
# ---------------------------------------------------------------------------

def _build_group_delay_figure(
    frequencies_hz,
    group_delay_s,
    *,
    title="On-Axis Group Delay",
    mesh_valid_hz=None,
    mesh_valid_radiating_hz=None,
    theme=None,
    colors=None,
):
    theme_obj = apply_theme_overrides(theme, colors=colors)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    with theme_rc_context(theme_obj):
        fig, ax = plt.subplots(figsize=(10.0, 4.5))
        fig.patch.set_facecolor(theme_obj.figure_bg)
        ax.semilogx(
            freqs,
            np.asarray(group_delay_s, dtype=np.float64) * 1000.0,
            color=theme_obj.response_colors["combined"],
            linewidth=2.0,
        )
        ax.set_xlim(float(np.min(freqs)), float(np.max(freqs)))
        _apply_mesh_valid_markers(
            ax,
            freqs,
            mesh_valid_hz=mesh_valid_hz,
            mesh_valid_radiating_hz=mesh_valid_radiating_hz,
            theme=theme_obj,
        )
        _style_line_axes(
            ax,
            title=title,
            xlabel="Frequency [Hz]",
            ylabel="Group delay [ms]",
            theme=theme_obj,
        )
        fig.tight_layout(pad=1.5)
        return fig


def save_group_delay_plot(
    output_path,
    frequencies_hz,
    group_delay_s,
    *,
    title="On-Axis Group Delay",
    mesh_valid_hz=None,
    mesh_valid_radiating_hz=None,
    dpi=150,
    theme=None,
    colors=None,
):
    """Save the group-delay-vs-frequency chart to PNG.

    Args:
        output_path: PNG destination (parents created).
        frequencies_hz: 1D array, ``(n_freq,)``.
        group_delay_s: Group delay per frequency in **seconds** (plotted in
            ms). NaN bins (phase-unwrap ambiguity) simply break the trace.
            Compute it from engineering-convention (``exp(+j*omega*t)``)
            complex pressure upstream; see the module docstring.
        title: Chart title (callers usually prefix a device label).
        mesh_valid_hz: Solid conservative fully-resolved frequency marker.
        mesh_valid_radiating_hz: Dashed radiating-aperture frequency marker.

    Returns:
        The output Path.
    """
    fig = _build_group_delay_figure(
        frequencies_hz,
        group_delay_s,
        title=title,
        mesh_valid_hz=mesh_valid_hz,
        mesh_valid_radiating_hz=mesh_valid_radiating_hz,
        theme=theme,
        colors=colors,
    )
    return _save_figure(fig, output_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Cone excursion vs frequency (with Xmax marker)
# ---------------------------------------------------------------------------

def _build_excursion_figure(
    frequencies_hz,
    excursion_m,
    *,
    label="Driver",
    xmax_m=None,
    title=None,
    theme=None,
    colors=None,
):
    theme_obj = apply_theme_overrides(theme, colors=colors)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    with theme_rc_context(theme_obj):
        fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
        fig.patch.set_facecolor(theme_obj.figure_bg)
        ax.semilogx(
            freqs,
            np.asarray(excursion_m, dtype=np.float64) * 1.0e3,
            color=theme_obj.response_colors["combined"],
            linewidth=2.0,
            label=str(label),
        )
        if xmax_m is not None:
            ax.axhline(
                float(xmax_m) * 1.0e3,
                color=theme_obj.mesh_limit_color,
                linestyle="--",
                label="Xmax",
            )
        ax.set_xlim(float(np.min(freqs)), float(np.max(freqs)))
        _style_line_axes(
            ax,
            title=title if title is not None else f"{label} Cone Excursion",
            xlabel="Frequency [Hz]",
            ylabel="Excursion [mm RMS]",
            theme=theme_obj,
        )
        legend = ax.legend(
            loc="best",
            fontsize=9,
            facecolor=theme_obj.axes_bg,
            edgecolor=theme_obj.spine_color,
            labelcolor=theme_obj.text_color,
        )
        legend.get_frame().set_alpha(0.92)
        return fig


def save_excursion_plot(
    output_path,
    frequencies_hz,
    excursion_m,
    *,
    label="Driver",
    xmax_m=None,
    title=None,
    dpi=160,
    theme=None,
    colors=None,
):
    """Save the cone-excursion-vs-frequency chart (with optional Xmax) to PNG.

    Args:
        output_path: PNG destination (parents created).
        frequencies_hz: 1D array, ``(n_freq,)``.
        excursion_m: RMS cone excursion per frequency in **meters** (plotted
            in mm RMS).
        label: Trace label; also used for the default title
            ``"{label} Cone Excursion"``.
        xmax_m: Optional linear-excursion limit in meters; drawn as a dashed
            horizontal marker in the theme's mesh-limit warning color.
        title: Override the default title.

    Returns:
        The output Path.
    """
    fig = _build_excursion_figure(
        frequencies_hz,
        excursion_m,
        label=label,
        xmax_m=xmax_m,
        title=title,
        theme=theme,
        colors=colors,
    )
    return _save_figure(fig, output_path, dpi=dpi)


__all__ = [
    "POLAR_POWER_APPROXIMATION_NOTE",
    "FULL_SPHERE_POWER_NOTE",
    "BEAMWIDTH_SYMMETRY_NOTE",
    "sphere_power_metrics",
    "interference_ratio_db",
    "save_interference_heatmap",
    "save_directivity_power_plot",
    "save_beamwidth_plot",
    "save_group_delay_plot",
    "save_excursion_plot",
]
