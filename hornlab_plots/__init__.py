"""Canonical visualization stack for HornLab.

The render half of the canonical BEM pipeline. Sibling to
``hornlab-waveguide-mesher`` (geometry → ``.msh``) and
``hornlab-metal-bem`` (``.msh`` → results); consumes the numpy arrays they
produce.

Public API:

- ``directivity_heatmap_from_legacy_dict(frequencies, directivity, ...)`` —
  API-compatible replacement for WG ``render_directivity_plot``.
  Accepts the dict-of-list payload shape ``{"horizontal": [[[angle, db], ...], ...], ...}``.
- ``save_directivity_plot(output_path, frequencies, directivity, ...)`` —
  same as the legacy-dict signature but writes to disk.
- ``frequency_response_b64(frequencies, spl)`` — on-axis SPL.
- ``directivity_index_b64(frequencies, di)`` — DI vs frequency.
- ``forward_beam_shape_b64(beam_shape)`` — fitted beam shape + spherical DI.
- ``impedance_b64(frequencies, real, imaginary)`` — \\|Z\\| line plot.
- ``render_all_charts_b64(payload)`` — convenience: returns dict with
  ``frequency_response``, ``directivity_index``, ``impedance``,
  ``directivity_map`` keys.

Derived-output renderers (ported from the Fusion addin):

- ``save_directivity_power_plot(output_path, freqs, di_db, power_db, ...)``
  — DI + power response on twin axes.
- ``sphere_power_metrics(pressure, theta_deg, phi_deg, ...)`` — solid-angle
  integration of a solved pressure balloon over a hemisphere (theta ending at
  90 deg) or a full sphere (180 deg). A theta axis ending anywhere else is a
  partial cap and is refused; pass ``cap_theta_max_deg`` to declare that cap's
  coverage and integrate it over its own solid angle.
- ``save_beamwidth_plot(output_path, freqs, beamwidths_deg, ...)`` —
  -6 dB beamwidth vs frequency per plane.

Lower-level primitives:

- ``render_single_heatmap(ax, ...)`` — render onto a caller-supplied Axes.
- ``prepare_heatmap_data(angles, freqs, values)``
- ``build_grid_from_legacy(freqs, patterns)``

Theme API and backwards-compatible constants live in
``hornlab_plots.style``. Project-specific plot templates live with their
owning tools, not in this public renderer.
"""

from ._grid import (
    contains_frequency,
    freq_formatter,
    log_grid_lines,
    preferred_frequency_ticks,
)
from ._heatmap import (
    build_grid_from_legacy,
    directivity_heatmap_from_legacy_dict,
    prepare_heatmap_data,
    render_single_heatmap,
    save_directivity_plot,
)
from .charts import (
    FrequencyResponseCurve,
    directivity_index_b64,
    forward_beam_shape_b64,
    frequency_response_b64,
    impedance_b64,
    render_all_charts_b64,
    save_frequency_response_plot,
    set_spl_window,
    spl_window,
)
from .derived import (
    FULL_SPHERE_POWER_NOTE,
    PARTIAL_CAP_POWER_NOTE,
    POLAR_POWER_APPROXIMATION_NOTE,
    save_beamwidth_plot,
    save_directivity_power_plot,
    sphere_power_metrics,
)
from .style import (
    ABYSS_THEME,
    BLUEPRINT_THEME,
    BUILTIN_THEMES,
    CLASSIC_THEME,
    CONTRAST_THEME,
    DARK_THEME,
    EMBER_THEME,
    GRANITE_THEME,
    HORNLAB_THEME,
    JOURNAL_THEME,
    PHOSPHOR_THEME,
    SEPIA_THEME,
    Theme,
    get_theme,
    theme,
)

__version__ = "0.1.3"

__all__ = [
    # Heatmap entry points
    "directivity_heatmap_from_legacy_dict",
    "save_directivity_plot",
    "render_single_heatmap",
    "prepare_heatmap_data",
    "build_grid_from_legacy",
    # Line charts
    "frequency_response_b64",
    "directivity_index_b64",
    "forward_beam_shape_b64",
    "impedance_b64",
    "save_frequency_response_plot",
    "render_all_charts_b64",
    "FrequencyResponseCurve",
    "set_spl_window",
    "spl_window",
    # Derived-output renderers (Fusion addin ports)
    "FULL_SPHERE_POWER_NOTE",
    "PARTIAL_CAP_POWER_NOTE",
    "POLAR_POWER_APPROXIMATION_NOTE",
    "sphere_power_metrics",
    "save_directivity_power_plot",
    "save_beamwidth_plot",
    # Themes
    "Theme",
    "HORNLAB_THEME",
    "DARK_THEME",
    "GRANITE_THEME",
    "ABYSS_THEME",
    "BLUEPRINT_THEME",
    "JOURNAL_THEME",
    "CONTRAST_THEME",
    "SEPIA_THEME",
    "PHOSPHOR_THEME",
    "EMBER_THEME",
    "CLASSIC_THEME",
    "BUILTIN_THEMES",
    "get_theme",
    "theme",
    # Tick / grid helpers
    "log_grid_lines",
    "preferred_frequency_ticks",
    "freq_formatter",
    "contains_frequency",
    "__version__",
]
