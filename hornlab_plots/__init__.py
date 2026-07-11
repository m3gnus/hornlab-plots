"""Canonical visualization stack for HornLab.

The render half of the canonical BEM pipeline. Sibling to
``hornlab-waveguide-mesher`` (geometry → ``.msh``) and
``hornlab-metal-bem`` (``.msh`` → results); consumes the numpy arrays they
produce.

Public API:

- ``directivity_heatmap_from_legacy_dict(frequencies, directivity, ...)`` —
  byte-equivalent replacement for WG ``render_directivity_plot``.
  Accepts the dict-of-list payload shape ``{"horizontal": [[[angle, db], ...], ...], ...}``.
- ``directivity_heatmap_b64(freq_hz, angles_deg, spl_db, ...)`` —
  numpy-array signature (preferred for new callers).
- ``save_directivity_plot(output_path, frequencies, directivity, ...)`` —
  same as the legacy-dict signature but writes to disk.
- ``frequency_response_b64(frequencies, spl)`` — on-axis SPL.
- ``directivity_index_b64(frequencies, di)`` — DI vs frequency.
- ``impedance_b64(frequencies, real, imaginary)`` — \\|Z\\| line plot.
- ``save_impedance_plot(output_path, ...)`` — impedance to disk.
- ``render_all_charts_b64(payload)`` — convenience: returns dict with
  ``frequency_response``, ``directivity_index``, ``impedance``,
  ``directivity_map`` keys.
- ``polar_line_b64(...)`` / ``save_polar_line_plot(...)`` — fixed-frequency
  polar line plots from complex pressure or dB arrays.
- ``hornlab_plots.complex_analysis`` — complex-pressure analysis plots used
  by WG's ``/api/render-complex-analysis`` route and CLI wrapper.

Derived-output renderers (ported from the Fusion addin; see
``hornlab_plots.derived`` for the engineering ``exp(+j*omega*t)``
complex-pressure contract):

- ``save_interference_heatmap(output_path, freqs, pressure_by_source, ...)``
  — coherent-vs-incoherent driver-sum ratio heatmap per plane
  (``interference_ratio_db`` exposes the underlying math).
- ``save_directivity_power_plot(output_path, freqs, di_db, power_db, ...)``
  — DI + power response on twin axes.
- ``save_beamwidth_plot(output_path, freqs, beamwidths_deg, ...)`` —
  -6 dB beamwidth vs frequency per plane.
- ``save_group_delay_plot(output_path, freqs, group_delay_s, ...)`` —
  on-axis group delay vs frequency.
- ``save_excursion_plot(output_path, freqs, excursion_m, ...)`` — cone
  excursion vs frequency with optional Xmax marker.

Lower-level primitives:

- ``render_single_heatmap(ax, ...)`` — render onto a caller-supplied Axes.
- ``prepare_heatmap_data(angles, freqs, values)``
- ``build_grid_from_legacy(freqs, patterns)``
- ``check_symmetry(h_values, v_values)``

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
    check_symmetry,
    directivity_heatmap_b64,
    directivity_heatmap_from_legacy_dict,
    prepare_heatmap_data,
    render_single_heatmap,
    save_directivity_plot,
)
from .charts import (
    FrequencyResponseCurve,
    directivity_index_b64,
    frequency_response_b64,
    frequency_response_multi_b64,
    impedance_b64,
    render_all_charts_b64,
    save_frequency_response_plot,
    save_impedance_plot,
    set_spl_window,
    spl_window,
)
from ._polar import (
    polar_line_b64,
    prepare_polar_line_data,
    save_polar_line_plot,
)
from .derived import (
    interference_ratio_db,
    save_beamwidth_plot,
    save_directivity_power_plot,
    save_excursion_plot,
    save_group_delay_plot,
    save_interference_heatmap,
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
    set_theme,
    theme,
    theme_context,
)

__version__ = "0.1.0"

__all__ = [
    # Heatmap entry points
    "directivity_heatmap_from_legacy_dict",
    "directivity_heatmap_b64",
    "save_directivity_plot",
    "render_single_heatmap",
    "prepare_heatmap_data",
    "build_grid_from_legacy",
    "check_symmetry",
    "prepare_polar_line_data",
    "polar_line_b64",
    "save_polar_line_plot",
    # Line charts
    "frequency_response_b64",
    "frequency_response_multi_b64",
    "directivity_index_b64",
    "impedance_b64",
    "save_frequency_response_plot",
    "save_impedance_plot",
    "render_all_charts_b64",
    "FrequencyResponseCurve",
    "set_spl_window",
    "spl_window",
    # Derived-output renderers (Fusion addin ports)
    "interference_ratio_db",
    "save_interference_heatmap",
    "save_directivity_power_plot",
    "save_beamwidth_plot",
    "save_group_delay_plot",
    "save_excursion_plot",
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
    "set_theme",
    "theme",
    "theme_context",
    # Tick / grid helpers
    "log_grid_lines",
    "preferred_frequency_ticks",
    "freq_formatter",
    "contains_frequency",
    "__version__",
]
