# Changes

## Unreleased

Breaking (public API; take at least a minor version at release):

- `frequency_response_b64` and `render_all_charts_b64` no longer accept
  `phase_degrees`, `phase_reference_distance_m`, `sound_speed_m_per_s` or
  `phase_time_convention` (the payload keys of the same names are ignored by
  `render_all_charts_b64`). The canonical frequency-response chart draws no
  phase; a caller passing these keyword arguments now gets `TypeError`. The
  private phase helpers (`_flatten_phase_for_display`, `_phase_propagation_sign`,
  `_passband_weights`, `_weighted_linfit`) are removed. Waveguide Generator, the
  only caller, references and draws phase itself through `chart_spec_b64`.

Added:

- `chart_spec_b64` / `build_chart_spec_figure`: a chart described as data
  (`lines`, `polar`, `beam_map`, `balloon`), drawn in the active theme, with
  bounded counts and ranges.

Changed:

- Legend and twin-axis styling are shared helpers (`_themed_legend`,
  `_style_twin_axis`); the canonical PNGs are byte-identical.
