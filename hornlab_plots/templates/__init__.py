"""Opinionated plot templates that compose the core renderers.

These are the "preset" entry points callers reach for when they want a
canonical look without configuring every styling kwarg. The current set:

- ``optimizer_isobar_jet`` — Optimizer-Dashboard jet-style isobar
  (custom_colors palette, gouraud shading, BoundaryNorm step colorbar).
  Replaces ``Optimizer-Dashboard/bem_optimizer/visualization/plots.py``.
- ``optimizer_impedance`` — Dashboard impedance line plot (no log-grid).
- ``optimizer_beamwidth`` — Dashboard beamwidth-vs-frequency with
  optional target overlay.

The directivity-heatmap "bigmeh_chamber_stacked" template is not added
here as a single function because each BIGMEH post script applies
slightly different overlay logic (baseline minus-6 contour, tube
resonance markers, β-damping legends). Instead, those scripts call
``hornlab_plots.render_single_heatmap`` plus the canonical Arctic Night
constants from ``hornlab_plots.style`` directly — same affordance the
WG-private call sites used, now via a clean import path.
"""

from .optimizer_jet import (
    optimizer_beamwidth,
    optimizer_impedance,
    optimizer_isobar_jet,
)

__all__ = [
    "optimizer_isobar_jet",
    "optimizer_impedance",
    "optimizer_beamwidth",
]
