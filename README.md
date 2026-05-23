# hornlab-plots

Canonical visualization stack for the HornLab BEM pipeline.

This is the **render** half of the canonical pipeline. It sits alongside
`hornlab-mesher` (geometry → `.msh`) and `hornlab-solver` (`.msh` →
result NPZs / dicts) and consumes the numpy arrays they produce to draw
directivity heatmaps, polar line plots, frequency response, DI, and
impedance.

## Why this package exists

The memory rule "Heatmaps via WG render-directivity, never matplotlib"
existed because the canonical plot code lived inside the Waveguide-
Generator server, only reachable via `/api/render-directivity` (which
needs the WG dev server up) or by importing leading-underscore private
helpers across a `sys.path.insert(WG_SERVER)` boundary. Every consumer
that wanted offline plotting fell back to hand-rolled matplotlib, which
broke cross-run comparability and visual consistency.

`hornlab-plots` is the canonical import — `pip install -e .` it into
any venv and call the functions directly. The WG HTTP routes become
thin wrappers over this package; everyone else (Optimizer-Dashboard,
MEH-Lab post scripts, forum-reply tooling) imports it directly.

## Public API

Numpy-first, opinionated styling baked in. All public functions accept
small numpy arrays plus a handful of styling kwargs and return either a
base64-encoded PNG string (for HTTP routes), a PIL Image, or write to
disk when an output path is supplied.

```python
import hornlab_plots as hlp

# Directivity heatmap from a dict-of-list payload (WG legacy shape).
b64 = hlp.directivity_heatmap_from_legacy_dict(
    frequencies=[1000, 2000, 4000],
    directivity={"horizontal": [...], "vertical": [...]},
    reference_level=-6.0,
)

# Or from numpy arrays directly (preferred).
b64 = hlp.directivity_heatmap_b64(
    freq_hz=np.array([1000, 2000, 4000]),
    angles_deg=np.array([-90, ..., 90]),
    spl_db=spl_matrix,        # (n_angle, n_freq)
    plane="h",
    reference_level=-6.0,
)

# Line plots.
b64 = hlp.frequency_response_b64(freqs, spl)
b64 = hlp.directivity_index_b64(freqs, di_per_plane)
b64 = hlp.impedance_b64(freqs, z_real, z_imag)

# Save to disk.
hlp.save_directivity_plot(path, frequencies, directivity)
hlp.save_impedance_plot(path, freqs, z_real, z_imag)
```

Lower-level helpers that take a matplotlib `Axes` for composition:

```python
hlp.render_single_heatmap(ax, freqs, angles, values, title, reference_level=-6.0)
hlp.prepare_heatmap_data(angles, freqs, values)
hlp.build_grid_from_legacy(freqs, patterns)
```

Theme constants for callers that compose figures around heatmap panels:

```python
from hornlab_plots.style import (
    FIGURE_BG, AXES_BG, TEXT_COLOR, TICK_COLOR, SPINE_COLOR,
    CONTOUR_OUTLINE, REFERENCE_CONTOUR_COLOR, HEATMAP_CMAP,
)
```

## Installation

```bash
pip install -e ~/Code/HornLab/hornlab-plots
```

Same pattern as `hornlab-mesher` and `hornlab-solver`: a sibling
package under `HornLab/`, installed editable into each consumer venv.
Not published to PyPI; lives in the private workspace.

## Migration status

This package is the canonical plot renderer for HornLab. The legacy WG
modules at `Waveguide-Generator/server/solver/directivity_plot.py` and
`server/solver/charts.py` were removed on 2026-05-23 after byte-identical
gate evidence.
