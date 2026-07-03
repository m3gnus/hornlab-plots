# hornlab-plots

Reusable visualization stack for acoustic simulation results.

This package renders directivity heatmaps, polar line plots, frequency
response, DI, and impedance from numpy arrays or lightweight payloads. It can
be used standalone or as the render layer for a BEM pipeline such as
`hornlab-waveguide-mesher` (geometry → `.msh`) and `hornlab-metal-bem`
(`.msh` → result NPZs / dicts).

## Why this package exists

The old memory rule "Heatmaps via WG render-directivity, never matplotlib"
existed because the plot code used to live inside the Waveguide-Generator
server, behind `/api/render-directivity` (which needed the WG dev server up)
or leading-underscore private helpers imported across a
`sys.path.insert(WG_SERVER)` boundary. That route is now only a WG browser UI
wrapper around this package; non-WG consumers import `hornlab_plots` directly.

`hornlab-plots` is the canonical import — `pip install -e .` it into
any venv and call the functions directly. The WG HTTP routes become
thin wrappers over this package; everyone else (Optimizer-Dashboard,
MEH-Lab post scripts, forum-reply tooling) imports it directly.

## Public API

Numpy-first, opinionated styling baked in. All public functions accept
small numpy arrays plus a handful of styling kwargs and return either a
base64-encoded PNG string (for HTTP routes), a PIL Image, or write to
disk when an output path is supplied.

Optimizer-Dashboard-specific templates, including the jet isobar,
impedance, and beamwidth leaderboard plots, moved back into the private
Optimizer-Dashboard package. `hornlab-plots` intentionally exports only
shared renderer APIs.

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
b64 = hlp.frequency_response_multi_b64(
    [
        hlp.FrequencyResponseCurve(freqs, lf_lp_db, "LF LR4 LP", role="lf", crossover=True),
        hlp.FrequencyResponseCurve(freqs, hf_hp_db, "HF LR4 HP", role="hf", crossover=True),
        hlp.FrequencyResponseCurve(freqs, combined_db, "Combined", role="combined"),
    ],
    title="LR4 combined on-axis response",
    crossover_hz=800.0,
)
b64 = hlp.directivity_index_b64(freqs, di_per_plane)
b64 = hlp.impedance_b64(freqs, z_real, z_imag)

# Save to disk.
hlp.save_directivity_plot(path, frequencies, directivity)
hlp.save_frequency_response_plot(path, curves, title="On-axis response")
hlp.save_impedance_plot(path, freqs, z_real, z_imag)
```

### Canonical Frequency-Response Rules

- Use `frequency_response_multi_b64()` or `save_frequency_response_plot()`
  for every new on-axis response plot. Local matplotlib response plots should
  be temporary only.
- Use the role colors: `lf` = blue/cyan, `mf` = amber, `hf` = red, and
  `combined` = bright near-white. These are defined in
  `hornlab_plots.style.RESPONSE_COLORS` and sit on the same Arctic Night
  background as the canonical heatmap.
- Use solid lines for raw/source responses. Set `crossover=True` for
  crossover-applied component traces (`LR4 LP`, `LR4 HP`, band-pass, etc.);
  those render as dotted traces. The summed response uses `role="combined"`
  and renders thicker/solid.
- Pass `crossover_hz` for crossover plots. The renderer draws a dashed
  vertical marker so the filtered component behavior is legible.
- Keep the visible y-axis focused: the renderer defaults to a 40 dB SPL
  window with a small headroom above the loudest finite curve point. Deep
  filter tails and numerical dropouts should not decide the plot scale.
- Component curves in combined plots should be physically or intentionally
  level matched before plotting. Do not hide a real gain mismatch by
  display-normalizing without saying so in the label.

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

## Themes

Ten built-in themes register in `hornlab_plots.style.BUILTIN_THEMES`. The
default is `hornlab` and its output is pixel-identical to the pre-theme-API
renderer; the others opt in per call (`theme="granite"`) or process-wide
(`set_theme(...)` / `theme_context(...)`).

| Theme | Intent |
| --- | --- |
| `hornlab` | Canonical Arctic Night look; the untouched, pixel-stable default for every HornLab renderer. |
| `dark` | Waveguide-Generator dark-axes variant of the Arctic palette; intentionally shares hornlab's values today. |
| `granite` | Calm, high-legibility warm-paper light theme for docs and posts; colorblind-distinct teal/orange lead pair. |
| `abyss` | Dark studio theme for long-session night work; cyan/amber lead pair stays separable for all common CVD types. |
| `blueprint` | Technical-drawing homage on drafting blue; presentation flair with the white primary trace as the pencil line. |
| `journal` | Publication print-first theme; grayscale-ordered leads keep a mono print of a 3-curve chart readable. |
| `contrast` | WCAG high-contrast theme for projection, low vision, and grayscale-safe output; 21:1 ink contrast. |
| `sepia` | Solarized-warm low-glare reading theme for long sessions and printed reports. |
| `phosphor` | CRT oscilloscope instrument aesthetic; green-screen lab-bench mood with luminance-separated leads. |
| `ember` | Warm charcoal studio wildcard; steel-blue vs stoked-ember lead pair, warm/cool alternating cycle. |

Previews for every theme live in the [theme gallery](docs/themes/README.md)
(`docs/themes/<name>.png`), regenerated with:

```bash
python scripts/render_theme_gallery.py
```

The designed themes' grid linestyle/weight ride in `Theme.rc_params`
(`grid.linestyle` / `grid.linewidth`); apply them with
`matplotlib.rc_context(theme.matplotlib_rc_params())` when composing your
own figures. The heatmap dB window stays a data decision (`MIN_DB`/`MAX_DB`),
never a per-theme style decision.

## Installation

```bash
pip install "hornlab-plots @ git+https://github.com/m3gnus/hornlab-plots.git"

# Local development checkout:
git clone https://github.com/m3gnus/hornlab-plots.git
cd hornlab-plots
pip install -e ".[dev]"
```

The package is not currently published to PyPI. During local HornLab
development, install it into each consumer venv from the workspace checkout.

## Migration status

Within HornLab, this package is the canonical plot renderer. The legacy WG
modules at `Waveguide-Generator/server/solver/directivity_plot.py` and
`server/solver/charts.py` were removed on 2026-05-23 after byte-identical
gate evidence.

## License

AGPL-3.0-or-later
