"""Canonical Arctic Night theme constants for HornLab plots.

These are the load-bearing styling tokens that every canonical plot
shares. Callers composing figures around `directivity_heatmap` panels
(BIGMEH stacked plots, complex-pressure analyses) import these to keep
the surrounding chrome consistent with the heatmap inside.

Kept as plain module-level constants (not a config object) so they can
be imported as `from hornlab_plots.style import FIGURE_BG, AXES_BG`.
"""

from matplotlib.colors import LinearSegmentedColormap


# Heatmap scale + smoothing defaults --------------------------------------
MIN_DB = -30.0
MAX_DB = 0.0
FRACTIONAL_OCTAVE = 24.0
ANGLE_SAMPLES = 361
FREQ_SAMPLES = 500


# Arctic Night palette — matches the WG app dark mode ---------------------
FIGURE_BG = "#080D16"   # --scene-bg dark
AXES_BG = "#0F1927"     # --surface dark
TEXT_COLOR = "#C8D8EC"  # --text dark (slightly dimmer for print)
TICK_COLOR = "#6A90B8"  # --text-sec dark
SPINE_COLOR = "#1E3050"  # --border dark
GRID_COLOR = "white"
PRIMARY_GRID_ALPHA = 0.22
SECONDARY_GRID_ALPHA = 0.08


# Heatmap colormap: deep arctic blue → indigo → warm amber at top ---------
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "waveguide_arctic",
    ["#050C18", "#0A1E36", "#0E3058", "#1A4A7A", "#2D6090", "#4D78A8",
     "#3A3A7A", "#5A3090", "#7A2060", "#A83040", "#C84428"],
    N=256,
)


# Contour styling ---------------------------------------------------------
CONTOUR_OUTLINE = "#050C18"
REFERENCE_CONTOUR_COLOR = "#E8A83A"  # --accent-warm

# Mesh-valid frequency marker: results above this frequency are under-resolved
# and increasingly inaccurate. Warm red, distinct from the amber reference.
MESH_LIMIT_COLOR = "#FF5A5A"


# Complex-pressure diverging / cyclic palettes ----------------------------
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "arctic_diverging",
    ["#1565C0", "#42A5F5", AXES_BG, "#E8A83A", "#E65100"],
    N=256,
)

CYCLIC_CMAP = LinearSegmentedColormap.from_list(
    "arctic_cyclic",
    ["#1565C0", "#42A5F5", "#80DEEA", "#C8D8EC",
     "#E8A83A", "#EF6C00", "#B71C1C", "#6A1B9A", "#1565C0"],
    N=256,
)


# Line palette for multi-trace plots --------------------------------------
LINE_COLORS = ["#4FC3F7", "#E8A83A", "#EF5350", "#66BB6A", "#AB47BC", "#78909C"]
RESPONSE_COLORS = {
    "lf": "#4FC3F7",
    "mf": "#E8A83A",
    "hf": "#EF5350",
    "combined": "#EAF2FF",
    "raw": "#78909C",
    "other": "#AB47BC",
}


# Common axis grid alpha settings -----------------------------------------
GRID_ALPHA = 0.15
GRID_ALPHA_MINOR = 0.07
FILL_ALPHA = 0.18
DPI = 200
