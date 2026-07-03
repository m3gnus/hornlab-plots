"""Theme registry tests: every built-in theme loads, parses, and resolves.

Color assertions go through ``matplotlib.colors`` parseability on purpose:
the legacy ``hornlab`` theme stores named colors (``grid_color="white"``),
so literal-hex checks would wrongly fail it.
"""

from __future__ import annotations

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pytest
from matplotlib.colors import Colormap

import hornlab_plots as hlp
from hornlab_plots import style

EXPECTED_THEME_NAMES = {
    "hornlab",
    "dark",
    "granite",
    "abyss",
    "blueprint",
    "journal",
    "contrast",
    "sepia",
    "phosphor",
    "ember",
}

THEME_NAMES = sorted(style.BUILTIN_THEMES)

# Field-by-field pins from the design specs (themes-fable.md / themes-opus.md):
# figure_bg, axes_bg, ink, grid color, grid (linestyle, linewidth),
# cycle, heatmap colormap name, accent, dpi.
SPEC_THEMES = {
    "granite": {
        "figure_bg": "#FAFAF7",
        "axes_bg": "#FFFFFF",
        "ink": "#1A1D21",
        "grid_color": "#D8D8D3",
        "grid_style": ("--", 0.6),
        "cycle": ("#00696D", "#C4501E", "#3B4CC0", "#5B8C2A", "#7B2D8B", "#556270"),
        "heatmap": "viridis",
        "accent": "#C4501E",
        "dpi": 160,
    },
    "abyss": {
        "figure_bg": "#0E1116",
        "axes_bg": "#161B22",
        "ink": "#E6EDF3",
        "grid_color": "#2D333B",
        "grid_style": ("-", 0.5),
        "cycle": ("#33C3DC", "#FFB454", "#3FB950", "#DB61A2", "#7B9DFF", "#D2A86A"),
        "heatmap": "magma",
        "accent": "#33C3DC",
        "dpi": 160,
    },
    "blueprint": {
        "figure_bg": "#10243E",
        "axes_bg": "#16304F",
        "ink": "#DCE9F5",
        "grid_color": "#2E4A6B",
        "grid_style": ("-", 0.4),
        "cycle": ("#F2F7FC", "#7FD4FF", "#FFD966", "#FF8A7A", "#8AE0C3", "#C3B5FF"),
        "heatmap": "cividis",
        "accent": "#FFD966",
        "dpi": 160,
    },
    "journal": {
        "figure_bg": "#FFFFFF",
        "axes_bg": "#FFFFFF",
        "ink": "#111111",
        "grid_color": "#CCCCCC",
        "grid_style": (":", 0.5),
        "cycle": ("#000000", "#666666", "#AAAAAA", "#B03A2E", "#1F618D", "#117A65"),
        "heatmap": "gray_r",
        "accent": "#B03A2E",
        "dpi": 300,
    },
    "contrast": {
        "figure_bg": "#FFFFFF",
        "axes_bg": "#FFFFFF",
        "ink": "#000000",
        "grid_color": "#4D4D4D",
        "grid_style": ("-", 0.9),
        "cycle": ("#005AB5", "#DC3220", "#000000", "#785EF0", "#FFB000", "#1AA179"),
        "heatmap": "cividis",
        "accent": "#DC3220",
        "dpi": 150,
    },
    "sepia": {
        "figure_bg": "#FDF6E3",
        "axes_bg": "#FBF1DA",
        "ink": "#586E75",
        "grid_color": "#93A1A1",
        "grid_style": (":", 0.7),
        "cycle": ("#268BD2", "#CB4B16", "#859900", "#6C71C4", "#B58900", "#2AA198"),
        "heatmap": "magma",
        "accent": "#CB4B16",
        "dpi": 140,
    },
    "phosphor": {
        "figure_bg": "#0A140F",
        "axes_bg": "#0D1A12",
        "ink": "#7DFFB0",
        "grid_color": "#1F7A4D",
        "grid_style": ("-", 0.6),
        "cycle": ("#00E676", "#FFB300", "#40C4FF", "#FF6E9C", "#B388FF", "#EEFF41"),
        "heatmap": "viridis",
        "accent": "#00E676",
        "dpi": 130,
    },
    "ember": {
        "figure_bg": "#1A1614",
        "axes_bg": "#221C19",
        "ink": "#E8DDD3",
        "grid_color": "#4A403A",
        "grid_style": ("--", 0.6),
        "cycle": ("#3B6EA5", "#D98324", "#5FB49C", "#C77DFF", "#E8C547", "#E5657A"),
        "heatmap": "inferno",
        "accent": "#D98324",
        "dpi": 140,
    },
}


def _iter_theme_colors(theme):
    """Yield every color value a Theme carries (scalar fields + mappings)."""
    yield from (
        theme.figure_bg,
        theme.axes_bg,
        theme.text_color,
        theme.tick_color,
        theme.spine_color,
        theme.grid_color,
        theme.contour_outline,
        theme.reference_contour_color,
        theme.mesh_limit_color,
    )
    yield from theme.palette
    yield from theme.line_colors
    yield from theme.role_colors.values()
    yield from theme.response_colors.values()
    yield from theme.plane_colors.values()
    yield from theme.impedance_colors.values()


def test_all_expected_themes_registered():
    assert set(style.BUILTIN_THEMES) == EXPECTED_THEME_NAMES
    assert hlp.BUILTIN_THEMES is style.BUILTIN_THEMES
    for name in EXPECTED_THEME_NAMES:
        theme = style.get_theme(name)
        assert isinstance(theme, style.Theme)
        assert theme.name == name
        assert style.resolve_theme(name) is theme


@pytest.mark.parametrize("name", THEME_NAMES)
def test_theme_colors_parse_via_matplotlib(name):
    """Every color must be matplotlib-parseable (hex or named, e.g. "white")."""
    theme = style.get_theme(name)
    for color in _iter_theme_colors(theme):
        rgb = mcolors.to_rgb(color)  # raises ValueError on junk
        assert len(rgb) == 3
        assert mcolors.to_hex(color).startswith("#")


@pytest.mark.parametrize("name", THEME_NAMES)
def test_theme_cycle_has_at_least_six_colors(name):
    theme = style.get_theme(name)
    assert len(theme.line_colors) >= 6
    assert len(theme.palette) >= 6


@pytest.mark.parametrize("name", THEME_NAMES)
def test_theme_colormaps_are_colormaps(name):
    theme = style.get_theme(name)
    for cmap in (theme.heatmap_cmap, theme.diverging_cmap, theme.cyclic_cmap):
        assert isinstance(cmap, Colormap)


@pytest.mark.parametrize("name", sorted(SPEC_THEMES))
def test_spec_theme_colormap_names_resolve(name):
    """Spec themes use stock colormaps; their names must resolve through the
    registry. (Legacy hornlab/dark use custom unregistered Arctic maps.)"""
    theme = style.get_theme(name)
    assert theme.heatmap_cmap.name == SPEC_THEMES[name]["heatmap"]
    for cmap in (theme.heatmap_cmap, theme.diverging_cmap, theme.cyclic_cmap):
        assert cmap.name in matplotlib.colormaps
        assert isinstance(matplotlib.colormaps[cmap.name], Colormap)


@pytest.mark.parametrize("name", sorted(SPEC_THEMES))
def test_spec_theme_fields_match_design_specs(name):
    theme = style.get_theme(name)
    spec = SPEC_THEMES[name]
    assert theme.figure_bg == spec["figure_bg"]
    assert theme.axes_bg == spec["axes_bg"]
    assert theme.text_color == spec["ink"]
    assert theme.tick_color == spec["ink"]
    assert theme.spine_color == spec["ink"]
    assert theme.grid_color == spec["grid_color"]
    assert theme.line_colors == spec["cycle"]
    assert theme.palette == spec["cycle"]
    assert theme.reference_contour_color == spec["accent"]
    assert theme.role_colors["accent"] == spec["accent"]
    assert theme.dpi == spec["dpi"]
    linestyle, linewidth = spec["grid_style"]
    assert theme.rc_params["grid.linestyle"] == linestyle
    assert theme.rc_params["grid.linewidth"] == pytest.approx(linewidth)
    assert theme.matplotlib_rc_params()["grid.linestyle"] == linestyle
    # Cycle order feeds the role colors: the CVD-safe lead pair carries
    # lf/mf, H/V, and real/imag.
    assert theme.response_colors["lf"] == spec["cycle"][0]
    assert theme.response_colors["mf"] == spec["cycle"][1]
    assert theme.response_colors["hf"] == spec["cycle"][2]
    assert theme.plane_colors["horizontal"] == spec["cycle"][0]
    assert theme.impedance_colors["real"] == spec["cycle"][0]


def test_default_theme_is_hornlab_with_unchanged_style_values():
    """Parity gate: registering new themes must not move the default."""
    assert style.get_theme().name == "hornlab"
    theme = style.HORNLAB_THEME
    assert style.BUILTIN_THEMES["hornlab"] is theme
    assert theme.figure_bg == "#080D16"
    assert theme.axes_bg == "#0F1927"
    assert theme.text_color == "#C8D8EC"
    assert theme.tick_color == "#6A90B8"
    assert theme.spine_color == "#1E3050"
    assert theme.grid_color == "white"
    assert theme.contour_outline == "#050C18"
    assert theme.reference_contour_color == "#E8A83A"
    assert theme.mesh_limit_color == "#FF5A5A"
    assert theme.primary_grid_alpha == pytest.approx(0.22)
    assert theme.secondary_grid_alpha == pytest.approx(0.08)
    assert theme.grid_alpha == pytest.approx(0.15)
    assert theme.grid_alpha_minor == pytest.approx(0.07)
    assert theme.fill_alpha == pytest.approx(0.18)
    assert theme.dpi == 200
    assert theme.heatmap_cmap.name == "waveguide_arctic"
    assert theme.diverging_cmap.name == "arctic_diverging"
    assert theme.cyclic_cmap.name == "arctic_cyclic"
    assert dict(theme.plane_colors) == {
        "horizontal": "#81c784",
        "vertical": "#64b5f6",
        "diagonal": "#ffb74d",
    }
    assert dict(theme.impedance_colors) == {
        "real": "#64b5f6",
        "imaginary": "#ffb74d",
    }
    assert dict(theme.rc_params) == {}
    # Legacy module constants stay pinned to hornlab.
    assert style.FIGURE_BG == theme.figure_bg
    assert style.GRID_COLOR == "white"
    assert style.DPI == 200


def test_default_render_is_pixel_identical_to_explicit_hornlab():
    """Parity gate: a default-theme render must byte-match theme="hornlab"."""
    freqs = np.geomspace(100.0, 20000.0, 24)
    spl = 100.0 + 2.0 * np.sin(np.log10(freqs))
    default_png = hlp.frequency_response_b64(freqs, spl)
    hornlab_png = hlp.frequency_response_b64(freqs, spl, theme="hornlab")
    assert default_png == hornlab_png


def test_unknown_theme_error_lists_registered_names():
    with pytest.raises(ValueError, match="granite"):
        style.get_theme("no-such-theme")
