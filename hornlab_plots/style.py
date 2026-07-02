"""Canonical theme API and backwards-compatible style constants.

``hornlab`` is the canonical HornLab plotting look and preserves the former
module-level constants exactly. ``dark`` is a distinct built-in Theme modeled
on the Waveguide Generator dark axes with the Arctic palette; it intentionally
shares the current values until WG needs a separate dark variant.

The legacy constants remain importable for existing callers. New renderers
should resolve a :class:`Theme` with ``get_theme`` / ``apply_theme_overrides``
instead of importing constants directly.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from matplotlib.colors import Colormap, LinearSegmentedColormap


# Heatmap scale + smoothing defaults --------------------------------------
MIN_DB = -30.0
MAX_DB = 0.0
FRACTIONAL_OCTAVE = 24.0
ANGLE_SAMPLES = 361
FREQ_SAMPLES = 500


_FIGURE_BG = "#080D16"   # --scene-bg dark
_AXES_BG = "#0F1927"     # --surface dark
_TEXT_COLOR = "#C8D8EC"  # --text dark (slightly dimmer for print)
_TICK_COLOR = "#6A90B8"  # --text-sec dark
_SPINE_COLOR = "#1E3050"  # --border dark
_GRID_COLOR = "white"
_PRIMARY_GRID_ALPHA = 0.22
_SECONDARY_GRID_ALPHA = 0.08

_CONTOUR_OUTLINE = "#050C18"
_REFERENCE_CONTOUR_COLOR = "#E8A83A"  # --accent-warm
_MESH_LIMIT_COLOR = "#FF5A5A"

_LINE_COLORS = ("#4FC3F7", "#E8A83A", "#EF5350", "#66BB6A", "#AB47BC", "#78909C")
_RESPONSE_COLORS = {
    "lf": "#4FC3F7",
    "mf": "#E8A83A",
    "hf": "#EF5350",
    "combined": "#EAF2FF",
    "raw": "#78909C",
    "other": "#AB47BC",
}
_PLANE_COLORS = {
    "horizontal": "#81c784",
    "vertical": "#64b5f6",
    "diagonal": "#ffb74d",
}
_IMPEDANCE_COLORS = {
    "real": "#64b5f6",
    "imaginary": "#ffb74d",
}

_GRID_ALPHA = 0.15
_GRID_ALPHA_MINOR = 0.07
_FILL_ALPHA = 0.18
_DPI = 200


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _heatmap_cmap() -> Colormap:
    return LinearSegmentedColormap.from_list(
        "waveguide_arctic",
        [
            "#050C18",
            "#0A1E36",
            "#0E3058",
            "#1A4A7A",
            "#2D6090",
            "#4D78A8",
            "#3A3A7A",
            "#5A3090",
            "#7A2060",
            "#A83040",
            "#C84428",
        ],
        N=256,
    )


def _diverging_cmap(axes_bg: str = _AXES_BG) -> Colormap:
    return LinearSegmentedColormap.from_list(
        "arctic_diverging",
        ["#1565C0", "#42A5F5", axes_bg, "#E8A83A", "#E65100"],
        N=256,
    )


def _cyclic_cmap() -> Colormap:
    return LinearSegmentedColormap.from_list(
        "arctic_cyclic",
        [
            "#1565C0",
            "#42A5F5",
            "#80DEEA",
            "#C8D8EC",
            "#E8A83A",
            "#EF6C00",
            "#B71C1C",
            "#6A1B9A",
            "#1565C0",
        ],
        N=256,
    )


@dataclass(frozen=True)
class Theme:
    """Complete styling surface for HornLab matplotlib renderers."""

    name: str
    figure_bg: str
    axes_bg: str
    text_color: str
    tick_color: str
    spine_color: str
    grid_color: str
    contour_outline: str
    reference_contour_color: str
    mesh_limit_color: str
    primary_grid_alpha: float
    secondary_grid_alpha: float
    grid_alpha: float
    grid_alpha_minor: float
    fill_alpha: float
    dpi: int
    palette: tuple[str, ...]
    role_colors: Mapping[str, str]
    line_colors: tuple[str, ...]
    response_colors: Mapping[str, str]
    plane_colors: Mapping[str, str]
    impedance_colors: Mapping[str, str]
    heatmap_cmap: Colormap
    diverging_cmap: Colormap
    cyclic_cmap: Colormap
    font_family: tuple[str, ...] = ()
    font_sans_serif: tuple[str, ...] = ()
    font_size: float | None = None
    rc_params: Mapping[str, Any] = MappingProxyType({})

    def matplotlib_rc_params(self) -> dict[str, Any]:
        """Return rcParams implied by font fields plus explicit overrides."""
        params = dict(self.rc_params)
        if self.font_family:
            params.setdefault("font.family", list(self.font_family))
        if self.font_sans_serif:
            params.setdefault("font.sans-serif", list(self.font_sans_serif))
        if self.font_size is not None:
            params.setdefault("font.size", self.font_size)
        return params


def _role_colors() -> dict[str, str]:
    return {
        "figure_bg": _FIGURE_BG,
        "background": _FIGURE_BG,
        "axes_bg": _AXES_BG,
        "axes": _AXES_BG,
        "text": _TEXT_COLOR,
        "tick": _TICK_COLOR,
        "spine": _SPINE_COLOR,
        "grid": _GRID_COLOR,
        "contour_outline": _CONTOUR_OUTLINE,
        "reference_contour": _REFERENCE_CONTOUR_COLOR,
        "mesh_limit": _MESH_LIMIT_COLOR,
    }


def _make_theme(name: str) -> Theme:
    return Theme(
        name=name,
        figure_bg=_FIGURE_BG,
        axes_bg=_AXES_BG,
        text_color=_TEXT_COLOR,
        tick_color=_TICK_COLOR,
        spine_color=_SPINE_COLOR,
        grid_color=_GRID_COLOR,
        contour_outline=_CONTOUR_OUTLINE,
        reference_contour_color=_REFERENCE_CONTOUR_COLOR,
        mesh_limit_color=_MESH_LIMIT_COLOR,
        primary_grid_alpha=_PRIMARY_GRID_ALPHA,
        secondary_grid_alpha=_SECONDARY_GRID_ALPHA,
        grid_alpha=_GRID_ALPHA,
        grid_alpha_minor=_GRID_ALPHA_MINOR,
        fill_alpha=_FILL_ALPHA,
        dpi=_DPI,
        palette=_LINE_COLORS,
        role_colors=_freeze_mapping(_role_colors()),
        line_colors=_LINE_COLORS,
        response_colors=_freeze_mapping(_RESPONSE_COLORS),
        plane_colors=_freeze_mapping(_PLANE_COLORS),
        impedance_colors=_freeze_mapping(_IMPEDANCE_COLORS),
        heatmap_cmap=_heatmap_cmap(),
        diverging_cmap=_diverging_cmap(),
        cyclic_cmap=_cyclic_cmap(),
        rc_params=MappingProxyType({}),
    )


HORNLAB_THEME = _make_theme("hornlab")
DARK_THEME = _make_theme("dark")
BUILTIN_THEMES: Mapping[str, Theme] = MappingProxyType(
    {
        HORNLAB_THEME.name: HORNLAB_THEME,
        DARK_THEME.name: DARK_THEME,
    }
)

_CURRENT_THEME: Theme = HORNLAB_THEME


def resolve_theme(theme: Theme | str | None = None) -> Theme:
    """Resolve ``None``, a built-in name, or a Theme object to a Theme."""
    if theme is None:
        return _CURRENT_THEME
    if isinstance(theme, Theme):
        return theme
    key = str(theme).strip().lower()
    try:
        return BUILTIN_THEMES[key]
    except KeyError as exc:
        names = ", ".join(sorted(BUILTIN_THEMES))
        raise ValueError(f"unknown HornLab plot theme {theme!r}; expected one of: {names}") from exc


def get_theme(theme: Theme | str | None = None) -> Theme:
    """Return the active Theme, or resolve a named/custom Theme when provided."""
    return resolve_theme(theme)


def set_theme(theme: Theme | str) -> Theme:
    """Set the active HornLab plot Theme and return it."""
    global _CURRENT_THEME
    _CURRENT_THEME = resolve_theme(theme)
    return _CURRENT_THEME


@contextmanager
def theme(theme: Theme | str) -> Iterator[Theme]:
    """Temporarily set the active Theme, restoring the previous Theme always."""
    previous = get_theme()
    active = set_theme(theme)
    try:
        yield active
    finally:
        set_theme(previous)


theme_context = theme


_COLOR_FIELD_ALIASES = {
    "figure_bg": "figure_bg",
    "background": "figure_bg",
    "axes_bg": "axes_bg",
    "axes": "axes_bg",
    "text": "text_color",
    "text_color": "text_color",
    "tick": "tick_color",
    "tick_color": "tick_color",
    "spine": "spine_color",
    "spine_color": "spine_color",
    "grid": "grid_color",
    "grid_color": "grid_color",
    "contour_outline": "contour_outline",
    "reference_contour": "reference_contour_color",
    "reference_contour_color": "reference_contour_color",
    "mesh_limit": "mesh_limit_color",
    "mesh_limit_color": "mesh_limit_color",
}


def apply_theme_overrides(
    theme: Theme | str | None = None,
    *,
    colors: Mapping[str, str] | None = None,
    line_colors: tuple[str, ...] | list[str] | None = None,
    response_colors: Mapping[str, str] | None = None,
) -> Theme:
    """Resolve a Theme and return a copy with per-call color overrides."""
    theme_obj = resolve_theme(theme)
    updates: dict[str, Any] = {}
    role_colors = dict(theme_obj.role_colors)

    for raw_key, value in (colors or {}).items():
        key = str(raw_key).strip()
        field_name = _COLOR_FIELD_ALIASES.get(key)
        if field_name is None:
            role_colors[key] = value
            continue
        updates[field_name] = value
        role_colors[key] = value

    if line_colors is not None:
        normalized_line_colors = tuple(line_colors)
        if normalized_line_colors:
            updates["line_colors"] = normalized_line_colors
            updates["palette"] = normalized_line_colors
    if response_colors is not None:
        merged = dict(theme_obj.response_colors)
        merged.update(response_colors)
        updates["response_colors"] = _freeze_mapping(merged)

    if not updates and role_colors == dict(theme_obj.role_colors):
        return theme_obj

    for role_key, field_name in _COLOR_FIELD_ALIASES.items():
        if field_name in updates:
            role_colors[role_key] = updates[field_name]
    updates["role_colors"] = _freeze_mapping(role_colors)
    return replace(theme_obj, **updates)


# Backwards-compatible constants. These are intentionally fixed to hornlab
# defaults; they do not track set_theme().
FIGURE_BG = HORNLAB_THEME.figure_bg
AXES_BG = HORNLAB_THEME.axes_bg
TEXT_COLOR = HORNLAB_THEME.text_color
TICK_COLOR = HORNLAB_THEME.tick_color
SPINE_COLOR = HORNLAB_THEME.spine_color
GRID_COLOR = HORNLAB_THEME.grid_color
PRIMARY_GRID_ALPHA = HORNLAB_THEME.primary_grid_alpha
SECONDARY_GRID_ALPHA = HORNLAB_THEME.secondary_grid_alpha

HEATMAP_CMAP = HORNLAB_THEME.heatmap_cmap

CONTOUR_OUTLINE = HORNLAB_THEME.contour_outline
REFERENCE_CONTOUR_COLOR = HORNLAB_THEME.reference_contour_color
MESH_LIMIT_COLOR = HORNLAB_THEME.mesh_limit_color

DIVERGING_CMAP = HORNLAB_THEME.diverging_cmap
CYCLIC_CMAP = HORNLAB_THEME.cyclic_cmap

LINE_COLORS = list(HORNLAB_THEME.line_colors)
RESPONSE_COLORS = dict(HORNLAB_THEME.response_colors)

GRID_ALPHA = HORNLAB_THEME.grid_alpha
GRID_ALPHA_MINOR = HORNLAB_THEME.grid_alpha_minor
FILL_ALPHA = HORNLAB_THEME.fill_alpha
DPI = HORNLAB_THEME.dpi


__all__ = [
    "Theme",
    "HORNLAB_THEME",
    "DARK_THEME",
    "BUILTIN_THEMES",
    "get_theme",
    "set_theme",
    "theme",
    "theme_context",
    "resolve_theme",
    "apply_theme_overrides",
    "MIN_DB",
    "MAX_DB",
    "FRACTIONAL_OCTAVE",
    "ANGLE_SAMPLES",
    "FREQ_SAMPLES",
    "FIGURE_BG",
    "AXES_BG",
    "TEXT_COLOR",
    "TICK_COLOR",
    "SPINE_COLOR",
    "GRID_COLOR",
    "PRIMARY_GRID_ALPHA",
    "SECONDARY_GRID_ALPHA",
    "HEATMAP_CMAP",
    "CONTOUR_OUTLINE",
    "REFERENCE_CONTOUR_COLOR",
    "MESH_LIMIT_COLOR",
    "DIVERGING_CMAP",
    "CYCLIC_CMAP",
    "LINE_COLORS",
    "RESPONSE_COLORS",
    "GRID_ALPHA",
    "GRID_ALPHA_MINOR",
    "FILL_ALPHA",
    "DPI",
]
