#!/usr/bin/env python3
"""Render one preview PNG per registered HornLab plot theme.

For every theme in ``hornlab_plots.style.BUILTIN_THEMES`` this renders a
two-panel figure — a synthetic frequency-response chart (three curves from
the theme's cycle) beside a small synthetic directivity heatmap (the theme's
heatmap colormap) — to ``docs/themes/<name>.png`` at the theme's DPI, then
writes ``docs/themes/README.md`` with an index table.

Each theme's ``rc_params`` (spec'd grid linestyle/weight) are applied via
``matplotlib.rc_context`` so the previews honor the designed grid styles.

Usage:
    python scripts/render_theme_gallery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hornlab_plots.style import BUILTIN_THEMES, MAX_DB, MIN_DB, Theme  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "themes"

# One-line intents, taken from the design specs (Fable + Opus sets) and the
# style-module docstring for the two legacy themes.
THEME_INTENTS: dict[str, str] = {
    "hornlab": "Canonical Arctic Night look; the untouched, pixel-stable default for every HornLab renderer.",
    "dark": "Waveguide-Generator dark-axes variant of the Arctic palette; intentionally shares hornlab's values today.",
    "granite": "Calm, high-legibility warm-paper light theme for docs and posts; colorblind-distinct teal/orange lead pair.",
    "abyss": "Dark studio theme for long-session night work; cyan/amber lead pair stays separable for all common CVD types.",
    "blueprint": "Technical-drawing homage on drafting blue; presentation flair with the white primary trace as the pencil line.",
    "journal": "Publication print-first theme; grayscale-ordered leads keep a mono print of a 3-curve chart readable.",
    "contrast": "WCAG high-contrast theme for projection, low vision, and grayscale-safe output; 21:1 ink contrast.",
    "sepia": "Solarized-warm low-glare reading theme for long sessions and printed reports.",
    "phosphor": "CRT oscilloscope instrument aesthetic; green-screen lab-bench mood with luminance-separated leads.",
    "ember": "Warm charcoal studio wildcard; steel-blue vs stoked-ember lead pair, warm/cool alternating cycle.",
}


def _synthetic_frequency_response() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LF lowpass / HF highpass / combined triple around an 800 Hz crossover."""
    freqs = np.geomspace(100.0, 20000.0, 240)
    lf = 100.0 - 20.0 * np.log10(1.0 + (freqs / 800.0) ** 2)
    hf = 100.0 - 20.0 * np.log10(1.0 + (800.0 / freqs) ** 2)
    combined = 20.0 * np.log10(10.0 ** (lf / 20.0) + 10.0 ** (hf / 20.0))
    ripple = 0.8 * np.sin(3.0 * np.log10(freqs))
    return freqs, lf + ripple, hf + ripple, combined + ripple


def _synthetic_directivity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Beamwidth narrowing with frequency plus a soft off-axis HF lobe."""
    freqs = np.geomspace(200.0, 16000.0, 160)
    angles = np.linspace(-90.0, 90.0, 121)
    octaves = np.log2(freqs / freqs[0])
    sigma = np.clip(65.0 - 6.5 * octaves, 24.0, None)
    spl = -20.0 * (1.0 - np.exp(-(angles[:, None] ** 2) / (2.0 * sigma[None, :] ** 2)))
    lobe = 3.0 * np.exp(-((np.abs(angles[:, None]) - 50.0) ** 2) / 250.0)
    spl += lobe * np.clip(octaves[None, :] - 3.0, 0.0, None) / 3.0
    return freqs, angles, np.clip(spl, MIN_DB, MAX_DB)


def _style_axes(ax: plt.Axes, theme: Theme) -> None:
    ax.set_facecolor(theme.axes_bg)
    ax.tick_params(colors=theme.tick_color, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(theme.spine_color)


_FREQ_TICKS = (200, 500, 1000, 2000, 5000, 10000, 20000)


def _freq_ticks(ax: plt.Axes, lo: float, hi: float) -> None:
    ticks = [t for t in _FREQ_TICKS if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks])
    ax.minorticks_off()


def render_theme_preview(theme: Theme, out_path: Path) -> None:
    """Render the two-panel preview for one theme."""
    freqs, lf, hf, combined = _synthetic_frequency_response()
    hfreqs, angles, spl = _synthetic_directivity()

    with matplotlib.rc_context(theme.matplotlib_rc_params()):
        fig, (ax_fr, ax_map) = plt.subplots(
            1, 2, figsize=(9.2, 3.4), width_ratios=(1.25, 1.0)
        )
        fig.patch.set_facecolor(theme.figure_bg)

        # Frequency response: three curves from the theme's cycle.
        cycle = theme.line_colors
        ax_fr.semilogx(freqs, lf, color=cycle[0], linewidth=1.5, label="LF LR4 LP")
        ax_fr.semilogx(freqs, hf, color=cycle[1], linewidth=1.5, label="HF LR4 HP")
        ax_fr.semilogx(freqs, combined, color=cycle[2], linewidth=2.0, label="Combined")
        ax_fr.set_xlim(freqs[0], freqs[-1])
        ax_fr.set_ylim(72.0, 108.0)
        _freq_ticks(ax_fr, freqs[0], freqs[-1])
        _style_axes(ax_fr, theme)
        ax_fr.grid(True, which="major", color=theme.grid_color, alpha=theme.grid_alpha)
        ax_fr.set_xlabel("Frequency [Hz]", color=theme.text_color, fontsize=9)
        ax_fr.set_ylabel("SPL [dB]", color=theme.text_color, fontsize=9)
        ax_fr.set_title("Frequency response", color=theme.text_color, fontsize=10)
        ax_fr.legend(
            loc="lower center",
            fontsize=7,
            facecolor=theme.axes_bg,
            edgecolor=theme.spine_color,
            labelcolor=theme.text_color,
        )

        # Directivity heatmap: the theme's colormap over the canonical window.
        mesh = ax_map.pcolormesh(
            hfreqs,
            angles,
            spl,
            cmap=theme.heatmap_cmap,
            vmin=MIN_DB,
            vmax=MAX_DB,
            shading="auto",
        )
        ax_map.set_xscale("log")
        ax_map.set_xlim(hfreqs[0], hfreqs[-1])
        _freq_ticks(ax_map, hfreqs[0], hfreqs[-1])
        _style_axes(ax_map, theme)
        ax_map.set_xlabel("Frequency [Hz]", color=theme.text_color, fontsize=9)
        ax_map.set_ylabel("Angle [deg]", color=theme.text_color, fontsize=9)
        ax_map.set_title("Directivity", color=theme.text_color, fontsize=10)
        cbar = fig.colorbar(mesh, ax=ax_map, pad=0.02)
        cbar.set_label("dB", color=theme.text_color, fontsize=8)
        cbar.ax.tick_params(colors=theme.tick_color, labelsize=7)
        cbar.outline.set_edgecolor(theme.spine_color)

        fig.suptitle(theme.name, color=theme.text_color, fontsize=12, fontweight="600")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        fig.savefig(
            out_path,
            dpi=theme.dpi,
            facecolor=theme.figure_bg,
            bbox_inches="tight",
            pad_inches=0.2,
        )
        plt.close(fig)


def write_index(names: list[str]) -> Path:
    lines = [
        "# HornLab plot theme gallery",
        "",
        "Generated by `scripts/render_theme_gallery.py` — rerun it after theme",
        "changes to refresh the previews. Each preview pairs a synthetic",
        "frequency-response chart (three curves from the theme's cycle) with a",
        "synthetic directivity heatmap (the theme's colormap) at the theme's DPI.",
        "",
        "| Theme | Intent | Preview |",
        "| --- | --- | --- |",
    ]
    for name in names:
        intent = THEME_INTENTS.get(name, "(no intent recorded)")
        lines.append(f"| `{name}` | {intent} | [{name}.png]({name}.png) |")
    lines.append("")
    for name in names:
        lines.extend([f"## {name}", "", f"![{name}]({name}.png)", ""])
    out = OUT_DIR / "README.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = list(BUILTIN_THEMES)
    for name in names:
        out_path = OUT_DIR / f"{name}.png"
        render_theme_preview(BUILTIN_THEMES[name], out_path)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    index = write_index(names)
    print(f"wrote {index.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
