"""PNG output shared by every renderer: base64 for HTTP routes, files on disk.

Both helpers save with the figure's own facecolor and no edge, then close the
figure. ``bbox_inches=None`` keeps the full figure canvas.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _write_png(fig, target, *, dpi, bbox_inches):
    kwargs = {} if bbox_inches is None else {"bbox_inches": bbox_inches}
    fig.savefig(
        target,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        **kwargs,
    )
    plt.close(fig)


def fig_to_b64(fig, dpi=150, *, bbox_inches="tight"):
    """Encode ``fig`` as a base64 PNG string (no data-URI prefix) and close it."""
    buf = io.BytesIO()
    _write_png(fig, buf, dpi=dpi, bbox_inches=bbox_inches)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_png(fig, output_path, *, dpi, bbox_inches="tight"):
    """Write ``fig`` to ``output_path`` as PNG, creating parents, and close it."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_png(fig, str(out), dpi=dpi, bbox_inches=bbox_inches)
    return out
