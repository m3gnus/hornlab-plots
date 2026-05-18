"""Log-frequency tick + grid helpers shared between heatmap and line plots."""

from __future__ import annotations

import numpy as np


def log_grid_lines(freq_min, freq_max):
    """Generate frequencies at decade sub-boundaries: 1, 2, 3, 5 x 10^n."""
    min_log = np.log10(freq_min)
    max_log = np.log10(freq_max)
    lines = []
    for decade in range(int(np.floor(min_log)), int(np.ceil(max_log)) + 1):
        for mantissa in [1, 2, 3, 5]:
            freq = mantissa * (10 ** decade)
            if freq_min <= freq <= freq_max:
                lines.append(freq)
    return sorted(set(lines))


def preferred_frequency_ticks(freq_min, freq_max):
    """
    Build denser frequency ticks for directivity charts.

    Requested density:
    - every 100 Hz between 100 and 1000
    - every 1 kHz between 1 kHz and 10 kHz
    """
    if freq_max <= freq_min:
        return []

    ticks = []
    ticks.extend(_linear_tick_range(freq_min, freq_max, 100.0, 1000.0, 100.0))
    ticks.extend(_linear_tick_range(freq_min, freq_max, 1000.0, 10000.0, 1000.0))

    # Outside the requested ranges, retain sparse log boundaries for orientation.
    if freq_min < 100.0:
        ticks.extend(log_grid_lines(freq_min, min(freq_max, 100.0)))
    if freq_max > 10000.0:
        ticks.extend(log_grid_lines(max(freq_min, 10000.0), freq_max))

    if not ticks:
        return log_grid_lines(freq_min, freq_max)

    return sorted({round(float(tick), 6) for tick in ticks})


def _linear_tick_range(freq_min, freq_max, domain_min, domain_max, step):
    lo = max(float(freq_min), float(domain_min))
    hi = min(float(freq_max), float(domain_max))
    if hi < lo:
        return []
    start = np.ceil(lo / step) * step
    if start > hi + (step * 1e-9):
        return []
    return list(np.arange(start, hi + (step * 0.5), step))


def contains_frequency(freqs, target):
    for freq in freqs:
        if np.isclose(freq, target, rtol=1e-6, atol=1e-6):
            return True
    return False


def freq_formatter(x, pos):
    """Format frequency ticks: 100, 200, 1k, 2k, 10k, 20k."""
    if x >= 1000:
        if x % 1000 == 0:
            return f"{int(x / 1000)}k"
        return f"{x / 1000:.1f}k"
    return f"{int(x)}"
