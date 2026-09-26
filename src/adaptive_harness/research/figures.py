"""Render the vector figures embedded in the paper.

Everything here is deterministic and seeded: the figures must be reproducible
from the same receipts, so a reader can re-run this and get identical SVGs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]


def _axes(width: int, height: int, margin: int, xlabel: str, ylabel: str) -> list[str]:
    left, right = margin, width - margin
    top, bottom = margin, height - margin
    return [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#11111b" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#11111b" stroke-width="1.2"/>',
        f'<text x="{(left + right) // 2}" y="{height - 8}" font-size="13" text-anchor="middle" '
        f'fill="#11111b">{xlabel}</text>',
        f'<text x="16" y="{(top + bottom) // 2}" font-size="13" text-anchor="middle" fill="#11111b" '
        f'transform="rotate(-90 16 {(top + bottom) // 2})">{ylabel}</text>',
        (left, right, top, bottom),
    ]


def variance_divergence_svg(path: str | Path, *, width: int = 620, height: int = 400,
                            margin: int = 62) -> Path:
    """The critical index: variance of a Pareto delay versus its tail index.

    This is the figure that carries the paper's central negative result — that
    the routing objective only exists above alpha = 2.
    """
    alphas = np.linspace(2.02, 6.0, 400)
    with np.errstate(divide="ignore", invalid="ignore"):
        theory = alphas / ((alphas - 1) ** 2 * (alphas - 2))
    cap = float(np.nanmax(theory[~np.isinf(theory)]))
    theory = np.minimum(theory, cap * 1.05)

    out = Path(path)
    lines = _svg_header(width, height)
    axis = _axes(width, height, margin, "tail index alpha", "variance of one delay")
    left, right, top, bottom = axis[-1]
    axis = axis[:-1]
    plot_h = bottom - top

    def sx(value: float) -> float:
        return left + (value - alphas[0]) / (alphas[-1] - alphas[0]) * (right - left)

    def sy(value: float) -> float:
        # log scale: the curve spans orders of magnitude near the pole
        lo, hi = 0.0, np.log10(cap * 1.05)
        safe = max(float(value), 1e-3)
        return bottom - (np.log10(safe) - lo) / (hi - lo) * plot_h

    for decade in range(int(np.log10(cap * 1.05)) + 1):
        value = 10.0 ** decade
        y = sy(value)
        if top <= y <= bottom:
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                         f'stroke="#e5e5ef" stroke-width="1"/>')
            lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
                         f'fill="#55556b">1e{decade}</text>')

    points = " ".join(f"{sx(a):.1f},{sy(v):.1f}" for a, v in zip(alphas, theory))
    lines.append(f'<polyline points="{points}" fill="none" stroke="#1e66f5" stroke-width="2.2"/>')

    critical_x = sx(2.0)
    lines.append(f'<line x1="{critical_x:.1f}" y1="{top}" x2="{critical_x:.1f}" y2="{bottom}" '
                 f'stroke="#c01c28" stroke-width="1.6" stroke-dasharray="5,4"/>')
    lines.append(f'<text x="{critical_x + 6:.1f}" y="{top + 16}" font-size="12" fill="#c01c28">'
                 f'alpha_c = 2 (infinite variance)</text>')
    lines.append(f'<text x="{right}" y="{bottom - 12}" font-size="11" text-anchor="end" '
                 f'fill="#55556b">Var = alpha / ((alpha-1)^2 (alpha-2))</text>')
    lines.append("</svg>")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def routing_variance_svg(path: str | Path, *, width: int = 620, height: int = 400,
                         margin: int = 62) -> Path:
    """Batch variance under the three load splits compared in EXP-002."""
    labels = ["balanced", "moderate", "greedy"]
    values = [105.576056, 3834.981492, 203317.643744]
    colors = ["#1e66f5", "#e8a33d", "#c01c28"]

    out = Path(path)
    lines = _svg_header(width, height)
    axis = _axes(width, height, margin, "routing policy", "batch variance (log scale)")
    left, right, top, bottom = axis[-1]
    axis = axis[:-1]
    plot_h = bottom - top
    ceiling = max(values) * 1.6
    lo, hi = 0.0, np.log10(ceiling)

    def sy(value: float) -> float:
        return bottom - (np.log10(max(value, 1.0)) - lo) / (hi - lo) * plot_h

    for decade in range(int(np.log10(ceiling)) + 1):
        y = sy(10.0 ** decade)
        if top <= y <= bottom:
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                         f'stroke="#e5e5ef" stroke-width="1"/>')
            lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
                         f'fill="#55556b">1e{decade}</text>')

    slot = (right - left) / len(labels)
    bar_w = slot * 0.42
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        centre = left + slot * (index + 0.5)
        y = sy(value)
        lines.append(f'<rect x="{centre - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                     f'height="{bottom - y:.1f}" fill="{color}" opacity="0.88"/>')
        lines.append(f'<text x="{centre:.1f}" y="{y - 8:.1f}" font-size="12" text-anchor="middle" '
                     f'fill="#11111b">{value:,.0f}</text>')
        lines.append(f'<text x="{centre:.1f}" y="{bottom + 18}" font-size="12" text-anchor="middle" '
                     f'fill="#11111b">{label}</text>')

    lines.append(f'<text x="{right}" y="{top + 14}" font-size="11" text-anchor="end" '
                 f'fill="#55556b">601 tasks, 4 servers, alpha = 3, 500 paired trials</text>')
    lines.append("</svg>")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":  # pragma: no cover - manual regeneration helper
    import sys
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
    target.mkdir(parents=True, exist_ok=True)
    variance_divergence_svg(target / "critical_index_divergence.svg")
    routing_variance_svg(target / "routing_split_variance.svg")
    print(f"wrote figures to {target}")
