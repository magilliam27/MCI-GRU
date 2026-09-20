#!/usr/bin/env python
"""Generate docs/assets/readme_evidence.svg from docs/assets/readme_evidence.json.

Two panels, drawn only from numbers already printed in current research reports, so the
figure needs no Drive access and anyone can regenerate it:

* Left: excess return over the benchmark by test year on the 110-name point-in-time
  universe (SP500_PIT_GICS_TOP10_MULTIYEAR_BASELINE_2026-06-23), with the roughly
  700-name panel's return compounded across its four yearly test windows against the
  benchmark's beside it (PIT_MASKED_PANEL_2022_2025_FULL_RUN_REPORT_2026-05-16). The
  line's interior points are compounded from the printed per-year rows; its endpoints
  are checked against the compounded figures the report prints.
* Right: the paired re-analysis (GRAPH_PAIRED_REANALYSIS_2026-09-02) as a forest plot:
  each arm's mean daily IC difference from the graph-zeroed control with its 95%
  block-bootstrap interval.

The JSON carries every plotted number and label with the report, section, and column it
is read from, and each panel's caveat in the report's own words.
tests/test_readme_figure.py parses those reports and refuses a JSON value that differs,
checks the README caption carries the caveats, and checks that this script reproduces
the committed SVG byte for byte.

Determinism, so that reproduction holds across machines and matplotlib versions: axes
are placed by figure fraction with no layout engine and no tight bounding box; every
piece of text is placed at an explicit baseline coordinate (no tick labels, no legend,
no axis labels, and no vertical alignment other than baseline, all of which move with
font metrics and moved between matplotlib 3.10 and 3.11); text is kept as text; the
hash-derived clip-path and marker ids matplotlib emits are renamed in order of
appearance; and no date or creator metadata is written. Checked byte-identical on
matplotlib 3.10.9 and 3.11.2.

Colour is by role. Each role has a light and a dark value from the reference palette
the repository's chart method documents (categorical slots 1 and 2, the blue/red
diverging poles, the ink and chrome tokens); the surface pair is GitHub's page
background in each theme. The light value is drawn, then rewritten as a CSS custom
property with a ``prefers-color-scheme: dark`` block, so one SVG renders on both
GitHub themes. Numeric text gets tabular numerals so value columns align.

Usage:
    python scripts/gen_readme_figure.py
    python scripts/gen_readme_figure.py --json docs/assets/readme_evidence.json --out docs/assets/readme_evidence.svg
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import BoxStyle, FancyBboxPatch, Rectangle

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = "docs/assets/readme_evidence.json"
DEFAULT_SVG = "docs/assets/readme_evidence.svg"

# role -> (light, dark)
TOKENS: dict[str, tuple[str, str]] = {
    "ink-primary": ("#0b0b0b", "#ffffff"),
    "ink-secondary": ("#52514e", "#c3c2b7"),
    "ink-muted": ("#898781", "#898781"),
    "grid": ("#e1e0d9", "#2c2c2a"),
    "baseline": ("#c3c2b7", "#383835"),
    "series-1": ("#2a78d6", "#3987e5"),
    "series-2": ("#eb6834", "#d95926"),
    "negative": ("#e34948", "#e66767"),
    "surface": ("#ffffff", "#0d1117"),
}
LIGHT = {role: light for role, (light, _dark) in TOKENS.items()}

Rect = tuple[float, float, float, float]  # left, bottom, width, height as figure fractions
Limits = tuple[float, float]

DPI = 100
PT_PER_PX = 72.0 / DPI  # one figure pixel, in points
FIG_W, FIG_H = 9.6, 3.9  # inches; 691 x 281 pt, close to the README column so it scales little
FONT = "DejaVu Sans"  # bundled with matplotlib; only its name is written, never its metrics
FONT_STACK = "system-ui, -apple-system, 'Segoe UI', sans-serif"
HASH_SALT = "mci-gru-readme-evidence"
MINUS = "−"
NUMERIC_TEXT = re.compile(rf"^[{MINUS}+]?\d[\d.,]*%?$")

RECT_BARS: Rect = (0.05, 0.19, 0.20, 0.58)
RECT_LINE: Rect = (0.30, 0.19, 0.22, 0.58)
RECT_FOREST: Rect = (0.715, 0.19, 0.205, 0.58)
X_RIGHT_BLOCK = 0.545  # where the right panel's title begins; its row labels end at the axes

SIZE_TITLE, SIZE_SUB, SIZE_LABEL, SIZE_TICK = 10.5, 9.0, 9.0, 8.5
CAP = 0.72  # cap height of the sans, as a fraction of the font size
MID = 0.36  # height of a digit's visual centre above the baseline, as a fraction

LINE_W = 2 * PT_PER_PX
HAIR_W = 1 * PT_PER_PX
MARKER = 8 * PT_PER_PX
RING = 2 * PT_PER_PX


def px(points: float) -> float:
    """Convert a font size in points to figure pixels."""
    return points * DPI / 72.0


def signed(value: float, decimals: int, suffix: str = "") -> str:
    """Format with an explicit sign and a true minus sign."""
    return f"{value:+.{decimals}f}".replace("-", MINUS) + suffix


def half_up(value: float) -> int:
    """Round to the nearest integer with halves away from zero.

    The baseline report prints 22.50; the README quotes it as 23 points, and Python's
    banker's rounding would print 22.
    """
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def compounded_pct(yearly_pct: Sequence[float]) -> list[float]:
    """Compound yearly percentage returns into a path of cumulative returns starting at 0."""
    path = [0.0]
    growth = 1.0
    for pct in yearly_pct:
        growth *= 1.0 + pct / 100.0
        path.append((growth - 1.0) * 100.0)
    return path


class Panel:
    """An axes with fixed limits, plus pixel-to-data conversions for placing marks and text."""

    def __init__(self, fig: Figure, rect: Rect, xlim: Limits, ylim: Limits) -> None:
        self.ax = fig.add_axes(rect)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_facecolor("none")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.xlim, self.ylim = xlim, ylim
        self.w_px = rect[2] * FIG_W * DPI
        self.h_px = rect[3] * FIG_H * DPI
        self.ppx = self.w_px / (xlim[1] - xlim[0])
        self.ppy = self.h_px / (ylim[1] - ylim[0])

    def dx(self, pixels: float) -> float:
        return pixels / self.ppx

    def dy(self, pixels: float) -> float:
        return pixels / self.ppy

    def hline(self, y: float, role: str, zorder: int) -> None:
        self.ax.axhline(y, color=LIGHT[role], linewidth=HAIR_W, zorder=zorder)

    def vline(self, x: float, role: str, zorder: int) -> None:
        self.ax.axvline(x, color=LIGHT[role], linewidth=HAIR_W, zorder=zorder)

    def text(self, x: float, y: float, s: str, *, size: float, role: str, ha: str) -> None:
        """Text at an explicit baseline; nothing about its position depends on font metrics."""
        self.ax.text(x, y, s, fontsize=size, color=LIGHT[role], ha=ha, va="baseline", clip_on=False)

    def centred(self, x: float, y: float, s: str, *, size: float, role: str, ha: str) -> None:
        """Text whose digits sit visually centred on y."""
        self.text(x, y - self.dy(MID * px(size)), s, size=size, role=role, ha=ha)

    def below(self, x: float, y: float, s: str, *, size: float, role: str, gap: float = 6) -> None:
        """Text hanging below y with a gap, as a tick label hangs below an axis."""
        self.text(x, y - self.dy(gap + CAP * px(size)), s, size=size, role=role, ha="center")

    def above(self, x: float, y: float, s: str, *, size: float, role: str, gap: float = 5) -> None:
        self.text(x, y + self.dy(gap), s, size=size, role=role, ha="center")

    def left_of(
        self, x: float, y: float, s: str, *, size: float, role: str, gap: float = 7
    ) -> None:
        self.centred(x - self.dx(gap), y, s, size=size, role=role, ha="right")

    def x_ticks(self, positions: Sequence[float], labels: Sequence[str]) -> None:
        for x, label in zip(positions, labels, strict=True):
            self.below(x, self.ylim[0], label, size=SIZE_TICK, role="ink-muted")

    def y_ticks(self, positions: Sequence[float], labels: Sequence[str]) -> None:
        for y, label in zip(positions, labels, strict=True):
            self.left_of(self.xlim[0], y, label, size=SIZE_TICK, role="ink-muted")

    def marker(self, x: float, y: float, role: str) -> None:
        """A >= 8px dot with a 2px surface ring, so it stays legible over a line."""
        self.ax.plot(
            [x],
            [y],
            marker="o",
            markersize=MARKER,
            markerfacecolor=LIGHT[role],
            markeredgecolor=LIGHT["surface"],
            markeredgewidth=RING,
            linestyle="none",
            zorder=3,
        )

    def line(self, xs: Sequence[float], ys: Sequence[float], role: str, zorder: int = 2) -> None:
        self.ax.plot(
            xs,
            ys,
            color=LIGHT[role],
            linewidth=LINE_W,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=zorder,
        )

    def key(self, y_axes: float, name: str, role: str) -> None:
        """One legend entry at a fixed axes-fraction height: a short line and its name."""
        self.ax.plot(
            [0.03, 0.095],
            [y_axes, y_axes],
            transform=self.ax.transAxes,
            color=LIGHT[role],
            linewidth=LINE_W,
            solid_capstyle="round",
            zorder=4,
        )
        self.ax.text(
            0.12,
            y_axes - MID * px(SIZE_LABEL) / self.h_px,
            name,
            transform=self.ax.transAxes,
            fontsize=SIZE_LABEL,
            color=LIGHT["ink-secondary"],
            ha="left",
            va="baseline",
        )


def draw_bars(fig: Figure, block: dict) -> None:
    rows = block["rows"]
    panel = Panel(fig, RECT_BARS, (-0.6, len(rows) - 0.4), (-36.0, 36.0))
    for y in (-30, 30):
        panel.hline(y, "grid", 0)
    panel.hline(0, "baseline", 1)

    width = panel.dx(24)  # bars are at most 24px thick
    radius_x, radius_y = panel.dx(4), panel.dy(4)  # 4px rounded data-end, square at the baseline
    aspect = panel.ppx / panel.ppy
    for x, row in enumerate(rows):
        value = float(row["excess_pct"])
        positive = value >= 0
        role = "series-1" if positive else "negative"
        x0 = x - width / 2
        height = abs(value)
        base_height = min(height, 2 * radius_y)
        for patch in (
            FancyBboxPatch(
                (x0, 0.0 if positive else -height),
                width,
                height,
                boxstyle=BoxStyle.Round(pad=0, rounding_size=radius_x),
                mutation_aspect=aspect,
            ),
            Rectangle((x0, 0.0 if positive else -base_height), width, base_height),
        ):
            patch.set(facecolor=LIGHT[role], edgecolor="none", linewidth=0, zorder=2)
            panel.ax.add_patch(patch)
        label = signed(half_up(value), 0)
        if positive:
            panel.above(x, value, label, size=SIZE_LABEL, role="ink-secondary")
        else:
            panel.below(x, value, label, size=SIZE_LABEL, role="ink-secondary", gap=5)
    panel.x_ticks(list(range(len(rows))), [str(row["year"]) for row in rows])
    panel.y_ticks([-30, 0, 30], [f"{MINUS}30", "0", "+30"])


def draw_compounded(fig: Figure, block: dict) -> None:
    rows = block["rows"]
    model = compounded_pct([float(r["model_total_pct"]) for r in rows])
    bench = compounded_pct([float(r["benchmark_pct"]) for r in rows])
    xs = list(range(len(model)))
    panel = Panel(fig, RECT_LINE, (-0.35, len(model) - 1 + 1.15), (-32.0, 112.0))
    for y in (50, 100):
        panel.hline(y, "grid", 0)
    panel.hline(0, "baseline", 1)

    series = (("benchmark", bench, "series-2"), ("model", model, "series-1"))
    for _name, path, role in series:
        panel.line(xs, path, role)
        panel.marker(xs[-1], path[-1], role)
        panel.centred(
            xs[-1] + panel.dx(9),
            path[-1],
            signed(half_up(path[-1]), 0, "%"),
            size=SIZE_LABEL,
            role="ink-secondary",
            ha="left",
        )
    for index, (name, _path, role) in enumerate(reversed(series)):
        panel.key(0.95 - index * 0.11, name, role)
    panel.x_ticks(xs, ["start"] + [str(row["year"]) for row in rows])
    panel.y_ticks([0, 50, 100], ["0", "+50%", "+100%"])


def draw_forest(fig: Figure, block: dict) -> None:
    rows = block["rows"]
    ys = list(range(len(rows) - 1, -1, -1))  # the report's first row at the top
    panel = Panel(fig, RECT_FOREST, (-0.03, 0.03), (-0.7, len(rows) - 0.3))
    for x in (-0.02, -0.01, 0.01):
        panel.vline(x, "grid", 0)
    panel.vline(0, "baseline", 1)
    for y, row in zip(ys, rows, strict=True):
        panel.line([float(row["ci_low"]), float(row["ci_high"])], [y, y], "series-1")
        panel.marker(float(row["mean_delta"]), y, "series-1")
        panel.centred(
            0.0165,
            y,
            signed(float(row["mean_delta"]), 4),
            size=SIZE_LABEL,
            role="ink-secondary",
            ha="left",
        )
        panel.left_of(panel.xlim[0], y, row["label"], size=SIZE_LABEL, role="ink-secondary")
    panel.x_ticks([-0.02, -0.01, 0, 0.01], [f"{MINUS}0.02", f"{MINUS}0.01", "0", "+0.01"])


def _fig_text(fig: Figure, x: float, y: float, s: str, *, size: float, role: str, **kwargs):
    fig.text(x, y, s, fontsize=size, color=LIGHT[role], va="baseline", **kwargs)


def draw_figure(spec: dict) -> Figure:
    fig = Figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_visible(False)

    forest = spec["paired_reanalysis"]
    draw_bars(fig, spec["reduced_universe_excess"])
    draw_compounded(fig, spec["full_panel_compounded"])
    draw_forest(fig, forest)

    y_title, y_sub1, y_sub2 = 0.915, 0.862, 0.822
    blocks = (
        (
            RECT_BARS[0],
            "Point-in-time replays against the benchmark",
            "excess return by test year, 110-name universe (left)",
            "compounded return, roughly 700-name panel (right)",
        ),
        (
            X_RIGHT_BLOCK,
            "Graph specifications against the graph-zeroed control",
            f"mean daily IC difference, {forest['test_days']} test days",
            "95% block-bootstrap interval; none distinguishable from zero",
        ),
    )
    for x, title, sub1, sub2 in blocks:
        _fig_text(fig, x, y_title, title, size=SIZE_TITLE, role="ink-primary", weight="bold")
        _fig_text(fig, x, y_sub1, sub1, size=SIZE_SUB, role="ink-secondary")
        _fig_text(fig, x, y_sub2, sub2, size=SIZE_SUB, role="ink-secondary")
    for rect, caption in (
        (RECT_BARS, "percentage points over the benchmark"),
        (RECT_LINE, "cumulative return, yearly test windows"),
        (RECT_FOREST, "arm minus control, daily cross-sectional IC"),
    ):
        _fig_text(
            fig,
            rect[0] + rect[2] / 2,
            0.045,
            caption,
            size=SIZE_TICK,
            role="ink-muted",
            ha="center",
        )
    return fig


def _style_block() -> str:
    light = "\n".join(f"    --{role}: {values[0]};" for role, values in TOKENS.items())
    dark = "\n".join(f"      --{role}: {values[1]};" for role, values in TOKENS.items())
    return (
        "<style>\n"
        "  svg {\n"
        f"{light}\n"
        "  }\n"
        "  @media (prefers-color-scheme: dark) {\n"
        "    svg {\n"
        f"{dark}\n"
        "    }\n"
        "  }\n"
        "</style>\n"
    )


def stable_ids(svg: str) -> str:
    """Rename matplotlib's hash-derived clip-path and marker ids in order of appearance.

    The hash changed between matplotlib versions; the order of appearance does not.
    """
    mapping: dict[str, str] = {}
    counts = {"clip": 0, "marker": 0}
    for match in re.finditer(r'id="([mp][0-9a-f]{8,})"', svg):
        old = match.group(1)
        if old in mapping:
            continue
        kind = "clip" if old.startswith("p") else "marker"
        counts[kind] += 1
        mapping[old] = f"{kind}-{counts[kind]}"
    for old, new in mapping.items():
        svg = svg.replace(f'"{old}"', f'"{new}"').replace(f"#{old}", f"#{new}")
    return svg


def tabular_numerals(svg: str) -> str:
    """Give every numeric text run tabular numerals, so value columns align."""

    def restyle(match: re.Match[str]) -> str:
        style, rest, content = match.group(1), match.group(2), match.group(3)
        if NUMERIC_TEXT.match(content):
            style += "; font-variant-numeric: tabular-nums"
        return f'<text style="{style}"{rest}>{content}</text>'

    return re.sub(r'<text style="([^"]*)"([^>]*)>([^<]*)</text>', restyle, svg)


def tokenise(svg: str) -> str:
    """Rewrite every drawn light colour as its role's CSS custom property."""
    for role, (light, _dark) in TOKENS.items():
        svg = re.sub(rf"(fill|stroke): {light}\b", rf"\1: var(--{role}, {light})", svg)
    svg = svg.replace(f"'{FONT}'", FONT_STACK).replace(f'"{FONT}"', FONT_STACK)
    opening_end = svg.index(">", svg.index("<svg")) + 1
    svg = svg[:opening_end] + "\n" + _style_block() + svg[opening_end:]
    stripped = svg
    for role, (light, dark) in TOKENS.items():
        stripped = (
            stripped.replace(f"var(--{role}, {light})", "")
            .replace(f"--{role}: {light}", "")
            .replace(f"--{role}: {dark}", "")
        )
    leaked = sorted({hex_ for _role, (hex_, _dark) in TOKENS.items() if hex_ in stripped})
    if leaked:
        raise RuntimeError(f"colours escaped the token mapping: {leaked}")
    return svg


def render_svg(spec: dict) -> bytes:
    with matplotlib.rc_context(
        {"svg.fonttype": "none", "svg.hashsalt": HASH_SALT, "font.family": FONT}
    ):
        fig = draw_figure(spec)
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="svg",
            dpi=DPI,
            facecolor="none",
            edgecolor="none",
            metadata={"Date": None, "Creator": None},
        )
    svg = buffer.getvalue().decode("utf-8")
    return tokenise(tabular_numerals(stable_ids(svg))).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", default=str(REPO_ROOT / DEFAULT_JSON))
    parser.add_argument("--out", default=str(REPO_ROOT / DEFAULT_SVG))
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.json).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render_svg(spec))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
