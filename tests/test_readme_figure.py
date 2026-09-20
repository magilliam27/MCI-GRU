"""Contract tests for scripts/gen_readme_figure.py and the README evidence figure.

Two seams. The committed figure must be reproducible from the committed JSON, so a
reader can regenerate it without Drive access. And every number, label, and caveat in
that JSON must be what the report it cites prints, parsed from the report rather than
retyped, so the figure and its caption cannot drift from the evidence they claim to show.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

from scripts.gen_readme_figure import (
    DEFAULT_JSON,
    DEFAULT_SVG,
    TOKENS,
    compounded_pct,
    main,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
YEAR_BLOCKS = ("reduced_universe_excess", "full_panel_compounded")


def _spec() -> dict:
    return json.loads((REPO_ROOT / DEFAULT_JSON).read_text(encoding="utf-8"))


def _report(block: dict) -> str:
    return (REPO_ROOT / block["report"]).read_text(encoding="utf-8")


def _number(cell: str) -> float:
    """Parse a report table cell: strips bold markers, percent signs, and the Unicode minus."""
    return float(cell.strip().strip("*").replace("−", "-").replace("%", "").replace("+", ""))


def _table(text: str, column_names: list[str]) -> tuple[list[str], list[list[str]]]:
    """Return (header, rows) of the first Markdown table whose header holds every named column.

    Matching on the full column set, not the first cell: the full-run report has a run-status
    table and a backtest table that both open with "Test year".
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(name in cells for name in column_names):
            rows: list[list[str]] = []
            for row in lines[index + 2 :]:
                if not row.strip().startswith("|"):
                    break
                rows.append([c.strip() for c in row.strip().strip("|").split("|")])
            return cells, rows
    raise AssertionError(f"no table with columns {column_names!r}")


def _columns(block: dict) -> tuple[dict[str, int], list[list[str]]]:
    header, rows = _table(_report(block), list(block["columns"].values()))
    return {key: header.index(name) for key, name in block["columns"].items()}, rows


def _caption() -> str:
    """The README paragraph that follows the figure."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    after = readme.split(f"]({DEFAULT_SVG})", 1)[1]
    return after.strip().split("\n\n", 1)[0]


def _assert_year_rows_match(block: dict) -> None:
    col, rows = _columns(block)
    by_year = {int(r[col["year"]]): r for r in rows if re.fullmatch(r"20\d\d", r[col["year"]])}
    assert [r["year"] for r in block["rows"]] == sorted(by_year), (
        "every year in the table is plotted"
    )
    for row in block["rows"]:
        cells = by_year[row["year"]]
        for key in ("model_total_pct", "benchmark_pct", "excess_pct"):
            assert _number(cells[col[key]]) == row[key], (row["year"], key)


def test_reduced_universe_rows_match_report():
    _assert_year_rows_match(_spec()["reduced_universe_excess"])


def test_full_panel_rows_and_compounded_figures_match_report():
    block = _spec()["full_panel_compounded"]
    _assert_year_rows_match(block)

    text = _report(block)
    model = re.search(r"Compounded model return[^.]*?about (\d+\.\d)%", text, re.S)
    benchmark = re.search(r"compounded benchmark return is about (\d+\.\d)%", text)
    assert model and benchmark, "the report's compounded sentences were not found"
    assert float(model.group(1)) == block["compounded_pct"]["model"]
    assert float(benchmark.group(1)) == block["compounded_pct"]["benchmark"]

    # The line the figure draws compounds the per-year rows; its endpoints must land on
    # the figures the report prints, or the rows and the prose disagree.
    model_path = compounded_pct([r["model_total_pct"] for r in block["rows"]])
    bench_path = compounded_pct([r["benchmark_pct"] for r in block["rows"]])
    assert round(model_path[-1], 1) == block["compounded_pct"]["model"]
    assert round(bench_path[-1], 1) == block["compounded_pct"]["benchmark"]
    assert model_path[0] == 0.0 and bench_path[0] == 0.0


def test_paired_reanalysis_rows_and_labels_match_report():
    block = _spec()["paired_reanalysis"]
    col, rows = _columns(block)
    by_arm = {r[col["arm"]].split(" ")[0]: r for r in rows}
    assert [r["arm"] for r in block["rows"]] == list(by_arm), (
        "arms in the report's order, all of them"
    )
    for row in block["rows"]:
        cells = by_arm[row["arm"]]
        _code, description = cells[col["arm"]].split(" — ", 1)
        assert row["label"] == description.strip(), row["arm"]
        assert _number(cells[col["mean_delta"]]) == row["mean_delta"], row["arm"]
        assert _number(cells[col["bhy_p"]]) == row["bhy_p"], row["arm"]
        low, high = cells[col["ci"]].strip("[]").split(",")
        assert _number(low) == row["ci_low"] and _number(high) == row["ci_high"], row["arm"]
    days = re.search(r"\| Test days \| (\d+),", _report(block))
    assert days and int(days.group(1)) == block["test_days"]


def test_caveats_are_the_reports_words_and_the_caption_carries_them():
    spec = _spec()
    caption = _caption()
    for name in (*YEAR_BLOCKS, "paired_reanalysis"):
        block = spec[name]
        report = re.sub(r"\s+", " ", _report(block).replace("**", ""))
        assert block["caveats"], name
        for sentence in block["caveats"]:
            assert sentence in report, (name, sentence)
            assert sentence in caption, (name, sentence)
        assert f"]({block['report']})" in caption, name
    assert f"{spec['paired_reanalysis']['test_days']} test days" in caption


def test_generator_reproduces_committed_svg(tmp_path):
    out = tmp_path / "readme_evidence.svg"
    assert main(["--json", str(REPO_ROOT / DEFAULT_JSON), "--out", str(out)]) == 0
    generated = out.read_bytes().replace(b"\r\n", b"\n")
    committed = (REPO_ROOT / DEFAULT_SVG).read_bytes().replace(b"\r\n", b"\n")
    assert generated == committed, (
        "docs/assets/readme_evidence.svg is not what scripts/gen_readme_figure.py produces "
        f"from docs/assets/readme_evidence.json (matplotlib {matplotlib.__version__}); "
        "regenerate it and commit both, or explain the difference."
    )


def test_committed_svg_carries_both_themes_tabular_numerals_and_no_volatile_metadata():
    svg = (REPO_ROOT / DEFAULT_SVG).read_text(encoding="utf-8")
    assert "@media (prefers-color-scheme: dark)" in svg
    for role, (light, dark) in TOKENS.items():
        assert f"--{role}: {light}" in svg, role
        assert f"--{role}: {dark}" in svg, role
    assert "<text" in svg, "text is kept as text, not converted to paths"
    numeric = re.findall(r'<text style="([^"]*)"[^>]*>([−+]?\d[\d.,]*%?)</text>', svg)
    assert numeric and all("tabular-nums" in style for style, _ in numeric)
    assert "<dc:date>" not in svg and "dc:creator" not in svg


def test_readme_embeds_the_committed_figure_once():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count(f"]({DEFAULT_SVG})") == 1
