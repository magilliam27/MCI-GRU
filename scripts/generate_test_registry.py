#!/usr/bin/env python
"""Generate docs/TEST_REGISTRY.md: a registry of every pytest test in tests/.

For each test file the registry records the module docstring, the first-party
modules it exercises (mci_gru / cockpit / paper_trade / scripts), pytest
markers, and every test function. When test_reports/junit.xml exists (written
by running pytest with --junitxml=test_reports/junit.xml), the last-run status
and duration of each test are merged in.

Usage:
    python scripts/generate_test_registry.py
    python scripts/generate_test_registry.py --junit test_reports/junit.xml --out docs/TEST_REGISTRY.md
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIRST_PARTY_PREFIXES = ("mci_gru", "cockpit", "paper_trade", "scripts", "run_experiment")


@dataclass
class TestCase:
    name: str
    doc: str
    markers: list[str] = field(default_factory=list)


@dataclass
class TestModule:
    path: Path
    doc: str
    covers: list[str]
    tests: list[TestCase]


def _first_line(docstring: str | None) -> str:
    if not docstring:
        return ""
    return docstring.strip().splitlines()[0].strip()


def _marker_names(decorators: list[ast.expr]) -> list[str]:
    names: list[str] = []
    for deco in decorators:
        target = deco.func if isinstance(deco, ast.Call) else deco
        text = ast.unparse(target)
        if text.startswith("pytest.mark."):
            names.append(text.removeprefix("pytest.mark."))
    return names


def _is_first_party(name: str) -> bool:
    # Boundary-aware: "scripts.foo" matches, "scriptsomething" does not.
    return any(name == p or name.startswith(p + ".") for p in FIRST_PARTY_PREFIXES)


def _first_party_imports(tree: ast.Module) -> list[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            candidates = [node.module]
        else:
            continue
        for name in candidates:
            if _is_first_party(name):
                modules.add(name)
    return sorted(modules)


def _collect_tests(body: list[ast.stmt], prefix: str = "") -> list[TestCase]:
    tests: list[TestCase] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            tests.append(
                TestCase(
                    name=f"{prefix}{node.name}",
                    doc=_first_line(ast.get_docstring(node)),
                    markers=_marker_names(node.decorator_list),
                )
            )
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            tests.extend(_collect_tests(node.body, prefix=f"{prefix}{node.name}."))
    return tests


def parse_test_module(path: Path) -> TestModule:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tests = _collect_tests(tree.body)
    return TestModule(
        path=path,
        doc=_first_line(ast.get_docstring(tree)),
        covers=_first_party_imports(tree),
        tests=tests,
    )


def load_junit_results(
    junit_path: Path,
) -> tuple[dict[tuple[str, str], tuple[str, float]], set[str]]:
    """Parse a junit XML file.

    Returns ``(results, module_skips)`` where results maps
    ``(module_stem, test_name) -> (status, seconds)`` and module_skips holds
    module stems that were skipped at collection time (e.g. missing optional
    dependency), so their tests never appear as individual testcases.
    """
    results: dict[tuple[str, str], tuple[str, float]] = {}
    module_skips: set[str] = set()
    root = ET.parse(junit_path).getroot()
    for case in root.iter("testcase"):
        # Module-level collection skips are emitted with an empty classname and
        # the dotted module path as the name (e.g. "tests.test_mlflow_tracking").
        if not case.get("classname") and case.find("skipped") is not None:
            module_skips.add(case.get("name", "").split(".")[-1])
            continue
        # classname examples: "tests.test_backtest_fairness" (function-style)
        # or "tests.test_dynamic_graph_updates.TestStaticGraph" (class-style).
        parts = case.get("classname", "").split(".")
        module_idx = next((i for i, p in enumerate(parts) if p.startswith("test_")), len(parts) - 1)
        module_stem = parts[module_idx]
        class_prefix = ".".join(parts[module_idx + 1 :])
        raw_name = case.get("name", "")
        base_name = raw_name.split("[")[0]  # collapse parametrized ids
        if class_prefix:
            base_name = f"{class_prefix}.{base_name}"
        seconds = float(case.get("time", "0") or 0.0)
        if case.find("failure") is not None:
            status = "FAILED"
        elif case.find("error") is not None:
            status = "ERROR"
        elif case.find("skipped") is not None:
            status = "SKIPPED"
        else:
            status = "PASSED"
        key = (module_stem, base_name)
        prev = results.get(key)
        # Parametrized cases collapse to one row: worst status wins, times sum.
        rank = {"ERROR": 3, "FAILED": 2, "SKIPPED": 1, "PASSED": 0}
        if prev is None:
            results[key] = (status, seconds)
        else:
            worst = status if rank[status] >= rank[prev[0]] else prev[0]
            results[key] = (worst, prev[1] + seconds)
    return results, module_skips


def _display_junit_path(junit_path: Path) -> str:
    """Repo-relative path when possible, to keep the committed doc stable."""
    try:
        return junit_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return junit_path.as_posix()


def render_registry(
    modules: list[TestModule],
    junit: dict[tuple[str, str], tuple[str, float]] | None,
    junit_path: Path | None,
    module_skips: set[str] | None = None,
) -> str:
    total_tests = sum(len(m.tests) for m in modules)
    lines: list[str] = []
    lines.append("# Test Registry")
    lines.append("")
    lines.append("> Auto-generated by `scripts/generate_test_registry.py` — do not edit by hand.")
    lines.append("> Regenerate with:")
    lines.append("> `.\\.venv\\Scripts\\python.exe scripts/generate_test_registry.py`")
    lines.append("")
    stamp = dt.date.today().isoformat()
    lines.append(f"Generated: {stamp}  ")
    lines.append(f"Test files: {len(modules)}  ")
    lines.append(f"Test functions: {total_tests} (parametrized cases collapsed)  ")
    if junit is not None and junit_path is not None:
        lines.append(f"Last-run results merged from `{_display_junit_path(junit_path)}`.")
    else:
        lines.append(
            "No junit results found. Run the suite with "
            "`--junitxml=test_reports/junit.xml` to include last-run status."
        )
    lines.append("")

    for module in sorted(modules, key=lambda m: m.path.name):
        rel = module.path.relative_to(REPO_ROOT).as_posix()
        lines.append(f"## `{rel}`")
        lines.append("")
        if module.doc:
            lines.append(f"{module.doc}")
            lines.append("")
        if module.covers:
            covered = ", ".join(f"`{c}`" for c in module.covers)
            lines.append(f"**Exercises:** {covered}")
            lines.append("")
        header = "| Test | Description | Markers |"
        divider = "|---|---|---|"
        if junit is not None:
            header += " Last run | Time (s) |"
            divider += "---|---|"
        lines.append(header)
        lines.append(divider)
        module_collection_skipped = module_skips is not None and module.path.stem in module_skips
        for test in module.tests:
            markers = ", ".join(test.markers) if test.markers else ""
            row = f"| `{test.name}` | {test.doc} | {markers} |"
            if junit is not None:
                key = (module.path.stem, test.name)
                default = ("SKIPPED (collection)", 0.0) if module_collection_skipped else (
                    "not run",
                    0.0,
                )
                status, seconds = junit.get(key, default)
                row += f" {status} | {seconds:.2f} |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines) + "\n"


def build_registry(tests_dir: Path, junit_path: Path | None, out_path: Path) -> str:
    modules = [parse_test_module(p) for p in sorted(tests_dir.glob("test_*.py"))]
    junit = None
    module_skips: set[str] = set()
    if junit_path is not None and junit_path.exists():
        junit, module_skips = load_junit_results(junit_path)
    else:
        junit_path = None
    content = render_registry(modules, junit, junit_path, module_skips)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-dir", type=Path, default=REPO_ROOT / "tests")
    parser.add_argument("--junit", type=Path, default=REPO_ROOT / "test_reports" / "junit.xml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "TEST_REGISTRY.md")
    args = parser.parse_args()

    build_registry(args.tests_dir, args.junit, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
