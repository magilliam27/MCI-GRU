#!/usr/bin/env python
"""Generate docs/TEST_REGISTRY.md: a registry of every pytest test in tests/.

For each test file the registry records the module docstring, the first-party
modules it exercises (mci_gru / scripts), and every test function with its
description and pytest markers.

A test's markers start with its own mark decorators, as written. Then come the
marks it inherits and does not already carry, first from the mark decorators and
``pytestmark`` of its enclosing classes, closest first, then from its module's
``pytestmark`` (#121). Marks applied any other way are not seen, such as through an
alias, a computed value, a ``pytest.param`` case, or a conftest hook.

The committed registry records that test inventory and nothing else: no counts,
no digest, no generation date, and no last-run status. Every line therefore
depends on a single test file, so branches that add tests to different files
change different sections and merge without a registry conflict (#149). Git
still reports one when both insert at the same point, for example two new files
whose names sort next to each other; take either side and regenerate.

``--check`` regenerates the registry in memory, parses the inventory back out of
both the regenerated and the committed copy, and fails on any genuine
difference: a test missing or left over, or a changed description, marker,
module docstring, or exercised module. Section order, row order, cell padding,
and the preamble are formatting and are ignored. It also fails when the
committed copy carries last-run status columns, which belong in the separate
report.

With ``--junit``, the last-run status and duration of each test are merged from a
junit XML report (written by running pytest with
``--junitxml=test_reports/junit.xml``) into a separate status report under the
gitignored ``test_reports/``, never into the committed registry.

Usage:
    python scripts/generate_test_registry.py
    python scripts/generate_test_registry.py --check
    python scripts/generate_test_registry.py --junit test_reports/junit.xml
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIRST_PARTY_PREFIXES = ("mci_gru", "scripts", "run_experiment")
INVENTORY_COLUMNS = ["Test", "Description", "Markers"]
STATUS_COLUMNS = ["Last run", "Time (s)"]
_SECTION_HEADING = re.compile(r"^## `(?P<path>[^`]+)`$")
_TEST_ROW = re.compile(r"^\|\s*`(?P<name>[^`]+)`\s*\|(?P<rest>.*)\|$")
_EXERCISES = "**Exercises:**"


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


def _marker_names(marks: list[ast.expr]) -> list[str]:
    """Names of the pytest marks among decorators or ``pytestmark`` elements, in order."""
    names: list[str] = []
    for mark in marks:
        target = mark.func if isinstance(mark, ast.Call) else mark
        text = ast.unparse(target)
        if text.startswith("pytest.mark."):
            names.append(text.removeprefix("pytest.mark."))
    return names


def _pytestmark_names(body: list[ast.stmt]) -> list[str]:
    """Names of the marks a module or class body assigns to ``pytestmark``.

    pytest applies them to every test that module or class holds. The value is a
    single mark or a list or tuple of them, each with or without arguments.
    """
    names: list[str] = []
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets):
            marks = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            names = _marker_names(marks)
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


def _collect_tests(
    body: list[ast.stmt], prefix: str = "", inherited: tuple[str, ...] = ()
) -> list[TestCase]:
    """List the tests in body with their own marks and the marks they inherit.

    ``inherited`` holds the marks of the enclosing classes and module, closest scope
    first. A test lists its own decorator marks as written, then each inherited mark
    it does not already carry.
    """
    tests: list[TestCase] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            markers = _marker_names(node.decorator_list)
            for mark in inherited:
                if mark not in markers:
                    markers.append(mark)
            tests.append(
                TestCase(
                    name=f"{prefix}{node.name}",
                    doc=_first_line(ast.get_docstring(node)),
                    markers=markers,
                )
            )
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            class_marks = _marker_names(node.decorator_list) + _pytestmark_names(node.body)
            tests.extend(
                _collect_tests(
                    node.body, prefix=f"{prefix}{node.name}.", inherited=(*class_marks, *inherited)
                )
            )
    return tests


def parse_test_module(path: Path) -> TestModule:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tests = _collect_tests(tree.body, inherited=tuple(_pytestmark_names(tree.body)))
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


@dataclass(frozen=True)
class RegistryEntry:
    """What the registry records for one test file, independent of formatting."""

    doc: str
    covers: tuple[str, ...]
    tests: tuple[tuple[str, str, tuple[str, ...]], ...]  # sorted (name, doc, markers)


@dataclass
class ParsedRegistry:
    modules: dict[str, RegistryEntry]  # keyed by the section heading, e.g. tests/test_x.py
    duplicates: list[str]  # headings that appear more than once
    has_status_columns: bool


@dataclass
class _Section:
    path: str
    doc_lines: list[str] = field(default_factory=list)
    covers: tuple[str, ...] = ()
    tests: list[tuple[str, str, tuple[str, ...]]] = field(default_factory=list)
    in_table: bool = False


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_registry(content: str) -> ParsedRegistry:
    """Read the test inventory back out of a rendered registry.

    Section order, row order, blank lines, cell padding, and everything before the
    first section are ignored. Rows are split from the right, because descriptions
    may contain ``|`` while markers and status cells never do.
    """
    sections: list[_Section] = []
    has_status_columns = False
    status_cells = 0
    for raw_line in content.splitlines():
        line = raw_line.strip()
        heading = _SECTION_HEADING.match(line)
        if heading:
            sections.append(_Section(heading["path"]))
            continue
        if not sections or not line:
            continue
        section = sections[-1]
        header = _cells(line) if line.startswith("|") else []
        if header[: len(INVENTORY_COLUMNS)] == INVENTORY_COLUMNS:
            section.in_table = True
            status_cells = len(header) - len(INVENTORY_COLUMNS)
            has_status_columns = has_status_columns or status_cells > 0
        elif section.in_table:
            row = _TEST_ROW.match(line)
            if row:
                cells = row["rest"].rsplit("|", 1 + status_cells)
                cells += [""] * (2 + status_cells - len(cells))
                markers = tuple(m.strip() for m in cells[1].split(",") if m.strip())
                section.tests.append((row["name"], cells[0].strip(), markers))
        elif line.startswith(_EXERCISES):
            section.covers = tuple(sorted(re.findall(r"`([^`]+)`", line[len(_EXERCISES) :])))
        else:
            section.doc_lines.append(line)

    modules: dict[str, RegistryEntry] = {}
    duplicates: set[str] = set()
    for section in sections:
        if section.path in modules:
            duplicates.add(section.path)
        modules[section.path] = RegistryEntry(
            " ".join(section.doc_lines), section.covers, tuple(sorted(section.tests))
        )
    return ParsedRegistry(modules, sorted(duplicates), has_status_columns)


def inventory_differences(
    expected: dict[str, RegistryEntry], recorded: dict[str, RegistryEntry]
) -> list[str]:
    """Describe every way the recorded inventory differs from the expected one."""
    problems = [f"{path}: missing from the registry" for path in sorted(expected.keys() - recorded)]
    problems += [
        f"{path}: listed, but not in the tests directory"
        for path in sorted(recorded.keys() - expected)
    ]
    for path in sorted(expected.keys() & recorded.keys()):
        want, have = expected[path], recorded[path]
        if want.doc != have.doc:
            problems.append(f"{path}: module description differs")
        if want.covers != have.covers:
            problems.append(f"{path}: exercised modules differ")
        missing = Counter(want.tests) - Counter(have.tests)
        extra = Counter(have.tests) - Counter(want.tests)
        missing_names = Counter(name for name, _, _ in missing.elements())
        extra_names = Counter(name for name, _, _ in extra.elements())
        changed = missing_names & extra_names
        problems += [
            f"{path}::{name}: missing from the registry"
            for name in sorted((missing_names - changed).elements())
        ]
        problems += [
            f"{path}::{name}: listed, but not in the test file"
            for name in sorted((extra_names - changed).elements())
        ]
        problems += [
            f"{path}::{name}: description or markers differ" for name in sorted(changed.elements())
        ]
    return problems


def registry_problems(tests_dir: Path, out_path: Path) -> list[str]:
    """Return why out_path does not record the tests now on disk; empty when current."""
    if not out_path.is_file():
        return [f"{out_path} does not exist"]
    expected = parse_registry(build_registry(tests_dir, None, out_path, write=False))
    recorded = parse_registry(out_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if recorded.has_status_columns:
        problems.append(
            "carries last-run status columns; the committed registry records the "
            "inventory only, and status belongs in the --junit report"
        )
    problems += [f"{path}: listed more than once" for path in recorded.duplicates]
    problems += inventory_differences(expected.modules, recorded.modules)
    return problems


def registry_is_current(tests_dir: Path, out_path: Path) -> bool:
    """Return whether out_path records the inventory of the tests now on disk."""
    return not registry_problems(tests_dir, out_path)


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
    display_root: Path,
    module_skips: set[str] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# Test Registry")
    lines.append("")
    lines.append("> Auto-generated by `scripts/generate_test_registry.py` — do not edit by hand.")
    lines.append("> Regenerate with:")
    lines.append("> `.\\.venv\\Scripts\\python.exe scripts/generate_test_registry.py`")
    lines.append("")
    if junit is not None and junit_path is not None:
        lines.append(f"Last-run results merged from `{_display_junit_path(junit_path)}`.")
    else:
        lines.append(
            "This file records the test inventory only, with no counts, dates, or "
            "last-run status, so each line depends on one test file. `--check` compares "
            "it with `tests/`; last-run status goes to a separate report "
            "(see `docs/TESTING_GUIDE.md`)."
        )
    lines.append("")

    for module in sorted(modules, key=lambda m: m.path.name):
        rel = module.path.relative_to(display_root).as_posix()
        lines.append(f"## `{rel}`")
        lines.append("")
        if module.doc:
            lines.append(f"{module.doc}")
            lines.append("")
        if module.covers:
            covered = ", ".join(f"`{c}`" for c in module.covers)
            lines.append(f"**Exercises:** {covered}")
            lines.append("")
        columns = INVENTORY_COLUMNS + (STATUS_COLUMNS if junit is not None else [])
        header = "| " + " | ".join(columns) + " |"
        divider = "|" + "---|" * len(columns)
        lines.append(header)
        lines.append(divider)
        module_collection_skipped = module_skips is not None and module.path.stem in module_skips
        for test in module.tests:
            markers = ", ".join(test.markers) if test.markers else ""
            row = f"| `{test.name}` | {test.doc} | {markers} |"
            if junit is not None:
                key = (module.path.stem, test.name)
                default = (
                    ("SKIPPED (collection)", 0.0)
                    if module_collection_skipped
                    else (
                        "not run",
                        0.0,
                    )
                )
                status, seconds = junit.get(key, default)
                row += f" {status} | {seconds:.2f} |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def build_registry(
    tests_dir: Path,
    junit_path: Path | None,
    out_path: Path,
    *,
    write: bool = True,
) -> str:
    """Render the registry, writing it to out_path unless write is False."""
    modules = [parse_test_module(p) for p in sorted(tests_dir.glob("test_*.py"))]
    junit = None
    module_skips: set[str] = set()
    if junit_path is not None and junit_path.exists():
        junit, module_skips = load_junit_results(junit_path)
    else:
        junit_path = None
    content = render_registry(
        modules,
        junit,
        junit_path,
        display_root=tests_dir.parent,
        module_skips=module_skips,
    )
    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" keeps the committed bytes identical on Windows and Linux.
        out_path.write_text(content, encoding="utf-8", newline="\n")
    return content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-dir", type=Path, default=REPO_ROOT / "tests")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "TEST_REGISTRY.md",
        help="The committed registry. It never carries last-run status.",
    )
    parser.add_argument(
        "--junit",
        type=Path,
        default=None,
        help="Also write a last-run status report merged from this junit XML file.",
    )
    parser.add_argument(
        "--status-out",
        type=Path,
        default=REPO_ROOT / "test_reports" / "TEST_REGISTRY_STATUS.md",
        help="Where --junit writes the status report (default under gitignored test_reports/).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail without writing when the committed registry's inventory is stale.",
    )
    args = parser.parse_args(argv)

    if args.check:
        problems = registry_problems(args.tests_dir, args.out)
        if not problems:
            print(f"Test registry inventory is current: {args.out}")
            return 0
        print(f"Test registry inventory is stale: {args.out}", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "Regenerate with scripts/generate_test_registry.py. On a merge conflict in "
            "the registry, take either side and regenerate.",
            file=sys.stderr,
        )
        return 1

    content = build_registry(args.tests_dir, None, args.out)
    modules = parse_registry(content).modules
    total_tests = sum(len(entry.tests) for entry in modules.values())
    print(f"Wrote {args.out} ({len(modules)} test files, {total_tests} test functions)")
    if args.junit is not None:
        if not args.junit.is_file():
            print(f"No status report written: {args.junit} does not exist.", file=sys.stderr)
            return 0
        build_registry(args.tests_dir, args.junit, args.status_out)
        print(f"Wrote {args.status_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
