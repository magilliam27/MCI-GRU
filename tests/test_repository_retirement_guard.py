"""Prevent retired repository surfaces from being reintroduced.

Two retirements are guarded. The cockpit automation was retired in July 2026.
The 2026-09 cleanup (map #211) retired ``paper_trade/``, ``seed_results/``,
the handoffs, the Cursor-era agent references, the superpowers plans, the
third-party papers, the pre-cleanup notebooks and scripts, and the two
compatibility shims. Everything retired is readable at the tag
``archive/pre-cleanup-2026-09``; nothing here is lost, it is just no longer in
``main``.

The guard exists because a retirement that is only a policy sentence drifts
back silently: a branch cut before the removal reintroduces the path on merge
with no failing test to say so.

Presence is judged by the git index, not the filesystem, so a stale
``__pycache__`` left under a retired directory on a developer machine does not
trip the guard, while a file staged or committed under one does. The test
requires git; there is no fallback, so an environment without it fails loudly
rather than reporting a verdict it cannot support.
"""

import ast
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RETIRED_PATHS = (
    # Cockpit automation, retired 2026-07.
    ".github/workflows/cockpit-overrides.yml",
    "cockpit",
    "docs/agents/cockpit",
    "docs/agents/workstreams.md",
    "scripts/apply_cockpit_overrides.py",
    "scripts/refresh_cockpit.py",
    # Cleanup, retired 2026-09 (map #211, tag archive/pre-cleanup-2026-09).
    "paper_trade",
    "seed_results",
    "cursor",
    "references/papers",
    "docs/handoffs",
    "docs/agent_references/cursor",
    "docs/superpowers",
    "docs/ARCHITECTURE_REVIEW.md",
    "docs/mci_gru_implementation_plan.md",
    "docs/REARCHITECTURE_TECHNICAL_SPEC_2026-07-01.md",
    "docs/REARCHITECTURE_TECHNICAL_SPEC_ADDENDUM_2026-07-04.md",
    "docs/workflows/COLAB_PLAYWRIGHT_MCP_GUIDE.md",
    "mci_gru/models/mci_gru.py",
    "mci_gru/training/metrics.py",
    "notebooks/colab_workflow.ipynb",
    "notebooks/train_test_backtest_workflow.ipynb",
    "scripts/scratch",
    "scripts/verify_baseline.py",
    "tests/backtest_sp500.py",
    "tests/backtest_sp500_daily.py",
)
RETIRED_IMPORT_ROOTS = ("cockpit", "paper_trade")
PYTHON_SOURCE_ROOTS = ("mci_gru", "scripts", "tests")
PYTHON_SOURCE_FILES = ("run_experiment.py",)


def _tracked_under(path: str) -> bool:
    """True when git's index holds at least one file at or under ``path``."""
    result = subprocess.run(
        ["git", "ls-files", "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def test_retired_repository_surfaces_stay_absent():
    present = [path for path in RETIRED_PATHS if _tracked_under(path)]

    assert not present, f"Retired repository surfaces were reintroduced: {present}"


def _imports_retired_root(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in RETIRED_IMPORT_ROOTS)


def test_retired_packages_stay_out_of_packaging_and_imports():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for section_name in ("tool.setuptools.packages.find", "tool.ruff.lint.isort"):
        section_start = pyproject.index(f"[{section_name}]")
        section_end = pyproject.find("\n[", section_start + 1)
        section = pyproject[section_start : section_end if section_end >= 0 else None]
        for root in RETIRED_IMPORT_ROOTS:
            assert root not in section.lower()

    this_file = Path(__file__).resolve()
    offenders: list[str] = []
    source_paths = [REPO_ROOT / path for path in PYTHON_SOURCE_FILES]
    for source_root in PYTHON_SOURCE_ROOTS:
        source_paths.extend((REPO_ROOT / source_root).rglob("*.py"))
    for path in sorted(source_paths):
        if path.resolve() == this_file:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            if any(_imports_retired_root(name) for name in imported):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
                break

    assert not offenders, f"Retired package imports were reintroduced: {offenders}"
