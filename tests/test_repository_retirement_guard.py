"""Prevent retired repository cockpit surfaces from being reintroduced."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RETIRED_PATHS = (
    ".github/workflows/cockpit-overrides.yml",
    "cockpit",
    "docs/agents/cockpit",
    "docs/agents/workstreams.md",
    "docs/handoffs/2026-07-13-automated-disposition-policy.md",
    "docs/superpowers/plans/2026-07-12-automated-disposition-policy-plan.md",
    "scripts/apply_cockpit_overrides.py",
    "scripts/refresh_cockpit.py",
)
PYTHON_SOURCE_ROOTS = ("mci_gru", "paper_trade", "scripts", "tests")
PYTHON_SOURCE_FILES = ("run_experiment.py",)


def test_retired_repository_surfaces_stay_absent():
    present = [path for path in RETIRED_PATHS if (REPO_ROOT / path).exists()]

    assert not present, f"Retired repository surfaces were reintroduced: {present}"


def test_retired_package_stays_out_of_packaging_and_imports():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for section_name in ("tool.setuptools.packages.find", "tool.ruff.lint.isort"):
        section_start = pyproject.index(f"[{section_name}]")
        section_end = pyproject.find("\n[", section_start + 1)
        section = pyproject[section_start : section_end if section_end >= 0 else None]
        assert "cockpit" not in section.lower()

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
            if any(name == "cockpit" or name.startswith("cockpit.") for name in imported):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
                break

    assert not offenders, f"Retired cockpit imports were reintroduced: {offenders}"
