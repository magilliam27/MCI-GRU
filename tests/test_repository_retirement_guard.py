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


def test_retired_repository_surfaces_stay_absent():
    present = [path for path in RETIRED_PATHS if (REPO_ROOT / path).exists()]

    assert not present, f"Retired repository surfaces were reintroduced: {present}"


def test_retired_package_stays_out_of_packaging_and_imports():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"cockpit*"' not in pyproject
    assert '"cockpit"' not in pyproject

    this_file = Path(__file__).resolve()
    offenders: list[str] = []
    for source_root in PYTHON_SOURCE_ROOTS:
        for path in sorted((REPO_ROOT / source_root).rglob("*.py")):
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
