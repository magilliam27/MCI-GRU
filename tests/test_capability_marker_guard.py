"""No test CI deselects by capability marker may pass without that capability.

CI deselects tests by marker alone (`.github/workflows/ci.yml`), so a
`requires_data` or `requires_lseg` marker on a test that does not need the
capability is a silent coverage hole: the test exists, passes, and never runs on
a pull request (#120).

The guard runs exactly the tests CI deselects, `-m "not (<CI's expression>)"`,
in a child pytest whose capabilities are the CI runner's rather than the
workstation's:

- no real market data: the child runs in a copy of the repository's tracked
  files, and every market-data format is gitignored;
- no LSEG client: `refinitiv` and `lseg` fail to import;
- no FRED key: `FRED_API_KEY` is removed, as on the runner.

Any deselected test that passes there does not need what its marker claims.
Control tests injected into the child prove both halves on every run: a
decorative control must be seen passing, and a control that genuinely needs the
capability must not pass. Without them an empty selection, or a denial that
leaks, would make this guard pass vacuously.

Tests that need data or LSEG keep their marker; this guard only rejects markers
their tests do not earn.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
GIT = shutil.which("git")
CHILD_RUN_ENV = "MCI_GRU_CAPABILITY_GUARD_CHILD"
DENIED_LSEG_PACKAGES = ("refinitiv", "lseg")
CONTROL_MODULE = "test_capability_marker_guard_controls.py"
CONTROL_CLASSNAME = "tests.test_capability_marker_guard_controls"
CONTROLS_BY_MARKER = {
    "requires_data": {
        "decorative": "test_requires_data_on_a_test_that_needs_nothing",
        "genuine": "test_requires_data_on_a_test_that_reads_market_data",
    },
    "requires_lseg": {
        "decorative": "test_requires_lseg_on_a_test_that_needs_nothing",
        "genuine": "test_requires_lseg_on_a_test_that_imports_the_lseg_client",
    },
}
CONTROL_SOURCE = '''"""Controls injected by tests/test_capability_marker_guard.py; never committed."""

import importlib

import pytest

from mci_gru.data.path_resolver import PROJECT_ROOT

MARKET_DATA_SUFFIXES = {".csv", ".parquet", ".feather", ".h5", ".hdf5"}


@pytest.mark.requires_data
def test_requires_data_on_a_test_that_needs_nothing():
    pass


@pytest.mark.requires_data
def test_requires_data_on_a_test_that_reads_market_data():
    data_files = [
        path for path in (PROJECT_ROOT / "data").rglob("*") if path.suffix in MARKET_DATA_SUFFIXES
    ]
    assert data_files, f"no market data under {PROJECT_ROOT / 'data'}"


@pytest.mark.requires_lseg
def test_requires_lseg_on_a_test_that_needs_nothing():
    pass


@pytest.mark.requires_lseg
def test_requires_lseg_on_a_test_that_imports_the_lseg_client():
    errors = []
    for client in ("refinitiv.data", "lseg.data"):
        try:
            importlib.import_module(client)
        except ImportError as exc:
            errors.append(f"{client}: {exc}")
        else:
            return
    pytest.fail("; ".join(errors))
'''


def _ci_marker_expression() -> str:
    """Return the one `-m` expression every pytest step in the CI workflow selects with.

    Each test job (the Python 3.10 floor and the Linux CPU lock, #143) must deselect
    the same markers, so this guard's single child run covers all of them.
    """
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    expressions = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            tokens = shlex.split(step.get("run") or "")
            if "pytest" not in tokens:
                continue
            arguments = tokens[tokens.index("pytest") + 1 :]
            if "-m" in arguments:
                expressions.append(arguments[arguments.index("-m") + 1])
    assert expressions and len(set(expressions)) == 1, (
        f"expected every pytest step in {CI_WORKFLOW} to select with one shared -m "
        f"expression, found {expressions}; adapt this guard to how CI now selects tests"
    )
    return expressions[0]


def _copy_tracked_files(destination: Path) -> None:
    """Copy tracked files as they stand in the working tree; ignored data stays behind."""
    listed = subprocess.run(
        [GIT, "ls-files", "--cached", "-z"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    for relative in filter(None, os.fsdecode(listed).split("\0")):
        source = REPOSITORY_ROOT / relative
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_lseg_import_denials(directory: Path) -> None:
    for package in DENIED_LSEG_PACKAGES:
        (directory / package).mkdir(parents=True)
        (directory / package / "__init__.py").write_text(
            f"raise ImportError({package!r} + ' is denied: the CI runner has no LSEG client')\n",
            encoding="utf-8",
        )


def _ci_runner_environment(denials: Path, snapshot: Path) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"FRED_API_KEY", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}
    }
    pythonpath = [str(denials), str(snapshot)]
    if inherited := environment.get("PYTHONPATH"):
        pythonpath.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    environment[CHILD_RUN_ENV] = "1"
    return environment


def _outcomes(junit_xml: Path) -> dict[str, str]:
    outcomes = {}
    for case in ET.parse(junit_xml).getroot().iter("testcase"):
        node = f"{case.get('classname')}::{case.get('name')}"
        if case.find("failure") is not None or case.find("error") is not None:
            outcomes[node] = "failed"
        elif case.find("skipped") is not None:
            outcomes[node] = "skipped"
        else:
            outcomes[node] = "passed"
    return outcomes


@pytest.mark.skipif(GIT is None, reason="git is required to copy only the tracked files")
def test_no_test_ci_deselects_passes_without_its_capability(tmp_path: Path) -> None:
    """Every test CI deselects fails or skips with the CI runner's capabilities."""
    if os.environ.get(CHILD_RUN_ENV):
        pytest.skip("already inside this guard's child run")

    expression = _ci_marker_expression()
    named = set(re.findall(r"[A-Za-z_]\w*", expression)) - {"not", "and", "or"}
    assert named and named <= set(CONTROLS_BY_MARKER), (
        f"CI's -m expression {expression!r} names {sorted(named)}; this guard can deny and "
        f"control only {sorted(CONTROLS_BY_MARKER)}. Audit the tests carrying any new marker "
        "and extend the guard before CI deselects them (#120)."
    )

    snapshot = tmp_path / "repo"
    denials = tmp_path / "denied"
    _copy_tracked_files(snapshot)
    _write_lseg_import_denials(denials)
    (snapshot / "tests" / CONTROL_MODULE).write_text(CONTROL_SOURCE, encoding="utf-8")
    junit_xml = tmp_path / "child-junit.xml"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-m",
            f"not ({expression})",
            "-q",
            "-p",
            "no:cacheprovider",
            "--continue-on-collection-errors",
            f"--basetemp={tmp_path / 'child-basetemp'}",
            f"--junitxml={junit_xml}",
        ],
        cwd=snapshot,
        env=_ci_runner_environment(denials, snapshot),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    child_output = (
        f"child pytest exit {completed.returncode}\n{completed.stdout}\n{completed.stderr}"
    )
    assert junit_xml.is_file(), child_output

    outcomes = _outcomes(junit_xml)
    passed = {node for node, outcome in outcomes.items() if outcome == "passed"}
    decorative = {
        f"{CONTROL_CLASSNAME}::{CONTROLS_BY_MARKER[marker]['decorative']}" for marker in named
    }
    genuine = {f"{CONTROL_CLASSNAME}::{CONTROLS_BY_MARKER[marker]['genuine']}" for marker in named}

    assert decorative <= passed, (
        "a control that needs nothing was not seen passing, so this guard cannot observe a "
        f"decorative marker: {sorted(decorative - passed)}\n{child_output}"
    )
    assert not genuine & passed, (
        "a control that needs the capability passed, so the child still has it and cannot "
        f"stand in for the CI runner: {sorted(genuine & passed)}\n{child_output}"
    )
    unearned = sorted(passed - decorative)
    assert not unearned, (
        f"CI deselects these tests with -m {expression!r}, yet they pass with no market data, "
        "no LSEG client and no FRED key, so CI never runs them for no reason. Remove the "
        "marker, or keep it only on a test that fails or skips without the capability "
        "(#120):\n" + "\n".join(unearned)
    )
