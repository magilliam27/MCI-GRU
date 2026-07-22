#!/usr/bin/env python
"""Run pytest with a pre-configuration guard for managed filesystem paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

BASETEMP_PATH_ENV = "MCI_GRU_PYTEST_BASETEMP"
CACHE_DIR_PATH_ENV = "MCI_GRU_PYTEST_CACHE_DIR"


def _required_path_from_environment(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable {name} is missing")
    return Path(value).expanduser().resolve()


def _resolve_cache_dir(value: str, rootpath: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = rootpath / path
    return path.resolve()


class _IsolationPathGuard:
    """Fail before configuration if pytest parsed paths outside the run root."""

    def __init__(self, *, expected_basetemp: Path, expected_cache_dir: Path) -> None:
        self.expected_basetemp = expected_basetemp
        self.expected_cache_dir = expected_cache_dir

    @pytest.hookimpl(wrapper=True)
    def pytest_cmdline_parse(self) -> Generator[None, pytest.Config, pytest.Config]:
        config = yield
        configured_basetemp = config.getoption("basetemp")
        actual_basetemp = (
            None
            if configured_basetemp is None
            else Path(configured_basetemp).expanduser().resolve()
        )
        actual_cache_dir = _resolve_cache_dir(config.getini("cache_dir"), config.rootpath)

        mismatches: list[str] = []
        if actual_basetemp != self.expected_basetemp:
            mismatches.append(
                f"basetemp resolved to {actual_basetemp}, expected {self.expected_basetemp}"
            )
        if actual_cache_dir != self.expected_cache_dir:
            mismatches.append(
                f"cache_dir resolved to {actual_cache_dir}, expected {self.expected_cache_dir}"
            )
        if mismatches:
            raise pytest.UsageError(
                "isolated pytest path validation failed: " + "; ".join(mismatches)
            )
        return config


def main(argv: Sequence[str] | None = None) -> int:
    """Run pytest only when parsed filesystem paths match the parent contract."""
    try:
        expected_basetemp = _required_path_from_environment(BASETEMP_PATH_ENV)
        expected_cache_dir = _required_path_from_environment(CACHE_DIR_PATH_ENV)
    except ValueError as exc:
        print(f"pytest_isolation_child: error: {exc}", file=sys.stderr)
        return int(pytest.ExitCode.USAGE_ERROR)

    guard = _IsolationPathGuard(
        expected_basetemp=expected_basetemp,
        expected_cache_dir=expected_cache_dir,
    )
    pytest_args = list(sys.argv[1:] if argv is None else argv)
    return int(pytest.main(pytest_args, plugins=[guard]))


if __name__ == "__main__":
    sys.exit(main())
