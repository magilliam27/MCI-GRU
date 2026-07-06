"""Golden-output regression tests for legacy backtest CLIs (WS-N step 2)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "backtest_golden"
GOLDENS_DIR = FIXTURE_ROOT / "goldens"

TEST_START = "2025-02-03"
TEST_END = "2025-04-30"
TOP_K = 5
LABEL_T = 5

ENGINE_CLI = REPO_ROOT / "scripts" / "backtest_sp500.py"
DAILY_CLI = REPO_ROOT / "scripts" / "backtest_sp500_daily.py"

# (case_name, cli_kind label for docs, extra_args, engines_to_run)
BACKTEST_CASES: list[tuple[str, str, list[str], tuple[str, ...]]] = [
    ("daily", "daily", [], ("daily", "engine")),
    ("staggered", "engine", ["--holding_period", "5"], ("engine",)),
    ("block", "engine", ["--holding_period", "5", "--rebalance_style", "block"], ("engine",)),
    (
        "pit",
        "daily",
        ["--pit_universe_csv", str(FIXTURE_ROOT / "pit_universe.csv")],
        ("daily", "engine"),
    ),
    (
        "tc",
        "daily",
        ["--transaction_costs", "--spread", "10", "--slippage", "5"],
        ("daily", "engine"),
    ),
]


def _normalize_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Round-trip through JSON so float formatting is stable across platforms."""
    return json.loads(json.dumps(payload, sort_keys=True))


def extract_backtest_metrics(backtest_dir: Path) -> dict[str, Any]:
    metrics_path = backtest_dir / "backtest_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with metrics_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return _normalize_metrics(payload)


def _base_cli_args(predictions_dir: Path, data_file: Path) -> list[str]:
    return [
        "--predictions_dir",
        str(predictions_dir),
        "--data_file",
        str(data_file),
        "--test_start",
        TEST_START,
        "--test_end",
        TEST_END,
        "--top_k",
        str(TOP_K),
        "--label_t",
        str(LABEL_T),
        "--auto_save",
    ]


def run_backtest_cli(
    cli_path: Path,
    work_root: Path,
    extra_args: list[str],
) -> dict[str, Any]:
    predictions_src = FIXTURE_ROOT / "run" / "averaged_predictions"
    run_dir = work_root / "run"
    predictions_dir = run_dir / "averaged_predictions"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    shutil.copytree(predictions_src, predictions_dir)

    data_file = FIXTURE_ROOT / "stock_data.csv"
    cmd = [
        sys.executable,
        str(cli_path),
        *_base_cli_args(predictions_dir, data_file),
        *extra_args,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Backtest CLI failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    backtest_dir = run_dir / "backtest"
    return extract_backtest_metrics(backtest_dir)


def _golden_path(case: str, engine: str) -> Path:
    return GOLDENS_DIR / f"{case}__{engine}.json"


def _assert_metrics_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in sorted(set(actual) | set(expected)):
        assert key in expected, f"Unexpected metric key: {key}"
        assert key in actual, f"Missing metric key: {key}"
        av = actual[key]
        ev = expected[key]
        if isinstance(ev, float):
            assert av == pytest.approx(ev, rel=1e-12, abs=0.0)
        else:
            assert av == ev


@pytest.mark.parametrize(
    ("case", "cli_kind", "extra_args", "engines"),
    BACKTEST_CASES,
)
def test_backtest_engine_golden(
    case: str,
    cli_kind: str,
    extra_args: list[str],
    engines: tuple[str, ...],
    tmp_path: Path,
) -> None:
    for engine in engines:
        cli_path = DAILY_CLI if engine == "daily" else ENGINE_CLI
        metrics = run_backtest_cli(cli_path, tmp_path / f"{case}_{engine}", extra_args)
        golden_path = _golden_path(case, engine)
        assert golden_path.is_file(), f"Missing golden fixture: {golden_path}"
        expected = json.loads(golden_path.read_text(encoding="utf-8"))
        _assert_metrics_equal(metrics, expected)


@pytest.mark.parametrize(
    ("case", "cli_kind", "extra_args", "engines"),
    BACKTEST_CASES,
)
def test_backtest_engine_deterministic(
    case: str,
    cli_kind: str,
    extra_args: list[str],
    engines: tuple[str, ...],
    tmp_path: Path,
) -> None:
    for engine in engines:
        cli_path = DAILY_CLI if engine == "daily" else ENGINE_CLI
        first = run_backtest_cli(cli_path, tmp_path / f"{case}_{engine}_a", extra_args)
        second = run_backtest_cli(cli_path, tmp_path / f"{case}_{engine}_b", extra_args)
        assert first == second
