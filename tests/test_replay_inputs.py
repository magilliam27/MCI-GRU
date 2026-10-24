"""The replay reads its inputs from the run it replays (#339).

Run as documented, the replay used paper-era defaults (an old data file, top 10, a
2025-01-01 start, no PIT file), substituted any same-named data file it could find,
dropped every PIT membership with no end date, and averaged a benchmark over names that
joined the universe after the candidates were ranked. These tests pin the fixes:

* ``--run_dir`` supplies the data file, PIT file and export cutoff, window, label, k and
  members, and refuses when a file's bytes differ from what the run consumed.
* Flags that remain are checked overrides; the bare documented command is refused.
* An open-ended PIT row keeps its member through the run's export cutoff.
* The benchmark is the PIT universe on the candidates' membership date, so a later
  joiner is out and a name the run never scored is out even with no PIT file.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from mci_gru.evaluation import backtest_engine
from mci_gru.evaluation.backtest_engine import (
    calculate_forward_returns,
    equal_weight_benchmark_daily_series,
    load_pit_universe_for_backtest,
    replay_universe,
    resolve_data_file,
)
from mci_gru.evaluation.replay_inputs import (
    ReplayInputError,
    resolve_replay_inputs,
)

if TYPE_CHECKING:
    from pathlib import Path

SESSIONS = pd.bdate_range("2025-03-03", periods=12).strftime("%Y-%m-%d").tolist()
MEMBERS = ["AAA", "BBB", "CCC", "DDD"]
# Not a run member: present in the data file, never scored, never in the universe.
OUTSIDER = "XXX"
TEST_START, TEST_END = SESSIONS[1], SESSIONS[-3]
LABEL_T = 3
TOP_K_VALUES = [1, 2]
HEADLINE_K = 2
CUTOFF = SESSIONS[-1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_panel(path: Path) -> None:
    rows = []
    for j, kdcode in enumerate([*MEMBERS, OUTSIDER]):
        for i, dt in enumerate(SESSIONS):
            # Members move under 2% a session; OUTSIDER moves +10% a session, so any
            # leak into the benchmark is obvious.
            if kdcode == OUTSIDER:
                price = 100.0 * 1.10**i
            else:
                price = 100.0 * (1.0 + 0.001 * (j + 1)) ** i * (1.01 if i % 2 else 1.0)
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": dt,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * 1.002,
                    "volume": 1_000_000,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_pit(path: Path) -> None:
    # AAA and BBB are open-ended (blank valid_to), as current members are in the
    # run's own PIT file. CCC leaves after session 5; DDD joins on session 6.
    pd.DataFrame(
        [
            {"kdcode": "AAA", "valid_from": SESSIONS[0], "valid_to": ""},
            {"kdcode": "BBB", "valid_from": SESSIONS[0], "valid_to": ""},
            {"kdcode": "CCC", "valid_from": SESSIONS[0], "valid_to": SESSIONS[5]},
            {"kdcode": "DDD", "valid_from": SESSIONS[6], "valid_to": CUTOFF},
        ]
    ).to_csv(path, index=False)


def _active(kdcode: str, dt: str) -> bool:
    if kdcode in ("AAA", "BBB"):
        return True
    if kdcode == "CCC":
        return dt <= SESSIONS[5]
    if kdcode == "DDD":
        return dt >= SESSIONS[6]
    return False


def _write_predictions(directory: Path) -> None:
    directory.mkdir(parents=True)
    for i, dt in enumerate(SESSIONS):
        if not TEST_START <= dt <= TEST_END:
            continue
        names = [k for k in MEMBERS if _active(k, dt)]
        scores = [((i + j) % 4) / 10.0 for j in range(len(names))]
        pd.DataFrame({"kdcode": names, "dt": dt, "score": scores}).to_csv(
            directory / f"{dt}.csv", index=False
        )


def _build_run(root: Path, *, use_pit: bool = True) -> Path:
    """A run directory shaped like ``run_experiment.py``'s output, data under ``root/data``."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    data_file = data_dir / "panel.csv"
    pit_file = data_dir / "pit_universe.csv"
    _write_panel(data_file)
    _write_pit(pit_file)

    run = root / "run"
    run.mkdir()
    _write_predictions(run / "averaged_predictions")
    resolved = {
        "data": {
            "filename": "<ABSOLUTE_PATH>",
            "use_pit_universe": use_pit,
            "pit_universe_csv": "<ABSOLUTE_PATH>",
            "pit_export_cutoff": CUTOFF,
            "test_start": TEST_START,
            "test_end": TEST_END,
        },
        "model": {"label_t": LABEL_T},
        "evaluation": {"top_k_values": TOP_K_VALUES},
    }
    (run / "resolved_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    data_inputs = {
        "data.filename": {
            "configured_path": str(data_file),
            "resolved_path": str(data_file),
            "sha256": _sha256(data_file),
        }
    }
    if use_pit:
        data_inputs["data.pit_universe_csv"] = {
            "configured_path": str(pit_file),
            "resolved_path": str(pit_file),
            "sha256": _sha256(pit_file),
        }
    metadata = {
        "kdcode_list": MEMBERS,
        "label_t": LABEL_T,
        "resolved_config_path": "resolved_config.json",
        "resolved_config_sha256": _sha256(run / "resolved_config.json"),
        "data_inputs": data_inputs,
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = {"label_t": LABEL_T, "metrics": {"top_k": HEADLINE_K}}
    (run / "evaluation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return _build_run(tmp_path)


# -- The run directory supplies every input ------------------------------------------


def test_run_dir_supplies_data_pit_window_label_k_and_members(run_dir: Path) -> None:
    inputs = resolve_replay_inputs(run_dir)

    data_dir = run_dir.parent / "data"
    assert inputs.data_file == (data_dir / "panel.csv").resolve()
    assert inputs.pit_universe_csv == (data_dir / "pit_universe.csv").resolve()
    assert inputs.pit_export_cutoff == CUTOFF
    assert (inputs.test_start, inputs.test_end) == (TEST_START, TEST_END)
    assert inputs.label_t == LABEL_T
    assert inputs.top_k == HEADLINE_K
    assert inputs.members == tuple(MEMBERS)
    assert inputs.predictions_dir == run_dir / "averaged_predictions"

    config = inputs.backtest_config()
    assert config["data_file"] == str(inputs.data_file)
    assert config["pit_universe_csv"] == str(inputs.pit_universe_csv)
    assert config["pit_export_cutoff"] == CUTOFF
    assert config["members"] == MEMBERS
    assert config["data_file_sha256"] == _sha256(data_dir / "panel.csv")


def test_refuses_a_data_file_whose_bytes_differ_from_what_the_run_consumed(
    run_dir: Path,
) -> None:
    panel = run_dir.parent / "data" / "panel.csv"
    panel.write_text(panel.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ReplayInputError, match="data.filename"):
        resolve_replay_inputs(run_dir)


def test_refuses_a_pit_file_whose_bytes_differ_from_what_the_run_consumed(
    run_dir: Path,
) -> None:
    pit = run_dir.parent / "data" / "pit_universe.csv"
    pit.write_text(pit.read_text(encoding="utf-8").replace(SESSIONS[5], SESSIONS[4]))

    with pytest.raises(ReplayInputError, match="data.pit_universe_csv"):
        resolve_replay_inputs(run_dir)


def test_refuses_an_edited_resolved_config(run_dir: Path) -> None:
    path = run_dir / "resolved_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["data"]["test_start"] = SESSIONS[0]
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    with pytest.raises(ReplayInputError, match="resolved_config"):
        resolve_replay_inputs(run_dir)


def test_refuses_predictions_for_a_name_outside_the_run(run_dir: Path) -> None:
    day = run_dir / "averaged_predictions" / f"{TEST_START}.csv"
    frame = pd.read_csv(day)
    frame.loc[len(frame)] = [OUTSIDER, TEST_START, 9.9]
    frame.to_csv(day, index=False)

    with pytest.raises(ReplayInputError, match=OUTSIDER):
        resolve_replay_inputs(run_dir)


# -- Flags remain only as checked overrides ------------------------------------------


def test_a_data_file_override_with_the_same_bytes_is_accepted(
    run_dir: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "elsewhere" / "staged_panel.csv"
    copy.parent.mkdir()
    copy.write_bytes((run_dir.parent / "data" / "panel.csv").read_bytes())

    inputs = resolve_replay_inputs(run_dir, data_file=str(copy))

    assert inputs.data_file == copy.resolve()


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"top_k": 3}, "top_k"),
        ({"label_t": LABEL_T + 1}, "label_t"),
        ({"test_start": SESSIONS[0]}, "test_start"),
        ({"test_end": SESSIONS[-1]}, "test_end"),
        ({"pit_export_cutoff": SESSIONS[-2]}, "pit_export_cutoff"),
    ],
)
def test_overrides_outside_the_run_are_refused(
    run_dir: Path, override: dict[str, object], match: str
) -> None:
    with pytest.raises(ReplayInputError, match=match):
        resolve_replay_inputs(run_dir, **override)


def test_overrides_inside_the_run_are_honoured(run_dir: Path) -> None:
    inputs = resolve_replay_inputs(run_dir, top_k=1, test_start=SESSIONS[3], test_end=SESSIONS[7])

    assert (inputs.top_k, inputs.test_start, inputs.test_end) == (1, SESSIONS[3], SESSIONS[7])


def test_a_pit_override_for_a_run_without_pit_is_refused(tmp_path: Path) -> None:
    run = _build_run(tmp_path, use_pit=False)

    with pytest.raises(ReplayInputError, match="PIT"):
        resolve_replay_inputs(run, pit_universe_csv=str(tmp_path / "data" / "pit_universe.csv"))


def test_the_bare_documented_command_is_refused(run_dir: Path) -> None:
    """``--predictions_dir`` alone used to replay on paper-era defaults."""
    with pytest.raises(SystemExit):
        backtest_engine.main(["--predictions_dir", str(run_dir / "averaged_predictions")])


def test_explicit_mode_without_a_run_names_every_missing_input(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        backtest_engine.main(
            [
                "--predictions_dir",
                str(run_dir / "averaged_predictions"),
                "--data_file",
                str(run_dir.parent / "data" / "panel.csv"),
            ]
        )
    err = capsys.readouterr().err
    for flag in ("--test_start", "--test_end", "--top_k", "--label_t"):
        assert flag in err


# -- No same-name substitute ---------------------------------------------------------


def test_a_same_named_file_elsewhere_is_not_substituted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "panel.csv").write_text("kdcode,dt,open,close\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        resolve_data_file(str(tmp_path / "not_here" / "panel.csv"))


# -- Open-ended PIT rows keep their members ------------------------------------------


def test_open_ended_pit_rows_keep_their_members_through_the_cutoff(run_dir: Path) -> None:
    pit = load_pit_universe_for_backtest(
        str(run_dir.parent / "data" / "pit_universe.csv"), open_valid_to=CUTOFF
    )

    assert set(pit["kdcode"]) == set(MEMBERS)
    open_rows = pit[pit["kdcode"].isin(["AAA", "BBB"])]
    assert (open_rows["valid_to"] == CUTOFF).all()


def test_open_ended_pit_rows_without_a_cutoff_are_refused_not_dropped(run_dir: Path) -> None:
    with pytest.raises(ValueError, match="pit_export_cutoff"):
        load_pit_universe_for_backtest(str(run_dir.parent / "data" / "pit_universe.csv"))


# -- The benchmark is the PIT universe on the candidates' membership date -----------


def _panel_with_returns(run_dir: Path) -> pd.DataFrame:
    panel = pd.read_csv(run_dir.parent / "data" / "panel.csv")
    return calculate_forward_returns(panel, label_t=LABEL_T)


def _expected_benchmark(panel: pd.DataFrame, members_on) -> pd.Series:
    """Equal-weight open-to-open return at each entry session, over members on the
    session before it (the date the candidates entering then were ranked)."""
    out = {}
    for prev, entry in zip(SESSIONS[:-1], SESSIONS[1:], strict=True):
        names = members_on(prev)
        rows = panel[(panel["dt"] == entry) & panel["kdcode"].isin(names)]
        values = rows["open_to_open_return"].dropna()
        if len(values):
            out[entry] = values.mean()
    return pd.Series(out).sort_index()


def test_benchmark_uses_the_candidates_membership_date(run_dir: Path) -> None:
    """DDD joins on session 6: it is out of the benchmark for the trade entering then
    (ranked on session 5) and in from the next one. CCC leaves after session 5: it is
    in for the trade entering on session 6."""
    panel = _panel_with_returns(run_dir)
    inputs = resolve_replay_inputs(run_dir)
    config = inputs.backtest_config()
    predictions = backtest_engine.load_predictions(str(inputs.predictions_dir))

    benchmark = equal_weight_benchmark_daily_series(panel, replay_universe(config, predictions))

    expected = _expected_benchmark(panel, lambda dt: [k for k in MEMBERS if _active(k, dt)])
    pd.testing.assert_series_equal(
        benchmark.dropna(), expected, check_names=False, check_index_type=False
    )
    joiner_entry = SESSIONS[6]
    ranked_then = panel[(panel["dt"] == joiner_entry) & panel["kdcode"].isin(["AAA", "BBB", "CCC"])]
    assert benchmark[joiner_entry] == pytest.approx(ranked_then["open_to_open_return"].mean())


def test_benchmark_without_a_pit_file_is_the_runs_members(tmp_path: Path) -> None:
    run = _build_run(tmp_path, use_pit=False)
    panel = _panel_with_returns(run)
    inputs = resolve_replay_inputs(run)
    config = inputs.backtest_config()
    predictions = backtest_engine.load_predictions(str(inputs.predictions_dir))

    benchmark = equal_weight_benchmark_daily_series(panel, replay_universe(config, predictions))

    expected = _expected_benchmark(panel, lambda _dt: MEMBERS)
    pd.testing.assert_series_equal(
        benchmark.dropna(), expected, check_names=False, check_index_type=False
    )
    assert np.all(benchmark.dropna() < 0.05), "the +10%/session outsider leaked in"


def test_explicit_mode_without_a_pit_file_benchmarks_the_scored_names(run_dir: Path) -> None:
    panel = _panel_with_returns(run_dir)
    predictions = backtest_engine.load_predictions(str(run_dir / "averaged_predictions"))

    universe = replay_universe({"pit_universe_csv": None}, predictions)

    assert set(universe["kdcode"]) == set(MEMBERS)
    benchmark = equal_weight_benchmark_daily_series(panel, universe)
    assert np.all(benchmark.dropna() < 0.05), "the +10%/session outsider leaked in"


# -- End to end through the CLI ------------------------------------------------------


def test_cli_replays_a_run_from_its_directory_alone(run_dir: Path) -> None:
    backtest_engine.main(["--run_dir", str(run_dir), "--auto_save", "--disable_mlflow_autolink"])

    backtest_dirs = list((run_dir / "averaged_predictions").parent.glob("backtest*"))
    assert backtest_dirs, "no backtest output written"
    metrics = json.loads((backtest_dirs[0] / "backtest_metrics.json").read_text(encoding="utf-8"))
    assert metrics
