"""The replay reads its inputs from the run it replays (#339).

Run as documented, the replay used paper-era defaults (an old data file, top 10, a
2025-01-01 start, no PIT file), substituted any same-named data file it could find,
dropped every PIT membership with no end date, and averaged a benchmark over names that
joined the universe after the candidates were ranked. These tests pin the fixes:

* ``--run_dir`` supplies the data file, PIT file and export cutoff, window, label, k and
  members, and refuses when a file's bytes differ from what the run consumed or a
  prediction file differs from what the run's receipt bound.
* Flags that remain are checked overrides; the bare documented command is refused.
* An open-ended PIT row keeps its member through the run's export cutoff.
* The benchmark is the PIT universe on the candidates' membership date, so a later
  joiner is out and a name the run never scored is out even with no PIT file (#340
  moved it onto the replay's tranche book).
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
    load_pit_universe_for_backtest,
    replay_universe,
    resolve_data_file,
)
from mci_gru.evaluation.replay_book import ReplayStrategy, build_replay_book
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
K = 2
ATTEMPT = "a" * 32
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


#: AAA pays 3% of its close on session 4, CCC 2% on session 7: inside the hold of
#: several tranches, so a total-return book differs from a price one.
DIVIDENDS = {("AAA", 4): 0.03, ("CCC", 7): 0.02}


def _add_dividends(path: Path) -> None:
    panel = pd.read_csv(path)
    session = panel["dt"].map(SESSIONS.index)
    panel["dividend"] = 0.0
    for (kdcode, i), share in DIVIDENDS.items():
        row = (panel["kdcode"] == kdcode) & (session == i)
        panel.loc[row, "dividend"] = panel.loc[row, "close"] * share
    panel.to_csv(path, index=False)


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


def _build_run(root: Path, *, use_pit: bool = True, return_basis: str | None = None) -> Path:
    """A run directory shaped like ``run_experiment.py``'s output, data under ``root/data``.

    With a ``return_basis``, the run records it and its panel carries a dividend column.
    """
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    data_file = data_dir / "panel.csv"
    pit_file = data_dir / "pit_universe.csv"
    _write_panel(data_file)
    if return_basis is not None:
        _add_dividends(data_file)
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
            **({"return_basis": return_basis} if return_basis is not None else {}),
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
    # As written by metrics.py's default, not chosen by the run: never a headline k.
    summary = {"label_t": LABEL_T, "metrics": {"top_k": 50}}
    (run / "evaluation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    start = run / "execution_provenance" / f"{ATTEMPT}.json"
    start.parent.mkdir()
    start.write_text('{"attempt_id": "' + ATTEMPT + '"}', encoding="utf-8")
    metadata["execution_start"] = {
        "path": f"execution_provenance/{ATTEMPT}.json",
        "sha256": _sha256(start),
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _bind_receipt(run)
    return run


def _bind_receipt(run: Path) -> None:
    """The window's terminal receipt, binding every saved prediction file by digest."""
    start = run / "execution_provenance" / f"{ATTEMPT}.json"
    artifacts = {
        f"averaged_predictions/{f.name}": _sha256(f)
        for f in sorted((run / "averaged_predictions").glob("*.csv"))
    }
    receipt = {"start": {"path": start.name, "sha256": _sha256(start)}, "artifacts": artifacts}
    (start.parent / f"{ATTEMPT}.receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return _build_run(tmp_path)


# -- The run directory supplies every input ------------------------------------------


def test_run_dir_supplies_data_pit_window_label_k_and_members(run_dir: Path) -> None:
    inputs = resolve_replay_inputs(run_dir, top_k=K)

    data_dir = run_dir.parent / "data"
    assert inputs.data_file == (data_dir / "panel.csv").resolve()
    assert inputs.pit_universe_csv == (data_dir / "pit_universe.csv").resolve()
    assert inputs.pit_export_cutoff == CUTOFF
    assert (inputs.test_start, inputs.test_end) == (TEST_START, TEST_END)
    assert inputs.label_t == LABEL_T
    assert inputs.top_k == K
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
        resolve_replay_inputs(run_dir, top_k=K)


def test_refuses_a_pit_file_whose_bytes_differ_from_what_the_run_consumed(
    run_dir: Path,
) -> None:
    pit = run_dir.parent / "data" / "pit_universe.csv"
    pit.write_text(pit.read_text(encoding="utf-8").replace(SESSIONS[5], SESSIONS[4]))

    with pytest.raises(ReplayInputError, match="data.pit_universe_csv"):
        resolve_replay_inputs(run_dir, top_k=K)


def test_refuses_an_edited_resolved_config(run_dir: Path) -> None:
    path = run_dir / "resolved_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["data"]["test_start"] = SESSIONS[0]
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    with pytest.raises(ReplayInputError, match="resolved_config"):
        resolve_replay_inputs(run_dir, top_k=K)


def test_refuses_predictions_for_a_name_outside_the_run(run_dir: Path) -> None:
    day = run_dir / "averaged_predictions" / f"{TEST_START}.csv"
    frame = pd.read_csv(day)
    frame.loc[len(frame)] = [OUTSIDER, TEST_START, 9.9]
    frame.to_csv(day, index=False)
    _bind_receipt(run_dir)

    with pytest.raises(ReplayInputError, match=OUTSIDER):
        resolve_replay_inputs(run_dir, top_k=K)


def test_k_is_required_when_the_run_reported_several(run_dir: Path) -> None:
    """metrics.top_k is the summary function's default (50), not a choice the run made."""
    with pytest.raises(ReplayInputError, match="--top_k"):
        resolve_replay_inputs(run_dir)


def test_refuses_a_prediction_file_the_receipt_did_not_bind(run_dir: Path) -> None:
    day = run_dir / "averaged_predictions" / f"{TEST_START}.csv"
    frame = pd.read_csv(day)
    frame["score"] = frame["score"][::-1].to_numpy()
    frame.to_csv(day, index=False)

    with pytest.raises(ReplayInputError, match="receipt"):
        resolve_replay_inputs(run_dir, top_k=K)


def test_refuses_another_runs_predictions(run_dir: Path, tmp_path: Path) -> None:
    other = _build_run(tmp_path / "other")

    with pytest.raises(ReplayInputError, match="not inside the run"):
        resolve_replay_inputs(run_dir, top_k=K, predictions_dir=str(other / "averaged_predictions"))


def test_legacy_fingerprint_is_used_only_for_csv_runs_without_data_inputs(
    run_dir: Path,
) -> None:
    path = run_dir / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    recorded = metadata["data_inputs"]["data.filename"]
    metadata["data_file"] = recorded["configured_path"]
    metadata["data_file_sha256"] = recorded["sha256"]
    # Another role is still recorded, so data_inputs exists and simply lacks this one.
    del metadata["data_inputs"]["data.filename"]
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with pytest.raises(ReplayInputError, match="no sha256 for data.filename"):
        resolve_replay_inputs(run_dir, top_k=K)


def test_the_recorded_resolved_path_is_tried_when_the_configured_one_differs(
    run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = run_dir / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["data_inputs"]["data.filename"]["configured_path"] = "panel.csv"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "panel.csv").write_text("kdcode,dt\n", encoding="utf-8")

    inputs = resolve_replay_inputs(run_dir, top_k=K)

    assert inputs.data_file == (run_dir.parent / "data" / "panel.csv").resolve()


# -- Flags remain only as checked overrides ------------------------------------------


def test_a_data_file_override_with_the_same_bytes_is_accepted(
    run_dir: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "elsewhere" / "staged_panel.csv"
    copy.parent.mkdir()
    copy.write_bytes((run_dir.parent / "data" / "panel.csv").read_bytes())

    inputs = resolve_replay_inputs(run_dir, top_k=K, data_file=str(copy))

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
        resolve_replay_inputs(run_dir, **{"top_k": K, **override})


def test_overrides_inside_the_run_are_honoured(run_dir: Path) -> None:
    inputs = resolve_replay_inputs(run_dir, top_k=1, test_start=SESSIONS[3], test_end=SESSIONS[7])

    assert (inputs.top_k, inputs.test_start, inputs.test_end) == (1, SESSIONS[3], SESSIONS[7])


def test_window_overrides_are_compared_as_dates_not_text(run_dir: Path) -> None:
    """'2025-3-5' sorts after '2025-03-17' as text; as a date it is 2025-03-05."""
    unpadded = SESSIONS[2].replace("-03-0", "-3-")

    inputs = resolve_replay_inputs(run_dir, top_k=K, test_start=unpadded)

    assert inputs.test_start == SESSIONS[2]


def test_multi_model_with_a_run_dir_is_refused(run_dir: Path) -> None:
    with pytest.raises(SystemExit):
        backtest_engine.main(
            ["--run_dir", str(run_dir), "--top_k", str(K), "--multi_model", str(run_dir)]
        )


def test_a_pit_override_for_a_run_without_pit_is_refused(tmp_path: Path) -> None:
    run = _build_run(tmp_path, use_pit=False)

    with pytest.raises(ReplayInputError, match="PIT"):
        resolve_replay_inputs(
            run, top_k=K, pit_universe_csv=str(tmp_path / "data" / "pit_universe.csv")
        )


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


def _replay(predictions_dir: Path, config: dict):
    return backtest_engine.replay(str(predictions_dir), config)[1]


def _expected_benchmark(run_dir: Path, book, members_on, *, total: bool = False) -> list[float]:
    """Equal-weight hold return (close D+1 to close D+label_t) for each signal date,
    over the members on that date: the date its candidates were ranked. With ``total``,
    each dividend in the hold is reinvested at its ex-date close."""
    panel = pd.read_csv(run_dir.parent / "data" / "panel.csv")
    close = panel.pivot(index="dt", columns="kdcode", values="close")
    if total:
        dividend = panel.pivot(index="dt", columns="kdcode", values="dividend")
        close = close * (1.0 + dividend / close).cumprod()
    out = []
    for date in book.signal_dates:
        i = SESSIONS.index(date)
        names = members_on(date)
        out.append(
            float(
                (
                    close.loc[SESSIONS[i + LABEL_T], names] / close.loc[SESSIONS[i + 1], names]
                    - 1.0
                ).mean()
            )
        )
    return out


def test_benchmark_uses_the_candidates_membership_date(run_dir: Path) -> None:
    """DDD joins on session 6: it is out of the benchmark for the tranche ranked on
    session 5 and in from session 6. CCC leaves after session 5: it is in for the
    tranche ranked on session 5."""
    inputs = resolve_replay_inputs(run_dir, top_k=K)

    book = _replay(inputs.predictions_dir, inputs.backtest_config())

    expected = _expected_benchmark(run_dir, book, lambda dt: [k for k in MEMBERS if _active(k, dt)])
    np.testing.assert_allclose(book.benchmark_period_returns, expected, rtol=1e-12)
    assert SESSIONS[5] in book.signal_dates and SESSIONS[6] in book.signal_dates
    ranked_on_5 = _expected_benchmark(run_dir, book, lambda _dt: ["AAA", "BBB", "CCC"])
    position = book.signal_dates.index(SESSIONS[5])
    assert book.benchmark_period_returns[position] == pytest.approx(ranked_on_5[position])


def test_benchmark_without_a_pit_file_is_the_runs_scored_members(tmp_path: Path) -> None:
    run = _build_run(tmp_path, use_pit=False)
    inputs = resolve_replay_inputs(run, top_k=K)

    book = _replay(inputs.predictions_dir, inputs.backtest_config())

    # The run scored only the names its tradable mask admitted: those are its universe.
    expected = _expected_benchmark(run, book, lambda dt: [k for k in MEMBERS if _active(k, dt)])
    np.testing.assert_allclose(book.benchmark_period_returns, expected, rtol=1e-12)
    assert np.all(book.benchmark_returns < 0.05), "the +10%/session outsider leaked in"


def test_a_total_return_run_is_replayed_on_total_returns(tmp_path: Path) -> None:
    """Issue 361: the replay reads closes on the run's return basis, like its labels."""
    run = _build_run(tmp_path, return_basis="total")
    inputs = resolve_replay_inputs(run, top_k=K)
    assert inputs.return_basis == "total"
    assert inputs.backtest_config()["return_basis"] == "total"

    book = _replay(inputs.predictions_dir, inputs.backtest_config())

    members_on = lambda dt: [k for k in MEMBERS if _active(k, dt)]  # noqa: E731
    total = _expected_benchmark(run, book, members_on, total=True)
    price = _expected_benchmark(run, book, members_on)
    np.testing.assert_allclose(book.benchmark_period_returns, total, rtol=1e-12)
    assert not np.allclose(total, price), "the dividends fall inside no hold"


def test_a_run_before_return_basis_existed_replays_on_prices(run_dir: Path) -> None:
    inputs = resolve_replay_inputs(run_dir, top_k=K)
    assert inputs.return_basis == "price"
    assert inputs.backtest_config()["return_basis"] == "price"


def test_numeric_kdcodes_match_a_string_universe(run_dir: Path) -> None:
    panel = pd.read_csv(run_dir.parent / "data" / "panel.csv")
    codes = {k: i + 1 for i, k in enumerate([*MEMBERS, OUTSIDER])}
    panel["kdcode"] = panel["kdcode"].map(codes)
    kdcodes = [str(codes[k]) for k in MEMBERS]
    dates = SESSIONS[1:9]
    scores = np.tile(np.arange(len(kdcodes), dtype=float), (len(dates), 1))

    book = build_replay_book(
        panel, scores, dates, kdcodes, strategy=ReplayStrategy.close_tranches(LABEL_T), top_k=K
    )

    assert book.benchmark_returns.size and np.all(np.isfinite(book.benchmark_returns))
    assert np.all(book.benchmark_returns < 0.05), "the +10%/session outsider leaked in"


def test_explicit_mode_without_a_pit_file_benchmarks_the_scored_names(run_dir: Path) -> None:
    predictions = backtest_engine.load_predictions(str(run_dir / "averaged_predictions"))
    config = {
        "data_file": str(run_dir.parent / "data" / "panel.csv"),
        "pit_universe_csv": None,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "label_t": LABEL_T,
        "top_k": K,
    }

    universe = replay_universe(config, predictions)
    book = _replay(run_dir / "averaged_predictions", config)

    assert set(universe["kdcode"]) == set(MEMBERS)
    assert np.all(book.benchmark_returns < 0.05), "the +10%/session outsider leaked in"


# -- End to end through the CLI ------------------------------------------------------


def test_cli_replays_a_run_from_its_directory_alone(run_dir: Path) -> None:
    backtest_engine.main(
        ["--run_dir", str(run_dir), "--top_k", str(K), "--auto_save", "--disable_mlflow_autolink"]
    )

    backtest_dirs = list((run_dir / "averaged_predictions").parent.glob("backtest*"))
    assert backtest_dirs, "no backtest output written"
    metrics = json.loads((backtest_dirs[0] / "backtest_metrics.json").read_text(encoding="utf-8"))
    assert metrics
