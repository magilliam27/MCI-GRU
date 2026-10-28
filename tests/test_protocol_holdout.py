"""The locked 2026 holdout and the window schedule (#370, PRD #369).

The holdout is the 2026 signal dates from 2026-01-22 to 2026-07-24. No run may train
on, validate on or score them, and no label may reach into them, until the promotion
path lands (#374). These tests pin:

* the holdout dates and the five window runs, W5 being the recipe as written (#367);
* the guard: a span inside the holdout is refused, and so is a span whose last label
  may exit inside it, while the latest legal window passes;
* the three places that apply it: every ``ExperimentConfig`` (training, walk-forward,
  rescoring), the replay's input resolver, and the selection-research protocol;
* the refusal is a ``HoldoutError``, which the walk-forward builder's
  ``except ValueError`` cannot swallow.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ProtocolConfig,
    create_config_from_dict,
)
from mci_gru.evaluation import backtest_engine
from mci_gru.evaluation.replay_inputs import resolve_replay_inputs
from mci_gru.evaluation.selection_audit import SelectionResearchProtocol
from mci_gru.protocol import (
    HOLDOUT_END,
    HOLDOUT_START,
    HOLDOUT_WINDOW,
    WINDOWS,
    HoldoutError,
    Window,
    latest_label_end_bound,
    require_outside_holdout,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE = REPO_ROOT / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md"
DATA_KEYS = ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")

#: NYSE full-day closures from the 2024 holidays to the holdout's front edge, for the
#: real label exits below. 2025-01-09 was the day of mourning for President Carter.
NYSE_CLOSURES = np.array(
    [
        "2024-11-28",
        "2024-12-25",
        "2025-01-01",
        "2025-01-09",
        "2025-01-20",
        "2025-02-17",
        "2025-04-18",
        "2025-05-26",
        "2025-06-19",
        "2025-07-04",
        "2025-09-01",
        "2025-11-27",
        "2025-12-25",
        "2026-01-01",
        "2026-01-19",
    ],
    dtype="datetime64[D]",
)
#: The New Year's Day closures before each window's first training session.
NEW_YEARS_CLOSURES = np.array(
    ["2016-01-01", "2017-01-02", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"],
    dtype="datetime64[D]",
)


def _real_exit(signal_date: str, label_t: int) -> str:
    """The session ``label_t`` sessions after ``signal_date`` on the NYSE calendar."""
    exit_day = np.busday_offset(
        np.datetime64(signal_date, "D"), label_t, roll="forward", holidays=NYSE_CLOSURES
    )
    return str(exit_day)


def _config(window: Window, *, label_t: int = 5) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(**{key: getattr(window, key) for key in DATA_KEYS}),
        model=ModelConfig(label_t=label_t),
    )


def _compose(overrides: list[str]):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))


def _recipe_overrides() -> list[str]:
    block = re.search(
        r"^## Hydra Overrides\n+```text\n(.*?)^```",
        RECIPE.read_text(encoding="utf-8"),
        re.M | re.S,
    )
    assert block, "the recipe has no Hydra override block"
    return [line.strip() for line in block.group(1).splitlines() if line.strip()]


# -- The dates ----------------------------------------------------------------------


def test_the_holdout_dates_are_pinned():
    """Moving the holdout takes a reviewed change to this test, by design."""
    assert (HOLDOUT_START, HOLDOUT_END) == ("2026-01-22", "2026-07-24")
    assert HOLDOUT_WINDOW.spans() == {
        "data.train_start": "2021-01-04",
        "data.train_end": "2024-12-31",
        "data.val_start": "2025-01-22",
        "data.val_end": "2025-12-31",
        "data.test_start": "2026-01-22",
        "data.test_end": "2026-07-24",
    }


def test_the_window_schedule_is_pinned():
    assert {name: w.spans() for name, w in WINDOWS.items()} == {
        "W1": {
            "data.train_start": "2016-01-04",
            "data.train_end": "2019-12-31",
            "data.val_start": "2020-01-22",
            "data.val_end": "2020-12-31",
            "data.test_start": "2021-01-22",
            "data.test_end": "2021-12-31",
        },
        "W2": {
            "data.train_start": "2017-01-03",
            "data.train_end": "2020-12-31",
            "data.val_start": "2021-01-22",
            "data.val_end": "2021-12-31",
            "data.test_start": "2022-01-22",
            "data.test_end": "2022-12-31",
        },
        "W3": {
            "data.train_start": "2018-01-02",
            "data.train_end": "2021-12-31",
            "data.val_start": "2022-01-22",
            "data.val_end": "2022-12-31",
            "data.test_start": "2023-01-22",
            "data.test_end": "2023-12-31",
        },
        "W4": {
            "data.train_start": "2019-01-02",
            "data.train_end": "2022-12-31",
            "data.val_start": "2023-01-22",
            "data.val_end": "2023-12-31",
            "data.test_start": "2024-01-22",
            "data.test_end": "2024-12-31",
        },
        "W5": {
            "data.train_start": "2020-01-02",
            "data.train_end": "2023-12-31",
            "data.val_start": "2024-01-22",
            "data.val_end": "2024-12-31",
            "data.test_start": "2025-01-22",
            "data.test_end": "2025-12-31",
        },
    }


@pytest.mark.parametrize("window", [*WINDOWS.values(), HOLDOUT_WINDOW], ids=lambda w: w.name)
def test_each_window_trains_four_calendar_years_then_validates_then_tests(window):
    train_first = int(window.train_start[:4])
    assert window.train_end == f"{train_first + 3}-12-31"
    assert window.val_start[:4] == window.val_end[:4] == str(train_first + 4)
    assert window.test_start[:4] == window.test_end[:4] == str(train_first + 5)
    assert window.train_end < window.val_start <= window.val_end < window.test_start


def test_each_window_starts_training_on_its_first_session():
    """January 1 is never a session; the training start is the first one after it."""
    for window in [*WINDOWS.values(), HOLDOUT_WINDOW]:
        first = np.busday_offset(
            np.datetime64(f"{window.train_start[:4]}-01-01", "D"),
            0,
            roll="forward",
            holidays=NEW_YEARS_CLOSURES,
        )
        assert window.train_start == str(first), window.name


def test_w5_is_the_recipe_as_written():
    """The recipe (#367) and the schedule must not drift apart."""
    data = _compose(_recipe_overrides()).data
    assert {f"data.{key}": getattr(data, key) for key in DATA_KEYS} == WINDOWS["W5"].spans()


# -- The guard ----------------------------------------------------------------------


@pytest.mark.parametrize("window", list(WINDOWS.values()), ids=lambda w: w.name)
def test_every_window_run_passes_the_guard(window):
    require_outside_holdout(window.spans(), label_t=5, what=window.name)
    _config(window)


def test_the_holdout_window_is_refused():
    with pytest.raises(HoldoutError, match="locked holdout"):
        require_outside_holdout(HOLDOUT_WINDOW.spans(), label_t=5, what="holdout")
    with pytest.raises(HoldoutError, match="This run"):
        _config(HOLDOUT_WINDOW)


@pytest.mark.parametrize("key", DATA_KEYS)
def test_any_single_date_inside_the_holdout_is_refused(key):
    """Each of the six dates is checked, not just test_end."""
    spans = dict(WINDOWS["W5"].spans())
    spans[f"data.{key}"] = "2026-03-02"
    with pytest.raises(HoldoutError, match=f"data.{key}=2026-03-02"):
        require_outside_holdout(spans, label_t=5, what="probe")


def test_a_date_after_the_holdout_is_refused_too():
    """Decision 3: the data after the holdout is the next holdout, not open ground."""
    with pytest.raises(HoldoutError):
        require_outside_holdout({"test_end": "2026-09-30"}, label_t=5, what="probe")


def test_a_label_that_may_exit_inside_the_holdout_is_refused():
    """2026-01-15 is before the holdout, but its 5-session label can end inside it."""
    with pytest.raises(HoldoutError, match="may exit as late as"):
        require_outside_holdout({"test_end": "2026-01-15"}, label_t=5, what="probe")
    # Control: a week earlier the label closes well before the holdout.
    require_outside_holdout({"test_end": "2026-01-08"}, label_t=5, what="probe")


def test_the_label_check_follows_label_t():
    """The same end date passes a 1-session label and fails a 10-session one."""
    require_outside_holdout({"test_end": "2026-01-14"}, label_t=1, what="probe")
    with pytest.raises(HoldoutError):
        require_outside_holdout({"test_end": "2026-01-14"}, label_t=10, what="probe")


def test_the_label_check_reads_the_latest_date_not_the_last_key():
    spans = {"data.train_start": "2026-01-15", "data.test_end": "2025-06-30"}
    with pytest.raises(HoldoutError, match="data.train_start=2026-01-15"):
        require_outside_holdout(spans, label_t=5, what="probe")


def test_the_bound_never_falls_before_the_real_exit():
    """The slack must cover every closure a label can cross, the densest cluster included.

    Signal dates run from the 2024 Christmas run-up, whose 15-session labels cross four
    closures, to the holdout's front edge.
    """
    signal = np.datetime64("2024-12-02", "D")
    while signal < np.datetime64(HOLDOUT_START, "D"):
        for label_t in (1, 2, 5, 10, 19, 21, 42, 63):
            day = str(signal)
            if _real_exit(day, label_t) > "2026-01-30":
                continue  # past the closures listed above
            assert latest_label_end_bound(day, label_t) >= _real_exit(day, label_t), (
                day,
                label_t,
            )
        signal += np.timedelta64(1, "D")


def test_the_last_legal_signal_date_for_the_recipe_label_is_open():
    """The test year's real last label (2026-01-08 exit) is far from the guard."""
    assert _real_exit("2025-12-31", 5) == "2026-01-08"
    assert latest_label_end_bound("2025-12-31", 5) < HOLDOUT_START


def test_label_t_below_one_is_rejected():
    with pytest.raises(ValueError, match="label_t"):
        latest_label_end_bound("2025-01-02", 0)


def test_an_empty_span_is_rejected():
    with pytest.raises(ValueError, match="at least one date"):
        require_outside_holdout({}, label_t=5, what="probe")


def test_holdout_error_is_not_a_value_error():
    """The walk-forward builder catches ValueError to skip a window; this must escape it."""
    assert not issubclass(HoldoutError, ValueError)


# -- Promotion stays closed until #374 ------------------------------------------------


def test_the_base_config_names_no_promotion():
    assert _compose([]).protocol == ProtocolConfig(holdout_promotion=None)


def test_any_promotion_is_refused_until_the_promotion_path_lands():
    with pytest.raises(HoldoutError, match="#374"):
        ProtocolConfig(holdout_promotion="candidate-a")
    with pytest.raises(HoldoutError, match="#374"):
        _compose(["protocol.holdout_promotion=candidate-a"])


def test_a_promotion_does_not_open_the_holdout_either():
    """Not even a promotion key alongside a holdout window composes today."""
    overrides = [f"{key}={value}" for key, value in HOLDOUT_WINDOW.spans().items()]
    with pytest.raises(HoldoutError):
        _compose(overrides)
    with pytest.raises(HoldoutError):
        _compose([*overrides, "protocol.holdout_promotion=candidate-a"])


def test_a_hydra_override_into_the_holdout_is_refused():
    """The runner composes this way; a data.test_end override must not get through."""
    with pytest.raises(HoldoutError, match="data.test_end=2026-03-31"):
        _compose(["data.test_end=2026-03-31"])


# -- The scoring paths ----------------------------------------------------------------


def _replay_run(root: Path, *, test_start: str, test_end: str) -> Path:
    """The minimum of a run directory the replay resolver reads before its data files."""
    run = root / "run"
    run.mkdir(parents=True)
    resolved = {
        "data": {"test_start": test_start, "test_end": test_end},
        "model": {"label_t": 5},
        "evaluation": {"top_k_values": [10]},
    }
    (run / "resolved_config.json").write_text(json.dumps(resolved), encoding="utf-8")
    digest = hashlib.sha256((run / "resolved_config.json").read_bytes()).hexdigest()
    metadata = {
        "kdcode_list": ["AAA"],
        "label_t": 5,
        "resolved_config_path": "resolved_config.json",
        "resolved_config_sha256": digest,
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run


def test_the_replay_refuses_a_run_that_scored_the_holdout(tmp_path):
    run = _replay_run(tmp_path, test_start="2026-01-22", test_end="2026-07-24")
    with pytest.raises(HoldoutError, match="Scoring"):
        resolve_replay_inputs(run)


def test_the_replay_refuses_a_window_whose_last_label_reaches_the_holdout(tmp_path):
    run = _replay_run(tmp_path, test_start="2025-01-22", test_end="2026-01-15")
    with pytest.raises(HoldoutError, match="may exit as late as"):
        resolve_replay_inputs(run)


def test_the_replay_guard_does_not_stop_a_legal_window(tmp_path):
    """Control: a 2025 window gets past the guard and fails later, on the missing data."""
    run = _replay_run(tmp_path, test_start="2025-01-22", test_end="2025-12-31")
    with pytest.raises(Exception) as caught:
        resolve_replay_inputs(run)
    assert not isinstance(caught.value, HoldoutError)


def _explicit_backtest(test_start: str, test_end: str) -> list[str]:
    """The backtest CLI with every input named, so no run's records are read."""
    return [
        "--predictions_dir",
        "no-such-predictions",
        "--data_file",
        "no-such-panel.csv",
        "--test_start",
        test_start,
        "--test_end",
        test_end,
        "--top_k",
        "10",
        "--label_t",
        "5",
    ]


def test_the_backtest_with_explicit_inputs_refuses_the_holdout():
    """The no-run_dir branch takes its window from flags, so it needs its own guard."""
    with pytest.raises(HoldoutError, match="This backtest"):
        backtest_engine.main(_explicit_backtest("2026-01-22", "2026-07-24"))
    with pytest.raises(HoldoutError, match="may exit as late as"):
        backtest_engine.main(_explicit_backtest("2025-01-22", "2026-01-15"))


def test_the_backtest_guard_does_not_stop_a_legal_window():
    """Control: a 2025 window gets past the guard and fails later, on the missing files."""
    with pytest.raises(BaseException) as caught:
        backtest_engine.main(_explicit_backtest("2025-01-22", "2025-12-31"))
    assert not isinstance(caught.value, HoldoutError)


def _selection_protocol(test_start: str, test_end: str, data_as_of: str):
    return SelectionResearchProtocol(
        research_semantics_version="selection-research-v1",
        study_name="holdout-probe",
        trial_family_id="fixture-family",
        predictions_dir="predictions",
        market_data_path="market.csv",
        pit_universe_path="pit.csv",
        expected_scorable_path="expected.csv",
        calendar_path="calendar.csv",
        label_horizon=5,
        test_start=test_start,
        test_end=test_end,
        data_as_of=data_as_of,
        top_k=2,
        price_basis="ADJUSTED_RESEARCH",
        price_adjustment_provenance="fixture-adjusted-close-v1",
        null_draws=1000,
        null_seed=17,
        hac_lag=3,
        bootstrap_block_length=5,
        bootstrap_resamples=100,
        bootstrap_seed=19,
        ci_level=0.95,
        alpha=0.05,
        prediction_source_run_id="fixture-run",
        prediction_ensemble_rule="averaged_predictions",
        prediction_ensemble_member_count=20,
        prediction_seed_id="314159",
        prediction_source_code_commit="d6b0f60",
        prediction_label_contract="MCI_GRU_FORWARD_CLOSE_V1",
        prediction_label_horizon=5,
    )


def test_selection_research_refuses_a_holdout_window():
    with pytest.raises(HoldoutError, match="holdout-probe"):
        _selection_protocol("2026-01-22", "2026-07-24", "2026-07-31")


def test_selection_research_reads_data_as_of_as_a_vintage_not_a_scored_date():
    """Control: a 2025 window on data pulled in 2026 is legal."""
    _selection_protocol("2025-01-22", "2025-12-31", "2026-07-31")
