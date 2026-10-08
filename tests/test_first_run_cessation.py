"""The first run declares no cessation file; this pins why that is safe, and when not.

``build_pit_masks`` uses a declared cessation only through ``eligible``, and
``tradable = eligible & feature_ready & price_observed``. So for a delisted stock
with no close after its last session, the tradable and loss masks are the same
with or without the file, and the recipe's ``pit_cessation_events_csv: null``
changes no prediction, label or metric. A vendor tail of carried closes after
delisting breaks that, which is what ``scripts/check_delisted_tails.py`` checks on
the real panel before the run. See docs/DEFAULT_EXPERIMENT_RECIPE.md.

Each identity test is paired with a control in which the masks must differ, so
the identity cannot pass because the cessation mask stopped being applied.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import create_config_from_dict
from mci_gru.data.pit import (
    CessationEvidence,
    PredictionClock,
    build_pit_masks,
    cessation_exclusion_mask,
    parse_cessation_events,
    resolve_cessation_events,
)
from mci_gru.data.preprocessing import label_session_axis

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSIONS = list(pd.bdate_range("2023-09-01", "2023-10-31").strftime("%Y-%m-%d"))
LAST = "2023-10-12"  # the delisted name's last real session
CODES = ["A", "B", "GONE"]
SAMPLE = SESSIONS[5:]
HIS_T = 5
LABEL_T = 5
INTERVALS = pd.DataFrame(
    {"kdcode": CODES, "valid_from": ["2023-09-01"] * 3, "valid_to": ["2023-10-31"] * 3}
)
EVENT = {
    "event_id": "e1",
    "kdcode": "GONE",
    "effective_at": "2023-10-13",
    "known_from": "2023-10-11",
    "acquired_at": "",
    "evidence": "",
}

_spec = importlib.util.spec_from_file_location(
    "check_delisted_tails", REPO_ROOT / "scripts" / "check_delisted_tails.py"
)
check_delisted_tails = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_delisted_tails)


def _panel(stale_tail: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for code in CODES:
        price = 100.0
        last_close = None
        for session in SESSIONS:
            price *= 1 + rng.normal(0, 0.01)
            if code == "GONE" and session > LAST:
                if stale_tail:
                    rows.append((code, session, last_close, 0.0))
                continue
            if code == "GONE":
                last_close = price
            rows.append((code, session, price, 1000.0))
    return pd.DataFrame(rows, columns=["kdcode", "dt", "close", "volume"])


def _masks(panel: pd.DataFrame, declare: bool):
    evidence = (
        parse_cessation_events(pd.DataFrame([EVENT]).astype(str), source="events.csv")
        if declare
        else CessationEvidence(configured_path=None, events=())
    )
    clock = PredictionClock()
    resolved = resolve_cessation_events(evidence, label_session_axis(panel, CODES), clock)
    excluded = cessation_exclusion_mask(resolved, CODES, SAMPLE, clock)
    return build_pit_masks(panel, panel, CODES, SAMPLE, HIS_T, LABEL_T, INTERVALS, excluded)


def test_without_a_price_tail_the_cessation_file_changes_no_tradable_or_loss_row():
    panel = _panel(stale_tail=False)
    undeclared, declared = _masks(panel, declare=False), _masks(panel, declare=True)
    assert declared.cessation_excluded[:, CODES.index("GONE")].any(), "control: the event holds"
    assert np.array_equal(undeclared.tradable, declared.tradable)
    assert np.array_equal(undeclared.loss, declared.loss)


def test_with_a_carried_price_tail_the_cessation_file_does_change_the_masks():
    """Control: carried closes after delisting make the undeclared run train on them."""
    panel = _panel(stale_tail=True)
    undeclared, declared = _masks(panel, declare=False), _masks(panel, declare=True)
    gone = CODES.index("GONE")
    after = [i for i, date in enumerate(SAMPLE) if date > LAST]
    assert undeclared.tradable[after, gone].all()
    assert not declared.tradable[after, gone].any()


def test_the_recipe_declares_no_cessation_file():
    """The decision this module justifies; a declared file needs the recipe note revisited."""
    text = (REPO_ROOT / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md").read_text(encoding="utf-8")
    block = text.split("## Hydra Overrides", 1)[1].split("```text", 1)[1].split("```", 1)[0]
    overrides = [line.strip() for line in block.splitlines() if line.strip()]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    config = create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))
    assert config.data.pit_cessation_events_csv is None


def test_tail_check_flags_a_carried_tail_and_passes_a_clean_history():
    clean = check_delisted_tails.tail_report(_panel(stale_tail=False), ["GONE", "A"])
    assert [r["stale"] for r in clean] == [False, False]
    assert clean[0]["last_finite_close_dt"] == LAST
    stale = check_delisted_tails.tail_report(_panel(stale_tail=True), ["GONE"])[0]
    assert stale["stale"]
    assert stale["trailing_repeat_sessions"] >= 2
    assert stale["trailing_zero_volume_sessions"] >= 2


def test_tail_check_exit_code_follows_the_finding(tmp_path, capsys):
    clean_csv = tmp_path / "clean.csv"
    stale_csv = tmp_path / "stale.csv"
    for path, stale in ((clean_csv, False), (stale_csv, True)):
        panel = _panel(stale_tail=stale)
        panel["kdcode"] = panel["kdcode"].replace({"GONE": "GONE.N^J23"})
        panel.to_csv(path, index=False)
    assert check_delisted_tails.main(["--market-csv", str(clean_csv)]) == 0
    assert check_delisted_tails.main(["--market-csv", str(stale_csv)]) == 1
    assert "GONE.N^J23" in capsys.readouterr().out


def _recipe_text(block: str, before: str = "") -> str:
    return f"{before}## Hydra Overrides\n\n```text\n{block}```\n"


def test_tail_check_reads_the_panel_the_recipe_selects(tmp_path):
    """The precondition must run on the panel the first run reads (issue 283)."""
    eodhd = REPO_ROOT / "configs" / "data" / "gics_top10_110_2016_eodhd.yaml"
    assert check_delisted_tails.recipe_data_config() == eodhd
    assert check_delisted_tails._default_market_csv() == REPO_ROOT / str(
        OmegaConf.load(eodhd).filename
    )
    # Control: it follows the selector rather than naming a fixed config.
    other = tmp_path / "recipe.md"
    other.write_text(_recipe_text("data=gics_top10_110_2016\nseed=1729\n"), encoding="utf-8")
    assert check_delisted_tails.recipe_data_config(other).name == "gics_top10_110_2016.yaml"


def test_tail_check_reads_the_selector_only_inside_the_override_block(tmp_path):
    other = tmp_path / "recipe.md"
    # A data= line in prose before the block is not the recipe's selection.
    other.write_text(
        _recipe_text("data=gics_top10_110_2016\n", before="```text\ndata=sp500\n```\n\n"),
        encoding="utf-8",
    )
    assert check_delisted_tails.recipe_data_config(other).name == "gics_top10_110_2016.yaml"
    # Neither a key override nor an indented or prefixed line selects a group.
    for block in ("seed=1729\ndata.source=csv\n", "  data=sp500\n", "xdata=sp500\n"):
        other.write_text(_recipe_text(block), encoding="utf-8")
        with pytest.raises(SystemExit):
            check_delisted_tails.recipe_data_config(other)


def _with_last_row(panel: pd.DataFrame, close_delta: float, volume: float) -> pd.DataFrame:
    """Append one GONE row on the session after its last, at last close + delta."""
    gone = panel.loc[panel["kdcode"] == "GONE"].sort_values("dt")
    next_session = SESSIONS[SESSIONS.index(LAST) + 1]
    row = {"kdcode": "GONE", "dt": next_session, "close": gone["close"].iloc[-1] + close_delta}
    return pd.concat([panel, pd.DataFrame([{**row, "volume": volume}])], ignore_index=True)


def test_tail_check_flags_one_carried_zero_volume_row():
    """A vendor row for the delisting day repeats the close at zero volume (EODHD, 2026-10-08)."""
    panel = _with_last_row(_panel(stale_tail=False), close_delta=0.0, volume=0.0)
    record = check_delisted_tails.tail_report(panel, ["GONE"])[0]
    assert (record["trailing_repeat_sessions"], record["trailing_zero_volume_sessions"]) == (1, 1)
    assert record["stale"]
    # Controls: a traded repeat of the close, or a zero-volume row at a new
    # price, is one coincidence and stays clean at the default threshold.
    traded = _with_last_row(_panel(stale_tail=False), close_delta=0.0, volume=500.0)
    assert not check_delisted_tails.tail_report(traded, ["GONE"])[0]["stale"]
    repriced = _with_last_row(_panel(stale_tail=False), close_delta=0.25, volume=0.0)
    assert not check_delisted_tails.tail_report(repriced, ["GONE"])[0]["stale"]
