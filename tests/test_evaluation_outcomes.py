"""The shared evaluation core: realized returns, the rankable set, exits and ties (#335).

Every path that scores a run (the in-run summary, the replay, the side modules) is
meant to ask this core what a stock returned and who could be ranked. These tests pin
the four rules it owns, on one small synthetic panel:

* ``KEEP`` trades every session.
* ``GAP`` has no bar on session 3, so per-stock row shifts and the session axis disagree.
* ``DLST`` stops trading after session 4, inside the hold of earlier signal dates.
* ``HALT`` misses sessions 4 and 6 and trades again on session 7, so "the last close
  in the hold" and "the next close after the exit" give different answers.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mci_gru.config import (
    EvaluationConfig,
    ExperimentConfig,
    Legs,
    ModelConfig,
    create_config_from_dict,
)
from mci_gru.data.pit import build_pit_masks
from mci_gru.data.preprocessing import compute_labels
from mci_gru.evaluation import portfolio
from mci_gru.evaluation.outcomes import (
    order_by_score,
    rankable,
    realized_returns,
    select_top_k,
    top_k_tranche_returns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSIONS = pd.bdate_range("2025-03-03", periods=10).strftime("%Y-%m-%d").tolist()
KDCODES = ["DLST", "GAP", "HALT", "KEEP"]
DLST, GAP, HALT, KEEP = range(4)
GAP_MISSING = 3
DLST_LAST = 4
HALT_MISSING = (4, 6)


def _close(kdcode: str, i: int) -> float:
    return {
        "KEEP": 100.0 + i,
        "GAP": 100.0 + 10.0 * i,
        "DLST": 50.0 - 5.0 * i,
        "HALT": 200.0 + 7.0 * i,
    }[kdcode]


def _panel() -> pd.DataFrame:
    rows = []
    for i, session in enumerate(SESSIONS):
        for kdcode in KDCODES:
            if kdcode == "GAP" and i == GAP_MISSING:
                continue
            if kdcode == "DLST" and i > DLST_LAST:
                continue
            if kdcode == "HALT" and i in HALT_MISSING:
                continue
            close = _close(kdcode, i)
            rows.append({"kdcode": kdcode, "dt": session, "open": close, "close": close})
    # Shuffled so nothing depends on the input's row order.
    return pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)


def _intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kdcode": KDCODES,
            "valid_from": ["2016-01-04"] * len(KDCODES),
            "valid_to": ["2030-12-31"] * len(KDCODES),
        }
    )


# --- realized returns -------------------------------------------------------------


def test_default_legs_equal_the_training_label_exactly_on_a_panel_with_a_missing_bar():
    panel = _panel()
    out = realized_returns(panel, SESSIONS, Legs(), kdcodes=KDCODES)
    label = compute_labels(panel, KDCODES, SESSIONS, label_t=5, fill_missing=False)

    observable = np.isfinite(label)
    assert observable.sum() >= 8  # the comparison is not vacuous
    np.testing.assert_array_equal(out.returns[observable].astype(np.float32), label[observable])

    # Signal on session 1: entry close session 2, exit close session 6. GAP's missing
    # session 3 lies inside the hold; a per-stock row shift would exit on session 7.
    d = 1
    assert out.entry_dates[d] == SESSIONS[2]
    assert out.exit_dates[d] == SESSIONS[6]
    assert out.returns[d, GAP] == _close("GAP", 6) / _close("GAP", 2) - 1.0
    assert out.returns[d, GAP] != _close("GAP", 7) / _close("GAP", 2) - 1.0
    assert out.returns[d, KEEP] == _close("KEEP", 6) / _close("KEEP", 2) - 1.0


def test_custom_legs_read_closes_at_their_own_sessions():
    out = realized_returns(
        _panel(), SESSIONS, Legs(entry_session=2, exit_session=4), kdcodes=KDCODES
    )
    d = 0
    assert out.entry_dates[d] == SESSIONS[2]
    assert out.exit_dates[d] == SESSIONS[4]
    assert out.returns[d, KEEP] == _close("KEEP", 4) / _close("KEEP", 2) - 1.0
    assert out.returns[d, GAP] == _close("GAP", 4) / _close("GAP", 2) - 1.0


def test_missing_exit_books_the_last_available_close_and_records_the_event():
    out = realized_returns(_panel(), SESSIONS, Legs(), kdcodes=KDCODES)

    # Signal on session 1: DLST is bought at session 2's close, has no close at
    # session 6, and its last close is session 4. That loss is booked, then cash.
    d = 1
    assert out.returns[d, DLST] == pytest.approx(_close("DLST", 4) / _close("DLST", 2) - 1.0)
    assert out.returns[d, DLST] < 0
    assert out.booked_exit_dates[d][DLST] == SESSIONS[DLST_LAST]

    events = {(e.date, e.kdcode): e for e in out.events}
    event = events[(SESSIONS[d], "DLST")]
    assert event.kind == "missing_exit"
    assert event.scheduled_session == SESSIONS[6]
    assert event.booked_session == SESSIONS[DLST_LAST]


def test_missing_exit_books_the_last_close_inside_the_hold_never_a_later_one():
    out = realized_returns(_panel(), SESSIONS, Legs(), kdcodes=KDCODES)

    # Signal on session 1: HALT is bought at session 2, has no close on session 4 or
    # on its exit (session 6), and trades again on session 7. The last close inside
    # the hold is session 5; session 7 is after the exit and is never read.
    d = 1
    assert out.returns[d, HALT] == _close("HALT", 5) / _close("HALT", 2) - 1.0
    assert out.booked_exit_dates[d][HALT] == SESSIONS[5]
    event = {(e.date, e.kdcode): e for e in out.events}[(SESSIONS[d], "HALT")]
    assert (event.kind, event.booked_session) == ("missing_exit", SESSIONS[5])


def test_the_session_axis_does_not_depend_on_which_columns_are_requested():
    panel = _panel()
    full = realized_returns(panel, SESSIONS, Legs(), kdcodes=KDCODES)
    alone = realized_returns(panel, SESSIONS, Legs(), kdcodes=["GAP"])

    # Asked for GAP alone, the axis must still include session 3 (GAP's missing bar),
    # or every endpoint across it would shift by a session.
    assert alone.exit_dates == full.exit_dates
    np.testing.assert_array_equal(alone.returns[:, 0], full.returns[:, GAP])
    assert alone.returns[1, 0] == _close("GAP", 6) / _close("GAP", 2) - 1.0


def test_missing_entry_close_holds_cash_and_records_the_event():
    out = realized_returns(_panel(), SESSIONS, Legs(), kdcodes=KDCODES)

    # Signal on session 2: GAP has a close on the signal date but none at its entry
    # (session 3), so no position opens. The slot earns nothing.
    d = 2
    assert out.returns[d, GAP] == 0.0
    events = {(e.date, e.kdcode): e for e in out.events}
    event = events[(SESSIONS[d], "GAP")]
    assert event.kind == "missing_entry"
    assert event.scheduled_session == SESSIONS[GAP_MISSING]
    assert event.booked_session is None


def test_a_name_with_no_close_on_the_signal_date_has_no_outcome_and_no_event():
    out = realized_returns(_panel(), SESSIONS, Legs(), kdcodes=KDCODES)
    d = DLST_LAST + 1
    assert math.isnan(out.returns[d, DLST])
    assert not [e for e in out.events if e.date == SESSIONS[d] and e.kdcode == "DLST"]


def test_a_hold_past_the_panel_end_is_unresolved_not_booked():
    out = realized_returns(_panel(), SESSIONS, Legs(), kdcodes=KDCODES)
    last_resolved = len(SESSIONS) - 1 - 5
    assert np.isfinite(out.returns[last_resolved, KEEP])
    tail = out.returns[last_resolved + 1 :]
    assert np.isnan(tail).all()
    assert not [e for e in out.events if e.date in SESSIONS[last_resolved + 1 :]]


@pytest.mark.parametrize(
    "entry, exit_",
    [(0, 5), (-1, 5), (3, 3), (4, 2), (True, 5), (1.0, 5)],
)
def test_legs_reject_impossible_or_non_integer_sessions(entry, exit_):
    with pytest.raises(ValueError):
        Legs(entry_session=entry, exit_session=exit_)


# --- who is rankable ------------------------------------------------------------


def test_rankable_is_the_tradable_mask_and_keeps_a_name_with_no_future_price():
    panel = _panel()
    masks = build_pit_masks(panel, panel, KDCODES, SESSIONS[1:], 1, 5, _intervals())
    ranks = rankable(masks)

    np.testing.assert_array_equal(ranks, masks.tradable)
    # Signal on session 2: DLST trades that day but has no close at its exit, so it
    # has no training label. It must still be ranked.
    d = SESSIONS[1:].index(SESSIONS[2])
    assert not masks.loss[d, DLST]
    assert ranks[d, DLST]


def test_a_top_pick_that_stops_trading_keeps_its_slot_and_books_its_loss():
    panel = _panel()
    sample = SESSIONS[1:]
    masks = build_pit_masks(panel, panel, KDCODES, sample, 1, 5, _intervals())
    out = realized_returns(panel, sample, Legs(), kdcodes=KDCODES)
    # DLST ranked first, KEEP second.
    scores = np.tile(np.array([0.9, 0.1, 0.05, 0.5]), (len(sample), 1))

    d = sample.index(SESSIONS[1])
    tranche = top_k_tranche_returns(scores, rankable(masks), out, top_k=2)
    dlst = _close("DLST", 4) / _close("DLST", 2) - 1.0  # -0.25, booked at its last close
    keep = _close("KEEP", 6) / _close("KEEP", 2) - 1.0
    assert tranche[d] == pytest.approx((dlst + keep) / 2)

    # Session 1 is the first sample date; once its hold runs past the panel end, the
    # date is unresolved rather than averaged over whoever happens to have a close.
    assert np.isnan(tranche[-1])


# --- ties -----------------------------------------------------------------------

# Column order puts each later kdcode first, so an index tiebreak picks the wrong name.
TIE_KDCODES = ["X.N", "GOOGL.OQ", "GOOG.OQ", "V.N", "MA.N", "Y.N"]
TIE_SCORES = np.array([0.9, 0.5, 0.5, 0.25263, 0.25263, 0.1])


def test_ties_order_by_score_then_kdcode():
    order = order_by_score(TIE_SCORES, TIE_KDCODES)
    assert [TIE_KDCODES[i] for i in order] == [
        "X.N",
        "GOOG.OQ",
        "GOOGL.OQ",
        "MA.N",
        "V.N",
        "Y.N",
    ]
    assert [TIE_KDCODES[i] for i in select_top_k(TIE_SCORES, TIE_KDCODES, 2)] == ["X.N", "GOOG.OQ"]
    assert [TIE_KDCODES[i] for i in select_top_k(TIE_SCORES, TIE_KDCODES, 4)] == [
        "X.N",
        "GOOG.OQ",
        "GOOGL.OQ",
        "MA.N",
    ]


def test_tie_order_does_not_depend_on_column_order():
    reversed_kd = TIE_KDCODES[::-1]
    reversed_scores = TIE_SCORES[::-1]
    picks = [reversed_kd[i] for i in select_top_k(reversed_scores, reversed_kd, 4)]
    assert picks == ["X.N", "GOOG.OQ", "GOOGL.OQ", "MA.N"]


def test_unscored_names_order_last():
    scores = np.array([np.nan, 0.5, np.nan, 0.5])
    kd = ["D", "C", "A", "B"]
    order = [kd[i] for i in order_by_score(scores, kd)]
    assert order[:2] == ["B", "C"]
    assert set(order[2:]) == {"A", "D"}
    ranked = portfolio.rank_scores(pd.DataFrame({"kdcode": kd, "score": scores}))
    assert ranked["kdcode"].tolist()[:2] == ["B", "C"]
    assert ranked["score"].iloc[2:].isna().all()


def test_unscored_and_unrankable_names_are_never_selected():
    scores = np.array([np.nan, 0.5, 0.4, 0.3])
    kd = ["A", "B", "C", "D"]
    allowed = np.array([True, False, True, True])
    assert [kd[i] for i in select_top_k(scores, kd, 3, rankable=allowed)] == ["C", "D"]


def test_portfolio_top_k_uses_the_core_tie_rule():
    picks = portfolio.select_top_k(TIE_SCORES, 2, kdcodes=TIE_KDCODES)
    assert [TIE_KDCODES[i] for i in picks] == ["X.N", "GOOG.OQ"]

    ranked = portfolio.rank_scores(pd.DataFrame({"kdcode": TIE_KDCODES, "score": TIE_SCORES}))
    assert ranked["kdcode"].tolist()[:3] == ["X.N", "GOOG.OQ", "GOOGL.OQ"]

    returns = np.array([[0.0, -0.10, 0.20, 0.0, 0.0, 0.0]])
    out = portfolio.top_k_returns(TIE_SCORES[None, :], returns, top_k=2, kdcodes=TIE_KDCODES)
    assert out.tolist() == [pytest.approx((0.0 + 0.20) / 2)]


# --- evaluation config ----------------------------------------------------------


def test_evaluation_legs_and_tranches_default_to_the_training_label():
    cfg = EvaluationConfig()
    assert cfg.legs is None
    assert cfg.tranches is None
    assert cfg.resolved_legs(label_t=5) == Legs(1, 5)
    assert cfg.resolved_tranches(label_t=5) == 4
    assert cfg.resolved_legs(label_t=10) == Legs(1, 10)

    explicit = EvaluationConfig(legs={"entry_session": 2, "exit_session": 6}, tranches=2)
    assert explicit.resolved_legs(label_t=10) == Legs(2, 6)
    assert explicit.resolved_tranches(label_t=10) == 2
    assert explicit.resolved_tranches(label_t=10) != EvaluationConfig(
        legs={"entry_session": 2, "exit_session": 6}
    ).resolved_tranches(label_t=10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"legs": {"entry_session": 1, "exit_session": 5, "price": "open"}},
        {"legs": {"entry_session": 1}},
        {"legs": {"entry_session": 0, "exit_session": 5}},
        {"legs": {"entry_session": 3, "exit_session": 3}},
        {"legs": {"entry_session": 1, "exit_session": "5"}},
        {"legs": {"entry_session": True, "exit_session": 5}},
        {"legs": [1, 5]},
        {"tranches": 0},
        {"tranches": True},
        {"tranches": 2.0},
    ],
)
def test_evaluation_rejects_unknown_or_invalid_legs_and_tranches(kwargs):
    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)


def test_tranches_must_divide_the_hold_the_label_implies():
    with pytest.raises(ValueError, match="tranches"):
        ExperimentConfig(model=ModelConfig(label_t=5), evaluation=EvaluationConfig(tranches=3))
    ok = ExperimentConfig(model=ModelConfig(label_t=5), evaluation=EvaluationConfig(tranches=2))
    assert ok.evaluation.resolved_tranches(ok.model.label_t) == 2

    # Explicit legs: close D+1 to close D+7 is a six-session hold.
    seven = ModelConfig(label_t=7)
    legs = {"entry_session": 1, "exit_session": 7}
    with pytest.raises(ValueError, match="tranches"):
        ExperimentConfig(model=seven, evaluation=EvaluationConfig(legs=legs, tranches=4))
    ExperimentConfig(model=seven, evaluation=EvaluationConfig(legs=legs, tranches=3))


def test_legs_may_not_exit_past_the_embargoed_label_horizon():
    legs = {"entry_session": 1, "exit_session": 6}
    with pytest.raises(ValueError, match="label_t"):
        ExperimentConfig(model=ModelConfig(label_t=5), evaluation=EvaluationConfig(legs=legs))
    shorter = {"entry_session": 2, "exit_session": 5}
    ExperimentConfig(model=ModelConfig(label_t=5), evaluation=EvaluationConfig(legs=shorter))


def test_legs_and_tranches_survive_the_hydra_construction_paths():
    section = OmegaConf.create(
        {"legs": {"entry_session": 1, "exit_session": 5}, "tranches": 2, "top_k_values": [10]}
    )
    from_dict = create_config_from_dict({"evaluation": OmegaConf.to_container(section)})
    assert from_dict.evaluation.resolved_legs(from_dict.model.label_t) == Legs(1, 5)
    assert from_dict.evaluation.tranches == 2

    from_dictconfig = ExperimentConfig(evaluation=section)
    assert isinstance(from_dictconfig.evaluation, EvaluationConfig)
    assert from_dictconfig.evaluation.legs == {"entry_session": 1, "exit_session": 5}


def test_unknown_evaluation_key_is_rejected():
    with pytest.raises(TypeError):
        EvaluationConfig(tranche_count=4)


def test_shipped_config_declares_legs_and_tranches_following_the_label():
    block = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "configs" / "config.yaml"))[
        "evaluation"
    ]
    assert block["legs"] is None
    assert block["tranches"] is None
    EvaluationConfig(**block)
