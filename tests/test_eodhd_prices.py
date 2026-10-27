"""EODHD daily prices for the point-in-time universe (#281).

The pull itself needs the vendor; these tests drive the same code with canned
vendor rows: the symbol rule, the split adjustment, the map's validation, the
proof of each mapping against a reference panel, and the whole export through
to a manifest the run can verify.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import scripts.data.check_eodhd_dividends as dividend_check
import scripts.data.check_eodhd_volume_basis as volume_check
import scripts.data.export_eodhd_pit_prices as export
from mci_gru.config import create_config_from_dict
from mci_gru.data.eodhd_prices import (
    DIVIDEND,
    PriceAdjustment,
    SymbolMapError,
    agreement_passes,
    apply_adjustments,
    attach_dividends,
    build_symbol_plans,
    clean_values,
    default_symbol,
    dividend_basis_findings,
    dividends_frame,
    eod_rows_frame,
    is_delisted,
    load_symbol_map,
    name_hint_candidates,
    needed_spans,
    parse_split_ratio,
    parse_symbol_map,
    reference_check,
    return_agreement,
    split_adjust,
    split_basis_findings,
    splits_frame,
    trim_carried_tail,
)
from mci_gru.data.input_manifest import (
    InputFileSpec,
    read_input_manifest,
    validate_input_package,
    write_input_manifest,
)
from mci_gru.data.input_observations import InputObservationContext
from mci_gru.data.quality_contract import Verdict, assess_market_panel
from mci_gru.evaluation.run_input_declarations import declare_window_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gen_eodhd_price_pull_nb as generator  # noqa: E402

COMMITTED_MAP = REPO_ROOT / "data" / "mappings" / "eodhd_symbols_gics_top10_110_2016.json"
AAPL_2020_SPLIT = REPO_ROOT / "tests" / "fixtures" / "eodhd" / "aapl_2020_split.json"
CHTR_2016_MERGER = REPO_ROOT / "tests" / "fixtures" / "eodhd" / "chtr_2016_merger.json"


def _sessions(start: str, periods: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=periods)]


def _raw_rows(dates, closes, volume=1000.0, adjusted=None):
    adjusted = closes if adjusted is None else adjusted
    return [
        {
            "date": d,
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "adjusted_close": a,
            "volume": volume,
        }
        for d, c, a in zip(dates, closes, adjusted, strict=True)
    ]


def _walk(seed: int, n: int, start: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.cumprod(1 + rng.normal(0, 0.015, n))


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ric", "symbol"),
    [
        ("AAPL.OQ", "AAPL.US"),
        ("D.N", "D.US"),
        ("BRKb.N", "BRK-B.US"),
        ("ATVI.OQ^J23", "ATVI.US"),
        ("GOOGL.OQ", "GOOGL.US"),
        ("XYZ^A20", "XYZ.US"),
    ],
)
def test_default_symbol_follows_the_ric_rule(ric, symbol):
    assert default_symbol(ric) == symbol


def test_committed_symbol_map_parses_and_names_only_ric_shaped_identifiers():
    plans = load_symbol_map(COMMITTED_MAP)
    assert plans, "the committed map should carry overrides"
    for kdcode, plan in plans.items():
        assert "." in kdcode and kdcode == kdcode.strip()
        assert plan.overridden
    # The reused tickers must be cut at the merger, not read whole. EODHD has no
    # pre-merger DuPont history at all: its DD.US rows before then are Dow Chemical's.
    assert plans["DD.N^I17"].segments == () and "Dow Chemical" in plans["DD.N^I17"].unavailable
    assert plans["DD.N"].segments[0].start == "2017-09-01"
    # BLL became BALL at the open on 2022-05-10.
    assert [(s.end, s.start) for s in plans["BALL.N"].segments] == [
        ("2022-05-09", None),
        (None, "2022-05-10"),
    ]
    # Corporate actions the first pull (2026-10-08) showed EODHD's split records miss.
    (charter,) = plans["CHTR.OQ"].adjustments
    assert (charter.date, charter.share_conversion) == ("2016-05-18", True)
    assert charter.factor == pytest.approx(5000 / 4521)  # 0.9042 new shares per legacy share
    (baker,) = plans["BKR.OQ"].adjustments
    assert (baker.date, baker.cash, baker.factor) == ("2017-07-05", 17.5, None)
    # Only Ford's supplemental is restated; the vendor step would take the regular too.
    (ford,) = plans["F.N"].adjustments
    assert (ford.date, ford.cash) == ("2023-02-10", 0.65)
    vendor_dated = {k: [a.date for a in plans[k].adjustments] for k in ("COST.OQ", "EQR.N")}
    assert vendor_dated == {
        "COST.OQ": ["2015-02-05", "2017-05-08"],
        "EQR.N": ["2016-03-01", "2016-09-22"],
    }


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        (
            [
                {"candidates": ["A.US"], "end": "2020-01-10"},
                {"candidates": ["B.US"], "start": "2020-01-10"},
            ],
            "overlap",
        ),
        ([{"candidates": ["A.US"]}, {"candidates": ["B.US"], "start": "2021-01-01"}], "open-ended"),
        ([{"candidates": []}], "candidates or a name_hint"),
        ([{"candidates": ["A"]}], "'.US' symbols"),
        ([{"candidates": ["A.US"], "start": "2020-1-1"}], "YYYY-MM-DD"),
        ([{"candidates": ["A.US"], "typo": 1}], "unknown fields"),
    ],
)
def test_symbol_map_rejects_ambiguous_segments(segments, message):
    with pytest.raises(SymbolMapError, match=message):
        parse_symbol_map({"schema": 1, "overrides": {"A.N": {"segments": segments}}})


def test_plans_reject_overrides_for_names_outside_the_universe():
    overrides = parse_symbol_map(
        {"schema": 1, "overrides": {"ZZZ.N": {"segments": [{"candidates": ["ZZZ.US"]}]}}}
    )
    with pytest.raises(SymbolMapError, match="outside the universe"):
        build_symbol_plans(["AAPL.OQ"], overrides)


def test_name_hint_matches_common_stock_names_case_insensitively():
    listing = [
        {"Code": "DD", "Name": "E. I. du Pont de Nemours and Company", "Type": "Common Stock"},
        {"Code": "DD", "Name": "DuPont de Nemours Inc", "Type": "Common Stock"},
        {"Code": "DDX", "Name": "du Pont warrants", "Type": "Warrant"},
    ]
    assert name_hint_candidates(listing, "DU PONT") == ["DD.US"]


# ---------------------------------------------------------------------------
# Split adjustment
# ---------------------------------------------------------------------------


def test_parse_split_ratio_reads_new_over_old():
    assert parse_split_ratio("4.000000/1.000000") == 4.0
    assert parse_split_ratio("1/10") == pytest.approx(0.1)
    with pytest.raises(ValueError):
        parse_split_ratio("0/1")


def test_split_adjust_divides_prices_and_leaves_volume_as_delivered():
    """Issue 347: EODHD's volume is already split-adjusted; multiplying it applied the split twice."""
    dates = _sessions("2020-08-26", 6)
    raw = [400.0, 404.0, 408.0, 102.5, 103.0, 104.0]  # 4-for-1 on the fourth session
    eod = eod_rows_frame(
        _raw_rows(dates, raw, adjusted=[c / 4 if i < 3 else c for i, c in enumerate(raw)])
    )
    splits = splits_frame([{"date": dates[3], "split": "4.000000/1.000000"}], as_of=dates[-1])

    adjusted = split_adjust(eod, splits)

    assert adjusted["close"].tolist() == pytest.approx([100.0, 101.0, 102.0, 102.5, 103.0, 104.0])
    assert adjusted["volume"].tolist() == eod["volume"].tolist()
    # The return across the split is the economic one, not -75%.
    assert adjusted["close"].pct_change().iloc[3] == pytest.approx(102.5 / 102.0 - 1)
    assert split_basis_findings("AAPL.OQ", "AAPL.US", adjusted) == []


def _median_volume_step(frame: pd.DataFrame, split_dt: str) -> float:
    """Median volume over the sessions from the split on, over the sessions before it."""
    after = frame.loc[frame["dt"] >= split_dt, "volume"]
    before = frame.loc[frame["dt"] < split_dt, "volume"]
    return float(after.median() / before.median())


def test_volume_is_continuous_across_a_real_vendor_split():
    """Issue 347: the vendor's own AAPL rows around 2020-08-31 (4-for-1), as served.

    Real volume does not step at a split once both sides are counted in the same
    shares. Applied twice, the 20 sessions before read four times too high and the
    step is about 1/4.
    """
    vendor = json.loads(AAPL_2020_SPLIT.read_text(encoding="utf-8"))
    eod = eod_rows_frame(vendor["eod"])
    splits = splits_frame(vendor["splits"], as_of=eod["dt"].iloc[-1])
    assert splits["dt"].tolist() == ["2020-08-31"] and len(eod) == 40

    adjusted = split_adjust(eod, splits)

    assert 0.5 < _median_volume_step(adjusted, "2020-08-31") < 2.0
    assert adjusted["volume"].tolist() == eod["volume"].tolist()
    # The shares traded on 2020-08-28 were about 46.9M; the vendor serves 187.63M, four times that.
    assert adjusted.loc[adjusted["dt"] == "2020-08-28", "volume"].item() == 187_630_000
    assert np.abs(np.diff(np.log(adjusted["close"].to_numpy()))).max() < 0.1
    assert split_basis_findings("AAPL.OQ", "AAPL.US", adjusted) == []


def test_splits_after_the_panel_end_come_out_of_volume_only():
    """Issue 347: the vendor's volume carries splits after the panel end; the prices do not.

    MNST split 2-for-1 on 2026-08-11, after the 2026-07-31 panel end, and the vendor's
    July volume already counts post-split shares. The panel's volume is put back on
    the panel end's basis so it matches the prices beside it.
    """
    dates = _sessions("2026-07-27", 5)
    eod = eod_rows_frame(_raw_rows(dates, [94.0, 95.0, 96.0, 95.0, 94.5], volume=12_000.0))
    splits = splits_frame([], as_of=dates[-1])
    vendor_splits = [{"date": "2026-08-11", "split": "2.000000/1.000000"}]
    later = splits_frame(vendor_splits, as_of="2026-10-08", after=dates[-1])

    adjusted = split_adjust(eod, splits, later)

    assert adjusted["close"].tolist() == eod["close"].tolist()
    assert adjusted["volume"].tolist() == [6_000.0] * 5
    assert splits_frame(vendor_splits, as_of=dates[-1]).empty


def test_splits_frame_keeps_only_the_dates_between_after_and_as_of():
    rows = [
        {"date": "2020-01-02", "split": "2/1"},
        {"date": "2026-08-11", "split": "2/1"},
        {"date": "2026-10-01", "split": "1333/200"},
        {"date": "2027-01-04", "split": "10/1"},
    ]
    later = splits_frame(rows, as_of="2026-10-08", after="2026-07-31")
    assert later["dt"].tolist() == ["2026-08-11", "2026-10-01"]
    assert later["ratio"].tolist() == pytest.approx([2.0, 6.665])


def test_chtrs_real_vendor_rows_restate_to_new_shares_across_the_merger():
    """Issue 347: CHTR's own vendor rows around the 2016-05-18 conversion, through the map's entry.

    EODHD's pre-merger closes are the legacy closes times 0.9042, and its 4521/5000
    split record divides that back out; the map's conversion then restates prices per
    new share. The vendor's volume is the legacy share count (21,642,600 on 2016-05-17,
    the same as Yahoo's), so only the conversion rescales it.
    """
    vendor = json.loads(CHTR_2016_MERGER.read_text(encoding="utf-8"))
    eod = eod_rows_frame(vendor["eod"])
    splits = splits_frame(vendor["splits"], as_of="2026-07-31")
    assert splits["dt"].tolist() == ["2016-05-18"]
    (conversion,) = load_symbol_map(COMMITTED_MAP)["CHTR.OQ"].adjustments

    out, applied, findings = apply_adjustments("CHTR.OQ", split_adjust(eod, splits), [conversion])

    assert not findings and applied[0]["share_conversion"]
    day = out.set_index("dt")
    # LSEG's return that day is +1.11%; with a 0.9 ratio it would read +0.65%.
    assert day.loc["2016-05-18", "close"] / day.loc["2016-05-17", "close"] - 1 == pytest.approx(
        0.0111, abs=5e-4
    )
    before = eod.set_index("dt").loc["2016-05-17", "volume"]
    assert before == 21_642_600
    assert day.loc["2016-05-17", "volume"] == pytest.approx(before * 4521 / 5000)
    assert day.loc["2016-05-18", "volume"] == eod.set_index("dt").loc["2016-05-18", "volume"]


def test_splits_after_the_panel_end_are_not_applied():
    frame = splits_frame(
        [{"date": "2020-01-02", "split": "2/1"}, {"date": "2027-01-04", "split": "10/1"}],
        as_of="2026-07-31",
    )
    assert frame["dt"].tolist() == ["2020-01-02"]


def test_a_split_the_raw_prices_already_carry_is_caught():
    dates = _sessions("2020-08-26", 6)
    already = [100.0, 101.0, 102.0, 102.5, 103.0, 104.0]
    eod = eod_rows_frame(_raw_rows(dates, already))
    splits = splits_frame([{"date": dates[3], "split": "4/1"}], as_of=dates[-1])

    findings = split_basis_findings("AAPL.OQ", "AAPL.US", split_adjust(eod, splits))

    assert [f.code for f in findings] == ["split_basis_jump"]
    assert findings[0].blocking
    assert findings[0].evidence["dates"] == [dates[3]]


def test_clean_values_blanks_nonpositive_prices_and_negative_volume():
    frame = eod_rows_frame(_raw_rows(_sessions("2021-01-04", 3), [10.0, 0.0, 11.0]))
    frame.loc[2, "volume"] = -5
    cleaned, blanked = clean_values(frame)
    assert np.isnan(cleaned.loc[1, "close"]) and np.isnan(cleaned.loc[2, "volume"])
    assert blanked == 4 + 1  # all four prices of row 1 are 0, plus one volume


def test_a_declared_spin_off_rescales_earlier_prices_only():
    dates = _sessions("2023-12-26", 6)
    raw = [100.0, 101.0, 102.0, 80.0, 81.0, 82.0]  # the spin-off removes 20 per share on day 3
    adjusted = [c * 0.8 if i < 3 else c for i, c in enumerate(raw)]
    frame = eod_rows_frame(_raw_rows(dates, raw, adjusted=adjusted))

    declared, applied, findings = apply_adjustments(
        "GE.N", frame, [PriceAdjustment(dates[3], "spin-off", factor=0.8)]
    )
    vendor, vendor_applied, _ = apply_adjustments(
        "GE.N", frame, [PriceAdjustment(dates[3], "spin-off")]
    )

    assert declared["close"].tolist() == pytest.approx([80.0, 80.8, 81.6, 80.0, 81.0, 82.0])
    assert declared["volume"].tolist() == frame["volume"].tolist()
    assert vendor["close"].tolist() == pytest.approx(declared["close"].tolist())
    assert vendor_applied[0]["factor"] == pytest.approx(0.8)
    assert vendor_applied[0]["basis"] == "vendor_adjusted_close"
    assert applied[0]["basis"] == "declared_factor" and not findings


def test_a_vendor_factor_with_no_rows_on_its_date_blocks():
    frame = eod_rows_frame(_raw_rows(_sessions("2024-01-02", 3), [10.0, 11.0, 12.0]))
    _, applied, findings = apply_adjustments(
        "X.N", frame, [PriceAdjustment("2025-06-02", "spin-off")]
    )
    assert not applied
    assert [(f.code, f.blocking) for f in findings] == [("adjustment_unresolved", True)]


def test_a_cash_distribution_uses_the_last_close_before_its_date():
    dates = _sessions("2017-06-28", 5)
    raw = [55.0, 54.0, 54.6, 35.3, 35.0]  # $17.50 cash plus a new share on day 3
    frame = eod_rows_frame(_raw_rows(dates, raw))
    out, applied, findings = apply_adjustments(
        "BKR.OQ", frame, [PriceAdjustment(dates[3], "special dividend", cash=17.5)]
    )
    factor = 1 - 17.5 / 54.6
    assert not findings
    assert applied[0]["basis"] == "declared_cash"
    assert applied[0]["factor"] == pytest.approx(factor)
    assert out["close"].tolist() == pytest.approx([c * factor for c in raw[:3]] + raw[3:])
    assert out["volume"].tolist() == frame["volume"].tolist()


def test_a_cash_distribution_above_the_prior_close_blocks():
    dates = _sessions("2017-06-28", 3)
    frame = eod_rows_frame(_raw_rows(dates, [10.0, 11.0, 12.0]))
    out, applied, findings = apply_adjustments(
        "X.N", frame, [PriceAdjustment(dates[2], "typo", cash=50.0)]
    )
    assert not applied
    assert [(f.code, f.blocking) for f in findings] == [("adjustment_unresolved", True)]
    assert out["close"].tolist() == frame["close"].tolist()


def test_a_share_conversion_rescales_volume_as_well_as_prices():
    dates = _sessions("2016-05-13", 5)
    raw = [180.0, 182.0, 184.0, 205.0, 206.0]  # 0.9 new shares per old share on day 3
    frame = eod_rows_frame(_raw_rows(dates, raw, volume=900.0))
    out, applied, _ = apply_adjustments(
        "CHTR.OQ",
        frame,
        [PriceAdjustment(dates[3], "merger", factor=1 / 0.9, share_conversion=True)],
    )
    assert out["close"].tolist() == pytest.approx(
        [200.0, 202.0 + 2 / 9, 204.0 + 4 / 9, 205.0, 206.0]
    )
    assert out["volume"].tolist() == pytest.approx([810.0] * 3 + [900.0] * 2)
    assert applied[0]["share_conversion"] is True


def test_rows_with_nothing_before_the_date_need_no_adjustment():
    dates = _sessions("2017-07-05", 3)
    frame = eod_rows_frame(_raw_rows(dates, [35.0, 36.0, 37.0]))
    out, applied, findings = apply_adjustments(
        "BKR.OQ",
        frame,
        [
            PriceAdjustment(dates[0], "special dividend", cash=17.5),
            PriceAdjustment(dates[0], "spin-off"),
        ],
    )
    assert [a["basis"] for a in applied] == ["no_rows_before", "no_rows_before"]
    assert [a["factor"] for a in applied] == [None, None]
    # Disclosed rather than silent, so a mistyped date stays visible; it never blocks.
    assert {f.code for f in findings} == {"adjustment_not_applied"}
    assert not [f for f in findings if f.blocking]
    assert out["close"].tolist() == frame["close"].tolist()


# ---------------------------------------------------------------------------
# Dividends (#361)
# ---------------------------------------------------------------------------


def _div(date: str, amount: float, adjusted: float | None = None) -> dict:
    """An EODHD /div record; ``value`` is restated for later splits, ``unadjustedValue`` is not."""
    return {
        "date": date,
        "value": amount if adjusted is None else adjusted,
        "unadjustedValue": amount,
        "period": "Quarterly",
        "currency": "USD",
    }


def _with_dividends(dates, closes, dividends, **kwargs) -> pd.DataFrame:
    """Raw vendor rows whose adjusted_close takes the step each dividend implies."""
    paid = dict(dividends)
    factor, adjusted = 1.0, []
    for i in range(len(dates) - 1, -1, -1):
        adjusted.append(closes[i] * factor)
        if dates[i] in paid and i > 0:
            factor *= 1 - paid[dates[i]] / closes[i - 1]
    frame = eod_rows_frame(_raw_rows(dates, closes, adjusted=adjusted[::-1], **kwargs))
    records = dividends_frame([_div(d, a) for d, a in dividends])
    out, findings = attach_dividends("X.N", "X.US", frame, records)
    assert not findings
    return out


def test_dividends_are_read_as_paid_and_summed_on_one_ex_date():
    frame = dividends_frame(
        [
            _div("2020-08-07", 0.82, adjusted=0.205),  # before AAPL's 4-for-1
            _div("2023-02-10", 0.15),
            _div("2023-02-10", 0.65),  # a supplemental listed as its own record
        ]
    )
    assert frame.to_dict("records") == [
        {"dt": "2020-08-07", "amount": 0.82},
        {"dt": "2023-02-10", "amount": pytest.approx(0.80)},
    ]
    with pytest.raises(ValueError, match="nonnegative"):
        dividends_frame([_div("2023-02-10", -0.15)])


def test_a_dividend_sits_on_its_ex_date_and_a_missing_session_moves_it_to_the_next():
    dates = _sessions("2024-03-04", 5)
    frame = eod_rows_frame(_raw_rows(dates, [10.0, 11.0, 12.0, 13.0, 14.0]))
    gap = frame.drop(index=2).reset_index(drop=True)  # no vendor row on dates[2]
    records = dividends_frame([_div(dates[1], 0.25), _div(dates[2], 0.5), _div("2024-03-29", 0.75)])

    out, findings = attach_dividends("X.N", "X.US", gap, records)

    assert out[DIVIDEND].tolist() == [0.0, 0.25, 0.5, 0.0]
    assert out.loc[out[DIVIDEND] == 0.5, "dt"].item() == dates[3]
    assert [(f.code, f.blocking) for f in findings] == [
        ("dividend_off_session", False),
        ("dividend_after_last_row", False),
    ]
    assert findings[0].evidence["moved_to"] == {dates[2]: dates[3]}
    assert findings[1].evidence["dividends"] == {"2024-03-29": 0.75}


def test_a_split_rescales_dividends_before_it_as_it_rescales_prices():
    """AAPL paid $0.82 on 2020-08-07, before its 4-for-1; on the panel's basis that is $0.205."""
    dates = _sessions("2020-08-05", 6)
    raw = [440.0, 455.0, 444.0, 450.0, 125.0, 126.0]  # ex-dividend on 2, 4-for-1 on 4
    frame = eod_rows_frame(_raw_rows(dates, raw))
    records = dividends_frame([_div(dates[2], 0.82, adjusted=0.205), _div(dates[5], 0.205)])
    frame, _ = attach_dividends("AAPL.OQ", "AAPL.US", frame, records)
    splits = splits_frame([{"date": dates[4], "split": "4.000000/1.000000"}], as_of=dates[-1])

    adjusted = split_adjust(frame, splits)

    assert adjusted[DIVIDEND].tolist() == pytest.approx([0, 0, 0.205, 0, 0, 0.205])
    assert adjusted["close"].tolist() == pytest.approx([110, 113.75, 111, 112.5, 125, 126])


def test_dividends_agree_with_the_vendors_adjusted_close_step():
    dates = _sessions("2024-03-04", 6)
    closes = [50.0, 50.5, 49.8, 50.2, 50.4, 50.1]
    frame = _with_dividends(dates, closes, [(dates[2], 0.5), (dates[4], 0.25)])
    assert dividend_basis_findings("X.N", "X.US", frame) == []

    missing = frame.assign(**{DIVIDEND: frame[DIVIDEND].where(frame["dt"] != dates[4], 0.0)})
    doubled = frame.assign(**{DIVIDEND: frame[DIVIDEND] * 2})
    early = frame.assign(**{DIVIDEND: frame[DIVIDEND].shift(-1, fill_value=0.0)})

    (finding,) = dividend_basis_findings("X.N", "X.US", missing)
    assert (finding.code, finding.blocking) == ("dividend_disagrees_with_adjusted_close", False)
    assert list(finding.evidence["dates"]) == [dates[4]]
    assert finding.evidence["dates"][dates[4]]["adjusted_close_yield"] == pytest.approx(
        0.25 / 50.2, abs=1e-6
    )
    assert dividend_basis_findings("X.N", "X.US", doubled)[0].evidence["sessions"] == 2
    assert list(dividend_basis_findings("X.N", "X.US", early)[0].evidence["dates"]) == [
        dates[1],
        dates[2],
        dates[3],
        dates[4],
    ]
    # A date the map declares is the declared action's step, not a dividend's.
    assert dividend_basis_findings("X.N", "X.US", missing, exempt=[dates[4]]) == []


def test_a_vendor_step_adjustment_leaves_no_dividend_on_its_date():
    """LYB 2022-06-03: one $6.39 record (special plus regular); the step restates both."""
    dates = _sessions("2022-05-31", 5)
    closes = [100.0, 101.0, 102.0, 96.0, 97.0]
    frame = _with_dividends(dates, closes, [(dates[0], 1.0), (dates[3], 6.39)])

    out, applied, findings = apply_adjustments(
        "LYB.N", frame, [PriceAdjustment(dates[3], "special dividend")]
    )

    factor = 1 - 6.39 / 102.0
    assert applied[0]["basis"] == "vendor_adjusted_close"
    assert applied[0]["factor"] == pytest.approx(factor)
    assert (applied[0]["dividend_vendor"], applied[0]["dividend_kept"]) == (6.39, 0.0)
    assert out[DIVIDEND].tolist() == pytest.approx([factor, 0, 0, 0, 0])  # earlier ones rescale
    assert out["close"].tolist() == pytest.approx([c * factor for c in closes[:3]] + closes[3:])
    assert not findings


def test_a_cash_adjustment_keeps_the_dividend_above_the_declared_cash():
    """Ford 2023-02-10: one $0.80 record; the map restates the $0.65 supplemental only."""
    dates = _sessions("2023-02-08", 4)
    closes = [13.0, 13.2, 12.4, 12.5]
    frame = _with_dividends(dates, closes, [(dates[2], 0.80)])

    out, applied, findings = apply_adjustments(
        "F.N", frame, [PriceAdjustment(dates[2], "supplemental", cash=0.65)]
    )

    assert applied[0]["basis"] == "declared_cash"
    assert applied[0]["dividend_kept"] == pytest.approx(0.15)
    assert out[DIVIDEND].tolist() == pytest.approx([0, 0, 0.15, 0])
    assert not findings
    # The price-only return across the date plus the dividend kept is the total return.
    total = (out["close"].iloc[2] + out[DIVIDEND].iloc[2]) / out["close"].iloc[1]
    assert total == pytest.approx((12.4 + 0.80) / 13.2)


def test_merger_cash_listed_as_a_dividend_is_not_counted_twice():
    """BKR 2017-07-05: /api/div lists the $17.50 merger cash the map already restates."""
    dates = _sessions("2017-06-30", 4)
    frame = _with_dividends(dates, [54.0, 54.6, 35.3, 35.0], [(dates[2], 17.5)])
    out, applied, findings = apply_adjustments(
        "BKR.OQ", frame, [PriceAdjustment(dates[2], "merger cash", cash=17.5)]
    )
    assert out[DIVIDEND].tolist() == [0.0] * 4
    assert applied[0]["dividend_kept"] == 0.0 and not findings

    short = frame.assign(**{DIVIDEND: frame[DIVIDEND] / 2})
    out, _, findings = apply_adjustments(
        "BKR.OQ", short, [PriceAdjustment(dates[2], "merger cash", cash=17.5)]
    )
    assert out[DIVIDEND].tolist() == [0.0] * 4
    assert [(f.code, f.blocking) for f in findings] == [("dividend_below_declared_cash", False)]


def test_a_share_conversion_keeps_the_dividend_on_its_date():
    dates = _sessions("2016-05-13", 4)
    frame = _with_dividends(dates, [180.0, 182.0, 205.0, 206.0], [(dates[0], 1.0), (dates[2], 0.5)])
    out, applied, _ = apply_adjustments(
        "CHTR.OQ",
        frame,
        [PriceAdjustment(dates[2], "merger", factor=1 / 0.9, share_conversion=True)],
    )
    assert out[DIVIDEND].tolist() == pytest.approx([1 / 0.9, 0, 0.5, 0])
    assert "dividend_kept" not in applied[0]


def test_clean_values_reads_a_bad_dividend_as_no_payment():
    frame = eod_rows_frame(_raw_rows(_sessions("2024-01-02", 3), [10.0, 11.0, 12.0]))
    frame[DIVIDEND] = [np.nan, -0.1, 0.2]
    out, blanked = clean_values(frame)
    assert out[DIVIDEND].tolist() == [0.0, 0.0, 0.2]
    assert blanked == 2


def _tail_frame(closes, volumes):
    dates = _sessions("2023-10-09", len(closes))
    rows = _raw_rows(dates, closes)
    for row, volume in zip(rows, volumes, strict=True):
        row["volume"] = volume
    return eod_rows_frame(rows), dates


def test_a_carried_close_at_zero_volume_is_trimmed_from_the_tail():
    frame, dates = _tail_frame([10.0, 11.0, 12.0, 12.0, 12.0], [5.0, 5.0, 5.0, 0.0, 0.0])
    out, dropped = trim_carried_tail(frame)
    assert dropped == dates[3:]
    assert out["dt"].tolist() == dates[:3]


@pytest.mark.parametrize(
    ("closes", "volumes"),
    [
        ([10.0, 11.0, 11.0], [5.0, 5.0, 3.0]),  # a repeat close that traded
        ([10.0, 11.0, 11.5], [5.0, 5.0, 0.0]),  # zero volume at a new price
        ([10.0, 11.0, 11.0], [5.0, 5.0, np.nan]),  # volume unknown, not zero
        ([10.0, 11.0, 11.0, 12.0], [5.0, 0.0, 0.0, 5.0]),  # carried rows mid-history
    ],
)
def test_only_a_final_zero_volume_repeat_counts_as_carried(closes, volumes):
    frame, _ = _tail_frame(closes, volumes)
    out, dropped = trim_carried_tail(frame)
    assert dropped == []
    assert len(out) == len(frame)


def test_only_lseg_delisted_codes_are_trimmed():
    assert is_delisted("ATVI.OQ^J23")
    assert not is_delisted("PSKY.OQ")


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"factor": 0.9, "cash": 1.0}, "not both"),
        ({"share_conversion": True}, "stated factor"),
        ({"factor": 1.1, "share_conversion": "yes"}, "true or false"),
        ({"cash": -1.0}, "cash must be a positive number"),
        ({"cash": True}, "cash must be a positive number"),
    ],
)
def test_symbol_map_rejects_contradictory_adjustments(entry, message):
    adjustment = {"date": "2016-05-18", "reason": "test", **entry}
    payload = {"schema": 1, "overrides": {"CHTR.OQ": {"adjustments": [adjustment]}}}
    with pytest.raises(SymbolMapError, match=message):
        parse_symbol_map(payload)


def test_symbol_map_reads_cash_and_share_conversion():
    payload = {
        "schema": 1,
        "overrides": {
            "CHTR.OQ": {
                "adjustments": [
                    {
                        "date": "2016-05-18",
                        "reason": "merger",
                        "factor": 1.25,
                        "share_conversion": True,
                    }
                ]
            },
            "BKR.OQ": {"adjustments": [{"date": "2017-07-05", "reason": "cash", "cash": 17.5}]},
        },
    }
    overrides = parse_symbol_map(payload)
    assert overrides["CHTR.OQ"].adjustments == (
        PriceAdjustment("2016-05-18", "merger", factor=1.25, share_conversion=True),
    )
    assert overrides["BKR.OQ"].adjustments == (PriceAdjustment("2017-07-05", "cash", cash=17.5),)


def test_a_dividend_sized_step_is_quiet_and_a_larger_one_is_reported():
    dates = _sessions("2024-03-25", 5)
    closes = [100.0, 100.0, 100.0, 100.0, 100.0]
    adjusted = [0.99 * 0.9, 0.99 * 0.9, 0.9, 1 * 100 / 100, 1.0]
    adjusted = [a * 100 for a in adjusted]  # a 1% dividend step, then a 10% step
    frame = eod_rows_frame(_raw_rows(dates, closes, adjusted=adjusted))
    findings = split_basis_findings("MMM.N", "MMM.US", frame)
    assert [(f.code, f.blocking) for f in findings] == [("large_adjustment_step", False)]
    assert list(findings[0].evidence["steps"]) == [dates[3]]
    assert split_basis_findings("MMM.N", "MMM.US", frame, exempt=[dates[3]]) == []


def test_needed_spans_refuse_blank_valid_to():
    pit = pd.DataFrame({"kdcode": ["A.N"], "valid_from": ["2020-01-02"], "valid_to": [""]})
    with pytest.raises(ValueError, match="blank"):
        needed_spans(pit)


def test_symbol_map_reads_adjustments_and_accepted_differences():
    plans = parse_symbol_map(
        {
            "schema": 1,
            "overrides": {
                "GE.N": {
                    "adjustments": [{"date": "2024-04-02", "reason": "GE Vernova spin-off"}],
                    "accepted_differences": [{"date": "2019-01-02", "reason": "bad tick"}],
                }
            },
        }
    )
    plan = plans["GE.N"]
    assert plan.segments[0].candidates == ("GE.US",)
    assert plan.adjustments == (PriceAdjustment("2024-04-02", "GE Vernova spin-off"),)
    assert plan.accepted_differences == ("2019-01-02",)
    with pytest.raises(SymbolMapError, match="reason"):
        parse_symbol_map(
            {"schema": 1, "overrides": {"GE.N": {"adjustments": [{"date": "2024-04-02"}]}}}
        )


# ---------------------------------------------------------------------------
# Proof against the reference panel
# ---------------------------------------------------------------------------


def _reference(kdcode: str, dates, closes) -> pd.DataFrame:
    return pd.DataFrame({"kdcode": kdcode, "dt": dates, "close": closes})


def test_matching_returns_at_a_different_price_level_pass():
    dates = _sessions("2016-01-04", 120)
    closes = _walk(1, 120)
    mine = pd.DataFrame({"dt": dates, "close": closes / 3})  # level differs, returns do not
    stats = return_agreement(mine, _reference("X.N", dates, closes), dates[0], dates[-1])
    assert stats["correlation"] == pytest.approx(1.0)
    assert agreement_passes(stats)


def test_another_company_with_the_same_ticker_fails():
    dates = _sessions("2016-01-04", 120)
    mine = pd.DataFrame({"dt": dates, "close": _walk(2, 120)})
    stats = return_agreement(mine, _reference("X.N", dates, _walk(1, 120)), dates[0], dates[-1])
    assert not agreement_passes(stats)


def test_uncorrelated_quiet_series_fail_even_when_differences_are_small():
    # Two unrelated low-volatility series differ by little each day, so only the
    # correlation can tell them apart.
    dates = _sessions("2016-01-04", 250)
    rng = np.random.default_rng(5)
    ref = 100 * np.cumprod(1 + rng.normal(0, 0.001, 250))
    mine = 100 * np.cumprod(1 + rng.normal(0, 0.001, 250))
    stats = return_agreement(
        pd.DataFrame({"dt": dates, "close": mine}),
        _reference("X.N", dates, ref),
        dates[0],
        dates[-1],
    )
    assert stats["median_abs_diff"] < 0.002
    assert not agreement_passes(stats)


def test_one_large_daily_difference_fails_unless_the_map_accepts_it():
    dates = _sessions("2016-01-04", 2600)
    closes = _walk(6, 2600)
    mine = closes.copy()
    mine[:1300] *= 1.25  # a 20% level gap no split explains, e.g. an unadjusted spin-off
    frame = pd.DataFrame({"dt": dates, "close": mine})
    reference = _reference("X.N", dates, closes)

    stats = return_agreement(frame, reference, dates[0], dates[-1])
    assert stats["median_abs_diff"] < 0.002  # the whole-window statistics look fine
    assert stats["unexplained_large_diffs"] == 1
    assert stats["unexplained_large_diff_dates"][0]["dt"] == dates[1300]
    assert not agreement_passes(stats)

    accepted = return_agreement(frame, reference, dates[0], dates[-1], accepted=[dates[1300]])
    assert accepted["accepted_large_diffs"] == 1
    assert agreement_passes(accepted)


def test_a_short_window_with_full_coverage_passes_without_a_correlation():
    dates = _sessions("2018-10-01", 12)
    closes = _walk(7, 12)
    stats = return_agreement(
        pd.DataFrame({"dt": dates, "close": closes}),
        _reference("X.N", dates, closes),
        dates[0],
        dates[-1],
    )
    assert stats["correlation"] is None and stats["return_pairs"] == 11
    assert agreement_passes(stats)


def test_missing_sessions_inside_the_needed_span_fail():
    dates = _sessions("2016-01-04", 300)
    closes = _walk(3, 300)
    mine = pd.DataFrame({"dt": dates[:250], "close": closes[:250]})
    stats = return_agreement(mine, _reference("X.N", dates, closes), dates[0], dates[-1])
    assert stats["missing_sessions"] == 50
    assert not agreement_passes(stats)


def test_reference_check_only_judges_the_span_the_universe_needs():
    dates = _sessions("2015-01-02", 600)
    closes = _walk(4, 600)
    reference = _reference("X.N", dates, closes)
    # EODHD lacks the first 200 sessions, but the name is needed only from 2016-06.
    panel = pd.DataFrame(
        {"kdcode": "X.N", "dt": dates[200:], "close": closes[200:], "adjusted_close": closes[200:]}
    )
    pit = pd.DataFrame(
        {"kdcode": ["X.N"], "valid_from": ["2017-01-03"], "valid_to": ["2017-03-31"]}
    )

    check, findings = reference_check(panel, reference, pit)

    assert check.loc[0, "check_start"] == "2016-01-04"
    assert check.loc[0, "verdict"] == "pass", check.to_dict("records")
    assert not [f for f in findings if f.blocking]


def test_rows_for_a_name_declared_unavailable_block():
    dates = _sessions("2016-01-04", 40)
    closes = _walk(5, 40)
    reference = _reference("X.N", dates, closes)
    panel = pd.DataFrame({"kdcode": "X.N", "dt": dates, "close": closes, "adjusted_close": closes})
    pit = pd.DataFrame(
        {"kdcode": ["X.N"], "valid_from": ["2016-01-04"], "valid_to": ["2016-02-26"]}
    )

    check, findings = reference_check(panel, reference, pit, unavailable=["X.N"])
    _, clean = reference_check(panel.iloc[0:0], reference, pit, unavailable=["X.N"])

    assert check.loc[0, "verdict"] == "declared_unavailable"
    assert [(f.code, f.blocking) for f in findings] == [("declared_unavailable_has_rows", True)]
    assert not clean


# ---------------------------------------------------------------------------
# The whole export, with a canned vendor
# ---------------------------------------------------------------------------


class FakeClient:
    """Stands in for EodhdClient; serves canned rows per symbol."""

    eod_rows: dict[str, list[dict]] = {}
    split_rows: dict[str, list[dict]] = {}
    div_rows: dict[str, list[dict]] = {}
    listings: dict[bool, list[dict]] = {}
    #: When the canned vendor "served" its rows.
    served_at = "2016-10-03T12:00:00+00:00"
    #: Every client the export made, so a test can read what was asked of it.
    instances: list[FakeClient] = []

    def __init__(self, api_key, cache_dir, raw_log):
        FakeClient.instances.append(self)
        self.calls = 0
        self._raw = raw_log
        self.split_requests: list[tuple[str, str, str]] = []
        self.last_retrieved_at = None

    def eod(self, symbol, start, end):
        self.calls += 1
        self.last_retrieved_at = self.served_at
        self._raw.append({"endpoint": f"eod/{symbol}", "body": self.eod_rows.get(symbol, [])})
        return self.eod_rows.get(symbol, [])

    def splits(self, symbol, start, end):
        self.calls += 1
        self.split_requests.append((symbol, start, end))
        return [r for r in self.split_rows.get(symbol, []) if start <= r["date"] <= end]

    def dividends(self, symbol, start, end):
        self.calls += 1
        return [r for r in self.div_rows.get(symbol, []) if start <= r["date"] <= end]

    def listing(self, delisted):
        self.calls += 1
        return self.listings.get(delisted, [])


@pytest.fixture
def vendor(tmp_path, monkeypatch):
    dates = _sessions("2015-01-02", 400)
    good = _walk(10, 400)
    old = _walk(11, 400)
    impostor = _walk(12, 400)

    source = tmp_path / "source"
    (source / "constituents").mkdir(parents=True)
    prefix = "toy_universe"
    pit = source / "constituents" / f"{prefix}_pit_universe.csv"
    pit.write_text(
        "kdcode,valid_from,valid_to\nGOOD.OQ,2015-06-01,\nOLD.N^A16,2015-06-01,2016-01-29\n",
        encoding="utf-8",
    )
    reference = source / "reference.csv"
    pd.concat(
        [
            pd.DataFrame({"kdcode": "GOOD.OQ", "dt": dates, "close": good}),
            pd.DataFrame({"kdcode": "OLD.N^A16", "dt": dates[:280], "close": old[:280]}),
        ]
    ).to_csv(reference, index=False)
    symbol_map = source / "map.json"
    symbol_map.write_text(
        json.dumps(
            {
                "schema": 1,
                "overrides": {
                    "OLD.N^A16": {
                        "segments": [
                            {"candidates": ["GONE.US", "OLD.US"], "name_hint": "Old Company"}
                        ]
                    },
                    "GOOD.OQ": {
                        "adjustments": [{"date": dates[300], "reason": "spin-off"}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    # GOOD: raw prices carry a 2-for-1 split on session 100 that the split record undoes,
    # and a spin-off on session 300 that only adjusted_close carries (step 0.8). It pays
    # $1.00 a share on session 50, before both, and $0.60 on session 350; adjusted_close
    # steps by each one's yield on the prices' basis.
    raw_good = np.where(np.arange(400) < 100, good * 2, good)
    raw_good = np.where(np.arange(400) < 300, raw_good / 0.8, raw_good)
    adjusted_good = good.copy()
    adjusted_good[:350] *= 1 - 0.6 / good[349]
    adjusted_good[:50] *= 1 - (1.0 / 2) / (good[49] / 0.8)
    FakeClient.eod_rows = {
        "GOOD.US": _raw_rows(dates, list(raw_good), adjusted=list(adjusted_good)),
        # OLD.US is now someone else; the old company sits under OLD_old in the delisted list.
        "OLD.US": _raw_rows(dates, list(impostor)),
        "OLD_OLD.US": _raw_rows(dates[:280], list(old[:280])),
    }
    FakeClient.split_rows = {"GOOD.US": [{"date": dates[100], "split": "2.000000/1.000000"}]}
    FakeClient.div_rows = {"GOOD.US": [_div(dates[50], 1.0, adjusted=0.5), _div(dates[350], 0.6)]}
    FakeClient.listings = {
        True: [{"Code": "OLD_OLD", "Name": "Old Company Inc", "Type": "Common Stock"}],
        False: [],
    }
    monkeypatch.setattr(export, "EodhdClient", FakeClient)
    return {
        "pit": pit,
        "reference": reference,
        "map": symbol_map,
        "prefix": prefix,
        "dates": dates,
        "tmp": tmp_path,
    }


def _argv(vendor, **extra):
    argv = [
        "--pit-universe", str(vendor["pit"]),
        "--reference-panel", str(vendor["reference"]),
        "--symbol-map", str(vendor["map"]),
        "--start", "2015-01-01",
        "--end", "2016-07-29",
        "--pit-export-cutoff", "2016-06-30",
        "--package-root", str(vendor["tmp"] / "package"),
        "--cache-dir", str(vendor["tmp"] / "cache"),
    ]  # fmt: skip
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return argv


def test_export_proves_each_mapping_and_publishes_a_verifiable_package(vendor):
    manifest = vendor["tmp"] / "out" / "toy.r1.json"
    manifest.parent.mkdir()

    assert export.main(_argv(vendor, manifest_output=manifest)) == 0

    root = vendor["tmp"] / "package"
    stem = f"{vendor['prefix']}_eodhd_20150101_20160729"
    panel = pd.read_csv(root / "market" / f"{stem}.csv", dtype={"dt": str})
    assert list(panel.columns) == [
        "kdcode", "dt", "open", "high", "low", "close", "volume", "dividend"
    ]  # fmt: skip
    findings, _ = assess_market_panel(panel, role="data.filename", configured_path=None)
    assert not [f for f in findings if f.verdict is Verdict.INVALID]

    symbols = json.loads((root / "market" / f"{stem}_symbols.json").read_text())
    old = symbols["OLD.N^A16"]["segments"][0]
    assert old["symbol"] == "OLD_OLD.US"
    assert old["chosen_by"] == "reference_match"
    assert [t["symbol"] for t in old["tried"]] == ["GONE.US", "OLD.US", "OLD_OLD.US"]
    assert old["tried"][0]["rows"] == 0  # a symbol EODHD does not know is skipped, not fatal

    assert symbols["GOOD.OQ"]["adjustments_applied"][0]["factor"] == pytest.approx(0.8)
    good = panel[panel["kdcode"] == "GOOD.OQ"]["close"].to_numpy()
    assert np.abs(np.diff(np.log(good))).max() < 0.2  # the split no longer shows

    snapshot = read_input_manifest(manifest)
    paths = [record.path for record in snapshot.manifest.files]
    assert f"constituents/{vendor['pit'].name}" in paths
    assert f"market/{stem}.csv" in paths
    validate_input_package(snapshot, root, expected_paths=paths)
    pit_copy = root / "constituents" / vendor["pit"].name
    assert pit_copy.read_bytes() == vendor["pit"].read_bytes()


def test_export_carries_dividends_on_the_prices_basis_without_moving_the_prices(vendor):
    """Issue 361: the dividend column is the only change; with it the closes give total return.

    GOOD's $1.00 on session 50 sits before a 2-for-1 split and a spin-off restated at
    0.8, so on the panel's basis it is $0.40; its $0.60 on session 350 is after both.
    The same pull with no dividend records writes the same prices and volume.
    """
    dates = vendor["dates"]
    assert export.main(_argv(vendor)) == 0
    root = vendor["tmp"] / "package" / "market"
    panel = pd.read_csv(next(root.glob("*_20160729.csv")), dtype={"dt": str})
    good = panel[panel["kdcode"] == "GOOD.OQ"].set_index("dt")
    paid = good.loc[good["dividend"] > 0, "dividend"]
    assert paid.to_dict() == pytest.approx({dates[50]: 0.4, dates[350]: 0.6})
    assert (panel.loc[panel["kdcode"] == "OLD.N^A16", "dividend"] == 0).all()

    # Close plus dividend over the previous close is the vendor's total return. On an
    # ex-date the vendor reinvests at the previous close less the dividend, so the two
    # differ by the yield times the day's return: under 1e-3 here.
    raw = pd.DataFrame(FakeClient.eod_rows["GOOD.US"]).set_index("date")["adjusted_close"]
    vendor_total = raw / raw.shift(1) - 1
    total = (good["close"] + good["dividend"]) / good["close"].shift(1) - 1
    assert total[dates[51]] == pytest.approx(vendor_total[dates[51]], abs=1e-12)
    for day in (dates[50], dates[350]):
        assert total[day] == pytest.approx(vendor_total[day], abs=1e-3)
        assert total[day] - good["close"].pct_change()[day] > 0.003  # the dividend counts
    meta = json.loads(next(root.glob("*.meta.json")).read_text())
    assert not [f for f in meta["findings"] if f["code"].startswith("dividend")]
    assert (meta["dividend_rows"], meta["names_with_dividends"]) == (2, 1)
    assert "/api/div" in meta["source"]

    FakeClient.div_rows = {}
    other = vendor["tmp"] / "no_dividends"
    assert export.main([*_argv(vendor)[:-4], "--package-root", str(other / "package"),
                        "--cache-dir", str(other / "cache")]) == 0  # fmt: skip
    without = pd.read_csv(next((other / "package" / "market").glob("*_20160729.csv")))
    assert (without["dividend"] == 0).all()
    columns = ["kdcode", "dt", "open", "high", "low", "close", "volume"]
    pd.testing.assert_frame_equal(without[columns], panel[columns], check_exact=True)


def _dividend_check(root: Path, previous: Path, out: Path) -> dict:
    code = dividend_check.main(
        ["--package-root", str(root), "--previous-root", str(previous), "--out-dir", str(out)]
    )
    return {"code": code, **json.loads((out / "summary.json").read_text())}


def test_the_dividend_check_passes_a_revision_that_only_adds_dividends(vendor):
    """Issue 361: r4 against r3 is the export with dividends against the one without."""
    tmp = vendor["tmp"]
    assert export.main(_argv(vendor)) == 0
    dividends = FakeClient.div_rows
    FakeClient.div_rows = {}
    previous = tmp / "previous"
    assert export.main([*_argv(vendor)[:-4], "--package-root", str(previous),
                        "--cache-dir", str(tmp / "cache_previous")]) == 0  # fmt: skip
    FakeClient.div_rows = dividends
    # The previous revision's CSV, as r3 wrote it: no dividend column.
    (old_panel,) = (previous / "market").glob("*_20160729.csv")
    pd.read_csv(old_panel).drop(columns="dividend").to_csv(old_panel, index=False)

    result = _dividend_check(tmp / "package", previous, tmp / "check")

    assert result["code"] == 0
    assert (result["changed_rows"], result["dividend_rows"], result["names_with_dividends"]) == (
        0, 2, 1
    )  # fmt: skip
    by_name = pd.read_csv(tmp / "check" / "dividends_by_name.csv")
    assert by_name["kdcode"].tolist() == ["GOOD.OQ"]


def test_the_dividend_check_fails_any_other_change(vendor):
    tmp = vendor["tmp"]
    assert export.main(_argv(vendor)) == 0
    (panel_path,) = (tmp / "package" / "market").glob("*_20160729.csv")
    (meta_path,) = (tmp / "package" / "market").glob("*.meta.json")
    previous = tmp / "previous" / "market"
    previous.mkdir(parents=True)
    panel = pd.read_csv(panel_path, dtype={"dt": str})
    panel.drop(columns="dividend").to_csv(previous / panel_path.name, index=False)
    assert _dividend_check(tmp / "package", tmp / "previous", tmp / "c0")["code"] == 0

    moved = panel.copy()
    moved.loc[5, "close"] *= 1.0000001
    moved.to_csv(panel_path, index=False)
    result = _dividend_check(tmp / "package", tmp / "previous", tmp / "c1")
    assert (result["code"], result["changed_rows"]) == (1, 1)

    negative = panel.copy()
    negative.loc[5, "dividend"] = -0.1
    negative.to_csv(panel_path, index=False)
    result = _dividend_check(tmp / "package", tmp / "previous", tmp / "c2")
    assert (result["code"], result["invalid_dividends"]) == (1, 1)

    panel.to_csv(panel_path, index=False)
    meta = json.loads(meta_path.read_text())
    meta["findings"].append({"code": "dividend_disagrees_with_adjusted_close", "kdcode": "X.N"})
    meta_path.write_text(json.dumps(meta))
    result = _dividend_check(tmp / "package", tmp / "previous", tmp / "c3")
    assert (result["code"], result["names_with_adjusted_close_disagreements"]) == (1, ["X.N"])


def test_export_takes_splits_after_the_panel_end_out_of_volume_only(vendor):
    """Issue 347: volume comes through as delivered, less splits from the panel end to serving.

    GOOD's vendor volume (1000 a day) already counts the session-100 split, so it is
    unchanged across it. A 3-for-1 dated after the 2016-07-29 panel end, before the
    vendor served the rows, is in that volume too and is divided back out; one dated
    after serving is not in it and is left alone.
    """
    FakeClient.split_rows["GOOD.US"] = [
        *FakeClient.split_rows["GOOD.US"],
        {"date": "2016-09-01", "split": "3.000000/1.000000"},
        {"date": "2017-03-01", "split": "5.000000/1.000000"},
    ]
    FakeClient.instances = []

    assert export.main(_argv(vendor)) == 0

    root = vendor["tmp"] / "package" / "market"
    panel = pd.read_csv(next(root.glob("*_20160729.csv")), dtype={"dt": str})
    good = panel[panel["kdcode"] == "GOOD.OQ"]
    assert good["volume"].to_numpy() == pytest.approx(1000.0 / 3)
    old = panel[panel["kdcode"] == "OLD.N^A16"]
    assert (old["volume"] == 1000.0).all()
    (client,) = FakeClient.instances
    assert ("GOOD.US", "2016-07-29", "2016-10-03") in client.split_requests
    meta = json.loads(next(root.glob("*.meta.json")).read_text())
    (rebased,) = [f for f in meta["findings"] if f["code"] == "volume_rebased_for_later_split"]
    assert rebased["kdcode"] == "GOOD.OQ" and not rebased["blocking"]
    assert rebased["evidence"]["splits"] == {"2016-09-01": 3.0}
    symbols = json.loads(next(root.glob("*_symbols.json")).read_text())
    (segment,) = symbols["GOOD.OQ"]["segments"]
    assert segment["volume_later_splits"] == {"2016-09-01": 3.0}
    assert segment["served"] == "2016-10-03"


def test_a_response_with_no_serving_time_blocks(vendor):
    """Without a serving time the later splits in the vendor's volume cannot be known."""
    FakeClient.served_at = None
    try:
        assert export.main(_argv(vendor)) == 1
    finally:
        FakeClient.served_at = "2016-10-03T12:00:00+00:00"
    meta = json.loads(next((vendor["tmp"] / "package" / "market").glob("*.meta.json")).read_text())
    codes = {(f["kdcode"], f["code"], f["blocking"]) for f in meta["findings"]}
    assert ("GOOD.OQ", "vendor_serve_time_unknown", True) in codes


def test_export_stops_without_a_manifest_when_no_candidate_matches(vendor):
    FakeClient.listings = {True: [], False: []}  # the old company cannot be found
    manifest = vendor["tmp"] / "out" / "toy.r1.json"
    manifest.parent.mkdir()

    assert export.main(_argv(vendor, manifest_output=manifest)) == 1

    assert not manifest.exists()
    meta = json.loads(next((vendor["tmp"] / "package" / "market").glob("*.meta.json")).read_text())
    codes = {(f["kdcode"], f["code"]) for f in meta["findings"] if f["blocking"]}
    assert ("OLD.N^A16", "no_candidate_matched") in codes
    assert ("OLD.N^A16", "reference_disagreement") in codes


def test_a_declared_unavailable_name_is_left_out_and_disclosed(vendor):
    payload = json.loads(vendor["map"].read_text())
    payload["overrides"]["OLD.N^A16"] = {"unavailable": "The vendor has no history for it"}
    vendor["map"].write_text(json.dumps(payload), encoding="utf-8")
    FakeClient.listings = {True: [], False: []}
    manifest = vendor["tmp"] / "out" / "toy.r1.json"
    manifest.parent.mkdir()

    assert export.main(_argv(vendor, manifest_output=manifest)) == 0

    market = vendor["tmp"] / "package" / "market"
    stem = f"{vendor['prefix']}_eodhd_20150101_20160729"
    panel = pd.read_csv(market / f"{stem}.csv")
    assert set(panel["kdcode"]) == {"GOOD.OQ"}
    meta = json.loads((market / f"{stem}.meta.json").read_text())
    assert meta["declared_unavailable"] == {"OLD.N^A16": "The vendor has no history for it"}
    assert meta["missing_identifiers"] == []
    check = pd.read_csv(market / f"{stem}_reference_check.csv").set_index("kdcode")
    assert check.loc["OLD.N^A16", "verdict"] == "declared_unavailable"
    unknowns = read_input_manifest(manifest).manifest.provenance["unknowns"]
    assert any("OLD.N^A16" in item for item in unknowns)


def test_a_name_with_no_rows_still_blocks_unless_declared(vendor):
    FakeClient.eod_rows = {k: v for k, v in FakeClient.eod_rows.items() if k == "GOOD.US"}
    FakeClient.listings = {True: [], False: []}
    assert export.main(_argv(vendor)) == 1
    meta = json.loads(next((vendor["tmp"] / "package" / "market").glob("*.meta.json")).read_text())
    assert meta["missing_identifiers"] == ["OLD.N^A16"]
    codes = {(f["kdcode"], f["code"]) for f in meta["findings"] if f["blocking"]}
    assert ("OLD.N^A16", "no_rows") in codes


def test_a_delisted_name_stops_at_its_last_trade(vendor):
    # EODHD carries the old company one session past its last trade, at zero volume.
    carried = dict(FakeClient.eod_rows["OLD_OLD.US"][-1])
    carried.update(date=vendor["dates"][280], volume=0.0)
    FakeClient.eod_rows = {**FakeClient.eod_rows}
    FakeClient.eod_rows["OLD_OLD.US"] = [*FakeClient.eod_rows["OLD_OLD.US"], carried]
    # A live name ending the same way keeps its row: a quiet session is not a cessation.
    good = [dict(row) for row in FakeClient.eod_rows["GOOD.US"]]
    last = max(i for i, row in enumerate(good) if row["date"] <= "2016-07-29")
    good[last].update(
        {k: good[last - 1][k] for k in ("open", "high", "low", "close", "adjusted_close")},
        volume=0.0,
    )
    FakeClient.eod_rows["GOOD.US"] = good

    assert export.main(_argv(vendor)) == 0

    market = vendor["tmp"] / "package" / "market"
    panel = pd.read_csv(next(market.glob("*_20160729.csv")), dtype={"dt": str})
    old = panel[panel["kdcode"] == "OLD.N^A16"]
    assert old["dt"].max() == vendor["dates"][279]
    symbols = json.loads(next(market.glob("*_symbols.json")).read_text())
    assert symbols["OLD.N^A16"]["carried_tail_dropped"] == [vendor["dates"][280]]
    assert symbols["GOOD.OQ"]["carried_tail_dropped"] == []
    assert panel[panel["kdcode"] == "GOOD.OQ"]["dt"].max() == good[last]["date"]
    meta = json.loads(next(market.glob("*.meta.json")).read_text())
    codes = {(f["kdcode"], f["code"], f["blocking"]) for f in meta["findings"]}
    assert ("OLD.N^A16", "carried_tail_dropped", False) in codes


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"unavailable": ""}, "nonempty reason"),
        ({"unavailable": "gone", "segments": [{"candidates": ["A.US"]}]}, "only a note"),
        ({"unavailable": "gone", "adjustments": []}, "only a note"),
    ],
)
def test_symbol_map_rejects_a_malformed_unavailable_entry(entry, message):
    with pytest.raises(SymbolMapError, match=message):
        parse_symbol_map({"schema": 1, "overrides": {"A.N": entry}})


def test_the_key_is_never_written(vendor, monkeypatch):
    monkeypatch.setenv(export.KEY_ENV, "sekret-key-value")
    manifest = vendor["tmp"] / "out" / "toy.r1.json"
    manifest.parent.mkdir()
    assert export.main(_argv(vendor, manifest_output=manifest)) == 0
    written = [p for p in (vendor["tmp"]).rglob("*") if p.is_file()]
    assert written
    for path in written:
        assert b"sekret-key-value" not in path.read_bytes(), path


# ---------------------------------------------------------------------------
# The real client and the input checks
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, body):
        self.status = 200
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_client_never_stores_or_reports_the_key(tmp_path, monkeypatch):
    key = "sekret-key-value"
    seen_urls = []

    def fake_urlopen(url, timeout):
        seen_urls.append(url)
        return _Response([{"date": "2020-01-02", "close": 1}])

    monkeypatch.setattr(export.urllib.request, "urlopen", fake_urlopen)
    raw_log: list[dict] = []
    client = export.EodhdClient(key, tmp_path / "cache", raw_log)

    assert client.eod("AAPL.US", "2020-01-01", "2020-01-31") == [{"date": "2020-01-02", "close": 1}]
    assert key in seen_urls[0]  # it does reach the vendor
    stored = b"".join(p.read_bytes() for p in (tmp_path / "cache").rglob("*") if p.is_file())
    assert stored and key.encode() not in stored
    assert key not in json.dumps(raw_log)

    # A second read is served from the cache without a call.
    client.eod("AAPL.US", "2020-01-01", "2020-01-31")
    assert len(seen_urls) == 1 and raw_log[-1]["from_cache"] is True

    # Another date range is another request, not the cached one.
    client.eod("AAPL.US", "2019-01-01", "2020-01-31")
    assert len(seen_urls) == 2

    def failing_urlopen(url, timeout):
        raise export.urllib.error.HTTPError(url, 500, f"boom {url}", {}, None)

    monkeypatch.setattr(export.urllib.request, "urlopen", failing_urlopen)
    monkeypatch.setattr(export.time, "sleep", lambda seconds: None)
    with pytest.raises(export.VendorError) as caught:
        client.eod("MSFT.US", "2020-01-01", "2020-01-31")
    assert key not in str(caught.value)

    def odd_urlopen(url, timeout):
        raise ValueError(f"URL can't contain control characters. {url!r}")

    monkeypatch.setattr(export.urllib.request, "urlopen", odd_urlopen)
    with pytest.raises(export.VendorError) as caught:
        client.eod("A B.US", "2020-01-01", "2020-01-31")
    assert key not in str(caught.value) and caught.value.__cause__ is None


def test_client_quotes_symbols_and_retries_dropped_connections(tmp_path, monkeypatch):
    attempts = []

    def flaky_urlopen(url, timeout):
        attempts.append(url)
        if len(attempts) == 1:
            raise ConnectionResetError("reset by peer")
        return _Response([])

    monkeypatch.setattr(export.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(export.time, "sleep", lambda seconds: None)
    client = export.EodhdClient("k", tmp_path / "cache", [])
    assert client.eod("A B.US", "2020-01-01", "2020-01-31") == []
    assert len(attempts) == 2
    assert "/eod/A%20B.US?" in attempts[-1]


def test_client_records_when_each_response_was_served(tmp_path, monkeypatch):
    """Issue 347: the later splits in volume run to when the vendor served the rows, cached or not."""
    monkeypatch.setattr(export.urllib.request, "urlopen", lambda url, timeout: _Response([]))
    client = export.EodhdClient("k", tmp_path / "cache", [])

    client.eod("AAPL.US", "2020-01-01", "2020-01-31")
    fresh = client.last_retrieved_at
    assert fresh and fresh[:4] >= "2026"

    cache_file = next((tmp_path / "cache" / "eod").glob("AAPL.US.*.json"))
    envelope = json.loads(cache_file.read_text())
    envelope["retrieved_at"] = "2026-10-08T05:00:00+00:00"
    cache_file.write_text(json.dumps(envelope))
    client.splits("AAPL.US", "2020-01-01", "2020-01-31")  # another response in between
    client.eod("AAPL.US", "2020-01-01", "2020-01-31")
    assert client.last_retrieved_at == "2026-10-08T05:00:00+00:00"


# ---------------------------------------------------------------------------
# The volume check, on a package built from a cache with no vendor calls
# ---------------------------------------------------------------------------

_CHECK_START, _CHECK_END, _CHECK_SERVED = "2020-01-01", "2020-01-31", "2020-03-02"


def _put(cache: Path, kind: str, symbol: str, params: dict, body) -> None:
    path = f"{kind}/{symbol}"
    digest = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    file = cache / kind / f"{symbol}.{digest[:16]}.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"status": 200, "body": body, "retrieved_at": f"{_CHECK_SERVED}T12:00:00+00:00"}
    file.write_text(json.dumps(envelope), encoding="utf-8")


def _cached_package(tmp_path: Path, *, x_close_shift: float = 0.0) -> dict[str, Path]:
    """X.OQ is read from X.US in two stretches; X.US splits 2-for-1 after the panel end."""
    cache = tmp_path / "cache"
    dates = _sessions(_CHECK_START, 22)
    for symbol, base in (("X.US", 50.0), ("Y.US", 80.0)):
        closes = [base + 0.1 * i + (x_close_shift if symbol == "X.US" else 0) for i in range(22)]
        _put(
            cache,
            "eod",
            symbol,
            {"period": "d", "from": _CHECK_START, "to": _CHECK_END},
            _raw_rows(dates, closes, volume=1000.0),
        )
        _put(cache, "splits", symbol, {"from": _CHECK_START, "to": _CHECK_END}, [])
        _put(cache, "div", symbol, {"from": _CHECK_START, "to": _CHECK_END}, [])
    later = [{"date": "2020-02-10", "split": "2.000000/1.000000"}]
    _put(cache, "splits", "X.US", {"from": _CHECK_END, "to": _CHECK_SERVED}, later)
    _put(cache, "splits", "Y.US", {"from": _CHECK_END, "to": _CHECK_SERVED}, [])
    pit = tmp_path / "toy_pit_universe.csv"
    pit.write_text("kdcode,valid_from,valid_to\nX.OQ,2020-01-01,\nY.OQ,2020-01-01,\n")
    symbol_map = tmp_path / "map.json"
    segments = [
        {"candidates": ["X.US"], "end": "2020-01-15"},
        {"candidates": ["X.US"], "start": "2020-01-16"},
    ]
    symbol_map.write_text(json.dumps({"schema": 1, "overrides": {"X.OQ": {"segments": segments}}}))
    return {"cache": cache, "pit": pit, "map": symbol_map}


def _build(tmp_path: Path, inputs: dict[str, Path], name: str, monkeypatch) -> Path:
    def no_network(url, timeout):
        raise AssertionError(f"unexpected vendor call: {url}")

    monkeypatch.setattr(export.urllib.request, "urlopen", no_network)
    monkeypatch.setenv(export.KEY_ENV, "unused")
    root = tmp_path / name
    argv = [
        "--pit-universe", str(inputs["pit"]),
        "--symbol-map", str(inputs["map"]),
        "--start", _CHECK_START,
        "--end", _CHECK_END,
        "--pit-export-cutoff", _CHECK_END,
        "--package-root", str(root),
        "--cache-dir", str(inputs["cache"]),
    ]  # fmt: skip
    assert export.main(argv) == 0
    return root


def _check(tmp_path: Path, root: Path, previous: Path, cache: Path, *allowed: str) -> dict:
    out = tmp_path / f"check_{root.name}_{previous.name}"
    argv = [
        "--package-root", str(root),
        "--previous-root", str(previous),
        "--cache-dir", str(cache),
        "--start", _CHECK_START,
        "--end", _CHECK_END,
        "--out-dir", str(out),
    ]  # fmt: skip
    for item in allowed:
        argv += ["--allow-price-change", item]
    code = volume_check.main(argv)
    return {"code": code, **json.loads((out / "summary.json").read_text())}


def test_the_volume_check_passes_a_package_built_from_its_cache(tmp_path, monkeypatch):
    """Two stretches of one symbol share its later split once each, never twice (#347 review)."""
    inputs = _cached_package(tmp_path)
    root = _build(tmp_path, inputs, "new", monkeypatch)
    panel = pd.read_csv(next((root / "market").glob("*_20200131.csv")), dtype={"dt": str})
    assert (panel.loc[panel["kdcode"] == "X.OQ", "volume"] == 500.0).all()
    assert (panel.loc[panel["kdcode"] == "Y.OQ", "volume"] == 1000.0).all()

    summary = _check(tmp_path, root, root, inputs["cache"])

    assert summary["code"] == 0
    assert summary["volume_mismatches"] == 0 and summary["rows_without_vendor_volume"] == 0
    assert summary["training_rows_above_vendor_volume"] == 0
    assert summary["price_difference_rows"] == 0


def test_the_volume_check_fails_doubled_volume_and_undeclared_price_changes(tmp_path, monkeypatch):
    inputs = _cached_package(tmp_path)
    root = _build(tmp_path, inputs, "new", monkeypatch)
    panel_path = next((root / "market").glob("*_20200131.csv"))
    previous = tmp_path / "previous"
    shutil.copytree(root, previous)
    doubled = pd.read_csv(panel_path, dtype={"dt": str})
    doubled["volume"] *= 2  # what a second split adjustment does
    doubled.to_csv(next((previous / "market").glob("*_20200131.csv")), index=False)

    as_built = _check(tmp_path, root, previous, inputs["cache"])
    assert as_built["code"] == 0  # volume may differ from the previous revision; prices may not

    shutil.copytree(root, tmp_path / "doubled")
    doubled.to_csv(next((tmp_path / "doubled" / "market").glob("*_20200131.csv")), index=False)
    bad = _check(tmp_path, tmp_path / "doubled", root, inputs["cache"])
    assert bad["code"] == 1
    assert bad["volume_mismatches"] == len(doubled)
    assert bad["training_rows_above_vendor_volume"] > 0

    halved = doubled.assign(volume=doubled["volume"] / 4)  # under the vendor's, still wrong
    shutil.copytree(root, tmp_path / "halved")
    halved.to_csv(next((tmp_path / "halved" / "market").glob("*_20200131.csv")), index=False)
    low = _check(tmp_path, tmp_path / "halved", root, inputs["cache"])
    assert low["code"] == 1 and low["volume_mismatches"] == len(halved)
    assert low["training_rows_above_vendor_volume"] == 0

    shifted = _build(
        tmp_path, _cached_package(tmp_path / "s", x_close_shift=1.0), "shifted", monkeypatch
    )
    moved = _check(tmp_path, shifted, root, tmp_path / "s" / "cache")
    assert moved["code"] == 1 and moved["unexpected_price_difference_rows"] == 22
    declared = _check(tmp_path, shifted, root, tmp_path / "s" / "cache", "X.OQ:2020-02-01")
    assert declared["code"] == 0 and declared["price_difference_rows"] == 22


def test_inputs_must_match_the_reference_manifest(tmp_path):
    root = tmp_path / "lseg"
    (root / "constituents").mkdir(parents=True)
    pit = root / "constituents" / "toy_pit_universe.csv"
    pit.write_text("kdcode,valid_from,valid_to\nA.N,2020-01-02,2020-12-31\n", encoding="utf-8")
    manifest = tmp_path / "toy.r1.json"
    snapshot = write_input_manifest(
        manifest,
        root,
        package_id="toy",
        package_revision="r1",
        files=[InputFileSpec("constituents/toy_pit_universe.csv", "membership")],
        provenance={
            "source": "test",
            "acquisition_mode": "test",
            "acquired_at": "2026-10-04",
            "producing_command": "test",
            "producing_arguments": [],
            "unknowns": [],
        },
    )

    assert export.verify_inputs(manifest, snapshot.sha256, [pit]) == snapshot.sha256
    pit.write_text("kdcode,valid_from,valid_to\nB.N,2020-01-02,2020-12-31\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from its record"):
        export.verify_inputs(manifest, snapshot.sha256, [pit])


# ---------------------------------------------------------------------------
# The Colab notebook
# ---------------------------------------------------------------------------

NOTEBOOK = REPO_ROOT / "notebooks" / "eodhd_price_pull_colab.ipynb"


def _notebook_source() -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells)


def test_notebook_is_the_generator_output(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "NOTEBOOK_PATH", tmp_path / "nb.ipynb")
    generator.main()
    assert (tmp_path / "nb.ipynb").read_bytes() == NOTEBOOK.read_bytes()


def test_notebook_pins_the_reference_package_the_lseg_config_pins():
    config = yaml.safe_load((REPO_ROOT / "configs/data/gics_top10_110_2016.yaml").read_text())
    source = _notebook_source()
    prefix = Path(config["pit_universe_csv"]).name.removesuffix("_pit_universe.csv")
    manifest = config["input_package_manifest"].replace(prefix, "{PREFIX}")
    sha = config["input_package_manifest_sha256"]
    assert f'PREFIX = "{prefix}"' in source
    assert f'REFERENCE_MANIFEST = f"{manifest}"' in source
    assert f'REFERENCE_MANIFEST_SHA256 = "{sha}"' in source
    assert Path(config["filename"]).name == f"{prefix}_lseg_20150101_20260731.csv"
    assert 'REFERENCE_PANEL = f"market/{PREFIX}_lseg_20150101_20260731.csv"' in source


def test_notebook_reads_the_key_from_secrets_and_never_prints_it():
    source = _notebook_source()
    assert 'userdata.get("EODHD_API_KEY")' in source
    assert "print(os.environ" not in source
    assert "EODHD_API_KEY=" not in source


# ---------------------------------------------------------------------------
# The published package and its data config
# ---------------------------------------------------------------------------

EODHD_CONFIG = REPO_ROOT / "configs" / "data" / "gics_top10_110_2016_eodhd.yaml"
LSEG_CONFIG = REPO_ROOT / "configs" / "data" / "gics_top10_110_2016.yaml"


def test_the_eodhd_config_changes_only_the_price_panel_and_its_package():
    eodhd = OmegaConf.to_container(OmegaConf.load(EODHD_CONFIG))
    lseg = OmegaConf.to_container(OmegaConf.load(LSEG_CONFIG))
    package_keys = {"filename", "input_package_manifest", "input_package_manifest_sha256"}
    assert set(eodhd) == set(lseg) | {"pit_absent_kdcodes"}
    assert {k for k in lseg if eodhd[k] != lseg[k]} == package_keys
    assert eodhd["pit_absent_kdcodes"] == ["DD.N^I17"]
    assert "_eodhd_" in eodhd["filename"]


def test_the_eodhd_package_carries_the_lseg_membership_byte_for_byte():
    eodhd = OmegaConf.load(EODHD_CONFIG)
    lseg = OmegaConf.load(LSEG_CONFIG)
    mine = read_input_manifest(
        REPO_ROOT / eodhd.input_package_manifest,
        expected_sha256=eodhd.input_package_manifest_sha256,
    ).manifest
    theirs = read_input_manifest(
        REPO_ROOT / lseg.input_package_manifest,
        expected_sha256=lseg.input_package_manifest_sha256,
    ).manifest
    pit = Path(eodhd.pit_universe_csv).name
    by_path = {record.path: record for record in mine.files}
    assert by_path[f"constituents/{pit}"].sha256 == (
        {record.path: record for record in theirs.files}[f"constituents/{pit}"].sha256
    )
    # The pull ran with the committed symbol map, and declared the one gap.
    map_record = next(r for r in mine.files if r.path.endswith("_symbol_map.json"))
    assert map_record.sha256 == hashlib.sha256(COMMITTED_MAP.read_bytes()).hexdigest()
    unknowns = [u.split(":")[0] for u in mine.provenance["unknowns"]]
    assert unknowns == ["DD.N^I17"]
    # Admission accepts exactly the gap the package records, no more.
    assert list(eodhd.pit_absent_kdcodes) == unknowns


def test_the_eodhd_config_binds_both_selected_files_to_its_package():
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["data=gics_top10_110_2016_eodhd"])
    config = create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))

    declarations = declare_window_inputs(config, InputObservationContext().freeze())

    assert config.data.pit_absent_kdcodes == ["DD.N^I17"]
    (package,) = declarations.manifests
    assert package.manifest.package_id.endswith("_eodhd")
    assert package.sha256 == config.data.input_package_manifest_sha256
    files = {record.path for record in package.manifest.files}
    bindings = {binding.role: binding for binding in declarations.required}
    for role in ("data.filename", "data.pit_universe_csv"):
        assert bindings[role].manifest_sha256 == package.sha256
        assert bindings[role].path in files
