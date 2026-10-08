"""EODHD daily prices for the point-in-time universe (#281).

The pull itself needs the vendor; these tests drive the same code with canned
vendor rows: the symbol rule, the split adjustment, the map's validation, the
proof of each mapping against a reference panel, and the whole export through
to a manifest the run can verify.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import scripts.data.export_eodhd_pit_prices as export
from mci_gru.config import create_config_from_dict
from mci_gru.data.eodhd_prices import (
    PriceAdjustment,
    SymbolMapError,
    agreement_passes,
    apply_adjustments,
    build_symbol_plans,
    clean_values,
    default_symbol,
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
    assert charter.factor == pytest.approx(1 / 0.9)
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


def test_split_adjust_divides_prices_and_multiplies_volume_before_the_split():
    dates = _sessions("2020-08-26", 6)
    raw = [400.0, 404.0, 408.0, 102.5, 103.0, 104.0]  # 4-for-1 on the fourth session
    eod = eod_rows_frame(
        _raw_rows(dates, raw, adjusted=[c / 4 if i < 3 else c for i, c in enumerate(raw)])
    )
    splits = splits_frame([{"date": dates[3], "split": "4.000000/1.000000"}], as_of=dates[-1])

    adjusted = split_adjust(eod, splits)

    assert adjusted["close"].tolist() == pytest.approx([100.0, 101.0, 102.0, 102.5, 103.0, 104.0])
    assert adjusted["volume"].tolist() == [4000.0] * 3 + [1000.0] * 3
    # The return across the split is the economic one, not -75%.
    assert adjusted["close"].pct_change().iloc[3] == pytest.approx(102.5 / 102.0 - 1)
    assert split_basis_findings("AAPL.OQ", "AAPL.US", adjusted) == []


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
    listings: dict[bool, list[dict]] = {}

    def __init__(self, api_key, cache_dir, raw_log):
        self.calls = 0
        self._raw = raw_log

    def eod(self, symbol, start, end):
        self.calls += 1
        self._raw.append({"endpoint": f"eod/{symbol}", "body": self.eod_rows.get(symbol, [])})
        return self.eod_rows.get(symbol, [])

    def splits(self, symbol, start, end):
        self.calls += 1
        return self.split_rows.get(symbol, [])

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
    # and a spin-off on session 300 that only adjusted_close carries (step 0.8).
    raw_good = np.where(np.arange(400) < 100, good * 2, good)
    raw_good = np.where(np.arange(400) < 300, raw_good / 0.8, raw_good)
    FakeClient.eod_rows = {
        "GOOD.US": _raw_rows(dates, list(raw_good), adjusted=list(good)),
        # OLD.US is now someone else; the old company sits under OLD_old in the delisted list.
        "OLD.US": _raw_rows(dates, list(impostor)),
        "OLD_OLD.US": _raw_rows(dates[:280], list(old[:280])),
    }
    FakeClient.split_rows = {"GOOD.US": [{"date": dates[100], "split": "2.000000/1.000000"}]}
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
    assert list(panel.columns) == ["kdcode", "dt", "open", "high", "low", "close", "volume"]
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
