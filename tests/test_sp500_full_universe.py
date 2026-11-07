"""The full point-in-time S&P 500 universe (#386): membership file and checks on the EODHD panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

import scripts.data.build_sp500_full_universe as build
from mci_gru.data.eodhd_prices import load_symbol_map
from mci_gru.data.input_manifest import InputFileSpec, read_input_manifest, write_input_manifest
from mci_gru.data.pit import normalise_pit_intervals

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "data" / "sp500_full_eodhd.yaml"
RECIPE_DATA = REPO_ROOT / "configs" / "data" / "gics_top10_110_2016_eodhd.yaml"
SYMBOL_MAP = REPO_ROOT / "data" / "mappings" / "eodhd_symbols_sp500_full.json"
MAP_110 = REPO_ROOT / "data" / "mappings" / "eodhd_symbols_gics_top10_110_2016.json"
REVIEW = REPO_ROOT / "data" / "mappings" / "eodhd_review_sp500_full.json"
INTERVALS = "lseg_sp500_membership_intervals.csv"


def _spans(*pairs):
    return pd.DataFrame(pairs, columns=["valid_from", "valid_to"])


@pytest.fixture
def lseg_package(tmp_path):
    root = tmp_path / "lseg"
    (root / "constituents").mkdir(parents=True)
    intervals = pd.DataFrame(
        {
            "constituent_ric": ["AAA.N", "BRKb.N", "OLD.OQ^C17", "AAA.N"],
            "valid_from": ["2016-01-04", "2016-01-04", "2016-01-04", "2018-01-02"],
            "valid_to": ["2016-06-30", "2026-07-31", "2017-03-15", "2026-07-31"],
        }
    )
    intervals.to_csv(root / "constituents" / INTERVALS, index=False)
    manifest = tmp_path / "lseg.r1.json"
    write_input_manifest(
        manifest,
        root,
        package_id="lseg",
        package_revision="r1",
        files=[InputFileSpec(f"constituents/{INTERVALS}", "membership intervals")],
        provenance={
            "source": "test",
            "acquisition_mode": "test",
            "acquired_at": "2026-11-07T00:00:00+00:00",
            "producing_command": "test",
            "producing_arguments": [],
            "unknowns": [],
        },
    )
    return root / "constituents" / INTERVALS, manifest


def test_membership_keeps_every_interval_under_the_kdcode_header(lseg_package, tmp_path):
    intervals, manifest = lseg_package
    out = tmp_path / "out"
    argv = ["membership", "--intervals", str(intervals), "--reference-manifest", str(manifest)]
    assert build.main([*argv, "--output-dir", str(out), "--prefix", "full"]) == 0
    pit = pd.read_csv(out / "constituents" / "full_pit_universe.csv", dtype=str)
    assert list(pit.columns) == ["kdcode", "valid_from", "valid_to"]
    assert len(pit) == 4  # a name with two spells keeps both
    original = pd.read_csv(intervals, dtype=str)
    assert pit["kdcode"].tolist() == original["constituent_ric"].tolist()
    assert len(normalise_pit_intervals(pit)) == 4
    summary = json.loads((out / "full_membership_check.json").read_text())
    assert summary["identifiers"] == 3 and summary["delisted_identifiers"] == 1


def test_membership_refuses_intervals_that_differ_from_the_manifest(lseg_package, tmp_path):
    intervals, manifest = lseg_package
    original = intervals.read_text()
    argv = ["membership", "--intervals", str(intervals), "--reference-manifest", str(manifest)]
    argv += ["--output-dir", str(tmp_path / "o"), "--prefix", "full"]
    for changed in (original + "NEW.N,2020-01-02,2020-01-03\n", original.replace("AAA", "AAB")):
        intervals.write_text(changed)  # a longer file, then one of the same size
        with pytest.raises(SystemExit, match="differs from its record"):
            build.main(argv)


def test_membership_agreement_counts_member_days_by_ticker():
    pit = pd.DataFrame(
        {
            "kdcode": ["AAA.N", "BRKb.N", "OLD.OQ^C17"],
            "valid_from": ["2020-01-02", "2020-01-02", "2020-01-02"],
            "valid_to": ["2020-01-03", "2020-01-03", "2020-01-02"],
        }
    )
    payload = {
        "HistoricalComponents": {
            "2019-12-31": {"0": {"Code": "AAA"}, "1": {"Code": "BRK-B"}, "2": {"Code": "OLD_old"}},
            "2020-01-03": {"0": {"Code": "AAA"}, "1": {"Code": "NEW"}},
        }
    }
    dates, members = build.eodhd_snapshots(payload)
    result = build.membership_agreement(pit, dates, members, ["2020-01-02", "2020-01-03"])
    # 2020-01-02: all three agree (the _old suffix is a label); 2020-01-03: BRK-B is missing.
    assert result["lseg_member_days"] == 5
    assert result["member_days_in_both"] == 4
    assert result["lseg_only_by_ticker"] == {"BRK-B": 1}
    assert result["eodhd_only_by_ticker"] == {"NEW": 1}


def _month_ends():
    return ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"]


def test_a_missed_split_is_a_price_step_and_a_merger_is_not():
    closes = pd.DataFrame({"dt": _month_ends(), "close": [10.0, 11.0, 5.5, 6.0]})
    # March: the close halves while the market cap holds (a split EODHD missed).
    # April: the market cap jumps with a steady close (shares issued in a merger).
    caps = pd.DataFrame({"as_of_date": _month_ends(), "company_market_cap": [100, 110, 110, 240]})
    spans = _spans(("2020-01-01", "2020-12-31"))
    pairs = build.mcap_pairs(closes, caps, spans, _month_ends())
    assert len(pairs) == 3
    assert pairs[0]["share_step"] == pytest.approx(0.0, abs=1e-4)
    assert pairs[1]["share_step"] == pytest.approx(0.6931, abs=1e-4)
    assert pairs[2]["share_step"] == pytest.approx(0.7, abs=0.01)
    assert [p["to"] for p in build.price_steps(pairs)] == ["2020-03-31"]


def test_mcap_pairs_skip_month_ends_outside_membership_and_across_a_gap():
    ends = _month_ends()[:3]
    closes = pd.DataFrame({"dt": ends, "close": [1.0, 2.0, 4.0]})
    caps = pd.DataFrame({"as_of_date": ends, "company_market_cap": [1, 9, 1]})
    spans = _spans(("2020-01-01", "2020-01-31"), ("2020-03-01", "2020-03-31"))
    assert build.mcap_pairs(closes, caps, spans, ends) == []  # January and March are not adjacent


def test_mcap_pairs_do_not_match_a_close_too_long_before_the_month_end():
    closes = pd.DataFrame({"dt": ["2020-01-31", "2020-02-10"], "close": [1.0, 1.0]})
    caps = pd.DataFrame({"as_of_date": ["2020-01-31", "2020-02-28"], "company_market_cap": [1, 5]})
    spans = _spans(("2020-01-01", "2020-12-31"))
    assert build.mcap_pairs(closes, caps, spans, ["2020-01-31", "2020-02-28"]) == []


def test_extreme_moves_count_only_inside_membership():
    closes = pd.DataFrame(
        {
            "dt": ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
            "close": [10.0, 20.0, 10.0, 10.1],
        }
    )
    moves = build.extreme_moves(closes, _spans(("2020-01-06", "2020-12-31")))
    assert [m["dt"] for m in moves] == ["2020-01-06"]
    assert moves[0]["move"] == pytest.approx(-0.5)


def test_an_extreme_move_is_corroborated_only_by_a_month_whose_market_cap_followed():
    move = {"dt": "2020-03-16", "move": -0.5}
    followed = {"from": "2020-02-28", "to": "2020-03-31", "share_step": 0.01}
    diverged = {"from": "2020-02-28", "to": "2020-03-31", "share_step": 0.6}
    other_month = {"from": "2020-03-31", "to": "2020-04-30", "share_step": 0.0}
    assert build.corroborating_pair(move, [other_month, followed]) == followed
    assert build.corroborating_pair(move, [diverged, other_month]) is None


def test_delisting_month_gap_reads_the_code():
    assert build.delisting_month_gap("ATVI.OQ^J23", "2023-10-12") == 0
    assert build.delisting_month_gap("CB.N^A16", "2026-07-31") == -126
    assert build.delisting_month_gap("AAPL.OQ", "2026-07-31") is None


def test_missing_sessions_are_calendar_days_inside_membership_without_a_row():
    calendar = pd.Series(["2020-01-02", "2020-01-03", "2020-01-06"]).to_numpy()
    dates = pd.Series(["2020-01-02"])
    spans = _spans(("2020-01-01", "2020-01-03"))
    assert build.missing_sessions(dates, calendar, spans, delisted=False) == (["2020-01-03"], [])


def test_sessions_after_a_delisted_names_last_trade_are_not_flagged():
    calendar = pd.Series(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]).to_numpy()
    dates = pd.Series(["2020-01-02", "2020-01-06"])
    spans = _spans(("2020-01-01", "2020-01-07"))
    flagged, after = build.missing_sessions(dates, calendar, spans, delisted=True)
    assert flagged == ["2020-01-03"] and after == ["2020-01-07"]


def test_review_entries_need_a_reason(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"entries": {"AAA.N": {"price_steps": {"2020-03-31": " "}}}}))
    with pytest.raises(SystemExit, match="needs a reason"):
        build.load_review(path, {"AAA.N"})
    path.write_text(json.dumps({"entries": {"ZZZ.N": {"identity": "x"}}}))
    with pytest.raises(SystemExit, match="outside the universe"):
        build.load_review(path, {"AAA.N"})


def test_check_exits_1_until_every_flag_is_reviewed(tmp_path):
    days = pd.bdate_range("2020-01-01", "2020-04-30").strftime("%Y-%m-%d")
    rows = []
    for code in [f"N{i:03d}.N" for i in range(build.CALENDAR_MIN_NAMES)]:
        rows += [{"kdcode": code, "dt": d, "close": 10.0} for d in days]
    rows += [
        {"kdcode": "AAA.N", "dt": d, "close": 10.0 if d < "2020-02-14" else 30.0} for d in days
    ]
    panel = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(panel, index=False)
    pit = tmp_path / "x_pit_universe.csv"
    codes = sorted({r["kdcode"] for r in rows})
    pd.DataFrame({"kdcode": codes, "valid_from": "2020-01-01", "valid_to": "2020-04-30"}).to_csv(
        pit, index=False
    )
    month_ends = _month_ends()
    meta = tmp_path / "meta.csv"
    pd.DataFrame(
        [
            {"as_of_date": m, "kdcode": c, "company_market_cap": 100.0}
            for m in month_ends
            for c in codes
        ]
    ).to_csv(meta, index=False)
    argv = ["check", "--panel", str(panel), "--pit-universe", str(pit), "--metadata", str(meta)]
    out = tmp_path / "report.json"
    assert build.main([*argv, "--output", str(out)]) == 1
    report = json.loads(out.read_text())
    # The close triples while the market cap stays flat: a price step, and an
    # extreme move the market cap does not corroborate.
    assert {u["check"] for u in report["unresolved"]} == {"price_step", "extreme_move"}
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "entries": {
                    "AAA.N": {
                        "price_steps": {"2020-02-28": "test"},
                        "extreme_moves": {"2020-02-14": "test"},
                    }
                }
            }
        )
    )
    assert build.main([*argv, "--review", str(review), "--output", str(out)]) == 0


# ---------------------------------------------------------------------------
# The committed package
# ---------------------------------------------------------------------------


def test_the_config_binds_its_files_to_the_committed_package():
    config = OmegaConf.load(CONFIG)
    manifest = read_input_manifest(
        REPO_ROOT / config.input_package_manifest,
        expected_sha256=config.input_package_manifest_sha256,
    ).manifest
    paths = {record.path: record for record in manifest.files}
    root = f"{config.input_package_root}/"
    assert config.filename.removeprefix(root) in paths
    assert config.pit_universe_csv.removeprefix(root) in paths
    map_record = next(r for r in manifest.files if r.path.endswith("_symbol_map.json"))
    assert map_record.sha256 == hashlib.sha256(SYMBOL_MAP.read_bytes()).hexdigest()
    # Declared-absent members are exactly the manifest's unknowns.
    unknowns = {u.split(":", 1)[0] for u in manifest.provenance.unknowns}
    assert set(config.pit_absent_kdcodes) == unknowns


#: 110-name decisions the full map replaces, each because the vendor's layout changed.
REPLACED_110_DECISIONS = {
    # EODHD now files Paramount's whole history under PSKY.US; PARA.US is another series.
    "PSKY.OQ",
}


def test_the_symbol_map_keeps_every_110_name_decision():
    """The 110-name map was proven against the LSEG panel; the full map keeps it except where noted."""
    full = json.loads(SYMBOL_MAP.read_text())["overrides"]
    for kdcode, entry in json.loads(MAP_110.read_text())["overrides"].items():
        if kdcode in REPLACED_110_DECISIONS:
            assert full[kdcode] != entry and "110-name map" in full[kdcode]["note"], kdcode
            continue
        assert full[kdcode] == entry, kdcode
    load_symbol_map(SYMBOL_MAP)


def test_the_review_file_is_accepted_by_the_checker():
    entries = json.loads(REVIEW.read_text())["entries"]
    assert entries
    build.load_review(REVIEW, set(entries))  # every entry carries a reason


def test_the_config_keeps_the_recipe_window():
    """Only the universe changes: train 2020-2023, validate 2024, test 2025."""
    config = OmegaConf.load(CONFIG)
    recipe = OmegaConf.load(RECIPE_DATA)
    assert config.train_start == "2020-01-02"
    for key in ("train_end", "val_start", "val_end", "test_start", "test_end"):
        assert config[key] == recipe[key], key
    assert config.use_pit_universe and config.pit_universe_mode == "masked_panel"
    assert config.pit_export_cutoff == recipe.pit_export_cutoff
