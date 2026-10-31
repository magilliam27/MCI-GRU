"""The small-caps universe draw (#379): frame, draw size, membership file and symbol map."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

import scripts.data.build_r2k_sample_universe as build
from mci_gru.data.eodhd_prices import load_symbol_map, parse_symbol_map
from mci_gru.data.input_manifest import read_input_manifest
from mci_gru.data.pit import normalise_pit_intervals

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "data" / "r2k_sample_400_eodhd.yaml"
RECIPE_DATA = REPO_ROOT / "configs" / "data" / "gics_top10_110_2016_eodhd.yaml"
SYMBOL_MAP = REPO_ROOT / "data" / "mappings" / "eodhd_symbols_r2k_sample_400.json"
CIK_A, CIK_B, CIK_C, CIK_D = "0000000001", "0000000002", "0000000003", "0000000004"


def _holding(snap, cik, symbol, how="price", value=100.0, name="Co"):
    return {"snap": snap, "cik": cik, "symbol": symbol, "how": how, "value": value, "name": name}


@pytest.fixture
def inputs(tmp_path):
    holdings = pd.DataFrame(
        [
            _holding("2019-12-31", CIK_A, "AAA.US"),
            _holding("2020-03-31", CIK_A, "AAA.US"),
            _holding("2020-06-30", CIK_A, "NEWA.US"),  # renamed between the two dates
            _holding("2019-12-31", CIK_B, "BBB.US"),
            _holding("2019-12-31", CIK_B, "BBB-B.US", value=5.0),  # a smaller second class
            _holding("2020-03-31", CIK_C, "CCC.US", how="closest_40pct"),  # never confirmed
            _holding("2018-12-31", CIK_D, "DDD.US"),  # left before the window
        ]
    )
    membership = pd.DataFrame(
        {
            "cik": [CIK_A, CIK_B, CIK_C, CIK_D],
            "spells": [
                "2019-12-31:",
                "2019-12-31:2020-03-30;2021-06-30:",
                "2020-03-31:",
                "2018-12-31:2019-03-30",
            ],
        }
    )
    holdings_path = tmp_path / "holdings_matched.csv.gz"
    holdings.to_csv(holdings_path, index=False, compression="gzip")
    membership_path = tmp_path / "r2k_membership.csv"
    membership.to_csv(membership_path, index=False)
    eod = tmp_path / "eod"
    eod.mkdir()
    (eod / "AAA.US.csv.gz").write_bytes(
        gzip.compress(b"Date,Close\n2020-05-14,9.0\n2020-05-15,9.1\n")
    )
    return {"holdings": holdings, "membership": membership, "tmp": tmp_path, "eod": eod}


def test_frame_keeps_window_members_with_a_price_confirmed_holding(inputs):
    frame, counts = build.frame_companies(
        inputs["membership"], inputs["holdings"], "2020-01-02", "2025-12-31"
    )
    assert frame == [CIK_A, CIK_B]  # C was never price-confirmed; D left in 2019
    assert counts == {
        "with_spell_in_window": 3,
        "left_out_no_price_confirmed_holding": 1,
        "frame": 2,
    }


def test_draw_size_scales_the_frame_to_the_target_members():
    assert build.draw_size(3600, 1950.0, 400) == 738
    assert build.draw_size(100, 50.0, 400) == 100  # never more than the frame


def test_draw_is_seeded_and_ignores_input_order():
    ciks = [f"{i:010d}" for i in range(50)]
    first = build.draw(ciks, 379, 10)
    assert first == build.draw(list(reversed(ciks)), 379, 10)
    assert first != build.draw(ciks, 380, 10)
    assert len(set(first)) == 10


def test_membership_file_clips_to_the_window_end_and_loads(inputs):
    pit = build.pit_rows(inputs["membership"], [CIK_A, CIK_B], "2021-12-31")
    assert pit.to_dict("records") == [
        {"kdcode": "CIK0000000001", "valid_from": "2019-12-31", "valid_to": "2021-12-31"},
        {"kdcode": "CIK0000000002", "valid_from": "2019-12-31", "valid_to": "2020-03-30"},
        {"kdcode": "CIK0000000002", "valid_from": "2021-06-30", "valid_to": "2021-12-31"},
    ]
    assert len(normalise_pit_intervals(pit)) == 3
    closed = pd.DataFrame({"cik": [CIK_A], "spells": ["2019-12-31:2022-06-29"]})
    assert build.pit_rows(closed, [CIK_A], "2021-12-31")["valid_to"].tolist() == ["2021-12-31"]
    later = build.pit_rows(inputs["membership"], [CIK_B], "2021-01-31")
    assert list(later["valid_from"]) == ["2019-12-31"]  # a spell after the end is dropped


def test_ticker_change_ends_the_old_symbol_on_its_last_close(inputs):
    mapping = build.symbol_map(
        inputs["holdings"], [CIK_A, CIK_B], build.price_cache_reader(inputs["eod"]), "2025-12-31"
    )
    a = mapping["overrides"]["CIK0000000001"]["segments"]
    assert a == [
        {"candidates": ["AAA.US"], "end": "2020-05-15"},
        {"candidates": ["NEWA.US"], "start": "2020-05-16"},
    ]
    # The larger class by value is the company's symbol on a shared report date.
    assert mapping["overrides"]["CIK0000000002"]["segments"] == [{"candidates": ["BBB.US"]}]
    plans = parse_symbol_map(mapping)  # the export's own reader accepts it
    assert plans["CIK0000000001"].segments[1].start == "2020-05-16"


def test_ticker_change_without_prices_ends_the_day_before_the_new_report(inputs):
    mapping = build.symbol_map(
        inputs["holdings"], [CIK_A], build.price_cache_reader(None), "2025-12-31"
    )
    segments = mapping["overrides"]["CIK0000000001"]["segments"]
    assert segments[0]["end"] == "2020-06-29" and segments[1]["start"] == "2020-06-30"


def test_main_writes_the_three_outputs(inputs):
    out = inputs["tmp"] / "out"
    argv = [
        "--holdings", str(inputs["tmp"] / "holdings_matched.csv.gz"),
        "--membership", str(inputs["tmp"] / "r2k_membership.csv"),
        "--price-cache", str(inputs["eod"]),
        "--output-dir", str(out),
        "--prefix", "toy",
        "--target-members", "1",
    ]  # fmt: skip
    assert build.main(argv) == 0
    report = json.loads((out / "toy_sample.json").read_text())
    assert report["counts"]["frame"] == 2
    # Only A is held on the window's report dates (B's rows are from 2019-12-31).
    assert report["median_frame_members_per_report_date"] == 1.0
    assert report["drawn"] == 2
    pit = pd.read_csv(out / "constituents" / "toy_pit_universe.csv", dtype=str)
    assert set(pit["kdcode"]) == set(report["draw_order"])
    mapping = json.loads((out / "eodhd_symbols_toy.json").read_text())
    assert set(mapping["overrides"]) == set(report["draw_order"])


def test_symbols_after_the_window_end_are_ignored(inputs):
    mapping = build.symbol_map(
        inputs["holdings"], [CIK_A], build.price_cache_reader(None), "2020-03-31"
    )
    assert mapping["overrides"]["CIK0000000001"]["segments"] == [{"candidates": ["AAA.US"]}]


def test_review_moves_the_start_and_declares_adjustments_with_a_reason(inputs):
    review = {
        "CIK0000000002": {
            "start": "2018-01-01",
            "adjustments": [{"date": "2020-02-03", "reason": "special dividend"}],
            "reason": "vendor rows before 2018 are wrong",
        }
    }
    mapping = build.symbol_map(
        inputs["holdings"], [CIK_B], build.price_cache_reader(None), "2025-12-31", review
    )
    entry = mapping["overrides"]["CIK0000000002"]
    assert entry["segments"] == [{"candidates": ["BBB.US"], "start": "2018-01-01"}]
    assert entry["adjustments"] == [{"date": "2020-02-03", "reason": "special dividend"}]
    assert entry["note"].endswith("Reviewed: vendor rows before 2018 are wrong")
    assert parse_symbol_map(mapping)["CIK0000000002"].adjustments[0].date == "2020-02-03"
    with pytest.raises(SystemExit, match="needs a reason"):
        build.symbol_map(
            inputs["holdings"], [CIK_B], build.price_cache_reader(None), "2025-12-31",
            {"CIK0000000002": {"start": "2018-01-01"}},
        )  # fmt: skip
    with pytest.raises(SystemExit, match="not drawn"):
        build.symbol_map(
            inputs["holdings"], [CIK_B], build.price_cache_reader(None), "2025-12-31",
            {"CIK0000000009": {"reason": "x"}},
        )  # fmt: skip


def test_review_drops_a_single_wrong_session_by_splitting_the_segment():
    segments = build.drop_sessions([{"candidates": ["X.US"]}], ["2025-09-29"])
    assert segments == [
        {"candidates": ["X.US"], "end": "2025-09-28"},
        {"candidates": ["X.US"], "start": "2025-09-30"},
    ]
    later = build.drop_sessions(
        [{"candidates": ["A.US"], "end": "2020-05-15"}, {"candidates": ["B.US"], "start": "2020-05-16"}],
        ["2021-01-04"],
    )  # fmt: skip
    assert later[1:] == [
        {"candidates": ["B.US"], "start": "2020-05-16", "end": "2021-01-03"},
        {"candidates": ["B.US"], "start": "2021-01-05"},
    ]
    assert (
        len(parse_symbol_map({"schema": 1, "overrides": {"K": {"segments": later}}})["K"].segments)
        == 3
    )


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
    # The pull ran with the committed symbol map, which the export's reader accepts.
    map_record = next(r for r in manifest.files if r.path.endswith("_symbol_map.json"))
    assert map_record.sha256 == hashlib.sha256(SYMBOL_MAP.read_bytes()).hexdigest()
    assert len(load_symbol_map(SYMBOL_MAP)) == 688


def test_the_config_keeps_the_recipe_window():
    """Only the universe changes: train 2020-2023 (the recipe's start), validate 2024, test 2025."""
    config = OmegaConf.load(CONFIG)
    recipe = OmegaConf.load(RECIPE_DATA)
    assert config.train_start == "2020-01-02"
    for key in ("train_end", "val_start", "val_end", "test_start", "test_end"):
        assert config[key] == recipe[key], key
    assert config.use_pit_universe and config.pit_universe_mode == "masked_panel"
