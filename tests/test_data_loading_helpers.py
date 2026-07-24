"""Unit tests for pure data-loading helpers with no prior coverage.

Covers ``mci_gru.data.reshape`` (LSEG MultiIndex -> flat OHLCV),
``mci_gru.data.universes`` (universe metadata), ``mci_gru.data.path_resolver``
(project-aware data path fallbacks), and ``mci_gru.graph.sector_edges``
(static sector relation used by the dual-GAT path). All are synthetic and
require no LSEG/FRED access.
"""

import pandas as pd
import pytest
import torch

from mci_gru.data.path_resolver import resolve_project_data_path
from mci_gru.data.reshape import STANDARD_COLS, reshape_lseg_to_standard
from mci_gru.data.universes import (
    UNIVERSES,
    get_chain_ric,
    get_universe_info,
    is_multi_country,
    list_universes,
)
from mci_gru.graph.sector_edges import build_sector_edges, load_sector_map_csv


def _vendor_multiindex_frame() -> pd.DataFrame:
    """Two instruments, two days, official LSEG daily-summary field names."""
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    fields = ["MKT_OPEN", "MKT_HIGH", "MKT_LOW", "TRDPRC_1", "ACVOL_UNS"]
    columns = pd.MultiIndex.from_product([["AAPL.O", "MSFT.O"], fields])
    rows = [
        # open, high, low, close, volume per instrument
        [100.0, 102.0, 99.0, 101.0, 1000.0, 200.0, 204.0, 198.0, 202.0, 2000.0],
        [101.0, 103.0, 100.0, 102.0, 1100.0, 202.0, 206.0, 200.0, 204.0, 2100.0],
    ]
    return pd.DataFrame(rows, index=dates, columns=columns)


# Note: class/test names avoid the substring "lseg" because tests/conftest.py
# auto-skips any nodeid containing it; these tests are pure and need no API.
class TestReshapeToStandard:
    def test_reshapes_multiindex_to_standard_columns(self):
        out = reshape_lseg_to_standard(_vendor_multiindex_frame())

        assert list(out.columns) == STANDARD_COLS
        assert sorted(out["kdcode"].unique()) == ["AAPL.O", "MSFT.O"]
        assert sorted(out["dt"].unique()) == ["2025-01-02", "2025-01-03"]

        aapl_day1 = out[(out["kdcode"] == "AAPL.O") & (out["dt"] == "2025-01-02")].iloc[0]
        assert aapl_day1["open"] == 100.0
        assert aapl_day1["close"] == 101.0
        assert aapl_day1["turnover"] == 101.0 * 1000.0  # volume * close

    def test_drops_rows_with_missing_close(self):
        frame = _vendor_multiindex_frame()
        frame.loc[frame.index[0], ("AAPL.O", "TRDPRC_1")] = float("nan")

        out = reshape_lseg_to_standard(frame)

        aapl = out[out["kdcode"] == "AAPL.O"]
        assert list(aapl["dt"]) == ["2025-01-03"]

    def test_rejects_flat_columns(self):
        flat = pd.DataFrame({"close": [1.0]})
        with pytest.raises(ValueError, match="MultiIndex"):
            reshape_lseg_to_standard(flat)

    def test_rejects_missing_required_fields(self):
        dates = pd.to_datetime(["2025-01-02"])
        columns = pd.MultiIndex.from_product([["AAPL.O"], ["TRDPRC_1"]])
        frame = pd.DataFrame([[101.0]], index=dates, columns=columns)
        with pytest.raises(KeyError, match="Missing required columns"):
            reshape_lseg_to_standard(frame)


class TestUniverses:
    def test_sp500_metadata(self):
        info = get_universe_info("sp500")
        assert info["chain_ric"] == "0#.SPX"
        assert get_chain_ric("sp500") == "0#.SPX"

    def test_unknown_universe_raises_with_available_list(self):
        with pytest.raises(ValueError, match="Unknown universe"):
            get_universe_info("ftse100")

    def test_list_universes_matches_registry(self):
        assert list_universes() == list(UNIVERSES.keys())

    def test_multi_country_flag(self):
        assert is_multi_country("msci_world") is True
        assert is_multi_country("sp500") is False


class TestPathResolver:
    def test_exact_path_wins(self, tmp_path):
        target = tmp_path / "prices.csv"
        target.write_text("kdcode,dt\n", encoding="utf-8")
        assert resolve_project_data_path(str(target)) == target.resolve()

    def test_missing_file_raises_filenotfound(self):
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            resolve_project_data_path("definitely_not_a_real_file_9a8b7c.csv")


class TestSectorEdges:
    def test_load_sector_map_csv_parses_headers_case_insensitively(self, tmp_path):
        csv_path = tmp_path / "sectors.csv"
        csv_path.write_text(
            "Kdcode,Sector\nAAA,Tech\nBBB,Tech\nCCC,Energy\n",
            encoding="utf-8",
        )
        assert load_sector_map_csv(str(csv_path)) == {
            "AAA": "Tech",
            "BBB": "Tech",
            "CCC": "Energy",
        }

    def test_load_sector_map_csv_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sector_map_csv(str(tmp_path / "nope.csv"))

    def test_load_sector_map_csv_requires_expected_headers(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("symbol,industry\nAAA,Tech\n", encoding="utf-8")
        with pytest.raises(ValueError, match="kdcode, sector"):
            load_sector_map_csv(str(csv_path))

    def test_build_sector_edges_links_same_sector_only(self):
        kdcodes = ["AAA", "BBB", "CCC"]
        sectors = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"}

        edge_index, edge_weight = build_sector_edges(kdcodes, sectors, top_k=5)

        pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
        # AAA (0) <-> BBB (1) share Tech; CCC (2) is alone in Energy.
        assert pairs == {(0, 1), (1, 0)}
        assert torch.all(edge_weight == 1.0)

    def test_build_sector_edges_respects_top_k(self):
        kdcodes = ["A", "B", "C", "D"]
        sectors = dict.fromkeys(kdcodes, "Tech")

        edge_index, _ = build_sector_edges(kdcodes, sectors, top_k=1)

        # Each node links to at most 1 same-sector peer.
        out_degree = pd.Series(edge_index[0].tolist()).value_counts()
        assert (out_degree <= 1).all()
        assert edge_index.shape[1] == 4

    def test_build_sector_edges_empty_universe(self):
        edge_index, edge_weight = build_sector_edges([], {}, top_k=3)
        assert edge_index.shape == (2, 0)
        assert edge_weight.shape == (0,)
