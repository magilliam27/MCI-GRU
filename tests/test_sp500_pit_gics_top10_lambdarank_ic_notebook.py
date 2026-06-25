import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py")


def _cell_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def _code_cell_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("top10_lambdarank_generator", GENERATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_top10_lambdarank_screen_notebook_matches_generator_output() -> None:
    checked_in_notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    generator = _load_generator_module()

    generated_notebook = generator.build_notebook(generator.cells)

    assert checked_in_notebook == generated_notebook


def test_top10_lambdarank_screen_pins_year_budget_and_complete_pair_contract() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Top-10 PIT LambdaRankIC Screen",
        "docs/superpowers/specs/2026-06-25-top10-lambdarank-screen-design.md",
        "YEARS = [2022]",
        "BASE_SEEDS = [314159]",
        "NUM_MODELS = 1",
        "NUM_EPOCHS = 40",
        "EARLY_STOPPING_PATIENCE = 8",
        "PAIR_CAP = 8192",
        "COMPLETE_PAIR_COUNT_110 = 5995",
        "assert PAIR_CAP >= COMPLETE_PAIR_COUNT_110",
        "training.loss_type=lambdarank_ic",
        "training.selection_metric=val_rank_ic",
        "training.lambdarank_ic_max_pairs_per_day=8192",
        "training.lambdarank_ic_temperature=1.0",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_top10_lambdarank_screen_avoids_unsupported_backtest_flags() -> None:
    sources = {
        "notebook": "\n".join(_cell_sources()),
        "generator": GENERATOR_PATH.read_text(encoding="utf-8"),
    }
    forbidden_tokens = [
        "--spread_bps",
        "--slippage_bps",
        "--output_dir",
    ]

    for source_name, source in sources.items():
        for token in forbidden_tokens:
            assert token not in source, f"{token} unexpectedly found in {source_name}"


def test_top10_lambdarank_screen_uses_reduced_2016_start_bundle_and_masked_panel() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_",
        "lseg_20150101_20260622.csv",
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv",
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_snapshots.csv",
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_meta.json",
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.meta.json",
        "data.use_pit_universe=true",
        "data.pit_universe_mode=masked_panel",
        "data.pit_min_scoreable_stocks=100",
        "data.pit_breadth_policy=error",
        "selector_start == '2016-01-04'",
        '"snapshot_dates": 127',
        '"snapshot_min_selected": 110',
        '"snapshot_max_selected": 110',
        '"pit_union_kdcodes": 205',
        '"missing_identifiers": []',
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_top10_lambdarank_screen_preserves_recipe_and_colab_runtime_contract() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1",
        "graph.update_frequency_months=0",
        "graph.top_k=0",
        "graph.top_k_metric=corr",
        "graph.use_multi_feature_edges=true",
        "graph.drop_edge_p=0.1",
        "features=with_momentum",
        "features.include_global_regime=true",
        "features.regime_strict=true",
        "features.regime_include_subsequent_returns=false",
        "G4/L4-class Colab runtime",
        "not T4/CPU",
        'BLOCKED_GPU_NAMES = ("T4",)',
        "ALLOWED_GPU_MARKERS = (",
        "FRED_API_KEY is required",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_top10_lambdarank_screen_writes_drive_truth_artifacts_and_backtests() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "sp500_gics_top10_lambdarank_ic_screen",
        'HEARTBEAT_PATH = DRIVE_RUN_ROOT / "heartbeat.json"',
        "data_audit.json",
        "lambdarank_ic_sp500_pit_gics_top10_screen_manifest.json",
        "training_results.csv",
        "training_results.json",
        "backtest_results.csv",
        "backtest_results.json",
        "run_summary.json",
        "logs",
        "summaries",
        "artifacts",
        "tests/backtest_sp500_daily.py",
        "--pit_universe_csv",
        "--top_k",
        "rank-drop",
        'write_heartbeat("FAILED", "failed"',
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_top10_lambdarank_screen_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
