import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/volatility_targeting_repeated_seed_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_volatility_targeting_repeated_seed_nb.py")


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


def test_issue8_repeated_seed_notebook_pins_candidate_grid() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Issue #8 Volatility-Targeting Repeated-Seed Validation",
        'BRANCH = "codex/colab-gpu-utilization-hardening-20260620"',
        "REPLICATION_BASE_SEEDS = [314159, 271828, 161803]",
        "YEARS = [2022, 2023, 2024, 2025]",
        '"baseline_vol"',
        '"vt_full_clip_0p25_4p0"',
        '"vt_clip_0p50_2p0"',
        '"vt_scale_only"',
        '"vt_ewm_only"',
        "EXPECTED_JOB_COUNT = len(REPLICATION_BASE_SEEDS) * len(YEARS) * len(CANDIDATE_VARIANTS)",
        "EXPECTED_TOTAL_MODELS = EXPECTED_JOB_COUNT * NUM_MODELS",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue8_repeated_seed_notebook_uses_current_pit_recipe_and_candidate_overrides() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "G4/L4-class Colab runtime",
        "not T4/CPU",
        "FRED_API_KEY loaded from Colab Secrets.",
        "features=with_momentum",
        "features.include_global_regime=true",
        "features.regime_include_subsequent_returns=false",
        "graph.update_frequency_months=0",
        "graph.top_k=0",
        "graph.use_multi_feature_edges=true",
        "graph.drop_edge_p=0.1",
        "training.loss_type=ic",
        "training.label_type=returns",
        "training.selection_metric=val_ic",
        "training.shuffle_train=true",
        "model.label_t=5",
        "data.pit_universe_mode=masked_panel",
        "data.pit_min_scoreable_stocks=450",
        "features.volatility_targeting_include_ewm_vol",
        "features.volatility_targeting_include_scale",
        "features.volatility_targeting_include_dynamics",
        "features.volatility_targeting_include_scaled_return",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue8_repeated_seed_notebook_writes_matched_delta_artifacts() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "issue8_vol_targeting_repeated_seed_manifest.json",
        "issue8_vol_targeting_repeated_seed_results.csv",
        "issue8_vol_targeting_repeated_seed_deltas_vs_baseline.csv",
        "issue8_vol_targeting_repeated_seed_seed_summary.csv",
        "issue8_vol_targeting_repeated_seed_year_variant_summary.csv",
        "issue8_vol_targeting_repeated_seed_summary.md",
        'for (base_seed, year), group in results_df.groupby(["base_seed", "year"]):',
        'baseline = group[group["variant"] == "baseline_vol"]',
        "same-seed/year baseline",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue8_repeated_seed_notebook_runs_cost_rank_gate_backtests() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        'BACKTEST_SUFFIX = "_pit_daily_tc_rank_gate"',
        "SPREAD_BPS = 10.0",
        "SLIPPAGE_BPS = 5.0",
        "MIN_RANK_DROP = 30",
        "--transaction_costs",
        "--enable_rank_drop_gate",
        "--min_rank_drop",
        'ADJUSTMENT_METHOD = "bhy"',
        "backtest_env = os.environ.copy()",
        "env=backtest_env",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue8_repeated_seed_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
