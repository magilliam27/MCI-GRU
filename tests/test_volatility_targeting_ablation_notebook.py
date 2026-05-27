import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/volatility_targeting_ablation_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_volatility_targeting_ablation_nb.py")


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


def test_issue8_ablation_notebook_defines_stage1_component_sweep() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Issue #8 Volatility-Targeting Ablation Sweep",
        'RUN_STAGE = "stage1_2023"',
        '"stage2_contrasts": [2024, 2025]',
        '"baseline_vol"',
        '"vt_full_clip_0p25_4p0"',
        '"vt_no_scaled_return"',
        '"vt_ewm_only"',
        '"vt_scale_only"',
        '"vt_no_dynamics"',
        '"vt_clip_0p50_2p0"',
        '"vt_clip_0p75_1p5"',
        "features.volatility_targeting_include_ewm_vol",
        "features.volatility_targeting_include_scale",
        "features.volatility_targeting_include_dynamics",
        "features.volatility_targeting_include_scaled_return",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue8_ablation_notebook_uses_current_pit_recipe_and_g4_preflight() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "G4/L4-class Colab runtime, not T4/CPU",
        "BRANCH = \"codex/issue8-vol-ablation-sweep\"",
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
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue8_ablation_notebook_runs_cost_rank_gate_backtests_and_writes_deltas() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "BACKTEST_SUFFIX = \"_pit_daily_tc_rank_gate\"",
        "SPREAD_BPS = 10.0",
        "SLIPPAGE_BPS = 5.0",
        "MIN_RANK_DROP = 30",
        "--transaction_costs",
        "--enable_rank_drop_gate",
        "--min_rank_drop",
        "issue8_vol_targeting_ablation_results.csv",
        "issue8_vol_targeting_ablation_deltas_vs_baseline.csv",
        "total_return_vs_baseline",
        "issue8_vol_targeting_ablation_summary.md",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue8_ablation_notebook_sets_repo_pythonpath_for_backtest_script() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "backtest_env = os.environ.copy()",
        'existing_pythonpath = backtest_env.get("PYTHONPATH")',
        'backtest_env["PYTHONPATH"] = str(REPO_DIR)',
        'backtest_env["PYTHONPATH"] = f"{REPO_DIR}{os.pathsep}{existing_pythonpath}"',
        "env=backtest_env",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue8_ablation_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
