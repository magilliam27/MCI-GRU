import ast
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.requires_lseg, pytest.mark.requires_fred]

NOTEBOOK_PATH = Path("notebooks/pit_repeated_seed_replication_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_pit_repeated_seed_replication_nb.py")


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


def test_issue31_notebook_pins_repeated_seed_full_pit_replication() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Repeated-Seed Full PIT Masked-Panel Replication",
        "Issue #31",
        "REFERENCE_RUN_TAG = '20260514_043539'",
        "REPLICATION_BASE_SEEDS = [314159, 271828, 161803]",
        "SMOKE_MODE = False",
        "RUN_TRAINING = True",
        "RUN_BACKTESTS = True",
        "RUN_REFERENCE_COST_BACKTESTS = True",
        "ALLOW_MISSING_REFERENCE = False",
        "docs/DEFAULT_EXPERIMENT_RECIPE.md",
        "static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1",
        "NUM_MODELS = 1 if SMOKE_MODE else 20",
        "NUM_EPOCHS = 1 if SMOKE_MODE else 100",
        "EARLY_STOPPING_PATIENCE = 2 if SMOKE_MODE else 15",
        "seed={job['base_seed']}",
        "features=with_momentum",
        "features.include_global_regime=true",
        "features.regime_include_subsequent_returns=false",
        "graph.update_frequency_months=0",
        "graph.top_k=0",
        "graph.use_multi_feature_edges=true",
        "graph.drop_edge_p={MODEL_RECIPE['drop_edge_p']}",
        "training.loss_type={MODEL_RECIPE['loss_type']}",
        "training.label_type={MODEL_RECIPE['label_type']}",
        "training.selection_metric={MODEL_RECIPE['selection_metric']}",
        "training.shuffle_train=true",
        "model.label_t={MODEL_RECIPE['label_t']}",
        "model.temporal_encoder=gru_attn",
        "data.use_pit_universe=true",
        "data.pit_universe_mode=masked_panel",
        "PIT_MIN_SCOREABLE_STOCKS = 450",
        "PIT_BREADTH_POLICY = 'error'",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue31_notebook_writes_required_decision_artifacts() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "pit_repeated_seed_replication_manifest.json",
        "training_results.csv",
        "pit_breadth_summary.csv",
        "prediction_count_checks.csv",
        "backtest_results.csv",
        "pit_repeated_seed_pooled_daily_returns.csv",
        "pit_repeated_seed_pooled_daily_significance.csv",
        "pit_repeated_seed_seed_summary.csv",
        "pit_repeated_seed_yearly_seed_summary.csv",
        "pit_repeated_seed_issue_closeout_summary.csv",
        "pit_repeated_seed_backtest_sensitivity_results.csv",
        "pit_repeated_seed_backtest_sensitivity_year_crosstab.csv",
        "pit_repeated_seed_backtest_sensitivity_metric_deltas.csv",
        "pit_repeated_seed_reference_comparison.csv",
        "pit_repeated_seed_2022_monthly_diagnostics.csv",
        "pit_repeated_seed_replication_summary.md",
        "assert_expected_training_complete(training_df)",
        "assert_expected_backtests_complete(backtest_df)",
        "raise RuntimeError('Reference run root not found",
        "raise RuntimeError('Reference cost/rank-gate backtest did not produce",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue31_notebook_can_resume_and_retry_transient_regime_fetch_failures() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "RUN_TAG_OVERRIDE = ''",
        "RUN_TAG = RUN_TAG_OVERRIDE.strip() or datetime.now().strftime('%Y%m%d_%H%M%S')",
        "TRAINING_RETRY_ON_REGIME_FETCH_FAILURE = 2",
        "TRAINING_RETRY_SLEEP_SECONDS = 60",
        "MCI_GRU_FRED_MAX_ATTEMPTS",
        "MCI_GRU_FRED_RETRY_SECONDS",
        "def is_regime_fetch_failure(stdout_text: str, stderr_text: str) -> bool:",
        "Unable to load required regime input series",
        "Retrying after transient regime input fetch failure",
        "USE_STATIC_REGIME_INPUTS = True",
        "STATIC_REGIME_INPUTS_CSV = ''",
        "def build_static_regime_inputs() -> tuple[Path | None, str, dict]:",
        "features.regime_inputs_csv={STATIC_REGIME_INPUTS_RELATIVE_PATH}",
        "static_regime_inputs_sha256",
        "static_regime_inputs_summary",
        "static_regime_inputs_marker.json",
        "def run_matches_static_regime_inputs(path: Path) -> bool:",
        "def write_static_regime_marker(run_dir: Path | None) -> None:",
        "def latest_run_dir(experiment_name: str, require_static_regime: bool = True) -> Path | None:",
        "latest_run_dir(job['name'], require_static_regime=False)",
        "Recovered static regime inputs from Drive",
        "Cached static regime inputs failed validation; redrawing from FRED-backed loader:",
        "def draw_static_regime_inputs(dest: Path) -> None:",
        "DRIVE_RUN_ROOT / 'inputs' / dest.name",
        "Static regime inputs:",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue31_notebook_uses_cost_rank_gate_promotion_path() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "BACKTEST_SUFFIX = '_pit_daily_tc_rank_gate'",
        "SPREAD_BPS = 10.0",
        "SLIPPAGE_BPS = 5.0",
        "MIN_RANK_DROP = 30",
        "--transaction_costs",
        "--enable_rank_drop_gate",
        "--min_rank_drop",
        "--num_tests",
        "--adjustment_method",
        "ADJUSTMENT_METHOD = 'bhy'",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue31_notebook_pastes_known_drive_locations_and_branch() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Known Drive Locations",
        "https://drive.google.com/drive/folders/1mlM6KQISlXl3Bnrk20ebEv7Mo8CLg8LO",
        "https://drive.google.com/file/d/1hbJ7lg45tgyDKv6ecyuh8NxuTr6GpCpm/view?usp=drivesdk",
        "https://drive.google.com/file/d/1d5NQC2y9JeKZF-90zh1ko04VzHUsb8PM/view?usp=drivesdk",
        "https://drive.google.com/file/d/1Zn4njEOgtfuchMesHlhNZYdz_emgbBz-/view?usp=drivesdk",
        "https://drive.google.com/drive/folders/1p1F2NqY5C6ISBzjm7-JBkbvsE4K2E2LF",
        "https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/pit-universe-validation/notebooks/pit_repeated_seed_replication_colab.ipynb",
        "BRANCH = 'codex/pit-universe-validation'",
        "runtime-generated PIT union presets",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue31_notebook_hardwires_colab_inputs_and_runtime_key() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Hardwired to avoid slow recursive Drive discovery in Colab.",
        "MARKET_CSV_PATH = '/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.csv'",
        "PIT_UNIVERSE_CSV_PATH = '/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv'",
        "MARKET_META_JSON_PATH = '/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.meta.json'",
        "Expected fixed input path does not exist:",
        "Upload {filename} to {DRIVE_DATA_DIR} or update the hardwired path at the top of this cell.",
        "Hardwired for this full-run Colab notebook.",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator

    assert "root.glob(f'**/{filename}')" not in combined
    assert "root.glob(f'**/{filename}')" not in generator
    assert "MY_FRED_KEY = ''" not in combined
    assert "MY_FRED_KEY = ''" not in generator


def test_issue31_notebook_includes_pooled_significance_for_issue29() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "newey_west_std",
        "moving_block_bootstrap_ci",
        "annualized_excess_return",
        "annualized_excess_volatility",
        "information_ratio",
        "adjust_p_values_bhy",
        "multiple_testing_adjusted_p_value",
        "Newey-West",
        "moving-block bootstrap",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue31_notebook_summarizes_three_seed_closeout_evidence() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "MIN_CLOSEOUT_REPLICATION_SEEDS = 3",
        "EXPECTED_REPLICATION_JOB_COUNT",
        "EXPECTED_TOTAL_MODELS",
        "build_seed_summary",
        "build_yearly_seed_summary",
        "build_issue_closeout_summary",
        "issue31_pit_pipeline_status",
        "issue29_pooled_significance_status",
        "issue30_2022_stress_status",
        "replication_all_seeds",
        "supports_closeout",
        "needs_more_evidence",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue31_notebook_runs_backtest_sensitivity_replay() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "RUN_BACKTEST_SENSITIVITY = True",
        "SENSITIVITY_INCLUDE_LABEL21_DIAGNOSTIC = True",
        "run_pit_repeated_seed_backtest_sensitivity.py",
        "--training-label-t",
        "spread5_only_label5",
        "spread5_only_label21_diagnostic",
        "label_t=21 is diagnostic",
        "Backtest Sensitivity: Scenario x Year",
        "Backtest sensitivity metric deltas",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_issue31_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
