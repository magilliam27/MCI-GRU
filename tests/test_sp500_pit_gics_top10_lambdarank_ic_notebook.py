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


def test_top10_lambdarank_full_default_pins_mode_budget_and_complete_pair_contract() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Top-10 PIT LambdaRankIC Full Run",
        "docs/superpowers/specs/2026-06-25-top10-lambdarank-screen-design.md",
        "SCREEN_MODE = False",
        "SKIP_COMPLETED_JOBS = True",
        "RESUME_PREVIOUS_DRIVE_RUNS = True",
        'RESUME_RUN_TAG = ""',
        "BUDGET_MODE = \"screen\" if SCREEN_MODE else \"full\"",
        "SCREEN_YEARS = [2022]",
        "FULL_YEARS = [2022, 2023, 2024]",
        "SCREEN_BASE_SEEDS = [314159]",
        "FULL_BASE_SEEDS = [314159]",
        "SCREEN_NUM_MODELS = 1",
        "FULL_NUM_MODELS = 20",
        "SCREEN_NUM_EPOCHS = 40",
        "FULL_NUM_EPOCHS = 100",
        "SCREEN_EARLY_STOPPING_PATIENCE = 8",
        "FULL_EARLY_STOPPING_PATIENCE = 15",
        "YEARS = SCREEN_YEARS if SCREEN_MODE else FULL_YEARS",
        "BASE_SEEDS = SCREEN_BASE_SEEDS if SCREEN_MODE else FULL_BASE_SEEDS",
        "NUM_MODELS = SCREEN_NUM_MODELS if SCREEN_MODE else FULL_NUM_MODELS",
        "NUM_EPOCHS = SCREEN_NUM_EPOCHS if SCREEN_MODE else FULL_NUM_EPOCHS",
        "EARLY_STOPPING_PATIENCE = (",
        "EXPECTED_JOB_COUNT = len(YEARS) * len(BASE_SEEDS)",
        "EXPECTED_TOTAL_MODELS = EXPECTED_JOB_COUNT * NUM_MODELS",
        "assert EXPECTED_JOB_COUNT == (1 if SCREEN_MODE else 3)",
        "assert EXPECTED_TOTAL_MODELS == (1 if SCREEN_MODE else 60)",
        "PAIR_CAP = 8192",
        "COMPLETE_PAIR_COUNT_110 = 5995",
        "assert PAIR_CAP >= COMPLETE_PAIR_COUNT_110",
        '"budget_mode": BUDGET_MODE',
        '"expected_job_count": EXPECTED_JOB_COUNT',
        '"expected_total_models": EXPECTED_TOTAL_MODELS',
        '"skip_completed_jobs": SKIP_COMPLETED_JOBS',
        '"resume_previous_drive_runs": RESUME_PREVIOUS_DRIVE_RUNS',
        '"resume_run_tag": RESUME_RUN_TAG',
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
        "REQUIRE_APPLES_TO_APPLES_SELECTOR_HISTORY = True",
        "selector_history_blockers",
        "selector snapshots begin after required train_start",
        "not apples-to-apples",
        "REFERENCE_2025 = {",
        '"reference_2025": REFERENCE_2025',
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
        "AUTO_DISCONNECT_RUNTIME = True",
        "if AUTO_DISCONNECT_RUNTIME and IN_COLAB and runtime is not None:",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_top10_lambdarank_screen_releases_runtime_after_final_summary() -> None:
    code_cells = _code_cell_sources()
    train_backtest_cell = code_cells[-2]
    final_summary_cell = code_cells[-1]
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    assert "runtime.unassign()" not in train_backtest_cell
    assert final_summary_cell.index('print("Backtest results:"') < final_summary_cell.index(
        "runtime.unassign()"
    )
    assert 'print("Backtest results:"' in generator
    assert "runtime.unassign()" in generator


def test_top10_lambdarank_screen_writes_drive_truth_artifacts_and_backtests() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "sp500_gics_top10_lambdarank_ic_screen",
        "sp500_gics_top10_lambdarank_ic_full",
        "DRIVE_EXPERIMENT_ROOT = (",
        "DRIVE_RUN_ROOT = DRIVE_EXPERIMENT_ROOT / RUN_TAG",
        'HEARTBEAT_PATH = DRIVE_RUN_ROOT / "heartbeat.json"',
        "data_audit.json",
        'SCREEN_MANIFEST_FILENAME = "lambdarank_ic_sp500_pit_gics_top10_screen_manifest.json"',
        'FULL_MANIFEST_FILENAME = "lambdarank_ic_sp500_pit_gics_top10_full_manifest.json"',
        "MANIFEST_FILENAME = SCREEN_MANIFEST_FILENAME if SCREEN_MODE else FULL_MANIFEST_FILENAME",
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


def test_top10_lambdarank_full_launcher_can_resume_completed_jobs() -> None:
    combined = "\n".join(_cell_sources())
    train_backtest_cell = _code_cell_sources()[-2]
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "def find_run_dir(year_root: Path, experiment_name: str) -> Path | None:",
        "def drive_job_root_for(year: int, base_seed: int) -> Path:",
        "def prior_drive_run_roots() -> list[Path]:",
        "def restore_drive_job_if_present(year_root: Path, drive_job_root: Path) -> None:",
        "def sync_job_to_drive(year_root: Path, drive_job_root: Path) -> None:",
        "def append_training_row(",
        "def append_backtest_row(",
        "skipped: bool",
        'ARTIFACT_DIR / "local_run_root" / str(year) / f"seed{base_seed}"',
        "resume_root = DRIVE_EXPERIMENT_ROOT / RESUME_RUN_TAG",
        'for prior_run_root in sorted(DRIVE_EXPERIMENT_ROOT.glob("20??????_??????"), reverse=True):',
        'relative_job_root = drive_job_root.relative_to(ARTIFACT_DIR / "local_run_root")',
        'source_job_roots.append(prior_run_root / "artifacts" / "local_run_root" / relative_job_root)',
        "drive_job_root = drive_job_root_for(year, base_seed)",
        "restore_drive_job_if_present(year_root, drive_job_root)",
        "sync_job_to_drive(year_root, drive_job_root)",
        "backtest_metrics_path = run_dir / \"backtest_top10_rankdrop\" / \"backtest_metrics.json\"",
        "if SKIP_COMPLETED_JOBS and existing_run_dir is not None:",
        "if existing_pred_dir.exists():",
        "append_training_row(",
        "existing_run_dir",
        "existing_pred_dir",
        "if existing_backtest_metrics.exists():",
        "append_backtest_row(",
        "continue",
        'print("Skipping completed job:", job_name)',
        'print("Skipping training; running missing backtest:", job_name)',
        '"skipped",',
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator

    restore_index = train_backtest_cell.index("restore_drive_job_if_present(year_root, drive_job_root)")
    find_index = train_backtest_cell.index("existing_run_dir = find_run_dir(year_root, job_name)")
    training_row_index = train_backtest_cell.index(
        "append_training_row(year, base_seed, run_dir, pred_dir, skipped=False)"
    )
    first_sync_index = train_backtest_cell.index("sync_job_to_drive(year_root, drive_job_root)")
    backtest_row_index = train_backtest_cell.index(
        "append_backtest_row(year, base_seed, run_dir, pred_dir, skipped=False)"
    )
    last_sync_index = train_backtest_cell.rindex("sync_job_to_drive(year_root, drive_job_root)")

    assert restore_index < find_index
    assert training_row_index < first_sync_index
    assert backtest_row_index < last_sync_index


def test_top10_lambdarank_screen_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
