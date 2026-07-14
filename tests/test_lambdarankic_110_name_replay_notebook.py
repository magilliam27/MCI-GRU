import ast
import json
import subprocess
import sys
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/lambdarankic_110_name_replay_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_lambdarankic_110_name_replay_nb.py")


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


def test_replay_notebook_pins_diagnostics_only_contract() -> None:
    combined = "\n".join(_cell_sources())
    code_cells = "\n".join(_code_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "LambdaRankIC 110-Name Replay Diagnostics",
        "DRY_RUN = True",
        "RUN_TRAINING = False",
        "REQUIRE_COMPLETE_MATRIX = True",
        "DISCONNECT_RUNTIME_WHEN_DONE = True",
        "BASE_SEEDS = [161803, 271828, 314159]",
        "CURRENT_PAIR_CAP = 8192",
        'LEGACY_PAIR_CAP = "legacy_unknown"',
        'BLOCKED_GPU_NAMES = ("T4", "L4")',
        "REQUIRE_GPU = False",
        "G4-class replay runtime",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator

    assert "baseline_cost_gate30_2024" not in combined
    assert "baseline_cost_gate30_2024" not in generator

    assert "run_experiment.py" not in code_cells
    assert "run_pit_saved_prediction_backtests.py" not in code_cells


def test_replay_notebook_preserves_pit_and_base_seed_contract() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "data.use_pit_universe=true",
        "data.pit_universe_mode=masked_panel",
        "data.pit_breadth_policy=error",
        "training.label_type=returns",
        "model.label_t=5",
        "features.regime_include_subsequent_returns=false",
        "training.loss_type=lambdarank_ic",
        "training.selection_metric=val_rank_ic",
        "base seeds (`161803`, `271828`, `314159`)",
        "Ensemble member seeds remain internal",
        "PIT_WINDOWS = {",
        '2024: {"test_start": "2024-01-08", "test_end": "2024-12-31", "expected_csv_count": 248}',
        'MARKET_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv"',
        'PIT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv"',
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_replay_notebook_contains_known_drive_prediction_rows() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "1fYmtPg97O52SgTRsU_XgwuVaWFbpj9W_",
        "1mO5dqZ6QMIRMDQHmrbd30so2ui7HeT5V",
        "1Co5Vd2dOSMrHUN5x_OzbJpkjJFocSHMo",
        "1lhL-tnUoShh8ImNdTED_sRBOf_dqcOim",
        "1MW8uCWjlarfJYnsOUzG2vBmZUaXjaDin",
        "1w_lFPx_JKginWf6-TsQoFY-Mlhs2XHuc",
        "1jCcZu-ENQKfbit2cdRjBucmCVklULERO",
        "1i2eiUSi0CGpatkzZ64hqAz04v71ob6FV",
        "1Gvtz8C3U6da1YtjA_bJ6SFyrcBNgfMeI",
        "1km8pF1mFCREktte26bnboKw8_aqSzLzL",
        "1ctmw-XztXVP8r_FGu81bE_V7k2FawVLO",
        "1Yg1yzcU9xKZ8FnjSK0KDUc6AWr3XlUON",
        "1IJ62jNdpLbFW4Kuc9l68LkG3NmTmS9bd",
        "1tp-BEvU2yPMxnE_c6ul3o1gVi-ZyeF3n",
        "1KHK3TSjtjz4Ft-XTU5DkmVtgcklgKTwt",
        "1PIV6uuwKDBKAGYMIRsvepCD7cgo9CgO3",
        "recovery_20260701_verified",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_replay_notebook_calls_daily_backtest_with_cost_and_gate_scenarios() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        'str(REPO_DIR / "scripts" / "backtest_sp500_daily.py")',
        '"--predictions_dir"',
        '"--data_file"',
        '"--pit_universe_csv"',
        '"--test_start"',
        '"--test_end"',
        '"--top_k"',
        '"--label_t"',
        '"--num_tests"',
        '"--adjustment_method"',
        'ADJUSTMENT_METHOD = "bhy"',
        '"--auto_save"',
        '"--backtest_suffix"',
        '"--transaction_costs"',
        '"--spread"',
        '"--slippage"',
        '"--enable_rank_drop_gate"',
        '"--min_rank_drop"',
        "baseline_cost_gate30",
        "gross_no_cost_no_gate",
        "cost_no_gate",
        "gate30_no_cost",
        "def row_ready_for_replay",
        'return row.get("stage_status") == "OK"',
        "GATE_SWEEP_VALUES = [10, 20, 30, 40, 60]",
        "COST_SWEEP_PAIRS = [(0.0, 0.0), (5.0, 2.0), (10.0, 5.0), (20.0, 10.0)]",
        "RUN_ALL_YEAR_CONFIRMATION = True",
        "def execute_backtests() -> list[dict]",
        'error_type=type(exc).__name__',
        'status="FAILED"',
        "runtime.unassign()",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_replay_notebook_declares_expected_artifacts() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "heartbeat.json",
        "lambdarankic_110_name_replay_manifest.json",
        "saved_prediction_inventory.csv",
        "drive_artifact_inventory.csv",
        "staged_prediction_inventory.csv",
        "planned_backtest_commands.csv",
        "backtest_results.csv",
        "backtest_results.json",
        "canonical_backtest_dir",
        "BACKTEST_ROOT",
        "rank_stability_summary.csv",
        "cross_seed_rank_correlation.csv",
        "cross_seed_jaccard.csv",
        "comparison_pair_cap",
        '["year", "left_pair_cap", "right_pair_cap", "left_base_seed", "right_base_seed"]',
        "top10_boundary_churn.csv",
        "rank_drop_cost_sensitivity.csv",
        "decision_gate_report.md",
        "backtest_metrics.json",
        "trade_journal.csv",
        "daily_holdings.csv",
        "MediaIoBaseDownload",
        "MediaFileUpload",
        "drive_publication_verification.json",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_replay_notebook_uses_authenticated_drive_api_without_mount() -> None:
    combined = "\n".join(_code_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "auth.authenticate_user()",
        'DRIVE_SERVICE = build("drive", "v3")',
        'DRIVE_PROJECT_FOLDER_ID = "1KUIj06ekfNpZa1IkkcAdhHXbVZt-PYT5"',
        'RUN_ROOT = Path("/content/mci_gru_runs")',
        'BRANCH = "codex/lambdarankic-saved-prediction-replay-20260713"',
        '"accelerator": "CPU"',
    ]
    for token in required_tokens:
        assert token in combined or token in generator

    assert combined.index("auth.authenticate_user()") < combined.index(
        'DRIVE_SERVICE = build("drive", "v3")'
    )
    assert "drive.mount(" not in combined
    assert "drive.mount(" not in generator
    assert "/content/drive" not in combined
    assert "/content/drive" not in generator


def test_replay_notebook_downloads_exact_pit_inputs_atomically() -> None:
    combined = "\n".join(_code_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        'MARKET_FILE_ID = "1e6aXtSkQGgsAjmytRsUt-xoJTYssWkPq"',
        "MARKET_FILE_SIZE = 39459457",
        'PIT_FILE_ID = "11WAppghYylyyBWLeisIhJ-505y2ptTr1"',
        "PIT_FILE_SIZE = 15940",
        'LOCAL_DATA_ROOT = Path("/content/mci_gru_inputs")',
        "def download_drive_file",
        'Path(f"{target_path}.part")',
        "actual_size != expected_size",
        "partial_path.replace(target_path)",
        "get_media(fileId=file_id, supportsAllDrives=True)",
    ]
    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_replay_notebook_publishes_and_verifies_drive_artifacts() -> None:
    combined = "\n".join(_code_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "def ensure_drive_folder",
        "def upload_or_update_drive_file",
        "def publish_run_artifacts",
        "def verify_published_artifacts",
        "supportsAllDrives=True",
        '"parents": [parent_id]',
        "PUBLISHED_FILE_STATS",
        "failure_publication = publish_run_artifacts()",
        'verify_published_artifacts(["heartbeat.json"])',
        "completion_publication = publish_run_artifacts()",
        "required_remote_artifacts",
        "verify_published_csv_directory",
        "executed_backtest_artifacts",
        '"backtest_metrics.json", "trade_journal.csv", "daily_holdings.csv"',
        "final_publication = publish_run_artifacts()",
        'remote_heartbeat = read_published_json("heartbeat.json")',
        "drive_publication_verification.json",
    ]
    for token in required_tokens:
        assert token in combined
        assert token in generator

    failure_cell = next(source for source in _code_cell_sources() if "failure_publication" in source)
    assert failure_cell.index('status="FAILED"') < failure_cell.index(
        "failure_publication = publish_run_artifacts()"
    )
    assert failure_cell.index("failure_publication = publish_run_artifacts()") < failure_cell.index(
        'verify_published_artifacts(["heartbeat.json"])'
    )
    assert failure_cell.index('verify_published_artifacts(["heartbeat.json"])') < failure_cell.index(
        "runtime.unassign()"
    )

    completion_cell = next(source for source in _code_cell_sources() if "completion_publication" in source)
    assert completion_cell.index('status="RUNNING"') < completion_cell.index(
        "completion_publication = publish_run_artifacts()"
    )
    assert completion_cell.index("completion_publication = publish_run_artifacts()") < completion_cell.index(
        "publication_verification = verify_published_artifacts"
    )
    assert completion_cell.index("publication_verification = verify_published_artifacts") < completion_cell.index(
        'status="COMPLETE"'
    )
    assert completion_cell.index("final_publication = publish_run_artifacts()") < completion_cell.index(
        'remote_heartbeat = read_published_json("heartbeat.json")'
    )
    assert completion_cell.index('remote_heartbeat = read_published_json("heartbeat.json")') < completion_cell.index(
        "runtime.unassign()"
    )


def test_generator_reproduces_committed_notebook(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR_PATH.resolve())],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    generated = json.loads(
        (tmp_path / "notebooks" / NOTEBOOK_PATH.name).read_text(encoding="utf-8")
    )
    committed = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert generated == committed


def test_replay_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
