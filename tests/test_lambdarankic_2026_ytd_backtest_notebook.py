import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/lambdarankic_2026_ytd_backtest_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_lambdarankic_2026_ytd_backtest_nb.py")


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


def test_notebook_is_cpu_replay_only_for_completed_run() -> None:
    combined = "\n".join(_cell_sources())
    code_cells = "\n".join(_code_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required = [
        "LambdaRankIC 2026-YTD replay-only backtests",
        "RUN_TRAINING = False",
        "REQUIRE_CPU = True",
        "Runtime policy: CPU replay-only",
        "Visible GPU: none",
        "20260714_215538",
        "1039LRjF_7mQ9v6g0iXzWU_5kh3WYKVcF",
        "EXPECTED_SEEDS = [314159, 271828, 161803, 141421, 173205]",
        "9bd17d5b7ff14594681c7bdbee3bb17a9882b264",
        "3d224de423ed7064cbc290f288af64a65fcd629f",
        "a34ae4b778b03a12c464f794f79a72caa2024a75d376f45b275175f5507768a8",
    ]
    for token in required:
        assert token in combined
        assert token in generator

    assert "run_experiment.py" not in code_cells
    assert '"accelerator": "GPU"' not in NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_notebook_starts_with_strict_read_only_eligibility_audit() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required = [
        "No replay output folder has been created; the Drive audit is read-only.",
        'root_heartbeat.get("phase") != "complete"',
        'root_heartbeat.get("remote_durability_verified")',
        'root_summary.get("completed_jobs", -1)',
        'row.get("remote_run_relative_path")',
        'row.get("remote_seed_manifest")',
        'durability.get("status") != "VERIFIED"',
        'durability.get("checkpoint_count", -1)',
        'durability.get("averaged_predictions", {})',
        'archive_proof.get("model_count", -1)',
        'artifact_validation.get("status") != "OK"',
        'training_summary.get("models_trained", -1)',
        "averaged_prediction_manifest",
        'required_columns = {"kdcode", "dt", "score"}',
        "AUDIT_COMPLETE = True",
    ]
    for token in required:
        assert token in combined
        assert token in generator

    audit_index = combined.index("AUDIT_COMPLETE = True")
    output_creation_index = combined.index("OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)")
    assert audit_index < output_creation_index


def test_notebook_uses_approved_data_and_cutoff_contract() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required = [
        "sp500_pit_gics_top10_mcap_monthly_20210104_20260713_",
        "lseg_20190101_20260713.csv",
        "pit_universe.csv",
        'TEST_START = "2026-01-01"',
        'TEST_END = "2026-07-13"',
        'FIRST_PREDICTION_SESSION = "2026-01-02"',
        'REALIZED_T5_CUTOFF = "2026-07-06"',
        'LABEL_T = 5',
        "market CSV SHA-256 mismatch",
        "PIT CSV SHA-256 mismatch",
        "The 2026-07-06 cutoff applies to realized t+5 label metrics",
        "last_strategy_score_date",
        "last_strategy_entry_date",
        "last_strategy_return_date",
    ]
    for token in required:
        assert token in combined
        assert token in generator


def test_notebook_invokes_canonical_daily_backtest_contract() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required = [
        'str(REPO_DIR / "scripts" / "backtest_sp500_daily.py")',
        '"--predictions_dir"',
        '"--data_file"',
        '"--pit_universe_csv"',
        '"--test_start"',
        '"--test_end"',
        '"--label_t"',
        '"--top_k"',
        '"--num_tests"',
        '"--adjustment_method"',
        '"--auto_save"',
        '"--backtest_suffix"',
        '"--transaction_costs"',
        '"--spread"',
        '"--slippage"',
        '"--enable_rank_drop_gate"',
        '"--min_rank_drop"',
        'BACKTEST_SUFFIX = "_top10_tc_rankdrop"',
        '"primary_backtest_input": "averaged_predictions only"',
    ]
    for token in required:
        assert token in combined
        assert token in generator

    assert "predictions_model_*" not in combined


def test_notebook_declares_non_overwriting_outputs_and_cleanup() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required = [
        "replay_only_top10_tc_rankdrop_",
        "exist_ok=False",
        "coverage_audit.json",
        "seed_eligibility.csv",
        "seed_eligibility.json",
        "backtest_commands.json",
        "backtest_rows.csv",
        "backtest_rows.json",
        "cross_seed_mean_sample_std.csv",
        "cross_seed_mean_sample_std.json",
        "run_summary.json",
        "heartbeat.json",
        "report.md",
        "colab_run_review.md",
        "source_manifests",
        "all_expected_seeds_accounted_for",
        "Drive artifact readback: OK",
        "runtime.unassign()",
    ]
    for token in required:
        assert token in combined
        assert token in generator

    readback_index = combined.index("Drive artifact readback: OK")
    cleanup_index = combined.index("runtime.unassign()")
    assert readback_index < cleanup_index


def test_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
