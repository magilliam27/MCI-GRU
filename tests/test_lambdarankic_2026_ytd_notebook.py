"""Contract tests for the approval-gated 2026-YTD LambdaRankIC campaign."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

NOTEBOOK_PATH = Path("notebooks/lambdarankic_2026_ytd_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_lambdarankic_2026_ytd_nb.py")
EXPERIMENT_PATH = Path("configs/experiment/lambdarankic_2026_ytd_110_name.yaml")
APPROVAL_PATH = Path("configs/launch_manifests/lambdarankic_2026_ytd_110_name.json")


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cell_sources(*, code_only: bool = False) -> list[str]:
    cells = _notebook()["cells"]
    if code_only:
        cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    return ["".join(cell.get("source", [])) for cell in cells]


def _approval_bundle() -> tuple[dict, str]:
    combined = "\n".join(_cell_sources(code_only=True))
    match = re.search(
        r"APPROVAL_BUNDLE = json\.loads\(r'''(.*?)'''\)",
        combined,
        flags=re.DOTALL,
    )
    assert match is not None
    bundle = json.loads(match.group(1))
    digest = hashlib.sha256(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return bundle, digest


def test_campaign_digest_binds_exact_resolved_config() -> None:
    bundle, digest = _approval_bundle()
    combined = "\n".join(_cell_sources())
    approval_artifact = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))

    assert digest == "a34ae4b778b03a12c464f794f79a72caa2024a75d376f45b275175f5507768a8"
    assert f'EXPECTED_CONFIG_SHA256 = "{digest}"' in combined
    assert approval_artifact == {"config_sha256": digest, **bundle}
    for relative, expected in bundle["campaign"]["source_files_sha256"].items():
        content = Path(relative).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == expected
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=lambdarankic_2026_ytd_110_name"],
        )
    assert OmegaConf.to_container(cfg, resolve=True) == bundle["resolved_hydra_config"]


def test_campaign_pins_requested_split_and_multi_seed_lambdarank() -> None:
    bundle, _ = _approval_bundle()
    campaign = bundle["campaign"]
    cfg = bundle["resolved_hydra_config"]

    assert campaign["base_seeds"] == [314159, 271828, 161803, 141421, 173205]
    assert campaign["expected_training_jobs"] == 5
    assert campaign["ensemble_models_per_seed"] == 20
    assert campaign["expected_model_fits"] == 100
    assert campaign["artifact_contract"]["resume_granularity"].startswith("no automatic resume")
    assert campaign["source_files_sha256"]
    for required_source in (
        "mci_gru/graph/builder.py",
        "mci_gru/data/data_manager.py",
        "mci_gru/data/pit.py",
        "mci_gru/models/mci_gru.py",
        "mci_gru/evaluation/metrics.py",
    ):
        assert required_source in campaign["source_files_sha256"]
    assert campaign["split_contract"] == {
        "boundary_policy": "actual-session embargo; predictions include the unlabeled YTD tail",
        "label_sessions": 5,
        "test": ["2026-01-01", "2026-07-13"],
        "train": ["2021-01-01", "2024-12-31"],
        "validation": ["2025-01-10", "2025-12-23"],
    }
    assert cfg["training"]["loss_type"] == "lambdarank_ic"
    assert cfg["training"]["selection_metric"] == "val_rank_ic"
    assert cfg["training"]["lambdarank_ic_max_pairs_per_day"] == 8192
    assert cfg["training"]["lambdarank_ic_temperature"] == 1.0
    assert cfg["training"]["num_models"] == 20
    assert cfg["training"]["num_epochs"] == 100
    assert cfg["training"]["early_stopping_patience"] == 15


def test_campaign_uses_true_pit_110_name_recipe_and_all_pairs() -> None:
    combined = "\n".join(_cell_sources())
    required_tokens = [
        "monthly PIT S&P 500 top-10 by market cap within each GICS sector",
        '"expected_active_names": 110',
        '"expected_sectors": 11',
        '"top_n_per_sector": 10',
        '"pit_universe_mode": "masked_panel"',
        '"pit_min_scoreable_stocks": 100',
        '"history_start": "2019-01-01"',
        '"test_label_complete_through"',
        '"train_per_stock_embargo"',
        '"validation_per_stock_embargo"',
        '"daily_complete_ohlcv_min"',
        '"static_graph_edge_count"',
        '"volume"',
        "Pair cap {pair_cap} is below the 110-name all-pairs count",
        "max_all_pairs",
    ]
    for token in required_tokens:
        assert token in combined


def test_notebook_gate_precedes_drive_mutation_and_training() -> None:
    launch = next(source for source in _cell_sources(code_only=True) if "RUN_TRAINING =" in source)

    assert "RUN_TRAINING = False" in launch
    assert 'APPROVED_CONFIG_SHA256 = ""' in launch
    gate_index = launch.index("if not RUN_TRAINING:")
    digest_index = launch.index("if APPROVED_CONFIG_SHA256 != launch_config_sha256:")
    source_index = launch.index("assert_approved_sources()")
    drive_create_index = launch.index("create_drive_folder(")
    training_index = launch.index('str(REPO_DIR / "run_experiment.py")')
    recompose_index = launch.index("launch_job_cfg = compose(")
    subprocess_index = launch.index("process = subprocess.Popen(")
    assert gate_index < digest_index < source_index < drive_create_index < training_index
    assert recompose_index < training_index < subprocess_index


def test_per_stock_embargo_checks_latest_label_with_a_finite_target() -> None:
    source = next(
        cell_source
        for cell_source in _cell_sources(code_only=True)
        if "def prove_per_stock_embargo(" in cell_source
    )
    tree = ast.parse(source)
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "prove_per_stock_embargo"
    )
    namespace = {
        "pit": pd.DataFrame(
            {
                "kdcode": ["A"],
                "valid_from": ["2025-12-01"],
                "valid_to": ["2025-12-23"],
            }
        ),
        "stock_dates": {
            "A": [
                "2025-12-19",
                "2025-12-22",
                "2025-12-23",
                "2025-12-24",
                "2025-12-26",
                "2026-01-02",
            ]
        },
        "label_t": 5,
    }
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), "<audit>", "exec"), namespace)

    with pytest.raises(RuntimeError, match="Per-stock label embargo failed"):
        namespace["prove_per_stock_embargo"](
            "2025-12-01",
            "2025-12-23",
            "2026-01-02",
        )


def test_notebook_is_visible_g4_only_and_persists_durable_artifacts() -> None:
    combined = "\n".join(_cell_sources())
    required_tokens = [
        'BRANCH = "codex/lambdarankic-ytd-drive-durability-20260714"',
        '"code_branch": "codex/lambdarankic-2026-ytd-20260713"',
        'BLOCKED_GPU_NAMES = ("T4", "L4")',
        'STRICT_GPU_MARKERS: list[str] = ["G4", "RTX PRO", "BLACKWELL"]',
        "visible Colab **G4 GPU** only",
        'CAMPAIGN_DRIVE_FOLDER_ID = "1iXHVKRwHBF3Jv_ruIMNWYIXFEyeYy4Po"',
        'DRIVE_SERVICE = build("drive", "v3")',
        "MediaFileUpload",
        "MediaIoBaseDownload",
        "resumable=True",
        '"md5Checksum"',
        'upload_json_verified("config_approval.json"',
        'upload_json_verified("data_audit.json"',
        '"runtime_gpu.txt"',
        'upload_json_verified("training_results.json"',
        'upload_json_verified("cross_seed_evaluation_summary.json"',
        'upload_json_verified("run_summary.json"',
        'upload_json_verified("heartbeat.json"',
        'run_dir / "graph_data.pt"',
        'run_dir / "averaged_predictions"',
        'run_dir.glob("predictions_model_*")',
        "per_model_predictions.zip",
        "model_artifacts",
        "upload_per_model_predictions=False",
        'read_remote_json("run_summary.json")',
        'read_remote_json("heartbeat.json")',
        "def verify_remote_run",
        "REMOTE_DURABILITY_VERIFIED = verify_remote_run",
        "runtime.unassign()",
        "Runtime intentionally left assigned because remote durability was not proved",
    ]
    for token in required_tokens:
        assert token in combined


def test_remote_durability_is_proved_before_success_and_runtime_release() -> None:
    launch = next(source for source in _cell_sources(code_only=True) if "RUN_TRAINING =" in source)

    assert "def sync_tree" not in launch
    assert "DRIVE_RUN_ROOT.mkdir" not in launch
    verify_index = launch.index("REMOTE_DURABILITY_VERIFIED = verify_remote_run")
    complete_index = launch.index('print("Complete. Durable run root:"')
    guarded_release_index = launch.index("if REMOTE_DURABILITY_VERIFIED:")
    unassign_index = launch.index("runtime.unassign()")
    assert verify_index < complete_index < guarded_release_index < unassign_index


def test_per_model_predictions_are_archived_without_raw_drive_fanout(tmp_path: Path) -> None:
    launch = next(source for source in _cell_sources(code_only=True) if "RUN_TRAINING =" in source)
    tree = ast.parse(launch)
    publish_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "publish_completed_seed"
    )
    publish_source = ast.get_source_segment(launch, publish_node)
    assert publish_source is not None
    assert "log_path" not in publish_source
    wanted = {"sha256_file", "build_per_model_predictions_archive"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"Path": Path, "hashlib": hashlib, "json": json, "zipfile": zipfile}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<archive>", "exec"), namespace)

    run_dir = tmp_path / "run"
    for model_index in range(2):
        prediction_dir = run_dir / f"predictions_model_{model_index}"
        prediction_dir.mkdir(parents=True)
        for day in ("2026-01-02", "2026-01-05"):
            (prediction_dir / f"{day}.csv").write_text("kdcode,prediction\nA,0.1\n", encoding="utf-8")

    archive_path, summary = namespace["build_per_model_predictions_archive"](
        run_dir,
        2,
        2,
    )
    assert archive_path == run_dir / "model_artifacts" / "per_model_predictions.zip"
    assert summary["model_count"] == 2
    assert summary["member_count"] == 4
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "per_model_predictions_manifest.json" in names
        assert "predictions_model_0/2026-01-02.csv" in names
        assert "predictions_model_1/2026-01-05.csv" in names

    assert "upload_per_model_predictions=False" in launch
    assert "Raw per-model predictions entered the Drive upload set" in launch


def test_drive_verifier_rejects_remote_size_and_md5_mismatch(tmp_path: Path) -> None:
    launch = next(source for source in _cell_sources(code_only=True) if "RUN_TRAINING =" in source)
    tree = ast.parse(launch)
    wanted = {"sha256_file", "md5_file", "verify_remote_file_metadata"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"Path": Path, "hashlib": hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<verify>", "exec"), namespace)

    local_path = tmp_path / "artifact.bin"
    local_path.write_bytes(b"durable")
    local_md5 = hashlib.md5(b"durable").hexdigest()
    verified = namespace["verify_remote_file_metadata"](
        local_path,
        {"id": "remote-id", "size": str(local_path.stat().st_size), "md5Checksum": local_md5},
        "artifact.bin",
    )
    assert verified["file_id"] == "remote-id"

    with pytest.raises(RuntimeError, match="Remote size mismatch"):
        namespace["verify_remote_file_metadata"](
            local_path,
            {"id": "remote-id", "size": "999", "md5Checksum": local_md5},
            "artifact.bin",
        )
    with pytest.raises(RuntimeError, match="Remote MD5 mismatch"):
        namespace["verify_remote_file_metadata"](
            local_path,
            {"id": "remote-id", "size": str(local_path.stat().st_size), "md5Checksum": "bad"},
            "artifact.bin",
        )


def test_seed_remote_verification_precedes_completion_accounting() -> None:
    launch = next(source for source in _cell_sources(code_only=True) if "RUN_TRAINING =" in source)

    publish_index = launch.index("seed_remote = publish_completed_seed(")
    append_index = launch.index("rows.append(row)")
    summary_readback_index = launch.index('remote_summary = read_remote_json("run_summary.json")')
    terminal_verify_index = launch.index("REMOTE_DURABILITY_VERIFIED = verify_remote_run(rows)")
    assert publish_index < append_index
    assert summary_readback_index < terminal_verify_index


def test_experiment_yaml_pins_no_lookahead_and_frozen_recipe() -> None:
    source = EXPERIMENT_PATH.read_text(encoding="utf-8")
    required_tokens = [
        'train_start: "2021-01-01"',
        'train_end: "2024-12-31"',
        'val_start: "2025-01-10"',
        'val_end: "2025-12-23"',
        'test_start: "2026-01-01"',
        'test_end: "2026-07-13"',
        "skip_embargo_check: false",
        "normalisation: zscore",
        "pit_universe_mode: masked_panel",
        "regime_strict: true",
        "regime_include_subsequent_returns: false",
        "update_frequency_months: 0",
        "use_multi_feature_edges: true",
        "drop_edge_p: 0.1",
        "loss_type: lambdarank_ic",
        "selection_metric: val_rank_ic",
        "label_type: returns",
        "shuffle_train: true",
    ]
    for token in required_tokens:
        assert token in source


def test_generator_and_notebook_code_parse() -> None:
    ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"))
    for source in _cell_sources(code_only=True):
        ast.parse(source)
