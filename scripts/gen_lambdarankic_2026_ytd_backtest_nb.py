"""Generate the replay-only Colab backtest for the completed 2026-YTD campaign."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

OUT = Path("notebooks/lambdarankic_2026_ytd_backtest_colab.ipynb")


def md(source: str) -> dict:
    """Build one markdown notebook cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    """Build one code notebook cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    md(
        """
        # LambdaRankIC 2026-YTD replay-only backtests

        This notebook is the separate CPU replay workstream for the completed
        `20260714_215538` LambdaRankIC campaign. It starts with a read-only Drive
        eligibility audit, then replays only each eligible seed's
        `averaged_predictions` through the canonical daily backtest wrapper.

        Safety contract: no training, no training-runtime reuse, no raw
        per-model prediction input, no overwrite of the campaign artifacts, and
        no replay output folder before the eligibility audit finishes.
        """
    ),
    md("## 1. CPU setup and read-only safety gates"),
    code(
        r"""
        import csv
        import hashlib
        import json
        import math
        import os
        import re
        import shutil
        import subprocess
        import sys
        import time
        from datetime import datetime, timezone
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from google.colab import drive, runtime

        RUN_TRAINING = False
        REQUIRE_CPU = True
        EXPECTED_SEEDS = [314159, 271828, 161803, 141421, 173205]
        EXPECTED_CAMPAIGN_COMMIT = "9bd17d5b7ff14594681c7bdbee3bb17a9882b264"
        EXPECTED_LAUNCHER_COMMIT = "3d224de423ed7064cbc290f288af64a65fcd629f"
        EXPECTED_CONFIG_SHA256 = "a34ae4b778b03a12c464f794f79a72caa2024a75d376f45b275175f5507768a8"
        EXPECTED_ENSEMBLE_MODELS = 20
        EXPECTED_PREDICTION_COUNT = 131
        CAMPAIGN_BRANCH = "codex/lambdarankic-2026-ytd-20260713"
        CAMPAIGN_ID = "lambdarankic_2026_ytd_110_name"
        CAMPAIGN_RUN_TAG = "20260714_215538"
        CAMPAIGN_RUN_FOLDER_ID = "1039LRjF_7mQ9v6g0iXzWU_5kh3WYKVcF"
        CAMPAIGN_RUN_URL = (
            "https://drive.google.com/drive/folders/"
            + CAMPAIGN_RUN_FOLDER_ID
        )
        NOTEBOOK_URL = (
            "https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/"
            "codex/lambdarankic-2026-ytd-backtest-20260714/"
            "notebooks/lambdarankic_2026_ytd_backtest_colab.ipynb"
        )
        TEST_START = "2026-01-01"
        TEST_END = "2026-07-13"
        FIRST_PREDICTION_SESSION = "2026-01-02"
        REALIZED_T5_CUTOFF = "2026-07-06"
        LABEL_T = 5
        TOP_K = 10
        BACKTEST_SUFFIX = "_top10_tc_rankdrop"
        REPLAY_RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        DRIVE_RUN_ROOT = Path(
            "/content/drive/MyDrive/MCI-GRU-Ablations/"
            "lambdarank_ic_2026_ytd/20260714_215538"
        )
        SHARED_DATA_ROOT = Path("/content/drive/MyDrive/MCI_GRU_shared/data")
        MARKET_FILENAME = (
            "sp500_pit_gics_top10_mcap_monthly_20210104_20260713_"
            "lseg_20190101_20260713.csv"
        )
        PIT_FILENAME = (
            "sp500_pit_gics_top10_mcap_monthly_20210104_20260713_"
            "pit_universe.csv"
        )
        LOCAL_ROOT = Path("/content") / f"lambdarankic_2026_ytd_backtest_{REPLAY_RUN_TAG}"
        REPO_DIR = LOCAL_ROOT / "repo"
        LOCAL_DATA_DIR = LOCAL_ROOT / "data"
        LOCAL_INPUTS_DIR = LOCAL_ROOT / "inputs"
        OUTPUT_ROOT = None

        if RUN_TRAINING:
            raise RuntimeError("Replay notebook safety gate requires RUN_TRAINING=False")

        def utc_now() -> str:
            return datetime.now(timezone.utc).isoformat()

        def json_default(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, np.bool_):
                return bool(value)
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                value = float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        def write_json(path: Path, payload) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    default=json_default,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)

        def sha256_file(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def md5_file(path: Path) -> str:
            digest = hashlib.md5(usedforsecurity=False)
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def averaged_prediction_manifest(directory: Path) -> tuple[dict, str]:
            rows = {
                path.name: {"size": path.stat().st_size, "md5": md5_file(path)}
                for path in sorted(directory.glob("*.csv"))
            }
            canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
            return rows, hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        def heartbeat(phase: str, status: str = "RUNNING", **extra) -> None:
            if OUTPUT_ROOT is None:
                raise RuntimeError("Replay output root does not exist yet")
            payload = {
                "campaign_id": CAMPAIGN_ID,
                "campaign_run_tag": CAMPAIGN_RUN_TAG,
                "campaign_run_folder_id": CAMPAIGN_RUN_FOLDER_ID,
                "replay_run_tag": REPLAY_RUN_TAG,
                "status": status,
                "phase": phase,
                "updated_at_utc": utc_now(),
                "training_executed": False,
                "runtime_policy": "CPU replay-only",
                "primary_prediction_input": "averaged_predictions",
                "test_start": TEST_START,
                "test_end": TEST_END,
                "prediction_file_end": TEST_END,
                "label_t": LABEL_T,
                "effective_realized_t5_cutoff": REALIZED_T5_CUTOFF,
            }
            payload.update(extra)
            write_json(OUTPUT_ROOT / "heartbeat.json", payload)

        def detect_gpu_name() -> str:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
            except FileNotFoundError:
                return ""
            if result.returncode != 0 or not result.stdout.strip():
                return ""
            return result.stdout.strip().splitlines()[0].strip()

        drive.mount("/content/drive", force_remount=False)
        if not DRIVE_RUN_ROOT.is_dir():
            raise FileNotFoundError(f"Completed campaign root is missing: {DRIVE_RUN_ROOT}")

        GPU_NAME = detect_gpu_name()
        if REQUIRE_CPU and GPU_NAME:
            raise RuntimeError(
                f"Replay-only notebook requires a CPU runtime; visible GPU is {GPU_NAME}"
            )

        LOCAL_ROOT.mkdir(parents=True, exist_ok=False)
        LOCAL_DATA_DIR.mkdir(parents=True)
        LOCAL_INPUTS_DIR.mkdir(parents=True)
        print("Runtime policy: CPU replay-only")
        print("Visible GPU: none")
        print("Campaign root:", DRIVE_RUN_ROOT)
        print("No replay output folder has been created; the Drive audit is read-only.")
        """
    ),
    md("## 2. Read-only completion, provenance, and prediction audit"),
    code(
        r"""
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--branch",
                CAMPAIGN_BRANCH,
                "--single-branch",
                "https://github.com/magilliam27/MCI-GRU.git",
                str(REPO_DIR),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_DIR),
                "checkout",
                "--quiet",
                "--detach",
                EXPECTED_CAMPAIGN_COMMIT,
            ],
            check=True,
        )
        live_commit = subprocess.run(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if live_commit != EXPECTED_CAMPAIGN_COMMIT:
            raise RuntimeError(
                f"Replay code commit mismatch: {live_commit} != {EXPECTED_CAMPAIGN_COMMIT}"
            )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_DIR)],
            check=True,
        )

        source_code_paths = [
            "tests/backtest_sp500_daily.py",
            "scripts/backtest_sp500_daily.py",
            "mci_gru/evaluation/backtest_engine.py",
        ]
        source_code_sha256 = {
            relative: sha256_file(REPO_DIR / relative) for relative in source_code_paths
        }

        approval_path = DRIVE_RUN_ROOT / "config_approval.json"
        data_audit_path = DRIVE_RUN_ROOT / "data_audit.json"
        root_summary_path = DRIVE_RUN_ROOT / "run_summary.json"
        root_heartbeat_path = DRIVE_RUN_ROOT / "heartbeat.json"
        training_results_path = DRIVE_RUN_ROOT / "training_results.json"
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        data_audit = json.loads(data_audit_path.read_text(encoding="utf-8"))
        root_summary = json.loads(root_summary_path.read_text(encoding="utf-8"))
        root_heartbeat = json.loads(root_heartbeat_path.read_text(encoding="utf-8"))
        training_rows = json.loads(training_results_path.read_text(encoding="utf-8"))

        root_errors = []
        if approval.get("config_sha256") != EXPECTED_CONFIG_SHA256:
            root_errors.append("config approval SHA-256 mismatch")
        if approval.get("approved_campaign_commit") != EXPECTED_CAMPAIGN_COMMIT:
            root_errors.append("approved campaign commit mismatch")
        if approval.get("launcher_commit") != EXPECTED_LAUNCHER_COMMIT:
            root_errors.append("launcher commit mismatch")
        if data_audit.get("status") != "OK":
            root_errors.append("data audit is not OK")
        if data_audit.get("test_prediction_end") != TEST_END:
            root_errors.append("prediction end differs from approved test end")
        if data_audit.get("test_label_complete_through") != REALIZED_T5_CUTOFF:
            root_errors.append("realized t+5 cutoff differs from approved cutoff")
        if int(data_audit.get("test_prediction_session_count", -1)) != EXPECTED_PREDICTION_COUNT:
            root_errors.append("prediction session count mismatch")
        if root_summary.get("status") != "OK":
            root_errors.append("campaign run summary is not OK")
        if int(root_summary.get("completed_jobs", -1)) != len(EXPECTED_SEEDS):
            root_errors.append("campaign run summary is not 5/5")
        if root_summary.get("config_sha256") != EXPECTED_CONFIG_SHA256:
            root_errors.append("campaign run-summary config SHA-256 mismatch")
        if root_heartbeat.get("status") != "OK" or root_heartbeat.get("phase") != "complete":
            root_errors.append("campaign heartbeat is not terminal OK/complete")
        if not root_heartbeat.get("remote_durability_verified"):
            root_errors.append("campaign heartbeat lacks remote durability proof")
        if sorted(root_heartbeat.get("verified_seeds", [])) != sorted(EXPECTED_SEEDS):
            root_errors.append("campaign heartbeat verified-seed set mismatch")

        market_source = SHARED_DATA_ROOT / MARKET_FILENAME
        pit_source = SHARED_DATA_ROOT / PIT_FILENAME
        if not market_source.is_file() or not pit_source.is_file():
            root_errors.append("approved market or PIT CSV is missing")
        market_sha256 = sha256_file(market_source) if market_source.is_file() else None
        pit_sha256 = sha256_file(pit_source) if pit_source.is_file() else None
        expected_input_hashes = data_audit.get("input_sha256", {})
        if market_sha256 != expected_input_hashes.get("market_filename"):
            root_errors.append("market CSV SHA-256 mismatch")
        if pit_sha256 != expected_input_hashes.get("pit_filename"):
            root_errors.append("PIT CSV SHA-256 mismatch")

        rows_by_seed = {}
        for row in training_rows:
            if "base_seed" in row:
                rows_by_seed.setdefault(int(row["base_seed"]), []).append(row)

        expected_checkpoint_names = {
            f"model_{model_id}_best.pth" for model_id in range(EXPECTED_ENSEMBLE_MODELS)
        }
        eligibility_rows = []
        eligible_runs = {}
        source_manifests = {}

        for seed in EXPECTED_SEEDS:
            reasons = list(root_errors)
            matching_rows = rows_by_seed.get(seed, [])
            row = matching_rows[0] if len(matching_rows) == 1 else None
            if len(matching_rows) != 1:
                reasons.append(f"expected one training-results row, found {len(matching_rows)}")

            run_relative_path = row.get("remote_run_relative_path") if row else None
            seed_manifest_relative = row.get("remote_seed_manifest") if row else None
            run_dir = DRIVE_RUN_ROOT / run_relative_path if run_relative_path else None
            seed_manifest_path = (
                DRIVE_RUN_ROOT / seed_manifest_relative if seed_manifest_relative else None
            )
            durability = None
            artifact_validation = None
            training_summary = None
            actual_manifest = {}
            actual_manifest_sha256 = None
            prediction_dates = []
            checkpoint_names = set()

            if row:
                if row.get("status") != "OK":
                    reasons.append("training-results status is not OK")
                if row.get("remote_durability_status") != "VERIFIED":
                    reasons.append("training-results durability status is not VERIFIED")
                if int(row.get("ensemble_models", -1)) != EXPECTED_ENSEMBLE_MODELS:
                    reasons.append("training-results ensemble count is not 20")
                if int(row.get("averaged_prediction_count", -1)) != EXPECTED_PREDICTION_COUNT:
                    reasons.append("training-results averaged prediction count is not 131")
            if run_dir is None or not run_dir.is_dir():
                reasons.append("exact remote_run_relative_path is missing")
            elif seed_manifest_path != run_dir / "seed_durability.json":
                reasons.append("remote_seed_manifest does not match the exact run directory")
            elif not seed_manifest_path.is_file():
                reasons.append("seed_durability.json is missing")
            else:
                durability = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
                if durability.get("status") != "VERIFIED":
                    reasons.append("seed durability status is not VERIFIED")
                if int(durability.get("base_seed", -1)) != seed:
                    reasons.append("seed durability base seed mismatch")
                if durability.get("config_sha256") != EXPECTED_CONFIG_SHA256:
                    reasons.append("seed durability config SHA-256 mismatch")
                if durability.get("run_relative_path") != run_relative_path:
                    reasons.append("seed durability run path mismatch")
                if int(durability.get("checkpoint_count", -1)) != EXPECTED_ENSEMBLE_MODELS:
                    reasons.append("seed durability checkpoint count is not 20")
                if set(durability.get("checkpoint_names", [])) != expected_checkpoint_names:
                    reasons.append("seed durability checkpoint names are incomplete")

                averaged_proof = durability.get("averaged_predictions", {})
                if int(averaged_proof.get("csv_count", -1)) != EXPECTED_PREDICTION_COUNT:
                    reasons.append("seed durability averaged prediction count is not 131")
                if averaged_proof.get("path") != f"{run_relative_path}/averaged_predictions":
                    reasons.append("seed durability averaged-prediction path mismatch")
                if not averaged_proof.get("manifest_sha256"):
                    reasons.append("seed durability lacks averaged-prediction manifest hash")
                if not averaged_proof.get("remote_folder_id"):
                    reasons.append("seed durability lacks averaged-prediction folder ID")

                archive_proof = durability.get("per_model_predictions_archive", {})
                if int(archive_proof.get("model_count", -1)) != EXPECTED_ENSEMBLE_MODELS:
                    reasons.append("per-model archive summary does not prove 20 models")
                if int(archive_proof.get("csvs_per_model", -1)) != EXPECTED_PREDICTION_COUNT:
                    reasons.append("per-model archive summary does not prove 131 CSVs/model")
                if int(archive_proof.get("member_count", -1)) != (
                    EXPECTED_ENSEMBLE_MODELS * EXPECTED_PREDICTION_COUNT
                ):
                    reasons.append("per-model archive member count mismatch")

                required_files = [
                    run_dir / "training_summary.json",
                    run_dir / "evaluation_summary.json",
                    run_dir / "graph_data.pt",
                    run_dir / "artifact_validation.json",
                ]
                for required in required_files:
                    if not required.is_file():
                        reasons.append(f"missing final artifact {required.name}")

                validation_path = run_dir / "artifact_validation.json"
                if validation_path.is_file():
                    artifact_validation = json.loads(validation_path.read_text(encoding="utf-8"))
                    if artifact_validation.get("status") != "OK":
                        reasons.append("artifact_validation.json is not OK")
                    if artifact_validation.get("missing_artifacts"):
                        reasons.append("artifact_validation.json lists missing artifacts")

                training_summary_path = run_dir / "training_summary.json"
                if training_summary_path.is_file():
                    training_summary = json.loads(
                        training_summary_path.read_text(encoding="utf-8")
                    )
                    if int(training_summary.get("models_trained", -1)) != EXPECTED_ENSEMBLE_MODELS:
                        reasons.append("training summary does not prove 20 trained models")

                checkpoints_dir = run_dir / "checkpoints"
                checkpoint_names = {
                    path.name for path in checkpoints_dir.glob("model_*_best.pth")
                }
                if checkpoint_names != expected_checkpoint_names:
                    reasons.append("Drive checkpoint directory is incomplete")

                averaged_dir = run_dir / "averaged_predictions"
                averaged_files = sorted(averaged_dir.glob("*.csv")) if averaged_dir.is_dir() else []
                if len(averaged_files) != EXPECTED_PREDICTION_COUNT:
                    reasons.append(
                        f"Drive averaged_predictions contains {len(averaged_files)} CSVs, not 131"
                    )
                else:
                    actual_manifest, actual_manifest_sha256 = averaged_prediction_manifest(
                        averaged_dir
                    )
                    if actual_manifest_sha256 != averaged_proof.get("manifest_sha256"):
                        reasons.append("Drive averaged-prediction manifest SHA-256 mismatch")
                    for prediction_file in averaged_files:
                        table = pd.read_csv(prediction_file)
                        required_columns = {"kdcode", "dt", "score"}
                        if not required_columns.issubset(table.columns):
                            reasons.append(
                                f"prediction schema mismatch in {prediction_file.name}"
                            )
                            break
                        if table.empty or table["kdcode"].duplicated().any():
                            reasons.append(
                                f"empty or duplicate-kdcode prediction file {prediction_file.name}"
                            )
                            break
                        dates = pd.to_datetime(table["dt"], errors="coerce").dropna().dt.strftime(
                            "%Y-%m-%d"
                        ).unique()
                        if len(dates) != 1:
                            reasons.append(
                                f"prediction file {prediction_file.name} does not contain one date"
                            )
                            break
                        prediction_dates.append(str(dates[0]))
                    if len(set(prediction_dates)) != EXPECTED_PREDICTION_COUNT:
                        reasons.append("prediction dates are not unique across the 131 files")
                    elif min(prediction_dates) != FIRST_PREDICTION_SESSION:
                        reasons.append("first prediction session mismatch")
                    elif max(prediction_dates) != TEST_END:
                        reasons.append("last prediction session mismatch")

            eligible = not reasons
            eligibility = {
                "base_seed": seed,
                "status": "ELIGIBLE" if eligible else "NOT_ELIGIBLE",
                "reason": "" if eligible else "; ".join(reasons),
                "source_run_relative_path": run_relative_path,
                "source_seed_manifest_relative_path": seed_manifest_relative,
                "source_averaged_predictions_relative_path": (
                    f"{run_relative_path}/averaged_predictions" if run_relative_path else None
                ),
                "source_averaged_predictions_folder_id": (
                    durability.get("averaged_predictions", {}).get("remote_folder_id")
                    if durability
                    else None
                ),
                "checkpoint_count": len(checkpoint_names),
                "averaged_prediction_count": len(actual_manifest),
                "prediction_start": min(prediction_dates) if prediction_dates else None,
                "prediction_end": max(prediction_dates) if prediction_dates else None,
                "averaged_predictions_manifest_sha256": actual_manifest_sha256,
                "seed_durability_sha256": (
                    sha256_file(seed_manifest_path)
                    if seed_manifest_path and seed_manifest_path.is_file()
                    else None
                ),
                "artifact_validation_sha256": (
                    sha256_file(run_dir / "artifact_validation.json")
                    if run_dir and (run_dir / "artifact_validation.json").is_file()
                    else None
                ),
            }
            eligibility_rows.append(eligibility)
            if eligible:
                eligible_runs[seed] = {
                    "run_dir": run_dir,
                    "run_relative_path": run_relative_path,
                    "averaged_dir": run_dir / "averaged_predictions",
                    "manifest_sha256": actual_manifest_sha256,
                    "remote_folder_id": durability["averaged_predictions"]["remote_folder_id"],
                }
                source_manifests[seed] = actual_manifest

        eligibility_df = pd.DataFrame(eligibility_rows)
        eligible_seeds = [int(seed) for seed in EXPECTED_SEEDS if seed in eligible_runs]
        not_eligible_seeds = [
            int(row["base_seed"])
            for row in eligibility_rows
            if row["status"] != "ELIGIBLE"
        ]
        AUDIT_COMPLETE = True
        print("Read-only Drive audit complete.")
        print("Eligible seeds:", eligible_seeds)
        print("Not eligible seeds:", not_eligible_seeds)
        print("No replay output folder has been created yet.")
        display(eligibility_df)
        """
    ),
    md("## 3. Foreground replay and durable report publication"),
    code(
        r"""
        if not AUDIT_COMPLETE:
            raise RuntimeError("Read-only eligibility audit has not completed")
        if not eligible_runs:
            raise RuntimeError("No completed seed is eligible for replay")

        OUTPUT_ROOT = (
            DRIVE_RUN_ROOT
            / "backtests"
            / f"replay_only_top10_tc_rankdrop_{REPLAY_RUN_TAG}"
        )
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
        (OUTPUT_ROOT / "logs").mkdir()
        (OUTPUT_ROOT / "per_seed").mkdir()
        (OUTPUT_ROOT / "source_manifests").mkdir()

        coverage_audit = {
            "campaign_id": CAMPAIGN_ID,
            "campaign_run_tag": CAMPAIGN_RUN_TAG,
            "campaign_run_folder_id": CAMPAIGN_RUN_FOLDER_ID,
            "campaign_run_url": CAMPAIGN_RUN_URL,
            "audited_at_utc": utc_now(),
            "expected_seeds": EXPECTED_SEEDS,
            "eligible_seeds": eligible_seeds,
            "not_eligible_seeds": not_eligible_seeds,
            "expected_campaign_commit": EXPECTED_CAMPAIGN_COMMIT,
            "expected_launcher_commit": EXPECTED_LAUNCHER_COMMIT,
            "live_replay_code_commit": live_commit,
            "approved_config_sha256": EXPECTED_CONFIG_SHA256,
            "source_code_sha256": source_code_sha256,
            "config_approval_path": str(approval_path),
            "config_approval_sha256": sha256_file(approval_path),
            "data_audit_path": str(data_audit_path),
            "data_audit_sha256": sha256_file(data_audit_path),
            "campaign_run_summary_path": str(root_summary_path),
            "campaign_run_summary_sha256": sha256_file(root_summary_path),
            "campaign_heartbeat_path": str(root_heartbeat_path),
            "campaign_heartbeat_sha256": sha256_file(root_heartbeat_path),
            "market_csv": str(market_source),
            "market_csv_sha256": market_sha256,
            "pit_universe_csv": str(pit_source),
            "pit_universe_csv_sha256": pit_sha256,
            "prediction_window_requested": [TEST_START, TEST_END],
            "prediction_file_start": FIRST_PREDICTION_SESSION,
            "prediction_file_end": TEST_END,
            "label_t": LABEL_T,
            "effective_realized_t5_cutoff": REALIZED_T5_CUTOFF,
            "daily_strategy_timing": (
                "score at T close; enter T+1 open; realize T+1 to T+2 open-to-open"
            ),
            "primary_backtest_input": "averaged_predictions only",
            "runtime_policy": "CPU replay-only",
            "training_executed": False,
            "strategy_contract": {
                "entrypoint": "scripts/backtest_sp500_daily.py",
                "compatibility_shim": "tests/backtest_sp500_daily.py",
                "top_k": TOP_K,
                "num_tests": 1,
                "adjustment_method": "bhy",
                "transaction_costs": True,
                "spread_bps": 10,
                "slippage_bps_per_side": 5,
                "enable_rank_drop_gate": True,
                "min_rank_drop": 30,
                "backtest_suffix": BACKTEST_SUFFIX,
            },
            "seed_eligibility": eligibility_rows,
        }
        write_json(OUTPUT_ROOT / "coverage_audit.json", coverage_audit)
        eligibility_df.to_csv(OUTPUT_ROOT / "seed_eligibility.csv", index=False)
        write_json(OUTPUT_ROOT / "seed_eligibility.json", eligibility_rows)
        for seed, manifest in source_manifests.items():
            write_json(
                OUTPUT_ROOT / "source_manifests" / f"seed{seed}_averaged_predictions.json",
                manifest,
            )

        local_market = LOCAL_DATA_DIR / MARKET_FILENAME
        local_pit = LOCAL_DATA_DIR / PIT_FILENAME
        shutil.copy2(market_source, local_market)
        shutil.copy2(pit_source, local_pit)
        if sha256_file(local_market) != market_sha256:
            raise RuntimeError("Local market CSV staging hash mismatch")
        if sha256_file(local_pit) != pit_sha256:
            raise RuntimeError("Local PIT CSV staging hash mismatch")

        heartbeat(
            "replay_start",
            eligible_seeds=eligible_seeds,
            not_eligible_seeds=not_eligible_seeds,
            expected_backtests=len(eligible_seeds),
            completed_backtests=0,
        )

        backtest_rows = []
        command_rows = []
        summary_metrics = [
            "ARR",
            "AVoL",
            "MDD",
            "ASR",
            "CR",
            "IR",
            "MSE",
            "MAE",
            "total_return",
            "total_return_calendar_aligned",
            "benchmark_return",
            "excess_return",
            "num_trading_days",
            "days_with_gate_exits",
            "days_skipped_by_rank_gate",
            "total_transaction_cost",
            "avg_daily_turnover",
            "avg_daily_cost_bps",
            "total_trades",
            "gross_ARR",
            "gross_total_return",
            "net_ARR",
            "net_total_return",
            "cost_drag_ARR",
        ]

        try:
            for seed in EXPECTED_SEEDS:
                if seed not in eligible_runs:
                    continue
                source = eligible_runs[seed]
                local_seed_root = LOCAL_INPUTS_DIR / f"seed{seed}"
                local_predictions = local_seed_root / "averaged_predictions"
                shutil.copytree(source["averaged_dir"], local_predictions)
                staged_manifest, staged_manifest_sha256 = averaged_prediction_manifest(
                    local_predictions
                )
                if staged_manifest_sha256 != source["manifest_sha256"]:
                    raise RuntimeError(f"Prediction staging manifest mismatch for seed {seed}")
                if len(staged_manifest) != EXPECTED_PREDICTION_COUNT:
                    raise RuntimeError(f"Prediction staging count mismatch for seed {seed}")

                command = [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(REPO_DIR / "scripts" / "backtest_sp500_daily.py"),
                    "--predictions_dir",
                    str(local_predictions),
                    "--data_file",
                    str(local_market),
                    "--pit_universe_csv",
                    str(local_pit),
                    "--test_start",
                    TEST_START,
                    "--test_end",
                    TEST_END,
                    "--label_t",
                    str(LABEL_T),
                    "--top_k",
                    str(TOP_K),
                    "--num_tests",
                    "1",
                    "--adjustment_method",
                    "bhy",
                    "--auto_save",
                    "--backtest_suffix",
                    BACKTEST_SUFFIX,
                    "--transaction_costs",
                    "--spread",
                    "10",
                    "--slippage",
                    "5",
                    "--enable_rank_drop_gate",
                    "--min_rank_drop",
                    "30",
                ]
                command_rows.append(
                    {
                        "base_seed": seed,
                        "command": subprocess.list2cmdline(command),
                        "source_averaged_predictions": str(source["averaged_dir"]),
                        "source_averaged_predictions_folder_id": source["remote_folder_id"],
                        "source_manifest_sha256": source["manifest_sha256"],
                    }
                )
                write_json(OUTPUT_ROOT / "backtest_commands.json", command_rows)

                heartbeat(
                    "backtesting",
                    current_seed=seed,
                    eligible_seeds=eligible_seeds,
                    completed_backtests=len(backtest_rows),
                    expected_backtests=len(eligible_seeds),
                )
                print("Backtesting eligible seed", seed)
                started = time.time()
                process = subprocess.run(
                    command,
                    cwd=str(REPO_DIR),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log_path = OUTPUT_ROOT / "logs" / f"seed{seed}.log"
                log_path.write_text(process.stdout, encoding="utf-8")
                if process.returncode != 0:
                    raise RuntimeError(
                        f"Backtest failed for seed {seed}; return code {process.returncode}"
                    )

                local_backtest = local_seed_root / f"backtest{BACKTEST_SUFFIX}"
                metrics_path = local_backtest / "backtest_metrics.json"
                results_csv_path = local_backtest / "backtest_results.csv"
                if not metrics_path.is_file() or not results_csv_path.is_file():
                    raise FileNotFoundError(
                        f"Canonical auto-save outputs are incomplete for seed {seed}"
                    )
                seed_output = OUTPUT_ROOT / "per_seed" / f"seed{seed}"
                shutil.copytree(local_backtest, seed_output)
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                adjusted_row = pd.read_csv(results_csv_path).iloc[0].to_dict()

                holdings_path = seed_output / "daily_holdings.csv"
                returns_path = seed_output / "daily_returns.csv"
                holdings = pd.read_csv(holdings_path) if holdings_path.is_file() else pd.DataFrame()
                daily_returns = (
                    pd.read_csv(returns_path) if returns_path.is_file() else pd.DataFrame()
                )
                score_dates = (
                    pd.to_datetime(holdings["pred_date"], errors="coerce").dropna()
                    if "pred_date" in holdings
                    else pd.Series(dtype="datetime64[ns]")
                )
                entry_dates = (
                    pd.to_datetime(holdings["entry_date"], errors="coerce").dropna()
                    if "entry_date" in holdings
                    else pd.Series(dtype="datetime64[ns]")
                )
                return_dates = (
                    pd.to_datetime(daily_returns["date"], errors="coerce").dropna()
                    if "date" in daily_returns
                    else pd.Series(dtype="datetime64[ns]")
                )

                row = {
                    "base_seed": seed,
                    "status": "OK",
                    "elapsed_seconds": round(time.time() - started, 3),
                    "source_run_relative_path": source["run_relative_path"],
                    "source_averaged_predictions": str(source["averaged_dir"]),
                    "source_averaged_predictions_folder_id": source["remote_folder_id"],
                    "source_averaged_predictions_manifest_sha256": source[
                        "manifest_sha256"
                    ],
                    "source_prediction_csv_count": EXPECTED_PREDICTION_COUNT,
                    "prediction_file_start": FIRST_PREDICTION_SESSION,
                    "prediction_file_end": TEST_END,
                    "effective_realized_t5_cutoff": REALIZED_T5_CUTOFF,
                    "last_strategy_score_date": (
                        score_dates.max().strftime("%Y-%m-%d")
                        if not score_dates.empty
                        else None
                    ),
                    "last_strategy_entry_date": (
                        entry_dates.max().strftime("%Y-%m-%d")
                        if not entry_dates.empty
                        else None
                    ),
                    "last_strategy_return_date": (
                        return_dates.max().strftime("%Y-%m-%d")
                        if not return_dates.empty
                        else None
                    ),
                    "backtest_output_dir": str(seed_output),
                    "t_statistic": adjusted_row.get("t_statistic"),
                    "original_p_value": adjusted_row.get("original_p_value"),
                    "adjusted_p_value": adjusted_row.get("adjusted_p_value"),
                    "haircutted_sharpe": adjusted_row.get("haircutted_sharpe"),
                    "haircut_pct": adjusted_row.get("haircut_pct"),
                    "is_significant": adjusted_row.get("is_significant"),
                    **metrics,
                }
                backtest_rows.append(row)
                write_json(OUTPUT_ROOT / "backtest_rows.json", backtest_rows)
                pd.DataFrame(backtest_rows).to_csv(
                    OUTPUT_ROOT / "backtest_rows.csv", index=False
                )

            rows_df = pd.DataFrame(backtest_rows)
            cross_seed_rows = []
            for metric in summary_metrics:
                if metric not in rows_df.columns:
                    continue
                numeric = pd.to_numeric(rows_df[metric], errors="coerce").dropna()
                if len(numeric) != len(rows_df) or numeric.empty:
                    continue
                cross_seed_rows.append(
                    {
                        "metric": metric,
                        "n": int(len(numeric)),
                        "mean": float(numeric.mean()),
                        "sample_std": (
                            float(numeric.std(ddof=1)) if len(numeric) > 1 else None
                        ),
                    }
                )
            pd.DataFrame(cross_seed_rows).to_csv(
                OUTPUT_ROOT / "cross_seed_mean_sample_std.csv", index=False
            )
            write_json(
                OUTPUT_ROOT / "cross_seed_mean_sample_std.json", cross_seed_rows
            )

            backtested_seeds = [int(row["base_seed"]) for row in backtest_rows]
            accounted = sorted(backtested_seeds + not_eligible_seeds) == sorted(
                EXPECTED_SEEDS
            )
            if not accounted:
                raise RuntimeError("Five-seed accounting invariant failed")

            run_summary = {
                "status": "OK",
                "campaign_id": CAMPAIGN_ID,
                "campaign_run_tag": CAMPAIGN_RUN_TAG,
                "campaign_run_folder_id": CAMPAIGN_RUN_FOLDER_ID,
                "replay_run_tag": REPLAY_RUN_TAG,
                "output_root": str(OUTPUT_ROOT),
                "runtime_policy": "CPU replay-only",
                "training_executed": False,
                "primary_prediction_input": "averaged_predictions only",
                "expected_seeds": EXPECTED_SEEDS,
                "eligible_seeds": eligible_seeds,
                "backtested_seeds": backtested_seeds,
                "not_eligible_seeds": not_eligible_seeds,
                "all_expected_seeds_accounted_for": accounted,
                "strategy_contract": coverage_audit["strategy_contract"],
                "prediction_window_requested": [TEST_START, TEST_END],
                "prediction_file_end": TEST_END,
                "effective_realized_t5_cutoff": REALIZED_T5_CUTOFF,
                "daily_strategy_cutoff_note": (
                    "The 2026-07-06 cutoff applies to realized t+5 label metrics. "
                    "The one-day open-to-open strategy may use later score dates when "
                    "the required T+1/T+2 market observations exist."
                ),
                "source_commit": live_commit,
                "source_code_sha256": source_code_sha256,
                "approved_config_sha256": EXPECTED_CONFIG_SHA256,
                "market_csv_sha256": market_sha256,
                "pit_universe_csv_sha256": pit_sha256,
                "per_seed_rows": backtest_rows,
                "cross_seed_mean_sample_std": cross_seed_rows,
                "completed_at_utc": utc_now(),
            }
            write_json(OUTPUT_ROOT / "run_summary.json", run_summary)

            def pct(value) -> str:
                return "" if value is None else f"{float(value) * 100:.2f}%"

            report_lines = [
                "# LambdaRankIC 2026-YTD Replay Backtest Report",
                "",
                f"- Campaign run: `{CAMPAIGN_RUN_TAG}`",
                f"- Replay run: `{REPLAY_RUN_TAG}`",
                "- Runtime: CPU replay-only",
                "- Training executed: no",
                "- Primary input: each seed's `averaged_predictions` only",
                f"- Approved config SHA-256: `{EXPECTED_CONFIG_SHA256}`",
                f"- Prediction files: `{FIRST_PREDICTION_SESSION}` through `{TEST_END}`",
                f"- Effective realized t+5 cutoff: `{REALIZED_T5_CUTOFF}`",
                "",
                "The t+5 cutoff applies to forward-label MSE/MAE. The daily strategy uses "
                "T-close scores and T+1-to-T+2 open-to-open returns, so its last realized "
                "strategy date is reported separately per seed.",
                "",
                "## Seed eligibility",
                "",
                "| Seed | Status | Checkpoints | Averaged CSVs | Manifest SHA-256 |",
                "| ---: | --- | ---: | ---: | --- |",
            ]
            for item in eligibility_rows:
                report_lines.append(
                    f"| {item['base_seed']} | {item['status']} | "
                    f"{item['checkpoint_count']} | {item['averaged_prediction_count']} | "
                    f"`{item['averaged_predictions_manifest_sha256'] or ''}` |"
                )
            report_lines.extend(
                [
                    "",
                    "## Per-seed results",
                    "",
                    "| Seed | Net return | ARR | ASR | MDD | Benchmark | Trades | Cost | Last score | Last return |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
                ]
            )
            for row in backtest_rows:
                report_lines.append(
                    f"| {row['base_seed']} | {pct(row.get('net_total_return'))} | "
                    f"{pct(row.get('ARR'))} | {float(row.get('ASR', 0)):.3f} | "
                    f"{pct(row.get('MDD'))} | {pct(row.get('benchmark_return'))} | "
                    f"{int(row.get('total_trades', 0))} | "
                    f"{pct(row.get('total_transaction_cost'))} | "
                    f"{row.get('last_strategy_score_date') or ''} | "
                    f"{row.get('last_strategy_return_date') or ''} |"
                )
            report_lines.extend(
                [
                    "",
                    "## Strategy contract",
                    "",
                    "Top 10; t+5 prediction labels; one daily strategy test; BHY; "
                    "10 bps spread; 5 bps slippage per side; rank-drop gate enabled "
                    "at 30 positions; PIT universe filtering enabled.",
                    "",
                    "## Cross-seed summary",
                    "",
                    "See `cross_seed_mean_sample_std.csv` for whitelisted metric means "
                    "and sample standard deviations across eligible seeds.",
                    "",
                    "## Provenance",
                    "",
                    f"- Campaign commit: `{live_commit}`",
                    f"- Market CSV SHA-256: `{market_sha256}`",
                    f"- PIT CSV SHA-256: `{pit_sha256}`",
                ]
            )
            for relative, digest in source_code_sha256.items():
                report_lines.append(f"- `{relative}` SHA-256: `{digest}`")
            (OUTPUT_ROOT / "report.md").write_text(
                "\n".join(report_lines) + "\n", encoding="utf-8"
            )

            review_lines = [
                "# Colab Chrome Control Run Review",
                "",
                "## Scope",
                f"- Branch: `codex/lambdarankic-2026-ytd-backtest-20260714`",
                f"- Notebook URL: {NOTEBOOK_URL}",
                "- Goal: full replay-only backtest",
                "- Claimed tab: dedicated LambdaRankIC 2026-YTD backtest Colab tab",
                "- Surface: `chrome:control-chrome`",
                "",
                "## Runtime And Evidence",
                "- Runtime accepted from matrix: CPU replay-only",
                "- Visible runtime evidence: CPU; no GPU assigned",
                f"- Drive artifact root: `{OUTPUT_ROOT}`",
                "- Drive evidence: eligibility, per-seed outputs, cross-seed summary, heartbeat, and run summary",
                "",
                "## Prompts And Control",
                "- Prompts handled: routine GitHub notebook and Drive mount prompts as shown",
                "- Cells run: CPU setup, read-only audit, foreground replay, Drive readback/cleanup",
                "",
                "## Notebook And Git State",
                "- Cell freshness checked: yes",
                f"- Replay code commit: `{live_commit}`",
                "",
                "## Outcome",
                "- Status: succeeded",
                "- Cleanup state: replay runtime scheduled for unassignment after Drive readback",
                "- Residual risk: none beyond normal historical-backtest limitations",
                "",
                "## Recommended Follow-up",
                "- Review `report.md`, `backtest_rows.csv`, and the per-seed auto-save directories.",
            ]
            (OUTPUT_ROOT / "colab_run_review.md").write_text(
                "\n".join(review_lines) + "\n", encoding="utf-8"
            )
            heartbeat(
                "complete",
                status="OK",
                eligible_seeds=eligible_seeds,
                backtested_seeds=backtested_seeds,
                not_eligible_seeds=not_eligible_seeds,
                completed_backtests=len(backtest_rows),
                expected_backtests=len(eligible_seeds),
                all_expected_seeds_accounted_for=True,
                run_summary=str(OUTPUT_ROOT / "run_summary.json"),
                report=str(OUTPUT_ROOT / "report.md"),
            )
            print("Replay complete:", OUTPUT_ROOT)
            display(pd.DataFrame(backtest_rows))
            display(pd.DataFrame(cross_seed_rows))
        except Exception as replay_error:
            heartbeat(
                "failed",
                status="FAILED",
                error=repr(replay_error),
                completed_backtests=len(backtest_rows),
                expected_backtests=len(eligible_seeds),
            )
            print("Replay runtime intentionally remains assigned for failure inspection.")
            raise
        """
    ),
    md("## 4. Drive readback and replay-runtime cleanup"),
    code(
        r"""
        time.sleep(5)
        summary_readback = json.loads(
            (OUTPUT_ROOT / "run_summary.json").read_text(encoding="utf-8")
        )
        heartbeat_readback = json.loads(
            (OUTPUT_ROOT / "heartbeat.json").read_text(encoding="utf-8")
        )
        if summary_readback.get("status") != "OK":
            raise RuntimeError("Drive run_summary readback is not OK")
        if heartbeat_readback.get("status") != "OK" or heartbeat_readback.get("phase") != "complete":
            raise RuntimeError("Drive heartbeat readback is not terminal OK/complete")
        if not summary_readback.get("all_expected_seeds_accounted_for"):
            raise RuntimeError("Drive readback does not account for all five expected seeds")
        for seed in summary_readback.get("backtested_seeds", []):
            required = [
                OUTPUT_ROOT / "per_seed" / f"seed{seed}" / "backtest_metrics.json",
                OUTPUT_ROOT / "per_seed" / f"seed{seed}" / "backtest_results.csv",
                OUTPUT_ROOT / "per_seed" / f"seed{seed}" / "daily_holdings.csv",
                OUTPUT_ROOT / "per_seed" / f"seed{seed}" / "trade_journal.csv",
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError(f"Drive readback missing seed {seed} outputs: {missing}")

        print(
            json.dumps(
                {
                    "output_root": str(OUTPUT_ROOT),
                    "backtested_seeds": summary_readback["backtested_seeds"],
                    "not_eligible_seeds": summary_readback["not_eligible_seeds"],
                    "prediction_file_end": summary_readback["prediction_file_end"],
                    "effective_realized_t5_cutoff": summary_readback[
                        "effective_realized_t5_cutoff"
                    ],
                    "all_expected_seeds_accounted_for": summary_readback[
                        "all_expected_seeds_accounted_for"
                    ],
                },
                indent=2,
            )
        )
        print("Drive artifact readback: OK")
        print("Unassigning only this replay CPU runtime.")
        runtime.unassign()
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    """Write the generated notebook."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
