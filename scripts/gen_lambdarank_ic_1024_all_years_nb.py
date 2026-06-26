"""Generate the Colab notebook for LambdaRankIC 1024 all-year PIT validation."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

OUT = Path("notebooks/lambdarank_ic_1024_all_years_colab.ipynb")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


cells = [
    md(
        """
        # LambdaRankIC 1024 All-Year PIT Validation

        Full-preset continuation of the completed 2022 pair-cap tranche. This
        notebook trains `lambdarank_ic` with `max_pairs_per_day=1024` for every
        configured PIT test year, then replays saved averaged predictions in:

        - `no_cost_no_gate`
        - `cost_rank_gate` with 10 bps spread, 5 bps slippage, and
          `min_rank_drop=30`

        Use a visible Colab G4/L4-class or better GPU runtime for training.
        Drive artifacts are the durable run truth.
        """
    ),
    md("## 1. Setup, GPU Gate, And Data"),
    code(
        r"""
        import csv
        import json
        import os
        import shutil
        import subprocess
        import sys
        import time
        from datetime import datetime
        from pathlib import Path

        import pandas as pd
        import torch

        IN_COLAB = "google.colab" in sys.modules
        REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
        BRANCH = "codex/colab-gpu-utilization-hardening-20260620"
        REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()
        REQUIRE_G4_L4_GPU = True
        BLOCKED_GPU_NAMES = ("T4",)
        ALLOWED_GPU_MARKERS = ("G4", "L4", "A100", "H100", "V100", "RTX PRO", "BLACKWELL")

        def detect_gpu_name() -> str:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "nvidia-smi failed. Expected G4/L4-class Colab runtime, not T4/CPU.\n"
                    + proc.stderr
                )
            gpu_name = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
            if not gpu_name:
                raise RuntimeError("nvidia-smi did not report a GPU name.")
            upper_gpu = gpu_name.upper()
            if any(blocked in upper_gpu for blocked in BLOCKED_GPU_NAMES):
                raise RuntimeError(f"Refusing blocked runtime GPU: {gpu_name}")
            if not any(marker in upper_gpu for marker in ALLOWED_GPU_MARKERS):
                raise RuntimeError(
                    f"Refusing runtime GPU {gpu_name}; allowed markers are {ALLOWED_GPU_MARKERS}."
                )
            return gpu_name

        if IN_COLAB:
            from google.colab import drive

            drive.mount("/content/drive")
            if not REPO_DIR.exists():
                subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)], check=True)
            else:
                subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "-B", BRANCH, f"origin/{BRANCH}"], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO_DIR}[dev,tracking,fred]"], check=True)

        os.chdir(REPO_DIR)
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))

        print("Repo:", REPO_DIR)
        print("Branch:", BRANCH)
        subprocess.run(["git", "rev-parse", "HEAD"], check=False)
        print("Python:", sys.executable)
        print("Torch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            GPU_NAME = detect_gpu_name()
            print("GPU:", GPU_NAME)
        elif REQUIRE_G4_L4_GPU:
            raise RuntimeError("Expected G4/L4-class Colab runtime, not T4/CPU.")

        if IN_COLAB and not os.environ.get("FRED_API_KEY"):
            try:
                from google.colab import userdata

                secret = userdata.get("FRED_API_KEY")
                if secret:
                    os.environ["FRED_API_KEY"] = secret
                    print("FRED_API_KEY loaded from Colab Secrets.")
            except Exception as exc:
                print("Could not read FRED_API_KEY from Colab Secrets:", exc)
        if not os.environ.get("FRED_API_KEY"):
            raise RuntimeError("FRED_API_KEY is required for the regime-enabled preset.")

        drive_data_dir = Path("/content/drive/MyDrive/MCI_GRU_shared/data") if IN_COLAB else REPO_DIR / "data/raw/market"
        drive_market_csv = drive_data_dir / "sp500_pit_union_lseg_20150101_20260513.csv"
        drive_pit_csv = drive_data_dir / "sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv"
        drive_market_meta = drive_data_dir / "sp500_pit_union_lseg_20150101_20260513.meta.json"

        if not drive_market_csv.exists():
            raise FileNotFoundError(f"Missing market CSV: {drive_market_csv}")
        if not drive_pit_csv.exists():
            raise FileNotFoundError(f"Missing PIT universe CSV: {drive_pit_csv}")

        repo_market_csv = REPO_DIR / "data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv"
        repo_pit_csv = REPO_DIR / "data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv"
        repo_market_csv.parent.mkdir(parents=True, exist_ok=True)
        repo_pit_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_market_csv, repo_market_csv)
        shutil.copy2(drive_pit_csv, repo_pit_csv)

        market_meta = {}
        if drive_market_meta.exists():
            market_meta = json.loads(drive_market_meta.read_text(encoding="utf-8"))
            print("Market date range:", market_meta.get("date_min"), "to", market_meta.get("date_max"))
            if str(market_meta.get("date_max")) < "2025-12-31":
                raise RuntimeError(f"Market data does not cover 2025-12-31: {market_meta.get('date_max')}")

        from mci_gru.config import TrainingConfig
        from mci_gru.training.losses import build_training_loss

        probe_cfg = TrainingConfig(
            loss_type="lambdarank_ic",
            selection_metric="val_rank_ic",
            lambdarank_ic_max_pairs_per_day=1024,
        )
        probe_loss, probe_name = build_training_loss(probe_cfg)
        print("LambdaRankIC branch probe:", probe_name, type(probe_loss).__name__)
        """
    ),
    md("## 2. Build All-Year Job Matrix"),
    code(
        r"""
        YEARS = [2022, 2023, 2024, 2025]
        BASE_SEED = 314159
        MAX_PAIRS_PER_DAY = 1024
        NUM_MODELS = 20
        NUM_EPOCHS = 100
        EARLY_STOPPING_PATIENCE = 15
        TOP_K = 10
        LABEL_T = 5
        NUM_TESTS = 4
        ADJUSTMENT_METHOD = "bhy"

        PIT_WINDOWS = {
            2022: {"test_start": "2022-01-22", "test_end": "2022-12-31"},
            2023: {"test_start": "2023-01-22", "test_end": "2023-12-31"},
            2024: {"test_start": "2024-01-22", "test_end": "2024-12-31"},
            2025: {"test_start": "2025-01-22", "test_end": "2025-12-31"},
        }

        RUN_TAG = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        RUN_ROOT = (
            Path("/content/drive/MyDrive/MCI-GRU-Ablations/lambdarank_ic_1024_all_years")
            if IN_COLAB
            else REPO_DIR / "results" / "lambdarank_ic_1024_all_years"
        ) / RUN_TAG
        TRAINING_OUTPUT_DIR = RUN_ROOT / "training"
        BACKTEST_OUTPUT_DIR = RUN_ROOT / "backtests"
        SUMMARY_DIR = RUN_ROOT / "summaries"
        HEARTBEAT_PATH = RUN_ROOT / "heartbeat.json"
        GPU_UTIL_PATH = RUN_ROOT / "gpu_util.csv"
        GPU_UTIL_STOP_PATH = RUN_ROOT / "gpu_util.stop"
        TRAINING_RESULTS_JSON = RUN_ROOT / "training_results.json"
        TRAINING_RESULTS_CSV = RUN_ROOT / "training_results.csv"
        BACKTEST_RESULTS_JSON = RUN_ROOT / "backtest_results.json"
        BACKTEST_RESULTS_CSV = RUN_ROOT / "backtest_results.csv"
        ALL_RESULTS_JSON = RUN_ROOT / "all_years_results.json"
        ALL_RESULTS_CSV = RUN_ROOT / "all_years_results.csv"
        MANIFEST_PATH = RUN_ROOT / "lambdarank_ic_1024_all_years_manifest.json"

        if RUN_ROOT.exists():
            raise RuntimeError(f"Refusing to reuse existing run root: {RUN_ROOT}")
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

        def write_heartbeat(
            phase: str,
            status: str = "RUNNING",
            current_job: str | None = None,
            completed_training_jobs: int = 0,
            completed_backtests: int = 0,
            error: str | None = None,
        ) -> None:
            payload = {
                "phase": phase,
                "status": status,
                "current_job": current_job,
                "completed_training_jobs": completed_training_jobs,
                "expected_training_jobs": len(YEARS),
                "completed_backtests": completed_backtests,
                "expected_backtests": len(YEARS) * 2,
                "branch": BRANCH,
                "run_root": str(RUN_ROOT),
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            if "GPU_NAME" in globals():
                payload["gpu_name"] = GPU_NAME
            if torch.cuda.is_available():
                payload["gpu"] = torch.cuda.get_device_name(0)
            if error is not None:
                payload["error"] = error
            HEARTBEAT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        def start_gpu_sampler() -> subprocess.Popen | None:
            if GPU_UTIL_STOP_PATH.exists():
                GPU_UTIL_STOP_PATH.unlink()
            monitor_script = REPO_DIR / "scripts/monitor_gpu_util.py"
            if not monitor_script.exists():
                raise FileNotFoundError(f"Missing GPU monitor: {monitor_script}")
            return subprocess.Popen(
                [
                    sys.executable,
                    str(monitor_script),
                    "--output",
                    str(GPU_UTIL_PATH),
                    "--interval",
                    "1",
                    "--stop-file",
                    str(GPU_UTIL_STOP_PATH),
                ],
                cwd=str(REPO_DIR),
            )

        def stop_gpu_sampler(proc: subprocess.Popen | None) -> None:
            if proc is None:
                return
            GPU_UTIL_STOP_PATH.write_text("stop", encoding="utf-8")
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)

        FROZEN_RECIPE_ID = (
            "static-threshold-shuffle__pure-ic-returns-5d-val-ic__"
            "regime-current-only__ensemble__drop-edge-0p1"
        )

        BASE_OVERRIDES = [
            "data.source=csv",
            "features=with_momentum",
            "features.include_momentum=true",
            "features.include_weekly_momentum=true",
            "features.momentum_encoding=binary",
            "features.momentum_blend_mode=static",
            "features.momentum_blend_fast_weight=0.5",
            "features.include_global_regime=true",
            "features.regime_strict=true",
            "features.regime_enforce_lag_days=0",
            "features.regime_include_subsequent_returns=false",
            "features.regime_change_months=12",
            "features.regime_norm_months=120",
            "features.regime_exclusion_months=1",
            "features.regime_similarity_quantile=0.2",
            "features.regime_min_history_months=24",
            "graph.judge_value=0.8",
            "graph.update_frequency_months=0",
            "graph.corr_lookback_days=252",
            "graph.top_k=0",
            "graph.top_k_metric=corr",
            "graph.use_multi_feature_edges=true",
            "graph.append_snapshot_age_days=false",
            "graph.use_lead_lag_features=false",
            "graph.drop_edge_p=0.1",
            "training.lr_scheduler=cosine",
            "training.learning_rate=5e-5",
            f"training.num_epochs={NUM_EPOCHS}",
            f"training.num_models={NUM_MODELS}",
            f"training.early_stopping_patience={EARLY_STOPPING_PATIENCE}",
            "training.label_type=returns",
            "training.shuffle_train=true",
            "training.dataloader_num_workers=2",
            "training.dataloader_pin_memory=true",
            "training.dataloader_persistent_workers=true",
            "training.dataloader_prefetch_factor=4",
            "training.profile_batches=0",
            "model.label_t=5",
            "model.temporal_encoder=gru_attn",
            "tracking.enabled=false",
            "tracking.log_artifacts=false",
            "tracking.log_checkpoints=false",
            "tracking.log_predictions=false",
            f"data.filename={repo_market_csv.relative_to(REPO_DIR).as_posix()}",
            f"data.pit_universe_csv={repo_pit_csv.relative_to(REPO_DIR).as_posix()}",
            "data.use_pit_universe=true",
            "data.pit_universe_mode=masked_panel",
            "data.pit_min_scoreable_stocks=450",
            "data.pit_breadth_policy=error",
            "training.loss_type=lambdarank_ic",
            "training.selection_metric=val_rank_ic",
            f"training.lambdarank_ic_max_pairs_per_day={MAX_PAIRS_PER_DAY}",
            "training.lambdarank_ic_temperature=1.0",
            f"seed={BASE_SEED}",
        ]

        training_jobs = []
        for year in YEARS:
            name = f"lambdarank_ic_pairs1024_{year}_seed{BASE_SEED}"
            training_jobs.append(
                {
                    "year": year,
                    "name": name,
                    "max_pairs_per_day": MAX_PAIRS_PER_DAY,
                    "base_seed": BASE_SEED,
                    "overrides": [
                        f"+experiment=pit_temporal_{year}",
                        *BASE_OVERRIDES,
                        f"experiment_name={name}",
                        f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                    ],
                }
            )

        backtest_scenarios = [
            {
                "name": "no_cost_no_gate",
                "suffix": "_pit_daily_no_cost_no_gate",
                "transaction_costs": False,
                "rank_gate": False,
            },
            {
                "name": "cost_rank_gate",
                "suffix": "_pit_daily_tc_rank_gate",
                "transaction_costs": True,
                "rank_gate": True,
                "spread_bps": 10,
                "slippage_bps": 5,
                "min_rank_drop": 30,
            },
        ]

        manifest = {
            "scope": "LambdaRankIC max_pairs_per_day=1024 all configured PIT test years",
            "recipe_id": FROZEN_RECIPE_ID,
            "branch": BRANCH,
            "run_tag": RUN_TAG,
            "run_root": str(RUN_ROOT),
            "years": YEARS,
            "pit_windows": PIT_WINDOWS,
            "base_seed": BASE_SEED,
            "max_pairs_per_day": MAX_PAIRS_PER_DAY,
            "num_models": NUM_MODELS,
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "selection_metric": "val_rank_ic",
            "label_t": LABEL_T,
            "top_k": TOP_K,
            "num_tests": NUM_TESTS,
            "adjustment_method": ADJUSTMENT_METHOD,
            "market_csv": str(repo_market_csv),
            "pit_universe_csv": str(repo_pit_csv),
            "market_meta": market_meta,
            "training_jobs": training_jobs,
            "backtest_scenarios": backtest_scenarios,
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        write_heartbeat("manifest")

        print("Run root:", RUN_ROOT)
        print("Manifest:", MANIFEST_PATH)
        print("Training jobs:", len(training_jobs))
        display(pd.DataFrame(training_jobs)[["year", "name", "max_pairs_per_day", "base_seed"]])
        """
    ),
    md("## 3. Run Training And Saved-Prediction Backtests"),
    code(
        r"""
        def latest_training_summary(job_name: str) -> Path | None:
            candidates = sorted(TRAINING_OUTPUT_DIR.glob(f"{job_name}/**/training_summary.json"))
            return candidates[-1] if candidates else None

        def latest_predictions_dir(job_name: str) -> Path:
            candidates = sorted(TRAINING_OUTPUT_DIR.glob(f"{job_name}/**/averaged_predictions"))
            if not candidates:
                raise FileNotFoundError(f"No averaged_predictions found for {job_name}")
            return candidates[-1]

        def write_table(path: Path, rows: list[dict]) -> None:
            if rows:
                pd.json_normalize(rows).to_csv(path, index=False)
            else:
                path.write_text("", encoding="utf-8")

        def training_command(job: dict) -> list[str]:
            return [sys.executable, "-u", str(REPO_DIR / "run_experiment.py"), *job["overrides"]]

        def backtest_command(job: dict, predictions_dir: Path, scenario: dict) -> list[str]:
            window = PIT_WINDOWS[job["year"]]
            cmd = [
                sys.executable,
                "-X",
                "utf8",
                str(REPO_DIR / "tests/backtest_sp500_daily.py"),
                "--predictions_dir",
                str(predictions_dir),
                "--data_file",
                str(repo_market_csv),
                "--pit_universe_csv",
                str(repo_pit_csv),
                "--test_start",
                window["test_start"],
                "--test_end",
                window["test_end"],
                "--top_k",
                str(TOP_K),
                "--label_t",
                str(LABEL_T),
                "--num_tests",
                str(NUM_TESTS),
                "--adjustment_method",
                ADJUSTMENT_METHOD,
                "--auto_save",
                "--backtest_suffix",
                scenario["suffix"],
            ]
            if scenario.get("transaction_costs"):
                cmd.extend(
                    [
                        "--transaction_costs",
                        "--spread",
                        str(scenario["spread_bps"]),
                        "--slippage",
                        str(scenario["slippage_bps"]),
                    ]
                )
            if scenario.get("rank_gate"):
                cmd.extend(["--enable_rank_drop_gate", "--min_rank_drop", str(scenario["min_rank_drop"])])
            return cmd

        def collect_backtest_result(job: dict, predictions_dir: Path, scenario: dict) -> dict:
            backtest_dir = predictions_dir.parent / f"backtest{scenario['suffix']}"
            result_csv = backtest_dir / "backtest_results.csv"
            row = {
                "year": job["year"],
                "training_job": job["name"],
                "scenario": scenario["name"],
                "predictions_dir": str(predictions_dir),
                "backtest_dir": str(backtest_dir),
                "test_start": PIT_WINDOWS[job["year"]]["test_start"],
                "test_end": PIT_WINDOWS[job["year"]]["test_end"],
            }
            if result_csv.exists():
                result_df = pd.read_csv(result_csv)
                if len(result_df):
                    row.update({f"backtest.{key}": value for key, value in result_df.iloc[0].to_dict().items()})
            return row

        training_rows = []
        backtest_rows = []
        gpu_sampler_proc = start_gpu_sampler()
        try:
            for job in training_jobs:
                write_heartbeat(
                    "training",
                    current_job=job["name"],
                    completed_training_jobs=len(training_rows),
                    completed_backtests=len(backtest_rows),
                )
                print("=" * 100)
                print("Training:", job["name"])
                cmd = training_command(job)
                print("Command:", " ".join(cmd[:4]), "... +", len(job["overrides"]), "overrides")
                start_time = time.perf_counter()
                proc = subprocess.run(
                    cmd,
                    cwd=str(REPO_DIR),
                    text=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    check=False,
                )
                elapsed_seconds = time.perf_counter() - start_time
                summary_path = latest_training_summary(job["name"])
                row = {
                    "year": job["year"],
                    "name": job["name"],
                    "base_seed": job["base_seed"],
                    "max_pairs_per_day": job["max_pairs_per_day"],
                    "returncode": int(proc.returncode),
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "training_summary_path": str(summary_path) if summary_path else None,
                }
                if summary_path is not None:
                    row["training_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
                training_rows.append(row)
                TRAINING_RESULTS_JSON.write_text(json.dumps(training_rows, indent=2), encoding="utf-8")
                write_table(TRAINING_RESULTS_CSV, training_rows)
                if proc.returncode != 0:
                    raise RuntimeError(f"Training job failed: {job['name']}")

                predictions_dir = latest_predictions_dir(job["name"])
                for scenario in backtest_scenarios:
                    current = f"{job['name']}::{scenario['name']}"
                    write_heartbeat(
                        "backtest",
                        current_job=current,
                        completed_training_jobs=len(training_rows),
                        completed_backtests=len(backtest_rows),
                    )
                    print("Backtest:", current)
                    cmd = backtest_command(job, predictions_dir, scenario)
                    print("Command:", subprocess.list2cmdline(cmd))
                    start_time = time.perf_counter()
                    proc = subprocess.run(
                        cmd,
                        cwd=str(REPO_DIR),
                        text=True,
                        env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"},
                        check=False,
                    )
                    elapsed_seconds = time.perf_counter() - start_time
                    row = collect_backtest_result(job, predictions_dir, scenario)
                    row["returncode"] = int(proc.returncode)
                    row["elapsed_seconds"] = round(elapsed_seconds, 3)
                    backtest_rows.append(row)
                    BACKTEST_RESULTS_JSON.write_text(json.dumps(backtest_rows, indent=2), encoding="utf-8")
                    write_table(BACKTEST_RESULTS_CSV, backtest_rows)
                    if proc.returncode != 0:
                        raise RuntimeError(f"Backtest failed: {current}")

            all_rows = []
            by_year_training = {row["year"]: row for row in training_rows}
            for row in backtest_rows:
                merged = dict(row)
                training = by_year_training.get(row["year"], {})
                for key, value in pd.json_normalize(training).iloc[0].to_dict().items():
                    merged[f"training.{key}"] = value
                all_rows.append(merged)
            ALL_RESULTS_JSON.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
            write_table(ALL_RESULTS_CSV, all_rows)
            write_heartbeat(
                "done",
                status="OK",
                completed_training_jobs=len(training_rows),
                completed_backtests=len(backtest_rows),
            )
        except Exception as exc:
            write_heartbeat(
                "failed",
                status="FAILED",
                current_job=job["name"] if "job" in locals() else None,
                completed_training_jobs=len(training_rows),
                completed_backtests=len(backtest_rows),
                error=str(exc),
            )
            raise
        finally:
            stop_gpu_sampler(gpu_sampler_proc)
            if IN_COLAB:
                try:
                    import google.colab.runtime

                    google.colab.runtime.unassign()
                    print("Released Colab runtime with google.colab.runtime.unassign().")
                except Exception as exc:
                    print("Manual Runtime > Disconnect and delete runtime may be needed:", exc)

        print("Training results:", TRAINING_RESULTS_JSON)
        print("Backtest results:", BACKTEST_RESULTS_JSON)
        print("All-year results:", ALL_RESULTS_CSV)
        """
    ),
    md("## 4. Results Snapshot"),
    code(
        r"""
        if ALL_RESULTS_CSV.exists():
            df = pd.read_csv(ALL_RESULTS_CSV)
            cols = [
                "year",
                "scenario",
                "training.training_summary.mean_best_val_ic",
                "training.training_summary.mean_best_val_rank_ic",
                "backtest.total_return",
                "backtest.ARR",
                "backtest.ASR",
                "backtest.MDD",
                "backtest.benchmark_return",
                "backtest.excess_return",
                "backtest.adjusted_p_value",
                "backtest.total_trades",
                "backtest.avg_daily_turnover",
                "backtest.total_transaction_cost",
            ]
            display(df[[col for col in cols if col in df.columns]])
        else:
            print("No all-year results file yet.")

        print("Run root:", RUN_ROOT)
        print("Heartbeat:", HEARTBEAT_PATH)
        print("Manifest:", MANIFEST_PATH)
        """
    ),
]


def build_notebook(notebook_cells: list[dict]) -> dict:
    return {
        "cells": notebook_cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build_notebook(cells), indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
