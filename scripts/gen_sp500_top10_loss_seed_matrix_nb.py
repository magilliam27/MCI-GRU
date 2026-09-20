"""Generate a Colab launcher for the 110-name loss/seed comparison matrix."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT = Path("notebooks/sp500_top10_loss_seed_matrix_colab.ipynb")


def md(source: str) -> dict:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


def code(source: str) -> dict:
    source = textwrap.dedent(source).strip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


cells = [
    md(
        """
        # S&P 500 PIT GICS Top-10 Loss/Seed Matrix

        Practical-minimum 110-name comparison for pure IC, LambdaRankIC, and
        Portfolio-IC weight50 across test years 2022-2025.

        The runner does not retrain year/loss/base-seed rows already counted as
        completed evidence. Existing rows are recorded as reused; missing rows
        are trained and backtested with transaction costs and the rank-drop
        gate. Relaunching the notebook resumes from Drive artifacts.
        """
    ),
    code(
        r"""
        from __future__ import annotations

        import csv
        import json
        import os
        import shutil
        import subprocess
        import sys
        import time
        from datetime import datetime, timezone
        from pathlib import Path

        import pandas as pd
        import torch

        try:
            from google.colab import drive, runtime, userdata
            IN_COLAB = True
        except Exception:
            drive = runtime = userdata = None
            IN_COLAB = False

        REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
        BRANCH = "main"
        RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        RUN_FAMILY = "sp500_gics_top10_loss_comparison_repeated_seeds"
        RECIPE = "static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1"

        YEARS = [2022, 2023, 2024, 2025]
        ADDITIONAL_SEEDS = [271828, 161803]
        PORTFOLIO_SEEDS = [314159, 271828, 161803]
        NUM_MODELS = 20
        NUM_EPOCHS = 100
        EARLY_STOPPING_PATIENCE = 15
        PIT_MIN_SCOREABLE_STOCKS = 100
        LAMBDARANK_PAIR_CAP = 8192
        BACKTEST_SUFFIX = "_top10_tc_rankdrop"

        MARKET_FILENAME = (
            "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_"
            "lseg_20150101_20260622.csv"
        )
        PIT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv"
        SNAPSHOT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_snapshots.csv"

        PIT_WINDOWS = {
            2022: {
                "train_start": "2016-01-01",
                "train_end": "2020-12-31",
                "val_start": "2021-01-08",
                "val_end": "2021-12-31",
                "test_start": "2022-01-08",
                "test_end": "2022-12-31",
            },
            2023: {
                "train_start": "2017-01-01",
                "train_end": "2021-12-31",
                "val_start": "2022-01-08",
                "val_end": "2022-12-31",
                "test_start": "2023-01-08",
                "test_end": "2023-12-31",
            },
            2024: {
                "train_start": "2018-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-08",
                "val_end": "2023-12-31",
                "test_start": "2024-01-08",
                "test_end": "2024-12-31",
            },
            2025: {
                "train_start": "2019-01-01",
                "train_end": "2023-12-31",
                "val_start": "2024-01-08",
                "val_end": "2024-12-31",
                "test_start": "2025-01-08",
                "test_end": "2025-12-31",
            },
        }

        KNOWN_EXISTING_ROWS = [
            {
                "loss_key": "pure_ic",
                "loss_type": "ic",
                "year": year,
                "base_seed": 1729,
                "source": "sp500_gics_top10_baseline_multiyear/20260623_011810",
                "needs_backtest_replay": True,
            }
            for year in [2022, 2023, 2024]
        ] + [
            {
                "loss_key": "pure_ic",
                "loss_type": "ic",
                "year": 2025,
                "base_seed": 1729,
                "source": "sp500_gics_top10_baseline/20260622_043728",
                "needs_backtest_replay": True,
                "note": "Older 2018-start data family; counted per no-repeat instruction.",
            }
        ] + [
            {
                "loss_key": "lambdarank_ic",
                "loss_type": "lambdarank_ic",
                "year": year,
                "base_seed": 314159,
                "source": (
                    "sp500_gics_top10_lambdarank_ic_full/20260626_172316"
                    if year < 2025
                    else "sp500_gics_top10_lambdarank_ic_2025/20260627_012647"
                ),
                "needs_backtest_replay": False,
            }
            for year in YEARS
        ]

        LOSS_VARIANTS = {
            "pure_ic": {
                "loss_type": "ic",
                "selection_metric": "val_ic",
                "seeds_to_train": ADDITIONAL_SEEDS,
            },
            "lambdarank_ic": {
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "seeds_to_train": ADDITIONAL_SEEDS,
                "lambdarank_ic_max_pairs_per_day": LAMBDARANK_PAIR_CAP,
                "lambdarank_ic_temperature": 1.0,
            },
            "portfolio_ic_weight50": {
                "loss_type": "portfolio_ic",
                "selection_metric": "val_loss",
                "seeds_to_train": PORTFOLIO_SEEDS,
                "portfolio_ic_top_k": 10,
                "portfolio_ic_weight": 0.50,
                "portfolio_ic_temperature": 0.25,
            },
        }

        if IN_COLAB:
            drive.mount("/content/drive")

        REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()
        DRIVE_ROOT = Path("/content/drive/MyDrive") if IN_COLAB else Path.cwd() / "drive_outputs"
        DRIVE_DATA_DIR = DRIVE_ROOT / "MCI_GRU_shared" / "data"
        RUN_ROOT = DRIVE_ROOT / "MCI-GRU-Ablations" / RUN_FAMILY / RUN_TAG
        LOCAL_RUN_ROOT = Path("/content/mci_gru_runs") / RUN_FAMILY / RUN_TAG if IN_COLAB else Path.cwd() / "results" / RUN_FAMILY / RUN_TAG
        TRAINING_ROOT = LOCAL_RUN_ROOT / "training"
        REUSED_REPLAY_ROOT = LOCAL_RUN_ROOT / "reused_prediction_replays"
        SUMMARY_DIR = RUN_ROOT / "summaries"
        LOG_DIR = RUN_ROOT / "logs"
        for path in [RUN_ROOT, LOCAL_RUN_ROOT, TRAINING_ROOT, REUSED_REPLAY_ROOT, SUMMARY_DIR, LOG_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        def utc_now() -> str:
            return datetime.now(timezone.utc).isoformat()

        def write_json(path: Path, payload: object) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        def heartbeat(status: str, phase: str, **extra) -> None:
            write_json(
                RUN_ROOT / "heartbeat.json",
                {
                    "status": status,
                    "phase": phase,
                    "updated_at_utc": utc_now(),
                    "run_tag": RUN_TAG,
                    "run_root": str(RUN_ROOT),
                    "recipe": RECIPE,
                    "years": YEARS,
                    "num_models": NUM_MODELS,
                    "num_epochs": NUM_EPOCHS,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "lambdarank_pair_cap": LAMBDARANK_PAIR_CAP,
                    **extra,
                },
            )

        def run_stream(cmd: list[str], *, cwd: Path, log_name: str, phase: str, env: dict | None = None) -> None:
            log_path = LOG_DIR / log_name
            heartbeat("RUNNING", phase, command=" ".join(cmd), log=str(log_path))
            with log_path.open("w", encoding="utf-8") as handle:
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                )
                assert proc.stdout is not None
                last_beat = 0.0
                for line in proc.stdout:
                    print(line, end="")
                    handle.write(line)
                    if time.time() - last_beat > 60:
                        heartbeat("RUNNING", phase, last_line=line.strip(), log=str(log_path))
                        last_beat = time.time()
                returncode = proc.wait()
            if returncode != 0:
                raise RuntimeError(f"{phase} failed with return code {returncode}; see {log_path}")

        def detect_gpu_name() -> str:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError("GPU is required for this matrix; nvidia-smi failed:\n" + proc.stderr)
            gpu_name = proc.stdout.strip().splitlines()[0].strip()
            if not gpu_name:
                raise RuntimeError("GPU is required for this matrix; no GPU name reported.")
            upper = gpu_name.upper()
            if "T4" in upper:
                raise RuntimeError(f"T4 runtime detected ({gpu_name}); switch to L4/G4-class or better.")
            allowed = ("L4", "G4", "A100", "H100", "V100", "RTX", "BLACKWELL")
            if not any(marker in upper for marker in allowed):
                raise RuntimeError(f"Unexpected GPU {gpu_name}; expected L4/G4-class or better.")
            return gpu_name

        heartbeat("RUNNING", "setup")
        if IN_COLAB:
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

        gpu_name = detect_gpu_name() if IN_COLAB else "local"
        if IN_COLAB and userdata is not None and not os.environ.get("FRED_API_KEY"):
            try:
                fred_key = userdata.get("FRED_API_KEY")
            except Exception:
                fred_key = None
            if fred_key:
                os.environ["FRED_API_KEY"] = fred_key
        if not os.environ.get("FRED_API_KEY"):
            raise RuntimeError("FRED_API_KEY is required for the current-only regime recipe.")

        def stage_named_file(source_name: str, dest: Path) -> Path:
            candidates = [
                DRIVE_DATA_DIR / source_name,
                DRIVE_DATA_DIR / "market" / source_name,
                DRIVE_DATA_DIR / "constituents" / source_name,
            ]
            source = next((candidate for candidate in candidates if candidate.exists()), None)
            if source is None:
                raise FileNotFoundError("Missing input " + source_name + "\nSearched:\n" + "\n".join(map(str, candidates)))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            return dest

        repo_market_csv = stage_named_file(MARKET_FILENAME, REPO_DIR / "data" / "raw" / "market" / MARKET_FILENAME)
        repo_pit_csv = stage_named_file(PIT_FILENAME, REPO_DIR / "data" / "raw" / "constituents" / PIT_FILENAME)
        repo_snapshot_csv = stage_named_file(SNAPSHOT_FILENAME, REPO_DIR / "data" / "raw" / "constituents" / SNAPSHOT_FILENAME)

        snapshots = pd.read_csv(repo_snapshot_csv)
        snapshot_counts = snapshots.groupby("as_of_date")["kdcode"].nunique()
        sector_counts = snapshots.groupby(["as_of_date", "gics_sector"])["kdcode"].nunique()
        data_audit = {
            "gpu_name": gpu_name,
            "market_csv": str(repo_market_csv),
            "pit_csv": str(repo_pit_csv),
            "snapshot_csv": str(repo_snapshot_csv),
            "snapshot_min_selected": int(snapshot_counts.min()),
            "snapshot_max_selected": int(snapshot_counts.max()),
            "bad_sector_cells": int((sector_counts != 10).sum()),
            "pit_min_scoreable_stocks": PIT_MIN_SCOREABLE_STOCKS,
        }
        if data_audit["snapshot_min_selected"] != 110 or data_audit["snapshot_max_selected"] != 110:
            raise RuntimeError(f"Unexpected PIT top-10 breadth: {data_audit}")
        if data_audit["bad_sector_cells"] != 0:
            raise RuntimeError(f"Sector top-10 validation failed: {data_audit}")

        def variant_overrides(loss_key: str) -> list[str]:
            variant = LOSS_VARIANTS[loss_key]
            overrides = [
                f"training.loss_type={variant['loss_type']}",
                f"training.selection_metric={variant['selection_metric']}",
            ]
            if variant["loss_type"] == "lambdarank_ic":
                overrides.extend(
                    [
                        f"training.lambdarank_ic_max_pairs_per_day={variant['lambdarank_ic_max_pairs_per_day']}",
                        f"training.lambdarank_ic_temperature={variant['lambdarank_ic_temperature']}",
                    ]
                )
            if variant["loss_type"] == "portfolio_ic":
                overrides.extend(
                    [
                        f"training.portfolio_ic_top_k={variant['portfolio_ic_top_k']}",
                        f"training.portfolio_ic_weight={variant['portfolio_ic_weight']}",
                        f"training.portfolio_ic_temperature={variant['portfolio_ic_temperature']}",
                    ]
                )
            return overrides

        BASE_OVERRIDES = [
            "data.source=csv",
            f"data.filename={repo_market_csv.relative_to(REPO_DIR).as_posix()}",
            f"data.pit_universe_csv={repo_pit_csv.relative_to(REPO_DIR).as_posix()}",
            "data.use_pit_universe=true",
            "data.pit_universe_mode=masked_panel",
            f"data.pit_min_scoreable_stocks={PIT_MIN_SCOREABLE_STOCKS}",
            "data.pit_breadth_policy=error",
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
            "model.label_t=5",
            "model.temporal_encoder=gru_attn",
            "tracking.enabled=false",
            "tracking.log_artifacts=false",
            "tracking.log_checkpoints=false",
            "tracking.log_predictions=false",
        ]

        training_jobs = []
        for year in YEARS:
            window = PIT_WINDOWS[year]
            for loss_key, variant in LOSS_VARIANTS.items():
                for seed in variant["seeds_to_train"]:
                    name = f"top10_{loss_key}_{year}_seed{seed}"
                    job_root = TRAINING_ROOT / loss_key / str(year) / f"seed{seed}"
                    training_jobs.append(
                        {
                            "name": name,
                            "loss_key": loss_key,
                            "loss_type": variant["loss_type"],
                            "year": year,
                            "base_seed": seed,
                            "job_root": str(job_root),
                            "overrides": [
                                *BASE_OVERRIDES,
                                *variant_overrides(loss_key),
                                f"seed={seed}",
                                f"experiment_name={name}",
                                f"output_dir={job_root.as_posix()}",
                                f"data.train_start={window['train_start']}",
                                f"data.train_end={window['train_end']}",
                                f"data.val_start={window['val_start']}",
                                f"data.val_end={window['val_end']}",
                                f"data.test_start={window['test_start']}",
                                f"data.test_end={window['test_end']}",
                            ],
                        }
                    )

        manifest = {
            "run_tag": RUN_TAG,
            "branch": BRANCH,
            "recipe": RECIPE,
            "years": YEARS,
            "pit_windows": PIT_WINDOWS,
            "data_audit": data_audit,
            "known_existing_rows": KNOWN_EXISTING_ROWS,
            "loss_variants": LOSS_VARIANTS,
            "training_jobs": training_jobs,
            "backtest": {
                "script": "tests/backtest_sp500_daily.py",
                "suffix": BACKTEST_SUFFIX,
                "top_k": 10,
                "label_t": 5,
                "transaction_costs": True,
                "spread_bps": 10,
                "slippage_bps": 5,
                "rank_drop_gate": True,
                "min_rank_drop": 30,
            },
        }
        write_json(RUN_ROOT / "manifest.json", manifest)
        write_json(SUMMARY_DIR / "known_existing_rows.json", KNOWN_EXISTING_ROWS)

        def latest_run_dir(job_root: Path, name: str) -> Path | None:
            candidates = sorted((job_root / name).glob("20*_??????"))
            if not candidates:
                candidates = sorted((job_root / name).glob("*"))
            return candidates[-1] if candidates else None

        def backtest_predictions(pred_dir: Path, year: int, out_prefix: str) -> dict:
            env = {**os.environ, "MPLBACKEND": "Agg"}
            window = PIT_WINDOWS[year]
            cmd = [
                sys.executable,
                "-X",
                "utf8",
                str(REPO_DIR / "tests" / "backtest_sp500_daily.py"),
                "--predictions_dir",
                str(pred_dir),
                "--data_file",
                str(repo_market_csv),
                "--pit_universe_csv",
                str(repo_pit_csv),
                "--test_start",
                window["test_start"],
                "--test_end",
                window["test_end"],
                "--label_t",
                "5",
                "--top_k",
                "10",
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
            run_stream(cmd, cwd=REPO_DIR, log_name=f"backtest_{out_prefix}.log", phase=f"backtest_{out_prefix}", env=env)
            metrics_path = pred_dir.parent / f"backtest{BACKTEST_SUFFIX}" / "backtest_metrics.json"
            return {
                "predictions_dir": str(pred_dir),
                "metrics_path": str(metrics_path),
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
            }

        def completed_row_path(job: dict) -> Path:
            return SUMMARY_DIR / "completed_rows" / job["loss_key"] / str(job["year"]) / f"seed{job['base_seed']}.json"

        def find_existing_prediction_dir(row: dict) -> Path | None:
            source_root = DRIVE_ROOT / "MCI-GRU-Ablations" / row["source"]
            if not source_root.exists():
                return None
            candidates = []
            for pred_dir in source_root.rglob("averaged_predictions"):
                path_text = pred_dir.as_posix()
                year_token = str(row["year"])
                if year_token in path_text or f"default_{year_token}" in path_text:
                    candidates.append(pred_dir)
            return sorted(candidates)[-1] if candidates else None

        def replay_existing_prediction_row(row: dict) -> dict:
            source_pred_dir = find_existing_prediction_dir(row)
            replay_row = {
                **row,
                "status": "MISSING_PREDICTIONS",
                "mode": "reused_existing_training",
                "source_predictions_dir": str(source_pred_dir) if source_pred_dir else None,
            }
            if source_pred_dir is None:
                return replay_row
            staged_pred_dir = (
                REUSED_REPLAY_ROOT
                / row["loss_key"]
                / str(row["year"])
                / f"seed{row['base_seed']}"
                / "averaged_predictions"
            )
            if not staged_pred_dir.exists():
                shutil.copytree(source_pred_dir, staged_pred_dir, dirs_exist_ok=True)
            backtest = backtest_predictions(
                staged_pred_dir,
                row["year"],
                f"reused_{row['loss_key']}_{row['year']}_seed{row['base_seed']}",
            )
            return {
                **replay_row,
                "status": "OK",
                "staged_predictions_dir": str(staged_pred_dir),
                **backtest,
            }

        training_rows = []
        backtest_rows = []
        reused_backtest_rows = []

        try:
            for existing_row in KNOWN_EXISTING_ROWS:
                if not existing_row.get("needs_backtest_replay"):
                    continue
                existing_path = (
                    SUMMARY_DIR
                    / "reused_backtests"
                    / existing_row["loss_key"]
                    / str(existing_row["year"])
                    / f"seed{existing_row['base_seed']}.json"
                )
                if existing_path.exists():
                    reused_backtest_rows.append(json.loads(existing_path.read_text(encoding="utf-8")))
                    continue
                heartbeat(
                    "RUNNING",
                    "replay_existing_predictions",
                    current_job=(
                        f"{existing_row['loss_key']}_{existing_row['year']}_"
                        f"seed{existing_row['base_seed']}"
                    ),
                    completed_reused_backtests=len(reused_backtest_rows),
                )
                replay_row = replay_existing_prediction_row(existing_row)
                write_json(existing_path, replay_row)
                reused_backtest_rows.append(replay_row)
                write_json(SUMMARY_DIR / "reused_backtest_rows.json", reused_backtest_rows)

            for job in training_jobs:
                row_path = completed_row_path(job)
                if row_path.exists():
                    row = json.loads(row_path.read_text(encoding="utf-8"))
                    training_rows.append(row)
                    continue
                heartbeat("RUNNING", "training", current_job=job["name"], completed_training_rows=len(training_rows))
                cmd = [sys.executable, "-u", str(REPO_DIR / "run_experiment.py"), *job["overrides"]]
                run_stream(cmd, cwd=REPO_DIR, log_name=f"training_{job['name']}.log", phase=f"training_{job['name']}")
                run_dir = latest_run_dir(Path(job["job_root"]), job["name"])
                if run_dir is None:
                    raise FileNotFoundError(f"Missing run dir for {job['name']}")
                pred_dir = run_dir / "averaged_predictions"
                if not pred_dir.exists():
                    raise FileNotFoundError(f"Missing averaged_predictions for {job['name']}: {pred_dir}")
                row = {
                    **{key: job[key] for key in ["name", "loss_key", "loss_type", "year", "base_seed"]},
                    "status": "OK",
                    "mode": "trained",
                    "run_dir": str(run_dir),
                    "predictions_dir": str(pred_dir),
                    "training_summary": json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8")),
                    "evaluation_summary": json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8")),
                }
                write_json(row_path, row)
                training_rows.append(row)
                write_json(SUMMARY_DIR / "training_rows.json", training_rows)

                backtest = backtest_predictions(pred_dir, job["year"], job["name"])
                bt_row = {
                    "name": job["name"],
                    "loss_key": job["loss_key"],
                    "loss_type": job["loss_type"],
                    "year": job["year"],
                    "base_seed": job["base_seed"],
                    "mode": "trained",
                    **backtest,
                }
                backtest_rows.append(bt_row)
                write_json(SUMMARY_DIR / "backtest_rows.json", backtest_rows)

            write_json(SUMMARY_DIR / "training_rows.json", training_rows)
            write_json(SUMMARY_DIR / "backtest_rows.json", backtest_rows)
            write_json(SUMMARY_DIR / "reused_backtest_rows.json", reused_backtest_rows)
            write_json(
                RUN_ROOT / "run_summary.json",
                {
                    "status": "OK",
                    "run_tag": RUN_TAG,
                    "run_root": str(RUN_ROOT),
                    "data_audit": data_audit,
                    "known_existing_rows": KNOWN_EXISTING_ROWS,
                    "training_rows": training_rows,
                    "backtest_rows": backtest_rows,
                    "reused_backtest_rows": reused_backtest_rows,
                },
            )
            heartbeat(
                "OK",
                "complete",
                completed_training_rows=len(training_rows),
                completed_backtest_rows=len(backtest_rows),
                completed_reused_backtests=len(reused_backtest_rows),
            )
        except Exception as exc:
            heartbeat("FAILED", "failed", error=repr(exc))
            raise
        finally:
            dest = RUN_ROOT / "artifacts"
            if LOCAL_RUN_ROOT.exists():
                shutil.copytree(LOCAL_RUN_ROOT, dest / "local_run_root", dirs_exist_ok=True)
            if IN_COLAB and runtime is not None:
                try:
                    runtime.unassign()
                except Exception as exc:
                    print("Manual Runtime > Disconnect and delete runtime may be needed:", exc)
        """
    ),
]


def build_notebook() -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
