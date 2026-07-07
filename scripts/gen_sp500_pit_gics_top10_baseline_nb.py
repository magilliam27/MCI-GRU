"""Generate a Colab launcher for reduced PIT GICS top-10 baseline runs."""

from __future__ import annotations

from pathlib import Path

from nb_lib import COLAB_GPU_METADATA_BARE_KERNEL, backtest_engine_path_expr, write_notebook
from nb_lib import code_lines as code
from nb_lib import md_lines as md

OUT = Path("notebooks/sp500_pit_gics_top10_baseline_colab.ipynb")


cells = [
    md(
        """
        # Reduced PIT GICS Top-10 Multiyear Baseline

        Runs the frozen default MCI-GRU recipe from
        `docs/DEFAULT_EXPERIMENT_RECIPE.md` on a point-in-time reduced
        S&P 500 universe: top 10 by market cap within each GICS sector at
        monthly PIT snapshots.

        The notebook is configured for test years 2022, 2023, and 2024 with
        rolling train/validation/test windows. It intentionally refuses to call
        the 2022/2023 setup apples-to-apples when the staged selector artifact
        starts after the required rolling train window. Extend the selector
        snapshots first, or explicitly set
        `REQUIRE_APPLES_TO_APPLES_SELECTOR_HISTORY = False` and label the run as
        not apples-to-apples.
        """
    ),
    code(
        r"""
        from __future__ import annotations

        import json
        import os
        import shutil
        import subprocess
        import sys
        import time
        from datetime import datetime, timezone
        from pathlib import Path

        import pandas as pd

        try:
            from google.colab import drive, runtime, userdata
            IN_COLAB = True
        except Exception:
            drive = runtime = userdata = None
            IN_COLAB = False

        REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
        BRANCH = "main"
        RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        MARKET_FILENAME = (
            "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_"
            "lseg_20150101_20260622.csv"
        )
        PIT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv"
        SNAPSHOT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_snapshots.csv"
        MARKET_META_FILENAME = MARKET_FILENAME.replace(".csv", ".meta.json")
        PIT_META_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_meta.json"

        YEARS = [2022, 2023, 2024]
        REQUIRE_APPLES_TO_APPLES_SELECTOR_HISTORY = True
        RECIPE = "static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1"

        PIT_WINDOWS = {
            2022: {
                'experiment_name': 'sp500_pit_gics_top10_baseline_default_2022',
                'train_start': '2016-01-01',
                'train_end': '2020-12-31',
                'val_start': '2021-01-08',
                'val_end': '2021-12-31',
                'test_start': '2022-01-08',
                'test_end': '2022-12-31',
            },
            2023: {
                'experiment_name': 'sp500_pit_gics_top10_baseline_default_2023',
                'train_start': '2017-01-01',
                'train_end': '2021-12-31',
                'val_start': '2022-01-08',
                'val_end': '2022-12-31',
                'test_start': '2023-01-08',
                'test_end': '2023-12-31',
            },
            2024: {
                'experiment_name': 'sp500_pit_gics_top10_baseline_default_2024',
                'train_start': '2018-01-01',
                'train_end': '2022-12-31',
                'val_start': '2023-01-08',
                'val_end': '2023-12-31',
                'test_start': '2024-01-08',
                'test_end': '2024-12-31',
            },
        }

        NUM_MODELS = 20
        NUM_EPOCHS = 100
        EARLY_STOPPING_PATIENCE = 15
        PIT_MIN_SCOREABLE_STOCKS = 100
        BACKTEST_SUFFIX = "_pit_daily"

        REFERENCE_2025 = {
            "drive_folder": "https://drive.google.com/drive/folders/1W1Ykd-gvPXcGQjuunKjx1Upn-Dnsv_mp",
            "mean_best_val_ic": 0.0337335076,
            "test_avg_ic": 0.0321129471,
            "test_avg_rank_ic": 0.0448513563,
            "top10_no_tc_total_return": 0.2659,
            "benchmark_return": 0.1575,
            "excess_return": 0.1083,
            "asr": 0.913,
            "max_drawdown": -0.2550,
        }

        if IN_COLAB:
            drive.mount("/content/drive")

        REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()
        DRIVE_DATA_DIR = (
            Path("/content/drive/MyDrive/MCI_GRU_shared/data")
            if IN_COLAB
            else Path.cwd() / "data" / "raw"
        )
        DRIVE_RUN_ROOT = (
            Path("/content/drive/MyDrive/MCI-GRU-Ablations/sp500_gics_top10_baseline_multiyear") / RUN_TAG
            if IN_COLAB
            else Path.cwd() / "drive_outputs" / "sp500_gics_top10_baseline_multiyear" / RUN_TAG
        )
        LOCAL_RUN_ROOT = (
            Path("/content/mci_gru_runs/sp500_gics_top10_baseline_multiyear") / RUN_TAG
            if IN_COLAB
            else Path.cwd() / "results" / "sp500_gics_top10_baseline_multiyear" / RUN_TAG
        )
        LOG_DIR = DRIVE_RUN_ROOT / "logs"
        SUMMARY_DIR = DRIVE_RUN_ROOT / "summaries"
        for directory in [DRIVE_RUN_ROOT, LOCAL_RUN_ROOT, LOG_DIR, SUMMARY_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        def utc_now() -> str:
            return datetime.now(timezone.utc).isoformat()

        def write_json(path: Path, payload: dict) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        def heartbeat(status: str, phase: str, **extra) -> None:
            payload = {
                "status": status,
                "phase": phase,
                "updated_at_utc": utc_now(),
                "run_tag": RUN_TAG,
                "recipe": RECIPE,
                "years": YEARS,
                **extra,
            }
            write_json(DRIVE_RUN_ROOT / "heartbeat.json", payload)

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
                last_write = 0.0
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="")
                    handle.write(line)
                    if time.time() - last_write > 60:
                        heartbeat("RUNNING", phase, last_line=line.strip(), log=str(log_path))
                        last_write = time.time()
                returncode = proc.wait()
            heartbeat("RUNNING", phase, returncode=returncode, log=str(log_path))
            if returncode != 0:
                raise RuntimeError(f"{phase} failed with return code {returncode}; see {log_path}")

        def stage_named_file(source_name: str, dest: Path) -> Path:
            candidates = [
                DRIVE_DATA_DIR / source_name,
                DRIVE_DATA_DIR / "market" / source_name,
                DRIVE_DATA_DIR / "constituents" / source_name,
            ]
            source = next((path for path in candidates if path.exists()), None)
            if source is None:
                searched = "\n".join(str(path) for path in candidates)
                raise FileNotFoundError(f"Missing input {source_name}. Searched:\n{searched}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            print("Staged:", source, "->", dest)
            return dest

        def selector_blockers(selector_start: str) -> list[dict]:
            start_ts = pd.Timestamp(selector_start)
            blockers = []
            for year in YEARS:
                train_start = pd.Timestamp(PIT_WINDOWS[year]["train_start"])
                if start_ts > train_start + pd.Timedelta(days=7):
                    blockers.append(
                        {
                            "year": year,
                            "selector_start": selector_start,
                            "required_train_start": PIT_WINDOWS[year]["train_start"],
                            "reason": "selector snapshots begin after required train_start",
                        }
                    )
            return blockers

        def base_overrides(year: int, run_root: Path) -> list[str]:
            window = PIT_WINDOWS[year]
            return [
                f"experiment_name={window['experiment_name']}",
                f"output_dir={run_root.as_posix()}",
                "seed=1729",
                "data.source=csv",
                f"data.filename={repo_market_csv.relative_to(REPO_DIR).as_posix()}",
                f"data.train_start={window['train_start']}",
                f"data.train_end={window['train_end']}",
                f"data.val_start={window['val_start']}",
                f"data.val_end={window['val_end']}",
                f"data.test_start={window['test_start']}",
                f"data.test_end={window['test_end']}",
                "data.use_pit_universe=true",
                f"data.pit_universe_csv={repo_pit_csv.relative_to(REPO_DIR).as_posix()}",
                "data.pit_universe_mode=masked_panel",
                f"data.pit_min_scoreable_stocks={PIT_MIN_SCOREABLE_STOCKS}",
                "data.pit_breadth_policy=error",
                "training.num_models=20",
                "training.num_epochs=100",
                "training.early_stopping_patience=15",
                "training.learning_rate=5e-5",
                "training.lr_scheduler=cosine",
                "training.loss_type=ic",
                "training.label_type=returns",
                "training.selection_metric=val_ic",
                "training.shuffle_train=true",
                "model.label_t=5",
                "graph.judge_value=0.8",
                "graph.update_frequency_months=0",
                "graph.corr_lookback_days=252",
                "graph.top_k=0",
                "graph.top_k_metric=corr",
                "graph.use_multi_feature_edges=true",
                "graph.append_snapshot_age_days=false",
                "graph.use_lead_lag_features=false",
                "graph.drop_edge_p=0.1",
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
                "tracking.enabled=false",
                "tracking.log_artifacts=false",
                "tracking.log_checkpoints=false",
                "tracking.log_predictions=false",
            ]

        heartbeat("RUNNING", "setup")

        if IN_COLAB:
            if not REPO_DIR.exists():
                subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)], check=True)
            else:
                subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "checkout", BRANCH], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "pull", "origin", BRANCH], check=True)

        os.chdir(REPO_DIR)

        if IN_COLAB:
            gpu_name = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            ).strip()
            print("GPU:", gpu_name)
            if "T4" in gpu_name.upper():
                raise RuntimeError(f"T4 runtime detected ({gpu_name}). Use an L4/G4-class runtime for this baseline.")
        else:
            gpu_name = "local"

        if IN_COLAB:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev,tracking,fred]"], check=True)

        if IN_COLAB and userdata is not None:
            try:
                fred_key = userdata.get("FRED_API_KEY")
            except Exception:
                fred_key = None
            if fred_key:
                os.environ["FRED_API_KEY"] = fred_key
        if not os.environ.get("FRED_API_KEY"):
            raise RuntimeError("FRED_API_KEY is required for the frozen default current-only regime recipe.")

        repo_market_csv = REPO_DIR / "data" / "raw" / "market" / MARKET_FILENAME
        repo_pit_csv = REPO_DIR / "data" / "raw" / "constituents" / PIT_FILENAME
        repo_snapshot_csv = REPO_DIR / "data" / "raw" / "constituents" / SNAPSHOT_FILENAME
        for source_name, dest in [
            (MARKET_FILENAME, repo_market_csv),
            (PIT_FILENAME, repo_pit_csv),
            (SNAPSHOT_FILENAME, repo_snapshot_csv),
            (MARKET_META_FILENAME, REPO_DIR / "data" / "raw" / "market" / MARKET_META_FILENAME),
            (PIT_META_FILENAME, REPO_DIR / "data" / "raw" / "constituents" / PIT_META_FILENAME),
        ]:
            stage_named_file(source_name, dest)

        market_preview = pd.read_csv(repo_market_csv, usecols=["kdcode", "dt"])
        pit_preview = pd.read_csv(repo_pit_csv)
        snapshots = pd.read_csv(repo_snapshot_csv)
        snapshot_counts = snapshots.groupby("as_of_date")["kdcode"].nunique()
        sector_counts = snapshots.groupby(["as_of_date", "gics_sector"])["kdcode"].nunique()
        selector_start = str(pd.to_datetime(snapshots["as_of_date"]).min().date())
        data_audit = {
            "market_rows": int(len(market_preview)),
            "market_unique_kdcodes": int(market_preview["kdcode"].nunique()),
            "market_date_min": str(market_preview["dt"].min()),
            "market_date_max": str(market_preview["dt"].max()),
            "pit_interval_rows": int(len(pit_preview)),
            "pit_union_kdcodes": int(pit_preview["kdcode"].nunique()),
            "selector_start": selector_start,
            "snapshot_dates": int(snapshot_counts.shape[0]),
            "snapshot_min_selected": int(snapshot_counts.min()),
            "snapshot_max_selected": int(snapshot_counts.max()),
            "bad_sector_cells": int((sector_counts != 10).sum()),
            "gpu_name": gpu_name,
            "pit_min_scoreable_stocks": PIT_MIN_SCOREABLE_STOCKS,
            "selector_history_blockers": selector_blockers(selector_start),
        }
        print("Data audit:", json.dumps(data_audit, indent=2))
        write_json(DRIVE_RUN_ROOT / "data_audit.json", data_audit)
        if data_audit["snapshot_min_selected"] != 110 or data_audit["snapshot_max_selected"] != 110:
            raise RuntimeError(f"Unexpected selected breadth: {data_audit}")
        if data_audit["bad_sector_cells"] != 0:
            raise RuntimeError(f"Sector top-10 validation failed: {data_audit}")
        if data_audit["selector_history_blockers"] and REQUIRE_APPLES_TO_APPLES_SELECTOR_HISTORY:
            raise RuntimeError(
                "selector snapshots begin after required train_start for one or more years; "
                "extend/re-pull LSEG selector snapshots before the full multiyear baseline. "
                "Set REQUIRE_APPLES_TO_APPLES_SELECTOR_HISTORY=False only for a clearly labeled "
                "not apples-to-apples shorter-history run. "
                f"Blockers: {data_audit['selector_history_blockers']}"
            )

        manifest = {
            "run_tag": RUN_TAG,
            "branch": BRANCH,
            "years": YEARS,
            "pit_windows": PIT_WINDOWS,
            "recipe": RECIPE,
            "reference_2025": REFERENCE_2025,
            "data_audit": data_audit,
            "require_apples_to_apples_selector_history": REQUIRE_APPLES_TO_APPLES_SELECTOR_HISTORY,
        }
        write_json(DRIVE_RUN_ROOT / "manifest.json", manifest)

        training_rows = []
        backtest_rows = []
        try:
            for year in YEARS:
                year_root = LOCAL_RUN_ROOT / str(year)
                overrides = base_overrides(year, year_root)
                write_json(DRIVE_RUN_ROOT / f"hydra_overrides_{year}.json", {"overrides": overrides})

                train_cmd = [sys.executable, "-u", str(REPO_DIR / "run_experiment.py"), *overrides]
                run_stream(train_cmd, cwd=REPO_DIR, log_name=f"training_{year}.log", phase=f"training_{year}")

                experiment_name = PIT_WINDOWS[year]["experiment_name"]
                run_dirs = sorted(Path(year_root / experiment_name).glob("20*_??????"))
                if not run_dirs:
                    run_dirs = sorted(Path(year_root / experiment_name).glob("*"))
                run_dir = run_dirs[-1]
                pred_dir = run_dir / "averaged_predictions"
                if not pred_dir.exists():
                    raise FileNotFoundError(f"Missing averaged_predictions: {pred_dir}")

                training_row = {
                    "year": year,
                    "status": "OK",
                    "run_dir": str(run_dir),
                    "training_summary": json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8")),
                    "evaluation_summary": json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8")),
                }
                training_rows.append(training_row)
                write_json(SUMMARY_DIR / "training_results.json", {"rows": training_rows})

                env = os.environ.copy()
                env["MPLBACKEND"] = "Agg"
                window = PIT_WINDOWS[year]
                backtest_cmd = [
                    sys.executable,
                    "-X",
                    "utf8",
                    __BACKTEST_ENGINE_PATH_EXPR__,
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
                    "--auto_save",
                    "--backtest_suffix",
                    BACKTEST_SUFFIX,
                ]
                run_stream(backtest_cmd, cwd=REPO_DIR, log_name=f"backtest_{year}.log", phase=f"backtest_{year}", env=env)

                backtest_metrics = run_dir / f"backtest{BACKTEST_SUFFIX}" / "backtest_metrics.json"
                backtest_rows.append(
                    {
                        "year": year,
                        "status": "OK",
                        "run_dir": str(run_dir),
                        "backtest_metrics": json.loads(backtest_metrics.read_text(encoding="utf-8")) if backtest_metrics.exists() else {},
                    }
                )
                write_json(SUMMARY_DIR / "backtest_results.json", {"rows": backtest_rows})

            summary = {
                "status": "OK",
                "run_tag": RUN_TAG,
                "drive_run_root": str(DRIVE_RUN_ROOT),
                "data_audit": data_audit,
                "training_rows": training_rows,
                "backtest_rows": backtest_rows,
                "reference_2025": REFERENCE_2025,
            }
            write_json(DRIVE_RUN_ROOT / "run_summary.json", summary)
            heartbeat("OK", "complete", drive_run_root=str(DRIVE_RUN_ROOT))
        except Exception as exc:
            heartbeat("FAILED", "failed", error=repr(exc))
            raise
        finally:
            dest = DRIVE_RUN_ROOT / "artifacts"
            if LOCAL_RUN_ROOT.exists():
                shutil.copytree(LOCAL_RUN_ROOT, dest / "local_run_root", dirs_exist_ok=True)
            if IN_COLAB and runtime is not None:
                runtime.unassign()
        """.replace(
            "__BACKTEST_ENGINE_PATH_EXPR__",
            backtest_engine_path_expr("backtest_sp500_daily", quote='"'),
        )
    ),
]

write_notebook(cells, OUT, metadata=COLAB_GPU_METADATA_BARE_KERNEL, indent=2)
