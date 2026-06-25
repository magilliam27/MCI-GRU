"""Generate a Colab launcher for reduced PIT GICS top-10 LambdaRankIC screen runs."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT = Path("notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb")


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    md(
        """
        # Top-10 PIT LambdaRankIC Screen

        This notebook launches the reduced S&P 500 PIT GICS top-10
        LambdaRankIC screen described in
        `docs/superpowers/specs/2026-06-25-top10-lambdarank-screen-design.md`.
        It is intentionally a screen run: 2022 only, one seed, one model,
        40 epochs, patience 8, and the complete 110-name pair cap.

        Operate it from the visible Colab UI on a G4/L4-class Colab runtime,
        not T4/CPU. Treat Drive artifacts as the source of truth: heartbeat,
        logs, manifests, summaries, copied local artifacts, and rank-drop
        backtest outputs are all written under the run root.
        """
    ),
    md("## 1. Environment Setup"),
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

        IN_COLAB = "google.colab" in sys.modules
        REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
        BRANCH = "codex/top10-lambdarank-screen-20260625"
        REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()
        REQUIRE_G4_L4_GPU = True
        BLOCKED_GPU_NAMES = ("T4",)
        ALLOWED_GPU_MARKERS = (
            "G4",
            "L4",
            "A100",
            "H100",
            "RTX PRO 6000",
            "BLACKWELL",
        )
        STRICT_GPU_MARKERS: list[str] = []

        if IN_COLAB:
            from google.colab import drive, runtime, userdata
        else:
            drive = runtime = userdata = None

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
                raise RuntimeError(
                    f"Expected G4/L4-class Colab runtime, not T4/CPU. Visible GPU: {gpu_name}"
                )
            if not any(marker in upper_gpu for marker in ALLOWED_GPU_MARKERS):
                raise RuntimeError(
                    f"Refusing runtime GPU {gpu_name}; allowed markers are {ALLOWED_GPU_MARKERS}."
                )
            if STRICT_GPU_MARKERS and not any(marker in upper_gpu for marker in STRICT_GPU_MARKERS):
                raise RuntimeError(
                    f"GPU {gpu_name} does not match STRICT_GPU_MARKERS={STRICT_GPU_MARKERS}."
                )
            return gpu_name

        if IN_COLAB:
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

        if IN_COLAB:
            GPU_NAME = detect_gpu_name()
        else:
            GPU_NAME = "local"

        print("Repo:", REPO_DIR)
        print("Branch:", BRANCH)
        subprocess.run(["git", "rev-parse", "HEAD"], check=False)
        print("Python:", sys.executable)
        print("GPU:", GPU_NAME)

        if IN_COLAB and userdata is not None and not os.environ.get("FRED_API_KEY"):
            try:
                fred_key = userdata.get("FRED_API_KEY")
            except Exception as exc:
                print("Could not read FRED_API_KEY from Colab Secrets:", exc)
                fred_key = None
            if fred_key:
                os.environ["FRED_API_KEY"] = fred_key
                print("FRED_API_KEY loaded from Colab Secrets.")

        if not os.environ.get("FRED_API_KEY"):
            raise RuntimeError("FRED_API_KEY is required for the current regime-enabled recipe.")
        """
    ),
    md("## 2. Run Constants And Helpers"),
    code(
        r"""
        RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        RECIPE_ID = "static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1"

        YEARS = [2022]
        BASE_SEEDS = [314159]
        NUM_MODELS = 1
        NUM_EPOCHS = 40
        EARLY_STOPPING_PATIENCE = 8
        PAIR_CAP = 8192
        COMPLETE_PAIR_COUNT_110 = 5995
        assert PAIR_CAP >= COMPLETE_PAIR_COUNT_110
        TOP_K = 10
        SPREAD_BPS = 10
        SLIPPAGE_BPS = 5
        MIN_RANK_DROP = 30
        PIT_MIN_SCOREABLE_STOCKS = 100

        MARKET_FILENAME = (
            "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_"
            "lseg_20150101_20260622.csv"
        )
        PIT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv"
        SNAPSHOT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_snapshots.csv"
        MARKET_META_FILENAME = MARKET_FILENAME.replace(".csv", ".meta.json")
        PIT_META_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_meta.json"
        EXPECTED_MARKET_META_FILENAME = (
            "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.meta.json"
        )
        assert MARKET_META_FILENAME == EXPECTED_MARKET_META_FILENAME

        PIT_WINDOWS = {
            2022: {
                "experiment_name": "sp500_pit_gics_top10_lambdarank_ic_screen_2022",
                "train_start": "2016-01-01",
                "train_end": "2020-12-31",
                "val_start": "2021-01-08",
                "val_end": "2021-12-31",
                "test_start": "2022-01-08",
                "test_end": "2022-12-31",
            }
        }

        DRIVE_DATA_DIR = (
            Path("/content/drive/MyDrive/MCI_GRU_shared/data")
            if IN_COLAB
            else Path.cwd() / "data" / "raw"
        )
        DRIVE_RUN_ROOT = (
            Path("/content/drive/MyDrive/MCI-GRU-Ablations/sp500_gics_top10_lambdarank_ic_screen") / RUN_TAG
            if IN_COLAB
            else Path.cwd() / "drive_outputs" / "sp500_gics_top10_lambdarank_ic_screen" / RUN_TAG
        )
        LOCAL_RUN_ROOT = (
            Path("/content/mci_gru_runs/sp500_gics_top10_lambdarank_ic_screen") / RUN_TAG
            if IN_COLAB
            else Path.cwd() / "results" / "sp500_gics_top10_lambdarank_ic_screen" / RUN_TAG
        )
        LOG_DIR = DRIVE_RUN_ROOT / "logs"
        SUMMARY_DIR = DRIVE_RUN_ROOT / "summaries"
        ARTIFACT_DIR = DRIVE_RUN_ROOT / "artifacts"
        HEARTBEAT_PATH = DRIVE_RUN_ROOT / "heartbeat.json"
        MANIFEST_FILENAME = "lambdarank_ic_sp500_pit_gics_top10_screen_manifest.json"

        for directory in [DRIVE_RUN_ROOT, LOCAL_RUN_ROOT, LOG_DIR, SUMMARY_DIR, ARTIFACT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        def utc_now() -> str:
            return datetime.now(timezone.utc).isoformat()

        def write_json(path: Path, payload: dict) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field) for field in fieldnames})

        def write_heartbeat(status: str, phase: str, **extra) -> None:
            payload = {
                "status": status,
                "phase": phase,
                "updated_at_utc": utc_now(),
                "run_tag": RUN_TAG,
                "branch": BRANCH,
                "recipe_id": RECIPE_ID,
                "years": YEARS,
                "pair_cap": PAIR_CAP,
                "complete_pair_count_110": COMPLETE_PAIR_COUNT_110,
                "drive_run_root": str(DRIVE_RUN_ROOT),
                "local_run_root": str(LOCAL_RUN_ROOT),
                **extra,
            }
            write_json(HEARTBEAT_PATH, payload)

        def run_stream(
            cmd: list[str],
            *,
            cwd: Path,
            log_name: str,
            phase: str,
            env: dict | None = None,
        ) -> None:
            log_path = LOG_DIR / log_name
            write_heartbeat("RUNNING", phase, command=" ".join(cmd), log=str(log_path))
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
                last_write = 0.0
                for line in proc.stdout:
                    print(line, end="")
                    handle.write(line)
                    if time.time() - last_write > 60:
                        write_heartbeat("RUNNING", phase, last_line=line.strip(), log=str(log_path))
                        last_write = time.time()
                returncode = proc.wait()
            write_heartbeat("RUNNING", phase, returncode=returncode, log=str(log_path))
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
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            print("Staged:", source, "->", dest)
            return dest
        """
    ),
    md("## 3. Data Staging And Audit"),
    code(
        r"""
        write_heartbeat("RUNNING", "data_staging")

        repo_market_csv = REPO_DIR / "data" / "raw" / "market" / MARKET_FILENAME
        repo_pit_csv = REPO_DIR / "data" / "raw" / "constituents" / PIT_FILENAME
        repo_snapshot_csv = REPO_DIR / "data" / "raw" / "constituents" / SNAPSHOT_FILENAME
        repo_market_meta_json = REPO_DIR / "data" / "raw" / "market" / MARKET_META_FILENAME
        repo_pit_meta_json = REPO_DIR / "data" / "raw" / "constituents" / PIT_META_FILENAME

        for source_name, dest in [
            (MARKET_FILENAME, repo_market_csv),
            (PIT_FILENAME, repo_pit_csv),
            (SNAPSHOT_FILENAME, repo_snapshot_csv),
            (MARKET_META_FILENAME, repo_market_meta_json),
            (PIT_META_FILENAME, repo_pit_meta_json),
        ]:
            stage_named_file(source_name, dest)

        market_preview = pd.read_csv(repo_market_csv, usecols=["kdcode", "dt"])
        pit_preview = pd.read_csv(repo_pit_csv)
        snapshots = pd.read_csv(repo_snapshot_csv)
        market_meta = json.loads(repo_market_meta_json.read_text(encoding="utf-8"))
        pit_meta = json.loads(repo_pit_meta_json.read_text(encoding="utf-8"))

        selector_start = str(pd.to_datetime(snapshots["as_of_date"]).min().date())
        assert selector_start == '2016-01-04', selector_start
        snapshot_counts = snapshots.groupby("as_of_date")["kdcode"].nunique()
        sector_counts = snapshots.groupby(["as_of_date", "gics_sector"])["kdcode"].nunique()
        data_audit = {
            "selector_start": selector_start,
            "snapshot_dates": int(snapshot_counts.shape[0]),
            "snapshot_min_selected": int(snapshot_counts.min()),
            "snapshot_max_selected": int(snapshot_counts.max()),
            "bad_sector_cells": int((sector_counts != 10).sum()),
            "pit_union_kdcodes": int(pit_preview["kdcode"].nunique()),
            "missing_identifiers": market_meta.get("missing_identifiers", []),
            "complete_pair_count_110": COMPLETE_PAIR_COUNT_110,
            "pair_cap": PAIR_CAP,
            "market_rows": int(len(market_preview)),
            "market_unique_kdcodes": int(market_preview["kdcode"].nunique()),
            "market_date_min": str(market_preview["dt"].min()),
            "market_date_max": str(market_preview["dt"].max()),
            "pit_interval_rows": int(len(pit_preview)),
            "pit_meta": pit_meta,
        }
        EXPECTED_DATA_AUDIT = {
            "snapshot_dates": 127,
            "snapshot_min_selected": 110,
            "snapshot_max_selected": 110,
            "bad_sector_cells": 0,
            "pit_union_kdcodes": 205,
            "missing_identifiers": [],
        }
        assert data_audit["snapshot_dates"] == EXPECTED_DATA_AUDIT["snapshot_dates"], data_audit
        assert data_audit["snapshot_min_selected"] == EXPECTED_DATA_AUDIT["snapshot_min_selected"], data_audit
        assert data_audit["snapshot_max_selected"] == EXPECTED_DATA_AUDIT["snapshot_max_selected"], data_audit
        assert data_audit["bad_sector_cells"] == EXPECTED_DATA_AUDIT["bad_sector_cells"], data_audit
        assert data_audit["pit_union_kdcodes"] == EXPECTED_DATA_AUDIT["pit_union_kdcodes"], data_audit
        assert data_audit["missing_identifiers"] == EXPECTED_DATA_AUDIT["missing_identifiers"], data_audit
        write_json(DRIVE_RUN_ROOT / "data_audit.json", data_audit)
        write_heartbeat("RUNNING", "data_audit", data_audit=data_audit)

        print("Data audit:", json.dumps(data_audit, indent=2, sort_keys=True))
        """
    ),
    md("## 4. Hydra Overrides And Manifest"),
    code(
        r"""
        def base_overrides(year: int, year_root: Path, repo_pit_csv: Path) -> list[str]:
            window = PIT_WINDOWS[year]
            return [
                f"experiment_name={window['experiment_name']}",
                f"output_dir={year_root.as_posix()}",
                "seed=314159",
                "data.source=csv",
                f"data.filename=data/raw/market/{MARKET_FILENAME}",
                f"data.train_start={window['train_start']}",
                f"data.train_end={window['train_end']}",
                f"data.val_start={window['val_start']}",
                f"data.val_end={window['val_end']}",
                f"data.test_start={window['test_start']}",
                f"data.test_end={window['test_end']}",
                "data.use_pit_universe=true",
                f"data.pit_universe_csv={repo_pit_csv.relative_to(REPO_DIR).as_posix()}",
                "data.pit_universe_mode=masked_panel",
                "data.pit_min_scoreable_stocks=100",
                "data.pit_breadth_policy=error",
                "training.num_models=1",
                "training.num_epochs=40",
                "training.early_stopping_patience=8",
                "training.learning_rate=5e-5",
                "training.lr_scheduler=cosine",
                "training.loss_type=lambdarank_ic",
                "training.selection_metric=val_rank_ic",
                "training.lambdarank_ic_max_pairs_per_day=8192",
                "training.lambdarank_ic_temperature=1.0",
                "training.label_type=returns",
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

        manifest = {
            "run_tag": RUN_TAG,
            "branch": BRANCH,
            "recipe_id": RECIPE_ID,
            "years": YEARS,
            "base_seeds": BASE_SEEDS,
            "num_models": NUM_MODELS,
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "pair_cap": PAIR_CAP,
            "complete_pair_count_110": COMPLETE_PAIR_COUNT_110,
            "top_k": TOP_K,
            "spread_bps": SPREAD_BPS,
            "slippage_bps": SLIPPAGE_BPS,
            "min_rank_drop": MIN_RANK_DROP,
            "pit_min_scoreable_stocks": PIT_MIN_SCOREABLE_STOCKS,
            "pit_windows": PIT_WINDOWS,
            "market_filename": MARKET_FILENAME,
            "pit_filename": PIT_FILENAME,
            "snapshot_filename": SNAPSHOT_FILENAME,
            "market_meta_filename": MARKET_META_FILENAME,
            "pit_meta_filename": PIT_META_FILENAME,
            "data_audit": data_audit,
        }
        manifest_path = DRIVE_RUN_ROOT / MANIFEST_FILENAME
        write_json(manifest_path, manifest)
        write_heartbeat("RUNNING", "manifest", manifest=str(manifest_path))
        print("Manifest:", manifest_path)
        """
    ),
    md("## 5. Training And Saved-Prediction Backtests"),
    code(
        r"""
        def latest_run_dir(year_root: Path, experiment_name: str) -> Path:
            candidates = sorted((year_root / experiment_name).glob("20*_??????"))
            if not candidates:
                candidates = sorted((year_root / experiment_name).glob("*"))
            if not candidates:
                raise FileNotFoundError(f"No run directory found under {year_root / experiment_name}")
            return candidates[-1]

        def read_json_if_exists(path: Path) -> dict:
            if not path.exists():
                return {}
            return json.loads(path.read_text(encoding="utf-8"))

        training_rows: list[dict] = []
        backtest_rows: list[dict] = []

        try:
            for year in YEARS:
                window = PIT_WINDOWS[year]
                year_root = LOCAL_RUN_ROOT / str(year)
                overrides = base_overrides(year, year_root, repo_pit_csv)
                write_json(DRIVE_RUN_ROOT / f"hydra_overrides_{year}.json", {"overrides": overrides})

                train_cmd = [sys.executable, "run_experiment.py", *overrides]
                run_stream(train_cmd, cwd=REPO_DIR, log_name=f"training_{year}.log", phase=f"training_{year}")

                run_dir = latest_run_dir(year_root, window["experiment_name"])
                pred_dir = run_dir / "averaged_predictions"
                if not pred_dir.exists():
                    raise FileNotFoundError(f"Missing averaged_predictions: {pred_dir}")

                training_row = {
                    "year": year,
                    "status": "OK",
                    "run_dir": str(run_dir),
                    "predictions_dir": str(pred_dir),
                    "training_summary": read_json_if_exists(run_dir / "training_summary.json"),
                    "evaluation_summary": read_json_if_exists(run_dir / "evaluation_summary.json"),
                }
                training_rows.append(training_row)
                write_json(SUMMARY_DIR / "training_results.json", {"rows": training_rows})
                write_csv(
                    SUMMARY_DIR / "training_results.csv",
                    training_rows,
                    ["year", "status", "run_dir", "predictions_dir"],
                )

                env = os.environ.copy()
                env["MPLBACKEND"] = "Agg"
                backtest_cmd = [
                    sys.executable,
                    "tests/backtest_sp500_daily.py",
                    "--predictions_dir",
                    str(year_root / "averaged_predictions"),
                    "--data_file",
                    str(repo_market_csv),
                    "--pit_universe_csv",
                    str(repo_pit_csv),
                    "--top_k",
                    str(TOP_K),
                    "--test_start",
                    PIT_WINDOWS[year]["test_start"],
                    "--test_end",
                    PIT_WINDOWS[year]["test_end"],
                    "--label_t",
                    "5",
                    "--transaction_costs",
                    "--spread",
                    str(SPREAD_BPS),
                    "--slippage",
                    str(SLIPPAGE_BPS),
                    "--enable_rank_drop_gate",
                    "--min_rank_drop",
                    str(MIN_RANK_DROP),
                    "--auto_save",
                    "--backtest_suffix",
                    "_top10_rankdrop",
                ]
                backtest_cmd[backtest_cmd.index(str(year_root / "averaged_predictions"))] = str(pred_dir)
                run_stream(
                    backtest_cmd,
                    cwd=REPO_DIR,
                    log_name=f"backtest_{year}.log",
                    phase=f"backtest_{year}_rank-drop",
                    env=env,
                )

                backtest_dir = run_dir / "backtest_top10_rankdrop"
                backtest_row = {
                    "year": year,
                    "status": "OK",
                    "run_dir": str(run_dir),
                    "predictions_dir": str(pred_dir),
                    "backtest_dir": str(backtest_dir),
                    "backtest_metrics": read_json_if_exists(backtest_dir / "backtest_metrics.json"),
                }
                backtest_rows.append(backtest_row)
                write_json(SUMMARY_DIR / "backtest_results.json", {"rows": backtest_rows})
                write_csv(
                    SUMMARY_DIR / "backtest_results.csv",
                    backtest_rows,
                    ["year", "status", "run_dir", "predictions_dir", "backtest_dir"],
                )

            summary = {
                "status": "OK",
                "run_tag": RUN_TAG,
                "branch": BRANCH,
                "recipe_id": RECIPE_ID,
                "drive_run_root": str(DRIVE_RUN_ROOT),
                "local_run_root": str(LOCAL_RUN_ROOT),
                "manifest": str(manifest_path),
                "data_audit": data_audit,
                "training_rows": training_rows,
                "backtest_rows": backtest_rows,
            }
            write_json(DRIVE_RUN_ROOT / "run_summary.json", summary)
            write_heartbeat("OK", "complete", run_summary=str(DRIVE_RUN_ROOT / "run_summary.json"))
        except Exception as exc:
            write_heartbeat("FAILED", "failed", error=repr(exc))
            raise
        finally:
            if LOCAL_RUN_ROOT.exists():
                shutil.copytree(LOCAL_RUN_ROOT, ARTIFACT_DIR / "local_run_root", dirs_exist_ok=True)
            if IN_COLAB and runtime is not None:
                try:
                    runtime.unassign()
                except Exception as exc:
                    print("Manual Runtime > Disconnect and delete runtime may still be needed:", exc)
        """
    ),
    md("## 6. Final Summary"),
    code(
        r"""
        summary_path = DRIVE_RUN_ROOT / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print("Run summary has not been written yet.")

        print("Drive run root:", DRIVE_RUN_ROOT)
        print("Heartbeat:", HEARTBEAT_PATH)
        print("Manifest:", DRIVE_RUN_ROOT / MANIFEST_FILENAME)
        print("Training results:", SUMMARY_DIR / "training_results.csv")
        print("Backtest results:", SUMMARY_DIR / "backtest_results.csv")
        print("Rank-drop backtest suffix: _top10_rankdrop")
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
