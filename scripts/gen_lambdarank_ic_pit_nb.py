"""Generate the Colab notebook for LambdaRankIC PIT comparison runs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from textwrap import dedent

OUT = Path("notebooks/lambdarank_ic_pit_colab.ipynb")
FULL_TRANCHE_OUT = Path("notebooks/lambdarank_ic_full_tranche_colab.ipynb")


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
        # LambdaRankIC Pairwise Rank IC PIT Grid

        This notebook turns `docs/research/current/LOSS_PATH_DECISION_2026-06-04.md`
        into a Colab-ready PIT comparison. It keeps the frozen pure-IC launch
        candidate in the grid, includes the Portfolio-IC hybrid as the current
        portfolio-utility branch, and adds the disabled-by-default
        `lambdarank_ic` candidate.

        Recipe reference: `docs/DEFAULT_EXPERIMENT_RECIPE.md`.

        Default mode is a lower-pair screen: one year, one seed, one model,
        40 epochs, patience 8, and LambdaRankIC pair caps
        `[512, 1024, 2048, 4096]`. This is intentionally cheaper than the full
        20-model ensemble and should be labeled as screening evidence.

        Operate this notebook from the visible Colab UI on a G4/L4-class Colab
        runtime, not T4/CPU. Use Drive artifacts as truth if output streaming
        stalls; Drive API fallback is preferred over repeated DriveFS remounts
        for compact artifacts such as heartbeat and result files. If cleanup
        does not run, use `Runtime > Disconnect and delete runtime` manually.
        """
    ),
    md("## 1. Setup"),
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

        import torch

        IN_COLAB = "google.colab" in sys.modules
        REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
        BRANCH = "codex/colab-gpu-utilization-hardening-20260620"
        REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()
        REQUIRE_G4_L4_GPU = True
        BLOCKED_GPU_NAMES = ("T4",)
        ALLOWED_GPU_MARKERS = (
            "G4",
            "L4",
            "A100",
            "H100",
            "V100",
            "RTX PRO",
            "BLACKWELL",
        )
        STRICT_GPU_MARKERS: list[str] = []

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
            raise RuntimeError(
                "Expected G4/L4-class Colab runtime, not T4/CPU. "
                "Switch Runtime -> Change runtime type -> G4 GPU before training."
            )

        from mci_gru.config import TrainingConfig
        from mci_gru.training.losses import build_training_loss

        probe_cfg = TrainingConfig(loss_type="lambdarank_ic", selection_metric="val_rank_ic")
        probe_loss, probe_name = build_training_loss(probe_cfg)
        print("LambdaRankIC branch probe:", probe_name, type(probe_loss).__name__)
        """
    ),
    md("## 2. FRED Key And PIT Data"),
    code(
        r"""
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
            raise RuntimeError("FRED_API_KEY is required for the current regime-enabled preset.")

        drive_data_dir = Path("/content/drive/MyDrive/MCI_GRU_shared/data") if IN_COLAB else REPO_DIR / "data/raw/market"
        drive_market_csv = drive_data_dir / "sp500_pit_union_lseg_20150101_20260513.csv"
        drive_pit_csv = drive_data_dir / "sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv"

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

        print("Market CSV:", repo_market_csv)
        print("PIT CSV:", repo_pit_csv)
        """
    ),
    md("## 3. Build Objective Grid"),
    code(
        r"""
        SMOKE_MODE = False
        SCREEN_MODE = True
        RUN_TRAINING = True
        MAX_JOBS = None
        SMOKE_YEARS = [2022]
        SCREEN_YEARS = [2022]
        FULL_YEARS = [2022, 2023, 2024, 2025]
        SMOKE_BASE_SEEDS = [314159]
        SCREEN_BASE_SEEDS = [314159]
        FULL_BASE_SEEDS = [314159, 271828, 161803]
        SMOKE_PAIR_CAPS = [512]
        SCREEN_PAIR_CAPS = [512, 1024, 2048, 4096]
        FULL_PAIR_CAPS = [4096]
        SMOKE_NUM_MODELS = 1
        SCREEN_NUM_MODELS = 1
        FULL_NUM_MODELS = 20
        SMOKE_NUM_EPOCHS = 2
        SCREEN_NUM_EPOCHS = 40
        FULL_NUM_EPOCHS = 100
        SMOKE_EARLY_STOPPING_PATIENCE = 2
        SCREEN_EARLY_STOPPING_PATIENCE = 8
        FULL_EARLY_STOPPING_PATIENCE = 15

        if SMOKE_MODE and SCREEN_MODE:
            raise ValueError("SMOKE_MODE and SCREEN_MODE are mutually exclusive.")

        BUDGET_MODE = "smoke" if SMOKE_MODE else ("screen" if SCREEN_MODE else "full")
        YEARS = SMOKE_YEARS if SMOKE_MODE else (SCREEN_YEARS if SCREEN_MODE else FULL_YEARS)
        BASE_SEEDS = (
            SMOKE_BASE_SEEDS
            if SMOKE_MODE
            else (SCREEN_BASE_SEEDS if SCREEN_MODE else FULL_BASE_SEEDS)
        )
        PAIR_CAPS = SCREEN_PAIR_CAPS if SCREEN_MODE else (SMOKE_PAIR_CAPS if SMOKE_MODE else FULL_PAIR_CAPS)
        NUM_MODELS = (
            SMOKE_NUM_MODELS
            if SMOKE_MODE
            else (SCREEN_NUM_MODELS if SCREEN_MODE else FULL_NUM_MODELS)
        )
        NUM_EPOCHS = (
            SMOKE_NUM_EPOCHS
            if SMOKE_MODE
            else (SCREEN_NUM_EPOCHS if SCREEN_MODE else FULL_NUM_EPOCHS)
        )
        EARLY_STOPPING_PATIENCE = (
            SMOKE_EARLY_STOPPING_PATIENCE
            if SMOKE_MODE
            else (
                SCREEN_EARLY_STOPPING_PATIENCE
                if SCREEN_MODE
                else FULL_EARLY_STOPPING_PATIENCE
            )
        )

        RUN_TAG = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        RUN_ROOT = (
            Path("/content/drive/MyDrive/MCI-GRU-Ablations/lambdarank_ic_pit")
            if IN_COLAB
            else REPO_DIR / "results" / "lambdarank_ic_pit"
        ) / RUN_TAG
        TRAINING_OUTPUT_DIR = RUN_ROOT / "training"
        HEARTBEAT_PATH = RUN_ROOT / "heartbeat.json"
        GPU_UTIL_PATH = RUN_ROOT / "gpu_util.csv"
        GPU_UTIL_STOP_PATH = RUN_ROOT / "gpu_util.stop"
        RESULTS_CSV_PATH = RUN_ROOT / "training_results.csv"
        RESULTS_JSON_PATH = RUN_ROOT / "training_results.json"
        LEGACY_RESULTS_JSON_PATH = RUN_ROOT / "lambdarank_ic_pit_training_results.json"
        if RUN_ROOT.exists():
            raise RuntimeError(f"Refusing to reuse existing run root: {RUN_ROOT}")
        RUN_ROOT.mkdir(parents=True, exist_ok=True)

        def write_heartbeat(
            phase: str,
            status: str = "RUNNING",
            current_job: str | None = None,
            completed_jobs: int = 0,
            error: str | None = None,
        ) -> None:
            payload = {
                "phase": phase,
                "status": status,
                "current_job": current_job,
                "completed_jobs": completed_jobs,
                "expected_jobs": None,
                "budget_mode": BUDGET_MODE,
                "branch": BRANCH,
                "run_root": str(RUN_ROOT),
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            if "jobs" in globals():
                payload["expected_jobs"] = len(jobs)
            if "GPU_NAME" in globals():
                payload["gpu_name"] = GPU_NAME
            if torch.cuda.is_available():
                payload["gpu"] = torch.cuda.get_device_name(0)
            if error is not None:
                payload["error"] = error
            HEARTBEAT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        def start_gpu_sampler() -> subprocess.Popen | None:
            if not RUN_TRAINING:
                return None
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

        OBJECTIVE_VARIANTS_FULL = {
            'pure_ic_baseline': {
                'loss_type': 'ic',
                'selection_metric': 'val_ic',
            },
            'portfolio_ic_hybrid': {
                'loss_type': 'portfolio_ic',
                'selection_metric': 'val_loss',
                'portfolio_ic_top_k': 10,
                'portfolio_ic_weight': 0.25,
                'portfolio_ic_temperature': 0.25,
            },
            'lambdarank_ic_candidate': {
                'loss_type': 'lambdarank_ic',
                'selection_metric': 'val_rank_ic',
                'lambdarank_ic_max_pairs_per_day': 4096,
                'lambdarank_ic_temperature': 1.0,
            },
        }
        OBJECTIVE_VARIANTS = (
            {
                'lambdarank_ic_pair_cap_screen': {
                    'loss_type': 'lambdarank_ic',
                    'selection_metric': 'val_rank_ic',
                    'lambdarank_ic_max_pairs_per_day': 4096,
                    'lambdarank_ic_temperature': 1.0,
                },
            }
            if SCREEN_MODE
            else OBJECTIVE_VARIANTS_FULL
        )

        pair_multiplier = sum(
            len(PAIR_CAPS) if variant["loss_type"] == "lambdarank_ic" else 1
            for variant in OBJECTIVE_VARIANTS.values()
        )
        EXPECTED_JOB_COUNT = len(YEARS) * len(BASE_SEEDS) * pair_multiplier
        EXPECTED_TOTAL_MODELS = EXPECTED_JOB_COUNT * NUM_MODELS
        expected_jobs_by_mode = len(SCREEN_PAIR_CAPS) if SCREEN_MODE else (3 if SMOKE_MODE else 36)
        expected_models_by_mode = len(SCREEN_PAIR_CAPS) if SCREEN_MODE else (3 if SMOKE_MODE else 720)
        assert EXPECTED_JOB_COUNT == expected_jobs_by_mode
        assert EXPECTED_TOTAL_MODELS == expected_models_by_mode

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
        ]

        def loss_overrides_for_variant(variant: dict) -> list[str]:
            overrides = [
                f"training.loss_type={variant['loss_type']}",
                f"training.selection_metric={variant['selection_metric']}",
            ]
            if variant["loss_type"] == "portfolio_ic":
                overrides.extend(
                    [
                        f"training.portfolio_ic_top_k={variant['portfolio_ic_top_k']}",
                        f"training.portfolio_ic_weight={variant['portfolio_ic_weight']}",
                        f"training.portfolio_ic_temperature={variant['portfolio_ic_temperature']}",
                    ]
                )
            if variant["loss_type"] == "lambdarank_ic":
                overrides.extend(
                    [
                        "training.lambdarank_ic_max_pairs_per_day="
                        f"{variant['lambdarank_ic_max_pairs_per_day']}",
                        "training.lambdarank_ic_temperature="
                        f"{variant['lambdarank_ic_temperature']}",
                    ]
                )
            return overrides

        jobs = []
        for year in YEARS:
            for base_seed in BASE_SEEDS:
                for variant_name, variant in OBJECTIVE_VARIANTS.items():
                    variant_pair_caps = PAIR_CAPS if variant["loss_type"] == "lambdarank_ic" else [None]
                    for max_pairs_per_day in variant_pair_caps:
                        experiment = f"pit_temporal_{year}"
                        job_variant = dict(variant)
                        if max_pairs_per_day is not None:
                            job_variant["lambdarank_ic_max_pairs_per_day"] = max_pairs_per_day
                        pair_suffix = (
                            f"_pairs{max_pairs_per_day}" if max_pairs_per_day is not None else ""
                        )
                        name = f"lambdarank_ic_{variant_name}{pair_suffix}_{year}_seed{base_seed}"
                        jobs.append(
                            {
                                "year": year,
                                "base_seed": base_seed,
                                "variant": variant_name,
                                "loss_type": job_variant["loss_type"],
                                "selection_metric": job_variant["selection_metric"],
                                "max_pairs_per_day": max_pairs_per_day,
                                "name": name,
                                "overrides": [
                                    f"+experiment={experiment}",
                                    *BASE_OVERRIDES,
                                    *loss_overrides_for_variant(job_variant),
                                    f"seed={base_seed}",
                                    f"experiment_name={name}",
                                    f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                                ],
                            }
                        )

        if MAX_JOBS is not None:
            jobs = jobs[:MAX_JOBS]

        ACTUAL_JOB_COUNT = len(jobs)
        ACTUAL_TOTAL_MODELS = ACTUAL_JOB_COUNT * NUM_MODELS

        manifest = {
            "research_basis": "LOSS_PATH_DECISION_2026-06-04.md",
            "recipe_id": FROZEN_RECIPE_ID,
            "branch": BRANCH,
            "run_tag": RUN_TAG,
            "smoke_mode": SMOKE_MODE,
            "screen_mode": SCREEN_MODE,
            "max_jobs": MAX_JOBS,
            "budget_mode": BUDGET_MODE,
            "years": YEARS,
            "base_seeds": BASE_SEEDS,
            "pair_caps": PAIR_CAPS,
            "num_models": NUM_MODELS,
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "expected_job_count": EXPECTED_JOB_COUNT,
            "expected_total_models": EXPECTED_TOTAL_MODELS,
            "actual_job_count": ACTUAL_JOB_COUNT,
            "actual_total_models": ACTUAL_TOTAL_MODELS,
            "objective_variants": OBJECTIVE_VARIANTS,
            "jobs": jobs,
        }
        manifest_path = RUN_ROOT / "lambdarank_ic_pit_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print("Run root:", RUN_ROOT)
        print("Recipe:", FROZEN_RECIPE_ID)
        print("Jobs:", ACTUAL_JOB_COUNT)
        print("Expected total models:", EXPECTED_TOTAL_MODELS)
        print("Actual total models:", ACTUAL_TOTAL_MODELS)
        for job in jobs:
            print("-", job["name"], job["loss_type"], job["selection_metric"])
        print("Manifest:", manifest_path)
        write_heartbeat("manifest", completed_jobs=0)
        """
    ),
    md("## 4. Run Training Jobs"),
    code(
        r"""
        def latest_training_summary(job_name: str) -> Path | None:
            candidates = sorted(TRAINING_OUTPUT_DIR.glob(f"{job_name}/**/training_summary.json"))
            return candidates[-1] if candidates else None

        def write_results_artifacts(rows: list[dict]) -> None:
            RESULTS_JSON_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            LEGACY_RESULTS_JSON_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            fieldnames = [
                "name",
                "variant",
                "loss_type",
                "selection_metric",
                "year",
                "base_seed",
                "max_pairs_per_day",
                "returncode",
                "elapsed_seconds",
                "training_summary_path",
            ]
            with RESULTS_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key) for key in fieldnames})

        results = []
        gpu_sampler_proc = start_gpu_sampler()
        try:
            if not RUN_TRAINING:
                print("RUN_TRAINING is false; grid manifest only.")
                write_heartbeat("skipped", status="OK", completed_jobs=0)
            for job in jobs if RUN_TRAINING else []:
                write_heartbeat(
                    "training",
                    current_job=job["name"],
                    completed_jobs=len(results),
                )
                print("=" * 100)
                print("Starting:", job["name"])
                cmd = [sys.executable, "-u", str(REPO_DIR / "run_experiment.py"), *job["overrides"]]
                print("Command:", " ".join(cmd[:4]), "... +", len(job["overrides"]), "overrides")
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                start_time = time.perf_counter()
                proc = subprocess.run(cmd, cwd=str(REPO_DIR), text=True, env=env)
                elapsed_seconds = time.perf_counter() - start_time
                summary_path = latest_training_summary(job["name"])
                result = {
                    "name": job["name"],
                    "variant": job["variant"],
                    "loss_type": job["loss_type"],
                    "selection_metric": job["selection_metric"],
                    "year": job["year"],
                    "base_seed": job["base_seed"],
                    "max_pairs_per_day": job["max_pairs_per_day"],
                    "returncode": int(proc.returncode),
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "training_summary_path": str(summary_path) if summary_path else None,
                }
                if summary_path is not None:
                    result["training_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
                results.append(result)
                write_results_artifacts(results)
                print("Return code:", proc.returncode)
                print("Training summary:", summary_path)
                if proc.returncode != 0:
                    raise RuntimeError(f"Job failed: {job['name']}")

            print("Training loop complete.")
            write_heartbeat("done", status="OK", completed_jobs=len(results))
        except Exception as exc:
            write_heartbeat(
                "failed",
                status="FAILED",
                current_job=job["name"] if "job" in locals() else None,
                completed_jobs=len(results),
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
                    print(
                        "Runtime > Disconnect and delete runtime manually if "
                        f"foreground cleanup did not complete: {exc}"
                    )

        print("Results:", RESULTS_JSON_PATH)
        print("Legacy results:", LEGACY_RESULTS_JSON_PATH)
        print("CSV results:", RESULTS_CSV_PATH)
        """
    ),
    md("## 5. Results Snapshot"),
    code(
        r"""
        results_path = RESULTS_JSON_PATH
        if results_path.exists():
            import pandas as pd

            df = pd.json_normalize(json.loads(results_path.read_text(encoding="utf-8")))
            keep_cols = [
                "variant",
                "loss_type",
                "selection_metric",
                "year",
                "base_seed",
                "returncode",
                "training_summary.mean_best_val_ic",
                "training_summary.mean_best_val_rank_ic",
                "training_summary.evaluation.avg_ic",
                "training_summary.evaluation.avg_rank_ic",
            ]
            display(df[[col for col in keep_cols if col in df.columns]])
        else:
            print("No training results file yet.")

        print("Primary go/no-go question:")
        print("Does lambdarank_ic improve Rank IC and net top-k/rank-drop backtests versus pure IC without increasing turnover, drawdown, or year instability?")
        """
    ),
]


def build_full_tranche_cells() -> list[dict]:
    tranche_cells = deepcopy(cells)
    tranche_cells[0] = md(
        """
        # LambdaRankIC Full-Recipe Confirmation Tranche

        This notebook is the next step after the lower-pair LambdaRankIC screen.
        It keeps the frozen pure-IC launch candidate, the Portfolio-IC hybrid,
        and the LambdaRankIC candidate in the grid, but runs only the first
        full-recipe tranche: year 2022, seed 314159, 20 models, 100 epochs,
        patience 15, and the screen-winning LambdaRankIC pair cap `[512]`.

        This is intentionally narrower than the whole 36-job / 720-model full
        grid. It should answer whether the screen winner survives ensemble
        scale before spending the full multi-year, multi-seed budget.

        Operate this notebook from the visible Colab UI on a G4/L4-class Colab
        runtime, not T4/CPU. Use Drive artifacts as truth if output streaming
        stalls; Drive API fallback is preferred over repeated DriveFS remounts
        for compact artifacts such as heartbeat and result files. If cleanup
        does not run, use `Runtime > Disconnect and delete runtime` manually.
        """
    )
    grid_source = "".join(tranche_cells[6]["source"])
    replacements = {
        "SCREEN_MODE = True": "SCREEN_MODE = False",
        "MAX_JOBS = None": "MAX_JOBS = 3",
        "FULL_PAIR_CAPS = [4096]": "FULL_PAIR_CAPS = [512]",
        "Path(\"/content/drive/MyDrive/MCI-GRU-Ablations/lambdarank_ic_pit\")": (
            "Path(\"/content/drive/MyDrive/MCI-GRU-Ablations/lambdarank_ic_full_tranche\")"
        ),
        'REPO_DIR / "results" / "lambdarank_ic_pit"': (
            'REPO_DIR / "results" / "lambdarank_ic_full_tranche"'
        ),
        'LEGACY_RESULTS_JSON_PATH = RUN_ROOT / "lambdarank_ic_pit_training_results.json"': (
            'LEGACY_RESULTS_JSON_PATH = RUN_ROOT / "lambdarank_ic_full_tranche_training_results.json"'
        ),
        'manifest_path = RUN_ROOT / "lambdarank_ic_pit_manifest.json"': (
            'manifest_path = RUN_ROOT / "lambdarank_ic_full_tranche_manifest.json"'
        ),
    }
    for old, new in replacements.items():
        if old not in grid_source:
            raise ValueError(f"Full-tranche source replacement missing: {old}")
        grid_source = grid_source.replace(old, new)
    tranche_cells[6]["source"] = grid_source.splitlines(keepends=True)
    return tranche_cells


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
    FULL_TRANCHE_OUT.write_text(
        json.dumps(build_notebook(build_full_tranche_cells()), indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {FULL_TRANCHE_OUT}")


if __name__ == "__main__":
    main()
