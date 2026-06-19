"""Generate the issue #8 volatility-targeting ablation Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

OUT = Path("notebooks/volatility_targeting_ablation_colab.ipynb")


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
        # Issue #8 Volatility-Targeting Ablation Sweep

        This notebook isolates why the Harvey-style volatility-targeting feature
        family helped some PIT years but severely hurt 2023. It runs a staged
        sweep against the current PIT recipe, writes a manifest, trains selected
        variants, runs the cost-aware rank-gated saved-prediction backtest, and
        emits a variant-vs-baseline delta table.

        Default stage is `efficiency_probe`: a small 2022 same-seed
        baseline/candidate pair for timing the Colab optimization settings
        before resuming the full repeated-seed sweep. Switch `RUN_STAGE` to
        `repeated_seed_validation` for the narrow candidate set from the Issue #8
        handoff across 2022-2025 and three seeds.
        """
    ),
    md("## 1. Setup"),
    code(
        r"""
        import hashlib
        import json
        import os
        import shutil
        import subprocess
        import sys
        from datetime import datetime
        from pathlib import Path

        import pandas as pd

        IN_COLAB = "google.colab" in sys.modules
        REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
        BRANCH = "codex/issue8-colab-efficiency-experiment"
        REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()
        REQUIRE_G4_GPU = True

        def require_expected_gpu() -> None:
            try:
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                gpu_name = proc.stdout.strip()
            except FileNotFoundError:
                gpu_name = ""
            print("GPU:", gpu_name or "not detected")
            if REQUIRE_G4_GPU and (not gpu_name or "T4" in gpu_name.upper()):
                raise RuntimeError(
                    "Issue #8 full ablation expects a G4/L4-class Colab runtime, not T4/CPU. "
                    "Use Runtime > Change runtime type before continuing."
                )

        if IN_COLAB:
            from google.colab import drive

            drive.mount("/content/drive")
            require_expected_gpu()
            if not REPO_DIR.exists():
                subprocess.run(
                    ["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)],
                    check=True,
                )
            else:
                subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=True)
                subprocess.run(
                    ["git", "-C", str(REPO_DIR), "checkout", "-B", BRANCH, f"origin/{BRANCH}"],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH],
                    check=True,
                )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")],
                check=True,
            )
        else:
            print("Local notebook generation/test mode; skipping Drive/GPU setup.")

        os.chdir(REPO_DIR)
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))

        print("Repo:", REPO_DIR)
        print("Branch:", BRANCH)
        subprocess.run(["git", "rev-parse", "HEAD"], check=False)
        """
    ),
    md("## 2. FRED Key And PIT Inputs"),
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

        def sha256_file(path: Path) -> str:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        drive_data_dir = Path("/content/drive/MyDrive/MCI_GRU_shared/data")
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

        print("Market CSV:", repo_market_csv, sha256_file(repo_market_csv)[:16])
        print("PIT CSV:", repo_pit_csv, sha256_file(repo_pit_csv)[:16])
        """
    ),
    md("## 3. Build Ablation Sweep"),
    code(
        r"""
        SMOKE_MODE = False
        RUN_STAGE = "efficiency_probe"  # efficiency_probe | repeated_seed_validation | stage1_2023 | stage2_contrasts | all_years
        EFFICIENCY_PROBE = RUN_STAGE == "efficiency_probe"
        BASE_SEED = 314159
        RUN_TAG_OVERRIDE = ""
        RUN_TAG = RUN_TAG_OVERRIDE.strip() or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        RUN_ROOT = Path("/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_ablation") / RUN_TAG
        DRIVE_TRAINING_OUTPUT_DIR = RUN_ROOT / "training"
        LOCAL_RUN_ROOT = (
            Path("/content/mci-gru-issue8-ablation-work") / RUN_TAG
            if IN_COLAB
            else RUN_ROOT
        )
        LOCAL_TRAINING_OUTPUT_DIR = LOCAL_RUN_ROOT / "training"
        TRAINING_OUTPUT_DIR = LOCAL_TRAINING_OUTPUT_DIR
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        LOCAL_TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        NUM_MODELS = 1 if SMOKE_MODE else (5 if EFFICIENCY_PROBE else 20)
        NUM_EPOCHS = 2 if SMOKE_MODE else 100
        PATIENCE = 2 if SMOKE_MODE else (10 if EFFICIENCY_PROBE else 15)
        BOOTSTRAP_RESAMPLES = 25 if SMOKE_MODE else 1000
        BATCH_SIZE = 64 if EFFICIENCY_PROBE else 32
        TEST_BATCH_SIZE = 32 if EFFICIENCY_PROBE else 1
        DATALOADER_NUM_WORKERS = 0
        DATALOADER_PIN_MEMORY = True
        DATALOADER_PERSISTENT_WORKERS = False
        DATALOADER_PREFETCH_FACTOR = None
        SAVE_MEMBER_PREDICTIONS = False
        SAVE_CHECKPOINTS = not EFFICIENCY_PROBE

        YEARS_BY_STAGE = {
            "efficiency_probe": [2022],
            "repeated_seed_validation": [2022, 2023, 2024, 2025],
            "stage1_2023": [2023],
            "stage2_contrasts": [2024, 2025],
            "all_years": [2022, 2023, 2024, 2025],
        }

        SEEDS_BY_STAGE = {
            "efficiency_probe": [BASE_SEED],
            "repeated_seed_validation": [314159, 271828, 161803],
            "stage1_2023": [BASE_SEED],
            "stage2_contrasts": [BASE_SEED],
            "all_years": [BASE_SEED],
        }

        VARIANTS_BY_STAGE = {
            "efficiency_probe": [
                "baseline_vol",
                "vt_scale_only",
            ],
            "repeated_seed_validation": [
                "baseline_vol",
                "vt_full_clip_0p25_4p0",
                "vt_clip_0p50_2p0",
                "vt_scale_only",
                "vt_ewm_only",
            ],
            "stage1_2023": [
                "baseline_vol",
                "vt_full_clip_0p25_4p0",
                "vt_no_scaled_return",
                "vt_ewm_only",
                "vt_scale_only",
                "vt_no_dynamics",
                "vt_clip_0p50_2p0",
                "vt_clip_0p75_1p5",
            ],
            "stage2_contrasts": [
                "baseline_vol",
                "vt_full_clip_0p25_4p0",
                "vt_no_scaled_return",
                "vt_ewm_only",
                "vt_clip_0p50_2p0",
            ],
            "all_years": [
                "baseline_vol",
                "vt_full_clip_0p25_4p0",
                "vt_no_scaled_return",
                "vt_ewm_only",
                "vt_scale_only",
                "vt_no_dynamics",
                "vt_clip_0p50_2p0",
                "vt_clip_0p75_1p5",
            ],
        }

        VARIANT_DEFS = {
            "baseline_vol": {
                "include_vol_targeting": False,
                "components": {},
                "scale_clip": [0.25, 4.0],
                "hypothesis": "Control: current realized-volatility features without issue #8 features.",
            },
            "vt_full_clip_0p25_4p0": {
                "include_vol_targeting": True,
                "components": {
                    "ewm_vol": True,
                    "scale": True,
                    "dynamics": True,
                    "scaled_return": True,
                },
                "scale_clip": [0.25, 4.0],
                "hypothesis": "Current issue #8 implementation and Harvey-style guard rail.",
            },
            "vt_no_scaled_return": {
                "include_vol_targeting": True,
                "components": {
                    "ewm_vol": True,
                    "scale": True,
                    "dynamics": True,
                    "scaled_return": False,
                },
                "scale_clip": [0.25, 4.0],
                "hypothesis": "Tests whether the lagged return x scale interaction caused the 2023 displacement.",
            },
            "vt_ewm_only": {
                "include_vol_targeting": True,
                "components": {
                    "ewm_vol": True,
                    "scale": False,
                    "dynamics": False,
                    "scaled_return": False,
                },
                "scale_clip": [0.25, 4.0],
                "hypothesis": "Keeps ex ante volatility state but removes target-scale proxies.",
            },
            "vt_scale_only": {
                "include_vol_targeting": True,
                "components": {
                    "ewm_vol": False,
                    "scale": True,
                    "dynamics": False,
                    "scaled_return": False,
                },
                "scale_clip": [0.25, 4.0],
                "hypothesis": "Isolates clipped target-vol scale proxies.",
            },
            "vt_no_dynamics": {
                "include_vol_targeting": True,
                "components": {
                    "ewm_vol": True,
                    "scale": True,
                    "dynamics": False,
                    "scaled_return": True,
                },
                "scale_clip": [0.25, 4.0],
                "hypothesis": "Tests whether vol-change and vol-of-vol destabilized 2023 ranks.",
            },
            "vt_clip_0p50_2p0": {
                "include_vol_targeting": True,
                "components": {
                    "ewm_vol": True,
                    "scale": True,
                    "dynamics": True,
                    "scaled_return": True,
                },
                "scale_clip": [0.50, 2.0],
                "hypothesis": "Moderates the target-scale guard rail while retaining all components.",
            },
            "vt_clip_0p75_1p5": {
                "include_vol_targeting": True,
                "components": {
                    "ewm_vol": True,
                    "scale": True,
                    "dynamics": True,
                    "scaled_return": True,
                },
                "scale_clip": [0.75, 1.5],
                "hypothesis": "Near-neutral scale proxy; should show whether scale magnitude is the problem.",
            },
        }

        COMPONENT_ORDER = ["ewm_vol", "scale", "dynamics", "scaled_return"]

        def bool_override(value: bool) -> str:
            return "true" if value else "false"

        def component_list_override(component_flags: dict[str, bool]) -> str:
            components = [name for name in COMPONENT_ORDER if component_flags[name]]
            return "[" + ",".join(components) + "]"

        def variant_overrides(name: str) -> list[str]:
            variant = VARIANT_DEFS[name]
            components = {
                "ewm_vol": True,
                "scale": True,
                "dynamics": True,
                "scaled_return": True,
                **variant["components"],
            }
            return [
                "features.include_volatility=true",
                f"features.include_volatility_targeting={bool_override(variant['include_vol_targeting'])}",
                "features.volatility_targeting_half_lives=[20,60,90]",
                "features.volatility_target_vol=0.10",
                f"features.volatility_target_scale_clip=[{variant['scale_clip'][0]},{variant['scale_clip'][1]}]",
                "features.volatility_targeting_interaction_return_window=21",
                f"features.volatility_targeting_components={component_list_override(components)}",
            ]

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
            f"training.batch_size={BATCH_SIZE}",
            f"training.test_batch_size={TEST_BATCH_SIZE}",
            f"training.dataloader_num_workers={DATALOADER_NUM_WORKERS}",
            f"training.dataloader_pin_memory={bool_override(DATALOADER_PIN_MEMORY)}",
            f"training.dataloader_persistent_workers={bool_override(DATALOADER_PERSISTENT_WORKERS)}",
            f"training.num_epochs={NUM_EPOCHS}",
            f"training.num_models={NUM_MODELS}",
            f"training.early_stopping_patience={PATIENCE}",
            "training.loss_type=ic",
            "training.label_type=returns",
            "training.selection_metric=val_ic",
            "training.shuffle_train=true",
            f"training.save_member_predictions={bool_override(SAVE_MEMBER_PREDICTIONS)}",
            f"training.save_checkpoints={bool_override(SAVE_CHECKPOINTS)}",
            "model.label_t=5",
            "model.temporal_encoder=gru_attn",
            f"evaluation.bootstrap_resamples={BOOTSTRAP_RESAMPLES}",
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
        if DATALOADER_PREFETCH_FACTOR is not None:
            BASE_OVERRIDES.append(
                f"training.dataloader_prefetch_factor={DATALOADER_PREFETCH_FACTOR}"
            )

        years = YEARS_BY_STAGE[RUN_STAGE]
        variant_names = VARIANTS_BY_STAGE[RUN_STAGE]
        seeds = SEEDS_BY_STAGE[RUN_STAGE]
        jobs = []
        for year in years:
            for seed in seeds:
                for variant_name in variant_names:
                    experiment = f"pit_temporal_{year}"
                    name = f"issue8_ablate_{variant_name}_{year}_seed{seed}"
                    jobs.append(
                        {
                            "year": year,
                            "seed": seed,
                            "variant": variant_name,
                            "name": name,
                            "hypothesis": VARIANT_DEFS[variant_name]["hypothesis"],
                            "overrides": [
                                f"+experiment={experiment}",
                                *BASE_OVERRIDES,
                                *variant_overrides(variant_name),
                                f"seed={seed}",
                                f"experiment_name={name}",
                                f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                            ],
                        }
                    )

        manifest = {
            "issue": 8,
            "branch": BRANCH,
            "run_tag": RUN_TAG,
            "run_stage": RUN_STAGE,
            "smoke_mode": SMOKE_MODE,
            "num_models": NUM_MODELS,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "test_batch_size": TEST_BATCH_SIZE,
            "dataloader_num_workers": DATALOADER_NUM_WORKERS,
            "dataloader_pin_memory": DATALOADER_PIN_MEMORY,
            "dataloader_persistent_workers": DATALOADER_PERSISTENT_WORKERS,
            "dataloader_prefetch_factor": DATALOADER_PREFETCH_FACTOR,
            "save_member_predictions": SAVE_MEMBER_PREDICTIONS,
            "save_checkpoints": SAVE_CHECKPOINTS,
            "drive_training_output_dir": str(DRIVE_TRAINING_OUTPUT_DIR),
            "local_training_output_dir": str(LOCAL_TRAINING_OUTPUT_DIR),
            "base_seed": BASE_SEED,
            "seeds": seeds,
            "years": years,
            "variants": {name: VARIANT_DEFS[name] for name in variant_names},
            "jobs": jobs,
            "market_csv": str(repo_market_csv),
            "pit_universe_csv": str(repo_pit_csv),
            "market_csv_sha256": sha256_file(repo_market_csv),
            "pit_universe_csv_sha256": sha256_file(repo_pit_csv),
        }
        manifest_path = RUN_ROOT / "issue8_vol_targeting_ablation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (RUN_ROOT / "pit_masked_panel_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print("Run root:", RUN_ROOT)
        print("Stage:", RUN_STAGE)
        print("Seeds:", seeds)
        print("Jobs:", len(jobs))
        for job in jobs:
            print("-", job["name"], "|", job["hypothesis"])
        print("Manifest:", manifest_path)
        """
    ),
    md("## 4. Train, Backtest, And Write Deltas"),
    code(
        r"""
        PIT_WINDOWS = {
            2022: ("2022-01-22", "2022-12-31"),
            2023: ("2023-01-22", "2023-12-31"),
            2024: ("2024-01-22", "2024-12-31"),
            2025: ("2025-01-22", "2025-12-31"),
        }
        BACKTEST_SUFFIX = "_pit_daily_tc_rank_gate"
        SPREAD_BPS = 10.0
        SLIPPAGE_BPS = 5.0
        MIN_RANK_DROP = 30
        TOP_K = 10
        LABEL_T = 5
        ADJUSTMENT_METHOD = "bhy"
        RESUME_COMPLETED_TRAINING = True

        results_path = RUN_ROOT / "issue8_vol_targeting_ablation_results.csv"
        deltas_path = RUN_ROOT / "issue8_vol_targeting_ablation_deltas_vs_baseline.csv"
        seed_summary_path = RUN_ROOT / "issue8_vol_targeting_repeated_seed_summary_by_variant.csv"
        year_variant_summary_path = RUN_ROOT / "issue8_vol_targeting_repeated_seed_summary_by_year_variant.csv"
        summary_md_path = RUN_ROOT / "issue8_vol_targeting_ablation_summary.md"

        def latest_run_dir(
            experiment_name: str,
            training_root: Path = TRAINING_OUTPUT_DIR,
        ) -> Path | None:
            base = training_root / experiment_name
            if not base.exists():
                return None
            candidates = sorted(path for path in base.iterdir() if path.is_dir())
            return candidates[-1] if candidates else None

        def drive_mirror_for_run_dir(run_dir: Path) -> Path:
            try:
                rel = run_dir.resolve().relative_to(LOCAL_TRAINING_OUTPUT_DIR.resolve())
            except ValueError:
                return run_dir
            return DRIVE_TRAINING_OUTPUT_DIR / rel

        def sync_run_to_drive(run_dir: Path) -> Path:
            drive_run_dir = drive_mirror_for_run_dir(run_dir)
            if drive_run_dir == run_dir:
                return run_dir
            print("Syncing completed run to Drive:", drive_run_dir)
            drive_run_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_dir, drive_run_dir, dirs_exist_ok=True)
            return drive_run_dir

        def tail(path: Path, n: int = 60) -> str:
            if not path.exists():
                return ""
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])

        def run_training(job: dict) -> Path:
            existing = latest_run_dir(job["name"], DRIVE_TRAINING_OUTPUT_DIR)
            if (
                RESUME_COMPLETED_TRAINING
                and existing is not None
                and (existing / "averaged_predictions").is_dir()
            ):
                print("Skipping completed Drive training:", existing)
                return existing

            existing = latest_run_dir(job["name"], LOCAL_TRAINING_OUTPUT_DIR)
            if (
                RESUME_COMPLETED_TRAINING
                and existing is not None
                and (existing / "averaged_predictions").is_dir()
            ):
                print("Skipping completed local training:", existing)
                return existing

            print("=" * 100)
            print("Training:", job["name"])
            cmd = [sys.executable, "-u", str(REPO_DIR / "run_experiment.py"), *job["overrides"]]
            logs_dir = RUN_ROOT / "logs" / job["name"] / "training"
            logs_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = logs_dir / "stdout.log"
            stderr_path = logs_dir / "stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w",
                encoding="utf-8",
            ) as stderr:
                proc = subprocess.run(cmd, cwd=str(REPO_DIR), stdout=stdout, stderr=stderr, text=True)
            print("Training return code:", proc.returncode)
            print("Training stdout:", stdout_path)
            print("Training stderr:", stderr_path)
            if proc.returncode != 0:
                print(tail(stdout_path))
                print(tail(stderr_path))
                raise RuntimeError(f"Training failed for {job['name']}")

            run_dir = latest_run_dir(job["name"], LOCAL_TRAINING_OUTPUT_DIR)
            if run_dir is None or not (run_dir / "averaged_predictions").is_dir():
                raise FileNotFoundError(f"Missing averaged_predictions for {job['name']}")
            return run_dir

        def run_backtest(job: dict, run_dir: Path) -> dict:
            test_start, test_end = PIT_WINDOWS[job["year"]]
            predictions_dir = run_dir / "averaged_predictions"
            backtest_dir = run_dir / f"backtest{BACKTEST_SUFFIX}"
            result_csv = backtest_dir / "backtest_results.csv"
            if RESUME_COMPLETED_TRAINING and result_csv.exists():
                print("Skipping completed backtest:", result_csv)
            else:
                print("Backtesting:", job["name"])
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
                    test_start,
                    "--test_end",
                    test_end,
                    "--top_k",
                    str(TOP_K),
                    "--label_t",
                    str(LABEL_T),
                    "--num_tests",
                    "1",
                    "--adjustment_method",
                    ADJUSTMENT_METHOD,
                    "--auto_save",
                    "--backtest_suffix",
                    BACKTEST_SUFFIX,
                    "--transaction_costs",
                    "--spread",
                    str(SPREAD_BPS),
                    "--slippage",
                    str(SLIPPAGE_BPS),
                    "--enable_rank_drop_gate",
                    "--min_rank_drop",
                    str(MIN_RANK_DROP),
                ]
                logs_dir = RUN_ROOT / "logs" / job["name"] / "backtest"
                logs_dir.mkdir(parents=True, exist_ok=True)
                stdout_path = logs_dir / "stdout.log"
                stderr_path = logs_dir / "stderr.log"
                backtest_env = os.environ.copy()
                existing_pythonpath = backtest_env.get("PYTHONPATH", "")
                backtest_env["PYTHONPATH"] = (
                    f"{REPO_DIR}{os.pathsep}{existing_pythonpath}"
                    if existing_pythonpath
                    else str(REPO_DIR)
                )
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                    "w",
                    encoding="utf-8",
                ) as stderr:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(REPO_DIR),
                        env=backtest_env,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                    )
                print("Backtest return code:", proc.returncode)
                print("Backtest stdout:", stdout_path)
                print("Backtest stderr:", stderr_path)
                if proc.returncode != 0:
                    print(tail(stdout_path))
                    print(tail(stderr_path))
                    raise RuntimeError(f"Backtest failed for {job['name']}")

            row = pd.read_csv(result_csv).iloc[0].to_dict()
            row.update(
                {
                    "year": job["year"],
                    "seed": job["seed"],
                    "variant": job["variant"],
                    "experiment_name": job["name"],
                    "run_dir": str(run_dir),
                    "predictions_dir": str(predictions_dir),
                    "backtest_dir": str(backtest_dir),
                    "hypothesis": job["hypothesis"],
                }
            )
            return row

        def build_deltas(results_df: pd.DataFrame) -> pd.DataFrame:
            metric_cols = [
                "total_return",
                "excess_return",
                "ARR",
                "ASR",
                "MDD",
                "avg_daily_turnover",
                "days_with_gate_exits",
                "days_skipped_by_rank_gate",
            ]
            rows = []
            for (year, seed), year_df in results_df.groupby(["year", "seed"]):
                baseline = year_df[year_df["variant"] == "baseline_vol"]
                if baseline.empty:
                    continue
                baseline_row = baseline.iloc[0]
                for _, row in year_df.iterrows():
                    out = {
                        "year": year,
                        "seed": seed,
                        "variant": row["variant"],
                        "run_dir": row["run_dir"],
                    }
                    for metric in metric_cols:
                        if metric in row and metric in baseline_row:
                            out[f"{metric}_vs_baseline"] = row[metric] - baseline_row[metric]
                    rows.append(out)
            return pd.DataFrame(rows)

        def write_repeated_seed_summaries(deltas_df: pd.DataFrame) -> None:
            if deltas_df.empty:
                pd.DataFrame().to_csv(seed_summary_path, index=False)
                pd.DataFrame().to_csv(year_variant_summary_path, index=False)
                return
            summary_metrics = [
                col
                for col in [
                    "total_return_vs_baseline",
                    "ASR_vs_baseline",
                    "MDD_vs_baseline",
                    "avg_daily_turnover_vs_baseline",
                ]
                if col in deltas_df.columns
            ]
            if not summary_metrics:
                pd.DataFrame().to_csv(seed_summary_path, index=False)
                pd.DataFrame().to_csv(year_variant_summary_path, index=False)
                return
            by_variant = (
                deltas_df.groupby("variant")[summary_metrics]
                .agg(["count", "mean", "median", "min", "max"])
                .reset_index()
            )
            by_variant.columns = [
                "_".join(str(part) for part in col if part).rstrip("_")
                if isinstance(col, tuple)
                else str(col)
                for col in by_variant.columns
            ]
            by_year_variant = (
                deltas_df.groupby(["year", "variant"])[summary_metrics]
                .agg(["count", "mean", "median", "min", "max"])
                .reset_index()
            )
            by_year_variant.columns = [
                "_".join(str(part) for part in col if part).rstrip("_")
                if isinstance(col, tuple)
                else str(col)
                for col in by_year_variant.columns
            ]
            by_variant.to_csv(seed_summary_path, index=False)
            by_year_variant.to_csv(year_variant_summary_path, index=False)

        rows = []
        for job in jobs:
            run_dir = run_training(job)
            row = run_backtest(job, run_dir)
            drive_run_dir = sync_run_to_drive(run_dir)
            row.update(
                {
                    "local_run_dir": str(run_dir),
                    "drive_run_dir": str(drive_run_dir),
                    "run_dir": str(drive_run_dir),
                    "predictions_dir": str(drive_run_dir / "averaged_predictions"),
                    "backtest_dir": str(drive_run_dir / f"backtest{BACKTEST_SUFFIX}"),
                }
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            interim_deltas = build_deltas(pd.DataFrame(rows))
            interim_deltas.to_csv(deltas_path, index=False)
            write_repeated_seed_summaries(interim_deltas)

        results_df = pd.DataFrame(rows)
        deltas_df = build_deltas(results_df)
        results_df.to_csv(results_path, index=False)
        deltas_df.to_csv(deltas_path, index=False)
        write_repeated_seed_summaries(deltas_df)

        display_cols = [
            col
            for col in [
                "year",
                "seed",
                "variant",
                "total_return",
                "excess_return",
                "ARR",
                "ASR",
                "MDD",
                "avg_daily_turnover",
            ]
            if col in results_df.columns
        ]
        print("Results:")
        display(results_df[display_cols].sort_values(["year", "total_return"], ascending=[True, False]))
        print("Deltas vs baseline:")
        display(deltas_df.sort_values(["year", "total_return_vs_baseline"], ascending=[True, False]))

        summary_lines = [
            "# Issue #8 Volatility-Targeting Ablation Sweep",
            "",
            f"- Run root: `{RUN_ROOT}`",
            f"- Stage: `{RUN_STAGE}`",
            f"- Smoke mode: `{SMOKE_MODE}`",
            f"- Models per job: `{NUM_MODELS}`",
            f"- Epochs per job: `{NUM_EPOCHS}`",
            f"- Batch size: `{BATCH_SIZE}`",
            f"- Test batch size: `{TEST_BATCH_SIZE}`",
            f"- DataLoader: workers={DATALOADER_NUM_WORKERS}, pin_memory={DATALOADER_PIN_MEMORY}, persistent_workers={DATALOADER_PERSISTENT_WORKERS}, prefetch_factor={DATALOADER_PREFETCH_FACTOR}",
            f"- Save per-member predictions: `{SAVE_MEMBER_PREDICTIONS}`",
            f"- Save checkpoints: `{SAVE_CHECKPOINTS}`",
            f"- Local training output: `{LOCAL_TRAINING_OUTPUT_DIR}`",
            f"- Drive training output: `{DRIVE_TRAINING_OUTPUT_DIR}`",
            f"- Seeds: `{seeds}`",
            f"- Backtest: top_k={TOP_K}, label_t={LABEL_T}, spread={SPREAD_BPS}, slippage={SLIPPAGE_BPS}, rank gate={MIN_RANK_DROP}",
            f"- Results CSV: `{results_path}`",
            f"- Delta CSV: `{deltas_path}`",
            f"- Seed summary CSV: `{seed_summary_path}`",
            f"- Year-variant summary CSV: `{year_variant_summary_path}`",
            "",
            "## Variant Hypotheses",
        ]
        for name in variant_names:
            summary_lines.append(f"- `{name}`: {VARIANT_DEFS[name]['hypothesis']}")
        if not deltas_df.empty:
            summary_lines.extend(["", "## Deltas Vs Baseline", ""])
            summary_lines.append(deltas_df.to_markdown(index=False))
        summary_md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print("Summary:", summary_md_path)
        """
    ),
]


def build_notebook() -> dict:
    return {
        "cells": cells,
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
    OUT.write_text(json.dumps(build_notebook(), indent=1), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
