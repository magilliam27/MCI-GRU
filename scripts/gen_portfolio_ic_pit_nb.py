"""Generate the Colab notebook for Portfolio-IC hybrid PIT grid runs."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

OUT = Path("notebooks/portfolio_ic_pit_colab.ipynb")


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
        # Portfolio-IC Hybrid Loss PIT Grid

        This notebook compares the frozen pure-IC baseline against the first
        Portfolio-IC Hybrid Loss implementation. The Portfolio-IC variant keeps
        IC as the anchor and adds a differentiable soft top-10 forward-return
        utility term.

        Research basis includes Machine Learning Meets Markowitz and the
        direct portfolio-objective / differentiable ranking literature already
        summarized in the implementation plan.

        The baseline recipe is still the frozen default:
        `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1`.
        See `docs/DEFAULT_EXPERIMENT_RECIPE.md`.

        Default mode is a mechanics smoke: one year, one seed, one model, one
        epoch. Set `SMOKE_MODE = False` only when launching the full explicit
        2022-2025 x 3-seed x 2-objective grid.
        """
    ),
    md("## 1. Setup"),
    code(
        r"""
        import json
        import os
        import shutil
        import subprocess
        import sys
        from datetime import datetime
        from pathlib import Path

        try:
            from google.colab import drive, userdata

            IN_COLAB = True
        except ImportError:
            drive = None
            userdata = None
            IN_COLAB = False

        REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
        BRANCH = "codex/portfolio-ic-hybrid-testing"
        REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()

        if IN_COLAB:
            drive.mount("/content/drive")
            if not REPO_DIR.exists():
                subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)], check=True)
            else:
                subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "-B", BRANCH, f"origin/{BRANCH}"], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)], check=True)

        os.chdir(REPO_DIR)
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))

        print("Repo:", REPO_DIR)
        print("Branch:", BRANCH)
        subprocess.run(["git", "rev-parse", "HEAD"], check=False)
        """
    ),
    md("## 2. Data And Regime Inputs"),
    code(
        r"""
        if IN_COLAB and not os.environ.get("FRED_API_KEY"):
            try:
                secret = userdata.get("FRED_API_KEY") if userdata is not None else None
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
        SMOKE_MODE = True
        SMOKE_YEARS = [2025]
        FULL_YEARS = [2022, 2023, 2024, 2025]
        SMOKE_BASE_SEEDS = [314159]
        FULL_BASE_SEEDS = [314159, 271828, 161803]

        YEARS = SMOKE_YEARS if SMOKE_MODE else FULL_YEARS
        BASE_SEEDS = SMOKE_BASE_SEEDS if SMOKE_MODE else FULL_BASE_SEEDS
        NUM_MODELS = 1 if SMOKE_MODE else 20
        NUM_EPOCHS = 1 if SMOKE_MODE else 100
        EARLY_STOPPING_PATIENCE = 2 if SMOKE_MODE else 15

        RUN_TAG = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        RUN_ROOT = (
            Path("/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid")
            if IN_COLAB
            else REPO_DIR / "results" / "portfolio_ic_hybrid"
        ) / RUN_TAG
        TRAINING_OUTPUT_DIR = RUN_ROOT / "training"
        RUN_ROOT.mkdir(parents=True, exist_ok=True)

        FROZEN_RECIPE_ID = (
            "static-threshold-shuffle__pure-ic-returns-5d-val-ic__"
            "regime-current-only__ensemble__drop-edge-0p1"
        )

        OBJECTIVE_VARIANTS = {
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
        }

        EXPECTED_JOB_COUNT = len(YEARS) * len(BASE_SEEDS) * len(OBJECTIVE_VARIANTS)
        EXPECTED_TOTAL_MODELS = EXPECTED_JOB_COUNT * NUM_MODELS
        assert EXPECTED_JOB_COUNT == (2 if SMOKE_MODE else 24)
        assert EXPECTED_TOTAL_MODELS == (2 if SMOKE_MODE else 480)

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
            return overrides

        jobs = []
        for year in YEARS:
            for base_seed in BASE_SEEDS:
                for variant_name, variant in OBJECTIVE_VARIANTS.items():
                    experiment = f"pit_temporal_{year}"
                    name = f"portfolio_ic_{variant_name}_{year}_seed{base_seed}"
                    jobs.append(
                        {
                            "year": year,
                            "base_seed": base_seed,
                            "variant": variant_name,
                            "loss_type": variant["loss_type"],
                            "selection_metric": variant["selection_metric"],
                            "name": name,
                            "overrides": [
                                f"+experiment={experiment}",
                                *BASE_OVERRIDES,
                                *loss_overrides_for_variant(variant),
                                f"seed={base_seed}",
                                f"experiment_name={name}",
                                f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                            ],
                        }
                    )

        manifest = {
            "research_basis": "Machine Learning Meets Markowitz",
            "recipe_id": FROZEN_RECIPE_ID,
            "branch": BRANCH,
            "run_tag": RUN_TAG,
            "smoke_mode": SMOKE_MODE,
            "years": YEARS,
            "base_seeds": BASE_SEEDS,
            "num_models": NUM_MODELS,
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "expected_job_count": EXPECTED_JOB_COUNT,
            "expected_total_models": EXPECTED_TOTAL_MODELS,
            "objective_variants": OBJECTIVE_VARIANTS,
            "jobs": jobs,
        }
        manifest_path = RUN_ROOT / "portfolio_ic_pit_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print("Run root:", RUN_ROOT)
        print("Recipe:", FROZEN_RECIPE_ID)
        print("Jobs:", len(jobs))
        print("Expected total models:", EXPECTED_TOTAL_MODELS)
        for job in jobs:
            print("-", job["name"], job["loss_type"], job["selection_metric"])
        print("Manifest:", manifest_path)
        """
    ),
    md("## 4. Run Training Jobs"),
    code(
        r"""
        results = []
        for job in jobs:
            print("=" * 100)
            print("Starting:", job["name"])
            cmd = [sys.executable, "-u", str(REPO_DIR / "run_experiment.py"), *job["overrides"]]
            print("Command:", " ".join(cmd[:4]), "... +", len(job["overrides"]), "overrides")
            proc = subprocess.run(cmd, cwd=str(REPO_DIR), text=True)
            result = {
                "name": job["name"],
                "variant": job["variant"],
                "loss_type": job["loss_type"],
                "selection_metric": job["selection_metric"],
                "year": job["year"],
                "base_seed": job["base_seed"],
                "returncode": int(proc.returncode),
            }
            results.append(result)
            results_path = RUN_ROOT / "portfolio_ic_pit_training_results.json"
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print("Return code:", proc.returncode)
            if proc.returncode != 0:
                raise RuntimeError(f"Job failed: {job['name']}")

        print("All jobs completed.")
        print("Results:", RUN_ROOT / "portfolio_ic_pit_training_results.json")
        """
    ),
    md("## 5. Post-Training Reminder"),
    code(
        r"""
        print("Training objective comparison complete.")
        print("Next evidence step: replay the saved predictions through the existing PIT portfolio reports.")
        print("Keep Sharpe, drawdown, turnover, and transaction-cost effects out of the v1 training loss.")
        print("Report those metrics after training from prediction/backtest artifacts.")
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
