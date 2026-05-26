"""Generate a Colab launcher for issue #8 volatility-targeting PIT smoke runs."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

OUT = Path("notebooks/volatility_targeting_pit_colab.ipynb")


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
        # Issue #8 Volatility-Targeting PIT Smoke

        This notebook starts the first Colab check for the Harvey-style
        volatility-targeting feature family. It uses current PIT temporal
        presets for 2022 and 2024 and compares:

        - `baseline_vol`: existing realized-volatility features only.
        - `vol_targeting`: existing realized-volatility features plus the
          issue #8 Harvey-style volatility-targeting family.

        Default mode is a mechanics smoke: one model, two epochs, and low
        bootstrap count. Set `SMOKE_MODE = False` for a full-budget run after
        the smoke proves the preset composes and training starts.
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

        IN_COLAB = "google.colab" in sys.modules
        REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
        BRANCH = "codex/pit-universe-validation"
        REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()

        if IN_COLAB:
            from google.colab import drive

            drive.mount("/content/drive")
            if not REPO_DIR.exists():
                subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)], check=True)
            else:
                subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "-B", BRANCH, f"origin/{BRANCH}"], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")], check=True)

        os.chdir(REPO_DIR)
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))

        print("Repo:", REPO_DIR)
        print("Branch:", BRANCH)
        subprocess.run(["git", "rev-parse", "HEAD"], check=False)
        """
    ),
    md("## 2. FRED Key And Data"),
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
    md("## 3. Build Current-Preset Smoke Matrix"),
    code(
        r"""
        SMOKE_MODE = True
        YEARS = [2022, 2024]
        BASE_SEED = 314159
        RUN_TAG = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        RUN_ROOT = Path("/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8") / RUN_TAG
        TRAINING_OUTPUT_DIR = RUN_ROOT / "training"
        RUN_ROOT.mkdir(parents=True, exist_ok=True)

        num_models = 1 if SMOKE_MODE else 20
        num_epochs = 2 if SMOKE_MODE else 100
        patience = 2 if SMOKE_MODE else 15
        bootstrap_resamples = 25 if SMOKE_MODE else 1000

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
            f"training.num_epochs={num_epochs}",
            f"training.num_models={num_models}",
            f"training.early_stopping_patience={patience}",
            "training.loss_type=ic",
            "training.label_type=returns",
            "training.selection_metric=val_ic",
            "training.shuffle_train=true",
            "model.label_t=5",
            "model.temporal_encoder=gru_attn",
            f"evaluation.bootstrap_resamples={bootstrap_resamples}",
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

        VARIANTS = {
            "baseline_vol": [
                "features.include_volatility=true",
                "features.include_volatility_targeting=false",
            ],
            "vol_targeting": [
                "features.include_volatility=true",
                "features.include_volatility_targeting=true",
                "features.volatility_targeting_half_lives=[20,60,90]",
                "features.volatility_target_vol=0.10",
                "features.volatility_target_scale_clip=[0.25,4.0]",
                "features.volatility_targeting_interaction_return_window=21",
            ],
        }

        jobs = []
        for year in YEARS:
            for variant, variant_overrides in VARIANTS.items():
                experiment = f"pit_temporal_{year}"
                name = f"issue8_{variant}_{year}_seed{BASE_SEED}"
                jobs.append(
                    {
                        "year": year,
                        "variant": variant,
                        "name": name,
                        "overrides": [
                            f"+experiment={experiment}",
                            *BASE_OVERRIDES,
                            *variant_overrides,
                            f"seed={BASE_SEED}",
                            f"experiment_name={name}",
                            f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                        ],
                    }
                )

        manifest = {
            "issue": 8,
            "branch": BRANCH,
            "run_tag": RUN_TAG,
            "smoke_mode": SMOKE_MODE,
            "years": YEARS,
            "base_seed": BASE_SEED,
            "jobs": jobs,
        }
        manifest_path = RUN_ROOT / "issue8_volatility_targeting_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Run root:", RUN_ROOT)
        print("Jobs:", len(jobs))
        for job in jobs:
            print("-", job["name"], job["variant"], job["year"])
        print("Manifest:", manifest_path)
        """
    ),
    md("## 4. Run Jobs"),
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
                "year": job["year"],
                "returncode": int(proc.returncode),
            }
            results.append(result)
            results_path = RUN_ROOT / "issue8_volatility_targeting_results.json"
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print("Return code:", proc.returncode)
            if proc.returncode != 0:
                raise RuntimeError(f"Job failed: {job['name']}")

        print("All jobs completed.")
        print("Results:", RUN_ROOT / "issue8_volatility_targeting_results.json")
        """
    ),
]


def main() -> None:
    notebook = {
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
