"""Foreground, resumable Colab runner for the Portfolio-IC upward weight sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

BRANCH = "codex/portfolio-ic-hybrid-testing"
FROZEN_RECIPE_ID = (
    "static-threshold-shuffle__pure-ic-returns-5d-val-ic__"
    "regime-current-only__ensemble__drop-edge-0p1"
)
DEFAULT_YEARS = [2022, 2023, 2024, 2025]
DEFAULT_BASE_SEEDS = [314159, 271828, 161803]
DEFAULT_WEIGHTS = [0.75, 1.0]
NUM_MODELS = 20
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
BACKTEST_SUFFIX = "_pit_daily_tc_rank_gate"
PIT_WINDOWS = {
    2022: ("2022-01-22", "2022-12-31"),
    2023: ("2023-01-22", "2023-12-31"),
    2024: ("2024-01-22", "2024-12-31"),
    2025: ("2025-01-22", "2025-12-31"),
}
ALLOWED_GPU_MARKERS = ("G4", "L4", "A100", "H100", "V100", "RTX PRO", "BLACKWELL")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def detect_gpu_name() -> str:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("nvidia-smi failed. Attach a GPU runtime before launching.\n" + proc.stderr)
    name = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
    if not name:
        raise RuntimeError("No GPU name returned by nvidia-smi.")
    upper = name.upper()
    if "T4" in upper:
        raise RuntimeError(f"Refusing runtime GPU {name}. This sweep requires non-T4 G4-class hardware.")
    if not any(marker in upper for marker in ALLOWED_GPU_MARKERS):
        raise RuntimeError(
            f"GPU {name} is not in the allowed non-T4 set {ALLOWED_GPU_MARKERS}. "
            "Switch to a G4-class or better Colab runtime before continuing."
        )
    return name


def weight_variant_name(weight: float) -> str:
    return f"portfolio_ic_weight{int(round(weight * 100)):02d}"


def build_base_overrides(repo_dir: Path, training_output_dir: Path) -> list[str]:
    market_csv = repo_dir / "data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv"
    pit_csv = repo_dir / "data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv"
    regime_csv = repo_dir / "data/raw/regime/pit_repeated_seed_regime_inputs_20260520_183538.csv"
    for path in (market_csv, pit_csv, regime_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    return [
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
        f"features.regime_inputs_csv={regime_csv.relative_to(repo_dir).as_posix()}",
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
        f"data.filename={market_csv.relative_to(repo_dir).as_posix()}",
        f"data.pit_universe_csv={pit_csv.relative_to(repo_dir).as_posix()}",
        "data.use_pit_universe=true",
        "data.pit_universe_mode=masked_panel",
        "data.pit_min_scoreable_stocks=450",
        "data.pit_breadth_policy=error",
        f"output_dir={training_output_dir.as_posix()}",
    ]


def build_jobs(
    repo_dir: Path,
    training_output_dir: Path,
    years: list[int],
    seeds: list[int],
    weights: list[float],
) -> list[dict[str, Any]]:
    base_overrides = build_base_overrides(repo_dir, training_output_dir)
    jobs: list[dict[str, Any]] = []
    for weight in weights:
        variant_name = weight_variant_name(weight)
        for year in years:
            if year not in PIT_WINDOWS:
                raise ValueError(f"No PIT window configured for {year}")
            for base_seed in seeds:
                name = f"{variant_name}_{year}_seed{base_seed}"
                jobs.append(
                    {
                        "name": name,
                        "variant": variant_name,
                        "portfolio_ic_weight": weight,
                        "year": year,
                        "base_seed": base_seed,
                        "loss_type": "portfolio_ic",
                        "selection_metric": "val_loss",
                        "overrides": [
                            f"+experiment=pit_temporal_{year}",
                            *base_overrides,
                            "training.loss_type=portfolio_ic",
                            "training.selection_metric=val_loss",
                            "training.portfolio_ic_top_k=10",
                            f"training.portfolio_ic_weight={weight}",
                            "training.portfolio_ic_temperature=0.25",
                            f"seed={base_seed}",
                            f"experiment_name={name}",
                        ],
                    }
                )
    return jobs


def read_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def write_result_rows(csv_path: Path, json_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_json(json_path, rows)


def count_prediction_dirs(run_dir: Path) -> int:
    return sum(
        1
        for idx in range(NUM_MODELS)
        if (run_dir / f"predictions_model_{idx}").is_dir()
        and any((run_dir / f"predictions_model_{idx}").glob("*.csv"))
    )


def count_checkpoints(run_dir: Path) -> int:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return 0
    return sum(1 for idx in range(NUM_MODELS) if (checkpoint_dir / f"model_{idx}_best.pth").is_file())


def existing_run_score(run_dir: Path) -> tuple[int, int, int, str]:
    averaged = 1 if (run_dir / "averaged_predictions").is_dir() else 0
    predictions = count_prediction_dirs(run_dir)
    checkpoints = count_checkpoints(run_dir)
    return (averaged, predictions, checkpoints, run_dir.name)


def best_existing_run_dir(training_output_dir: Path, job_name: str) -> Path | None:
    base = training_output_dir / job_name
    if not base.is_dir():
        return None
    candidates = [path for path in base.iterdir() if path.is_dir()]
    candidates = [path for path in candidates if count_prediction_dirs(path) or count_checkpoints(path)]
    if not candidates:
        return None
    return max(candidates, key=existing_run_score)


def new_run_dir(training_output_dir: Path, job_name: str) -> Path:
    return training_output_dir / job_name / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def write_manifest(run_root: Path, jobs: list[dict[str, Any]], gpu_name: str) -> Path:
    manifest = {
        "recipe_id": FROZEN_RECIPE_ID,
        "branch": BRANCH,
        "run_root": str(run_root),
        "gpu_name": gpu_name,
        "started_at": utc_now(),
        "years": sorted({job["year"] for job in jobs}),
        "base_seeds": sorted({job["base_seed"] for job in jobs}),
        "weights": sorted({job["portfolio_ic_weight"] for job in jobs}),
        "num_models": NUM_MODELS,
        "num_epochs": NUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "expected_job_count": len(jobs),
        "expected_total_models": len(jobs) * NUM_MODELS,
        "jobs": jobs,
    }
    path = run_root / "portfolio_ic_upward_sweep_manifest.json"
    write_json(path, manifest)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_dir = args.repo_dir.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    training_output_dir = run_root / "training_runs"
    summary_dir = run_root / "summaries"
    logs_dir = summary_dir / "logs" / "training"
    training_output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = detect_gpu_name() if args.require_gpu_gate else "not_checked"
    print("GPU:", gpu_name, flush=True)
    jobs = build_jobs(repo_dir, training_output_dir, args.years, args.seeds, args.weights)
    manifest_path = write_manifest(run_root, jobs, gpu_name)
    print("Run root:", run_root, flush=True)
    print("Manifest:", manifest_path, flush=True)
    print("Jobs:", len(jobs), "total models:", len(jobs) * NUM_MODELS, flush=True)

    csv_path = summary_dir / "training_results.csv"
    json_path = summary_dir / "training_results.json"
    rows = read_result_rows(csv_path)
    completed_ok = {row["name"] for row in rows if str(row.get("status", "")).upper() == "OK"}
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["MCI_GRU_RESUME_ENSEMBLE"] = "1" if args.resume else "0"

    for job_index, job in enumerate(jobs, start=1):
        if job["name"] in completed_ok:
            print(f"[{job_index}/{len(jobs)}] skipping OK job: {job['name']}", flush=True)
            continue

        run_dir = best_existing_run_dir(training_output_dir, job["name"]) if args.resume else None
        if run_dir is None:
            run_dir = new_run_dir(training_output_dir, job["name"])
        run_dir.mkdir(parents=True, exist_ok=True)

        logs_job_dir = logs_dir / job["variant"] / f"{job['year']}_seed{job['base_seed']}"
        logs_job_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python_executable,
            "-u",
            str(repo_dir / "run_experiment.py"),
            *job["overrides"],
            f"hydra.run.dir={run_dir.as_posix()}",
        ]
        write_json(logs_job_dir / "cmd.json", cmd)
        write_json(
            summary_dir / "heartbeat.json",
            {
                "updated_at": utc_now(),
                "phase": "training",
                "status": "RUNNING",
                "job_index": job_index,
                "current_job": job["name"],
                "run_dir": str(run_dir),
                "gpu_name": gpu_name,
            },
        )

        print("=" * 100, flush=True)
        print(f"[{job_index}/{len(jobs)}] Starting: {job['name']}", flush=True)
        print("Run dir:", run_dir, flush=True)
        print("Resume ensemble:", args.resume, flush=True)
        start = time.time()
        proc = subprocess.run(cmd, cwd=repo_dir, text=True, env=env, check=False)
        elapsed_minutes = round((time.time() - start) / 60.0, 3)
        summary_path = run_dir / "training_summary.json"
        eval_path = run_dir / "evaluation_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
        result = {
            "name": job["name"],
            "variant": job["variant"],
            "portfolio_ic_weight": job["portfolio_ic_weight"],
            "year": job["year"],
            "base_seed": job["base_seed"],
            "loss_type": "portfolio_ic",
            "selection_metric": "val_loss",
            "status": "OK" if proc.returncode == 0 else "FAILED",
            "returncode": int(proc.returncode),
            "elapsed_minutes": elapsed_minutes,
            "run_dir": str(run_dir),
            "predictions_dir": str(run_dir / "averaged_predictions"),
            "training_summary.mean_best_val_loss": summary.get("mean_best_val_loss"),
            "training_summary.mean_best_val_ic": summary.get("mean_best_val_ic"),
            "training_summary.models_resumed_from_predictions": summary.get(
                "models_resumed_from_predictions"
            ),
            "training_summary.models_resumed_from_checkpoints": summary.get(
                "models_resumed_from_checkpoints"
            ),
            "evaluation": evaluation.get("metrics"),
        }
        rows = [row for row in rows if row.get("name") != job["name"]]
        rows.append(result)
        write_result_rows(csv_path, json_path, rows)
        print("Return code:", proc.returncode, "elapsed_minutes:", elapsed_minutes, flush=True)

        if proc.returncode != 0:
            write_json(summary_dir / "failure_report.json", result)
            write_json(
                summary_dir / "heartbeat.json",
                {
                    "updated_at": utc_now(),
                    "phase": "training",
                    "status": "FAILED",
                    "failed_job": job["name"],
                    "returncode": proc.returncode,
                    "run_dir": str(run_dir),
                    "gpu_name": gpu_name,
                },
            )
            raise RuntimeError(f"Training job failed: {job['name']}")
        completed_ok.add(job["name"])

    ok_rows = [row for row in rows if str(row.get("status", "")).upper() == "OK"]
    write_json(
        summary_dir / "heartbeat.json",
        {
            "updated_at": utc_now(),
            "phase": "training",
            "status": "OK",
            "completed_jobs": len(ok_rows),
            "expected_jobs": len(jobs),
            "gpu_name": gpu_name,
        },
    )
    if len(ok_rows) != len(jobs):
        raise RuntimeError(f"Expected {len(jobs)} OK jobs, got {len(ok_rows)}")
    print("All training jobs completed.", flush=True)
    print("Training results:", csv_path, flush=True)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_BASE_SEEDS)
    parser.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--require-gpu-gate", action="store_true", default=True)
    parser.add_argument("--no-gpu-gate", dest="require_gpu_gate", action="store_false")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
