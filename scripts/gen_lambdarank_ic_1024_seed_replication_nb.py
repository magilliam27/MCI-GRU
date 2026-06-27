"""Generate the Colab notebook for LambdaRankIC 1024 seed replication."""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path("notebooks/lambdarank_ic_1024_all_years_colab.ipynb")
OUT = Path("notebooks/lambdarank_ic_1024_seed_replication_colab.ipynb")


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Expected text not found: {old[:120]!r}")
    return text.replace(old, new, 1)


def replace_all(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Expected text not found: {old[:120]!r}")
    return text.replace(old, new)


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_cell_text(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def transform_intro(text: str) -> str:
    text = text.replace(
        "# LambdaRankIC 1024 All-Year PIT Validation",
        "# LambdaRankIC 1024 Five-Seed PIT Replication",
    )
    text = text.replace(
        "Full-preset continuation of the completed 2022 pair-cap tranche. This\n"
        "notebook trains `lambdarank_ic` with `max_pairs_per_day=1024` for every\n"
        "configured PIT test year, then replays saved averaged predictions in:",
        "Five-seed continuation of the completed all-year 1024 cap probe. This\n"
        "notebook trains `lambdarank_ic` with `max_pairs_per_day=1024` for every\n"
        "configured PIT test year and five base seeds, then replays saved averaged\n"
        "predictions in:",
    )
    return text


def transform_setup(text: str) -> str:
    return text.replace(
        "probe_cfg = TrainingConfig(\n"
        '    loss_type="lambdarank_ic",\n'
        '    selection_metric="val_rank_ic",\n'
        "    lambdarank_ic_max_pairs_per_day=1024,\n"
        ")",
        "probe_cfg = TrainingConfig(\n"
        '    loss_type="lambdarank_ic",\n'
        '    selection_metric="val_rank_ic",\n'
        "    lambdarank_ic_max_pairs_per_day=1024,\n"
        ")",
    )


def transform_manifest(text: str) -> str:
    text = text.replace("## 2. Build All-Year Job Matrix", "## 2. Build Five-Seed Job Matrix")
    text = replace_once(
        text, "BASE_SEED = 314159", "BASE_SEEDS = [314159, 271828, 161803, 141421, 173205]"
    )
    text = replace_all(text, "lambdarank_ic_1024_all_years", "lambdarank_ic_1024_seed_replication")
    text = replace_once(
        text,
        '"expected_training_jobs": len(YEARS),',
        '"expected_training_jobs": len(YEARS) * len(BASE_SEEDS),',
    )
    text = replace_once(
        text,
        '"expected_backtests": len(YEARS) * 2,',
        '"expected_backtests": len(YEARS) * len(BASE_SEEDS) * 2,',
    )
    text = replace_once(
        text,
        'f"seed={BASE_SEED}",\n',
        "",
    )
    text = replace_once(
        text,
        """training_jobs = []
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
""",
        """training_jobs = []
for year in YEARS:
    for base_seed in BASE_SEEDS:
        name = f"lambdarank_ic_pairs1024_{year}_seed{base_seed}"
        training_jobs.append(
            {
                "year": year,
                "name": name,
                "max_pairs_per_day": MAX_PAIRS_PER_DAY,
                "base_seed": base_seed,
                "overrides": [
                    f"+experiment=pit_temporal_{year}",
                    *BASE_OVERRIDES,
                    f"seed={base_seed}",
                    f"experiment_name={name}",
                    f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                ],
            }
        )
""",
    )
    text = replace_once(
        text,
        '"scope": "LambdaRankIC max_pairs_per_day=1024 all configured PIT test years",',
        '"scope": "LambdaRankIC max_pairs_per_day=1024 five-seed replication across all configured PIT test years",',
    )
    text = replace_once(text, '"base_seed": BASE_SEED,', '"base_seeds": BASE_SEEDS,')
    return text


def transform_training(text: str) -> str:
    text = text.replace(
        "## 3. Run Training And Saved-Prediction Backtests",
        "## 3. Run Resumable Training And Saved-Prediction Backtests",
    )
    text = replace_once(
        text,
        """    row = {
        "year": job["year"],
        "training_job": job["name"],
        "scenario": scenario["name"],
        "predictions_dir": str(predictions_dir),
        "backtest_dir": str(backtest_dir),
        "test_start": PIT_WINDOWS[job["year"]]["test_start"],
        "test_end": PIT_WINDOWS[job["year"]]["test_end"],
    }
""",
        """    row = {
        "year": job["year"],
        "base_seed": job["base_seed"],
        "training_job": job["name"],
        "scenario": scenario["name"],
        "predictions_dir": str(predictions_dir),
        "backtest_dir": str(backtest_dir),
        "test_start": PIT_WINDOWS[job["year"]]["test_start"],
        "test_end": PIT_WINDOWS[job["year"]]["test_end"],
    }
""",
    )
    text = replace_once(
        text,
        """training_rows = []
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
""",
        """training_rows = json.loads(TRAINING_RESULTS_JSON.read_text(encoding="utf-8")) if TRAINING_RESULTS_JSON.exists() else []
backtest_rows = json.loads(BACKTEST_RESULTS_JSON.read_text(encoding="utf-8")) if BACKTEST_RESULTS_JSON.exists() else []

def completed_training_names() -> set[str]:
    return {row["name"] for row in training_rows if int(row.get("returncode", 1)) == 0}

def completed_backtest_keys() -> set[tuple[str, str]]:
    return {
        (row["training_job"], row["scenario"])
        for row in backtest_rows
        if int(row.get("returncode", 1)) == 0
    }

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
        if job["name"] in completed_training_names() and latest_training_summary(job["name"]) is not None:
            print("Skipping completed training job:", job["name"])
        else:
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
            training_rows[:] = [old for old in training_rows if old["name"] != job["name"]]
            training_rows.append(row)
            TRAINING_RESULTS_JSON.write_text(json.dumps(training_rows, indent=2), encoding="utf-8")
            write_table(TRAINING_RESULTS_CSV, training_rows)
            if proc.returncode != 0:
                raise RuntimeError(f"Training job failed: {job['name']}")

        predictions_dir = latest_predictions_dir(job["name"])
        for scenario in backtest_scenarios:
            current = f"{job['name']}::{scenario['name']}"
            if (job["name"], scenario["name"]) in completed_backtest_keys():
                print("Skipping completed backtest:", current)
                continue
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
            backtest_rows[:] = [
                old
                for old in backtest_rows
                if not (old["training_job"] == job["name"] and old["scenario"] == scenario["name"])
            ]
            backtest_rows.append(row)
            BACKTEST_RESULTS_JSON.write_text(json.dumps(backtest_rows, indent=2), encoding="utf-8")
            write_table(BACKTEST_RESULTS_CSV, backtest_rows)
            if proc.returncode != 0:
                raise RuntimeError(f"Backtest failed: {current}")

    all_rows = []
    by_training_key = {(row["year"], row["base_seed"]): row for row in training_rows}
    for row in backtest_rows:
        merged = dict(row)
        training = by_training_key.get((row["year"], row.get("base_seed")), {})
        if training:
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
""",
    )
    return text


def transform_snapshot(text: str) -> str:
    return text.replace(
        """cols = [
        "year",
        "scenario",
""",
        """cols = [
        "year",
        "base_seed",
        "scenario",
""",
    )


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source notebook: {SRC}")
    notebook = json.loads(SRC.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        text = cell_text(cell)
        if "# LambdaRankIC 1024 All-Year PIT Validation" in text:
            set_cell_text(cell, transform_intro(text))
        elif "## 1. Setup, GPU Gate, And Data" in text:
            continue
        elif "probe_cfg = TrainingConfig(" in text:
            set_cell_text(cell, transform_setup(text))
        elif cell.get("cell_type") == "markdown" and "## 2. Build All-Year Job Matrix" in text:
            set_cell_text(
                cell,
                text.replace("## 2. Build All-Year Job Matrix", "## 2. Build Five-Seed Job Matrix"),
            )
        elif "YEARS = [2022, 2023, 2024, 2025]" in text:
            set_cell_text(cell, transform_manifest(text))
        elif (
            cell.get("cell_type") == "markdown"
            and "## 3. Run Training And Saved-Prediction Backtests" in text
        ):
            set_cell_text(
                cell,
                text.replace(
                    "## 3. Run Training And Saved-Prediction Backtests",
                    "## 3. Run Resumable Training And Saved-Prediction Backtests",
                ),
            )
        elif "training_rows = []" in text:
            set_cell_text(cell, transform_training(text))
        elif "cols = [" in text:
            set_cell_text(cell, transform_snapshot(text))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
