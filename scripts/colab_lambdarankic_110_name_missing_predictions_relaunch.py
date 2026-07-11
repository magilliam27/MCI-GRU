"""Colab-only relaunch for missing 110-name LambdaRankIC predictions.

This script is intended to be pasted into the visible Colab notebook and run on
a G4 High-RAM runtime. It regenerates only missing saved-prediction rows for the
110-name PIT GICS top-10 LambdaRankIC campaign, uploads a narrow artifact set,
and skips rows already recovered in Drive.
"""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from google.colab import auth, runtime, userdata

    IN_COLAB = True
except Exception:
    auth = runtime = userdata = None
    IN_COLAB = False


if not IN_COLAB:
    raise RuntimeError("This relaunch script is Colab-only. Do not run it on the local PC.")


REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
BRANCH = "main"
SOURCE_NOTEBOOK = "https://colab.research.google.com/drive/1uRIERr_OFE8HC79FV6GHhR1isZ3AjFIt"
SOURCE_RUN_FAMILY = "sp500_gics_top10_loss_comparison_repeated_seeds"
SOURCE_RUN_TAG = "20260629_011839"
DRIVE_ABLATIONS_PARENT_ID = "1KUIj06ekfNpZa1IkkcAdhHXbVZt-PYT5"
RUN_FAMILY = "lambdarankic_110_name_missing_predictions_g4"
RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RECIPE = (
    "static-threshold-shuffle__pure-ic-returns-5d-val-ic__"
    "regime-current-only__ensemble__drop-edge-0p1"
)

YEARS = [2022, 2023, 2024, 2025]
BASE_SEEDS = [271828, 161803]
NUM_MODELS = 20
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
PIT_MIN_SCOREABLE_STOCKS = 100
LAMBDARANK_PAIR_CAP = 8192
LAMBDARANK_TEMPERATURE = 1.0
RUN_BACKTESTS = False
UPLOAD_CHECKPOINTS = False
UPLOAD_PER_MODEL_PREDICTIONS = False

MARKET_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv"
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

REPO_DIR = Path("/content/MCI-GRU")
LOCAL_RUN_ROOT = Path("/content/mci_gru_runs") / RUN_FAMILY / RUN_TAG
TRAINING_ROOT = LOCAL_RUN_ROOT / "training"
LOG_DIR = LOCAL_RUN_ROOT / "logs"
SUMMARY_DIR = LOCAL_RUN_ROOT / "summaries"
for directory in [LOCAL_RUN_ROOT, TRAINING_ROOT, LOG_DIR, SUMMARY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

BLOCKED_GPU_NAMES = ("T4", "L4")
ALLOWED_GPU_MARKERS = ("G4", "RTX PRO", "BLACKWELL")

ROOT_RUN_FILES = {
    "config.yaml",
    "run_metadata.json",
    "training_summary.json",
    "evaluation_summary.json",
    "timing_summary.json",
    "feature_reference.json",
    "run_summary.json",
}
HYDRA_FILES = {"config.yaml", "hydra.yaml", "overrides.yaml"}
CHECKPOINT_SUFFIXES = {".pt", ".pth"}

drive_service: Any = None
run_folder_id: str | None = None
folder_cache: dict[tuple[str, str], str] = {}
file_cache: dict[tuple[str, str], str] = {}
training_rows: list[dict[str, Any]] = []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def drive_folder_url(folder_id: str | None) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}" if folder_id else ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def escape_drive_value(value: str) -> str:
    return value.replace("'", "\\'")


def execute_with_retries(request: Any, *, label: str, attempts: int = 4) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return request.execute()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(60, 2**attempt)
            print(f"{label} failed on attempt {attempt}/{attempts}: {exc!r}; retrying in {delay}s")
            time.sleep(delay)


def detect_gpu_name() -> str:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "nvidia-smi failed; select a visible Colab G4 runtime before running.\n" + proc.stderr
        )
    gpu_name = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
    if not gpu_name:
        raise RuntimeError("nvidia-smi did not report a GPU name.")
    upper_gpu = gpu_name.upper()
    if any(blocked in upper_gpu for blocked in BLOCKED_GPU_NAMES):
        raise RuntimeError(f"Refusing GPU {gpu_name}; this recovery run is G4-only, not L4/T4.")
    if not any(marker in upper_gpu for marker in ALLOWED_GPU_MARKERS):
        raise RuntimeError(
            f"Refusing GPU {gpu_name}; expected a visible G4/RTX PRO/Blackwell runtime."
        )
    return gpu_name


def build_drive_service() -> Any:
    global drive_service
    assert auth is not None
    auth.authenticate_user()
    from googleapiclient.discovery import build

    drive_service = build("drive", "v3")
    return drive_service


def find_child(parent_id: str, name: str, *, mime_type: str | None = None) -> dict[str, Any] | None:
    assert drive_service is not None
    clauses = [
        f"'{parent_id}' in parents",
        f"name = '{escape_drive_value(name)}'",
        "trashed = false",
    ]
    if mime_type:
        clauses.append(f"mimeType = '{mime_type}'")
    response = execute_with_retries(
        drive_service.files().list(
            q=" and ".join(clauses),
            fields="files(id,name,mimeType,modifiedTime,size)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        label=f"find_child {name}",
    )
    files = response.get("files", [])
    return files[0] if files else None


def ensure_folder(parent_id: str, name: str) -> str:
    assert drive_service is not None
    key = (parent_id, name)
    if key in folder_cache:
        return folder_cache[key]
    existing = find_child(parent_id, name, mime_type="application/vnd.google-apps.folder")
    if existing:
        folder_cache[key] = existing["id"]
        return existing["id"]
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    created = execute_with_retries(
        drive_service.files().create(body=metadata, fields="id", supportsAllDrives=True),
        label=f"create folder {name}",
    )
    folder_cache[key] = created["id"]
    return created["id"]


def ensure_folder_path(parent_id: str, parts: list[str]) -> str:
    current_id = parent_id
    for part in parts:
        current_id = ensure_folder(current_id, part)
    return current_id


def find_drive_file_by_name(name: str) -> dict[str, Any]:
    assert drive_service is not None
    response = execute_with_retries(
        drive_service.files().list(
            q=f"name = '{escape_drive_value(name)}' and trashed = false",
            fields="files(id,name,mimeType,modifiedTime,size,parents)",
            orderBy="modifiedTime desc",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        label=f"find Drive file {name}",
    )
    files = response.get("files", [])
    if not files:
        raise FileNotFoundError(f"Could not find Drive file named {name}")
    return files[0]


def download_drive_file(file_id: str, dest: Path) -> None:
    assert drive_service is not None
    from googleapiclient.http import MediaIoBaseDownload

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with dest.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def upload_or_update_file(
    local_path: Path,
    parent_id: str,
    *,
    name: str | None = None,
    mime_type: str | None = None,
) -> str:
    assert drive_service is not None
    from googleapiclient.http import MediaFileUpload

    file_name = name or local_path.name
    mime = mime_type or mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
    cache_key = (parent_id, file_name)
    file_id = file_cache.get(cache_key)
    if file_id is None:
        existing = find_child(parent_id, file_name)
        file_id = existing["id"] if existing else None
    for attempt in range(1, 5):
        try:
            media = MediaFileUpload(str(local_path), mimetype=mime, resumable=False)
            if file_id:
                response = (
                    drive_service.files()
                    .update(
                        fileId=file_id,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
            else:
                metadata = {"name": file_name, "parents": [parent_id]}
                response = (
                    drive_service.files()
                    .create(
                        body=metadata,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
            file_cache[cache_key] = response["id"]
            return response["id"]
        except Exception as exc:
            if attempt == 4:
                raise
            delay = min(60, 2**attempt)
            print(
                f"upload {file_name} failed on attempt {attempt}/4: {exc!r}; retrying in {delay}s"
            )
            time.sleep(delay)
    raise RuntimeError(f"Upload failed unexpectedly for {local_path}")


def heartbeat(status: str, phase: str, **extra: Any) -> None:
    payload = {
        "status": status,
        "phase": phase,
        "updated_at_utc": utc_now(),
        "run_family": RUN_FAMILY,
        "run_tag": RUN_TAG,
        "drive_run_root_url": drive_folder_url(run_folder_id),
        "source_notebook": SOURCE_NOTEBOOK,
        "source_run_family": SOURCE_RUN_FAMILY,
        "source_run_tag": SOURCE_RUN_TAG,
        "years": YEARS,
        "base_seeds": BASE_SEEDS,
        "pair_cap": LAMBDARANK_PAIR_CAP,
        "run_backtests": RUN_BACKTESTS,
        "runtime_requirement": "Visible Colab G4 runtime. L4/T4 are blocked by nvidia-smi gate.",
        **extra,
    }
    heartbeat_path = SUMMARY_DIR / "heartbeat.json"
    write_json(heartbeat_path, payload)
    if run_folder_id:
        upload_or_update_file(
            heartbeat_path,
            run_folder_id,
            name="heartbeat.json",
            mime_type="application/json",
        )


def run_stream(
    cmd: list[str],
    *,
    cwd: Path,
    log_name: str,
    phase: str,
    env: dict[str, str] | None = None,
) -> None:
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
            if time.time() - last_write > 90:
                heartbeat("RUNNING", phase, last_line=line.strip(), log=str(log_path))
                last_write = time.time()
        returncode = proc.wait()
    if run_folder_id:
        logs_folder_id = ensure_folder(run_folder_id, "logs")
        upload_or_update_file(log_path, logs_folder_id, mime_type="text/plain")
    heartbeat("RUNNING", phase, returncode=returncode, log=str(log_path))
    if returncode != 0:
        raise RuntimeError(f"{phase} failed with return code {returncode}; see {log_path}")


def is_timestamp_dir(path: Path) -> bool:
    return path.is_dir() and bool(re.fullmatch(r"\d{8}_\d{6}", path.name))


def latest_run_dir(job_root: Path, name: str) -> Path | None:
    base = job_root / name
    if not base.exists():
        return None
    candidates = sorted([path for path in base.iterdir() if is_timestamp_dir(path)])
    return candidates[-1] if candidates else None


def validation_summary(pred_dir: Path) -> dict[str, Any]:
    csvs = sorted(pred_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No prediction CSVs found in {pred_dir}")
    return {
        "prediction_csv_count": len(csvs),
        "first_prediction_csv": csvs[0].name,
        "last_prediction_csv": csvs[-1].name,
    }


def should_upload_run_file(path: Path, run_dir: Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(run_dir)
    parts = rel.parts
    if not parts:
        return False
    top = parts[0]
    suffix = path.suffix.lower()
    if top == "averaged_predictions":
        return True
    if top.startswith("predictions_model_"):
        return UPLOAD_PER_MODEL_PREDICTIONS and suffix == ".csv"
    if top == "checkpoints":
        return UPLOAD_CHECKPOINTS and suffix in CHECKPOINT_SUFFIXES
    if top == ".hydra" and len(parts) == 2:
        return parts[1] in HYDRA_FILES
    if len(parts) == 1:
        return rel.name in ROOT_RUN_FILES or (rel.name.startswith("training_") and suffix == ".log")
    return False


def iter_recovery_upload_files(run_dir: Path) -> list[Path]:
    return [path for path in sorted(run_dir.rglob("*")) if should_upload_run_file(path, run_dir)]


def count_local_averaged_prediction_csvs(run_dir: Path) -> int:
    return len(list((run_dir / "averaged_predictions").glob("*.csv")))


def count_drive_csv_files(folder_id: str) -> int:
    assert drive_service is not None
    count = 0
    page_token = None
    while True:
        response = execute_with_retries(
            drive_service.files().list(
                q=(
                    f"'{folder_id}' in parents and trashed = false "
                    "and mimeType != 'application/vnd.google-apps.folder'"
                ),
                fields="nextPageToken,files(id,name,mimeType)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ),
            label=f"count csv files in {folder_id}",
        )
        count += sum(
            1 for item in response.get("files", []) if item.get("name", "").endswith(".csv")
        )
        page_token = response.get("nextPageToken")
        if not page_token:
            return count


def upload_selected_run_tree(
    run_dir: Path,
    drive_job_root_id: str,
    *,
    job_label: str,
) -> tuple[str, str | None, int]:
    exp_folder_id = ensure_folder(drive_job_root_id, run_dir.parent.name)
    drive_run_id = ensure_folder(exp_folder_id, run_dir.name)
    folder_ids: dict[Path, str] = {Path("."): drive_run_id}
    upload_files = iter_recovery_upload_files(run_dir)
    expected_avg_count = count_local_averaged_prediction_csvs(run_dir)

    heartbeat(
        "RUNNING",
        "upload_start",
        current_job=job_label,
        upload_file_count=len(upload_files),
        expected_averaged_prediction_csv_count=expected_avg_count,
    )

    uploaded = 0
    for file_path in upload_files:
        rel = file_path.relative_to(run_dir)
        parent_rel = rel.parent
        if parent_rel not in folder_ids:
            current_id = drive_run_id
            current_rel = Path(".")
            for part in parent_rel.parts:
                current_rel = current_rel / part
                if current_rel not in folder_ids:
                    current_id = ensure_folder(current_id, part)
                    folder_ids[current_rel] = current_id
                else:
                    current_id = folder_ids[current_rel]
        upload_or_update_file(file_path, folder_ids[parent_rel])
        uploaded += 1
        if uploaded == 1 or uploaded % 25 == 0 or uploaded == len(upload_files):
            heartbeat(
                "RUNNING",
                "uploading",
                current_job=job_label,
                uploaded_file_count=uploaded,
                upload_file_count=len(upload_files),
                expected_averaged_prediction_csv_count=expected_avg_count,
                last_uploaded_file=rel.as_posix(),
            )

    pred_folder_id = folder_ids.get(Path("averaged_predictions"))
    heartbeat(
        "RUNNING",
        "upload_complete",
        current_job=job_label,
        uploaded_file_count=uploaded,
        expected_averaged_prediction_csv_count=expected_avg_count,
    )
    return drive_run_id, pred_folder_id, uploaded


def write_training_rows(rows: list[dict[str, Any]]) -> None:
    write_json(SUMMARY_DIR / "training_rows.json", {"rows": rows})
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (SUMMARY_DIR / "training_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if run_folder_id:
        summaries_id = ensure_folder(run_folder_id, "summaries")
        upload_or_update_file(
            SUMMARY_DIR / "training_rows.json",
            summaries_id,
            mime_type="application/json",
        )
        if rows:
            upload_or_update_file(
                SUMMARY_DIR / "training_rows.csv",
                summaries_id,
                mime_type="text/csv",
            )


def stage_named_file(name: str, dest: Path) -> Path:
    source = find_drive_file_by_name(name)
    heartbeat("RUNNING", "download_input", input_name=name, drive_file_id=source["id"])
    download_drive_file(source["id"], dest)
    return dest


def variant_overrides() -> list[str]:
    return [
        "training.loss_type=lambdarank_ic",
        "training.selection_metric=val_rank_ic",
        f"training.lambdarank_ic_max_pairs_per_day={LAMBDARANK_PAIR_CAP}",
        f"training.lambdarank_ic_temperature={LAMBDARANK_TEMPERATURE}",
    ]


def build_jobs(repo_market_csv: Path, repo_pit_csv: Path) -> list[dict[str, Any]]:
    base_overrides = [
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
    jobs: list[dict[str, Any]] = []
    for year in YEARS:
        window = PIT_WINDOWS[year]
        for seed in BASE_SEEDS:
            name = f"top10_lambdarank_ic_{year}_seed{seed}"
            job_root = TRAINING_ROOT / "lambdarank_ic" / str(year) / f"seed{seed}"
            jobs.append(
                {
                    "name": name,
                    "loss_key": "lambdarank_ic",
                    "loss_type": "lambdarank_ic",
                    "year": year,
                    "base_seed": seed,
                    "pair_cap": LAMBDARANK_PAIR_CAP,
                    "job_root": str(job_root),
                    "overrides": [
                        *base_overrides,
                        *variant_overrides(),
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
    return jobs


RECOVERED_ROWS = {
    (2022, 271828),
    (2022, 161803),
}
SALVAGEABLE_AVERAGED_PREDICTIONS: dict[tuple[int, int], dict[str, Any]] = {
    (2023, 271828): {
        "folder_id": "1i2eiUSi0CGpatkzZ64hqAz04v71ob6FV",
        "expected_csv_count": 246,
        "url": "https://drive.google.com/drive/folders/1i2eiUSi0CGpatkzZ64hqAz04v71ob6FV",
    },
}


def recovered_or_salvageable(job: dict[str, Any]) -> bool:
    key = (job["year"], job["base_seed"])
    if key in RECOVERED_ROWS:
        return True
    salvage = SALVAGEABLE_AVERAGED_PREDICTIONS.get(key)
    if salvage is None:
        return False
    observed = count_drive_csv_files(salvage["folder_id"])
    salvage["observed_csv_count"] = observed
    return observed == salvage["expected_csv_count"]


def main() -> None:
    global run_folder_id

    gpu_name = detect_gpu_name()
    print("GPU:", gpu_name)
    build_drive_service()
    family_folder_id = ensure_folder(DRIVE_ABLATIONS_PARENT_ID, RUN_FAMILY)
    run_folder_id = ensure_folder(family_folder_id, RUN_TAG)
    training_folder_id = ensure_folder(run_folder_id, "training")
    heartbeat("RUNNING", "drive_api_authenticated", gpu_name=gpu_name)

    if not os.environ.get("FRED_API_KEY") and userdata is not None:
        secret = userdata.get("FRED_API_KEY")
        if secret:
            os.environ["FRED_API_KEY"] = secret
            print("FRED_API_KEY loaded from Colab Secrets.")
    if not os.environ.get("FRED_API_KEY"):
        raise RuntimeError("FRED_API_KEY is required for the regime-enabled recipe.")

    if not REPO_DIR.exists():
        run_stream(
            ["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)],
            cwd=Path("/content"),
            log_name="setup_git_clone.log",
            phase="setup_git_clone",
        )
    else:
        run_stream(
            ["git", "-C", str(REPO_DIR), "fetch", "origin"],
            cwd=Path("/content"),
            log_name="setup_git_fetch.log",
            phase="setup_git_fetch",
        )
        run_stream(
            ["git", "-C", str(REPO_DIR), "checkout", "-B", BRANCH, f"origin/{BRANCH}"],
            cwd=Path("/content"),
            log_name="setup_git_checkout.log",
            phase="setup_git_checkout",
        )
        run_stream(
            ["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH],
            cwd=Path("/content"),
            log_name="setup_git_pull.log",
            phase="setup_git_pull",
        )

    run_stream(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=REPO_DIR,
        log_name="setup_pip_upgrade.log",
        phase="setup_pip_upgrade",
    )
    run_stream(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")],
        cwd=REPO_DIR,
        log_name="setup_requirements.log",
        phase="setup_requirements",
    )
    run_stream(
        [sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO_DIR}[dev,tracking,fred]"],
        cwd=REPO_DIR,
        log_name="setup_editable.log",
        phase="setup_editable",
    )

    import pandas as pd

    repo_market_csv = stage_named_file(
        MARKET_FILENAME, REPO_DIR / "data/raw/market" / MARKET_FILENAME
    )
    repo_pit_csv = stage_named_file(PIT_FILENAME, REPO_DIR / "data/raw/constituents" / PIT_FILENAME)
    repo_snapshot_csv = stage_named_file(
        SNAPSHOT_FILENAME,
        REPO_DIR / "data/raw/constituents" / SNAPSHOT_FILENAME,
    )

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

    all_jobs = build_jobs(repo_market_csv, repo_pit_csv)
    original_job_count = len(all_jobs)
    jobs = [job for job in all_jobs if not recovered_or_salvageable(job)]
    manifest = {
        "run_family": RUN_FAMILY,
        "run_tag": RUN_TAG,
        "branch": BRANCH,
        "recipe": RECIPE,
        "source_notebook": SOURCE_NOTEBOOK,
        "source_run_family": SOURCE_RUN_FAMILY,
        "source_run_tag": SOURCE_RUN_TAG,
        "years": YEARS,
        "base_seeds": BASE_SEEDS,
        "num_models": NUM_MODELS,
        "num_epochs": NUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "pair_cap": LAMBDARANK_PAIR_CAP,
        "run_backtests": RUN_BACKTESTS,
        "data_audit": data_audit,
        "original_job_count": original_job_count,
        "remaining_job_count": len(jobs),
        "recovered_rows": sorted([f"{year}_seed{seed}" for year, seed in RECOVERED_ROWS]),
        "salvageable_rows": {
            f"{year}_seed{seed}": metadata
            for (year, seed), metadata in SALVAGEABLE_AVERAGED_PREDICTIONS.items()
        },
        "jobs": jobs,
    }
    write_json(LOCAL_RUN_ROOT / "manifest.json", manifest)
    upload_or_update_file(
        LOCAL_RUN_ROOT / "manifest.json",
        run_folder_id,
        name="manifest.json",
        mime_type="application/json",
    )
    heartbeat(
        "RUNNING",
        "resume_matrix_filtered",
        gpu_name=gpu_name,
        original_job_count=original_job_count,
        remaining_job_count=len(jobs),
        recovered_rows=manifest["recovered_rows"],
        salvageable_rows=manifest["salvageable_rows"],
    )
    print("Drive run root:", drive_folder_url(run_folder_id))
    print("Remaining training jobs:")
    for job in jobs:
        print("-", job["year"], job["base_seed"], job["name"])

    try:
        for job in jobs:
            job_label = f"{job['year']}_seed{job['base_seed']}"
            row_path = (
                SUMMARY_DIR
                / "completed_rows"
                / job["loss_key"]
                / str(job["year"])
                / f"seed{job['base_seed']}.json"
            )
            if row_path.exists():
                row = json.loads(row_path.read_text(encoding="utf-8"))
                training_rows.append(row)
                write_training_rows(training_rows)
                continue

            heartbeat(
                "RUNNING",
                "training",
                current_job=job_label,
                completed_training_rows=len(training_rows),
            )
            cmd = [sys.executable, "-u", str(REPO_DIR / "run_experiment.py"), *job["overrides"]]
            run_stream(
                cmd,
                cwd=REPO_DIR,
                log_name=f"training_{job['name']}.log",
                phase=f"training_{job_label}",
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            run_dir = latest_run_dir(Path(job["job_root"]), job["name"])
            if run_dir is None:
                raise FileNotFoundError(f"Missing run dir for {job['name']}")
            pred_dir = run_dir / "averaged_predictions"
            if not pred_dir.exists():
                raise FileNotFoundError(
                    f"Missing averaged_predictions for {job['name']}: {pred_dir}"
                )

            row = {
                "name": job["name"],
                "loss_key": job["loss_key"],
                "loss_type": job["loss_type"],
                "year": job["year"],
                "base_seed": job["base_seed"],
                "pair_cap": job["pair_cap"],
                "status": "TRAINED_UPLOAD_PENDING",
                "mode": "trained",
                "run_dir": str(run_dir),
                "predictions_dir": str(pred_dir),
                "training_summary": json.loads(
                    (run_dir / "training_summary.json").read_text(encoding="utf-8")
                ),
                "evaluation_summary": json.loads(
                    (run_dir / "evaluation_summary.json").read_text(encoding="utf-8")
                ),
                **validation_summary(pred_dir),
            }

            drive_job_root_id = ensure_folder_path(
                training_folder_id,
                [job["loss_key"], str(job["year"]), f"seed{job['base_seed']}"],
            )
            drive_run_id, pred_folder_id, uploaded = upload_selected_run_tree(
                run_dir,
                drive_job_root_id,
                job_label=job_label,
            )
            row.update(
                {
                    "status": "OK",
                    "drive_run_folder_id": drive_run_id,
                    "drive_run_url": drive_folder_url(drive_run_id),
                    "averaged_predictions_folder_id": pred_folder_id,
                    "averaged_predictions_url": drive_folder_url(pred_folder_id),
                    "uploaded_file_count": uploaded,
                }
            )
            write_json(row_path, row)
            training_rows.append(row)
            write_training_rows(training_rows)
            heartbeat(
                "RUNNING",
                "row_complete",
                current_job=job_label,
                completed_training_rows=len(training_rows),
                prediction_csv_count=row["prediction_csv_count"],
                uploaded_file_count=uploaded,
            )

        write_json(
            LOCAL_RUN_ROOT / "run_summary.json",
            {
                "status": "OK",
                "run_family": RUN_FAMILY,
                "run_tag": RUN_TAG,
                "run_root": str(LOCAL_RUN_ROOT),
                "drive_run_root_url": drive_folder_url(run_folder_id),
                "data_audit": data_audit,
                "training_rows": training_rows,
                "run_backtests": RUN_BACKTESTS,
            },
        )
        upload_or_update_file(
            LOCAL_RUN_ROOT / "run_summary.json",
            run_folder_id,
            mime_type="application/json",
        )
        heartbeat(
            "OK",
            "complete",
            completed_training_rows=len(training_rows),
            remaining_job_count=len(jobs),
        )
    except Exception as exc:
        heartbeat("FAILED", "failed", error=repr(exc), completed_training_rows=len(training_rows))
        raise
    finally:
        if runtime is not None:
            try:
                runtime.unassign()
            except Exception as exc:
                print("Manual Runtime > Disconnect and delete runtime may be needed:", exc)


main()
