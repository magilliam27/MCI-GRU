"""Generate the Colab notebook for LambdaRankIC 110-name replay diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

OUT = Path("notebooks/lambdarankic_110_name_replay_colab.ipynb")


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
        # LambdaRankIC 110-Name Replay Diagnostics

        This notebook runs saved-prediction replay tests for the 110-name PIT
        GICS top-10-per-sector LambdaRankIC campaign. It is diagnostics-only:
        no model training, no `run_experiment.py`, no per-model prediction
        archival, and no default change away from pure IC.

        The notebook treats base seeds (`161803`, `271828`, `314159`) as the
        repeated-seed axis. Ensemble member seeds remain internal to a single
        trained model ensemble and are not counted as repeated base seeds.

        Default mode is `DRY_RUN = True`. Review the inventory and command
        plan, confirm the Drive folders, then set `DRY_RUN = False` to execute
        replay backtests and rank-stability diagnostics in Colab.
        """
    ),
    md("## 1. Setup And Safety Gates"),
    code(
        r"""
        import csv
        import io
        import json
        import math
        import mimetypes
        import os
        import re
        import shutil
        import subprocess
        import sys
        import time
        from datetime import datetime, timezone
        from itertools import combinations
        from pathlib import Path

        import numpy as np
        import pandas as pd

        try:
            from google.colab import auth, runtime
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            IN_COLAB = True
        except Exception:
            auth = None
            runtime = None
            build = None
            MediaFileUpload = None
            IN_COLAB = False

        REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
        BRANCH = "codex/lambdarankic-saved-prediction-replay-20260713"
        REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()
        RUN_FAMILY = "lambdarankic_110_name_replay_diagnostics"
        RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        DRIVE_PROJECT_FOLDER_ID = "1KUIj06ekfNpZa1IkkcAdhHXbVZt-PYT5"
        MARKET_FILE_ID = "1e6aXtSkQGgsAjmytRsUt-xoJTYssWkPq"
        MARKET_FILE_SIZE = 39459457
        PIT_FILE_ID = "11WAppghYylyyBWLeisIhJ-505y2ptTr1"
        PIT_FILE_SIZE = 15940
        RUN_ROOT = Path("/content/mci_gru_runs") / RUN_FAMILY / RUN_TAG if IN_COLAB else REPO_DIR / "outputs" / RUN_FAMILY / RUN_TAG
        LOCAL_DATA_ROOT = Path("/content/mci_gru_inputs") if IN_COLAB else REPO_DIR / "outputs" / "lambdarankic_replay_inputs"
        SUMMARY_DIR = RUN_ROOT / "summaries"
        PREDICTIONS_ROOT = RUN_ROOT / "predictions"
        BACKTEST_ROOT = RUN_ROOT / "backtests"
        LOG_DIR = RUN_ROOT / "logs"
        HEARTBEAT_PATH = RUN_ROOT / "heartbeat.json"
        MANIFEST_PATH = RUN_ROOT / "lambdarankic_110_name_replay_manifest.json"
        DRIVE_SERVICE = None
        DRIVE_RUN_FOLDER_ID = None
        DRIVE_FOLDER_CACHE = {}
        DRIVE_FILE_CACHE = {}
        PUBLISHED_FILE_STATS = {}

        # Safety defaults. Flip DRY_RUN only after the manifest/inventory cells look right.
        DRY_RUN = True
        RUN_TRAINING = False
        REQUIRE_COLAB = True
        REQUIRE_GPU = False
        REQUIRE_COMPLETE_MATRIX = True
        RUN_BASELINE_REPLAY = True
        RUN_2024_DECOMPOSITION = True
        RUN_2024_GATE_SWEEP = False
        RUN_2024_COST_SWEEP = False
        RUN_ALL_YEAR_CONFIRMATION = True
        DISCONNECT_RUNTIME_WHEN_DONE = True

        # Runtime guardrails. Replay is CPU/file I/O heavy, so a GPU is optional.
        # If REQUIRE_GPU is enabled for operator policy, T4 and L4 are blocked.
        BLOCKED_GPU_NAMES = ("T4", "L4")
        ALLOWED_GPU_MARKERS = ("G4", "RTX PRO", "BLACKWELL", "A100", "H100", "V100")

        YEARS = [2022, 2023, 2024, 2025]
        BASE_SEEDS = [161803, 271828, 314159]
        CURRENT_PAIR_CAP = 8192
        LEGACY_PAIR_CAP = "legacy_unknown"
        TOP_K = 10
        LABEL_T = 5
        ADJUSTMENT_METHOD = "bhy"
        BASELINE_SPREAD_BPS = 10.0
        BASELINE_SLIPPAGE_BPS = 5.0
        BASELINE_MIN_RANK_DROP = 30
        GATE_SWEEP_VALUES = [10, 20, 30, 40, 60]
        COST_SWEEP_PAIRS = [(0.0, 0.0), (5.0, 2.0), (10.0, 5.0), (20.0, 10.0)]

        MARKET_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv"
        PIT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv"
        PIT_WINDOWS = {
            2022: {"test_start": "2022-01-08", "test_end": "2022-12-31", "expected_csv_count": 246},
            2023: {"test_start": "2023-01-08", "test_end": "2023-12-31", "expected_csv_count": 246},
            2024: {"test_start": "2024-01-08", "test_end": "2024-12-31", "expected_csv_count": 248},
            2025: {"test_start": "2025-01-08", "test_end": "2025-12-31", "expected_csv_count": 246},
        }

        # PIT masked-panel/no-lookahead contract tokens retained in the manifest.
        PIT_RECIPE_OVERRIDES = [
            "data.use_pit_universe=true",
            "data.pit_universe_mode=masked_panel",
            "data.pit_breadth_policy=error",
            "training.label_type=returns",
            "model.label_t=5",
            "features.regime_include_subsequent_returns=false",
            "training.loss_type=lambdarank_ic",
            "training.selection_metric=val_rank_ic",
        ]

        if RUN_TRAINING:
            raise RuntimeError("This replay notebook must not train models. RUN_TRAINING must stay False.")
        if REQUIRE_COLAB and not IN_COLAB:
            raise RuntimeError("Run this notebook in Colab so Drive folders and output artifacts are canonical.")

        def utc_now() -> str:
            return datetime.now(timezone.utc).isoformat()

        def json_default(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, (np.ndarray,)):
                return value.tolist()
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return str(value)

        def write_json(path: Path, payload: dict) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False), encoding="utf-8")

        def write_heartbeat(phase: str, status: str = "RUNNING", **extra) -> None:
            payload = {
                "run_family": RUN_FAMILY,
                "run_tag": RUN_TAG,
                "phase": phase,
                "status": status,
                "updated_at_utc": utc_now(),
                "run_training": RUN_TRAINING,
                "dry_run": DRY_RUN,
            }
            payload.update(extra)
            write_json(HEARTBEAT_PATH, payload)

        def detect_gpu_name() -> str:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                return ""
            return proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""

        def drive_folder_url(folder_id: str | None) -> str:
            return f"https://drive.google.com/drive/folders/{folder_id}" if folder_id else ""

        def escape_drive_value(value: str) -> str:
            return value.replace("'", "\\'")

        def execute_with_retries(request, *, label: str, attempts: int = 4):
            for attempt in range(1, attempts + 1):
                try:
                    return request.execute()
                except Exception as exc:
                    if attempt == attempts:
                        raise
                    delay = min(60, 2**attempt)
                    print(f"{label} failed on attempt {attempt}/{attempts}: {exc!r}; retrying in {delay}s")
                    time.sleep(delay)

        def find_drive_child(parent_id: str, name: str, *, mime_type: str | None = None):
            clauses = [
                f"'{parent_id}' in parents",
                f"name = '{escape_drive_value(name)}'",
                "trashed = false",
            ]
            if mime_type:
                clauses.append(f"mimeType = '{mime_type}'")
            response = execute_with_retries(
                DRIVE_SERVICE.files().list(
                    q=" and ".join(clauses),
                    fields="files(id,name,mimeType,size)",
                    pageSize=10,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                ),
                label=f"find Drive child {name}",
            )
            files = response.get("files", [])
            return files[0] if files else None

        def ensure_drive_folder(parent_id: str, name: str) -> str:
            key = (parent_id, name)
            if key in DRIVE_FOLDER_CACHE:
                return DRIVE_FOLDER_CACHE[key]
            existing = find_drive_child(parent_id, name, mime_type="application/vnd.google-apps.folder")
            if existing:
                DRIVE_FOLDER_CACHE[key] = existing["id"]
                return existing["id"]
            metadata = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            }
            created = execute_with_retries(
                DRIVE_SERVICE.files().create(body=metadata, fields="id", supportsAllDrives=True),
                label=f"create Drive folder {name}",
            )
            DRIVE_FOLDER_CACHE[key] = created["id"]
            return created["id"]

        def ensure_drive_folder_path(parent_id: str, parts: tuple[str, ...]) -> str:
            current_id = parent_id
            for part in parts:
                current_id = ensure_drive_folder(current_id, part)
            return current_id

        def upload_or_update_drive_file(local_path: Path, parent_id: str) -> str:
            file_name = local_path.name
            cache_key = (parent_id, file_name)
            file_id = DRIVE_FILE_CACHE.get(cache_key)
            if file_id is None:
                existing = find_drive_child(parent_id, file_name)
                file_id = existing["id"] if existing else None
            mime_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
            media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=False)
            if file_id:
                response = execute_with_retries(
                    DRIVE_SERVICE.files().update(
                        fileId=file_id,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    ),
                    label=f"update Drive file {file_name}",
                )
            else:
                response = execute_with_retries(
                    DRIVE_SERVICE.files().create(
                        body={"name": file_name, "parents": [parent_id]},
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    ),
                    label=f"create Drive file {file_name}",
                )
            DRIVE_FILE_CACHE[cache_key] = response["id"]
            return response["id"]

        def publish_run_artifacts() -> dict:
            if DRIVE_SERVICE is None or DRIVE_RUN_FOLDER_ID is None:
                return {"uploaded": 0, "skipped": 0, "total": 0}
            uploaded = 0
            skipped = 0
            local_files = sorted(path for path in RUN_ROOT.rglob("*") if path.is_file())
            for local_path in local_files:
                relative_path = local_path.relative_to(RUN_ROOT)
                stat = local_path.stat()
                fingerprint = (stat.st_size, stat.st_mtime_ns)
                if PUBLISHED_FILE_STATS.get(str(relative_path)) == fingerprint:
                    skipped += 1
                    continue
                parent_id = ensure_drive_folder_path(DRIVE_RUN_FOLDER_ID, tuple(relative_path.parts[:-1]))
                upload_or_update_drive_file(local_path, parent_id)
                PUBLISHED_FILE_STATS[str(relative_path)] = fingerprint
                uploaded += 1
            return {"uploaded": uploaded, "skipped": skipped, "total": len(local_files)}

        def verify_published_artifacts(relative_paths: list[str]) -> list[dict]:
            verified = []
            for relative_value in relative_paths:
                relative_path = Path(relative_value)
                local_path = RUN_ROOT / relative_path
                if not local_path.exists():
                    raise FileNotFoundError(f"Missing local publication artifact: {local_path}")
                parent_id = DRIVE_RUN_FOLDER_ID
                for part in relative_path.parts[:-1]:
                    folder = find_drive_child(parent_id, part, mime_type="application/vnd.google-apps.folder")
                    if not folder:
                        raise FileNotFoundError(f"Missing remote publication folder: {relative_path.parent}")
                    parent_id = folder["id"]
                remote_file = find_drive_child(parent_id, relative_path.name)
                if not remote_file:
                    raise FileNotFoundError(f"Missing remote publication artifact: {relative_value}")
                remote_size = int(remote_file.get("size", -1))
                if remote_size != local_path.stat().st_size:
                    raise RuntimeError(
                        f"Remote size mismatch for {relative_value}: local={local_path.stat().st_size} remote={remote_size}"
                    )
                verified.append({"path": relative_value, "file_id": remote_file["id"], "size": remote_size})
            return verified

        def resolve_remote_folder(relative_parts: tuple[str, ...]) -> str:
            parent_id = DRIVE_RUN_FOLDER_ID
            for part in relative_parts:
                folder = find_drive_child(parent_id, part, mime_type="application/vnd.google-apps.folder")
                if not folder:
                    raise FileNotFoundError(f"Missing remote publication folder: {'/'.join(relative_parts)}")
                parent_id = folder["id"]
            return parent_id

        def verify_published_csv_directory(relative_directory: str, expected_count: int) -> dict:
            relative_path = Path(relative_directory)
            local_directory = RUN_ROOT / relative_path
            remote_folder_id = resolve_remote_folder(tuple(relative_path.parts))
            remote_csvs = {
                child["name"]: int(child.get("size", -1))
                for child in list_drive_children(remote_folder_id)
                if child["name"].lower().endswith(".csv")
            }
            local_csvs = {path.name: path.stat().st_size for path in local_directory.glob("*.csv")}
            if len(local_csvs) != expected_count or len(remote_csvs) != expected_count:
                raise RuntimeError(
                    f"Published CSV count mismatch for {relative_directory}: "
                    f"expected={expected_count} local={len(local_csvs)} remote={len(remote_csvs)}"
                )
            if local_csvs != remote_csvs:
                raise RuntimeError(f"Published CSV names or sizes differ for {relative_directory}")
            return {
                "path": relative_directory,
                "remote_folder_id": remote_folder_id,
                "csv_count": len(remote_csvs),
            }

        def read_published_json(relative_value: str) -> dict:
            relative_path = Path(relative_value)
            parent_id = resolve_remote_folder(tuple(relative_path.parts[:-1]))
            remote_file = find_drive_child(parent_id, relative_path.name)
            if not remote_file:
                raise FileNotFoundError(f"Missing remote JSON artifact: {relative_value}")
            request = DRIVE_SERVICE.files().get_media(fileId=remote_file["id"], supportsAllDrives=True)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return json.loads(buffer.getvalue().decode("utf-8"))

        if IN_COLAB:
            auth.authenticate_user()
            DRIVE_SERVICE = build("drive", "v3")
            family_folder_id = ensure_drive_folder(DRIVE_PROJECT_FOLDER_ID, RUN_FAMILY)
            DRIVE_RUN_FOLDER_ID = ensure_drive_folder(family_folder_id, RUN_TAG)

        RUN_ROOT.mkdir(parents=True, exist_ok=False)
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
        BACKTEST_ROOT.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        write_heartbeat("setup_start")

        if IN_COLAB:
            if not REPO_DIR.exists():
                subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)], check=True)
            else:
                subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "-B", BRANCH, f"origin/{BRANCH}"], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO_DIR}[dev,tracking,fred]"], check=True)

        write_heartbeat("setup")
        setup_publication = publish_run_artifacts()

        os.chdir(REPO_DIR)
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))

        GPU_NAME = detect_gpu_name()
        if REQUIRE_GPU:
            upper_gpu = GPU_NAME.upper()
            if not GPU_NAME or any(blocked in upper_gpu for blocked in BLOCKED_GPU_NAMES):
                raise RuntimeError(f"Expected G4-class replay runtime, not T4/L4/CPU. Visible GPU: {GPU_NAME or 'NONE'}")
            if not any(marker in upper_gpu for marker in ALLOWED_GPU_MARKERS):
                raise RuntimeError(f"Refusing GPU {GPU_NAME}; allowed markers are {ALLOWED_GPU_MARKERS}.")

        git_head = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
        git_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
        print("Repo:", REPO_DIR)
        print("Branch:", git_branch)
        print("Commit:", git_head)
        print("Run root:", RUN_ROOT)
        print("Drive run root:", drive_folder_url(DRIVE_RUN_FOLDER_ID))
        print("Setup publication:", setup_publication)
        print("GPU:", GPU_NAME or "not required")
        print("DRY_RUN:", DRY_RUN)
        """
    ),
    md("## 2. Prediction Manifest And Drive Inventory"),
    code(
        r"""
        from googleapiclient.http import MediaIoBaseDownload

        # Explicit verified LambdaRankIC 110-name averaged_prediction folders.
        PREDICTION_ROWS = [
            {
                "row_id": "lambdarank_ic_2022_seed161803_pair8192",
                "year": 2022,
                "base_seed": 161803,
                "pair_cap": CURRENT_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2022]["expected_csv_count"],
                "folder_id": "1MW8uCWjlarfJYnsOUzG2vBmZUaXjaDin",
                "folder_url": "https://drive.google.com/drive/folders/1MW8uCWjlarfJYnsOUzG2vBmZUaXjaDin",
                "source": "recovery_20260701_verified",
                "required": True,
            },
            {
                "row_id": "lambdarank_ic_2022_seed271828_pair8192",
                "year": 2022,
                "base_seed": 271828,
                "pair_cap": CURRENT_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2022]["expected_csv_count"],
                "folder_id": "1w_lFPx_JKginWf6-TsQoFY-Mlhs2XHuc",
                "folder_url": "https://drive.google.com/drive/folders/1w_lFPx_JKginWf6-TsQoFY-Mlhs2XHuc",
                "source": "recovery_20260701_verified",
                "required": True,
            },
            {
                "row_id": "lambdarank_ic_2023_seed161803_pair8192",
                "year": 2023,
                "base_seed": 161803,
                "pair_cap": CURRENT_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2023]["expected_csv_count"],
                "folder_id": "1jCcZu-ENQKfbit2cdRjBucmCVklULERO",
                "folder_url": "https://drive.google.com/drive/folders/1jCcZu-ENQKfbit2cdRjBucmCVklULERO",
                "source": "g4_recovery_20260707",
                "required": True,
            },
            {
                "row_id": "lambdarank_ic_2023_seed271828_pair8192",
                "year": 2023,
                "base_seed": 271828,
                "pair_cap": CURRENT_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2023]["expected_csv_count"],
                "folder_id": "1i2eiUSi0CGpatkzZ64hqAz04v71ob6FV",
                "folder_url": "https://drive.google.com/drive/folders/1i2eiUSi0CGpatkzZ64hqAz04v71ob6FV",
                "source": "recovery_20260701_salvaged",
                "required": True,
            },
            {
                "row_id": "lambdarank_ic_2024_seed161803_pair8192",
                "year": 2024,
                "base_seed": 161803,
                "pair_cap": CURRENT_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2024]["expected_csv_count"],
                "folder_id": "1Gvtz8C3U6da1YtjA_bJ6SFyrcBNgfMeI",
                "folder_url": "https://drive.google.com/drive/folders/1Gvtz8C3U6da1YtjA_bJ6SFyrcBNgfMeI",
                "source": "g4_recovery_20260707",
                "required": True,
            },
            {
                "row_id": "lambdarank_ic_2024_seed271828_pair8192",
                "year": 2024,
                "base_seed": 271828,
                "pair_cap": CURRENT_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2024]["expected_csv_count"],
                "folder_id": "1km8pF1mFCREktte26bnboKw8_aqSzLzL",
                "folder_url": "https://drive.google.com/drive/folders/1km8pF1mFCREktte26bnboKw8_aqSzLzL",
                "source": "g4_recovery_20260707",
                "required": True,
            },
            {
                "row_id": "lambdarank_ic_2025_seed161803_pair8192",
                "year": 2025,
                "base_seed": 161803,
                "pair_cap": CURRENT_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2025]["expected_csv_count"],
                "folder_id": "1ctmw-XztXVP8r_FGu81bE_V7k2FawVLO",
                "folder_url": "https://drive.google.com/drive/folders/1ctmw-XztXVP8r_FGu81bE_V7k2FawVLO",
                "source": "g4_recovery_20260707",
                "required": True,
            },
            {
                "row_id": "lambdarank_ic_2025_seed271828_pair8192",
                "year": 2025,
                "base_seed": 271828,
                "pair_cap": CURRENT_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2025]["expected_csv_count"],
                "folder_id": "1Yg1yzcU9xKZ8FnjSK0KDUc6AWr3XlUON",
                "folder_url": "https://drive.google.com/drive/folders/1Yg1yzcU9xKZ8FnjSK0KDUc6AWr3XlUON",
                "source": "g4_recovery_20260707",
                "required": True,
            },
            {
                "row_id": "lambdarank_ic_2022_seed314159_legacy",
                "year": 2022,
                "base_seed": 314159,
                "pair_cap": LEGACY_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2022]["expected_csv_count"],
                "folder_id": "1IJ62jNdpLbFW4Kuc9l68LkG3NmTmS9bd",
                "folder_url": "https://drive.google.com/drive/folders/1IJ62jNdpLbFW4Kuc9l68LkG3NmTmS9bd",
                "source": "legacy_314159_20260626",
                "required": True,
            },
            {
                "row_id": "lambdarank_ic_2023_seed314159_legacy",
                "year": 2023,
                "base_seed": 314159,
                "pair_cap": LEGACY_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2023]["expected_csv_count"],
                "folder_id": "1tp-BEvU2yPMxnE_c6ul3o1gVi-ZyeF3n",
                "folder_url": "https://drive.google.com/drive/folders/1tp-BEvU2yPMxnE_c6ul3o1gVi-ZyeF3n",
                "source": "legacy_314159_20260626",
                "required": True,
            },
            {
                "row_id": "lambdarank_ic_2024_seed314159_legacy",
                "year": 2024,
                "base_seed": 314159,
                "pair_cap": LEGACY_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2024]["expected_csv_count"],
                "folder_id": "1KHK3TSjtjz4Ft-XTU5DkmVtgcklgKTwt",
                "folder_url": "https://drive.google.com/drive/folders/1KHK3TSjtjz4Ft-XTU5DkmVtgcklgKTwt",
                "source": "legacy_314159_20260626",
                "required": True,
            },
            {
                "row_id": "lambdarank_ic_2025_seed314159_legacy",
                "year": 2025,
                "base_seed": 314159,
                "pair_cap": LEGACY_PAIR_CAP,
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "expected_csv_count": PIT_WINDOWS[2025]["expected_csv_count"],
                "folder_id": "1PIV6uuwKDBKAGYMIRsvepCD7cgo9CgO3",
                "folder_url": "https://drive.google.com/drive/folders/1PIV6uuwKDBKAGYMIRsvepCD7cgo9CgO3",
                "source": "legacy_314159_20260627",
                "required": True,
            },
        ]

        DRIVE_REFERENCE_FOLDERS = [
            {"label": "repeated_seed_campaign_root", "folder_id": "1lhL-tnUoShh8ImNdTED_sRBOf_dqcOim"},
            {"label": "completed_backtest_run_20260629_011839", "folder_id": "1Co5Vd2dOSMrHUN5x_OzbJpkjJFocSHMo"},
            {"label": "missing_prediction_recovery_20260701_185554", "folder_id": "1mO5dqZ6QMIRMDQHmrbd30so2ui7HeT5V"},
            {"label": "g4_recovery_20260707_014129", "folder_id": "1fYmtPg97O52SgTRsU_XgwuVaWFbpj9W_"},
        ]

        def list_drive_children(folder_id: str) -> list[dict]:
            if DRIVE_SERVICE is None:
                return []
            files: list[dict] = []
            page_token = None
            while True:
                response = (
                    DRIVE_SERVICE.files()
                    .list(
                        q=f"'{folder_id}' in parents and trashed=false",
                        fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, webViewLink)",
                        pageToken=page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
                files.extend(response.get("files", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
            return files

        def count_drive_csvs(folder_id: str) -> tuple[int, str, str]:
            csv_names = sorted(
                child["name"] for child in list_drive_children(folder_id) if child["name"].lower().endswith(".csv")
            )
            if not csv_names:
                return 0, "", ""
            return len(csv_names), csv_names[0], csv_names[-1]

        inventory_rows = []
        for row in PREDICTION_ROWS:
            csv_count = 0
            first_csv = ""
            last_csv = ""
            status = "MISSING_FOLDER_ID"
            if row["folder_id"]:
                csv_count, first_csv, last_csv = count_drive_csvs(row["folder_id"])
                status = "OK" if csv_count == row["expected_csv_count"] else "CSV_COUNT_MISMATCH"
            elif not row.get("required", True):
                status = "OPTIONAL_FOLDER_ID_MISSING"
            inventory_rows.append({**row, "csv_count": csv_count, "first_csv": first_csv, "last_csv": last_csv, "status": status})

        inventory_df = pd.DataFrame(inventory_rows)
        inventory_path = SUMMARY_DIR / "saved_prediction_inventory.csv"
        inventory_df.to_csv(inventory_path, index=False)

        drive_inventory_rows = []
        for reference in DRIVE_REFERENCE_FOLDERS:
            children = list_drive_children(reference["folder_id"]) if DRIVE_SERVICE is not None else []
            drive_inventory_rows.append(
                {
                    "label": reference["label"],
                    "folder_id": reference["folder_id"],
                    "url": f"https://drive.google.com/drive/folders/{reference['folder_id']}",
                    "child_count": len(children),
                    "sample_children": "; ".join(child["name"] for child in children[:10]),
                }
            )
        drive_inventory_path = SUMMARY_DIR / "drive_artifact_inventory.csv"
        pd.DataFrame(drive_inventory_rows).to_csv(drive_inventory_path, index=False)

        missing_required = inventory_df[(inventory_df["required"]) & (inventory_df["status"] != "OK")]
        if REQUIRE_COMPLETE_MATRIX and not missing_required.empty:
            write_heartbeat(
                "inventory_failed",
                status="FAILED",
                missing_required_rows=missing_required["row_id"].tolist(),
            )
            publish_run_artifacts()
            raise RuntimeError(f"Required prediction rows are missing or incomplete:\n{missing_required.to_string(index=False)}")

        manifest = {
            "run_family": RUN_FAMILY,
            "run_tag": RUN_TAG,
            "repo_url": REPO_URL,
            "branch": BRANCH,
            "git_head": git_head,
            "git_branch": git_branch,
            "drive_project_folder_id": DRIVE_PROJECT_FOLDER_ID,
            "drive_run_folder_id": DRIVE_RUN_FOLDER_ID,
            "drive_run_root_url": drive_folder_url(DRIVE_RUN_FOLDER_ID),
            "dry_run": DRY_RUN,
            "run_training": RUN_TRAINING,
            "years": YEARS,
            "base_seeds": BASE_SEEDS,
            "current_pair_cap": CURRENT_PAIR_CAP,
            "legacy_pair_cap": LEGACY_PAIR_CAP,
            "top_k": TOP_K,
            "label_t": LABEL_T,
            "adjustment_method": ADJUSTMENT_METHOD,
            "pit_recipe_overrides": PIT_RECIPE_OVERRIDES,
            "market_filename": MARKET_FILENAME,
            "market_file_id": MARKET_FILE_ID,
            "pit_filename": PIT_FILENAME,
            "pit_file_id": PIT_FILE_ID,
            "prediction_rows": inventory_rows,
            "drive_reference_folders": DRIVE_REFERENCE_FOLDERS,
        }
        write_json(MANIFEST_PATH, manifest)
        write_heartbeat("inventory", inventory_path=str(inventory_path), drive_inventory_path=str(drive_inventory_path))
        inventory_publication = publish_run_artifacts()
        print("Inventory publication:", inventory_publication)
        display(inventory_df)
        """
    ),
    md("## 3. Stage Predictions And Resolve PIT Data"),
    code(
        r"""
        def safe_token(value) -> str:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

        def expected_prediction_dir(row: dict) -> Path:
            return (
                PREDICTIONS_ROOT
                / str(row["year"])
                / f"seed{row['base_seed']}"
                / f"pair_cap_{safe_token(row['pair_cap'])}"
                / "averaged_predictions"
            )

        def download_drive_file(file_id: str, target_path: Path, expected_size: int) -> str:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() and target_path.stat().st_size == expected_size:
                return "EXISTS"
            partial_path = Path(f"{target_path}.part")
            partial_path.unlink(missing_ok=True)
            request = DRIVE_SERVICE.files().get_media(fileId=file_id, supportsAllDrives=True)
            with partial_path.open("wb") as handle:
                downloader = MediaIoBaseDownload(handle, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            actual_size = partial_path.stat().st_size
            if actual_size != expected_size:
                partial_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Drive download size mismatch for {target_path.name}: expected={expected_size} actual={actual_size}"
                )
            partial_path.replace(target_path)
            return "DOWNLOADED"

        def download_drive_folder_csvs(folder_id: str, target_dir: Path) -> list[dict]:
            target_dir.mkdir(parents=True, exist_ok=True)
            downloaded = []
            children = list_drive_children(folder_id)
            for child in sorted(children, key=lambda item: item["name"]):
                if not child["name"].lower().endswith(".csv"):
                    continue
                target_path = target_dir / child["name"]
                expected_size = int(child["size"])
                status = download_drive_file(child["id"], target_path, expected_size)
                downloaded.append({"name": child["name"], "path": str(target_path), "status": status})
            return downloaded

        def resolve_or_download_input(
            candidates: list[Path],
            *,
            file_id: str,
            expected_size: int,
            filename: str,
            label: str,
        ) -> Path:
            for candidate in candidates:
                if candidate.exists() and candidate.stat().st_size == expected_size:
                    return candidate
            target_path = LOCAL_DATA_ROOT / filename
            download_drive_file(file_id, target_path, expected_size)
            if not target_path.exists() or target_path.stat().st_size != expected_size:
                raise FileNotFoundError(f"Could not materialize exact {label} from Drive file {file_id}")
            return target_path

        MARKET_CSV = resolve_or_download_input(
            [
                REPO_DIR / "data" / "raw" / "market" / MARKET_FILENAME,
            ],
            file_id=MARKET_FILE_ID,
            expected_size=MARKET_FILE_SIZE,
            filename=MARKET_FILENAME,
            label="110-name PIT market CSV",
        )
        PIT_CSV = resolve_or_download_input(
            [
                REPO_DIR / "data" / "raw" / "constituents" / PIT_FILENAME,
            ],
            file_id=PIT_FILE_ID,
            expected_size=PIT_FILE_SIZE,
            filename=PIT_FILENAME,
            label="110-name PIT universe CSV",
        )

        staged_rows = []
        for row in inventory_rows:
            if row["status"] != "OK":
                staged_rows.append({**row, "staged_predictions_dir": "", "stage_status": "SKIPPED_NOT_OK"})
                continue
            target_dir = expected_prediction_dir(row)
            if DRY_RUN:
                staged_rows.append({**row, "staged_predictions_dir": str(target_dir), "stage_status": "DRY_RUN"})
                continue
            downloaded = download_drive_folder_csvs(row["folder_id"], target_dir)
            local_count = len(list(target_dir.glob("*.csv")))
            stage_status = "OK" if local_count == row["expected_csv_count"] else "LOCAL_CSV_COUNT_MISMATCH"
            staged_rows.append(
                {
                    **row,
                    "staged_predictions_dir": str(target_dir),
                    "downloaded_file_count": len(downloaded),
                    "local_csv_count": local_count,
                    "stage_status": stage_status,
                }
            )

        staged_df = pd.DataFrame(staged_rows)
        staged_path = SUMMARY_DIR / "staged_prediction_inventory.csv"
        staged_df.to_csv(staged_path, index=False)
        write_heartbeat("stage_predictions", staged_path=str(staged_path), market_csv=str(MARKET_CSV), pit_csv=str(PIT_CSV))
        stage_publication = publish_run_artifacts()
        print("Stage publication:", stage_publication)
        display(staged_df)
        """
    ),
    md("## 4. Replay Backtest Scenarios"),
    code(
        r"""
        def scenario(name: str, scope: str, transaction_costs: bool, rank_gate: bool, spread_bps: float, slippage_bps: float, min_rank_drop: int | None) -> dict:
            return {
                "name": name,
                "scope": scope,
                "transaction_costs": transaction_costs,
                "rank_gate": rank_gate,
                "spread_bps": spread_bps,
                "slippage_bps": slippage_bps,
                "min_rank_drop": min_rank_drop,
            }

        scenarios_by_name: dict[str, dict] = {}

        def add_scenario(item: dict) -> None:
            scenarios_by_name[item["name"]] = item

        if RUN_BASELINE_REPLAY:
            add_scenario(
                scenario(
                    "baseline_cost_gate30",
                    "all_years",
                    True,
                    True,
                    BASELINE_SPREAD_BPS,
                    BASELINE_SLIPPAGE_BPS,
                    BASELINE_MIN_RANK_DROP,
                )
            )
        if RUN_2024_DECOMPOSITION:
            add_scenario(scenario("gross_no_cost_no_gate", "stress_2024", False, False, 0.0, 0.0, None))
            add_scenario(scenario("cost_no_gate", "stress_2024", True, False, BASELINE_SPREAD_BPS, BASELINE_SLIPPAGE_BPS, None))
            add_scenario(scenario("gate30_no_cost", "stress_2024", False, True, 0.0, 0.0, BASELINE_MIN_RANK_DROP))
        if RUN_2024_GATE_SWEEP:
            add_scenario(scenario("gate_none_cost_baseline", "stress_2024", True, False, BASELINE_SPREAD_BPS, BASELINE_SLIPPAGE_BPS, None))
            for gate_value in GATE_SWEEP_VALUES:
                add_scenario(
                    scenario(
                        f"gate{gate_value}_cost_baseline",
                        "stress_2024",
                        True,
                        True,
                        BASELINE_SPREAD_BPS,
                        BASELINE_SLIPPAGE_BPS,
                        gate_value,
                    )
                )
        if RUN_2024_COST_SWEEP:
            for spread_bps, slippage_bps in COST_SWEEP_PAIRS:
                add_scenario(
                    scenario(
                        f"cost_s{safe_token(spread_bps)}_sl{safe_token(slippage_bps)}_gate30",
                        "stress_2024",
                        True,
                        True,
                        spread_bps,
                        slippage_bps,
                        BASELINE_MIN_RANK_DROP,
                    )
                )

        SCENARIOS = list(scenarios_by_name.values())

        def row_ready_for_replay(row: dict) -> bool:
            if row["status"] != "OK":
                return False
            if DRY_RUN:
                return row.get("stage_status") == "DRY_RUN"
            return row.get("stage_status") == "OK"

        def rows_for_scenario(scenario_row: dict) -> list[dict]:
            ok_rows = [row for row in staged_rows if row_ready_for_replay(row)]
            if scenario_row["scope"] == "stress_2024":
                return [row for row in ok_rows if row["year"] == 2024]
            if scenario_row["scope"] == "all_years":
                return ok_rows if RUN_ALL_YEAR_CONFIRMATION else [row for row in ok_rows if row["year"] == 2024]
            return ok_rows

        def backtest_suffix(row: dict, scenario_row: dict) -> str:
            return f"_replay_{safe_token(scenario_row['name'])}_seed{row['base_seed']}_pair{safe_token(row['pair_cap'])}"

        def build_backtest_command(row: dict, scenario_row: dict) -> list[str]:
            window = PIT_WINDOWS[int(row["year"])]
            cmd = [
                sys.executable,
                "-X",
                "utf8",
                str(REPO_DIR / "scripts" / "backtest_sp500_daily.py"),
                "--predictions_dir",
                row["staged_predictions_dir"],
                "--data_file",
                str(MARKET_CSV),
                "--pit_universe_csv",
                str(PIT_CSV),
                "--test_start",
                window["test_start"],
                "--test_end",
                window["test_end"],
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
                backtest_suffix(row, scenario_row),
            ]
            if scenario_row["transaction_costs"]:
                cmd.extend(["--transaction_costs", "--spread", str(scenario_row["spread_bps"]), "--slippage", str(scenario_row["slippage_bps"])])
            if scenario_row["rank_gate"]:
                cmd.extend(["--enable_rank_drop_gate", "--min_rank_drop", str(scenario_row["min_rank_drop"])])
            return cmd

        def expected_backtest_dir(row: dict, scenario_row: dict) -> Path:
            return Path(row["staged_predictions_dir"]).parent / f"backtest{backtest_suffix(row, scenario_row)}"

        def canonical_backtest_dir(row: dict, scenario_row: dict) -> Path:
            return (
                BACKTEST_ROOT
                / safe_token(scenario_row["name"])
                / str(row["year"])
                / f"seed{row['base_seed']}"
                / f"pair_cap_{safe_token(row['pair_cap'])}"
            )

        planned_commands = []
        for scenario_row in SCENARIOS:
            for row in rows_for_scenario(scenario_row):
                planned_commands.append(
                    {
                        "row_id": row["row_id"],
                        "year": row["year"],
                        "base_seed": row["base_seed"],
                        "pair_cap": row["pair_cap"],
                        "scenario": scenario_row["name"],
                        "scope": scenario_row["scope"],
                        "command": " ".join(build_backtest_command(row, scenario_row)),
                        "script_backtest_dir": str(expected_backtest_dir(row, scenario_row)),
                        "canonical_backtest_dir": str(canonical_backtest_dir(row, scenario_row)),
                    }
                )

        planned_path = SUMMARY_DIR / "planned_backtest_commands.csv"
        pd.DataFrame(planned_commands).to_csv(planned_path, index=False)
        planning_publication = publish_run_artifacts()
        print("Planning publication:", planning_publication)
        display(pd.DataFrame(planned_commands))

        def execute_backtests() -> list[dict]:
            result_rows = []
            for scenario_row in SCENARIOS:
                for row in rows_for_scenario(scenario_row):
                    cmd = build_backtest_command(row, scenario_row)
                    bt_dir = expected_backtest_dir(row, scenario_row)
                    canonical_dir = canonical_backtest_dir(row, scenario_row)
                    log_path = LOG_DIR / f"backtest_{row['row_id']}_{scenario_row['name']}.log"
                    write_heartbeat(
                        "backtest",
                        current_row=row["row_id"],
                        current_scenario=scenario_row["name"],
                        command=" ".join(cmd),
                    )

                    if DRY_RUN:
                        result_rows.append(
                            {
                                **row,
                                **{f"scenario.{key}": value for key, value in scenario_row.items()},
                                "script_backtest_dir": str(bt_dir),
                                "backtest_dir": str(canonical_dir),
                                "returncode": None,
                                "status": "DRY_RUN",
                            }
                        )
                        continue

                    started = time.perf_counter()
                    proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True, check=False)
                    elapsed_seconds = time.perf_counter() - started
                    log_path.write_text(proc.stdout + "\n\nSTDERR:\n" + proc.stderr, encoding="utf-8")
                    metrics_path = bt_dir / "backtest_metrics.json"
                    metrics = {}
                    if metrics_path.exists():
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if bt_dir.exists():
                        shutil.copytree(bt_dir, canonical_dir, dirs_exist_ok=True)
                    status = "OK" if proc.returncode == 0 else "FAILED"
                    result_rows.append(
                        {
                            **row,
                            **{f"scenario.{key}": value for key, value in scenario_row.items()},
                            "script_backtest_dir": str(bt_dir),
                            "backtest_dir": str(canonical_dir),
                            "metrics_path": str(canonical_dir / "backtest_metrics.json"),
                            "log_path": str(log_path),
                            "returncode": proc.returncode,
                            "elapsed_seconds": round(elapsed_seconds, 3),
                            "status": status,
                            **{f"backtest.{key}": value for key, value in metrics.items()},
                        }
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"Backtest failed for {row['row_id']} {scenario_row['name']}; see {log_path}"
                        )
            return result_rows

        try:
            backtest_rows = execute_backtests()
        except Exception as exc:
            write_heartbeat(
                "failed",
                status="FAILED",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            failure_publication = publish_run_artifacts()
            verify_published_artifacts(["heartbeat.json"])
            print("Failure publication:", failure_publication)
            if IN_COLAB and DISCONNECT_RUNTIME_WHEN_DONE:
                runtime.unassign()
            raise

        backtest_df = pd.DataFrame(backtest_rows)
        backtest_csv = SUMMARY_DIR / "backtest_results.csv"
        backtest_json = SUMMARY_DIR / "backtest_results.json"
        backtest_df.to_csv(backtest_csv, index=False)
        write_json(backtest_json, {"rows": backtest_rows})
        write_heartbeat("backtests_done", backtest_csv=str(backtest_csv), row_count=len(backtest_rows))
        backtest_publication = publish_run_artifacts()
        print("Backtest publication:", backtest_publication)
        display(backtest_df)
        """
    ),
    md("## 5. Cross-Seed Rank Stability"),
    code(
        r"""
        ID_CANDIDATES = ["kdcode", "symbol", "ticker", "stock", "stock_id", "code"]
        SCORE_CANDIDATES = ["prediction", "predicted_return", "pred", "score", "y_pred", "model_prediction"]

        def detect_id_column(df: pd.DataFrame) -> str:
            for column in ID_CANDIDATES:
                if column in df.columns:
                    return column
            return df.columns[0]

        def detect_score_column(df: pd.DataFrame) -> str:
            for column in SCORE_CANDIDATES:
                if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                    return column
            numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
            excluded = {"rank", "target", "label", "return", "actual"}
            for column in numeric_columns:
                if column.lower() not in excluded:
                    return column
            raise ValueError(f"Could not detect prediction score column in columns={list(df.columns)}")

        def load_daily_prediction_tables(prediction_dir: Path) -> dict[str, pd.DataFrame]:
            tables = {}
            for csv_path in sorted(prediction_dir.glob("*.csv")):
                frame = pd.read_csv(csv_path)
                if frame.empty:
                    continue
                id_col = detect_id_column(frame)
                score_col = detect_score_column(frame)
                date_value = csv_path.stem
                table = frame[[id_col, score_col]].copy()
                table.columns = ["asset_id", "score"]
                table["score"] = pd.to_numeric(table["score"], errors="coerce")
                table = table.dropna(subset=["score"]).sort_values("score", ascending=False).reset_index(drop=True)
                table["rank"] = np.arange(1, len(table) + 1)
                table["date"] = date_value
                tables[date_value] = table
            return tables

        def jaccard(left: set, right: set) -> float:
            if not left and not right:
                return 1.0
            union = left | right
            return len(left & right) / len(union) if union else 0.0

        rank_rows = []
        jaccard_rows = []
        churn_rows = []
        table_cache: dict[str, dict[str, pd.DataFrame]] = {}

        diagnostic_rows = [
            row
            for row in staged_rows
            if row_ready_for_replay(row) and row.get("staged_predictions_dir")
        ]

        if DRY_RUN:
            diagnostic_rows = []

        for row in diagnostic_rows:
            prediction_dir = Path(row["staged_predictions_dir"])
            tables = load_daily_prediction_tables(prediction_dir)
            table_cache[row["row_id"]] = tables
            prev_top10: set | None = None
            for date_value, table in sorted(tables.items()):
                top10 = set(table.head(10)["asset_id"])
                boundary = table.iloc[7:12].copy()
                score_rank10 = table.iloc[9]["score"] if len(table) >= 10 else np.nan
                score_rank11 = table.iloc[10]["score"] if len(table) >= 11 else np.nan
                additions = len(top10 - prev_top10) if prev_top10 is not None else np.nan
                removals = len(prev_top10 - top10) if prev_top10 is not None else np.nan
                churn_rows.append(
                    {
                        "row_id": row["row_id"],
                        "year": row["year"],
                        "base_seed": row["base_seed"],
                        "pair_cap": row["pair_cap"],
                        "date": date_value,
                        "top10_additions": additions,
                        "top10_removals": removals,
                        "score_rank10": score_rank10,
                        "score_rank11": score_rank11,
                        "rank10_rank11_margin": score_rank10 - score_rank11,
                        "boundary_assets_ranks_8_12": "|".join(boundary["asset_id"].astype(str).tolist()),
                    }
                )
                prev_top10 = top10

        grouped: dict[tuple[int, str, str], list[dict]] = {}
        for row in diagnostic_rows:
            grouped.setdefault((int(row["year"]), str(row["loss_type"]), str(row["pair_cap"])), []).append(row)

        for (year, loss_type, comparison_pair_cap), rows in grouped.items():
            for left, right in combinations(rows, 2):
                left_tables = table_cache[left["row_id"]]
                right_tables = table_cache[right["row_id"]]
                for date_value in sorted(set(left_tables) & set(right_tables)):
                    left_table = left_tables[date_value]
                    right_table = right_tables[date_value]
                    merged = left_table.merge(right_table, on="asset_id", suffixes=("_left", "_right"))
                    spearman = merged["rank_left"].corr(merged["rank_right"], method="spearman") if len(merged) > 1 else np.nan
                    kendall = merged["rank_left"].corr(merged["rank_right"], method="kendall") if len(merged) > 1 else np.nan
                    rank_rows.append(
                        {
                            "year": year,
                            "loss_type": loss_type,
                            "comparison_pair_cap": comparison_pair_cap,
                            "date": date_value,
                            "left_row_id": left["row_id"],
                            "right_row_id": right["row_id"],
                            "left_base_seed": left["base_seed"],
                            "right_base_seed": right["base_seed"],
                            "left_pair_cap": left["pair_cap"],
                            "right_pair_cap": right["pair_cap"],
                            "common_asset_count": len(merged),
                            "spearman_rank_corr": spearman,
                            "kendall_rank_corr": kendall,
                        }
                    )
                    for k_value in [10, 20, 30]:
                        jaccard_rows.append(
                            {
                                "year": year,
                                "loss_type": loss_type,
                                "comparison_pair_cap": comparison_pair_cap,
                                "date": date_value,
                                "left_row_id": left["row_id"],
                                "right_row_id": right["row_id"],
                                "left_base_seed": left["base_seed"],
                                "right_base_seed": right["base_seed"],
                                "left_pair_cap": left["pair_cap"],
                                "right_pair_cap": right["pair_cap"],
                                "top_k": k_value,
                                "jaccard": jaccard(
                                    set(left_table.head(k_value)["asset_id"]),
                                    set(right_table.head(k_value)["asset_id"]),
                                ),
                            }
                        )

        rank_corr_df = pd.DataFrame(rank_rows)
        jaccard_df = pd.DataFrame(jaccard_rows)
        churn_df = pd.DataFrame(churn_rows)
        rank_summary_df = (
            rank_corr_df.groupby(["year", "left_pair_cap", "right_pair_cap", "left_base_seed", "right_base_seed"], dropna=False)
            .agg(
                mean_spearman_rank_corr=("spearman_rank_corr", "mean"),
                median_spearman_rank_corr=("spearman_rank_corr", "median"),
                mean_kendall_rank_corr=("kendall_rank_corr", "mean"),
                date_count=("date", "count"),
            )
            .reset_index()
            if not rank_corr_df.empty
            else pd.DataFrame()
        )

        rank_corr_path = SUMMARY_DIR / "cross_seed_rank_correlation.csv"
        jaccard_path = SUMMARY_DIR / "cross_seed_jaccard.csv"
        churn_path = SUMMARY_DIR / "top10_boundary_churn.csv"
        rank_summary_path = SUMMARY_DIR / "rank_stability_summary.csv"
        rank_corr_df.to_csv(rank_corr_path, index=False)
        jaccard_df.to_csv(jaccard_path, index=False)
        churn_df.to_csv(churn_path, index=False)
        rank_summary_df.to_csv(rank_summary_path, index=False)

        # Also materialize this expected filename for the sensitivity grid owner.
        sensitivity_path = SUMMARY_DIR / "rank_drop_cost_sensitivity.csv"
        backtest_df.to_csv(sensitivity_path, index=False)

        write_heartbeat(
            "rank_stability_done",
            rank_corr_path=str(rank_corr_path),
            jaccard_path=str(jaccard_path),
            churn_path=str(churn_path),
            rank_summary_path=str(rank_summary_path),
        )
        diagnostics_publication = publish_run_artifacts()
        print("Diagnostics publication:", diagnostics_publication)
        display(rank_summary_df)
        """
    ),
    md("## 6. Decision Gate Report"),
    code(
        r"""
        def write_decision_gate_report() -> Path:
            report_path = SUMMARY_DIR / "decision_gate_report.md"
            lines = [
                "# LambdaRankIC 110-Name Replay Decision Gate Report",
                "",
                f"- Run tag: `{RUN_TAG}`",
                f"- Dry run: `{DRY_RUN}`",
                f"- Training executed: `{RUN_TRAINING}`",
                f"- Baseline cost model: spread `{BASELINE_SPREAD_BPS}` bps, slippage `{BASELINE_SLIPPAGE_BPS}` bps",
                f"- Baseline rank-drop gate: `{BASELINE_MIN_RANK_DROP}`",
                "",
                "## Promotion Gate",
                "",
                "Promote LambdaRankIC only if the saved-prediction matrix is complete, 2024 no longer shows a seed-specific churn cliff, net results beat or match pure IC under cost + gate30, worst seed-year drawdown is acceptable, turnover is controlled, and cross-seed Top-10/20/30 agreement is stable.",
                "",
                "## Experimental Gate",
                "",
                "Keep LambdaRankIC experimental if 2024 remains dominated by weak gross ranking, top-10 boundary churn, one seed, one pair cap, or one cost/gate assumption.",
                "",
                "## Hybrid Gate",
                "",
                "Consider second-stage or hybrid use only if broad-rank quality improves while a pre-declared blend, reranker, or hysteresis rule fixes top-10 churn across all years and base seeds.",
                "",
                "## Expected Artifacts",
                "",
                "- `heartbeat.json`",
                "- `lambdarankic_110_name_replay_manifest.json`",
                "- `summaries/saved_prediction_inventory.csv`",
                "- `summaries/drive_artifact_inventory.csv`",
                "- `summaries/backtest_results.csv`",
                "- `summaries/rank_stability_summary.csv`",
                "- `summaries/cross_seed_rank_correlation.csv`",
                "- `summaries/cross_seed_jaccard.csv`",
                "- `summaries/top10_boundary_churn.csv`",
                "- `summaries/rank_drop_cost_sensitivity.csv`",
                "- `predictions/<year>/seed<seed>/pair_cap_<cap>/averaged_predictions/*.csv`",
                "- `backtests/<scenario>/<year>/seed<seed>/pair_cap_<cap>/backtest_metrics.json`",
                "- `backtests/<scenario>/<year>/seed<seed>/pair_cap_<cap>/trade_journal.csv`",
                "- `backtests/<scenario>/<year>/seed<seed>/pair_cap_<cap>/daily_holdings.csv`",
            ]
            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return report_path

        report_path = write_decision_gate_report()
        write_heartbeat(
            "publication_verification",
            status="RUNNING",
            decision_gate_report=str(report_path),
        )
        completion_publication = publish_run_artifacts()
        required_remote_artifacts = [
            "heartbeat.json",
            "lambdarankic_110_name_replay_manifest.json",
            "summaries/saved_prediction_inventory.csv",
            "summaries/drive_artifact_inventory.csv",
            "summaries/staged_prediction_inventory.csv",
            "summaries/planned_backtest_commands.csv",
            "summaries/backtest_results.csv",
            "summaries/backtest_results.json",
            "summaries/rank_stability_summary.csv",
            "summaries/cross_seed_rank_correlation.csv",
            "summaries/cross_seed_jaccard.csv",
            "summaries/top10_boundary_churn.csv",
            "summaries/rank_drop_cost_sensitivity.csv",
            "summaries/decision_gate_report.md",
        ]
        publication_verification = verify_published_artifacts(required_remote_artifacts)
        prediction_directory_verification = []
        executed_backtest_artifacts = []
        if not DRY_RUN:
            for row in staged_rows:
                if row.get("stage_status") != "OK":
                    continue
                relative_prediction_dir = str(Path(row["staged_predictions_dir"]).relative_to(RUN_ROOT))
                prediction_directory_verification.append(
                    verify_published_csv_directory(relative_prediction_dir, int(row["expected_csv_count"]))
                )
            for row in backtest_rows:
                if row.get("status") != "OK":
                    continue
                relative_backtest_dir = Path(row["backtest_dir"]).relative_to(RUN_ROOT)
                for filename in ["backtest_metrics.json", "trade_journal.csv", "daily_holdings.csv"]:
                    executed_backtest_artifacts.append(str(relative_backtest_dir / filename))
            publication_verification.extend(verify_published_artifacts(executed_backtest_artifacts))
        publication_verification_path = SUMMARY_DIR / "drive_publication_verification.json"
        write_json(
            publication_verification_path,
            {
                "drive_run_folder_id": DRIVE_RUN_FOLDER_ID,
                "drive_run_root_url": drive_folder_url(DRIVE_RUN_FOLDER_ID),
                "artifacts": publication_verification,
                "prediction_directories": prediction_directory_verification,
                "executed_backtest_artifacts": executed_backtest_artifacts,
            },
        )
        write_heartbeat(
            "complete",
            status="COMPLETE",
            decision_gate_report=str(report_path),
            drive_run_root_url=drive_folder_url(DRIVE_RUN_FOLDER_ID),
            verified_remote_artifact_count=len(publication_verification),
        )
        final_publication = publish_run_artifacts()
        verify_published_artifacts(
            required_remote_artifacts
            + executed_backtest_artifacts
            + ["summaries/drive_publication_verification.json"]
        )
        remote_heartbeat = read_published_json("heartbeat.json")
        if remote_heartbeat.get("status") != "COMPLETE" or remote_heartbeat.get("phase") != "complete":
            raise RuntimeError(f"Remote heartbeat did not read back COMPLETE: {remote_heartbeat}")
        print("Report:", report_path)
        print("Drive run root:", drive_folder_url(DRIVE_RUN_FOLDER_ID))
        print("Completion publication:", completion_publication)
        print("Final publication:", final_publication)

        if IN_COLAB and DISCONNECT_RUNTIME_WHEN_DONE:
            runtime.unassign()
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {"name": OUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
