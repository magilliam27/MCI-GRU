# LambdaRankIC G4 Recovery Relaunch Patch

Use this note before relaunching `lambdarankic_110_name_missing_predictions_g4`.
It is intentionally narrow: fix the upload policy, salvage the completed rows,
and resume only missing 110-name LambdaRankIC `pair_cap=8192` saved-prediction
rows. Do not run this without explicit user approval in the visible Colab UI.

## Current State

- Runtime from run tag `20260701_185554` was disconnected on 2026-07-02 after
  an opaque post-training upload loop kept the G4 backend attached.
- Manifest-complete rows:
  - 2022 seed `271828`
  - 2022 seed `161803`
- Likely salvageable row:
  - 2023 seed `271828`
  - Drive run folder: `https://drive.google.com/drive/folders/1Xtli0wHhLEeLxtVuaRJqSf-9DlZ4_DCy`
  - Drive `averaged_predictions`: `https://drive.google.com/drive/folders/1i2eiUSi0CGpatkzZ64hqAz04v71ob6FV`
  - Visible date range from connector listing: `2023-01-09.csv` through
    `2023-12-29.csv`
- No fourth training log was present, so there is no evidence that 2023 seed
  `161803`, 2024, or 2025 rows started.

## Patch The Recovery Cell

Replace the old broad upload filter and upload tree function with this block.
It assumes the existing recovery cell already defines `Path`, `drive_service`,
`ensure_folder`, `upload_or_update_file`, and `heartbeat(status, phase, **extra)`.

```python
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
UPLOAD_PER_MODEL_PREDICTIONS = False
UPLOAD_CHECKPOINTS = False


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
        return rel.name in ROOT_RUN_FILES or (
            rel.name.startswith("training_") and suffix == ".log"
        )
    return False


def iter_recovery_upload_files(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.rglob("*"))
        if should_upload_run_file(path, run_dir)
    ]


def count_local_averaged_prediction_csvs(run_dir: Path) -> int:
    return len(list((run_dir / "averaged_predictions").glob("*.csv")))


def count_drive_csv_files(folder_id: str) -> int:
    assert drive_service is not None
    count = 0
    page_token = None
    while True:
        response = (
            drive_service.files()
            .list(
                q=(
                    f"'{folder_id}' in parents and trashed = false "
                    "and mimeType != 'application/vnd.google-apps.folder'"
                ),
                fields="nextPageToken,files(id,name,mimeType)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        count += sum(
            1
            for item in response.get("files", [])
            if item.get("name", "").endswith(".csv")
        )
        page_token = response.get("nextPageToken")
        if not page_token:
            return count


def upload_selected_run_tree(
    run_dir: Path,
    drive_job_root_id: str,
    *,
    job_label: str = "",
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
```

If the call site currently uses:

```python
drive_run_id, pred_folder_id, uploaded = upload_selected_run_tree(run_dir, drive_job_root_id)
```

change it to:

```python
drive_run_id, pred_folder_id, uploaded = upload_selected_run_tree(
    run_dir,
    drive_job_root_id,
    job_label=f"{job['year']}_seed{job['base_seed']}",
)
```

## Resume Matrix Guard

Add this before launching jobs. It prevents rerunning rows that already have
durable averaged predictions.

```python
RECOVERED_ROWS = {
    (2022, 271828),
    (2022, 161803),
}
SALVAGEABLE_AVERAGED_PREDICTIONS = {
    (2023, 271828): {
        "folder_id": "1i2eiUSi0CGpatkzZ64hqAz04v71ob6FV",
        "expected_csv_count": 246,
        "url": "https://drive.google.com/drive/folders/1i2eiUSi0CGpatkzZ64hqAz04v71ob6FV",
    },
}


def recovered_or_salvageable(job: dict) -> bool:
    key = (job["year"], job["base_seed"])
    if key in RECOVERED_ROWS:
        return True
    salvage = SALVAGEABLE_AVERAGED_PREDICTIONS.get(key)
    if salvage is None:
        return False
    observed = count_drive_csv_files(salvage["folder_id"])
    salvage["observed_csv_count"] = observed
    return observed == salvage["expected_csv_count"]


original_job_count = len(training_jobs)
training_jobs = [job for job in training_jobs if not recovered_or_salvageable(job)]
heartbeat(
    "RUNNING",
    "resume_matrix_filtered",
    original_job_count=original_job_count,
    remaining_job_count=len(training_jobs),
    recovered_rows=sorted([f"{year}_seed{seed}" for year, seed in RECOVERED_ROWS]),
    salvageable_rows={
        f"{year}_seed{seed}": metadata
        for (year, seed), metadata in SALVAGEABLE_AVERAGED_PREDICTIONS.items()
    },
)
print("Remaining training jobs:")
for job in training_jobs:
    print("-", job["year"], job["base_seed"], job.get("name", ""))
```

Expected remaining rows if 2023 seed `271828` counts exactly 246 CSVs:

- 2023 seed `161803`
- 2024 seed `271828`
- 2024 seed `161803`
- 2025 seed `271828`
- 2025 seed `161803`

## 2026-07-02 Safe Relaunch Attempt

- `scripts/colab_lambdarankic_110_name_missing_predictions_relaunch.py` now contains the full patched launcher used for the visible Colab attempt. It is safer than manually rebuilding the older cell.
- The launcher was pasted into the notebook as Cell 2 and run on a G4 backend. The in-cell gate printed `GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition`.
- The attempt blocked in `google.colab.auth.authenticate_user()` after Colab's in-notebook credential prompt. The external `accounts.google.com` OAuth flow was not accessible to the agent, so the run was interrupted before any Drive heartbeat, run root, or training row was created.
- The G4 runtime was then disconnected/deleted and the toolbar returned to `Reconnect G4 High-RAM`.
- To retry, the user must manually complete the Google/Drive OAuth prompt, then rerun only the patched Cell 2 launcher. Do not run the older broad-upload cell.

## 2026-07-07 Retry Status

- After the Codex update, Chrome control worked again and the same notebook was reopened on G4 High-RAM.
- Important UI trap: the notebook now contains a stale upper recovery cell plus a lower base64 launcher cell. Do **not** run the stale upper cell. It still trains recovered 2022 rows.
- One accidental stale-upper-cell run was started and immediately interrupted:
  - Abandoned run folder: `https://drive.google.com/drive/folders/1AbGldDnV6SzTuXuooy1mkYG_skOmj9hZ`
  - Run tag: `20260707_013458`
  - It began `2022_seed271828`; it was interrupted before completing a model row and should not be treated as evidence.
- The Colab session was restarted, a fresh wrapper generated from local `scripts/colab_lambdarankic_110_name_missing_predictions_relaunch.py` was pasted into the lower launcher cell, and that lower cell was run.
- Active fixed run:
  - Run folder: `https://drive.google.com/drive/folders/1fYmtPg97O52SgTRsU_XgwuVaWFbpj9W_`
  - Heartbeat: `https://drive.google.com/file/d/1WfegZoL2M-pM1KZabssWSOa3DhL9LIdh/view`
  - Manifest: `https://drive.google.com/file/d/18SJLx-wVP2vGHb6EgucOIB2q2ErMCKx1/view`
  - Run tag: `20260707_014129`
  - GPU proof in manifest: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
  - Manifest guardrails: `original_job_count=8`, `remaining_job_count=5`, `run_backtests=false`, recovered rows `2022_seed161803` and `2022_seed271828`, salvageable `2023_seed271828` observed at `246` CSVs.
  - First active job at handoff: `training_2023_seed161803`. Visible output around `2026-07-07T01:45Z` showed it had completed model 3/20 and started model 4/20.
- Resume by reading the heartbeat first. If the run reaches upload, expected upload counts should be around averaged predictions plus metadata, not thousands of per-model CSVs.
- If heartbeat stops advancing for a long time outside normal training, interrupt and disconnect/delete the runtime rather than letting G4 burn.

## Relaunch Checklist

1. Visible runtime must show G4 high-RAM/G4-class GPU. L4/T4 should remain blocked.
2. Run only the patched recovery cell; do not run the older unpatched cell.
3. Confirm the first heartbeat after filtering reports five remaining jobs if the
   2023 seed `271828` folder has 246 CSVs.
4. During each upload, heartbeat must advance at least every 25 files and the
   per-row upload count should be roughly the averaged CSV count plus metadata,
   not thousands of files.
5. After completion or failure, verify Colab toolbar returns to reconnect state.
