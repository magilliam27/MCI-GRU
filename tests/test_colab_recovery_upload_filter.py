from pathlib import Path

from scripts.colab_recovery_upload_filter import (
    count_averaged_prediction_csvs,
    iter_recovery_upload_files,
    should_upload_recovery_artifact,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def test_default_recovery_upload_excludes_per_model_csvs_and_broad_csvs(tmp_path: Path) -> None:
    run_dir = tmp_path / "top10_lambdarank_ic_2023_seed271828" / "20260702_011015"

    averaged = _touch(run_dir / "averaged_predictions" / "2023-01-09.csv")
    root_config = _touch(run_dir / "config.yaml")
    hydra_config = _touch(run_dir / ".hydra" / "config.yaml")
    training_log = _touch(run_dir / "training_20260702_011015.log")

    per_model = _touch(run_dir / "predictions_model_0" / "2023-01-09.csv")
    root_csv = _touch(run_dir / "training_rows.csv")
    broad_csv = _touch(run_dir / "diagnostics" / "rank_snapshot.csv")
    checkpoint = _touch(run_dir / "checkpoints" / "model_0_best.pth")

    included = {path.relative_to(run_dir).as_posix() for path in iter_recovery_upload_files(run_dir)}

    assert averaged.relative_to(run_dir).as_posix() in included
    assert root_config.relative_to(run_dir).as_posix() in included
    assert hydra_config.relative_to(run_dir).as_posix() in included
    assert training_log.relative_to(run_dir).as_posix() in included
    assert per_model.relative_to(run_dir).as_posix() not in included
    assert root_csv.relative_to(run_dir).as_posix() not in included
    assert broad_csv.relative_to(run_dir).as_posix() not in included
    assert checkpoint.relative_to(run_dir).as_posix() not in included


def test_per_model_and_checkpoint_uploads_are_explicit_opt_in(tmp_path: Path) -> None:
    run_dir = tmp_path / "row"
    per_model = _touch(run_dir / "predictions_model_19" / "2024-12-31.csv")
    checkpoint = _touch(run_dir / "checkpoints" / "model_19_best.pth")

    assert not should_upload_recovery_artifact(per_model, run_dir)
    assert not should_upload_recovery_artifact(checkpoint, run_dir)

    assert should_upload_recovery_artifact(
        per_model,
        run_dir,
        upload_per_model_predictions=True,
    )
    assert should_upload_recovery_artifact(
        checkpoint,
        run_dir,
        upload_checkpoints=True,
    )


def test_count_averaged_prediction_csvs_counts_only_ensemble_predictions(tmp_path: Path) -> None:
    run_dir = tmp_path / "row"
    _touch(run_dir / "averaged_predictions" / "2023-01-09.csv")
    _touch(run_dir / "averaged_predictions" / "2023-01-10.csv")
    _touch(run_dir / "averaged_predictions" / "notes.txt")
    _touch(run_dir / "predictions_model_0" / "2023-01-09.csv")

    assert count_averaged_prediction_csvs(run_dir) == 2
