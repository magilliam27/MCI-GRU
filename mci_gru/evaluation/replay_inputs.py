"""Derive a saved-prediction replay's inputs from the run it replays (#339).

A run directory (one walk-forward window of ``run_experiment.py`` output) already
records everything the replay needs: ``resolved_config.json`` for the window, label,
reported k values and PIT export cutoff; ``run_metadata.json`` for the stock axis
(``kdcode_list``) and the identity of every data file the run consumed
(``data_inputs``); ``evaluation_summary.json`` for the headline k. The replay used to
take all of these from paper-era flag defaults instead.

Every file is checked against the SHA-256 the run recorded when it consumed it, so a
replay cannot silently price a run on different data or a different universe. Flags
remain only as overrides, and each is checked against the run: a file override must
carry the same bytes, a window override must lie inside the run's test window, and a
k override must be one the run reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.data.path_resolver import resolve_project_data_path
from mci_gru.utils.hashing import sha256_file

DATA_ROLE = "data.filename"
PIT_ROLE = "data.pit_universe_csv"
PREDICTIONS_DIRNAME = "averaged_predictions"


class ReplayInputError(ValueError):
    """The run directory and the replay's inputs do not describe the same run."""


@dataclass(frozen=True)
class ReplayInputs:
    """Inputs for one replay, each derived from (or checked against) the run."""

    run_dir: Path
    predictions_dir: Path
    data_file: Path
    data_file_sha256: str
    pit_universe_csv: Path | None
    pit_universe_sha256: str | None
    pit_export_cutoff: str | None
    test_start: str
    test_end: str
    label_t: int
    top_k: int
    top_k_values: tuple[int, ...]
    members: tuple[str, ...]

    def backtest_config(self) -> dict[str, Any]:
        """The input keys of the replay engine's config dict."""
        return {
            "run_dir": str(self.run_dir),
            "data_file": str(self.data_file),
            "data_file_sha256": self.data_file_sha256,
            "pit_universe_csv": str(self.pit_universe_csv) if self.pit_universe_csv else None,
            "pit_universe_sha256": self.pit_universe_sha256,
            "pit_export_cutoff": self.pit_export_cutoff,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "label_t": self.label_t,
            "top_k": self.top_k,
            "members": list(self.members),
        }


def resolve_replay_inputs(
    run_dir: str | Path,
    *,
    predictions_dir: str | None = None,
    data_file: str | None = None,
    pit_universe_csv: str | None = None,
    pit_export_cutoff: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    label_t: int | None = None,
    top_k: int | None = None,
) -> ReplayInputs:
    """Read the replay's inputs from ``run_dir``; every keyword is a checked override.

    Raises :class:`ReplayInputError` when a file's bytes differ from the run's record,
    when an override falls outside what the run did, or when the saved predictions
    name a stock or date outside the run.
    """
    run = Path(run_dir)
    metadata = _read_json(run / "run_metadata.json")
    resolved = _read_resolved_config(run, metadata)
    data_cfg = resolved.get("data", {})
    data_inputs = metadata.get("data_inputs") or {}

    run_label_t = int(resolved["model"]["label_t"])
    if int(metadata.get("label_t", run_label_t)) != run_label_t:
        raise ReplayInputError(
            f"run_metadata.json label_t {metadata.get('label_t')} differs from the resolved "
            f"config's {run_label_t}"
        )
    if label_t is not None and int(label_t) != run_label_t:
        raise ReplayInputError(f"label_t override {label_t} differs from the run's {run_label_t}")

    run_start, run_end = str(data_cfg["test_start"]), str(data_cfg["test_end"])
    start = run_start if test_start is None else str(test_start)
    end = run_end if test_end is None else str(test_end)
    if start < run_start:
        raise ReplayInputError(f"test_start override {start} precedes the run's {run_start}")
    if end > run_end:
        raise ReplayInputError(f"test_end override {end} is after the run's {run_end}")
    if start > end:
        raise ReplayInputError(f"test_start {start} is after test_end {end}")

    top_k_values = tuple(int(k) for k in resolved.get("evaluation", {}).get("top_k_values", ()))
    k = _headline_k(run) if top_k is None else int(top_k)
    if k is None:
        raise ReplayInputError(
            "the run's evaluation_summary.json records no headline top_k; pass --top_k, "
            f"one of the run's top_k_values {list(top_k_values)}"
        )
    if k not in top_k_values:
        raise ReplayInputError(
            f"top_k {k} is not one the run reported (top_k_values {list(top_k_values)})"
        )

    members = tuple(str(code) for code in metadata.get("kdcode_list") or ())
    if not members:
        raise ReplayInputError("run_metadata.json records no kdcode_list (the run's members)")
    if k > len(members):
        raise ReplayInputError(f"top_k {k} exceeds the run's {len(members)} members")

    data_path, data_sha = _verified_input(
        DATA_ROLE,
        data_inputs.get(DATA_ROLE),
        data_file,
        legacy_sha256=metadata.get("data_file_sha256"),
        legacy_path=metadata.get("data_file"),
    )

    run_cutoff = data_cfg.get("pit_export_cutoff")
    if data_cfg.get("use_pit_universe"):
        pit_path, pit_sha = _verified_input(PIT_ROLE, data_inputs.get(PIT_ROLE), pit_universe_csv)
        if pit_export_cutoff is not None and str(pit_export_cutoff) != str(run_cutoff):
            raise ReplayInputError(
                f"pit_export_cutoff override {pit_export_cutoff} differs from the run's "
                f"{run_cutoff}"
            )
        cutoff = None if run_cutoff is None else str(run_cutoff)
    else:
        if pit_universe_csv is not None or pit_export_cutoff is not None:
            raise ReplayInputError(
                "the run used no PIT universe, so a PIT override has nothing to be checked "
                "against; its universe is its members"
            )
        pit_path, pit_sha, cutoff = None, None, None

    preds_dir = Path(predictions_dir) if predictions_dir else run / PREDICTIONS_DIRNAME
    _check_predictions(preds_dir, members, run_start, run_end)

    return ReplayInputs(
        run_dir=run,
        predictions_dir=preds_dir,
        data_file=data_path,
        data_file_sha256=data_sha,
        pit_universe_csv=pit_path,
        pit_universe_sha256=pit_sha,
        pit_export_cutoff=cutoff,
        test_start=start,
        test_end=end,
        label_t=run_label_t,
        top_k=k,
        top_k_values=top_k_values,
        members=members,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReplayInputError(
            f"{path.name} not found in {path.parent}; --run_dir must be one window's output "
            "directory from run_experiment.py"
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_resolved_config(run: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    name = metadata.get("resolved_config_path", "resolved_config.json")
    path = run / name
    resolved = _read_json(path)
    recorded = metadata.get("resolved_config_sha256")
    if not recorded:
        raise ReplayInputError("run_metadata.json records no resolved_config_sha256")
    actual = sha256_file(path)
    if actual != recorded:
        raise ReplayInputError(
            f"{name} sha256 {actual} differs from run_metadata.json's resolved_config_sha256 "
            f"{recorded}"
        )
    return resolved


def _headline_k(run: Path) -> int | None:
    path = run / "evaluation_summary.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    k = summary.get("metrics", {}).get("top_k", summary.get("top_k"))
    return None if k is None else int(k)


def _verified_input(
    role: str,
    recorded: dict[str, Any] | None,
    override: str | None,
    *,
    legacy_sha256: str | None = None,
    legacy_path: str | None = None,
) -> tuple[Path, str]:
    """Locate ``role``'s file and prove it carries the bytes the run consumed."""
    expected = (recorded or {}).get("sha256") or legacy_sha256
    configured = (recorded or {}).get("configured_path") or legacy_path
    if not expected:
        raise ReplayInputError(
            f"run_metadata.json records no sha256 for {role}; the replay cannot check the "
            "file it would read"
        )
    candidate = override if override is not None else configured
    if not candidate:
        raise ReplayInputError(f"run_metadata.json records no path for {role}; pass an override")
    try:
        path = resolve_project_data_path(str(candidate), allow_basename_fallback=False)
    except FileNotFoundError as error:
        raise ReplayInputError(f"{role}: {error}") from error
    actual = sha256_file(path)
    if actual != expected:
        raise ReplayInputError(
            f"{role} at {path} has sha256 {actual}, but the run consumed {expected}"
        )
    return path, actual


def _check_predictions(directory: Path, members: tuple[str, ...], start: str, end: str) -> None:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise ReplayInputError(f"no prediction CSVs in {directory}")
    frame = pd.concat(
        (pd.read_csv(f, usecols=["kdcode", "dt"], dtype={"kdcode": str}) for f in files),
        ignore_index=True,
    )
    outsiders = sorted(set(frame["kdcode"]) - set(members))
    if outsiders:
        raise ReplayInputError(f"predictions name stocks outside the run's members: {outsiders}")
    dates = pd.to_datetime(frame["dt"]).dt.strftime("%Y-%m-%d")
    if dates.min() < start or dates.max() > end:
        raise ReplayInputError(
            f"predictions span {dates.min()}..{dates.max()}, outside the run's test window "
            f"{start}..{end}"
        )
