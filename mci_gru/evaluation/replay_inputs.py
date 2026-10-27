"""Derive a saved-prediction replay's inputs from the run it replays (#339).

A run directory (one walk-forward window of ``run_experiment.py`` output) already
records everything the replay needs: ``resolved_config.json`` for the window, label,
reported k values and PIT export cutoff; ``run_metadata.json`` for the stock axis
(``kdcode_list``) and the identity of every data file the run consumed
(``data_inputs``); the window's terminal receipt under ``execution_provenance/`` for
the digest of every saved prediction file. The replay used to take all of these from
paper-era flag defaults instead.

Every file is checked against the SHA-256 the run recorded, so a replay cannot
silently price a run on different data, a different universe or another run's
predictions. Flags remain only as overrides, and each is checked against the run: a
file override must carry the same bytes, a window override must lie inside the run's
test window, and k must be one the run reported. The run records no chosen headline
k (``metrics.top_k`` is a function default), so k is required unless the run
reported exactly one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
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
    #: What the run's labels and returns measured; the replay reads closes on it (#361).
    return_basis: str = "price"

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
            "return_basis": self.return_basis,
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

    run_start, run_end = iso_date(data_cfg["test_start"]), iso_date(data_cfg["test_end"])
    start = run_start if test_start is None else iso_date(test_start)
    end = run_end if test_end is None else iso_date(test_end)
    if start < run_start:
        raise ReplayInputError(f"test_start override {start} precedes the run's {run_start}")
    if end > run_end:
        raise ReplayInputError(f"test_end override {end} is after the run's {run_end}")
    if start > end:
        raise ReplayInputError(f"test_start {start} is after test_end {end}")

    top_k_values = tuple(int(k) for k in resolved.get("evaluation", {}).get("top_k_values", ()))
    if top_k is None and len(top_k_values) != 1:
        raise ReplayInputError(
            "the run records no chosen headline k; pass --top_k, one of the run's "
            f"top_k_values {list(top_k_values)}"
        )
    k = top_k_values[0] if top_k is None else int(top_k)
    if k not in top_k_values:
        raise ReplayInputError(
            f"top_k {k} is not one the run reported (top_k_values {list(top_k_values)})"
        )

    members = tuple(str(code) for code in metadata.get("kdcode_list") or ())
    if not members:
        raise ReplayInputError("run_metadata.json records no kdcode_list (the run's members)")
    if k > len(members):
        raise ReplayInputError(f"top_k {k} exceeds the run's {len(members)} members")

    recorded_data = data_inputs.get(DATA_ROLE)
    if recorded_data is None and not data_inputs and data_cfg.get("source") == "csv":
        # Runs before data_inputs existed fingerprinted the configured CSV directly.
        recorded_data = {
            "sha256": metadata.get("data_file_sha256"),
            "configured_path": metadata.get("data_file"),
        }
    data_path, data_sha = _verified_input(DATA_ROLE, recorded_data, data_file)

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
    _check_predictions_recorded(run, metadata, preds_dir)
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
        # Runs before #361 had no return basis; they measured prices.
        return_basis=str(data_cfg.get("return_basis", "price")),
    )


def iso_date(value: Any) -> str:
    """``YYYY-MM-DD`` for a date given as text; the replay compares dates as text."""
    text = str(value)
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as error:
        raise ReplayInputError(f"date {text!r} is not YYYY-MM-DD") from error


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


def _verified_input(
    role: str, recorded: dict[str, Any] | None, override: str | None
) -> tuple[Path, str]:
    """Locate ``role``'s file and prove it carries the bytes the run consumed.

    Without an override, the configured path is tried first, then the path the run
    resolved it to; the first that carries the recorded bytes is used.
    """
    recorded = recorded or {}
    expected = recorded.get("sha256")
    if not expected:
        raise ReplayInputError(
            f"run_metadata.json records no sha256 for {role}; the replay cannot check the "
            "file it would read"
        )
    if override is not None:
        candidates = [override]
    else:
        candidates = [
            c for c in (recorded.get("configured_path"), recorded.get("resolved_path")) if c
        ]
    if not candidates:
        raise ReplayInputError(f"run_metadata.json records no path for {role}; pass an override")
    problems = []
    for candidate in dict.fromkeys(str(c) for c in candidates):
        try:
            path = resolve_project_data_path(candidate, allow_basename_fallback=False)
        except FileNotFoundError as error:
            problems.append(str(error))
            continue
        actual = sha256_file(path)
        if actual == expected:
            return path, actual
        problems.append(f"{path} has sha256 {actual}")
    raise ReplayInputError(
        f"{role}: no file carries the bytes the run consumed (sha256 {expected}): "
        + "; ".join(problems)
    )


def _check_predictions_recorded(run: Path, metadata: dict[str, Any], directory: Path) -> None:
    """Every prediction file must be one the run's terminal receipt bound by digest."""
    start = (metadata.get("execution_start") or {}).get("path")
    if not start:
        raise ReplayInputError(
            "run_metadata.json records no execution_start, so no receipt binds the predictions"
        )
    receipt_path = run / Path(start).with_name(f"{Path(start).stem}.receipt.json")
    receipt = _read_json(receipt_path)
    if (receipt.get("start") or {}).get("sha256") != metadata["execution_start"].get("sha256"):
        raise ReplayInputError(f"{receipt_path.name} does not belong to this run's start")
    artifacts: dict[str, str] = receipt.get("artifacts") or {}
    try:
        prefix = directory.resolve().relative_to(run.resolve()).as_posix()
    except ValueError as error:
        raise ReplayInputError(
            f"predictions directory {directory} is not inside the run {run}"
        ) from error
    bound = {k: v for k, v in artifacts.items() if k.startswith(f"{prefix}/")}
    present = {f"{prefix}/{f.name}": f for f in directory.glob("*.csv")}
    if not bound:
        raise ReplayInputError(f"the run's receipt binds no files under {prefix}/")
    if set(present) != set(bound):
        raise ReplayInputError(
            f"prediction files differ from the run's receipt: missing "
            f"{sorted(set(bound) - set(present))}, unexpected {sorted(set(present) - set(bound))}"
        )
    changed = sorted(name for name, f in present.items() if sha256_file(f) != bound[name])
    if changed:
        raise ReplayInputError(f"prediction files differ from the run's receipt: {changed}")


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
