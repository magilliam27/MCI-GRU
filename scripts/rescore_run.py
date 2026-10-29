"""Re-score a saved run with the current evaluation code, without retraining (#337).

A run directory (one walk-forward window of ``run_experiment.py`` output) keeps its
ensemble's averaged predictions, its resolved config and the identity of every input
it consumed. That is enough to score it again: the labels, the PIT tradable mask and
the panel the summary reads are rebuilt from the run's own data package, and the saved
scores are put back on the run's stock axis.

Every input is checked before anything is scored. ``resolve_replay_inputs`` (#339)
proves the resolved config, the data file, the PIT file and every saved prediction
file carry the bytes the run recorded; any other input the rebuild reads (a cessation
event file) must match its recorded SHA-256 too.

Re-scoring must reproduce the stored IC and rank IC within ``--tolerance`` (default
1e-4: saved scores are rounded to 5 decimals). That is the control that the rebuilt
labels are the ones the run trained on; if it fails, nothing is written. On a run
with few names, rounding can tie two scores the run ranked apart and move a day's
rank IC by more than that; Pearson IC barely moves, so compare the two.

The result goes beside the old summary, never over it (E8):
``evaluation_summary.rescored.json`` (the current summary schema, plus a ``rescore``
block naming the code commit and every input digest) and ``daily_ic.rescored.csv``
(the dated IC series, written through ``mci_gru.evaluation.dated_ic``). Existing
re-score files are refused unless ``--force``. ``linked_score_ties`` needs the model's
graph and member predictions and is not re-scored.

Usage::

    python scripts/rescore_run.py --run_dir path/to/run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import mci_gru  # noqa: E402
from mci_gru.config import ExperimentConfig, create_config_from_dict  # noqa: E402
from mci_gru.data.data_manager import DataManager  # noqa: E402
from mci_gru.data.pit import (  # noqa: E402
    CESSATION_EVENTS_ROLE,
    CessationEvidence,
    PredictionClock,
    apply_label_mask,
    build_pit_masks,
    cessation_exclusion_mask,
    resolve_cessation_events,
)
from mci_gru.data.preprocessing import (  # noqa: E402
    apply_rank_labels,
    compute_labels,
    label_session_axis,
    return_basis_panel,
)
from mci_gru.data.quality_contract import AdmissionLedger  # noqa: E402
from mci_gru.evaluation.artifacts import write_json_artifact  # noqa: E402
from mci_gru.evaluation.dated_ic import dated_ic_frame  # noqa: E402
from mci_gru.evaluation.experiment_summary import (  # noqa: E402
    REDACTED_ABSOLUTE_PATH,
    compute_evaluation_summary,
)
from mci_gru.evaluation.replay_inputs import (  # noqa: E402
    DATA_ROLE,
    ReplayInputs,
    resolve_replay_inputs,
)
from mci_gru.pipeline import resolve_pit_context, select_universe  # noqa: E402
from mci_gru.utils.hashing import sha256_file  # noqa: E402

STORED_SUMMARY_FILENAME = "evaluation_summary.json"
RESCORED_SUMMARY_FILENAME = "evaluation_summary.rescored.json"
RESCORED_DAILY_IC_FILENAME = "daily_ic.rescored.csv"
RESCORE_SCHEMA = "mci_gru.rescore.v1"
DEFAULT_TOLERANCE = 1e-4
IC_KEYS = ("avg_ic", "avg_rank_ic")
NOT_RESCORED = ("linked_score_ties",)


class RescoreError(ValueError):
    """The run cannot be re-scored as recorded; nothing was written."""


@dataclass(frozen=True)
class ScoringInputs:
    """What the in-run summary scored against, rebuilt from the run's own inputs."""

    panel: pd.DataFrame  # closes on the run's data.return_basis (#361)
    kdcodes: list[str]
    dates: list[str]
    labels: np.ndarray
    rankable_mask: np.ndarray | None
    consumed: dict[str, set[str]]  # role -> sha256 of every read the rebuild made


def rescore_run(
    run_dir: str | Path,
    *,
    data_file: str | None = None,
    pit_universe_csv: str | None = None,
    cessation_events_csv: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    force: bool = False,
) -> Path:
    """Re-score ``run_dir`` and return the path of the new summary.

    The overrides name a copy of an input at another path (a Colab staging
    directory, say); each must carry the bytes the run recorded.
    """
    _check_code_root()
    run = Path(run_dir)
    summary_path = run / RESCORED_SUMMARY_FILENAME
    daily_ic_path = run / RESCORED_DAILY_IC_FILENAME
    if not force:
        for path in (summary_path, daily_ic_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    saved = resolve_run_config(
        run,
        data_file=data_file,
        pit_universe_csv=pit_universe_csv,
        cessation_events_csv=cessation_events_csv,
    )
    config, inputs, metadata, recorded = saved.config, saved.inputs, saved.metadata, saved.recorded

    scoring = rebuild_scoring_inputs(config)
    consumed = check_consumed(scoring.consumed, recorded)
    if scoring.kdcodes != list(inputs.members):
        raise RescoreError(
            f"the rebuilt stock axis ({len(scoring.kdcodes)} names) differs from the run's "
            f"kdcode_list ({len(inputs.members)} names)"
        )
    predictions = load_saved_predictions(inputs.predictions_dir, scoring.kdcodes, scoring.dates)
    if scoring.rankable_mask is not None:
        outside = np.isfinite(predictions) & ~scoring.rankable_mask
        if outside.any():
            i, j = (int(x[0]) for x in np.nonzero(outside))
            raise RescoreError(
                f"{int(outside.sum())} saved scores fall outside the rebuilt PIT tradable mask "
                f"(first: {scoring.kdcodes[j]} on {scoring.dates[i]})"
            )

    summary = compute_evaluation_summary(
        predictions,
        scoring.labels,
        config,
        panel=scoring.panel,
        dates=scoring.dates,
        kdcodes=scoring.kdcodes,
        rankable_mask=scoring.rankable_mask,
    )
    stored_path = run / STORED_SUMMARY_FILENAME
    stored_sha256 = _check_stored_summary(run, metadata, stored_path)
    stored = _read_json(stored_path)
    ic_check = check_ic_reproduced(stored.get("metrics", {}), summary["metrics"], tolerance)
    daily_ic = dated_ic_frame(predictions, scoring.labels, scoring.dates)

    summary["rescore"] = {
        "schema": RESCORE_SCHEMA,
        "rescored_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": _code_identity(),
        "stored_summary": {
            "path": STORED_SUMMARY_FILENAME,
            "sha256": stored_sha256,
            "schema_version": stored.get("schema_version"),
        },
        "resolved_config_sha256": metadata.get("resolved_config_sha256"),
        "inputs": consumed,
        "predictions": {
            "directory": inputs.predictions_dir.name,
            "files": len(scoring.dates),
            "scores_rounded_to_decimals": 5,
        },
        "ic_check": ic_check,
        "daily_ic": RESCORED_DAILY_IC_FILENAME,
        "not_rescored": list(NOT_RESCORED),
    }
    # Both files or neither: the dated IC waits under a temporary name until the
    # summary is written.
    pending = daily_ic_path.with_name(f".{daily_ic_path.name}.pending")
    daily_ic.to_csv(pending, index=False, float_format="%.17g")
    try:
        path = write_json_artifact(summary_path, summary, force=force)
    except BaseException:
        pending.unlink()
        raise
    pending.replace(daily_ic_path)
    return path


@dataclass(frozen=True)
class SavedRunConfig:
    """A saved run's resolved config, its inputs pointed at files with the recorded bytes."""

    config: ExperimentConfig
    inputs: ReplayInputs
    metadata: dict[str, Any]
    recorded: dict[str, Any]  # role -> the identity the run recorded for that input


def resolve_run_config(
    run_dir: str | Path,
    *,
    data_file: str | None = None,
    pit_universe_csv: str | None = None,
    cessation_events_csv: str | None = None,
) -> SavedRunConfig:
    """The run's resolved config, rebuilt to read its own inputs from where they are now.

    ``resolve_replay_inputs`` (#339) proves the resolved config, the data file, the PIT
    file and every saved prediction file carry the bytes the run recorded. Any other
    input a rebuild reads (a cessation event file) is checked by ``check_consumed``
    against ``recorded`` once it has been read.
    """
    run = Path(run_dir)
    metadata = _read_json(run / "run_metadata.json")
    resolved = _read_json(run / metadata.get("resolved_config_path", "resolved_config.json"))
    top_k_values = resolved.get("evaluation", {}).get("top_k_values") or [None]
    # k plays no part in a re-score (every k is scored); the replay asks for one.
    inputs = resolve_replay_inputs(
        run,
        data_file=data_file,
        pit_universe_csv=pit_universe_csv,
        top_k=top_k_values[0],
    )
    recorded = metadata.get("data_inputs") or {}
    if not recorded and resolved["data"].get("source") == "csv":
        # Runs before data_inputs existed fingerprinted the configured CSV directly,
        # the record resolve_replay_inputs has just checked the data file against.
        recorded = {DATA_ROLE: {"sha256": metadata.get("data_file_sha256")}}

    data_cfg = dict(resolved["data"])
    data_cfg["filename"] = str(inputs.data_file)
    if inputs.pit_universe_csv is not None:
        data_cfg["pit_universe_csv"] = str(inputs.pit_universe_csv)
    if data_cfg.get("pit_cessation_events_csv"):
        data_cfg["pit_cessation_events_csv"] = cessation_events_csv or _recorded_path(
            recorded, CESSATION_EVENTS_ROLE, data_cfg["pit_cessation_events_csv"]
        )
    config = create_config_from_dict({**resolved, "data": data_cfg})
    return SavedRunConfig(config, inputs, metadata, recorded)


def rebuild_scoring_inputs(config: ExperimentConfig) -> ScoringInputs:
    """The panel, stock axis, test dates, labels and tradable mask the run scored on.

    Uses the pipeline's own loaders and label and mask functions, in the order
    ``prepare_data`` applies them, but skips features, normalisation, the graph and
    the tensors: none of them changes which rows exist or what a stock returned.
    """
    admission = AdmissionLedger()
    data_manager = DataManager(config.data, admission=admission)
    panel = data_manager.load()
    observations = data_manager.input_observations
    admission.require_admitted(observations)
    pit = resolve_pit_context(config, observations, admission)
    admission.require_admitted(observations)

    filtered, kdcodes = select_universe(panel, data_manager, config, pit)
    _, _, test_df = data_manager.split_by_period(filtered)
    dates = sorted(str(d) for d in test_df["dt"].unique())
    his_t, label_t = config.model.his_t, config.model.label_t
    # Labels and the summary read closes on the run's return basis (#361).
    returns = return_basis_panel(panel, config.data.return_basis)

    if pit.masked_panel:
        assert pit.intervals is not None
        clock = pit.clock or PredictionClock()
        cessation = pit.cessation or CessationEvidence(configured_path=None, events=())
        events = resolve_cessation_events(cessation, label_session_axis(panel, kdcodes), clock)
        masks = build_pit_masks(
            filtered,
            panel,
            kdcodes,
            dates,
            his_t,
            label_t,
            pit.intervals,
            cessation_exclusion_mask(events, kdcodes, dates, clock),
        )
        # A masked panel's returns are unfilled and masked to the loss set before any
        # rank transform, as apply_pit_masks_to_tensors does.
        labels = compute_labels(returns, kdcodes, dates, label_t, fill_missing=False)
        labels = apply_label_mask(labels, masks.loss)
        if config.training.label_type == "rank":
            labels = apply_rank_labels(labels, masks.loss)
        rankable_mask = np.asarray(masks.tradable, dtype=bool)
    else:
        labels = compute_labels(returns, kdcodes, dates, label_t, fill_missing=True)
        if config.training.label_type == "rank":
            labels = apply_rank_labels(labels)
        rankable_mask = None

    consumed: dict[str, set[str]] = {}
    sealed = observations.freeze().to_dict()
    for use in sealed["uses"]:
        identity = sealed["observations"][use["observation_id"]].get("identity", {})
        consumed.setdefault(use["role"], set()).add(str(identity.get("sha256")))
    return ScoringInputs(returns, kdcodes, dates, labels, rankable_mask, consumed)


def load_saved_predictions(directory: Path, kdcodes: list[str], dates: list[str]) -> np.ndarray:
    """Saved scores on the ``(dates, kdcodes)`` axis; NaN where the run wrote no row.

    One file per test date, named ``<date>.csv``, as the ensemble writes them.
    """
    files = {path.stem: path for path in Path(directory).glob("*.csv")}
    if sorted(files) != list(dates):
        raise RescoreError(
            f"prediction files differ from the rebuilt test dates: missing "
            f"{sorted(set(dates) - set(files))}, unexpected {sorted(set(files) - set(dates))}"
        )
    column = {kdcode: j for j, kdcode in enumerate(kdcodes)}
    out = np.full((len(dates), len(kdcodes)), np.nan)
    for i, date in enumerate(dates):
        frame = pd.read_csv(files[date], dtype={"kdcode": str})
        if frame["kdcode"].duplicated().any():
            raise RescoreError(f"{files[date].name} scores a stock more than once")
        unknown = sorted(set(frame["kdcode"]) - set(column))
        if unknown:
            raise RescoreError(f"{files[date].name} scores stocks off the run's axis: {unknown}")
        out[i, [column[k] for k in frame["kdcode"]]] = frame["score"].to_numpy(np.float64)
    return out


def check_ic_reproduced(
    stored: dict[str, Any], rescored: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    """Each stored IC mean must come back within ``tolerance``, or the re-score stops."""
    check: dict[str, Any] = {"tolerance": tolerance}
    for key in IC_KEYS:
        if key not in stored:
            raise RescoreError(f"the stored summary has no {key} to check the re-score against")
        before, after = float(stored[key]), float(rescored.get(key, np.nan))
        difference = abs(after - before)
        if not difference <= tolerance:
            raise RescoreError(
                f"re-scored {key} {after!r} differs from the stored {before!r} by "
                f"{difference!r} (tolerance {tolerance}). Either the rebuilt labels are not "
                "the ones the run scored, or rounding the saved scores to 5 decimals tied "
                "names the run had ranked apart. Nothing was written"
            )
        check[key] = {"stored": before, "rescored": after, "abs_difference": difference}
    return check


def check_consumed(consumed: dict[str, set[str]], recorded: dict[str, Any]) -> dict[str, str]:
    """Every read the rebuild made must carry the bytes the run consumed for that role."""
    if DATA_ROLE not in consumed:
        raise RescoreError(f"the rebuild read no {DATA_ROLE}, so nothing ties it to the run")
    checked = {}
    for role, digests in sorted(consumed.items()):
        expected = (recorded.get(role) or {}).get("sha256")
        if expected is None:
            raise RescoreError(f"the run recorded no sha256 for {role}, which the re-score read")
        if digests != {expected}:
            raise RescoreError(
                f"{role} has sha256 {sorted(digests)}, not the {expected} the run consumed"
            )
        checked[role] = expected
    return checked


def _recorded_path(recorded: dict[str, Any], role: str, configured: str) -> str:
    """The first of the configured and recorded paths that exists; its bytes are checked later."""
    record = recorded.get(role) or {}
    candidates = [
        c
        for c in (configured, record.get("configured_path"), record.get("resolved_path"))
        if c and c != REDACTED_ABSOLUTE_PATH
    ]
    for candidate in candidates:
        if Path(candidate).is_file() or (REPO_DIR / candidate).is_file():
            return str(candidate)
    raise RescoreError(
        f"no file for {role} at {candidates or 'any recorded path'}; pass an override"
    )


def _check_stored_summary(run: Path, metadata: dict[str, Any], path: Path) -> str:
    """The summary the IC is checked against must be the one the run's receipt bound."""
    actual = sha256_file(path)
    start = (metadata.get("execution_start") or {}).get("path")
    if start:
        receipt = run / Path(start).with_name(f"{Path(start).stem}.receipt.json")
        bound = (_read_json(receipt).get("artifacts") or {}).get(path.name)
        if bound is not None and bound != actual:
            raise RescoreError(
                f"{path.name} has sha256 {actual}, not the {bound} the run's receipt bound"
            )
    return actual


def _check_code_root() -> None:
    """The evaluation code imported must be this checkout's, whose commit is recorded."""
    package = Path(mci_gru.__file__).resolve().parent
    if not package.is_relative_to(REPO_DIR):
        raise RescoreError(
            f"mci_gru is imported from {package}, not from {REPO_DIR}, so the recorded "
            "commit would not describe the code that scored the run"
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RescoreError(f"{path.name} not found in {path.parent}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _code_identity() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(REPO_DIR), *args], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = git("status", "--porcelain", "--", "mci_gru", "scripts")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run_dir", required=True, help="one window's run output directory")
    parser.add_argument("--data_file", help="a copy of the run's data file at another path")
    parser.add_argument("--pit_universe_csv", help="a copy of the run's PIT file")
    parser.add_argument("--cessation_events_csv", help="a copy of the run's cessation file")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--force", action="store_true", help="replace earlier re-score files")
    args = parser.parse_args(argv)
    path = rescore_run(
        args.run_dir,
        data_file=args.data_file,
        pit_universe_csv=args.pit_universe_csv,
        cessation_events_csv=args.cessation_events_csv,
        tolerance=args.tolerance,
        force=args.force,
    )
    check = json.loads(path.read_text(encoding="utf-8"))["rescore"]["ic_check"]
    for key in IC_KEYS:
        print(f"{key}: stored {check[key]['stored']:.6f}, re-scored {check[key]['rescored']:.6f}")
    print(f"Re-scored summary written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
