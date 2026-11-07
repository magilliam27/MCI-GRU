"""Score a candidate model on a saved run's own axis, through the run's own evaluation (#384).

A candidate (``mci_gru.evaluation.candidates``) is any model other than MCI-GRU. This
script hands it one window's run as a ``CandidateContext``, takes its test scores and
scores them with ``compute_evaluation_summary`` and the run's resolved config, on the
dates, names, labels and PIT tradable mask the run was scored on: the measurement
MCI-GRU's own ``evaluation_summary.json`` is. It then pairs the candidate against the
run's saved predictions, day by day.

The checks are ``scripts/score_baselines.py``'s (#360), and must pass before anything
is written:

* every input carries the bytes the run recorded;
* the run's saved predictions, scored here, reproduce its stored IC and rank IC;
* when the candidate reads ``context.data``, the stock-level rebuild carries the run's
  exact train-period normalisation and sits on its stock axis and test dates.

The candidate sees the run's panel only up to its last test date, so it cannot read
the locked 2026 holdout; ``resolve_replay_inputs`` already refuses a run that scored it.

**Another universe, with no MCI-GRU run on it** (``--template_run``). There is no saved
run to score against on, say, the full S&P 500, so the run of the same window on the
original universe serves as a template: its resolved config (the recipe and the
window's dates) is kept, and the data block is replaced by ``--data_config``'s (a
``configs/data`` file) except for the dates. The labels, dates, names and PIT mask are
then rebuilt from the new universe with the same code. ``--package_manifest`` must be
the package that data config pins (its sha256), and the data and PIT files must carry
the digests that manifest records. There is no model arm, so no IC-reproduction
control and no pairing; the holdout guard is applied to the window's dates directly.

Every summary records a ``scoring_fingerprint`` (a digest of the dates, names, labels
and tradable mask), so ``pool_candidates.py`` can check that folders of one window and
universe scored the same thing.

Results go to ``<run_dir>/candidates/<arm>/`` (or ``--out_dir``), never over an
existing folder unless ``--force``: ``candidate_summary.json``, ``daily_ic/<arm>.csv``
and, with a run, ``daily_ic/model.csv`` (the shape ``scripts/pool_candidates.py`` and
the baselines folders share), and ``scores/<arm>.csv``.

Usage::

    python scripts/score_candidates.py --run_dir path/to/W5_run --arm residual_linear \\
        --data_file path/to/panel.csv --pit_universe_csv path/to/pit_universe.csv

    python scripts/score_candidates.py --template_run path/to/W5_run --arm residual_linear \\
        --data_config configs/data/sp500_full_eodhd.yaml --package_manifest path/to/r1.json \\
        --data_file path/to/full_panel.csv --pit_universe_csv path/to/full_pit.csv \\
        --out_dir path/to/out
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from mci_gru.config import ExperimentConfig, create_config_from_dict  # noqa: E402
from mci_gru.data.pit import load_pit_intervals  # noqa: E402
from mci_gru.data.quality_contract import AdmissionLedger  # noqa: E402
from mci_gru.evaluation.artifacts import write_json_artifact  # noqa: E402
from mci_gru.evaluation.candidate_registry import CANDIDATES, get_candidate  # noqa: E402
from mci_gru.evaluation.candidates import (  # noqa: E402
    CandidateContext,
    check_result,
    panel_up_to,
)
from mci_gru.evaluation.dated_ic import dated_ic_frame  # noqa: E402
from mci_gru.evaluation.experiment_summary import compute_evaluation_summary  # noqa: E402
from mci_gru.features import FeatureEngineer  # noqa: E402
from mci_gru.pipeline import prepare_data  # noqa: E402
from mci_gru.protocol import WINDOWS, require_outside_holdout  # noqa: E402
from mci_gru.utils.hashing import sha256_file  # noqa: E402
from scripts.rescore_run import (  # noqa: E402
    DEFAULT_TOLERANCE,
    STORED_SUMMARY_FILENAME,
    ScoringInputs,
    _check_code_root,
    _check_stored_summary,
    _code_identity,
    _read_json,
    check_consumed,
    check_ic_reproduced,
    load_saved_predictions,
    rebuild_scoring_inputs,
    resolve_run_config,
)
from scripts.score_baselines import (  # noqa: E402
    MODEL_ARM,
    BaselineError,
    _coverage,
    _long_scores,
    _strict_json,
    paired_comparison,
    stock_level_config,
    stock_level_data,
)

if TYPE_CHECKING:
    import pandas as pd

CANDIDATES_SCHEMA = "mci_gru.candidates.v1"
#: Data-config keys a template keeps from its own run: the window's dates.
WINDOW_DATE_KEYS = (
    "train_start",
    "train_end",
    "val_start",
    "val_end",
    "test_start",
    "test_end",
)
DEFAULT_OUT_DIRNAME = "candidates"
SUMMARY_FILENAME = "candidate_summary.json"
RUN_RECORD_FILENAME = "colab_run_record.json"


class CandidateError(BaselineError):
    """The candidate cannot be scored as the run was; nothing was written."""


def run_window(run: Path, test_start: str | None) -> str | None:
    """The protocol window a run scored: its run record's, else the one its test start opens."""
    record = run / RUN_RECORD_FILENAME
    if record.is_file():
        window = _read_json(record).get("window")
        if window:
            return str(window)
    for name, window in WINDOWS.items():
        if window.test_start == test_start:
            return name
    return None


def read_pit_intervals(path: str | None, open_valid_to: str | None) -> pd.DataFrame | None:
    """The PIT membership file read as the pipeline reads it: ISO dates, open ends filled."""
    if not path:
        return None
    frame = load_pit_intervals(path, open_valid_to=open_valid_to)
    return frame.loc[:, ["kdcode", "valid_from", "valid_to"]].reset_index(drop=True)


def scoring_fingerprint(scoring: ScoringInputs) -> str:
    """A digest of what a candidate is scored on: dates, names, labels and tradable mask."""
    digest = hashlib.sha256()
    digest.update("\n".join(scoring.dates).encode())
    digest.update(b"\0")
    digest.update("\n".join(scoring.kdcodes).encode())
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(scoring.labels, dtype=np.float64).tobytes())
    if scoring.rankable_mask is not None:
        digest.update(np.ascontiguousarray(scoring.rankable_mask, dtype=bool).tobytes())
    return digest.hexdigest()


def _check_target(target: Path, run: Path, *, force: bool) -> None:
    resolved, run_resolved = target.resolve(), run.resolve()
    if run_resolved == resolved or run_resolved.is_relative_to(resolved):
        raise CandidateError(f"the output folder {target} holds the run itself")
    if not target.exists():
        return
    if not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
    summary = target / SUMMARY_FILENAME
    if not summary.is_file() or _read_json(summary).get("schema") != CANDIDATES_SCHEMA:
        raise CandidateError(
            f"--force replaces only an earlier candidate result, and {target} is not one"
        )


@dataclasses.dataclass
class Prepared:
    """Everything a candidate is scored with, from a run or from a template."""

    config: ExperimentConfig
    scoring: ScoringInputs
    load_data: Any  # () -> (prepare_data output, controls)
    run_info: dict[str, Any]
    inputs: dict[str, Any]
    model: np.ndarray | None = None
    model_controls: dict[str, Any] | None = None


def _summarise(prepared: Prepared, scores: np.ndarray) -> dict[str, Any]:
    scoring = prepared.scoring
    return compute_evaluation_summary(
        scores,
        scoring.labels,
        prepared.config,
        panel=scoring.panel,
        dates=scoring.dates,
        kdcodes=scoring.kdcodes,
        rankable_mask=scoring.rankable_mask,
    )


def prepare_from_run(
    run: Path,
    *,
    data_file: str | None,
    pit_universe_csv: str | None,
    cessation_events_csv: str | None,
    tolerance: float,
) -> Prepared:
    """The run's own axis, with its saved predictions as the model arm (checked first)."""
    saved = resolve_run_config(
        run,
        data_file=data_file,
        pit_universe_csv=pit_universe_csv,
        cessation_events_csv=cessation_events_csv,
    )
    config, metadata = saved.config, saved.metadata
    scoring = rebuild_scoring_inputs(config)
    consumed = check_consumed(scoring.consumed, saved.recorded)
    if scoring.kdcodes != list(saved.inputs.members):
        raise CandidateError("the rebuilt stock axis differs from the run's kdcode_list")
    model = load_saved_predictions(saved.inputs.predictions_dir, scoring.kdcodes, scoring.dates)
    prepared = Prepared(
        config=config,
        scoring=scoring,
        load_data=lambda: stock_level_data(config, metadata, saved.recorded, scoring),
        run_info={
            "mode": "run",
            "run_dir": run.name,
            "window": run_window(run, config.data.test_start),
            "resolved_config_sha256": metadata.get("resolved_config_sha256"),
        },
        inputs=consumed,
        model=model,
    )
    stored_path = run / STORED_SUMMARY_FILENAME
    stored_sha256 = _check_stored_summary(run, metadata, stored_path)
    model_summary = _summarise(prepared, model)
    ic_check = check_ic_reproduced(
        _read_json(stored_path).get("metrics", {}), model_summary["metrics"], tolerance
    )
    ic_check["stored_summary_sha256"] = stored_sha256
    prepared.model_controls = {"summary": model_summary, "ic_check": ic_check}
    return prepared


def _manifest_digest(manifest: dict[str, Any], configured: str) -> str:
    name = Path(configured).name
    entries = [f for f in manifest.get("files", []) if Path(str(f.get("path"))).name == name]
    if len(entries) != 1:
        raise CandidateError(f"the package manifest lists {len(entries)} files named {name}")
    return str(entries[0]["sha256"])


def template_config(
    template_run: Path,
    data_config: Path,
    package_manifest: Path | None,
    data_file: str,
    pit_universe_csv: str | None,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """The template run's resolved config with another universe's data block.

    The window's dates stay the template's. The data and PIT files must carry the
    digests the data config's pinned package manifest records.
    """
    metadata = _read_json(template_run / "run_metadata.json")
    resolved = _read_json(
        template_run / metadata.get("resolved_config_path", "resolved_config.json")
    )
    block = yaml.safe_load(data_config.read_text(encoding="utf-8")) or {}
    data_cfg = dict(resolved["data"])
    data_cfg.update({k: v for k, v in block.items() if k not in WINDOW_DATE_KEYS})

    checked: dict[str, Any] = {"data_config": str(data_config)}
    pinned = block.get("input_package_manifest")
    if pinned:
        if package_manifest is None:
            raise CandidateError(
                f"{data_config} pins the package manifest {pinned}; pass --package_manifest"
            )
        manifest_sha = sha256_file(package_manifest)
        expected = str(block.get("input_package_manifest_sha256") or "")
        if manifest_sha != expected:
            raise CandidateError(
                f"{package_manifest} has sha256 {manifest_sha}, not the {expected} "
                f"{data_config} pins"
            )
        manifest = json.loads(package_manifest.read_text(encoding="utf-8"))
        checked["package_manifest_sha256"] = manifest_sha
        files = {"data.filename": (block["filename"], data_file)}
        if block.get("use_pit_universe"):
            if not pit_universe_csv:
                raise CandidateError(f"{data_config} uses a PIT universe; pass --pit_universe_csv")
            files["data.pit_universe_csv"] = (block["pit_universe_csv"], pit_universe_csv)
        for role, (configured, actual) in files.items():
            want = _manifest_digest(manifest, configured)
            have = sha256_file(Path(actual))
            if have != want:
                raise CandidateError(
                    f"{actual} has sha256 {have}; the manifest records {want} for {configured}"
                )
            checked[role] = have
    # The digests are checked above; the loader reads the files where they are now.
    data_cfg.update(
        {
            "filename": str(data_file),
            "pit_universe_csv": str(pit_universe_csv) if pit_universe_csv else None,
            "input_package_manifest": None,
            "input_package_manifest_sha256": None,
            "input_package_root": None,
        }
    )
    config = create_config_from_dict({**resolved, "data": data_cfg})
    spans = {key: getattr(config.data, key) for key in WINDOW_DATE_KEYS}
    require_outside_holdout(spans, label_t=config.model.label_t, what="candidate scoring")
    checked["window"] = run_window(template_run, config.data.test_start)
    checked["template_run"] = template_run.name
    checked["template_resolved_config_sha256"] = metadata.get("resolved_config_sha256")
    return config, checked


def prepare_from_template(
    template_run: Path,
    *,
    data_config: Path,
    package_manifest: Path | None,
    data_file: str,
    pit_universe_csv: str | None,
) -> Prepared:
    """Another universe's axis, built with a template run's config and window."""
    config, checked = template_config(
        template_run, data_config, package_manifest, data_file, pit_universe_csv
    )
    scoring = rebuild_scoring_inputs(config)

    def load() -> tuple[dict[str, Any], dict[str, Any]]:
        stock_config = stock_level_config(config)
        data = prepare_data(
            stock_config, FeatureEngineer(stock_config.features), admission=AdmissionLedger()
        )
        if list(data["kdcode_list"]) != scoring.kdcodes:
            raise CandidateError("the rebuilt inputs' stock axis differs from the scoring axis")
        if [str(d) for d in data["test_dates"]] != scoring.dates:
            raise CandidateError("the rebuilt inputs' test dates differ from the scoring dates")
        return data, {
            "stock_level_columns": list(data["feature_cols"]),
            "normalisation_matches_run": None,
            "note": "no run on this universe, so nothing to check the normalisation against",
        }

    return Prepared(
        config=config,
        scoring=scoring,
        load_data=load,
        run_info={"mode": "template", **checked},
        inputs={role: sorted(digests) for role, digests in scoring.consumed.items()},
    )


def score_prepared(prepared: Prepared, arm: str, target: Path) -> Path:
    """Score ``arm`` on a prepared axis and write the result folder; return its summary."""
    candidate = get_candidate(arm)
    config, scoring = prepared.config, prepared.scoring
    data_controls: dict[str, Any] = {}

    def load_data() -> dict[str, Any]:
        data, controls = prepared.load_data()
        data_controls.update(controls)
        return data

    context = CandidateContext(
        config=config,
        panel=panel_up_to(scoring.panel, scoring.dates[-1]),
        kdcodes=list(scoring.kdcodes),
        test_dates=list(scoring.dates),
        rankable_mask=scoring.rankable_mask,
        pit_intervals=read_pit_intervals(
            config.data.pit_universe_csv if config.data.use_pit_universe else None,
            config.data.pit_export_cutoff,
        ),
        data_loader=load_data,
    )
    started = time.perf_counter()
    result = candidate(context)
    seconds = time.perf_counter() - started
    scores = check_result(result, context)
    if scoring.rankable_mask is not None:
        scores = np.where(scoring.rankable_mask, scores, np.nan)

    arms = {arm: scores}
    summaries = {arm: _summarise(prepared, scores)}
    if prepared.model is not None:
        arms = {MODEL_ARM: prepared.model, **arms}
        summaries[MODEL_ARM] = prepared.model_controls["summary"]
    daily = {name: dated_ic_frame(s, scoring.labels, scoring.dates) for name, s in arms.items()}
    label_t = config.model.label_t
    payload = {
        "schema": CANDIDATES_SCHEMA,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": _code_identity(),
        "candidate": arm,
        "run": {
            **prepared.run_info,
            "test_dates": [scoring.dates[0], scoring.dates[-1]],
            "n_test_dates": len(scoring.dates),
            "n_stocks": len(scoring.kdcodes),
            "label_t": label_t,
            "label_type": config.training.label_type,
            "data_file": Path(config.data.filename).name,
            "scoring_fingerprint": scoring_fingerprint(scoring),
        },
        "inputs": prepared.inputs,
        "controls": {
            "model_ic_reproduced": (prepared.model_controls or {}).get("ic_check"),
            "stock_level_inputs": data_controls or None,
            "candidate": result.controls,
            "candidate_seconds": seconds,
        },
        "arms": {
            name: {"coverage": _coverage(s, scoring), "summary": summaries[name]}
            for name, s in arms.items()
        },
        "paired_direction": "model_minus_candidate",
        "paired_daily_difference": (
            paired_comparison(
                arms, scoring.labels, scoring.dates, label_t, config.evaluation.ci_level
            )
            if prepared.model is not None
            else []
        ),
        "note": "Exploratory, not a verdict. Every arm is scored by compute_evaluation_summary "
        "on the labels, dates, names and PIT tradable mask rebuilt from the run's (or the "
        "template's) config. Paired rows, when there is a run, are the run minus the "
        "candidate per date, on the cells the candidate scored.",
    }

    pending = target.with_name(f".{target.name}.pending")
    if pending.exists():
        raise CandidateError(f"{pending} is left from an interrupted scoring; move it aside first")
    (pending / "daily_ic").mkdir(parents=True)
    (pending / "scores").mkdir()
    try:
        for name, frame in daily.items():
            frame.to_csv(pending / "daily_ic" / f"{name}.csv", index=False, float_format="%.17g")
        _long_scores(scores, scoring.dates, scoring.kdcodes).to_csv(
            pending / "scores" / f"{arm}.csv", index=False, float_format="%.17g"
        )
        write_json_artifact(pending / SUMMARY_FILENAME, _strict_json(payload))
        if target.exists():
            shutil.rmtree(target)
        pending.replace(target)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return target / SUMMARY_FILENAME


def score_candidate(
    run_dir: str | Path,
    arm: str,
    *,
    out_dir: str | Path | None = None,
    data_file: str | None = None,
    pit_universe_csv: str | None = None,
    cessation_events_csv: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    force: bool = False,
) -> Path:
    """Score candidate ``arm`` and the run on the run's axis; return the summary's path."""
    _check_code_root()
    if arm == MODEL_ARM:
        raise CandidateError(f"{MODEL_ARM!r} names the run's own predictions, not a candidate")
    get_candidate(arm)
    run = Path(run_dir)
    target = Path(out_dir) if out_dir is not None else run / DEFAULT_OUT_DIRNAME / arm
    _check_target(target, run, force=force)
    prepared = prepare_from_run(
        run,
        data_file=data_file,
        pit_universe_csv=pit_universe_csv,
        cessation_events_csv=cessation_events_csv,
        tolerance=tolerance,
    )
    return score_prepared(prepared, arm, target)


def score_candidate_on_universe(
    template_run: str | Path,
    arm: str,
    *,
    out_dir: str | Path,
    data_config: str | Path,
    data_file: str,
    pit_universe_csv: str | None = None,
    package_manifest: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Score candidate ``arm`` on another universe, with a template run's window."""
    _check_code_root()
    if arm == MODEL_ARM:
        raise CandidateError(f"{MODEL_ARM!r} names a run's own predictions, not a candidate")
    get_candidate(arm)
    template = Path(template_run)
    target = Path(out_dir)
    _check_target(target, template, force=force)
    prepared = prepare_from_template(
        template,
        data_config=Path(data_config),
        package_manifest=Path(package_manifest) if package_manifest else None,
        data_file=data_file,
        pit_universe_csv=pit_universe_csv,
    )
    return score_prepared(prepared, arm, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run_dir", help="one window's run output directory")
    source.add_argument(
        "--template_run", help="a run of the window on another universe, used as a template"
    )
    parser.add_argument("--arm", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--out_dir", help="where to write (default <run_dir>/candidates/<arm>)")
    parser.add_argument("--data_file", help="a copy of the run's data file at another path")
    parser.add_argument("--pit_universe_csv", help="a copy of the run's PIT file")
    parser.add_argument("--cessation_events_csv", help="a copy of the run's cessation file")
    parser.add_argument("--data_config", help="with --template_run: the universe's data yaml")
    parser.add_argument(
        "--package_manifest", help="with --template_run: the package the data yaml pins"
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--force", action="store_true", help="replace an earlier result")
    args = parser.parse_args(argv)
    if args.template_run:
        if not (args.data_config and args.data_file and args.out_dir):
            parser.error("--template_run needs --data_config, --data_file and --out_dir")
        path = score_candidate_on_universe(
            args.template_run,
            args.arm,
            out_dir=args.out_dir,
            data_config=args.data_config,
            data_file=args.data_file,
            pit_universe_csv=args.pit_universe_csv,
            package_manifest=args.package_manifest,
            force=args.force,
        )
    else:
        path = score_candidate(
            args.run_dir,
            args.arm,
            out_dir=args.out_dir,
            data_file=args.data_file,
            pit_universe_csv=args.pit_universe_csv,
            cessation_events_csv=args.cessation_events_csv,
            tolerance=args.tolerance,
            force=args.force,
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    for name, entry in summary["arms"].items():
        m = entry["summary"]["metrics"]
        print(
            f"{name:>22}: IC {m.get('avg_ic')}, rank IC {m.get('avg_rank_ic')}, "
            f"coverage {entry['coverage']['share_scored']}"
        )
    for row in summary["paired_daily_difference"]:
        print(
            f"model - {row['control']} ({row['metric']}): mean {row.get('mean_delta')}, "
            f"HAC t {row.get('hac_t')}, p {row.get('hac_p')}, dates {row['n_common_dates']}"
        )
    print(f"Candidate summary written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
