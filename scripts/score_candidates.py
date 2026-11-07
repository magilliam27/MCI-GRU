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

Results go to ``<run_dir>/candidates/<arm>/`` (or ``--out_dir``), never over an
existing folder unless ``--force``: ``candidate_summary.json``, ``daily_ic/<arm>.csv``
and ``daily_ic/model.csv`` (the shape ``scripts/pool_candidates.py`` and the baselines
folders share), and ``scores/<arm>.csv``.

Usage::

    python scripts/score_candidates.py --run_dir path/to/W5_run --arm residual_linear \\
        --data_file path/to/panel.csv --pit_universe_csv path/to/pit_universe.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from mci_gru.evaluation.artifacts import write_json_artifact  # noqa: E402
from mci_gru.evaluation.candidate_registry import CANDIDATES, get_candidate  # noqa: E402
from mci_gru.evaluation.candidates import (  # noqa: E402
    CandidateContext,
    check_result,
    panel_up_to,
)
from mci_gru.evaluation.dated_ic import dated_ic_frame  # noqa: E402
from mci_gru.evaluation.experiment_summary import compute_evaluation_summary  # noqa: E402
from mci_gru.protocol import WINDOWS  # noqa: E402
from scripts.rescore_run import (  # noqa: E402
    DEFAULT_TOLERANCE,
    STORED_SUMMARY_FILENAME,
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
    stock_level_data,
)

CANDIDATES_SCHEMA = "mci_gru.candidates.v1"
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


def read_pit_intervals(path: str | None) -> pd.DataFrame | None:
    """The PIT membership file's kdcode, valid_from and valid_to, as strings."""
    if not path:
        return None
    frame = pd.read_csv(path, dtype=str)
    return frame.loc[:, ["kdcode", "valid_from", "valid_to"]].reset_index(drop=True)


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
    candidate = get_candidate(arm)
    run = Path(run_dir)
    target = Path(out_dir) if out_dir is not None else run / DEFAULT_OUT_DIRNAME / arm
    _check_target(target, run, force=force)

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

    def summarise(scores: np.ndarray) -> dict[str, Any]:
        return compute_evaluation_summary(
            scores,
            scoring.labels,
            config,
            panel=scoring.panel,
            dates=scoring.dates,
            kdcodes=scoring.kdcodes,
            rankable_mask=scoring.rankable_mask,
        )

    stored_path = run / STORED_SUMMARY_FILENAME
    stored_sha256 = _check_stored_summary(run, metadata, stored_path)
    summaries = {MODEL_ARM: summarise(model)}
    ic_check = check_ic_reproduced(
        _read_json(stored_path).get("metrics", {}), summaries[MODEL_ARM]["metrics"], tolerance
    )
    ic_check["stored_summary_sha256"] = stored_sha256

    data_controls: dict[str, Any] = {}

    def load_data() -> dict[str, Any]:
        data, controls = stock_level_data(config, metadata, saved.recorded, scoring)
        data_controls.update(controls)
        return data

    context = CandidateContext(
        config=config,
        panel=panel_up_to(scoring.panel, scoring.dates[-1]),
        kdcodes=list(scoring.kdcodes),
        test_dates=list(scoring.dates),
        rankable_mask=scoring.rankable_mask,
        pit_intervals=read_pit_intervals(config.data.pit_universe_csv),
        data_loader=load_data,
    )
    started = time.perf_counter()
    result = candidate(context)
    seconds = time.perf_counter() - started
    scores = check_result(result, context)
    if scoring.rankable_mask is not None:
        scores = np.where(scoring.rankable_mask, scores, np.nan)

    arms = {MODEL_ARM: model, arm: scores}
    summaries[arm] = summarise(scores)
    daily = {name: dated_ic_frame(s, scoring.labels, scoring.dates) for name, s in arms.items()}
    label_t = config.model.label_t
    payload = {
        "schema": CANDIDATES_SCHEMA,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": _code_identity(),
        "candidate": arm,
        "run": {
            "window": run_window(run, config.data.test_start),
            "resolved_config_sha256": metadata.get("resolved_config_sha256"),
            "test_dates": [scoring.dates[0], scoring.dates[-1]],
            "n_test_dates": len(scoring.dates),
            "n_stocks": len(scoring.kdcodes),
            "label_t": label_t,
            "label_type": config.training.label_type,
        },
        "inputs": consumed,
        "controls": {
            "model_ic_reproduced": ic_check,
            "stock_level_inputs": data_controls or None,
            "candidate": result.controls,
            "candidate_seconds": seconds,
        },
        "arms": {
            name: {
                "coverage": _coverage(s, scoring),
                "summary": summaries[name],
            }
            for name, s in arms.items()
        },
        "paired_daily_difference": paired_comparison(
            arms, scoring.labels, scoring.dates, label_t, config.evaluation.ci_level
        ),
        "note": "Exploratory, not a verdict. Both arms are scored by compute_evaluation_summary "
        "on the run's labels, dates, names and PIT tradable mask. Paired rows are the run "
        "minus the candidate per date, on the cells the candidate scored.",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run_dir", required=True, help="one window's run output directory")
    parser.add_argument("--arm", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--out_dir", help="where to write (default <run_dir>/candidates/<arm>)")
    parser.add_argument("--data_file", help="a copy of the run's data file at another path")
    parser.add_argument("--pit_universe_csv", help="a copy of the run's PIT file")
    parser.add_argument("--cessation_events_csv", help="a copy of the run's cessation file")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--force", action="store_true", help="replace an earlier result")
    args = parser.parse_args(argv)
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
