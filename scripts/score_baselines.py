"""Score simple baselines on a saved run's own axis, through the run's own evaluation (#360).

A model has to beat signals anyone could compute in one line before an arm difference
means much. This script scores four of them on the dates, names, labels and PIT
tradable mask a saved run was scored on, with the code that wrote the run's
``evaluation_summary.json`` (``compute_evaluation_summary`` and the run's resolved
config), and compares the run with each:

* ``reversal_1w`` and ``reversal_1m``: minus the return over the last 5 and 21 sessions;
* ``momentum_12_1``: the return from close ``D-252`` to close ``D-21``;
* ``linear_ridge``: a cross-sectional ridge regression on the run's own stock-level
  input window, fitted on the run's training dates, penalty chosen on validation IC.

See ``mci_gru.evaluation.baselines`` for the definitions.

Every input is checked as the re-score checks it (#337): the run's resolved config,
data file, PIT file and saved predictions must carry the recorded bytes, and every
other read must match the digest the run recorded for its role. Two controls must pass
before anything is written:

* the run's saved predictions, scored here, reproduce its stored IC and rank IC
  within ``--tolerance`` (the labels and mask are the run's);
* the linear model's inputs, rebuilt with the pipeline, carry exactly the train-period
  normalisation the run recorded for every stock-level column (they are the run's
  inputs). The global columns (regime, VIX, credit) are left out of that rebuild: they
  hold one value per date for every stock, so a linear ranking cannot use them.

The paired comparison is the run's daily IC and rank IC minus each baseline's, on the
dates both scored, with the overlap-aware HAC t-test of ``paired_inference``. It is
evidence, not a gate.

Results go to ``<run_dir>/baselines/`` (or ``--out_dir``), never over an existing
folder unless ``--force``: ``baselines_summary.json``, ``daily_ic/<arm>.csv`` and
``scores/<arm>.csv`` (``dt, kdcode, score``, the shape ``--baseline`` in
``scripts/evaluation/tsfm_prediction_report.py`` reads).

Usage::

    python scripts/score_baselines.py --run_dir path/to/run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from mci_gru.data.quality_contract import AdmissionLedger  # noqa: E402
from mci_gru.evaluation.artifacts import write_json_artifact  # noqa: E402
from mci_gru.evaluation.baselines import (  # noqa: E402
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP,
    REVERSAL_LOOKBACKS,
    RIDGE_ALPHAS,
    fit_cross_sectional_ridge,
    linear_scores,
    momentum_scores,
    reversal_scores,
)
from mci_gru.evaluation.dated_ic import dated_ic_frame  # noqa: E402
from mci_gru.evaluation.experiment_summary import compute_evaluation_summary  # noqa: E402
from mci_gru.evaluation.paired_inference import (  # noqa: E402
    align_daily_series,
    paired_mean_inference,
)
from mci_gru.features import FeatureEngineer  # noqa: E402
from mci_gru.pipeline import prepare_data  # noqa: E402
from scripts.rescore_run import (  # noqa: E402
    DEFAULT_TOLERANCE,
    STORED_SUMMARY_FILENAME,
    RescoreError,
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

if TYPE_CHECKING:
    from mci_gru.config import ExperimentConfig

BASELINES_SCHEMA = "mci_gru.baselines.v1"
DEFAULT_OUT_DIRNAME = "baselines"
SUMMARY_FILENAME = "baselines_summary.json"
MODEL_ARM = "model"
#: The rebuilt stock-level normalisation must equal the run's to this relative tolerance.
NORM_RTOL = 1e-12
GLOBAL_FEATURE_FLAGS = ("include_global_regime", "include_vix", "include_credit_spread")

DEFINITIONS = {
    **{
        name: f"minus the return from close D-{lookback} to close D (sessions)"
        for name, lookback in REVERSAL_LOOKBACKS.items()
    },
    "momentum_12_1": f"return from close D-{MOMENTUM_LOOKBACK} to close D-{MOMENTUM_SKIP} "
    "(sessions)",
    "linear_ridge": "cross-sectional ridge on the run's stock-level input window; fitted on "
    "training dates, penalty chosen on validation mean IC",
}


class BaselineError(RescoreError):
    """The baselines cannot be scored as the run was; nothing was written."""


@dataclasses.dataclass(frozen=True)
class LinearBaseline:
    scores: np.ndarray  # (test dates, stocks), before the rankable mask
    controls: dict[str, Any]


def stock_level_config(config: ExperimentConfig) -> ExperimentConfig:
    """The run's config without its global inputs, read from no auxiliary source."""
    features = dataclasses.replace(config.features, **dict.fromkeys(GLOBAL_FEATURE_FLAGS, False))
    data = dataclasses.replace(
        config.data, auxiliary_snapshot_mode="source", auxiliary_snapshot_directory=None
    )
    return dataclasses.replace(config, features=features, data=data)


def _consumed_digests(frozen_observations: Any) -> dict[str, set[str]]:
    sealed = frozen_observations.to_dict()
    consumed: dict[str, set[str]] = {}
    for use in sealed["uses"]:
        identity = sealed["observations"][use["observation_id"]].get("identity", {})
        consumed.setdefault(use["role"], set()).add(str(identity.get("sha256")))
    return consumed


def check_stock_level_inputs(
    metadata: dict[str, Any],
    rebuilt_columns: list[str],
    global_columns: set[str],
    norm_means: dict[str, float],
    norm_stds: dict[str, float],
) -> dict[str, Any]:
    """The rebuilt columns are the run's minus its global ones, normalised as the run was."""
    run_columns = list(metadata["feature_cols"])
    expected = [column for column in run_columns if column not in global_columns]
    if rebuilt_columns != expected:
        raise BaselineError(
            f"rebuilt stock-level columns {rebuilt_columns} are not the run's {run_columns} "
            f"without its global columns {sorted(global_columns)}"
        )
    for name, rebuilt, stored in (
        ("norm_means", norm_means, metadata.get("norm_means") or {}),
        ("norm_stds", norm_stds, metadata.get("norm_stds") or {}),
    ):
        for column in rebuilt_columns:
            if column not in stored:
                raise BaselineError(f"the run recorded no {name} for {column}")
            if not np.isclose(
                float(rebuilt[column]), float(stored[column]), rtol=NORM_RTOL, atol=0
            ):
                raise BaselineError(
                    f"rebuilt {name}[{column}] {float(rebuilt[column])!r} differs from the "
                    f"run's {float(stored[column])!r}: these are not the inputs it trained on"
                )
    return {
        "stock_level_columns": rebuilt_columns,
        "global_columns_left_out": [c for c in run_columns if c in global_columns],
        "normalisation_matches_run": True,
        "normalisation_rtol": NORM_RTOL,
    }


def linear_baseline(
    config: ExperimentConfig,
    metadata: dict[str, Any],
    recorded: dict[str, Any],
    scoring: ScoringInputs,
) -> LinearBaseline:
    """Rebuild the run's stock-level inputs, fit the ridge on its training dates, score test."""
    if config.data.normalisation != "zscore":
        raise BaselineError(
            f"the linear baseline checks a z-score normalisation; the run used "
            f"{config.data.normalisation!r}"
        )
    stock_config = stock_level_config(config)
    run_engineer = FeatureEngineer(config.features)
    stock_engineer = FeatureEngineer(stock_config.features)
    global_columns = set(run_engineer.get_feature_columns()) - set(
        stock_engineer.get_feature_columns()
    )
    data = prepare_data(stock_config, stock_engineer, admission=AdmissionLedger())
    check_consumed(_consumed_digests(data["input_observations"]), recorded)
    controls = check_stock_level_inputs(
        metadata, list(data["feature_cols"]), global_columns, data["norm_means"], data["norm_stds"]
    )
    if list(data["kdcode_list"]) != scoring.kdcodes:
        raise BaselineError("the rebuilt inputs' stock axis differs from the run's")
    if [str(d) for d in data["test_dates"]] != scoring.dates:
        raise BaselineError("the rebuilt inputs' test dates differ from the run's")

    fit = fit_cross_sectional_ridge(
        data["stock_features_train"],
        data["train_labels"],
        data["stock_features_val"],
        data["val_labels"],
        train_mask=data.get("train_loss_mask"),
        val_mask=data.get("val_tradable_mask"),
    )
    controls.update(
        {
            "ridge_alpha": fit.alpha,
            "ridge_alphas": list(RIDGE_ALPHAS),
            "validation_mean_ic": {str(a): ic for a, ic in fit.validation_ic.items()},
            "train_rows": fit.train_rows,
            "train_dates": fit.train_dates,
            "inputs_per_stock": int(fit.coefficients.size),
            "his_t": config.model.his_t,
        }
    )
    return LinearBaseline(linear_scores(data["stock_features_test"], fit.coefficients), controls)


def baseline_scores(scoring: ScoringInputs, linear: LinearBaseline | None) -> dict[str, np.ndarray]:
    """Every baseline's scores on the run's axis, kept only where the run could rank."""
    signals = {
        name: reversal_scores(scoring.panel, scoring.kdcodes, scoring.dates, lookback)
        for name, lookback in REVERSAL_LOOKBACKS.items()
    }
    signals["momentum_12_1"] = momentum_scores(scoring.panel, scoring.kdcodes, scoring.dates)
    if linear is not None:
        signals["linear_ridge"] = linear.scores
    if scoring.rankable_mask is None:
        return signals
    return {name: np.where(scoring.rankable_mask, s, np.nan) for name, s in signals.items()}


def paired_comparison(
    daily: dict[str, pd.DataFrame], label_t: int, ci_level: float
) -> list[dict[str, Any]]:
    """The run minus each baseline, per date, on the dates both scored."""
    rows = []
    for column in ("ic", "rank_ic"):
        for arm in daily:
            if arm == MODEL_ARM:
                continue
            try:
                frame = align_daily_series(
                    {
                        MODEL_ARM: daily[MODEL_ARM].set_index("dt")[column],
                        arm: daily[arm].set_index("dt")[column],
                    }
                )
            except ValueError:  # no date both scored: nothing to compare
                rows.append(
                    {"metric": column, "n_common_dates": 0, "arm": MODEL_ARM, "control": arm}
                )
                continue
            inference = paired_mean_inference(
                frame[MODEL_ARM] - frame[arm],
                arm=MODEL_ARM,
                control=arm,
                label_horizon=label_t,
                ci_level=ci_level,
            )
            rows.append(
                {"metric": column, "n_common_dates": len(frame), **dataclasses.asdict(inference)}
            )
    return rows


def _coverage(scores: np.ndarray, scoring: ScoringInputs) -> dict[str, Any]:
    rankable = (
        np.ones(scores.shape, dtype=bool)
        if scoring.rankable_mask is None
        else np.asarray(scoring.rankable_mask, dtype=bool)
    )
    scored = int((np.isfinite(scores) & rankable).sum())
    return {
        "scored_cells": scored,
        "rankable_cells": int(rankable.sum()),
        "share_scored": scored / int(rankable.sum()) if rankable.any() else None,
    }


def _long_scores(scores: np.ndarray, dates: list[str], kdcodes: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(scores, index=dates, columns=kdcodes).rename_axis("dt")
    long = frame.stack(future_stack=True).rename("score").reset_index()
    long.columns = ["dt", "kdcode", "score"]
    return long[np.isfinite(long["score"])]


def _strict_json(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    return value


def score_baselines(
    run_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    data_file: str | None = None,
    pit_universe_csv: str | None = None,
    cessation_events_csv: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    include_linear: bool = True,
    force: bool = False,
) -> Path:
    """Score the baselines and the run on the run's axis; return the summary's path."""
    _check_code_root()
    run = Path(run_dir)
    target = Path(out_dir) if out_dir is not None else run / DEFAULT_OUT_DIRNAME
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")

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
        raise BaselineError("the rebuilt stock axis differs from the run's kdcode_list")
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

    # The cheap control first: the labels and mask are the run's only if its own
    # predictions give back the IC its receipt-bound summary stored.
    stored_path = run / STORED_SUMMARY_FILENAME
    stored_sha256 = _check_stored_summary(run, metadata, stored_path)
    summaries = {MODEL_ARM: summarise(model)}
    ic_check = check_ic_reproduced(
        _read_json(stored_path).get("metrics", {}), summaries[MODEL_ARM]["metrics"], tolerance
    )
    ic_check["stored_summary_sha256"] = stored_sha256

    linear = linear_baseline(config, metadata, saved.recorded, scoring) if include_linear else None
    arms = {MODEL_ARM: model, **baseline_scores(scoring, linear)}
    summaries.update({arm: summarise(s) for arm, s in arms.items() if arm != MODEL_ARM})
    daily = {
        arm: dated_ic_frame(scores, scoring.labels, scoring.dates) for arm, scores in arms.items()
    }
    label_t = config.model.label_t
    payload = {
        "schema": BASELINES_SCHEMA,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": _code_identity(),
        "run": {
            "resolved_config_sha256": metadata.get("resolved_config_sha256"),
            "test_dates": [scoring.dates[0], scoring.dates[-1]] if scoring.dates else None,
            "n_test_dates": len(scoring.dates),
            "n_stocks": len(scoring.kdcodes),
            "label_t": label_t,
            "label_type": config.training.label_type,
        },
        "inputs": consumed,
        "controls": {
            "model_ic_reproduced": ic_check,
            "linear_inputs": None if linear is None else linear.controls,
        },
        "arms": {
            arm: {
                "definition": "the run's saved averaged predictions"
                if arm == MODEL_ARM
                else DEFINITIONS[arm],
                "coverage": _coverage(scores, scoring),
                "summary": summaries[arm],
            }
            for arm, scores in arms.items()
        },
        "paired_daily_difference": paired_comparison(daily, label_t, config.evaluation.ci_level),
        "note": "Evidence, not a gate. Every arm is scored by compute_evaluation_summary on the "
        "run's labels, dates, names and PIT tradable mask. Paired rows are the run minus the "
        "baseline per date, on the dates both scored.",
    }

    # Everything goes to a pending folder first, so a failure leaves no partial result.
    pending = target.with_name(f".{target.name}.pending")
    if pending.exists():
        shutil.rmtree(pending)
    (pending / "daily_ic").mkdir(parents=True)
    (pending / "scores").mkdir()
    try:
        for arm, scores in arms.items():
            daily[arm].to_csv(
                pending / "daily_ic" / f"{arm}.csv", index=False, float_format="%.17g"
            )
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
    parser.add_argument("--out_dir", help="where to write (default <run_dir>/baselines)")
    parser.add_argument("--data_file", help="a copy of the run's data file at another path")
    parser.add_argument("--pit_universe_csv", help="a copy of the run's PIT file")
    parser.add_argument("--cessation_events_csv", help="a copy of the run's cessation file")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--no_linear", action="store_true", help="skip the linear model (no feature rebuild)"
    )
    parser.add_argument("--force", action="store_true", help="replace an earlier result")
    args = parser.parse_args(argv)
    path = score_baselines(
        args.run_dir,
        out_dir=args.out_dir,
        data_file=args.data_file,
        pit_universe_csv=args.pit_universe_csv,
        cessation_events_csv=args.cessation_events_csv,
        tolerance=args.tolerance,
        include_linear=not args.no_linear,
        force=args.force,
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    for arm, entry in summary["arms"].items():
        m = entry["summary"]["metrics"]
        print(
            f"{arm:>14}: IC {m.get('avg_ic')}, rank IC {m.get('avg_rank_ic')}, "
            f"coverage {entry['coverage']['share_scored']}"
        )
    for row in summary["paired_daily_difference"]:
        print(
            f"model - {row['control']} ({row['metric']}): mean {row.get('mean_delta')}, "
            f"HAC t {row.get('hac_t')}, p {row.get('hac_p')}, dates {row['n_common_dates']}"
        )
    print(f"Baseline summary written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
