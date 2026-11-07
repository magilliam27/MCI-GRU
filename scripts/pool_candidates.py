"""Pool candidate and baseline scorings over the five protocol windows (#384).

``scripts/score_candidates.py`` and ``scripts/score_baselines.py`` each write one
window's dated IC series per arm, as ``daily_ic/<arm>.csv`` inside a result folder
whose summary names the run's window and label horizon. This script reads any number
of those folders, joins each arm's windows into one dated series, and reports:

* per arm and window, the mean IC and rank IC;
* per arm, the IC and rank IC pooled over every date it scored, with the
  overlap-aware Hansen-Hodrick t-statistic on ``label_t - 2`` lags (the run's own IC
  interval's convention, ``overlap_hac_lags``), and
  the Newey-West (Bartlett) t on ``label_t - 2`` lags that the protocol's five-window
  table reports, so the two can be read side by side;
* per arm against the reference arm (``model``, MCI-GRU's saved predictions by
  default), the arm minus the reference per date, pooled over every window, with the
  HAC t-test of ``paired_inference``.

The same ``model`` series appears in every folder of a window. It must be identical in
each, and every candidate folder of a window must record the same scoring fingerprint
(dates, names, labels and tradable mask): the controls that every folder scored the same
run, or on a universe with no run, the same universe and window. A window or date
scored twice for one arm is refused. On a universe with no run there is no ``model``
arm; name another reference with ``--reference``. Arms need not share windows; each row says which
windows it covers, and a paired row uses only the dates both arms scored.

Exploratory, never a verdict: a verdict needs a pre-registered experiment (#372).

Usage::

    python scripts/pool_candidates.py --result W1/candidates/residual_linear \\
        --result W1/baselines ... --out pooled.json [--markdown pooled.md]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from mci_gru.evaluation.paired_inference import paired_mean_inference  # noqa: E402
from mci_gru.evaluation.statistics import (  # noqa: E402
    hansen_hodrick_mean_inference,
    newey_west_std,
    overlap_hac_lags,
)
from mci_gru.protocol import WINDOWS  # noqa: E402

POOLED_SCHEMA = "mci_gru.candidates_pooled.v1"
SUMMARY_FILES = ("candidate_summary.json", "baselines_summary.json")
REFERENCE_ARM = "model"
METRICS = ("ic", "rank_ic")


def _window_of(summary: dict[str, Any]) -> str:
    run = summary.get("run", {})
    window = run.get("window")
    if window:
        return str(window)
    first = (run.get("test_dates") or [None])[0]
    for name, spec in WINDOWS.items():
        if spec.test_start <= str(first) <= spec.test_end:
            return name
    raise ValueError(f"cannot tell which window scored test dates starting {first}")


def read_result(folder: Path) -> tuple[str, int, dict[str, pd.DataFrame], str | None]:
    """A result folder's window, label horizon, dated IC frame per arm and fingerprint."""
    summary_path = next((folder / n for n in SUMMARY_FILES if (folder / n).is_file()), None)
    if summary_path is None:
        raise ValueError(f"{folder} holds neither of {SUMMARY_FILES}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    window = _window_of(summary)
    label_t = int(summary["run"]["label_t"])
    daily = {}
    for path in sorted((folder / "daily_ic").glob("*.csv")):
        frame = pd.read_csv(path, dtype={"dt": str}).set_index("dt")
        daily[path.stem] = frame.loc[:, list(METRICS)]
    if not daily:
        raise ValueError(f"{folder} has no daily_ic/*.csv")
    return window, label_t, daily, summary["run"].get("scoring_fingerprint")


def collect(folders: list[Path]) -> tuple[dict[str, dict[str, pd.DataFrame]], int]:
    """arm -> window -> dated IC frame, checking the folders of a window scored one thing.

    Two checks: a candidate summary's ``scoring_fingerprint`` (dates, names, labels and
    tradable mask) must agree across every folder of a window that records one, and the
    run's own ``model`` series, where folders carry it, must be identical.
    """
    by_arm: dict[str, dict[str, pd.DataFrame]] = {}
    fingerprints: dict[str, str] = {}
    horizons = set()
    for folder in folders:
        window, label_t, daily, fingerprint = read_result(folder)
        horizons.add(label_t)
        if fingerprint is not None and fingerprints.setdefault(window, fingerprint) != fingerprint:
            raise ValueError(
                f"{folder} scored other labels, names or dates on {window} than an earlier "
                "folder: the folders did not score the same run or universe"
            )
        for arm, frame in daily.items():
            windows = by_arm.setdefault(arm, {})
            if window not in windows:
                windows[window] = frame
                continue
            if arm != REFERENCE_ARM:
                raise ValueError(f"arm {arm!r} is scored twice on {window}")
            if not frame.equals(windows[window]):
                raise ValueError(
                    f"{folder} scored a different {REFERENCE_ARM!r} series on {window} than an "
                    "earlier folder: the folders did not score the same run"
                )
    if len(horizons) != 1:
        raise ValueError(f"the results score different label horizons: {sorted(horizons)}")
    return by_arm, horizons.pop()


def pooled_series(windows: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pooled = pd.concat([windows[w] for w in sorted(windows, key=list(WINDOWS).index)])
    if not pooled.index.is_unique:
        raise ValueError(f"date {pooled.index[pooled.index.duplicated()][0]} is scored twice")
    return pooled.sort_index()


def _inference(values: pd.Series, label_t: int) -> dict[str, Any]:
    finite = values.to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    lags = overlap_hac_lags(label_t)
    # Uniform weights on the overlap's own lag count, as the run's IC interval and the
    # paired rows use; hansen_hodrick_mean_inference takes lags = label_horizon - 1.
    result = hansen_hodrick_mean_inference(finite, label_horizon=lags + 1)
    nw_se = newey_west_std(finite, lags=lags) / np.sqrt(finite.size) if finite.size else 0.0
    return {
        "n_dates": result.n_obs,
        "mean": result.mean,
        "t": result.t_stat,
        "p": result.p_value,
        "t_newey_west": float(result.mean / nw_se) if nw_se > 0 else None,
        "newey_west_lags": lags,
    }


def pool(
    folders: list[Path], *, reference: str = REFERENCE_ARM, ci_level: float = 0.95
) -> dict[str, Any]:
    """The pooled report over every folder given, paired against ``reference``."""
    by_arm, label_t = collect(folders)
    if reference not in by_arm:
        raise ValueError(
            f"no folder scored the reference arm {reference!r}; arms: {sorted(by_arm)}. "
            "On a universe with no run, name one with --reference."
        )
    order = [reference, *sorted(a for a in by_arm if a != reference)]
    pooled = {arm: pooled_series(by_arm[arm]) for arm in order}
    arms = []
    for arm in order:
        windows = sorted(by_arm[arm], key=list(WINDOWS).index)
        arms.append(
            {
                "arm": arm,
                "windows": windows,
                "per_window": {
                    w: {m: float(by_arm[arm][w][m].mean()) for m in METRICS} for w in windows
                },
                **{m: _inference(pooled[arm][m], label_t) for m in METRICS},
            }
        )
    paired = []
    for arm in order[1:]:
        for metric in METRICS:
            frame = pd.concat(
                {"arm": pooled[arm][metric], "reference": pooled[reference][metric]}, axis=1
            ).dropna()
            inference = paired_mean_inference(
                frame["arm"] - frame["reference"],
                arm=arm,
                control=reference,
                label_horizon=label_t,
                ci_level=ci_level,
            )
            paired.append(
                {
                    "metric": metric,
                    "windows": sorted(by_arm[arm], key=list(WINDOWS).index),
                    "n_common_dates": len(frame),
                    **dataclasses.asdict(inference),
                }
            )
    return {
        "schema": POOLED_SCHEMA,
        "label_t": label_t,
        "reference_arm": reference,
        "results": [str(f) for f in folders],
        "arms": arms,
        "paired_direction": "arm_minus_reference",
        "paired_vs_reference": paired,
        "note": "Exploratory, not a verdict (#372). Pooled t is Hansen-Hodrick on label_t - 2 "
        "lags (overlap_hac_lags, as the paired rows) on the arm's own dates; t_newey_west is Bartlett on label_t - 2 lags, as the protocol's "
        "five-window table reports. Paired rows are the arm minus the reference per date, on the dates both "
        "scored. A baseline folder's paired rows inside score_baselines.py score the model on "
        "the baseline's cells only; these use each arm's whole daily series.",
    }


def _fmt(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown(report: dict[str, Any]) -> str:
    """The report as a Markdown table."""
    windows = list(WINDOWS)
    head = "| Arm | " + " | ".join(windows) + " | Pooled IC | HH t | NW t | Pooled rank IC | HH t |"
    lines = [head, "|" + "---|" * (len(windows) + 6)]
    for arm in report["arms"]:
        cells = [
            f"{arm['per_window'][w]['ic']:.4f}" if w in arm["per_window"] else "" for w in windows
        ]
        ic, rank = arm["ic"], arm["rank_ic"]
        lines.append(
            f"| {arm['arm']} | " + " | ".join(cells) + f" | {_fmt(ic['mean'], 4)} | "
            f"{_fmt(ic['t'], 2)} | {_fmt(ic['t_newey_west'], 2)} | {_fmt(rank['mean'], 4)} | "
            f"{_fmt(rank['t'], 2)} |"
        )
    lines += [
        "",
        f"| Arm minus {report['reference_arm']} | Metric | Dates | Mean | HAC t | p |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["paired_vs_reference"]:
        lines.append(
            f"| {row['arm']} | {row['metric']} | {row['n_common_dates']} | "
            f"{_fmt(row['mean_delta'], 4)} | {_fmt(row['hac_t'], 2)} | {_fmt(row['hac_p'], 3)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        type=Path,
        help="a candidate or baseline result folder; repeat per folder",
    )
    parser.add_argument("--out", required=True, type=Path, help="pooled report JSON")
    parser.add_argument("--markdown", type=Path, help="also write the table as Markdown")
    parser.add_argument(
        "--reference", default=REFERENCE_ARM, help="the arm every other is paired against"
    )
    args = parser.parse_args(argv)
    for path in (args.out, args.markdown):
        if path is not None and path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    report = pool(args.result, reference=args.reference)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    table = markdown(report)
    if args.markdown is not None:
        args.markdown.write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
