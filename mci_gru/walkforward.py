"""Walk-forward experiment windows: rolling or expanding train, sliding val/test.

Each window is a full :class:`~mci_gru.config.ExperimentConfig` clone with
``data.*`` dates rewritten.  Windows are rejected if embargo rules fail
(:meth:`~mci_gru.config.ExperimentConfig._validate_embargo`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from mci_gru.config import ExperimentConfig, WalkforwardConfig
from mci_gru.training.summary import CHECKPOINT_METRICS, mean_of_available


def _parse(d: str) -> date:
    y, m, dd = d.split("-")
    return date(int(y), int(m), int(dd))


def _fmt(d: date) -> str:
    return d.isoformat()


def generate_walkforward_configs(base: ExperimentConfig) -> list[ExperimentConfig]:
    """Build one ExperimentConfig per walk-forward window (empty if disabled)."""
    wf = base.training.walkforward
    if not wf.enabled:
        return [base]

    lt = base.model.label_t
    d0 = _parse(base.data.train_start)
    global_end = _parse(base.data.test_end)

    windows: list[ExperimentConfig] = []
    if wf.expanding:
        train_start = d0
        train_end = train_start + relativedelta(years=wf.window_train_years)
        widx = 0
        while train_end < global_end and (wf.max_windows is None or widx < wf.max_windows):
            cfg = _one_window_from_train_end(base, train_start, train_end, lt, wf, global_end)
            if cfg is not None:
                windows.append(cfg)
                widx += 1
            train_end = train_end + relativedelta(months=wf.step_months)
    else:
        train_start = d0
        widx = 0
        while True:
            train_end = train_start + relativedelta(years=wf.window_train_years)
            if train_end >= global_end:
                break
            cfg = _one_window_from_train_end(base, train_start, train_end, lt, wf, global_end)
            if cfg is not None:
                windows.append(cfg)
                widx += 1
            if wf.max_windows is not None and widx >= wf.max_windows:
                break
            train_start = train_start + relativedelta(months=wf.step_months)

    if not windows:
        raise ValueError(
            "Walk-forward enabled but no embargo-valid window fits the configured date range."
        )
    return windows


def _one_window_from_train_end(
    base: ExperimentConfig,
    train_start: date,
    train_end: date,
    label_t: int,
    wf: WalkforwardConfig,
    global_end: date,
) -> ExperimentConfig | None:
    # Calendar-day gaps only. Six calendar days can hold three sessions, so the
    # pipeline purges, on the session axis, the training tail whose labels reach
    # val_start and the validation tail whose labels reach test_start (#342).
    gap = timedelta(days=label_t + 1)
    val_start = train_end + gap
    val_end = val_start + relativedelta(months=wf.window_val_months)
    test_start = val_end + gap
    if test_start > global_end:
        return None
    test_end = min(test_start + relativedelta(months=wf.test_span_months), global_end)
    if test_end <= test_start:
        return None

    new_data = replace(
        base.data,
        train_start=_fmt(train_start),
        train_end=_fmt(train_end),
        val_start=_fmt(val_start),
        val_end=_fmt(val_end),
        test_start=_fmt(test_start),
        test_end=_fmt(test_end),
    )
    try:
        cfg = ExperimentConfig(
            data=new_data,
            features=base.features,
            graph=base.graph,
            model=base.model,
            training=base.training,
            evaluation=base.evaluation,
            tracking=base.tracking,
            experiment_name=base.experiment_name,
            output_dir=base.output_dir,
            seed=base.seed,
        )
    except ValueError:
        return None
    return cfg


@dataclass(frozen=True)
class WindowTestSet:
    """One window's test predictions and what the in-run summary scores them on."""

    predictions: np.ndarray  # (dates, kdcodes)
    labels: np.ndarray  # (dates, kdcodes), the training labels
    dates: list[str]
    kdcodes: list[str]
    rankable_mask: np.ndarray | None
    closes: pd.DataFrame  # kdcode, dt, close


def window_test_set(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    panel: pd.DataFrame,
    dates: list[str],
    kdcodes: list[str],
    rankable_mask: np.ndarray | None,
    label_t: int,
) -> WindowTestSet:
    """Keep what pooling needs: the test arrays and the closes their holds read.

    The closes run from the first test date to ``label_t`` sessions past the last,
    on the window panel's own session axis, so every hold the window booked is kept.
    """
    dates = [str(day) for day in dates]
    closes = panel.loc[:, ["kdcode", "dt", "close"]].copy()
    closes["dt"] = closes["dt"].astype(str)
    closes["kdcode"] = closes["kdcode"].astype(str)
    closes = closes.loc[closes["kdcode"].isin(list(kdcodes))]
    if dates:
        sessions = sorted(closes["dt"].unique())
        last = sessions.index(dates[-1]) if dates[-1] in sessions else len(sessions) - 1
        end = sessions[min(last + label_t, len(sessions) - 1)]
        closes = closes.loc[(closes["dt"] >= dates[0]) & (closes["dt"] <= end)]
    return WindowTestSet(
        predictions=np.asarray(predictions, dtype=np.float64).copy(),
        labels=np.asarray(labels, dtype=np.float64).copy(),
        dates=dates,
        kdcodes=[str(code) for code in kdcodes],
        rankable_mask=None
        if rankable_mask is None
        else np.asarray(rankable_mask, dtype=bool).copy(),
        closes=closes.reset_index(drop=True),
    )


def pool_window_test_sets(windows: list[WindowTestSet]) -> WindowTestSet:
    """Concatenate the windows' test dates on the union of their kdcodes.

    A name a window did not carry is unscored and unrankable there. Windows whose
    test dates overlap cannot be pooled (two models would score one date) and raise.
    """
    if not windows:
        raise ValueError("no walk-forward windows to pool")
    dates = [day for window in windows for day in window.dates]
    if len(set(dates)) != len(dates):
        raise ValueError("walk-forward test windows overlap; their test dates cannot be pooled")
    with_mask = {window.rankable_mask is not None for window in windows}
    if len(with_mask) != 1:
        raise ValueError("either every walk-forward window has a rankable mask or none does")
    kdcodes = sorted({code for window in windows for code in window.kdcodes})
    column = {code: j for j, code in enumerate(kdcodes)}

    def widen(window: WindowTestSet, values: np.ndarray, fill: float | bool) -> np.ndarray:
        out = np.full((len(window.dates), len(kdcodes)), fill, dtype=np.asarray(values).dtype)
        out[:, [column[code] for code in window.kdcodes]] = values
        return out

    order = np.argsort(np.asarray(dates), kind="stable")

    def stack(arrays: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(arrays, axis=0)[order]

    closes = pd.concat([window.closes for window in windows], ignore_index=True)
    closes = closes.drop_duplicates(["kdcode", "dt"]).sort_values(["dt", "kdcode"])
    return WindowTestSet(
        predictions=stack([widen(w, w.predictions, np.nan) for w in windows]),
        labels=stack([widen(w, w.labels, np.nan) for w in windows]),
        dates=[dates[i] for i in order],
        kdcodes=kdcodes,
        rankable_mask=None
        if with_mask == {False}
        else stack([widen(w, w.rankable_mask, False) for w in windows]),
        closes=closes.reset_index(drop=True),
    )


def merge_walkforward_summary(
    summaries: list[dict],
    pooled_evaluation: dict | None = None,
    pooling_status: str | None = None,
) -> dict:
    """Aggregate per-window training_summary dicts.

    Each ``mean_best_val_*_across_windows`` is the unweighted mean of the windows
    whose own ``mean_best_val_*`` is available, so a window counts once whatever
    its member count. ``window_coverage`` records how many windows contributed out
    of the total; member coverage stays in each window's summary. Missing and
    non-finite window values, including historical ``-inf`` sentinels, are
    unavailable, and an aggregate with no available window is ``None``.

    Evaluation metrics are never averaged across windows: a mean of Sharpes, of CI
    bounds or of cumulative returns is not the walk-forward's Sharpe, interval or
    return, and a short last window would count as much as a full one (#342).
    ``evaluation`` is ``pooled_evaluation``, the in-run summary scored once on every
    window's test dates together (see :func:`pool_window_test_sets`), and
    ``evaluation_pooling`` says whether it was available. Each window keeps its own
    ``evaluation``.
    """
    if not summaries:
        return {}
    merged: dict = {"n_windows": len(summaries)}
    window_coverage: dict[str, dict[str, int]] = {}
    for metric in CHECKPOINT_METRICS:
        merged[f"mean_{metric}_across_windows"], window_coverage[metric] = mean_of_available(
            [summary.get(f"mean_{metric}") for summary in summaries]
        )
    merged["window_coverage"] = window_coverage
    merged["windows"] = summaries
    if pooled_evaluation is not None:
        merged["evaluation"] = pooled_evaluation
        merged["evaluation_pooling"] = "POOLED_TEST_DATES"
    elif any(summary.get("evaluation") for summary in summaries):
        merged["evaluation_pooling"] = pooling_status or "NOT_POOLED"
    return merged
