"""Walk-forward experiment windows: rolling or expanding train, sliding val/test.

Each window is a full :class:`~mci_gru.config.ExperimentConfig` clone with
``data.*`` dates rewritten.  Windows are rejected if embargo rules fail
(:meth:`~mci_gru.config.ExperimentConfig._validate_embargo`).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from dateutil.relativedelta import relativedelta

from mci_gru.config import ExperimentConfig, WalkforwardConfig


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
            cfg = _one_window_from_train_end(
                base, train_start, train_end, lt, wf, global_end
            )
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
            cfg = _one_window_from_train_end(
                base, train_start, train_end, lt, wf, global_end
            )
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
            tracking=base.tracking,
            experiment_name=base.experiment_name,
            output_dir=base.output_dir,
            seed=base.seed,
        )
    except ValueError:
        return None
    return cfg


def merge_walkforward_summary(summaries: list[dict]) -> dict:
    """Aggregate per-window training_summary dicts."""
    if not summaries:
        return {}
    losses = [s["mean_best_val_loss"] for s in summaries if s.get("mean_best_val_loss") is not None]
    ics = [s["mean_best_val_ic"] for s in summaries if s.get("mean_best_val_ic") is not None]
    merged = {
        "n_windows": len(summaries),
        "mean_best_val_loss_across_windows": float(sum(losses) / len(losses)) if losses else None,
        "mean_best_val_ic_across_windows": float(sum(ics) / len(ics)) if ics else None,
        "windows": summaries,
    }
    eval_keys: set[str] = set()
    for summary in summaries:
        eval_keys.update((summary.get("evaluation") or {}).keys())
    eval_summary = {}
    for key in sorted(eval_keys):
        vals = [
            summary.get("evaluation", {}).get(key)
            for summary in summaries
            if isinstance(summary.get("evaluation", {}).get(key), (int, float))
        ]
        if vals:
            eval_summary[f"mean_{key}_across_windows"] = float(sum(vals) / len(vals))
    if eval_summary:
        merged["evaluation"] = eval_summary
    return merged
