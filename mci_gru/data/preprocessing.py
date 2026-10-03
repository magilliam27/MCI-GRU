"""
Data preprocessing utilities for MCI-GRU.

Contains pure data-transformation functions extracted from run_experiment.py:
- generate_time_series_features: sliding-window tensor construction
- generate_graph_features: per-day graph node features
- resolve_label_endpoints / resolve_labels / compute_labels: fixed-session
  forward-return labels (the one endpoint resolver, #225)
- apply_rank_labels: cross-sectional rank percentile conversion
- purge_training_sessions_for_embargo / assert_training_labels_respect_embargo:
  session-level train/val embargo on the same endpoint resolver as the labels
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from tqdm import tqdm


def fit_rank_gaussian_reference(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    """Sorted train values per feature for rank-Gaussian inverse-CDF mapping."""
    ref: dict[str, np.ndarray] = {}
    for col in feature_cols:
        if col not in train_df.columns:
            continue
        arr = train_df[col].dropna().to_numpy(dtype=np.float64)
        if arr.size == 0:
            continue
        ref[col] = np.sort(arr)
    return ref


def apply_rank_gaussian(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Map each feature through empirical rank → Gaussian quantiles (train ``reference``)."""
    out = df.copy()
    for col in feature_cols:
        if col not in reference or col not in out.columns:
            continue
        sv = reference[col]
        n = len(sv)
        if n == 0:
            continue
        vals = out[col].to_numpy(dtype=np.float64)
        ranks = np.searchsorted(sv, vals, side="right").astype(np.float64)
        u = np.clip((ranks + 0.5) / (n + 1.0), 1e-6, 1.0 - 1e-6)
        out[col] = stats.norm.ppf(u)
    return out


def generate_time_series_features(
    df: pd.DataFrame,
    kdcode_list: list[str],
    feature_cols: list[str],
    his_t: int,
    use_polars: bool = False,
) -> np.ndarray:
    """Build sliding-window feature tensors for all stocks.

    Returns array of shape (num_usable_days, num_stocks, his_t, num_features).
    """
    all_dates = sorted(df["dt"].unique())
    num_stocks = len(kdcode_list)
    num_features = len(feature_cols)
    num_usable_days = len(all_dates) - his_t

    print(f"  Allocating feature array: ({num_usable_days}, {num_stocks}, {his_t}, {num_features})")

    df_subset = df[df["kdcode"].isin(kdcode_list)][["kdcode", "dt"] + feature_cols].copy()
    # Last row wins for duplicate (dt, kdcode), matching legacy iterrows overwrite semantics.
    df_subset = df_subset.drop_duplicates(subset=["dt", "kdcode"], keep="last")

    pivot_data = np.zeros((len(all_dates), num_stocks, num_features), dtype=np.float32)
    pl = None
    if use_polars:
        try:
            import polars as pl_mod  # noqa: PLC0415

            pl = pl_mod
        except ImportError:
            pl = None

    for fi, col in enumerate(
        tqdm(feature_cols, desc="  Building pivot (per-feature)", leave=False)
    ):
        if pl is not None:
            pdf = pl.from_pandas(df_subset[["dt", "kdcode", col]].copy())
            wide = pdf.pivot(on="kdcode", index="dt", values=col, aggregate_function="last")
            wide = wide.fill_null(0.0)
            wide_pd = wide.to_pandas()
            if "dt" in wide_pd.columns:
                wide_pd = wide_pd.set_index("dt")
            wide_pd = wide_pd.reindex(index=all_dates)
            wide_pd = wide_pd.reindex(columns=kdcode_list, fill_value=0.0)
            pivot_data[:, :, fi] = wide_pd.to_numpy(dtype=np.float32, copy=False)
        else:
            wide = df_subset.pivot_table(
                index="dt",
                columns="kdcode",
                values=col,
                aggfunc="last",
                fill_value=0.0,
            )
            wide = wide.reindex(index=all_dates, columns=kdcode_list, fill_value=0.0)
            pivot_data[:, :, fi] = wide.to_numpy(dtype=np.float32, copy=False)

    # (T, S, F) -> sliding windows along time -> (T - his_t + 1, S, F, his_t) -> keep num_usable_days
    windows = sliding_window_view(pivot_data, his_t, axis=0)
    windows = windows[:num_usable_days, ...]
    # (num_usable_days, S, F, his_t) -> (num_usable_days, S, his_t, F)
    stock_features = np.transpose(windows, (0, 1, 3, 2)).astype(np.float32, copy=False)

    return stock_features


def generate_graph_features(
    df: pd.DataFrame,
    kdcode_list: list[str],
    feature_cols: list[str],
    dates: list[str],
) -> np.ndarray:
    """Build per-day graph node feature tensors.

    Returns array of shape (num_dates, num_stocks, num_features).
    """
    num_dates = len(dates)
    num_stocks = len(kdcode_list)
    num_features = len(feature_cols)

    x_graph = np.zeros((num_dates, num_stocks, num_features), dtype=np.float32)
    stock_to_idx = {stock: idx for idx, stock in enumerate(kdcode_list)}

    df_subset = df[df["dt"].isin(dates) & df["kdcode"].isin(kdcode_list)]

    for date_idx, date in enumerate(dates):
        df_day = df_subset[df_subset["dt"] == date]
        for _, row in df_day.iterrows():
            stock_idx = stock_to_idx.get(row["kdcode"])
            if stock_idx is not None:
                x_graph[date_idx, stock_idx, :] = row[feature_cols].values.astype(np.float32)

    return x_graph


def apply_rank_labels(labels: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Convert raw return labels to cross-sectional rank percentiles per day.

    Each day's returns are ranked across stocks and divided by the stock count
    to yield percentiles in (0, 1].  Only same-day information is used, so this
    does **not** introduce look-ahead bias.
    """
    from scipy.stats import rankdata

    ranked = np.full_like(labels, np.nan, dtype=np.float32)
    mask = np.isfinite(labels)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool)
    for i in range(labels.shape[0]):
        row_mask = mask[i]
        if not row_mask.any():
            continue
        ranked[i, row_mask] = rankdata(labels[i, row_mask]) / row_mask.sum()
    return ranked.astype(np.float32)


def label_session_axis(df: pd.DataFrame, kdcode_list: list[str]) -> list[str]:
    """The panel's own trading sessions: every date any selected stock has a row on.

    Labels, ``label_available_mask`` and the embargo check all count sessions on this
    axis, so they agree on which closes a label consumes.
    """
    rows = df.loc[df["kdcode"].isin(kdcode_list), "dt"]
    return sorted(rows.astype(str).unique())


def resolve_label_endpoints(
    sessions: list[str],
    dates: list[str],
    label_t: int,
) -> tuple[list[str | None], list[str | None]]:
    """Entry and exit sessions for each signal date (#225 ruling 4).

    Entry is the close of session ``D+1`` and exit the close of session ``D+label_t``
    on ``sessions``, the panel's own trading dates. With ``label_t=5`` the return spans
    four close-to-close session intervals. An endpoint past the end of the axis, or a
    signal date that is not on the axis, resolves to ``None``.
    """
    position = {session: i for i, session in enumerate(sessions)}
    entries: list[str | None] = []
    exits: list[str | None] = []
    for date in dates:
        i = position.get(str(date))
        if i is None:
            entries.append(None)
            exits.append(None)
            continue
        entries.append(sessions[i + 1] if i + 1 < len(sessions) else None)
        exits.append(sessions[i + label_t] if i + label_t < len(sessions) else None)
    return entries, exits


@dataclass(frozen=True)
class LabelResolution:
    """Fixed-session label endpoints and the closes observed at them."""

    dates: list[str]
    kdcodes: list[str]
    entry_dates: list[str | None]
    exit_dates: list[str | None]
    entry_close: np.ndarray  # (dates, stocks); NaN when the stock has no close there
    exit_close: np.ndarray

    @property
    def returns(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.exit_close / self.entry_close - 1.0

    @property
    def observable(self) -> np.ndarray:
        return np.isfinite(self.returns)


def resolve_labels(
    df: pd.DataFrame,
    kdcode_list: list[str],
    dates: list[str],
    label_t: int,
) -> LabelResolution:
    """Resolve fixed-session endpoints and read each stock's close at them.

    A stock with no close at its entry or exit session has an unobservable label.
    There is no fill and no next-available substitute: a gap never moves an endpoint.
    """
    subset = df.loc[df["kdcode"].isin(kdcode_list), ["kdcode", "dt", "close"]].copy()
    subset["dt"] = subset["dt"].astype(str)
    sessions = label_session_axis(subset, kdcode_list)
    closes = subset.pivot_table(index="dt", columns="kdcode", values="close", aggfunc="mean")
    closes = closes.reindex(index=sessions, columns=kdcode_list)
    close_matrix = closes.to_numpy(dtype=np.float64)

    entries, exits = resolve_label_endpoints(sessions, dates, label_t)
    position = {session: i for i, session in enumerate(sessions)}

    def _closes_at(endpoints: list[str | None]) -> np.ndarray:
        out = np.full((len(endpoints), len(kdcode_list)), np.nan, dtype=np.float64)
        for row, endpoint in enumerate(endpoints):
            if endpoint is not None:
                out[row] = close_matrix[position[endpoint]]
        return out

    return LabelResolution(
        dates=[str(date) for date in dates],
        kdcodes=list(kdcode_list),
        entry_dates=entries,
        exit_dates=exits,
        entry_close=_closes_at(entries),
        exit_close=_closes_at(exits),
    )


def compute_labels(
    df: pd.DataFrame,
    kdcode_list: list[str],
    dates: list[str],
    label_t: int,
    fill_missing: bool = True,
) -> np.ndarray:
    """Compute forward-return labels for the given dates.

    For each (stock, date) pair the label is:
        close[session D + label_t] / close[session D + 1] - 1

    where sessions are the panel's own trading dates (``resolve_label_endpoints``).
    A stock missing either close has no observable label.

    When ``fill_missing`` is true, NaN labels (e.g. near the end of the dataset)
    are filled with the cross-sectional mean for that day, then with zero as a
    final fallback. Masked PIT mode passes ``fill_missing=False`` so unobservable
    labels stay excluded from loss/evaluation.
    """
    labels = resolve_labels(df, kdcode_list, dates, label_t).returns
    labels = np.where(np.isfinite(labels), labels, np.nan)

    if fill_missing:
        pivot = pd.DataFrame(labels, index=dates, columns=kdcode_list)
        for date in dates:
            row_mean = pivot.loc[date].mean()
            pivot.loc[date] = pivot.loc[date].fillna(row_mean)
        labels = pivot.fillna(0).to_numpy()

    return labels.astype(np.float32)


def purge_training_sessions_for_embargo(
    train_dates: list[str],
    his_t: int,
    label_t: int,
) -> list[str]:
    """Drop the final ``label_t`` trading sessions from the training session axis.

    ``compute_labels`` builds the label for signal date ``D`` as
    ``close[D + label_t] / close[D + 1] - 1`` where the offsets count sessions on the
    panel's own trading dates.  The label for the last training session
    therefore matures ``label_t`` sessions later, which lands inside the validation
    window whenever the configured gap spans fewer than ``label_t`` sessions -- a
    calendar-day gap check cannot see this because weekends and holidays are absent
    from the panel.

    Purging the last ``label_t`` sessions of training *signal* leaves the configured
    split dates untouched and guarantees the final training label matures no later
    than the last training session, for any gap width.
    """
    if label_t <= 0:
        return list(train_dates)

    kept = list(train_dates[: max(0, len(train_dates) - label_t)])
    if len(kept) <= his_t:
        raise ValueError(
            f"Embargo purge of {label_t} session(s) leaves no training labels: "
            f"{len(train_dates)} training sessions, his_t={his_t}, label_t={label_t}. "
            "Widen data.train_start..train_end or reduce model.his_t / model.label_t."
        )
    return kept


def assert_training_labels_respect_embargo(
    df_for_labels: pd.DataFrame,
    kdcode_list: list[str],
    train_label_dates: list[str],
    val_start: str,
    label_t: int,
) -> dict[str, object]:
    """Fail closed if any training label would mature on or after ``val_start``.

    This is the authoritative, data-backed counterpart to the cheap calendar-day check
    in ``ExperimentConfig._validate_embargo``: it runs on the real session axis, so it
    measures sessions rather than calendar days.  It is deliberately unconditional and
    takes no ``skip_embargo_check`` flag -- that flag governs the calendar check only.

    Outcome dates come from ``resolve_label_endpoints`` on ``label_session_axis``, the
    same resolver ``compute_labels`` and ``label_available_mask`` use (#225 ruling 4),
    so the check measures exactly the exit close each label consumes.  A stock with a
    gap does not move its exit: an endpoint it lacks makes its label unobservable,
    never later.

    Returns a summary dict for logging.  Raises ``ValueError`` on any violation, on a
    training label date missing from the panel, or when the panel is too short to prove
    compliance.
    """
    summary: dict[str, object] = {
        "label_t": label_t,
        "val_start": val_start,
        "train_label_dates": len(train_label_dates),
    }
    if label_t <= 0 or not train_label_dates:
        return summary

    sessions = label_session_axis(df_for_labels, kdcode_list)
    if not sessions:
        raise ValueError(
            "Cannot verify the train/val embargo: no label panel rows for the selected "
            f"universe of {len(kdcode_list)} stock(s)."
        )

    label_dates = sorted({str(date) for date in train_label_dates})
    session_set = set(sessions)
    missing = [date for date in label_dates if date not in session_set]
    if missing:
        raise ValueError(
            f"Cannot verify the train/val embargo: {len(missing)} training label date(s) "
            f"are absent from the label panel (first: {missing[0]})."
        )

    _entries, exits = resolve_label_endpoints(sessions, label_dates, label_t)
    if exits[-1] is None:
        raise ValueError(
            "Cannot verify the train/val embargo: the label panel ends before the last "
            f"training label matures ({label_dates[-1]} + {label_t} sessions, panel ends "
            f"{sessions[-1]}). Refusing to treat unverifiable labels as compliant."
        )
    violations = [
        (date, exit_date)
        for date, exit_date in zip(label_dates, exits, strict=True)
        if exit_date is not None and exit_date >= val_start
    ]

    summary.update(
        {
            "last_train_label_date": label_dates[-1],
            # label_dates is sorted ascending, so the last exit is the latest.
            "last_outcome_date": exits[-1],
        }
    )

    if violations:
        first_date, first_exit = violations[0]
        raise ValueError(
            "Train/val embargo violated on the session axis: "
            f"{len(violations)} training label date(s) mature at or after {val_start} "
            f"(first: {first_date} -> {first_exit}). label_t={label_t} is a session "
            "count, not a calendar-day count; the training signal must be purged so "
            "labels mature before val_start."
        )

    return summary
