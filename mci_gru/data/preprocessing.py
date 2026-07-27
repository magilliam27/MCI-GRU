"""
Data preprocessing utilities for MCI-GRU.

Contains pure data-transformation functions extracted from run_experiment.py:
- generate_time_series_features: sliding-window tensor construction
- generate_graph_features: per-day graph node features
- compute_labels: forward-return label computation
- apply_rank_labels: cross-sectional rank percentile conversion
- purge_training_sessions_for_embargo / assert_training_labels_respect_embargo:
  session-level train/val embargo (labels are row shifts, not calendar offsets)
"""

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


def compute_labels(
    df: pd.DataFrame,
    kdcode_list: list[str],
    dates: list[str],
    label_t: int,
    fill_missing: bool = True,
) -> np.ndarray:
    """Compute forward-return labels for the given dates.

    For each (stock, date) pair the label is:
        close[date + label_t] / close[date + 1] - 1

    When ``fill_missing`` is true, NaN labels (e.g. near the end of the dataset)
    are filled with the cross-sectional mean for that day, then with zero as a
    final fallback. Masked PIT mode passes ``fill_missing=False`` so unobservable
    labels stay excluded from loss/evaluation.
    """
    df_subset = df[df["kdcode"].isin(kdcode_list)].copy()
    df_subset = df_subset.sort_values(["kdcode", "dt"])

    df_subset["future_close"] = df_subset.groupby("kdcode")["close"].shift(-label_t)
    df_subset["next_close"] = df_subset.groupby("kdcode")["close"].shift(-1)
    df_subset["forward_return"] = df_subset["future_close"] / df_subset["next_close"] - 1

    df_subset = df_subset[df_subset["dt"].isin(dates)]
    pivot = df_subset.pivot_table(index="dt", columns="kdcode", values="forward_return")
    pivot = pivot.reindex(index=dates, columns=kdcode_list)

    if fill_missing:
        for date in dates:
            if date in pivot.index:
                row_mean = pivot.loc[date].mean()
                pivot.loc[date] = pivot.loc[date].fillna(row_mean)
        pivot = pivot.fillna(0)

    return pivot.values.astype(np.float32)


def purge_training_sessions_for_embargo(
    train_dates: list[str],
    his_t: int,
    label_t: int,
) -> list[str]:
    """Drop the final ``label_t`` trading sessions from the training session axis.

    ``compute_labels`` builds the label for signal date ``D`` as
    ``close[D + label_t] / close[D + 1] - 1`` where the offsets are per-stock **row**
    shifts over a session-indexed panel.  The label for the last training session
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

    Two bases are checked, both vectorised over the panel:

    * **per-stock outcome dates** -- ``groupby('kdcode')['dt'].shift(-label_t)``, exactly
      the rows ``compute_labels`` consumes, including stocks whose own panel has gaps;
    * **union-session outcome dates** -- the label-date position on the panel's union
      session axis plus ``label_t``.  A stock's rows are a subsequence of the union axis,
      so this is a lower bound on its true outcome date, and unlike the per-stock basis
      it stays defined when a stock's rows stop before its label matures.  Without it,
      truncated stocks would look compliant and the check's strictness would depend on
      how much future data happened to be loaded.

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

    panel = df_for_labels.loc[df_for_labels["kdcode"].isin(kdcode_list), ["kdcode", "dt"]]
    if panel.empty:
        raise ValueError(
            "Cannot verify the train/val embargo: no label panel rows for the selected "
            f"universe of {len(kdcode_list)} stock(s)."
        )
    panel = panel.sort_values(["kdcode", "dt"], kind="mergesort")

    sessions = np.asarray(sorted(panel["dt"].unique()))
    label_dates = np.asarray(sorted(set(train_label_dates)))
    positions = np.searchsorted(sessions, label_dates)
    clipped = np.minimum(positions, len(sessions) - 1)
    missing = label_dates[(positions >= len(sessions)) | (sessions[clipped] != label_dates)]
    if missing.size:
        raise ValueError(
            f"Cannot verify the train/val embargo: {missing.size} training label date(s) "
            f"are absent from the label panel (first: {missing[0]})."
        )

    outcome_positions = positions + label_t
    if int(outcome_positions.max()) >= len(sessions):
        raise ValueError(
            "Cannot verify the train/val embargo: the label panel ends before the last "
            f"training label matures ({label_dates[-1]} + {label_t} sessions, panel ends "
            f"{sessions[-1]}). Refusing to treat unverifiable labels as compliant."
        )
    union_outcomes = sessions[outcome_positions]
    union_violations = label_dates[union_outcomes >= val_start]

    panel = panel.copy()
    panel["outcome_dt"] = panel.groupby("kdcode", sort=False)["dt"].shift(-label_t)
    train_rows = panel[panel["dt"].isin(set(label_dates.tolist()))]
    matured = train_rows["outcome_dt"].notna()
    per_stock_violations = train_rows[matured & (train_rows["outcome_dt"] >= val_start)]

    summary.update(
        {
            "last_train_label_date": str(label_dates[-1]),
            # label_dates is sorted ascending, so the last outcome is the latest.
            "last_union_outcome_date": str(union_outcomes[-1]),
            "last_per_stock_outcome_date": (
                str(train_rows.loc[matured, "outcome_dt"].max()) if bool(matured.any()) else None
            ),
            "rows_without_matured_label": int((~matured).sum()),
        }
    )

    if union_violations.size or len(per_stock_violations):
        detail = []
        if union_violations.size:
            detail.append(
                f"{union_violations.size} training label date(s) mature at or after "
                f"{val_start} on the union session axis (first: {union_violations[0]} "
                f"-> {sessions[np.searchsorted(sessions, union_violations[0]) + label_t]})"
            )
        if len(per_stock_violations):
            worst = per_stock_violations.iloc[0]
            detail.append(
                f"{len(per_stock_violations)} (stock, date) label(s) mature at or after "
                f"{val_start} on their own session axis (first: {worst['kdcode']} "
                f"{worst['dt']} -> {worst['outcome_dt']})"
            )
        raise ValueError(
            "Train/val embargo violated on the session axis: "
            + "; ".join(detail)
            + f". label_t={label_t} is a session count, not a calendar-day count; the "
            "training signal must be purged so labels mature before val_start."
        )

    return summary
