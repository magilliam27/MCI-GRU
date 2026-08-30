"""Pure correlation-matrix and edge-selection math for graph construction."""

import numpy as np
import pandas as pd
import torch


def _daily_returns_pivot(
    df: pd.DataFrame, kdcode_list: list[str], end_date: str, corr_lookback_days: int
) -> pd.DataFrame:
    """Aligned daily return matrix (dates × stocks) up to *end_date* (exclusive)."""
    df = df.copy()

    if "prev_close" in df.columns:
        df["daily_return"] = df["close"] / df["prev_close"] - 1
    else:
        df = df.sort_values(["kdcode", "dt"])
        df["daily_return"] = df.groupby("kdcode")["close"].pct_change()

    df = df[df["dt"] < end_date]
    all_dates = sorted(df["dt"].unique())
    if len(all_dates) > corr_lookback_days:
        start_date = all_dates[-corr_lookback_days]
        df = df[df["dt"] >= start_date]

    df = df[df["kdcode"].isin(kdcode_list)]
    pivot = df.pivot_table(index="dt", columns="kdcode", values="daily_return")
    pivot = pivot.reindex(columns=kdcode_list)
    pivot = pivot.fillna(0)
    return pivot


def compute_correlation_matrix(
    df: pd.DataFrame, kdcode_list: list[str], end_date: str, corr_lookback_days: int
) -> pd.DataFrame:
    """Per paper Section 3.3.2: use past year of returns for correlation."""
    pivot = _daily_returns_pivot(df, kdcode_list, end_date, corr_lookback_days)
    return pivot.corr()


def build_edges(
    corr_matrix: pd.DataFrame,
    kdcode_list: list[str],
    show_progress: bool,
    returns_pivot: pd.DataFrame | None,
    judge_value: float,
    top_k: int,
    top_k_metric: str,
    use_multi_feature_edges: bool,
    use_lead_lag_features: bool,
    lead_lag_days: list[int],
    admissible_mask: np.ndarray | None = None,
    exclude_pairs: list[tuple[str, str]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build edge tensors from a correlation matrix using vectorised numpy ops.

    Two selection paths and two output shapes are supported:

    Selection (controlled by ``top_k``):
    - ``top_k == 0`` (legacy): keep edges with ``corr > judge_value``.
    - ``top_k > 0``: per-row K-nearest selection by ``top_k_metric``
      ("corr" => K most-positive; "abs_corr" => K largest by |corr|, which
      recovers strong negative correlations - the "1b-lite" path).

    Output (controlled by ``use_multi_feature_edges``):
    - ``False`` (legacy): returns ``(edge_index (2,E), edge_weight (E,))`` with
      the signed correlation as the scalar weight.
    - ``True``: returns ``(edge_index (2,E), edge_attr (E,4+))`` with columns
      ``[corr, |corr|, corr^2, rank_pct]`` plus optional lead–lag columns when
      ``use_lead_lag_features`` and *returns_pivot* are set.

    *admissible_mask* is an optional boolean array over *kdcode_list* marking
    the names in the investable universe at this graph's date (issue #123).
    When supplied, inadmissible entries are excluded **before** selection
    rather than after: both their rows and their columns are set to NaN, which
    the threshold path skips via its ``~isnan`` guard and the top-K path maps
    to ``-inf`` so they are never counted as valid candidates.

    Filtering before selection is the point. ``combined_collate_fn`` already
    drops edges touching inactive nodes per sample date, so filtering *output*
    changes nothing. On the top-K path, however, an inadmissible name that wins
    a slot spends that slot and is then discarded, leaving the node with fewer
    neighbours than configured and a ``rank_pct`` ranked against a contaminated
    candidate set. Excluding candidates up front is the only place that is
    repairable.

    *exclude_pairs* names kdcode pairs to exclude from the adjacency (issue 164
    hygiene rule: the GOOG.OQ/GOOGL.OQ same-company twin). It rides the same
    candidate-level mechanism for the same reason: the pair's entries are set
    to NaN before selection, so the excluded name can never spend a top-K slot
    and the node's budget is refilled by its next clean candidate. Names absent
    from *kdcode_list* are ignored here; the caller owns any warning.
    """
    corr = corr_matrix.values

    if admissible_mask is not None:
        admissible_mask = np.asarray(admissible_mask, dtype=bool)
        if admissible_mask.shape != (corr.shape[0],):
            raise ValueError(
                f"admissible_mask has shape {admissible_mask.shape}, expected "
                f"({corr.shape[0]},) to match the correlation matrix axis"
            )
        corr = corr.astype(float, copy=True)
        corr[~admissible_mask, :] = np.nan
        corr[:, ~admissible_mask] = np.nan

    if exclude_pairs:
        index_by_code = {code: idx for idx, code in enumerate(kdcode_list)}
        positions = [
            (index_by_code[a], index_by_code[b])
            for a, b in exclude_pairs
            if a in index_by_code and b in index_by_code
        ]
        if positions:
            if admissible_mask is None:
                # Not yet copied above; never NaN the caller's matrix in place.
                corr = corr.astype(float, copy=True)
            for i, j in positions:
                corr[i, j] = np.nan
                corr[j, i] = np.nan

    if top_k > 0:
        rows, cols, kept_corr, rank_pct = _select_edges_topk(corr, top_k, top_k_metric)
    else:
        rows, cols, kept_corr = _select_edges_threshold(corr, judge_value)
        rank_pct = np.zeros_like(kept_corr)

    n_extra = 2 if (use_lead_lag_features and use_multi_feature_edges) else 0
    base_f = 4

    if len(rows) == 0:
        empty_attr = (
            torch.zeros((0, base_f + n_extra), dtype=torch.float)
            if use_multi_feature_edges
            else torch.zeros(0, dtype=torch.float)
        )
        return torch.zeros((2, 0), dtype=torch.long), empty_attr

    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)

    if use_multi_feature_edges:
        edge_attr = np.stack(
            [kept_corr, np.abs(kept_corr), kept_corr * kept_corr, rank_pct],
            axis=1,
        ).astype(np.float32)
        if use_lead_lag_features and returns_pivot is not None:
            lag_n, lag_c = _lead_lag_columns(returns_pivot, rows, cols, lead_lag_days)
            edge_attr = np.concatenate([edge_attr, lag_n[:, None], lag_c[:, None]], axis=1)
        return edge_index, torch.from_numpy(edge_attr)

    return edge_index, torch.tensor(kept_corr, dtype=torch.float)


def _lead_lag_columns(
    returns_pivot: pd.DataFrame,
    rows: np.ndarray,
    cols: np.ndarray,
    lead_lag_days: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Best lag (among 0 and ``lead_lag_days``) by strongest |corr|; signed corr stored."""
    vals = returns_pivot.to_numpy(dtype=np.float64)
    if vals.size == 0:
        z = np.zeros(len(rows), dtype=np.float32)
        return z, z
    max_lag = max(lead_lag_days) if lead_lag_days else 1
    lag_norms = np.zeros(len(rows), dtype=np.float32)
    lag_corrs = np.zeros(len(rows), dtype=np.float32)
    for e, (ri, ci) in enumerate(zip(rows, cols, strict=False)):
        a = vals[:, int(ri)]
        b = vals[:, int(ci)]
        L = min(len(a), len(b))
        if L < 5:
            continue
        a = a[:L]
        b = b[:L]
        best_abs = -1.0
        best_lag = 0
        best_c = 0.0
        for lag in [0, *lead_lag_days]:
            if lag == 0:
                aa, bb = a, b
            else:
                if lag >= L:
                    continue
                aa = a[: L - lag]
                bb = b[lag:L]
            m = len(aa)
            if m < 5 or np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
                continue
            cc = float(np.corrcoef(aa, bb)[0, 1])
            if abs(cc) > best_abs:
                best_abs = abs(cc)
                best_lag = lag
                best_c = cc
        lag_norms[e] = float(best_lag) / float(max_lag) if max_lag > 0 else 0.0
        lag_corrs[e] = float(best_c)
    return lag_norms, lag_corrs


def _select_edges_threshold(
    corr: np.ndarray, judge_value: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy threshold-based selection: keep ``corr > judge_value`` off-diagonal."""
    mask = (~np.isnan(corr)) & (corr > judge_value)
    np.fill_diagonal(mask, False)
    rows, cols = np.where(mask)
    kept_corr = corr[rows, cols].astype(np.float64)
    return rows, cols, kept_corr


def _select_edges_topk(
    corr: np.ndarray, top_k: int, top_k_metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row top-K neighbour selection by ``top_k_metric``.

    Diagonal entries and NaNs are masked to ``-inf`` so they cannot win a
    slot.  When a row has fewer than ``top_k`` valid candidates (rare; e.g.
    a tiny universe), only the valid ones are kept and ``rank_pct`` is
    scaled against the actual kept count.

    Returns
    -------
    rows, cols : np.ndarray
        Index pairs of kept directed edges.
    kept_corr : np.ndarray
        Signed correlation ``corr[row, col]`` for each kept edge.  This is
        the *signed* value even when ``top_k_metric == "abs_corr"`` so that
        column 0 of the multi-feature edge tensor preserves sign.
    rank_pct : np.ndarray
        Within-row percentile rank by the selection metric, in ``(0, 1]``,
        with 1.0 = strongest neighbour in that row.
    """
    n = corr.shape[0]
    score = np.where(np.isnan(corr), -np.inf, corr)
    if top_k_metric == "abs_corr":
        score = np.where(np.isfinite(score), np.abs(score), -np.inf)
    np.fill_diagonal(score, -np.inf)

    valid_mask = np.isfinite(score)
    valid_per_row = valid_mask.sum(axis=1)
    k_per_row = np.minimum(valid_per_row, top_k)

    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []
    kept_corr_out: list[np.ndarray] = []
    rank_pct_out: list[np.ndarray] = []

    for i in range(n):
        k_i = int(k_per_row[i])
        if k_i == 0:
            continue
        row_score = score[i]
        sorted_idx = np.argpartition(-row_score, kth=k_i - 1)[:k_i]
        sorted_idx = sorted_idx[np.argsort(-row_score[sorted_idx])]

        positions = np.arange(1, k_i + 1, dtype=np.float64)
        rank_pct_row = (k_i - positions + 1) / k_i

        rows_out.append(np.full(k_i, i, dtype=np.int64))
        cols_out.append(sorted_idx.astype(np.int64))
        kept_corr_out.append(corr[i, sorted_idx].astype(np.float64))
        rank_pct_out.append(rank_pct_row)

    if not rows_out:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty_i, empty_i, empty_f, empty_f

    rows = np.concatenate(rows_out)
    cols = np.concatenate(cols_out)
    kept_corr = np.concatenate(kept_corr_out)
    rank_pct = np.concatenate(rank_pct_out)
    return rows, cols, kept_corr, rank_pct
