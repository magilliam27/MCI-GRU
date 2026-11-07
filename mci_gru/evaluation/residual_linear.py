"""A linear model on residual returns: market and peer moves removed (#384).

The model survey's first candidate. Reversal measured on returns after removing
common moves earns about twice what plain reversal does among large caps (Blitz,
Huij, Lansdorp and Verbeek 2013), and a linear model is hard to overfit on ~110
names. Everything here is built from the run's own prices and volume.

**Residual returns.** On the panel's session axis (``label_session_axis``, the axis
labels count on), a stock's daily return ``r`` loses its market move and then its peer
move:

* the market return ``m`` on a session is the equal-weight mean return of that
  session's point-in-time members;
* ``u = r - beta * m``, with ``beta`` the stock's regression slope on ``m`` over the
  last ``ESTIMATION_SESSIONS`` sessions, shrunk a third of the way to one;
* peers are clusters of the members' ``u`` correlations over the same trailing
  sessions (average-linkage clustering into ``N_CLUSTERS``): there are no
  point-in-time sectors, and current sectors would leak;
* ``e = u - mean(u of the stock's cluster peers that are members that session)``;
  a name with fewer than ``MIN_PEERS`` such peers (a singleton cluster) keeps ``u``.

Betas and clusters are re-estimated on the first session of each calendar month from
the sessions strictly before it, and held for that month. A name with too little
history keeps a beta of one and no peers.

**Signals** at the close of date ``D``, from sessions up to ``D`` only: residual
reversal over 5 and 21 sessions, residual 12-1 momentum scaled by its own residual
volatility, 21-session total volatility, 63-session residual volatility, the largest
daily return in 21 sessions, the change in trading volume (5 over 63 sessions) and the
beta. Each is ranked within the date among the date's scorable names, onto
``[-0.5, 0.5]``; a name with a signal missing gets the middle rank for it.

**Target and fit.** The run's own 5-session training label (cut at the split end as
the pipeline cuts it), residualised the same way with that date's betas and clusters,
then ranked within the date. ``fit_cross_sectional_ridge`` fits on training dates only
and picks its penalty on validation IC against the run's raw validation labels, so
the choice is made on the quantity the evaluation scores.

``residual_reversal_1w`` is the single signal on its own, unfitted: minus the
5-session sum of ``e``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from mci_gru.data.preprocessing import label_session_axis
from mci_gru.evaluation.baselines import fit_cross_sectional_ridge, linear_scores
from mci_gru.evaluation.candidates import CandidateContext, CandidateResult

#: Trailing sessions for betas and clusters, and the fewest a name needs for either.
ESTIMATION_SESSIONS = 252
MIN_ESTIMATION_SESSIONS = 126
#: Beta is shrunk this share of the way to one (Vasicek-style).
BETA_SHRINK = 1.0 / 3.0
N_CLUSTERS = 10
#: A peer mean needs this many peers with a return that session; fewer means no peer move.
MIN_PEERS = 2

#: name -> (description). Order is the column order of the model's inputs.
SIGNALS: dict[str, str] = {
    "rrev_5": "minus the 5-session sum of residual returns",
    "rrev_21": "minus the 21-session sum of residual returns",
    "rmom_12_1": "residual return summed over sessions D-251..D-21, over its residual volatility",
    "vol_21": "standard deviation of daily returns, 21 sessions",
    "ivol_63": "standard deviation of residual returns, 63 sessions",
    "max_21": "largest daily return, 21 sessions",
    "volume_change": "log of mean volume over 5 sessions over mean volume over 63 sessions",
    "beta": "market beta in force on the date",
}


@dataclass(frozen=True)
class ResidualState:
    """Every session's residual quantities on the panel's session axis."""

    sessions: list[str]
    returns: np.ndarray  # (sessions, stocks) daily return r
    residuals: np.ndarray  # (sessions, stocks) residual return e
    betas: np.ndarray  # (sessions, stocks) beta in force
    clusters: np.ndarray  # (sessions, stocks) cluster id in force; 0 means none
    members: np.ndarray  # (sessions, stocks) bool, point-in-time member
    volume: np.ndarray  # (sessions, stocks)
    refreshes: int
    peered_share: float  # share of member cells whose peer move was removed


def membership(
    pit_intervals: pd.DataFrame | None, kdcodes: list[str], sessions: list[str]
) -> np.ndarray:
    """``True`` where a stock was a point-in-time member on a session (all, without PIT)."""
    members = np.zeros((len(sessions), len(kdcodes)), dtype=bool)
    if pit_intervals is None:
        members[:] = True
        return members
    axis = np.asarray(sessions, dtype=object)
    column = {code: j for j, code in enumerate(kdcodes)}
    for row in pit_intervals.itertuples(index=False):
        j = column.get(str(row.kdcode))
        if j is None:
            continue
        members[:, j] |= (axis >= str(row.valid_from)) & (axis <= str(row.valid_to))
    return members


def _pivot(panel: pd.DataFrame, value: str, sessions: list[str], kdcodes: list[str]) -> np.ndarray:
    table = panel.pivot_table(index="dt", columns="kdcode", values=value, aggfunc="mean")
    return table.reindex(index=sessions, columns=kdcodes).to_numpy(dtype=np.float64)


def _row_mean(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    use = keep & np.isfinite(values)
    total = np.where(use, values, 0.0).sum(axis=1)
    count = use.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _betas(r: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Shrunk slope of each column of ``r`` on ``m``; one where history is short."""
    ok = np.isfinite(r) & np.isfinite(m)[:, None]
    n = ok.sum(axis=0)
    mm = np.where(ok, m[:, None], 0.0)
    rr = np.where(ok, r, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_m = mm.sum(axis=0) / n
        mean_r = rr.sum(axis=0) / n
        cov = (np.where(ok, (mm - mean_m) * (rr - mean_r), 0.0)).sum(axis=0)
        var = (np.where(ok, (mm - mean_m) ** 2, 0.0)).sum(axis=0)
        raw = cov / var
    beta = (1.0 - BETA_SHRINK) * raw + BETA_SHRINK
    usable = (n >= MIN_ESTIMATION_SESSIONS) & np.isfinite(beta)
    return np.where(usable, beta, 1.0)


def _clusters(u: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Cluster id (1..N_CLUSTERS) per column, 0 where a name is not clustered."""
    labels = np.zeros(u.shape[1], dtype=np.int64)
    columns = np.flatnonzero(eligible & (np.isfinite(u).sum(axis=0) >= MIN_ESTIMATION_SESSIONS))
    if columns.size <= N_CLUSTERS:
        return labels
    corr = pd.DataFrame(u[:, columns]).corr(min_periods=MIN_ESTIMATION_SESSIONS).to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    distance = np.clip(1.0 - corr, 0.0, 2.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    labels[columns] = fcluster(tree, t=N_CLUSTERS, criterion="maxclust")
    return labels


def _peer_mean(
    u: np.ndarray, clusters: np.ndarray, members: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each name's leave-one-out mean of its cluster peers' ``u`` on each session.

    Also returns where that mean exists: a name with fewer than ``MIN_PEERS`` peers
    holding a return that session gets a peer move of zero.
    """
    peer = np.zeros_like(u)
    has_peers = np.zeros(u.shape, dtype=bool)
    use = members & np.isfinite(u)
    values = np.where(use, u, 0.0)
    for cluster in np.unique(clusters[clusters > 0]):
        columns = np.flatnonzero(clusters == cluster)
        total = values[:, columns].sum(axis=1, keepdims=True)
        count = use[:, columns].sum(axis=1, keepdims=True)
        others_total = total - values[:, columns]
        others_count = count - use[:, columns]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = others_total / others_count
        enough = others_count >= MIN_PEERS
        peer[:, columns] = np.where(enough, mean, 0.0)
        has_peers[:, columns] = enough
    return peer, has_peers


def residual_state(
    panel: pd.DataFrame, kdcodes: list[str], pit_intervals: pd.DataFrame | None
) -> ResidualState:
    """Daily returns, betas, clusters and residual returns on the panel's session axis."""
    rows = panel.loc[:, ["kdcode", "dt", "close", "volume"]].copy()
    rows["dt"] = rows["dt"].astype(str)
    rows["kdcode"] = rows["kdcode"].astype(str)
    sessions = label_session_axis(rows, list(kdcodes))
    close = _pivot(rows, "close", sessions, list(kdcodes))
    volume = _pivot(rows, "volume", sessions, list(kdcodes))
    r = np.full_like(close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = close[1:] / close[:-1] - 1.0
    r[~np.isfinite(r)] = np.nan
    members = membership(pit_intervals, list(kdcodes), sessions)
    m = _row_mean(r, members)

    months = [s[:7] for s in sessions]
    starts = [0] + [i for i in range(1, len(sessions)) if months[i] != months[i - 1]]
    bounds = list(zip(starts, starts[1:] + [len(sessions)], strict=True))
    betas = np.ones_like(r)
    clusters = np.zeros(r.shape, dtype=np.int64)
    residuals = np.full_like(r, np.nan)
    peered = np.zeros(r.shape, dtype=bool)
    for start, stop in bounds:
        lo = max(0, start - ESTIMATION_SESSIONS)
        window_r, window_m = r[lo:start], m[lo:start]
        beta = _betas(window_r, window_m) if start > lo else np.ones(r.shape[1])
        if start > lo:
            u_window = window_r - beta * window_m[:, None]
            label = _clusters(u_window, members[start])
        else:
            label = np.zeros(r.shape[1], dtype=np.int64)
        betas[start:stop] = beta
        clusters[start:stop] = label
        u = r[start:stop] - beta * m[start:stop, None]
        peer, has_peers = _peer_mean(u, label, members[start:stop])
        residuals[start:stop] = u - peer
        peered[start:stop] = has_peers
    residuals[~np.isfinite(r)] = np.nan
    member_cells = int(members.sum())
    with_peers = int((members & peered).sum())
    return ResidualState(
        sessions=sessions,
        returns=r,
        residuals=residuals,
        betas=betas,
        clusters=clusters,
        members=members,
        volume=volume,
        refreshes=len(bounds),
        peered_share=with_peers / member_cells if member_cells else 0.0,
    )


def _rolling(values: np.ndarray, window: int, min_periods: int, how: str) -> np.ndarray:
    frame = pd.DataFrame(values).rolling(window, min_periods=min_periods)
    return np.array(getattr(frame, how)().to_numpy(dtype=np.float64), copy=True)


def signal_panel(state: ResidualState) -> dict[str, np.ndarray]:
    """Every signal per session and stock, each using sessions up to its own only."""
    e, r = state.residuals, state.returns
    with np.errstate(invalid="ignore", divide="ignore"):
        momentum_span = ESTIMATION_SESSIONS - 21
        lagged = np.full_like(e, np.nan)
        lagged[21:] = e[:-21]
        mom_sum = _rolling(lagged, momentum_span, 150, "sum")
        mom_sd = _rolling(lagged, momentum_span, 150, "std")
        log_volume_ratio = np.log(
            _rolling(state.volume, 5, 4, "mean") / _rolling(state.volume, 63, 40, "mean")
        )
        signals = {
            "rrev_5": -_rolling(e, 5, 4, "sum"),
            "rrev_21": -_rolling(e, 21, 15, "sum"),
            "rmom_12_1": mom_sum / mom_sd,
            "vol_21": _rolling(r, 21, 15, "std"),
            "ivol_63": _rolling(e, 63, 40, "std"),
            "max_21": _rolling(r, 21, 15, "max"),
            "volume_change": log_volume_ratio,
            "beta": state.betas.copy(),
        }
    for name, values in signals.items():
        values[~np.isfinite(values)] = np.nan
        signals[name] = values
    return signals


def _session_rows(state: ResidualState, dates: list[str]) -> np.ndarray:
    position = {session: i for i, session in enumerate(state.sessions)}
    missing = [d for d in dates if d not in position]
    if missing:
        raise ValueError(f"{len(missing)} dates are not on the panel's sessions: {missing[:3]}")
    return np.asarray([position[d] for d in dates], dtype=np.int64)


def centred_ranks(values: np.ndarray, scorable: np.ndarray) -> np.ndarray:
    """Each row ranked among its scorable finite cells onto ``[-0.5, 0.5]``.

    Scorable cells without a value get 0, the middle; unscorable cells are NaN.
    """
    out = np.full(values.shape, np.nan)
    for i in range(values.shape[0]):
        keep = scorable[i]
        out[i, keep] = 0.0
        finite = keep & np.isfinite(values[i])
        n = int(finite.sum())
        if n >= 2:
            ranks = pd.Series(values[i, finite]).rank(method="average").to_numpy()
            out[i, finite] = (ranks - 1.0) / (n - 1.0) - 0.5
    return out


def model_inputs(
    signals: dict[str, np.ndarray], rows: np.ndarray, scorable: np.ndarray
) -> np.ndarray:
    """``(dates, stocks, signals)`` ranked inputs on the given session rows."""
    columns = [centred_ranks(signals[name][rows], scorable) for name in SIGNALS]
    return np.stack(columns, axis=-1)


def residual_labels(state: ResidualState, labels: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Forward labels with the market and peer move removed, as daily returns are.

    The market move is the date's equal-weight mean label; each name's beta and
    cluster are those in force on the date.
    """
    y = np.asarray(labels, dtype=np.float64)
    keep = np.isfinite(y)
    market = _row_mean(y, keep)
    betas = state.betas[rows]
    u = y - betas * market[:, None]
    out = np.full_like(y, np.nan)
    for i, row in enumerate(rows):
        peer, _ = _peer_mean(u[i : i + 1], state.clusters[row], keep[i : i + 1])
        out[i] = u[i] - peer[0]
    out[~keep] = np.nan
    return out


def _split_scorable(context: CandidateContext, key: str, labels: np.ndarray) -> np.ndarray:
    mask = context.data.get(key)
    finite = np.isfinite(np.asarray(labels, dtype=np.float64))
    return finite if mask is None else finite & np.asarray(mask, dtype=bool)


def _test_scorable(context: CandidateContext, state: ResidualState, rows: np.ndarray) -> np.ndarray:
    if context.rankable_mask is not None:
        return np.asarray(context.rankable_mask, dtype=bool)
    previous = np.where(rows > 0, rows - 1, rows)
    return np.isfinite(state.returns[rows]) | np.isfinite(state.returns[previous])


def residual_linear(context: CandidateContext) -> CandidateResult:
    """Ridge on ranked residual signals, fitted to ranked residual 5-session labels."""
    label_type = context.config.training.label_type
    if label_type != "returns":
        # Rank labels sit in [0, 1]: removing beta times their mean would plant a beta tilt.
        raise ValueError(f"residual_linear residualises return labels; the run used {label_type!r}")
    state = residual_state(context.panel, context.kdcodes, context.pit_intervals)
    signals = signal_panel(state)
    data = context.data
    train_rows = _session_rows(state, context.train_dates)
    val_rows = _session_rows(state, context.val_dates)
    test_rows = _session_rows(state, context.test_dates)

    train_labels = np.asarray(data["train_labels"], dtype=np.float64)
    val_labels = np.asarray(data["val_labels"], dtype=np.float64)
    train_scorable = _split_scorable(context, "train_loss_mask", train_labels)
    val_scorable = _split_scorable(context, "val_tradable_mask", val_labels)
    test_scorable = _test_scorable(context, state, test_rows)
    # Inputs are ranked over the tradable names, as on test dates; the fit then uses only
    # the rows whose label exists (the loss mask), a subset of them.
    train_tradable = data.get("train_tradable_mask")
    train_universe = (
        train_scorable | np.asarray(train_tradable, dtype=bool)
        if train_tradable is not None
        else train_scorable
    )

    target = residual_labels(state, np.where(train_scorable, train_labels, np.nan), train_rows)
    ranked_target = centred_ranks(target, train_scorable & np.isfinite(target))
    fit = fit_cross_sectional_ridge(
        model_inputs(signals, train_rows, train_universe),
        ranked_target,
        model_inputs(signals, val_rows, val_scorable),
        val_labels,
        train_mask=train_scorable,
        val_mask=val_scorable,
    )
    test_inputs = model_inputs(signals, test_rows, test_scorable)
    scores = linear_scores(np.nan_to_num(test_inputs, nan=0.0), fit.coefficients)
    scores = np.where(test_scorable, scores, np.nan)
    return CandidateResult(
        scores=scores,
        controls={
            "definition": "ridge on ranked residual signals, ranked residual 5-session target",
            "signals": SIGNALS,
            "coefficients": dict(zip(SIGNALS, map(float, fit.coefficients), strict=True)),
            "ridge_alpha": fit.alpha,
            "validation_mean_ic": {str(a): ic for a, ic in fit.validation_ic.items()},
            "train_rows": fit.train_rows,
            "train_dates": fit.train_dates,
            **_state_controls(state),
        },
    )


def residual_reversal_1w(context: CandidateContext) -> CandidateResult:
    """Minus the 5-session sum of residual returns, unfitted."""
    state = residual_state(context.panel, context.kdcodes, context.pit_intervals)
    rows = _session_rows(state, context.test_dates)
    scores = -_rolling(state.residuals, 5, 4, "sum")[rows]
    scores[~np.isfinite(scores)] = np.nan
    scores = np.where(_test_scorable(context, state, rows), scores, np.nan)
    return CandidateResult(
        scores=scores,
        controls={"definition": SIGNALS["rrev_5"], **_state_controls(state)},
    )


def _state_controls(state: ResidualState) -> dict[str, object]:
    return {
        "estimation_sessions": ESTIMATION_SESSIONS,
        "min_estimation_sessions": MIN_ESTIMATION_SESSIONS,
        "beta_shrink": BETA_SHRINK,
        "n_clusters": N_CLUSTERS,
        "refreshes": state.refreshes,
        "min_peers": MIN_PEERS,
        "member_cells_with_peer_move_share": state.peered_share,
    }
