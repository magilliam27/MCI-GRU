"""A small neural net fed only each stock's return history (#385).

Murray, Xia and Xiao (2024, JFE), "Charting by Machines", report that a network given
nothing but a stock's past returns predicts the cross-section of returns, including
among the 500 largest stocks. This module is that idea at MCI-GRU's own horizon and
axis, as a candidate arm for the #384 harness. It is far smaller than MCI-GRU and has
no graph, no volume, and no market-wide inputs.

Inputs at the close of signal date ``D``, from closes up to ``D`` only, counted in
sessions on the label's own axis (``label_session_axis``), as ``past_return`` counts
them:

* the cumulative return from close ``D-j`` to close ``D`` for ``j = 1..20`` (a daily
  path, for the weekly label);
* the cumulative return from close ``D-21k`` to close ``D`` for ``k = 1..12`` (the
  paper's twelve cumulative monthly returns).

Each input is ranked within its date to ``[-1, 1]``; a missing close gives a neutral 0.
A stock without a close on ``D`` itself has no score that date.

The target is the run's own training label, standardised within each training date
over the run's loss mask, fitted by MSE with every date weighted equally. Each member
stops early on validation mean daily IC (the recipe's ``selection_metric=val_ic``), and
the arm's score is the mean of the members' within-date standardised predictions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from mci_gru.data.preprocessing import label_session_axis
from mci_gru.evaluation.candidates import CandidateContext, CandidateResult
from mci_gru.evaluation.statistics import daily_ic_series

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

#: Daily path: close ``D-j`` to close ``D``.
DAILY_LAGS: tuple[int, ...] = tuple(range(1, 21))
#: Monthly path: close ``D-21k`` to close ``D``, twelve months of 21 sessions.
MONTH_SESSIONS = 21
MONTHLY_LAGS: tuple[int, ...] = tuple(MONTH_SESSIONS * k for k in range(1, 13))
INPUT_LAGS: tuple[int, ...] = DAILY_LAGS + MONTHLY_LAGS


@dataclass(frozen=True)
class NetSpec:
    """Training settings, fixed before any window is scored (#385)."""

    architecture: str = "fnn"  # "fnn" or "cnn_lstm"
    members: int = 10
    max_epochs: int = 100
    patience: int = 10
    dates_per_batch: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 385


@dataclass
class NetFit:
    """What the arm reports about its own training, for the harness's controls."""

    scores: np.ndarray  # (test dates, stocks); NaN where unscored
    member_best_epoch: list[int] = field(default_factory=list)
    member_val_ic: list[float] = field(default_factory=list)
    ensemble_val_ic: float = float("nan")
    train_dates: int = 0
    train_rows: int = 0


# ── inputs ───────────────────────────────────────────────────────────────


def _close_matrix(panel: pd.DataFrame, kdcodes: list[str]) -> tuple[list[str], np.ndarray]:
    rows = panel.loc[:, ["kdcode", "dt", "close"]].copy()
    rows["dt"] = rows["dt"].astype(str)
    rows["kdcode"] = rows["kdcode"].astype(str)
    sessions = label_session_axis(rows, list(kdcodes))
    closes = rows.pivot_table(index="dt", columns="kdcode", values="close", aggfunc="mean")
    closes = closes.reindex(index=sessions, columns=list(kdcodes)).to_numpy(dtype=np.float64)
    return sessions, closes


def return_history_inputs(
    panel: pd.DataFrame,
    kdcodes: list[str],
    dates: list[str],
    lags: tuple[int, ...] = INPUT_LAGS,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw cumulative returns ``(dates, stocks, lags)`` and the scoreable mask.

    Entry ``[i, s, l]`` is close ``D`` over close ``D-lags[l]`` minus one, where ``D`` is
    ``dates[i]``, NaN when either close is missing or the lag runs off the axis. The
    mask ``(dates, stocks)`` is true where the stock has a close on ``D`` itself.
    Nothing after ``D`` is read.
    """
    sessions, closes = _close_matrix(panel, kdcodes)
    position = {session: i for i, session in enumerate(sessions)}
    lag_array = np.asarray(lags, dtype=np.int64)
    out = np.full((len(dates), len(kdcodes), len(lags)), np.nan)
    scoreable = np.zeros((len(dates), len(kdcodes)), dtype=bool)
    for row, date in enumerate(dates):
        at = position.get(str(date))
        if at is None:
            continue
        now = closes[at]
        scoreable[row] = np.isfinite(now) & (now > 0)
        starts = at - lag_array
        valid = starts >= 0
        past = np.full((len(lags), len(kdcodes)), np.nan)
        past[valid] = closes[starts[valid]]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[row] = (now[None, :] / past - 1.0).T
    out[~np.isfinite(out)] = np.nan
    return out, scoreable


def rank_within_date(raw: np.ndarray) -> np.ndarray:
    """Each input ranked within its date to ``[-1, 1]`` (ties averaged); NaN becomes 0."""
    out = np.zeros_like(raw, dtype=np.float32)
    for i in range(raw.shape[0]):
        for k in range(raw.shape[2]):
            column = raw[i, :, k]
            finite = np.isfinite(column)
            n = int(finite.sum())
            if n < 2:
                continue
            values = column[finite]
            order = values.argsort(kind="mergesort")
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64)
            # average tied ranks
            sorted_values = values[order]
            boundaries = np.flatnonzero(np.diff(sorted_values)) + 1
            for group in np.split(order, boundaries):
                if group.size > 1:
                    ranks[group] = ranks[group].mean()
            out[i, finite, k] = (2.0 * ranks / (n - 1) - 1.0).astype(np.float32)
    return out


def standardise_within_date(labels: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Labels z-scored within each date over the masked finite rows; others NaN."""
    y = np.asarray(labels, dtype=np.float64)
    keep = np.isfinite(y)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    out = np.full(y.shape, np.nan)
    for i in range(y.shape[0]):
        rows = keep[i]
        if int(rows.sum()) < 2:
            continue
        values = y[i, rows]
        sd = values.std()
        if sd > 0:
            out[i, rows] = (values - values.mean()) / sd
    return out


# ── networks ─────────────────────────────────────────────────────────────


class ReturnFNN(nn.Module):
    """32-32-16-1 ReLU network over the ranked return path."""

    def __init__(self, n_inputs: int = len(INPUT_LAGS)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _PathEncoder(nn.Module):
    def __init__(self, channels: int = 8, hidden: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)

    def forward(self, path: torch.Tensor) -> torch.Tensor:  # (rows, steps)
        h = torch.relu(self.conv(path.unsqueeze(1)))  # (rows, channels, steps)
        _, (last, _) = self.lstm(h.transpose(1, 2))
        return last[-1]


class ReturnCNNLSTM(nn.Module):
    """A 1-D CNN plus LSTM over each path, oldest step first, then a small head."""

    def __init__(self) -> None:
        super().__init__()
        self.daily = _PathEncoder()
        self.monthly = _PathEncoder()
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.n_daily = len(DAILY_LAGS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lags run newest first; flip so each sequence ends at D.
        daily = torch.flip(x[:, : self.n_daily], dims=[1])
        monthly = torch.flip(x[:, self.n_daily :], dims=[1])
        return self.head(torch.cat([self.daily(daily), self.monthly(monthly)], dim=1)).squeeze(-1)


ARCHITECTURES: dict[str, Callable[[], nn.Module]] = {
    "fnn": ReturnFNN,
    "cnn_lstm": ReturnCNNLSTM,
}


# ── training ─────────────────────────────────────────────────────────────


def _predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        flat = torch.from_numpy(x.reshape(-1, x.shape[-1]))
        return model(flat).numpy().astype(np.float64).reshape(x.shape[:2])


def _mean_ic(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    values = daily_ic_series(np.where(mask, scores, np.nan), labels, "pearson")
    return float(np.mean(values)) if values.size else float("nan")


def _standardise_scores(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return standardise_within_date(scores, mask)


def _train_member(
    spec: NetSpec,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    val_labels: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[nn.Module, int, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ARCHITECTURES[spec.architecture]()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    usable = [i for i in range(len(y_train)) if np.isfinite(y_train[i]).sum() >= 2]
    x_t = torch.from_numpy(x_train)
    y_t = torch.from_numpy(np.nan_to_num(y_train, nan=0.0).astype(np.float32))
    w_t = torch.from_numpy(np.isfinite(y_train).astype(np.float32))

    best_ic, best_epoch, best_state, stale = -np.inf, -1, None, 0
    for epoch in range(spec.max_epochs):
        model.train()
        order = rng.permutation(usable)
        for start in range(0, len(order), spec.dates_per_batch):
            batch = torch.from_numpy(order[start : start + spec.dates_per_batch])
            xb, yb, wb = x_t[batch], y_t[batch], w_t[batch]
            pred = model(xb.reshape(-1, xb.shape[-1])).reshape(yb.shape)
            per_date = ((pred - yb) ** 2 * wb).sum(dim=1) / wb.sum(dim=1)
            loss = per_date.mean()  # every date weighted equally
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        ic = _mean_ic(_predict(model, x_val), val_labels, val_mask)
        if np.isfinite(ic) and ic > best_ic:
            best_ic, best_epoch, stale = ic, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= spec.patience:
                break
    if best_state is None:
        raise ValueError("no epoch gave a finite validation IC")
    model.load_state_dict(best_state)
    return model, best_epoch, float(best_ic)


def fit_predict(*args, **kwargs) -> NetFit:
    """``_fit_predict`` on one CPU thread, restoring the caller's thread count after.

    These networks are tiny: one thread is faster than several (thread hand-offs
    dominate an LSTM step this small), and it lets the five windows run side by side.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _fit_predict(*args, **kwargs)
    finally:
        torch.set_num_threads(threads)


def _fit_predict(
    panel: pd.DataFrame,
    kdcodes: list[str],
    *,
    train_dates: list[str],
    train_labels: np.ndarray,
    train_mask: np.ndarray | None,
    val_dates: list[str],
    val_labels: np.ndarray,
    val_mask: np.ndarray | None,
    test_dates: list[str],
    spec: NetSpec | None = None,
) -> NetFit:
    """Train ``spec.members`` networks on the training dates; score the test dates.

    Training rows are those with a finite label inside ``train_mask`` (the run's loss
    mask). Validation IC is read where ``val_mask`` (the run's tradable mask) holds and
    the stock is scoreable. Test scores are NaN where the stock has no close on ``D``;
    the harness applies the run's PIT mask on top.
    """
    spec = spec or NetSpec()
    if spec.architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {spec.architecture!r}")

    raw_tr, ok_tr = return_history_inputs(panel, kdcodes, train_dates)
    raw_va, ok_va = return_history_inputs(panel, kdcodes, val_dates)
    raw_te, ok_te = return_history_inputs(panel, kdcodes, test_dates)
    x_tr, x_va, x_te = rank_within_date(raw_tr), rank_within_date(raw_va), rank_within_date(raw_te)

    train_keep = ok_tr if train_mask is None else ok_tr & np.asarray(train_mask, dtype=bool)
    y_tr = standardise_within_date(train_labels, train_keep)
    val_keep = ok_va if val_mask is None else ok_va & np.asarray(val_mask, dtype=bool)
    val_y = np.asarray(val_labels, dtype=np.float64)

    fit = NetFit(scores=np.full((len(test_dates), len(kdcodes)), np.nan))
    fit.train_dates = int((np.isfinite(y_tr).sum(axis=1) >= 2).sum())
    fit.train_rows = int(np.isfinite(y_tr).sum())
    val_members, test_members = [], []
    for member in range(spec.members):
        model, epoch, ic = _train_member(
            spec, spec.seed + member, x_tr, y_tr, x_va, val_y, val_keep
        )
        fit.member_best_epoch.append(epoch)
        fit.member_val_ic.append(ic)
        val_members.append(_standardise_scores(_predict(model, x_va), val_keep))
        test_members.append(_standardise_scores(_predict(model, x_te), ok_te))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN cells stay NaN
        val_scores = np.nanmean(np.stack(val_members), axis=0)
        scores = np.nanmean(np.stack(test_members), axis=0)
    fit.ensemble_val_ic = _mean_ic(val_scores, val_y, val_keep)
    fit.scores = np.where(ok_te, scores, np.nan)
    return fit


# ── candidate arms (#384 harness) ────────────────────────────────────────


def _score_candidate(context: CandidateContext, spec: NetSpec) -> CandidateResult:
    data = context.data
    fit = fit_predict(
        context.panel,
        context.kdcodes,
        train_dates=context.train_dates,
        train_labels=np.asarray(data["train_labels"], dtype=np.float64),
        train_mask=data.get("train_loss_mask"),
        val_dates=context.val_dates,
        val_labels=np.asarray(data["val_labels"], dtype=np.float64),
        val_mask=data.get("val_tradable_mask"),
        test_dates=context.test_dates,
        spec=spec,
    )
    scores = fit.scores
    if context.rankable_mask is not None:
        scores = np.where(np.asarray(context.rankable_mask, dtype=bool), scores, np.nan)
    return CandidateResult(
        scores=scores,
        controls={
            "definition": f"{spec.architecture} on ranked cumulative return paths "
            "(20 daily, 12 monthly lags); target the training label standardised within date",
            "input_lags_sessions": list(INPUT_LAGS),
            "spec": {
                "architecture": spec.architecture,
                "members": spec.members,
                "max_epochs": spec.max_epochs,
                "patience": spec.patience,
                "dates_per_batch": spec.dates_per_batch,
                "learning_rate": spec.learning_rate,
                "weight_decay": spec.weight_decay,
                "seed": spec.seed,
            },
            "member_best_epoch": fit.member_best_epoch,
            "member_validation_ic": fit.member_val_ic,
            "ensemble_validation_ic": fit.ensemble_val_ic,
            "train_dates": fit.train_dates,
            "train_rows": fit.train_rows,
        },
    )


def return_net_fnn(context: CandidateContext) -> CandidateResult:
    """The primary arm: a 32-32-16-1 network over the ranked return path."""
    return _score_candidate(context, NetSpec(architecture="fnn"))


def return_net_cnn_lstm(context: CandidateContext) -> CandidateResult:
    """The secondary arm: a CNN plus LSTM over the daily and monthly paths."""
    return _score_candidate(context, NetSpec(architecture="cnn_lstm"))
