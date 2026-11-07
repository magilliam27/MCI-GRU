"""Simple baseline signals scored on a run's own axis (#360).

The evaluation core (#335, #336) tells us whether a run beat zero and the
equal-weight universe. It does not say whether the run beat signals anyone could
compute in one line. These are those signals, built only from information known at
the close of each signal date ``D``:

* **Reversal**: minus the stock's return over the last ``lookback`` sessions,
  close ``D-lookback`` to close ``D``.
* **12-1 momentum**: the return from close ``D-252`` to close ``D-21``, skipping the
  most recent month, as Jegadeesh and Titman define it.
* **Linear model**: a cross-sectional ridge regression of the training label on the
  run's own stock-level input window, fitted on the run's training dates and with
  its penalty chosen on validation IC.

Past returns read closes on the panel's own session axis, as labels do
(``label_session_axis``), so a missing bar never moves an endpoint: a stock without a
close at either end has no score that date.

The linear model sees every stock-level input the run's encoder sees. Global inputs
(regime scores, VIX, credit spreads) are meant to hold one value per date for every
stock, so a linear score shifts by the same amount for all names and its ranking
cannot use them; the fit demeans each date's rows, which removes any such input
exactly. The caller leaves them out of the inputs it rebuilds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mci_gru.data.preprocessing import label_session_axis
from mci_gru.evaluation.statistics import daily_ic_series

if TYPE_CHECKING:
    import pandas as pd

#: One week (the label's own horizon) and one month, in sessions.
REVERSAL_LOOKBACKS = {"reversal_1w": 5, "reversal_1m": 21}
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21
#: Ridge penalties, as multiples of the mean nonzero diagonal of the demeaned X'X.
RIDGE_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
#: An input whose demeaned sum of squares is below this share of the largest varies by
#: rounding only.
VARYING_INPUT_RTOL = 1e-12
#: Validation ICs this close are a tie, which goes to the larger penalty. Rescaling a
#: score leaves its IC unchanged up to rounding, so exact equality would never tie.
IC_TIE_TOLERANCE = 1e-12


def past_return(
    panel: pd.DataFrame,
    kdcodes: list[str],
    dates: list[str],
    start_lag: int,
    end_lag: int = 0,
) -> np.ndarray:
    """Close ``D-end_lag`` over close ``D-start_lag``, minus one, per date and stock.

    Lags count sessions on the axis the training label counts on (every date any of
    ``kdcodes`` printed, ``label_session_axis``). A date off the axis, a start before the axis begins, or a missing close
    at either end gives NaN.
    """
    if not 0 <= end_lag < start_lag:
        raise ValueError(f"need 0 <= end_lag < start_lag, got {end_lag} and {start_lag}")
    rows = panel.loc[:, ["kdcode", "dt", "close"]].copy()
    rows["dt"] = rows["dt"].astype(str)
    rows["kdcode"] = rows["kdcode"].astype(str)
    sessions = label_session_axis(rows, list(kdcodes))
    closes = rows.pivot_table(index="dt", columns="kdcode", values="close", aggfunc="mean")
    closes = closes.reindex(index=sessions, columns=list(kdcodes)).to_numpy(dtype=np.float64)

    position = {session: i for i, session in enumerate(sessions)}
    out = np.full((len(dates), len(kdcodes)), np.nan)
    for row, date in enumerate(dates):
        at = position.get(str(date))
        if at is None or at - start_lag < 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            out[row] = closes[at - end_lag] / closes[at - start_lag] - 1.0
    out[~np.isfinite(out)] = np.nan
    return out


def reversal_scores(
    panel: pd.DataFrame, kdcodes: list[str], dates: list[str], lookback: int
) -> np.ndarray:
    """Minus the return over the last ``lookback`` sessions: recent losers rank first."""
    return -past_return(panel, kdcodes, dates, start_lag=lookback)


def momentum_scores(
    panel: pd.DataFrame,
    kdcodes: list[str],
    dates: list[str],
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> np.ndarray:
    """Return from close ``D-lookback`` to close ``D-skip``."""
    return past_return(panel, kdcodes, dates, start_lag=lookback, end_lag=skip)


def _date_inputs(features: np.ndarray, i: int) -> np.ndarray:
    """Date ``i``'s ``(stocks, inputs)`` matrix in float64, its window flattened.

    One date at a time, so a float32 training tensor is never copied whole.
    """
    x = np.asarray(features[i], dtype=np.float64)
    return x.reshape(x.shape[0], -1)


def _rows(labels: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    rows = np.isfinite(np.asarray(labels, dtype=np.float64))
    if mask is not None:
        rows &= np.asarray(mask, dtype=bool)
    return rows


@dataclass(frozen=True)
class RidgeFit:
    """A cross-sectional ridge fit and how its penalty was chosen."""

    alpha: float
    coefficients: np.ndarray  # (inputs,)
    validation_ic: dict[float, float]  # alpha -> mean daily Pearson IC on validation
    train_rows: int
    train_dates: int


def _demeaned_moments(
    features: np.ndarray, labels: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """X'X and X'y over ``rows``, each date's X and y demeaned over its own rows."""
    y = np.asarray(labels, dtype=np.float64)
    width = int(np.prod(np.shape(features)[2:]))
    xtx = np.zeros((width, width))
    xty = np.zeros(width)
    used_rows = used_dates = 0
    for i in range(len(features)):
        keep = rows[i]
        if int(keep.sum()) < 2:
            continue
        xi = _date_inputs(features, i)[keep]
        yi = y[i, keep]
        xi = xi - xi.mean(axis=0)
        yi = yi - yi.mean()
        xtx += xi.T @ xi
        xty += xi.T @ yi
        used_rows += int(keep.sum())
        used_dates += 1
    return xtx, xty, used_rows, used_dates


def _solve(xtx: np.ndarray, xty: np.ndarray, alpha: float) -> np.ndarray:
    # Scale the penalty by inputs that vary across stocks: one that holds a single value
    # per date demeans to zero and must not change the penalty the others get.
    # Demeaning leaves such an input with rounding noise, not an exact zero.
    diagonal = np.diag(xtx)
    varying = diagonal[diagonal > VARYING_INPUT_RTOL * diagonal.max(initial=0.0)]
    scale = float(np.mean(varying)) if varying.size else 0.0
    penalised = xtx + alpha * (scale if scale > 0 else 1.0) * np.eye(xtx.shape[0])
    return np.linalg.lstsq(penalised, xty, rcond=None)[0]


def linear_scores(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Each stock's score: its input window times the coefficients."""
    beta = np.asarray(coefficients, dtype=np.float64)
    return np.stack([_date_inputs(features, i) @ beta for i in range(len(features))])


def fit_cross_sectional_ridge(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    *,
    train_mask: np.ndarray | None = None,
    val_mask: np.ndarray | None = None,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
) -> RidgeFit:
    """Fit on training rows only; keep the penalty with the best validation mean IC.

    A row is used where its label is finite and its mask (the PIT loss mask for the
    labels, the tradable mask for the validation scores) is true. Validation ICs within
    ``IC_TIE_TOLERANCE`` of the best are a tie, which goes to the larger penalty.
    """
    if not alphas:
        raise ValueError("alphas must name at least one penalty")
    rows = _rows(train_labels, train_mask)
    xtx, xty, used_rows, used_dates = _demeaned_moments(train_features, train_labels, rows)
    if used_dates == 0:
        raise ValueError("no training date has two or more usable rows")

    val_scores_mask = np.ones(np.shape(val_labels), dtype=bool) if val_mask is None else val_mask
    validation_ic: dict[float, float] = {}
    best: tuple[float, float, np.ndarray] | None = None
    for alpha in sorted(float(a) for a in alphas):
        beta = _solve(xtx, xty, alpha)
        scores = np.where(val_scores_mask, linear_scores(val_features, beta), np.nan)
        values = daily_ic_series(scores, val_labels, "pearson")
        ic = float(np.mean(values)) if values.size else float("nan")
        validation_ic[alpha] = ic
        if np.isfinite(ic) and (best is None or ic >= best[1] - IC_TIE_TOLERANCE):
            best = (alpha, ic, beta)
    if best is None:
        raise ValueError("no penalty gave a finite validation IC")
    return RidgeFit(
        alpha=best[0],
        coefficients=best[2],
        validation_ic=validation_ic,
        train_rows=used_rows,
        train_dates=used_dates,
    )
