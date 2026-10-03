"""Percentage change with pandas 2's gap handling, identical under pandas 2 and 3."""

import pandas as pd


def padded_pct_change(
    values: pd.Series, groups: pd.Series | None = None, periods: int = 1
) -> pd.Series:
    """Return ``values.pct_change`` after carrying the last observation over gaps.

    pandas 2's default ``fill_method='pad'`` did this implicitly; pandas 3 leaves a
    gap NaN instead, which silently changes every return measured across it. The
    project keeps pandas 2's behaviour (#245): a gap day has zero change and the
    first day after a gap spans the whole gap. Leading NaN stays NaN. With
    ``groups``, the fill and the change never cross from one group to the next.
    """
    if groups is None:
        return values.ffill().pct_change(periods=periods, fill_method=None)
    filled = values.groupby(groups).ffill()
    return filled.groupby(groups).pct_change(periods=periods, fill_method=None)
