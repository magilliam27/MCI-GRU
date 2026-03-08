"""
Momentum feature engineering for MCI-GRU.

This module provides momentum features based on the Goulding, Harvey, Mazzoleni paper
"Momentum Turning Points" (SSRN-3489539).

Three encoding variants are available:
- Binary: Original +1/-1 signals
- Continuous: Raw momentum values (normalized)
- Buffered: With no-trade zones for weak signals
"""

import numpy as np
import pandas as pd


# Momentum feature columns
MOMENTUM_BASE_FEATURES = [
    'slow_momentum', 'fast_momentum',
    'slow_signal', 'fast_signal', 'momentum_blend',
    'cycle_bull', 'cycle_correction', 'cycle_bear'
]
MOMENTUM_WEEKLY_FEATURES = ['weekly_momentum', 'weekly_signal']
MOMENTUM_FEATURES = MOMENTUM_BASE_FEATURES + MOMENTUM_WEEKLY_FEATURES
MOMENTUM_BLEND_MODES = ['static', 'dynamic']

DEFAULT_BLEND_FAST_WEIGHT = 0.5
DEFAULT_DYNAMIC_CORRECTION_FAST_WEIGHT = 0.15
DEFAULT_DYNAMIC_REBOUND_FAST_WEIGHT = 0.70
DEFAULT_DYNAMIC_LOOKBACK_PERIODS = 0
DEFAULT_DYNAMIC_MIN_HISTORY = 252
DEFAULT_DYNAMIC_MIN_STATE_OBSERVATIONS = 3

_DYNAMIC_STATES = ('Bu', 'Co', 'Be', 'Re')
_EPSILON = 1e-12


def get_momentum_features(include_weekly_momentum: bool = True) -> list[str]:
    """Return momentum feature columns with optional weekly terms."""
    if include_weekly_momentum:
        return list(MOMENTUM_FEATURES)
    return list(MOMENTUM_BASE_FEATURES)


def _validate_fast_weight(name: str, value: float) -> None:
    """Validate a blend weight expressed as the FAST allocation."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_dynamic_history_param(name: str, value: int, allow_zero: bool = False) -> None:
    """Validate integer parameters used by the dynamic estimator."""
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    elif value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _blend_signals(
    slow_signal: pd.Series,
    fast_signal: pd.Series,
    fast_weight: pd.Series | float,
) -> pd.Series:
    """Blend slow and fast signals using a FAST allocation weight."""
    return (1.0 - fast_weight) * slow_signal + fast_weight * fast_signal


def _compute_binary_position(momentum: pd.Series) -> pd.Series:
    """Return +1/-1 positions with 0 reserved for insufficient history."""
    position = pd.Series(0.0, index=momentum.index, dtype=float)
    valid = momentum.notna()
    position.loc[valid] = np.where(momentum.loc[valid] >= 0, 1.0, -1.0)
    return position


def _compute_market_states(
    slow_position: pd.Series,
    fast_position: pd.Series,
) -> pd.Series:
    """Map slow/fast positions into Bull, Correction, Bear, and Rebound states."""
    state = pd.Series(pd.NA, index=slow_position.index, dtype='object')
    state.loc[(slow_position == 1.0) & (fast_position == 1.0)] = 'Bu'
    state.loc[(slow_position == 1.0) & (fast_position == -1.0)] = 'Co'
    state.loc[(slow_position == -1.0) & (fast_position == -1.0)] = 'Be'
    state.loc[(slow_position == -1.0) & (fast_position == 1.0)] = 'Re'
    return state


def _window_history(series: pd.Series, lookback_periods: int) -> pd.Series:
    """
    Return expanding/rolling history that excludes the current row.

    A lookback of 0 means use all prior observations (expanding history).
    """
    history = series.shift(1, fill_value=0.0)
    if lookback_periods <= 0:
        return history
    return history - history.shift(lookback_periods, fill_value=0.0)


def _solve_dynamic_speed(
    mu_bu: pd.Series,
    mu_be: pd.Series,
    mu_co: pd.Series,
    mu_re: pd.Series,
    second_bu: pd.Series,
    second_be: pd.Series,
    second_co: pd.Series,
    second_re: pd.Series,
    p_bu: pd.Series,
    p_be: pd.Series,
    fallback_correction_fast_weight: float,
    fallback_rebound_fast_weight: float,
    valid_correction_mask: pd.Series,
    valid_rebound_mask: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Solve Proposition 9 using trailing sample moments.

    a_Co = 1/2 * (1 - (E[r^2|Bu] P[Bu]) / (E[r|Bu]P[Bu] - E[r|Be]P[Be]) * E[r|Co]/E[r^2|Co])
    a_Re = 1/2 * (1 + (E[r^2|Be] P[Be]) / (E[r|Bu]P[Bu] - E[r|Be]P[Be]) * E[r|Re]/E[r^2|Re])
    """
    denominator = mu_bu * p_bu - mu_be * p_be

    a_co = pd.Series(fallback_correction_fast_weight, index=mu_bu.index, dtype=float)
    co_formula_mask = valid_correction_mask & (denominator > _EPSILON) & (second_co > _EPSILON)
    if co_formula_mask.any():
        co_component = (
            (second_bu.loc[co_formula_mask] * p_bu.loc[co_formula_mask])
            / denominator.loc[co_formula_mask]
            * (mu_co.loc[co_formula_mask] / second_co.loc[co_formula_mask])
        )
        a_co.loc[co_formula_mask] = 0.5 * (1.0 - co_component)
    a_co = a_co.clip(0.0, 1.0)

    a_re = pd.Series(fallback_rebound_fast_weight, index=mu_bu.index, dtype=float)
    re_formula_mask = valid_rebound_mask & (denominator > _EPSILON) & (second_re > _EPSILON)
    if re_formula_mask.any():
        re_component = (
            (second_be.loc[re_formula_mask] * p_be.loc[re_formula_mask])
            / denominator.loc[re_formula_mask]
            * (mu_re.loc[re_formula_mask] / second_re.loc[re_formula_mask])
        )
        a_re.loc[re_formula_mask] = 0.5 * (1.0 + re_component)
    a_re = a_re.clip(0.0, 1.0)

    return a_co, a_re


def _estimate_dynamic_fast_weights_for_group(
    group: pd.DataFrame,
    slow_position: pd.Series,
    fast_position: pd.Series,
    fallback_correction_fast_weight: float,
    fallback_rebound_fast_weight: float,
    lookback_periods: int,
    min_history: int,
    min_state_observations: int,
) -> pd.Series:
    """Estimate DYN FAST allocations for one asset using only prior observations."""
    state = _compute_market_states(slow_position, fast_position)
    forward_return = group['_daily_return'].shift(-1)
    valid_pair = state.notna() & forward_return.notna()

    total_count = _window_history(valid_pair.astype(float), lookback_periods)
    forward_sq = forward_return.pow(2)

    counts: dict[str, pd.Series] = {}
    sums: dict[str, pd.Series] = {}
    second_sums: dict[str, pd.Series] = {}
    for state_code in _DYNAMIC_STATES:
        state_mask = (state == state_code) & valid_pair
        counts[state_code] = _window_history(state_mask.astype(float), lookback_periods)
        sums[state_code] = _window_history(forward_return.where(state_mask, 0.0), lookback_periods)
        second_sums[state_code] = _window_history(forward_sq.where(state_mask, 0.0), lookback_periods)

    mu = {
        state_code: sums[state_code] / counts[state_code].where(counts[state_code] > 0, np.nan)
        for state_code in _DYNAMIC_STATES
    }
    second = {
        state_code: second_sums[state_code] / counts[state_code].where(counts[state_code] > 0, np.nan)
        for state_code in _DYNAMIC_STATES
    }
    probabilities = {
        state_code: counts[state_code] / total_count.where(total_count > 0, np.nan)
        for state_code in _DYNAMIC_STATES
    }

    enough_history = total_count >= float(min_history)
    valid_correction_mask = (
        enough_history
        & (counts['Bu'] >= float(min_state_observations))
        & (counts['Be'] >= float(min_state_observations))
        & (counts['Co'] >= float(min_state_observations))
    )
    valid_rebound_mask = (
        enough_history
        & (counts['Bu'] >= float(min_state_observations))
        & (counts['Be'] >= float(min_state_observations))
        & (counts['Re'] >= float(min_state_observations))
    )

    a_co, a_re = _solve_dynamic_speed(
        mu_bu=mu['Bu'],
        mu_be=mu['Be'],
        mu_co=mu['Co'],
        mu_re=mu['Re'],
        second_bu=second['Bu'],
        second_be=second['Be'],
        second_co=second['Co'],
        second_re=second['Re'],
        p_bu=probabilities['Bu'],
        p_be=probabilities['Be'],
        fallback_correction_fast_weight=fallback_correction_fast_weight,
        fallback_rebound_fast_weight=fallback_rebound_fast_weight,
        valid_correction_mask=valid_correction_mask,
        valid_rebound_mask=valid_rebound_mask,
    )

    weights = pd.Series(0.5, index=group.index, dtype=float)
    weights.loc[state == 'Co'] = a_co.loc[state == 'Co']
    weights.loc[state == 'Re'] = a_re.loc[state == 'Re']
    return weights


def _compute_blend_weights(
    df: pd.DataFrame,
    slow_position: pd.Series,
    fast_position: pd.Series,
    blend_mode: str,
    blend_fast_weight: float,
    dynamic_correction_fast_weight: float,
    dynamic_rebound_fast_weight: float,
    dynamic_lookback_periods: int,
    dynamic_min_history: int,
    dynamic_min_state_observations: int,
) -> pd.Series:
    """Return per-row FAST allocation weights for static or paper-style dynamic blending."""
    if blend_mode not in MOMENTUM_BLEND_MODES:
        raise ValueError(
            f"blend_mode must be one of {MOMENTUM_BLEND_MODES}, got {blend_mode!r}"
        )

    _validate_fast_weight("blend_fast_weight", blend_fast_weight)
    _validate_fast_weight(
        "dynamic_correction_fast_weight",
        dynamic_correction_fast_weight,
    )
    _validate_fast_weight(
        "dynamic_rebound_fast_weight",
        dynamic_rebound_fast_weight,
    )
    _validate_dynamic_history_param(
        "dynamic_lookback_periods",
        dynamic_lookback_periods,
        allow_zero=True,
    )
    _validate_dynamic_history_param("dynamic_min_history", dynamic_min_history)
    _validate_dynamic_history_param(
        "dynamic_min_state_observations",
        dynamic_min_state_observations,
    )

    if blend_mode == 'static':
        return pd.Series(blend_fast_weight, index=df.index, dtype=float)

    return df.groupby('kdcode', group_keys=False).apply(
        lambda g: _estimate_dynamic_fast_weights_for_group(
            group=g,
            slow_position=slow_position.loc[g.index],
            fast_position=fast_position.loc[g.index],
            fallback_correction_fast_weight=dynamic_correction_fast_weight,
            fallback_rebound_fast_weight=dynamic_rebound_fast_weight,
            lookback_periods=dynamic_lookback_periods,
            min_history=dynamic_min_history,
            min_state_observations=dynamic_min_state_observations,
        )
    )


def _compute_raw_momentum(
    df: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    include_weekly_momentum: bool,
) -> pd.DataFrame:
    """Compute raw daily-return momentum columns."""
    df = df.sort_values(['kdcode', 'dt']).copy()

    # Calculate daily returns per stock.
    df['_daily_return'] = df.groupby('kdcode')['close'].pct_change()

    # Fast momentum: trailing return.
    df['fast_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=fast_window, min_periods=fast_window).sum()
    )

    # Slow momentum: trailing return.
    df['slow_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=slow_window, min_periods=slow_window).sum()
    )

    if include_weekly_momentum:
        # Weekly momentum: 5-day trailing return.
        df['weekly_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
            lambda x: x.rolling(window=5, min_periods=5).sum()
        )

    return df


def _add_cycle_features(
    df: pd.DataFrame,
    slow_position: pd.Series,
    fast_position: pd.Series,
) -> pd.DataFrame:
    """Add Bull / Correction / Bear one-hot indicators with neutral handling."""
    df['cycle_bull'] = ((slow_position == 1.0) & (fast_position == 1.0)).astype(float)
    df['cycle_correction'] = ((slow_position == 1.0) & (fast_position == -1.0)).astype(float)
    df['cycle_bear'] = ((slow_position == -1.0) & (fast_position == -1.0)).astype(float)
    return df


def _compute_static_or_dynamic_blend(
    df: pd.DataFrame,
    slow_signal: pd.Series,
    fast_signal: pd.Series,
    slow_position: pd.Series,
    fast_position: pd.Series,
    blend_mode: str,
    blend_fast_weight: float,
    dynamic_correction_fast_weight: float,
    dynamic_rebound_fast_weight: float,
    dynamic_lookback_periods: int,
    dynamic_min_history: int,
    dynamic_min_state_observations: int,
) -> pd.Series:
    """Compute the speed-selected blend signal for the current encoding."""
    blend_weights = _compute_blend_weights(
        df=df,
        slow_position=slow_position,
        fast_position=fast_position,
        blend_mode=blend_mode,
        blend_fast_weight=blend_fast_weight,
        dynamic_correction_fast_weight=dynamic_correction_fast_weight,
        dynamic_rebound_fast_weight=dynamic_rebound_fast_weight,
        dynamic_lookback_periods=dynamic_lookback_periods,
        dynamic_min_history=dynamic_min_history,
        dynamic_min_state_observations=dynamic_min_state_observations,
    )
    blend = _blend_signals(slow_signal, fast_signal, blend_weights)
    invalid_mask = (slow_position == 0.0) | (fast_position == 0.0)
    blend.loc[invalid_mask] = 0.0
    return blend


def add_momentum_binary(df: pd.DataFrame,
                        fast_window: int = 21,
                        slow_window: int = 252,
                        blend_mode: str = 'static',
                        blend_fast_weight: float = DEFAULT_BLEND_FAST_WEIGHT,
                        dynamic_correction_fast_weight: float = DEFAULT_DYNAMIC_CORRECTION_FAST_WEIGHT,
                        dynamic_rebound_fast_weight: float = DEFAULT_DYNAMIC_REBOUND_FAST_WEIGHT,
                        dynamic_lookback_periods: int = DEFAULT_DYNAMIC_LOOKBACK_PERIODS,
                        dynamic_min_history: int = DEFAULT_DYNAMIC_MIN_HISTORY,
                        dynamic_min_state_observations: int = DEFAULT_DYNAMIC_MIN_STATE_OBSERVATIONS,
                        include_weekly_momentum: bool = True) -> pd.DataFrame:
    """
    Add binary momentum features from the Momentum Turning Points paper.

    Paper methodology:
    - Slow momentum: 12-month (252 trading days) trailing return
    - Fast momentum: 1-month (21 trading days) trailing return
    - MED: static a = 1/2 blend of SLOW and FAST
    - DYN: no-lookahead estimate of a_Co and a_Re from historical state-conditioned moments

    Features added:
        - slow_momentum: 252-day cumulative return
        - fast_momentum: 21-day cumulative return
        - slow_signal: +1 if slow_momentum >= 0, else -1
        - fast_signal: +1 if fast_momentum >= 0, else -1
        - momentum_blend: Static-speed weight w_t(a) or paper-style DYN weight w_t(a_s(t))
        - cycle_bull: 1 if Bull state, else 0
        - cycle_correction: 1 if Correction state, else 0
        - cycle_bear: 1 if Bear state, else 0
    """
    print("Computing binary momentum features...")
    df = _compute_raw_momentum(df, fast_window, slow_window, include_weekly_momentum)

    slow_position = _compute_binary_position(df['slow_momentum'])
    fast_position = _compute_binary_position(df['fast_momentum'])
    df['slow_signal'] = slow_position
    df['fast_signal'] = fast_position

    if include_weekly_momentum:
        df['weekly_signal'] = _compute_binary_position(df['weekly_momentum'])

    df['momentum_blend'] = _compute_static_or_dynamic_blend(
        df=df,
        slow_signal=df['slow_signal'],
        fast_signal=df['fast_signal'],
        slow_position=slow_position,
        fast_position=fast_position,
        blend_mode=blend_mode,
        blend_fast_weight=blend_fast_weight,
        dynamic_correction_fast_weight=dynamic_correction_fast_weight,
        dynamic_rebound_fast_weight=dynamic_rebound_fast_weight,
        dynamic_lookback_periods=dynamic_lookback_periods,
        dynamic_min_history=dynamic_min_history,
        dynamic_min_state_observations=dynamic_min_state_observations,
    )

    df['slow_momentum'] = df['slow_momentum'].fillna(0.0)
    df['fast_momentum'] = df['fast_momentum'].fillna(0.0)
    if include_weekly_momentum:
        df['weekly_momentum'] = df['weekly_momentum'].fillna(0.0)

    df = _add_cycle_features(df, slow_position, fast_position)
    df = df.drop(columns=['_daily_return'])

    print(f"  Added momentum features: {get_momentum_features(include_weekly_momentum)}")
    print(f"  Rows with valid slow momentum: {(df['slow_momentum'] != 0).sum()} / {len(df)}")

    return df


def add_momentum_continuous(df: pd.DataFrame,
                            fast_window: int = 21,
                            slow_window: int = 252,
                            blend_mode: str = 'static',
                            blend_fast_weight: float = DEFAULT_BLEND_FAST_WEIGHT,
                            dynamic_correction_fast_weight: float = DEFAULT_DYNAMIC_CORRECTION_FAST_WEIGHT,
                            dynamic_rebound_fast_weight: float = DEFAULT_DYNAMIC_REBOUND_FAST_WEIGHT,
                            dynamic_lookback_periods: int = DEFAULT_DYNAMIC_LOOKBACK_PERIODS,
                            dynamic_min_history: int = DEFAULT_DYNAMIC_MIN_HISTORY,
                            dynamic_min_state_observations: int = DEFAULT_DYNAMIC_MIN_STATE_OBSERVATIONS,
                            include_weekly_momentum: bool = True) -> pd.DataFrame:
    """
    Add continuous momentum features (raw values, not binary signals).

    This variant keeps the actual momentum values instead of converting
    to binary +1/-1 signals, allowing the model to learn from momentum magnitude.
    """
    print("Computing continuous momentum features...")
    df = _compute_raw_momentum(df, fast_window, slow_window, include_weekly_momentum)

    slow_position = _compute_binary_position(df['slow_momentum'])
    fast_position = _compute_binary_position(df['fast_momentum'])

    df['slow_momentum'] = df['slow_momentum'].fillna(0.0)
    df['fast_momentum'] = df['fast_momentum'].fillna(0.0)
    if include_weekly_momentum:
        df['weekly_momentum'] = df['weekly_momentum'].fillna(0.0)

    # Normalize signals per day (cross-sectional z-score).
    df['slow_signal'] = df.groupby('dt')['slow_momentum'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    df['fast_signal'] = df.groupby('dt')['fast_momentum'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    if include_weekly_momentum:
        df['weekly_signal'] = df.groupby('dt')['weekly_momentum'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    df['slow_signal'] = df['slow_signal'].clip(-3, 3).fillna(0.0)
    df['fast_signal'] = df['fast_signal'].clip(-3, 3).fillna(0.0)
    if include_weekly_momentum:
        df['weekly_signal'] = df['weekly_signal'].clip(-3, 3).fillna(0.0)

    df.loc[slow_position == 0.0, 'slow_signal'] = 0.0
    df.loc[fast_position == 0.0, 'fast_signal'] = 0.0
    if include_weekly_momentum:
        weekly_position = _compute_binary_position(df['weekly_momentum'])
        df.loc[weekly_position == 0.0, 'weekly_signal'] = 0.0

    df['momentum_blend'] = _compute_static_or_dynamic_blend(
        df=df,
        slow_signal=df['slow_signal'],
        fast_signal=df['fast_signal'],
        slow_position=slow_position,
        fast_position=fast_position,
        blend_mode=blend_mode,
        blend_fast_weight=blend_fast_weight,
        dynamic_correction_fast_weight=dynamic_correction_fast_weight,
        dynamic_rebound_fast_weight=dynamic_rebound_fast_weight,
        dynamic_lookback_periods=dynamic_lookback_periods,
        dynamic_min_history=dynamic_min_history,
        dynamic_min_state_observations=dynamic_min_state_observations,
    )

    df = _add_cycle_features(df, slow_position, fast_position)
    df = df.drop(columns=['_daily_return'])

    print("  Added continuous momentum features")
    return df


def add_momentum_buffered(df: pd.DataFrame,
                          fast_window: int = 21,
                          slow_window: int = 252,
                          buffer_low: float = 0.1,
                          buffer_high: float = 0.9,
                          blend_mode: str = 'static',
                          blend_fast_weight: float = DEFAULT_BLEND_FAST_WEIGHT,
                          dynamic_correction_fast_weight: float = DEFAULT_DYNAMIC_CORRECTION_FAST_WEIGHT,
                          dynamic_rebound_fast_weight: float = DEFAULT_DYNAMIC_REBOUND_FAST_WEIGHT,
                          dynamic_lookback_periods: int = DEFAULT_DYNAMIC_LOOKBACK_PERIODS,
                          dynamic_min_history: int = DEFAULT_DYNAMIC_MIN_HISTORY,
                          dynamic_min_state_observations: int = DEFAULT_DYNAMIC_MIN_STATE_OBSERVATIONS,
                          include_weekly_momentum: bool = True) -> pd.DataFrame:
    """
    Add buffered momentum features with no-trade zones.

    This variant introduces buffers at the extremes:
    - Weak signals (near zero): Set to 0 (no trade)
    - Moderate signals: Scaled linearly
    - Extreme signals: Clipped (potential mean reversion concern)
    """
    print(f"Computing buffered momentum features (buffer_low={buffer_low}, buffer_high={buffer_high})...")
    df = _compute_raw_momentum(df, fast_window, slow_window, include_weekly_momentum)

    slow_position = _compute_binary_position(df['slow_momentum'])
    fast_position = _compute_binary_position(df['fast_momentum'])

    df['slow_momentum'] = df['slow_momentum'].fillna(0.0)
    df['fast_momentum'] = df['fast_momentum'].fillna(0.0)
    if include_weekly_momentum:
        df['weekly_momentum'] = df['weekly_momentum'].fillna(0.0)

    def compute_buffered_signal(group: pd.DataFrame, col: str) -> pd.Series:
        """Compute buffered signal for a day's cross-section."""
        values = group[col].values
        if len(values) == 0 or np.all(values == 0):
            return pd.Series(np.zeros(len(values)), index=group.index)

        ranks = pd.Series(values).rank(pct=True).values
        signals = np.zeros(len(values))

        for i, rank in enumerate(ranks):
            if rank < buffer_low:
                signals[i] = 0.0
            elif rank > buffer_high:
                signals[i] = 1.0
            else:
                signals[i] = (rank - buffer_low) / (buffer_high - buffer_low) * 2 - 1

        return pd.Series(signals, index=group.index)

    df['slow_signal'] = df.groupby('dt', group_keys=False).apply(
        lambda g: compute_buffered_signal(g, 'slow_momentum')
    )
    df['fast_signal'] = df.groupby('dt', group_keys=False).apply(
        lambda g: compute_buffered_signal(g, 'fast_momentum')
    )
    if include_weekly_momentum:
        df['weekly_signal'] = df.groupby('dt', group_keys=False).apply(
            lambda g: compute_buffered_signal(g, 'weekly_momentum')
        )

    df.loc[slow_position == 0.0, 'slow_signal'] = 0.0
    df.loc[fast_position == 0.0, 'fast_signal'] = 0.0
    if include_weekly_momentum:
        weekly_position = _compute_binary_position(df['weekly_momentum'])
        df.loc[weekly_position == 0.0, 'weekly_signal'] = 0.0

    df['momentum_blend'] = _compute_static_or_dynamic_blend(
        df=df,
        slow_signal=df['slow_signal'],
        fast_signal=df['fast_signal'],
        slow_position=slow_position,
        fast_position=fast_position,
        blend_mode=blend_mode,
        blend_fast_weight=blend_fast_weight,
        dynamic_correction_fast_weight=dynamic_correction_fast_weight,
        dynamic_rebound_fast_weight=dynamic_rebound_fast_weight,
        dynamic_lookback_periods=dynamic_lookback_periods,
        dynamic_min_history=dynamic_min_history,
        dynamic_min_state_observations=dynamic_min_state_observations,
    )

    df['trade_signal'] = np.where(
        df['momentum_blend'].abs() < 0.2,
        0,
        np.sign(df['momentum_blend'])
    )

    df = _add_cycle_features(df, slow_position, fast_position)
    df = df.drop(columns=['_daily_return'])

    print("  Added buffered momentum features")
    return df


# Alias for backwards compatibility
add_momentum_features = add_momentum_binary
