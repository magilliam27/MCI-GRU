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
from typing import Tuple


# Momentum feature columns
MOMENTUM_BASE_FEATURES = [
    'slow_momentum', 'fast_momentum',
    'slow_signal', 'fast_signal', 'momentum_blend',
    'cycle_bull', 'cycle_correction', 'cycle_bear'
]
MOMENTUM_WEEKLY_FEATURES = ['weekly_momentum', 'weekly_signal']
MOMENTUM_FEATURES = MOMENTUM_BASE_FEATURES + MOMENTUM_WEEKLY_FEATURES


def get_momentum_features(include_weekly_momentum: bool = True) -> list[str]:
    """Return momentum feature columns with optional weekly terms."""
    if include_weekly_momentum:
        return list(MOMENTUM_FEATURES)
    return list(MOMENTUM_BASE_FEATURES)


def add_momentum_binary(df: pd.DataFrame, 
                        fast_window: int = 21,
                        slow_window: int = 252,
                        include_weekly_momentum: bool = True) -> pd.DataFrame:
    """
    Add binary momentum features from the Momentum Turning Points paper.
    
    Paper methodology:
    - Slow momentum: 12-month (252 trading days) trailing return
    - Fast momentum: 1-month (21 trading days) trailing return
    - Market cycles: Bull, Correction, Bear, Rebound based on slow/fast agreement
    
    Features added:
        - slow_momentum: 252-day cumulative return
        - fast_momentum: 21-day cumulative return
        - slow_signal: +1 if slow_momentum >= 0, else -1
        - fast_signal: +1 if fast_momentum >= 0, else -1
        - momentum_blend: MED strategy signal = (slow_signal + fast_signal) / 2
        - cycle_bull: 1 if Bull state, else 0
        - cycle_correction: 1 if Correction state, else 0
        - cycle_bear: 1 if Bear state, else 0
        
    Args:
        df: DataFrame with columns ['kdcode', 'dt', 'close', ...]
        fast_window: Window for fast momentum (default 21 days)
        slow_window: Window for slow momentum (default 252 days)
        
    Returns:
        DataFrame with momentum features added
    """
    print("Computing binary momentum features...")
    df = df.sort_values(['kdcode', 'dt']).copy()
    
    # Calculate daily returns per stock
    df['_daily_return'] = df.groupby('kdcode')['close'].pct_change()
    
    # Fast momentum: trailing return
    df['fast_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=fast_window, min_periods=fast_window).sum()
    )
    
    # Slow momentum: trailing return
    df['slow_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=slow_window, min_periods=slow_window).sum()
    )
    
    if include_weekly_momentum:
        # Weekly momentum: 5-day trailing return (last week's return)
        df['weekly_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
            lambda x: x.rolling(window=5, min_periods=5).sum()
        )
    
    # Binary signals: +1 for positive/zero, -1 for negative
    df['slow_signal'] = np.where(df['slow_momentum'] >= 0, 1.0, -1.0)
    df['fast_signal'] = np.where(df['fast_momentum'] >= 0, 1.0, -1.0)
    if include_weekly_momentum:
        df['weekly_signal'] = np.where(df['weekly_momentum'] >= 0, 1.0, -1.0)
    
    # Handle NaN values (insufficient history) - fill with 0 (neutral)
    df['slow_momentum'] = df['slow_momentum'].fillna(0)
    df['fast_momentum'] = df['fast_momentum'].fillna(0)
    if include_weekly_momentum:
        df['weekly_momentum'] = df['weekly_momentum'].fillna(0)
    df['slow_signal'] = df['slow_signal'].fillna(0)
    df['fast_signal'] = df['fast_signal'].fillna(0)
    if include_weekly_momentum:
        df['weekly_signal'] = df['weekly_signal'].fillna(0)
    
    # MED: Intermediate-speed momentum (blend of slow and fast)
    df['momentum_blend'] = (df['slow_signal'] + df['fast_signal']) / 2
    
    # Market Cycle States (one-hot encoded)
    df['cycle_bull'] = ((df['slow_signal'] == 1) & (df['fast_signal'] == 1)).astype(float)
    df['cycle_correction'] = ((df['slow_signal'] == 1) & (df['fast_signal'] == -1)).astype(float)
    df['cycle_bear'] = ((df['slow_signal'] == -1) & (df['fast_signal'] == -1)).astype(float)
    
    # For rows with insufficient history, set all cycle indicators to 0
    neutral_mask = (df['slow_signal'] == 0) | (df['fast_signal'] == 0)
    df.loc[neutral_mask, ['cycle_bull', 'cycle_correction', 'cycle_bear']] = 0
    
    # Clean up intermediate column
    df = df.drop(columns=['_daily_return'])
    
    print(f"  Added momentum features: {get_momentum_features(include_weekly_momentum)}")
    print(f"  Rows with valid slow momentum: {(df['slow_momentum'] != 0).sum()} / {len(df)}")
    
    return df


def add_momentum_continuous(df: pd.DataFrame,
                            fast_window: int = 21,
                            slow_window: int = 252,
                            include_weekly_momentum: bool = True) -> pd.DataFrame:
    """
    Add continuous momentum features (raw values, not binary signals).
    
    This variant keeps the actual momentum values instead of converting
    to binary +1/-1 signals, allowing the model to learn from momentum magnitude.
    
    Features added:
        - slow_momentum: 252-day cumulative return (raw)
        - fast_momentum: 21-day cumulative return (raw)
        - slow_signal: Normalized slow momentum (z-score per day)
        - fast_signal: Normalized fast momentum (z-score per day)
        - momentum_blend: Average of normalized signals
        - cycle_bull, cycle_correction, cycle_bear: Same as binary
        
    Args:
        df: DataFrame with columns ['kdcode', 'dt', 'close', ...]
        fast_window: Window for fast momentum
        slow_window: Window for slow momentum
        
    Returns:
        DataFrame with continuous momentum features
    """
    print("Computing continuous momentum features...")
    df = df.sort_values(['kdcode', 'dt']).copy()
    
    # Calculate daily returns per stock
    df['_daily_return'] = df.groupby('kdcode')['close'].pct_change()
    
    # Raw momentum values
    df['fast_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=fast_window, min_periods=fast_window).sum()
    )
    df['slow_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=slow_window, min_periods=slow_window).sum()
    )
    if include_weekly_momentum:
        df['weekly_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
            lambda x: x.rolling(window=5, min_periods=5).sum()
        )
    
    # Fill NaN with 0
    df['slow_momentum'] = df['slow_momentum'].fillna(0)
    df['fast_momentum'] = df['fast_momentum'].fillna(0)
    if include_weekly_momentum:
        df['weekly_momentum'] = df['weekly_momentum'].fillna(0)
    
    # Normalize signals per day (cross-sectional z-score)
    # This captures relative momentum vs other stocks on the same day
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
    
    # Clip to reasonable range
    df['slow_signal'] = df['slow_signal'].clip(-3, 3)
    df['fast_signal'] = df['fast_signal'].clip(-3, 3)
    if include_weekly_momentum:
        df['weekly_signal'] = df['weekly_signal'].clip(-3, 3)
    
    # Fill any remaining NaN
    df['slow_signal'] = df['slow_signal'].fillna(0)
    df['fast_signal'] = df['fast_signal'].fillna(0)
    if include_weekly_momentum:
        df['weekly_signal'] = df['weekly_signal'].fillna(0)
    
    # Blend
    df['momentum_blend'] = (df['slow_signal'] + df['fast_signal']) / 2
    
    # Cycle indicators (still binary based on sign)
    slow_pos = df['slow_momentum'] >= 0
    fast_pos = df['fast_momentum'] >= 0
    df['cycle_bull'] = (slow_pos & fast_pos).astype(float)
    df['cycle_correction'] = (slow_pos & ~fast_pos).astype(float)
    df['cycle_bear'] = (~slow_pos & ~fast_pos).astype(float)
    
    # Clean up
    df = df.drop(columns=['_daily_return'])
    
    print(f"  Added continuous momentum features")
    
    return df


def add_momentum_buffered(df: pd.DataFrame,
                          fast_window: int = 21,
                          slow_window: int = 252,
                          buffer_low: float = 0.1,
                          buffer_high: float = 0.9,
                          include_weekly_momentum: bool = True) -> pd.DataFrame:
    """
    Add buffered momentum features with no-trade zones.
    
    This variant introduces buffers at the extremes:
    - Weak signals (near zero): Set to 0 (no trade)
    - Moderate signals: Scaled linearly
    - Extreme signals: Clipped (potential mean reversion concern)
    
    The buffer_low and buffer_high are percentiles within each day's
    cross-sectional distribution.
    
    Features added:
        - slow_momentum, fast_momentum: Raw values
        - slow_signal, fast_signal: Buffered signals
        - momentum_blend: Average of buffered signals
        - cycle_*: Same as binary
        - trade_signal: -1, 0, or 1 indicating trade direction
        
    Args:
        df: DataFrame with columns ['kdcode', 'dt', 'close', ...]
        fast_window: Window for fast momentum
        slow_window: Window for slow momentum
        buffer_low: Percentile below which signal is 0 (default 0.1 = 10th percentile)
        buffer_high: Percentile above which signal is clipped (default 0.9 = 90th percentile)
        
    Returns:
        DataFrame with buffered momentum features
    """
    print(f"Computing buffered momentum features (buffer_low={buffer_low}, buffer_high={buffer_high})...")
    df = df.sort_values(['kdcode', 'dt']).copy()
    
    # Calculate daily returns per stock
    df['_daily_return'] = df.groupby('kdcode')['close'].pct_change()
    
    # Raw momentum values
    df['fast_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=fast_window, min_periods=fast_window).sum()
    )
    df['slow_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=slow_window, min_periods=slow_window).sum()
    )
    if include_weekly_momentum:
        df['weekly_momentum'] = df.groupby('kdcode')['_daily_return'].transform(
            lambda x: x.rolling(window=5, min_periods=5).sum()
        )
    
    # Fill NaN with 0
    df['slow_momentum'] = df['slow_momentum'].fillna(0)
    df['fast_momentum'] = df['fast_momentum'].fillna(0)
    if include_weekly_momentum:
        df['weekly_momentum'] = df['weekly_momentum'].fillna(0)
    
    def compute_buffered_signal(group, col):
        """Compute buffered signal for a day's cross-section."""
        values = group[col].values
        if len(values) == 0 or np.all(values == 0):
            return pd.Series(np.zeros(len(values)), index=group.index)
        
        # Compute percentile ranks
        ranks = pd.Series(values).rank(pct=True).values
        
        # Initialize signals
        signals = np.zeros(len(values))
        
        # Below buffer_low: no trade (weak negative)
        # Above buffer_high: clipped (strong positive)
        # In between: scaled
        
        for i, (val, rank) in enumerate(zip(values, ranks)):
            if rank < buffer_low:
                # Weak negative momentum - no trade
                signals[i] = 0
            elif rank > buffer_high:
                # Strong positive momentum - clipped
                signals[i] = 1.0
            else:
                # Scale linearly between buffer_low and buffer_high
                signals[i] = (rank - buffer_low) / (buffer_high - buffer_low) * 2 - 1
        
        return pd.Series(signals, index=group.index)
    
    # Apply buffered signal computation
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
    
    # Blend
    df['momentum_blend'] = (df['slow_signal'] + df['fast_signal']) / 2
    
    # Trade signal: simplified -1, 0, 1
    df['trade_signal'] = np.where(
        df['momentum_blend'].abs() < 0.2, 0,  # No trade zone
        np.sign(df['momentum_blend'])
    )
    
    # Cycle indicators (based on raw momentum sign)
    slow_pos = df['slow_momentum'] >= 0
    fast_pos = df['fast_momentum'] >= 0
    df['cycle_bull'] = (slow_pos & fast_pos).astype(float)
    df['cycle_correction'] = (slow_pos & ~fast_pos).astype(float)
    df['cycle_bear'] = (~slow_pos & ~fast_pos).astype(float)
    
    # Clean up
    df = df.drop(columns=['_daily_return'])
    
    print(f"  Added buffered momentum features")
    
    return df


# Alias for backwards compatibility
add_momentum_features = add_momentum_binary
