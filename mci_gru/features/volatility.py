"""
Volatility feature engineering for MCI-GRU.

This module provides volatility-related features including:
- Realized volatility (historical)
- Volatility ratio (short-term vs long-term)
- VIX integration (market-level fear gauge)
"""

import numpy as np
import pandas as pd
from typing import Optional


# Volatility feature columns
VOLATILITY_FEATURES = ['volatility_5d', 'volatility_21d', 'vol_ratio']
VIX_FEATURES = ['vix', 'vix_change', 'vix_regime']


def add_volatility_features(df: pd.DataFrame,
                            short_window: int = 5,
                            long_window: int = 21,
                            annualization_factor: float = np.sqrt(252)) -> pd.DataFrame:
    """
    Add realized volatility features.
    
    Features added:
        - volatility_5d: 5-day annualized volatility
        - volatility_21d: 21-day annualized volatility
        - vol_ratio: volatility_5d / volatility_21d (increasing = rising uncertainty)
        
    Args:
        df: DataFrame with columns ['kdcode', 'dt', 'close', ...]
        short_window: Window for short-term volatility (default 5 days)
        long_window: Window for long-term volatility (default 21 days)
        annualization_factor: Factor to annualize volatility (default sqrt(252))
        
    Returns:
        DataFrame with volatility features added
    """
    print("Computing volatility features...")
    df = df.sort_values(['kdcode', 'dt']).copy()
    
    # Calculate daily returns if not present
    if '_daily_return' not in df.columns:
        df['_daily_return'] = df.groupby('kdcode')['close'].pct_change()
    
    # Short-term volatility (5-day, annualized)
    short_col = f'volatility_{short_window}d'
    df[short_col] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=short_window, min_periods=short_window).std() * annualization_factor
    )
    
    # Long-term volatility (21-day, annualized)
    long_col = f'volatility_{long_window}d'
    df[long_col] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=long_window, min_periods=long_window).std() * annualization_factor
    )
    
    # Fill NaN with median volatility per stock
    df[short_col] = df.groupby('kdcode')[short_col].transform(
        lambda x: x.fillna(x.median() if not x.isna().all() else 0.2)
    )
    df[long_col] = df.groupby('kdcode')[long_col].transform(
        lambda x: x.fillna(x.median() if not x.isna().all() else 0.2)
    )
    
    # Volatility ratio (short / long)
    # Values > 1 indicate rising volatility
    df['vol_ratio'] = df[short_col] / (df[long_col] + 1e-8)
    df['vol_ratio'] = df['vol_ratio'].clip(0.1, 10)  # Reasonable bounds
    
    # Clean up temporary column if we created it
    if '_daily_return' in df.columns:
        # Check if other functions might need it
        pass  # Keep it for now
    
    print(f"  Added volatility features: {short_col}, {long_col}, vol_ratio")
    
    return df


def add_vix_features(df: pd.DataFrame, 
                     vix_df: pd.DataFrame,
                     vix_ma_window: int = 10) -> pd.DataFrame:
    """
    Merge VIX data and add derived features.
    
    VIX is a market-level indicator, so it's replicated for all stocks on each date.
    
    Features added:
        - vix: VIX index level
        - vix_change: Daily percentage change in VIX
        - vix_regime: 1 if VIX > MA, else 0 (high/low vol regime)
        
    Args:
        df: DataFrame with stock data (must have 'dt' column)
        vix_df: DataFrame with VIX data (index or 'dt' column, 'close' or 'vix' column)
        vix_ma_window: Window for VIX moving average (default 10 days)
        
    Returns:
        DataFrame with VIX features merged
    """
    print("Merging VIX features...")
    df = df.copy()
    
    # Prepare VIX data
    vix = vix_df.copy()
    
    # Handle different column names
    if 'vix' not in vix.columns:
        if 'close' in vix.columns:
            vix = vix.rename(columns={'close': 'vix'})
        elif 'CLOSE' in vix.columns:
            vix = vix.rename(columns={'CLOSE': 'vix'})
    
    # Handle index vs column for date
    if 'dt' not in vix.columns:
        if vix.index.name == 'dt' or vix.index.name == 'Date':
            vix = vix.reset_index()
            vix = vix.rename(columns={vix.columns[0]: 'dt'})
    
    # Ensure dt is string for merging
    vix['dt'] = pd.to_datetime(vix['dt']).dt.strftime('%Y-%m-%d')
    
    # Compute VIX features
    vix = vix.sort_values('dt')
    vix['vix_change'] = vix['vix'].pct_change().fillna(0)
    vix['vix_ma'] = vix['vix'].rolling(window=vix_ma_window, min_periods=1).mean()
    vix['vix_regime'] = (vix['vix'] > vix['vix_ma']).astype(float)
    
    # Select columns to merge
    vix_cols = ['dt', 'vix', 'vix_change', 'vix_regime']
    vix_merge = vix[vix_cols]
    
    # Merge with stock data (VIX is same for all stocks on each date)
    df['dt'] = pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d')
    df = df.merge(vix_merge, on='dt', how='left')
    
    # Fill missing VIX values (for dates not in VIX data)
    df['vix'] = df['vix'].fillna(method='ffill').fillna(20)  # Forward fill, default 20
    df['vix_change'] = df['vix_change'].fillna(0)
    df['vix_regime'] = df['vix_regime'].fillna(0)
    
    print(f"  Added VIX features: vix, vix_change, vix_regime")
    print(f"  VIX range: {df['vix'].min():.1f} to {df['vix'].max():.1f}")
    
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) feature.
    
    RSI is a mean-reversion indicator that complements momentum signals.
    
    Features added:
        - rsi_{period}: RSI value (0-100)
        - rsi_normalized: RSI normalized to [-1, 1] range
        
    Args:
        df: DataFrame with columns ['kdcode', 'dt', 'close', ...]
        period: RSI calculation period (default 14 days)
        
    Returns:
        DataFrame with RSI features added
    """
    print(f"Computing RSI ({period}-day)...")
    df = df.sort_values(['kdcode', 'dt']).copy()
    
    def compute_rsi(series):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    rsi_col = f'rsi_{period}'
    df[rsi_col] = df.groupby('kdcode')['close'].transform(compute_rsi)
    
    # Normalized RSI (-1 to 1)
    df['rsi_normalized'] = (df[rsi_col] - 50) / 50
    
    # Fill NaN
    df[rsi_col] = df[rsi_col].fillna(50)
    df['rsi_normalized'] = df['rsi_normalized'].fillna(0)
    
    print(f"  Added RSI features: {rsi_col}, rsi_normalized")
    
    return df


def add_moving_average_features(df: pd.DataFrame,
                                 short_window: int = 50,
                                 long_window: int = 200) -> pd.DataFrame:
    """
    Add moving average distance features.
    
    Features added:
        - dist_ma{short}: % distance from short MA
        - dist_ma{long}: % distance from long MA
        - ma_cross: 1 if short MA > long MA, else 0 (golden/death cross)
        
    Args:
        df: DataFrame with columns ['kdcode', 'dt', 'close', ...]
        short_window: Short moving average window (default 50)
        long_window: Long moving average window (default 200)
        
    Returns:
        DataFrame with MA features added
    """
    print(f"Computing moving average features (MA{short_window}, MA{long_window})...")
    df = df.sort_values(['kdcode', 'dt']).copy()
    
    # Compute moving averages
    ma_short = f'ma_{short_window}'
    ma_long = f'ma_{long_window}'
    
    df[ma_short] = df.groupby('kdcode')['close'].transform(
        lambda x: x.rolling(window=short_window, min_periods=1).mean()
    )
    df[ma_long] = df.groupby('kdcode')['close'].transform(
        lambda x: x.rolling(window=long_window, min_periods=1).mean()
    )
    
    # Distance from MAs (percentage)
    df[f'dist_ma{short_window}'] = (df['close'] - df[ma_short]) / df[ma_short]
    df[f'dist_ma{long_window}'] = (df['close'] - df[ma_long]) / df[ma_long]
    
    # MA cross indicator
    df['ma_cross'] = (df[ma_short] > df[ma_long]).astype(float)
    
    # Clean up intermediate columns
    df = df.drop(columns=[ma_short, ma_long])
    
    print(f"  Added MA features: dist_ma{short_window}, dist_ma{long_window}, ma_cross")
    
    return df
