"""
Base feature engineering for MCI-GRU.

This module provides base OHLCV features and turnover calculation.
"""

import numpy as np
import pandas as pd
from typing import List


# Base feature columns
BASE_FEATURES = ['close', 'open', 'high', 'low', 'volume', 'turnover']


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add base features to the dataframe.
    
    Computes turnover (Close x Volume) if not already present.
    
    Args:
        df: DataFrame with columns ['kdcode', 'dt', 'close', 'open', 'high', 'low', 'volume']
        
    Returns:
        DataFrame with turnover column added
    """
    df = df.copy()
    
    # Add Turnover feature (Close x Volume) per paper specification
    if 'turnover' not in df.columns:
        df['turnover'] = df['close'] * df['volume']
        print("  Added turnover feature (close * volume)")
    
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived price features.
    
    Features added:
        - daily_range: (high - low) / close - normalized intraday range
        - body_ratio: |close - open| / (high - low) - candlestick body ratio
        - overnight_return: open / prev_close - 1 - gap/overnight return
        - intraday_return: close / open - 1 - within-day return
        
    Args:
        df: DataFrame with OHLC columns
        
    Returns:
        DataFrame with price features added
    """
    df = df.copy()
    df = df.sort_values(['kdcode', 'dt'])
    
    # Normalized daily range (intraday volatility proxy)
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    
    # Candlestick body ratio
    range_val = df['high'] - df['low']
    df['body_ratio'] = np.abs(df['close'] - df['open']) / (range_val + 1e-8)
    
    # Overnight return (gap)
    df['prev_close'] = df.groupby('kdcode')['close'].shift(1)
    df['overnight_return'] = df['open'] / df['prev_close'] - 1
    df['overnight_return'] = df['overnight_return'].fillna(0)
    
    # Intraday return
    df['intraday_return'] = df['close'] / df['open'] - 1
    df['intraday_return'] = df['intraday_return'].fillna(0)
    
    # Clean up
    df = df.drop(columns=['prev_close'], errors='ignore')
    
    print("  Added price features: daily_range, body_ratio, overnight_return, intraday_return")
    
    return df


def add_volume_features(df: pd.DataFrame, ma_window: int = 20) -> pd.DataFrame:
    """
    Add volume-based features.
    
    Features added:
        - volume_ma{window}: Moving average of volume
        - volume_ratio: volume / volume_ma - unusual activity indicator
        - dollar_volume: close * volume - dollar trading volume
        
    Args:
        df: DataFrame with volume and close columns
        ma_window: Window for moving average (default 20 days)
        
    Returns:
        DataFrame with volume features added
    """
    df = df.copy()
    df = df.sort_values(['kdcode', 'dt'])
    
    # Volume moving average
    ma_col = f'volume_ma{ma_window}'
    df[ma_col] = df.groupby('kdcode')['volume'].transform(
        lambda x: x.rolling(window=ma_window, min_periods=1).mean()
    )
    
    # Volume ratio (>1 = unusual activity)
    df['volume_ratio'] = df['volume'] / (df[ma_col] + 1e-8)
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
    
    # Dollar volume
    df['dollar_volume'] = df['close'] * df['volume']
    
    print(f"  Added volume features: volume_ma{ma_window}, volume_ratio, dollar_volume")
    
    return df
