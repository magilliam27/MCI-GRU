"""Base OHLCV features and turnover."""

import numpy as np
import pandas as pd
from typing import List


# Base feature columns
BASE_FEATURES = ['close', 'open', 'high', 'low', 'volume', 'turnover']


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Turnover = Close x Volume per paper specification
    if 'turnover' not in df.columns:
        df['turnover'] = df['close'] * df['volume']
        print("  Added turnover feature (close * volume)")
    
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['kdcode', 'dt'])
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    range_val = df['high'] - df['low']
    df['body_ratio'] = np.abs(df['close'] - df['open']) / (range_val + 1e-8)
    df['prev_close'] = df.groupby('kdcode')['close'].shift(1)
    df['overnight_return'] = df['open'] / df['prev_close'] - 1
    df['overnight_return'] = df['overnight_return'].fillna(0)
    df['intraday_return'] = df['close'] / df['open'] - 1
    df['intraday_return'] = df['intraday_return'].fillna(0)
    df = df.drop(columns=['prev_close'], errors='ignore')
    
    print("  Added price features: daily_range, body_ratio, overnight_return, intraday_return")
    
    return df


def add_volume_features(df: pd.DataFrame, ma_window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['kdcode', 'dt'])
    
    ma_col = f'volume_ma{ma_window}'
    df[ma_col] = df.groupby('kdcode')['volume'].transform(
        lambda x: x.rolling(window=ma_window, min_periods=1).mean()
    )
    df['volume_ratio'] = df['volume'] / (df[ma_col] + 1e-8)
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
    df['dollar_volume'] = df['close'] * df['volume']
    
    print(f"  Added volume features: volume_ma{ma_window}, volume_ratio, dollar_volume")
    
    return df
