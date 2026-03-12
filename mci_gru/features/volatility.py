"""Volatility features: realized volatility, vol ratio, VIX, RSI, moving averages."""

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
    print("Computing volatility features...")
    df = df.sort_values(['kdcode', 'dt']).copy()
    
    if '_daily_return' not in df.columns:
        df['_daily_return'] = df.groupby('kdcode')['close'].pct_change()
    
    short_col = f'volatility_{short_window}d'
    df[short_col] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=short_window, min_periods=short_window).std() * annualization_factor
    )
    
    long_col = f'volatility_{long_window}d'
    df[long_col] = df.groupby('kdcode')['_daily_return'].transform(
        lambda x: x.rolling(window=long_window, min_periods=long_window).std() * annualization_factor
    )
    
    df[short_col] = df.groupby('kdcode')[short_col].transform(
        lambda x: x.fillna(x.median() if not x.isna().all() else 0.2)
    )
    df[long_col] = df.groupby('kdcode')[long_col].transform(
        lambda x: x.fillna(x.median() if not x.isna().all() else 0.2)
    )
    
    df['vol_ratio'] = df[short_col] / (df[long_col] + 1e-8)
    df['vol_ratio'] = df['vol_ratio'].clip(0.1, 10)
    
    print(f"  Added volatility features: {short_col}, {long_col}, vol_ratio")
    
    return df


def add_vix_features(df: pd.DataFrame, 
                     vix_df: pd.DataFrame,
                     vix_ma_window: int = 10) -> pd.DataFrame:
    print("Merging VIX features...")
    df = df.copy()
    vix = vix_df.copy()
    
    if 'vix' not in vix.columns:
        if 'close' in vix.columns:
            vix = vix.rename(columns={'close': 'vix'})
        elif 'CLOSE' in vix.columns:
            vix = vix.rename(columns={'CLOSE': 'vix'})
    
    if 'dt' not in vix.columns:
        if vix.index.name == 'dt' or vix.index.name == 'Date':
            vix = vix.reset_index()
            vix = vix.rename(columns={vix.columns[0]: 'dt'})
    
    vix['dt'] = pd.to_datetime(vix['dt']).dt.strftime('%Y-%m-%d')
    vix = vix.sort_values('dt')
    vix['vix_change'] = vix['vix'].pct_change().fillna(0)
    vix['vix_ma'] = vix['vix'].rolling(window=vix_ma_window, min_periods=1).mean()
    vix['vix_regime'] = (vix['vix'] > vix['vix_ma']).astype(float)
    
    vix_cols = ['dt', 'vix', 'vix_change', 'vix_regime']
    vix_merge = vix[vix_cols]
    df['dt'] = pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d')
    df = df.merge(vix_merge, on='dt', how='left')
    df['vix'] = df['vix'].fillna(method='ffill').fillna(20)
    df['vix_change'] = df['vix_change'].fillna(0)
    df['vix_regime'] = df['vix_regime'].fillna(0)
    
    print(f"  Added VIX features: vix, vix_change, vix_regime")
    print(f"  VIX range: {df['vix'].min():.1f} to {df['vix'].max():.1f}")
    
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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
    df['rsi_normalized'] = (df[rsi_col] - 50) / 50
    df[rsi_col] = df[rsi_col].fillna(50)
    df['rsi_normalized'] = df['rsi_normalized'].fillna(0)
    
    print(f"  Added RSI features: {rsi_col}, rsi_normalized")
    
    return df


def add_moving_average_features(df: pd.DataFrame,
                                 short_window: int = 50,
                                 long_window: int = 200) -> pd.DataFrame:
    print(f"Computing moving average features (MA{short_window}, MA{long_window})...")
    df = df.sort_values(['kdcode', 'dt']).copy()
    
    ma_short = f'ma_{short_window}'
    ma_long = f'ma_{long_window}'
    
    df[ma_short] = df.groupby('kdcode')['close'].transform(
        lambda x: x.rolling(window=short_window, min_periods=1).mean()
    )
    df[ma_long] = df.groupby('kdcode')['close'].transform(
        lambda x: x.rolling(window=long_window, min_periods=1).mean()
    )
    
    df[f'dist_ma{short_window}'] = (df['close'] - df[ma_short]) / df[ma_short]
    df[f'dist_ma{long_window}'] = (df['close'] - df[ma_long]) / df[ma_long]
    df['ma_cross'] = (df[ma_short] > df[ma_long]).astype(float)
    df = df.drop(columns=[ma_short, ma_long])
    
    print(f"  Added MA features: dist_ma{short_window}, dist_ma{long_window}, ma_cross")
    
    return df
