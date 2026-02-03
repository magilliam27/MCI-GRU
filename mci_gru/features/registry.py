"""
Feature registry and composition for MCI-GRU.

This module provides:
- FEATURE_SETS: Pre-defined feature set configurations
- FeatureEngineer: Class for applying feature transformations
- build_feature_list: Helper to build feature column list from config
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from mci_gru.features.base import (
    BASE_FEATURES,
    add_base_features,
    add_price_features,
    add_volume_features,
)
from mci_gru.features.momentum import (
    MOMENTUM_FEATURES,
    add_momentum_binary,
    add_momentum_continuous,
    add_momentum_buffered,
)
from mci_gru.features.volatility import (
    VOLATILITY_FEATURES,
    VIX_FEATURES,
    add_volatility_features,
    add_vix_features,
    add_rsi,
    add_moving_average_features,
)


# Pre-defined feature sets
FEATURE_SETS = {
    'base': BASE_FEATURES,
    'momentum': MOMENTUM_FEATURES,
    'volatility': VOLATILITY_FEATURES,
    'vix': VIX_FEATURES,
    'base_momentum': BASE_FEATURES + MOMENTUM_FEATURES,
    'full': BASE_FEATURES + MOMENTUM_FEATURES + VOLATILITY_FEATURES + VIX_FEATURES,
}


def build_feature_list(
    include_base: bool = True,
    include_momentum: bool = True,
    include_volatility: bool = False,
    include_vix: bool = False,
    additional_features: Optional[List[str]] = None
) -> List[str]:
    """
    Build feature column list based on configuration flags.
    
    Args:
        include_base: Include base OHLCV features
        include_momentum: Include momentum features
        include_volatility: Include volatility features
        include_vix: Include VIX features
        additional_features: Additional feature column names to include
        
    Returns:
        List of feature column names
    """
    features = []
    
    if include_base:
        features.extend(BASE_FEATURES)
    if include_momentum:
        features.extend(MOMENTUM_FEATURES)
    if include_volatility:
        features.extend(VOLATILITY_FEATURES)
    if include_vix:
        features.extend(VIX_FEATURES)
    if additional_features:
        features.extend(additional_features)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in features:
        if f not in seen:
            seen.add(f)
            unique_features.append(f)
    
    return unique_features


class FeatureEngineer:
    """
    Feature engineering pipeline for MCI-GRU.
    
    Applies feature transformations based on configuration.
    """
    
    def __init__(
        self,
        include_momentum: bool = True,
        momentum_encoding: str = 'binary',  # 'binary', 'continuous', 'buffered'
        momentum_buffer_low: float = 0.1,
        momentum_buffer_high: float = 0.9,
        include_volatility: bool = False,
        include_vix: bool = False,
        include_rsi: bool = False,
        include_ma_features: bool = False,
        include_price_features: bool = False,
        include_volume_features: bool = False,
    ):
        """
        Initialize feature engineer.
        
        Args:
            include_momentum: Whether to add momentum features
            momentum_encoding: Type of momentum encoding
            momentum_buffer_low: Low buffer for buffered momentum
            momentum_buffer_high: High buffer for buffered momentum
            include_volatility: Whether to add volatility features
            include_vix: Whether to add VIX features (requires vix_df in transform)
            include_rsi: Whether to add RSI features
            include_ma_features: Whether to add moving average features
            include_price_features: Whether to add derived price features
            include_volume_features: Whether to add volume features
        """
        self.include_momentum = include_momentum
        self.momentum_encoding = momentum_encoding
        self.momentum_buffer_low = momentum_buffer_low
        self.momentum_buffer_high = momentum_buffer_high
        self.include_volatility = include_volatility
        self.include_vix = include_vix
        self.include_rsi = include_rsi
        self.include_ma_features = include_ma_features
        self.include_price_features = include_price_features
        self.include_volume_features = include_volume_features
    
    def transform(self, df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply feature transformations to dataframe.
        
        Args:
            df: Input dataframe with OHLCV columns
            vix_df: Optional VIX data for VIX features
            
        Returns:
            DataFrame with features added
        """
        print("=" * 60)
        print("Feature Engineering Pipeline")
        print("=" * 60)
        
        # Always add base features (turnover)
        df = add_base_features(df)
        
        # Momentum features
        if self.include_momentum:
            if self.momentum_encoding == 'binary':
                df = add_momentum_binary(df)
            elif self.momentum_encoding == 'continuous':
                df = add_momentum_continuous(df)
            elif self.momentum_encoding == 'buffered':
                df = add_momentum_buffered(
                    df,
                    buffer_low=self.momentum_buffer_low,
                    buffer_high=self.momentum_buffer_high
                )
            else:
                raise ValueError(f"Unknown momentum encoding: {self.momentum_encoding}")
        
        # Volatility features
        if self.include_volatility:
            df = add_volatility_features(df)
        
        # VIX features
        if self.include_vix:
            if vix_df is None:
                raise ValueError("include_vix=True but vix_df not provided")
            df = add_vix_features(df, vix_df)
        
        # RSI
        if self.include_rsi:
            df = add_rsi(df)
        
        # Moving average features
        if self.include_ma_features:
            df = add_moving_average_features(df)
        
        # Price features
        if self.include_price_features:
            df = add_price_features(df)
        
        # Volume features
        if self.include_volume_features:
            df = add_volume_features(df)
        
        print("=" * 60)
        print("Feature engineering complete")
        print("=" * 60)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns that will be created.
        
        Returns:
            List of feature column names
        """
        features = list(BASE_FEATURES)
        
        if self.include_momentum:
            features.extend(MOMENTUM_FEATURES)
            if self.momentum_encoding == 'buffered':
                features.append('trade_signal')
        
        if self.include_volatility:
            features.extend(VOLATILITY_FEATURES)
        
        if self.include_vix:
            features.extend(VIX_FEATURES)
        
        if self.include_rsi:
            features.extend(['rsi_14', 'rsi_normalized'])
        
        if self.include_ma_features:
            features.extend(['dist_ma50', 'dist_ma200', 'ma_cross'])
        
        if self.include_price_features:
            features.extend(['daily_range', 'body_ratio', 'overnight_return', 'intraday_return'])
        
        if self.include_volume_features:
            features.extend(['volume_ma20', 'volume_ratio', 'dollar_volume'])
        
        return features


# Factory function for creating feature engineer from config
def create_feature_engineer_from_config(config: Dict[str, Any]) -> FeatureEngineer:
    """
    Create FeatureEngineer from configuration dictionary.
    
    Args:
        config: Configuration dict with feature settings
        
    Returns:
        Configured FeatureEngineer instance
    """
    return FeatureEngineer(
        include_momentum=config.get('include_momentum', True),
        momentum_encoding=config.get('momentum_encoding', 'binary'),
        momentum_buffer_low=config.get('momentum_buffer_low', 0.1),
        momentum_buffer_high=config.get('momentum_buffer_high', 0.9),
        include_volatility=config.get('include_volatility', False),
        include_vix=config.get('include_vix', False),
        include_rsi=config.get('include_rsi', False),
        include_ma_features=config.get('include_ma_features', False),
        include_price_features=config.get('include_price_features', False),
        include_volume_features=config.get('include_volume_features', False),
    )
