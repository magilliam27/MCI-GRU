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
    get_momentum_features,
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
from mci_gru.features.credit import (
    CREDIT_FEATURES,
    add_credit_features,
)
from mci_gru.features.regime import (
    REGIME_FEATURES,
    add_regime_features,
)


# Pre-defined feature sets
FEATURE_SETS = {
    'base': BASE_FEATURES,
    'momentum': MOMENTUM_FEATURES,
    'volatility': VOLATILITY_FEATURES,
    'vix': VIX_FEATURES,
    'credit': CREDIT_FEATURES,
    'regime': REGIME_FEATURES,
    'base_momentum': BASE_FEATURES + MOMENTUM_FEATURES,
    'full': BASE_FEATURES + MOMENTUM_FEATURES + VOLATILITY_FEATURES + VIX_FEATURES + CREDIT_FEATURES + REGIME_FEATURES,
}


def build_feature_list(
    include_base: bool = True,
    include_momentum: bool = True,
    include_weekly_momentum: bool = True,
    include_volatility: bool = False,
    include_vix: bool = False,
    include_credit_spread: bool = False,
    include_global_regime: bool = False,
    additional_features: Optional[List[str]] = None
) -> List[str]:
    """
    Build feature column list based on configuration flags.

    Args:
        include_base: Include base OHLCV features
        include_momentum: Include momentum features
        include_weekly_momentum: Include weekly momentum features (5-day return/signal)
        include_volatility: Include volatility features
        include_vix: Include VIX features
        include_credit_spread: Include credit spread features (IG/HY from FRED)
        include_global_regime: Include global scalar regime features
        additional_features: Additional feature column names to include

    Returns:
        List of feature column names
    """
    features = []

    if include_base:
        features.extend(BASE_FEATURES)
    if include_momentum:
        features.extend(get_momentum_features(include_weekly_momentum))
    if include_volatility:
        features.extend(VOLATILITY_FEATURES)
    if include_vix:
        features.extend(VIX_FEATURES)
    if include_credit_spread:
        features.extend(CREDIT_FEATURES)
    if include_global_regime:
        features.extend(REGIME_FEATURES)
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
        include_weekly_momentum: bool = True,
        momentum_encoding: str = 'binary',  # 'binary', 'continuous', 'buffered'
        momentum_blend_mode: str = 'static',  # 'static', 'dynamic'
        momentum_blend_fast_weight: float = 0.5,
        momentum_dynamic_correction_fast_weight: float = 0.15,
        momentum_dynamic_rebound_fast_weight: float = 0.70,
        momentum_buffer_low: float = 0.1,
        momentum_buffer_high: float = 0.9,
        include_volatility: bool = False,
        include_vix: bool = False,
        include_credit_spread: bool = False,
        include_global_regime: bool = False,
        regime_change_months: int = 12,
        regime_norm_months: int = 120,
        regime_clip_z: float = 3.0,
        regime_exclusion_months: int = 1,
        regime_similarity_quantile: float = 0.2,
        regime_min_history_months: int = 24,
        regime_strict: bool = False,
        include_rsi: bool = False,
        include_ma_features: bool = False,
        include_price_features: bool = False,
        include_volume_features: bool = False,
    ):
        """
        Initialize feature engineer.

        Args:
            include_momentum: Whether to add momentum features
            include_weekly_momentum: Whether to include weekly momentum terms (5-day return/signal)
            momentum_encoding: Type of momentum encoding
            momentum_blend_mode: Whether momentum_blend uses a fixed or cycle-aware FAST allocation
            momentum_blend_fast_weight: FAST allocation for static blending and agreement states
            momentum_dynamic_correction_fast_weight: FAST allocation after Correction states
            momentum_dynamic_rebound_fast_weight: FAST allocation after Rebound states
            momentum_buffer_low: Low buffer for buffered momentum
            momentum_buffer_high: High buffer for buffered momentum
            include_volatility: Whether to add volatility features
            include_vix: Whether to add VIX features (requires vix_df in transform)
            include_credit_spread: Whether to add credit spread features (requires credit_df in transform)
            include_global_regime: Whether to add global scalar regime features (requires regime_df in transform)
            include_rsi: Whether to add RSI features
            include_ma_features: Whether to add moving average features
            include_price_features: Whether to add derived price features
            include_volume_features: Whether to add volume features
        """
        self.include_momentum = include_momentum
        self.include_weekly_momentum = include_weekly_momentum
        self.momentum_encoding = momentum_encoding
        self.momentum_blend_mode = momentum_blend_mode
        self.momentum_blend_fast_weight = momentum_blend_fast_weight
        self.momentum_dynamic_correction_fast_weight = momentum_dynamic_correction_fast_weight
        self.momentum_dynamic_rebound_fast_weight = momentum_dynamic_rebound_fast_weight
        self.momentum_buffer_low = momentum_buffer_low
        self.momentum_buffer_high = momentum_buffer_high
        self.include_volatility = include_volatility
        self.include_vix = include_vix
        self.include_credit_spread = include_credit_spread
        self.include_global_regime = include_global_regime
        self.regime_change_months = regime_change_months
        self.regime_norm_months = regime_norm_months
        self.regime_clip_z = regime_clip_z
        self.regime_exclusion_months = regime_exclusion_months
        self.regime_similarity_quantile = regime_similarity_quantile
        self.regime_min_history_months = regime_min_history_months
        self.regime_strict = regime_strict
        self.include_rsi = include_rsi
        self.include_ma_features = include_ma_features
        self.include_price_features = include_price_features
        self.include_volume_features = include_volume_features
    
    def transform(
        self,
        df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
        credit_df: Optional[pd.DataFrame] = None,
        regime_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Apply feature transformations to dataframe.

        Args:
            df: Input dataframe with OHLCV columns
            vix_df: Optional VIX data for VIX features
            credit_df: Optional credit spread data for credit features (from FRED)
            regime_df: Optional regime input data for global scalar regime features

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
                df = add_momentum_binary(
                    df,
                    blend_mode=self.momentum_blend_mode,
                    blend_fast_weight=self.momentum_blend_fast_weight,
                    dynamic_correction_fast_weight=self.momentum_dynamic_correction_fast_weight,
                    dynamic_rebound_fast_weight=self.momentum_dynamic_rebound_fast_weight,
                    include_weekly_momentum=self.include_weekly_momentum,
                )
            elif self.momentum_encoding == 'continuous':
                df = add_momentum_continuous(
                    df,
                    blend_mode=self.momentum_blend_mode,
                    blend_fast_weight=self.momentum_blend_fast_weight,
                    dynamic_correction_fast_weight=self.momentum_dynamic_correction_fast_weight,
                    dynamic_rebound_fast_weight=self.momentum_dynamic_rebound_fast_weight,
                    include_weekly_momentum=self.include_weekly_momentum,
                )
            elif self.momentum_encoding == 'buffered':
                df = add_momentum_buffered(
                    df,
                    blend_mode=self.momentum_blend_mode,
                    blend_fast_weight=self.momentum_blend_fast_weight,
                    dynamic_correction_fast_weight=self.momentum_dynamic_correction_fast_weight,
                    dynamic_rebound_fast_weight=self.momentum_dynamic_rebound_fast_weight,
                    buffer_low=self.momentum_buffer_low,
                    buffer_high=self.momentum_buffer_high,
                    include_weekly_momentum=self.include_weekly_momentum,
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

        # Credit spread features (skipped if credit_df is None, e.g. FRED load failed)
        if self.include_credit_spread:
            if credit_df is not None:
                df = add_credit_features(df, credit_df)
            else:
                # Soft-fail: add zero columns so feature_cols and downstream pipeline stay consistent
                for col in CREDIT_FEATURES:
                    df[col] = 0.0

        # Global scalar regime features
        if self.include_global_regime:
            if regime_df is not None:
                df = add_regime_features(
                    df=df,
                    regime_df=regime_df,
                    change_months=self.regime_change_months,
                    norm_window_months=self.regime_norm_months,
                    clip_z=self.regime_clip_z,
                    exclusion_months=self.regime_exclusion_months,
                    similarity_quantile=self.regime_similarity_quantile,
                    min_history_months=self.regime_min_history_months,
                )
            elif self.regime_strict:
                raise ValueError("include_global_regime=True but regime_df not provided and regime_strict=True")
            else:
                for col in REGIME_FEATURES:
                    df[col] = 0.0

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
            features.extend(get_momentum_features(self.include_weekly_momentum))
            if self.momentum_encoding == 'buffered':
                features.append('trade_signal')
        
        if self.include_volatility:
            features.extend(VOLATILITY_FEATURES)
        
        if self.include_vix:
            features.extend(VIX_FEATURES)

        if self.include_credit_spread:
            features.extend(CREDIT_FEATURES)
        
        if self.include_global_regime:
            features.extend(REGIME_FEATURES)

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
        include_weekly_momentum=config.get('include_weekly_momentum', True),
        momentum_encoding=config.get('momentum_encoding', 'binary'),
        momentum_blend_mode=config.get('momentum_blend_mode', 'static'),
        momentum_blend_fast_weight=config.get('momentum_blend_fast_weight', 0.5),
        momentum_dynamic_correction_fast_weight=config.get(
            'momentum_dynamic_correction_fast_weight', 0.15
        ),
        momentum_dynamic_rebound_fast_weight=config.get(
            'momentum_dynamic_rebound_fast_weight', 0.70
        ),
        momentum_buffer_low=config.get('momentum_buffer_low', 0.1),
        momentum_buffer_high=config.get('momentum_buffer_high', 0.9),
        include_volatility=config.get('include_volatility', False),
        include_vix=config.get('include_vix', False),
        include_credit_spread=config.get('include_credit_spread', False),
        include_global_regime=config.get('include_global_regime', False),
        regime_change_months=config.get('regime_change_months', 12),
        regime_norm_months=config.get('regime_norm_months', 120),
        regime_clip_z=config.get('regime_clip_z', 3.0),
        regime_exclusion_months=config.get('regime_exclusion_months', 1),
        regime_similarity_quantile=config.get('regime_similarity_quantile', 0.2),
        regime_min_history_months=config.get('regime_min_history_months', 24),
        regime_strict=config.get('regime_strict', False),
        include_rsi=config.get('include_rsi', False),
        include_ma_features=config.get('include_ma_features', False),
        include_price_features=config.get('include_price_features', False),
        include_volume_features=config.get('include_volume_features', False),
    )
