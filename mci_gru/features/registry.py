"""
Feature registry and composition for MCI-GRU.

Provides:
- FEATURE_SETS: Pre-defined feature set configurations
- FeatureEngineer: Pipeline that applies feature transformations
- build_feature_list: Helper to build feature column list from flags
"""

from __future__ import annotations

import pandas as pd
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from mci_gru.config import FeatureConfig

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
    get_regime_features,
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
    regime_include_subsequent_returns: bool = True,
    regime_subsequent_return_horizons: Optional[List[int]] = None,
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
        regime_include_subsequent_returns: Include similarity-conditioned subsequent-return features
        regime_subsequent_return_horizons: Monthly forward-return horizons for regime features
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
        features.extend(
            get_regime_features(
                include_subsequent_returns=regime_include_subsequent_returns,
                horizons=regime_subsequent_return_horizons,
            )
        )
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
    """Feature engineering pipeline for MCI-GRU.

    Preferred usage — pass a ``FeatureConfig`` dataclass::

        engineer = FeatureEngineer(config.features)

    Legacy usage with individual kwargs is still supported for backward
    compatibility but is discouraged for new code.
    """

    def __init__(
        self,
        config: "FeatureConfig | None" = None,
        *,
        # Legacy kwargs — used only when *config* is None.
        include_momentum: bool = True,
        include_weekly_momentum: bool = True,
        momentum_encoding: str = 'binary',
        momentum_blend_mode: str = 'static',
        momentum_blend_fast_weight: float = 0.5,
        momentum_dynamic_correction_fast_weight: float = 0.15,
        momentum_dynamic_rebound_fast_weight: float = 0.70,
        momentum_dynamic_lookback_periods: int = 0,
        momentum_dynamic_min_history: int = 252,
        momentum_dynamic_min_state_observations: int = 3,
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
        regime_include_subsequent_returns: bool = True,
        regime_subsequent_return_horizons: Optional[List[int]] = None,
        include_rsi: bool = False,
        include_ma_features: bool = False,
        include_price_features: bool = False,
        include_volume_features: bool = False,
    ):
        if config is not None:
            # Pull every attribute from the dataclass — single source of truth.
            self.include_momentum = config.include_momentum
            self.include_weekly_momentum = config.include_weekly_momentum
            self.momentum_encoding = config.momentum_encoding
            self.momentum_blend_mode = config.momentum_blend_mode
            self.momentum_blend_fast_weight = config.momentum_blend_fast_weight
            self.momentum_dynamic_correction_fast_weight = config.momentum_dynamic_correction_fast_weight
            self.momentum_dynamic_rebound_fast_weight = config.momentum_dynamic_rebound_fast_weight
            self.momentum_dynamic_lookback_periods = config.momentum_dynamic_lookback_periods
            self.momentum_dynamic_min_history = config.momentum_dynamic_min_history
            self.momentum_dynamic_min_state_observations = config.momentum_dynamic_min_state_observations
            self.momentum_buffer_low = config.momentum_buffer_low
            self.momentum_buffer_high = config.momentum_buffer_high
            self.include_volatility = config.include_volatility
            self.include_vix = config.include_vix
            self.include_credit_spread = config.include_credit_spread
            self.include_global_regime = config.include_global_regime
            self.regime_change_months = config.regime_change_months
            self.regime_norm_months = config.regime_norm_months
            self.regime_clip_z = config.regime_clip_z
            self.regime_exclusion_months = config.regime_exclusion_months
            self.regime_similarity_quantile = config.regime_similarity_quantile
            self.regime_min_history_months = config.regime_min_history_months
            self.regime_strict = config.regime_strict
            self.regime_include_subsequent_returns = config.regime_include_subsequent_returns
            self.regime_subsequent_return_horizons = list(config.regime_subsequent_return_horizons)
            self.include_rsi = config.include_rsi
            self.include_ma_features = config.include_ma_features
            self.include_price_features = config.include_price_features
            self.include_volume_features = config.include_volume_features
        else:
            # Legacy path: accept individual kwargs.
            self.include_momentum = include_momentum
            self.include_weekly_momentum = include_weekly_momentum
            self.momentum_encoding = momentum_encoding
            self.momentum_blend_mode = momentum_blend_mode
            self.momentum_blend_fast_weight = momentum_blend_fast_weight
            self.momentum_dynamic_correction_fast_weight = momentum_dynamic_correction_fast_weight
            self.momentum_dynamic_rebound_fast_weight = momentum_dynamic_rebound_fast_weight
            self.momentum_dynamic_lookback_periods = momentum_dynamic_lookback_periods
            self.momentum_dynamic_min_history = momentum_dynamic_min_history
            self.momentum_dynamic_min_state_observations = momentum_dynamic_min_state_observations
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
            self.regime_include_subsequent_returns = regime_include_subsequent_returns
            self.regime_subsequent_return_horizons = list(
                regime_subsequent_return_horizons if regime_subsequent_return_horizons is not None else [1, 3]
            )
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
                    dynamic_lookback_periods=self.momentum_dynamic_lookback_periods,
                    dynamic_min_history=self.momentum_dynamic_min_history,
                    dynamic_min_state_observations=self.momentum_dynamic_min_state_observations,
                    include_weekly_momentum=self.include_weekly_momentum,
                )
            elif self.momentum_encoding == 'continuous':
                df = add_momentum_continuous(
                    df,
                    blend_mode=self.momentum_blend_mode,
                    blend_fast_weight=self.momentum_blend_fast_weight,
                    dynamic_correction_fast_weight=self.momentum_dynamic_correction_fast_weight,
                    dynamic_rebound_fast_weight=self.momentum_dynamic_rebound_fast_weight,
                    dynamic_lookback_periods=self.momentum_dynamic_lookback_periods,
                    dynamic_min_history=self.momentum_dynamic_min_history,
                    dynamic_min_state_observations=self.momentum_dynamic_min_state_observations,
                    include_weekly_momentum=self.include_weekly_momentum,
                )
            elif self.momentum_encoding == 'buffered':
                df = add_momentum_buffered(
                    df,
                    blend_mode=self.momentum_blend_mode,
                    blend_fast_weight=self.momentum_blend_fast_weight,
                    dynamic_correction_fast_weight=self.momentum_dynamic_correction_fast_weight,
                    dynamic_rebound_fast_weight=self.momentum_dynamic_rebound_fast_weight,
                    dynamic_lookback_periods=self.momentum_dynamic_lookback_periods,
                    dynamic_min_history=self.momentum_dynamic_min_history,
                    dynamic_min_state_observations=self.momentum_dynamic_min_state_observations,
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
                    include_subsequent_returns=self.regime_include_subsequent_returns,
                    subsequent_return_horizons=self.regime_subsequent_return_horizons,
                )
            elif self.regime_strict:
                raise ValueError("include_global_regime=True but regime_df not provided and regime_strict=True")
            else:
                for col in self._get_regime_feature_columns():
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
            features.extend(self._get_regime_feature_columns())

        if self.include_rsi:
            features.extend(['rsi_14', 'rsi_normalized'])
        
        if self.include_ma_features:
            features.extend(['dist_ma50', 'dist_ma200', 'ma_cross'])
        
        if self.include_price_features:
            features.extend(['daily_range', 'body_ratio', 'overnight_return', 'intraday_return'])
        
        if self.include_volume_features:
            features.extend(['volume_ma20', 'volume_ratio', 'dollar_volume'])
        
        return features

    def _get_regime_feature_columns(self) -> List[str]:
        return get_regime_features(
            include_subsequent_returns=self.regime_include_subsequent_returns,
            horizons=self.regime_subsequent_return_horizons,
        )


def create_feature_engineer_from_config(config: Dict[str, Any]) -> FeatureEngineer:
    """Create FeatureEngineer from a plain dictionary.

    .. deprecated::
        Prefer ``FeatureEngineer(FeatureConfig(**config_dict))`` directly.
    """
    from mci_gru.config import FeatureConfig
    return FeatureEngineer(FeatureConfig(**config))
