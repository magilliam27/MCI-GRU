"""
Configuration dataclasses for MCI-GRU experiments.

This module provides structured configuration classes that work with Hydra
for experiment configuration and management.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from omegaconf import MISSING


@dataclass
class DataConfig:
    """
    Data source and date range configuration.
    
    Attributes:
        universe: Stock universe name (sp500, russell1000, msci_world)
        source: Data source type (csv, lseg)
        filename: Path to CSV file (used when source='csv')
        experiment_mode: 'stock_level' (cross-sectional stocks) or 'index_level' (single index series; no survivorship bias)
        index_filename: Path to index CSV with dt, close (used when experiment_mode='index_level'); if None, use FRED SP500
        train_start: Training period start date
        train_end: Training period end date
        val_start: Validation period start date
        val_end: Validation period end date
        test_start: Test period start date
        test_end: Test period end date
    """
    universe: str = "sp500"
    source: str = "csv"
    filename: str = "data/raw/market/sp500_data.csv"
    experiment_mode: str = "stock_level"
    index_filename: Optional[str] = None
    train_start: str = "2019-01-01"
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"
    
    def __post_init__(self):
        if self.experiment_mode not in ("stock_level", "index_level"):
            raise ValueError(f"experiment_mode must be 'stock_level' or 'index_level', got {self.experiment_mode!r}")
        dates = [self.train_start, self.train_end, self.val_start,
                 self.val_end, self.test_start, self.test_end]
        for i in range(len(dates) - 1):
            if dates[i] > dates[i + 1]:
                raise ValueError(f"Dates must be in chronological order: {dates}")


@dataclass
class FeatureConfig:
    """
    Feature engineering configuration.
    
    Attributes:
        base_features: List of base OHLCV feature column names
        include_momentum: Whether to add momentum features
        include_weekly_momentum: Whether to include weekly momentum features (5-day return/signal)
        momentum_encoding: Type of momentum encoding (binary, continuous, buffered)
        momentum_blend_mode: Blend mode for momentum_blend (static or dynamic)
        momentum_blend_fast_weight: FAST allocation for static intermediate-speed blending
        momentum_dynamic_correction_fast_weight: Fallback FAST allocation for Correction when DYN is not estimable
        momentum_dynamic_rebound_fast_weight: Fallback FAST allocation for Rebound when DYN is not estimable
        momentum_dynamic_lookback_periods: Prior periods used by DYN (0 = expanding history)
        momentum_dynamic_min_history: Minimum prior observations before DYN estimation activates
        momentum_dynamic_min_state_observations: Minimum prior observations required per state
        momentum_buffer_low: Low buffer percentile for buffered encoding
        momentum_buffer_high: High buffer percentile for buffered encoding
        include_volatility: Whether to add volatility features
        include_vix: Whether to add VIX features
        include_credit_spread: Whether to add credit spread features (IG/HY from FRED)
        include_global_regime: Whether to add global scalar regime features
        regime_change_months: Months for delta transform (default 12)
        regime_norm_months: Rolling normalization window in months (default 120)
        regime_clip_z: Clip bound for transformed values
        regime_exclusion_months: Exclusion window before matching historical months
        regime_similarity_quantile: Quantile size for similar/dissimilar buckets
        regime_min_history_months: Minimum history required before emitting non-zero regime features
        regime_strict: If true, fail run when regime loading fails; otherwise soft-fill zeros
        regime_lseg_market_ric: LSEG RIC for market proxy (LSEG-primary, FRED fallback)
        regime_lseg_copper_ric: LSEG RIC for copper proxy (LSEG-primary, FRED fallback)
        regime_lseg_yield_10y_ric: LSEG RIC for 10Y yield fallback
        regime_lseg_yield_3m_ric: LSEG RIC for 3M yield fallback
        regime_lseg_oil_ric: LSEG RIC for oil fallback
        regime_lseg_vix_ric: LSEG RIC for volatility fallback
        regime_inputs_csv: Optional path to canonical regime CSV (if set, bypass live API for regime)
        regime_enforce_lag_days: If regime_inputs_csv set, shift dates by this many days (0 or 1) to avoid look-ahead
        regime_include_subsequent_returns: Whether to emit post-similarity return features
        regime_subsequent_return_horizons: Forward monthly return horizons used for similarity-conditioned features
        include_rsi: Whether to add RSI features
        include_ma_features: Whether to add moving average features
        include_price_features: Whether to add derived price features
        include_volume_features: Whether to add volume features
    """
    base_features: List[str] = field(
        default_factory=lambda: ['close', 'open', 'high', 'low', 'volume', 'turnover']
    )
    include_momentum: bool = True
    include_weekly_momentum: bool = True
    momentum_encoding: str = "binary"  # binary, continuous, buffered
    momentum_blend_mode: str = "static"  # static, dynamic
    momentum_blend_fast_weight: float = 0.5
    momentum_dynamic_correction_fast_weight: float = 0.15
    momentum_dynamic_rebound_fast_weight: float = 0.70
    momentum_dynamic_lookback_periods: int = 0
    momentum_dynamic_min_history: int = 252
    momentum_dynamic_min_state_observations: int = 3
    momentum_buffer_low: float = 0.1
    momentum_buffer_high: float = 0.9
    include_volatility: bool = False
    include_vix: bool = False
    include_credit_spread: bool = False
    include_global_regime: bool = False
    regime_change_months: int = 12
    regime_norm_months: int = 120
    regime_clip_z: float = 3.0
    regime_exclusion_months: int = 1
    regime_similarity_quantile: float = 0.2
    regime_min_history_months: int = 24
    regime_strict: bool = False
    regime_lseg_market_ric: str = ".SPX"
    regime_lseg_copper_ric: str = ".MXCOPPFE"
    regime_lseg_yield_10y_ric: str = "US10YT=RR"
    regime_lseg_yield_3m_ric: str = "US3MT=RR"
    regime_lseg_oil_ric: str = "CLc1"
    regime_lseg_vix_ric: str = "VIX"
    regime_inputs_csv: Optional[str] = None
    regime_enforce_lag_days: int = 0
    regime_include_subsequent_returns: bool = True
    regime_subsequent_return_horizons: List[int] = field(default_factory=lambda: [1, 3])
    include_rsi: bool = False
    include_ma_features: bool = False
    include_price_features: bool = False
    include_volume_features: bool = False
    
    def __post_init__(self):
        valid_encodings = ['binary', 'continuous', 'buffered']
        valid_blend_modes = ['static', 'dynamic']
        if self.momentum_encoding not in valid_encodings:
            raise ValueError(f"momentum_encoding must be one of {valid_encodings}")
        if self.momentum_blend_mode not in valid_blend_modes:
            raise ValueError(f"momentum_blend_mode must be one of {valid_blend_modes}")
        for name, value in [
            ("momentum_blend_fast_weight", self.momentum_blend_fast_weight),
            (
                "momentum_dynamic_correction_fast_weight",
                self.momentum_dynamic_correction_fast_weight,
            ),
            (
                "momentum_dynamic_rebound_fast_weight",
                self.momentum_dynamic_rebound_fast_weight,
            ),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.momentum_dynamic_lookback_periods < 0:
            raise ValueError("momentum_dynamic_lookback_periods must be >= 0")
        if self.momentum_dynamic_min_history <= 0:
            raise ValueError("momentum_dynamic_min_history must be > 0")
        if self.momentum_dynamic_min_state_observations <= 0:
            raise ValueError("momentum_dynamic_min_state_observations must be > 0")
        if not (0 < self.regime_similarity_quantile < 0.5):
            raise ValueError("regime_similarity_quantile must be in (0, 0.5)")
        if self.regime_change_months <= 0:
            raise ValueError("regime_change_months must be > 0")
        if self.regime_norm_months <= 0:
            raise ValueError("regime_norm_months must be > 0")
        if self.regime_exclusion_months < 0:
            raise ValueError("regime_exclusion_months must be >= 0")
        if self.regime_min_history_months <= 0:
            raise ValueError("regime_min_history_months must be > 0")
        if self.regime_enforce_lag_days < 0:
            raise ValueError("regime_enforce_lag_days must be >= 0")
        if any(horizon <= 0 for horizon in self.regime_subsequent_return_horizons):
            raise ValueError("regime_subsequent_return_horizons must contain only positive integers")
        if self.regime_include_subsequent_returns and self.regime_exclusion_months < 1:
            raise ValueError(
                "regime_exclusion_months must be >= 1 when regime_include_subsequent_returns=True"
            )


@dataclass
class GraphConfig:
    """
    Graph construction configuration.
    
    Attributes:
        judge_value: Correlation threshold for edge creation
        update_frequency_months: How often to update graph (0 = never)
        corr_lookback_days: Days of history for correlation computation
    """
    judge_value: float = 0.8
    update_frequency_months: int = 0
    corr_lookback_days: int = 252
    
    def __post_init__(self):
        if not (0 < self.judge_value < 1):
            raise ValueError(f"judge_value must be between 0 and 1, got {self.judge_value}")
        if self.update_frequency_months < 0:
            raise ValueError(f"update_frequency_months must be >= 0")
        if self.corr_lookback_days <= 0:
            raise ValueError(f"corr_lookback_days must be > 0")


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Attributes:
        his_t: Historical lookback period (days)
        label_t: Forward return period (days)
        gru_hidden_sizes: Hidden sizes for GRU layers
        hidden_size_gat1: Hidden size for first GAT layer
        output_gat1: Output size for first GAT layer
        gat_heads: Number of attention heads in GAT
        hidden_size_gat2: Hidden size for second GAT layer
        num_hidden_states: Number of latent state vectors
        cross_attn_heads: Number of heads in cross-attention
        slow_kernel: Kernel size for slow temporal aggregation
        slow_stride: Stride for slow temporal downsampling
        use_multi_scale: Use MultiScaleTemporalEncoder (True) or plain ImprovedGRU (False)
        use_self_attention: Apply self-attention before final prediction GAT
        activation: Activation function ("elu" or "relu")
        latent_init_scale: Std for latent state initialisation (original code used 1.0)
    """
    his_t: int = 10
    label_t: int = 5
    gru_hidden_sizes: List[int] = field(default_factory=lambda: [32, 10])
    hidden_size_gat1: int = 32
    output_gat1: int = 4
    gat_heads: int = 4
    hidden_size_gat2: int = 32
    num_hidden_states: int = 32
    cross_attn_heads: int = 4
    slow_kernel: int = 5
    slow_stride: int = 2
    use_multi_scale: bool = True
    use_self_attention: bool = True
    activation: str = "elu"
    latent_init_scale: float = 0.02

    def __post_init__(self):
        if self.activation not in ("elu", "relu"):
            raise ValueError(f"activation must be 'elu' or 'relu', got {self.activation!r}")
        if self.latent_init_scale <= 0:
            raise ValueError("latent_init_scale must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'gru_hidden_sizes': self.gru_hidden_sizes,
            'hidden_size_gat1': self.hidden_size_gat1,
            'output_gat1': self.output_gat1,
            'gat_heads': self.gat_heads,
            'hidden_size_gat2': self.hidden_size_gat2,
            'num_hidden_states': self.num_hidden_states,
            'cross_attn_heads': self.cross_attn_heads,
            'slow_kernel': self.slow_kernel,
            'slow_stride': self.slow_stride,
            'use_multi_scale': self.use_multi_scale,
            'use_self_attention': self.use_self_attention,
            'activation': self.activation,
            'latent_init_scale': self.latent_init_scale,
        }


@dataclass
class TrainingConfig:
    """
    Training configuration.

    Attributes:
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        num_epochs: Maximum number of training epochs
        num_models: Number of models to train (for averaging)
        early_stopping_patience: Epochs to wait before early stopping
        weight_decay: L2 regularization weight
        gradient_clip: Maximum gradient norm (0 = no clipping)
        loss_type: Loss function (mse, ic, combined)
        ic_loss_alpha: Weight for IC component when loss_type=combined (0 to 1)
        label_type: Label representation -- "returns" for raw forward returns,
                     "rank" for cross-sectional rank percentiles per day.
                     Rank labels use only same-day information so they
                     do not introduce look-ahead bias.
    """
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 100
    num_models: int = 10
    early_stopping_patience: int = 10
    weight_decay: float = 1e-3
    gradient_clip: float = 1.0
    loss_type: str = "mse"
    ic_loss_alpha: float = 0.5
    label_type: str = "returns"

    _VALID_LOSS_TYPES = ("mse", "ic", "combined")
    _VALID_LABEL_TYPES = ("returns", "rank")

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.num_models <= 0:
            raise ValueError("num_models must be > 0")
        if self.loss_type not in self._VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type must be one of {self._VALID_LOSS_TYPES}, got {self.loss_type!r}"
            )
        if not 0 <= self.ic_loss_alpha <= 1:
            raise ValueError(f"ic_loss_alpha must be in [0, 1], got {self.ic_loss_alpha}")
        if self.label_type not in self._VALID_LABEL_TYPES:
            raise ValueError(
                f"label_type must be one of {self._VALID_LABEL_TYPES}, got {self.label_type!r}"
            )


@dataclass
class TrackingConfig:
    """Optional MLflow experiment tracking configuration.

    Attributes:
        enabled: Whether MLflow tracking is active
        tracking_uri: Local or remote MLflow tracking URI
        experiment_name: Optional MLflow experiment override
        run_name: Optional MLflow run name override
        log_artifacts: Whether to log run artifacts to MLflow
        log_checkpoints: Whether to log checkpoint files
        log_predictions: Whether to log prediction directories (off by default
            to avoid duplicating a large number of CSV artifacts)
    """
    enabled: bool = False
    tracking_uri: str = "mlruns"
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    log_artifacts: bool = True
    log_checkpoints: bool = True
    log_predictions: bool = False


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.
    
    Combines all sub-configurations into a single config object.
    
    Attributes:
        data: Data source and date configuration
        features: Feature engineering configuration
        graph: Graph construction configuration
        model: Model architecture configuration
        training: Training configuration
        tracking: MLflow tracking configuration
        experiment_name: Name for this experiment
        output_dir: Directory for saving outputs
        seed: Random seed for reproducibility
    """
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    experiment_name: str = "baseline"
    output_dir: str = "results"
    seed: int = 42
    
    def get_output_path(self) -> str:
        import os
        return os.path.join(self.output_dir, self.experiment_name)
    
    def to_flat_dict(self) -> Dict[str, Any]:
        flat = {
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            # Data
            'data.universe': self.data.universe,
            'data.source': self.data.source,
            'data.train_start': self.data.train_start,
            'data.train_end': self.data.train_end,
            'data.val_start': self.data.val_start,
            'data.val_end': self.data.val_end,
            'data.test_start': self.data.test_start,
            'data.test_end': self.data.test_end,
            # Features
            'features.include_momentum': self.features.include_momentum,
            'features.momentum_encoding': self.features.momentum_encoding,
            'features.momentum_blend_mode': self.features.momentum_blend_mode,
            'features.momentum_blend_fast_weight': self.features.momentum_blend_fast_weight,
            'features.momentum_dynamic_correction_fast_weight': self.features.momentum_dynamic_correction_fast_weight,
            'features.momentum_dynamic_rebound_fast_weight': self.features.momentum_dynamic_rebound_fast_weight,
            'features.momentum_dynamic_lookback_periods': self.features.momentum_dynamic_lookback_periods,
            'features.momentum_dynamic_min_history': self.features.momentum_dynamic_min_history,
            'features.momentum_dynamic_min_state_observations': self.features.momentum_dynamic_min_state_observations,
            'features.include_weekly_momentum': self.features.include_weekly_momentum,
            'features.include_vix': self.features.include_vix,
            'features.include_credit_spread': self.features.include_credit_spread,
            'features.include_global_regime': self.features.include_global_regime,
            'features.include_volatility': self.features.include_volatility,
            'features.regime_include_subsequent_returns': self.features.regime_include_subsequent_returns,
            'features.regime_subsequent_return_horizons': str(self.features.regime_subsequent_return_horizons),
            # Graph
            'graph.judge_value': self.graph.judge_value,
            'graph.update_frequency_months': self.graph.update_frequency_months,
            # Model
            'model.his_t': self.model.his_t,
            'model.label_t': self.model.label_t,
            'model.gru_hidden_sizes': str(self.model.gru_hidden_sizes),
            # Training
            'training.batch_size': self.training.batch_size,
            'training.learning_rate': self.training.learning_rate,
            'training.num_epochs': self.training.num_epochs,
            'training.num_models': self.training.num_models,
            'training.loss_type': self.training.loss_type,
            'training.ic_loss_alpha': self.training.ic_loss_alpha,
            # Tracking
            'tracking.enabled': self.tracking.enabled,
            'tracking.tracking_uri': self.tracking.tracking_uri,
            'tracking.log_artifacts': self.tracking.log_artifacts,
            'tracking.log_checkpoints': self.tracking.log_checkpoints,
            'tracking.log_predictions': self.tracking.log_predictions,
        }
        return flat


# Pre-defined experiment configurations
EXPERIMENT_PRESETS = {
    'baseline': ExperimentConfig(),
    
    'lookback_5': ExperimentConfig(
        model=ModelConfig(his_t=5),
        experiment_name='lookback_5'
    ),
    
    'lookback_20': ExperimentConfig(
        model=ModelConfig(his_t=20),
        experiment_name='lookback_20'
    ),
    
    'training_10yr': ExperimentConfig(
        data=DataConfig(train_start='2014-01-01'),
        experiment_name='training_10yr'
    ),
    
    'with_vix': ExperimentConfig(
        features=FeatureConfig(include_vix=True, include_volatility=True),
        experiment_name='with_vix'
    ),
    
    'correlation_6mo': ExperimentConfig(
        graph=GraphConfig(update_frequency_months=6),
        experiment_name='correlation_6mo'
    ),
    
    'momentum_buffered': ExperimentConfig(
        features=FeatureConfig(momentum_encoding='buffered'),
        experiment_name='momentum_buffered'
    ),
    
    'momentum_continuous': ExperimentConfig(
        features=FeatureConfig(momentum_encoding='continuous'),
        experiment_name='momentum_continuous'
    ),

    'momentum_dynamic': ExperimentConfig(
        features=FeatureConfig(
            momentum_blend_mode='dynamic',
            momentum_dynamic_correction_fast_weight=0.15,
            momentum_dynamic_rebound_fast_weight=0.70,
            momentum_dynamic_lookback_periods=0,
            momentum_dynamic_min_history=252,
            momentum_dynamic_min_state_observations=3,
        ),
        experiment_name='momentum_dynamic'
    ),
}


def get_preset(name: str) -> ExperimentConfig:
    """
    Get a pre-defined experiment configuration.
    
    Args:
        name: Preset name
        
    Returns:
        ExperimentConfig instance
        
    Raises:
        ValueError: If preset name not found
    """
    if name not in EXPERIMENT_PRESETS:
        available = list(EXPERIMENT_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return EXPERIMENT_PRESETS[name]


def create_config_from_dict(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """
    Create ExperimentConfig from a flat or nested dictionary.
    
    Useful for creating configs from Hydra or command-line args.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        ExperimentConfig instance
    """
    # Handle nested structure
    data_dict = config_dict.get('data', {})
    features_dict = config_dict.get('features', {})
    graph_dict = config_dict.get('graph', {})
    model_dict = config_dict.get('model', {})
    training_dict = config_dict.get('training', {})
    tracking_dict = config_dict.get('tracking', {})
    
    return ExperimentConfig(
        data=DataConfig(**data_dict) if data_dict else DataConfig(),
        features=FeatureConfig(**features_dict) if features_dict else FeatureConfig(),
        graph=GraphConfig(**graph_dict) if graph_dict else GraphConfig(),
        model=ModelConfig(**model_dict) if model_dict else ModelConfig(),
        training=TrainingConfig(**training_dict) if training_dict else TrainingConfig(),
        tracking=TrackingConfig(**tracking_dict) if tracking_dict else TrackingConfig(),
        experiment_name=config_dict.get('experiment_name', 'baseline'),
        output_dir=config_dict.get('output_dir', 'results'),
        seed=config_dict.get('seed', 42),
    )
