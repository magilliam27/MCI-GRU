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
        train_start: Training period start date
        train_end: Training period end date
        val_start: Validation period start date
        val_end: Validation period end date
        test_start: Test period start date
        test_end: Test period end date
    """
    universe: str = "sp500"
    source: str = "csv"
    filename: str = "sp500_yf_download.csv"
    train_start: str = "2019-01-01"
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"
    
    def __post_init__(self):
        """Validate date ordering."""
        # Basic validation - dates should be in order
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
        momentum_encoding: Type of momentum encoding (binary, continuous, buffered)
        momentum_buffer_low: Low buffer percentile for buffered encoding
        momentum_buffer_high: High buffer percentile for buffered encoding
        include_volatility: Whether to add volatility features
        include_vix: Whether to add VIX features
        include_credit_spread: Whether to add credit spread features (IG/HY from FRED)
        include_rsi: Whether to add RSI features
        include_ma_features: Whether to add moving average features
        include_price_features: Whether to add derived price features
        include_volume_features: Whether to add volume features
    """
    base_features: List[str] = field(
        default_factory=lambda: ['close', 'open', 'high', 'low', 'volume', 'turnover']
    )
    include_momentum: bool = True
    momentum_encoding: str = "binary"  # binary, continuous, buffered
    momentum_buffer_low: float = 0.1
    momentum_buffer_high: float = 0.9
    include_volatility: bool = False
    include_vix: bool = False
    include_credit_spread: bool = False
    include_rsi: bool = False
    include_ma_features: bool = False
    include_price_features: bool = False
    include_volume_features: bool = False
    
    def __post_init__(self):
        """Validate momentum encoding."""
        valid_encodings = ['binary', 'continuous', 'buffered']
        if self.momentum_encoding not in valid_encodings:
            raise ValueError(f"momentum_encoding must be one of {valid_encodings}")


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
        """Validate parameters."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model creation."""
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
    """
    batch_size: int = 32
    learning_rate: float = 0.0002
    num_epochs: int = 100
    num_models: int = 10
    early_stopping_patience: int = 10
    weight_decay: float = 0.0
    gradient_clip: float = 0.0
    loss_type: str = "mse"
    ic_loss_alpha: float = 0.5

    _VALID_LOSS_TYPES = ("mse", "ic", "combined")

    def __post_init__(self):
        """Validate parameters."""
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
        experiment_name: Name for this experiment
        output_dir: Directory for saving outputs
        seed: Random seed for reproducibility
    """
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "baseline"
    output_dir: str = "results"
    seed: int = 42
    
    def get_output_path(self) -> str:
        """Get full output path for this experiment."""
        import os
        return os.path.join(self.output_dir, self.experiment_name)
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for logging."""
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
            'features.momentum_encoding': self.features.momentum_encoding,
            'features.include_vix': self.features.include_vix,
            'features.include_credit_spread': self.features.include_credit_spread,
            'features.include_volatility': self.features.include_volatility,
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
    
    return ExperimentConfig(
        data=DataConfig(**data_dict) if data_dict else DataConfig(),
        features=FeatureConfig(**features_dict) if features_dict else FeatureConfig(),
        graph=GraphConfig(**graph_dict) if graph_dict else GraphConfig(),
        model=ModelConfig(**model_dict) if model_dict else ModelConfig(),
        training=TrainingConfig(**training_dict) if training_dict else TrainingConfig(),
        experiment_name=config_dict.get('experiment_name', 'baseline'),
        output_dir=config_dict.get('output_dir', 'results'),
        seed=config_dict.get('seed', 42),
    )
