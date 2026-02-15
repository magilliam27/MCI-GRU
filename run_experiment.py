#!/usr/bin/env python
"""
MCI-GRU Experiment Runner

Main entry point for running MCI-GRU experiments with Hydra configuration.

Usage:
    # Run baseline experiment
    python run_experiment.py
    
    # Override lookback period
    python run_experiment.py model.his_t=5
    
    # Use a different experiment preset
    python run_experiment.py +experiment=with_vix
    
    # Sweep over multiple values
    python run_experiment.py --multirun model.his_t=5,10,15,20
    
    # Use Russell 1000 data
    python run_experiment.py +data=russell1000
    
    # Combine overrides
    python run_experiment.py +experiment=with_vix +data=russell1000 model.his_t=20
"""

import os
import gc
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mci_gru.config import (
    ExperimentConfig,
    DataConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    TrainingConfig,
)
from mci_gru.data.data_manager import DataManager, create_data_loaders
from mci_gru.features import FeatureEngineer
from mci_gru.graph import GraphBuilder
from mci_gru.models import create_model, StockPredictionModel
from mci_gru.training import Trainer, train_multiple_models
from mci_gru.training.metrics import evaluate_predictions, print_metrics


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str, experiment_name: str) -> logging.Logger:
    """
    Setup logging to both file and console.
    
    Args:
        output_dir: Base output directory
        experiment_name: Name of the experiment
        
    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = output_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing configuration
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def dict_to_config(cfg: DictConfig) -> ExperimentConfig:
    """Convert Hydra DictConfig to ExperimentConfig dataclass."""
    # Convert to plain dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Create sub-configs
    data_cfg = DataConfig(**cfg_dict.get('data', {}))
    feature_cfg = FeatureConfig(**cfg_dict.get('features', {}))
    graph_cfg = GraphConfig(**cfg_dict.get('graph', {}))
    model_cfg = ModelConfig(**cfg_dict.get('model', {}))
    training_cfg = TrainingConfig(**cfg_dict.get('training', {}))
    
    return ExperimentConfig(
        data=data_cfg,
        features=feature_cfg,
        graph=graph_cfg,
        model=model_cfg,
        training=training_cfg,
        experiment_name=cfg_dict.get('experiment_name', 'baseline'),
        output_dir=cfg_dict.get('output_dir', 'results'),
        seed=cfg_dict.get('seed', 42),
    )


def prepare_data(
    config: ExperimentConfig,
    feature_engineer: FeatureEngineer
) -> Dict[str, Any]:
    """
    Load and prepare data for training.
    
    This function mirrors the data preparation logic from sp500.py
    but uses the modular components.
    """
    print("=" * 80)
    print("Preparing Data")
    print("=" * 80)
    
    # Load data
    data_manager = DataManager(config.data)
    df = data_manager.load()
    
    # Load VIX if needed
    vix_df = None
    if config.features.include_vix:
        try:
            vix_df = data_manager.load_vix()
            print(f"Loaded VIX data: {len(vix_df)} observations")
        except Exception as e:
            print(f"Warning: Could not load VIX data: {e}")
            print("Continuing without VIX features")

    # Load credit spreads (FRED) if needed
    credit_df = None
    if config.features.include_credit_spread:
        try:
            credit_df = data_manager.load_credit_spreads()
            print(f"Loaded credit spread data: {len(credit_df)} observations")
        except Exception as e:
            print(f"Warning: Could not load credit spread data: {e}")
            print("Continuing without credit spread features")

    # Apply feature engineering
    df = feature_engineer.transform(df, vix_df, credit_df)
    
    # Get feature columns
    feature_cols = feature_engineer.get_feature_columns()
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    
    # Fill NaN values per day (mean imputation)
    print("Filling NaN values...")
    df_features_grouped = df.groupby('dt')
    res = []
    for dt in df_features_grouped.groups:
        df_day = df_features_grouped.get_group(dt).copy()
        for column in feature_cols:
            if column in df_day.columns:
                mean_val = df_day[column].mean()
                df_day[column] = df_day[column].fillna(mean_val)
        df_day = df_day.fillna(0.0)
        res.append(df_day)
    df_filled = pd.concat(res)
    del res
    gc.collect()
    
    # Compute normalization statistics from training period
    print(f"Computing normalization statistics from training period...")
    train_df = df_filled[df_filled['dt'] <= config.data.train_end]
    means = {}
    stds = {}
    for col in feature_cols:
        if col in train_df.columns:
            means[col] = train_df[col].mean()
            stds[col] = train_df[col].std()
            if stds[col] == 0:
                stds[col] = 1.0
    
    # Apply normalization
    df_norm = df_filled.copy()
    for col in feature_cols:
        if col in df_norm.columns:
            mean = means[col]
            std = stds[col]
            # 3-sigma clipping
            max_range = mean + 3 * std
            min_range = mean - 3 * std
            df_norm[col] = np.clip(df_norm[col], min_range, max_range)
            # Z-score
            df_norm[col] = (df_norm[col] - mean) / std
    
    del df_filled
    gc.collect()
    
    # Filter to complete stocks
    df_filtered, kdcode_list = data_manager.filter_complete_stocks(df_norm)
    
    # Split by period
    train_df, val_df, test_df = data_manager.split_by_period(df_filtered)
    
    train_dates = sorted(train_df['dt'].unique())
    val_dates = sorted(val_df['dt'].unique())
    test_dates = sorted(test_df['dt'].unique())
    
    # Generate time series features
    print("Generating time series features...")
    his_t = config.model.his_t
    
    # This is a simplified version - in production you'd use the full
    # generate_time_series_features_paper function from sp500.py
    stock_features = generate_time_series_features(
        df_filtered, kdcode_list, feature_cols, his_t
    )
    
    # Split features by period
    effective_train_days = len(train_dates) - his_t
    
    stock_features_train = stock_features[:effective_train_days]
    stock_features_val = stock_features[effective_train_days:effective_train_days + len(val_dates)]
    stock_features_test = stock_features[effective_train_days + len(val_dates):]
    
    # Generate graph features
    print("Generating graph features...")
    x_graph_train = generate_graph_features(train_df, kdcode_list, feature_cols, train_dates[his_t:])
    x_graph_val = generate_graph_features(val_df, kdcode_list, feature_cols, val_dates)
    x_graph_test = generate_graph_features(test_df, kdcode_list, feature_cols, test_dates)
    
    # Compute labels
    print("Computing labels...")
    train_labels = compute_labels(df, kdcode_list, train_dates[his_t:], config.model.label_t)
    val_labels = compute_labels(df, kdcode_list, val_dates, config.model.label_t)
    
    # Build graph
    print("Building correlation graph...")
    graph_builder = GraphBuilder(
        judge_value=config.graph.judge_value,
        update_frequency_months=config.graph.update_frequency_months,
        corr_lookback_days=config.graph.corr_lookback_days
    )
    edge_index, edge_weight = graph_builder.build_graph(
        df, kdcode_list, config.data.train_start
    )
    
    return {
        'kdcode_list': kdcode_list,
        'train_dates': train_dates[his_t:],
        'val_dates': val_dates,
        'test_dates': test_dates,
        'stock_features_train': stock_features_train,
        'stock_features_val': stock_features_val,
        'stock_features_test': stock_features_test,
        'x_graph_train': x_graph_train,
        'x_graph_val': x_graph_val,
        'x_graph_test': x_graph_test,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'feature_cols': feature_cols,
        'graph_builder': graph_builder,
        'df': df,  # Keep for dynamic graph updates
    }


def generate_time_series_features(
    df: pd.DataFrame,
    kdcode_list: List[str],
    feature_cols: List[str],
    his_t: int
) -> np.ndarray:
    """
    Generate time series features for all stocks.
    
    Returns array of shape (num_usable_days, num_stocks, his_t, num_features)
    """
    all_dates = sorted(df['dt'].unique())
    num_stocks = len(kdcode_list)
    num_features = len(feature_cols)
    num_usable_days = len(all_dates) - his_t
    
    print(f"  Allocating feature array: ({num_usable_days}, {num_stocks}, {his_t}, {num_features})")
    
    # Pre-allocate array
    stock_features = np.zeros((num_usable_days, num_stocks, his_t, num_features), dtype=np.float32)
    
    # Build lookup
    stock_to_idx = {stock: idx for idx, stock in enumerate(kdcode_list)}
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    
    # Build pivot data
    df_subset = df[df['kdcode'].isin(kdcode_list)][['kdcode', 'dt'] + feature_cols].copy()
    pivot_data = np.zeros((len(all_dates), num_stocks, num_features), dtype=np.float32)
    
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="  Building pivot"):
        kdcode = row['kdcode']
        dt = row['dt']
        if kdcode in stock_to_idx and dt in date_to_idx:
            stock_idx = stock_to_idx[kdcode]
            date_idx = date_to_idx[dt]
            pivot_data[date_idx, stock_idx, :] = row[feature_cols].values.astype(np.float32)
    
    # Generate sliding windows
    for day_offset in tqdm(range(num_usable_days), desc="  Processing days"):
        stock_features[day_offset, :, :, :] = pivot_data[day_offset:day_offset + his_t, :, :].transpose(1, 0, 2)
    
    return stock_features


def generate_graph_features(
    df: pd.DataFrame,
    kdcode_list: List[str],
    feature_cols: List[str],
    dates: List[str]
) -> np.ndarray:
    """Generate graph node features for each day."""
    num_dates = len(dates)
    num_stocks = len(kdcode_list)
    num_features = len(feature_cols)
    
    x_graph = np.zeros((num_dates, num_stocks, num_features), dtype=np.float32)
    stock_to_idx = {stock: idx for idx, stock in enumerate(kdcode_list)}
    
    df_subset = df[df['dt'].isin(dates) & df['kdcode'].isin(kdcode_list)]
    
    for date_idx, date in enumerate(dates):
        df_day = df_subset[df_subset['dt'] == date]
        for _, row in df_day.iterrows():
            stock_idx = stock_to_idx.get(row['kdcode'])
            if stock_idx is not None:
                x_graph[date_idx, stock_idx, :] = row[feature_cols].values.astype(np.float32)
    
    return x_graph


def compute_labels(
    df: pd.DataFrame,
    kdcode_list: List[str],
    dates: List[str],
    label_t: int
) -> np.ndarray:
    """Compute forward return labels."""
    df_subset = df[df['kdcode'].isin(kdcode_list)].copy()
    df_subset = df_subset.sort_values(['kdcode', 'dt'])
    
    # Compute forward returns
    df_subset['future_close'] = df_subset.groupby('kdcode')['close'].shift(-label_t)
    df_subset['next_close'] = df_subset.groupby('kdcode')['close'].shift(-1)
    df_subset['forward_return'] = df_subset['future_close'] / df_subset['next_close'] - 1
    
    # Pivot
    df_subset = df_subset[df_subset['dt'].isin(dates)]
    pivot = df_subset.pivot_table(index='dt', columns='kdcode', values='forward_return')
    pivot = pivot.reindex(index=dates, columns=kdcode_list)
    
    # Fill NaN
    for date in dates:
        if date in pivot.index:
            row_mean = pivot.loc[date].mean()
            pivot.loc[date] = pivot.loc[date].fillna(row_mean)
    pivot = pivot.fillna(0)
    
    return pivot.values.astype(np.float32)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    # Get Hydra's output directory (respects output_dir override from command line)
    from hydra.core.hydra_config import HydraConfig
    try:
        hydra_cfg = HydraConfig.get()
        output_path = hydra_cfg.runtime.output_dir
    except:
        # Fallback if Hydra config not available
        output_path = os.getcwd()
    
    # Setup logging first
    logger = setup_logging(output_path, cfg.get('experiment_name', 'baseline'))
    
    logger.info("=" * 80)
    logger.info("MCI-GRU Experiment Runner")
    logger.info("=" * 80)
    
    # Print configuration
    logger.info("\nConfiguration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    
    # Convert to typed config
    config = dict_to_config(cfg)
    
    # Use Hydra's managed output path (preserves user's output_dir setting)
    # Don't override config.output_dir - it contains the base path
    # output_path contains the full timestamped path
    
    # Set seed
    set_seed(config.seed)
    logger.info(f"\nRandom seed: {config.seed}")
    logger.info(f"Output directory: {output_path}")
    
    # Save config
    config_path = os.path.join(output_path, 'config.yaml')
    OmegaConf.save(cfg, config_path)
    logger.info(f"Configuration saved to: {config_path}")
    
    # Create feature engineer
    logger.info("\nInitializing feature engineer...")
    feature_engineer = FeatureEngineer(
        include_momentum=config.features.include_momentum,
        include_weekly_momentum=config.features.include_weekly_momentum,
        momentum_encoding=config.features.momentum_encoding,
        momentum_buffer_low=config.features.momentum_buffer_low,
        momentum_buffer_high=config.features.momentum_buffer_high,
        include_volatility=config.features.include_volatility,
        include_vix=config.features.include_vix,
        include_credit_spread=config.features.include_credit_spread,
        include_rsi=config.features.include_rsi,
        include_ma_features=config.features.include_ma_features,
        include_price_features=config.features.include_price_features,
        include_volume_features=config.features.include_volume_features,
    )
    
    # Prepare data
    data = prepare_data(config, feature_engineer)
    
    # Create data loaders
    logger.info("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        stock_features_train=data['stock_features_train'],
        x_graph_train=data['x_graph_train'],
        train_labels=data['train_labels'],
        stock_features_val=data['stock_features_val'],
        x_graph_val=data['x_graph_val'],
        val_labels=data['val_labels'],
        stock_features_test=data['stock_features_test'],
        x_graph_test=data['x_graph_test'],
        edge_index=data['edge_index'],
        edge_weight=data['edge_weight'],
        batch_size=config.training.batch_size
    )
    
    # Model factory
    num_features = len(data['feature_cols'])
    
    def model_factory():
        return create_model(num_features, config.model.to_dict())
    
    # Train multiple models
    logger.info("\n" + "=" * 80)
    logger.info("Training")
    logger.info("=" * 80)
    
    results, avg_predictions = train_multiple_models(
        model_factory=model_factory,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        kdcode_list=data['kdcode_list'],
        test_dates=data['test_dates'],
        graph_builder=data['graph_builder'],
        df=data['df'],
        train_dates=data['train_dates'],
        output_path=output_path,  # Pass Hydra's output path
    )
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Experiment Complete")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Models trained: {len(results)}")
    logger.info(f"Best validation losses: {[r.best_val_loss for r in results]}")
    logger.info(f"Mean best val loss: {np.mean([r.best_val_loss for r in results]):.6f}")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
