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
import sys
import json
import random
import logging

import numpy as np
import torch
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mci_gru.config import (
    ExperimentConfig,
    DataConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    TrainingConfig,
)
from mci_gru.data.data_manager import create_data_loaders
from mci_gru.features import FeatureEngineer
from mci_gru.models import create_model
from mci_gru.pipeline import prepare_data, prepare_data_index_level
from mci_gru.training import train_multiple_models


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str, experiment_name: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'training_{timestamp}.log')
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
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    from hydra.core.hydra_config import HydraConfig
    try:
        hydra_cfg = HydraConfig.get()
        output_path = hydra_cfg.runtime.output_dir
    except:
        output_path = os.getcwd()
    
    logger = setup_logging(output_path, cfg.get('experiment_name', 'baseline'))
    
    logger.info("=" * 80)
    logger.info("MCI-GRU Experiment Runner")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    config = dict_to_config(cfg)
    set_seed(config.seed)
    logger.info(f"\nRandom seed: {config.seed}")
    logger.info(f"Output directory: {output_path}")
    config_path = os.path.join(output_path, 'config.yaml')
    OmegaConf.save(cfg, config_path)
    logger.info(f"Configuration saved to: {config_path}")
    
    logger.info("\nInitializing feature engineer...")
    feature_engineer = FeatureEngineer(config.features)
    
    if config.data.experiment_mode == "index_level":
        data = prepare_data_index_level(config, feature_engineer)
    else:
        data = prepare_data(config, feature_engineer)
    
    metadata = {
        'norm_means': {k: float(v) for k, v in data['norm_means'].items()},
        'norm_stds': {k: float(v) for k, v in data['norm_stds'].items()},
        'feature_cols': data['feature_cols'],
        'kdcode_list': data['kdcode_list'],
        'his_t': config.model.his_t,
        'label_t': config.model.label_t,
        'seed': config.seed,
        'train_end': config.data.train_end,
        'data_file': config.data.filename,
    }
    metadata_path = os.path.join(output_path, 'run_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Run metadata saved to: {metadata_path}")

    graph_data_path = os.path.join(output_path, 'graph_data.pt')
    torch.save({
        'edge_index': data['edge_index'],
        'edge_weight': data['edge_weight'],
    }, graph_data_path)
    logger.info(f"Graph data saved to: {graph_data_path}")

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
    
    num_features = len(data['feature_cols'])
    
    def model_factory():
        return create_model(num_features, config.model.to_dict())
    
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
        output_path=output_path,
    )
    
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
