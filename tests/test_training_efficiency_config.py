from pathlib import Path

import numpy as np
import torch

from mci_gru.config import TrainingConfig, create_config_from_dict
from mci_gru.data.data_manager import create_data_loaders


def test_training_config_exposes_dataloader_and_profile_defaults() -> None:
    cfg = TrainingConfig()

    assert cfg.dataloader_num_workers == 0
    assert cfg.dataloader_pin_memory is False
    assert cfg.dataloader_persistent_workers is False
    assert cfg.dataloader_prefetch_factor is None
    assert cfg.profile_batches == 0


def test_create_config_from_dict_accepts_efficiency_overrides() -> None:
    cfg = create_config_from_dict(
        {
            "training": {
                "dataloader_num_workers": 2,
                "dataloader_pin_memory": True,
                "dataloader_persistent_workers": True,
                "dataloader_prefetch_factor": 4,
                "profile_batches": 8,
            }
        }
    )

    assert cfg.training.dataloader_num_workers == 2
    assert cfg.training.dataloader_pin_memory is True
    assert cfg.training.dataloader_persistent_workers is True
    assert cfg.training.dataloader_prefetch_factor == 4
    assert cfg.training.profile_batches == 8


def test_create_data_loaders_preserves_default_loader_behavior() -> None:
    stock_features = np.zeros((3, 4, 2, 2), dtype=np.float32)
    graph_features = np.zeros((3, 4, 2), dtype=np.float32)
    labels = np.zeros((3, 4), dtype=np.float32)
    dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.ones(2, dtype=torch.float32)

    train_loader, val_loader, test_loader = create_data_loaders(
        stock_features_train=stock_features,
        x_graph_train=graph_features,
        train_labels=labels,
        stock_features_val=stock_features,
        x_graph_val=graph_features,
        val_labels=labels,
        stock_features_test=stock_features,
        x_graph_test=graph_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch_size=2,
        train_dates=dates,
        val_dates=dates,
        test_dates=dates,
    )

    assert train_loader.num_workers == 0
    assert train_loader.pin_memory is False
    assert val_loader.num_workers == 0
    assert test_loader.batch_size == 1


def test_create_data_loaders_accepts_efficiency_overrides(tmp_path: Path) -> None:
    del tmp_path
    stock_features = np.zeros((3, 4, 2, 2), dtype=np.float32)
    graph_features = np.zeros((3, 4, 2), dtype=np.float32)
    labels = np.zeros((3, 4), dtype=np.float32)
    dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.ones(2, dtype=torch.float32)

    train_loader, val_loader, _test_loader = create_data_loaders(
        stock_features_train=stock_features,
        x_graph_train=graph_features,
        train_labels=labels,
        stock_features_val=stock_features,
        x_graph_val=graph_features,
        val_labels=labels,
        stock_features_test=stock_features,
        x_graph_test=graph_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch_size=2,
        train_dates=dates,
        val_dates=dates,
        test_dates=dates,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
    )

    assert train_loader.num_workers == 2
    assert train_loader.pin_memory is True
    assert train_loader.persistent_workers is True
    assert train_loader.prefetch_factor == 4
    assert val_loader.num_workers == 2
