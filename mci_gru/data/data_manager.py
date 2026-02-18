"""
Unified data manager for MCI-GRU.

This module provides a unified interface for loading data from
different sources (CSV, LSEG) and preparing it for model training.
"""

import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from functools import partial

import torch
from torch.utils.data import Dataset

from mci_gru.config import DataConfig, FeatureConfig, ExperimentConfig
from mci_gru.data.path_resolver import resolve_project_data_path


class DataManager:
    """
    Unified data manager for loading and preparing stock data.
    
    Supports loading from CSV files or LSEG/Refinitiv API.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize data manager.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.vix_df: Optional[pd.DataFrame] = None
        self.credit_df: Optional[pd.DataFrame] = None
        self.kdcode_list: Optional[List[str]] = None
    
    def load(self) -> pd.DataFrame:
        """
        Load data from configured source.
        
        Returns:
            DataFrame with stock data
        """
        if self.config.source == "csv":
            return self._load_from_csv()
        elif self.config.source == "lseg":
            return self._load_from_lseg()
        else:
            raise ValueError(f"Unknown data source: {self.config.source}")
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load data from CSV file."""
        resolved_path = resolve_project_data_path(self.config.filename)
        print(f"Loading data from {resolved_path}...")

        df = pd.read_csv(resolved_path)
        
        print(f"  Loaded {len(df)} rows")
        print(f"  Date range: {df['dt'].min()} to {df['dt'].max()}")
        print(f"  Stocks: {df['kdcode'].nunique()}")
        
        self.df = df
        return df
    
    def _load_from_lseg(self) -> pd.DataFrame:
        """Load data from LSEG/Refinitiv."""
        from mci_gru.data.lseg_loader import LSEGLoader
        
        loader = LSEGLoader()
        try:
            loader.connect()
            
            # Determine date range (need data before training for correlation)
            start_date = self.config.train_start
            end_date = self.config.test_end
            
            df = loader.fetch_universe_data(
                universe=self.config.universe,
                start=start_date,
                end=end_date,
                include_vix=False  # We'll handle VIX separately
            )
            
            self.df = df
            return df
            
        finally:
            loader.disconnect()
    
    def load_vix(self) -> pd.DataFrame:
        """
        Load VIX data (for VIX feature integration).
        
        Returns:
            DataFrame with VIX data
        """
        if self.config.source == "lseg":
            from mci_gru.data.lseg_loader import LSEGLoader
            
            loader = LSEGLoader()
            try:
                loader.connect()
                vix_df = loader.get_vix(
                    self.config.train_start,
                    self.config.test_end
                )
                self.vix_df = vix_df
                return vix_df
            finally:
                loader.disconnect()
        else:
            # Try to load from a VIX CSV file
            try:
                vix_path = resolve_project_data_path("vix_data.csv")
            except FileNotFoundError:
                raise FileNotFoundError(
                    "VIX data not found. Create vix_data.csv under data/raw/market "
                    "or use source='lseg'"
                )
            vix_df = pd.read_csv(vix_path)
            self.vix_df = vix_df
            return vix_df

    def load_credit_spreads(self) -> pd.DataFrame:
        """
        Load credit spread data (IG/HY OAS) from FRED API.

        Requires FRED_API_KEY environment variable. If the key is missing or
        the fetch fails, raises an exception; the caller should catch and
        soft-fail (e.g. continue without credit features).

        Returns:
            DataFrame with columns [dt, ig_spread, hy_spread]
        """
        from mci_gru.data.fred_loader import FREDLoader

        loader = FREDLoader()
        credit_df = loader.get_credit_spreads(
            start=self.config.train_start,
            end=self.config.test_end,
        )
        self.credit_df = credit_df
        return credit_df

    def filter_complete_stocks(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Filter to stocks with complete data across all periods.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame and list of stock codes
        """
        print("Filtering stocks with complete data...")
        
        # Get date range from train_start through test_end
        date_mask = (df['dt'] >= self.config.train_start) & (df['dt'] <= self.config.test_end)
        df_period = df[date_mask].copy()
        period_dates = sorted(df_period['dt'].unique())
        
        print(f"  Period: {len(period_dates)} trading days from {period_dates[0]} to {period_dates[-1]}")
        
        # Count occurrences per stock
        kdcode_counts = df_period['kdcode'].value_counts()
        
        # Keep only stocks present on all trading days
        kdcode_list = kdcode_counts[kdcode_counts == len(period_dates)].index.tolist()
        kdcode_list = sorted(kdcode_list)
        
        print(f"  Stocks with complete data: {len(kdcode_list)}")
        
        if len(kdcode_list) == 0:
            raise ValueError("No stocks have complete data across the entire period!")
        
        # Filter DataFrame
        df_filtered = df_period[df_period['kdcode'].isin(kdcode_list)].copy()
        df_filtered = df_filtered.sort_values(['dt', 'kdcode']).reset_index(drop=True)
        
        self.kdcode_list = kdcode_list
        return df_filtered, kdcode_list
    
    def split_by_period(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test periods.
        
        Args:
            df: Filtered DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_mask = (df['dt'] >= self.config.train_start) & (df['dt'] <= self.config.train_end)
        val_mask = (df['dt'] >= self.config.val_start) & (df['dt'] <= self.config.val_end)
        test_mask = (df['dt'] >= self.config.test_start) & (df['dt'] <= self.config.test_end)
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()
        
        train_dates = sorted(train_df['dt'].unique())
        val_dates = sorted(val_df['dt'].unique())
        test_dates = sorted(test_df['dt'].unique())
        
        print(f"  Training: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
        print(f"  Validation: {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})")
        print(f"  Test: {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")
        
        return train_df, val_df, test_df


class CombinedDataset(Dataset):
    """
    Combined dataset for synchronized time series, graph features, and labels.
    
    This ensures time series and graph data stay aligned when shuffling.
    """
    
    def __init__(self, X_time_series, X_graph, y):
        """
        Initialize dataset.
        
        Args:
            X_time_series: Time series features tensor
            X_graph: Graph node features tensor
            y: Labels tensor
        """
        self.X_time_series = X_time_series
        self.X_graph = X_graph
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'time_series': self.X_time_series[idx],
            'graph_features': self.X_graph[idx],
            'label': self.y[idx]
        }


def combined_collate_fn(batch, edge_index, edge_weight):
    """
    Custom collate function to create properly batched graph data.
    
    PyG batches graphs by concatenating nodes and shifting edge indices.
    This function replicates that behavior while keeping time series aligned.
    
    Args:
        batch: List of dicts with 'time_series', 'graph_features', 'label'
        edge_index: Original edge index tensor (2, num_edges)
        edge_weight: Original edge weight tensor (num_edges,)
    
    Returns:
        time_series: (batch_size, num_stocks, seq_len, features)
        labels: (batch_size, num_stocks)
        graph_features: (batch_size * num_stocks, features)
        batched_edge_index: (2, batch_size * num_edges)
        batched_edge_weight: (batch_size * num_edges,)
        num_stocks: int
    """
    batch_size = len(batch)
    num_stocks = batch[0]['graph_features'].shape[0]
    
    # Stack time series and labels (standard batching)
    time_series = torch.stack([item['time_series'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Create batched graph structure
    graph_features_list = []
    edge_index_list = []
    edge_weight_list = []
    
    for i, item in enumerate(batch):
        graph_features_list.append(item['graph_features'])
        # Shift edge indices for each graph in the batch
        shifted_edge_index = edge_index + (i * num_stocks)
        edge_index_list.append(shifted_edge_index)
        edge_weight_list.append(edge_weight)
    
    # Concatenate all graph data
    batched_graph_features = torch.cat(graph_features_list, dim=0)
    batched_edge_index = torch.cat(edge_index_list, dim=1)
    batched_edge_weight = torch.cat(edge_weight_list, dim=0)
    
    return time_series, labels, batched_graph_features, batched_edge_index, batched_edge_weight, num_stocks


def create_data_loaders(
    stock_features_train: np.ndarray,
    x_graph_train: np.ndarray,
    train_labels: np.ndarray,
    stock_features_val: np.ndarray,
    x_graph_val: np.ndarray,
    val_labels: np.ndarray,
    stock_features_test: np.ndarray,
    x_graph_test: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    batch_size: int = 32
) -> Tuple:
    """
    Create train/val/test data loaders.
    
    Args:
        stock_features_train: Training time series (days, stocks, seq_len, features)
        x_graph_train: Training graph features (days, stocks, features)
        train_labels: Training labels (days, stocks)
        stock_features_val: Validation time series
        x_graph_val: Validation graph features
        val_labels: Validation labels
        stock_features_test: Test time series
        x_graph_test: Test graph features
        edge_index: Graph edge indices
        edge_weight: Graph edge weights
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("Creating data loaders...")
    
    # Convert to tensors
    X_train_ts = torch.from_numpy(stock_features_train).float()
    X_train_graph = torch.from_numpy(x_graph_train).float()
    y_train = torch.from_numpy(train_labels).float()
    
    X_val_ts = torch.from_numpy(stock_features_val).float()
    X_val_graph = torch.from_numpy(x_graph_val).float()
    y_val = torch.from_numpy(val_labels).float()
    
    X_test_ts = torch.from_numpy(stock_features_test).float()
    X_test_graph = torch.from_numpy(x_graph_test).float()
    y_test_dummy = torch.zeros(len(X_test_ts), X_test_graph.shape[1], dtype=torch.float32)
    
    print(f"  Train: ts={X_train_ts.shape}, graph={X_train_graph.shape}, labels={y_train.shape}")
    print(f"  Val: ts={X_val_ts.shape}, graph={X_val_graph.shape}, labels={y_val.shape}")
    print(f"  Test: ts={X_test_ts.shape}, graph={X_test_graph.shape}")
    
    # Create datasets
    train_dataset = CombinedDataset(X_train_ts, X_train_graph, y_train)
    val_dataset = CombinedDataset(X_val_ts, X_val_graph, y_val)
    test_dataset = CombinedDataset(X_test_ts, X_test_graph, y_test_dummy)
    
    # Create collate function with edge data
    collate_fn = partial(combined_collate_fn, edge_index=edge_index, edge_weight=edge_weight)
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"  Created loaders: train={len(train_loader)} batches, val={len(val_loader)} batches, test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
