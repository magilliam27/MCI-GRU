"""
Graph construction for MCI-GRU.

This module provides graph building functionality:
- Static graph construction from correlation matrix
- Dynamic graph updates at configurable intervals
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Optional


class GraphBuilder:
    """
    Build and maintain correlation-based stock graphs.
    
    Supports both static (build once) and dynamic (periodic update) modes.
    
    Args:
        judge_value: Correlation threshold for edge creation (default 0.8)
        update_frequency_months: How often to update graph (0 = never)
        corr_lookback_days: Days of history for correlation (default 252)
    """
    
    def __init__(
        self,
        judge_value: float = 0.8,
        update_frequency_months: int = 0,
        corr_lookback_days: int = 252
    ):
        self.judge_value = judge_value
        self.update_frequency_months = update_frequency_months
        self.corr_lookback_days = corr_lookback_days
        
        # State tracking
        self.last_update_date: Optional[str] = None
        self.current_edge_index: Optional[torch.Tensor] = None
        self.current_edge_weight: Optional[torch.Tensor] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
    
    def compute_correlation_matrix(
        self,
        df: pd.DataFrame,
        kdcode_list: List[str],
        end_date: str
    ) -> pd.DataFrame:
        """
        Compute stock return correlation matrix using historical data.
        
        Per paper Section 3.3.2: Use past year of returns for correlation.
        
        Args:
            df: DataFrame with columns ['kdcode', 'dt', 'close', ...]
            kdcode_list: List of stock codes to include
            end_date: Compute correlations using data before this date
            
        Returns:
            Correlation matrix as DataFrame
        """
        df = df.copy()
        
        # Compute daily returns if not present
        if 'prev_close' in df.columns:
            df['daily_return'] = df['close'] / df['prev_close'] - 1
        else:
            df = df.sort_values(['kdcode', 'dt'])
            df['daily_return'] = df.groupby('kdcode')['close'].pct_change()
        
        # Filter to before end_date
        df = df[df['dt'] < end_date]
        
        # Get last lookback_days
        all_dates = sorted(df['dt'].unique())
        if len(all_dates) > self.corr_lookback_days:
            start_date = all_dates[-self.corr_lookback_days]
            df = df[df['dt'] >= start_date]
        
        # Filter to our stocks
        df = df[df['kdcode'].isin(kdcode_list)]
        
        # Pivot to get returns matrix (dates x stocks)
        pivot = df.pivot_table(index='dt', columns='kdcode', values='daily_return')
        
        # Ensure column order matches kdcode_list
        pivot = pivot.reindex(columns=kdcode_list)
        
        # Fill missing values with 0
        pivot = pivot.fillna(0)
        
        # Compute correlation matrix
        corr_matrix = pivot.corr()
        
        return corr_matrix
    
    def build_edges(
        self,
        corr_matrix: pd.DataFrame,
        kdcode_list: List[str],
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph edges from correlation matrix.
        
        Creates undirected edges between stocks with correlation > judge_value.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            kdcode_list: List of stock codes (determines node indices)
            show_progress: Whether to show progress bar
            
        Returns:
            edge_index: (2, num_edges) tensor of edge indices
            edge_weight: (num_edges,) tensor of edge weights
        """
        matrix_values = corr_matrix.values.tolist()
        n_stocks = len(kdcode_list)
        
        edge_index = []
        edge_weight = []
        
        iterator = range(n_stocks)
        if show_progress:
            iterator = tqdm(iterator, desc="Building graph edges")
        
        for i in iterator:
            for j in range(i + 1, n_stocks):
                weight = matrix_values[i][j]
                # Check for valid weight (not NaN) and above threshold
                if not np.isnan(weight) and weight > self.judge_value:
                    # Add both directions (undirected graph)
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_weight.append(weight)
                    edge_weight.append(weight)
        
        # Handle empty edge case
        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        return edge_index, edge_weight
    
    def build_graph(
        self,
        df: pd.DataFrame,
        kdcode_list: List[str],
        end_date: str,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build complete graph from data.
        
        Args:
            df: DataFrame with stock data
            kdcode_list: List of stock codes
            end_date: Compute correlations using data before this date
            show_progress: Whether to show progress bar
            
        Returns:
            edge_index: (2, num_edges) tensor
            edge_weight: (num_edges,) tensor
        """
        print(f"Building graph (judge_value={self.judge_value}, lookback={self.corr_lookback_days} days)...")
        
        # Compute correlation matrix
        self.correlation_matrix = self.compute_correlation_matrix(df, kdcode_list, end_date)
        
        # Build edges
        edge_index, edge_weight = self.build_edges(
            self.correlation_matrix, kdcode_list, show_progress
        )
        
        # Store state
        self.last_update_date = end_date
        self.current_edge_index = edge_index
        self.current_edge_weight = edge_weight
        
        print(f"  Graph built: {edge_index.shape[1]} edges for {len(kdcode_list)} nodes")
        
        return edge_index, edge_weight
    
    def should_update(self, current_date: str) -> bool:
        """
        Check if graph should be updated based on time elapsed.
        
        Args:
            current_date: Current date string (YYYY-MM-DD)
            
        Returns:
            True if graph should be updated
        """
        if self.update_frequency_months == 0:
            return False
        
        if self.last_update_date is None:
            return True
        
        # Parse dates
        try:
            last_update = datetime.strptime(self.last_update_date, '%Y-%m-%d')
            current = datetime.strptime(current_date, '%Y-%m-%d')
        except ValueError:
            # Try alternate format
            last_update = pd.to_datetime(self.last_update_date)
            current = pd.to_datetime(current_date)
        
        # Calculate months elapsed
        months_elapsed = (current.year - last_update.year) * 12 + (current.month - last_update.month)
        
        return months_elapsed >= self.update_frequency_months
    
    def update_if_needed(
        self,
        df: pd.DataFrame,
        kdcode_list: List[str],
        current_date: str,
        show_progress: bool = False
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Update graph if needed based on update frequency.
        
        Args:
            df: DataFrame with stock data
            kdcode_list: List of stock codes
            current_date: Current date
            show_progress: Whether to show progress bar
            
        Returns:
            (edge_index, edge_weight) if updated, (None, None) if not
        """
        if not self.should_update(current_date):
            return None, None
        
        print(f"Updating graph (last update: {self.last_update_date}, current: {current_date})")
        return self.build_graph(df, kdcode_list, current_date, show_progress)
    
    def get_current_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current graph tensors.
        
        Returns:
            (edge_index, edge_weight) tuple
            
        Raises:
            ValueError: If graph has not been built yet
        """
        if self.current_edge_index is None or self.current_edge_weight is None:
            raise ValueError("Graph has not been built yet. Call build_graph() first.")
        return self.current_edge_index, self.current_edge_weight
    
    def get_update_dates(
        self,
        start_date: str,
        end_date: str
    ) -> List[str]:
        """
        Get list of dates when graph should be updated.
        
        Useful for planning dynamic updates during training.
        
        Args:
            start_date: Start of period (YYYY-MM-DD)
            end_date: End of period (YYYY-MM-DD)
            
        Returns:
            List of update dates
        """
        if self.update_frequency_months == 0:
            return [start_date]  # Just initial build
        
        update_dates = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            update_dates.append(current.strftime('%Y-%m-%d'))
            current = current + relativedelta(months=self.update_frequency_months)
        
        return update_dates
    
    def get_stats(self) -> dict:
        """
        Get statistics about current graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if self.current_edge_index is None:
            return {"built": False}
        
        n_edges = self.current_edge_index.shape[1]
        n_unique_edges = n_edges // 2  # Undirected, so divide by 2
        
        stats = {
            "built": True,
            "last_update_date": self.last_update_date,
            "n_edges": n_edges,
            "n_unique_edges": n_unique_edges,
            "judge_value": self.judge_value,
            "update_frequency_months": self.update_frequency_months,
        }
        
        if self.current_edge_weight is not None and len(self.current_edge_weight) > 0:
            stats["avg_edge_weight"] = float(self.current_edge_weight.mean())
            stats["min_edge_weight"] = float(self.current_edge_weight.min())
            stats["max_edge_weight"] = float(self.current_edge_weight.max())
        
        return stats


def create_graph_builder_from_config(config: dict) -> GraphBuilder:
    """
    Create GraphBuilder from configuration dictionary.
    
    Args:
        config: Configuration dict with keys:
            - judge_value: float
            - update_frequency_months: int
            - corr_lookback_days: int
            
    Returns:
        Configured GraphBuilder instance
    """
    return GraphBuilder(
        judge_value=config.get('judge_value', 0.8),
        update_frequency_months=config.get('update_frequency_months', 0),
        corr_lookback_days=config.get('corr_lookback_days', 252)
    )
