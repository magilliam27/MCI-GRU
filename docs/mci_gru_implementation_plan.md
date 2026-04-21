# MCI-GRU Implementation Plan for S&P 500

## Overview

This document provides a step-by-step implementation plan for the MCI-GRU stock prediction model on the S&P 500 universe in Google Colab. The model combines an attention-enhanced GRU, Graph Attention Networks, and multi-head cross-attention for latent market state learning.

**Paper Reference:** "MCI-GRU: Stock Prediction Model Based on Multi-Head Cross-Attention and Improved GRU" (Neurocomputing 2025)

**Repo note (April 2026):** The maintained training stack lives under `mci_gru/` and `run_experiment.py` (Hydra configs, **AdamW** + warmup/cosine LR, optional AMP, IC-aware checkpointing, combined loss defaults, MLflow on by default). Treat this Colab-oriented plan as a **pedagogical / paper-aligned** walkthrough; for production behaviour see `docs/ARCHITECTURE.md` and `docs/CONFIGURATION_GUIDE.md`.

---

## 1. Environment Setup

### 1.1 Install Dependencies

```python
!pip install yfinance pandas numpy torch torch-geometric scikit-learn matplotlib tqdm
```

### 1.2 Required Imports

```python
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### 1.3 (Optional) Generate S&P 500 Tickers CSV

If you don't already have a tickers CSV, you can generate one for the S&P 500:

```python
def generate_sp500_tickers_csv(output_path='/content/sp500_tickers.csv'):
    """
    Fetch S&P 500 constituents from Wikipedia and save to CSV.
    Run this once to create your tickers file.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]
    
    # Create clean DataFrame with ticker and company name
    tickers_df = pd.DataFrame({
        'ticker': sp500_table['Symbol'].str.replace('.', '-', regex=False),
        'company_name': sp500_table['Security'],
        'sector': sp500_table['GICS Sector']
    })
    
    # Save to CSV
    tickers_df.to_csv(output_path, index=False)
    print(f"Saved {len(tickers_df)} tickers to {output_path}")
    
    return output_path

# Uncomment to generate:
# TICKER_CSV_PATH = generate_sp500_tickers_csv()
```

---

## 2. Data Acquisition & Preprocessing

### 2.1 Load Tickers from CSV

The pipeline expects a CSV file containing stock tickers. The CSV should have a column named `ticker` (or `symbol`, `Ticker`, `Symbol` - the loader will auto-detect).

**Example CSV format (`tickers.csv`):**
```
ticker
AAPL
MSFT
GOOGL
AMZN
...
```

**Alternative formats also supported:**
```
symbol,company_name
AAPL,Apple Inc.
MSFT,Microsoft Corporation
...
```

```python
def load_tickers_from_csv(filepath):
    """
    Load ticker symbols from a CSV file.
    
    Args:
        filepath: Path to CSV file containing tickers
        
    Returns:
        list: List of ticker symbols
        
    Expected CSV format:
        - Must have a column named 'ticker', 'symbol', 'Ticker', or 'Symbol'
        - One ticker per row
    """
    df = pd.read_csv(filepath)
    
    # Auto-detect ticker column
    possible_columns = ['ticker', 'symbol', 'Ticker', 'Symbol', 'TICKER', 'SYMBOL']
    ticker_column = None
    
    for col in possible_columns:
        if col in df.columns:
            ticker_column = col
            break
    
    if ticker_column is None:
        # If no standard column name, assume first column contains tickers
        ticker_column = df.columns[0]
        print(f"Warning: No standard ticker column found. Using first column: '{ticker_column}'")
    
    tickers = df[ticker_column].astype(str).tolist()
    
    # Clean tickers (replace '.' with '-' for yfinance compatibility, e.g., BRK.B -> BRK-B)
    tickers = [t.strip().replace('.', '-') for t in tickers]
    
    # Remove any empty or invalid tickers
    tickers = [t for t in tickers if t and t != 'nan']
    
    print(f"Loaded {len(tickers)} tickers from {filepath}")
    
    return tickers
```

**Upload your ticker CSV to Colab:**
```python
# Option 1: Upload via Colab UI
from google.colab import files
uploaded = files.upload()  # This opens a file picker
TICKER_CSV_PATH = list(uploaded.keys())[0]

# Option 2: If CSV is in Google Drive
from google.colab import drive
drive.mount('/content/drive')
TICKER_CSV_PATH = '/content/drive/MyDrive/your_folder/tickers.csv'

# Option 3: Direct path if already uploaded
TICKER_CSV_PATH = '/content/tickers.csv'
```

### 2.2 Download Historical Data

```python
def download_stock_data(tickers, start_date, end_date):
    """
    Download OHLCV data for all tickers.
    
    Returns:
        dict: {ticker: DataFrame with columns [Open, High, Low, Close, Volume]}
    """
    stock_data = {}
    failed_tickers = []
    
    for ticker in tqdm(tickers, desc="Downloading stock data"):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 200:  # Require minimum trading days
                stock_data[ticker] = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        except Exception as e:
            failed_tickers.append(ticker)
    
    print(f"Successfully downloaded: {len(stock_data)} stocks")
    print(f"Failed: {len(failed_tickers)} stocks")
    return stock_data
```

### 2.3 Data Configuration

```python
# ==============================================
# CONFIGURATION - UPDATE THESE PATHS AS NEEDED
# ==============================================

# Path to your ticker CSV file
TICKER_CSV_PATH = '/content/tickers.csv'  # <-- UPDATE THIS

# Time periods (following paper's setup)
TRAIN_START = "2018-01-01"
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2023-12-31"

# Download with buffer for correlation calculation
DATA_START = "2017-01-01"  # 1 year buffer for correlation
DATA_END = "2023-12-31"
```

### 2.4 Feature Engineering

```python
def compute_features(stock_data):
    """
    Compute 6 features per stock per day:
    1. Open price (normalized)
    2. Close price (normalized)
    3. High price (normalized)
    4. Low price (normalized)
    5. Volume (normalized)
    6. Turnover proxy (Close * Volume, normalized)
    
    Also compute daily returns as labels.
    """
    processed_data = {}
    
    for ticker, df in stock_data.items():
        df = df.copy()
        
        # Compute turnover proxy (Close * Volume)
        df['Turnover'] = df['Close'] * df['Volume']
        
        # Compute daily returns (label)
        df['Return'] = df['Close'].pct_change()
        
        # Drop NaN rows
        df = df.dropna()
        
        # Feature columns
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
        
        processed_data[ticker] = df
    
    return processed_data
```

### 2.5 Align Stock Data to Common Dates

```python
def align_stock_data(processed_data):
    """
    Align all stocks to common trading dates.
    Remove stocks with missing data on common dates.
    """
    # Find common dates across all stocks
    date_sets = [set(df.index) for df in processed_data.values()]
    common_dates = sorted(set.intersection(*date_sets))
    
    aligned_data = {}
    for ticker, df in processed_data.items():
        aligned_data[ticker] = df.loc[common_dates].copy()
    
    print(f"Common trading dates: {len(common_dates)}")
    print(f"Aligned stocks: {len(aligned_data)}")
    
    return aligned_data, common_dates
```

### 2.6 Normalize Features

```python
def normalize_features(aligned_data, train_end_date):
    """
    Z-score normalize features using training period statistics only.
    This prevents look-ahead bias.
    """
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    
    # Compute normalization stats from training period
    train_data = []
    for ticker, df in aligned_data.items():
        train_df = df.loc[:train_end_date, feature_cols]
        train_data.append(train_df.values)
    
    train_array = np.vstack(train_data)
    means = train_array.mean(axis=0)
    stds = train_array.std(axis=0)
    stds[stds == 0] = 1  # Prevent division by zero
    
    # Apply normalization to all data
    normalized_data = {}
    for ticker, df in aligned_data.items():
        df_norm = df.copy()
        df_norm[feature_cols] = (df[feature_cols].values - means) / stds
        normalized_data[ticker] = df_norm
    
    return normalized_data, means, stds
```

---

## 3. Graph Construction (GAT Adjacency)

### 3.1 Compute Correlation Matrix

```python
def compute_correlation_matrix(normalized_data, end_date, lookback_days=252):
    """
    Compute Pearson correlation matrix based on trailing returns.
    
    Args:
        normalized_data: dict of DataFrames
        end_date: date to compute correlation as of
        lookback_days: number of trading days to look back (default 252 = 1 year)
    
    Returns:
        correlation_matrix: np.array of shape (N, N)
        tickers: list of ticker symbols in order
    """
    tickers = sorted(normalized_data.keys())
    
    # Collect returns for lookback period
    returns_dict = {}
    for ticker in tickers:
        df = normalized_data[ticker]
        mask = df.index <= end_date
        returns = df.loc[mask, 'Return'].iloc[-lookback_days:]
        returns_dict[ticker] = returns.values
    
    # Build returns matrix (days x stocks)
    returns_matrix = np.column_stack([returns_dict[t] for t in tickers])
    
    # Compute Pearson correlation
    correlation_matrix = np.corrcoef(returns_matrix.T)
    
    # Handle NaN (set to 0)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    
    return correlation_matrix, tickers
```

### 3.2 Build Edge Index with Threshold

```python
def build_edge_index(correlation_matrix, judge_value=0.8):
    """
    Build PyTorch Geometric edge_index from correlation matrix.
    
    Args:
        correlation_matrix: np.array of shape (N, N)
        judge_value: threshold for including edges (default 0.8)
    
    Returns:
        edge_index: torch.LongTensor of shape (2, E)
        edge_weight: torch.FloatTensor of shape (E,)
    """
    N = correlation_matrix.shape[0]
    
    # Find edges above threshold (excluding self-loops)
    sources = []
    targets = []
    weights = []
    
    for i in range(N):
        for j in range(N):
            if i != j and correlation_matrix[i, j] >= judge_value:
                sources.append(i)
                targets.append(j)
                weights.append(correlation_matrix[i, j])
    
    if len(sources) == 0:
        # If no edges pass threshold, use top-k connections per node
        print(f"Warning: No edges pass threshold {judge_value}. Using top-5 connections per node.")
        k = 5
        for i in range(N):
            corrs = correlation_matrix[i].copy()
            corrs[i] = -np.inf  # Exclude self
            top_k_idx = np.argsort(corrs)[-k:]
            for j in top_k_idx:
                sources.append(i)
                targets.append(j)
                weights.append(correlation_matrix[i, j])
    
    edge_index = torch.LongTensor([sources, targets])
    edge_weight = torch.FloatTensor(weights)
    
    print(f"Graph edges: {edge_index.shape[1]} (avg degree: {edge_index.shape[1] / N:.2f})")
    
    return edge_index, edge_weight
```

---

## 4. Dataset Construction

### 4.1 Create Sliding Window Samples

```python
def create_dataset(normalized_data, tickers, dates, hist_days=10, label_days=5):
    """
    Create dataset with sliding window approach.
    
    Args:
        normalized_data: dict of normalized DataFrames
        tickers: list of tickers in consistent order
        dates: list of dates to create samples for
        hist_days: number of historical days as input (default 10)
        label_days: prediction horizon in days (default 5)
    
    Returns:
        samples: list of dicts with 'features', 'labels', 'date'
    """
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    N = len(tickers)
    dx = len(feature_cols)
    
    # Build aligned arrays
    all_dates = normalized_data[tickers[0]].index.tolist()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    
    # Stack all data: (T, N, dx)
    data_array = np.stack([
        normalized_data[t][feature_cols].values for t in tickers
    ], axis=1)
    
    # Stack returns: (T, N)
    returns_array = np.stack([
        normalized_data[t]['Return'].values for t in tickers
    ], axis=1)
    
    samples = []
    
    for date in dates:
        if date not in date_to_idx:
            continue
        
        t = date_to_idx[date]
        
        # Check if we have enough history and future
        if t < hist_days or t + label_days >= len(all_dates):
            continue
        
        # Features: (hist_days, N, dx)
        features = data_array[t - hist_days:t]
        
        # Labels: average return over next label_days for each stock
        future_returns = returns_array[t:t + label_days]
        labels = future_returns.mean(axis=0)  # (N,)
        
        samples.append({
            'features': torch.FloatTensor(features),  # (hist_days, N, dx)
            'labels': torch.FloatTensor(labels),      # (N,)
            'date': date
        })
    
    print(f"Created {len(samples)} samples")
    return samples
```

### 4.2 Split into Train/Val/Test

```python
def split_dataset(samples, train_end, val_end):
    """Split samples by date into train/val/test sets."""
    train_samples = [s for s in samples if s['date'] <= pd.Timestamp(train_end)]
    val_samples = [s for s in samples if pd.Timestamp(train_end) < s['date'] <= pd.Timestamp(val_end)]
    test_samples = [s for s in samples if s['date'] > pd.Timestamp(val_end)]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    return train_samples, val_samples, test_samples
```

---

## 5. Model Architecture

### 5.1 Improved GRU with Attention Reset Gate

```python
class AttentionResetGRUCell(nn.Module):
    """
    GRU cell with attention mechanism replacing the reset gate.
    
    Instead of: r_t = sigmoid(W_r * x_t + U_r * h_{t-1})
    We use:     r'_t = Attention(h_{t-1}, x_t)
    """
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Update gate (unchanged from standard GRU)
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size)
        
        # Attention mechanism (replaces reset gate)
        self.W_q = nn.Linear(hidden_size, hidden_size)  # Query from h_{t-1}
        self.W_k = nn.Linear(input_size, hidden_size)   # Key from x_t
        self.W_v = nn.Linear(input_size, hidden_size)   # Value from x_t
        
        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x_t, h_prev):
        """
        Args:
            x_t: input at time t, shape (batch, input_size)
            h_prev: hidden state from t-1, shape (batch, hidden_size)
        
        Returns:
            h_t: new hidden state, shape (batch, hidden_size)
        """
        # Update gate (standard)
        z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_prev))
        
        # Attention-based reset gate
        q_t = self.W_q(h_prev)  # (batch, hidden_size)
        k_t = self.W_k(x_t)     # (batch, hidden_size)
        v_t = self.W_v(x_t)     # (batch, hidden_size)
        
        # Scaled dot-product attention score
        attn_score = torch.sum(q_t * k_t, dim=-1, keepdim=True) / np.sqrt(self.hidden_size)
        alpha_t = torch.softmax(attn_score, dim=-1)
        
        # Attention-weighted value as reset signal
        r_prime_t = alpha_t * v_t  # (batch, hidden_size)
        
        # Candidate hidden state
        h_tilde = torch.tanh(self.W_h(x_t) + r_prime_t * self.U_h(h_prev))
        
        # Final hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t


class ImprovedGRU(nn.Module):
    """
    Multi-layer Improved GRU for temporal feature extraction.
    """
    
    def __init__(self, input_size, hidden_sizes=[32, 10]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(AttentionResetGRUCell(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.output_size = hidden_sizes[-1]
    
    def forward(self, x):
        """
        Args:
            x: input sequence, shape (batch, seq_len, input_size)
        
        Returns:
            output: final hidden state, shape (batch, output_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Process through each layer
        layer_input = x
        for layer in self.layers:
            hidden_size = layer.hidden_size
            h = torch.zeros(batch_size, hidden_size, device=x.device)
            
            outputs = []
            for t in range(seq_len):
                h = layer(layer_input[:, t, :], h)
                outputs.append(h)
            
            layer_input = torch.stack(outputs, dim=1)
        
        # Return final hidden state
        return layer_input[:, -1, :]
```

### 5.2 GAT for Cross-Sectional Features

```python
class CrossSectionalGAT(nn.Module):
    """
    Graph Attention Network for extracting cross-sectional features.
    Two-layer GAT following the paper's architecture.
    """
    
    def __init__(self, input_size, hidden_size=32, output_size=4, num_heads=4):
        super().__init__()
        
        # First GAT layer
        self.gat1 = GATConv(
            in_channels=input_size,
            out_channels=hidden_size,
            heads=num_heads,
            concat=True,
            dropout=0.1
        )
        
        # Second GAT layer
        self.gat2 = GATConv(
            in_channels=hidden_size * num_heads,
            out_channels=output_size,
            heads=1,
            concat=False,
            dropout=0.1
        )
        
        self.output_size = output_size
    
    def forward(self, x, edge_index):
        """
        Args:
            x: node features, shape (N, input_size)
            edge_index: graph connectivity, shape (2, E)
        
        Returns:
            output: node embeddings, shape (N, output_size)
        """
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        
        return x
```

### 5.3 Multi-Head Cross-Attention for Latent States

```python
class MarketLatentStateLearner(nn.Module):
    """
    Multi-head cross-attention mechanism for learning latent market states.
    
    Learns two sets of latent state vectors (R1, R2) that interact with
    temporal features (A1) and cross-sectional features (A2).
    """
    
    def __init__(self, feature_dim, num_latent_states=32, latent_dim=16, num_heads=4):
        super().__init__()
        
        self.num_latent_states = num_latent_states
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Learnable latent state vectors
        self.R1 = nn.Parameter(torch.randn(num_latent_states, feature_dim) * 0.02)
        self.R2 = nn.Parameter(torch.randn(num_latent_states, feature_dim) * 0.02)
        
        # Multi-head attention projections for R1-A1 interaction
        self.W_Q1 = nn.Linear(feature_dim, feature_dim)
        self.W_K1 = nn.Linear(feature_dim, feature_dim)
        self.W_V1 = nn.Linear(feature_dim, feature_dim)
        self.W_O1 = nn.Linear(feature_dim, feature_dim)
        
        # Multi-head attention projections for R2-A2 interaction
        self.W_Q2 = nn.Linear(feature_dim, feature_dim)
        self.W_K2 = nn.Linear(feature_dim, feature_dim)
        self.W_V2 = nn.Linear(feature_dim, feature_dim)
        self.W_O2 = nn.Linear(feature_dim, feature_dim)
    
    def multi_head_cross_attention(self, query, key_value, W_Q, W_K, W_V, W_O):
        """
        Multi-head cross-attention.
        
        Args:
            query: (N, feature_dim) - A1 or A2 as query
            key_value: (num_latent_states, feature_dim) - R1 or R2 as key/value
            W_Q, W_K, W_V, W_O: projection layers
        
        Returns:
            output: (N, feature_dim) - enriched features
        """
        N = query.shape[0]
        
        # Project to Q, K, V
        Q = W_Q(query)      # (N, feature_dim)
        K = W_K(key_value)  # (num_latent_states, feature_dim)
        V = W_V(key_value)  # (num_latent_states, feature_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (heads, N, head_dim)
        K = K.view(self.num_latent_states, self.num_heads, self.head_dim).transpose(0, 1)  # (heads, num_latent, head_dim)
        V = V.view(self.num_latent_states, self.num_heads, self.head_dim).transpose(0, 1)  # (heads, num_latent, head_dim)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (heads, N, num_latent)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (heads, N, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 1).contiguous().view(N, -1)  # (N, feature_dim)
        
        # Output projection
        output = W_O(attn_output)
        
        return output
    
    def forward(self, A1, A2):
        """
        Args:
            A1: temporal features, shape (N, feature_dim)
            A2: cross-sectional features, shape (N, feature_dim)
        
        Returns:
            B1: enriched temporal features, shape (N, feature_dim)
            B2: enriched cross-sectional features, shape (N, feature_dim)
        """
        # Cross-attention between A1 (query) and R1 (key/value)
        B1 = self.multi_head_cross_attention(A1, self.R1, self.W_Q1, self.W_K1, self.W_V1, self.W_O1)
        
        # Cross-attention between A2 (query) and R2 (key/value)
        B2 = self.multi_head_cross_attention(A2, self.R2, self.W_Q2, self.W_K2, self.W_V2, self.W_O2)
        
        return B1, B2
```

### 5.4 Prediction Layer

```python
class PredictionLayer(nn.Module):
    """
    Final prediction layer using GAT on concatenated features.
    """
    
    def __init__(self, input_size, hidden_size=32, num_heads=4):
        super().__init__()
        
        # GAT for final prediction
        self.gat = GATConv(
            in_channels=input_size,
            out_channels=hidden_size,
            heads=num_heads,
            concat=True,
            dropout=0.1
        )
        
        # Output projection to scalar prediction
        self.fc = nn.Linear(hidden_size * num_heads, 1)
    
    def forward(self, Z, edge_index):
        """
        Args:
            Z: concatenated features, shape (N, input_size)
            edge_index: graph connectivity, shape (2, E)
        
        Returns:
            predictions: shape (N,)
        """
        # GAT layer
        x = self.gat(Z, edge_index)
        x = F.elu(x)
        
        # Final prediction
        out = self.fc(x).squeeze(-1)
        
        return out
```

### 5.5 Complete MCI-GRU Model

```python
class MCIGRU(nn.Module):
    """
    Complete MCI-GRU model integrating all components.
    """
    
    def __init__(
        self,
        num_stocks,
        input_features=6,
        gru_hidden_sizes=[32, 10],
        gat_hidden_size=32,
        gat_output_size=4,
        gat_heads=4,
        num_latent_states=32,
        latent_dim=16,
        cross_attn_heads=4
    ):
        super().__init__()
        
        self.num_stocks = num_stocks
        
        # Part (a): Improved GRU for temporal features
        self.temporal_gru = ImprovedGRU(input_features, gru_hidden_sizes)
        gru_output_size = self.temporal_gru.output_size
        
        # Part (b): GAT for cross-sectional features
        self.cross_sectional_gat = CrossSectionalGAT(
            input_size=input_features,
            hidden_size=gat_hidden_size,
            output_size=gat_output_size,
            num_heads=gat_heads
        )
        
        # Projection layers to align dimensions
        self.proj_temporal = nn.Linear(gru_output_size, gat_hidden_size)
        self.proj_cross = nn.Linear(gat_output_size, gat_hidden_size)
        
        # Part (c): Multi-head cross-attention for latent states
        self.latent_learner = MarketLatentStateLearner(
            feature_dim=gat_hidden_size,
            num_latent_states=num_latent_states,
            latent_dim=latent_dim,
            num_heads=cross_attn_heads
        )
        
        # Part (d): Prediction layer
        # Concatenate A1, A2, B1, B2 -> 4 * gat_hidden_size
        concat_size = 4 * gat_hidden_size
        self.prediction_layer = PredictionLayer(
            input_size=concat_size,
            hidden_size=gat_hidden_size,
            num_heads=gat_heads
        )
    
    def forward(self, x, edge_index):
        """
        Args:
            x: input features, shape (seq_len, N, input_features)
            edge_index: graph connectivity, shape (2, E)
        
        Returns:
            predictions: shape (N,)
        """
        seq_len, N, dx = x.shape
        
        # Part (a): Temporal features via Improved GRU
        # Reshape for GRU: (N, seq_len, dx)
        x_temporal = x.permute(1, 0, 2)
        A1_raw = self.temporal_gru(x_temporal)  # (N, gru_output_size)
        A1 = self.proj_temporal(A1_raw)         # (N, gat_hidden_size)
        
        # Part (b): Cross-sectional features via GAT
        # Use the last time step's features
        x_last = x[-1]  # (N, dx)
        A2_raw = self.cross_sectional_gat(x_last, edge_index)  # (N, gat_output_size)
        A2 = self.proj_cross(A2_raw)                            # (N, gat_hidden_size)
        
        # Part (c): Latent state learning via multi-head cross-attention
        B1, B2 = self.latent_learner(A1, A2)  # Both (N, gat_hidden_size)
        
        # Part (d): Concatenate and predict
        Z = torch.cat([A1, A2, B1, B2], dim=-1)  # (N, 4 * gat_hidden_size)
        predictions = self.prediction_layer(Z, edge_index)  # (N,)
        
        return predictions
```

---

## 6. Training Loop

### 6.1 Training Configuration

```python
# Hyperparameters (from paper)
CONFIG = {
    'hist_days': 10,           # Historical window
    'label_days': 5,           # Prediction horizon
    'judge_value': 0.8,        # Correlation threshold
    'batch_size': 32,
    'learning_rate': 0.0002,
    'num_epochs': 100,
    'early_stopping_patience': 10,
    'gru_hidden_sizes': [32, 10],
    'gat_hidden_size': 32,
    'gat_output_size': 4,
    'gat_heads': 4,
    'num_latent_states': 32,
    'latent_dim': 16,
    'cross_attn_heads': 4,
    'top_k': 10,               # Top-k stocks for portfolio
}
```

### 6.2 Training Function

```python
def train_model(model, train_samples, val_samples, edge_index, config, device):
    """
    Train the MCI-GRU model.
    """
    model = model.to(device)
    edge_index = edge_index.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        # Shuffle training samples
        np.random.shuffle(train_samples)
        
        for sample in train_samples:
            features = sample['features'].to(device)  # (seq_len, N, dx)
            labels = sample['labels'].to(device)      # (N,)
            
            optimizer.zero_grad()
            predictions = model(features, edge_index)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_samples)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for sample in val_samples:
                features = sample['features'].to(device)
                labels = sample['labels'].to(device)
                
                predictions = model(features, edge_index)
                loss = criterion(predictions, labels)
                epoch_val_loss += loss.item()
        
        epoch_val_loss /= len(val_samples)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Train Loss = {epoch_train_loss:.6f}, Val Loss = {epoch_val_loss:.6f}")
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    return model, train_losses, val_losses
```

---

## 7. Evaluation & Backtesting

### 7.1 Generate Predictions

```python
def generate_predictions(model, samples, edge_index, device):
    """
    Generate predictions for all samples.
    
    Returns:
        results: list of dicts with 'date', 'predictions', 'labels'
    """
    model.eval()
    edge_index = edge_index.to(device)
    results = []
    
    with torch.no_grad():
        for sample in samples:
            features = sample['features'].to(device)
            labels = sample['labels'].to(device)
            
            predictions = model(features, edge_index)
            
            results.append({
                'date': sample['date'],
                'predictions': predictions.cpu().numpy(),
                'labels': labels.cpu().numpy()
            })
    
    return results
```

### 7.2 Backtest Strategy

```python
def backtest_strategy(results, tickers, top_k=10):
    """
    Backtest the top-k stock selection strategy.
    
    Strategy:
    - At each time step, select top-k stocks by predicted return
    - Equal-weight portfolio
    - Rebalance daily
    
    Returns:
        portfolio_returns: pd.Series of daily returns
        metrics: dict of performance metrics
    """
    portfolio_returns = []
    dates = []
    
    for result in results:
        date = result['date']
        preds = result['predictions']
        actual_returns = result['labels']
        
        # Select top-k stocks by prediction
        top_k_indices = np.argsort(preds)[-top_k:]
        
        # Equal-weight portfolio return
        portfolio_return = np.mean(actual_returns[top_k_indices])
        
        portfolio_returns.append(portfolio_return)
        dates.append(date)
    
    # Create return series
    returns_series = pd.Series(portfolio_returns, index=dates)
    
    return returns_series
```

### 7.3 Performance Metrics

```python
def calculate_metrics(returns_series):
    """
    Calculate all performance metrics from the paper.
    
    Metrics:
    - ARR: Annualized Rate of Return
    - AVol: Annualized Volatility
    - MDD: Maximum Drawdown
    - ASR: Annualized Sharpe Ratio
    - CR: Calmar Ratio
    - IR: Information Ratio
    """
    # Convert to numpy array
    returns = returns_series.values
    
    # Number of trading days
    T = len(returns)
    
    # Cumulative returns
    cumulative = np.cumprod(1 + returns)
    
    # ARR: Annualized Rate of Return
    total_return = cumulative[-1] - 1
    ARR = (1 + total_return) ** (252 / T) - 1
    
    # AVol: Annualized Volatility
    AVol = np.std(returns) * np.sqrt(252)
    
    # MDD: Maximum Drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    MDD = np.min(drawdown)
    
    # ASR: Annualized Sharpe Ratio (assuming risk-free rate = 0)
    ASR = ARR / AVol if AVol > 0 else 0
    
    # CR: Calmar Ratio
    CR = ARR / abs(MDD) if MDD != 0 else 0
    
    # IR: Information Ratio (using daily returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    IR = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    # MSE and MAE (prediction error metrics)
    # These need actual vs predicted, computed separately
    
    metrics = {
        'ARR': ARR,
        'AVol': AVol,
        'MDD': MDD,
        'ASR': ASR,
        'CR': CR,
        'IR': IR,
        'Total_Return': total_return,
        'Num_Days': T
    }
    
    return metrics


def calculate_prediction_metrics(results):
    """Calculate MSE and MAE for predictions."""
    all_preds = []
    all_labels = []
    
    for result in results:
        all_preds.extend(result['predictions'])
        all_labels.extend(result['labels'])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))
    
    return {'MSE': mse, 'MAE': mae}
```

### 7.4 Visualization

```python
def plot_results(returns_series, benchmark_returns=None, title="MCI-GRU Portfolio Performance"):
    """
    Plot cumulative returns and drawdown.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Cumulative returns
    cumulative = (1 + returns_series).cumprod()
    axes[0].plot(cumulative.index, cumulative.values, label='MCI-GRU Strategy', color='red')
    
    if benchmark_returns is not None:
        bench_cumulative = (1 + benchmark_returns).cumprod()
        axes[0].plot(bench_cumulative.index, bench_cumulative.values, 
                     label='S&P 500 Benchmark', color='blue', alpha=0.7)
    
    axes[0].set_title(title)
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Drawdown
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    axes[1].plot(drawdown.index, drawdown.values, color='red', linewidth=0.5)
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('portfolio_performance.png', dpi=150)
    plt.show()


def print_metrics(metrics, pred_metrics, title="Performance Metrics"):
    """Pretty print all metrics."""
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50)
    print(f"  ARR (Annualized Return):     {metrics['ARR']*100:>8.2f}%")
    print(f"  AVol (Annualized Volatility): {metrics['AVol']*100:>8.2f}%")
    print(f"  MDD (Maximum Drawdown):       {metrics['MDD']*100:>8.2f}%")
    print(f"  ASR (Sharpe Ratio):           {metrics['ASR']:>8.3f}")
    print(f"  CR (Calmar Ratio):            {metrics['CR']:>8.3f}")
    print(f"  IR (Information Ratio):       {metrics['IR']:>8.3f}")
    print("-"*50)
    print(f"  MSE:                          {pred_metrics['MSE']:>8.6f}")
    print(f"  MAE:                          {pred_metrics['MAE']:>8.6f}")
    print("="*50 + "\n")
```

---

## 8. Main Execution Pipeline

```python
def main(ticker_csv_path):
    """
    Main execution pipeline for MCI-GRU on S&P 500.
    
    Args:
        ticker_csv_path: Path to CSV file containing ticker symbols
    """
    print("="*60)
    print("  MCI-GRU Stock Prediction Model - S&P 500")
    print("="*60)
    
    # =============================================
    # Step 1: Data Acquisition
    # =============================================
    print("\n[Step 1] Loading tickers from CSV...")
    tickers = load_tickers_from_csv(ticker_csv_path)
    
    print(f"\n[Step 1] Downloading historical data...")
    stock_data = download_stock_data(tickers, DATA_START, DATA_END)
    
    # =============================================
    # Step 2: Data Preprocessing
    # =============================================
    print("\n[Step 2] Computing features...")
    processed_data = compute_features(stock_data)
    
    print("\n[Step 2] Aligning stock data...")
    aligned_data, common_dates = align_stock_data(processed_data)
    tickers = sorted(aligned_data.keys())  # Update ticker list
    
    print("\n[Step 2] Normalizing features...")
    normalized_data, means, stds = normalize_features(aligned_data, TRAIN_END)
    
    # =============================================
    # Step 3: Graph Construction
    # =============================================
    print("\n[Step 3] Computing correlation matrix...")
    corr_matrix, tickers = compute_correlation_matrix(normalized_data, TRAIN_END)
    
    print(f"\n[Step 3] Building graph with judge_value={CONFIG['judge_value']}...")
    edge_index, edge_weight = build_edge_index(corr_matrix, CONFIG['judge_value'])
    
    # =============================================
    # Step 4: Dataset Creation
    # =============================================
    print("\n[Step 4] Creating dataset samples...")
    all_dates = [d for d in common_dates if d >= pd.Timestamp(TRAIN_START)]
    samples = create_dataset(
        normalized_data, tickers, all_dates,
        hist_days=CONFIG['hist_days'],
        label_days=CONFIG['label_days']
    )
    
    print("\n[Step 4] Splitting into train/val/test...")
    train_samples, val_samples, test_samples = split_dataset(samples, TRAIN_END, VAL_END)
    
    # =============================================
    # Step 5: Model Initialization
    # =============================================
    print("\n[Step 5] Initializing MCI-GRU model...")
    model = MCIGRU(
        num_stocks=len(tickers),
        input_features=6,
        gru_hidden_sizes=CONFIG['gru_hidden_sizes'],
        gat_hidden_size=CONFIG['gat_hidden_size'],
        gat_output_size=CONFIG['gat_output_size'],
        gat_heads=CONFIG['gat_heads'],
        num_latent_states=CONFIG['num_latent_states'],
        latent_dim=CONFIG['latent_dim'],
        cross_attn_heads=CONFIG['cross_attn_heads']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # =============================================
    # Step 6: Training
    # =============================================
    print("\n[Step 6] Training model...")
    model, train_losses, val_losses = train_model(
        model, train_samples, val_samples, edge_index, CONFIG, device
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    
    # =============================================
    # Step 7: Evaluation on Test Set
    # =============================================
    print("\n[Step 7] Generating test predictions...")
    test_results = generate_predictions(model, test_samples, edge_index, device)
    
    print("\n[Step 7] Running backtest...")
    test_returns = backtest_strategy(test_results, tickers, top_k=CONFIG['top_k'])
    
    # Calculate metrics
    metrics = calculate_metrics(test_returns)
    pred_metrics = calculate_prediction_metrics(test_results)
    
    # Print results
    print_metrics(metrics, pred_metrics, "Test Set Performance (2023)")
    
    # =============================================
    # Step 8: Visualization
    # =============================================
    print("\n[Step 8] Generating visualizations...")
    
    # Get S&P 500 benchmark returns for comparison
    spy = yf.download('SPY', start=TEST_START, end=TEST_END, progress=False)
    spy_returns = spy['Close'].pct_change().dropna()
    # Align dates
    common_test_dates = test_returns.index.intersection(spy_returns.index)
    spy_returns_aligned = spy_returns.loc[common_test_dates]
    test_returns_aligned = test_returns.loc[common_test_dates]
    
    plot_results(test_returns_aligned, spy_returns_aligned)
    
    # =============================================
    # Step 9: Multiple Runs for Robustness
    # =============================================
    print("\n[Step 9] Running multiple trials for robustness...")
    NUM_TRIALS = 10
    all_metrics = []
    
    for trial in range(NUM_TRIALS):
        print(f"   Trial {trial+1}/{NUM_TRIALS}...")
        
        # Reinitialize model
        model_trial = MCIGRU(
            num_stocks=len(tickers),
            input_features=6,
            gru_hidden_sizes=CONFIG['gru_hidden_sizes'],
            gat_hidden_size=CONFIG['gat_hidden_size'],
            gat_output_size=CONFIG['gat_output_size'],
            gat_heads=CONFIG['gat_heads'],
            num_latent_states=CONFIG['num_latent_states'],
            latent_dim=CONFIG['latent_dim'],
            cross_attn_heads=CONFIG['cross_attn_heads']
        )
        
        # Train
        model_trial, _, _ = train_model(
            model_trial, train_samples, val_samples, edge_index, CONFIG, device
        )
        
        # Evaluate
        results = generate_predictions(model_trial, test_samples, edge_index, device)
        returns = backtest_strategy(results, tickers, top_k=CONFIG['top_k'])
        trial_metrics = calculate_metrics(returns)
        all_metrics.append(trial_metrics)
    
    # Average metrics
    print("\n" + "="*50)
    print("  Average Metrics Over 10 Trials")
    print("="*50)
    for key in ['ARR', 'AVol', 'MDD', 'ASR', 'CR', 'IR']:
        values = [m[key] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        if key in ['ARR', 'AVol', 'MDD']:
            print(f"  {key}: {mean_val*100:.2f}% ± {std_val*100:.2f}%")
        else:
            print(f"  {key}: {mean_val:.3f} ± {std_val:.3f}")
    print("="*50)
    
    print("\n✓ Pipeline complete!")
    
    return model, test_returns, metrics


# =============================================
# Run the pipeline
# =============================================
if __name__ == "__main__":
    # Set path to your ticker CSV file
    TICKER_CSV_PATH = '/content/tickers.csv'  # <-- UPDATE THIS PATH
    
    model, test_returns, metrics = main(TICKER_CSV_PATH)
```

---

## 9. Appendix: Hyperparameter Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hist_days` | 10 | Number of historical days as input |
| `label_days` | 5 | Prediction horizon (days) |
| `judge_value` | 0.8 | Correlation threshold for graph edges |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.0002 | Adam optimizer learning rate |
| `gru_hidden_sizes` | [32, 10] | Hidden layer sizes for Improved GRU |
| `gat_hidden_size` | 32 | Hidden dimension for GAT layers |
| `gat_output_size` | 4 | Output dimension for cross-sectional GAT |
| `gat_heads` | 4 | Number of attention heads in GAT |
| `num_latent_states` | 32 | Number of learnable market latent vectors |
| `latent_dim` | 16 | Dimension of each latent state |
| `cross_attn_heads` | 4 | Number of heads in cross-attention |
| `top_k` | 10 | Number of stocks in portfolio |

---

## 10. Expected Output

When run successfully, the pipeline should produce:

1. **Training curves plot** (`training_curves.png`)
2. **Portfolio performance plot** (`portfolio_performance.png`) showing:
   - Cumulative returns vs S&P 500 benchmark
   - Drawdown chart
3. **Console output** with metrics:
   - Single run metrics
   - Average metrics over 10 trials with standard deviations

**Target benchmark from paper (S&P 500 test set):**
- ARR: ~45.6%
- ASR: ~2.55
- CR: ~3.54
- MDD: ~-12.9%

---

## 11. Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or `num_latent_states`
2. **No edges in graph**: Lower `judge_value` (try 0.6 or 0.7)
3. **Poor convergence**: Try lower learning rate (1e-4) or more epochs
4. **Dimension mismatch**: Ensure `gat_hidden_size` is divisible by `cross_attn_heads`

### Validation Checks

- Verify graph has reasonable connectivity (avg degree > 1)
- Check that training loss decreases steadily
- Ensure test dates don't overlap with training dates
