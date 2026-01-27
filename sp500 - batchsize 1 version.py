import os
import math
import time
import pandas
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from subprocess import run
from datetime import datetime
from collections import Counter
from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch, DataLoader


def count_elements(lst):
    element_count = {}
    for element in lst:
        if element in element_count:
            element_count[element] += 1
        else:
            element_count[element] = 1
    return element_count

def rank_labeling(df, col_label='label', col_return='t2_am-15m_return_rate'):
    df[col_label] = df[col_return].rank(ascending=True, pct=True)
    return df

def process_daily_df_std(df, feature_cols):
    """Legacy per-day normalization (kept for compatibility)."""
    df = df.copy()
    for c in feature_cols:
        df[c] = filter_extreme_3sigma(df[c])
        df[c] = standardize_zscore(df[c])
    return df

def filter_extreme_3sigma(series, n=3):  # 3 sigma
    mean = series.mean()
    std = series.std()
    max_range = mean + n * std
    min_range = mean - n * std
    return np.clip(series, min_range, max_range)

def standardize_zscore(series):
    std = series.std()
    mean = series.mean()
    return (series - mean) / std

def compute_training_stats(df, feature_cols, train_end_date):
    """
    Compute normalization statistics from training period only.
    This prevents look-ahead bias per the paper's methodology.
    
    Args:
        df: DataFrame with all data
        feature_cols: List of feature column names
        train_end_date: End date for training period (YYYY-MM-DD string)
    
    Returns:
        means: dict of mean values per feature
        stds: dict of std values per feature
    """
    train_df = df[df['dt'] <= train_end_date]
    means = {}
    stds = {}
    for col in feature_cols:
        means[col] = train_df[col].mean()
        stds[col] = train_df[col].std()
        if stds[col] == 0:
            stds[col] = 1.0  # Prevent division by zero
    return means, stds

def normalize_with_training_stats(df, feature_cols, means, stds, n_sigma=3):
    """
    Normalize features using pre-computed training statistics.
    Apply 3-sigma clipping then Z-score normalization.
    
    Args:
        df: DataFrame to normalize
        feature_cols: List of feature column names
        means: dict of mean values per feature (from training)
        stds: dict of std values per feature (from training)
        n_sigma: Number of standard deviations for clipping (default 3)
    
    Returns:
        Normalized DataFrame
    """
    df = df.copy()
    for col in feature_cols:
        mean = means[col]
        std = stds[col]
        # 3-sigma clipping using training stats
        max_range = mean + n_sigma * std
        min_range = mean - n_sigma * std
        df[col] = np.clip(df[col], min_range, max_range)
        # Z-score normalization using training stats
        df[col] = (df[col] - mean) / std
    return df  

def generate_dataset(df_comp, feature_cols, hist_len, date_range):
    ds = []
    id_vals = df_comp.index.values
    df_comp = df_comp.reset_index(drop=True)
    dt_vals = df_comp['dt'].values
    feature_vals = df_comp[feature_cols].values

    for idx, row in df_comp.iterrows():
        dt = dt_vals[idx]
        if idx < hist_len or dt < date_range[0] or dt > date_range[1]:
            continue
        else:
            seq_features = feature_vals[idx + 1 - hist_len: idx + 1]
            ds.append((id_vals[idx], seq_features))
    return ds

def fun_train_test_data(dts_one, df, his_t):
    df1 = df.loc[df['dt'] >= dts_one[1]]
    df2 = df1.loc[df1['dt'] <= dts_one[5]]
    df2_test = df2.loc[df2['dt'] >= dts_one[4]]
    dts_test = sorted(list(set(df2_test['dt'].values.tolist())))

    kdcode_list = df2['kdcode'].values.tolist()
    dts = sorted(list(set(df2['dt'].values.tolist())))

    dict_list = count_elements(kdcode_list)
    kdcode_last = []
    for key in dict_list:
        if dict_list[key] == len(dts):
            kdcode_last.append(key)

    df3 = df2[df2['kdcode'].isin(kdcode_last)]
    len_test = len(dts_test)
    len_train = len(dts) - len(dts_test) - his_t
    print('Total days: ' + str(len(dts)))
    print('Number of stocks: ' + str(len(kdcode_last)))
    print('Training days: ' + str(len_train))
    print('Testing days: ' + str(len_test))
    print('Expected total rows: ' + str(len(kdcode_last) * len(dts)))
    print('Actual total rows: ' + str(len(df3)))

    df3 = df3[['kdcode','dt'] + feature_cols]
    df3 = df3.reset_index(drop=True)
    date_range_list = sorted(list(set(df3['dt'].values.tolist())))

    df_group = df3.groupby('kdcode')
    param_list = []
    for kdcode in df_group.groups.keys():
        df_comp = df_group.get_group(kdcode)
        param_list.append((df_comp, feature_cols, his_t, (date_range_list[0], date_range_list[-1])))
    
    pool = multiprocessing.Pool(10)
    result = pool.starmap(generate_dataset, param_list)
    pool.close()
    pool.join()
    ds_data = [item for sub in result for item in sub]
    ds_data = np.array(ds_data, dtype=object)
    
    idx_data = np.array([x[0] for x in ds_data])
    X_data = np.array([x[1] for x in ds_data])
    s_idx = pd.Series(index=idx_data, data=list(range(len(idx_data))))
    idx_train = s_idx[[i for i in df3.index if i in s_idx.index]].values
    X_train = X_data[idx_train]
    
    df3_1 = df3.loc[df3['dt']>=dts_one[2]]
    df3_1 = df3_1.loc[df3_1['dt']<=dts_one[3]]
    df3_1 = df3_1.reset_index(drop=True)
    df3_1_dt = sorted(list(set(df3_1['dt'].values.tolist())))
    df4_1 = df3_1.reset_index().sort_values(['dt', 'kdcode'])
    df4_1 = df3_1[feature_cols]
    df4_1_list = df4_1.values.tolist()
    x_graph_train = []
    for i in range(len(df3_1_dt)):
        x_graph_train.append(df4_1_list[i*len(kdcode_last):(i+1)*len(kdcode_last)])

    df3_2 = df3.loc[df3['dt']>=dts_one[4]]
    df3_2 = df3_2.loc[df3_2['dt']<=dts_one[5]]
    df3_2 = df3_2.reset_index(drop=True)
    df3_2_dt = sorted(list(set(df3_2['dt'].values.tolist())))
    df4_2 = df3_2.reset_index().sort_values(['dt', 'kdcode'])
    df4_2 = df3_2[feature_cols]
    df4_2_list = df4_2.values.tolist()
    x_graph_test = []
    for i in range(len(df3_2_dt)):
        x_graph_test.append(df4_2_list[i*len(kdcode_last):(i+1)*len(kdcode_last)])

    stock_features_all = []
    for i in range(len(dts)-his_t):
        stock_features_all.append(X_train[i*len(kdcode_last):(i+1)*len(kdcode_last)])
    stock_features_all_1 = stock_features_all[len(stock_features_all)-len(df3_1_dt)-len(df3_2_dt):]
    stock_features_train = stock_features_all_1[0:len(df3_1_dt)]
    stock_features_test = stock_features_all_1[len(df3_1_dt):]
    return kdcode_last, df3_1_dt, df3_2_dt, stock_features_train, stock_features_test, x_graph_train, x_graph_test

def fun_relation(kdcode_list, df):
    df5 = df.loc[df['dt']<=dts_one[0]]
    df5_dts = sorted(list(set(df5['dt'].values.tolist())))
    df5 = df5.loc[df5['dt']>=df5_dts[-250]]
    df5['t1_return_rate'] = df5['close']/df5['prev_close'] - 1    
    df5 = df5[df5['kdcode'].isin(kdcode_list)]
    df5 = df5.reset_index(drop=True)
    df_factors_2 = df5[['kdcode','dt','t1_return_rate']]
    col_name = 't1_return_rate'
    df0 = df_factors_2[df_factors_2['kdcode']==kdcode_list[0]].reset_index(drop=True)
    df1 = df0[[col_name]]
    df1 = df1.rename(columns={col_name: kdcode_list[0]})
    df_features_grouped = df_factors_2.groupby('kdcode')
    for kdcode in df_features_grouped.groups:
        if kdcode == kdcode_list[0]:
            continue
        else:
            df2 = df_features_grouped.get_group(kdcode).reset_index(drop=True)
            if len(df2)!=len(df1):
                df_tmp = df0[['kdcode','dt']]
                df_tmp = df_tmp.merge(df2, how='left', left_on=['dt'],right_on=['dt'])
                df_tmp[col_name]=df_tmp[col_name].fillna(df_tmp[col_name].mean())
                df1[kdcode] = df_tmp[col_name]
            else:
                df1[kdcode] = df2[col_name]
    matrx = df1.corr()
    return matrx

def fun_graph(matrx, kdcode_last, judge_value):
    df_jbm_matrx_2_list = matrx.values.tolist()
#     print(df_jbm_matrx_2_list)
    edge_index = []
    edge_weight = []
    for i in tqdm(range(len(kdcode_last))):
        for j in range(i + 1, len(kdcode_last)):
            weight = df_jbm_matrx_2_list[i][j]
            if weight>judge_value:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_weight.append(weight)
                edge_weight.append(weight)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    return edge_index, edge_weight

def fun_label(df, kdcode_last, df3_1_dt, label_t, dts_one):
    """
    Compute labels for training.
    
    Paper methodology: Use actual returns as labels, not rank percentiles.
    Ranking is only used for portfolio selection at inference time.
    """
    n = label_t
    c = 'close'
    label_column = 't'+str(n)+'_close_return_rate'
    df = df[df['kdcode'].isin(kdcode_last)]
    df = df.loc[df['dt']>=dts_one[0]]
    df = df.loc[df['dt']<=dts_one[5]]
    df_vwap_sorted = df.reset_index().sort_values(['kdcode', 'dt'])
    df_vwap_sorted['t1_{}'.format(c)] = df_vwap_sorted.groupby('kdcode')[c].shift(-1)
    df_vwap_sorted['t{}_{}'.format(n, c)] = df_vwap_sorted.groupby('kdcode')[c].shift(-n)
    df_vwap_sorted['t{}_{}_return_rate'.format(n, c)] = (df_vwap_sorted['t{}_{}'.format(n, c)]) / (df_vwap_sorted['t1_{}'.format(c)]) - 1
    df_vwap_sorted['dt'] = pd.to_datetime(df_vwap_sorted['dt'])
    df_vwap_sorted['dt'] =df_vwap_sorted['dt'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df_vwap_sorted = df_vwap_sorted.loc[df_vwap_sorted['dt']>=dts_one[2]]
    df_vwap_sorted = df_vwap_sorted.loc[df_vwap_sorted['dt']<=dts_one[3]]
    df_vwap_sorted_1 = df_vwap_sorted[['kdcode','dt',label_column]]
    df_features_grouped = df_vwap_sorted_1.groupby('dt')
    res = []
    for dt in df_features_grouped.groups:
        df_day = df_features_grouped.get_group(dt).copy()
        mean_val = df_day[label_column].mean()
        df_day[label_column].fillna(mean_val, inplace=True)
        res.append(df_day)
    df_label = pd.concat(res)
    df_label = df_label.sort_values(['dt','kdcode'])
    df_label = df_label.reset_index(drop=True)
    
    # Paper: Use actual returns as labels, NOT rank percentiles
    # The ranking is only used for portfolio selection at inference time
    if len(df_label) == len(kdcode_last)*len(df3_1_dt):
        label_list = df_label[label_column].values.tolist()  # Use actual returns
        true_returns=[]
        for i in range(len(df3_1_dt)):
            true_returns.append(label_list[i*len(kdcode_last):(i+1)*len(kdcode_last)])
        true_returns = np.array(true_returns)
    else:
        print("error")
    return true_returns

def fun_process_data_all(dts_one, filename, feature_cols, judge_value, label_t, his_t):
    df_org = pd.read_csv(filename)
    
    # Add Turnover feature (Close x Volume) per paper specification
    df_org['turnover'] = df_org['close'] * df_org['volume']
    
    # Fill NaN values first (before computing stats)
    df_features_grouped = df_org.groupby('dt')
    res = []
    for dt in df_features_grouped.groups:
        df_day = df_features_grouped.get_group(dt).copy()
        for column in feature_cols:
            mean_val = df_day[column].mean()
            df_day[column].fillna(mean_val, inplace=True)
        df_day = df_day.fillna(0.0)
        res.append(df_day)
    df_filled = pd.concat(res)
    
    # Paper methodology: Compute normalization stats from training period only
    # dts_one[0] is the correlation end date (before training starts)
    # Use data up to this date to compute normalization statistics
    train_end_for_norm = dts_one[0]
    means, stds = compute_training_stats(df_filled, feature_cols, train_end_for_norm)
    
    # Apply training-period normalization to all data
    df = normalize_with_training_stats(df_filled, feature_cols, means, stds)
    
    kdcode_last, df3_1_dt, df3_2_dt, stock_features_train, stock_features_test, x_graph_train, x_graph_test = fun_train_test_data(dts_one, df, his_t)
    
    matrx = fun_relation(kdcode_last, df)
    
    edge_index, edge_weight = fun_graph(matrx, kdcode_last, judge_value)
    
    true_returns = fun_label(df_org, kdcode_last, df3_1_dt, label_t, dts_one)
    
    return kdcode_last, df3_1_dt, df3_2_dt, stock_features_train, stock_features_test, x_graph_train, x_graph_test, edge_index, edge_weight, true_returns

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GraphDataset(Dataset):
    def __init__(self, X, edge_index, edge_weight):
        self.X = X
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        data = Data(x=self.X[idx], edge_index=self.edge_index, edge_weight=self.edge_weight)
        return data

class AttentionResetGRUCell(nn.Module):
    """
    GRU cell with attention mechanism replacing the reset gate.
    
    Paper methodology:
    - Instead of: r_t = sigmoid(W_r * x_t + U_r * h_{t-1})
    - We use: r'_t = Attention(h_{t-1}, x_t)
    - Query from h_{t-1}, Key/Value from x_t
    - Candidate: h_tilde = tanh(W_h(x) + r' * U_h(h))
    """
    
    def __init__(self, input_size, hidden_size):
        super(AttentionResetGRUCell, self).__init__()
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
            x_t: input at time t, shape (batch, num_stocks, input_size)
            h_prev: hidden state from t-1, shape (batch, num_stocks, hidden_size)
        
        Returns:
            h_t: new hidden state, shape (batch, num_stocks, hidden_size)
        """
        # Update gate (standard GRU)
        z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_prev))
        
        # Attention-based reset gate
        # Query from hidden state, Key/Value from input
        q_t = self.W_q(h_prev)  # (batch, num_stocks, hidden_size)
        k_t = self.W_k(x_t)     # (batch, num_stocks, hidden_size)
        v_t = self.W_v(x_t)     # (batch, num_stocks, hidden_size)
        
        # Scaled dot-product attention per paper Equation 6
        attn_score = torch.sum(q_t * k_t, dim=-1, keepdim=True) / np.sqrt(self.hidden_size)
        alpha_t = F.softmax(attn_score, dim=-1)  # Paper uses softmax

        # Attention-weighted value as reset signal (paper Equation 7: r'_t = a_t * v_t)
        r_prime_t = alpha_t * v_t  # (batch, num_stocks, hidden_size)
        
        # Candidate hidden state: h_tilde = tanh(W_h(x) + r' * U_h(h))
        h_tilde = torch.tanh(self.W_h(x_t) + r_prime_t * self.U_h(h_prev))
        
        # Final hidden state: h_t = (1 - z_t) * h + z_t * h_tilde
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t


class ImprovedGRU(nn.Module):
    """
    Multi-layer Improved GRU for temporal feature extraction.
    Paper uses two layers with hidden sizes [32, 10].
    """
    
    def __init__(self, input_size, hidden_sizes=[32, 10]):
        super(ImprovedGRU, self).__init__()
        self.layers = nn.ModuleList()
        self.hidden_sizes = hidden_sizes
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(AttentionResetGRUCell(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.output_size = hidden_sizes[-1]
    
    def forward(self, x):
        """
        Args:
            x: input sequence, shape (batch, num_stocks, seq_len, input_size)
        
        Returns:
            output: final hidden state, shape (batch, num_stocks, output_size)
        """
        batch_size, num_stocks, seq_len, _ = x.shape
        device = x.device
        
        # Process through each layer
        layer_input = x
        for layer_idx, layer in enumerate(self.layers):
            hidden_size = self.hidden_sizes[layer_idx]
            h = torch.zeros(batch_size, num_stocks, hidden_size, device=device)
            
            outputs = []
            for t in range(seq_len):
                h = layer(layer_input[:, :, t, :], h)
                outputs.append(h)
            
            # Stack outputs as input for next layer
            layer_input = torch.stack(outputs, dim=2)
        
        # Return final hidden state (last time step of last layer)
        return layer_input[:, :, -1, :]


class GATLayer(nn.Module):
    """
    Two-layer GAT for cross-sectional feature extraction.
    Paper uses ELU activation (not ReLU).
    """
    def __init__(self, hidden_size_gat1, output_gat1, in_channels, out_channels, heads=1):
        super(GATLayer, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_size_gat1, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_size_gat1 * heads, output_gat1, heads=1, concat=False, edge_dim=1)

    def forward(self, x, edge_index, edge_weight):
        x = self.gat1(x, edge_index, edge_weight)
        x = F.elu(x)  # Paper uses ELU
        x = self.gat2(x, edge_index, edge_weight)
        return x
    

class GATLayer_1(nn.Module):
    """
    Final prediction GAT layer.
    Paper uses ELU activation (not ReLU).
    """
    def __init__(self, hidden_size_gat2, in_channels, out_channels, heads=1):
        super(GATLayer_1, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_size_gat2, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_size_gat2 * heads, out_channels, heads=1, concat=False, edge_dim=1)

    def forward(self, x, edge_index, edge_weight):
        x = self.gat1(x, edge_index, edge_weight)
        x = F.elu(x)  # Paper uses ELU
        x = self.gat2(x, edge_index, edge_weight)
        return x 

class MarketLatentStateLearner(nn.Module):
    """
    Multi-head cross-attention mechanism for learning latent market states.
    
    Paper methodology:
    - Learns two sets of latent state vectors (R1, R2) 
    - R1 interacts with temporal features (A1)
    - R2 interacts with cross-sectional features (A2)
    - Uses multi-head attention (4 heads per paper)
    """
    
    def __init__(self, feature_dim, num_latent_states=32, num_heads=4):
        super(MarketLatentStateLearner, self).__init__()
        
        self.num_latent_states = num_latent_states
        self.feature_dim = feature_dim
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
    
class StockPredictionModel(nn.Module):
    """
    MCI-GRU Model: Multi-head Cross-attention and Improved GRU for Stock Prediction.
    
    Architecture per paper:
    1. Part A: Improved GRU for temporal features (two layers: [32, 10])
    2. Part B: GAT for cross-sectional features
    3. Part C: Multi-head cross-attention for latent market states
    4. Part D: Concatenate A1, A2, B1, B2 -> Prediction GAT
    """
    def __init__(
        self, 
        input_size, 
        gru_hidden_sizes=[32, 10],  # Paper: two-layer GRU
        hidden_size_gat1=32,         # Paper: 32
        output_gat1=4,               # Paper: 4
        gat_heads=4,                 # Paper: 4 heads
        hidden_size_gat2=32,         # Paper: 32
        num_hidden_states=32,        # Paper: 32 latent vectors
        cross_attn_heads=4           # Paper: 4 heads for cross-attention
    ):
        super(StockPredictionModel, self).__init__()
        
        # Part A: Improved GRU for temporal features
        self.temporal_gru = ImprovedGRU(input_size, gru_hidden_sizes)
        gru_output_size = self.temporal_gru.output_size  # 10 for paper config
        
        # Part B: GAT for cross-sectional features
        self.gat_layer = GATLayer(hidden_size_gat1, output_gat1, input_size, output_gat1, gat_heads)
        
        # Projection layers to align dimensions for cross-attention
        # Both A1 and A2 should have same dimension for the latent state learner
        self.align_dim = hidden_size_gat1  # Use GAT hidden size as alignment dimension
        self.proj_temporal = nn.Linear(gru_output_size, self.align_dim)
        self.proj_cross = nn.Linear(output_gat1, self.align_dim)
        
        # Part C: Multi-head cross-attention for latent market states
        self.latent_learner = MarketLatentStateLearner(
            feature_dim=self.align_dim,
            num_latent_states=num_hidden_states,
            num_heads=cross_attn_heads
        )
        
        # Part D: Prediction layer
        # Paper: Concatenate A1, A2, B1, B2 -> 4 * align_dim, then final GAT
        concat_size = 4 * self.align_dim
        self.final_gat = GATLayer_1(hidden_size_gat2, concat_size, 1, gat_heads)
        self.elu = nn.ELU()  # Paper uses ELU, not ReLU
        
    def forward(self, x_time_series, x_graph, edge_index, edge_weight):
        """
        Args:
            x_time_series: (batch, num_stocks, seq_len, input_size)
            x_graph: (num_stocks, input_size) - last time step features for graph
            edge_index: Graph edge indices
            edge_weight: Graph edge weights
        
        Returns:
            predictions: (num_stocks,) - predicted returns for each stock
        """
        # Part A: Temporal features via Improved GRU
        A1_raw = self.temporal_gru(x_time_series)  # (batch, num_stocks, gru_output_size)
        A1_raw = A1_raw[-1, :, :]  # Take last batch: (num_stocks, gru_output_size)
        A1 = self.proj_temporal(A1_raw)  # (num_stocks, align_dim)
        
        # Part B: Cross-sectional features via GAT
        A2_raw = self.gat_layer(x_graph, edge_index, edge_weight)  # (num_stocks, output_gat1)
        A2 = self.proj_cross(A2_raw)  # (num_stocks, align_dim)
        
        # Part C: Latent state learning via multi-head cross-attention
        B1, B2 = self.latent_learner(A1, A2)  # Both (num_stocks, align_dim)
        
        # Part D: Concatenate and predict (NO self-attention per paper)
        Z = torch.cat([A1, A2, B1, B2], dim=-1)  # (num_stocks, 4 * align_dim)
        
        # Final GAT for prediction
        out = self.final_gat(Z, edge_index, edge_weight)  # (num_stocks, 1)
        out = self.elu(out)  # Paper uses ELU activation
        
        return out.squeeze(-1)  # (num_stocks,)  


def model_data(stock_features_train, x_graph_train, true_returns, stock_features_test, x_graph_test):
    X_train_time_series=torch.Tensor(stock_features_train) 
    X_train_graph=torch.Tensor(x_graph_train) 
    y_train=torch.Tensor(true_returns) 

    train_time_series_dataset = TimeSeriesDataset(X_train_time_series, y_train)
    train_time_series_loader = DataLoader(train_time_series_dataset, batch_size=1, shuffle=True)
    train_graph_dataset = GraphDataset(X_train_graph, edge_index, edge_weight)
    train_graph_loader = DataLoader(train_graph_dataset, batch_size=1, shuffle=True)

    X_test_time_series=torch.Tensor(stock_features_test) 
    X_test_graph=torch.Tensor(x_graph_test) 

    test_graph_dataset = GraphDataset(X_test_graph, edge_index, edge_weight)
    test_graph_loader = DataLoader(test_graph_dataset, batch_size=1, shuffle=False)
    
    return train_time_series_loader, train_graph_loader, X_test_time_series, test_graph_loader

def create_model(input_size, config):
    """Create model using paper hyperparameters from CONFIG."""
    return StockPredictionModel(
        input_size=input_size,
        gru_hidden_sizes=config['gru_hidden_sizes'],
        hidden_size_gat1=config['hidden_size_gat1'],
        output_gat1=config['output_gat1'],
        gat_heads=config['gat_heads'],
        hidden_size_gat2=config['hidden_size_gat2'],
        num_hidden_states=config['num_hidden_states'],
        cross_attn_heads=config['cross_attn_heads']
    )


def model_train_predict(num_models, num_epochs, save_path, model_dt, kdcode_last, df3_2_dt, train_time_series_loader, train_graph_loader, X_test_time_series, test_graph_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    for num in range(num_models):
        # Create model using paper hyperparameters
        model = create_model(num_features, CONFIG).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for (X_time_series_batch, y_batch), graph_batch in zip(train_time_series_loader, train_graph_loader):
                X_time_series_batch, y_batch = X_time_series_batch.to(device), y_batch.to(device)
                graph_batch = graph_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_time_series_batch, graph_batch.x, graph_batch.edge_index, graph_batch.edge_weight)
                loss = criterion(outputs, y_batch.view(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_time_series_batch.size(0)

            epoch_loss = running_loss / len(train_time_series_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                save_path_1 = save_path + 'model_' + str(num) + '/' + model_dt + '_best.pth'
                torch.save(model.state_dict(), save_path_1)
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Also save per-epoch for compatibility
            save_path_1 = save_path + 'model_' + str(num) + '/' + model_dt + '_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), save_path_1)
            
        print('Finished Training With Number ' + str(num))
    
    # Inference using best models
    for num in tqdm(range(num_models)):
        model = create_model(num_features, CONFIG).to(device)
        
        # Try to load best model, fall back to last epoch
        best_path = save_path + 'model_' + str(num) + '/' + model_dt + '_best.pth'
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path))
        else:
            # Fall back to last available epoch
            for epoch in range(num_epochs - 1, -1, -1):
                epoch_path = save_path + 'model_' + str(num) + '/' + model_dt + '_' + str(epoch) + '.pth'
                if os.path.exists(epoch_path):
                    model.load_state_dict(torch.load(epoch_path))
                    break
        
        model.eval()
        with torch.no_grad():
            index = 0
            for X_test_time_series_batch, graph_batch in zip(X_test_time_series, test_graph_loader):
                X_test_time_series_batch = X_test_time_series_batch.unsqueeze(0).to(device)
                graph_batch = graph_batch.to(device)

                outputs = model(X_test_time_series_batch, graph_batch.x, graph_batch.edge_index, graph_batch.edge_weight)
                prediction = outputs.cpu().numpy().tolist()
                data_all = []
                for i in range(len(prediction)):
                    one = []
                    one.append(kdcode_last[i])
                    one.append(df3_2_dt[index])
                    one.append(round(prediction[i], 5))
                    data_all.append(one)
                df = pd.DataFrame(columns=['kdcode', 'dt', 'score'], data=data_all)
                
                # Save predictions (simplified to just best model output)
                pred_path = save_path + 'prediction_' + str(num) + '/0/'
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)
                df.to_csv(pred_path + df3_2_dt[index] + '.csv', header=True, index=False, encoding='utf_8_sig')
                index += 1
                    

dts_all =[
['2022-11-30', '2022-11-01', '2022-12-01', '2022-12-31', '2023-01-01', '2023-01-31'],
['2022-12-31', '2022-12-01', '2023-01-01', '2023-01-31', '2023-02-01', '2023-02-28'],
['2023-01-31', '2023-01-01', '2023-02-01', '2023-02-28', '2023-03-01', '2023-03-31'],
['2023-02-28', '2023-02-01', '2023-03-01', '2023-03-31', '2023-04-01', '2023-04-30'],
['2023-03-31', '2023-03-01', '2023-04-01', '2023-04-30', '2023-05-01', '2023-05-31'],
['2023-04-30', '2023-04-01', '2023-05-01', '2023-05-31', '2023-06-01', '2023-06-30'],
['2023-05-31', '2023-05-01', '2023-06-01', '2023-06-30', '2023-07-01', '2023-07-31'],
['2023-06-30', '2023-06-01', '2023-07-01', '2023-07-31', '2023-08-01', '2023-08-31'],
['2023-07-31', '2023-07-01', '2023-08-01', '2023-08-31', '2023-09-01', '2023-09-30'],
['2023-08-31', '2023-08-01', '2023-09-01', '2023-09-30', '2023-10-01', '2023-10-31'],
['2023-09-30', '2023-09-01', '2023-10-01', '2023-10-31', '2023-11-01', '2023-11-30'], 
['2023-10-31', '2023-10-01', '2023-11-01', '2023-11-30', '2023-12-01', '2023-12-31']]
filename = 'sp500_yf_download.csv'
# Paper uses 6 features: Open, High, Low, Close, Volume, Turnover
feature_cols = ['close','open','high','low','volume','turnover']
num_features = len(feature_cols)
judge_value = 0.8
label_t = 5
his_t = 10
num_models = 10  # Paper uses 10 runs for averaging
num_epochs = 100  # Paper uses more epochs with early stopping

# Paper hyperparameters
CONFIG = {
    'gru_hidden_sizes': [32, 10],  # Two-layer GRU per paper
    'hidden_size_gat1': 32,        # Paper: 32 (was 5)
    'output_gat1': 4,              # Paper: 4 (was 256)
    'gat_heads': 4,                # Paper: 4 heads
    'hidden_size_gat2': 32,        # Paper: 32 (was 5)
    'num_hidden_states': 32,       # Paper: 32 latent vectors (was 4)
    'cross_attn_heads': 4,         # Paper: 4 heads for cross-attention
    'learning_rate': 0.0002,       # Paper: 0.0002 (was 0.001)
    'early_stopping_patience': 10,
    'batch_size': 1
}
save_path = '4_20240707_sp_hs256_' + str(judge_value) + '_' + str(label_t) + '_' + str(his_t) + '/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path_prediction=save_path+'prediction/'
if not os.path.exists(save_path_prediction):
    os.makedirs(save_path_prediction)

for i in range(num_models):
    save_path_1  = save_path+'model_'+str(i)
    if not os.path.exists(save_path_1):
        os.makedirs(save_path_1)
    save_path_1  = save_path+'prediction_'+str(i)
    if not os.path.exists(save_path_1):
        os.makedirs(save_path_1)
    for j in range(num_epochs):
        save_path_2  = save_path+'prediction_'+str(i)+'/'+str(j)
        if not os.path.exists(save_path_2):
            os.makedirs(save_path_2)
        
dts_all = dts_all[0:12]

for dts_one in tqdm(dts_all):
    # print(dts_one)
    kdcode_last, df3_1_dt, df3_2_dt, stock_features_train, stock_features_test, x_graph_train, x_graph_test, edge_index, edge_weight, true_returns = fun_process_data_all(dts_one, filename, feature_cols, judge_value, label_t, his_t)
    
    train_time_series_loader, train_graph_loader, X_test_time_series, test_graph_loader = model_data(stock_features_train, x_graph_train, true_returns, stock_features_test, x_graph_test)
    
    model_train_predict(num_models, num_epochs, save_path, dts_one[3], kdcode_last, df3_2_dt, train_time_series_loader, train_graph_loader, X_test_time_series, test_graph_loader)

# nohup /home/liyuante/miniconda3/envs/py38/bin/python /home/liyuante/sp500.py >> /home/liyuante/log_for_all/sp5_test.txt 2>&1 &