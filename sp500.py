import os
import gc
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
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from functools import partial


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
    
    # Debug: Check for potential slicing issues
    print(f"DEBUG: len(stock_features_all)={len(stock_features_all)}, len(df3_1_dt)={len(df3_1_dt)}, len(df3_2_dt)={len(df3_2_dt)}")
    slice_start = len(stock_features_all) - len(df3_1_dt) - len(df3_2_dt)
    print(f"DEBUG: Slicing stock_features_all from index {slice_start}")
    
    if slice_start < 0:
        raise ValueError(
            f"Not enough historical data! Need at least {len(df3_1_dt) + len(df3_2_dt)} days after his_t offset, "
            f"but only have {len(stock_features_all)} days. "
            f"Total days: {len(dts)}, his_t: {his_t}, "
            f"training dates: {len(df3_1_dt)}, test dates: {len(df3_2_dt)}"
        )
    
    stock_features_all_1 = stock_features_all[slice_start:]
    stock_features_train = stock_features_all_1[0:len(df3_1_dt)]
    stock_features_test = stock_features_all_1[len(df3_1_dt):]
    
    # Final validation
    print(f"DEBUG: stock_features_train length={len(stock_features_train)}, stock_features_test length={len(stock_features_test)}")
    print(f"DEBUG: x_graph_train length={len(x_graph_train)}, x_graph_test length={len(x_graph_test)}")
    
    if len(stock_features_train) == 0:
        raise ValueError(
            f"stock_features_train is empty! "
            f"Check date range: training should be from {dts_one[2]} to {dts_one[3]}"
        )
    
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
            # Check for valid weight (not NaN) and above threshold
            if not np.isnan(weight) and weight > judge_value:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_weight.append(weight)
                edge_weight.append(weight)
    
    # Handle empty edge case - create proper 2D tensor with shape (2, 0)
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
    else:
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
        # Fixed: Use proper pandas syntax to avoid FutureWarning
        df_day[label_column] = df_day[label_column].fillna(mean_val)
        res.append(df_day)
    df_label = pd.concat(res)
    df_label = df_label.sort_values(['dt','kdcode'])
    df_label = df_label.reset_index(drop=True)
    
    # Debug: Print diagnostic info about label data
    print(f"DEBUG fun_label: df_label rows = {len(df_label)}, expected = {len(kdcode_last)}*{len(df3_1_dt)} = {len(kdcode_last)*len(df3_1_dt)}")
    print(f"DEBUG fun_label: num stocks = {len(kdcode_last)}, num training dates = {len(df3_1_dt)}")
    
    # Paper: Use actual returns as labels, NOT rank percentiles
    # The ranking is only used for portfolio selection at inference time
    if len(kdcode_last) == 0 or len(df3_1_dt) == 0:
        raise ValueError(
            f"Cannot create labels: kdcode_last has {len(kdcode_last)} stocks, "
            f"df3_1_dt has {len(df3_1_dt)} training dates. "
            f"Check that your data covers the date range {dts_one[2]} to {dts_one[3]}."
        )
    
    if len(df_label) == len(kdcode_last)*len(df3_1_dt):
        label_list = df_label[label_column].values.tolist()  # Use actual returns
        true_returns = []
        for i in range(len(df3_1_dt)):
            true_returns.append(label_list[i*len(kdcode_last):(i+1)*len(kdcode_last)])
        true_returns = np.array(true_returns)
    else:
        raise ValueError(
            f"Label data mismatch: got {len(df_label)} rows, expected {len(kdcode_last)*len(df3_1_dt)}. "
            f"This usually means some stocks are missing data for certain dates."
        )
    
    print(f"DEBUG fun_label: true_returns shape = {true_returns.shape}")
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
            # Fixed: Use proper pandas syntax to avoid FutureWarning
            df_day[column] = df_day[column].fillna(mean_val)
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


# ============================================================================
# PAPER METHODOLOGY: Fixed Train/Val/Test Split (matching paper Section 4.1.1)
# Training: 2017-01-01 to 2021-12-31 (~1000+ trading days)
# Validation: 2022-01-01 to 2022-12-31 (~252 trading days)
# Testing: 2023-01-01 to 2023-12-31 (~252 trading days)
# ============================================================================

def prepare_data_paper_style(filename, feature_cols, his_t, 
                              train_start='2017-01-01', train_end='2021-12-31',
                              val_start='2022-01-01', val_end='2022-12-31',
                              test_start='2023-01-01', test_end='2023-12-31',
                              corr_lookback_days=252):
    """
    Prepare data following the paper's methodology with fixed train/val/test split.
    
    Paper Section 4.1.1:
    - Training: Jan 1, 2018 to Dec 31, 2021 (we use 2017 for more data)
    - Validation: Jan 1, 2022 to Dec 31, 2022
    - Testing: Jan 1, 2023 to Dec 31, 2023
    
    Args:
        filename: Path to CSV file with stock data
        feature_cols: List of feature column names
        his_t: Historical window size (10 days per paper)
        train_start/end: Training period boundaries
        val_start/end: Validation period boundaries
        test_start/end: Test period boundaries
        corr_lookback_days: Days to use for correlation computation (252 = 1 year)
    
    Returns:
        Dictionary containing all processed data for training pipeline
    """
    print(f"Loading data from {filename}...")
    df_org = pd.read_csv(filename)
    
    # Add Turnover feature (Close x Volume) per paper specification
    df_org['turnover'] = df_org['close'] * df_org['volume']
    
    print(f"Data loaded: {len(df_org)} rows")
    print(f"Date range: {df_org['dt'].min()} to {df_org['dt'].max()}")
    
    # Fill NaN values per day (mean imputation)
    print("Filling NaN values...")
    df_features_grouped = df_org.groupby('dt')
    res = []
    for dt in df_features_grouped.groups:
        df_day = df_features_grouped.get_group(dt).copy()
        for column in feature_cols:
            mean_val = df_day[column].mean()
            df_day[column] = df_day[column].fillna(mean_val)
        df_day = df_day.fillna(0.0)
        res.append(df_day)
    df_filled = pd.concat(res)
    del res  # Free memory
    gc.collect()
    
    # Paper methodology: Compute normalization stats from training period only
    print(f"Computing normalization statistics from training period ({train_start} to {train_end})...")
    means, stds = compute_training_stats(df_filled, feature_cols, train_end)
    
    # Apply training-period normalization to all data (prevents look-ahead bias)
    df = normalize_with_training_stats(df_filled, feature_cols, means, stds)
    
    # Free memory from intermediate DataFrame
    del df_filled
    gc.collect()
    
    # Filter to stocks that have complete data across all periods
    # This is important to ensure consistent stock universe
    print("Filtering stocks with complete data...")
    all_dates = sorted(df['dt'].unique())
    
    # Get date range from train_start through test_end
    date_mask = (df['dt'] >= train_start) & (df['dt'] <= test_end)
    df_period = df[date_mask].copy()
    period_dates = sorted(df_period['dt'].unique())
    
    print(f"Period dates: {len(period_dates)} trading days from {period_dates[0]} to {period_dates[-1]}")
    
    # Count occurrences per stock
    kdcode_counts = df_period['kdcode'].value_counts()
    # Keep only stocks present on all trading days
    kdcode_last = kdcode_counts[kdcode_counts == len(period_dates)].index.tolist()
    kdcode_last = sorted(kdcode_last)
    
    print(f"Stocks with complete data: {len(kdcode_last)}")
    
    if len(kdcode_last) == 0:
        raise ValueError("No stocks have complete data across the entire period!")
    
    # Filter to selected stocks
    df_filtered = df_period[df_period['kdcode'].isin(kdcode_last)].copy()
    df_filtered = df_filtered.sort_values(['dt', 'kdcode']).reset_index(drop=True)
    
    # Split into train/val/test periods
    train_mask = (df_filtered['dt'] >= train_start) & (df_filtered['dt'] <= train_end)
    val_mask = (df_filtered['dt'] >= val_start) & (df_filtered['dt'] <= val_end)
    test_mask = (df_filtered['dt'] >= test_start) & (df_filtered['dt'] <= test_end)
    
    df_train = df_filtered[train_mask].copy()
    df_val = df_filtered[val_mask].copy()
    df_test = df_filtered[test_mask].copy()
    
    train_dates = sorted(df_train['dt'].unique())
    val_dates = sorted(df_val['dt'].unique())
    test_dates = sorted(df_test['dt'].unique())
    
    print(f"Training period: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"Validation period: {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})")
    print(f"Test period: {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")
    
    # Generate time series features for each stock
    print("Generating time series features...")
    stock_features = generate_time_series_features_paper(df_filtered, kdcode_last, feature_cols, his_t, period_dates)
    
    # Split features by period (accounting for his_t offset)
    # We need his_t days of history before each prediction day
    train_start_idx = his_t  # First usable index after history window
    train_end_idx = len(train_dates)
    val_end_idx = train_end_idx + len(val_dates)
    test_end_idx = val_end_idx + len(test_dates)
    
    # Effective training days = total train days - his_t (need history)
    effective_train_days = len(train_dates) - his_t
    
    print(f"Effective training samples: {effective_train_days} days")
    print(f"Validation samples: {len(val_dates)} days")
    print(f"Test samples: {len(test_dates)} days")
    
    # Extract features for each period (stock_features is now a numpy array)
    stock_features_train = stock_features[0:effective_train_days]
    stock_features_val = stock_features[effective_train_days:effective_train_days + len(val_dates)]
    stock_features_test = stock_features[effective_train_days + len(val_dates):]
    
    # Free the original large array since we've sliced it
    del stock_features
    gc.collect()
    
    print(f"stock_features_train shape: {stock_features_train.shape}")
    print(f"stock_features_val shape: {stock_features_val.shape}")
    print(f"stock_features_test shape: {stock_features_test.shape}")
    
    # Generate graph node features (daily features for each stock)
    print("Generating graph features...")
    x_graph_train = generate_graph_features(df_train, kdcode_last, feature_cols, train_dates[his_t:])
    x_graph_val = generate_graph_features(df_val, kdcode_last, feature_cols, val_dates)
    x_graph_test = generate_graph_features(df_test, kdcode_last, feature_cols, test_dates)
    
    # Free DataFrames we no longer need
    del df_train, df_val, df_test, df_filtered, df_period, df
    gc.collect()
    print("Cleaned up intermediate DataFrames")
    
    # Compute stock correlation matrix for graph construction
    # Per paper: correlations based on "past year" returns
    # Note: If no data exists before train_start, use early training data instead
    print("Computing stock correlation matrix...")
    data_start = df_org['dt'].min()
    if data_start >= train_start:
        # No pre-training data available, use first year of training data
        # This is a reasonable fallback when historical data is limited
        corr_end_date = train_dates[min(corr_lookback_days, len(train_dates) - 1)]
        print(f"  Note: No pre-training data available. Using training data up to {corr_end_date} for correlations.")
    else:
        corr_end_date = train_start  # Use data before training starts
    matrx = compute_correlation_matrix(df_org, kdcode_last, corr_end_date, corr_lookback_days)
    
    # Build graph edges
    edge_index, edge_weight = fun_graph(matrx, kdcode_last, judge_value)
    print(f"Graph edges: {edge_index.shape[1]} (with judge_value={judge_value})")
    
    # Compute labels (returns) for training and validation
    print("Computing labels...")
    train_labels = compute_labels_paper(df_org, kdcode_last, train_dates[his_t:], label_t)
    val_labels = compute_labels_paper(df_org, kdcode_last, val_dates, label_t)
    
    print(f"train_labels shape: {train_labels.shape}")
    print(f"val_labels shape: {val_labels.shape}")
    
    return {
        'kdcode_last': kdcode_last,
        'train_dates': train_dates[his_t:],  # Dates we can actually train on
        'val_dates': val_dates,
        'test_dates': test_dates,
        'stock_features_train': stock_features_train,
        'stock_features_val': stock_features_val,
        'stock_features_test': stock_features_test,
        'x_graph_train': x_graph_train,
        'x_graph_val': x_graph_val,
        'x_graph_test': x_graph_test,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'train_labels': train_labels,
        'val_labels': val_labels,
    }


def generate_time_series_features_paper(df, kdcode_last, feature_cols, his_t, all_dates):
    """
    Generate time series features for all stocks across all dates.
    
    OPTIMIZED VERSION: Uses pre-allocated numpy arrays instead of nested Python lists.
    This reduces memory usage from ~50GB+ to ~2-4GB.
    
    Returns numpy array of shape (num_usable_days, num_stocks, his_t, num_features)
    """
    num_stocks = len(kdcode_last)
    num_features = len(feature_cols)
    num_usable_days = len(all_dates) - his_t
    
    print(f"  Allocating feature array: ({num_usable_days}, {num_stocks}, {his_t}, {num_features})")
    print(f"  Expected memory: {num_usable_days * num_stocks * his_t * num_features * 4 / 1e9:.2f} GB")
    
    # Pre-allocate single numpy array (float32 to save memory)
    stock_features_all = np.zeros((num_usable_days, num_stocks, his_t, num_features), dtype=np.float32)
    
    # Create efficient lookup structures
    print("  Building lookup table...")
    
    # Create stock index mapping
    stock_to_idx = {stock: idx for idx, stock in enumerate(kdcode_last)}
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    
    # Pivot the dataframe for fast vectorized access
    # Shape: (num_dates, num_stocks, num_features)
    df_subset = df[df['kdcode'].isin(kdcode_last)].copy()
    df_subset = df_subset[['kdcode', 'dt'] + feature_cols]
    
    # Create a 3D array: (dates, stocks, features)
    print("  Creating pivoted data structure...")
    pivot_data = np.zeros((len(all_dates), num_stocks, num_features), dtype=np.float32)
    
    # Fill pivot_data efficiently using groupby
    for (kdcode, dt), group in tqdm(df_subset.groupby(['kdcode', 'dt']), 
                                      desc="  Building pivot table", 
                                      total=len(df_subset.groupby(['kdcode', 'dt']))):
        if kdcode in stock_to_idx and dt in date_to_idx:
            stock_idx = stock_to_idx[kdcode]
            date_idx = date_to_idx[dt]
            pivot_data[date_idx, stock_idx, :] = group[feature_cols].values[0].astype(np.float32)
    
    # Free memory
    del df_subset
    gc.collect()
    
    # Now fill the feature array using vectorized slicing
    print("  Generating sliding window features...")
    for day_offset in tqdm(range(num_usable_days), desc="  Processing days"):
        # Get his_t consecutive days of data for all stocks at once
        # day_offset corresponds to predicting day (his_t + day_offset)
        # We need history from day_offset to day_offset + his_t
        stock_features_all[day_offset, :, :, :] = pivot_data[day_offset:day_offset + his_t, :, :].transpose(1, 0, 2)
    
    # Free pivot data
    del pivot_data
    gc.collect()
    
    print(f"  Feature generation complete. Shape: {stock_features_all.shape}")
    
    return stock_features_all


def generate_graph_features(df, kdcode_last, feature_cols, dates):
    """
    Generate graph node features for each day.
    
    OPTIMIZED VERSION: Returns numpy array instead of nested lists.
    Shape: (num_dates, num_stocks, num_features)
    """
    num_dates = len(dates)
    num_stocks = len(kdcode_last)
    num_features = len(feature_cols)
    
    # Pre-allocate array
    x_graph = np.zeros((num_dates, num_stocks, num_features), dtype=np.float32)
    
    # Create lookup
    stock_to_idx = {stock: idx for idx, stock in enumerate(kdcode_last)}
    
    # Filter dataframe once
    df_subset = df[df['dt'].isin(dates) & df['kdcode'].isin(kdcode_last)][['kdcode', 'dt'] + feature_cols].copy()
    
    # Group by date for efficiency
    for date_idx, date in enumerate(dates):
        df_day = df_subset[df_subset['dt'] == date]
        for _, row in df_day.iterrows():
            stock_idx = stock_to_idx.get(row['kdcode'])
            if stock_idx is not None:
                x_graph[date_idx, stock_idx, :] = row[feature_cols].values.astype(np.float32)
    
    del df_subset
    gc.collect()
    
    return x_graph


def compute_correlation_matrix(df_org, kdcode_last, end_date, lookback_days=252):
    """
    Compute stock return correlation matrix using historical data.
    
    Per paper Section 3.3.2: Use past year of returns for correlation.
    """
    # Add return column if not present
    df = df_org.copy()
    if 'prev_close' in df.columns:
        df['daily_return'] = df['close'] / df['prev_close'] - 1
    else:
        # Compute returns per stock
        df = df.sort_values(['kdcode', 'dt'])
        df['daily_return'] = df.groupby('kdcode')['close'].pct_change()
    
    # Filter to before end_date
    df = df[df['dt'] < end_date]
    
    # Get last lookback_days
    all_dates = sorted(df['dt'].unique())
    if len(all_dates) > lookback_days:
        start_date = all_dates[-lookback_days]
        df = df[df['dt'] >= start_date]
    
    # Filter to our stocks
    df = df[df['kdcode'].isin(kdcode_last)]
    
    # Pivot to get returns matrix (dates x stocks)
    pivot = df.pivot_table(index='dt', columns='kdcode', values='daily_return')
    
    # Ensure column order matches kdcode_last
    pivot = pivot.reindex(columns=kdcode_last)
    
    # Fill missing values with 0
    pivot = pivot.fillna(0)
    
    # Compute correlation matrix
    corr_matrix = pivot.corr()
    
    return corr_matrix


def compute_labels_paper(df_org, kdcode_last, dates, label_t):
    """
    Compute return labels for given dates.
    
    Paper uses next-day returns as labels (or n-day forward returns).
    
    Args:
        df_org: Original dataframe with price data
        kdcode_last: List of stock codes
        dates: List of dates to compute labels for
        label_t: Forward return period (days)
    
    Returns:
        numpy array of shape (num_dates, num_stocks) with dtype float32
    """
    # Filter to our stocks only (don't copy entire df)
    df = df_org[df_org['kdcode'].isin(kdcode_last)].copy()
    df = df.sort_values(['kdcode', 'dt'])
    
    # Compute forward returns
    df['future_close'] = df.groupby('kdcode')['close'].shift(-label_t)
    df['next_close'] = df.groupby('kdcode')['close'].shift(-1)
    df['forward_return'] = df['future_close'] / df['next_close'] - 1
    
    # Filter to our dates
    df = df[df['dt'].isin(dates)]
    
    # Pivot to (dates, stocks)
    pivot = df.pivot_table(index='dt', columns='kdcode', values='forward_return')
    pivot = pivot.reindex(index=dates, columns=kdcode_last)
    
    # Fill NaN with mean per day
    for date in dates:
        if date in pivot.index:
            row_mean = pivot.loc[date].mean()
            pivot.loc[date] = pivot.loc[date].fillna(row_mean)
    
    # Fill any remaining NaN with 0
    pivot = pivot.fillna(0)
    
    # Return as float32 to save memory
    return pivot.values.astype(np.float32)


def create_data_loaders_paper(data_dict, edge_index, edge_weight, batch_size=32):
    """
    Create train/val/test data loaders from prepared data.
    
    OPTIMIZED: Data is already numpy float32 arrays, convert directly to tensors.
    
    Args:
        data_dict: Dictionary from prepare_data_paper_style
        edge_index: Graph edge indices
        edge_weight: Graph edge weights
        batch_size: Batch size for training
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Convert to tensors (data is already numpy float32)
    print("Converting data to tensors...")
    X_train_ts = torch.from_numpy(data_dict['stock_features_train'])
    X_train_graph = torch.from_numpy(data_dict['x_graph_train'])
    y_train = torch.from_numpy(data_dict['train_labels'])
    
    X_val_ts = torch.from_numpy(data_dict['stock_features_val'])
    X_val_graph = torch.from_numpy(data_dict['x_graph_val'])
    y_val = torch.from_numpy(data_dict['val_labels'])
    
    X_test_ts = torch.from_numpy(data_dict['stock_features_test'])
    X_test_graph = torch.from_numpy(data_dict['x_graph_test'])
    y_test_dummy = torch.zeros(len(X_test_ts), X_test_graph.shape[1], dtype=torch.float32)
    
    # Free the numpy arrays from data_dict to save memory
    del data_dict['stock_features_train']
    del data_dict['stock_features_val']
    del data_dict['stock_features_test']
    del data_dict['x_graph_train']
    del data_dict['x_graph_val']
    del data_dict['x_graph_test']
    del data_dict['train_labels']
    del data_dict['val_labels']
    gc.collect()
    
    print(f"Train tensors: ts={X_train_ts.shape}, graph={X_train_graph.shape}, labels={y_train.shape}")
    print(f"Val tensors: ts={X_val_ts.shape}, graph={X_val_graph.shape}, labels={y_val.shape}")
    print(f"Test tensors: ts={X_test_ts.shape}, graph={X_test_graph.shape}")
    
    # Create datasets
    train_dataset = CombinedDataset(X_train_ts, X_train_graph, y_train)
    val_dataset = CombinedDataset(X_val_ts, X_val_graph, y_val)
    test_dataset = CombinedDataset(X_test_ts, X_test_graph, y_test_dummy)
    
    # Create collate functions
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
    
    print(f"Created loaders: train={len(train_loader)} batches, val={len(val_loader)} batches, test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def train_model_paper_style(model, train_loader, val_loader, optimizer, criterion, 
                            num_epochs, patience, device, save_path, model_id):
    """
    Train model with validation-based early stopping (matching paper methodology).
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader  
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Maximum epochs
        patience: Early stopping patience
        device: Device to train on
        save_path: Path to save best model
        model_id: Model identifier for saving
    
    Returns:
        best_val_loss: Best validation loss achieved
    """
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for time_series, labels, graph_features, edge_index, edge_weight, n_stocks in train_loader:
            batch_size = time_series.shape[0]
            
            time_series = time_series.to(device)
            labels = labels.to(device)
            graph_features = graph_features.to(device)
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)
            
            optimizer.zero_grad()
            outputs = model(time_series, graph_features, edge_index, edge_weight, n_stocks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_size
            train_samples += batch_size
        
        avg_train_loss = train_loss / train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for time_series, labels, graph_features, edge_index, edge_weight, n_stocks in val_loader:
                batch_size = time_series.shape[0]
                
                time_series = time_series.to(device)
                labels = labels.to(device)
                graph_features = graph_features.to(device)
                edge_index = edge_index.to(device)
                edge_weight = edge_weight.to(device)
                
                outputs = model(time_series, graph_features, edge_index, edge_weight, n_stocks)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * batch_size
                val_samples += batch_size
        
        avg_val_loss = val_loss / val_samples
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            model_path = os.path.join(save_path, f'model_{model_id}', 'best_model.pth')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  -> New best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break
    
    return best_val_loss


def run_inference_paper_style(model, test_loader, kdcode_last, test_dates, device, save_path, model_id):
    """
    Run inference on test set and save predictions.
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for idx, (time_series, _, graph_features, edge_index, edge_weight, n_stocks) in enumerate(test_loader):
            time_series = time_series.to(device)
            graph_features = graph_features.to(device)
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)
            
            outputs = model(time_series, graph_features, edge_index, edge_weight, n_stocks)
            predictions = outputs.squeeze().cpu().numpy()
            all_predictions.append(predictions)
            
            # Save per-day predictions
            if idx < len(test_dates):
                date = test_dates[idx]
                data = [[kdcode_last[i], date, round(float(predictions[i]), 5)] 
                        for i in range(len(kdcode_last))]
                df = pd.DataFrame(columns=['kdcode', 'dt', 'score'], data=data)
                
                pred_path = os.path.join(save_path, f'predictions_{model_id}')
                os.makedirs(pred_path, exist_ok=True)
                df.to_csv(os.path.join(pred_path, f'{date}.csv'), index=False)
    
    return np.array(all_predictions)


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


class CombinedDataset(Dataset):
    """
    Combined dataset for synchronized time series, graph features, and labels.
    This ensures time series and graph data stay aligned when shuffling.
    """
    def __init__(self, X_time_series, X_graph, y):
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
        # Node indices need to be offset by i * num_stocks
        shifted_edge_index = edge_index + (i * num_stocks)
        edge_index_list.append(shifted_edge_index)
        edge_weight_list.append(edge_weight)
    
    # Concatenate all graph data
    batched_graph_features = torch.cat(graph_features_list, dim=0)
    batched_edge_index = torch.cat(edge_index_list, dim=1)
    batched_edge_weight = torch.cat(edge_weight_list, dim=0)
    
    return time_series, labels, batched_graph_features, batched_edge_index, batched_edge_weight, num_stocks


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
        
    def forward(self, x_time_series, x_graph, edge_index, edge_weight, num_stocks=None):
        """
        Batched forward pass supporting batch_size > 1.
        
        Args:
            x_time_series: (batch, num_stocks, seq_len, input_size)
            x_graph: (batch * num_stocks, input_size) - PyG batched graph nodes
            edge_index: (2, batch * num_edges) - PyG batched edge indices
            edge_weight: (batch * num_edges,) - PyG batched edge weights
            num_stocks: int - number of stocks per graph (required for batch > 1)
        
        Returns:
            predictions: (batch, num_stocks) - predicted returns for each stock
        """
        batch_size = x_time_series.shape[0]
        if num_stocks is None:
            num_stocks = x_time_series.shape[1]
        
        # Part A: Temporal features via Improved GRU
        # Input: (batch, num_stocks, seq_len, input_size)
        # Output: (batch, num_stocks, gru_output_size)
        A1_raw = self.temporal_gru(x_time_series)
        
        # Flatten to (batch * num_stocks, gru_output_size) to match batched graph structure
        A1_raw = A1_raw.reshape(batch_size * num_stocks, -1)
        A1 = self.proj_temporal(A1_raw)  # (batch * num_stocks, align_dim)
        
        # Part B: Cross-sectional features via GAT
        # x_graph is already (batch * num_stocks, input_size) from batched collate
        A2_raw = self.gat_layer(x_graph, edge_index, edge_weight)  # (batch * num_stocks, output_gat1)
        A2 = self.proj_cross(A2_raw)  # (batch * num_stocks, align_dim)
        
        # Part C: Latent state learning via multi-head cross-attention
        B1, B2 = self.latent_learner(A1, A2)  # Both (batch * num_stocks, align_dim)
        
        # Part D: Concatenate and predict (NO self-attention per paper)
        Z = torch.cat([A1, A2, B1, B2], dim=-1)  # (batch * num_stocks, 4 * align_dim)
        
        # Final GAT for prediction
        out = self.final_gat(Z, edge_index, edge_weight)  # (batch * num_stocks, 1)
        out = self.elu(out)  # Paper uses ELU activation
        
        # Reshape back to (batch, num_stocks)
        return out.view(batch_size, num_stocks)  


def model_data(stock_features_train, x_graph_train, true_returns, stock_features_test, x_graph_test, edge_index, edge_weight, batch_size=32):
    """
    Create data loaders for training and testing.
    
    Uses CombinedDataset to keep time series and graph data synchronized when shuffling.
    Custom collate function properly batches graph data with shifted edge indices.
    
    Args:
        stock_features_train: Training time series features
        x_graph_train: Training graph node features
        true_returns: Training labels
        stock_features_test: Test time series features
        x_graph_test: Test graph node features
        edge_index: Graph edge indices
        edge_weight: Graph edge weights
        batch_size: Batch size for training (default 32 per paper)
    
    Returns:
        train_loader: Combined training data loader
        test_loader: Combined test data loader
    """
    # Convert to tensors properly (fixes numpy array warning)
    X_train_time_series = torch.tensor(np.array(stock_features_train), dtype=torch.float32)
    X_train_graph = torch.tensor(np.array(x_graph_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(true_returns), dtype=torch.float32)
    
    X_test_time_series = torch.tensor(np.array(stock_features_test), dtype=torch.float32)
    X_test_graph = torch.tensor(np.array(x_graph_test), dtype=torch.float32)
    # Dummy labels for test (we don't use them)
    y_test_dummy = torch.zeros(len(X_test_time_series), X_test_graph.shape[1], dtype=torch.float32)
    
    # Create combined training dataset
    train_dataset = CombinedDataset(X_train_time_series, X_train_graph, y_train)
    
    num_train_samples = len(train_dataset)
    print(f"DEBUG: num_train_samples = {num_train_samples}, requested batch_size = {batch_size}")
    
    # Handle edge case of empty dataset
    if num_train_samples == 0:
        raise ValueError("Training dataset is empty! Check data processing pipeline.")
    
    # Adjust batch size if we have fewer samples than batch_size
    effective_batch_size = min(batch_size, num_train_samples)
    if effective_batch_size < batch_size:
        print(f"Note: Adjusted batch_size from {batch_size} to {effective_batch_size} (only {num_train_samples} training samples)")
    
    # Create collate function with edge data bound
    train_collate_fn = partial(combined_collate_fn, edge_index=edge_index, edge_weight=edge_weight)
    
    # Training loader - never drop last batch to ensure we use all data
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=False,  # Always keep all data
        collate_fn=train_collate_fn
    )
    
    print(f"DEBUG: Created train_loader with {len(train_loader)} batches")
    
    # Create combined test dataset
    test_dataset = CombinedDataset(X_test_time_series, X_test_graph, y_test_dummy)
    
    # Test collate function
    test_collate_fn = partial(combined_collate_fn, edge_index=edge_index, edge_weight=edge_weight)
    
    # Test loader with batch_size=1 for inference
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_collate_fn
    )
    
    return train_loader, test_loader

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


def model_train_predict(num_models, num_epochs, save_path, model_dt, kdcode_last, df3_2_dt, train_loader, test_loader):
    """
    Train and evaluate models with batch_size=32 support.
    
    Uses combined data loader that keeps time series and graph data synchronized.
    
    Args:
        num_models: Number of model runs for averaging
        num_epochs: Maximum training epochs
        save_path: Path to save models and predictions
        model_dt: Date identifier for model files
        kdcode_last: List of stock codes
        df3_2_dt: List of test dates
        train_loader: Combined training data loader
        test_loader: Combined test data loader
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_stocks = len(kdcode_last)
    
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
            num_samples = 0
            
            # Iterate through combined loader
            for time_series, labels, graph_features, edge_index, edge_weight, n_stocks in train_loader:
                batch_size_actual = time_series.shape[0]
                
                # Move data to device
                time_series = time_series.to(device)
                labels = labels.to(device)
                graph_features = graph_features.to(device)
                edge_index = edge_index.to(device)
                edge_weight = edge_weight.to(device)

                optimizer.zero_grad()
                
                # Forward pass with batched data
                outputs = model(time_series, graph_features, edge_index, edge_weight, n_stocks)
                
                # outputs: (batch, num_stocks), labels: (batch, num_stocks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * batch_size_actual
                num_samples += batch_size_actual

            # Safeguard against empty loader
            if num_samples == 0:
                raise ValueError(f"No training samples processed in epoch {epoch+1}! train_loader appears empty.")
            
            epoch_loss = running_loss / num_samples
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
            model.load_state_dict(torch.load(best_path, weights_only=True))
        else:
            # Fall back to last available epoch
            for epoch in range(num_epochs - 1, -1, -1):
                epoch_path = save_path + 'model_' + str(num) + '/' + model_dt + '_' + str(epoch) + '.pth'
                if os.path.exists(epoch_path):
                    model.load_state_dict(torch.load(epoch_path, weights_only=True))
                    break
        
        model.eval()
        with torch.no_grad():
            for idx, (time_series, _, graph_features, edge_index, edge_weight, n_stocks) in enumerate(test_loader):
                # Move data to device
                time_series = time_series.to(device)
                graph_features = graph_features.to(device)
                edge_index = edge_index.to(device)
                edge_weight = edge_weight.to(device)

                # Forward pass (batch_size=1 for test)
                outputs = model(time_series, graph_features, edge_index, edge_weight, n_stocks)
                
                # outputs: (1, num_stocks) -> squeeze to (num_stocks,)
                prediction = outputs.squeeze(0).cpu().numpy().tolist()
                
                # Build prediction dataframe
                data_all = []
                for i in range(len(prediction)):
                    data_all.append([kdcode_last[i], df3_2_dt[idx], round(prediction[i], 5)])
                df = pd.DataFrame(columns=['kdcode', 'dt', 'score'], data=data_all)
                
                # Save predictions
                pred_path = save_path + 'prediction_' + str(num) + '/0/'
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)
                df.to_csv(pred_path + df3_2_dt[idx] + '.csv', header=True, index=False, encoding='utf_8_sig')
                    

# ============================================================================
# MAIN EXECUTION - Paper Methodology (Section 4.1.1)
# ============================================================================
# Paper splits:
# - Training: Jan 1, 2018 to Dec 31, 2021 (we use 2017 for more data with S&P 500)
# - Validation: Jan 1, 2022 to Dec 31, 2022
# - Testing: Jan 1, 2023 to Dec 31, 2023
# ============================================================================

filename = 'sp500_yf_download.csv'

# Paper uses 6 features: Open, High, Low, Close, Volume, Turnover
feature_cols = ['close', 'open', 'high', 'low', 'volume', 'turnover']
num_features = len(feature_cols)

# Paper hyperparameters (Table 2)
judge_value = 0.7   # Correlation threshold for graph edges
label_t = 5         # Forward return period (days) - adjust based on paper
his_t = 10          # Historical window size (10 days per paper Section 4.1.3)
num_models = 10     # Paper uses 10 runs for averaging (Section 4.1.2)
num_epochs = 100    # Maximum epochs with early stopping

# Paper hyperparameters (Table 2)
CONFIG = {
    'gru_hidden_sizes': [32, 10],  # Two-layer GRU per paper
    'hidden_size_gat1': 32,        # Paper: 32
    'output_gat1': 4,              # Paper: 4
    'gat_heads': 4,                # Paper: 4 heads
    'hidden_size_gat2': 32,        # Paper: 32
    'num_hidden_states': 32,       # Paper: 32 latent vectors (d_r)
    'cross_attn_heads': 4,         # Paper: 4 heads for cross-attention
    'learning_rate': 0.0002,       # Paper: 0.0002
    'early_stopping_patience': 10, # Early stopping patience
    'batch_size': 32,              # Paper Table 2: batch_size = 32
}

# Date splits following paper methodology (Section 4.1.1)
# Using 2017 as training start since our S&P 500 data starts from 2017
TRAIN_START = '2017-01-01'
TRAIN_END = '2021-12-31'
VAL_START = '2022-01-01'
VAL_END = '2022-12-31'
TEST_START = '2023-01-01'
TEST_END = '2023-12-31'

# Output directory
save_path = f'paper_style_output_{judge_value}_{label_t}_{his_t}/'
os.makedirs(save_path, exist_ok=True)

print("=" * 80)
print("MCI-GRU Training - Paper Methodology")
print("=" * 80)
print(f"Training period: {TRAIN_START} to {TRAIN_END}")
print(f"Validation period: {VAL_START} to {VAL_END}")
print(f"Test period: {TEST_START} to {TEST_END}")
print(f"Historical window (his_t): {his_t} days")
print(f"Forward return period (label_t): {label_t} days")
print(f"Number of model runs: {num_models}")
print(f"Batch size: {CONFIG['batch_size']}")
print("=" * 80)

# Step 1: Prepare data with fixed train/val/test split
print("\n[Step 1] Preparing data...")
data_dict = prepare_data_paper_style(
    filename=filename,
    feature_cols=feature_cols,
    his_t=his_t,
    train_start=TRAIN_START,
    train_end=TRAIN_END,
    val_start=VAL_START,
    val_end=VAL_END,
    test_start=TEST_START,
    test_end=TEST_END,
    corr_lookback_days=252  # 1 year of data for correlation per paper
)

kdcode_last = data_dict['kdcode_last']
test_dates = data_dict['test_dates']
edge_index = data_dict['edge_index']
edge_weight = data_dict['edge_weight']

# Step 2: Create data loaders
print("\n[Step 2] Creating data loaders...")
train_loader, val_loader, test_loader = create_data_loaders_paper(
    data_dict=data_dict,
    edge_index=edge_index,
    edge_weight=edge_weight,
    batch_size=CONFIG['batch_size']
)

# Step 3: Train multiple models (paper uses 10 runs and averages)
print("\n[Step 3] Training models...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

all_predictions = []

for model_id in range(num_models):
    print(f"\n{'='*60}")
    print(f"Training Model {model_id + 1}/{num_models}")
    print(f"{'='*60}")
    
    # Create fresh model
    model = create_model(num_features, CONFIG).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Train with validation-based early stopping
    best_val_loss = train_model_paper_style(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        patience=CONFIG['early_stopping_patience'],
        device=device,
        save_path=save_path,
        model_id=model_id
    )
    
    print(f"Model {model_id + 1} training complete. Best val loss: {best_val_loss:.6f}")
    
    # Load best model for inference
    best_model_path = os.path.join(save_path, f'model_{model_id}', 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    # Run inference on test set
    print(f"Running inference on test set...")
    predictions = run_inference_paper_style(
        model=model,
        test_loader=test_loader,
        kdcode_last=kdcode_last,
        test_dates=test_dates,
        device=device,
        save_path=save_path,
        model_id=model_id
    )
    all_predictions.append(predictions)

# Step 4: Average predictions across all models (as per paper Section 4.1.2)
print("\n[Step 4] Averaging predictions across models...")
avg_predictions = np.mean(all_predictions, axis=0)

# Save averaged predictions
avg_pred_path = os.path.join(save_path, 'averaged_predictions')
os.makedirs(avg_pred_path, exist_ok=True)

for idx, date in enumerate(test_dates):
    if idx < len(avg_predictions):
        data = [[kdcode_last[i], date, round(float(avg_predictions[idx][i]), 5)] 
                for i in range(len(kdcode_last))]
        df = pd.DataFrame(columns=['kdcode', 'dt', 'score'], data=data)
        df.to_csv(os.path.join(avg_pred_path, f'{date}.csv'), index=False)

print("\n" + "=" * 80)
print("Training Complete!")
print("=" * 80)
print(f"Models saved to: {save_path}")
print(f"Averaged predictions saved to: {avg_pred_path}")
print(f"Number of test dates: {len(test_dates)}")
print(f"Number of stocks: {len(kdcode_last)}")
print("=" * 80)


# ============================================================================
# LEGACY CODE (kept for reference - rolling window approach)
# Uncomment below to use the original rolling monthly window approach
# ============================================================================
"""
dts_all =[
['2022-11-30', '2022-11-01', '2022-12-01', '2022-12-31', '2023-01-01', '2023-01-31'],
['2022-12-31', '2022-12-01', '2023-01-01', '2023-01-31', '2023-02-01', '2023-02-28'],
...
]

for dts_one in tqdm(dts_all):
    kdcode_last, df3_1_dt, df3_2_dt, stock_features_train, stock_features_test, x_graph_train, x_graph_test, edge_index, edge_weight, true_returns = fun_process_data_all(dts_one, filename, feature_cols, judge_value, label_t, his_t)
    train_loader, test_loader = model_data(stock_features_train, x_graph_train, true_returns, stock_features_test, x_graph_test, edge_index, edge_weight, batch_size=CONFIG['batch_size'])
    model_train_predict(num_models, num_epochs, save_path, dts_one[3], kdcode_last, df3_2_dt, train_loader, test_loader)
"""