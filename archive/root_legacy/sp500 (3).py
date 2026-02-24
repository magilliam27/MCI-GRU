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
    
    result = []
    pool = multiprocessing.Pool(10)
    result = pool.starmap(generate_dataset, param_list)
    pool.close()
    pool.join()
    ds_data = np.concatenate([x for x in result if len(x) > 0])
    
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
    n = label_t
    c = 'close'
    label_column = 't'+str(n)+'_close_return_rate'
    label_last = 't'+str(n)+'_label'
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
        df = df_features_grouped.get_group(dt)
        mean_val = df[label_column].mean()
        df[label_column].fillna(mean_val, inplace=True)
        res.append(df)
    df_label = pd.concat(res)
    df_label = df_label.sort_values(['dt','kdcode'])
    df_label = df_label.reset_index(drop=True)
    df_label = df_label.groupby('dt').apply(lambda df: rank_labeling(df, col_label=label_last, col_return=label_column))
    if len(df_label) == len(kdcode_last)*len(df3_1_dt):
        label_list = df_label[label_last].values.tolist()
        true_returns=[]
        for i in range(len(df3_1_dt)):
            true_returns.append(label_list[i*len(kdcode_last):(i+1)*len(kdcode_last)])
        true_returns = np.array(true_returns)
    else:
        print("error")
    return true_returns

def fun_process_data_all(dts_one, filename, feature_cols, judge_value, label_t, his_t):
    df_org = pd.read_csv(filename)
    
    df_features_grouped = df_org.groupby('dt')
    res = []
    for dt in df_features_grouped.groups:
        df = df_features_grouped.get_group(dt)
        for column in feature_cols:
            mean_val = df[column].mean()
            df[column].fillna(mean_val, inplace=True)
        df = df.fillna(0.0)
        processed_df = process_daily_df_std(df, feature_cols)  
        res.append(processed_df)
    df = pd.concat(res)
    
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

class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.w_ih = nn.Linear(input_size, hidden_size * 2, bias=False)
        self.w_hh = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.attention = nn.Linear(hidden_size, input_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        attn_scores = self.attention(hidden)
        attn_weights = F.softmax(attn_scores, dim=1)
        x = x * attn_weights

        gates = self.w_ih(x) + self.w_hh(hidden)
        r_gate, u_gate = gates.chunk(2, 2)

        r_gate = torch.sigmoid(r_gate)
        u_gate = torch.sigmoid(u_gate)

        h_hat = self.tanh(r_gate * hidden)
        new_hidden = u_gate * hidden + (1 - u_gate) * h_hat

        return new_hidden

class GATLayer(nn.Module):
    def __init__(self, hidden_size_gat1, output_gat1, in_channels, out_channels, heads=1):
        super(GATLayer, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_size_gat1, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_size_gat1 * heads, output_gat1, heads=1, concat=False, edge_dim=1)

    def forward(self, x, edge_index, edge_weight):
        x = self.gat1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x
    
class GATLayer_1(nn.Module):
    def __init__(self, hidden_size_gat2, in_channels, out_channels, heads=1):
        super(GATLayer_1, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_size_gat2, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_size_gat2 * heads, out_channels, heads=1, concat=False, edge_dim=1)

    def forward(self, x, edge_index, edge_weight):
        x = self.gat1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x 

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, query, key, value):
        # print(f"Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        
        k = k.transpose(-2, -1)
        
        attn_weights = torch.matmul(q, k) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        return attn_output

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        k = k.transpose(-2, -1)
        
        attn_weights = torch.matmul(q, k) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        return attn_output
    
class StockPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_gat1, output_gat1, gat_in_channels, gat_out_channels, gat_heads, hidden_size_gat2, embed_dim, num_hidden_states):
        super(StockPredictionModel, self).__init__()
        self.attention_gru = AttentionGRUCell(input_size, hidden_size)
        self.gat_layer = GATLayer(hidden_size_gat1, output_gat1,  gat_in_channels, gat_out_channels, gat_heads)
        self.cross_attention = CrossAttention(hidden_size)
        self.num_hidden_states = num_hidden_states
        self.market_hidden_states_1 = nn.Parameter(torch.randn(num_hidden_states, hidden_size))
        self.market_hidden_states_2 = nn.Parameter(torch.randn(num_hidden_states, hidden_size))
        self.self_attention = SelfAttention(hidden_size * 4)
        self.final_gat = GATLayer_1(hidden_size_gat2, hidden_size * 4, 1, 1)
        self.relu = nn.ReLU()  
        # self.dim_reduction = nn.Linear(32, 16)
        
    def forward(self, x_time_series, x_graph, edge_index, edge_weight):
        batch_size, num_samples, num_time_steps, num_features = x_time_series.size()
        h_gru = torch.zeros(batch_size, num_samples, self.attention_gru.hidden_size).to(x_time_series.device)
        for t in range(num_time_steps):
            h_gru = self.attention_gru(x_time_series[:, :, t, :], h_gru)
        h_gru_1 = h_gru[-1,:,:]
        # print(h_gru_1.shape)

        x_gat = self.gat_layer(x_graph, edge_index, edge_weight)

        stock_rep_1 = self.cross_attention(h_gru_1.unsqueeze(1), self.market_hidden_states_1, self.market_hidden_states_1).squeeze(1)
        stock_rep_2 = self.cross_attention(x_gat.unsqueeze(1), self.market_hidden_states_2, self.market_hidden_states_2).squeeze(1)

        concatenated_output = torch.cat([h_gru_1, x_gat, stock_rep_1, stock_rep_2], dim=1)

        attention_output = self.self_attention(concatenated_output.unsqueeze(1)).squeeze(1)

        out = self.final_gat(attention_output, edge_index, edge_weight)
        out = self.relu(out)
        
        return out.squeeze(1)  


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

def model_train_predict(num_models, num_epochs, save_path, model_dt, kdcode_last, df3_2_dt, train_time_series_loader, train_graph_loader, X_test_time_series, test_graph_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for num in range(num_models):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = StockPredictionModel(
            input_size=num_features,
            hidden_size=256,
            hidden_size_gat1=5,
            output_gat1=256,
            gat_in_channels=num_features,
            gat_out_channels=4,
            gat_heads=4,
            hidden_size_gat2=5,
            embed_dim=256,
            num_hidden_states=4
        ).to(device)
        # print(model)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

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
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            save_path_1 = save_path +  'model_'+str(num) + '/' + model_dt + '_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), save_path_1)
            print(f"Model saved to {save_path_1}")
        print('Finished Training With Number ' + str(num))
    
    for num in tqdm(range(num_models)):
        for epoch in range(num_epochs):
            # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model = StockPredictionModel(
                input_size=num_features,
                hidden_size=256,
                hidden_size_gat1=5,
                output_gat1=256,
                gat_in_channels=num_features,
                gat_out_channels=4,
                gat_heads=4,
                hidden_size_gat2=5,
                embed_dim=256,
                num_hidden_states=4
            ).to(device)
            model.load_state_dict(torch.load(save_path +  'model_'+str(num) + '/' + model_dt + '_' + str(epoch) + '.pth'))
            model.eval()
            with torch.no_grad():
                index=0
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
                        one.append(round(prediction[i],5))
                        data_all.append(one)
                    df = pd.DataFrame()
                    df = pd.DataFrame(columns=['kdcode', 'dt', 'score'], data=data_all)
                    df.to_csv(save_path+'prediction_'+str(num)+'/'+str(epoch)+'/'+df3_2_dt[index]+'.csv', header=True, index=False, encoding='utf_8_sig')
                    index+=1
                    

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
filename = '/home/liyuante/dataset/sp500_2018_2023_new_1.csv'
feature_cols = ['close','open','high','low','volume']
num_features = len(feature_cols)
judge_value = 0.8
label_t = 5
his_t = 10
num_models = 20
num_epochs = 5
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