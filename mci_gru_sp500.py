import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
from torch_geometric.nn import GATConv
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================
# CONFIGURATION - UPDATE THESE PATHS AS NEEDED
# ==============================================
TICKER_CSV_PATH = "tickers.csv"  # <-- UPDATE THIS PATH

TRAIN_START = "2018-01-01"
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2023-12-31"

DATA_START = "2017-01-01"  # 1 year buffer for correlation
DATA_END = "2023-12-31"

CONFIG = {
    "hist_days": 10,
    "label_days": 5,
    "judge_value": 0.8,
    "batch_size": 1,
    "learning_rate": 0.001,
    "num_epochs": 5,
    "early_stopping_patience": 10,
    "hidden_size": 256,
    "hidden_size_gat1": 5,
    "output_gat1": 256,
    "gat_heads": 4,
    "hidden_size_gat2": 5,
    "num_hidden_states": 4,
    "top_k": 10,
}


def generate_sp500_tickers_csv(output_path="sp500_tickers.csv"):
    """
    Fetch S&P 500 constituents from Wikipedia and save to CSV.
    Run this once to create your tickers file.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]

    tickers_df = pd.DataFrame(
        {
            "ticker": sp500_table["Symbol"].str.replace(".", "-", regex=False),
            "company_name": sp500_table["Security"],
            "sector": sp500_table["GICS Sector"],
        }
    )

    tickers_df.to_csv(output_path, index=False)
    print(f"Saved {len(tickers_df)} tickers to {output_path}")

    return output_path


def load_tickers_from_csv(filepath):
    """
    Load ticker symbols from a CSV file.

    Expected CSV format:
        - Must have a column named 'ticker', 'symbol', 'Ticker', or 'Symbol'
        - One ticker per row
    """
    df = pd.read_csv(filepath)

    possible_columns = ["ticker", "symbol", "Ticker", "Symbol", "TICKER", "SYMBOL"]
    ticker_column = None
    for col in possible_columns:
        if col in df.columns:
            ticker_column = col
            break

    if ticker_column is None:
        ticker_column = df.columns[0]
        print(
            f"Warning: No standard ticker column found. Using first column: '{ticker_column}'"
        )

    tickers = df[ticker_column].astype(str).tolist()
    tickers = [t.strip().replace(".", "-") for t in tickers]
    tickers = [t for t in tickers if t and t != "nan"]

    print(f"Loaded {len(tickers)} tickers from {filepath}")
    return tickers


def download_stock_data(tickers, start_date, end_date, min_days=200):
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
            if len(df) > min_days:
                stock_data[ticker] = df[
                    ["Open", "High", "Low", "Close", "Volume"]
                ].copy()
            else:
                failed_tickers.append(ticker)
        except Exception:
            failed_tickers.append(ticker)

    print(f"Successfully downloaded: {len(stock_data)} stocks")
    print(f"Failed: {len(failed_tickers)} stocks")
    return stock_data


def compute_features(stock_data):
    """
    Compute 5 features per stock per day:
    1. Close price
    2. Open price
    3. High price
    4. Low price
    5. Volume

    Also compute daily returns as labels.
    """
    processed_data = {}

    for ticker, df in stock_data.items():
        df = df.copy()
        df["Return"] = df["Close"].pct_change()
        df = df.dropna()
        processed_data[ticker] = df

    return processed_data


def filter_extreme_3sigma(series, n=3):
    values = series.values if hasattr(series, "values") else series
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std == 0:
        return series
    max_range = mean + n * std
    min_range = mean - n * std
    return np.clip(series, min_range, max_range)


def standardize_zscore(series):
    values = series.values if hasattr(series, "values") else series
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std == 0:
        return series - mean
    return (series - mean) / std


def process_daily_df_std(df, feature_cols):
    df = df.copy()
    for c in feature_cols:
        df[c] = filter_extreme_3sigma(df[c])
        df[c] = standardize_zscore(df[c])
    return df


def align_stock_data(processed_data, required_start=None, required_end=None):
    """
    Align all stocks to common trading dates.
    Remove stocks with missing data on common dates.
    """
    if required_start is not None:
        required_start = pd.Timestamp(required_start)
    if required_end is not None:
        required_end = pd.Timestamp(required_end)

    if not processed_data:
        raise ValueError("No stock data available after download/processing.")

    global_min = min(df.index.min() for df in processed_data.values())
    global_max = max(df.index.max() for df in processed_data.values())

    if required_start is not None and required_start < global_min:
        print(
            f"Adjusting required_start from {required_start.date()} "
            f"to {global_min.date()} based on available data."
        )
        required_start = global_min

    if required_end is not None and required_end > global_max:
        print(
            f"Adjusting required_end from {required_end.date()} "
            f"to {global_max.date()} based on available data."
        )
        required_end = global_max

    filtered_data = {}
    for ticker, df in processed_data.items():
        if required_start is not None and df.index.min() > required_start:
            continue
        if required_end is not None and df.index.max() < required_end:
            continue
        filtered_data[ticker] = df

    if not filtered_data:
        raise ValueError(
            "No tickers cover the required date range. "
            "This can happen if the end date is a non-trading day or "
            "if many tickers lack full coverage. "
            "Check your CSV tickers and date configuration."
        )

    removed_count = len(processed_data) - len(filtered_data)
    if removed_count > 0:
        print(f"Removed {removed_count} tickers without full date coverage.")

    date_sets = [set(df.index) for df in filtered_data.values()]
    common_dates = sorted(set.intersection(*date_sets))

    aligned_data = {}
    for ticker, df in filtered_data.items():
        aligned_data[ticker] = df.loc[common_dates].copy()

    print(f"Common trading dates: {len(common_dates)}")
    print(f"Aligned stocks: {len(aligned_data)}")

    return aligned_data, common_dates


def normalize_features(aligned_data, common_dates):
    """
    Per-day standardization with 3-sigma clipping (paper implementation).
    """
    feature_cols = ["Close", "Open", "High", "Low", "Volume"]
    tickers = sorted(aligned_data.keys())

    normalized_data = {t: df.copy() for t, df in aligned_data.items()}

    for dt in common_dates:
        day_df = pd.DataFrame(
            {t: aligned_data[t].loc[dt, feature_cols] for t in tickers}
        ).T
        day_df = process_daily_df_std(day_df, feature_cols)
        for t in tickers:
            normalized_data[t].loc[dt, feature_cols] = day_df.loc[t, feature_cols]

    return normalized_data


def compute_correlation_matrix(aligned_data, end_date, lookback_days=252):
    """
    Compute Pearson correlation matrix based on trailing returns.
    """
    tickers = sorted(aligned_data.keys())

    returns_dict = {}
    for ticker in tickers:
        df = aligned_data[ticker]
        mask = df.index <= end_date
        returns = df.loc[mask, "Return"].iloc[-lookback_days:]
        returns_dict[ticker] = returns.values

    returns_matrix = np.column_stack([returns_dict[t] for t in tickers])
    correlation_matrix = np.corrcoef(returns_matrix.T)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

    return correlation_matrix, tickers


def build_edge_index(correlation_matrix, judge_value=0.8):
    """
    Build PyTorch Geometric edge_index from correlation matrix.
    """
    N = correlation_matrix.shape[0]

    sources = []
    targets = []
    weights = []

    for i in range(N):
        for j in range(i + 1, N):
            weight = correlation_matrix[i, j]
            if weight > judge_value:
                sources.append(i)
                targets.append(j)
                weights.append(weight)
                sources.append(j)
                targets.append(i)
                weights.append(weight)

    edge_index = torch.LongTensor([sources, targets])
    edge_weight = torch.FloatTensor(weights)

    print(f"Graph edges: {edge_index.shape[1]} (avg degree: {edge_index.shape[1] / N:.2f})")

    return edge_index, edge_weight


def create_dataset(normalized_data, aligned_data, tickers, dates, hist_days=10, label_days=5):
    """
    Create dataset with sliding window approach.
    """
    feature_cols = ["Close", "Open", "High", "Low", "Volume"]

    all_dates = normalized_data[tickers[0]].index.tolist()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    data_array = np.stack(
        [normalized_data[t][feature_cols].values for t in tickers], axis=1
    )
    close_array = np.stack(
        [aligned_data[t]["Close"].values for t in tickers], axis=1
    )

    samples = []

    for date in dates:
        if date not in date_to_idx:
            continue

        t = date_to_idx[date]
        if t < hist_days - 1 or t + label_days >= len(all_dates) or t + 1 >= len(all_dates):
            continue

        features = data_array[t - hist_days + 1 : t + 1]
        label_returns = close_array[t + label_days] / close_array[t + 1] - 1
        labels = (
            pd.Series(label_returns)
            .rank(ascending=True, pct=True)
            .values
        )

        samples.append(
            {
                "features": torch.FloatTensor(features),
                "labels": torch.FloatTensor(labels),
                "future_returns": torch.FloatTensor(label_returns),
                "date": date,
            }
        )

    print(f"Created {len(samples)} samples")
    return samples


def split_dataset(samples, train_end, val_end):
    """Split samples by date into train/val/test sets."""
    train_samples = [s for s in samples if s["date"] <= pd.Timestamp(train_end)]
    val_samples = [
        s
        for s in samples
        if pd.Timestamp(train_end) < s["date"] <= pd.Timestamp(val_end)
    ]
    test_samples = [s for s in samples if s["date"] > pd.Timestamp(val_end)]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    if len(train_samples) == 0 or len(val_samples) == 0:
        raise ValueError(
            "Train/validation splits are empty. "
            "This usually means the aligned date range doesn't cover TRAIN/VAL. "
            "Check ticker coverage and date settings."
        )
    return train_samples, val_samples, test_samples


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
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
        r_gate, u_gate = gates.chunk(2, dim=2)
        r_gate = torch.sigmoid(r_gate)
        u_gate = torch.sigmoid(u_gate)

        h_hat = self.tanh(r_gate * hidden)
        new_hidden = u_gate * hidden + (1 - u_gate) * h_hat
        return new_hidden


class GATLayer(nn.Module):
    def __init__(self, hidden_size_gat1, output_gat1, in_channels, heads=1):
        super().__init__()
        self.gat1 = GATConv(
            in_channels, hidden_size_gat1, heads=heads, concat=True, edge_dim=1
        )
        self.gat2 = GATConv(
            hidden_size_gat1 * heads, output_gat1, heads=1, concat=False, edge_dim=1
        )

    def forward(self, x, edge_index, edge_weight):
        x = self.gat1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x


class GATLayer_1(nn.Module):
    def __init__(self, hidden_size_gat2, in_channels, out_channels, heads=1):
        super().__init__()
        self.gat1 = GATConv(
            in_channels, hidden_size_gat2, heads=heads, concat=True, edge_dim=1
        )
        self.gat2 = GATConv(
            hidden_size_gat2 * heads, out_channels, heads=1, concat=False, edge_dim=1
        )

    def forward(self, x, edge_index, edge_weight):
        x = self.gat1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5

    def forward(self, query, key, value):
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
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5

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
    def __init__(
        self,
        input_size,
        hidden_size,
        hidden_size_gat1,
        output_gat1,
        gat_heads,
        hidden_size_gat2,
        num_hidden_states,
    ):
        super().__init__()
        self.attention_gru = AttentionGRUCell(input_size, hidden_size)
        self.gat_layer = GATLayer(
            hidden_size_gat1, output_gat1, in_channels=input_size, heads=gat_heads
        )
        self.cross_attention = CrossAttention(hidden_size)
        self.num_hidden_states = num_hidden_states
        self.market_hidden_states_1 = nn.Parameter(
            torch.randn(num_hidden_states, hidden_size)
        )
        self.market_hidden_states_2 = nn.Parameter(
            torch.randn(num_hidden_states, hidden_size)
        )
        self.self_attention = SelfAttention(hidden_size * 4)
        self.final_gat = GATLayer_1(hidden_size_gat2, hidden_size * 4, 1)
        self.relu = nn.ReLU()

    def forward(self, x_time_series, x_graph, edge_index, edge_weight):
        batch_size, num_samples, num_time_steps, _ = x_time_series.size()
        h_gru = torch.zeros(
            batch_size, num_samples, self.attention_gru.hidden_size, device=x_time_series.device
        )
        for t in range(num_time_steps):
            h_gru = self.attention_gru(x_time_series[:, :, t, :], h_gru)
        h_gru_1 = h_gru[-1, :, :]

        x_gat = self.gat_layer(x_graph, edge_index, edge_weight)
        stock_rep_1 = self.cross_attention(
            h_gru_1.unsqueeze(1), self.market_hidden_states_1, self.market_hidden_states_1
        ).squeeze(1)
        stock_rep_2 = self.cross_attention(
            x_gat.unsqueeze(1), self.market_hidden_states_2, self.market_hidden_states_2
        ).squeeze(1)

        concatenated_output = torch.cat(
            [h_gru_1, x_gat, stock_rep_1, stock_rep_2], dim=1
        )
        attention_output = self.self_attention(concatenated_output.unsqueeze(1)).squeeze(1)
        out = self.final_gat(attention_output, edge_index, edge_weight)
        out = self.relu(out)
        return out.squeeze(1)


def train_model(model, train_samples, val_samples, edge_index, edge_weight, config, device):
    """
    Train the MCI-GRU model.
    """
    model = model.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_train_loss = 0.0

        np.random.shuffle(train_samples)

        for sample in train_samples:
            features = sample["features"].to(device).unsqueeze(0)
            labels = sample["labels"].to(device)

            optimizer.zero_grad()
            predictions = model(features, features[0, -1], edge_index, edge_weight)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_samples)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for sample in val_samples:
                features = sample["features"].to(device).unsqueeze(0)
                labels = sample["labels"].to(device)

                predictions = model(features, features[0, -1], edge_index, edge_weight)
                loss = criterion(predictions, labels)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_samples)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch {epoch+1}/{config['num_epochs']}: "
            f"Train Loss = {epoch_train_loss:.6f}, Val Loss = {epoch_val_loss:.6f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    return model, train_losses, val_losses


def generate_predictions(model, samples, edge_index, edge_weight, device):
    """
    Generate predictions for all samples.
    """
    model.eval()
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    results = []

    with torch.no_grad():
        for sample in samples:
            features = sample["features"].to(device).unsqueeze(0)
            labels = sample["labels"].to(device)
            predictions = model(features, features[0, -1], edge_index, edge_weight)

            results.append(
                {
                    "date": sample["date"],
                    "predictions": predictions.cpu().numpy(),
                    "labels": labels.cpu().numpy(),
                    "future_returns": sample["future_returns"].numpy(),
                }
            )

    return results


def backtest_strategy(results, tickers, top_k=10):
    """
    Backtest the top-k stock selection strategy.
    """
    portfolio_returns = []
    dates = []

    for result in results:
        date = result["date"]
        preds = result["predictions"]
        actual_returns = result["future_returns"]

        top_k_indices = np.argsort(preds)[-top_k:]
        portfolio_return = np.mean(actual_returns[top_k_indices])

        portfolio_returns.append(portfolio_return)
        dates.append(date)

    returns_series = pd.Series(portfolio_returns, index=dates)
    return returns_series


def calculate_metrics(returns_series):
    """
    Calculate all performance metrics from the paper.
    """
    returns = returns_series.values
    T = len(returns)
    cumulative = np.cumprod(1 + returns)

    total_return = cumulative[-1] - 1
    ARR = (1 + total_return) ** (252 / T) - 1
    AVol = np.std(returns) * np.sqrt(252)

    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    MDD = np.min(drawdown)

    ASR = ARR / AVol if AVol > 0 else 0
    CR = ARR / abs(MDD) if MDD != 0 else 0

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    IR = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

    metrics = {
        "ARR": ARR,
        "AVol": AVol,
        "MDD": MDD,
        "ASR": ASR,
        "CR": CR,
        "IR": IR,
        "Total_Return": total_return,
        "Num_Days": T,
    }

    return metrics


def calculate_prediction_metrics(results):
    """Calculate MSE and MAE for predictions."""
    all_preds = []
    all_labels = []

    for result in results:
        all_preds.extend(result["predictions"])
        all_labels.extend(result["labels"])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))

    return {"MSE": mse, "MAE": mae}


def plot_results(
    returns_series, benchmark_returns=None, title="MCI-GRU Portfolio Performance"
):
    """
    Plot cumulative returns and drawdown.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    cumulative = (1 + returns_series).cumprod()
    axes[0].plot(
        cumulative.index, cumulative.values, label="MCI-GRU Strategy", color="red"
    )

    if benchmark_returns is not None:
        bench_cumulative = (1 + benchmark_returns).cumprod()
        axes[0].plot(
            bench_cumulative.index,
            bench_cumulative.values,
            label="S&P 500 Benchmark",
            color="blue",
            alpha=0.7,
        )

    axes[0].set_title(title)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    axes[1].plot(drawdown.index, drawdown.values, color="red", linewidth=0.5)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("portfolio_performance.png", dpi=150)
    plt.show()


def print_metrics(metrics, pred_metrics, title="Performance Metrics"):
    """Pretty print all metrics."""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print(f"  ARR (Annualized Return):     {metrics['ARR']*100:>8.2f}%")
    print(f"  AVol (Annualized Volatility): {metrics['AVol']*100:>8.2f}%")
    print(f"  MDD (Maximum Drawdown):       {metrics['MDD']*100:>8.2f}%")
    print(f"  ASR (Sharpe Ratio):           {metrics['ASR']:>8.3f}")
    print(f"  CR (Calmar Ratio):            {metrics['CR']:>8.3f}")
    print(f"  IR (Information Ratio):       {metrics['IR']:>8.3f}")
    print("-" * 50)
    print(f"  MSE:                          {pred_metrics['MSE']:>8.6f}")
    print(f"  MAE:                          {pred_metrics['MAE']:>8.6f}")
    print("=" * 50 + "\n")


def main(ticker_csv_path):
    """
    Main execution pipeline for MCI-GRU on S&P 500 with rolling monthly windows.
    """
    device = get_device()
    print(f"Using device: {device}")
    print("=" * 60)
    print("  MCI-GRU Stock Prediction Model - S&P 500")
    print("=" * 60)

    print("\n[Step 1] Loading tickers from CSV...")
    tickers = load_tickers_from_csv(ticker_csv_path)

    print(f"\n[Step 1] Downloading historical data...")
    stock_data = download_stock_data(tickers, DATA_START, DATA_END)

    print("\n[Step 2] Computing features...")
    processed_data = compute_features(stock_data)

    print("\n[Step 2] Aligning stock data...")
    aligned_data, common_dates = align_stock_data(
        processed_data, required_start=TRAIN_START, required_end=TEST_END
    )
    tickers = sorted(aligned_data.keys())

    print("\n[Step 2] Normalizing features...")
    normalized_data = normalize_features(aligned_data, common_dates)

    print("\n[Step 3] Creating dataset samples...")
    all_dates = [d for d in common_dates if d >= pd.Timestamp(TRAIN_START)]
    samples = create_dataset(
        normalized_data,
        aligned_data,
        tickers,
        all_dates,
        hist_days=CONFIG["hist_days"],
        label_days=CONFIG["label_days"],
    )

    dts_all = [
        ["2022-11-30", "2022-11-01", "2022-12-01", "2022-12-31", "2023-01-01", "2023-01-31"],
        ["2022-12-31", "2022-12-01", "2023-01-01", "2023-01-31", "2023-02-01", "2023-02-28"],
        ["2023-01-31", "2023-01-01", "2023-02-01", "2023-02-28", "2023-03-01", "2023-03-31"],
        ["2023-02-28", "2023-02-01", "2023-03-01", "2023-03-31", "2023-04-01", "2023-04-30"],
        ["2023-03-31", "2023-03-01", "2023-04-01", "2023-04-30", "2023-05-01", "2023-05-31"],
        ["2023-04-30", "2023-04-01", "2023-05-01", "2023-05-31", "2023-06-01", "2023-06-30"],
        ["2023-05-31", "2023-05-01", "2023-06-01", "2023-06-30", "2023-07-01", "2023-07-31"],
        ["2023-06-30", "2023-06-01", "2023-07-01", "2023-07-31", "2023-08-01", "2023-08-31"],
        ["2023-07-31", "2023-07-01", "2023-08-01", "2023-08-31", "2023-09-01", "2023-09-30"],
        ["2023-08-31", "2023-08-01", "2023-09-01", "2023-09-30", "2023-10-01", "2023-10-31"],
        ["2023-09-30", "2023-09-01", "2023-10-01", "2023-10-31", "2023-11-01", "2023-11-30"],
        ["2023-10-31", "2023-10-01", "2023-11-01", "2023-11-30", "2023-12-01", "2023-12-31"],
    ]

    all_results = []

    for window_idx, dts_one in enumerate(dts_all, start=1):
        corr_end = pd.Timestamp(dts_one[0])
        train_start = pd.Timestamp(dts_one[1])
        val_start = pd.Timestamp(dts_one[2])
        val_end = pd.Timestamp(dts_one[3])
        test_start = pd.Timestamp(dts_one[4])
        test_end = pd.Timestamp(dts_one[5])

        print(f"\n[Window {window_idx}/{len(dts_all)}] {test_start.date()} to {test_end.date()}")

        corr_matrix, _ = compute_correlation_matrix(aligned_data, corr_end)
        edge_index, edge_weight = build_edge_index(corr_matrix, CONFIG["judge_value"])

        train_samples = [
            s for s in samples if train_start <= s["date"] <= val_end
        ]
        val_samples = [
            s for s in samples if val_start <= s["date"] <= val_end
        ]
        test_samples = [
            s for s in samples if test_start <= s["date"] <= test_end
        ]

        if len(val_samples) == 0:
            val_samples = train_samples

        model = StockPredictionModel(
            input_size=5,
            hidden_size=CONFIG["hidden_size"],
            hidden_size_gat1=CONFIG["hidden_size_gat1"],
            output_gat1=CONFIG["output_gat1"],
            gat_heads=CONFIG["gat_heads"],
            hidden_size_gat2=CONFIG["hidden_size_gat2"],
            num_hidden_states=CONFIG["num_hidden_states"],
        )

        model, train_losses, val_losses = train_model(
            model, train_samples, val_samples, edge_index, edge_weight, CONFIG, device
        )

        window_results = generate_predictions(
            model, test_samples, edge_index, edge_weight, device
        )
        all_results.extend(window_results)

    print("\n[Step 4] Running backtest across all windows...")
    test_returns = backtest_strategy(all_results, tickers, top_k=CONFIG["top_k"])

    metrics = calculate_metrics(test_returns)
    pred_metrics = calculate_prediction_metrics(all_results)
    print_metrics(metrics, pred_metrics, "Rolling Window Performance (2023)")

    print("\n[Step 5] Generating visualizations...")
    spy = yf.download("SPY", start=TEST_START, end=TEST_END, progress=False)
    spy_returns = spy["Close"].pct_change().dropna()
    common_test_dates = test_returns.index.intersection(spy_returns.index)
    spy_returns_aligned = spy_returns.loc[common_test_dates]
    test_returns_aligned = test_returns.loc[common_test_dates]
    plot_results(test_returns_aligned, spy_returns_aligned)

    print("\n✓ Pipeline complete!")
    return all_results, test_returns, metrics


if __name__ == "__main__":
    model, test_returns, metrics = main(TICKER_CSV_PATH)
