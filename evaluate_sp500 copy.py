"""
evaluate_sp500.py - Backtest and Evaluation Metrics for MCI-GRU Model

Implements the exact testing methodology from paper:
"MCI-GRU: Stock Prediction Model Based on Multi-Head Cross-Attention and Improved GRU"
(arXiv:2410.20679v3)

Paper Section 4.1.2 - Evaluation Metrics:
- ARR (Annualized Rate of Return)
- AVoL (Annualized Volatility)  
- MDD (Maximum Drawdown)
- ASR (Annualized Sharpe Ratio)
- CR (Calmar Ratio)
- IR (Information Ratio)
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)

Trading Strategy (Paper Section 4.1.2):
1. At close of day t, model generates prediction scores for each stock
2. Rank stocks by predicted score (expected return)
3. At open of day t+1, buy top-k stocks (paper uses k=10)
4. Equal-weighted portfolio, daily rebalancing
5. Transaction costs excluded
6. Run 10 times and average results

Usage:
    python evaluate_sp500.py --predictions_dir paper_style_output_0.8_5_10/averaged_predictions
"""

import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION (Paper Section 4.1.1 and 4.1.2)
# ============================================================================

DEFAULT_CONFIG = {
    'top_k': 10,                    # Paper: k=10 stocks in portfolio
    'trading_days_per_year': 252,   # Standard trading days
    'test_start': '2023-01-01',     # Paper Section 4.1.1
    'test_end': '2023-12-31',       # Paper Section 4.1.1
    'data_file': 'sp500_yf_download.csv',
    'label_t': 5,                   # Forward return period (days)
}


# ============================================================================
# METRIC CALCULATIONS (Paper Section 4.1.2)
# ============================================================================

def calculate_arr(daily_returns, trading_days=252):
    """
    Annualized Rate of Return (Paper Equation).
    
    Paper formula: ARR = (∏_{t=1}^{T}(1 + r_t))^(252/T) - 1
    
    Args:
        daily_returns: Array of daily portfolio returns
        trading_days: Number of trading days per year (252)
    
    Returns:
        Annualized rate of return
    """
    if len(daily_returns) == 0:
        return 0.0
    
    # Remove NaN values
    daily_returns = daily_returns[~np.isnan(daily_returns)]
    
    if len(daily_returns) == 0:
        return 0.0
    
    # Cumulative return
    cumulative = np.prod(1 + daily_returns)
    T = len(daily_returns)
    
    # Annualize
    arr = (cumulative ** (trading_days / T)) - 1
    
    return arr


def calculate_avol(daily_returns, trading_days=252):
    """
    Annualized Volatility (Paper Equation).
    
    Paper formula: AVoL = std(P_t/P_{t-1} - 1) * √252
    
    Args:
        daily_returns: Array of daily portfolio returns
        trading_days: Number of trading days per year (252)
    
    Returns:
        Annualized volatility (standard deviation)
    """
    if len(daily_returns) == 0:
        return 0.0
    
    # Remove NaN values
    daily_returns = daily_returns[~np.isnan(daily_returns)]
    
    if len(daily_returns) == 0:
        return 0.0
    
    return np.std(daily_returns, ddof=1) * np.sqrt(trading_days)


def calculate_mdd(portfolio_values):
    """
    Maximum Drawdown (Paper Equation).
    
    Paper formula: MDD = max_t((max_{i∈[1,t]} P_i - P_t) / max_{i∈[1,t]} P_i)
    
    Args:
        portfolio_values: Array of cumulative portfolio values (not returns)
    
    Returns:
        Maximum drawdown (negative value by convention)
    """
    if len(portfolio_values) == 0:
        return 0.0
    
    # Running maximum (peak)
    running_max = np.maximum.accumulate(portfolio_values)
    
    # Drawdown at each point
    drawdowns = (running_max - portfolio_values) / running_max
    
    # Maximum drawdown
    mdd = np.max(drawdowns)
    
    return -mdd  # Return as negative value (convention)


def calculate_asr(arr, avol):
    """
    Annualized Sharpe Ratio (Paper Equation).
    
    Paper formula: ASR = ARR / AVoL
    
    Args:
        arr: Annualized rate of return
        avol: Annualized volatility
    
    Returns:
        Annualized Sharpe ratio
    """
    if avol == 0 or np.isnan(avol):
        return 0.0
    
    return arr / avol


def calculate_cr(arr, mdd):
    """
    Calmar Ratio (Paper Equation).
    
    Paper formula: CR = ARR / |MDD|
    
    Args:
        arr: Annualized rate of return
        mdd: Maximum drawdown (negative value)
    
    Returns:
        Calmar ratio
    """
    if mdd == 0 or np.isnan(mdd):
        return 0.0
    
    return arr / abs(mdd)


def calculate_ir(portfolio_returns, benchmark_returns):
    """
    Information Ratio (Paper Equation).
    
    Paper formula: IR = mean(r_t - r_f,t) / std(r_t - r_f,t)
    where r_t is portfolio return and r_f,t is benchmark return
    
    Args:
        portfolio_returns: Array of daily portfolio returns
        benchmark_returns: Array of daily benchmark returns (equal-weighted market)
    
    Returns:
        Information ratio
    """
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    # Align lengths
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    portfolio_returns = portfolio_returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    # Excess returns
    excess_returns = portfolio_returns - benchmark_returns
    
    # Remove NaN
    excess_returns = excess_returns[~np.isnan(excess_returns)]
    
    if len(excess_returns) == 0:
        return 0.0
    
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    return np.mean(excess_returns) / std_excess


def calculate_mse(predictions, actuals):
    """
    Mean Squared Error.
    
    Args:
        predictions: Array of predicted values
        actuals: Array of actual values
    
    Returns:
        MSE value
    """
    if len(predictions) == 0:
        return 0.0
    
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions = predictions[mask]
    actuals = actuals[mask]
    
    if len(predictions) == 0:
        return 0.0
    
    return np.mean((predictions - actuals) ** 2)


def calculate_mae(predictions, actuals):
    """
    Mean Absolute Error.
    
    Args:
        predictions: Array of predicted values
        actuals: Array of actual values
    
    Returns:
        MAE value
    """
    if len(predictions) == 0:
        return 0.0
    
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions = predictions[mask]
    actuals = actuals[mask]
    
    if len(predictions) == 0:
        return 0.0
    
    return np.mean(np.abs(predictions - actuals))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_stock_data(filename, start_date=None, end_date=None):
    """
    Load stock data from CSV file.
    
    Args:
        filename: Path to stock data CSV
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
    
    Returns:
        DataFrame with stock data
    """
    print(f"Loading stock data from {filename}...")
    df = pd.read_csv(filename)
    
    # Ensure date format
    df['dt'] = pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d')
    
    # Filter by date range if specified
    if start_date:
        df = df[df['dt'] >= start_date]
    if end_date:
        df = df[df['dt'] <= end_date]
    
    # Sort by stock and date
    df = df.sort_values(['kdcode', 'dt']).reset_index(drop=True)
    
    print(f"  Loaded {len(df)} rows, {df['kdcode'].nunique()} stocks")
    print(f"  Date range: {df['dt'].min()} to {df['dt'].max()}")
    
    return df


def calculate_forward_returns(df, label_t=5):
    """
    Calculate forward returns (next label_t days).
    
    Paper uses the return from day t+1 to day t+label_t.
    
    Args:
        df: DataFrame with stock data
        label_t: Number of days for forward return
    
    Returns:
        DataFrame with forward returns added
    """
    print(f"Calculating {label_t}-day forward returns...")
    
    df = df.copy()
    df = df.sort_values(['kdcode', 'dt'])
    
    # Calculate forward close price
    df['close_t1'] = df.groupby('kdcode')['close'].shift(-1)
    df[f'close_t{label_t}'] = df.groupby('kdcode')['close'].shift(-label_t)
    
    # Forward return = close_{t+label_t} / close_{t+1} - 1
    # This matches the paper's label calculation
    df[f'forward_return_{label_t}d'] = df[f'close_t{label_t}'] / df['close_t1'] - 1
    
    # Also calculate 1-day return for daily portfolio tracking
    df['next_day_return'] = df['close_t1'] / df['close'] - 1
    
    return df


def load_predictions(predictions_dir):
    """
    Load model predictions from CSV files.
    
    Each file is named as YYYY-MM-DD.csv with columns: kdcode, dt, score
    
    Args:
        predictions_dir: Directory containing prediction CSV files
    
    Returns:
        DataFrame with all predictions
    """
    print(f"Loading predictions from {predictions_dir}...")
    
    pred_files = sorted(glob(os.path.join(predictions_dir, '*.csv')))
    
    if len(pred_files) == 0:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")
    
    all_preds = []
    for f in pred_files:
        df = pd.read_csv(f)
        all_preds.append(df)
    
    predictions_df = pd.concat(all_preds, ignore_index=True)
    predictions_df['dt'] = pd.to_datetime(predictions_df['dt']).dt.strftime('%Y-%m-%d')
    
    print(f"  Loaded {len(predictions_df)} predictions")
    print(f"  {len(pred_files)} dates, {predictions_df['kdcode'].nunique()} stocks")
    print(f"  Date range: {predictions_df['dt'].min()} to {predictions_df['dt'].max()}")
    
    return predictions_df


# ============================================================================
# TRADING STRATEGY SIMULATION (Paper Section 4.1.2)
# ============================================================================

def simulate_trading_strategy(predictions_df, stock_data_df, top_k=10, label_t=5):
    """
    Simulate the paper's trading strategy.
    
    Paper methodology:
    1. At close of day t, rank stocks by predicted score
    2. At open of day t+1, buy top-k stocks
    3. Equal-weighted portfolio
    4. Hold for 1 day (daily rebalancing)
    5. Transaction costs excluded
    
    Args:
        predictions_df: DataFrame with predictions (kdcode, dt, score)
        stock_data_df: DataFrame with stock data and returns
        top_k: Number of top stocks to select
        label_t: Forward return period for accuracy metrics
    
    Returns:
        Dictionary with daily returns and related data
    """
    print(f"\nSimulating trading strategy (top-{top_k} stocks)...")
    
    # Get unique prediction dates
    pred_dates = sorted(predictions_df['dt'].unique())
    print(f"  {len(pred_dates)} trading days")
    
    portfolio_returns = []
    benchmark_returns = []
    dates = []
    all_predictions = []
    all_actuals = []
    
    for date in pred_dates:
        # Get predictions for this date
        day_preds = predictions_df[predictions_df['dt'] == date].copy()
        
        if len(day_preds) < top_k:
            continue
        
        # Rank and select top-k stocks
        day_preds = day_preds.sort_values('score', ascending=False)
        top_stocks = day_preds.head(top_k)['kdcode'].tolist()
        
        # Get actual returns for these stocks
        # We use next_day_return for daily portfolio tracking
        day_stock_data = stock_data_df[stock_data_df['dt'] == date]
        
        if len(day_stock_data) == 0:
            continue
        
        # Portfolio return (equal-weighted top-k)
        top_k_returns = day_stock_data[
            day_stock_data['kdcode'].isin(top_stocks)
        ]['next_day_return'].dropna()
        
        if len(top_k_returns) == 0:
            continue
        
        portfolio_return = top_k_returns.mean()
        
        # Benchmark return (equal-weighted all stocks)
        all_returns = day_stock_data['next_day_return'].dropna()
        benchmark_return = all_returns.mean() if len(all_returns) > 0 else 0.0
        
        portfolio_returns.append(portfolio_return)
        benchmark_returns.append(benchmark_return)
        dates.append(date)
        
        # Collect predictions vs actuals for MSE/MAE
        for _, row in day_preds.iterrows():
            kdcode = row['kdcode']
            score = row['score']
            
            actual = day_stock_data[
                day_stock_data['kdcode'] == kdcode
            ][f'forward_return_{label_t}d']
            
            if len(actual) > 0 and not pd.isna(actual.values[0]):
                all_predictions.append(score)
                all_actuals.append(actual.values[0])
    
    print(f"  Completed simulation: {len(dates)} valid trading days")
    
    return {
        'portfolio_returns': np.array(portfolio_returns),
        'benchmark_returns': np.array(benchmark_returns),
        'dates': dates,
        'predictions': np.array(all_predictions),
        'actuals': np.array(all_actuals)
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate(predictions_dir, config=None):
    """
    Run full evaluation per paper methodology.
    
    Args:
        predictions_dir: Directory containing prediction CSV files
        config: Configuration dictionary (uses defaults if None)
    
    Returns:
        Dictionary of all evaluation metrics
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    print("\n" + "=" * 70)
    print("MCI-GRU MODEL EVALUATION")
    print("Paper: arXiv:2410.20679v3 (Section 4.1.2)")
    print("=" * 70)
    
    # Load predictions
    predictions_df = load_predictions(predictions_dir)
    
    # Load stock data
    stock_data = load_stock_data(
        config['data_file'],
        start_date=config['test_start'],
        end_date=config['test_end']
    )
    
    # Calculate forward returns
    stock_data = calculate_forward_returns(stock_data, label_t=config['label_t'])
    
    # Simulate trading strategy
    sim_results = simulate_trading_strategy(
        predictions_df=predictions_df,
        stock_data_df=stock_data,
        top_k=config['top_k'],
        label_t=config['label_t']
    )
    
    portfolio_returns = sim_results['portfolio_returns']
    benchmark_returns = sim_results['benchmark_returns']
    dates = sim_results['dates']
    
    if len(portfolio_returns) == 0:
        print("ERROR: No valid trading days found!")
        return None
    
    # Calculate cumulative portfolio values (starting at 1.0)
    portfolio_values = np.cumprod(1 + portfolio_returns)
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    
    arr = calculate_arr(portfolio_returns, config['trading_days_per_year'])
    avol = calculate_avol(portfolio_returns, config['trading_days_per_year'])
    mdd = calculate_mdd(portfolio_values)
    asr = calculate_asr(arr, avol)
    cr = calculate_cr(arr, mdd)
    ir = calculate_ir(portfolio_returns, benchmark_returns)
    
    # Prediction accuracy metrics
    mse = calculate_mse(sim_results['predictions'], sim_results['actuals'])
    mae = calculate_mae(sim_results['predictions'], sim_results['actuals'])
    
    # Additional metrics
    total_return = portfolio_values[-1] - 1 if len(portfolio_values) > 0 else 0
    benchmark_total = np.prod(1 + benchmark_returns) - 1
    excess_return = total_return - benchmark_total
    
    results = {
        'ARR': arr,
        'AVoL': avol,
        'MDD': mdd,
        'ASR': asr,
        'CR': cr,
        'IR': ir,
        'MSE': mse,
        'MAE': mae,
        'total_return': total_return,
        'benchmark_return': benchmark_total,
        'excess_return': excess_return,
        'num_trading_days': len(dates),
        'start_date': dates[0] if dates else None,
        'end_date': dates[-1] if dates else None,
    }
    
    return results


def print_results(results, model_name="MCI-GRU"):
    """
    Pretty print evaluation results in paper format.
    
    Args:
        results: Dictionary of evaluation metrics
        model_name: Name to display
    """
    print("\n" + "=" * 70)
    print(f"  EVALUATION RESULTS: {model_name}")
    print("=" * 70)
    print()
    
    # Paper metrics table format
    print("  Paper Metrics (Table 3/4 format):")
    print("  " + "-" * 66)
    print(f"  {'Metric':<25} {'Value':>12}  {'Goal':>20}")
    print("  " + "-" * 66)
    print(f"  {'ARR (Ann. Return)':<25} {results['ARR']:>12.4f}  {'Higher is better':>20}")
    print(f"  {'AVoL (Ann. Volatility)':<25} {results['AVoL']:>12.4f}  {'Lower is better':>20}")
    print(f"  {'MDD (Max Drawdown)':<25} {results['MDD']:>12.4f}  {'Less negative better':>20}")
    print(f"  {'ASR (Sharpe Ratio)':<25} {results['ASR']:>12.4f}  {'Higher is better':>20}")
    print(f"  {'CR (Calmar Ratio)':<25} {results['CR']:>12.4f}  {'Higher is better':>20}")
    print(f"  {'IR (Info Ratio)':<25} {results['IR']:>12.4f}  {'Higher is better':>20}")
    print("  " + "-" * 66)
    print(f"  {'MSE':<25} {results['MSE']:>12.6f}")
    print(f"  {'MAE':<25} {results['MAE']:>12.6f}")
    print("  " + "-" * 66)
    print()
    
    # Additional info
    print("  Additional Information:")
    print("  " + "-" * 66)
    print(f"  {'Total Portfolio Return':<25} {results['total_return']:>12.4f}  ({results['total_return']*100:.2f}%)")
    print(f"  {'Benchmark Return':<25} {results['benchmark_return']:>12.4f}  ({results['benchmark_return']*100:.2f}%)")
    print(f"  {'Excess Return':<25} {results['excess_return']:>12.4f}  ({results['excess_return']*100:.2f}%)")
    print(f"  {'Trading Days':<25} {results['num_trading_days']:>12}")
    print(f"  {'Test Period':<25} {results['start_date']} to {results['end_date']}")
    print("  " + "-" * 66)
    print()
    
    # Comparison hint
    print("  Paper S&P 500 Results (Table 4):")
    print("  " + "-" * 66)
    print("  MCI-GRU: ARR=0.456, AVoL=0.179, MDD=-0.129, ASR=2.549, CR=3.543, IR=2.197")
    print("  " + "-" * 66)
    print("=" * 70 + "\n")


def save_results(results, output_path):
    """
    Save results to CSV file.
    
    Args:
        results: Dictionary of evaluation metrics
        output_path: Path to save CSV
    """
    df = pd.DataFrame([results])
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def plot_equity_curve(predictions_dir, stock_data, config, output_path=None):
    """
    Plot equity curve similar to paper Figure 2.
    
    Args:
        predictions_dir: Directory with predictions
        stock_data: DataFrame with stock data
        config: Configuration dictionary
        output_path: Path to save plot (None = display)
    """
    import matplotlib.pyplot as plt
    
    predictions_df = load_predictions(predictions_dir)
    stock_data = calculate_forward_returns(stock_data, label_t=config['label_t'])
    
    sim_results = simulate_trading_strategy(
        predictions_df=predictions_df,
        stock_data_df=stock_data,
        top_k=config['top_k'],
        label_t=config['label_t']
    )
    
    dates = pd.to_datetime(sim_results['dates'])
    portfolio_values = np.cumprod(1 + sim_results['portfolio_returns'])
    benchmark_values = np.cumprod(1 + sim_results['benchmark_returns'])
    excess_values = portfolio_values / benchmark_values
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    # Top plot: Returns
    ax1 = axes[0]
    ax1.plot(dates, portfolio_values, 'r-', label='MCI-GRU Portfolio', linewidth=2)
    ax1.plot(dates, benchmark_values, 'b-', label='S&P 500 Benchmark', linewidth=1.5)
    ax1.plot(dates, excess_values, 'orange', label='Excess Return', linewidth=1.5)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('MCI-GRU S&P 500 Backtest Performance (Paper Methodology)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Drawdown
    ax2 = axes[1]
    running_max = np.maximum.accumulate(excess_values)
    drawdown = (running_max - excess_values) / running_max * 100
    ax2.fill_between(dates, 0, -drawdown, color='red', alpha=0.3)
    ax2.plot(dates, -drawdown, 'r-', linewidth=1)
    ax2.set_ylabel('Excess Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Equity curve saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# MULTI-MODEL EVALUATION (Paper Section 4.1.2: 10 runs averaged)
# ============================================================================

def evaluate_multiple_models(save_path, num_models=10, config=None):
    """
    Evaluate multiple model runs and average results (per paper methodology).
    
    Paper: "we conduct ten training and predictions for each method and 
    take the average result of the ten times as the final prediction result"
    
    Args:
        save_path: Base save path containing model folders
        num_models: Number of model runs to evaluate
        config: Configuration dictionary
    
    Returns:
        Dictionary with individual and averaged results
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    print("\n" + "=" * 70)
    print(f"MULTI-MODEL EVALUATION ({num_models} runs)")
    print("=" * 70)
    
    all_results = []
    
    for model_id in range(num_models):
        pred_dir = os.path.join(save_path, f'prediction_{model_id}', '0')
        
        if not os.path.exists(pred_dir):
            # Try alternate structure
            pred_dir = os.path.join(save_path, f'model_{model_id}', 'predictions')
            
        if not os.path.exists(pred_dir):
            print(f"  Model {model_id}: predictions not found, skipping")
            continue
        
        print(f"\n  Evaluating Model {model_id}...")
        try:
            results = evaluate(pred_dir, config)
            if results:
                all_results.append(results)
                print(f"    ARR={results['ARR']:.4f}, ASR={results['ASR']:.4f}, IR={results['IR']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
    
    if not all_results:
        print("No valid model results found!")
        return None
    
    # Average results
    avg_results = {}
    for key in all_results[0].keys():
        if isinstance(all_results[0][key], (int, float)):
            avg_results[key] = np.mean([r[key] for r in all_results])
        else:
            avg_results[key] = all_results[0][key]  # Keep non-numeric as-is
    
    print(f"\n  Averaged {len(all_results)} model runs")
    
    return {
        'individual': all_results,
        'averaged': avg_results
    }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MCI-GRU model predictions per paper methodology'
    )
    
    parser.add_argument(
        '--predictions_dir',
        type=str,
        required=True,
        help='Directory containing prediction CSV files'
    )
    
    parser.add_argument(
        '--data_file',
        type=str,
        default='sp500_yf_download.csv',
        help='Path to stock data CSV file'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of top stocks to select (default: 10)'
    )
    
    parser.add_argument(
        '--test_start',
        type=str,
        default='2023-01-01',
        help='Test period start date (default: 2023-01-01)'
    )
    
    parser.add_argument(
        '--test_end',
        type=str,
        default='2023-12-31',
        help='Test period end date (default: 2023-12-31)'
    )
    
    parser.add_argument(
        '--label_t',
        type=int,
        default=5,
        help='Forward return period in days (default: 5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for results CSV'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate equity curve plot'
    )
    
    parser.add_argument(
        '--multi_model',
        type=str,
        default=None,
        help='Base path for multi-model evaluation (10 runs)'
    )
    
    parser.add_argument(
        '--num_models',
        type=int,
        default=10,
        help='Number of model runs for multi-model evaluation (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Build config
    config = {
        'top_k': args.top_k,
        'trading_days_per_year': 252,
        'test_start': args.test_start,
        'test_end': args.test_end,
        'data_file': args.data_file,
        'label_t': args.label_t,
    }
    
    # Run evaluation
    if args.multi_model:
        # Multi-model evaluation (paper style: 10 runs averaged)
        multi_results = evaluate_multiple_models(
            args.multi_model, 
            num_models=args.num_models,
            config=config
        )
        
        if multi_results:
            print_results(multi_results['averaged'], "MCI-GRU (Averaged)")
            
            if args.output:
                save_results(multi_results['averaged'], args.output)
    else:
        # Single evaluation
        results = evaluate(args.predictions_dir, config)
        
        if results:
            print_results(results, "MCI-GRU")
            
            if args.output:
                save_results(results, args.output)
            
            if args.plot:
                stock_data = load_stock_data(
                    config['data_file'],
                    config['test_start'],
                    config['test_end']
                )
                plot_output = args.output.replace('.csv', '_equity.png') if args.output else None
                plot_equity_curve(args.predictions_dir, stock_data, config, plot_output)


if __name__ == '__main__':
    main()
