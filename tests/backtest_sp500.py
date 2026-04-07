"""
evaluate_sp500.py - Backtest and Evaluation Metrics for MCI-GRU Model

Implements backtesting methodology based on paper:
"MCI-GRU: Stock Prediction Model Based on Multi-Head Cross-Attention and Improved GRU"
(arXiv:2410.20679v3)

Trading Timeline:
1. Day T, 4:00 PM: Model generates predictions using data through close
2. Day T+1, 9:30 AM: Execute trades at market OPEN (ENTRY)
3. Day T+1, 4:00 PM: New prediction generated, continue holding overnight
4. Day T+2, 9:30 AM: Rebalance at market OPEN (EXIT old positions, ENTRY new)

Portfolio Returns:
- Holding period: open_{T+1} to open_{T+2}
- Captures: Day T+1 intraday + overnight gap to Day T+2
- Formula: open_{T+2} / open_{T+1} - 1

Benchmark Returns:
- Uses open-to-open returns (same window as portfolio for fair comparison)
- Formula: open_{T+2} / open_{T+1} - 1
- Equal-weighted across all stocks (apples-to-apples with portfolio)
- Note: This differs from published S&P 500 figures (cap-weighted, close-to-close)

Look-Ahead Bias Prevention:
- We do NOT claim returns from close_T to open_{T+1} (happens BEFORE we can trade)
- We DO capture returns from open_{T+1} to open_{T+2} (happens AFTER entry)

Paper Section 4.1.2 - Evaluation Metrics:
- ARR (Annualized Rate of Return)
- AVoL (Annualized Volatility)  
- MDD (Maximum Drawdown)
- ASR (Annualized Sharpe Ratio)
- CR (Calmar Ratio)
- IR (Information Ratio)
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)

Multiple Testing Adjustment (Harvey & Liu, 2014):
- When multiple strategies/hyperparameters are tested, Sharpe Ratios are overstated
- Implements three adjustment methods: BHY (recommended), Bonferroni, Holm
- Calculates t-statistics and adjusted p-values
- "Haircuts" Sharpe Ratios based on number of tests tried

Trading Strategy:
1. At close of day T, model generates prediction scores for each stock
2. Rank stocks by predicted score (expected return)
3. At open of day T+1, buy top-k stocks (paper uses k=10)
4. Equal-weighted portfolio, hold overnight
5. At open of day T+2, rebalance based on day T+1 predictions
6. Return = open_{T+2} / open_{T+1} - 1 (includes overnight)
7. Optional transaction costs (bid-ask spread + slippage)
8. Run 10 times and average results

Transaction Costs (Retail Investor Model):
- Bid-ask spread: Cost of crossing the spread when buying (at ask) or selling (at bid)
- Slippage: Execution price deviation from expected price for market orders
- Excludes market impact (negligible for retail-size trades)

Usage:
    # Basic evaluation (no transaction costs, matches paper)
    python evaluate_sp500.py --predictions_dir output/averaged_predictions
    
    # With transaction costs (retail investor model)
    python evaluate_sp500.py --predictions_dir output/averaged_predictions --transaction_costs
    
    # Custom transaction cost parameters (in basis points)
    python evaluate_sp500.py --predictions_dir output/averaged_predictions --transaction_costs --spread 15 --slippage 3
    
    # With multiple testing adjustment (if you tried 50 hyperparameter combinations):
    python evaluate_sp500.py --predictions_dir output/averaged_predictions --num_tests 50
    
    # Using different adjustment method (Bonferroni instead of BHY):
    python evaluate_sp500.py --predictions_dir output/averaged_predictions --num_tests 50 --adjustment_method bonferroni
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from mci_gru.tracking import MLflowTrackingManager, load_run_metadata_from_predictions_dir


# ============================================================================
# CONFIGURATION (Paper Section 4.1.1 and 4.1.2)
# ============================================================================

DEFAULT_CONFIG = {
    'top_k': 10,                    # Paper: k=10 stocks in portfolio
    'trading_days_per_year': 252,   # Standard trading days
    'test_start': '2025-01-01',     # Updated to match training config
    'test_end': '2025-12-31',       # Updated to match training config
    'data_file': 'data/raw/market/sp500_data.csv',  # Updated to reorganized path
    'label_t': 5,                   # Forward return period (days)
    # Transaction costs for retail investor
    'transaction_costs': {
        'enabled': False,           # Disabled by default (paper excludes costs)
        'bid_ask_spread': 0.0010,   # 10 bps round-trip (5 bps each side)
        'slippage': 0.0005,         # 5 bps per trade
    },
    # Rank-drop sell gate: only exit held stocks that worsened by >= min_rank_drop vs previous prediction day
    'rank_drop_gate': {
        'enabled': False,           # Disabled by default (backward compatibility)
        'min_rank_drop': 10,        # Minimum rank worsening (current_rank - prev_rank) to trigger exit
    },
    # When holding_period > 1: 'staggered' = 1/N of portfolio rebalances daily; 'block' = whole portfolio rebalances every N days (retail)
    'rebalance_style': 'staggered',
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
# MULTIPLE TESTING ADJUSTMENTS (Harvey & Liu, 2014)
# ============================================================================
# Reference: "Evaluating Trading Strategies" - Campbell R. Harvey & Yan Liu
# Key insight: When multiple strategies are tested, Sharpe Ratios are overstated.
# We need to adjust ("haircut") reported metrics based on the number of tests.

from scipy import stats


def calculate_t_statistic(sharpe_ratio, num_years):
    """
    Convert Sharpe Ratio to t-statistic.
    
    Paper formula: t-statistic = Sharpe Ratio × √(Number of years)
    
    This allows us to assess statistical significance of the strategy.
    A t-stat > 2.0 is traditionally considered significant for a single test,
    but with multiple tests, higher thresholds are required.
    
    Args:
        sharpe_ratio: Annualized Sharpe Ratio
        num_years: Number of years in the test period
    
    Returns:
        t-statistic value
    """
    if num_years <= 0:
        return 0.0
    return sharpe_ratio * np.sqrt(num_years)


def t_stat_to_p_value(t_stat):
    """
    Convert t-statistic to two-tailed p-value.
    
    Args:
        t_stat: t-statistic value
    
    Returns:
        Two-tailed p-value
    """
    # Use large degrees of freedom (approximates normal distribution)
    return 2 * (1 - stats.norm.cdf(abs(t_stat)))


def p_value_to_t_stat(p_value):
    """
    Convert p-value to t-statistic.
    
    Args:
        p_value: p-value (two-tailed)
    
    Returns:
        t-statistic value
    """
    if p_value >= 1.0:
        return 0.0
    if p_value <= 0.0:
        return np.inf
    return stats.norm.ppf(1 - p_value / 2)


def bhy_c_factor(num_tests):
    """
    Calculate the c(M) factor for BHY adjustment.
    
    c(M) = 1 + 1/2 + 1/3 + ... + 1/M ≈ log(M) for large M
    
    This is the harmonic series sum used in the BHY procedure.
    
    Args:
        num_tests: Number of tests (M)
    
    Returns:
        c(M) factor
    """
    if num_tests <= 0:
        return 1.0
    return sum(1.0 / k for k in range(1, num_tests + 1))


def adjust_p_value_bhy(p_value, rank, num_tests, alpha=0.05):
    """
    BHY (Benjamini-Hochberg-Yekutieli) p-value adjustment.
    
    The BHY method controls the False Discovery Rate (FDR) and is recommended
    by Harvey & Liu (2014) for evaluating trading strategies because:
    - It's less stringent than FWER methods (Bonferroni, Holm)
    - Allows for an expected proportion of false discoveries
    - More appropriate when a single false discovery isn't catastrophic
    
    Paper formula: threshold_k = k × α / (M × c(M))
    
    The procedure:
    1. Sort tests by p-value (lowest to highest)
    2. Starting from the LEAST significant (highest p-value), compare to threshold
    3. When p-value falls below its threshold, declare it and all lower p-values significant
    
    Args:
        p_value: Original p-value for this test
        rank: Rank of this test when sorted by p-value (1 = most significant)
        num_tests: Total number of tests (M)
        alpha: Significance level (default 0.05)
    
    Returns:
        Adjusted p-value (multiply original p-value by adjustment factor)
    """
    if num_tests <= 1:
        return p_value
    
    c_m = bhy_c_factor(num_tests)
    
    # BHY adjustment factor: (M × c(M)) / rank
    # Adjusted p-value = original_p × M × c(M) / rank
    adjustment_factor = (num_tests * c_m) / rank
    adjusted_p = min(p_value * adjustment_factor, 1.0)
    
    return adjusted_p


def adjust_p_value_bonferroni(p_value, num_tests):
    """
    Bonferroni p-value adjustment (most stringent FWER method).
    
    The Bonferroni method controls the Family-Wise Error Rate (FWER),
    meaning it's unacceptable to make even a single false discovery.
    
    This is the most conservative approach - appropriate for high-stakes
    decisions (e.g., space missions) but may be too stringent for trading.
    
    Paper formula: adjusted_p = original_p × M
    Equivalently: significance threshold = α / M
    
    Example: With 200 tests and α=0.05, you need p < 0.00025 to be significant.
    
    Args:
        p_value: Original p-value
        num_tests: Total number of tests (M)
    
    Returns:
        Adjusted p-value
    """
    if num_tests <= 1:
        return p_value
    
    adjusted_p = min(p_value * num_tests, 1.0)
    return adjusted_p


def adjust_p_value_holm(p_value, rank, num_tests):
    """
    Holm p-value adjustment (step-down FWER method).
    
    The Holm method is less stringent than Bonferroni but still controls
    the Family-Wise Error Rate. It uses information from the distribution
    of test statistics to provide a sharper test.
    
    Paper formula: threshold_k = α / (M + 1 - k)
    where k is the rank (1 = most significant)
    
    The procedure:
    1. Sort tests by p-value (lowest to highest)
    2. Starting from the MOST significant, compare to threshold
    3. When a test fails its threshold, reject it and all less significant tests
    
    Args:
        p_value: Original p-value
        rank: Rank of this test when sorted by p-value (1 = most significant)
        num_tests: Total number of tests (M)
    
    Returns:
        Adjusted p-value
    """
    if num_tests <= 1:
        return p_value
    
    # Holm adjustment factor: (M + 1 - rank)
    # Note: For rank=1 (most significant), this equals Bonferroni
    adjustment_factor = num_tests + 1 - rank
    adjusted_p = min(p_value * adjustment_factor, 1.0)
    
    return adjusted_p


def haircut_sharpe_ratio(sharpe_ratio, num_years, num_tests, method='bhy', 
                         rank=1, alpha=0.05):
    """
    Adjust ("haircut") Sharpe Ratio for multiple testing.
    
    This is the main function implementing Harvey & Liu (2014) methodology.
    When you've tried multiple strategies and are reporting the best one,
    the Sharpe Ratio should be adjusted downward to account for selection bias.
    
    The process:
    1. Convert Sharpe Ratio to t-statistic
    2. Convert t-statistic to p-value
    3. Adjust p-value using selected method (BHY/Bonferroni/Holm)
    4. Convert adjusted p-value back to t-statistic
    5. Convert adjusted t-statistic back to Sharpe Ratio
    
    Example from paper:
    - Original SR = 0.92, 10 years, 200 tests
    - t-stat = 0.92 × √10 = 2.91
    - p-value = 0.004
    - Bonferroni adjusted p-value = 0.004 × 200 = 0.80
    - Adjusted t-stat = 0.25
    - Haircutted SR = 0.25 / √10 = 0.08
    - Haircut = 91%!
    
    Args:
        sharpe_ratio: Original annualized Sharpe Ratio
        num_years: Number of years in test period
        num_tests: Number of strategies/configurations tested
        method: Adjustment method - 'bhy' (recommended), 'bonferroni', or 'holm'
        rank: Rank of this strategy (1 = best performer, used for BHY/Holm)
        alpha: Significance level (default 0.05)
    
    Returns:
        Dictionary with:
        - haircutted_sharpe: Adjusted Sharpe Ratio
        - original_sharpe: Original Sharpe Ratio (for reference)
        - t_statistic: Original t-statistic
        - original_p_value: Original p-value
        - adjusted_p_value: Multiple-testing adjusted p-value
        - is_significant: Whether result is significant after adjustment
        - haircut_pct: Percentage reduction in Sharpe Ratio
        - method: Method used for adjustment
    """
    # Step 1: Convert Sharpe to t-statistic
    t_stat = calculate_t_statistic(sharpe_ratio, num_years)
    
    # Step 2: Convert t-statistic to p-value
    original_p = t_stat_to_p_value(t_stat)
    
    # Step 3: Adjust p-value based on method
    if num_tests <= 1:
        adjusted_p = original_p
    elif method.lower() == 'bonferroni':
        adjusted_p = adjust_p_value_bonferroni(original_p, num_tests)
    elif method.lower() == 'holm':
        adjusted_p = adjust_p_value_holm(original_p, rank, num_tests)
    elif method.lower() == 'bhy':
        adjusted_p = adjust_p_value_bhy(original_p, rank, num_tests, alpha)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bhy', 'bonferroni', or 'holm'")
    
    # Step 4: Convert adjusted p-value back to t-statistic
    adjusted_t_stat = p_value_to_t_stat(adjusted_p)
    
    # Step 5: Convert adjusted t-statistic back to Sharpe Ratio
    if num_years > 0:
        haircutted_sharpe = adjusted_t_stat / np.sqrt(num_years)
    else:
        haircutted_sharpe = 0.0
    
    # Ensure haircutted Sharpe is non-negative and <= original
    haircutted_sharpe = max(0.0, min(haircutted_sharpe, abs(sharpe_ratio)))
    
    # Preserve sign of original Sharpe
    if sharpe_ratio < 0:
        haircutted_sharpe = -haircutted_sharpe
    
    # Calculate haircut percentage
    if sharpe_ratio != 0:
        haircut_pct = (abs(sharpe_ratio) - abs(haircutted_sharpe)) / abs(sharpe_ratio) * 100
    else:
        haircut_pct = 0.0
    
    # Determine significance
    is_significant = adjusted_p < alpha
    
    return {
        'haircutted_sharpe': haircutted_sharpe,
        'original_sharpe': sharpe_ratio,
        't_statistic': t_stat,
        'original_p_value': original_p,
        'adjusted_p_value': adjusted_p,
        'adjusted_t_statistic': adjusted_t_stat,
        'is_significant': is_significant,
        'haircut_pct': haircut_pct,
        'method': method,
        'num_tests': num_tests,
        'num_years': num_years
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def resolve_data_file(filename):
    """Resolve data file from reorganized and legacy locations."""
    if os.path.exists(filename):
        return filename

    basename = os.path.basename(filename)
    candidates = [
        os.path.join("data", "raw", "market", basename),
        basename,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find data file: {filename}")


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
    resolved_file = resolve_data_file(filename)
    print(f"Loading stock data from {resolved_file}...")
    df = pd.read_csv(resolved_file)
    
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
    Calculate forward returns for backtesting and model evaluation.
    
    Trading Timeline:
    - Day T, 4:00 PM: Model generates predictions using data through close
    - Day T+1, 9:30 AM: Execute trades at market open (ENTRY)
    - Day T+1, overnight: Hold positions
    - Day T+2, 9:30 AM: Rebalance at market open (EXIT)
    
    Portfolio return = open_{T+2} / open_{T+1} - 1 (captures intraday + overnight)
    Benchmark return = close_{T+1} / close_T - 1 (standard S&P 500 methodology)
    
    Args:
        df: DataFrame with stock data (must have 'open' and 'close' columns)
        label_t: Number of days for forward return (for MSE/MAE calculation)
    
    Returns:
        DataFrame with multiple return columns:
        - forward_return_{label_t}d: For model training/evaluation (close_t+label_t / close_t+1 - 1)
        - open_to_open_return: For BOTH portfolio AND benchmark backtesting (open_{t+1} / open_t - 1)
        - daily_return: For reference only (close_t / close_{t-1} - 1, standard close-to-close)
        - tradeable_return: Intraday return (close / open - 1)
        - overnight_gap: Gap from previous close to current open (for analysis)
    """
    print(f"Calculating {label_t}-day forward returns...")
    
    df = df.copy()
    df = df.sort_values(['kdcode', 'dt'])
    
    # Calculate forward close prices (for MSE/MAE metrics during training)
    df['close_t1'] = df.groupby('kdcode')['close'].shift(-1)
    df[f'close_t{label_t}'] = df.groupby('kdcode')['close'].shift(-label_t)
    
    # Forward return = close_{t+label_t} / close_{t+1} - 1
    # This matches the paper's label calculation for model training
    df[f'forward_return_{label_t}d'] = df[f'close_t{label_t}'] / df['close_t1'] - 1
    
    # DEPRECATED: Close-to-close return (WRONG for trading - includes look-ahead bias)
    # Kept for backward compatibility but should NOT be used for backtest
    df['next_day_return'] = df['close_t1'] / df['close'] - 1
    
    # CORRECT: Tradeable intraday return (what you can actually capture)
    # If prediction made at day t close, you trade at day t+1 open
    # Return = (close_t+1 - open_t+1) / open_t+1
    # This is represented as current day's intraday return
    df['tradeable_return'] = (df['close'] - df['open']) / df['open']
    df['tradeable_return'] = df['tradeable_return'].fillna(0)
    
    # Calculate overnight gap (for analysis)
    df['prev_close'] = df.groupby('kdcode')['close'].shift(1)
    df['overnight_gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['overnight_gap'] = df['overnight_gap'].fillna(0)
    
    # For verification: total return should equal overnight_gap + tradeable_return (approximately)
    df['total_return_check'] = (df['close'] - df['prev_close']) / df['prev_close']
    
    # Open-to-open return (for portfolio holding period)
    # This is the return from today's open to tomorrow's open
    # Captures: intraday (open to close) + overnight gap (close to next open)
    # Used for BOTH portfolio AND benchmark returns (apples-to-apples comparison)
    df['next_open'] = df.groupby('kdcode')['open'].shift(-1)
    df['open_to_open_return'] = df['next_open'] / df['open'] - 1
    # NaN values handled by dropna() in simulation - do NOT fillna(0)
    
    # Daily return (close-to-close) for benchmark calculation
    # This is the standard way S&P 500 returns are reported
    df['daily_return'] = df['close'] / df['prev_close'] - 1
    df['daily_return'] = df['daily_return'].fillna(0)
    
    # Clean up temporary columns
    df = df.drop(columns=['prev_close', 'next_open'], errors='ignore')
    
    print(f"  Added returns: forward_return_{label_t}d, open_to_open_return, daily_return, overnight_gap")
    print(f"  Portfolio & Benchmark use 'open_to_open_return' (same entry/exit window)")
    print(f"  'daily_return' kept for reference (standard S&P 500 methodology)")
    
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
# TRANSACTION COST CALCULATIONS (Retail Investor Model)
# ============================================================================
# Models realistic trading costs for retail investors:
# - Bid-ask spread: Cost of crossing the spread (buy at ask, sell at bid)
# - Slippage: Execution price deviation from expected price
# Excludes market impact (negligible for retail-size trades)

def calculate_turnover(prev_holdings, curr_holdings, target_k=None):
    """
    Calculate portfolio turnover between two consecutive days.
    
    Turnover measures how much of the portfolio changes each day.
    For an equal-weighted portfolio with k stocks:
    - If n_sold are sold and n_bought are bought:
    - One-way turnover = (n_sold + n_bought) / (2k)
    - Two-way turnover = (n_sold + n_bought) / k
    
    Args:
        prev_holdings: Set of stock codes held on previous day
        curr_holdings: Set of stock codes held on current day
        target_k: Optional fixed portfolio size used for turnover normalization.
                  If None, uses current holding count (backward-compatible).
    
    Returns:
        Dictionary with:
        - stocks_sold: Set of stocks exiting the portfolio
        - stocks_bought: Set of stocks entering the portfolio
        - stocks_held: Set of stocks remaining in portfolio
        - num_changes: Number of position changes
        - one_way_turnover: Fraction of portfolio traded (one side)
        - two_way_turnover: Total fraction traded (both sides)
    """
    if prev_holdings is None:
        prev_holdings = set()
    
    prev_set = set(prev_holdings)
    curr_set = set(curr_holdings)
    
    stocks_sold = prev_set - curr_set   # Stocks we exit
    stocks_bought = curr_set - prev_set  # Stocks we enter
    stocks_held = prev_set & curr_set    # Stocks we keep
    
    num_sold = len(stocks_sold)
    num_bought = len(stocks_bought)
    num_changes = max(num_sold, num_bought)
    traded_names = num_sold + num_bought

    k = int(target_k) if target_k is not None else len(curr_set)
    k = max(k, 1)

    # One-way turnover: average of sell/buy notional as a fraction of portfolio
    one_way_turnover = traded_names / (2 * k)
    # Two-way turnover: total traded notional fraction (sell + buy)
    two_way_turnover = traded_names / k
    
    return {
        'stocks_sold': stocks_sold,
        'stocks_bought': stocks_bought,
        'stocks_held': stocks_held,
        'num_sold': num_sold,
        'num_bought': num_bought,
        'num_trades': traded_names,
        'num_changes': num_changes,
        'one_way_turnover': one_way_turnover,
        'two_way_turnover': two_way_turnover,
    }


def calculate_transaction_cost(turnover_info, bid_ask_spread, slippage):
    """
    Calculate transaction costs based on turnover and cost parameters.
    
    For retail investors trading S&P 500 stocks:
    - Bid-ask spread: ~5-15 bps for large caps, split between buy and sell
    - Slippage: ~2-5 bps per trade for market orders
    
    Cost calculation:
    - When selling: incur half-spread + slippage on sold portion
    - When buying: incur half-spread + slippage on bought portion
    - Total cost = turnover_fraction * (spread + 2*slippage)
    
    The spread is the round-trip cost, so each side (buy or sell) pays half.
    Slippage applies independently to both buys and sells.
    
    Args:
        turnover_info: Dictionary from calculate_turnover()
        bid_ask_spread: Round-trip bid-ask spread as decimal (e.g., 0.001 = 10 bps)
        slippage: Per-trade slippage as decimal (e.g., 0.0005 = 5 bps)
    
    Returns:
        Dictionary with:
        - sell_cost: Cost from selling positions (as fraction of portfolio)
        - buy_cost: Cost from buying positions (as fraction of portfolio)
        - total_cost: Total transaction cost (as fraction of portfolio)
        - cost_breakdown: Detailed breakdown of costs
    """
    one_way_turnover = turnover_info['one_way_turnover']
    
    # Cost per side = half-spread + slippage
    cost_per_side = (bid_ask_spread / 2) + slippage
    
    # Sell cost: applied to the sold fraction of portfolio
    sell_cost = one_way_turnover * cost_per_side
    
    # Buy cost: applied to the bought fraction of portfolio
    buy_cost = one_way_turnover * cost_per_side
    
    # Total cost
    total_cost = sell_cost + buy_cost
    
    # Alternative formula: total_cost = one_way_turnover * (spread + 2*slippage)
    # This is equivalent since: 2 * (spread/2 + slippage) = spread + 2*slippage
    
    return {
        'sell_cost': sell_cost,
        'buy_cost': buy_cost,
        'total_cost': total_cost,
        'cost_breakdown': {
            'spread_component': one_way_turnover * bid_ask_spread,
            'slippage_component': 2 * one_way_turnover * slippage,
            'one_way_turnover': one_way_turnover,
            'num_trades': turnover_info['num_changes'] * 2,  # sells + buys
        }
    }


# ============================================================================
# TRADING STRATEGY SIMULATION (Paper Section 4.1.2)
# ============================================================================

def simulate_trading_strategy(predictions_df, stock_data_df, top_k=10, label_t=5,
                              transaction_costs=None, rank_drop_gate=None):
    """
    Simulate trading strategy with realistic timing and overnight holding.
    
    Trading Timeline:
    1. Day T, 4:00 PM: Model generates predictions using data through close
    2. Day T+1, 9:30 AM: Execute trades at market OPEN (ENTRY)
    3. Day T+1, 4:00 PM: New prediction generated, continue holding overnight
    4. Day T+2, 9:30 AM: Rebalance at market OPEN (EXIT old, ENTRY new)
    
    Returns Captured (Portfolio):
    - Holding period: open_{T+1} to open_{T+2}
    - Includes: Day T+1 intraday + overnight gap to Day T+2
    - Return formula: open_{T+2} / open_{T+1} - 1
    
    Benchmark:
    - Uses close-to-close daily returns (standard S&P 500 methodology)
    - Return formula: close_{T+1} / close_T - 1
    
    Paper methodology:
    1. At close of day T, rank stocks by predicted score
    2. At open of day T+1, buy top-k stocks (equal-weighted)
    3. Hold overnight until open of day T+2
    4. At open of day T+2, rebalance based on day T+1 predictions
    5. Optional transaction costs (bid-ask spread + slippage)
    
    Transaction Cost Model (for retail investors):
    - Bid-ask spread: Cost of crossing the spread when buying/selling
    - Slippage: Execution price deviation from expected price
    - Applied based on daily turnover (position changes)
    
    Rank-drop gate (optional): When enabled, this acts as a sell gate on held names.
    A held stock is exited only if its rank worsens by at least min_rank_drop vs
    the previous prediction day; otherwise it is kept. Vacated slots are refilled
    from today's highest-ranked names until top_k holdings are restored.
    
    Args:
        predictions_df: DataFrame with predictions (kdcode, dt, score)
        stock_data_df: DataFrame with stock data and returns (must have 'open_to_open_return')
        top_k: Number of top stocks to select
        label_t: Forward return period for accuracy metrics (MSE/MAE)
        transaction_costs: Dict with 'enabled', 'bid_ask_spread', 'slippage'
                          If None or not enabled, no costs applied
        rank_drop_gate: Dict with 'enabled' and 'min_rank_drop'. If None or not enabled, no gate.
    
    Returns:
        Dictionary with daily returns (gross and net), turnover stats, and related data
    """
    # Parse transaction cost settings
    tc_enabled = False
    bid_ask_spread = 0.0
    slippage = 0.0
    
    if transaction_costs and transaction_costs.get('enabled', False):
        tc_enabled = True
        bid_ask_spread = transaction_costs.get('bid_ask_spread', 0.001)
        slippage = transaction_costs.get('slippage', 0.0005)
    
    # Parse rank-drop gate settings
    gate_enabled = False
    min_rank_drop = 10
    if rank_drop_gate and rank_drop_gate.get('enabled', False):
        gate_enabled = True
        min_rank_drop = rank_drop_gate.get('min_rank_drop', 10)
    
    cost_status = "ENABLED" if tc_enabled else "DISABLED"
    print(f"\nSimulating trading strategy (top-{top_k} stocks)...")
    print(f"  Portfolio: open-to-open returns (entry at T+1 open, exit at T+2 open)")
    print(f"  Benchmark: open-to-open returns (equal-weighted, same window as portfolio)")
    print(f"  Transaction costs: {cost_status}")
    if tc_enabled:
        print(f"    Bid-ask spread: {bid_ask_spread*10000:.1f} bps (round-trip)")
        print(f"    Slippage: {slippage*10000:.1f} bps (per trade)")
    gate_status = "ENABLED" if gate_enabled else "DISABLED"
    print(f"  Rank-drop gate: {gate_status}")
    if gate_enabled:
        print(f"    Min rank drop: {min_rank_drop} (held stock exits only if rank worsens by >= {min_rank_drop})")
    
    # Get unique prediction dates and create date mapping
    pred_dates = sorted(predictions_df['dt'].unique())
    all_stock_dates = sorted(stock_data_df['dt'].unique())
    
    print(f"  {len(pred_dates)} prediction dates")
    
    # Verify required columns
    if 'open_to_open_return' not in stock_data_df.columns:
        raise ValueError(
            "stock_data_df must have 'open_to_open_return' column. "
            "Run calculate_forward_returns() first!"
        )
    if 'daily_return' not in stock_data_df.columns:
        raise ValueError(
            "stock_data_df must have 'daily_return' column. "
            "Run calculate_forward_returns() first!"
        )
    
    # Tracking arrays
    gross_portfolio_returns = []  # Returns before transaction costs
    net_portfolio_returns = []    # Returns after transaction costs
    benchmark_returns = []
    dates = []
    all_predictions = []
    all_actuals = []
    
    # Transaction cost tracking
    daily_turnover = []
    daily_costs = []
    daily_num_trades = []
    # Portfolio tracking outputs (additive, does not affect return calculations)
    daily_holdings_records = []
    trade_records = []
    
    # Rank-drop gate: track diagnostics and previous-date ranks (kdcode -> rank, 1-based)
    days_skipped_by_rank_gate = 0
    days_with_gate_exits = 0
    prev_date_ranks = None  # dict kdcode -> rank from previous prediction date
    
    # Track previous day's holdings for turnover calculation
    prev_holdings = None
    
    # CRITICAL FIX: Process predictions with next-day returns
    # Loop through predictions, but measure returns on the NEXT trading day
    for i, pred_date in enumerate(pred_dates):
        # Day t: Get predictions made at close
        day_preds = predictions_df[predictions_df['dt'] == pred_date].copy()
        
        if len(day_preds) < top_k:
            prev_date_ranks = None
            continue
        
        # Rank by score descending (best = rank 1)
        day_preds = day_preds.sort_values('score', ascending=False).reset_index(drop=True)
        day_preds['_rank'] = np.arange(1, len(day_preds) + 1, dtype=int)
        current_ranks = day_preds.set_index('kdcode')['_rank'].to_dict()
        
        # Build target holdings.
        # Gate disabled -> preserve legacy behavior (daily top-k refresh).
        if not gate_enabled:
            top_stocks = day_preds.head(top_k)['kdcode'].tolist()
        # Gate enabled -> only sell currently held names that worsen by >= min_rank_drop.
        elif prev_holdings is None or prev_date_ranks is None:
            # Initialize holdings on first valid gated day.
            top_stocks = day_preds.head(top_k)['kdcode'].tolist()
        else:
            current_held = [kd for kd in prev_holdings if kd in current_ranks]
            survivors = []
            gate_exits_today = 0

            for kdcode in current_held:
                prev_rank = prev_date_ranks.get(kdcode)
                curr_rank = current_ranks.get(kdcode)

                # If rank history is missing, keep the position to avoid forced churn.
                if prev_rank is None or curr_rank is None:
                    survivors.append(kdcode)
                    continue

                rank_drop = curr_rank - prev_rank  # positive = rank worsened
                if rank_drop >= min_rank_drop:
                    gate_exits_today += 1
                    continue

                survivors.append(kdcode)

            if gate_exits_today > 0:
                days_with_gate_exits += 1

            survivor_set = set(survivors)
            refill_candidates = [
                kd for kd in day_preds['kdcode'].tolist()
                if kd not in survivor_set
            ]
            slots_needed = max(0, top_k - len(survivors))
            top_stocks = survivors + refill_candidates[:slots_needed]

        if len(top_stocks) == 0:
            days_skipped_by_rank_gate += 1
            prev_date_ranks = current_ranks
            continue
        curr_holdings = set(top_stocks)
        
        # Find the NEXT trading day (day T+1) - this is when we enter positions
        # We trade at day T+1 open and hold until day T+2 open (when we rebalance)
        try:
            entry_date_idx = all_stock_dates.index(pred_date) + 1
            if entry_date_idx >= len(all_stock_dates):
                # No next day available (end of data)
                prev_date_ranks = current_ranks
                continue
            entry_date = all_stock_dates[entry_date_idx]
        except (ValueError, IndexError):
            # pred_date not in stock data or no next day
            prev_date_ranks = current_ranks
            continue
        
        # Day T+1: Get data for entry day
        # The open_to_open_return on this day gives us: open_{T+2} / open_{T+1} - 1
        entry_day_data = stock_data_df[stock_data_df['dt'] == entry_date]
        
        if len(entry_day_data) == 0:
            prev_date_ranks = current_ranks
            continue
        
        # Portfolio return: open-to-open (entry at T+1 open, exit at T+2 open)
        # This captures intraday T+1 + overnight gap to T+2
        top_k_data = entry_day_data[entry_day_data['kdcode'].isin(top_stocks)]
        top_k_returns = top_k_data['open_to_open_return'].dropna()
        
        if len(top_k_returns) == 0:
            prev_date_ranks = current_ranks
            continue
        
        # GROSS return (before transaction costs)
        gross_return = top_k_returns.mean()
        
        # Calculate turnover and transaction costs
        turnover_info = calculate_turnover(prev_holdings, curr_holdings, target_k=top_k)
        
        # Capture per-stock holdings details for this realized entry day.
        # Uses only same-day realized returns already used in portfolio return.
        day_score_map = day_preds.set_index('kdcode')['score'].to_dict()
        realized_holdings = top_k_data[['kdcode', 'open_to_open_return']].dropna(subset=['open_to_open_return'])
        if len(realized_holdings) > 0:
            realized_weight = 1.0 / len(realized_holdings)
            for _, held_row in realized_holdings.iterrows():
                held_kdcode = held_row['kdcode']
                held_return = float(held_row['open_to_open_return'])
                daily_holdings_records.append({
                    'pred_date': pred_date,
                    'entry_date': entry_date,
                    'kdcode': held_kdcode,
                    'rank': int(current_ranks.get(held_kdcode, np.nan)) if held_kdcode in current_ranks else np.nan,
                    'score': float(day_score_map.get(held_kdcode, np.nan)),
                    'weight': realized_weight,
                    'stock_return': held_return,
                    'contribution': held_return * realized_weight,
                    'rank_gate_enabled': gate_enabled,
                    'min_rank_drop': min_rank_drop if gate_enabled else np.nan,
                    'transaction_costs_enabled': tc_enabled,
                })
        
        # Capture buy/sell events from turnover (same-day decision state only).
        for bought_kdcode in sorted(turnover_info['stocks_bought']):
            trade_records.append({
                'date': entry_date,
                'pred_date': pred_date,
                'kdcode': bought_kdcode,
                'action': 'BUY',
                'rank': int(current_ranks.get(bought_kdcode, np.nan)) if bought_kdcode in current_ranks else np.nan,
                'score': float(day_score_map.get(bought_kdcode, np.nan)),
            })
        for sold_kdcode in sorted(turnover_info['stocks_sold']):
            trade_records.append({
                'date': entry_date,
                'pred_date': pred_date,
                'kdcode': sold_kdcode,
                'action': 'SELL',
                'rank': int(current_ranks.get(sold_kdcode, np.nan)) if sold_kdcode in current_ranks else np.nan,
                'score': float(day_score_map.get(sold_kdcode, np.nan)),
            })
        
        if tc_enabled:
            cost_info = calculate_transaction_cost(
                turnover_info, bid_ask_spread, slippage
            )
            transaction_cost = cost_info['total_cost']
        else:
            transaction_cost = 0.0
            cost_info = {'total_cost': 0.0, 'cost_breakdown': {
                'spread_component': 0.0, 'slippage_component': 0.0,
                'one_way_turnover': turnover_info['one_way_turnover'],
                'num_trades': 0
            }}
        
        # NET return = GROSS return - transaction costs
        net_return = gross_return - transaction_cost
        
        # Benchmark return: open-to-open (same window as portfolio for apples-to-apples comparison)
        # Equal-weighted across all stocks, measured over same period as portfolio
        all_returns = entry_day_data['open_to_open_return'].dropna()
        benchmark_return = all_returns.mean() if len(all_returns) > 0 else 0.0
        
        # Store results
        gross_portfolio_returns.append(gross_return)
        net_portfolio_returns.append(net_return)
        benchmark_returns.append(benchmark_return)
        dates.append(entry_date)  # Store the entry date (T+1)
        
        # Store turnover and cost metrics
        daily_turnover.append(turnover_info['one_way_turnover'])
        daily_costs.append(transaction_cost)
        daily_num_trades.append(turnover_info['num_trades'])
        
        # Update holdings for next iteration
        prev_holdings = curr_holdings
        prev_date_ranks = current_ranks
        
        # Collect predictions vs actuals for MSE/MAE (use prediction date's forward returns)
        pred_date_data = stock_data_df[stock_data_df['dt'] == pred_date]
        for _, row in day_preds.iterrows():
            kdcode = row['kdcode']
            score = row['score']
            
            actual = pred_date_data[
                pred_date_data['kdcode'] == kdcode
            ][f'forward_return_{label_t}d']
            
            if len(actual) > 0 and not pd.isna(actual.values[0]):
                all_predictions.append(score)
                all_actuals.append(actual.values[0])
    
    print(f"  Completed simulation: {len(dates)} valid trading days")
    if gate_enabled:
        print(f"  Rank-drop gate: {days_with_gate_exits} days with exits triggered")
        print(f"  Rank-drop gate: {days_skipped_by_rank_gate} days skipped (empty post-gate portfolio)")
    
    # Calculate transaction cost summary statistics
    total_tc = sum(daily_costs)
    avg_daily_turnover = np.mean(daily_turnover) if daily_turnover else 0.0
    avg_daily_cost = np.mean(daily_costs) if daily_costs else 0.0
    total_trades = sum(daily_num_trades)
    
    if tc_enabled:
        print(f"  Transaction cost summary:")
        print(f"    Total costs: {total_tc*100:.4f}% of portfolio")
        print(f"    Avg daily turnover: {avg_daily_turnover*100:.2f}%")
        print(f"    Avg daily cost: {avg_daily_cost*10000:.2f} bps")
        print(f"    Total trades: {total_trades}")
    
    return {
        # Primary returns (use net if costs enabled, gross otherwise for backwards compatibility)
        'portfolio_returns': np.array(net_portfolio_returns) if tc_enabled else np.array(gross_portfolio_returns),
        'gross_portfolio_returns': np.array(gross_portfolio_returns),
        'net_portfolio_returns': np.array(net_portfolio_returns),
        'benchmark_returns': np.array(benchmark_returns),
        'dates': dates,
        'predictions': np.array(all_predictions),
        'actuals': np.array(all_actuals),
        # Transaction cost details
        'transaction_costs_enabled': tc_enabled,
        'daily_turnover': np.array(daily_turnover),
        'daily_costs': np.array(daily_costs),
        'daily_num_trades': np.array(daily_num_trades),
        'total_transaction_cost': total_tc,
        'avg_daily_turnover': avg_daily_turnover,
        'avg_daily_cost': avg_daily_cost,
        'total_trades': total_trades,
        'tc_settings': {
            'bid_ask_spread': bid_ask_spread,
            'slippage': slippage,
        },
        # Rank-drop gate diagnostics
        'rank_gate_enabled': gate_enabled,
        'min_rank_drop': min_rank_drop if gate_enabled else None,
        'days_with_gate_exits': days_with_gate_exits if gate_enabled else 0,
        'days_skipped_by_rank_gate': days_skipped_by_rank_gate,
        # Additive portfolio tracking outputs
        'daily_holdings': daily_holdings_records,
        'trade_records': trade_records,
    }


# ============================================================================
# STAGGERED MULTI-DAY HOLDING PERIOD SIMULATION
# ============================================================================

def simulate_trading_strategy_staggered(predictions_df, stock_data_df, top_k=10,
                                        label_t=5, holding_period=21,
                                        transaction_costs=None, rank_drop_gate=None):
    """
    Simulate trading strategy with staggered multi-day tranche rebalancing.

    Splits capital equally into `holding_period` tranches. Each tranche
    rebalances on its assigned day (tranche 0 on day_index % holding_period == 0,
    tranche 1 on day_index % holding_period == 1, etc.). On non-rebalance days
    a tranche simply holds its existing positions. The portfolio daily return is
    the equal-weighted average across all active tranches.

    This reduces turnover to approximately 1/holding_period of the daily-rebalance
    strategy and aligns the trading horizon with the model's prediction horizon.

    When holding_period=1 this function is NOT called -- the caller dispatches
    to the original simulate_trading_strategy() instead.

    Args:
        predictions_df: DataFrame with predictions (kdcode, dt, score)
        stock_data_df: DataFrame with stock data (must have 'open_to_open_return')
        top_k: Number of top stocks each tranche selects
        label_t: Forward return period for accuracy metrics (MSE/MAE)
        holding_period: Number of days each tranche holds before rebalancing
        transaction_costs: Dict with 'enabled', 'bid_ask_spread', 'slippage'
        rank_drop_gate: Dict with 'enabled' and 'min_rank_drop'

    Returns:
        Dictionary with same schema as simulate_trading_strategy()
    """
    # Parse transaction cost settings
    tc_enabled = False
    bid_ask_spread = 0.0
    slippage = 0.0
    if transaction_costs and transaction_costs.get('enabled', False):
        tc_enabled = True
        bid_ask_spread = transaction_costs.get('bid_ask_spread', 0.001)
        slippage = transaction_costs.get('slippage', 0.0005)

    # Parse rank-drop gate settings
    gate_enabled = False
    min_rank_drop = 10
    if rank_drop_gate and rank_drop_gate.get('enabled', False):
        gate_enabled = True
        min_rank_drop = rank_drop_gate.get('min_rank_drop', 10)

    cost_status = "ENABLED" if tc_enabled else "DISABLED"
    gate_status = "ENABLED" if gate_enabled else "DISABLED"
    print(f"\nSimulating staggered trading strategy (top-{top_k} stocks)...")
    print(f"  Staggered rebalancing: {holding_period} tranches, each held {holding_period} days")
    print(f"  Effective rebalance rate: 1/{holding_period} of portfolio per day")
    print(f"  Warm-up: first {holding_period - 1} days have partial tranche coverage")
    print(f"  Portfolio: open-to-open returns (entry at T+1 open, exit at T+2 open)")
    print(f"  Benchmark: open-to-open returns (equal-weighted, same window as portfolio)")
    print(f"  Transaction costs: {cost_status}")
    if tc_enabled:
        print(f"    Bid-ask spread: {bid_ask_spread*10000:.1f} bps (round-trip)")
        print(f"    Slippage: {slippage*10000:.1f} bps (per trade)")
    print(f"  Rank-drop gate: {gate_status}")
    if gate_enabled:
        print(f"    Min rank drop: {min_rank_drop} (per-tranche rank history)")

    # Get unique prediction dates and stock trading dates
    pred_dates = sorted(predictions_df['dt'].unique())
    all_stock_dates = sorted(stock_data_df['dt'].unique())
    print(f"  {len(pred_dates)} prediction dates")

    # Verify required columns
    if 'open_to_open_return' not in stock_data_df.columns:
        raise ValueError(
            "stock_data_df must have 'open_to_open_return' column. "
            "Run calculate_forward_returns() first!"
        )
    if 'daily_return' not in stock_data_df.columns:
        raise ValueError(
            "stock_data_df must have 'daily_return' column. "
            "Run calculate_forward_returns() first!"
        )

    # Per-tranche state: each tranche tracks its own holdings and rank history
    tranches = {
        tid: {
            'holdings': set(),
            'prev_holdings': None,
            'prev_ranks': None,
        }
        for tid in range(holding_period)
    }

    # Tracking arrays (same schema as original function)
    gross_portfolio_returns = []
    net_portfolio_returns = []
    benchmark_returns = []
    dates = []
    all_predictions = []
    all_actuals = []

    # Transaction cost / turnover tracking
    daily_turnover = []
    daily_costs = []
    daily_num_trades = []

    # Portfolio tracking outputs
    daily_holdings_records = []
    trade_records = []

    # Rank-drop gate diagnostics
    days_skipped_no_return = 0
    days_with_gate_exits = 0

    for i, pred_date in enumerate(pred_dates):
        tranche_id = i % holding_period

        # Get predictions for this date and rank them
        day_preds = predictions_df[predictions_df['dt'] == pred_date].copy()
        if len(day_preds) < top_k:
            continue

        day_preds = day_preds.sort_values('score', ascending=False).reset_index(drop=True)
        day_preds['_rank'] = np.arange(1, len(day_preds) + 1, dtype=int)
        current_ranks = day_preds.set_index('kdcode')['_rank'].to_dict()

        # ---- Rebalance the active tranche ----
        tranche = tranches[tranche_id]

        if not gate_enabled:
            top_stocks = day_preds.head(top_k)['kdcode'].tolist()
        elif not tranche['holdings'] or tranche['prev_ranks'] is None:
            top_stocks = day_preds.head(top_k)['kdcode'].tolist()
        else:
            current_held = [kd for kd in tranche['holdings'] if kd in current_ranks]
            survivors = []
            gate_exits_today = 0

            for kdcode in current_held:
                prev_rank = tranche['prev_ranks'].get(kdcode)
                curr_rank = current_ranks.get(kdcode)

                if prev_rank is None or curr_rank is None:
                    survivors.append(kdcode)
                    continue

                rank_drop = curr_rank - prev_rank
                if rank_drop >= min_rank_drop:
                    gate_exits_today += 1
                    continue

                survivors.append(kdcode)

            if gate_exits_today > 0:
                days_with_gate_exits += 1

            survivor_set = set(survivors)
            refill_candidates = [
                kd for kd in day_preds['kdcode'].tolist()
                if kd not in survivor_set
            ]
            slots_needed = max(0, top_k - len(survivors))
            top_stocks = survivors + refill_candidates[:slots_needed]

        new_holdings = set(top_stocks)

        # Compute tranche-level turnover: compare what the tranche currently
        # holds against what it will hold after rebalancing
        tranche_turnover_info = calculate_turnover(
            tranche['holdings'], new_holdings, target_k=top_k
        )

        # Update tranche state
        tranche['prev_holdings'] = tranche['holdings']
        tranche['holdings'] = new_holdings
        tranche['prev_ranks'] = current_ranks

        # ---- Find entry date (T+1) ----
        try:
            entry_date_idx = all_stock_dates.index(pred_date) + 1
            if entry_date_idx >= len(all_stock_dates):
                continue
            entry_date = all_stock_dates[entry_date_idx]
        except (ValueError, IndexError):
            continue

        entry_day_data = stock_data_df[stock_data_df['dt'] == entry_date]
        if len(entry_day_data) == 0:
            continue

        # ---- Compute daily return across ALL active tranches ----
        active_tranche_returns = []
        num_active_tranches = 0
        for tid, t in tranches.items():
            if t['holdings']:
                t_returns = entry_day_data[
                    entry_day_data['kdcode'].isin(t['holdings'])
                ]['open_to_open_return'].dropna()
                if len(t_returns) > 0:
                    active_tranche_returns.append(t_returns.mean())
                    num_active_tranches += 1

        if len(active_tranche_returns) == 0:
            days_skipped_no_return += 1
            continue

        gross_return = np.mean(active_tranche_returns)

        # ---- Transaction costs (rebalancing tranche only, scaled to portfolio) ----
        portfolio_turnover = tranche_turnover_info['one_way_turnover'] / holding_period

        if tc_enabled:
            tranche_cost_info = calculate_transaction_cost(
                tranche_turnover_info, bid_ask_spread, slippage
            )
            portfolio_cost = tranche_cost_info['total_cost'] / holding_period
        else:
            portfolio_cost = 0.0

        net_return = gross_return - portfolio_cost

        # ---- Benchmark return (same as original: daily equal-weighted all stocks) ----
        all_returns = entry_day_data['open_to_open_return'].dropna()
        benchmark_return = all_returns.mean() if len(all_returns) > 0 else 0.0

        # ---- Store results ----
        gross_portfolio_returns.append(gross_return)
        net_portfolio_returns.append(net_return)
        benchmark_returns.append(benchmark_return)
        dates.append(entry_date)

        daily_turnover.append(portfolio_turnover)
        daily_costs.append(portfolio_cost)
        daily_num_trades.append(tranche_turnover_info['num_trades'])

        # ---- Per-stock holdings detail (all tranches) ----
        day_score_map = day_preds.set_index('kdcode')['score'].to_dict()
        for tid, t in tranches.items():
            if not t['holdings']:
                continue
            t_data = entry_day_data[entry_day_data['kdcode'].isin(t['holdings'])]
            realized = t_data[['kdcode', 'open_to_open_return']].dropna(subset=['open_to_open_return'])
            if len(realized) == 0:
                continue
            tranche_weight = 1.0 / num_active_tranches
            stock_weight_in_tranche = 1.0 / len(realized)
            portfolio_weight = tranche_weight * stock_weight_in_tranche
            for _, held_row in realized.iterrows():
                held_kdcode = held_row['kdcode']
                held_return = float(held_row['open_to_open_return'])
                daily_holdings_records.append({
                    'pred_date': pred_date,
                    'entry_date': entry_date,
                    'kdcode': held_kdcode,
                    'tranche_id': tid,
                    'rank': int(current_ranks.get(held_kdcode, 0)) if held_kdcode in current_ranks else np.nan,
                    'score': float(day_score_map.get(held_kdcode, np.nan)),
                    'weight': portfolio_weight,
                    'stock_return': held_return,
                    'contribution': held_return * portfolio_weight,
                    'rank_gate_enabled': gate_enabled,
                    'min_rank_drop': min_rank_drop if gate_enabled else np.nan,
                    'transaction_costs_enabled': tc_enabled,
                })

        # ---- Trade records (rebalancing tranche only) ----
        for bought_kdcode in sorted(tranche_turnover_info['stocks_bought']):
            trade_records.append({
                'date': entry_date,
                'pred_date': pred_date,
                'kdcode': bought_kdcode,
                'action': 'BUY',
                'tranche_id': tranche_id,
                'rank': int(current_ranks.get(bought_kdcode, 0)) if bought_kdcode in current_ranks else np.nan,
                'score': float(day_score_map.get(bought_kdcode, np.nan)),
            })
        for sold_kdcode in sorted(tranche_turnover_info['stocks_sold']):
            trade_records.append({
                'date': entry_date,
                'pred_date': pred_date,
                'kdcode': sold_kdcode,
                'action': 'SELL',
                'tranche_id': tranche_id,
                'rank': int(current_ranks.get(sold_kdcode, 0)) if sold_kdcode in current_ranks else np.nan,
                'score': float(day_score_map.get(sold_kdcode, np.nan)),
            })

        # ---- Collect predictions vs actuals for MSE/MAE ----
        pred_date_data = stock_data_df[stock_data_df['dt'] == pred_date]
        for _, row in day_preds.iterrows():
            kdcode = row['kdcode']
            score = row['score']
            actual = pred_date_data[
                pred_date_data['kdcode'] == kdcode
            ][f'forward_return_{label_t}d']
            if len(actual) > 0 and not pd.isna(actual.values[0]):
                all_predictions.append(score)
                all_actuals.append(actual.values[0])

    print(f"  Completed staggered simulation: {len(dates)} valid trading days")
    if gate_enabled:
        print(f"  Rank-drop gate: {days_with_gate_exits} days with tranche-level exits triggered")

    # Transaction cost summary
    total_tc = sum(daily_costs)
    avg_daily_turnover = np.mean(daily_turnover) if daily_turnover else 0.0
    avg_daily_cost = np.mean(daily_costs) if daily_costs else 0.0
    total_trades = sum(daily_num_trades)

    if tc_enabled:
        print(f"  Transaction cost summary:")
        print(f"    Total costs: {total_tc*100:.4f}% of portfolio")
        print(f"    Avg daily turnover: {avg_daily_turnover*100:.2f}% (portfolio-level)")
        print(f"    Avg daily cost: {avg_daily_cost*10000:.2f} bps")
        print(f"    Total trades: {total_trades}")

    return {
        'portfolio_returns': np.array(net_portfolio_returns) if tc_enabled else np.array(gross_portfolio_returns),
        'gross_portfolio_returns': np.array(gross_portfolio_returns),
        'net_portfolio_returns': np.array(net_portfolio_returns),
        'benchmark_returns': np.array(benchmark_returns),
        'dates': dates,
        'predictions': np.array(all_predictions),
        'actuals': np.array(all_actuals),
        'transaction_costs_enabled': tc_enabled,
        'daily_turnover': np.array(daily_turnover),
        'daily_costs': np.array(daily_costs),
        'daily_num_trades': np.array(daily_num_trades),
        'total_transaction_cost': total_tc,
        'avg_daily_turnover': avg_daily_turnover,
        'avg_daily_cost': avg_daily_cost,
        'total_trades': total_trades,
        'tc_settings': {
            'bid_ask_spread': bid_ask_spread,
            'slippage': slippage,
        },
        'rank_gate_enabled': gate_enabled,
        'min_rank_drop': min_rank_drop if gate_enabled else None,
        'days_with_gate_exits': days_with_gate_exits if gate_enabled else 0,
        'days_skipped_by_rank_gate': 0,
        'daily_holdings': daily_holdings_records,
        'trade_records': trade_records,
    }


# ============================================================================
# BLOCK REBALANCE (Retail-style: one portfolio, rebalance every N days)
# ============================================================================

def simulate_trading_strategy_block(predictions_df, stock_data_df, top_k=10,
                                    label_t=5, holding_period=21,
                                    transaction_costs=None, rank_drop_gate=None):
    """
    Simulate trading strategy with block rebalancing (retail-realistic).

    Single portfolio of top_k stocks. Rebalance the entire portfolio only on
    days where day_index % holding_period == 0 (e.g. every 21 days). On other
    days, hold current positions with no trading. At any time only top_k
    positions are held; total rebalances per year ~252/holding_period.

    Args:
        predictions_df: DataFrame with predictions (kdcode, dt, score)
        stock_data_df: DataFrame with stock data (must have 'open_to_open_return')
        top_k: Number of top stocks to hold
        label_t: Forward return period for accuracy metrics (MSE/MAE)
        holding_period: Rebalance whole portfolio every this many prediction days
        transaction_costs: Dict with 'enabled', 'bid_ask_spread', 'slippage'
        rank_drop_gate: Dict with 'enabled' and 'min_rank_drop'

    Returns:
        Dictionary with same schema as simulate_trading_strategy()
    """
    tc_enabled = False
    bid_ask_spread = 0.0
    slippage = 0.0
    if transaction_costs and transaction_costs.get('enabled', False):
        tc_enabled = True
        bid_ask_spread = transaction_costs.get('bid_ask_spread', 0.001)
        slippage = transaction_costs.get('slippage', 0.0005)

    gate_enabled = False
    min_rank_drop = 10
    if rank_drop_gate and rank_drop_gate.get('enabled', False):
        gate_enabled = True
        min_rank_drop = rank_drop_gate.get('min_rank_drop', 10)

    cost_status = "ENABLED" if tc_enabled else "DISABLED"
    gate_status = "ENABLED" if gate_enabled else "DISABLED"
    print(f"\nSimulating block rebalance strategy (top-{top_k} stocks)...")
    print(f"  Block rebalance: entire portfolio of {top_k} stocks rebalanced every {holding_period} days (retail-style)")
    print(f"  Portfolio: open-to-open returns (entry at T+1 open, exit at T+2 open)")
    print(f"  Benchmark: open-to-open returns (equal-weighted, same window as portfolio)")
    print(f"  Transaction costs: {cost_status}")
    if tc_enabled:
        print(f"    Bid-ask spread: {bid_ask_spread*10000:.1f} bps (round-trip)")
        print(f"    Slippage: {slippage*10000:.1f} bps (per trade)")
    print(f"  Rank-drop gate: {gate_status}")
    if gate_enabled:
        print(f"    Min rank drop: {min_rank_drop} (vs ranks from last rebalance)")

    pred_dates = sorted(predictions_df['dt'].unique())
    all_stock_dates = sorted(stock_data_df['dt'].unique())
    print(f"  {len(pred_dates)} prediction dates")

    if 'open_to_open_return' not in stock_data_df.columns:
        raise ValueError(
            "stock_data_df must have 'open_to_open_return' column. "
            "Run calculate_forward_returns() first!"
        )
    if 'daily_return' not in stock_data_df.columns:
        raise ValueError(
            "stock_data_df must have 'daily_return' column. "
            "Run calculate_forward_returns() first!"
        )

    # Single portfolio state
    current_holdings = set()
    prev_ranks = None

    gross_portfolio_returns = []
    net_portfolio_returns = []
    benchmark_returns = []
    dates = []
    all_predictions = []
    all_actuals = []
    daily_turnover = []
    daily_costs = []
    daily_num_trades = []
    daily_holdings_records = []
    trade_records = []
    days_with_gate_exits = 0

    for i, pred_date in enumerate(pred_dates):
        day_preds = predictions_df[predictions_df['dt'] == pred_date].copy()
        if len(day_preds) < top_k:
            # Still need entry_date for benchmark / skip day
            try:
                entry_date_idx = all_stock_dates.index(pred_date) + 1
                if entry_date_idx >= len(all_stock_dates):
                    continue
                entry_date = all_stock_dates[entry_date_idx]
            except (ValueError, IndexError):
                continue
            entry_day_data = stock_data_df[stock_data_df['dt'] == entry_date]
            if len(entry_day_data) == 0 or not current_holdings:
                continue
            # Have holdings but few preds: still book return for this day
            held_returns = entry_day_data[
                entry_day_data['kdcode'].isin(current_holdings)
            ]['open_to_open_return'].dropna()
            if len(held_returns) == 0:
                continue
            gross_return = held_returns.mean()
            net_return = gross_return
            all_ret = entry_day_data['open_to_open_return'].dropna()
            benchmark_return = all_ret.mean() if len(all_ret) > 0 else 0.0
            gross_portfolio_returns.append(gross_return)
            net_portfolio_returns.append(net_return)
            benchmark_returns.append(benchmark_return)
            dates.append(entry_date)
            daily_turnover.append(0.0)
            daily_costs.append(0.0)
            daily_num_trades.append(0)
            # daily_holdings
            day_score_map = day_preds.set_index('kdcode')['score'].to_dict() if len(day_preds) > 0 else {}
            w = 1.0 / len(current_holdings)
            for kd in current_holdings:
                row = entry_day_data[entry_day_data['kdcode'] == kd]
                if len(row) > 0 and not pd.isna(row['open_to_open_return'].values[0]):
                    r = float(row['open_to_open_return'].values[0])
                    daily_holdings_records.append({
                        'pred_date': pred_date, 'entry_date': entry_date, 'kdcode': kd,
                        'tranche_id': 0, 'rank': np.nan, 'score': day_score_map.get(kd, np.nan),
                        'weight': w, 'stock_return': r, 'contribution': r * w,
                        'rank_gate_enabled': gate_enabled, 'min_rank_drop': min_rank_drop if gate_enabled else np.nan,
                        'transaction_costs_enabled': tc_enabled,
                    })
            # MSE/MAE
            pred_date_data = stock_data_df[stock_data_df['dt'] == pred_date]
            for _, row in day_preds.iterrows():
                kdcode, score = row['kdcode'], row['score']
                actual = pred_date_data[pred_date_data['kdcode'] == kdcode][f'forward_return_{label_t}d']
                if len(actual) > 0 and not pd.isna(actual.values[0]):
                    all_predictions.append(score)
                    all_actuals.append(actual.values[0])
            continue

        day_preds = day_preds.sort_values('score', ascending=False).reset_index(drop=True)
        day_preds['_rank'] = np.arange(1, len(day_preds) + 1, dtype=int)
        current_ranks = day_preds.set_index('kdcode')['_rank'].to_dict()

        is_rebalance_day = (i % holding_period == 0)

        try:
            entry_date_idx = all_stock_dates.index(pred_date) + 1
            entry_date_this = all_stock_dates[entry_date_idx] if entry_date_idx < len(all_stock_dates) else None
        except (ValueError, IndexError):
            entry_date_this = None

        if is_rebalance_day:
            if not gate_enabled:
                top_stocks = day_preds.head(top_k)['kdcode'].tolist()
            elif not current_holdings or prev_ranks is None:
                top_stocks = day_preds.head(top_k)['kdcode'].tolist()
            else:
                current_held = [kd for kd in current_holdings if kd in current_ranks]
                survivors = []
                for kdcode in current_held:
                    pr, cr = prev_ranks.get(kdcode), current_ranks.get(kdcode)
                    if pr is None or cr is None:
                        survivors.append(kdcode)
                        continue
                    if (cr - pr) >= min_rank_drop:
                        continue
                    survivors.append(kdcode)
                if len(survivors) < top_k:
                    days_with_gate_exits += 1
                survivor_set = set(survivors)
                refill = [kd for kd in day_preds['kdcode'].tolist() if kd not in survivor_set]
                slots = max(0, top_k - len(survivors))
                top_stocks = survivors + refill[:slots]

            new_holdings = set(top_stocks)
            turnover_info = calculate_turnover(current_holdings, new_holdings, target_k=top_k)

            if tc_enabled:
                cost_info = calculate_transaction_cost(turnover_info, bid_ask_spread, slippage)
                portfolio_cost = cost_info['total_cost']
            else:
                portfolio_cost = 0.0

            day_score_map = day_preds.set_index('kdcode')['score'].to_dict()
            trade_date = entry_date_this if entry_date_this else pred_date
            for bought_kd in sorted(turnover_info['stocks_bought']):
                trade_records.append({
                    'date': trade_date, 'pred_date': pred_date, 'kdcode': bought_kd, 'action': 'BUY',
                    'tranche_id': 0,
                    'rank': int(current_ranks.get(bought_kd, 0)) if bought_kd in current_ranks else np.nan,
                    'score': float(day_score_map.get(bought_kd, np.nan)),
                })
            for sold_kd in sorted(turnover_info['stocks_sold']):
                trade_records.append({
                    'date': trade_date, 'pred_date': pred_date, 'kdcode': sold_kd, 'action': 'SELL',
                    'tranche_id': 0,
                    'rank': int(current_ranks.get(sold_kd, 0)) if sold_kd in current_ranks else np.nan,
                    'score': float(day_score_map.get(sold_kd, np.nan)),
                })

            current_holdings = new_holdings
            prev_ranks = current_ranks

            daily_turnover.append(turnover_info['one_way_turnover'])
            daily_costs.append(portfolio_cost)
            daily_num_trades.append(turnover_info['num_trades'])
        else:
            daily_turnover.append(0.0)
            daily_costs.append(0.0)
            daily_num_trades.append(0)
            portfolio_cost = 0.0

        entry_date = entry_date_this
        if entry_date is None:
            continue

        entry_day_data = stock_data_df[stock_data_df['dt'] == entry_date]
        if len(entry_day_data) == 0:
            continue

        if not current_holdings:
            continue

        held_returns = entry_day_data[
            entry_day_data['kdcode'].isin(current_holdings)
        ]['open_to_open_return'].dropna()
        if len(held_returns) == 0:
            continue

        gross_return = held_returns.mean()
        net_return = gross_return - (portfolio_cost if is_rebalance_day else 0.0)

        all_ret = entry_day_data['open_to_open_return'].dropna()
        benchmark_return = all_ret.mean() if len(all_ret) > 0 else 0.0

        gross_portfolio_returns.append(gross_return)
        net_portfolio_returns.append(net_return)
        benchmark_returns.append(benchmark_return)
        dates.append(entry_date)

        day_score_map = day_preds.set_index('kdcode')['score'].to_dict()
        w = 1.0 / len(current_holdings)
        for kd in current_holdings:
            row = entry_day_data[entry_day_data['kdcode'] == kd]
            if len(row) > 0 and not pd.isna(row['open_to_open_return'].values[0]):
                r = float(row['open_to_open_return'].values[0])
                daily_holdings_records.append({
                    'pred_date': pred_date, 'entry_date': entry_date, 'kdcode': kd,
                    'tranche_id': 0, 'rank': int(current_ranks.get(kd, 0)) if kd in current_ranks else np.nan,
                    'score': float(day_score_map.get(kd, np.nan)), 'weight': w, 'stock_return': r,
                    'contribution': r * w, 'rank_gate_enabled': gate_enabled,
                    'min_rank_drop': min_rank_drop if gate_enabled else np.nan,
                    'transaction_costs_enabled': tc_enabled,
                })

        pred_date_data = stock_data_df[stock_data_df['dt'] == pred_date]
        for _, row in day_preds.iterrows():
            kdcode, score = row['kdcode'], row['score']
            actual = pred_date_data[pred_date_data['kdcode'] == kdcode][f'forward_return_{label_t}d']
            if len(actual) > 0 and not pd.isna(actual.values[0]):
                all_predictions.append(score)
                all_actuals.append(actual.values[0])

    total_tc = sum(daily_costs)
    avg_daily_turnover = np.mean(daily_turnover) if daily_turnover else 0.0
    avg_daily_cost = np.mean(daily_costs) if daily_costs else 0.0
    total_trades = sum(daily_num_trades)

    print(f"  Completed block simulation: {len(dates)} valid trading days")
    if gate_enabled:
        print(f"  Rank-drop gate: {days_with_gate_exits} days with exits triggered")
    if tc_enabled:
        print(f"  Transaction cost summary:")
        print(f"    Total costs: {total_tc*100:.4f}% of portfolio")
        print(f"    Avg daily turnover: {avg_daily_turnover*100:.2f}%")
        print(f"    Avg daily cost: {avg_daily_cost*10000:.2f} bps")
        print(f"    Total trades: {total_trades}")

    return {
        'portfolio_returns': np.array(net_portfolio_returns) if tc_enabled else np.array(gross_portfolio_returns),
        'gross_portfolio_returns': np.array(gross_portfolio_returns),
        'net_portfolio_returns': np.array(net_portfolio_returns),
        'benchmark_returns': np.array(benchmark_returns),
        'dates': dates,
        'predictions': np.array(all_predictions),
        'actuals': np.array(all_actuals),
        'transaction_costs_enabled': tc_enabled,
        'daily_turnover': np.array(daily_turnover),
        'daily_costs': np.array(daily_costs),
        'daily_num_trades': np.array(daily_num_trades),
        'total_transaction_cost': total_tc,
        'avg_daily_turnover': avg_daily_turnover,
        'avg_daily_cost': avg_daily_cost,
        'total_trades': total_trades,
        'tc_settings': {'bid_ask_spread': bid_ask_spread, 'slippage': slippage},
        'rank_gate_enabled': gate_enabled,
        'min_rank_drop': min_rank_drop if gate_enabled else None,
        'days_with_gate_exits': days_with_gate_exits if gate_enabled else 0,
        'days_skipped_by_rank_gate': 0,
        'daily_holdings': daily_holdings_records,
        'trade_records': trade_records,
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
    
    # Get transaction cost settings (with fallback to defaults)
    tc_config = config.get('transaction_costs', DEFAULT_CONFIG['transaction_costs'])
    rank_drop_config = config.get('rank_drop_gate', DEFAULT_CONFIG['rank_drop_gate'])
    
    # Simulate trading strategy (dispatch based on holding period and rebalance style)
    holding_period = config.get('holding_period', 1)
    rebalance_style = config.get('rebalance_style', 'staggered')
    if holding_period == 1:
        sim_results = simulate_trading_strategy(
            predictions_df=predictions_df,
            stock_data_df=stock_data,
            top_k=config['top_k'],
            label_t=config['label_t'],
            transaction_costs=tc_config,
            rank_drop_gate=rank_drop_config
        )
    elif rebalance_style == 'block':
        sim_results = simulate_trading_strategy_block(
            predictions_df=predictions_df,
            stock_data_df=stock_data,
            top_k=config['top_k'],
            label_t=config['label_t'],
            holding_period=holding_period,
            transaction_costs=tc_config,
            rank_drop_gate=rank_drop_config
        )
    else:
        sim_results = simulate_trading_strategy_staggered(
            predictions_df=predictions_df,
            stock_data_df=stock_data,
            top_k=config['top_k'],
            label_t=config['label_t'],
            holding_period=holding_period,
            transaction_costs=tc_config,
            rank_drop_gate=rank_drop_config
        )
    
    portfolio_returns = sim_results['portfolio_returns']
    benchmark_returns = sim_results['benchmark_returns']
    dates = sim_results['dates']
    tc_enabled = sim_results['transaction_costs_enabled']
    
    if len(portfolio_returns) == 0:
        print("ERROR: No valid trading days found!")
        return None
    
    # Calculate cumulative portfolio values (starting at 1.0)
    portfolio_values = np.cumprod(1 + portfolio_returns)
    
    # Calculate metrics (based on net returns if costs enabled)
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
        'rank_gate_enabled': sim_results.get('rank_gate_enabled', False),
        'min_rank_drop': sim_results.get('min_rank_drop'),
        'days_with_gate_exits': sim_results.get('days_with_gate_exits', 0),
        'days_skipped_by_rank_gate': sim_results.get('days_skipped_by_rank_gate', 0),
    }
    
    # Add transaction cost metrics if enabled
    if tc_enabled:
        # Calculate gross metrics for comparison
        gross_returns = sim_results['gross_portfolio_returns']
        gross_values = np.cumprod(1 + gross_returns)
        gross_arr = calculate_arr(gross_returns, config['trading_days_per_year'])
        gross_total = gross_values[-1] - 1 if len(gross_values) > 0 else 0
        
        results.update({
            # Transaction cost summary
            'transaction_costs_enabled': True,
            'tc_bid_ask_spread_bps': sim_results['tc_settings']['bid_ask_spread'] * 10000,
            'tc_slippage_bps': sim_results['tc_settings']['slippage'] * 10000,
            'total_transaction_cost': sim_results['total_transaction_cost'],
            'avg_daily_turnover': sim_results['avg_daily_turnover'],
            'avg_daily_cost_bps': sim_results['avg_daily_cost'] * 10000,
            'total_trades': sim_results['total_trades'],
            # Gross vs Net comparison
            'gross_ARR': gross_arr,
            'gross_total_return': gross_total,
            'net_ARR': arr,  # Same as 'ARR' but explicit
            'net_total_return': total_return,  # Same as 'total_return' but explicit
            'cost_drag_ARR': gross_arr - arr,  # How much costs reduced annual return
        })
    else:
        results['transaction_costs_enabled'] = False
    
    return results


def print_results(results, model_name="MCI-GRU", num_tests=1, adjustment_method='bhy'):
    """
    Pretty print evaluation results in paper format.
    
    Args:
        results: Dictionary of evaluation metrics
        model_name: Name to display
        num_tests: Number of strategies/configurations tested (for haircut)
        adjustment_method: Method for multiple testing adjustment ('bhy', 'bonferroni', 'holm')
    """
    print("\n" + "=" * 70)
    print(f"  EVALUATION RESULTS: {model_name}")
    print("=" * 70)
    print()
    print("  BACKTESTING METHODOLOGY:")
    print("  " + "-" * 66)
    print("  Portfolio: Open-to-open returns (entry T+1 open, exit T+2 open)")
    print("  Benchmark: Open-to-open returns (equal-weighted, same window)")
    print("  Holding period: Intraday + overnight (realistic portfolio behavior)")
    print("  " + "-" * 66)
    print()
    
    # Calculate t-statistic and haircut if applicable
    num_years = results['num_trading_days'] / 252  # Trading days to years
    t_stat = calculate_t_statistic(results['ASR'], num_years)
    
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
    
    # Statistical significance (Harvey & Liu, 2014)
    print("  Statistical Significance (Harvey & Liu, 2014):")
    print("  " + "-" * 66)
    print(f"  {'t-statistic':<25} {t_stat:>12.4f}  (SR x sqrt(years))")
    print(f"  {'Test Period (years)':<25} {num_years:>12.2f}")
    
    original_p = t_stat_to_p_value(t_stat)
    print(f"  {'Original p-value':<25} {original_p:>12.6f}")
    
    if num_tests > 1:
        # Apply multiple testing adjustment
        haircut_result = haircut_sharpe_ratio(
            results['ASR'], num_years, num_tests, 
            method=adjustment_method, rank=1
        )
        
        print("  " + "-" * 66)
        print(f"  Multiple Testing Adjustment ({adjustment_method.upper()} method):")
        print("  " + "-" * 66)
        print(f"  {'Number of Tests':<25} {num_tests:>12}")
        print(f"  {'Adjusted p-value':<25} {haircut_result['adjusted_p_value']:>12.6f}")
        print(f"  {'Adjusted t-statistic':<25} {haircut_result['adjusted_t_statistic']:>12.4f}")
        print(f"  {'Haircutted Sharpe Ratio':<25} {haircut_result['haircutted_sharpe']:>12.4f}")
        print(f"  {'Haircut Percentage':<25} {haircut_result['haircut_pct']:>11.1f}%")
        
        sig_status = "YES" if haircut_result['is_significant'] else "NO"
        print(f"  {'Significant (p<0.05)?':<25} {sig_status:>12}")
        
        if not haircut_result['is_significant']:
            print()
            print("  WARNING: After adjusting for multiple testing, the strategy")
            print("           is NOT statistically significant at the 5% level.")
    else:
        sig_status = "YES" if original_p < 0.05 else "NO"
        print(f"  {'Significant (p<0.05)?':<25} {sig_status:>12}")
        print()
        print("  Note: No multiple testing adjustment applied (num_tests=1).")
        print("        If you tried multiple configurations, use --num_tests N")
    
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
    
    # Transaction cost information (if enabled)
    if results.get('transaction_costs_enabled', False):
        print("  Transaction Costs (Retail Investor Model):")
        print("  " + "-" * 66)
        print(f"  {'Bid-Ask Spread':<25} {results['tc_bid_ask_spread_bps']:>12.1f} bps (round-trip)")
        print(f"  {'Slippage':<25} {results['tc_slippage_bps']:>12.1f} bps (per trade)")
        print("  " + "-" * 66)
        print(f"  {'Total Transaction Cost':<25} {results['total_transaction_cost']*100:>11.4f}%  (cumulative)")
        print(f"  {'Avg Daily Turnover':<25} {results['avg_daily_turnover']*100:>11.2f}%")
        print(f"  {'Avg Daily Cost':<25} {results['avg_daily_cost_bps']:>11.2f} bps")
        print(f"  {'Total Trades':<25} {results['total_trades']:>12}")
        print("  " + "-" * 66)
        print("  Gross vs Net Performance:")
        print("  " + "-" * 66)
        print(f"  {'Gross ARR (before costs)':<25} {results['gross_ARR']:>12.4f}  ({results['gross_ARR']*100:.2f}%)")
        print(f"  {'Net ARR (after costs)':<25} {results['net_ARR']:>12.4f}  ({results['net_ARR']*100:.2f}%)")
        print(f"  {'Cost Drag on ARR':<25} {results['cost_drag_ARR']:>12.4f}  ({results['cost_drag_ARR']*100:.2f}%)")
        print(f"  {'Gross Total Return':<25} {results['gross_total_return']:>12.4f}  ({results['gross_total_return']*100:.2f}%)")
        print(f"  {'Net Total Return':<25} {results['net_total_return']:>12.4f}  ({results['net_total_return']*100:.2f}%)")
        print("  " + "-" * 66)
        print()
    else:
        print("  Note: Transaction costs not included (use --transaction_costs to enable)")
        print()
    
    # Comparison hint
    print("  Paper S&P 500 Results (Table 4):")
    print("  " + "-" * 66)
    print("  MCI-GRU: ARR=0.456, AVoL=0.179, MDD=-0.129, ASR=2.549, CR=3.543, IR=2.197")
    print("  (Paper excludes transaction costs)")
    print("  NOTE: Paper's benchmark uses close-to-close returns; ours uses open-to-open")
    print("        for apples-to-apples comparison. IR values may differ slightly.")
    print("  " + "-" * 66)
    print("=" * 70 + "\n")


def save_results(results, output_path, num_tests=1, adjustment_method='bhy'):
    """
    Save results to CSV file, including multiple testing adjustments.
    
    Args:
        results: Dictionary of evaluation metrics
        output_path: Path to save CSV
        num_tests: Number of strategies tested (for haircut calculation)
        adjustment_method: Method used for adjustment
    """
    # Create a copy to avoid modifying original
    save_data = dict(results)
    
    # Calculate and add statistical significance metrics
    num_years = results['num_trading_days'] / 252
    t_stat = calculate_t_statistic(results['ASR'], num_years)
    original_p = t_stat_to_p_value(t_stat)
    
    save_data['t_statistic'] = t_stat
    save_data['original_p_value'] = original_p
    save_data['num_tests'] = num_tests
    save_data['adjustment_method'] = adjustment_method
    
    # Add haircut metrics if multiple tests
    if num_tests > 1:
        haircut_result = haircut_sharpe_ratio(
            results['ASR'], num_years, num_tests, 
            method=adjustment_method, rank=1
        )
        save_data['adjusted_p_value'] = haircut_result['adjusted_p_value']
        save_data['haircutted_sharpe'] = haircut_result['haircutted_sharpe']
        save_data['haircut_pct'] = haircut_result['haircut_pct']
        save_data['is_significant'] = haircut_result['is_significant']
    else:
        save_data['adjusted_p_value'] = original_p
        save_data['haircutted_sharpe'] = results['ASR']
        save_data['haircut_pct'] = 0.0
        save_data['is_significant'] = original_p < 0.05
    
    df = pd.DataFrame([save_data])
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def setup_backtest_output_dir(predictions_dir: str, suffix: str = "") -> str:
    """
    Create backtest output directory alongside predictions.
    
    Args:
        predictions_dir: Path to averaged_predictions folder
        suffix: Optional suffix for multiple backtests (e.g., "_with_costs")
    
    Returns:
        Path to backtest output directory
    """
    # predictions_dir is typically: {output_dir}/{experiment_name}/{timestamp}/averaged_predictions
    # We want: {output_dir}/{experiment_name}/{timestamp}/backtest{suffix}
    
    parent_dir = os.path.dirname(predictions_dir)
    backtest_dir = os.path.join(parent_dir, f'backtest{suffix}')
    os.makedirs(backtest_dir, exist_ok=True)
    
    return backtest_dir


def derive_portfolio_composition(holdings_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-day composition labels from holdings and trade events.
    Marks holdings as NEW if bought on that entry date, otherwise HELD.
    """
    output_cols = [
        'entry_date', 'pred_date', 'kdcode', 'status',
        'rank', 'score', 'weight', 'stock_return', 'contribution'
    ]
    if holdings_df is None or len(holdings_df) == 0:
        return pd.DataFrame(columns=output_cols)
    
    comp_df = holdings_df.copy()
    comp_df['status'] = 'HELD'
    
    if trades_df is not None and len(trades_df) > 0 and 'action' in trades_df.columns:
        buys_df = trades_df[trades_df['action'] == 'BUY'].copy()
        if len(buys_df) > 0:
            buys_df = buys_df[['date', 'kdcode']].drop_duplicates()
            buys_df['status'] = 'NEW'
            comp_df = comp_df.merge(
                buys_df,
                how='left',
                left_on=['entry_date', 'kdcode'],
                right_on=['date', 'kdcode'],
                suffixes=('', '_buy')
            )
            comp_df['status'] = comp_df['status_buy'].fillna(comp_df['status'])
            comp_df = comp_df.drop(columns=['date', 'status_buy'], errors='ignore')
    
    return comp_df.reindex(columns=output_cols)


def derive_holdings_summary(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive aggregate per-stock statistics across the simulation horizon.
    """
    output_cols = [
        'kdcode', 'times_held', 'avg_score', 'avg_return',
        'total_contribution', 'win_rate'
    ]
    if holdings_df is None or len(holdings_df) == 0:
        return pd.DataFrame(columns=output_cols)
    
    summary_df = holdings_df.groupby('kdcode', as_index=False).agg(
        times_held=('stock_return', 'count'),
        avg_score=('score', 'mean'),
        avg_return=('stock_return', 'mean'),
        total_contribution=('contribution', 'sum')
    )
    win_rate_df = holdings_df.groupby('kdcode', as_index=False)['stock_return'].apply(
        lambda x: float((x > 0).mean())
    ).rename(columns={'stock_return': 'win_rate'})
    
    summary_df = summary_df.merge(win_rate_df, on='kdcode', how='left')
    summary_df = summary_df.sort_values('total_contribution', ascending=False).reset_index(drop=True)
    return summary_df.reindex(columns=output_cols)


def save_backtest_results(
    results: dict, 
    backtest_dir: str,
    sim_results: dict = None,
    config: dict = None,
    num_tests: int = 1,
    adjustment_method: str = 'bhy'
) -> str:
    """
    Save comprehensive backtest results in organized structure.
    
    Args:
        results: Dictionary of metrics from evaluate()
        backtest_dir: Directory to save results
        sim_results: Raw simulation results (optional, for time series)
        config: Configuration used (optional)
        num_tests: Number of tests for multiple testing adjustment
        adjustment_method: Adjustment method for multiple testing
    
    Returns:
        Path to backtest directory
    """
    import json
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nSaving backtest results to: {backtest_dir}")
    
    # 1. Save main results CSV (with statistical adjustments)
    results_file = os.path.join(backtest_dir, 'backtest_results.csv')
    save_results(results, results_file, num_tests, adjustment_method)
    
    # 2. Save detailed metrics as JSON (easier to read)
    metrics_file = os.path.join(backtest_dir, 'backtest_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Metrics JSON: {metrics_file}")
    
    # 3. Save configuration
    if config:
        config_file = os.path.join(backtest_dir, 'backtest_config.json')
        with open(config_file, 'w') as f:
            # Convert any non-serializable objects
            config_copy = {}
            for k, v in config.items():
                if isinstance(v, dict):
                    config_copy[k] = {str(k2): v2 for k2, v2 in v.items()}
                else:
                    config_copy[k] = v
            json.dump(config_copy, f, indent=2, default=str)
        print(f"  Config: {config_file}")
    
    # 4. Save time series data (portfolio returns, cumulative performance)
    if sim_results:
        # Daily returns
        returns_df = pd.DataFrame({
            'date': sim_results['dates'],
            'portfolio_return': sim_results['portfolio_returns'],
            'benchmark_return': sim_results['benchmark_returns'],
            'excess_return': sim_results['portfolio_returns'] - sim_results['benchmark_returns']
        })
        
        if sim_results.get('transaction_costs_enabled', False):
            returns_df['gross_portfolio_return'] = sim_results.get('gross_portfolio_returns', sim_results['portfolio_returns'])
            returns_df['transaction_cost'] = returns_df['gross_portfolio_return'] - returns_df['portfolio_return']
        
        returns_file = os.path.join(backtest_dir, 'daily_returns.csv')
        returns_df.to_csv(returns_file, index=False)
        print(f"  Daily returns: {returns_file}")
        
        # Cumulative performance
        cum_portfolio = np.cumprod(1 + sim_results['portfolio_returns'])
        cum_benchmark = np.cumprod(1 + sim_results['benchmark_returns'])
        
        perf_df = pd.DataFrame({
            'date': sim_results['dates'],
            'portfolio_value': cum_portfolio,
            'benchmark_value': cum_benchmark,
            'relative_performance': cum_portfolio / cum_benchmark,
            'portfolio_drawdown': (cum_portfolio / np.maximum.accumulate(cum_portfolio) - 1)
        })
        perf_file = os.path.join(backtest_dir, 'cumulative_performance.csv')
        perf_df.to_csv(perf_file, index=False)
        print(f"  Cumulative performance: {perf_file}")
        
        # Monthly aggregation for easier analysis
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        returns_df['year_month'] = returns_df['date'].dt.to_period('M')
        
        monthly_stats = returns_df.groupby('year_month').agg({
            'portfolio_return': ['sum', 'std', 'count'],
            'benchmark_return': ['sum', 'std'],
            'excess_return': ['sum', 'mean']
        }).reset_index()
        monthly_stats.columns = ['year_month', 'portfolio_return', 'portfolio_volatility', 'trading_days',
                                 'benchmark_return', 'benchmark_volatility', 'excess_return', 'avg_excess_return']
        monthly_file = os.path.join(backtest_dir, 'monthly_performance.csv')
        monthly_stats.to_csv(monthly_file, index=False)
        print(f"  Monthly performance: {monthly_file}")
        
        # 5. Additive portfolio tracking outputs (if simulation detail is available)
        holdings_df = pd.DataFrame(sim_results.get('daily_holdings', []))
        trades_df = pd.DataFrame(sim_results.get('trade_records', []))
        
        if len(holdings_df) > 0:
            holdings_file = os.path.join(backtest_dir, 'daily_holdings.csv')
            holdings_df.to_csv(holdings_file, index=False)
            print(f"  Daily holdings: {holdings_file}")
            
            # Return attribution: within-day contribution split
            attribution_df = holdings_df.copy()
            day_contribution_sum = attribution_df.groupby('entry_date')['contribution'].transform('sum')
            attribution_df['pct_of_portfolio_return'] = np.where(
                day_contribution_sum.abs() > 1e-12,
                attribution_df['contribution'] / day_contribution_sum * 100.0,
                np.nan
            )
            attribution_cols = [
                'entry_date', 'pred_date', 'kdcode', 'stock_return',
                'weight', 'contribution', 'pct_of_portfolio_return'
            ]
            attribution_file = os.path.join(backtest_dir, 'return_attribution.csv')
            attribution_df.reindex(columns=attribution_cols).to_csv(attribution_file, index=False)
            print(f"  Return attribution: {attribution_file}")
            
            composition_df = derive_portfolio_composition(holdings_df, trades_df)
            if len(composition_df) > 0:
                composition_file = os.path.join(backtest_dir, 'portfolio_composition.csv')
                composition_df.to_csv(composition_file, index=False)
                print(f"  Portfolio composition: {composition_file}")
            
            summary_df = derive_holdings_summary(holdings_df)
            if len(summary_df) > 0:
                summary_file = os.path.join(backtest_dir, 'holdings_summary.csv')
                summary_df.to_csv(summary_file, index=False)
                print(f"  Holdings summary: {summary_file}")
        
        if len(trades_df) > 0:
            trades_file = os.path.join(backtest_dir, 'trade_journal.csv')
            trades_df.to_csv(trades_file, index=False)
            print(f"  Trade journal: {trades_file}")
    
    # 6. Create summary text file
    summary_file = os.path.join(backtest_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BACKTEST SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        if config:
            f.write(f"Test Period: {config.get('test_start', 'N/A')} to {config.get('test_end', 'N/A')}\n")
            f.write(f"Top-K Stocks: {config.get('top_k', 'N/A')}\n")
            f.write(f"Label Period: {config.get('label_t', 'N/A')} days\n")
            
            if config.get('transaction_costs', {}).get('enabled', False):
                tc = config['transaction_costs']
                f.write(f"Transaction Costs: Enabled\n")
                f.write(f"  Bid-Ask Spread: {tc.get('bid_ask_spread', 0) * 10000:.1f} bps\n")
                f.write(f"  Slippage: {tc.get('slippage', 0) * 10000:.1f} bps\n")
            else:
                f.write(f"Transaction Costs: Disabled\n")
            if results.get('rank_gate_enabled', False):
                f.write(f"Rank-Drop Gate: Enabled (min_rank_drop={results.get('min_rank_drop', 'N/A')})\n")
                f.write(f"  Days with gate-triggered exits: {results.get('days_with_gate_exits', 0)}\n")
                f.write(f"  Days skipped (empty post-gate portfolio): {results.get('days_skipped_by_rank_gate', 0)}\n")
            else:
                f.write(f"Rank-Drop Gate: Disabled\n")
            f.write("\n")
        
        f.write("Key Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  ARR (Annualized Return):       {results.get('ARR', 0):.4f} ({results.get('ARR', 0)*100:.2f}%)\n")
        f.write(f"  AVoL (Annualized Volatility):  {results.get('AVoL', 0):.4f} ({results.get('AVoL', 0)*100:.2f}%)\n")
        f.write(f"  MDD (Maximum Drawdown):        {results.get('MDD', 0):.4f} ({results.get('MDD', 0)*100:.2f}%)\n")
        f.write(f"  ASR (Annualized Sharpe):       {results.get('ASR', 0):.4f}\n")
        f.write(f"  CR (Calmar Ratio):             {results.get('CR', 0):.4f}\n")
        f.write(f"  IR (Information Ratio):        {results.get('IR', 0):.4f}\n")
        f.write("\n")
        
        f.write("Prediction Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  MSE (Mean Squared Error):      {results.get('MSE', 0):.6f}\n")
        f.write(f"  MAE (Mean Absolute Error):     {results.get('MAE', 0):.6f}\n")
        f.write("\n")
        
        if num_tests > 1:
            f.write("Multiple Testing Adjustment:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Number of tests: {num_tests}\n")
            f.write(f"  Adjustment method: {adjustment_method.upper()}\n")
            
            # Calculate adjustments
            num_years = results['num_trading_days'] / 252
            haircut_result = haircut_sharpe_ratio(
                results['ASR'], num_years, num_tests,
                method=adjustment_method, rank=1
            )
            f.write(f"  Original Sharpe: {results['ASR']:.4f}\n")
            f.write(f"  Haircutted Sharpe: {haircut_result['haircutted_sharpe']:.4f}\n")
            f.write(f"  Haircut: {haircut_result['haircut_pct']:.2f}%\n")
            f.write(f"  Statistical Significance: {'Yes' if haircut_result['is_significant'] else 'No'} (p={haircut_result['adjusted_p_value']:.4f})\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {timestamp}\n")
    
    print(f"  Summary: {summary_file}")
    print("\nAll backtest outputs saved successfully.")
    
    return backtest_dir


def setup_backtest_logging(backtest_dir: str) -> logging.Logger:
    """
    Setup logging for backtest evaluation.
    
    Args:
        backtest_dir: Directory for backtest outputs
        
    Returns:
        Configured logger
    """
    import logging
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(backtest_dir, f'backtest_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('backtest')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Backtest logging initialized: {log_file}")
    
    return logger


def _infer_experiment_name(predictions_dir: str) -> str:
    """Infer experiment name from a standard averaged_predictions path."""
    predictions_path = Path(predictions_dir).resolve()
    if predictions_path.parent.parent.exists():
        return predictions_path.parent.parent.name
    return "backtest"


def setup_backtest_tracking(
    predictions_dir: str,
    config: dict,
    enable_mlflow: bool = False,
    tracking_uri: str = None,
    experiment_name: str = None,
    backtest_suffix: str = "",
) -> tuple:
    """Create an optional MLflow child run for backtest tracking.

    If the predictions directory belongs to a tracked training run, the
    backtest automatically links itself beneath that parent run.
    """
    linked_metadata = load_run_metadata_from_predictions_dir(predictions_dir)
    tracking_enabled = enable_mlflow or linked_metadata is not None

    if not tracking_enabled:
        return MLflowTrackingManager(enabled=False), linked_metadata

    resolved_tracking_uri = tracking_uri
    parent_run_id = None
    resolved_experiment_name = experiment_name

    if linked_metadata is not None:
        resolved_tracking_uri = resolved_tracking_uri or linked_metadata.get("tracking_uri")
        resolved_experiment_name = (
            resolved_experiment_name
            or linked_metadata.get("experiment_name")
            or _infer_experiment_name(predictions_dir)
        )
        parent_run_id = linked_metadata.get("run_id")
    else:
        resolved_experiment_name = resolved_experiment_name or _infer_experiment_name(predictions_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"backtest{backtest_suffix or ''}-{timestamp}"

    manager = MLflowTrackingManager(
        enabled=True,
        tracking_uri=resolved_tracking_uri,
        experiment_name=resolved_experiment_name,
        run_name=run_name,
        output_path=Path(predictions_dir).resolve().parent,
        parent_run_id=parent_run_id,
        tags={
            "run_kind": "backtest",
            "linked_training_run": linked_metadata is not None,
            "predictions_dir": str(Path(predictions_dir).resolve()),
            "top_k": config.get("top_k"),
            "label_t": config.get("label_t"),
        },
    )
    return manager, linked_metadata


def _log_backtest_artifacts(tracking_manager: MLflowTrackingManager, backtest_dir: str):
    """Log the highest-value backtest artifacts."""
    if not tracking_manager.is_active:
        return

    artifact_names = [
        "backtest_results.csv", "backtest_metrics.json", "backtest_config.json",
        "summary.txt", "daily_returns.csv", "cumulative_performance.csv",
        "monthly_performance.csv", "equity_curve.png",
    ]
    for name in artifact_names:
        tracking_manager.log_artifact(Path(backtest_dir) / name, artifact_path="backtest_artifacts")


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
    
    # Get transaction cost settings
    tc_config = config.get('transaction_costs', DEFAULT_CONFIG['transaction_costs'])
    
    rank_drop_config = config.get('rank_drop_gate', DEFAULT_CONFIG['rank_drop_gate'])
    holding_period = config.get('holding_period', 1)
    rebalance_style = config.get('rebalance_style', 'staggered')
    if holding_period == 1:
        sim_results = simulate_trading_strategy(
            predictions_df=predictions_df,
            stock_data_df=stock_data,
            top_k=config['top_k'],
            label_t=config['label_t'],
            transaction_costs=tc_config,
            rank_drop_gate=rank_drop_config
        )
    elif rebalance_style == 'block':
        sim_results = simulate_trading_strategy_block(
            predictions_df=predictions_df,
            stock_data_df=stock_data,
            top_k=config['top_k'],
            label_t=config['label_t'],
            holding_period=holding_period,
            transaction_costs=tc_config,
            rank_drop_gate=rank_drop_config
        )
    else:
        sim_results = simulate_trading_strategy_staggered(
            predictions_df=predictions_df,
            stock_data_df=stock_data,
            top_k=config['top_k'],
            label_t=config['label_t'],
            holding_period=holding_period,
            transaction_costs=tc_config,
            rank_drop_gate=rank_drop_config
        )
    
    dates = pd.to_datetime(sim_results['dates'])
    portfolio_values = np.cumprod(1 + sim_results['portfolio_returns'])
    benchmark_values = np.cumprod(1 + sim_results['benchmark_returns'])
    excess_values = portfolio_values / benchmark_values
    
    # Determine plot title based on whether transaction costs are enabled
    tc_enabled = sim_results['transaction_costs_enabled']
    if tc_enabled:
        tc_info = f" (with TC: {sim_results['tc_settings']['bid_ask_spread']*10000:.0f}+{sim_results['tc_settings']['slippage']*10000:.0f} bps)"
    else:
        tc_info = " (Paper Methodology, no TC)"
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    # Top plot: Returns
    ax1 = axes[0]
    ax1.plot(dates, portfolio_values, 'r-', label='MCI-GRU Portfolio', linewidth=2)
    ax1.plot(dates, benchmark_values, 'b-', label='S&P 500 Benchmark', linewidth=1.5)
    ax1.plot(dates, excess_values, 'orange', label='Excess Return', linewidth=1.5)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title(f'MCI-GRU S&P 500 Backtest Performance{tc_info}')
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
        default='data/raw/market/sp500_data.csv',
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
        default='2025-01-01',
        help='Test period start date (default: 2025-01-01)'
    )
    
    parser.add_argument(
        '--test_end',
        type=str,
        default='2025-12-31',
        help='Test period end date (default: 2025-12-31)'
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
    
    # Multiple testing adjustment arguments (Harvey & Liu, 2014)
    parser.add_argument(
        '--num_tests',
        type=int,
        default=1,
        help='Number of strategies/configurations tested for multiple testing '
             'adjustment (default: 1 = no adjustment). If you tried 50 '
             'hyperparameter combinations, set --num_tests 50'
    )
    
    parser.add_argument(
        '--adjustment_method',
        type=str,
        default='bhy',
        choices=['bhy', 'bonferroni', 'holm'],
        help='Multiple testing adjustment method (default: bhy). '
             'bhy = Benjamini-Hochberg-Yekutieli (recommended for trading), '
             'bonferroni = most stringent (FWER), '
             'holm = step-down FWER method'
    )
    
    # Transaction cost arguments (retail investor model)
    parser.add_argument(
        '--transaction_costs',
        action='store_true',
        help='Enable transaction cost modeling (bid-ask spread + slippage)'
    )
    
    parser.add_argument(
        '--spread',
        type=float,
        default=10.0,
        help='Bid-ask spread in basis points (default: 10 bps = 0.10%% round-trip). '
             'For S&P 500 large-caps, typical spreads are 5-15 bps.'
    )
    
    parser.add_argument(
        '--slippage',
        type=float,
        default=5.0,
        help='Slippage in basis points per trade (default: 5 bps = 0.05%%). '
             'Represents execution price deviation for market orders.'
    )
    
    # New arguments for enhanced output management
    parser.add_argument(
        '--auto_save',
        action='store_true',
        help='Automatically save all outputs in organized structure'
    )
    
    parser.add_argument(
        '--backtest_suffix',
        type=str,
        default='',
        help='Suffix for backtest directory (e.g., "_with_costs" or "_tc")'
    )
    
    parser.add_argument(
        '--enable_mlflow', action='store_true',
        help='Enable MLflow tracking for this backtest. Auto-links to training '
             'run if mlflow_run.json exists in the predictions directory.'
    )
    parser.add_argument(
        '--mlflow_tracking_uri', type=str, default=None,
        help='Optional MLflow tracking URI override'
    )
    parser.add_argument(
        '--mlflow_experiment_name', type=str, default=None,
        help='Optional MLflow experiment name override'
    )

    # Rank-drop sell gate (exit held names only if rank worsens by >= N vs previous prediction day)
    parser.add_argument(
        '--enable_rank_drop_gate',
        action='store_true',
        help='Enable rank-drop sell gate: only exit held stocks whose prediction rank '
             'worsened by at least --min_rank_drop vs the previous prediction day'
    )
    parser.add_argument(
        '--min_rank_drop',
        type=int,
        default=10,
        help='Minimum rank worsening (current_rank - prev_rank) required to exit a held stock (default: 10)'
    )
    
    # Multi-day holding period (staggered tranche rebalancing)
    parser.add_argument(
        '--holding_period',
        type=int,
        default=1,
        help='Number of days to hold each tranche (default: 1 = daily rebalance). '
             'E.g., 21 = split portfolio into 21 tranches, each held 21 days.'
    )
    
    # Rebalance style when holding_period > 1
    parser.add_argument(
        '--rebalance_style',
        type=str,
        default='staggered',
        choices=['staggered', 'block'],
        help="When holding_period > 1: 'staggered' = 1/N of portfolio rebalances daily (institutional). "
             "'block' = whole portfolio of top_k stocks rebalances every holding_period days (retail)."
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
        'transaction_costs': {
            'enabled': args.transaction_costs,
            'bid_ask_spread': args.spread / 10000.0,  # Convert bps to decimal
            'slippage': args.slippage / 10000.0,      # Convert bps to decimal
        },
        'rank_drop_gate': {
            'enabled': args.enable_rank_drop_gate,
            'min_rank_drop': args.min_rank_drop,
        },
        'holding_period': args.holding_period,
        'rebalance_style': args.rebalance_style,
    }
    
    # Setup output directory if auto_save is enabled
    backtest_dir = None
    logger = None
    if args.auto_save:
        backtest_dir = setup_backtest_output_dir(args.predictions_dir, args.backtest_suffix)
        logger = setup_backtest_logging(backtest_dir)
        logger.info("Starting backtest evaluation with auto-save")
        logger.info(f"Predictions directory: {args.predictions_dir}")
        logger.info(f"Backtest output directory: {backtest_dir}")
    
    tracking_manager, linked_metadata = setup_backtest_tracking(
        predictions_dir=args.predictions_dir,
        config=config,
        enable_mlflow=args.enable_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        backtest_suffix=args.backtest_suffix,
    )

    if tracking_manager.enabled:
        tracking_manager.log_params({
            "backtest": config,
            "num_tests": args.num_tests,
            "adjustment_method": args.adjustment_method,
        })

    # Run evaluation
    if args.multi_model:
        # Multi-model evaluation (paper style: 10 runs averaged)
        if logger:
            logger.info(f"Running multi-model evaluation ({args.num_models} models)")
        
        multi_results = evaluate_multiple_models(
            args.multi_model, 
            num_models=args.num_models,
            config=config
        )
        
        if multi_results:
            print_results(
                multi_results['averaged'], 
                "MCI-GRU (Averaged)",
                num_tests=args.num_tests,
                adjustment_method=args.adjustment_method
            )
            
            if args.auto_save and backtest_dir:
                # Save comprehensive results
                save_backtest_results(
                    multi_results['averaged'],
                    backtest_dir,
                    config=config,
                    num_tests=args.num_tests,
                    adjustment_method=args.adjustment_method
                )
                
                # Save individual model results
                import json
                for i, result in enumerate(multi_results['individual']):
                    model_file = os.path.join(backtest_dir, f'model_{i}_results.json')
                    with open(model_file, 'w') as f:
                        json.dump(result, f, indent=2, default=float)
                print(f"  Individual model results saved: {len(multi_results['individual'])} files")
                
            elif args.output:
                save_results(
                    multi_results['averaged'], 
                    args.output,
                    num_tests=args.num_tests,
                    adjustment_method=args.adjustment_method
                )
    else:
        # Single evaluation
        if logger:
            logger.info("Running single model evaluation")
        
        results = evaluate(args.predictions_dir, config)
        
        if results:
            print_results(
                results, 
                "MCI-GRU",
                num_tests=args.num_tests,
                adjustment_method=args.adjustment_method
            )
            
            if args.auto_save and backtest_dir:
                # Load stock data for time series outputs
                stock_data = load_stock_data(
                    config['data_file'],
                    config['test_start'],
                    config['test_end']
                )
                
                # Load predictions
                predictions_df = load_predictions(args.predictions_dir)
                stock_data = calculate_forward_returns(stock_data, label_t=config['label_t'])
                
                # Get full simulation results
                holding_period = config.get('holding_period', 1)
                rebalance_style = config.get('rebalance_style', 'staggered')
                if holding_period == 1:
                    sim_results = simulate_trading_strategy(
                        predictions_df=predictions_df,
                        stock_data_df=stock_data,
                        top_k=config['top_k'],
                        label_t=config['label_t'],
                        transaction_costs=config['transaction_costs'],
                        rank_drop_gate=config['rank_drop_gate']
                    )
                elif rebalance_style == 'block':
                    sim_results = simulate_trading_strategy_block(
                        predictions_df=predictions_df,
                        stock_data_df=stock_data,
                        top_k=config['top_k'],
                        label_t=config['label_t'],
                        holding_period=holding_period,
                        transaction_costs=config['transaction_costs'],
                        rank_drop_gate=config['rank_drop_gate']
                    )
                else:
                    sim_results = simulate_trading_strategy_staggered(
                        predictions_df=predictions_df,
                        stock_data_df=stock_data,
                        top_k=config['top_k'],
                        label_t=config['label_t'],
                        holding_period=holding_period,
                        transaction_costs=config['transaction_costs'],
                        rank_drop_gate=config['rank_drop_gate']
                    )
                
                # Save comprehensive results
                save_backtest_results(
                    results,
                    backtest_dir,
                    sim_results=sim_results,
                    config=config,
                    num_tests=args.num_tests,
                    adjustment_method=args.adjustment_method
                )
                
                # Generate and save plot
                if args.plot or args.auto_save:
                    plot_file = os.path.join(backtest_dir, 'equity_curve.png')
                    plot_equity_curve(
                        args.predictions_dir,
                        stock_data,
                        config,
                        output_path=plot_file
                    )
                    print(f"  Equity curve: {plot_file}")
                    
            elif args.output:
                save_results(
                    results, 
                    args.output,
                    num_tests=args.num_tests,
                    adjustment_method=args.adjustment_method
                )
            
                if args.plot:
                    stock_data = load_stock_data(
                        config['data_file'],
                        config['test_start'],
                        config['test_end']
                    )
                    plot_output = args.output.replace('.csv', '_equity.png') if args.output else None
                    plot_equity_curve(args.predictions_dir, stock_data, config, plot_output)
    
    if tracking_manager.enabled:
        eval_results = None
        if args.multi_model and multi_results:
            eval_results = multi_results['averaged']
        elif not args.multi_model and results:
            eval_results = results
        if eval_results is not None:
            tracking_manager.log_metrics(eval_results, prefix="backtest.")
        if backtest_dir:
            _log_backtest_artifacts(tracking_manager, backtest_dir)
        tracking_manager.close()

    if logger:
        logger.info("Backtest evaluation completed successfully")


if __name__ == '__main__':
    main()
