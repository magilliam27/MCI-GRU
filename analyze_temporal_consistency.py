#!/usr/bin/env python
"""
Temporal Consistency Analysis

Aggregates results from temporal experiments and generates:
- Comparison tables across time periods
- Statistical consistency metrics
- Visualizations for presentation
- Summary report for professor meeting

Usage:
    # Analyze experiments in results directory
    python analyze_temporal_consistency.py
    
    # Custom results directory (e.g., for Google Colab)
    python analyze_temporal_consistency.py --results_dir /content/drive/MyDrive/MCI-GRU-Experiments
    
    # Custom output directory for analysis
    python analyze_temporal_consistency.py --output_dir results/temporal_analysis
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")


# Temporal experiment metadata
TEMPORAL_EXPERIMENTS = {
    'temporal_2017': {
        'name': 'temporal_2017',
        'label': '2017-2021 Train',
        'test_year': 2023,
        'train_period': '2017-2021',
        'test_period': '2023',
    },
    'temporal_2018': {
        'name': 'temporal_2018',
        'label': '2018-2022 Train',
        'test_year': 2024,
        'train_period': '2018-2022',
        'test_period': '2024',
    },
    'temporal_2019': {
        'name': 'temporal_2019',
        'label': '2019-2023 Train',
        'test_year': 2025,
        'train_period': '2019-2023',
        'test_period': '2025',
    },
}


def find_latest_run(base_dir: str, experiment_name: str) -> Optional[str]:
    """Find the most recent run directory for an experiment."""
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # Look for timestamped directories
    run_dirs = sorted(glob.glob(os.path.join(experiment_dir, '*/')))
    
    if run_dirs:
        return run_dirs[-1].rstrip(os.sep)
    elif os.path.exists(experiment_dir):
        return experiment_dir
    
    return None


def load_backtest_results(backtest_dir: str) -> Optional[Dict]:
    """Load backtest results from a directory."""
    results_file = os.path.join(backtest_dir, 'backtest_results.csv')
    metrics_file = os.path.join(backtest_dir, 'backtest_metrics.json')
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    elif os.path.exists(results_file):
        df = pd.read_csv(results_file)
        return df.iloc[0].to_dict() if len(df) > 0 else None
    
    return None


def load_daily_returns(backtest_dir: str) -> Optional[pd.DataFrame]:
    """Load daily returns from backtest output."""
    returns_file = os.path.join(backtest_dir, 'daily_returns.csv')
    
    if os.path.exists(returns_file):
        df = pd.read_csv(returns_file)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    return None


def load_all_experiment_results(results_dir: str) -> Dict[str, Dict]:
    """Load results from all temporal experiments."""
    all_results = {}
    
    for exp_name, exp_meta in TEMPORAL_EXPERIMENTS.items():
        run_dir = find_latest_run(results_dir, exp_name)
        
        if not run_dir:
            print(f"  Warning: No run found for {exp_name}")
            continue
        
        # Check for backtest directory
        backtest_dir = os.path.join(run_dir, 'backtest')
        if not os.path.exists(backtest_dir):
            print(f"  Warning: No backtest found for {exp_name}")
            continue
        
        # Load results
        metrics = load_backtest_results(backtest_dir)
        daily_returns = load_daily_returns(backtest_dir)
        
        if metrics:
            all_results[exp_name] = {
                'metadata': exp_meta,
                'metrics': metrics,
                'daily_returns': daily_returns,
                'run_dir': run_dir,
                'backtest_dir': backtest_dir,
            }
            print(f"  Loaded {exp_name}: ARR={metrics.get('ARR', 0):.4f}, ASR={metrics.get('ASR', 0):.4f}")
        else:
            print(f"  Warning: Could not load metrics for {exp_name}")
    
    return all_results


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table of key metrics across experiments."""
    rows = []
    
    for exp_name, data in results.items():
        metrics = data['metrics']
        meta = data['metadata']
        
        row = {
            'Experiment': exp_name,
            'Train Period': meta['train_period'],
            'Test Period': meta['test_period'],
            'Test Year': meta['test_year'],
            'ARR': metrics.get('ARR', 0),
            'ARR (%)': metrics.get('ARR', 0) * 100,
            'AVoL': metrics.get('AVoL', 0),
            'AVoL (%)': metrics.get('AVoL', 0) * 100,
            'MDD': metrics.get('MDD', 0),
            'MDD (%)': metrics.get('MDD', 0) * 100,
            'ASR': metrics.get('ASR', 0),
            'CR': metrics.get('CR', 0),
            'IR': metrics.get('IR', 0),
            'MSE': metrics.get('MSE', 0),
            'MAE': metrics.get('MAE', 0),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Test Year')
    
    return df


def calculate_consistency_metrics(comparison_df: pd.DataFrame) -> Dict:
    """Calculate statistical consistency metrics across periods."""
    metrics = {}
    
    # Key columns to analyze
    key_cols = ['ARR', 'ASR', 'MDD', 'IR', 'CR']
    
    for col in key_cols:
        if col in comparison_df.columns:
            values = comparison_df[col].values
            metrics[col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'cv': float(np.std(values) / np.abs(np.mean(values))) if np.mean(values) != 0 else 0,
                'range': float(np.max(values) - np.min(values)),
            }
    
    # Overall consistency score (lower CV = more consistent)
    arr_cv = metrics.get('ARR', {}).get('cv', 0)
    asr_cv = metrics.get('ASR', {}).get('cv', 0)
    metrics['overall_consistency'] = {
        'arr_cv': arr_cv,
        'asr_cv': asr_cv,
        'is_consistent': arr_cv < 0.5 and asr_cv < 0.5,  # CV < 50% threshold
    }
    
    return metrics


def generate_comparison_plot(
    comparison_df: pd.DataFrame,
    output_path: str
):
    """Generate bar charts comparing key metrics across periods."""
    if not HAS_MATPLOTLIB:
        print("  Skipping plot (matplotlib not available)")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    experiments = comparison_df['Test Period'].values
    x = np.arange(len(experiments))
    width = 0.6
    
    # ARR
    ax = axes[0, 0]
    values = comparison_df['ARR (%)'].values
    colors = ['green' if v > 0 else 'red' for v in values]
    ax.bar(x, values, width, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=np.mean(values), color='blue', linestyle='--', linewidth=1, label=f'Mean: {np.mean(values):.1f}%')
    ax.set_ylabel('ARR (%)')
    ax.set_title('Annualized Return by Test Period')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    
    # ASR (Sharpe)
    ax = axes[0, 1]
    values = comparison_df['ASR'].values
    colors = ['green' if v > 0 else 'red' for v in values]
    ax.bar(x, values, width, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=2.0, color='orange', linestyle='--', linewidth=1, label='Target: 2.0')
    ax.axhline(y=np.mean(values), color='blue', linestyle='--', linewidth=1, label=f'Mean: {np.mean(values):.2f}')
    ax.set_ylabel('ASR')
    ax.set_title('Annualized Sharpe Ratio by Test Period')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    
    # MDD
    ax = axes[0, 2]
    values = comparison_df['MDD (%)'].values
    ax.bar(x, values, width, color='red', alpha=0.7)
    ax.axhline(y=np.mean(values), color='blue', linestyle='--', linewidth=1, label=f'Mean: {np.mean(values):.1f}%')
    ax.set_ylabel('MDD (%)')
    ax.set_title('Maximum Drawdown by Test Period')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    
    # IR
    ax = axes[1, 0]
    values = comparison_df['IR'].values
    colors = ['green' if v > 0 else 'red' for v in values]
    ax.bar(x, values, width, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=np.mean(values), color='blue', linestyle='--', linewidth=1, label=f'Mean: {np.mean(values):.2f}')
    ax.set_ylabel('IR')
    ax.set_title('Information Ratio by Test Period')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    
    # CR (Calmar)
    ax = axes[1, 1]
    values = comparison_df['CR'].values
    colors = ['green' if v > 0 else 'red' for v in values]
    ax.bar(x, values, width, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=np.mean(values), color='blue', linestyle='--', linewidth=1, label=f'Mean: {np.mean(values):.2f}')
    ax.set_ylabel('CR')
    ax.set_title('Calmar Ratio by Test Period')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    
    # MSE/MAE
    ax = axes[1, 2]
    mse_values = comparison_df['MSE'].values * 1000  # Scale for visibility
    mae_values = comparison_df['MAE'].values * 1000
    x_offset = 0.2
    ax.bar(x - x_offset, mse_values, width/2, label='MSE (x1000)', alpha=0.7)
    ax.bar(x + x_offset, mae_values, width/2, label='MAE (x1000)', alpha=0.7)
    ax.set_ylabel('Error (x1000)')
    ax.set_title('Prediction Error by Test Period')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    
    plt.suptitle('Temporal Robustness Analysis: MCI-GRU Performance Across Test Periods', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def generate_equity_curves_plot(
    results: Dict[str, Dict],
    output_path: str
):
    """Generate combined equity curves for all periods."""
    if not HAS_MATPLOTLIB:
        print("  Skipping plot (matplotlib not available)")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, (exp_name, data) in enumerate(sorted(results.items())):
        daily_returns = data.get('daily_returns')
        if daily_returns is None:
            continue
        
        meta = data['metadata']
        
        # Calculate cumulative returns
        daily_returns = daily_returns.sort_values('date')
        cum_returns = (1 + daily_returns['portfolio_return']).cumprod()
        
        label = f"{meta['test_period']} ({meta['train_period']} train)"
        ax.plot(daily_returns['date'], cum_returns, 
                label=label, linewidth=2, color=colors[i % len(colors)])
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title('Equity Curves Across All Test Periods', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def generate_consistency_report(
    comparison_df: pd.DataFrame,
    consistency_metrics: Dict,
    output_path: str
):
    """Generate markdown consistency report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'w') as f:
        f.write("# Temporal Robustness Analysis Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        
        f.write("## Executive Summary\n\n")
        
        # Calculate summary stats
        arr_mean = comparison_df['ARR (%)'].mean()
        arr_std = comparison_df['ARR (%)'].std()
        asr_mean = comparison_df['ASR'].mean()
        asr_std = comparison_df['ASR'].std()
        
        consistency = consistency_metrics.get('overall_consistency', {})
        is_consistent = consistency.get('is_consistent', False)
        
        f.write(f"The MCI-GRU model was evaluated across **{len(comparison_df)} independent test periods** ")
        f.write(f"({', '.join(comparison_df['Test Period'].values)}) to assess temporal robustness.\n\n")
        
        f.write("**Key Findings:**\n\n")
        f.write(f"- **Average Annualized Return:** {arr_mean:.2f}% (std: {arr_std:.2f}%)\n")
        f.write(f"- **Average Sharpe Ratio:** {asr_mean:.2f} (std: {asr_std:.2f})\n")
        f.write(f"- **Consistency Assessment:** {'CONSISTENT' if is_consistent else 'VARIABLE'} ")
        f.write(f"(ARR CV: {consistency.get('arr_cv', 0)*100:.1f}%, ASR CV: {consistency.get('asr_cv', 0)*100:.1f}%)\n\n")
        
        f.write("## Performance Comparison\n\n")
        
        # Create markdown table
        f.write("| Test Period | Train Period | ARR (%) | ASR | MDD (%) | IR | CR |\n")
        f.write("|-------------|--------------|---------|-----|---------|----|----|")
        f.write("\n")
        
        for _, row in comparison_df.iterrows():
            f.write(f"| {row['Test Period']} | {row['Train Period']} | ")
            f.write(f"{row['ARR (%)']:.2f} | {row['ASR']:.2f} | ")
            f.write(f"{row['MDD (%)']:.2f} | {row['IR']:.2f} | {row['CR']:.2f} |\n")
        
        f.write("\n")
        
        # Statistics
        f.write("### Summary Statistics\n\n")
        f.write("| Metric | Mean | Std Dev | Min | Max | CV |\n")
        f.write("|--------|------|---------|-----|-----|----|\n")
        
        for metric_name in ['ARR', 'ASR', 'MDD', 'IR', 'CR']:
            if metric_name in consistency_metrics:
                m = consistency_metrics[metric_name]
                scale = 100 if metric_name in ['ARR', 'MDD'] else 1
                f.write(f"| {metric_name} | {m['mean']*scale:.2f} | {m['std']*scale:.2f} | ")
                f.write(f"{m['min']*scale:.2f} | {m['max']*scale:.2f} | {m['cv']*100:.1f}% |\n")
        
        f.write("\n")
        
        f.write("## Interpretation\n\n")
        
        if is_consistent:
            f.write("The model demonstrates **temporal consistency**, with coefficient of variation ")
            f.write("below 50% for both returns and risk-adjusted metrics. This suggests:\n\n")
            f.write("1. The learned patterns generalize across different market regimes\n")
            f.write("2. The model is not overfit to a specific time period\n")
            f.write("3. Annual retraining can maintain predictive power\n\n")
        else:
            f.write("The model shows **variable performance** across time periods, with ")
            f.write("coefficient of variation above 50%. This may indicate:\n\n")
            f.write("1. Market regime sensitivity in the learned patterns\n")
            f.write("2. Potential overfitting to training period characteristics\n")
            f.write("3. Need for more robust feature engineering or regularization\n\n")
        
        f.write("## Methodology\n\n")
        f.write("- **Training Setup:** 5-year training, 1-year validation, 1-year test\n")
        f.write("- **Models per Period:** 10 (ensemble averaged)\n")
        f.write("- **Portfolio Strategy:** Top-10 stocks, equal-weighted, daily rebalancing\n")
        f.write("- **Benchmark:** S&P 500 index\n")
        f.write("- **Metrics:** Paper-standard (ARR, AVoL, MDD, ASR, CR, IR, MSE, MAE)\n\n")
        
        f.write("---\n")
        f.write(f"*Report generated by analyze_temporal_consistency.py*\n")
    
    print(f"  Saved: {output_path}")


def generate_summary_text(
    comparison_df: pd.DataFrame,
    consistency_metrics: Dict,
    output_path: str
):
    """Generate plain text summary for quick reference."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEMPORAL ROBUSTNESS ANALYSIS - SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {timestamp}\n\n")
        
        f.write("PERFORMANCE BY TEST PERIOD:\n")
        f.write("-" * 80 + "\n")
        
        for _, row in comparison_df.iterrows():
            f.write(f"\n{row['Test Period']} (trained on {row['Train Period']}):\n")
            f.write(f"  Annualized Return:  {row['ARR (%)']:7.2f}%\n")
            f.write(f"  Sharpe Ratio:       {row['ASR']:7.2f}\n")
            f.write(f"  Max Drawdown:       {row['MDD (%)']:7.2f}%\n")
            f.write(f"  Information Ratio:  {row['IR']:7.2f}\n")
            f.write(f"  Calmar Ratio:       {row['CR']:7.2f}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("CONSISTENCY ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        
        arr_mean = comparison_df['ARR (%)'].mean()
        arr_std = comparison_df['ARR (%)'].std()
        asr_mean = comparison_df['ASR'].mean()
        asr_std = comparison_df['ASR'].std()
        
        f.write(f"\nAverage ARR:    {arr_mean:7.2f}% (std: {arr_std:.2f}%)\n")
        f.write(f"Average Sharpe: {asr_mean:7.2f}   (std: {asr_std:.2f})\n")
        
        consistency = consistency_metrics.get('overall_consistency', {})
        f.write(f"\nARR Coefficient of Variation:   {consistency.get('arr_cv', 0)*100:.1f}%\n")
        f.write(f"Sharpe Coefficient of Variation: {consistency.get('asr_cv', 0)*100:.1f}%\n")
        
        is_consistent = consistency.get('is_consistent', False)
        f.write(f"\nOverall Assessment: {'CONSISTENT' if is_consistent else 'VARIABLE'}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"  Saved: {output_path}")


def run_analysis(
    results_dir: str,
    output_dir: str
) -> bool:
    """
    Run the full temporal consistency analysis.
    
    Returns:
        True if successful, False otherwise
    """
    print("=" * 80)
    print("TEMPORAL CONSISTENCY ANALYSIS")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all experiment results
    print("\nLoading experiment results...")
    results = load_all_experiment_results(results_dir)
    
    if len(results) == 0:
        print("\nError: No experiment results found!")
        print("Make sure experiments have been run and backtested.")
        return False
    
    print(f"\nLoaded {len(results)} experiments")
    
    # Create comparison table
    print("\nCreating comparison table...")
    comparison_df = create_comparison_table(results)
    
    # Save comparison table
    comparison_csv = os.path.join(output_dir, 'comparison_table.csv')
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"  Saved: {comparison_csv}")
    
    # Calculate consistency metrics
    print("\nCalculating consistency metrics...")
    consistency_metrics = calculate_consistency_metrics(comparison_df)
    
    # Save consistency metrics
    consistency_json = os.path.join(output_dir, 'consistency_metrics.json')
    with open(consistency_json, 'w') as f:
        json.dump(consistency_metrics, f, indent=2)
    print(f"  Saved: {consistency_json}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    metrics_plot = os.path.join(output_dir, 'metrics_comparison.png')
    generate_comparison_plot(comparison_df, metrics_plot)
    
    equity_plot = os.path.join(output_dir, 'equity_curves_all_periods.png')
    generate_equity_curves_plot(results, equity_plot)
    
    # Generate reports
    print("\nGenerating reports...")
    
    report_md = os.path.join(output_dir, 'consistency_report.md')
    generate_consistency_report(comparison_df, consistency_metrics, report_md)
    
    summary_txt = os.path.join(output_dir, 'temporal_robustness_summary.txt')
    generate_summary_text(comparison_df, consistency_metrics, summary_txt)
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Results:")
    print(f"  Experiments analyzed: {len(results)}")
    print(f"  Average ARR: {comparison_df['ARR (%)'].mean():.2f}%")
    print(f"  Average Sharpe: {comparison_df['ASR'].mean():.2f}")
    
    consistency = consistency_metrics.get('overall_consistency', {})
    print(f"  Consistency: {'CONSISTENT' if consistency.get('is_consistent', False) else 'VARIABLE'}")
    
    print(f"\nOutputs saved to: {output_dir}")
    print("=" * 80)
    
    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze temporal consistency across MCI-GRU experiments'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Base directory containing experiment results (default: results)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for analysis (default: results_dir/temporal_analysis)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.results_dir, 'temporal_analysis')
    
    success = run_analysis(
        results_dir=args.results_dir,
        output_dir=output_dir
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
