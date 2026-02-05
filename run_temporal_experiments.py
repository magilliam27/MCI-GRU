#!/usr/bin/env python
"""
Temporal Robustness Experiments Runner

Runs a series of time-shifted training experiments to validate model consistency
across different time periods and simulate incremental model updates.

Experiments:
- temporal_2017: Train 2017-2021, Val 2022, Test 2023
- temporal_2018: Train 2018-2022, Val 2023, Test 2024  
- temporal_2019: Train 2019-2023, Val 2024, Test 2025

Usage:
    # Run all temporal experiments
    python run_temporal_experiments.py
    
    # Custom output directory (e.g., for Google Colab)
    python run_temporal_experiments.py --output_dir /content/drive/MyDrive/MCI-GRU-Experiments
    
    # Run specific experiments only
    python run_temporal_experiments.py --experiments temporal_2017 temporal_2018
    
    # Skip training, only run backtests
    python run_temporal_experiments.py --backtest_only
    
    # Skip backtests, only run training
    python run_temporal_experiments.py --train_only
"""

import os
import sys
import glob
import json
import argparse
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple


# Temporal experiment configurations
TEMPORAL_EXPERIMENTS = {
    'temporal_2017': {
        'name': 'temporal_2017',
        'description': 'Train 2017-2021, Val 2022, Test 2023',
        'train_start': '2017-01-01',
        'train_end': '2021-12-31',
        'val_start': '2022-01-01',
        'val_end': '2022-12-31',
        'test_start': '2023-01-01',
        'test_end': '2023-12-31',
    },
    'temporal_2018': {
        'name': 'temporal_2018',
        'description': 'Train 2018-2022, Val 2023, Test 2024',
        'train_start': '2018-01-01',
        'train_end': '2022-12-31',
        'val_start': '2023-01-01',
        'val_end': '2023-12-31',
        'test_start': '2024-01-01',
        'test_end': '2024-12-31',
    },
    'temporal_2019': {
        'name': 'temporal_2019',
        'description': 'Train 2019-2023, Val 2024, Test 2025',
        'train_start': '2019-01-01',
        'train_end': '2023-12-31',
        'val_start': '2024-01-01',
        'val_end': '2024-12-31',
        'test_start': '2025-01-01',
        'test_end': '2025-12-31',
    },
}


def log_message(message: str, log_file: Optional[str] = None):
    """Print message and optionally write to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted + '\n')


def find_latest_run(base_dir: str, experiment_name: str) -> Optional[str]:
    """Find the most recent run directory for an experiment."""
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # Look for timestamped directories (format: YYYYMMDD_HHMMSS)
    run_dirs = sorted(glob.glob(os.path.join(experiment_dir, '*/')))
    
    if run_dirs:
        return run_dirs[-1].rstrip(os.sep)
    elif os.path.exists(experiment_dir):
        return experiment_dir
    
    return None


def run_training(
    experiment_name: str,
    output_dir: str,
    num_epochs: int = 100,
    num_models: int = 10,
    log_file: Optional[str] = None
) -> Tuple[bool, Optional[str], float]:
    """
    Run training for a single temporal experiment.
    
    Returns:
        Tuple of (success, run_directory, elapsed_time_minutes)
    """
    log_message(f"Starting training: {experiment_name}", log_file)
    log_message(f"  Config: +data={experiment_name}", log_file)
    
    start_time = time.time()
    
    # Build command
    cmd = [
        sys.executable,
        'run_experiment.py',
        f'output_dir={output_dir}',
        f'data={experiment_name}',
        f'experiment_name={experiment_name}',
        f'training.num_epochs={num_epochs}',
        f'training.num_models={num_models}',
    ]
    
    log_message(f"  Command: {' '.join(cmd)}", log_file)
    
    try:
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        
        elapsed = (time.time() - start_time) / 60
        run_dir = find_latest_run(output_dir, experiment_name)
        
        log_message(f"  Training completed in {elapsed:.1f} minutes", log_file)
        log_message(f"  Output directory: {run_dir}", log_file)
        
        return True, run_dir, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = (time.time() - start_time) / 60
        log_message(f"  Training FAILED after {elapsed:.1f} minutes", log_file)
        log_message(f"  Error: {e}", log_file)
        return False, None, elapsed
    except Exception as e:
        elapsed = (time.time() - start_time) / 60
        log_message(f"  Training FAILED with exception: {e}", log_file)
        return False, None, elapsed


def run_backtest(
    predictions_dir: str,
    experiment_config: Dict,
    log_file: Optional[str] = None
) -> Tuple[bool, float]:
    """
    Run backtesting for a completed experiment.
    
    Returns:
        Tuple of (success, elapsed_time_minutes)
    """
    experiment_name = experiment_config['name']
    log_message(f"Starting backtest: {experiment_name}", log_file)
    log_message(f"  Predictions: {predictions_dir}", log_file)
    
    start_time = time.time()
    
    # Build command with date overrides for correct test period
    cmd = [
        sys.executable,
        'evaluate_sp500.py',
        '--predictions_dir', predictions_dir,
        '--test_start', experiment_config['test_start'],
        '--test_end', experiment_config['test_end'],
        '--auto_save',
        '--plot',
    ]
    
    log_message(f"  Command: {' '.join(cmd)}", log_file)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed = (time.time() - start_time) / 60
        log_message(f"  Backtest completed in {elapsed:.1f} minutes", log_file)
        
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = (time.time() - start_time) / 60
        log_message(f"  Backtest FAILED after {elapsed:.1f} minutes", log_file)
        log_message(f"  Error: {e}", log_file)
        return False, elapsed
    except Exception as e:
        elapsed = (time.time() - start_time) / 60
        log_message(f"  Backtest FAILED with exception: {e}", log_file)
        return False, elapsed


def run_all_experiments(
    output_dir: str,
    experiments: List[str],
    num_epochs: int = 100,
    num_models: int = 10,
    train_only: bool = False,
    backtest_only: bool = False,
) -> Dict:
    """
    Run all specified temporal experiments.
    
    Returns:
        Summary dictionary with results for each experiment
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'temporal_experiments_{timestamp}.log')
    
    log_message("=" * 80, log_file)
    log_message("TEMPORAL ROBUSTNESS EXPERIMENTS", log_file)
    log_message("=" * 80, log_file)
    log_message(f"Output directory: {output_dir}", log_file)
    log_message(f"Experiments to run: {experiments}", log_file)
    log_message(f"Training epochs: {num_epochs}", log_file)
    log_message(f"Number of models: {num_models}", log_file)
    log_message(f"Train only: {train_only}", log_file)
    log_message(f"Backtest only: {backtest_only}", log_file)
    log_message("=" * 80, log_file)
    
    results = {}
    total_start = time.time()
    
    for i, exp_name in enumerate(experiments, 1):
        log_message("", log_file)
        log_message(f"[{i}/{len(experiments)}] Processing {exp_name}", log_file)
        log_message("-" * 60, log_file)
        
        if exp_name not in TEMPORAL_EXPERIMENTS:
            log_message(f"  Unknown experiment: {exp_name}, skipping", log_file)
            results[exp_name] = {'status': 'skipped', 'error': 'Unknown experiment'}
            continue
        
        exp_config = TEMPORAL_EXPERIMENTS[exp_name]
        log_message(f"  {exp_config['description']}", log_file)
        
        exp_result = {
            'config': exp_config,
            'training': None,
            'backtest': None,
        }
        
        # Training phase
        if not backtest_only:
            train_success, run_dir, train_time = run_training(
                experiment_name=exp_name,
                output_dir=output_dir,
                num_epochs=num_epochs,
                num_models=num_models,
                log_file=log_file
            )
            
            exp_result['training'] = {
                'success': train_success,
                'run_dir': run_dir,
                'elapsed_minutes': train_time,
            }
            
            if not train_success:
                log_message(f"  Skipping backtest due to training failure", log_file)
                results[exp_name] = exp_result
                continue
        else:
            # Find existing run for backtest-only mode
            run_dir = find_latest_run(output_dir, exp_name)
            if not run_dir:
                log_message(f"  No existing run found for {exp_name}, skipping", log_file)
                exp_result['training'] = {'success': False, 'error': 'No existing run found'}
                results[exp_name] = exp_result
                continue
            
            log_message(f"  Using existing run: {run_dir}", log_file)
            exp_result['training'] = {'success': True, 'run_dir': run_dir}
        
        # Backtest phase
        if not train_only:
            # Find predictions directory
            predictions_dir = os.path.join(run_dir, 'averaged_predictions')
            
            if not os.path.exists(predictions_dir):
                log_message(f"  Predictions not found: {predictions_dir}", log_file)
                exp_result['backtest'] = {'success': False, 'error': 'Predictions not found'}
                results[exp_name] = exp_result
                continue
            
            backtest_success, backtest_time = run_backtest(
                predictions_dir=predictions_dir,
                experiment_config=exp_config,
                log_file=log_file
            )
            
            exp_result['backtest'] = {
                'success': backtest_success,
                'elapsed_minutes': backtest_time,
            }
        
        results[exp_name] = exp_result
    
    # Summary
    total_elapsed = (time.time() - total_start) / 60
    
    log_message("", log_file)
    log_message("=" * 80, log_file)
    log_message("SUMMARY", log_file)
    log_message("=" * 80, log_file)
    log_message(f"Total elapsed time: {total_elapsed:.1f} minutes", log_file)
    log_message("", log_file)
    
    for exp_name, result in results.items():
        train_status = "N/A"
        backtest_status = "N/A"
        
        if result.get('training'):
            train_status = "OK" if result['training'].get('success') else "FAILED"
        if result.get('backtest'):
            backtest_status = "OK" if result['backtest'].get('success') else "FAILED"
        
        log_message(f"  {exp_name}: Training={train_status}, Backtest={backtest_status}", log_file)
    
    log_message("", log_file)
    log_message(f"Log file: {log_file}", log_file)
    log_message("=" * 80, log_file)
    
    # Save results summary as JSON
    summary_file = os.path.join(output_dir, f'temporal_experiments_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_elapsed_minutes': total_elapsed,
            'experiments': experiments,
            'results': results,
        }, f, indent=2, default=str)
    
    log_message(f"Summary saved to: {summary_file}", log_file)
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run temporal robustness experiments for MCI-GRU'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Base output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        type=str,
        default=list(TEMPORAL_EXPERIMENTS.keys()),
        choices=list(TEMPORAL_EXPERIMENTS.keys()),
        help='Experiments to run (default: all)'
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--num_models',
        type=int,
        default=10,
        help='Number of models to train per experiment (default: 10)'
    )
    
    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Only run training, skip backtests'
    )
    
    parser.add_argument(
        '--backtest_only',
        action='store_true',
        help='Only run backtests on existing runs, skip training'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.train_only and args.backtest_only:
        print("Error: Cannot specify both --train_only and --backtest_only")
        sys.exit(1)
    
    results = run_all_experiments(
        output_dir=args.output_dir,
        experiments=args.experiments,
        num_epochs=args.num_epochs,
        num_models=args.num_models,
        train_only=args.train_only,
        backtest_only=args.backtest_only,
    )
    
    # Exit with error code if any experiment failed
    all_success = all(
        (result.get('training', {}).get('success', True) and 
         result.get('backtest', {}).get('success', True))
        for result in results.values()
    )
    
    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()
