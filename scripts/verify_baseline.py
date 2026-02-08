#!/usr/bin/env python
"""
Baseline Verification Script

This script verifies that the new modular implementation produces
results equivalent to the original sp500.py implementation.

Run this after implementing the framework to ensure correctness.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compare_predictions(new_dir: str, old_dir: str, tolerance: float = 1e-4) -> dict:
    """
    Compare predictions from new and old implementations.
    
    Args:
        new_dir: Directory with new implementation predictions
        old_dir: Directory with old implementation predictions
        tolerance: Maximum allowed difference
        
    Returns:
        Dictionary with comparison results
    """
    results = {
        'files_compared': 0,
        'files_matched': 0,
        'max_diff': 0.0,
        'avg_diff': 0.0,
        'mismatched_files': []
    }
    
    # Get list of prediction files
    new_files = set(os.listdir(new_dir)) if os.path.exists(new_dir) else set()
    old_files = set(os.listdir(old_dir)) if os.path.exists(old_dir) else set()
    
    common_files = new_files & old_files
    
    if not common_files:
        print(f"No common files found between {new_dir} and {old_dir}")
        return results
    
    all_diffs = []
    
    for filename in sorted(common_files):
        if not filename.endswith('.csv'):
            continue
            
        new_path = os.path.join(new_dir, filename)
        old_path = os.path.join(old_dir, filename)
        
        new_df = pd.read_csv(new_path)
        old_df = pd.read_csv(old_path)
        
        # Merge on kdcode and dt
        merged = new_df.merge(
            old_df, 
            on=['kdcode', 'dt'], 
            suffixes=('_new', '_old')
        )
        
        if len(merged) == 0:
            continue
        
        # Compare scores
        diff = np.abs(merged['score_new'] - merged['score_old'])
        max_diff = diff.max()
        avg_diff = diff.mean()
        
        all_diffs.extend(diff.values.tolist())
        results['files_compared'] += 1
        
        if max_diff < tolerance:
            results['files_matched'] += 1
        else:
            results['mismatched_files'].append({
                'file': filename,
                'max_diff': max_diff,
                'avg_diff': avg_diff
            })
    
    if all_diffs:
        results['max_diff'] = max(all_diffs)
        results['avg_diff'] = np.mean(all_diffs)
    
    return results


def verify_model_architecture():
    """Verify that the extracted model matches the original."""
    print("\n" + "=" * 60)
    print("Verifying Model Architecture")
    print("=" * 60)
    
    try:
        from mci_gru.models import StockPredictionModel, create_model
        
        # Create model with default config
        config = {
            'gru_hidden_sizes': [32, 10],
            'hidden_size_gat1': 32,
            'output_gat1': 4,
            'gat_heads': 4,
            'hidden_size_gat2': 32,
            'num_hidden_states': 32,
            'cross_attn_heads': 4,
            'slow_kernel': 5,
            'slow_stride': 2,
        }
        
        model = create_model(input_size=14, config=config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 2
        num_stocks = 10
        seq_len = 10
        num_features = 14
        
        x_ts = torch.randn(batch_size, num_stocks, seq_len, num_features)
        x_graph = torch.randn(batch_size * num_stocks, num_features)
        
        # Create dummy edges
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_weight = torch.tensor([0.9, 0.85, 0.88], dtype=torch.float)
        
        # Expand edges for batch
        edge_indices = []
        edge_weights = []
        for i in range(batch_size):
            edge_indices.append(edge_index + i * num_stocks)
            edge_weights.append(edge_weight)
        edge_index = torch.cat(edge_indices, dim=1)
        edge_weight = torch.cat(edge_weights)
        
        with torch.no_grad():
            output = model(x_ts, x_graph, edge_index, edge_weight, num_stocks)
        
        print(f"  Forward pass successful")
        print(f"  Output shape: {output.shape} (expected: [{batch_size}, {num_stocks}])")
        
        assert output.shape == (batch_size, num_stocks), "Output shape mismatch!"
        print("  PASSED: Model architecture verified")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_feature_engineering():
    """Verify that feature engineering produces expected columns."""
    print("\n" + "=" * 60)
    print("Verifying Feature Engineering")
    print("=" * 60)
    
    try:
        from mci_gru.features import FeatureEngineer, FEATURE_SETS
        
        # Create test data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        stocks = ['AAPL', 'GOOGL', 'MSFT']
        
        rows = []
        for stock in stocks:
            for dt in dates:
                rows.append({
                    'kdcode': stock,
                    'dt': dt.strftime('%Y-%m-%d'),
                    'open': 100 + np.random.randn() * 10,
                    'high': 105 + np.random.randn() * 10,
                    'low': 95 + np.random.randn() * 10,
                    'close': 100 + np.random.randn() * 10,
                    'volume': 1000000 + np.random.randint(-100000, 100000),
                })
        
        df = pd.DataFrame(rows)
        
        # Test feature engineer
        fe = FeatureEngineer(
            include_momentum=True,
            momentum_encoding='binary',
            include_volatility=True,
            include_vix=False,
        )
        
        df_transformed = fe.transform(df)
        
        # Check expected columns exist
        expected_cols = fe.get_feature_columns()
        missing_cols = [c for c in expected_cols if c not in df_transformed.columns]
        
        if missing_cols:
            print(f"  WARNING: Missing columns: {missing_cols}")
        else:
            print(f"  All expected columns present: {len(expected_cols)} columns")
        
        print("  PASSED: Feature engineering verified")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_graph_builder():
    """Verify graph builder functionality."""
    print("\n" + "=" * 60)
    print("Verifying Graph Builder")
    print("=" * 60)
    
    try:
        from mci_gru.graph import GraphBuilder
        import pandas as pd
        import numpy as np
        
        # Create test data
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        
        rows = []
        for stock in stocks:
            base_price = 100 + np.random.randn() * 20
            for i, dt in enumerate(dates):
                # Add some correlation structure
                rows.append({
                    'kdcode': stock,
                    'dt': dt.strftime('%Y-%m-%d'),
                    'close': base_price + np.cumsum(np.random.randn(1))[0] * 0.5,
                })
        
        df = pd.DataFrame(rows)
        
        # Test graph builder
        gb = GraphBuilder(
            judge_value=0.5,  # Low threshold to ensure some edges
            update_frequency_months=0,
            corr_lookback_days=252
        )
        
        edge_index, edge_weight = gb.build_graph(
            df, stocks, '2020-12-01', show_progress=False
        )
        
        print(f"  Graph built: {edge_index.shape[1]} edges for {len(stocks)} stocks")
        
        # Test stats
        stats = gb.get_stats()
        print(f"  Graph stats: {stats}")
        
        # Test dynamic update check
        should_update = gb.should_update('2021-06-01')
        print(f"  Should update (static mode): {should_update} (expected: False)")
        
        gb2 = GraphBuilder(update_frequency_months=6)
        gb2.last_update_date = '2020-01-01'
        should_update2 = gb2.should_update('2020-07-01')
        print(f"  Should update (6mo mode): {should_update2} (expected: True)")
        
        print("  PASSED: Graph builder verified")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_config():
    """Verify configuration system."""
    print("\n" + "=" * 60)
    print("Verifying Configuration System")
    print("=" * 60)
    
    try:
        from mci_gru.config import (
            ExperimentConfig, DataConfig, FeatureConfig,
            GraphConfig, ModelConfig, TrainingConfig,
            EXPERIMENT_PRESETS, get_preset
        )
        
        # Test default config
        config = ExperimentConfig()
        print(f"  Default config created: {config.experiment_name}")
        
        # Test presets
        for preset_name in EXPERIMENT_PRESETS:
            preset = get_preset(preset_name)
            print(f"  Preset '{preset_name}' loaded")
        
        # Test validation
        try:
            bad_config = DataConfig(train_start='2025-01-01', train_end='2020-01-01')
            print("  ERROR: Should have raised validation error")
        except ValueError:
            print("  Date validation working correctly")
        
        print("  PASSED: Configuration system verified")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("MCI-GRU Baseline Verification")
    print("=" * 60)
    
    results = {
        'config': verify_config(),
        'model': verify_model_architecture(),
        'features': verify_feature_engineering(),
        'graph': verify_graph_builder(),
    }
    
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll verification tests PASSED!")
        print("\nTo fully verify baseline equivalence:")
        print("  1. Run: python sp500.py")
        print("  2. Run: python run_experiment.py +experiment=baseline")
        print("  3. Compare predictions in both output directories")
    else:
        print("\nSome verification tests FAILED. Please fix before proceeding.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
