"""
Test script to verify backtest fairness fixes.

This script creates synthetic data to test that:
1. Overnight returns are NOT included in portfolio returns
2. Only intraday returns are captured
3. The timing logic correctly maps predictions to next-day returns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def create_synthetic_test_data():
    """
    Create synthetic data with known overnight gaps and intraday returns.
    
    Returns:
        stock_data_df: DataFrame with OHLC data
        predictions_df: DataFrame with predictions
        expected_results: Dict with expected portfolio returns
    """
    dates = pd.date_range('2025-01-01', '2025-01-10', freq='B')  # Business days only
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    data = []
    
    # Create data with predictable patterns:
    # - Consistent overnight gaps: +1% for all stocks
    # - Variable intraday returns: +0.5% for top-ranked stocks, -0.5% for others
    
    prev_close = {stock: 100.0 for stock in stocks}
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        
        for stock in stocks:
            # Apply overnight gap (+1%)
            open_price = prev_close[stock] * 1.01
            
            # Intraday move depends on stock
            if stock in ['AAPL', 'MSFT']:  # Top stocks
                close_price = open_price * 1.005  # +0.5% intraday
            else:  # Other stocks
                close_price = open_price * 0.995  # -0.5% intraday
            
            high = max(open_price, close_price) * 1.002
            low = min(open_price, close_price) * 0.998
            
            data.append({
                'kdcode': stock,
                'dt': date_str,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': 1000000
            })
            
            prev_close[stock] = close_price
    
    stock_data_df = pd.DataFrame(data)
    
    # Create predictions that rank AAPL and MSFT highest
    pred_data = []
    for date in dates[:-1]:  # Predictions for all but last day
        date_str = date.strftime('%Y-%m-%d')
        for i, stock in enumerate(stocks):
            if stock in ['AAPL', 'MSFT']:
                score = 0.02  # High prediction
            else:
                score = -0.01  # Low prediction
            
            pred_data.append({
                'kdcode': stock,
                'dt': date_str,
                'score': score
            })
    
    predictions_df = pd.DataFrame(pred_data)
    
    # Expected results:
    # If we select top-2 stocks (AAPL, MSFT) based on predictions:
    # - Both have intraday return of +0.5% each day
    # - Average portfolio return per day = 0.5%
    # - We should NOT capture the +1% overnight gap
    
    expected_results = {
        'avg_daily_return': 0.005,  # 0.5% per day
        'num_days': len(dates) - 1,  # Can't trade on last prediction
        'should_not_include_overnight': 0.01  # Overnight gap we should NOT capture
    }
    
    return stock_data_df, predictions_df, expected_results


def test_return_calculation():
    """Test that calculate_forward_returns adds the correct columns."""
    print("=" * 70)
    print("TEST 1: Return Calculation")
    print("=" * 70)
    
    stock_data_df, predictions_df, expected = create_synthetic_test_data()
    
    # Import the function
    import sys
    sys.path.insert(0, '.')
    from evaluate_sp500 import calculate_forward_returns
    
    # Calculate returns
    stock_data_df = calculate_forward_returns(stock_data_df, label_t=5)
    
    # Check that required columns exist
    required_cols = ['tradeable_return', 'overnight_gap', 'next_day_return']
    for col in required_cols:
        assert col in stock_data_df.columns, f"Missing column: {col}"
        print(f"  ✓ Column '{col}' exists")
    
    # Verify tradeable_return calculation
    sample = stock_data_df[stock_data_df['kdcode'] == 'AAPL'].iloc[0]
    expected_tradeable = (sample['close'] - sample['open']) / sample['open']
    actual_tradeable = sample['tradeable_return']
    
    assert abs(expected_tradeable - actual_tradeable) < 1e-6, \
        f"Tradeable return mismatch: {expected_tradeable} != {actual_tradeable}"
    print(f"  ✓ Tradeable return calculation correct: {actual_tradeable:.6f}")
    
    # Verify overnight_gap calculation
    sample2 = stock_data_df[stock_data_df['kdcode'] == 'AAPL'].iloc[1]
    prev_close = stock_data_df[stock_data_df['kdcode'] == 'AAPL'].iloc[0]['close']
    expected_gap = (sample2['open'] - prev_close) / prev_close
    actual_gap = sample2['overnight_gap']
    
    assert abs(expected_gap - actual_gap) < 1e-6, \
        f"Overnight gap mismatch: {expected_gap} != {actual_gap}"
    print(f"  ✓ Overnight gap calculation correct: {actual_gap:.6f}")
    
    print("  ✓ TEST 1 PASSED\n")
    return stock_data_df, predictions_df, expected


def test_simulation_timing():
    """Test that simulation uses correct timing (next-day intraday returns)."""
    print("=" * 70)
    print("TEST 2: Simulation Timing")
    print("=" * 70)
    
    stock_data_df, predictions_df, expected = create_synthetic_test_data()
    
    # Import functions
    import sys
    sys.path.insert(0, '.')
    from evaluate_sp500 import calculate_forward_returns, simulate_trading_strategy
    
    # Prepare data
    stock_data_df = calculate_forward_returns(stock_data_df, label_t=5)
    
    # Run simulation
    try:
        sim_results = simulate_trading_strategy(
            predictions_df=predictions_df,
            stock_data_df=stock_data_df,
            top_k=2,
            label_t=5,
            transaction_costs=None
        )
        
        # Check results
        portfolio_returns = sim_results['portfolio_returns']
        
        print(f"  Number of trading days: {len(portfolio_returns)}")
        print(f"  Expected: {expected['num_days']}")
        
        # Calculate average return
        avg_return = np.mean(portfolio_returns)
        print(f"  Average daily return: {avg_return:.6f}")
        print(f"  Expected: {expected['avg_daily_return']:.6f}")
        
        # Verify we're getting intraday returns only
        tolerance = 0.001  # 0.1% tolerance
        assert abs(avg_return - expected['avg_daily_return']) < tolerance, \
            f"Average return mismatch! Got {avg_return}, expected {expected['avg_daily_return']}"
        
        print(f"  ✓ Returns match intraday expectation (within {tolerance*100}%)")
        
        # Verify we're NOT getting close-to-close returns
        close_to_close_return = expected['avg_daily_return'] + expected['should_not_include_overnight']
        if abs(avg_return - close_to_close_return) < tolerance:
            print(f"  ✗ ERROR: Still capturing overnight gaps!")
            print(f"     Got {avg_return}, but that includes overnight gap")
            return False
        else:
            print(f"  ✓ NOT capturing overnight gaps (correct!)")
        
        print("  ✓ TEST 2 PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ✗ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_prediction_date_mapping():
    """Test that predictions on day t use returns from day t+1."""
    print("=" * 70)
    print("TEST 3: Prediction-to-Return Date Mapping")
    print("=" * 70)
    
    # Create simple 3-day dataset
    dates = ['2025-01-01', '2025-01-02', '2025-01-03']
    
    data = []
    for i, date in enumerate(dates):
        for stock in ['AAPL', 'MSFT']:
            # Each day has a unique intraday return pattern we can identify
            intraday_return_pct = (i + 1) * 0.01  # Day 0: 1%, Day 1: 2%, Day 2: 3%
            
            open_price = 100.0
            close_price = open_price * (1 + intraday_return_pct)
            
            data.append({
                'kdcode': stock,
                'dt': date,
                'open': open_price,
                'high': close_price,
                'low': open_price,
                'close': close_price,
                'volume': 1000000
            })
    
    stock_data_df = pd.DataFrame(data)
    
    # Predictions on Jan 1 and Jan 2
    pred_data = [
        {'kdcode': 'AAPL', 'dt': '2025-01-01', 'score': 0.1},
        {'kdcode': 'MSFT', 'dt': '2025-01-01', 'score': 0.1},
        {'kdcode': 'AAPL', 'dt': '2025-01-02', 'score': 0.1},
        {'kdcode': 'MSFT', 'dt': '2025-01-02', 'score': 0.1},
    ]
    predictions_df = pd.DataFrame(pred_data)
    
    # Import functions
    import sys
    sys.path.insert(0, '.')
    from evaluate_sp500 import calculate_forward_returns, simulate_trading_strategy
    
    # Prepare data
    stock_data_df = calculate_forward_returns(stock_data_df, label_t=5)
    
    # Run simulation
    try:
        sim_results = simulate_trading_strategy(
            predictions_df=predictions_df,
            stock_data_df=stock_data_df,
            top_k=2,
            label_t=5,
            transaction_costs=None
        )
        
        portfolio_returns = sim_results['portfolio_returns']
        sim_dates = sim_results['dates']
        
        print(f"  Prediction dates: {predictions_df['dt'].unique().tolist()}")
        print(f"  Simulation dates (actual trading): {sim_dates}")
        
        # Prediction on Jan 1 should use Jan 2 returns (2%)
        # Prediction on Jan 2 should use Jan 3 returns (3%)
        
        if len(portfolio_returns) >= 2:
            print(f"  Return for Jan 2 (from Jan 1 prediction): {portfolio_returns[0]:.4f}")
            print(f"    Expected: 0.0200 (2%)")
            assert abs(portfolio_returns[0] - 0.02) < 0.0001, "Date mapping error!"
            
            print(f"  Return for Jan 3 (from Jan 2 prediction): {portfolio_returns[1]:.4f}")
            print(f"    Expected: 0.0300 (3%)")
            assert abs(portfolio_returns[1] - 0.03) < 0.0001, "Date mapping error!"
            
            print("  ✓ Date mapping correct: predictions use NEXT day returns")
        else:
            print(f"  ✗ ERROR: Expected 2 returns, got {len(portfolio_returns)}")
            return False
        
        print("  ✓ TEST 3 PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ✗ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BACKTEST FAIRNESS VERIFICATION TESTS")
    print("=" * 70 + "\n")
    
    results = []
    
    # Test 1: Return calculation
    try:
        test_return_calculation()
        results.append(("Return Calculation", True))
    except Exception as e:
        print(f"✗ TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        results.append(("Return Calculation", False))
    
    # Test 2: Simulation timing
    try:
        passed = test_simulation_timing()
        results.append(("Simulation Timing", passed))
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        results.append(("Simulation Timing", False))
    
    # Test 3: Date mapping
    try:
        passed = test_prediction_date_mapping()
        results.append(("Date Mapping", passed))
    except Exception as e:
        print(f"✗ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        results.append(("Date Mapping", False))
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:<30} {status}")
    
    all_passed = all(passed for _, passed in results)
    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Backtest fairness verified!")
    else:
        print("✗ SOME TESTS FAILED - Review issues above")
    print("=" * 70 + "\n")
    
    return all_passed


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
