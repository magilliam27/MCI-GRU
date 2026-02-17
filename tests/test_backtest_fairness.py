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
    
    bp = _import_backtest()
    # Calculate returns
    stock_data_df = bp.calculate_forward_returns(stock_data_df, label_t=5)
    
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
    
    bp = _import_backtest()
    # Prepare data
    stock_data_df = bp.calculate_forward_returns(stock_data_df, label_t=5)
    
    # Run simulation
    try:
        sim_results = bp.simulate_trading_strategy(
            predictions_df=predictions_df,
            stock_data_df=stock_data_df,
            top_k=2,
            label_t=5,
            transaction_costs=None,
            rank_drop_gate=None
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
    
    bp = _import_backtest()
    # Prepare data
    stock_data_df = bp.calculate_forward_returns(stock_data_df, label_t=5)
    
    # Run simulation
    try:
        sim_results = bp.simulate_trading_strategy(
            predictions_df=predictions_df,
            stock_data_df=stock_data_df,
            top_k=2,
            label_t=5,
            transaction_costs=None,
            rank_drop_gate=None
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


def _import_backtest():
    """Import backtest module from tests directory (file is backtest_sp500.py)."""
    _tests_dir = os.path.dirname(os.path.abspath(__file__))
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)
    import backtest_sp500 as bp
    return bp


def test_rank_drop_gate_eligible():
    """With rank-drop gate: stock that fell >= 10 ranks is eligible and can be selected."""
    print("=" * 70)
    print("TEST 4: Rank-Drop Gate - Eligible (rank fell >= 10)")
    print("=" * 70)
    bp = _import_backtest()
    # 3 dates, 15 stocks. Day1: S0=rank1. Day2: S0=rank11 (10 stocks with higher score) -> rank_drop=10
    dates = ['2025-01-01', '2025-01-02', '2025-01-03']
    stocks = [f'S{i}' for i in range(15)]
    data = []
    for dt in dates:
        for s in stocks:
            data.append({
                'kdcode': s, 'dt': dt, 'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 1e6
            })
    stock_df = pd.DataFrame(data)
    stock_df = bp.calculate_forward_returns(stock_df, label_t=5)
    preds = []
    for dt in dates:
        for r, s in enumerate(stocks):
            if dt == '2025-01-01':
                score = 1.0 - r * 0.01  # S0=1, S1=0.99, ... S14=0.86
            elif dt == '2025-01-02':
                # S0 at rank 11: give S1..S10 scores above S0, S0=0.0, rest below
                if s == 'S0':
                    score = 0.0
                elif stocks.index(s) < 10:
                    score = 0.5 - stocks.index(s) * 0.01  # S1=0.49 .. S10=0.41
                else:
                    score = -0.1 - stocks.index(s) * 0.01
            else:
                score = 1.0 - r * 0.01
            preds.append({'kdcode': s, 'dt': dt, 'score': score})
    pred_df = pd.DataFrame(preds)
    gate = {'enabled': True, 'min_rank_drop': 10}
    sim = bp.simulate_trading_strategy(
        pred_df, stock_df, top_k=2, label_t=5, transaction_costs=None, rank_drop_gate=gate
    )
    assert sim['rank_gate_enabled'] is True
    assert len(sim['portfolio_returns']) >= 1, "Should have at least one trading day when gate allows"
    print("  ✓ Rank-drop gate enabled; simulation produced trading days when stock had rank drop >= 10")
    print("  ✓ TEST 4 PASSED\n")
    return True


def test_rank_drop_gate_excluded():
    """With rank-drop gate: when no stock fell >= 10 ranks, day is skipped."""
    print("=" * 70)
    print("TEST 5: Rank-Drop Gate - Excluded (rank fell < 10)")
    print("=" * 70)
    bp = _import_backtest()
    dates = ['2025-01-01', '2025-01-02', '2025-01-03']
    stocks = [f'S{i}' for i in range(12)]
    data = []
    for dt in dates:
        for s in stocks:
            data.append({
                'kdcode': s, 'dt': dt, 'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 1e6
            })
    stock_df = pd.DataFrame(data)
    stock_df = bp.calculate_forward_returns(stock_df, label_t=5)
    # Same rank order both days -> rank_drop=0 for all; no one eligible on day2
    preds = []
    for dt in dates:
        for r, s in enumerate(stocks):
            score = 1.0 - r * 0.01
            preds.append({'kdcode': s, 'dt': dt, 'score': score})
    pred_df = pd.DataFrame(preds)
    gate = {'enabled': True, 'min_rank_drop': 10}
    sim = bp.simulate_trading_strategy(
        pred_df, stock_df, top_k=2, label_t=5, transaction_costs=None, rank_drop_gate=gate
    )
    assert sim['rank_gate_enabled'] is True
    assert sim['days_skipped_by_rank_gate'] >= 1, "Should skip at least one day when no stock has rank drop >= 10"
    print("  ✓ Day skipped when no stock had rank drop >= 10")
    print("  ✓ TEST 5 PASSED\n")
    return True


def test_rank_drop_gate_disabled_regression():
    """With rank-drop gate disabled, behavior matches no-gate (same number of trading days as without gate)."""
    print("=" * 70)
    print("TEST 6: Rank-Drop Gate Disabled - Regression")
    print("=" * 70)
    bp = _import_backtest()
    stock_data_df, predictions_df, _ = create_synthetic_test_data()
    stock_data_df = bp.calculate_forward_returns(stock_data_df, label_t=5)
    sim_no_gate = bp.simulate_trading_strategy(
        predictions_df, stock_data_df, top_k=2, label_t=5, transaction_costs=None, rank_drop_gate=None
    )
    sim_gate_off = bp.simulate_trading_strategy(
        predictions_df, stock_data_df, top_k=2, label_t=5, transaction_costs=None,
        rank_drop_gate={'enabled': False, 'min_rank_drop': 10}
    )
    assert sim_no_gate['rank_gate_enabled'] is False
    assert sim_gate_off['rank_gate_enabled'] is False
    assert len(sim_no_gate['portfolio_returns']) == len(sim_gate_off['portfolio_returns']), \
        "Disabled gate should give same number of trading days as no gate"
    np.testing.assert_array_almost_equal(
        sim_no_gate['portfolio_returns'], sim_gate_off['portfolio_returns'],
        err_msg="Disabled gate should match no-gate returns"
    )
    print("  ✓ Disabled gate matches no-gate (same returns and days)")
    print("  ✓ TEST 6 PASSED\n")
    return True


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
    
    # Test 4: Rank-drop gate eligible
    try:
        passed = test_rank_drop_gate_eligible()
        results.append(("Rank-Drop Gate Eligible", passed))
    except Exception as e:
        print(f"✗ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        results.append(("Rank-Drop Gate Eligible", False))
    
    # Test 5: Rank-drop gate excluded / skip day
    try:
        passed = test_rank_drop_gate_excluded()
        results.append(("Rank-Drop Gate Excluded", passed))
    except Exception as e:
        print(f"✗ TEST 5 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        results.append(("Rank-Drop Gate Excluded", False))
    
    # Test 6: Rank-drop gate disabled regression
    try:
        passed = test_rank_drop_gate_disabled_regression()
        results.append(("Rank-Drop Gate Disabled Regression", passed))
    except Exception as e:
        print(f"✗ TEST 6 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        results.append(("Rank-Drop Gate Disabled Regression", False))
    
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
