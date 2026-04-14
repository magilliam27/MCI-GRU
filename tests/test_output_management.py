#!/usr/bin/env python
"""
Test script to verify output management features.

This script tests:
1. Logging setup
2. Output directory creation
3. Backtest output organization
"""

import os
import shutil
import sys
import tempfile
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows console encoding fix
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def test_output_structure():
    """Test that output directories are created correctly."""
    print("Testing output directory structure...")

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="mci_gru_test_")
    print(f"  Using temp directory: {temp_dir}")

    try:
        # Simulate Hydra output structure
        experiment_name = "test_experiment"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_dir = os.path.join(temp_dir, experiment_name, timestamp)
        os.makedirs(run_dir, exist_ok=True)

        # Create subdirectories
        dirs = ["models", "predictions", "averaged_predictions"]
        for d in dirs:
            os.makedirs(os.path.join(run_dir, d), exist_ok=True)

        # Verify structure
        assert os.path.exists(run_dir), "Run directory not created"
        assert os.path.exists(os.path.join(run_dir, "models")), "Models dir not created"
        assert os.path.exists(os.path.join(run_dir, "averaged_predictions")), (
            "Predictions dir not created"
        )

        print("  [PASS] Output structure created successfully")

        # Test backtest directory creation
        from evaluate_sp500 import setup_backtest_output_dir

        predictions_dir = os.path.join(run_dir, "averaged_predictions")
        backtest_dir = setup_backtest_output_dir(predictions_dir, suffix="")

        assert os.path.exists(backtest_dir), "Backtest directory not created"
        print("  [PASS] Backtest directory created successfully")

        # Test with suffix
        backtest_dir_tc = setup_backtest_output_dir(predictions_dir, suffix="_with_costs")
        assert os.path.exists(backtest_dir_tc), "Backtest directory with suffix not created"
        assert "with_costs" in backtest_dir_tc, "Suffix not in directory name"
        print("  [PASS] Backtest directory with suffix created successfully")

        print("\n[SUCCESS] All output structure tests passed!\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}\n")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"  Cleaned up temp directory: {temp_dir}")


def test_logging_setup():
    """Test logging configuration."""
    print("Testing logging setup...")

    temp_dir = tempfile.mkdtemp(prefix="mci_gru_log_test_")

    try:
        # Test training logger
        from run_experiment import setup_logging

        logger = setup_logging(temp_dir, "test_experiment")
        logger.info("Test log message")

        # Close logger handlers to release file locks
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Check log file was created
        log_files = [
            f for f in os.listdir(temp_dir) if f.startswith("training_") and f.endswith(".log")
        ]
        assert len(log_files) > 0, "No log file created"

        # Check log file has content
        log_file = os.path.join(temp_dir, log_files[0])
        with open(log_file) as f:
            content = f.read()
            assert "Test log message" in content, "Log message not found in file"

        print("  [PASS] Training logger working")

        # Test backtest logger
        from evaluate_sp500 import setup_backtest_logging

        backtest_logger = setup_backtest_logging(temp_dir)
        backtest_logger.info("Test backtest log")

        # Close backtest logger handlers
        for handler in backtest_logger.handlers[:]:
            handler.close()
            backtest_logger.removeHandler(handler)

        # Check backtest log file
        log_files = [
            f for f in os.listdir(temp_dir) if f.startswith("backtest_") and f.endswith(".log")
        ]
        assert len(log_files) > 0, "No backtest log file created"

        print("  [PASS] Backtest logger working")

        print("\n[SUCCESS] All logging tests passed!\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}\n")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Clean up logging handlers
        import logging

        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

        # Give Windows a moment to release file handles
        import time

        time.sleep(0.1)

        # Clean up temp directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                # On Windows, sometimes files are still locked
                print(
                    f"  [WARNING] Could not remove temp directory: {temp_dir} (files still locked)"
                )
                pass


def test_config_hydra():
    """Test Hydra configuration."""
    print("Testing Hydra configuration...")

    try:
        from omegaconf import OmegaConf

        # Load base config
        config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
        if not os.path.exists(config_path):
            print("  [SKIP] Config file not found, skipping Hydra test")
            return True

        cfg = OmegaConf.load(config_path)

        # Check Hydra section exists
        assert "hydra" in cfg, "Hydra configuration not found"
        assert "run" in cfg.hydra, "Hydra run configuration not found"
        assert "dir" in cfg.hydra.run, "Hydra output directory not configured"

        print("  [PASS] Hydra configuration valid")
        print(f"  [PASS] Output pattern: {cfg.hydra.run.dir}")

        print("\n[SUCCESS] Hydra configuration test passed!\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("MCI-GRU Output Management Tests")
    print("=" * 80)
    print()

    tests = [
        ("Output Structure", test_output_structure),
        ("Logging Setup", test_logging_setup),
        ("Hydra Configuration", test_config_hydra),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'=' * 80}")
        print(f"Running: {name}")
        print("=" * 80)
        result = test_func()
        results.append((name, result))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")

    all_passed = all(result for _, result in results)

    print("=" * 80)
    if all_passed:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print("[WARNING] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
