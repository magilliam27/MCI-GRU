#!/usr/bin/env python
"""
Configuration Validation Script

Checks if configuration files are properly aligned and identifies common issues.
Run this before starting experiments to catch configuration mismatches early.

Usage:
    python check_config.py
"""

import os
import sys

# Fix Windows console encoding issues
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def check_config():
    """Validate configuration and report issues."""
    print("=" * 80)
    print("MCI-GRU Configuration Validation")
    print("=" * 80)
    print()

    checks = []
    errors = []
    warnings = []

    # 1. Check config files exist
    print("1. Checking configuration files...")
    config_files = {
        "Main config": "configs/config.yaml",
        "Data config": "configs/data/sp500.yaml",
        "Features config": "configs/features/with_momentum.yaml",
    }

    for name, path in config_files.items():
        if os.path.exists(path):
            checks.append(f"  ✓ {name}: {path}")
        else:
            errors.append(f"  ✗ {name}: {path} NOT FOUND")

    # 2. Load and check main config
    print("\n2. Validating main configuration...")
    try:
        # Load without resolving interpolations (to avoid Hydra context issues)
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("now", lambda x: "TIMESTAMP")
        cfg = OmegaConf.load("configs/config.yaml")
        checks.append("  ✓ Main config loaded successfully")

        # Check Hydra config (without resolving interpolations)
        if "hydra" in cfg:
            if "run" in cfg.hydra and "dir" in cfg.hydra.run:
                # Don't try to resolve the interpolation, just check it exists
                checks.append("  ✓ Hydra output directory configured")
            else:
                warnings.append("  ⚠  Hydra output directory not configured")
        else:
            warnings.append("  ⚠  Hydra configuration section missing")

    except Exception as e:
        errors.append(f"  ✗ Failed to load main config: {e}")
        import traceback

        print(traceback.format_exc())
        return 1

    # 3. Check data configuration
    print("\n3. Validating data configuration...")
    try:
        data_cfg = OmegaConf.load("configs/data/sp500.yaml")
        checks.append("  ✓ Data config loaded successfully")

        # Check dates are in order
        dates = [
            data_cfg.train_start,
            data_cfg.train_end,
            data_cfg.val_start,
            data_cfg.val_end,
            data_cfg.test_start,
            data_cfg.test_end,
        ]
        if dates == sorted(dates):
            checks.append("  ✓ Dates are in chronological order")
        else:
            errors.append("  ✗ Dates are not in chronological order!")

        # Check data source
        if data_cfg.source == "csv":
            checks.append(f"  ✓ Data source: CSV ({data_cfg.filename})")
            if os.path.exists(data_cfg.filename):
                checks.append(f"  ✓ CSV file found: {data_cfg.filename}")
            else:
                warnings.append(f"  ⚠  CSV file not found: {data_cfg.filename}")
                warnings.append("     Make sure to upload or generate this file before training")
        elif data_cfg.source == "lseg":
            checks.append("  ✓ Data source: LSEG API")
            if "LSEG_API_KEY" in os.environ:
                checks.append("  ✓ LSEG API key found in environment")
            else:
                warnings.append("  ⚠  LSEG API key not set in environment")
                warnings.append("     Set LSEG_API_KEY environment variable or use CSV source")
        else:
            warnings.append(f"  ⚠  Unknown data source: {data_cfg.source}")

    except Exception as e:
        errors.append(f"  ✗ Failed to load data config: {e}")
        import traceback

        print(traceback.format_exc())

    # 4. Check evaluate_sp500.py defaults
    print("\n4. Checking backtest defaults alignment...")
    try:
        # Read evaluate_sp500.py to check DEFAULT_CONFIG
        with open("evaluate_sp500.py", encoding="utf-8") as f:
            content = f.read()

        # Check for updated defaults
        if "'test_start': '2025-01-01'" in content:
            checks.append("  ✓ Backtest test_start matches training config (2025-01-01)")
        elif "'test_start': '2023-01-01'" in content:
            errors.append("  ✗ Backtest test_start outdated (2023 instead of 2025)")
        else:
            warnings.append("  ⚠  Could not verify backtest test_start")

        if "'data_file': 'data/raw/market/sp500_data.csv'" in content:
            checks.append(
                "  ✓ Backtest data_file matches reorganized path (data/raw/market/sp500_data.csv)"
            )
        elif "'data_file': 'sp500_data.csv'" in content:
            checks.append("  ✓ Backtest data_file uses legacy root path (still supported)")
        elif "'data_file': 'sp500_yf_download.csv'" in content:
            errors.append("  ✗ Backtest data_file outdated (sp500_yf_download.csv)")
        else:
            warnings.append("  ⚠  Could not verify backtest data_file")

    except Exception as e:
        warnings.append(f"  ⚠  Could not check evaluate_sp500.py: {e}")

    # 5. Check for common data files
    print("\n5. Checking for data files...")
    data_files = [
        "data/raw/market/sp500_data.csv",
        "data/raw/market/sp500_yf_download.csv",
        "data/raw/market/russell1000_data.csv",
        "sp500_data.csv",
        "sp500_yf_download.csv",
        "russell1000_data.csv",
    ]
    found_data = False
    for df in data_files:
        if os.path.exists(df):
            checks.append(f"  ✓ Found data file: {df}")
            found_data = True

    if not found_data:
        warnings.append("  ⚠  No known data files found in reorganized or legacy locations")
        warnings.append("     Upload data files before running experiments")

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if checks:
        print("\n✓ Passed Checks:")
        for check in checks:
            print(check)

    if warnings:
        print("\n⚠  Warnings:")
        for warning in warnings:
            print(warning)

    if errors:
        print("\n✗ Errors:")
        for error in errors:
            print(error)

    print("\n" + "=" * 80)

    # Return status
    if errors:
        print("❌ Configuration validation FAILED")
        print("   Please fix the errors above before running experiments")
        return 1
    elif warnings:
        print("⚠️  Configuration validation passed with WARNINGS")
        print("   Review warnings above - you may want to fix them")
        return 0
    else:
        print("✅ Configuration validation PASSED")
        print("   All checks passed! Ready to run experiments")
        return 0


def main():
    """Main entry point."""
    try:
        return check_config()
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
