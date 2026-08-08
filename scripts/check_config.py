#!/usr/bin/env python
"""
Configuration Validation Script

Checks if configuration files are properly aligned and identifies common issues.
Run this before starting experiments to catch configuration mismatches early.

Usage:
    python check_config.py
"""

import importlib.util
import io
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fallback used only when the main config carries no `data:` entry in its
# Hydra defaults list. It tracks the base default in `configs/config.yaml`;
# `test_fallback_data_group_names_a_config_that_exists` pins that it resolves.
FALLBACK_DATA_GROUP = "gics_top10_110_2016"


def selected_data_config(cfg) -> str:
    """Return the data config path the main config's Hydra defaults list selects.

    `check_config.py` must validate the config an experiment would actually
    compose, not a hard-coded guess. Hydra defaults entries are either the
    literal string ``_self_`` or a single ``{group: option}`` mapping.
    """
    group = FALLBACK_DATA_GROUP
    raw = cfg.get("defaults")
    # `or []` would be wrong here: an empty ListConfig is falsy, and passing a
    # plain list to to_container raises.
    for entry in OmegaConf.to_container(raw) if raw is not None else []:
        if isinstance(entry, dict) and isinstance(entry.get("data"), str):
            group = entry["data"]
    return f"configs/data/{group}.yaml"


def refinitiv_data_available() -> bool:
    """True when `import refinitiv.data` would succeed.

    `find_spec` on a dotted name imports the parent package first, so when
    `refinitiv` is absent entirely it raises ModuleNotFoundError rather than
    returning None. Letting that escape would turn "the optional LSEG SDK is
    not installed" into a hard validation error.
    """
    try:
        return importlib.util.find_spec("refinitiv.data") is not None
    except ModuleNotFoundError:
        return False


def find_configured_csv(filename: str) -> str | None:
    """Return the location of a configured CSV, or None if it is not present.

    Checks the path as given and relative to the project root. The runtime
    loader resolves through `mci_gru.data.path_resolver`, which also searches
    the project data directories; importing it here would pull in torch and
    pandas, so a miss is reported as a hint rather than as a verdict.
    """
    for candidate in (Path(filename), PROJECT_ROOT / filename):
        if candidate.exists():
            return str(candidate)
    return None


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
    # The data config is not listed here: which one is selected is only known
    # after the main config's defaults list is read, so section 3 checks it.
    config_files = {
        "Main config": "configs/config.yaml",
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
    data_config_path = selected_data_config(cfg)
    try:
        if os.path.exists(data_config_path):
            checks.append(f"  ✓ Data config selected by defaults: {data_config_path}")
        else:
            errors.append(f"  ✗ Data config: {data_config_path} NOT FOUND")

        data_cfg = OmegaConf.load(data_config_path)
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
            located = find_configured_csv(data_cfg.filename)
            if located:
                checks.append(f"  ✓ CSV file found: {located}")
            else:
                warnings.append(f"  ⚠  CSV file not found at configured path: {data_cfg.filename}")
                warnings.append("     Checked the path as given and relative to the project root.")
                warnings.append("     The loader also searches the project data directories.")
        elif data_cfg.source == "lseg":
            checks.append("  ✓ Data source: LSEG API")
            # LSEGLoader.connect() imports refinitiv.data, so that package -- not
            # any environment variable -- is what actually gates this source.
            if refinitiv_data_available():
                checks.append("  ✓ refinitiv-data package is importable")
            else:
                warnings.append("  ⚠  refinitiv-data package not installed")
                warnings.append("     Install it with `pip install refinitiv-data`, or use a")
                warnings.append("     CSV data config such as configs/data/csv_sp500.yaml.")
        else:
            warnings.append(f"  ⚠  Unknown data source: {data_cfg.source}")

    except Exception as e:
        errors.append(f"  ✗ Failed to load data config: {e}")
        import traceback

        print(traceback.format_exc())

    # The former section 4 read `evaluate_sp500.py`, which is not in the tree, so
    # it always fell into its except branch and emitted a bogus warning. A former
    # section 5 scanned a hard-coded list of legacy CSV names that no current data
    # config points at, so it reported "no data files" even when the configured
    # file was present. Whether the configured CSV exists is now answered once,
    # in section 3, against the config actually selected. See issue 146.

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


def _force_utf8_console() -> None:
    """Fix Windows console encoding for the tick and warning glyphs.

    Done here rather than at import time: rebinding `sys.stdout` as an import
    side effect would clobber pytest's capture when this module is imported by
    a test.
    """
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    """Main entry point."""
    _force_utf8_console()
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
