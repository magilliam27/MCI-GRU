# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

MCI-GRU is a Python ML research framework for stock return prediction using Multi-head Cross-attention and Improved GRU models with Graph Attention Networks. It is a standalone Python project with no web servers, databases, Docker, or external services required.

### Environment

- Python 3.12 is available as `python3`; a `python` symlink should exist at `/usr/local/bin/python`.
- Dependencies are listed in `requirements.txt` and installed via `pip install -r requirements.txt`.
- `torch-geometric` is installed from PyPI alongside PyTorch; no special wheel URL is needed on this VM.

### Running tests

- **pytest tests**: `python3 -m pytest tests/test_regime_features.py tests/test_index_level_mode.py -v` (9 tests, all pass).
- **Backtest fairness tests**: `python3 tests/test_backtest_fairness.py` (7 tests, all pass).
- **Output management tests**: `python3 tests/test_output_management.py` — 2 of 3 subtests fail because they import the old `evaluate_sp500` module name. The current backtest lives at `tests/backtest_sp500_daily.py` (imported as `backtest_sp500` within the tests directory). This test file has not been updated to use the new import path.

### Running the experiment

The main entry point is `python3 run_experiment.py` with Hydra configuration (see `docs/CONFIGURATION_GUIDE.md`). Running a full training experiment requires stock data at `data/raw/market/sp500_data.csv` or an LSEG API key. Without data, the experiment will fail at the data-loading step. Quick sanity check of the config: `python3 scripts/check_config.py`.

### Backtest evaluation

The current backtest/evaluation module is `tests/backtest_sp500_daily.py`. It can be run standalone as `python3 tests/backtest_sp500_daily.py --predictions_dir path/to/predictions --auto_save`. The old `evaluate_sp500.py` no longer exists; some docs and tests still reference it.

### Key gotchas

- FRED API features (`include_credit_spread`, `include_global_regime`) require `FRED_API_KEY` env var. They soft-fail to zero-filled features when unavailable.
- LSEG/Refinitiv features require `LSEG_API_KEY` env var and a running Refinitiv Workspace desktop app. Use `+data=csv_sp500` as a fallback when LSEG is unavailable.
- Hydra manages output directories; pass `output_dir=...` on the CLI to override the default `results/` directory.
- The `_uncertain/` and `seed_results/` directories contain historical experiment outputs and are not part of the active codebase.
