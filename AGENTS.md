# AGENTS.md

## Cursor Cloud specific instructions

### Branch note
The active development branch is `Corrected_model` (modular codebase). The `main` branch contains legacy flat scripts (`mci_gru_sp500.py`, `sp500.py`) with known dimension-ordering bugs. Always base work off `Corrected_model`.

### Project overview
MCI-GRU stock prediction model. Single Python package (`mci_gru/`) with Hydra config management. No web server, no database, no Docker — pure ML training pipeline.

### Running commands

- **Lint:** `flake8 --max-line-length=120 mci_gru/ run_experiment.py`
- **Tests:** `python -m pytest tests/ -v` (19 tests, all unit/integration, no external data needed)
- **Experiment:** `python run_experiment.py data=csv_sp500` (requires CSV data; see below)
- **Config reference:** see `docs/QUICK_REFERENCE.md` and `docs/CONFIGURATION_GUIDE.md`

### Data setup
CSV data files are gitignored (`*.csv`). To build a dataset from Yahoo Finance:
```bash
python scripts/build_sp500_dataset.py --tickers <tickers.csv> --start 2017-01-01 --end 2025-12-31 --out data/raw/market/sp500_data.csv
```
Then run with `data=csv_sp500`. LSEG source (`data=sp500`) requires Refinitiv credentials.

### Dependency gotchas
- **pandas:** Use `pandas>=2.0,<3.0`. Pandas 3.0 breaks the normalization pipeline (implicit float-to-int coercion rejected).
- **yfinance:** Version 1.x returns MultiIndex columns even for single-ticker downloads. The `build_sp500_dataset.py` script handles this (line 67-68), but the legacy `mci_gru_sp500.py` in `main` does not. A `usercustomize.py` shim is installed in the dev environment to patch `yf.download(multi_level_index=False)` by default.
- **torch:** CPU-only is sufficient for development/testing. GPU (CUDA) optional for production training.
- **fredapi:** Required by `requirements.txt` but only used when `include_credit_spread=true` or `include_global_regime=true`. Needs `FRED_API_KEY` env var.

### Legacy `mci_gru_sp500.py` (main branch)
The standalone `mci_gru_sp500.py` in `main` has a data dimension bug: `create_dataset` produces `[time, stocks, features]` but the model expects `[stocks, time, features]`. The modular codebase (`mci_gru/` package on `Corrected_model`) fixes this with a proper `.transpose(1, 0, 2)` in `generate_time_series_features`.
