# Data load path audit (2026-09-05)

Research report for GitHub issue #189: "Audit what the data load path checks today, with reproductions of its silent failures". This is a measurement of `origin/main @ 125abda`. Nothing was fixed, and no tracked file was modified.

## Question

What does the load path check today, from a config to a DataFrame in memory and on to tensors, and which failure modes around data identity (which file was loaded, is it recorded) and data quality (columns, dtypes, dates, duplicates, NaNs, PIT invariants, sector map) fail silently?

## Method

- Interpreter: `C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe` (Python 3.12.11, pandas 2.3.3).
- Working directory for every reproduction: `C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05` (HEAD `125abda007a76bb451af85cc2ab9f0a50b9e4fdd`, branch `claude/187-research-reports`, whose tree is identical to `origin/main` at that commit).
- Package identity: every script prints `mci_gru.__file__` at the top and asserts it contains `ecstatic-chaplygin-010f05`. Observed value in every run: `C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\mci_gru\__init__.py`.
- Resolution caveat found while running: with `python.exe <script>` the script directory becomes `sys.path[0]`, the cwd is not on `sys.path`, and the venv's editable install then resolved `mci_gru` to the protected checkout `C:\Users\magil\MCI-GRU\mci_gru\__init__.py`. The assertion in the shared header caught this on the first attempt (the scripts died at import, before touching any file); all reported runs were made with `PYTHONPATH=<worktree>` so the worktree package was imported. `PYTHONIOENCODING=utf-8` was set; `-u` was used so stderr and stdout interleave in order.
- Fixtures: every reproduction writes tiny synthetic CSVs under the scratchpad directory `C:\Users\magil\AppData\Local\Temp\claude\...\scratchpad\r189\` (rendered as `<SCRATCH>` in the quoted outputs). No real data file was read or written. Two scripts (items 2 and 3) plant a uniquely named `zz_audit_189_<pid>.csv` under `<worktree>/data/raw/market/`, which `.gitignore` line 1 (`*.csv`) hides from git (`git check-ignore -v data/raw/market/zz_audit_189_test.csv` reports `.gitignore:1:*.csv`), and remove it in a `finally` block.
- Protected checkout `C:\Users\magil\MCI-GRU`: only `Path.exists()` was called on it (item 2c). Its `data/raw/market` mtime is unchanged (2026-07-31 23:09:53).
- `git status --porcelain` in the worktree: empty before the first run, empty after items 2 and 3 individually, and empty at the end. `find data -name '*.csv'` in the worktree returns 0 files at the end; `find . -name 'zz_audit_189_*'` returns 0.
- Output rendering: the fingerprint warning in `mci_gru/evaluation/experiment_summary.py:79` contains an em-dash in the source; it is rendered as `--` below. The tqdm progress line from `generate_time_series_features` carried carriage returns and was collapsed. Everything else is verbatim.

## Summary table

| code path | check | enforced or absent | fails closed or silent | reference (file:line) |
|---|---|---|---|---|
| `DataManager.load` | `source` in {csv, lseg} | enforced | fails closed (ValueError) | mci_gru/data/data_manager.py:53-59 |
| `DataManager._load_from_csv` | file exists (via resolver) | enforced | fails closed (FileNotFoundError) | mci_gru/data/path_resolver.py:42-45; data_manager.py:107 |
| `DataManager._load_from_csv` | `dt`, `kdcode` columns present | absent; a KeyError escapes from a logging f-string | fails closed by accident (KeyError 'dt') | data_manager.py:113-114 |
| `DataManager._load_from_csv` | OHLCV columns present, numeric dtypes | absent | silent (S4) | data_manager.py:110-116 |
| `DataManager._load_from_csv` | `dt` parsed or canonical `YYYY-MM-DD` | absent (strings compared lexicographically downstream) | silent (S3) | data_manager.py:110-116 |
| `DataManager._load_from_csv` | duplicate `(dt, kdcode)` | absent | silent (S1) | data_manager.py:110-116 |
| `DataManager._load_from_csv` | NaN or negative prices | absent | silent (S2) | data_manager.py:110-116 |
| `DataManager.load_index_series` | `close` column present | enforced | fails closed (ValueError) | data_manager.py:77-78 |
| `DataManager.load_index_series` | `dt` column present | absent (KeyError at line 76, before the `close` check) | fails closed by accident | data_manager.py:76 |
| `DataManager.load_index_series` | duplicate `dt` | absent (keep last) | silent (S5) | data_manager.py:100 |
| `DataManager.load_index_series` | NaN or negative `close` | absent | silent (S6) | data_manager.py:75-103 |
| `DataManager.load_vix` | `vix_data.csv` exists somewhere | enforced | fails closed (FileNotFoundError with hint) | data_manager.py:157-162 |
| `DataManager.load_vix` | which `vix_data.csv` (cwd first, then seven fallback dirs) | absent | silent (S7) | data_manager.py:157; path_resolver.py:19-40 |
| `DataManager.load_vix` | columns, dtypes, date coverage | absent | silent (S7, S8) | data_manager.py:163; mci_gru/features/volatility.py:232-233 |
| `DataManager.load_credit_spreads` | `FRED_API_KEY` present | enforced in `FREDLoader.__init__` | fails closed in isolation; swallowed by `load_auxiliary_data`, zero-filled by `FeatureEngineer.transform` (S11) | mci_gru/data/fred_loader.py:29-34; mci_gru/pipeline.py:160-164; mci_gru/features/registry.py:397-402 |
| `DataManager.load_regime_inputs` (csv) | `dt` plus seven regime columns | enforced | fails closed (ValueError) | data_manager.py:243-251 |
| `DataManager.load_regime_inputs` (csv) | numeric values | absent (`to_numeric(errors="coerce")` then `ffill`) | silent (S9) | data_manager.py:253-259 |
| `DataManager.load_regime_inputs` (csv) | duplicate `dt` | absent (keep last) | silent (S10) | data_manager.py:242 |
| `DataManager.load_regime_inputs` (live) | six source series non-empty | enforced | fails closed (ValueError) | data_manager.py:373-389 |
| `load_auxiliary_data` (regime) | regime load failure | configurable via `features.regime_strict` (default False) | fails closed if strict, else warning and zero-fill | pipeline.py:166-183; mci_gru/config.py:218 |
| `resolve_project_data_path` | exact path or `PROJECT_ROOT/path` exists | enforced | fails closed on total miss | path_resolver.py:19-25, 42-45 |
| `resolve_project_data_path` | basename fallback across seven directories, no log, no record | absent (module has no logger) | silent (S12) | path_resolver.py:27-40 |
| `resolve_project_data_path` | cwd-relative file preferred over `PROJECT_ROOT` | absent | silent (S13) | path_resolver.py:19-21 |
| `data_file_fingerprint` | configured path exists relative to cwd | absent (warning, three nulls) | warning only (S14, S16) | mci_gru/evaluation/experiment_summary.py:75-84 |
| `run_experiment.py` metadata | fingerprint of the file actually loaded (resolved path) | absent (`data_file` records the configured string) | warning only (S15) | run_experiment.py:187, 194 |
| `run_experiment.py` metadata | fingerprint of `pit_universe_csv`, `sector_map_csv`, `index_filename` | absent (only `cfg_w.data.filename` is fingerprinted) | silent | run_experiment.py:194 |
| `normalise_pit_intervals` | `kdcode`, `valid_from`, `valid_to` present | enforced | fails closed (ValueError) | mci_gru/data/pit.py:71-74 |
| `normalise_pit_intervals` | dates parseable | enforced by pandas only | fails closed (DateParseError) | pit.py:81-82 |
| `normalise_pit_intervals` | rows with NaN key or dates | absent (dropped) | silent (S17) | pit.py:79 |
| `normalise_pit_intervals` | overlapping intervals per name | absent | silent (S18) | pit.py:55-90 |
| `normalise_pit_intervals` | `valid_from <= valid_to` | absent | silent (S19) | pit.py:55-90 |
| `normalise_pit_intervals` | `kdcode` case normalisation | absent (strip only) | silent (S22) | pit.py:80 |
| `active_kdcodes_in_period` / `select_universe` | PIT names present in the panel | absent (set intersection) | silent (S20) | pit.py:149-150; pipeline.py:503-509 |
| `build_pit_masks` | name absent from panel becomes active-but-never-tradable | absent | silent (S20) | pit.py:233-237 |
| `_apply_pit_universe` (row_filter, norm source) | overlapping intervals must not duplicate rows | absent (inner merge on `kdcode`) | silent, rows duplicated (S21) | pipeline.py:226-236 |
| `load_pit_intervals`, `_apply_pit_universe` | path resolution | absent (bare `pd.read_csv`, cwd-relative, no resolver) | fails closed on miss via pandas | pit.py:136; pipeline.py:228 |
| `load_sector_map_csv` | file exists | enforced | fails closed (FileNotFoundError) | mci_gru/graph/sector_edges.py:44-45 |
| `load_sector_map_csv` | zero-byte file | enforced | fails closed (ValueError "Empty CSV") | sector_edges.py:48-49 |
| `load_sector_map_csv` | `kdcode` and `sector`/`gics_sector` columns | enforced | fails closed (ValueError) | sector_edges.py:51-55 |
| `load_sector_map_csv` | header-only file (zero mappings) | absent | silent, info count only (S26) | sector_edges.py:80 |
| `load_sector_map_csv` | duplicate `kdcode`, same sector | absent | silent (S24) | sector_edges.py:68-73 |
| `load_sector_map_csv` | duplicate `kdcode`, different sector | absent (newest `as_of_date` wins; last row wins without one) | warning only (S23) | sector_edges.py:69-79 |
| `load_sector_map_csv` | missing or placeholder sector | absent (row skipped) | silent (S25) | sector_edges.py:65-66 |
| `build_sector_edges` | map covers the panel; case match | absent | silent, info coverage line (S27) | sector_edges.py:116-120, 137-142 |
| `filter_complete_stocks` | rows per `kdcode` == session count (a duplicate counts as coverage) | absent | silent (S28) | data_manager.py:470-471 |
| `generate_time_series_features` | duplicate `(dt, kdcode)` | absent (keep last) | silent (S29) | mci_gru/data/preprocessing.py:76-77 |
| `compute_labels` | duplicate `(dt, kdcode)` | absent (row shift then `pivot_table` mean) | silent, wrong labels (S30) | preprocessing.py:190-197 |
| `compute_labels` | NaN label | absent (row mean then 0 when `fill_missing`) | silent (S32) | preprocessing.py:200-205 |
| `impute_feature_nans_by_day` | NaN features | absent (day mean then 0) | silent (S32) | mci_gru/data/transforms.py:14-27 |
| `prepare_data` (stock level) | `dt` parsed, canonical, monotone; duplicates; negative prices | absent (no `to_datetime`, `drop_duplicates`, or `duplicated` anywhere in the function) | silent (S31) | pipeline.py:780-905 |
| `CombinedDataset` | `sample_dates` canonical `YYYY-MM-DD` | enforced | fails closed (ValueError) | data_manager.py:573-577; mci_gru/graph/schedule.py:38-63 |
| `_audit_pit_breadth` | tradable count >= `pit_min_scoreable_stocks` | enforced when `pit_breadth_policy=error` (default) | fails closed (error), warning (warn), skipped (off) | pipeline.py:251-274; config.py:63-64, 91-97 |
| `ExperimentConfig._validate_embargo` | calendar gap > `label_t` | enforced unless `skip_embargo_check` | fails closed, or UserWarning when skipped | config.py:838-862 |
| `assert_training_labels_respect_embargo` | session-axis embargo, unverifiable panel | enforced, unconditional | fails closed (ValueError) | preprocessing.py:242-348; pipeline.py:317-342 |
| `purge_training_sessions_for_embargo` | sessions remain after purge | enforced | fails closed (ValueError) | preprocessing.py:229-239 |
| `resolve_pit_context` | `pit_universe_csv` set when `use_pit_universe` | enforced | fails closed (ValueError) | pipeline.py:453-462 |
| `DataConfig.__post_init__` | enum values, date order, `pit_min_scoreable_stocks >= 0` | enforced | fails closed (ValueError) | config.py:66-97 |

Silent-failure identifiers S1..S32 are defined in the "Silent failures reproduced" list at the end.

## Item 1: DataManager.load, _load_from_csv, load_index_series, load_vix, load_credit_spreads, load_regime_inputs

What the code asserts, from reading:

- `_load_from_csv` (data_manager.py:106-117) is `resolve_project_data_path` + `pd.read_csv` + three `logger.info` lines. The log lines index `df['dt']` and `df['kdcode']`, so a missing `dt` or `kdcode` column surfaces as a `KeyError` raised from a logging statement, not from a check. Nothing else is asserted: no OHLCV column check, no dtype check, `dt` stays a string, duplicates and NaNs pass through.
- `load_index_series` (data_manager.py:61-103) indexes `df["dt"]` at line 76 (KeyError if absent), requires `close` at lines 77-78, synthesises missing `open/high/low/volume/turnover`, and dedups on `dt` with `keep="last"` at line 100. No NaN or sign check.
- `load_vix` (data_manager.py:143-165) resolves the bare basename `"vix_data.csv"`, so the resolver's first rule (exists relative to cwd) applies to whatever directory the process was launched from; the file content is read with a bare `pd.read_csv` and no column check. The consumer `add_vix_features` (volatility.py:206-240) left-merges on `dt`, forward-fills, then fills the remainder with the constant 20.
- `load_credit_spreads` (data_manager.py:167-199) has no CSV path; `FREDLoader.__init__` raises without `FRED_API_KEY` (fred_loader.py:29-34). `load_auxiliary_data` catches every exception, logs a warning and returns `None` (pipeline.py:159-164); `FeatureEngineer.transform` then writes 0.0 into all seven credit columns (registry.py:397-402).
- `load_regime_inputs` with `regime_inputs_csv` (data_manager.py:222-262) requires `dt` plus the seven `REGIME_VARIABLES` (fails closed), dedups on `dt` keep-last (line 242), coerces every variable with `errors="coerce"` (line 253) and forward-fills (line 258). The live path (lines 264-448) fails closed when any of six source series is missing (lines 373-389) and otherwise ffill/bfills.

Reproduction (`<SCRATCH>/repro1_loaders.py`):

```python
"""Item 1: DataManager.load / _load_from_csv / load_index_series / load_vix /
load_credit_spreads / load_regime_inputs against tiny synthetic CSVs."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hdr import SCRATCH, section, show_exc  # noqa: E402

import pandas as pd  # noqa: E402

from mci_gru.config import DataConfig  # noqa: E402
from mci_gru.data.data_manager import DataManager  # noqa: E402

section("1a _load_from_csv: dirty panel (dup key, NaN close, negative close, mixed date format, text volume)")
dirty = SCRATCH / "r1_dirty_panel.csv"
dirty.write_text(
    "kdcode,dt,open,high,low,close,volume\n"
    "AAA,2020-01-02,10,11,9,10,100\n"
    "AAA,2020-01-02,10,11,9,99,100\n"
    "AAA,2020-01-03,10,11,9,,100\n"
    "AAA,2020/01/06,10,11,9,-5,100\n"
    "BBB,2020-01-02,10,11,9,10,abc\n"
)
df = DataManager(DataConfig(filename=str(dirty))).load()
print("returned rows:", len(df))
print("dtypes:", dict(df.dtypes.astype(str)))
print("duplicate (dt,kdcode) rows kept:", int(df.duplicated(["dt", "kdcode"]).sum()))
print("close NaN:", int(df["close"].isna().sum()), "| close < 0:", int((df["close"] < 0).sum()))
print("dt unique (string-sorted):", sorted(df["dt"].unique()))

section("1b _load_from_csv: CSV without a dt column")
nodt = SCRATCH / "r1_no_dt.csv"
nodt.write_text("kdcode,date,close\nAAA,2020-01-02,10\n")
show_exc(DataManager(DataConfig(filename=str(nodt))).load)

section("1c load_index_series: duplicate dt rows and NaN close")
idx = SCRATCH / "r1_index.csv"
idx.write_text("dt,close\n2020-01-02,100\n2020-01-02,200\n2020-01-03,\n2020-01-06,-1\n")
dm = DataManager(DataConfig(index_filename=str(idx), train_start="2020-01-01", train_end="2020-06-30",
                            val_start="2020-07-08", val_end="2020-09-30", test_start="2020-10-08", test_end="2020-12-31"))
out = dm.load_index_series()
print(out[["kdcode", "dt", "close", "volume"]].to_string(index=False))

section("1d load_vix: not found anywhere -> FileNotFoundError (fail closed)")
dm = DataManager(DataConfig())
show_exc(dm.load_vix)

section("1d' load_vix: a vix_data.csv in the *current working directory* wins, any columns accepted")
shadow_dir = SCRATCH / "r1_cwd_shadow"
shadow_dir.mkdir(exist_ok=True)
(shadow_dir / "vix_data.csv").write_text("foo,bar\n1,2\n")
here = os.getcwd()
os.chdir(shadow_dir)
try:
    vix_df = dm.load_vix()
    print("loaded from:", os.path.join(os.getcwd(), "vix_data.csv"))
    print("columns:", list(vix_df.columns), "rows:", len(vix_df))
finally:
    os.chdir(here)

section("1d'' add_vix_features: VIX dates that do not overlap the panel -> every row gets the constant 20 fill")
from mci_gru.features.volatility import add_vix_features  # noqa: E402
panel = pd.DataFrame({"kdcode": ["AAA"] * 3, "dt": ["2020-01-02", "2020-01-03", "2020-01-06"], "close": [10, 11, 12]})
vix = pd.DataFrame({"dt": ["2019-01-02", "2019-01-03"], "close": [30.0, 31.0]})
merged = add_vix_features(panel, vix)
print(merged[["dt", "vix", "vix_change", "vix_regime"]].to_string(index=False))

section("1e load_regime_inputs(regime_inputs_csv): missing columns -> ValueError (fail closed)")
bad = SCRATCH / "r1_regime_missing.csv"
bad.write_text("dt,regime_market\n2020-01-02,1\n")
show_exc(dm.load_regime_inputs, regime_inputs_csv=str(bad))

section("1e' load_regime_inputs(regime_inputs_csv): non-numeric cell and duplicate dt")
from mci_gru.regime_contract import REGIME_VARIABLES  # noqa: E402
cols = ["dt"] + list(REGIME_VARIABLES)
rows = [
    ["2020-01-02"] + [1.0] * len(REGIME_VARIABLES),
    ["2020-01-03"] + ["abc"] + [2.0] * (len(REGIME_VARIABLES) - 1),
    ["2020-01-03"] + [3.0] * len(REGIME_VARIABLES),
    ["2020-01-06"] + ["abc"] + [4.0] * (len(REGIME_VARIABLES) - 1),
]
reg = SCRATCH / "r1_regime_dirty.csv"
pd.DataFrame(rows, columns=cols).to_csv(reg, index=False)
print("input rows:", len(rows), "| input has text 'abc' in", REGIME_VARIABLES[0], "on 2020-01-03 and 2020-01-06")
res = dm.load_regime_inputs(regime_inputs_csv=str(reg))
print(res[["dt", REGIME_VARIABLES[0], REGIME_VARIABLES[1]]].to_string(index=False))

section("1f load_credit_spreads with no FRED_API_KEY: raises; load_auxiliary_data swallows it")
os.environ.pop("FRED_API_KEY", None)
show_exc(dm.load_credit_spreads)
from mci_gru.config import ExperimentConfig, FeatureConfig  # noqa: E402
from mci_gru.pipeline import load_auxiliary_data  # noqa: E402
cfg = ExperimentConfig(features=FeatureConfig(include_credit_spread=True))
vix_df, credit_df, regime_df = load_auxiliary_data(dm, cfg)
print("credit_df is None:", credit_df is None)
from mci_gru.features import FeatureEngineer  # noqa: E402
from mci_gru.features.credit import CREDIT_FEATURES  # noqa: E402
fe = FeatureEngineer(include_momentum=False, include_credit_spread=True)
panel = pd.DataFrame({"kdcode": ["AAA"] * 3, "dt": ["2020-01-02", "2020-01-03", "2020-01-06"],
                      "open": [10, 11, 12], "high": [10, 11, 12], "low": [10, 11, 12], "close": [10, 11, 12], "volume": [1, 1, 1]})
feat = fe.transform(panel, None, credit_df, None)
print("credit feature columns after soft-fail:", {c: feat[c].unique().tolist() for c in CREDIT_FEATURES})
```

Output:

```text
python      : C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe
cwd         : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05
mci_gru     : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\mci_gru\__init__.py

======== 1a _load_from_csv: dirty panel (dup key, NaN close, negative close, mixed date format, text volume) ========
LOG INFO mci_gru.data.data_manager: Loading data from <SCRATCH>\r1_dirty_panel.csv...
LOG INFO mci_gru.data.data_manager:   Loaded 5 rows
LOG INFO mci_gru.data.data_manager:   Date range: 2020-01-02 to 2020/01/06
LOG INFO mci_gru.data.data_manager:   Stocks: 2
returned rows: 5
dtypes: {'kdcode': 'object', 'dt': 'object', 'open': 'int64', 'high': 'int64', 'low': 'int64', 'close': 'float64', 'volume': 'object'}
duplicate (dt,kdcode) rows kept: 1
close NaN: 1 | close < 0: 1
dt unique (string-sorted): ['2020-01-02', '2020-01-03', '2020/01/06']

======== 1b _load_from_csv: CSV without a dt column ========
LOG INFO mci_gru.data.data_manager: Loading data from <SCRATCH>\r1_no_dt.csv...
LOG INFO mci_gru.data.data_manager:   Loaded 1 rows
RAISED KeyError: 'dt'

======== 1c load_index_series: duplicate dt rows and NaN close ========
kdcode         dt  close  volume
 INDEX 2020-01-02  200.0     0.0
 INDEX 2020-01-03    NaN     0.0
 INDEX 2020-01-06   -1.0     0.0

======== 1d load_vix: not found anywhere -> FileNotFoundError (fail closed) ========
RAISED FileNotFoundError: VIX data not found. Create vix_data.csv under data/raw/market or use source='lseg'

======== 1d' load_vix: a vix_data.csv in the *current working directory* wins, any columns accepted ========
loaded from: <SCRATCH>\r1_cwd_shadow\vix_data.csv
columns: ['foo', 'bar'] rows: 1

======== 1d'' add_vix_features: VIX dates that do not overlap the panel -> every row gets the constant 20 fill ========
Merging VIX features...
C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\mci_gru\features\volatility.py:233: FutureWarning: Series.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df["vix"] = df["vix"].fillna(method="ffill").fillna(20)
  Added VIX features: vix, vix_change, vix_regime
  VIX range: 20.0 to 20.0
        dt  vix  vix_change  vix_regime
2020-01-02 20.0         0.0         0.0
2020-01-03 20.0         0.0         0.0
2020-01-06 20.0         0.0         0.0

======== 1e load_regime_inputs(regime_inputs_csv): missing columns -> ValueError (fail closed) ========
<SCRATCH>\_hdr.py:35: DeprecationWarning: regime_inputs_csv is deprecated; use the live FRED/LSEG regime input path with FRED_API_KEY instead. If this legacy escape hatch is used, the CSV must contain dt plus all seven regime variables.
  out = fn(*args, **kwargs)
RAISED ValueError: Regime CSV <SCRATCH>\r1_regime_missing.csv is missing required columns: ['regime_copper', 'regime_monetary_policy', 'regime_oil', 'regime_stock_bond_corr', 'regime_volatility', 'regime_yield_curve']. CSV regime inputs are deprecated; if used as a legacy override, they must provide the full seven-variable contract. See docs/REGIME_DATA_CONTRACT.md.

======== 1e' load_regime_inputs(regime_inputs_csv): non-numeric cell and duplicate dt ========
input rows: 4 | input has text 'abc' in regime_market on 2020-01-03 and 2020-01-06
<SCRATCH>\repro1_loaders.py:84: DeprecationWarning: regime_inputs_csv is deprecated; use the live FRED/LSEG regime input path with FRED_API_KEY instead. If this legacy escape hatch is used, the CSV must contain dt plus all seven regime variables.
  res = dm.load_regime_inputs(regime_inputs_csv=str(reg))
        dt  regime_market  regime_yield_curve
2020-01-02            1.0                 1.0
2020-01-03            3.0                 3.0
2020-01-06            3.0                 4.0

======== 1f load_credit_spreads with no FRED_API_KEY: raises; load_auxiliary_data swallows it ========
RAISED ValueError: FRED API key required. Set FRED_API_KEY environment variable or pass api_key to FREDLoader.
LOG WARNING mci_gru.pipeline: Warning: Could not load credit spread data: FRED API key required. Set FRED_API_KEY environment variable or pass api_key to FREDLoader.
credit_df is None: True
LOG INFO mci_gru.features.registry: ============================================================
LOG INFO mci_gru.features.registry: Feature Engineering Pipeline
LOG INFO mci_gru.features.registry: ============================================================
  Added turnover feature (close * volume)
LOG INFO mci_gru.features.registry: ============================================================
LOG INFO mci_gru.features.registry: Feature engineering complete
LOG INFO mci_gru.features.registry: ============================================================
credit feature columns after soft-fail: {'ig_spread': [0.0], 'hy_spread': [0.0], 'ig_spread_change': [0.0], 'hy_spread_change': [0.0], 'ig_spread_zscore': [0.0], 'hy_spread_zscore': [0.0], 'credit_spread_diff': [0.0]}
```

What the user would have seen: three INFO lines ("Loaded 5 rows", "Date range: 2020-01-02 to 2020/01/06", "Stocks: 2") for a panel containing a duplicate key, a NaN close, a negative close, a non-ISO date and a text volume; a `KeyError: 'dt'` with no contract message for a missing date column; "VIX range: 20.0 to 20.0" printed once when the VIX file did not cover the panel; a regime value of 3.0 where the CSV said "abc"; and one WARNING line before the run continued with all-zero credit features.

## Item 2: resolve_project_data_path

From reading: `PROJECT_ROOT = Path(__file__).resolve().parents[2]` (path_resolver.py:7) is the package parent, so in a worktree it is the worktree. Resolution order is exact path relative to cwd (lines 19-21), then `PROJECT_ROOT / configured_path` (23-25), then `basename` in seven directories (27-40). The module imports no `logging` and defines no logger; the only diagnostic is the final `FileNotFoundError` (42-45). `DataManager._load_from_csv` logs only the resolved path (data_manager.py:108) and `run_experiment.py:187` records the configured string as `data_file`, so a substitution is not visible in any artifact.

Reproduction (`<SCRATCH>/repro2_path_resolver.py`):

```python
"""Item 2: resolve_project_data_path basename fallback and the worktree case."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hdr import SCRATCH, section, show_exc  # noqa: E402

from pathlib import Path  # noqa: E402

from mci_gru.config import DataConfig  # noqa: E402
from mci_gru.data import path_resolver  # noqa: E402
from mci_gru.data.data_manager import DataManager  # noqa: E402
from mci_gru.data.path_resolver import PROJECT_ROOT, resolve_project_data_path  # noqa: E402

print("PROJECT_ROOT:", PROJECT_ROOT)
print("path_resolver defines a logger:", hasattr(path_resolver, "logger"), "| imports logging:", "logging" in path_resolver.__dict__)

section("2a basename fallback: configured data/processed/<name> does not exist; same-named file planted in data/raw/market")
name = f"zz_audit_189_{os.getpid()}.csv"
configured = f"data/processed/{name}"
planted = PROJECT_ROOT / "data" / "raw" / "market" / name
planted.write_text("kdcode,dt,open,high,low,close,volume\nZZZ,2020-01-02,1,1,1,1,1\n")
try:
    print("configured   :", configured, "| exists:", Path(configured).exists(), "| under root:", (PROJECT_ROOT / configured).exists())
    print("--- calling resolve_project_data_path (any log lines would appear between these markers) ---")
    resolved = resolve_project_data_path(configured)
    print("--- returned ---")
    print("resolved     :", resolved)
    print("configured dir == resolved dir:", (PROJECT_ROOT / configured).parent == resolved.parent)
    print("--- DataManager.load with the same configured path ---")
    df = DataManager(DataConfig(filename=configured)).load()
    print("loaded rows:", len(df), "kdcode:", df["kdcode"].tolist())
finally:
    planted.unlink()
    print("planted file removed:", not planted.exists())

section("2b cwd-first: a same-relative-path file under the *current working directory* shadows the project one")
shadow = SCRATCH / "r2_cwd" / "data" / "raw" / "market"
shadow.mkdir(parents=True, exist_ok=True)
(shadow / "sp500_data.csv").write_text("kdcode,dt,close\nSHADOW,2020-01-02,1\n")
here = os.getcwd()
os.chdir(SCRATCH / "r2_cwd")
try:
    print("cwd now      :", os.getcwd())
    print("resolved     :", resolve_project_data_path(DataConfig().filename))
finally:
    os.chdir(here)

section("2c worktree case: default DataConfig().filename from a worktree whose data dir has no corpus")
cfg = DataConfig()
print("configured   :", cfg.filename)
tried = [Path(cfg.filename), PROJECT_ROOT / cfg.filename] + [d / Path(cfg.filename).name for d in [
    PROJECT_ROOT / "data" / "raw" / "market", PROJECT_ROOT / "data" / "raw" / "constituents",
    PROJECT_ROOT / "data" / "raw" / "reference", PROJECT_ROOT / "data" / "external",
    PROJECT_ROOT / "data" / "interim", PROJECT_ROOT / "data" / "processed", PROJECT_ROOT]]
for t in tried:
    print("  tried:", t, "| exists:", t.exists())
protected = Path(r"C:\Users\magil\MCI-GRU\data\raw\market\sp500_data.csv")
print("corpus in protected checkout exists (read-only stat):", protected.exists())
show_exc(resolve_project_data_path, cfg.filename)
show_exc(DataManager(cfg).load)
```

Output:

```text
python      : C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe
cwd         : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05
mci_gru     : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\mci_gru\__init__.py
PROJECT_ROOT: C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05
path_resolver defines a logger: False | imports logging: False

======== 2a basename fallback: configured data/processed/<name> does not exist; same-named file planted in data/raw/market ========
configured   : data/processed/zz_audit_189_28896.csv | exists: False | under root: False
--- calling resolve_project_data_path (any log lines would appear between these markers) ---
--- returned ---
resolved     : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\raw\market\zz_audit_189_28896.csv
configured dir == resolved dir: False
--- DataManager.load with the same configured path ---
LOG INFO mci_gru.data.data_manager: Loading data from C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\raw\market\zz_audit_189_28896.csv...
LOG INFO mci_gru.data.data_manager:   Loaded 1 rows
LOG INFO mci_gru.data.data_manager:   Date range: 2020-01-02 to 2020-01-02
LOG INFO mci_gru.data.data_manager:   Stocks: 1
loaded rows: 1 kdcode: ['ZZZ']
planted file removed: True

======== 2b cwd-first: a same-relative-path file under the *current working directory* shadows the project one ========
cwd now      : <SCRATCH>\r2_cwd
resolved     : <SCRATCH>\r2_cwd\data\raw\market\sp500_data.csv

======== 2c worktree case: default DataConfig().filename from a worktree whose data dir has no corpus ========
configured   : data/raw/market/sp500_data.csv
  tried: data\raw\market\sp500_data.csv | exists: False
  tried: C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\raw\market\sp500_data.csv | exists: False
  tried: C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\raw\market\sp500_data.csv | exists: False
  tried: C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\raw\constituents\sp500_data.csv | exists: False
  tried: C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\raw\reference\sp500_data.csv | exists: False
  tried: C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\external\sp500_data.csv | exists: False
  tried: C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\interim\sp500_data.csv | exists: False
  tried: C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\processed\sp500_data.csv | exists: False
  tried: C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\sp500_data.csv | exists: False
corpus in protected checkout exists (read-only stat): True
RAISED FileNotFoundError: Data file not found for configured path 'data/raw/market/sp500_data.csv'. Tried explicit path and project fallbacks.
RAISED FileNotFoundError: Data file not found for configured path 'data/raw/market/sp500_data.csv'. Tried explicit path and project fallbacks.
```

What the user would have seen: for 2a, one INFO line "Loading data from ...\data\raw\market\zz_audit_189_28896.csv..." while the config said `data/processed/zz_audit_189_28896.csv`, with nothing else logged and no record of the substitution; for 2b, the file under the launch directory used instead of the project one; for 2c, `FileNotFoundError: Data file not found for configured path 'data/raw/market/sp500_data.csv'. Tried explicit path and project fallbacks.` from a worktree whose `data/raw/market` holds only `.meta.json` files, while the corpus exists in the protected checkout.

## Item 3: data_file_fingerprint

From reading: `data_file_fingerprint` (experiment_summary.py:73-96) joins a relative path onto `Path.cwd()` (lines 76-77), and on `not path.is_file()` logs a warning and returns three `None` values (78-84). It does not call `resolve_project_data_path`. The only call site is `run_experiment.py:194` with `cfg_w.data.filename`; `index_filename`, `pit_universe_csv` and `graph.sector_map_csv` are never fingerprinted.

Reproduction (`<SCRATCH>/repro3_fingerprint.py`):

```python
"""Item 3: data_file_fingerprint on a missing file, and divergence from what was loaded."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hdr import SCRATCH, section  # noqa: E402

import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from mci_gru.config import DataConfig  # noqa: E402
from mci_gru.data.data_manager import DataManager  # noqa: E402
from mci_gru.data.path_resolver import PROJECT_ROOT  # noqa: E402
from mci_gru.evaluation.experiment_summary import data_file_fingerprint  # noqa: E402

logger = logging.getLogger("run_experiment")

section("3a missing file: warning + nulls, no exception")
fp = data_file_fingerprint("data/raw/market/does_not_exist.csv", logger)
print(json.dumps(fp))

section("3b loaded via basename fallback, fingerprint of the configured path is null")
name = f"zz_audit_189_{os.getpid()}.csv"
configured = f"data/processed/{name}"
planted = PROJECT_ROOT / "data" / "raw" / "market" / name
planted.write_text("kdcode,dt,open,high,low,close,volume\nZZZ,2020-01-02,1,1,1,1,1\n")
try:
    df = DataManager(DataConfig(filename=configured)).load()
    print("DataManager.load succeeded, rows:", len(df))
    metadata = {"data_file": configured, **data_file_fingerprint(configured, logger)}
    print("run_metadata.json fragment (run_experiment.py:187,194):", json.dumps(metadata))
    print("sha256 of the bytes actually loaded:", hashlib.sha256(planted.read_bytes()).hexdigest())
finally:
    planted.unlink()
    print("planted file removed:", not planted.exists())

section("3c fingerprint is cwd-relative, not PROJECT_ROOT-relative")
def short(d):
    return {k: (v[:16] + "..." if isinstance(v, str) and k.endswith("sha256") else v) for k, v in d.items()}
here = os.getcwd()
os.chdir(SCRATCH)
try:
    print("cwd:", os.getcwd())
    print(json.dumps(short(data_file_fingerprint("mci_gru/__init__.py", logger))))
finally:
    os.chdir(here)
print("cwd:", os.getcwd())
print(json.dumps(short(data_file_fingerprint("mci_gru/__init__.py", logger))))
```

Output:

```text
python      : C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe
cwd         : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05
mci_gru     : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\mci_gru\__init__.py

======== 3a missing file: warning + nulls, no exception ========
LOG WARNING run_experiment: Data file not found at C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\raw\market\does_not_exist.csv -- skipping sha256
{"data_file_sha256": null, "data_file_size_bytes": null, "data_file_mtime_iso": null}

======== 3b loaded via basename fallback, fingerprint of the configured path is null ========
LOG INFO mci_gru.data.data_manager: Loading data from C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\raw\market\zz_audit_189_34460.csv...
LOG INFO mci_gru.data.data_manager:   Loaded 1 rows
LOG INFO mci_gru.data.data_manager:   Date range: 2020-01-02 to 2020-01-02
LOG INFO mci_gru.data.data_manager:   Stocks: 1
DataManager.load succeeded, rows: 1
LOG WARNING run_experiment: Data file not found at C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\data\processed\zz_audit_189_34460.csv -- skipping sha256
run_metadata.json fragment (run_experiment.py:187,194): {"data_file": "data/processed/zz_audit_189_34460.csv", "data_file_sha256": null, "data_file_size_bytes": null, "data_file_mtime_iso": null}
sha256 of the bytes actually loaded: 61b445efe0e92be9afc728c8f926c4ca5ec1437533dd968cb2dc40cfcf83e367
planted file removed: True

======== 3c fingerprint is cwd-relative, not PROJECT_ROOT-relative ========
cwd: <SCRATCH>
LOG WARNING run_experiment: Data file not found at <SCRATCH>\mci_gru\__init__.py -- skipping sha256
{"data_file_sha256": null, "data_file_size_bytes": null, "data_file_mtime_iso": null}
cwd: C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05
{"data_file_sha256": "42e2f2ddb08adab8...", "data_file_size_bytes": 816, "data_file_mtime_iso": "2026-08-01T23:20:56.745188+00:00"}
```

What the user would have seen: one WARNING line "Data file not found at ... -- skipping sha256" and a `run_metadata.json` carrying `"data_file_sha256": null, "data_file_size_bytes": null, "data_file_mtime_iso": null` next to a `data_file` string naming a path that was never opened, while the run trained on a file whose sha256 (`61b445ef...`) appears nowhere.
## Item 4: normalise_pit_intervals and build_pit_masks

From reading: `normalise_pit_intervals` (pit.py:55-90) lower-cases column names, accepts `constituent_ric` as an alias, requires `{kdcode, valid_from, valid_to}` (71-74), drops rows with NaN in any of the three (79), strips `kdcode` (80), formats dates through `pd.to_datetime` (81-82) and drops empty `kdcode`. It does not check ordering of `valid_from`/`valid_to`, overlap between intervals of one name, or presence of the name in any panel. `active_membership_mask` (154-172) is `any(start <= date <= end)` over a name's intervals, so overlaps are absorbed and an inverted interval is simply never true. `build_pit_masks` (224-243) computes `tradable = active & ready`; a name with no panel rows has `ready` False on every date. `select_universe` (pipeline.py:496-528) intersects the PIT names with the panel names via `available_kdcodes` (pit.py:149-150) without logging the difference. The row-filter path `_apply_pit_universe` (pipeline.py:226-236) is a separate implementation that inner-merges the panel with the intervals on `kdcode`, so a name with two overlapping intervals yields two copies of every row in the overlap.

Reproduction (`<SCRATCH>/repro4_pit.py`):

```python
"""Item 4: normalise_pit_intervals and build_pit_masks invariants."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hdr import SCRATCH, section, show_exc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mci_gru.data.pit import (  # noqa: E402
    active_kdcodes_in_period,
    active_membership_mask,
    build_pit_masks,
    load_pit_intervals,
    normalise_pit_intervals,
)
from mci_gru.pipeline import _apply_pit_universe  # noqa: E402

section("4a required columns are enforced")
show_exc(normalise_pit_intervals, pd.DataFrame({"kdcode": ["AAA"], "valid_from": ["2020-01-01"]}))

section("4b overlapping intervals (AAA), inverted interval (BBB), name absent from panel (CCC), NaN valid_to (DDD)")
pit_csv = SCRATCH / "r4_pit.csv"
pit_csv.write_text(
    "kdcode,valid_from,valid_to\n"
    "AAA,2020-01-01,2020-01-31\n"
    "AAA,2020-01-03,2020-01-20\n"
    "BBB,2020-01-10,2020-01-05\n"
    "CCC,2020-01-01,2020-01-31\n"
    "DDD,2020-01-01,\n"
)
norm = load_pit_intervals(str(pit_csv))
print(norm.to_string(index=False))
print("rows in:", 5, "| rows out:", len(norm), "| DDD present:", "DDD" in set(norm["kdcode"]))
print("AAA intervals:", len(norm[norm["kdcode"] == "AAA"]), "| BBB valid_from > valid_to:",
      bool((norm.loc[norm["kdcode"] == "BBB", "valid_from"] > norm.loc[norm["kdcode"] == "BBB", "valid_to"]).all()))

section("4c build_pit_masks on a panel that only has AAA and BBB")
dates = pd.bdate_range("2020-01-02", "2020-01-15").strftime("%Y-%m-%d").tolist()
panel = pd.DataFrame([(k, d, 10.0) for k in ["AAA", "BBB"] for d in dates], columns=["kdcode", "dt", "close"])
kd = ["AAA", "BBB", "CCC"]
sample_dates = dates[3:]
masks = build_pit_masks(panel, panel, kd, sample_dates, his_t=3, label_t=1, pit_intervals=norm)
for j, k in enumerate(kd):
    print(f"{k}: active={masks.active_member[:, j].astype(int).tolist()} ready={masks.feature_ready[:, j].astype(int).tolist()} tradable={masks.tradable[:, j].astype(int).tolist()}")
print("sample dates:", sample_dates)

section("4d select_universe basis: active_kdcodes_in_period intersects with the panel silently")
print("without available_kdcodes:", active_kdcodes_in_period(norm, "2020-01-01", "2020-01-31"))
print("with available_kdcodes={AAA,BBB}:", active_kdcodes_in_period(norm, "2020-01-01", "2020-01-31", {"AAA", "BBB"}))

section("4e row_filter path pipeline._apply_pit_universe: same CSV, no invariant beyond columns")
out = _apply_pit_universe(panel, str(pit_csv))
print("rows in:", len(panel), "| rows out:", len(out), "| per kdcode:", out["kdcode"].value_counts().to_dict())
print("AAA rows duplicated by the overlapping interval merge:", int(out.duplicated(["kdcode", "dt"]).sum()))

section("4f kdcode normalisation: strip only, no case fold; unparsable date raises from pandas")
print(normalise_pit_intervals(pd.DataFrame({"kdcode": [" aaa ", "AAA"], "valid_from": ["2020-01-01"] * 2, "valid_to": ["2020-01-31"] * 2}))["kdcode"].tolist())
show_exc(normalise_pit_intervals, pd.DataFrame({"kdcode": ["AAA"], "valid_from": ["not-a-date"], "valid_to": ["2020-01-31"]}))
```

Output:

```text
python      : C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe
cwd         : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05
mci_gru     : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\mci_gru\__init__.py

======== 4a required columns are enforced ========
RAISED ValueError: PIT intervals missing columns: ['valid_to']

======== 4b overlapping intervals (AAA), inverted interval (BBB), name absent from panel (CCC), NaN valid_to (DDD) ========
kdcode valid_from   valid_to
   AAA 2020-01-01 2020-01-31
   AAA 2020-01-03 2020-01-20
   BBB 2020-01-10 2020-01-05
   CCC 2020-01-01 2020-01-31
rows in: 5 | rows out: 4 | DDD present: False
AAA intervals: 2 | BBB valid_from > valid_to: True

======== 4c build_pit_masks on a panel that only has AAA and BBB ========
AAA: active=[1, 1, 1, 1, 1, 1, 1] ready=[1, 1, 1, 1, 1, 1, 1] tradable=[1, 1, 1, 1, 1, 1, 1]
BBB: active=[0, 0, 0, 0, 0, 0, 0] ready=[1, 1, 1, 1, 1, 1, 1] tradable=[0, 0, 0, 0, 0, 0, 0]
CCC: active=[1, 1, 1, 1, 1, 1, 1] ready=[0, 0, 0, 0, 0, 0, 0] tradable=[0, 0, 0, 0, 0, 0, 0]
sample dates: ['2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-13', '2020-01-14', '2020-01-15']

======== 4d select_universe basis: active_kdcodes_in_period intersects with the panel silently ========
without available_kdcodes: ['AAA', 'BBB', 'CCC']
with available_kdcodes={AAA,BBB}: ['AAA', 'BBB']

======== 4e row_filter path pipeline._apply_pit_universe: same CSV, no invariant beyond columns ========
rows in: 20 | rows out: 19 | per kdcode: {'AAA': 19}
AAA rows duplicated by the overlapping interval merge: 9

======== 4f kdcode normalisation: strip only, no case fold; unparsable date raises from pandas ========
['aaa', 'AAA']
RAISED DateParseError: Unknown datetime string format, unable to parse: not-a-date, at position 0
```

What the user would have seen: nothing at all for the four bad rows (one dropped, one overlapping, one inverted, one naming a stock the panel does not have); in masked-panel mode BBB and CCC would appear in `pit_breadth` diagnostics as never scoreable with no reason given, and in row-filter or normalisation-source mode the 10 AAA rows would become 19 rows with 9 duplicated `(kdcode, dt)` keys and no log line.

## Item 5: load_sector_map_csv

From reading: `load_sector_map_csv` (sector_edges.py:28-81) fails closed on a missing file (44-45), a zero-byte file (48-49) and missing `kdcode`/`sector` columns (51-55). Rows whose sector is blank or one of the placeholders in `_MISSING_SECTOR_VALUES` are skipped with `continue` (65-66) and never counted. A repeated `kdcode` with a different sector is collected into `conflicts` and reported once as a WARNING (69-79); with no `as_of_date` column both `as_of` values are `""`, `"" >= ""` is true, and the last row wins. A header-only file returns `{}` with an INFO line "0 kdcode(s)". `build_sector_edges` (84-148) isolates unmapped names and reports coverage on one INFO line (137-142); `kdcode` comparison is exact, so `aaa` and `AAA` are different names.

Reproduction (`<SCRATCH>/repro5_sector_map.py`):

```python
"""Item 5: load_sector_map_csv with duplicates, missing sector, empty file."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hdr import SCRATCH, section, show_exc  # noqa: E402

from mci_gru.graph.sector_edges import build_sector_edges, load_sector_map_csv  # noqa: E402

section("5a duplicate kdcode with different sectors (no as_of_date column)")
p = SCRATCH / "r5_dup_diff.csv"
p.write_text("kdcode,sector\nAAA,Tech\nAAA,Energy\nBBB,Tech\n")
print(load_sector_map_csv(str(p)))

section("5b duplicate kdcode with the same sector")
p = SCRATCH / "r5_dup_same.csv"
p.write_text("kdcode,sector\nAAA,Tech\nAAA,Tech\n")
print(load_sector_map_csv(str(p)))

section("5c missing / placeholder sector values")
p = SCRATCH / "r5_missing.csv"
p.write_text("kdcode,sector\nAAA,\nBBB,nan\nCCC,Unknown\nDDD,Tech\n")
m = load_sector_map_csv(str(p))
print(m)

section("5d zero-byte file")
p = SCRATCH / "r5_empty.csv"
p.write_text("")
show_exc(load_sector_map_csv, str(p))

section("5e header-only file, then build_sector_edges on a 3-name panel")
p = SCRATCH / "r5_header_only.csv"
p.write_text("kdcode,sector\n")
m = load_sector_map_csv(str(p))
print("map:", m)
ei, ew = build_sector_edges(["AAA", "BBB", "CCC"], m)
print("edge_index shape:", tuple(ei.shape), "| edge_weight shape:", tuple(ew.shape))

section("5f map names not on the panel / panel names not in the map / case sensitivity")
p = SCRATCH / "r5_mismatch.csv"
p.write_text("kdcode,sector\naaa,Tech\nBBB,Tech\nZZZ,Tech\n")
m = load_sector_map_csv(str(p))
ei, ew = build_sector_edges(["AAA", "BBB", "CCC"], m)
print("map:", m, "| edges:", ei.tolist())

section("5g wrong columns")
p = SCRATCH / "r5_wrong_cols.csv"
p.write_text("ticker,industry\nAAA,Tech\n")
show_exc(load_sector_map_csv, str(p))
```

Output:

```text
python      : C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe
cwd         : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05
mci_gru     : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\mci_gru\__init__.py

======== 5a duplicate kdcode with different sectors (no as_of_date column) ========
LOG WARNING mci_gru.graph.sector_edges: Sector map: 1 kdcode(s) carry more than one sector across snapshots (newest wins), e.g. ['AAA']
LOG INFO mci_gru.graph.sector_edges: Sector map: 2 kdcode(s) with a known sector from r5_dup_diff.csv
{'AAA': 'Energy', 'BBB': 'Tech'}

======== 5b duplicate kdcode with the same sector ========
LOG INFO mci_gru.graph.sector_edges: Sector map: 1 kdcode(s) with a known sector from r5_dup_same.csv
{'AAA': 'Tech'}

======== 5c missing / placeholder sector values ========
LOG INFO mci_gru.graph.sector_edges: Sector map: 1 kdcode(s) with a known sector from r5_missing.csv
{'DDD': 'Tech'}

======== 5d zero-byte file ========
RAISED ValueError: Empty CSV: <SCRATCH>\r5_empty.csv

======== 5e header-only file, then build_sector_edges on a 3-name panel ========
LOG INFO mci_gru.graph.sector_edges: Sector map: 0 kdcode(s) with a known sector from r5_header_only.csv
map: {}
LOG INFO mci_gru.graph.sector_edges: Sector edges: 0/3 name(s) mapped (0.0% coverage) across 0 sector(s); 3 name(s) isolated with no sector edges; 0 directed edge(s)
edge_index shape: (2, 0) | edge_weight shape: (0,)

======== 5f map names not on the panel / panel names not in the map / case sensitivity ========
LOG INFO mci_gru.graph.sector_edges: Sector map: 3 kdcode(s) with a known sector from r5_mismatch.csv
LOG INFO mci_gru.graph.sector_edges: Sector edges: 1/3 name(s) mapped (33.3% coverage) across 1 sector(s); 2 name(s) isolated with no sector edges; 0 directed edge(s)
map: {'aaa': 'Tech', 'BBB': 'Tech', 'ZZZ': 'Tech'} | edges: [[], []]

======== 5g wrong columns ========
RAISED ValueError: sector_map_csv must have columns kdcode, sector
```

What the user would have seen: one WARNING naming AAA for the conflicting duplicate; nothing for the same-sector duplicate or for the three names with missing sectors beyond the aggregate "1 kdcode(s) with a known sector"; a `ValueError: Empty CSV` for a zero-byte file; and for a header-only or case-mismatched map, an INFO line "0/3 name(s) mapped (0.0% coverage)" followed by a run that proceeds with an empty sector edge set.

## Item 6: preprocessing stage

From reading: `prepare_data` (pipeline.py:780-905) never calls `pd.to_datetime`, `drop_duplicates` or `duplicated` on the panel; dates are compared as strings everywhere (`filter_complete_stocks` data_manager.py:462, `split_by_period` 530-532, `compute_zscore_norm_stats` transforms.py:36). `filter_complete_stocks` (data_manager.py:450-483) counts rows per `kdcode` and keeps names whose count equals the number of distinct dates (470-471), so one duplicated row compensates for one missing session. `generate_time_series_features` drops duplicates with `keep="last"` (preprocessing.py:76-77). `compute_labels` (172-207) computes `groupby("kdcode")["close"].shift(-label_t)` over the row axis (192-193), so a duplicate row shifts every earlier label of that name by one session, and `pivot_table` (197) averages the two rows that share a `(dt, kdcode)` key with its default `aggfunc="mean"`. NaN labels are filled with the day mean and then 0 when `fill_missing` is true (200-205). NaN features are filled with the day mean and then 0 by `impute_feature_nans_by_day` (transforms.py:14-27). No code path tests `close <= 0`. The only date-format guard is `canonical_date` at the `CombinedDataset` boundary (data_manager.py:573-577, schedule.py:38-63), which sees `sample_dates` only. Row order in the CSV is not a failure mode: every consumer sorts (`filter_complete_stocks` data_manager.py:481, `generate_time_series_features` preprocessing.py:68, `compute_labels` 190).

Reproduction (`<SCRATCH>/repro6_preprocessing.py`):

```python
"""Item 6: duplicate (dt,kdcode), non-monotone/mixed dates, negative prices before tensors."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hdr import SCRATCH, section, show_exc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mci_gru.config import DataConfig  # noqa: E402
from mci_gru.data.data_manager import CombinedDataset, DataManager  # noqa: E402
from mci_gru.data.preprocessing import compute_labels, generate_time_series_features  # noqa: E402
from mci_gru.data.transforms import impute_feature_nans_by_day  # noqa: E402
from mci_gru.features.base import add_price_features  # noqa: E402

dates = pd.bdate_range("2020-01-02", "2020-01-10").strftime("%Y-%m-%d").tolist()  # 7 sessions
print("sessions:", dates)

section("6a duplicate (dt,kdcode) row + one missing session passes filter_complete_stocks")
rows = []
for d in dates:
    rows.append(("GOOD", d, 10.0))
for d in dates:
    if d == "2020-01-07":
        continue  # missing session
    rows.append(("DUPE", d, 10.0))
rows.append(("DUPE", "2020-01-03", 99.0))  # duplicate key, different close
panel = pd.DataFrame(rows, columns=["kdcode", "dt", "close"])
print("DUPE rows:", int((panel["kdcode"] == "DUPE").sum()), "| DUPE distinct sessions:", panel[panel["kdcode"] == "DUPE"]["dt"].nunique(), "| sessions:", len(dates))
dm = DataManager(DataConfig(train_start="2020-01-01", train_end="2020-01-10", val_start="2020-01-20", val_end="2020-01-31",
                            test_start="2020-02-10", test_end="2020-02-28"))
_, kd = dm.filter_complete_stocks(panel)
print("kdcode_list:", kd)

section("6b generate_time_series_features: last duplicate wins, no log")
feats = generate_time_series_features(panel, ["DUPE", "GOOD"], ["close"], his_t=2)
print("window ending 2020-01-03 for DUPE, close values:", feats[1, 0, :, 0].tolist(), "(source rows had 10.0 then 99.0)")

section("6c compute_labels: duplicate row shifts the per-stock row axis and pivot_table averages (label_t=2)")
print("DUPE rows sorted by dt:", panel[panel["kdcode"] == "DUPE"].sort_values("dt")[["dt", "close"]].values.tolist())
lab = compute_labels(panel, ["DUPE", "GOOD"], ["2020-01-02", "2020-01-03"], label_t=2, fill_missing=False)
print("labels (rows=dates 01-02, 01-03; cols=DUPE, GOOD):")
print(lab)
print("expected for a clean panel at close=10 throughout: all 0.0")
print("pivot_table default aggfunc (preprocessing.py:197):", "mean")

section("6d mixed date formats are compared as strings")
mixed = pd.DataFrame({"kdcode": ["AAA"] * 4, "dt": ["2020-01-08", "2020-01-09", "2020-1-10", "2020-01-13"], "close": [1, 2, 3, 4]})
print("sorted(dt.unique()):", sorted(mixed["dt"].unique()))
sub = mixed[(mixed["dt"] >= "2020-01-09") & (mixed["dt"] <= "2020-01-13")]
print("rows selected for [2020-01-09, 2020-01-13]:", sub["dt"].tolist(), "(2020-1-10 excluded)")
print("CombinedDataset boundary check (schedule.canonical_date):")
show_exc(CombinedDataset, np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), sample_dates=["2020-1-10"])

section("6e negative and NaN close: no detection, features and labels still produced")
neg = pd.DataFrame({"kdcode": ["AAA"] * 4, "dt": dates[:4], "open": [10, -10, 10, np.nan], "high": [11, -9, 11, 11],
                    "low": [9, -11, 9, 9], "close": [10, -10, 10, np.nan], "volume": [1, 1, 1, 1]})
pf = add_price_features(neg)
print(pf[["dt", "close", "daily_range", "overnight_return", "intraday_return"]].to_string(index=False))
lab = compute_labels(neg, ["AAA"], dates[:3], label_t=1, fill_missing=True)
print("labels with fill_missing=True:", lab.ravel().tolist())
print("impute_feature_nans_by_day on close:", impute_feature_nans_by_day(neg, ["close"])["close"].tolist())

section("6f is dt ever parsed or dedup'd on the stock-level path before tensors?")
import inspect  # noqa: E402
from mci_gru import pipeline  # noqa: E402
src = inspect.getsource(pipeline.prepare_data)
print("prepare_data mentions to_datetime:", "to_datetime" in src, "| drop_duplicates:", "drop_duplicates" in src, "| duplicated:", "duplicated" in src)
```

Output:

```text
python      : C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe
cwd         : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05
mci_gru     : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\mci_gru\__init__.py
sessions: ['2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10']

======== 6a duplicate (dt,kdcode) row + one missing session passes filter_complete_stocks ========
DUPE rows: 7 | DUPE distinct sessions: 6 | sessions: 7
LOG INFO mci_gru.data.data_manager: Filtering stocks with complete data...
LOG INFO mci_gru.data.data_manager:   Period: 7 trading days from 2020-01-02 to 2020-01-10
LOG INFO mci_gru.data.data_manager:   Stocks with complete data: 2
kdcode_list: ['DUPE', 'GOOD']

======== 6b generate_time_series_features: last duplicate wins, no log ========
  Allocating feature array: (5, 2, 2, 1)
  Building pivot (per-feature): [tqdm progress line collapsed]
                                                                     
window ending 2020-01-03 for DUPE, close values: [99.0, 10.0] (source rows had 10.0 then 99.0)

======== 6c compute_labels: duplicate row shifts the per-stock row axis and pivot_table averages (label_t=2) ========
DUPE rows sorted by dt: [['2020-01-02', 10.0], ['2020-01-03', 10.0], ['2020-01-03', 99.0], ['2020-01-06', 10.0], ['2020-01-08', 10.0], ['2020-01-09', 10.0], ['2020-01-10', 10.0]]
labels (rows=dates 01-02, 01-03; cols=DUPE, GOOD):
[[ 8.9         0.        ]
 [-0.44949496  0.        ]]
expected for a clean panel at close=10 throughout: all 0.0
pivot_table default aggfunc (preprocessing.py:197): mean

======== 6d mixed date formats are compared as strings ========
sorted(dt.unique()): ['2020-01-08', '2020-01-09', '2020-01-13', '2020-1-10']
rows selected for [2020-01-09, 2020-01-13]: ['2020-01-09', '2020-01-13'] (2020-1-10 excluded)
CombinedDataset boundary check (schedule.canonical_date):
RAISED ValueError: CombinedDataset sample_dates must be an unambiguous YYYY-MM-DD date (optionally with a time suffix), got '2020-1-10'

======== 6e negative and NaN close: no detection, features and labels still produced ========
  Added price features: daily_range, body_ratio, overnight_return, intraday_return
        dt  close  daily_range  overnight_return  intraday_return
2020-01-02   10.0          0.2               0.0              0.0
2020-01-03  -10.0         -0.2              -2.0              0.0
2020-01-06   10.0          0.2              -2.0              0.0
2020-01-07    NaN          NaN               0.0              0.0
labels with fill_missing=True: [0.0, 0.0, 0.0]
impute_feature_nans_by_day on close: [10.0, -10.0, 10.0, 0.0]

======== 6f is dt ever parsed or dedup'd on the stock-level path before tensors? ========
prepare_data mentions to_datetime: False | drop_duplicates: False | duplicated: False
```

What the user would have seen: "Stocks with complete data: 2" for a panel where DUPE is missing 2020-01-07; a feature window carrying 99.0 for a session whose first row said 10.0; training labels of 8.9 and -0.449 where a clean panel gives 0.0, with no line mentioning duplicates; a row dated `2020-1-10` silently excluded from a date range that contains it (the `CombinedDataset` guard only fires if such a date survives into `sample_dates`); and price features and labels computed through a close of -10.0 with nothing logged.

## Item 7: existing fail-closed precedents

Reproduction (`<SCRATCH>/repro7_fail_closed.py`):

```python
"""Item 7: existing fail-closed precedents and how they are configured."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hdr import section, show_exc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mci_gru.config import DataConfig, ExperimentConfig, ModelConfig  # noqa: E402
from mci_gru.data.preprocessing import (  # noqa: E402
    assert_training_labels_respect_embargo,
    purge_training_sessions_for_embargo,
)
from mci_gru.pipeline import _audit_pit_breadth, resolve_pit_context  # noqa: E402

section("7a pit_breadth_policy: config validation and the three policies")
show_exc(DataConfig, pit_breadth_policy="bogus")
show_exc(DataConfig, pit_min_scoreable_stocks=-1)
mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)  # 2 dates x 3 stocks: counts 2 and 1
for policy in ("error", "warn", "off"):
    print(f"policy={policy}:", end=" ")
    show_exc(_audit_pit_breadth, "train", ["2020-01-02", "2020-01-03"], mask, 2, policy)
print("defaults: pit_min_scoreable_stocks =", DataConfig().pit_min_scoreable_stocks, "| pit_breadth_policy =", repr(DataConfig().pit_breadth_policy))

section("7b embargo: calendar check in ExperimentConfig._validate_embargo (config-time, skippable)")
show_exc(ExperimentConfig, data=DataConfig(train_end="2023-12-31", val_start="2024-01-02"), model=ModelConfig(label_t=5))
cfg = ExperimentConfig(data=DataConfig(train_end="2023-12-31", val_start="2024-01-02", skip_embargo_check=True), model=ModelConfig(label_t=5))
print("with skip_embargo_check=True: constructed, label_t =", cfg.model.label_t)

section("7c embargo: session-axis check (unconditional, not skippable)")
dates = pd.bdate_range("2020-01-02", "2020-01-17").strftime("%Y-%m-%d").tolist()
panel = pd.DataFrame([("AAA", d) for d in dates], columns=["kdcode", "dt"])
print("purge:", purge_training_sessions_for_embargo(dates[:8], his_t=2, label_t=2))
show_exc(purge_training_sessions_for_embargo, dates[:3], 2, 2)
print("violation:", end=" ")
show_exc(assert_training_labels_respect_embargo, panel, ["AAA"], dates[:8], dates[8], 2)
print("compliant:", end=" ")
show_exc(assert_training_labels_respect_embargo, panel, ["AAA"], dates[:6], dates[8], 2)
print("unverifiable (panel too short):", end=" ")
show_exc(assert_training_labels_respect_embargo, panel.iloc[:6], ["AAA"], dates[:6], dates[8], 2)

section("7d other fail-closed guards on the load path")
show_exc(resolve_pit_context, ExperimentConfig(data=DataConfig(use_pit_universe=True)))
show_exc(DataConfig, source="csv", experiment_mode="bogus")
```

Output:

```text
python      : C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe
cwd         : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05
mci_gru     : C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\mci_gru\__init__.py

======== 7a pit_breadth_policy: config validation and the three policies ========
RAISED ValueError: pit_breadth_policy must be 'error', 'warn', or 'off', got 'bogus'
RAISED ValueError: pit_min_scoreable_stocks must be >= 0
policy=error: RAISED ValueError: PIT train breadth below 2 on 1 dates (first: 2020-01-03=1). Check ticker mapping/data outages or lower data.pit_min_scoreable_stocks with an explicit explanation.
policy=warn: LOG WARNING mci_gru.pipeline: Warning: PIT train breadth below 2 on 1 dates (first: 2020-01-03=1). Check ticker mapping/data outages or lower data.pit_min_scoreable_stocks with an explicit explanation.
C:\Users\magil\MCI-GRU\.claude\worktrees\ecstatic-chaplygin-010f05\mci_gru\config.py:825: UserWarning: Embargo check skipped but gaps are tight: train->val=2d, val->test=8d (label_t=5). Consider widening splits.
  self._validate_embargo()
returned: list
policy=off: returned: list
defaults: pit_min_scoreable_stocks = 450 | pit_breadth_policy = 'error'

======== 7b embargo: calendar check in ExperimentConfig._validate_embargo (config-time, skippable) ========
RAISED ValueError: Train/val gap (2 days) must be > label_t (5). Shift data.val_start later (e.g. after 2023-12-31 + 6 days) or set data.skip_embargo_check=true (not recommended).
with skip_embargo_check=True: constructed, label_t = 5

======== 7c embargo: session-axis check (unconditional, not skippable) ========
purge: ['2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09']
RAISED ValueError: Embargo purge of 2 session(s) leaves no training labels: 3 training sessions, his_t=2, label_t=2. Widen data.train_start..train_end or reduce model.his_t / model.label_t.
violation: RAISED ValueError: Train/val embargo violated on the session axis: 2 training label date(s) mature at or after 2020-01-14 on the union session axis (first: 2020-01-10 -> 2020-01-14); 2 (stock, date) label(s) mature at or after 2020-01-14 on their own session axis (first: AAA 2020-01-10 -> 2020-01-14). label_t=2 is a session count, not a calendar-day count; the training signal must be purged so labels mature before val_start.
compliant: returned: dict
unverifiable (panel too short): RAISED ValueError: Cannot verify the train/val embargo: the label panel ends before the last training label matures (2020-01-09 + 2 sessions, panel ends 2020-01-09). Refusing to treat unverifiable labels as compliant.

======== 7d other fail-closed guards on the load path ========
RAISED ValueError: data.use_pit_universe=true requires data.pit_universe_csv
RAISED ValueError: experiment_mode must be 'stock_level' or 'index_level', got 'bogus'
```

What the user would have seen: a `ValueError` at config construction for a bad policy value or a tight calendar gap, a `ValueError` naming the first offending date when PIT breadth falls below the threshold under the default `error` policy, and a `ValueError` that refuses to proceed when the embargo cannot be verified.

## Fail-closed precedents

The repository already has three shapes of fail-closed check on this path. Each is the pattern a data contract would follow.

1. Config-validated policy enum with a default of `error`. `DataConfig.pit_breadth_policy` (config.py:64) defaults to `"error"` and `__post_init__` (91-97) rejects anything outside `{"error", "warn", "off"}` and a negative `pit_min_scoreable_stocks` (default 450, line 63). The enforcement point `_audit_pit_breadth` (pipeline.py:251-274) computes a per-date diagnostic first, then branches: `off` or threshold 0 returns the diagnostic; `error` raises a `ValueError` whose message names the first five failing dates and tells the operator which config key to change "with an explicit explanation"; `warn` logs the same text. The diagnostic is written to `run_metadata.json` as `pit_breadth` regardless (run_experiment.py:193; pipeline.py:648-652). Reproduced in 7a.

2. Cheap calendar check at config time, skippable, paired with an authoritative data-backed check that is not skippable. `ExperimentConfig._validate_embargo` (config.py:838-862) raises unless `data.skip_embargo_check` is set, in which case it emits a `UserWarning`. `_embargo_training_sessions` (pipeline.py:317-342) then purges the training tail and calls `assert_training_labels_respect_embargo` (preprocessing.py:242-348), which takes no skip flag, raises on any violation on either the per-stock or the union session axis, raises when a label date is absent from the panel, and raises when the panel is too short to prove compliance ("Refusing to treat unverifiable labels as compliant", 299-304). The docstring at 253-254 states the design: the flag governs the calendar check only. Reproduced in 7b and 7c.

3. Strictness flag on a soft-fail. `features.regime_strict` (config.py:218, default False) turns the regime soft-fail in `load_auxiliary_data` (pipeline.py:166-183) and in `FeatureEngineer.transform` (registry.py:419-425) into a re-raise. The credit-spread soft-fail (pipeline.py:159-164; registry.py:397-402) and the VIX soft-fail (pipeline.py:152-157) have no such flag. Reproduced in 1f.

Other fail-closed guards on the path, all unconditional: `resolve_pit_context` (pipeline.py:453-462), `select_universe` when the PIT union is empty (511), `filter_complete_stocks` when no stock is complete (data_manager.py:476-477), `load_regime_inputs` column and series checks (data_manager.py:243-251, 385-389), `load_sector_map_csv` file/empty/column checks (sector_edges.py:44-55), `normalise_pit_intervals` column check (pit.py:71-74), `load_vix` not-found (data_manager.py:158-162), `canonical_date` at the dataset boundary (schedule.py:38-63), and `DataConfig.__post_init__` enums and date ordering (config.py:66-97). Reproduced in 7d, 1d, 1e, 4a, 5d, 5g, 6d.
## Observations

- The stock-level path from `DataConfig.filename` to tensors contains no check on the panel's columns beyond the accidental `KeyError` in two logging lines, no dtype coercion, no date parsing, no duplicate-key detection and no price-sign or NaN detection. The first place a non-canonical date is rejected is `CombinedDataset`, which sees only `sample_dates`.
- `resolve_project_data_path` can return a file from any of nine locations (cwd, project root, seven fallbacks) for one configured string; nothing records which one was chosen except the INFO line in `_load_from_csv`, and `run_metadata.json` stores the configured string, not the resolved path. `load_vix` passes a bare basename, so the launch directory participates in resolution.
- `data_file_fingerprint` resolves against cwd, not `PROJECT_ROOT`, and does not use the resolver, so the fingerprint is null for a run that loaded data through the fallback (3b), and the same call returns null or a digest depending on the launch directory (3c). Only `data.filename` is fingerprinted; `index_filename`, `pit_universe_csv` and `sector_map_csv` are not.
- `pit_universe_csv` is read with bare `pd.read_csv` in two places (pit.py:136, pipeline.py:228) and never through `resolve_project_data_path`, so its resolution rules differ from `data.filename`.
- The PIT interval loader enforces column presence only. Overlaps, inverted ranges and names absent from the panel are accepted; masked-panel mode absorbs them into `active`/`ready` masks with no diagnostic, and row-filter mode duplicates rows under overlaps.
- `filter_complete_stocks` uses row counts as a proxy for session coverage, so duplicate rows can mask missing sessions. `generate_time_series_features` and `compute_labels` resolve duplicates differently (`keep="last"` versus row shift plus mean), so a duplicated key produces features and labels from different rows.
- The sector map loader reports conflicts once as a WARNING and everything else as aggregate INFO counts; a map that matches zero names yields an empty edge set and the run continues.
- Every soft-fail on the auxiliary path (VIX, credit, regime) is a `logger.warning` followed by a constant fill (20 for VIX, 0.0 for credit and regime), and only the regime branch has a strictness flag.
- The package identity under test depends on `sys.path` order: running a script from outside the worktree resolved the editable install to the protected checkout until `PYTHONPATH` was set. The reproduction header asserts the resolved path for this reason.

## Silent failures reproduced

Fully silent (no log line about the condition):

- S1 (1a) `_load_from_csv` keeps duplicate `(dt, kdcode)` rows.
- S2 (1a) `_load_from_csv` keeps NaN and negative `close`.
- S3 (1a) `_load_from_csv` keeps mixed date formats as strings.
- S4 (1a) `_load_from_csv` keeps a non-numeric `volume` column (object dtype).
- S5 (1c) `load_index_series` keeps the last of duplicate `dt` rows.
- S6 (1c) `load_index_series` keeps NaN and negative `close`.
- S7 (1d') `load_vix` takes any `vix_data.csv` under the launch directory with any columns.
- S8 (1d'') `add_vix_features` fills a non-overlapping VIX series with the constant 20 on every row.
- S9 (1e') `load_regime_inputs` coerces a text cell to NaN and forward-fills the prior value.
- S10 (1e') `load_regime_inputs` keeps the last of duplicate `dt` rows.
- S12 (2a) basename fallback substitutes a same-named file from a different directory with no record.
- S13 (2b) a cwd-relative file shadows the project-root file.
- S17 (4b) a PIT row with NaN `valid_to` is dropped.
- S18 (4b, 4c) overlapping PIT intervals for one name are accepted.
- S19 (4b, 4c) an inverted PIT interval is accepted and never activates.
- S20 (4c, 4d) a PIT name absent from the panel is active-but-never-tradable and is intersected out of the union axis.
- S21 (4e) `_apply_pit_universe` duplicates panel rows under overlapping intervals.
- S22 (4f) `kdcode` case is not normalised in PIT intervals.
- S24 (5b) a duplicate `kdcode` with the same sector passes.
- S25 (5c) rows with missing or placeholder sectors are omitted without a per-row line.
- S26 (5e) a header-only sector map yields zero edges with an INFO count only.
- S27 (5f) unmapped or case-mismatched names are isolated with an INFO coverage line only.
- S28 (6a) `filter_complete_stocks` counts a duplicate row as coverage for a missing session.
- S29 (6b) `generate_time_series_features` keeps the last duplicate.
- S30 (6c) `compute_labels` produces 8.9 and -0.449 instead of 0.0 under one duplicate row.
- S31 (6d) a `2020-1-10` row is excluded from a string-compared date range.
- S32 (6e) negative `close` produces features and labels; NaN `close` is filled with the day mean then 0.

Warning-only (one `logger.warning`, run continues):

- S11 (1f) credit-spread load failure is swallowed and all seven credit features are 0.0.
- S14 (3a) `data_file_fingerprint` returns three nulls for a missing file.
- S15 (3b) fingerprint of the configured path is null while a different file was loaded via fallback.
- S16 (3c) `data_file_fingerprint` resolves against cwd, so the same call gives null or a digest depending on the launch directory.
- S23 (5a) duplicate `kdcode` with different sectors: last row wins.

Total: 32 silent or warning-only failure modes reproduced (27 fully silent, 5 warning-only), plus one stated from source without a script (only `data.filename` is fingerprinted, run_experiment.py:194).

## Appendix: shared header used by every reproduction

`<SCRATCH>/_hdr.py`:

```python
"""Shared header for issue-189 reproductions: prints interpreter identity and
captures every log record (DEBUG and up) to stdout so the report can show
exactly what a user would have seen."""
import logging
import os
import sys
import warnings
from pathlib import Path

import mci_gru

SCRATCH = Path(__file__).resolve().parent
print("python      :", sys.executable)
print("cwd         :", os.getcwd())
print("mci_gru     :", mci_gru.__file__)
assert "ecstatic-chaplygin-010f05" in mci_gru.__file__, "package did not resolve to the worktree"

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="LOG %(levelname)s %(name)s: %(message)s",
    force=True,
)
warnings.simplefilter("always")


def section(title: str) -> None:
    print()
    print("=" * 8, title, "=" * 8)


def show_exc(fn, *args, **kwargs):
    """Run fn and print the exception class and message instead of a traceback."""
    try:
        out = fn(*args, **kwargs)
        print("returned:", type(out).__name__)
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"RAISED {type(exc).__name__}: {exc}")
        return None
```
