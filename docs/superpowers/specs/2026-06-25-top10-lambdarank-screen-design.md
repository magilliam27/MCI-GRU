# Top-10 PIT LambdaRankIC Screen Design

Date: 2026-06-25
Status: Approved design

## Purpose

Run a first, cheap LambdaRankIC screen on the reduced S&P 500 PIT GICS top-10
universe instead of the full PIT S&P 500 union. The goal is to test whether a
complete same-day pairwise Rank IC objective behaves better on the 110-active-name
masked panel before spending a full 2022-2024, 20-model ensemble budget.

This design combines two existing workstreams:

- Reduced PIT GICS top-10 universe and multiyear Colab baseline.
- LambdaRankIC loss, `val_rank_ic` checkpointing, and pair-cap Colab runs.

## Goals

- Add a dedicated Colab launcher for a screen run on the reduced PIT GICS top-10
  universe.
- Keep the pure-IC reduced-universe baseline notebook unchanged.
- Keep LambdaRankIC disabled by default in base config.
- Use `data.pit_universe_mode=masked_panel` with the fixed PIT union axis.
- Use a pair cap that is complete for the reduced daily breadth.
- Produce Drive-backed artifacts that can be monitored without relying on
  notebook scrollback.
- Include saved-prediction backtest evidence, not only validation Rank IC.
- Guard the launcher with local notebook contract tests before any live Colab
  execution.

## Non-Goals

- Do not change `LambdaRankICLoss`, trainer checkpointing, metrics, or base
  training config for this screen.
- Do not replace the current pure-IC launch default.
- Do not use the static current 110-name top-10 file.
- Do not switch to complete-stock, stayer-only, or row-filter PIT behavior.
- Do not run the full 2022-2024 confirmation matrix in this first screen slice.
- Do not treat `val_rank_ic` as portfolio proof without backtest artifacts.

## Design Choice

Create a new generator and generated notebook:

- `scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py`
- `notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb`

This keeps the two existing launchers clean:

- `scripts/gen_sp500_pit_gics_top10_baseline_nb.py` remains the reduced-universe
  pure-IC baseline.
- `scripts/gen_lambdarank_ic_pit_nb.py` remains the full-PIT LambdaRankIC
  comparison and pair-cap launcher.

The new launcher borrows the reduced-universe data audit and rolling-window
contract from the top-10 baseline, and borrows GPU/heartbeat/result contracts
from the LambdaRankIC notebook.

## Screen Run

The first screen is one year, one objective, one complete-pair cap:

| Field | Value |
| --- | --- |
| Test year | 2022 |
| Train window | 2016-01-01 to 2020-12-31 |
| Validation window | 2021-01-08 to 2021-12-31 |
| Test window | 2022-01-08 to 2022-12-31 |
| Universe | Monthly PIT S&P 500 GICS top 10 by market cap per sector |
| Active breadth | 110 names per snapshot |
| PIT mode | `masked_panel` |
| Loss | `lambdarank_ic` |
| Checkpoint metric | `val_rank_ic` |
| Label type | raw 5-day returns |
| Pair cap | `8192` |
| Temperature | `1.0` |
| Screen budget | seed `314159`, 1 model, 40 epochs, patience 8 |

For 110 active names, the full same-day pair count is:

```text
110 * 109 / 2 = 5995
```

Because `8192 >= 5995`, the existing LambdaRankIC implementation will use all
valid same-day pairs for normal 110-name rows instead of deterministic sampling.
Rows with fewer finite labels also remain complete because they have even fewer
pairs.

## Data Contract

The launcher points at the intended 2016-start reduced PIT bundle:

- Market CSV:
  `data/raw/market/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv`
- PIT universe CSV:
  `data/raw/constituents/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv`
- Snapshot CSV:
  `data/raw/constituents/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_snapshots.csv`
- Constituent metadata:
  `data/raw/constituents/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_meta.json`
- Market metadata:
  `data/raw/market/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.meta.json`

The large 2016-start CSV files may be staged from Drive in Colab rather than
tracked in git. The notebook must fail fast if any required staged file is
missing.

The data audit must require:

- selector start is `2016-01-04`;
- snapshot count is `127`;
- min and max selected per snapshot are both `110`;
- PIT union contains `205` kdcodes;
- market metadata has no missing identifiers;
- selector-history blockers are empty for the 2022 train window.

## Hydra Overrides

The screen run preserves the reduced pure-IC baseline recipe except for the
objective fields and output naming:

```text
data.source=csv
data.filename=data/raw/market/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv
data.use_pit_universe=true
data.pit_universe_csv=data/raw/constituents/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv
data.pit_universe_mode=masked_panel
data.pit_min_scoreable_stocks=100
data.pit_breadth_policy=error
seed=314159
training.num_models=1
training.num_epochs=40
training.early_stopping_patience=8
training.learning_rate=5e-5
training.lr_scheduler=cosine
training.loss_type=lambdarank_ic
training.selection_metric=val_rank_ic
training.lambdarank_ic_max_pairs_per_day=8192
training.lambdarank_ic_temperature=1.0
training.label_type=returns
training.shuffle_train=true
model.label_t=5
graph.update_frequency_months=0
graph.top_k=0
graph.top_k_metric=corr
graph.use_multi_feature_edges=true
graph.drop_edge_p=0.1
features=with_momentum
features.include_global_regime=true
features.regime_strict=true
features.regime_include_subsequent_returns=false
tracking.enabled=false
tracking.log_predictions=false
```

The generated manifest should include the full override list for the run.

## Runtime Contract

The notebook must be run from visible Colab on a non-T4 G4/L4-class runtime.
It should:

- reject CPU and T4 runtime before training;
- load or require `FRED_API_KEY`;
- mount or read Drive data;
- refuse to reuse an existing run root;
- write heartbeat updates during setup, training, backtest, completion, and
  failure;
- leave enough Drive artifacts for another thread to monitor or resume.

## Artifacts

The run root should live under this pattern, where `RUN_TAG` is a UTC timestamp
such as `20260625_153000`:

```text
/content/drive/MyDrive/MCI-GRU-Ablations/sp500_gics_top10_lambdarank_ic_screen/RUN_TAG
```

Required top-level artifacts:

- `heartbeat.json`
- `data_audit.json`
- `lambdarank_ic_sp500_pit_gics_top10_screen_manifest.json`
- `training_results.csv`
- `training_results.json`
- `backtest_results.csv`
- `backtest_results.json`
- `run_summary.json`
- `logs/`
- `summaries/`
- `artifacts/`

Per-job artifacts should include Hydra overrides, training logs, run metadata,
training summary, evaluation summary, timing summary, averaged predictions, and
backtest outputs.

## Backtest Contract

The screen must run saved-prediction backtests using the same reduced PIT universe
CSV. The go/no-go question is:

Does complete-pair LambdaRankIC improve 2022 Rank IC and top-k/rank-drop backtest
behavior versus the existing reduced pure-IC baseline without worsening drawdown
or churn?

The screen report should compare at least:

- `mean_best_val_rank_ic`;
- test `avg_rank_ic`;
- test `avg_ic`;
- top-k total return;
- excess return versus benchmark;
- annualized Sharpe;
- max drawdown;
- turnover or rank-drop churn when available.

## Error Handling

- Missing staged data files fail before training.
- Data audit mismatch fails before training.
- T4, CPU, or unknown disallowed GPU fails before training.
- Missing `FRED_API_KEY` fails before training.
- Any failed training or backtest subprocess marks the heartbeat `FAILED` with
  the active job and log path.
- A partially completed run must still leave heartbeat, manifest, and logs.

## Local Tests

Add a notebook contract test:

- `tests/test_sp500_pit_gics_top10_lambdarank_ic_notebook.py`

It should assert:

- generator and notebook paths;
- reduced 2016-start GICS top-10 filenames;
- `YEARS = [2022]` or equivalent screen-year contract;
- `training.loss_type=lambdarank_ic`;
- `training.selection_metric=val_rank_ic`;
- `training.lambdarank_ic_max_pairs_per_day=8192`;
- `data.pit_universe_mode=masked_panel`;
- `data.pit_min_scoreable_stocks=100`;
- selector/history audit tokens;
- G4/L4 GPU guard and T4 rejection;
- `heartbeat.json`, manifest, training results, backtest results, and run
  summary artifact names;
- generated notebook code cells parse with `ast.parse`.

Focused verification commands:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_sp500_pit_gics_top10_lambdarank_ic_notebook.py tests\test_lambdarank_ic_loss.py tests\test_lambdarank_ic_config.py tests\test_lambdarank_ic_trainer.py tests\test_sp500_pit_gics_top10_baseline_notebook.py tests\test_sp500_pit_gics_top10_mcap_export.py tests\test_sp500_gics_top10_mcap_export.py tests\test_pit_saved_prediction_backtests.py -v --basetemp .tmp_pytest\pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m py_compile scripts\gen_sp500_pit_gics_top10_lambdarank_ic_nb.py
.\.venv\Scripts\ruff.exe check scripts\gen_sp500_pit_gics_top10_lambdarank_ic_nb.py tests\test_sp500_pit_gics_top10_lambdarank_ic_notebook.py
```

## Implementation Notes

Implement the notebook generator test-first. The first red test should fail
because the new generator and notebook do not exist. Then add the generator,
generate the notebook, and run the focused verification commands above.

The implementation should not modify the loss, trainer, metric, base config,
paper-trade code, or existing top-10 baseline notebook unless tests reveal a
direct contract gap.

## Completion Criteria

The design is implemented when:

- the new generator exists;
- the new notebook is generated from it;
- local contract tests and focused LambdaRankIC/top-10 regression tests pass;
- the generator encodes a 2022 screen run with complete-pair LambdaRankIC on the
  reduced PIT GICS top-10 universe;
- live Colab launch remains a separate explicit execution step after local
  validation.
