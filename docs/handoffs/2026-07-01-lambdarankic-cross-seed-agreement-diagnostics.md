# LambdaRankIC Cross-Seed Agreement Diagnostics

Date: 2026-07-01

Scope: Item 4, saved-artifact-only cross-seed agreement diagnostics for the
110-name PIT GICS top-10-per-sector universe. No retraining, Colab training,
GitHub mutation, Drive download, or push was performed.

## Bottom Line

The requested agreement metrics were blocked locally. This checkout does not
contain the per-date `averaged_predictions/` CSVs or LambdaRankIC
`trade_journal.csv` outputs needed to compute daily top-10 Jaccard, overlap,
cross-stock rank correlation, trade/exit overlap, or stock-level score/rank
dispersion.

The local evidence that is available still argues against promoting
LambdaRankIC to default today. The recovered backtest report shows a sharp
2024 base-seed split: LambdaRankIC seed `161803` returned `40.22%` with `120`
trades, while seed `271828` returned `15.88%` with `362` trades and `3.62%`
cumulative cost. That is not a direct selection-overlap metric, but it is a
strong warning signal that the 2024 LambdaRankIC ranking/holdings path may not
be stable across repeated base seeds.

Keep pure IC as the launch/default objective. Keep LambdaRankIC experimental
until the saved predictions or trade journals are mounted locally and the
agreement metrics below are computed.

## What Was Computed vs Blocked

| Diagnostic | Status | Reason |
| --- | --- | --- |
| Daily top-10 Jaccard / overlap by date | Blocked | No local `averaged_predictions/` directories for the LambdaRankIC or pure-IC base-seed runs. |
| Rank correlation across stocks per date | Blocked | Requires same-date score vectors from prediction CSVs. |
| Trade / exit overlap | Blocked | No local `trade_journal.csv`, `daily_holdings.csv`, or `portfolio_composition.csv` for these runs. |
| Score/rank dispersion by stock | Blocked | Requires per-stock predictions from both base seeds. |
| Summary-level base-seed divergence | Available from reports | Uses recovered markdown/report rows only; not a substitute for daily agreement. |

## Local Artifact Inventory

Current workspace:
`C:\Users\magil\.codex\worktrees\fd6c\MCI-GRU`

Findings:

- `docs/handoffs/2026-06-30-sp500-top10-loss-backtest-all-years-recovered.md`
  is present and contains the recovered all-year backtest rows.
- No `averaged_predictions/` directories were found under this workspace.
- No `training_runs/` directories were found under this workspace.
- No LambdaRankIC trade journals or daily holdings outputs were found locally.
- `seed_results/`, `_uncertain/`, and `paper_trade/` contain older or unrelated
  backtest artifacts, not the 110-name LambdaRankIC repeated-base-seed
  prediction folders.
- The older local consolidation bundle exists at
  `C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation`,
  but it contains summary JSON/CSV files only, not prediction CSVs or trade
  journals.
- The exact Colab market/PIT CSV names from the run summary are not present in
  this checkout; only related `.meta.json` files were found under `data/raw`.
- While finalizing, two sibling untracked handoffs were also present:
  `docs/handoffs/2026-06-30-lambdarankic-rank-stability-diagnostics.md` and
  `docs/handoffs/2026-07-01-lambdarankic-rank-drop-sensitivity-replay.md`.
  They independently report the same missing prediction/data blocker and the
  same 2024 LambdaRankIC seed `271828` churn stress case.

## Summary-Level Seed Divergence

These rows come from
`docs/handoffs/2026-06-30-sp500-top10-loss-backtest-all-years-recovered.md`.
They are repeated experiment/base seeds, not the 20 ensemble member seeds.

| Year | Loss | Seed 161803 net | Seed 271828 net | Absolute gap | Trades 161803 | Trades 271828 | Read |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2022 | pure IC | -39.01% | -42.09% | 3.08 pp | 14 | 10 | Similar weak outcome. |
| 2022 | LambdaRankIC | -21.96% | -23.39% | 1.43 pp | 16 | 16 | Best apparent cross-seed agreement. |
| 2023 | pure IC | 22.51% | 19.12% | 3.39 pp | 70 | 26 | Returns close, turnover path differs. |
| 2023 | LambdaRankIC | 54.10% | 33.59% | 20.51 pp | 116 | 24 | Material base-seed divergence. |
| 2024 | pure IC | 46.84% | 34.32% | 12.52 pp | 48 | 62 | Return dispersion, moderate trade dispersion. |
| 2024 | LambdaRankIC | 40.22% | 15.88% | 24.34 pp | 120 | 362 | Main instability hotspot. |
| 2025 | pure IC | 26.89% | 35.86% | 8.97 pp | 20 | 26 | Pure IC still cost-controlled. |
| 2025 | LambdaRankIC | 32.28% | 22.10% | 10.18 pp | 70 | 144 | Positive, but higher turnover dispersion. |

Implication: LambdaRankIC has attractive mean rows, especially 2022 and 2023,
but the 2023/2024/2025 trade-count and return gaps are large enough that
default-readiness depends on the missing daily agreement diagnostics. In
particular, 2024 seed `271828` may be selecting or exiting very different names
from seed `161803`.

The sibling rank-stability handoff also frames the 2024 LambdaRankIC turnover as
`trade_names / (2 * top_k)`: seed `161803` is about `6.0x` cumulative one-way
top-10 turnover, while seed `271828` is about `18.1x`.

## Exact Inputs Needed

Drive/run root from the recovered report:

- Drive folder:
  `https://drive.google.com/drive/folders/1Co5Vd2dOSMrHUN5x_OzbJpkjJFocSHMo`
- Colab run root:
  `/content/drive/MyDrive/MCI-GRU-Ablations/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839`
- Local summary bundle:
  `C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation`

Key 2024 prediction folders that must be mounted or copied locally:

| Loss | Seed | Required `averaged_predictions` path |
| --- | ---: | --- |
| LambdaRankIC | 161803 | `/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/lambdarank_ic/2024/seed161803/top10_lambdarank_ic_2024_seed161803/20260629_181302/averaged_predictions` |
| LambdaRankIC | 271828 | `/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/lambdarank_ic/2024/seed271828/top10_lambdarank_ic_2024_seed271828/20260629_165356/averaged_predictions` |
| pure IC | 161803 | `/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/pure_ic/2024/seed161803/top10_pure_ic_2024_seed161803/20260629_160629/averaged_predictions` |
| pure IC | 271828 | `/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/pure_ic/2024/seed271828/top10_pure_ic_2024_seed271828/20260629_151620/averaged_predictions` |

All-year path extraction command from the local summary bundle:

```powershell
Import-Csv -LiteralPath 'C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation\training_rows_flat.csv' |
  Where-Object { $_.loss_key -in @('lambdarank_ic','pure_ic') } |
  Sort-Object loss_key,year,base_seed |
  Select-Object loss_key,year,base_seed,status,run_dir,predictions_dir |
  ConvertTo-Json -Depth 3
```

If trade journals are not already available beside the run directories, rerun
only the saved-prediction backtest after the prediction and data CSV folders are
local. Example for 2024 LambdaRankIC seed `161803`:

```powershell
.\.venv\Scripts\python.exe tests\backtest_sp500_daily.py `
  --predictions_dir '<LOCAL_RUN_ROOT>\training\lambdarank_ic\2024\seed161803\top10_lambdarank_ic_2024_seed161803\20260629_181302\averaged_predictions' `
  --data_file '<LOCAL_DATA_DIR>\sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv' `
  --pit_universe_csv '<LOCAL_CONSTITUENTS_DIR>\sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv' `
  --test_start 2024-01-22 `
  --test_end 2024-12-31 `
  --top_k 10 `
  --label_t 5 `
  --num_tests 1 `
  --adjustment_method bhy `
  --auto_save `
  --backtest_suffix _top10_tc_rankdrop `
  --transaction_costs `
  --spread 10 `
  --slippage 5 `
  --enable_rank_drop_gate `
  --min_rank_drop 30
```

Repeat that command for the paired seed paths. It emits
`daily_holdings.csv`, `portfolio_composition.csv`, `holdings_summary.csv`, and
`trade_journal.csv` beside each run's `averaged_predictions` parent directory.

## Commands Run

```powershell
git status --short
Get-ChildItem -Path . -Directory -Force | Select-Object -ExpandProperty Name
rg -n "prediction|averaged|trade|journal|seed|161803|271828|314159|LambdaRankIC|lambdarank|pure IC|pure_ic|backtest" "docs/handoffs/2026-06-30-sp500-top10-loss-backtest-all-years-recovered.md"
rg --files -g "*prediction*" -g "*predictions*" -g "*trade*" -g "*journal*" -g "*backtest*" -g "*lambdarank*" -g "*rankdrop*"
Get-ChildItem -Path . -Recurse -File -Force -ErrorAction SilentlyContinue -Include '*prediction*','*predictions*','*averaged*','*trade*','*journal*','*portfolio*','*position*','*holding*' | Select-Object -ExpandProperty FullName
Get-ChildItem -Path . -Recurse -Directory -Force -ErrorAction SilentlyContinue -Filter 'averaged_predictions' | Select-Object -ExpandProperty FullName
Get-ChildItem -Path . -Recurse -Directory -Force -ErrorAction SilentlyContinue -Filter 'training_runs' | Select-Object -ExpandProperty FullName
Test-Path -LiteralPath 'C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation'
Get-ChildItem -LiteralPath 'C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation' -Recurse -File -Force -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
Import-Csv -LiteralPath 'C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation\training_rows_flat.csv' | Where-Object { $_.loss_key -in @('lambdarank_ic','pure_ic') } | Sort-Object loss_key,year,base_seed | Select-Object loss_key,year,base_seed,status,run_dir,predictions_dir | ConvertTo-Json -Depth 3
Test-Path -LiteralPath 'data\raw\market\sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv'
Test-Path -LiteralPath 'data\raw\constituents\sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv'
Get-ChildItem -Path data\raw\market,data\raw\constituents -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'sp500_pit_gics_top10_mcap_monthly_20160104_20260622' } | ForEach-Object { $_.FullName }
```

## Decision Implication

The repeated/base-seed question remains open at the daily ranking level, but the
summary-level signal is already enough for a conservative decision: LambdaRankIC
should not become the default for the 110-name PIT GICS top-10-per-sector
universe yet. The next no-retrain step is to mount/copy the saved prediction
folders and compute:

1. Same-date top-10 set overlap and Jaccard for seeds `161803` vs `271828`.
2. Same-date Spearman rank correlation across all common stocks.
3. Trade/exit overlap from `trade_journal.csv`, especially 2024 LambdaRankIC.
4. Per-stock score/rank standard deviation across seeds, aggregated by year.
