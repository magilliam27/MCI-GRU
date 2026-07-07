# LambdaRankIC Stability Diagnostics Coordination

Date: 2026-07-01

Purpose: coordinate the four no-retraining stability checks requested after the recovered 110-name PIT GICS top-10-per-sector loss backtest comparison.

## Bottom Line

Four subagents checked the first four stability items:

- Cost sensitivity replay
- Rank-stability diagnostics
- Rank-drop gate sensitivity replay
- Cross-seed agreement diagnostics

No new replay rows or prediction-level diagnostics were produced in this workspace. The shared blocker is real and consistent: this worktree has summary artifacts and Drive logs, but not the daily `averaged_predictions/` CSV folders or the top-10 PIT market/universe CSVs required for saved-prediction replay.

The next step is not more training. It is to run saved-prediction-only replay
and diagnostics in Colab/Drive, where the run artifacts live. Do not run these
replays on the user's local PC.

## Current Evidence From The Four Checks

### 1. Cost Sensitivity

Status: blocked locally.

Missing inputs:

- No local `averaged_predictions/` folders under `C:\Users\magil\.codex\worktrees\fd6c\MCI-GRU`.
- Missing top-10 PIT market CSV:
  `data/raw/market/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv`
- Missing top-10 PIT universe CSV:
  `data/raw/constituents/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv`

Decision implication: lower personal-account costs would help LambdaRankIC, but cost alone does not explain the bad 2024 seed. The recovered 2024 LambdaRankIC seed `271828` log had:

- Net total return: `15.88%`
- Gross total return before costs: `20.14%`
- Transaction cost: `3.62%`

Even at zero modeled transaction cost, that row remains well below 2024 pure IC and Portfolio-IC weight50 rows.

Detailed note: `docs/handoffs/2026-07-01-lambdarankic-cost-sensitivity` was not written by that worker, but its final finding is captured in this coordination note.

### 2. Rank-Stability Diagnostics

Status: daily prediction metrics blocked locally; summary-level turnover evidence computed.

Key summary-level result for 2024:

| Loss | Seed | Net return | Trades | Cost | Cumulative one-way top-10 turnover |
|---|---:|---:|---:|---:|---:|
| pure IC | 161803 | 46.84% | 48 | 0.48% | 2.4x |
| pure IC | 271828 | 34.32% | 62 | 0.62% | 3.1x |
| LambdaRankIC | 161803 | 40.22% | 120 | 1.20% | 6.0x |
| LambdaRankIC | 271828 | 15.88% | 362 | 3.62% | 18.1x |
| Portfolio-IC weight50 | 161803 | 41.20% | 26 | 0.26% | 1.3x |
| Portfolio-IC weight50 | 271828 | 44.16% | 22 | 0.22% | 1.1x |
| Portfolio-IC weight50 | 314159 | 41.86% | 24 | 0.24% | 1.2x |

Formula used: `trade_names / (2 * top_k)` with `top_k=10`. This matches the cost model because `18.1 * (10 bps spread + 2 * 5 bps slippage) = 3.62%`.

Decision implication: the 2024 LambdaRankIC seed `271828` issue is a genuine churn outlier, not a small cost-accounting artifact. Prediction CSVs are still needed to prove whether the churn came from broad rank instability, top-10 boundary churn, or large held-name rank drops.

Detailed note: `docs/handoffs/2026-06-30-lambdarankic-rank-stability-diagnostics.md`

### 3. Rank-Drop Gate Sensitivity

Status: replay blocked locally; exact commands prepared.

The current recovered rows all used `min_rank_drop=30`. The proposed replay grid is:

- `20`
- `30`
- `50`
- `75`

Priority rows:

- 2024 LambdaRankIC seed `271828`
- 2024 LambdaRankIC seed `161803`
- 2025 LambdaRankIC seeds if cheap
- 2023 LambdaRankIC seeds if cheap

Decision implication: stronger hysteresis is plausible, but not proven. LambdaRankIC default becomes more defensible only if stricter gates tame the 2024 churn case without giving up the 2022/2023/2025 alpha.

Detailed note and commands: `docs/handoffs/2026-07-01-lambdarankic-rank-drop-sensitivity-replay.md`

### 4. Cross-Seed Agreement

Status: prediction-level daily agreement metrics blocked locally.

Summary-level 2024 base-seed split:

- LambdaRankIC seed `161803`: `40.22%` net, `120` trades.
- LambdaRankIC seed `271828`: `15.88%` net, `362` trades.

Decision implication: this does not prove low top-10 Jaccard or low rank correlation, but it is enough to require daily cross-seed agreement checks before making LambdaRankIC the project default.

Detailed note: `docs/handoffs/2026-07-01-lambdarankic-cross-seed-agreement-diagnostics.md`

## Artifact Status

Readable local consolidation bundle:

```text
C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation
```

This contains summary JSON/CSV files only, including `training_rows.json` and `training_rows_flat.csv`; it does not contain prediction CSVs.

Drive run root:

```text
https://drive.google.com/drive/folders/1Co5Vd2dOSMrHUN5x_OzbJpkjJFocSHMo
```

Drive listing shows top-level `heartbeat.json`, `manifest.json`, `run_summary.json`, `logs`, `summaries`, and `artifacts`. The `artifacts/local_run_root/training` tree is present, but targeted Drive search for `top10_lambdarank_ic_2024_seed271828` found logs only:

- `training_top10_lambdarank_ic_2024_seed271828.log`
- `backtest_top10_lambdarank_ic_2024_seed271828.log`

It did not expose the `averaged_predictions` CSV folder via search.

### Drive CSV Prediction Search Update

Follow-up Drive search on 2026-07-01 found real saved-prediction CSV folders,
but not for the current repeated-seed LambdaRankIC 2023/2024 rows.

Current run, final artifact upload:

- `averaged_predictions` folder found:
  `https://drive.google.com/drive/folders/1hEh23PUtfD2104gxuVF_fKDJshYzFg8v`
- Parent chain:
  `artifacts/local_run_root/training/portfolio_ic_weight50/2025/seed161803/top10_portfolio_ic_weight50_2025_seed161803/20260630_090258/averaged_predictions`
- This is a real CSV prediction tree, but it is only the final visible
  `portfolio_ic_weight50` 2025 seed `161803` artifact. Drive search for
  current-run `top10_lambdarank_ic_*` folders, `seed271828` folders, and
  timestamp folders such as `20260629_122641`, `20260629_165356`, and
  `20260629_181302` did not find corresponding CSV artifact folders.

Current run, summary-only LambdaRankIC rows:

- `summaries/completed_rows/lambdarank_ic` folder:
  `https://drive.google.com/drive/folders/1Mh-AQ6E_qoEfeqtofam8dBNN8KuF5zu8`
- It contains year folders `2022`, `2023`, `2024`, and `2025`, with
  `seed161803.json` / `seed271828.json` summary files.
- Those JSON files record Colab-local `predictions_dir` paths, for example
  `/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/lambdarank_ic/2023/seed161803/top10_lambdarank_ic_2023_seed161803/20260629_122641/averaged_predictions`.
- The corresponding Drive folders were not found by exact timestamp, seed,
  or `top10_lambdarank_ic_*` folder searches.

Reusable older LambdaRankIC full seed `314159` prediction CSV folders were
found in Drive:

| Year | Run folder | Timestamp folder | `averaged_predictions` folder |
|---:|---|---|---|
| 2022 | `sp500_pit_gics_top10_lambdarank_ic_2022_full_seed314159` / `1sG5Q43kJWy5neGvquXda2IwXu5LfFQgo` | `20260626_172341` / `1xNHe-wfPZDCqdVYMFLRjbYuFu9e2YkyJ` | `1IJ62jNdpLbFW4Kuc9l68LkG3NmTmS9bd` |
| 2023 | `sp500_pit_gics_top10_lambdarank_ic_2023_full_seed314159` / `13y10NndupBoVrJR1FeYv3Kit2bjjuT2A` | `20260626_184201` / `1A-3uIw5Q9mvjbv5Kl2-RWzurhVxGC7Os` | `1tp-BEvU2yPMxnE_c6ul3o1gVi-ZyeF3n` |
| 2024 | `sp500_pit_gics_top10_lambdarank_ic_2024_full_seed314159` / `1MMbuUqRq0XK6fyZxnkashKeP_bXNHFpZ` | `20260626_192244` / `1EWMDtxP7JvC56anK0Bv4SfvM7xoEiriX` | `1KHK3TSjtjz4Ft-XTU5DkmVtgcklgKTwt` |
| 2025 | `sp500_pit_gics_top10_lambdarank_ic_2025_full_seed314159` / `1Se-SGZFgWjNAlWkj8-VjV9sLjqQaTq5b` | `20260627_012707` / `1myeVj1kbiVEOxt0CLTQf_-9-u3enc4wE` | `1PIV6uuwKDBKAGYMIRsvepCD7cgo9CgO3` |

Sample CSV verification:

- `2024-03-27.csv` in the 2024 full seed `314159` folder:
  `https://drive.google.com/file/d/1fTPB0GLrJXGCbPN2SATAQtuLtJxqAS7M/view`
- Fetched content starts with `kdcode,dt,score` and contains the expected
  110-name PIT universe rows for that date.

Implication: Drive has usable prediction CSVs for the older LambdaRankIC
seed `314159` full runs, and one current Portfolio-IC 2025 seed `161803`
artifact. It still does not expose the current repeated-seed LambdaRankIC
`161803` / `271828` 2023-2024 CSV folders needed for the main cross-seed
and rank-stability diagnostics.

Additional older Drive search:

- Drive family root found:
  `sp500_gics_top10_loss_comparison_repeated_seeds` /
  `https://drive.google.com/drive/folders/1lhL-tnUoShh8ImNdTED_sRBOf_dqcOim`
- It contains `20260629_011839`, `20260629_005700`, and `20260629_005322`.
  The two earlier timestamp roots are summary/log/heartbeat only. They do not
  contain training artifact trees.
- Content search for `165356` confirmed that the 2024 LambdaRankIC seed
  `271828` backtest loaded predictions successfully from the Colab-local path
  and read `27262` predictions over `248` dates / `132` stocks, but the
  corresponding Drive `averaged_predictions` folder still did not surface.
- Searches for trashed `top10_lambdarank_ic` and post-2026-06-29
  `averaged_predictions` folders also did not surface the missing current
  repeated-seed folders.

Older pair-cap-1024 LambdaRankIC prediction folders were found for the same
base seeds, but they are not the 110-name top10-per-sector loss-matrix universe:

| Run | Timestamp folder | `averaged_predictions` folder | Caveat |
|---|---|---|---|
| `lambdarank_ic_pairs1024_2023_seed271828` / `1MlN5D0SOcHlbm312nVrMNlqJ2KB8QVAM` | `20260626_004442` / `1AwDf1tQmB8rsZmEODGfU0nSK7yMPg6B7` | `1x9pJqekte8IMxiPo4WGcWwjvr7Ev8hsP` | broader PIT universe |
| `lambdarank_ic_pairs1024_2023_seed161803` / `1Pjk8m0ZxBUd-L1JLe-2N_jJue8_y0X6o` | `20260626_012206` / `16F9WLjeuBoSyOfEFIKZBNOSC_mbvnOHQ` | `1SLdysWpUnJ6hiPr31A42S6Ekg63wufp4` | broader PIT universe |
| `lambdarank_ic_pairs1024_2024_seed271828` / `1v5nd23_PURZLX5zn-9K0UvImN2qFOTQO` | `20260626_033721` / `1QdFjLS5uH82oPa2FOnCfhKxKX67CglQv` | `156jEfyFJ52I5CXIkvTjX4d9jsdTnPtYq` | broader PIT universe |
| `lambdarank_ic_pairs1024_2024_seed161803` / `10Aep0-S9LtShLAIAb8rov_pWjhHzUVwV` | `20260626_041124` / `1hWlrDHARJeq-geVjf6JiEzvKmpx5D8Fq` | `13KOtrHhOE3xCnvCeUxRNOpRLzUEEWrNm` | broader PIT universe |

Sample verification for the pair-cap-1024 block:

- `2024-12-26.csv` in `156jEfyFJ52I5CXIkvTjX4d9jsdTnPtYq` starts with
  `kdcode,dt,score`, but includes broad-universe names such as `A.N`,
  `ACGL.OQ`, `ADM.N`, etc. This is not the compact 110-name top10 universe
  seen in the older top10 seed `314159` CSVs.

Net: the old Drive files exist, including same-seed LambdaRankIC CSVs from the
pair-cap-1024 experiment. For the 110-name top10-per-sector objective comparison,
the only older LambdaRankIC top10 CSV folders found remain seed `314159`
for 2022-2025; current repeated-seed top10 LambdaRankIC CSV folders for
`161803` / `271828` remain missing from Drive search.

## Exact Inputs To Stage

Primary problem row:

```text
/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/lambdarank_ic/2024/seed271828/top10_lambdarank_ic_2024_seed271828/20260629_165356/averaged_predictions
```

Core comparators:

```text
/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/lambdarank_ic/2024/seed161803/top10_lambdarank_ic_2024_seed161803/20260629_181302/averaged_predictions
/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/pure_ic/2024/seed271828/top10_pure_ic_2024_seed271828/20260629_151620/averaged_predictions
/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/pure_ic/2024/seed161803/top10_pure_ic_2024_seed161803/20260629_160629/averaged_predictions
```

Required data CSVs:

```text
data/raw/market/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv
data/raw/constituents/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv
```

## Recommended Next Step

Run a saved-prediction-only diagnostic job in Colab/Drive against the existing
run root. Do not retrain. Do not run the heavy replay/diagnostic jobs on the
user's local PC.

Order:

1. Run rank-stability and cross-seed agreement on 2024 LambdaRankIC `161803` vs `271828`.
2. Replay 2024 LambdaRankIC `271828` at rank-drop gates `20/30/50/75`.
3. Replay the same row at cost assumptions `10+5`, `5+2.5`, `1+0.5`, and `0+0`.
4. Only if 2024 becomes controlled, expand to 2023 and 2025 LambdaRankIC rows.

Decision gate for promotion:

- LambdaRankIC can become the personal-trading default only if the 2024 seed `271828` instability is explained or mitigated.
- Cost-only mitigation is probably insufficient because gross 2024 return was still only `20.14%`.
- A stronger-rank-gate or stability-filter mitigation could change the conclusion if it materially reduces churn while keeping LambdaRankIC's 2022/2023 advantage.
