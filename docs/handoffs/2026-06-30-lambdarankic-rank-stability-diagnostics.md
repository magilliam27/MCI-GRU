# LambdaRankIC Rank-Stability Diagnostics

Date: 2026-06-30

Scope: Item 2 rank-stability diagnostics for saved predictions in the 110-name
PIT GICS top-10-per-sector loss comparison. This pass did not retrain, did not
launch Colab, did not fetch Drive artifacts, and did not mutate GitHub.

## Bottom Line

Daily prediction CSVs were not local in this worktree, so the full rank-stability
diagnostics could not be computed here.

What could be computed from recovered backtest summaries is the rank-drop-gated
turnover implied by total trades and transaction costs. That partial evidence is
already enough to confirm that LambdaRankIC 2024 seed `271828` is a genuine
churn outlier:

| Loss | Seed | 2024 net return | Trade names | Cost | Cumulative one-way top-10 turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| pure IC | 161803 | 46.84% | 48 | 0.48% | 2.4x |
| pure IC | 271828 | 34.32% | 62 | 0.62% | 3.1x |
| LambdaRankIC | 161803 | 40.22% | 120 | 1.20% | 6.0x |
| LambdaRankIC | 271828 | 15.88% | 362 | 3.62% | 18.1x |
| Portfolio-IC weight50 | 161803 | 41.20% | 26 | 0.26% | 1.3x |
| Portfolio-IC weight50 | 271828 | 44.16% | 22 | 0.22% | 1.1x |
| Portfolio-IC weight50 | 314159 | 41.86% | 24 | 0.24% | 1.2x |

Formula: `cumulative one-way turnover = trade_names / (2 * top_k)`, with
`top_k=10`. This matches the cost-implied turnover because the backtest cost
model uses `one_way_turnover * (spread + 2 * slippage)`, and the run used
`spread=0.001`, `slippage=0.0005`, so `18.1 * 0.002 = 0.0362`.

Interpretation: the 2024 seed `271828` LambdaRankIC result is not just a small
cost-accounting artifact. Under the rank-drop gate it still caused about 3x the
cumulative turnover of the other LambdaRankIC 2024 seed, about 5.8x the pure-IC
seed `271828` turnover, and about 15x the Portfolio-IC weight50 seed `271828`
turnover. The missing prediction CSVs are still required to prove whether those
trades came from broad rank instability, top-10 boundary churn, or large
held-name rank drops.

## Availability Check

Local searches found no `averaged_predictions` directories or prediction CSVs in
the current workspace, including ignored/untracked files.

Checked:

```powershell
rg --files -g "*averaged_predictions*" -g "*predictions*.csv" -g "*prediction*.csv"
rg --files -u -g "*averaged_predictions*" -g "*predictions*.csv" -g "*prediction*.csv"
Get-ChildItem -Force -Recurse -Directory -Filter averaged_predictions
```

The external consolidation bundle referenced by the recovered report was
readable, but it contains summary JSON/CSV only:

```text
C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation
```

It has `training_rows.json`, `training_rows_flat.csv`, `run_summary.json`, and
backtest summary files, but no daily prediction directories.

No local Drive mount was visible through `Get-PSDrive -PSProvider FileSystem`,
and these likely paths were absent:

```text
G:\My Drive\MCI-GRU-Ablations
C:\Users\magil\My Drive\MCI-GRU-Ablations
C:\Users\magil\Google Drive\My Drive\MCI-GRU-Ablations
C:\Users\magil\OneDrive\MCI-GRU-Ablations
```

## Exact Inputs Needed

Primary target:

```text
/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/lambdarank_ic/2024/seed271828/top10_lambdarank_ic_2024_seed271828/20260629_165356/averaged_predictions
```

Comparators:

```text
/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/lambdarank_ic/2024/seed161803/top10_lambdarank_ic_2024_seed161803/20260629_181302/averaged_predictions
/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/pure_ic/2024/seed271828/top10_pure_ic_2024_seed271828/20260629_151620/averaged_predictions
/content/mci_gru_runs/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839/training/pure_ic/2024/seed161803/top10_pure_ic_2024_seed161803/20260629_160629/averaged_predictions
```

Each prediction file is expected to be a date CSV loaded by
`tests/backtest_sp500_daily.py::load_predictions`, with columns:

```text
kdcode, dt, score
```

## Diagnostics To Run

Use the same rank convention as the backtest: sort `score` descending and use
one-based ranks. For ties, use `kdcode` as a deterministic secondary key.

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TOP_K = 10
MIN_RANK_DROP = 30
SPREAD = 0.001
SLIPPAGE = 0.0005


def load_predictions(predictions_dir: str | Path) -> pd.DataFrame:
    files = sorted(Path(predictions_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction CSV files found in {predictions_dir}")
    df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    required = {"kdcode", "dt", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction files missing columns: {sorted(missing)}")
    df = df[list(required)].copy()
    df["kdcode"] = df["kdcode"].astype(str)
    df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
    return df


def ranked_days(predictions: pd.DataFrame):
    for dt, day in predictions.groupby("dt", sort=True):
        ranked = day.sort_values(["score", "kdcode"], ascending=[False, True]).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        yield dt, ranked[["kdcode", "score", "rank"]]


def summarize_rank_stability(predictions_dir: str | Path) -> dict:
    previous_ranks = None
    previous_top = None
    previous_holdings = None

    rows = []
    all_rank_jumps = []
    prev_top_rank_jumps = []
    held_rank_jumps = []
    held_drop_checks = 0
    held_drop_gt_min = 0
    gated_trade_names = 0
    ungated_trade_names = 0

    for dt, ranked in ranked_days(load_predictions(predictions_dir)):
        ranks = dict(zip(ranked["kdcode"], ranked["rank"], strict=True))
        top = ranked.head(TOP_K)["kdcode"].tolist()

        if previous_ranks is None:
            holdings = top
            rows.append(
                {
                    "dt": dt,
                    "top10_overlap": np.nan,
                    "spearman_rank_autocorr": np.nan,
                    "avg_abs_rank_jump_all_common": np.nan,
                    "avg_abs_rank_jump_prev_top10": np.nan,
                    "held_drop_gt_30_count": 0,
                    "held_drop_gt_30_share": np.nan,
                    "gated_trade_names": TOP_K,
                    "ungated_trade_names": TOP_K,
                    "gated_one_way_turnover": TOP_K / (2 * TOP_K),
                    "ungated_one_way_turnover": TOP_K / (2 * TOP_K),
                }
            )
            previous_ranks = ranks
            previous_top = top
            previous_holdings = holdings
            gated_trade_names += TOP_K
            ungated_trade_names += TOP_K
            continue

        common = sorted(set(previous_ranks).intersection(ranks))
        prev_rank_values = pd.Series([previous_ranks[k] for k in common], dtype=float)
        curr_rank_values = pd.Series([ranks[k] for k in common], dtype=float)
        rank_jumps = (curr_rank_values - prev_rank_values).abs()
        all_rank_jumps.extend(rank_jumps.tolist())

        prev_top_common = [k for k in previous_top if k in previous_ranks and k in ranks]
        prev_top_jumps = [abs(ranks[k] - previous_ranks[k]) for k in prev_top_common]
        prev_top_rank_jumps.extend(prev_top_jumps)

        survivors = []
        held_drops_today = 0
        for kdcode in previous_holdings:
            if kdcode not in ranks:
                continue
            prev_rank = previous_ranks.get(kdcode)
            if prev_rank is None:
                survivors.append(kdcode)
                continue
            rank_drop = ranks[kdcode] - prev_rank
            held_rank_jumps.append(abs(rank_drop))
            held_drop_checks += 1
            if rank_drop >= MIN_RANK_DROP:
                held_drop_gt_min += 1
                held_drops_today += 1
            else:
                survivors.append(kdcode)

        survivor_set = set(survivors)
        refill = [kdcode for kdcode in ranked["kdcode"].tolist() if kdcode not in survivor_set]
        holdings = survivors + refill[: max(0, TOP_K - len(survivors))]

        gated_trades_today = len(set(previous_holdings) - set(holdings)) + len(
            set(holdings) - set(previous_holdings)
        )
        ungated_trades_today = len(set(previous_top) - set(top)) + len(set(top) - set(previous_top))
        gated_trade_names += gated_trades_today
        ungated_trade_names += ungated_trades_today

        rows.append(
            {
                "dt": dt,
                "top10_overlap": len(set(previous_top).intersection(top)) / TOP_K,
                "spearman_rank_autocorr": prev_rank_values.corr(curr_rank_values, method="spearman"),
                "avg_abs_rank_jump_all_common": float(rank_jumps.mean()) if len(rank_jumps) else np.nan,
                "avg_abs_rank_jump_prev_top10": float(np.mean(prev_top_jumps)) if prev_top_jumps else np.nan,
                "held_drop_gt_30_count": held_drops_today,
                "held_drop_gt_30_share": held_drops_today / max(len(previous_holdings), 1),
                "gated_trade_names": gated_trades_today,
                "ungated_trade_names": ungated_trades_today,
                "gated_one_way_turnover": gated_trades_today / (2 * TOP_K),
                "ungated_one_way_turnover": ungated_trades_today / (2 * TOP_K),
            }
        )

        previous_ranks = ranks
        previous_top = top
        previous_holdings = holdings

    daily = pd.DataFrame(rows)
    cumulative_gated_turnover = gated_trade_names / (2 * TOP_K)
    cumulative_ungated_turnover = ungated_trade_names / (2 * TOP_K)
    return {
        "prediction_days": int(len(daily)),
        "mean_daily_top10_overlap": float(daily["top10_overlap"].mean()),
        "mean_spearman_rank_autocorr": float(daily["spearman_rank_autocorr"].mean()),
        "mean_abs_rank_jump_all_common": float(np.mean(all_rank_jumps)) if all_rank_jumps else np.nan,
        "mean_abs_rank_jump_prev_top10": float(np.mean(prev_top_rank_jumps)) if prev_top_rank_jumps else np.nan,
        "mean_abs_rank_jump_held": float(np.mean(held_rank_jumps)) if held_rank_jumps else np.nan,
        "held_drop_gt_30_count": int(held_drop_gt_min),
        "held_drop_gt_30_share": held_drop_gt_min / held_drop_checks if held_drop_checks else np.nan,
        "days_with_held_drop_gt_30": int((daily["held_drop_gt_30_count"] > 0).sum()),
        "gated_trade_names": int(gated_trade_names),
        "ungated_trade_names": int(ungated_trade_names),
        "gated_cumulative_one_way_turnover": float(cumulative_gated_turnover),
        "ungated_cumulative_one_way_turnover": float(cumulative_ungated_turnover),
        "gated_cost_at_20bps": float(cumulative_gated_turnover * (SPREAD + 2 * SLIPPAGE)),
        "ungated_cost_at_20bps": float(cumulative_ungated_turnover * (SPREAD + 2 * SLIPPAGE)),
        "daily": daily,
    }
```

Compare the returned summary for the four 2024 input directories above. The
decision-critical contrasts are:

- LambdaRankIC `271828` versus LambdaRankIC `161803`.
- LambdaRankIC `271828` versus pure IC `271828`.
- LambdaRankIC `271828` gated turnover versus its ungated top-10 turnover.
- Held-drop share and days-with-drop for `rank_drop >= 30`.

## What The Evidence Implies Now

The summary-level turnover evidence supports the user's hypothesis directionally:
LambdaRankIC seed `271828` did not merely pay a few extra basis points from
near-cutoff noise. It produced enough rank-gated position changes to turn the
top-10 book over 18.1 cumulative one-way times in 2024.

However, the exact mechanism remains unproven until daily predictions are staged:

- Daily top-10 overlap will show whether the desired top-10 set itself is
  unstable.
- Consecutive-day Spearman rank autocorrelation will show whether the whole
  100-plus-name ranking is unstable.
- Average absolute rank jumps, especially for previous top-10 and held names,
  will show whether changes are broad and large.
- Held-name `rank_drop >= 30` count/share will directly test whether the
  rank-drop gate is being triggered by large rank collapses.
- Gated versus ungated implied turnover will show how much churn the gate
  prevents and how much it fails to prevent.

Until those CSV-backed diagnostics are computed, LambdaRankIC should remain
experimental rather than default for the 110-name PIT universe. The strongest
current statement is narrower: 2024 seed `271828` is a severe turnover outlier
under the exact rank-drop-gated backtest semantics.

## Commands Run

```powershell
git status --short
rg --files
rg --files -u -g "*averaged_predictions*" -g "*predictions*.csv" -g "*prediction*.csv" -g "*backtest_results*.csv" -g "*training_results*.csv" -g "*backtest_rows*.json" -g "*backtest_rows*.csv" -g "*run_summary*.json" -g "*.parquet"
Get-ChildItem -Force -Recurse -Directory -Filter averaged_predictions
Get-Content -Raw docs/ARCHITECTURE.md
Get-Content -Raw docs/handoffs/2026-06-30-sp500-top10-loss-backtest-all-years-recovered.md
Get-Content -Raw docs/handoffs/2026-06-30-sp500-top10-loss-backtest-comparison.md
rg -n "averaged_predictions|rank_drop|rank-drop|rank_drop_gate|min_rank_drop|top_k|turnover|transaction|pred" tests/backtest_sp500_daily.py scripts/run_pit_saved_prediction_backtests.py scripts/run_saved_prediction_selection_audit.py mci_gru/evaluation/portfolio.py
Get-ChildItem -Force C:/Users/magil/.codex/worktrees/559c/MCI-GRU/artifacts/2026-06-30-sp500-top10-loss-seed-matrix-consolidation
Get-ChildItem -Force -Recurse -Directory -Filter averaged_predictions C:/Users/magil/.codex/worktrees/559c/MCI-GRU/artifacts/2026-06-30-sp500-top10-loss-seed-matrix-consolidation
rg --files -u C:/Users/magil/.codex/worktrees/559c/MCI-GRU/artifacts/2026-06-30-sp500-top10-loss-seed-matrix-consolidation
Import-Csv C:/Users/magil/.codex/worktrees/559c/MCI-GRU/artifacts/2026-06-30-sp500-top10-loss-seed-matrix-consolidation/training_rows_flat.csv
Get-PSDrive -PSProvider FileSystem
```
