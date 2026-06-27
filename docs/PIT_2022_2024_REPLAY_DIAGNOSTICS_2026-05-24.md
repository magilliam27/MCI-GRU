# PIT 2022/2024 Replay Diagnostics

Date: 2026-05-24

## Scope

This memo follows `docs/handoffs/2026-05-24-pit-2022-2024-diagnostics.md`.
It uses saved Option A predictions and backtest artifacts from
`20260520_183538`. It does not retrain models, change the frozen default
recipe, or treat the diagnostic `label_t=21` replay as training-matched
evidence.

The goal is to separate the two weak-year stories:

- 2022: regime/style exposure failure.
- 2024: valid gross signal weakened by turnover, rank-gate churn, and costs.

## Provenance

- Frozen recipe:
  `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1`
- Saved run root:
  `.codex_tmp/pit_option_a_extract_20260520_183538/20260520_183538`
- Local diagnostic output:
  `.codex_tmp/pit_option_a_extract_20260520_183538/20260520_183538/summaries/pit_2022_2024_diagnostics_20260524/`
- Colab confirmation output:
  `/content/drive/MyDrive/MCI-GRU-Ablations/pit_repeated_seed_replication/20260520_183538/summaries/pit_2022_2024_diagnostics_20260524_colab/`
- Runner mode: replay-only, CPU-compatible saved-artifact analysis.

The local artifact is the full report source for the 2024 churn table below:
it contains the freshly computed no-gate/gate50 rows plus the existing gate30
baseline rows from the saved sensitivity replay. The recovered Colab notebook
returned `returncode 0` and wrote the Drive path above; because it resumed from
an interrupted cached output directory, its summary contains the freshly
computed no-gate/gate50 churn rows but not the appended gate30 baseline rows.

## Diagnostics Run

1. 2024 churn grid on saved predictions:
   baseline gate30, no rank gate, softer gate50, current `tc10_slip5`, and
   lower-cost `spread5` execution.
2. Periodic hold overlay:
   rebalance every 5 or 10 prediction days under current and lower-cost
   assumptions.
3. 2022 regime throttle overlay:
   VIX >= 25 half exposure and VIX >= 30 cash overlays.
4. Stress-month regime overlay:
   April/June/September/December 2022 and April/July 2024.
5. Ticker-level style attribution:
   trailing volatility, beta, and momentum for top/worst contributors.

## Findings

### 2024 Is Not A Gross Signal Failure

Mean 2024 results across the three Option A base seeds:

| scenario | total | gross | benchmark | excess | avg daily turnover | avg cost bps | trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tc10_slip5_gate30` | 11.75% | 21.82% | 11.73% | 0.02% | 18.10% | 3.62 | 858 |
| `spread5_gate30` | 19.22% | 21.82% | 11.73% | 7.49% | 18.10% | 0.91 | 858 |
| `tc10_slip5_gate50` | 20.27% | 28.06% | 11.73% | 8.53% | 13.07% | 2.61 | 619 |
| `spread5_gate50` | 26.07% | 28.06% | 11.73% | 14.34% | 13.07% | 0.65 | 619 |
| `tc10_slip5_no_gate` | 7.01% | 28.92% | 11.73% | -4.72% | 39.17% | 7.83 | 1857 |
| `spread5_no_gate` | 23.06% | 28.92% | 11.73% | 11.32% | 39.17% | 1.96 | 1857 |

The no-gate run has the best gross signal, but current costs turn it into a
negative excess result. The softer gate50 setting keeps most of the gross
signal while reducing turnover by about 5 percentage points versus gate30 and
about 26 percentage points versus no gate. Under current costs it improves
2024 excess by about 8.5 percentage points versus the promoted gate30
baseline.

### Naive Calendar Holding Does Not Fix 2024

Mean 2024 periodic hold results:

| scenario | total | gross | benchmark | excess | avg daily turnover | avg cost bps | trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tc10_slip5_rebalance_every_5d` | 2.57% | 9.95% | 11.73% | -9.16% | 29.40% | 2.94 | 707 |
| `spread5_rebalance_every_5d` | 8.05% | 9.95% | 11.73% | -3.68% | 29.40% | 0.73 | 707 |
| `tc10_slip5_rebalance_every_10d` | -4.21% | -0.65% | 11.73% | -15.94% | 15.44% | 1.54 | 376 |
| `spread5_rebalance_every_10d` | -1.55% | -0.65% | 11.73% | -13.28% | 15.44% | 0.39 | 376 |

The model needs execution discipline that preserves rank signal, not a blunt
calendar throttle. Gate50 is a better direction than fixed 5-day or 10-day
rebalance cadence.

### 2022 Is Not Solved By Simple VIX Throttles

Mean 2022 regime throttle results:

| scenario | risk-off days | total | baseline total | excess | baseline excess | delta excess |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vix_ge_25_half_exposure` | 130 | -29.30% | -22.93% | -22.85% | -16.48% | -6.37% |
| `vix_ge_30_cash` | 48 | -30.15% | -22.93% | -23.70% | -16.48% | -7.22% |

Both simple VIX overlays worsened 2022. That argues against solving the year
with a broad volatility switch. The failure looks more like the model held the
wrong style/exposure set during a regime transition.

### Stress Months Split The Failure Modes

The 2022 weak months are broad gross drawdowns, not mostly cost drag. April,
June, September, and December 2022 repeatedly lose before costs, with
transaction costs small relative to gross losses. Volatility was elevated in
April/June/September, but December had only moderate VIX with a deeply inverted
yield curve.

The 2024 weak months look different. April and July 2024 had low-to-moderate
VIX, and the gross signal remained positive in the yearly aggregate. The
problem is execution/churn under the current cost model, not a high-volatility
regime wipeout.

### Ticker Attribution Points To High-Beta Growth Pain In 2022

Worst 2022 contributors included:

| ticker | contribution | held days | avg vol63 | avg beta63 | avg mom63 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `META.OQ` | -9.67% | 314 | 52.75% | 1.16 | -22.66% |
| `TSLA.OQ` | -9.33% | 354 | 68.41% | 1.88 | -19.33% |
| `F.N` | -8.72% | 201 | 54.04% | 1.78 | -17.88% |
| `AMZN.OQ` | -8.25% | 315 | 52.56% | 1.48 | -19.41% |
| `AMD.OQ` | -7.66% | 499 | 60.44% | 1.89 | -18.10% |
| `NVDA.OQ` | -4.38% | 641 | 64.73% | 2.07 | -14.89% |

Top 2024 contributors included `NVDA.OQ`, `NFLX.OQ`, `NCLH.N`, `SMCI.OQ`,
`HPE.N`, and `AVGO.OQ`. The same willingness to hold volatile/high-upside
names can work well in 2024 but was punished in 2022. The next 2022 diagnostic
should be exposure-aware, not just cheaper execution.

## Decisions

- Preserve the current recipe while diagnosing. These are saved-prediction
  replays, not a new training search.
- Treat 2024 as an execution and turnover-control problem. Softer rank-gate
  logic is the best tested direction so far.
- Treat 2022 as a regime/style exposure problem. Simple VIX throttles are not
  sufficient and worsened the replay.
- Keep 2023/2025 as preservation targets: the prior Option A readout showed
  positive excess in all three seeds for both years.

## Follow-Up Work

1. Promote the replay diagnostics into a durable script if these tables will be
   re-run often.
2. Add sector/industry and market-cap proxy attribution if a reliable mapping is
   available.
3. Design a turnover-aware rank-gate experiment around gate50-style behavior,
   then test it with the same saved-prediction replay before retraining.
4. Design a 2022 exposure-control experiment that targets beta/volatility/style
   concentration directly, not just VIX risk-off days.

No production code was changed for this report.
