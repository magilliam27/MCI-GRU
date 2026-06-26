# LambdaRankIC 1024 All-Year PIT Validation

## Scope

- Branch/worktree: `codex/lambdarankic-1024-all-years-20260625` in `C:\Users\magil\.codex\worktrees\602a\MCI-GRU`.
- Goal: train `lambdarank_ic` with `training.lambdarank_ic_max_pairs_per_day=1024` across every configured PIT test year, then replay saved averaged predictions in two modes.
- Notebook: `notebooks/lambdarank_ic_1024_all_years_colab.ipynb`.
- Generator: `scripts/gen_lambdarank_ic_1024_all_years_nb.py`.

## Confirmed Year Universe

Configured PIT test years are `2022, 2023, 2024, 2025`.

Evidence:

- `configs/experiment/` contains `pit_temporal_2022.yaml` through `pit_temporal_2025.yaml` only.
- `notebooks/pit_repeated_seed_replication_colab.ipynb`, `notebooks/volatility_targeting_repeated_seed_colab.ipynb`, and `scripts/run_pit_saved_prediction_backtests.py` use the same `2022..2025` PIT windows.
- Market metadata for `sp500_pit_union_lseg_20150101_20260513.csv` reports `date_max=2026-05-13`, but no configured 2026 PIT test year exists.

Backtest windows:

- 2022: `2022-01-22` to `2022-12-31`
- 2023: `2023-01-22` to `2023-12-31`
- 2024: `2024-01-22` to `2024-12-31`
- 2025: `2025-01-22` to `2025-12-31`

The January 22 starts are the established shortened first-month PIT windows.

## Recipe

- Clone/training branch in Colab: `codex/colab-gpu-utilization-hardening-20260620`, matching the completed 2022 higher-pair tranche.
- Seed: `314159`
- Years: `[2022, 2023, 2024, 2025]`
- Pair cap: `1024`
- Models/year: `20`
- Epochs: `100`
- Early stopping patience: `15`
- Loss: `lambdarank_ic`
- Selection metric: `val_rank_ic`
- Labels: raw returns with `model.label_t=5`
- PIT mode: `data.use_pit_universe=true`, `data.pit_universe_mode=masked_panel`, `data.pit_min_scoreable_stocks=450`, `data.pit_breadth_policy=error`
- Graph/features: frozen recipe static threshold graph, multi-feature edges, `drop_edge_p=0.1`, static weekly momentum, strict current-only global regime features
- Training runtime: visible Colab G4/L4-class or better GPU, rejecting T4/CPU

## Backtests

Each completed yearly `averaged_predictions` folder is replayed with `tests/backtest_sp500_daily.py`:

- `no_cost_no_gate`: `top_k=10`, `label_t=5`, `num_tests=4`, `adjustment_method=bhy`, no transaction costs, no rank-drop gate
- `cost_rank_gate`: same base settings, plus `--transaction_costs --spread 10 --slippage 5 --enable_rank_drop_gate --min_rank_drop 30`

## Drive Contract

Uploaded notebook:

```text
https://drive.google.com/file/d/1lod-AYy0p9F5UcjTW-CM_Gn2yBdSQ3iN/view?usp=drivesdk
```

Notebook run root:

```text
/content/drive/MyDrive/MCI-GRU-Ablations/lambdarank_ic_1024_all_years/<RUN_TAG>
```

Primary artifacts:

- `heartbeat.json`
- `lambdarank_ic_1024_all_years_manifest.json`
- `training_results.json`
- `training_results.csv`
- `backtest_results.json`
- `backtest_results.csv`
- `all_years_results.json`
- `all_years_results.csv`
- `gpu_util.csv`

Do not overwrite the completed 2022 higher-pair tranche at:

```text
/content/drive/MyDrive/MCI-GRU-Ablations/lambdarank_ic_pair_cap_escalation_full_tranche/20260624_033128
```

## Verification

- `C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe scripts\gen_lambdarank_ic_1024_all_years_nb.py`
- `C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m py_compile scripts\gen_lambdarank_ic_1024_all_years_nb.py`
- Notebook code-cell parse via `ast.parse`: 4 code cells parsed.

## Launch Status On 2026-06-25

Initial launch failed before manifest creation because Colab Secrets timed out
while reading `FRED_API_KEY`. The secret existed and notebook access was enabled,
so the notebook was rerun from the top on the same visible G4-class runtime.

Successful retry evidence:

- Colab toolbar: connected Google Compute Engine GPU backend, about `176.88 GB`
  RAM.
- Setup output: `Torch: 2.11.0+cu128`, `CUDA available: True`,
  `GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition`.
- Setup output: `FRED_API_KEY loaded from Colab Secrets.`
- Setup output: `Market date range: 2015-01-02 to 2026-05-13`.
- Setup output: `LambdaRankIC branch probe: lambdarank_ic
  (max_pairs_per_day=1024, temperature=1.0) LambdaRankICLoss`.
- Visible notebook check after retry: running, setup success present, training
  section active, no active FRED `RuntimeError`.

Drive run root:

```text
https://drive.google.com/drive/folders/1hu65DrY-UmbkH1DVG6qRYSU0ZNsTc0am
/content/drive/MyDrive/MCI-GRU-Ablations/lambdarank_ic_1024_all_years/20260625_180953
```

Heartbeat:

```text
https://drive.google.com/file/d/1VCpM9QTIjUX1pdulJUocectBHD_EDd3L/view?usp=drivesdk
```

Manifest:

```text
https://drive.google.com/file/d/1tXHiO0Lil9ziLJaYxYfe1Vr0OI5-rIxF/view?usp=drivesdk
```

Heartbeat at launch:

- `phase=training`
- `status=RUNNING`
- `current_job=lambdarank_ic_pairs1024_2022_seed314159`
- `completed_training_jobs=0`
- `expected_training_jobs=4`
- `completed_backtests=0`
- `expected_backtests=8`
- `gpu_name=NVIDIA RTX PRO 6000 Blackwell Server Edition`

Training artifact evidence:

- 2022 job folder:
  `https://drive.google.com/drive/folders/1lz3R17NKLQZYBfT587vQ8OwgQWv1U8WL`
- 2022 timestamped output folder:
  `https://drive.google.com/drive/folders/1YbSTD5X2S_O9MpTCwuGv2nNCXXSxzk_3`
- The timestamped folder contains `.hydra/`, `config.yaml`, and
  `run_experiment.log`, confirming the first `run_experiment.py` invocation
  reached the training output path.

Next continuation:

1. Monitor `heartbeat.json`; Drive artifacts remain the source of truth.
2. When training completes, collect `training_results.json/csv` and
   `backtest_results.json/csv`.
3. Produce the final table by year for `val_ic`, `val_rank_ic`,
   `no_cost_no_gate`, and `cost_rank_gate`.
4. Recommend whether 1024 is robust, needs seed replication, or should be
   abandoned/changed.

## Seed Replication Launch On 2026-06-25

User approved proceeding with the follow-up seed replication run after the
all-year 1024 readout. The launched replication uses five base seeds:

```text
314159, 271828, 161803, 141421, 173205
```

Recipe stays matched to the completed all-year 1024 run:

- Years: `2022, 2023, 2024, 2025`
- Pair cap: `training.lambdarank_ic_max_pairs_per_day=1024`
- Jobs: `4` years x `5` seeds = `20` training jobs
- Ensemble/job: `20` models, `100` epochs, patience `15`
- Loss/selection: `lambdarank_ic`, `selection_metric=val_rank_ic`
- Labels/backtest: `label_t=5`, `top_k=10`, `num_tests=4`,
  `adjustment_method=bhy`
- Backtest modes: `no_cost_no_gate` and `cost_rank_gate` with 10 bps spread,
  5 bps slippage, `min_rank_drop=30`
- PIT mode remains `masked_panel`; no completed 2022 tranche artifacts are
  overwritten.

Local artifacts:

- Notebook:
  `notebooks/lambdarank_ic_1024_seed_replication_colab.ipynb`
- Generator:
  `scripts/gen_lambdarank_ic_1024_seed_replication_nb.py`
- Verification:
  `C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe scripts\gen_lambdarank_ic_1024_seed_replication_nb.py`
- Verification:
  `C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m py_compile scripts\gen_lambdarank_ic_1024_seed_replication_nb.py`
- Generated notebook AST parse: 4 code cells parsed; seed list,
  `len(YEARS) * len(BASE_SEEDS)`, run root name, and resume merge keys checked.

Uploaded notebook:

```text
https://drive.google.com/file/d/1rcAbSmt2z5pvpxJ9go9AlGWDbsq1m7og/view?usp=drivesdk
```

Drive run root:

```text
https://drive.google.com/drive/folders/1cNogvkAQAK-HFg4-vdbTLuuY29VP1XX8
/content/drive/MyDrive/MCI-GRU-Ablations/lambdarank_ic_1024_seed_replication/20260625_211134
```

Heartbeat:

```text
https://drive.google.com/file/d/1kaRglgK7hfoUQ-BKK9YZWvIx8SH6AA1E/view?usp=drivesdk
```

Manifest:

```text
https://drive.google.com/file/d/1tpFL5H2m59iW1y_a2RV5DL0FcL-r7PIC/view?usp=drivesdk
```

Training folder:

```text
https://drive.google.com/drive/folders/1Y-I0Rzh647gTTzgeZJX6B1iP0jcw1-mT
```

Backtests folder:

```text
https://drive.google.com/drive/folders/15aZ9H4iFZhabUzfd4HzdZuOFRu_jhhB7
```

Summaries folder:

```text
https://drive.google.com/drive/folders/1mlgGdH6SC0A_zsdaNQPhe3S_vjDZ1IuQ
```

Visible Colab launch evidence:

- Runtime was changed from visible `T4 GPU` to visible `G4 GPU` with High-RAM.
- Toolbar/runtime state showed `G4 (Python 3)`.
- Setup output on retry showed `FRED_API_KEY loaded from Colab Secrets`.
- Setup output showed GPU
  `NVIDIA RTX PRO 6000 Blackwell Server Edition`.
- Manifest output showed run root
  `/content/drive/MyDrive/MCI-GRU-Ablations/lambdarank_ic_1024_seed_replication/20260625_211134`
  and `Training jobs: 20`.

Heartbeat at launch:

- `phase=training`
- `status=RUNNING`
- `current_job=lambdarank_ic_pairs1024_2022_seed314159`
- `completed_training_jobs=0`
- `expected_training_jobs=20`
- `completed_backtests=0`
- `expected_backtests=40`
- `gpu_name=NVIDIA RTX PRO 6000 Blackwell Server Edition`

Training artifact evidence for the first seed/job:

- Job folder:
  `https://drive.google.com/drive/folders/1lK0kfNOAGiQKIAX_MjgOAsEChB7SZfeM`
- Timestamped output folder:
  `https://drive.google.com/drive/folders/1P28G6ajn4g_NFPRgnEWeE3Snbvl__I4l`
- Timestamped folder contains `.hydra/`, `config.yaml`, `run_experiment.log`,
  `run_metadata.json`, `feature_reference.json`, `graph_data.pt`, and
  `checkpoints/`, confirming the first `run_experiment.py` invocation reached
  graph materialization/checkpoint setup and is in the actual training path.

Continuation notes:

1. Monitor the heartbeat and summary files in Drive; Drive artifacts remain the
   source of truth.
2. Current notebook resume logic skips completed jobs within the active run
   root. If a fresh runtime restarts from scratch, reuse the same run tag/root
   instead of silently creating a second run folder.
3. When complete, aggregate by year and seed, then compare seed dispersion
   against the single-seed all-years pattern, especially 2024.

## Seed Replication Final Readout On 2026-06-26

Drive heartbeat now reports:

- `phase=done`
- `status=OK`
- `completed_training_jobs=20`
- `expected_training_jobs=20`
- `completed_backtests=40`
- `expected_backtests=40`
- `gpu_name=NVIDIA RTX PRO 6000 Blackwell Server Edition`
- `updated_at=2026-06-26T08:47:54.107594Z`

Final artifacts:

- `training_results.csv`:
  `https://drive.google.com/file/d/1USJ7xZYjILBeIzHUfnM4ndLXWXEYzmoe/view?usp=drivesdk`
- `training_results.json`:
  `https://drive.google.com/file/d/1grMat5ssCzjcK_VWw6TjS8bdcFjAljU5/view?usp=drivesdk`
- `backtest_results.csv`:
  `https://drive.google.com/file/d/1MNNDBeoumVdbVzyiK9tlbzISyyIvSrG5/view?usp=drivesdk`
- `backtest_results.json`:
  `https://drive.google.com/file/d/1L7dCQuCk2IcLmbNgM-IINhn0FdKTn0GS/view?usp=drivesdk`
- `all_years_results.csv`:
  `https://drive.google.com/file/d/1QZiKmPwbCqncKlBXFDgAGHkJ9kjB9GJd/view?usp=drivesdk`
- `all_years_results.json`:
  `https://drive.google.com/file/d/10PNroot26ng-KtIC4ipNmn6-0aB_iYfY/view?usp=drivesdk`

Year-level five-seed means:

| Year | Mean val IC | Mean val rank IC | No-cost total | No-cost ASR | No-cost excess | Cost+gate total | Cost+gate ASR | Cost+gate excess | Cost+gate trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 0.00751 | 0.01317 | -11.09% | -0.338 | -4.64% | -17.61% | -0.542 | -11.16% | 15.2 |
| 2023 | 0.01056 | 0.01506 | +38.98% | 1.943 | +31.86% | +37.65% | 1.826 | +30.06% | 48.8 |
| 2024 | 0.01367 | 0.01825 | +16.93% | 0.859 | +5.20% | +8.93% | 0.488 | -2.80% | 546.0 |
| 2025 | 0.01350 | 0.01661 | +9.26% | 0.315 | +2.12% | +5.78% | 0.214 | -1.37% | 764.8 |

Seed-level totals:

| Year | Seed | Val IC | Rank IC | No-cost total / ASR / excess | Cost+gate total / ASR / excess |
| --- | ---: | ---: | ---: | --- | --- |
| 2022 | 314159 | 0.00624 | 0.01197 | -0.61% / -0.025 / +5.84% | -5.26% / -0.204 / +1.19% |
| 2022 | 271828 | 0.00625 | 0.01619 | -5.69% / -0.235 / +0.76% | -15.81% / -0.515 / -9.36% |
| 2022 | 161803 | 0.00702 | 0.01206 | -4.18% / -0.174 / +2.27% | -13.27% / -0.479 / -6.82% |
| 2022 | 141421 | 0.00714 | 0.01223 | -10.66% / -0.370 / -4.21% | -18.10% / -0.606 / -11.65% |
| 2022 | 173205 | 0.01090 | 0.01341 | -34.33% / -0.887 / -27.88% | -35.62% / -0.903 / -29.17% |
| 2023 | 314159 | 0.00903 | 0.01528 | +26.64% / 1.893 / +19.52% | +28.66% / 1.814 / +21.54% |
| 2023 | 271828 | 0.01585 | 0.01820 | +4.57% / 0.314 / -2.55% | +6.49% / 0.434 / -0.63% |
| 2023 | 161803 | 0.00680 | 0.01167 | +17.04% / 1.233 / +9.91% | +15.91% / 1.129 / +8.79% |
| 2023 | 141421 | 0.00502 | 0.01082 | +70.41% / 3.085 / +63.29% | +70.60% / 3.127 / +63.48% |
| 2023 | 173205 | 0.01608 | 0.01933 | +76.25% / 3.191 / +69.12% | +64.25% / 2.627 / +57.12% |
| 2024 | 314159 | 0.01367 | 0.01715 | -0.35% / -0.023 / -12.08% | -2.29% / -0.130 / -14.02% |
| 2024 | 271828 | 0.01079 | 0.01609 | +1.69% / 0.109 / -10.04% | +10.01% / 0.615 / -1.72% |
| 2024 | 161803 | 0.01579 | 0.01892 | +47.62% / 2.359 / +35.88% | +28.79% / 1.414 / +17.06% |
| 2024 | 141421 | 0.01434 | 0.02078 | +18.55% / 1.102 / +6.81% | +10.37% / 0.634 / -1.36% |
| 2024 | 173205 | 0.01377 | 0.01829 | +17.15% / 0.747 / +5.42% | -2.23% / -0.091 / -13.96% |
| 2025 | 314159 | 0.01454 | 0.01527 | +22.10% / 0.835 / +14.95% | +26.83% / 1.010 / +19.69% |
| 2025 | 271828 | 0.01220 | 0.01568 | -3.46% / -0.145 / -10.60% | -3.45% / -0.137 / -10.59% |
| 2025 | 161803 | 0.01298 | 0.01601 | +17.00% / 0.592 / +9.86% | +9.09% / 0.326 / +1.94% |
| 2025 | 141421 | 0.01119 | 0.01524 | -3.07% / -0.168 / -10.21% | -0.54% / -0.029 / -7.68% |
| 2025 | 173205 | 0.01661 | 0.02082 | +13.75% / 0.461 / +6.61% | -3.06% / -0.100 / -10.20% |

Interpretation:

- 2023 is the only clearly robust year: all five seeds are positive in both
  modes and four of five beat benchmark in both modes.
- 2022 is clearly not robust: all five seeds are negative in both modes.
- 2024 has good average no-cost performance, but it depends heavily on one
  strong seed and cost/rank-gate excess is negative on average.
- 2025 is mixed: positive mean return, but only three of five no-cost seeds and
  two of five cost/rank-gate seeds are positive; cost/gate excess is negative on
  average.
- Validation IC/rank IC rises into 2024/2025, but that did not translate into
  consistent excess return after costs, so validation rank IC alone is not a
  sufficient acceptance gate for this setup.

Recommendation:

Do not treat cap 1024 as robust enough to promote as-is. It is good enough to
keep as a research candidate, but the next work should focus on stabilizing
realized portfolio behavior: cross-seed ensembling, turnover/rank-gate design,
and year/regime diagnostics. More pair-cap sweeps alone are unlikely to answer
the main question now.
