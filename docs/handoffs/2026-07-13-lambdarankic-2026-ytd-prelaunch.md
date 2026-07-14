# LambdaRankIC 2026-YTD pre-launch approval

Status: **PAUSED — training is not authorized yet.**

No Colab runtime has been allocated, no G4 session has been started, no fresh
LSEG export has been requested, and no Drive run folder has been created.

## Approval identity

- Branch: `codex/lambdarankic-2026-ytd-20260713`
- Worktree: `C:\Users\magil\MCI-GRU\.codex\worktrees\lambdarankic-2026-ytd-20260713`
- Hydra experiment: `configs/experiment/lambdarankic_2026_ytd_110_name.yaml`
- Complete resolved approval bundle:
  `configs/launch_manifests/lambdarankic_2026_ytd_110_name.json`
- Notebook: `notebooks/lambdarankic_2026_ytd_colab.ipynb`
- Approval SHA-256:
  `a34ae4b778b03a12c464f794f79a72caa2024a75d376f45b275175f5507768a8`

The digest covers the complete resolved Hydra config plus the campaign matrix,
universe/export contract, runtime restriction, evaluation scope, and artifact
policy, plus normalized hashes of the critical source/config/dependency files.
The notebook recomputes it, recomposes each seed immediately before launch,
and refuses to proceed when any approved source or resolved value differs.

## Proposed experiment

### Data and split

- Data source: refreshed LSEG CSV export, with FRED-backed strict global regime
  inputs during training.
- Universe: monthly PIT S&P 500 top 10 by market cap within each of 11 GICS
  sectors (110 active names per snapshot), carried as a true PIT masked panel.
- Selector range: `2021-01-04` through `2026-07-13`.
- Price-history buffer: `2019-01-01` through `2026-07-13`.
- Train: `2021-01-01` through `2024-12-31`.
- Validation: `2025-01-10` through `2025-12-23`.
- Test/predictions: `2026-01-01` through `2026-07-13` (first actual session is
  expected to be `2026-01-02`).
- Label: raw five-session forward return.
- Tail policy: predictions are retained through `2026-07-13`; label-based test
  metrics stop at the latest date with a complete five-session forward label.

The validation boundary starts on January 10 because the December 31, 2024
training label reaches January 8 and January 9 was not an NYSE session. The
validation end is December 23 because its five-session target reaches December
31, leaving the January 2, 2026 test session strictly out of validation labels.

### LambdaRank and replication

- Objective: `lambdarank_ic` only.
- Pair cap: `8192`; temperature: `1.0`.
- At 110 names, the full daily pair set is `110 * 109 / 2 = 5,995`, so this cap
  is an all-pairs run rather than a sampled-pair approximation.
- Base seeds: `314159`, `271828`, `161803`, `141421`, `173205`.
- Ensemble members per base seed: `20`, using internal member seeds
  `base_seed + model_id`.
- Total: 5 training jobs and 100 model fits.

### Frozen model/training recipe

- Epochs `100`; patience `15`; batch size `32`.
- Learning rate `5e-5`; weight decay `1e-3`; gradient clip `1.0`.
- Cosine scheduler with `1000` warm-up optimizer steps; AMP enabled.
- Checkpoint selection: `val_rank_ic`; shuffled static-graph training.
- `his_t=10`; `label_t=5`; `gru_attn` temporal encoder.
- Static threshold graph: correlation threshold `0.8`, 252-session lookback,
  multi-feature edges, edge dropout `0.1`, no sector relation, no lead-lag or
  snapshot-age columns.
- Binary weekly momentum with static 0.5 fast weight.
- Strict current-only global regime features; subsequent-return regime inputs
  disabled.
- Z-score statistics fitted only inside the inclusive 2021-2024 training range.
- Single-process data loader (`num_workers=0`, pin-memory off) to preserve the
  frozen comparison recipe.
- MLflow disabled. Repo-native checkpoints, predictions, graph data, logs, and
  summaries remain enabled and are copied to Drive.

With the expected 995 usable training samples, batch size 32 yields about 32
steps per epoch; the 1000-step warm-up therefore lasts roughly 31 epochs. This
is intentionally unchanged from the frozen recipe and needs explicit approval.

### Evaluation and artifacts

- Built-in test metrics at top K `10, 20, 50, 100`.
- 1,000 bootstrap resamples, 95% intervals, Newey-West adjustment.
- Cross-seed mean and sample standard deviation for numeric evaluation metrics.
- No matched pure-IC control in this launch.
- No strategy backtest in this launch; averaged predictions are preserved for a
  separate replay after training.
- G4-only visible Colab run through Chrome; T4, L4, and CPU are rejected.
- Drive root:
  `MyDrive/MCI-GRU-Ablations/lambdarank_ic_2026_ytd/<run_tag>/`.
- Durable artifacts include the approval bundle, input hashes/data audit,
  `nvidia-smi` evidence, heartbeat, resolved configs, checkpoints, per-model and
  averaged predictions, `graph_data.pt`, logs, per-seed results, cross-seed
  summary, and final run summary.
- Preflight additionally requires complete daily OHLCV breadth of at least 100,
  exact 110-name PIT activity, matching market/PIT/snapshot identifiers,
  per-stock label embargo proof, and a non-empty four-channel static graph.
- Artifacts sync every 60 seconds. There is no automatic resume: synced files
  are salvage evidence, and any interrupted campaign requires an explicit
  recovery plan before relaunch. The core trainer cannot resume inside a
  partially completed 20-model ensemble.

## Required approval

Confirm or change all of the following before launch:

1. The 110-name monthly PIT sector-balanced universe rather than the full
   roughly 500-name PIT panel.
2. Five base seeds and 20 ensemble members per seed (100 fits total).
3. LambdaRank-only training, with no matched pure-IC control and no strategy
   backtest in this run.
4. The session-safe train/validation/test boundaries above.
5. Pair cap 8192, temperature 1.0, and the full frozen model/training recipe,
   including the approximately 31-epoch warm-up.
6. G4-only Colab execution and the Drive artifact/resume policy.

After approval, the next actions are: refresh and audit LSEG data through July
13; stop and request re-approval if the final fully covered date or any config
changes; push the pinned branch; open Colab visibly in Chrome; verify G4 plus
`nvidia-smi`; enter the exact approved digest in the closed launch cell; and run
the foreground campaign through durable completion and runtime cleanup.
