# MCI-GRU Architecture Review (Updated 2026-04)

This document is the current architecture review for the MCI-GRU codebase. It
replaces the original pre-change audit with a status-oriented review: what the
system now does, what has been fixed since the first review, and where the next
engineering and research work should go.

Sister document: `docs/ARCHITECTURE.md` describes the system as built. This
review is the critique and roadmap.

Scope reviewed: `mci_gru/`, `run_experiment.py`, `configs/`, `paper_trade/`,
`scripts/`, `tests/`, `.github/workflows/ci.yml`, and the architecture docs.

## TL;DR

The repo has moved from a clean research implementation with several known
training, evaluation, and MLOps gaps into a much more credible experimentation
platform. The largest upgrades are now in place:

- Modern training defaults: `AdamW`, warmup plus cosine scheduling, CUDA AMP,
  validation-IC checkpoint selection, combined MSE+IC loss, and per-member
  ensemble seeds.
- Model uplift: trunk `LayerNorm` and dropout, fused `nn.GRU` plus attention,
  optional transformer encoder, optional `nn.MultiheadAttention` latent learner,
  optional stream type embeddings, output activation control, DropEdge, sector
  branch, and optional A1/A2 cross-stream attention.
- Graph and data upgrades: multi-feature edge attributes, top-k graph mode,
  lead-lag edge features, snapshot-age features, per-split completeness filtering,
  optional PIT universe filtering, optional Polars hot path, and rank-Gaussian
  normalisation.
- Evaluation and production trust layer: daily IC series, Newey-West Sharpe,
  moving-block bootstrap CIs, shared portfolio utilities, feature drift metrics,
  MLflow defaults, run metadata fingerprints, and CI smoke coverage.

The remaining work is less about "basic hygiene" and more about proving which
new knobs deserve to become defaults. The next highest-ROI work is controlled
A/B testing against `configs/experiment/paper_faithful.yaml`, improving missing
data handling, adding graph snapshot caching, adding gradient accumulation, and
hardening dependency/reproducibility locks.

## Current Architecture Posture

The codebase now has a good separation between research, evaluation, and
paper-trade concerns.

- `mci_gru/pipeline.py` is still the central orchestration layer for feature
  generation, normalisation, split handling, graph construction, tensor creation,
  and feature-reference creation.
- `mci_gru/models/mci_gru.py` contains the full model family behind config flags:
  legacy paper-style GRU, fused GRU+attention, causal transformer, GAT streams,
  latent market state learner, four-stream fusion, optional sector branch, and
  final prediction GAT.
- `mci_gru/graph/builder.py` owns graph schedule construction and edge feature
  generation. `GraphSchedule` preserves dynamic graphs without forcing
  `batch_size=1`.
- `mci_gru/evaluation/` is now the statistical trust layer for IC, Sharpe,
  bootstrap intervals, top-k returns, turnover, rank-drop selection, and feature
  drift.
- `paper_trade/scripts/` remains isolated around frozen checkpoints and live
  inference/monitoring workflows.
- `configs/config.yaml` is the modern default surface; `paper_faithful.yaml`
  preserves legacy behavior for replication and ablation.

The overall direction is right: retain the original four-stream MCI-GRU
inductive bias, but wrap it in better training dynamics, more realistic graph
signals, and statistically defensible evaluation.

## Shipped Since The Original Review

### Phase 1: Training, Leakage, and Reproducibility

Status: shipped.

- Replaced plain Adam with `AdamW`.
- Added linear warmup plus cosine LR scheduling through `TrainingConfig`.
- Added CUDA AMP with `torch.amp.autocast` and `GradScaler`.
- Added validation IC computation and `selection_metric=val_ic` checkpointing.
- Made combined MSE+IC loss the default (`loss_type=combined`,
  `ic_loss_alpha=0.5`).
- Defaulted graph edges to multi-feature attributes instead of scalar
  correlation only.
- Seeded ensemble members with `seed + model_id`.
- Added strict calendar embargo checks between train/val and val/test based on
  `label_t`, with an explicit opt-out.
- Vectorised time-series feature tensor construction.
- Added data file metadata and SHA-256 fingerprinting to run metadata when the
  source file is available.
- Enabled MLflow tracking by default.

Impact: the platform is now faster, less leak-prone, easier to reproduce, and
better aligned with the ranking objective used downstream.

### Phase 2: Model Trunk Uplift

Status: shipped behind config flags, with modern defaults enabled in
`configs/config.yaml` where appropriate.

- Added trunk `LayerNorm` and dropout through `model.use_trunk_regularisation`
  and `model.trunk_dropout`.
- Added inter-GAT dropout inside `GATBlock`.
- Added fused `GRUWithAttention` using `nn.GRU` plus post-hoc attention readout.
- Preserved the original `AttentionResetGRUCell` path as `temporal_encoder:
  legacy`.
- Added optional `nn.MultiheadAttention` path in `MarketLatentStateLearner`.
- Added learned four-stream type embeddings in `SelfAttention`.
- Added configurable final output activation with `"none"` as the modern
  default.
- Added train-time DropEdge via `graph.drop_edge_p`.
- Added optional A1/A2 cross-stream attention:
  `A2 = A2 + CrossAttn(Q=A2, KV=A1_sequence)`.

Impact: the model no longer relies only on the slow Python-loop GRU and
late-concat fusion. It can now use fused recurrent kernels, stronger
regularisation, richer stream identity, and direct temporal-to-graph transfer.

### Phase 3: Data and Graph Modernisation

Status: shipped, mostly optional.

- Added lead-lag graph edge features through `graph.use_lead_lag_features` and
  `graph.lead_lag_days`.
- Added snapshot age features through `graph.append_snapshot_age_days`.
- Added top-k graph mode with signed correlation preservation and optional
  absolute-correlation ranking.
- Added static sector relation support via a second GAT branch and linear fuse.
- Added rank-Gaussian normalisation fit only on the train period.
- Added per-split stock completeness filtering as a survivorship-bias mitigation.
- Added optional PIT universe filtering through `data.use_pit_universe` and
  `pit_universe_csv`.
- Added optional Polars-backed pivot path for selected preprocessing hot paths.
- Added causal transformer temporal encoder behind `model.temporal_encoder:
  transformer`.
- Added walk-forward orchestration through `training.walkforward`.

Impact: the graph can now carry more than same-day Pearson correlation, and the
data pipeline has knobs for more realistic universe handling. Most of these
features should remain gated until tested against the paper-faithful baseline.

### Phase 4: Evaluation, Monitoring, and CI

Status: shipped.

- Added daily IC series and top-k return helpers.
- Added Newey-West Sharpe for overlapping forward-return windows.
- Added moving-block bootstrap confidence intervals.
- Added shared portfolio helpers for rank sorting, top-k selection, turnover,
  and rank-drop gates.
- Added feature-reference artifacts from the train window.
- Added PSI and KS-style feature drift checks.
- Updated paper-trade inference to persist normalized inference features.
- Added monitor and report drift outputs.
- Added Optuna/Hydra sweep example config.
- Added GitHub Actions lint, tests, and end-to-end smoke via
  `scripts/ci_smoke.py`.

Impact: evaluation is no longer only a point-estimate exercise. The repo can now
ask "did this change help with uncertainty attached?" and CI is much more likely
to catch broken wiring.

## What Is Strong Now

- The four-stream design is still intact and now better regularised.
- The paper-faithful config gives a stable baseline for ablations.
- Dynamic graph lookup is precomputed and batch-safe.
- Training defaults match modern PyTorch practice much more closely.
- Validation selection aligns with the cross-sectional ranking goal.
- Evaluation now handles overlapping `label_t` return windows more honestly.
- Paper-trade monitoring has a real feature-drift surface.
- Tests cover model phase flags, dynamic graph behavior, preprocessing, drift,
  portfolio helpers, MLflow, and paper-trade monitor behavior.
- CI now runs lint, unit tests, and a small end-to-end smoke.

These are meaningful changes. The repo is no longer just a model script; it is
becoming an experimentation system.

## Remaining Gaps

### 1. Prove Which Phase 2/3 Features Should Be Defaults

Many upgrades are implemented, but not all should be promoted blindly. The
project still needs a disciplined ablation table comparing:

- `paper_faithful`
- modern defaults
- modern defaults plus A1/A2 cross-attention
- modern defaults plus transformer encoder
- top-k graph variants
- lead-lag and snapshot-age graph variants
- sector relation branch
- rank-Gaussian normalisation
- per-split or PIT universe filtering

Decision criterion should be out-of-sample IC, top-k return, Newey-West Sharpe,
turnover, and bootstrap confidence intervals, preferably under walk-forward
evaluation.

### 2. Missing Data Handling Is Still Too Crude

`pipeline.py` still fills feature NaNs by same-day cross-sectional mean and then
falls back to zero. In the inference path, missing feature values are filled with
zero. This is simple and stable, but it loses information and can leak
cross-sectional structure into imputed values.

Recommended next step:

- Forward-fill within each `kdcode` first.
- Use cross-sectional median/mean only as fallback.
- Add `{feature}_is_missing` indicator columns for high-value feature groups.
- Persist imputation policy metadata in run artifacts.

This is one of the remaining high-impact data-quality improvements.

### 3. Graph Snapshots Are Not Cached Across Runs

`GraphBuilder.precompute_snapshots()` rebuilds graph schedules inside each data
prep run. That is correct but wasteful for sweeps over model and training
parameters.

Recommended next step:

- Add a content-addressed graph cache under `data/cache/graphs/`.
- Key by universe/date range, `kdcode_list` hash, graph config, and source data
  fingerprint.
- Include edge feature dimensionality and lead-lag settings in the cache key.
- Log cache hits/misses in run metadata.

This will matter once Optuna or walk-forward sweeps become routine.

### 4. Gradient Accumulation Is Still Missing

AMP and scheduling are in place, but there is no gradient accumulation. Larger
effective batches can reduce IC noise without requiring larger GPU memory.

Recommended next step:

- Add `training.gradient_accumulation_steps`.
- Scale loss before backward.
- Step optimizer, scheduler, and scaler only on accumulation boundaries.
- Preserve current behavior when the value is `1`.

### 5. Dependency Reproducibility Is Not Locked

`pyproject.toml` and `requirements.txt` specify ranges, not a lockfile. CI and
local experiments can drift as PyTorch, PyG, pandas, scipy, or Hydra versions
move.

Recommended next step:

- Add `uv.lock` or another explicit lockfile.
- Document CPU CI install versus CUDA training install.
- Pin known-good PyTorch/PyG compatibility in docs.

### 6. Universe Bias Is Mitigated, Not Solved

Per-split completeness and PIT universe filtering are important improvements,
but the default still depends on how the input dataset was assembled. If the raw
source is already survivorship-biased, config flags cannot fully repair it.

Recommended next step:

- Prefer PIT universe files for serious backtests.
- Record nominal universe size, filtered universe size, and dropped tickers in
  run metadata.
- Treat backtests without PIT data as research diagnostics, not production-grade
  claims.

### 7. Global Regime Features Are Still Broadcast Per Stock

Global regime, VIX, and credit variables are still feature columns that get
broadcast into each stock row. That works, but it is parameter-inefficient and
forces the model to rediscover that those values are market-level.

Recommended next step:

- Add an optional market-token or FiLM conditioning path.
- Feed global features once per batch/date.
- Modulate A1/A2/B1/B2 latent states instead of repeating global scalars across
  all stocks.

This is a good Phase 5 model experiment after the current flags are ablated.

### 8. The Graph Stream Still Sees One Cross-Section Per Date

The temporal stream sees `his_t` days. The graph stream still receives a single
feature vector per stock for the prediction date. This matches the paper more
closely, but it leaves temporal context mostly outside message passing.

Recommended next step:

- Add an optional graph input summarizer: last-day, rolling mean, rolling
  volatility, or learned pooling over the last `k` days.
- Keep the default last-day path for paper comparison.
- Compare pooled graph input against A1/A2 cross-attention; they may be
  complementary.

### 9. Repo Hygiene Still Needs Attention

The package is workable, but the repo root still contains generated or bulky
artifacts such as PDFs, notebooks, cached bytecode, `_uncertain/`, and large
seed outputs.

Recommended next step:

- Move generated experiment artifacts out of the import/repo root path.
- Keep only curated benchmark outputs under version control.
- Consider Git LFS or external artifact storage for PDFs and large result
  bundles.
- Move script-like backtests out of `tests/` if they are not intended as pytest
  tests.

## Updated Roadmap

### Near Term: Make The Current System Measurable

- [ ] Build an ablation matrix around `paper_faithful` vs modern defaults.
- [ ] Run at least one walk-forward comparison with bootstrap CIs.
- [ ] Record results in a small decision table under `docs/` or `seed_results/`.
- [ ] Promote only statistically supported flags into default configs.

### Near Term: Data Robustness

- [ ] Improve NaN handling with same-stock forward fill plus missing indicators.
- [ ] Add universe/drop-count metadata to each run.
- [ ] Prefer PIT universe filters for any headline backtest.

### Mid Term: Runtime And Sweep Efficiency

- [ ] Add graph snapshot caching.
- [ ] Add gradient accumulation.
- [ ] Add dependency lockfile and environment notes.
- [ ] Add a reduced-cost Optuna profile for quick search and a full-cost profile
  for final verification.

### Mid Term: Model Research

- [ ] Test A1/A2 cross-attention as an enabled default.
- [ ] Compare `gru_attn` versus `transformer` under identical walk-forward
  windows.
- [ ] Test graph pooled temporal summaries.
- [ ] Add market-token or FiLM conditioning for regime/macro features.
- [ ] Explore multi-horizon heads for 1d/5d/20d auxiliary supervision.

### Later: Production-Grade Backtesting

- [ ] Add sector/factor neutralisation options to portfolio construction.
- [ ] Add long/short portfolio evaluation, not only long top-k.
- [ ] Add volatility targeting and exposure constraints.
- [ ] Align paper-trade, research backtest, and training evaluation around one
  shared portfolio interface where practical.

## What Not To Change Yet

- Do not remove the four-stream A1/A2/B1/B2 structure. It remains a useful
  inductive bias and a good experimental scaffold.
- Do not remove `paper_faithful.yaml`; it is the control group.
- Do not collapse typed dataclass config into loose dictionaries. The validation
  has already paid for itself through embargo, graph, and model-shape checks.
- Do not regress dynamic graphs to per-batch graph rebuilds.
- Do not let paper-trade inference depend on training-time graph construction.
- Do not promote every new flag into the default config without ablation.

## Bottom Line

The first architecture review identified basic modernization gaps. Most of
those are now fixed. The new challenge is evidence: deciding which of the
implemented upgrades actually improve out-of-sample behavior after costs,
turnover, overlapping-return corrections, and uncertainty intervals.

The repo is ready for new feature work, but the best next feature is probably
not another model knob. It is a tight ablation and evaluation loop that tells us
which knobs deserve to stay on.
