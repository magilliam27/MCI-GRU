# MCI-GRU Research Opportunity Scan - 2026-06-19

Companion document: `docs/research/current/MCI_GRU_PROGRAM_MAP_2026-06-19.md`.

This scan fans out from the program map into 17 component lanes. Each lane was
handled as a read-only research task, with emphasis on practical techniques that
are already used in recent finance ML, time-series ML, portfolio construction,
or production ML systems, and that MCI-GRU is not already using as a first-class
contract. More speculative ideas are separated from the near-term queue.

Representative source links were spot-checked on 2026-06-19. Several source
names without direct links below should be treated as research leads to verify
again before opening implementation tickets.

## Executive Takeaways

The strongest opportunities are not all new model architectures. A lot of the
highest-confidence upside is in cleaner data admissibility, execution-aware
evaluation, and run governance. That is good news: many of the first experiments
can be done with saved predictions or additive metadata before burning training
GPU.

Highest-confidence near-term changes:

1. Add a tradability overlay and event audit around the PIT masked panel.
2. Add OHLCV-only market-state, realized-risk, liquidity, and staleness features.
3. Try same-day cross-sectional rank or robust normalization as a direct
   alternative to train-window z-score.
4. Try volatility-scaled forward-return labels.
5. Add snapshot-bucket batching for dynamic graphs.
6. Add EMA checkpoint evaluation and smoothed early stopping.
7. Run a saved-prediction model-selection audit with trial ledger, DSR/PSR/PBO,
   paired dependent tests, factor/regime slices, and rank-decay diagnostics.
8. Run an execution-cost-capacity replay before more alpha tuning.
9. Add run-bundle manifests, artifact validation, and reproducibility ledgers.
10. Add a shadow-mode pre-trade health gate before live portfolio construction.

Highest-upside model/loss ablations:

1. Replace or augment A1 with a small PatchTST-lite temporal encoder.
2. Add a ListNet/ListMLE-style daily listwise ranking loss.
3. Add Spearman/Kendall dependence metrics for dynamic graph construction.
4. Add right-aligned multi-resolution context summaries.
5. Replay ensemble aggregation alternatives: rank aggregation, validation
   weighting, uncertainty-shrunk scores, and trimmed or median aggregation.

## Priority Matrix

| Priority | Component | First practical experiment | Why it comes early |
| --- | --- | --- | --- |
| P0 | Evaluation/tracking | Model-selection audit v0 plus run-bundle audit | No retraining; raises trust in all later comparisons. |
| P0 | Backtesting | Execution-cost-capacity replay | Tests whether current alpha survives more realistic execution. |
| P0 | Paper trading | Shadow-mode pre-trade guard | Improves live safety without changing model scores. |
| P1 | Data/PIT | Tradability overlay plus observation-staleness channels | Fixes investability and missingness before model changes. |
| P1 | Features | OHLCV market-state, realized-risk, liquidity pack | Uses existing data and likely improves regime sensitivity. |
| P1 | Normalization | Cross-sectional rank/robust normalization | Directly matches ranking objective and recent stock-ranking practice. |
| P1 | Labels | Volatility-scaled 5-day returns | Reduces noisy high-volatility label domination. |
| P1 | Training | EMA checkpoint evaluation | Low-risk stabilizer for noisy validation metrics. |
| P2 | Graph | Spearman/Kendall graph metric | Minimal surface change; tests nonlinear/rank dependence. |
| P2 | Loss | ListNet daily rank loss | Matches portfolio ranking use case better than pointwise returns. |
| P2 | Architecture | PatchTST-lite A1 replacement | Larger model-risk surface, but strong time-series precedent. |
| P2 | Ensemble | Saved-prediction aggregation replay | No retraining if per-model predictions exist. |
| P3 | External data | Delisting returns, fundamentals, news/text | Potentially valuable, but higher vendor/provenance risk. |

## Suggested Implementation Waves

Wave 0 - evidence hygiene before more model search:

- Add `trial_ledger.csv` / `trial_ledger.jsonl` for all candidate runs and
  backtest replays.
- Generate additive `artifact_manifest.json`, `artifact_validation.json`,
  `reproducibility_manifest.json`, and dry-run `promotion_decision.json`.
- Add saved-prediction evaluation reports for DSR/PSR/PBO-lite, paired HAC or
  block-bootstrap comparisons, rank decay, factor exposure, regime slices, and
  capacity diagnostics.

Wave 1 - no-new-vendor-data model inputs:

- Tradability overlay: rolling dollar volume, minimum price, stale/zero-volume
  flags, and next-open availability.
- Missingness/staleness channels for masked PIT panels.
- OHLCV feature pack: market breadth, realized-risk shape, liquidity/friction,
  sector-relative and market-residual transforms.
- `normalisation=cs_rank_gauss` or `cs_robust_z`.
- `label_transform=vol_scaled` for 5-day returns.

Wave 2 - training/model ablations:

- EMA checkpointing and smoothed early stopping.
- Dynamic graph `dependence_metric=spearman|kendall`.
- Snapshot-bucket batch sampler for dynamic graph snapshots.
- ListNet/ListMLE rank loss.
- PatchTST-lite A1 encoder.
- Saved-prediction ensemble aggregation replay.

Wave 3 - higher-risk or external-data work:

- Delisting-aware total-return labels and terminal event P&L.
- Permanent security master / identifier lineage.
- Fundamentals, text/news, or external factor exposures as PIT features.
- Distributional heads, conformal abstention, and utility/path-aware losses.
- Learned graph or graph-temporal encoders beyond the current correlation graph.

## Component Lane Findings

### 1. Data Source And Universe

Current gap:

- MCI-GRU supports CSV/LSEG/index data and true PIT masked panels, but does not
  yet have a first-class tradability overlay, permanent security master, or
  delisting-aware total-return contract.

First experiment:

- Add a strict past-only tradability overlay: rolling 20/60-day median dollar
  volume, minimum price, zero-volume and stale-bar flags, and next-open
  availability. Keep it separate from PIT membership so membership breadth is
  preserved while non-investable rows are masked from loss/trading.

Implementation surfaces:

- `mci_gru/config.py` `DataConfig`
- `mci_gru/data/pit.py`
- `mci_gru/pipeline.py`
- New Hydra preset such as
  `configs/experiment/pit_temporal_2025_tradability_overlay.yaml`

Guardrails:

- Do not replace masked-panel PIT breadth with complete-stock filtering.
- Use only data available before the prediction/trade date.
- Keep event/effective-date semantics explicit.

Source leads:

- CRSP delisting/corporate-action practice.
- S&P US Indices Methodology, June 2026.
- Harjot Singh Ranse, "Survivorship Bias in Emerging Market Small-Cap
  Indices", 2026, https://arxiv.org/abs/2603.19380.
- "Implementation Risk in Portfolio Backtesting", 2026,
  https://arxiv.org/abs/2603.20319.

### 2. PIT Handling

Current gap:

- Masked-panel PIT preserves breadth, but missingness, staleness, and terminal
  exit behavior are not yet exposed as first-class model/evaluation features.

First experiment:

- Add observation-mask and staleness channels:
  `feature_missing_any`, `missing_count`, and `days_since_observed`.

Implementation surfaces:

- `DataConfig`
- imputation / masked-panel branch in `mci_gru/pipeline.py`
- `tests/test_pit_masked_panel.py`
- `configs/experiment/pit_temporal_2024_missingness.yaml`

Guardrails:

- Missingness features must be computed from the input panel as observed on or
  before the sample date.
- Loss and prediction export must still ignore non-tradable or inactive names.

Source leads:

- Ranse 2026 survivorship-bias study, https://arxiv.org/abs/2603.19380.
- Chen and McCoy, revised 2024 missing-values work.
- SAITS and PyPOTS missing-time-series tooling.
- QuantConnect delisting and liquidity universe documentation.
- Nasdaq trade halt data/documentation.

### 3. Feature Engineering

Current gap:

- The current feature set has momentum, volatility, regime, VIX/credit, RSI, MA,
  price, and volume features, but it does not yet have a compact OHLCV
  microstructure pack for cross-sectional market state and investability.

First experiment:

- Add an OHLCV-only feature pack:
  market-state breadth, realized-risk shape, liquidity/trading-friction
  proxies, sector-relative features, and market-residual transforms.

Implementation surfaces:

- New feature modules under `mci_gru/features/`
- `mci_gru/features/registry.py`
- `FeatureConfig`
- `configs/features/with_panel_microstructure.yaml`

Guardrails:

- Trailing windows only.
- Cross-sectional ranks or residuals must be same-day and computed only across
  scoreable active names.
- Do not mix feature-family ablations when estimating lift.

Source leads:

- Fang and Slepaczuk, 2026, OHLCV/financial time-series evidence.
- Wang, 2023, market-state and trading signals.
- Chen, Koike, and Shau, 2024, realized-risk features.
- GRU-PFG, DGDNN, and High-Throughput Asset Pricing.

### 4. Normalization

Current gap:

- MCI-GRU has train-window z-score and rank-Gaussian references. It does not yet
  expose a same-date cross-sectional robust/rank normalization mode as the
  default way to match ranking objectives.

First experiment:

- Add `normalisation=cs_rank_gauss`: for each date, rank active PIT stocks per
  feature and map ranks to a Gaussian or centered rank score.

Implementation surfaces:

- `mci_gru/data/preprocessing.py`
- `mci_gru/pipeline.py`
- `DataConfig.normalisation`
- `configs/experiment/normalisation_cs_rank_gauss.yaml`

Guardrails:

- Same-day cross-sectional transforms may use all active names at that date, but
  must not use future dates.
- Paper-trade inference needs parity for the same transform.

Source leads:

- RankGLU, 2026, https://arxiv.org/abs/2606.08930.
- Qlib processor conventions.
- GAS-Norm, 2024.
- QuantRocket sector-neutralization documentation.
- Dish-TS, FAN, IN-Flow, WDAN, and TimeAPN normalization work.

### 5. Labels

Current gap:

- Labels are raw forward returns or same-day rank labels. The current contract
  does not yet normalize labels by ex-ante risk, residualize them, or expose
  multi-horizon targets.

First experiment:

- Add `training.label_transform=vol_scaled`: divide the forward 5-day return by
  trailing ex-ante realized volatility, with floor and clip parameters.

Implementation surfaces:

- `mci_gru/data/preprocessing.py`
- `mci_gru/pipeline.py`
- `TrainingConfig`
- `configs/experiment/label_vol_scaled_5d.yaml`

Guardrails:

- The volatility denominator must end before the label period begins.
- Keep raw labels available for audit and backtest attribution.

Source leads:

- Deep Momentum Networks.
- Ong and Herremans, 2023.
- Liu, Roberts, and Zohren, 2023.
- Numerai target-normalization documentation.
- Barunik, Hronec, and Tobek; Petursson and Oskarsdottir.

### 6. Tensor And Sample Construction

Current gap:

- The temporal tensor uses one `his_t` window. It does not yet expose
  right-aligned multi-resolution summaries or masks as explicit sample context.

First experiment:

- Add right-aligned multi-resolution context summaries, for example 10-day,
  21-day, and 63-day windows ending on the same label date, summarized by last,
  mean, and standard deviation.

Implementation surfaces:

- `mci_gru/data/preprocessing.py`
- `pipeline._build_tensors`
- `ModelConfig.context_windows`
- `ModelConfig.context_summary_stats`

Guardrails:

- Keep all windows right-aligned to the same label date.
- Start by appending summaries to graph features, leaving the temporal tensor
  stable.

Source leads:

- PatchTST, 2022, https://arxiv.org/abs/2211.14730.
- MTST, TimeMixer, and long-context time-series work.
- DGRCL, S4M, MissTSM, graph mini-batching work.

### 7. Correlation Graph

Current gap:

- Current graph construction is Pearson correlation with optional top-k,
  multi-feature edges, lead-lag features, snapshot age, and sector relation. It
  does not yet test rank, robust, partial, or nonlinear dependence metrics.

First experiment:

- Add `graph.dependence_metric=pearson|spearman|kendall`, keeping edge feature
  width unchanged at first.

Implementation surfaces:

- `mci_gru/graph/builder.py`
- `GraphConfig`
- `configs/experiment/correlation_dynamic_spearman.yaml`
- `configs/experiment/correlation_dynamic_kendall.yaml`

Guardrails:

- Keep dynamic graph snapshots strictly prior to the sample date.
- If edge columns change later, update `run_experiment._edge_feature_dim`.

Source leads:

- THGNN, DGDNN, MDGNN, GRU-PFG, and DGRCL.
- DGT S&P 500, 2025, https://arxiv.org/abs/2506.18717.
- DeltaLag, Hermes Hypergraph, PCGLASSO, sklearn sparse inverse covariance.

### 8. DataLoader, Collate, And Batching

Current gap:

- The 9-tuple collate contract is solid, but dynamic graph batches can mix dates
  that resolve to different snapshots. There is no snapshot-aware sampler.

First experiment:

- Add `SnapshotBucketBatchSampler` so dynamic-graph batches group dates by graph
  snapshot, reducing repeated graph lookup/collation work and making snapshot
  behavior easier to audit.

Implementation surfaces:

- `mci_gru/data/data_manager.py`
- `TrainingConfig.batch_sampler=none|snapshot_bucket`
- `tests/test_dynamic_graph_updates.py`

Guardrails:

- Preserve the 9-tuple return contract.
- Never let inactive PIT nodes send graph messages.
- Keep shuffle determinism explicit.

Source leads:

- Temporal Graph Benchmark and DyGLib.
- THGNN and MDGNN batching practices.
- PyTorch DataLoader reproducibility documentation.
- PyG loader and sparse edge-index documentation.

### 9. Model Architecture

Current gap:

- The four-stream MCI-GRU architecture is custom and graph-aware, but A1 is
  still mostly GRU-attention style rather than patch/variate-token transformer
  style. The prediction head is also not explicitly bounded/gated for stable
  rank formation.

First experiment:

- Add a `PatchTST-lite` A1 option with no external pretraining. Keep A2, B1/B2,
  final GAT, loss, PIT handling, and graph construction fixed.

Implementation surfaces:

- `mci_gru/models/mci_gru.py`
- `ModelConfig.temporal_encoder`
- `run_experiment.create_model`
- model shape tests

Guardrails:

- Output remains `(batch, stocks)`.
- Stock masks must still zero inactive nodes across temporal, graph, attention,
  and output paths.
- Evaluate against the frozen recipe before adding foundation-model pretraining.

Source leads:

- PatchTST, iTransformer, TimeMixer.
- Chronos, TimesFM, Time-MoE, Chronos-2 finance, and finance TSFM studies.
- MASTER, GRU-PFG, THGNN, SAMformer, S-Mamba, Mamba4Cast.
- RankGLU score-head evidence, https://arxiv.org/abs/2606.08930.

### 10. Loss And Objectives

Current gap:

- Current losses include MSE, IC, combined MSE/IC, PortfolioIC, and
  LambdaRankIC. MCI-GRU does not yet have a simple daily listwise probability
  ranking loss such as ListNet/ListMLE as a first-class baseline.

First experiment:

- Add `loss_type=listnet_rank`: per-date finite mask, label ranks or z-scored
  returns as softmax targets, and a temperature grid of `0.05, 0.1, 0.25, 0.5`.

Implementation surfaces:

- `mci_gru/training/losses.py`
- `TrainingConfig.loss_type`
- validation metrics for `val_ndcg_at_10`, `precision_at_10`, and top-bottom
  spread

Guardrails:

- Compute listwise loss only within each date and finite mask.
- Keep pure IC and LambdaRankIC as baselines.
- Do not promote path-dependent utility losses without a leakage review.

Source leads:

- Stock-ranking loss benchmark by Kwiatkowski and Chudziak, 2025,
  https://arxiv.org/abs/2510.14156.
- ListFold, Smooth-NDCG, diffNDCG, adaptive top-k ranking.
- Barunik, Hronec, and Tobek; Petursson and Oskarsdottir.

### 11. Training Loop

Current gap:

- Trainer uses AdamW, optional AMP, gradient clipping, warmup/cosine scheduler,
  and raw early stopping. It does not yet evaluate EMA weights, schedule-free
  optimizers, late SAM, or smoothed early stopping.

First experiment:

- Add EMA shadow weights and raw-vs-EMA evaluation/checkpointing.

Implementation surfaces:

- `mci_gru/config.py` `TrainingConfig`
- `mci_gru/training/trainer.py` optimizer step and checkpoint paths

Guardrails:

- Log raw and EMA metrics side by side.
- Do not replace the raw checkpoint until EMA wins on held-out evidence.
- Keep raw validation metrics even if smoothed metrics drive checkpointing.

Source leads:

- Schedule-Free AdamW, "The Road Less Scheduled", 2024,
  https://arxiv.org/abs/2405.15682.
- EMA weight studies, 2024.
- SAMformer and late-SAM/ASAM studies.
- Noisy Early Stopping, 2024.
- Lion and AdEMAMix optimizer papers as later ablations.

### 12. Ensemble

Current gap:

- MCI-GRU averages independent seed models uniformly. It does not yet exploit
  per-model disagreement for uncertainty, rank aggregation, trimmed means, or
  validation-constrained nonnegative weights.

First experiment:

- Replay existing per-model predictions with equal mean, validation-IC softmax
  weights, Borda/average-rank/RRF aggregation, median/trimmed aggregation, and
  `mean - lambda * std` uncertainty-shrunk scores.

Implementation surfaces:

- `mci_gru/training/trainer.py` `train_multiple_models`
- saved-prediction evaluation scripts
- optional ensemble combiner helper

Guardrails:

- Fit any weights only on train/validation windows.
- Compare every combiner to uniform mean.
- Report prediction dispersion, rank disagreement, turnover, and top-k returns.

Source leads:

- Uncertainty-Adjusted Sorting, 2026, https://arxiv.org/abs/2601.00593.
- Forecast-stability and adaptive robust time-series ensemble studies.
- Borda count and reciprocal-rank-fusion literature.
- Deep ensembles and GNN uncertainty studies.

### 13. Walk-Forward And Time-Split Validation

Current gap:

- Walk-forward window generation exists, but there is not yet a central
  split-audit report with label-interval purging, embargo verification, and
  regime-stratified stability summaries.

First experiment:

- Build a read-only split audit: for each sample, compute information interval,
  verify embargo/purge safety, and summarize OOS IC/rank IC/top-k returns across
  rolling or expanding windows.

Implementation surfaces:

- `mci_gru/walkforward.py`
- saved-prediction evaluation report
- `evaluation_summary.json`

Guardrails:

- Purge at least `label_t` trading sessions, and account for overlapping labels.
- Keep hyperparameter selection nested or predeclared.

Source leads:

- Bailey et al. CSCV/PBO work.
- Spurious Predictability in Financial ML, 2026,
  https://arxiv.org/abs/2604.15531.
- Adaptive window-selection and rolling-origin evaluation studies.
- sklearn `TimeSeriesSplit` documentation as a baseline contrast.

### 14. Evaluation

Current gap:

- MCI-GRU already reports IC, rank IC, top-k returns, Newey-West Sharpe, and
  bootstrap CIs. It lacks a central trial-ledger, DSR/PSR/PBO contract, paired
  dependent model-comparison tests, rank-decay diagnostics, and factor/regime
  attribution.

First experiment:

- Run saved-prediction "model-selection audit v0" on an existing sweep:
  trial ledger, DSR/PSR or Harvey-Liu haircuts, PBO-lite, paired HAC/bootstrap
  comparisons, rank decay, turnover/cost diagnostics, and factor/regime slices.

Implementation surfaces:

- `mci_gru/evaluation/statistics.py`
- `mci_gru/evaluation/prediction_report.py`
- `evaluation_summary.json`
- backtest summary artifacts

Guardrails:

- The trial ledger must include ugly and failed variants.
- Align prediction/label rows exactly for paired tests.
- Treat regime slices as diagnostics unless predeclared.

Source leads:

- Harvey and Liu, "Evaluating Trading Strategies", 2014,
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2474755.
- Harvey and Liu, "Backtesting", 2015,
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2345489.
- Bailey et al., "Probability of Backtest Overfitting", 2015,
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253.
- Nikolopoulos, "Spurious Predictability in Financial ML", 2026,
  https://arxiv.org/abs/2604.15531.
- Alphalens documentation, https://alphalens.ml4trading.io/notebooks/overview.html.
- Kenneth French Data Library,
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html.

### 15. Backtesting

Current gap:

- Current backtests have strong timing and cost foundations, but the gate is
  mostly rank-drop based. There is no standardized execution-timing matrix,
  cost-aware no-trade buffer, ADV capacity replay, event audit, or central
  implementation-risk report.

First experiment:

- Run saved-prediction execution-cost-capacity replay: daily open-to-open,
  top-10, PIT masked panel, fixed transaction-cost baseline, rank-drop thresholds
  `off, 10, 30, 60`, cost-aware no-trade buffer, and AUM/ADV capacity tiers.

Implementation surfaces:

- `tests/backtest_sp500.py`
- `mci_gru/evaluation/portfolio.py`
- `mci_gru/evaluation/statistics.py`
- new outputs such as `capacity_report.csv`,
  `backtest_trial_ledger.csv`, and `event_audit.csv`

Guardrails:

- Freeze timing/cost policies before seeing results.
- Prediction at date `t` can trade only from the next valid open onward.
- Use trailing ADV/volatility ending at prediction date for capacity estimates.

Source leads:

- AutoQuant, 2025, https://arxiv.org/abs/2512.22476.
- Bysik and Slepaczuk, 2026, transaction-cost threshold study.
- Realistic Market Impact Modeling, 2026,
  https://arxiv.org/abs/2603.29086.
- Cost-aware Portfolios, 2024, https://arxiv.org/abs/2412.11575.
- Markowitz Portfolio Construction at Seventy, 2024,
  https://arxiv.org/abs/2401.05080.
- Implementation Risk in Portfolio Backtesting, 2026,
  https://arxiv.org/abs/2603.20319.

### 16. Paper Trading And Live Inference

Current gap:

- Paper-trade inference correctly loads frozen metadata/checkpoints/graph data,
  but live controls are mostly post-inference/post-portfolio. There is no
  shadow-mode pre-trade health gate or uncertainty/capacity-aware score overlay.

First experiment:

- Add a shadow-mode pre-trade guard after inference and before portfolio
  construction. Emit `guard_status.json` with freshness, coverage,
  missing/stale/zero-volume counts, PSI/KS drift, score-distribution shift,
  projected turnover, and ADV participation.

Implementation surfaces:

- `paper_trade/scripts/run_nightly.py`
- `paper_trade/scripts/monitor.py`
- `mci_gru/evaluation/portfolio.py`
- paper-trade output artifacts

Guardrails:

- Keep paper-trade inference frozen: do not call `GraphBuilder`.
- Start in shadow mode; only later promote `WARN` to `NO_NEW_BUYS` or
  `NO_TRADE`.

Source leads:

- NIST AI Risk Management Framework,
  https://www.nist.gov/itl/ai-risk-management-framework.
- PRA SS1/23 model risk principles,
  https://www.bankofengland.co.uk/prudential-regulation/publication/2023/may/model-risk-management-principles-for-banks-ss.
- Cost-aware portfolio work, 2024, https://arxiv.org/abs/2412.11575.
- QuantConnect liquidity universe, execution, portfolio construction, and risk
  management documentation.
- Conformal predictive portfolio selection and uncertainty quantification
  papers.

### 17. Outputs, Tracking, And Reproducibility

Current gap:

- Filesystem artifacts are the source of truth, with MLflow additive. The run
  folder does not yet have content-addressed manifests, schema validation,
  promotion decision records, or a reproducibility envelope.

First experiment:

- Run a read-only run-bundle audit on one promoted baseline and one candidate.
  Generate `artifact_manifest.json`, `artifact_validation.json`,
  `reproducibility_manifest.json`, and `promotion_decision.dry_run.json`.

Implementation surfaces:

- output closeout helper or script
- MLflow artifact logging
- paper-trade promotion tooling
- optional `paper_trade_candidate_registry.jsonl`

Guardrails:

- Additive artifacts first; do not block older valid runs until schema versions
  are settled.
- Record determinism settings before enforcing them.
- Thresholds for promotion must compare seed/window-matched baselines.

Source leads:

- MLflow Tracking, Dataset Tracking, Registry, and Model Signatures docs.
- OpenLineage object model docs.
- PyTorch reproducibility docs,
  https://docs.pytorch.org/docs/2.12/notes/randomness.html.
- Great Expectations validation checkpoints.
- TFX Evaluator.
- Evidently drift reports.
- DVC experiment/data versioning.
- SLSA provenance and in-toto attestations.

## Practical Versus Speculative Queue

Practical, lower-friction:

- Tradability overlay.
- Observation-mask and staleness features.
- OHLCV microstructure feature pack.
- Cross-sectional robust/rank normalization.
- Volatility-scaled labels.
- Snapshot-bucket sampler.
- EMA checkpointing.
- ListNet rank loss.
- Spearman/Kendall graph.
- Saved-prediction ensemble replay.
- Split audit, model-selection audit, and execution-cost-capacity replay.
- Paper-trade shadow guard.
- Run-bundle manifests and validation.

Speculative or design-first:

- Learned dynamic graphs or graph-temporal encoders beyond correlation snapshots.
- Distributional return heads and conformal abstention.
- Path-dependent utility, CVaR, drawdown, or turnover-aware training losses.
- External text/news/fundamental features.
- Full security-master reconstruction with delisting return integration.
- OpenLineage event graph and signed SLSA/in-toto run attestations.
- Cross-engine replay harness.
- Long-short borrow/locate feasibility layer.

## Research Control Principles

- Keep the frozen default recipe unchanged unless a change is explicitly being
  promoted.
- Prefer saved-prediction replay before retraining when the question is
  evaluation, backtesting, ensemble aggregation, or run governance.
- Promote one component at a time. Use the program map to name the changed layer:
  data, PIT, feature, normalization, label, tensor, graph, batching, model, loss,
  training, ensemble, split, evaluation, backtest, paper-trade, or tracking.
- For model-changing experiments, keep the graph, label, loss, and evaluation
  constant unless the experiment is specifically about those layers.
- Every research result should state whether it is alpha evidence, execution
  evidence, governance evidence, or live-safety evidence.
- Any result used for promotion should identify the trial family and include all
  tried variants, not just the winner.
