# MCI-GRU Top-University Research Scan

Date: 2026-06-21

Status: new scan from scratch, stricter source gate than
`MCI_GRU_RESEARCH_OPPORTUNITY_SCAN_2026-06-19.md`.

Purpose: identify practical MCI-GRU research opportunities supported only by
primary research sources whose authors include verified top-university
affiliations.

## Source Gate

This scan used a fail-closed gate:

- Include only papers, proceedings, technical reports, official project pages,
  journal pages, SSRN pages, arXiv PDFs, PMLR/OpenReview/ACM-style pages, or
  official university/author pages.
- Include only sources where at least one primary author was verified as
  affiliated, at publication or preprint time, with a globally top research
  university.
- Record author and institution evidence in this report.
- Exclude useful-looking papers when affiliation was missing, industry-only, or
  not clearly in the top-university set.
- Treat current code, `AGENTS.md`, and repo invariants as stronger than research
  claims.

Subagent process: 17 component lanes were covered with independent read-only
subagents. The first six lanes launched immediately; new lanes were launched as
the session concurrency limit freed slots. The main synthesis also checked repo
anchors and selected primary-source evidence.

Repo anchors reviewed:

- `AGENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/CONFIGURATION_GUIDE.md`
- `docs/BACKTEST_FAIRNESS_AUDIT.md`
- `docs/OUTPUT_MANAGEMENT.md`
- `docs/MLFLOW_TRACKING.md`
- `docs/research/current/MCI_GRU_PROGRAM_MAP_2026-06-19.md`
- `mci_gru/config.py`
- `mci_gru/pipeline.py`
- `mci_gru/data/data_manager.py`
- `mci_gru/data/preprocessing.py`
- `mci_gru/features/registry.py`
- `mci_gru/graph/builder.py`
- `mci_gru/models/mci_gru.py`
- `mci_gru/training/losses.py`
- `mci_gru/training/trainer.py`
- `mci_gru/evaluation/statistics.py`
- `mci_gru/evaluation/portfolio.py`
- `mci_gru/walkforward.py`
- `paper_trade/scripts/infer.py`
- `paper_trade/scripts/portfolio.py`
- `paper_trade/scripts/monitor.py`
- `paper_trade/scripts/run_nightly.py`

## Executive Priority Matrix

| Priority | Opportunity | Why now | First evidence action |
| --- | --- | --- | --- |
| P0 | Split audit, trial ledger, and run-bundle manifest | Prevents another research-map-to-experiment loop from overclaiming. | Docs/tooling over existing configs and artifacts; no retraining. |
| P0 | Saved-prediction ensemble dispersion replay | Uses already-produced `predictions_model_*` and tests uncertainty value cheaply. | Tune dispersion penalty on validation only; replay backtests on frozen predictions. |
| P0 | PIT availability, tradability, and staleness audit | Strengthens masked-panel PIT without reducing breadth. | Emit breadth/staleness tables before changing masks or features. |
| P1 | Cross-sectional feature rank/gauss transform | Finance-native normalization supported by top asset-pricing papers. | Add as config experiment; no default change. |
| P1 | Observation/staleness channels | Preserves missingness signal before imputation erases it. | Synthetic no-lookahead tests, then PIT ablation. |
| P1 | Execution-cost/capacity replay and corrected timing | Backtest fairness is prerequisite to portfolio claims. | Saved-prediction replay with explicit prediction, execution, and return timing. |
| P2 | Snapshot-bucket dynamic batching | Runtime win that can preserve the 9-tuple contract. | Instrument current dynamic batches, then default-off sampler. |
| P2 | SWA/SWAG-lite or snapshot prediction ensembles | Training-loop improvement with limited data-surface risk. | New experiment preset after governance artifacts exist. |
| P2 | Market-state feature gate | Practical model ablation from MASTER/TFT-style gating. | Small gated model option, no frozen default change. |
| P3 | Forward-correlation edge features | Potential graph upgrade, but high leakage risk. | Graph-zero/residual-correlation diagnostics before edge forecaster. |
| P3 | Listwise soft-rank objective | Interesting, but existing LambdaRankIC already covers part of this seam. | Compare only after validation ledger and rank metrics are firm. |
| P3 | Differentiable portfolio/objective layers | High blast radius and timing-sensitive. | Design note only until execution replay is trusted. |

## Component Findings

### 1. Data Source and Universe

Practical gap: MCI-GRU has true PIT masked-panel breadth, but not a separate
security-master/tradability layer. Additive audits should distinguish active
membership, feature readiness, loss eligibility, next-open availability, stale
bars, min price, rolling dollar volume, primary/common listing flags, and
execution eligibility.

Recommended first experiment: PIT tradability and staleness overlay using only
existing OHLCV/PIT data. Keep `masked_panel` breadth; narrow only a separate
tradability/export/evaluation mask.

Repo surfaces: `mci_gru/data/pit.py`, `mci_gru/pipeline.py`,
`mci_gru/config.py`, `mci_gru/features/registry.py`, `run_metadata.json`.

Guardrails: do not replace masked-panel PIT with complete-stock filtering.
Rolling liquidity/staleness windows must end before or at the prediction date.

Source evidence:

- Jensen, Kelly, Pedersen, Global Factor Data documentation:
  Theis Jensen and Bryan Kelly are Yale-affiliated on the documentation/source
  pages; the data rules include common/primary/main-exchange screens and lag
  conventions. Passes via Yale. Source:
  https://jkpfactors-data.s3.amazonaws.com/documents/Documentation.pdf
- Gu, Kelly, Xiu, "Empirical Asset Pricing via Machine Learning":
  Chicago Booth, Yale SOM, Chicago Booth affiliations are shown on SSRN/PDF.
  Practical for large cross-sectional equity ML, characteristic handling, and
  liquidity/volatility signals. Source:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577
- Abe and Nakayama, "Deep Learning for Forecasting Stock Returns in the
  Cross-Section": Hideki Nakayama is University of Tokyo in the PDF. Practical
  evidence for reporting lags, missing thresholds, and cross-sectional rank
  scaling. Source: https://arxiv.org/pdf/1801.01777

### 2. PIT Handling

Practical gap: masked-panel PIT is strong, but feature-level `available_at`
calendars and artifact-level proof are not first-class. A feature should carry
observed date, release date, usable date, and staleness policy when it depends
on macro/fundamental/external data.

Recommended first experiment: PIT availability audit report over existing
masked-panel presets, with per-date counts for `active_member`, `feature_ready`,
`loss`, `tradable`, `next_open_available`, days since membership change, and
feature staleness.

Repo surfaces: `mci_gru/pipeline.py`, `mci_gru/data/fred_loader.py`,
`mci_gru/config.py`, `mci_gru/walkforward.py`, `run_experiment.py`.

Guardrails: this supplements label embargo and PIT masks; it does not weaken
them.

Source evidence:

- Chen, Pelger, Zhu, "Deep Learning in Asset Pricing": all authors are Stanford
  in the PDF; practical relevance is distinct monthly/yearly update timing.
  Source: https://arxiv.org/pdf/1904.00745
- Bailey, Borwein, Lopez de Prado, Zhu, "The Probability of Backtest
  Overfitting": Lopez de Prado is Cornell ORIE on source pages; practical
  relevance is CSCV/PBO model-selection leakage. Source:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Gao, Jiang, Yan, "Detecting Lookahead Bias in LLM Forecasts": CUHK Business
  School affiliation in PDF; relevant if text/LLM features are ever added.
  Source: https://arxiv.org/pdf/2512.23847

### 3. Feature Engineering

Practical gaps: MCI-GRU is OHLCV/momentum/vol/macro-heavy but lacks
chart-path descriptors, broad PIT characteristic families, learned macro-state
embeddings, and explicit feature-selection/interaction audits.

Recommended first experiment: add low-dimensional OHLCV chart-path descriptors
before any CNN/image infrastructure: drawdown, rebound from local low, return
acceleration, range compression, realized skew, volume shock, and distance from
rolling high/low.

Repo surfaces: `mci_gru/features/registry.py`, new optional feature module,
`FeatureConfig`, `configs/features/`, no-lookahead feature tests.

Guardrails: trailing windows only; no future-normalized charts; keep frozen
recipe unchanged.

Source evidence:

- Jiang, Kelly, Xiu, "Re-(Imag)ining Price Trends": University of Chicago,
  Yale SOM, and Chicago Booth affiliations are shown on SSRN. Practical
  evidence for learned price-path patterns. Source:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587
- Jensen, Kelly, Pedersen, "Is There a Replication Crisis in Finance?":
  Yale/Yale SOM evidence on SSRN/source pages; practical for characteristic
  theme families. Source:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3774514
- Gu, Kelly, Xiu, "Autoencoder Asset Pricing Models": Chicago Booth, Yale SOM,
  and Chicago Booth affiliations on SSRN/PDF. Practical for nonlinear exposure
  embeddings. Source:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3335536

### 4. Normalization

Practical gaps: current z-score and train-fit rank Gaussian are useful, but
there is no first-class same-date cross-sectional feature-rank transform over
PIT-active names, no per-window reversible normalization, and no robust
adaptive preprocessing ladder.

Recommended first experiment: frozen recipe baseline vs existing
`data.normalisation=rank_gauss` vs new same-date cross-sectional feature
rank/gauss mode. Rank over same-date PIT-active names only.

Repo surfaces: `mci_gru/pipeline.py`, `mci_gru/data/preprocessing.py`,
`DataConfig.normalisation`, `run_metadata.json`, paper-trade metadata parity.

Guardrails: fit nothing on val/test; persist transform type and tie/mask
policy; keep paper-trade promotion blocked until frozen metadata replay works.

Source evidence:

- Gu, Kelly, Xiu, "Empirical Asset Pricing via Machine Learning": Chicago/Yale
  affiliations in PDF; uses cross-sectional characteristic handling. Source:
  https://dachxiu.chicagobooth.edu/download/ML.pdf
- Chen, Pelger, Zhu, "Deep Learning in Asset Pricing": Stanford affiliations
  in PDF; monthly cross-sectional characteristic quantiles. Source:
  https://arxiv.org/pdf/1904.00745
- Kim et al., "RevIN": KAIST AI affiliations in OpenReview/PDF; reversible
  instance normalization for distribution shift. Source:
  https://openreview.net/pdf?id=cGDAkQo1C0p
- Liu et al., "Non-stationary Transformers": Tsinghua affiliations in PDF;
  stationarization/de-stationary attention. Source:
  https://arxiv.org/pdf/2205.14415

### 5. Labels

Practical gaps: labels are raw forward returns or same-day rank percentiles.
Missing first-class variants include volatility-scaled returns, neutral-zone
labels, multi-horizon label panels, and top-focused relevance targets.

Recommended first experiment: volatility-scaled 5-day forward-return labels:
raw 5-day forward return divided by ex-ante trailing realized volatility.
Keep pure IC, `selection_metric=val_ic`, PIT masked-panel, and the frozen recipe
otherwise fixed.

Repo surfaces: `compute_labels`, `TrainingConfig.label_type`, `_build_tensors`,
embargo validation, evaluation summary.

Guardrails: volatility denominator must be trailing and available at prediction
time; embargo must exceed max horizon for multi-horizon labels; overlapping
labels need Newey-West/block bootstrap evidence.

Source evidence:

- Lim, Zohren, Roberts, "Enhancing Time Series Momentum Strategies Using Deep
  Neural Networks": Oxford affiliations in PDF; volatility scaling and
  risk-adjusted objectives. Source: https://arxiv.org/pdf/1904.04912
- Poh, Lim, Zohren, Roberts, "Building Cross-Sectional Systematic Strategies By
  Learning to Rank": Oxford affiliations in PDF; 21-day listwise stock ranking.
  Source: https://arxiv.org/pdf/2012.07149
- Zhang, Zohren, Roberts, "DeepLOB": University of Oxford affiliations in PDF;
  denoised threshold labels and horizon tests. Source:
  https://arxiv.org/pdf/1808.03668

### 6. Tensor/Sample Construction

Practical gaps: MCI-GRU builds `(days, stocks, his_t, features)` tensors, but
missingness is mostly filled/masked away rather than exposed as information.
Long-history tensorization is also not patch-tokenized.

Recommended first experiment: add observation/staleness channels from the
pre-fill panel: `is_observed` and `days_since_observed`, with no label changes.

Repo surfaces: `mci_gru/pipeline.py`, `mci_gru/data/preprocessing.py`,
`mci_gru/data/data_manager.py`, `ModelConfig.his_t`, temporal encoder options.

Guardrails: compute missingness age using data known at the sample date; do not
impute labels; preserve 9-tuple collate and PIT inactive masks.

Source evidence:

- Nie et al., "PatchTST": Yuqi Nie is Princeton-affiliated in PDF; subseries
  patch tokens for long lookbacks. Source: https://arxiv.org/pdf/2211.14730
- Wu et al., "TimesNet": Tsinghua affiliations in PDF; period-aware 2D tensor
  construction. Source: https://arxiv.org/pdf/2210.02186
- Tashiro et al., "CSDI": Stanford affiliations in PDF; explicit values, masks,
  and timestamps for conditional imputation/forecasting. Source:
  https://arxiv.org/pdf/2107.03502
- Morrill et al., "Neural Controlled Differential Equations": Oxford
  affiliations in PDF; irregular/partially observed sequence modeling. Source:
  https://arxiv.org/pdf/2005.08926

### 7. Correlation Graph

Practical gaps: current graph uses trailing Pearson threshold/top-K snapshots.
Potential upgrades include residualized correlations, adaptive edge budgets,
graph diffusion, temporal-correlation pretraining, and relation-aware graph
modeling.

Recommended first experiment: not an edge forecaster yet. Start with graph-zero
and residual-correlation diagnostics to prove graph branch marginal value and
to test whether market/sector-neutral correlations are better than raw Pearson.

Repo surfaces: `mci_gru/graph/builder.py`, `GraphConfig`,
`combined_collate_fn`, `edge_feature_dim`, `tests/test_dynamic_graph_updates.py`.

Guardrails: do not use realized future correlations as live edge weights.
Dynamic snapshots must use data strictly before sample dates. Paper-trade still
loads frozen `graph_data.pt`.

Source evidence:

- Yan and Tan, TCGPN: Peking University affiliation in PDF; graph/temporal masks
  and correlation pretraining. Source: https://arxiv.org/pdf/2407.18519
- Feng et al., "Temporal Relational Ranking for Stock Prediction": NUS and
  Tsinghua affiliations in PDF; sector/supply-chain relation ranking. Source:
  https://arxiv.org/pdf/1809.09441
- You et al., DGDNN: University of Bristol affiliation in PDF; dynamic edges and
  graph diffusion. Source: https://arxiv.org/pdf/2401.01846
- Pei, Zheng, Cartlidge, DGRCL: University of Bristol affiliation in PDF;
  dynamic/static graph contrastive learning. Source:
  https://arxiv.org/pdf/2412.04034

### 8. DataLoader/Collate/Batching

Practical gaps: current collate correctly resolves `GraphSchedule` per sample
and returns the 9-tuple, but dynamic graph batches may pay unnecessary per-sample
lookup/collate cost. There is no snapshot-bucket sampler or memory-budgeted
batcher.

Recommended first experiment: design and instrument snapshot-bucketed batch
sampling for dynamic graphs, default off. Group train indices by graph snapshot
valid-from bucket while preserving current graph timing.

Repo surfaces: `CombinedDataset`, `combined_collate_fn`, `create_data_loaders`,
DataLoader worker/prefetch knobs, dynamic graph tests.

Guardrails: keep the 9-tuple contract; do not shuffle in a way that makes graph
state ambiguous; inactive PIT nodes cannot send messages.

Source evidence:

- You, Du, Leskovec, "ROLAND": Stanford affiliations in PDF; snapshot-based
  dynamic graph training. Source: https://arxiv.org/pdf/2208.07239
- Zhou et al., "TGL": USC affiliation in PDF; Temporal-CSR and parallel temporal
  sampling. Source: https://arxiv.org/pdf/2203.14883
- Huang et al., "Temporal Graph Benchmark": McGill, Stanford, Oxford, Imperial
  affiliations in PDF; chronological temporal loaders/evaluation. Source:
  https://arxiv.org/pdf/2307.01026
- Zeng et al., "GraphSAINT": USC affiliations in PDF; subgraph minibatching.
  Source: https://arxiv.org/pdf/1907.04931

### 9. Model Architecture

Practical gaps: MCI-GRU has temporal, graph, latent-state, and self-attention
streams. Lower-risk missing architecture ideas are market-state feature gating,
patch/channel-independent temporal encoding, and latent-factor residual streams.

Recommended first experiment: MarketStateFeatureGate computed from masked
cross-sectional current graph features, applied before A1/A2 streams. Compare
gate off vs simple sigmoid gate; do not change defaults.

Repo surfaces: `StockPredictionModel`, `MultiScaleTemporalEncoder`,
`CausalTransformerEncoder`, `MarketLatentStateLearner`, `ModelConfig`.

Guardrails: gate inputs must be sample-date/trailing only; inactive PIT nodes
cannot contribute to market-state means; output remains `(batch, stocks)`.

Source evidence:

- Li and Shen, MASTER: Shanghai Jiao Tong University affiliations in PDF;
  market-guided gating and intra/inter-stock attention. Source:
  https://arxiv.org/pdf/2312.15235
- Nie et al., PatchTST: Princeton affiliation in PDF; patch/channel-independent
  temporal encoder. Source: https://arxiv.org/pdf/2211.14730
- Liu et al., iTransformer: Tsinghua affiliations in PDF; inverted variable
  token attention. Source: https://arxiv.org/pdf/2310.06625
- Lim et al., Temporal Fusion Transformer: Oxford affiliation in PDF;
  interpretable variable selection/gating. Source:
  https://arxiv.org/pdf/1912.09363
- Guijarro-Ordonez, Pelger, Zanotti, "Deep Learning Statistical Arbitrage":
  Stanford affiliations in PDF; residual alpha/risk-adjusted modeling. Source:
  https://arxiv.org/pdf/2106.04028

### 10. Loss/Objectives

Practical gaps: MCI-GRU already has MSE, IC, combined, PortfolioIC, and
LambdaRankIC. Missing ideas include full-list differentiable ranking,
SDF/no-arbitrage regularization, uncertainty-adjusted ranking, and constrained
portfolio layers.

Recommended first experiment: defer train-time loss changes until trial-ledger
and evaluation governance exist. Then compare pure IC, LambdaRankIC, and a
listwise soft-rank-IC objective under the same PIT run.

Repo surfaces: `mci_gru/training/losses.py`, `TrainingConfig.loss_type`,
`selection_metric`, `mci_gru/evaluation/statistics.py`.

Guardrails: loss consumes only same-day finite PIT-active predictions/labels.
Portfolio/path objectives need explicit prediction, execution, holding-period,
cost, and return-attribution contracts.

Source evidence:

- Grover et al., "NeuralSort": Stanford affiliations in PDF; differentiable
  sorting. Source: https://arxiv.org/pdf/1903.08850
- Swezey et al., "PiRank": Stanford/UCLA affiliations in PDF; scalable listwise
  differentiable ranking. Source: https://arxiv.org/pdf/2012.06731
- Chen, Pelger, Zhu, "Deep Learning in Asset Pricing": Stanford affiliations;
  no-arbitrage SDF criterion. Source: https://arxiv.org/pdf/1904.00745
- Agrawal et al., "Differentiable Convex Optimization Layers": Stanford/CMU
  affiliations in PDF. Source: https://arxiv.org/pdf/1910.12430
- Zhang, Zhang, Cucuringu, Zohren, "A Universal End-to-End Approach to Portfolio
  Optimization via Deep Learning": Oxford affiliations in PDF. Source:
  https://arxiv.org/pdf/2111.09170

### 11. Training Loop and Optimization

Practical gaps: current training uses AdamW, cosine warmup, AMP, clipping, and
early stopping. Missing contained options include SWA/SWAG-lite, snapshot
prediction ensembles, robust early-stopping smoothing, and validation-only
calibration.

Recommended first experiment: SWA tail averaging per ensemble member compared
against normal best-checkpoint selection under the frozen recipe.

Repo surfaces: `mci_gru/training/trainer.py`, checkpoint save/load,
`TrainingConfig`, `configs/experiment/`, evaluation summaries.

Guardrails: choose SWA/snapshot variants using validation only. Do not
weight-average independently initialized ensemble members without basin checks.
Paper-trade promotion remains frozen-checkpoint/frozen-graph only.

Source evidence:

- Izmailov et al., "Averaging Weights Leads to Wider Optima and Better
  Generalization": Cornell affiliation in PDF. Source:
  https://arxiv.org/pdf/1803.05407
- Maddox et al., "A Simple Baseline for Bayesian Uncertainty in Deep Learning":
  NYU affiliations in PDF. Source: https://arxiv.org/pdf/1902.02476
- Huang et al., "Snapshot Ensembles": Cornell and Tsinghua affiliations in PDF.
  Source: https://arxiv.org/pdf/1704.00109
- Romano, Patterson, Candes, "Conformalized Quantile Regression": Stanford
  affiliations in PDF. Source: https://arxiv.org/pdf/1706.04599

### 12. Ensemble

Practical gaps: `train_multiple_models` trains independent seeds and averages
predictions. It does not expose dispersion, uncertainty-adjusted ranks,
validation-weighted member averaging, or diversity diagnostics.

Recommended first experiment: saved-prediction replay. From existing
`predictions_model_*`, compute mean, standard deviation, and percentile spread
per date/stock; tune a dispersion penalty on validation only; replay top-k and
rank-drop outcomes without retraining.

Repo surfaces: `mci_gru/training/trainer.py`, `averaged_predictions/`,
`predictions_model_*`, `mci_gru/evaluation/`, backtest scripts.

Guardrails: calibration lambda and member weights are train/validation only;
preserve PIT masks; keep single-member vs ensemble comparisons explicit.

Source evidence:

- Ma et al., "Uncertainty of Machine Learning Predictions in Asset Pricing":
  National University of Singapore affiliation in PDF; forecast confidence and
  uncertainty-aware selection. Source: https://arxiv.org/pdf/2503.00549
- Uncertainty-adjusted sorting paper: Tsinghua, HKU, and HKUST affiliations on
  arXiv/source pages; model-agnostic uncertainty-adjusted asset sorting. Source:
  https://arxiv.org/abs/2601.00593
- Filipovic and Pasricha, "Empirical Asset Pricing via Ensemble Gaussian
  Process Regression": EPFL/Swiss Finance Institute evidence on source pages;
  temporal experts, equal/MSE weighting, and uncertainty-aware construction.
  Source: https://arxiv.org/abs/2212.01048
- Huang et al., "Snapshot Ensembles": Cornell/Tsinghua affiliations in PDF.
  Source: https://arxiv.org/pdf/1704.00109

### 13. Walk-Forward/Time-Split Validation

Practical gaps: MCI-GRU has basic rolling/expanding walk-forward and calendar
label embargo, but not split-interval audits, PBO-style model-selection audits,
nested selection intervals, or a complete trial ledger.

Recommended first experiment: read-only `split_audit_v0` over existing configs
and run artifacts. Compute feature interval, graph valid-from, prediction date,
label interval, and split id. Verify no train/val/test label-interval overlap
after trading-session purging.

Repo surfaces: `mci_gru/walkforward.py`, `ExperimentConfig._validate_embargo`,
`mci_gru/pipeline.py`, `mci_gru/data/preprocessing.py`,
`mci_gru/evaluation/statistics.py`.

Guardrails: purge in trading sessions, not just calendar days; final test
windows cannot steer defaults; ledger includes failed and abandoned variants.

Source evidence:

- Gort, Liu et al., "Deep Reinforcement Learning for Cryptocurrency Trading":
  Columbia University evidence in PDF; combinatorial validation and PBO
  framing. Source: https://arxiv.org/pdf/2209.05559
- Harvey, Liu, Zhu, "... and the Cross-Section of Expected Returns": Campbell
  Harvey is Duke/NBER on Oxford/journal pages; multiple-testing threshold
  control. Source: https://doi.org/10.1093/rfs/hhv059
- Bates, Hastie, Tibshirani, "Cross-validation: what does it estimate and how
  well does it do it?": Berkeley and Stanford affiliations in PDF; nested CV
  interval coverage. Source: https://arxiv.org/pdf/2104.00673
- Buerkner, Gabry, Vehtari, "Approximate leave-future-out cross-validation":
  Columbia evidence on source pages; future-conditioned validation for time
  series. Source: https://arxiv.org/abs/1902.06281

### 14. Evaluation/Statistical Testing

Practical gaps: MCI-GRU has IC, Newey-West Sharpe, and moving-block bootstrap
CIs. Missing pieces are multiple-testing haircuts, model confidence sets,
deflated Sharpe/PBO reporting, IC p-values, block-size sensitivity, effective
breadth diagnostics, and trial-aware promotion thresholds.

Recommended first experiment: add a report-only trial ledger and saved-prediction
selection audit over existing PIT run folders. Collect `evaluation_summary.json`,
`training_summary.json`, config metadata, and `averaged_predictions/`; compute
daily rank IC, daily Pearson IC, top-20/top-50 returns, Newey-West t/p-values,
moving-block bootstrap CIs, BHY-adjusted p-values across the real trial count,
and deflated Sharpe for Sharpe-style claims.

Repo surfaces: `mci_gru/evaluation/statistics.py`,
`mci_gru/evaluation/portfolio.py`, `evaluation_summary.json`,
`training_summary.json`, MLflow child runs.

Guardrails: do not compute promotion metrics from test while choosing variants.
Overlapping labels need HAC or block bootstrap.

Source evidence:

- Harvey, Liu, Zhu, "... and the Cross-Section of Expected Returns": Duke/NBER
  evidence; multiple testing in expected-return research. Source:
  https://doi.org/10.1093/rfs/hhv059
- Bailey and Lopez de Prado, "The Deflated Sharpe Ratio": Cornell affiliation
  evidence for Lopez de Prado on SSRN/source pages; Sharpe-style overfitting
  disclosure. Source:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Lo, "The Statistics of Sharpe Ratios": MIT Sloan affiliation on author/source
  pages; Sharpe inference under non-IID returns. Source:
  https://doi.org/10.2469/faj.v58.n4.2453
- Romano, Shaikh, Wolf, "Multiple Testing in Economics": Stanford and Zurich
  affiliations in publication/source pages; stepwise multiple-testing control.
  Source: https://doi.org/10.1146/annurev-economics-080217-053456
- Hansen, Lunde, Nason, "The Model Confidence Set": Stanford/academic source
  evidence for model comparison under dependent loss series. Source:
  https://doi.org/10.3982/ECTA5771

### 15. Backtesting/Execution Simulation

Practical gaps: current backtests have explicit prediction-at-close, next-open
entry, open-to-open timing, spread/slippage, rebalancing, and rank-drop gates.
The missing first-class layer is capacity: portfolio AUM, per-name dollar trade,
ADV participation, volatility-scaled nonlinear impact, unfillable/clipped
notional, and persistent/transient impact state.

Recommended first experiment: saved-prediction capacity replay over existing
PIT predictions. Add an AUM/top-k/rank-drop/cost grid; compute target dollar
trades from equal weights, lagged rolling dollar ADV and volatility known by the
prediction date, participation rate, capacity breaches, gross/net return,
turnover, total cost, and clipped/unfillable diagnostics.

Repo surfaces: `docs/BACKTEST_FAIRNESS_AUDIT.md`,
`mci_gru/evaluation/portfolio.py`, backtest scripts under `tests/`,
paper-trade `portfolio.py`.

Guardrails: score at T close, enter T+1 open, and hold open-to-open unless a
different timing contract is named. Pre-trade ADV/vol/capacity gates must use
data available by T close; realized T+1 volume is ex-post diagnostic only.
Predeclare AUM/cost grids and do not tune capacity assumptions on test winners.

Source evidence:

- Boyd, Busseti, Diamond et al., "Multi-Period Trading via Convex
  Optimization": Stanford affiliations in PDF; bid-ask, volatility, volume, and
  nonlinear transaction-cost modeling. Source: https://arxiv.org/pdf/1705.00109
- Guijarro-Ordonez, Pelger, Zanotti, "Deep Learning Statistical Arbitrage":
  Stanford affiliations in PDF; constrained/risk-adjusted portfolio pipeline.
  Source: https://arxiv.org/pdf/2106.04028
- Epstein, Wang, Choi, Pelger, "Attention Factors for Statistical Arbitrage":
  Stanford affiliations in PDF; net-utility and adaptive trading-policy ideas.
  Source: https://arxiv.org/pdf/2510.11616
- Chan, Sircar, Zimbidis, "Optimal Trading under Instantaneous and Persistent
  Price Impact": Princeton affiliations in PDF; persistent/transient impact
  state. Source: https://arxiv.org/pdf/2507.17162
- Zhang, Zhang, Cucuringu, Zohren, Oxford portfolio optimization paper:
  Oxford affiliations in PDF; end-to-end constrained portfolio construction.
  Source: https://arxiv.org/pdf/2111.09170
- Micheli and Monod, "Deep Reinforcement Learning for Online Optimal Execution
  Strategies": Imperial College London affiliations in PDF; child-order and
  implementation-shortfall simulator axis. Source:
  https://arxiv.org/pdf/2410.13493

### 16. Paper Trading/Live Inference Controls

Practical gaps: paper-trade inference correctly loads frozen checkpoints,
metadata, and `graph_data.pt`. Missing controls are sequential harmful-shift
kill switches, conformal pre-trade abstention, martingale score/feature drift
monitoring, shadow-mode/canary authority, frozen-artifact checks, fallback
policy, and operator-readable halt reasons.

Recommended first experiment: offline shadow-mode pre-trade guard replay. For
historical paper-trade days, replay `scores.csv -> target_portfolio.csv ->
realized open-to-open outcome`, then compute a delayed-label sequential risk
monitor against a backtest/source reference. First output only
`safety_status.json` with `OK`, `SHADOW_ONLY`, or `HALT`.

Repo surfaces: `paper_trade/scripts/infer.py`, `portfolio.py`, `monitor.py`,
`run_nightly.py`, `feature_drift.json`, `run_metadata.json`.

Guardrails: do not use same-day or future realized returns to gate today's
trades; delayed labels update monitors only after outcomes are knowable. Keep
paper-trade inference frozen: no live `GraphBuilder`, no refit normalization,
no surprise dynamic graph. Missing references, stale data, feature mismatch, or
artifact mismatch should fail closed to `SHADOW_ONLY` or `HALT`.

Source evidence:

- Podkopaev and Ramdas, "Tracking the risk of a deployed model": Carnegie
  Mellon affiliations in source/PDF; delayed-label continuous risk alarms.
  Source: https://ar5iv.labs.arxiv.org/html/2110.06177
- Angelopoulos et al., "Conformal Risk Control": UC Berkeley, MIT, and Stanford
  affiliations in PDF; calibrated action thresholds with risk control. Source:
  https://arxiv.org/pdf/2208.02814
- Gibbs and Candes, "Adaptive Conformal Inference Under Distribution Shift":
  Stanford affiliations in PDF; distribution-shift-aware conformal calibration.
  Source: https://arxiv.org/pdf/2106.00170
- Prinster et al., "WATCH: Adaptive Monitoring for AI Deployments": Johns
  Hopkins affiliation in PDF; online martingale monitoring and root-cause
  diagnosis. Source: https://arxiv.org/pdf/2505.04608
- Cohen, Snow, Szpruch, "Black-box model risk in finance": Samuel Cohen at
  University of Oxford in PDF; finance-specific model-risk controls. Source:
  https://arxiv.org/pdf/2102.04757

### 17. Outputs/Tracking/Reproducibility

Practical gaps: MCI-GRU has filesystem source-of-truth artifacts and additive
MLflow. Missing pieces are a first-class run-bundle manifest, trial ledger,
dataset/model cards, and reproducibility checklists for financial experiments.

Recommended first experiment: `run_manifest.json` v0 beside each run, written
after existing artifacts are produced. It should include config hash, command,
git commit/dirty state, data fingerprints, PIT file hash, feature lag policies,
normalization reference hash, graph artifact/policy, checkpoint hashes,
prediction folder hash, evaluation summary path, MLflow run id, seed policy,
environment summary, selection rule, sibling trial ids, and paper-trade
eligibility flag.

Repo surfaces: `run_experiment.py`, `mci_gru/tracking/`, `docs/OUTPUT_MANAGEMENT.md`,
`docs/MLFLOW_TRACKING.md`, `run_metadata.json`, `training_summary.json`,
`evaluation_summary.json`, `timing_summary.json`.

Guardrails: filesystem artifacts remain source of truth; MLflow is additive.
Do not reference ignored `results/` as source-of-truth research evidence unless
the exact artifact is preserved and hashed.

Source evidence:

- Gebru et al., "Datasheets for Datasets": Stanford and Cornell affiliations
  visible on arXiv/PDF; dataset documentation and intended-use framing. Source:
  https://arxiv.org/pdf/1803.09010
- Mitchell et al., "Model Cards for Model Reporting": University of Toronto
  and Stanford evidence among authors/source pages; model reporting template.
  Source: https://arxiv.org/pdf/1810.03993
- Pineau et al., "Improving Reproducibility in Machine Learning Research":
  McGill/Mila affiliations in JMLR/source pages; reproducibility checklist.
  Source: https://jmlr.org/papers/v22/20-303.html
- Hao et al., "MGit": Columbia and Stanford affiliations in PDF; model lineage
  and version-graph ideas. Source: https://arxiv.org/pdf/2307.07507
- Bhardwaj et al., "DataHub": MIT/CSAIL affiliations in PDF; dataset versioning
  and provenance concepts. Source: https://arxiv.org/pdf/1409.0798
- Kapoor and Narayanan, "Leakage and the Reproducibility Crisis in ML-based
  Science": Princeton affiliation evidence in source pages; leakage taxonomy.
  Source: https://arxiv.org/pdf/2207.07048

## Practical vs Speculative

Practical first-pass items:

- Split audit and trial ledger.
- Run-bundle manifest.
- Saved-prediction ensemble dispersion replay.
- PIT availability/tradability/staleness audit.
- Execution-timing/cost/capacity replay.
- Cross-sectional feature rank/gauss transform.
- Observation/staleness channels.
- Snapshot-bucket batching instrumentation.

Practical but after first governance wave:

- SWA/SWAG-lite checkpoint construction.
- Market-state feature gate.
- Volatility-scaled labels.
- OHLCV chart-path feature descriptors.
- Listwise soft-rank objective comparison.

Speculative or design-first:

- Forward-correlation edge forecaster.
- True multi-relation graph attention.
- Differentiable portfolio optimizer layer.
- Distributional/heteroscedastic MCI-GRU head.
- Dynamic paper-trade graph updates.
- Text/LLM-derived trading features.

## Implementation Wave Plan

Wave 0: docs-only and saved-artifact governance

1. Add a run-bundle manifest design note and trial-ledger schema.
2. Add `split_audit_v0` design over existing artifacts.
3. Add PIT availability/tradability/staleness report spec.
4. Define saved-prediction replay contracts for ensemble dispersion and
   execution timing.

Wave 1: no-retraining replay and diagnostics

1. Replay existing per-model predictions with dispersion-adjusted ranks.
2. Replay execution-cost/capacity/timing assumptions on saved predictions.
3. Add graph-zero and residual-correlation diagnostics.
4. Add dynamic-batch instrumentation: active nodes, edges, edge bytes, collate
   milliseconds, snapshot age.

Wave 2: additive data/feature experiments

1. Cross-sectional feature rank/gauss normalization preset.
2. Observation/staleness channels.
3. OHLCV chart-path descriptors.
4. Tradability overlay as separate mask and report, not PIT replacement.

Wave 3: contained training/model experiments

1. SWA/SWAG-lite per ensemble member.
2. MarketStateFeatureGate.
3. Volatility-scaled labels.
4. Listwise soft-rank objective, compared with pure IC and LambdaRankIC.

Wave 4: higher-risk research

1. Forward-correlation edge feature pilot after graph diagnostics.
2. Multi-relation graph attention only with PIT relation intervals.
3. Differentiable portfolio objectives only after execution replay is trusted.
4. Paper-trade uncertainty/drift gates only after offline saved-prediction
   validation.

## Exclusions

Rejected because affiliation or top-university gate failed, despite useful ideas:

- Ritzmann Junior and Nievola, GA/SVM automatic feature engineering: accessible
  arXiv source did not verify top-university affiliation.
- Huang, Capretz, and Ho, fundamental-analysis ML: useful feature-selection
  angle, but no verified top-university affiliation in accessible primary
  source.
- Ranse survivorship-bias paper: useful practical survivorship work, but
  Cluster University of Jammu did not pass this conservative gate.
- RankGLU 2026: relevant rank/head-normalization idea, but first-author gate was
  not clearly in the top-university set from primary evidence.
- Original DAIN: useful normalization paper, but primary affiliations found in
  the scan did not satisfy the conservative gate.
- SAITS: useful imputation method, but accessible PDF affiliations were
  Concordia/Ciena.
- Temporal Graph Networks: useful temporal memory/event framework, but PDF
  author block was Twitter-only.
- MDGNN: relevant multi-relational dynamic graph, but primary affiliation was
  Ant Group/Alibaba.
- FinGAT and TRA: potentially relevant, but this scan did not verify qualifying
  primary-source affiliation.
- HIST: useful hidden-concept graph idea, but primary affiliation evidence found
  in the scan did not pass the gate.
- Lakshminarayanan et al., "Deep Ensembles": strong uncertainty baseline, but
  PDF affiliation was DeepMind-only.
- SAM and NFNets/AGC: practically interesting optimization methods, but source
  affiliations found were Google/DeepMind-only.
- Generic meta-labeling sources: relevant, but a primary top-university
  affiliation source was not verified in this pass.
- Qlib-style processor conventions: useful implementation context but industry
  platform documentation, not gated research evidence.
- AQR/Frazzini-Israel-Moskowitz trading-cost material: tempting for capacity,
  but primary affiliation evidence was not verified in this scan.
- Google "ML Test Score" style production-readiness work and NIST AI RMF:
  useful engineering context, but not included as top-university academic
  evidence.

## Caveats

- This scan emphasizes source quality over breadth. Some good ideas are omitted
  because the affiliation evidence was not visible enough.
- Several recent arXiv graph/finance papers pass the university gate but still
  need replication skepticism; they are not treated as proven alpha.
- Vendor/fundamental/analyst/relationship data ideas are blocked until
  point-in-time provenance and lag policy are explicit.
- Paper-trade ideas remain late-stage: `paper_trade/` must keep frozen
  checkpoint, `run_metadata.json`, and `graph_data.pt` behavior.
