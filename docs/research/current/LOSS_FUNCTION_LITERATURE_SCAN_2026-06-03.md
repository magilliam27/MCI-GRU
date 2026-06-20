---
title: "Loss Function Literature Scan for Stock Prediction"
evaluated_on: "2026-06-03"
scope: "Recent work from 2021-06-03 through 2026-06-03, with emphasis on arXiv/SSRN-style sources"
status: "current research evidence"
primary_landing_zone: "Training/evaluation"
---

# Loss Function Literature Scan for Stock Prediction

## Scope

This scan focuses on recent research into training objectives for stock return
prediction, stock ranking, and trading/portfolio decision models. The current
MCI-GRU baseline already covers MSE, IC, combined MSE/IC, and a hybrid
`portfolio_ic` loss with soft top-k utility, so the notes below emphasize
objectives that are meaningfully different from those.

The most useful recent shift is away from generic prediction error and toward
losses aligned with one of four downstream goals:

- full cross-sectional rank quality,
- top-k or long-short selection quality,
- utility/risk-adjusted portfolio performance,
- calibrated predictive distributions or uncertainty.

## Current MCI-GRU Baseline

Current training losses live in `mci_gru/training/losses.py`:

- `MaskedMSELoss`: finite-label MSE.
- `ICLoss`: negative cross-sectional Pearson IC per date.
- `CombinedMSEICLoss`: weighted MSE plus negative IC.
- `SoftTopKForwardReturnLoss`: differentiable soft top-k standardized forward-return utility.
- `PortfolioICLoss`: IC plus soft top-k utility blend.

Current config support is limited to `mse`, `ic`, `combined`, and `portfolio_ic`
via `TrainingConfig._VALID_LOSS_TYPES`.

## Loss Families Worth Knowing

### 1. Direct Rank IC / Learning-to-Rank Losses

Recent work is explicitly attacking the train/evaluation mismatch between MSE or
Pearson IC and rank-based stock selection.

- `LambdaRankIC: Directly Optimizing Rank IC for Financial Prediction`
  (arXiv:2605.00501, 2026-05-01) proposes a LambdaRank-style objective that
  directly targets Rank IC by using pairwise rank-swap gradients. It reports
  stronger Rank IC, ICIR, returns, and Sharpe than regression or NDCG-style
  ranking baselines in the paper's empirical setting.
- `On Evaluating Loss Functions for Stock Ranking` (arXiv:2510.14156,
  2025-10-15) benchmarks pointwise, pairwise, and listwise objectives for daily
  S&P 500 return ranking with Transformer models.

MCI-GRU interpretation: this is the cleanest next objective family because it
can stay per-date and PIT-mask-safe. It does not require chronological portfolio
state. A PyTorch pairwise RankIC surrogate or differentiable Spearman surrogate
is a natural experimental sibling to the existing `ICLoss`.

Main risk: pairwise/listwise losses can become O(n^2) over stocks. First pass
should sample pairs or restrict to a manageable cross-section.

### 2. Listwise Top/Bottom and Soft Ranking Objectives

Listwise ranking losses train the model to order the whole daily stock list, and
some variants emphasize the top and bottom tails used by long-short strategies.
This is different from IC because the objective can weight rank positions and
tail mistakes explicitly.

MCI-GRU interpretation: useful if the goal is better top-k names or better
top-minus-bottom spread rather than full-list Pearson correlation. It is also
safer than a full Sharpe or transaction-cost loss because each date can still be
trained independently.

Main risk: listwise losses imported from search/recommendation often optimize
NDCG-like metrics, which may not match finance's Rank IC, top-k return, or net
portfolio utility. LambdaRankIC is more finance-native.

### 3. Direct Profit / Trading-Action Losses

Some recent papers train the network to emit buy, short, hold, or allocation
actions and use losses based directly on realized profit.

- `Directly Learning Stock Trading Strategies Through Profit Guided Loss
  Functions` (arXiv:2507.19639, 2025-07-25) proposes four profit-guided losses
  for buy/short decisions over a stock portfolio, allowing ordinary time-series
  models to learn trading decisions directly.
- `A Novel Loss Function for Deep Learning Based Daily Stock Trading System`
  (arXiv:2502.17493, 2025-02-20; revised 2025-11-07) proposes a return-weighted
  loss meant to emphasize top growth opportunities from daily OHLCV, sector, and
  technical-indicator inputs.

MCI-GRU interpretation: conceptually close to the current `PortfolioICLoss`, but
more aggressive. These losses should be treated as experimental because they can
overfit execution assumptions and may reward unstable score magnitudes.

Main risk: direct profit losses are easy to make subtly unfair unless the label
horizon, tradable entry/exit convention, PIT mask, and costs are pinned down.
They should be evaluated through saved-prediction replay before real training
runs.

### 4. Sharpe, Sortino, Utility, and Decision-Focused Losses

This family trains through a risk/utility objective instead of predicting returns
first and optimizing later.

- `Deep Learning for Financial Time Series: A Large-Scale Benchmark of
  Risk-Adjusted Performance` (arXiv:2603.01820, 2026-03-02) focuses on
  Sharpe-ratio optimization and evaluates downside/tail risk, transaction-cost
  breakevens, seed robustness, and compute efficiency.
- `Return Prediction for Mean-Variance Portfolio Selection: How
  Decision-Focused Learning Shapes Forecasting Models` (arXiv:2409.09684,
  2024-09-15; revised 2025-11-09) shows why decision-focused learning can beat
  MSE for mean-variance optimization even when prediction error gets worse.
- `Decision by Supervised Learning with Deep Ensembles` (arXiv:2503.13544,
  2025-03-16; revised 2025-10-21) reframes portfolio optimization as supervised
  learning over optimal portfolio weights, using cross-entropy against weights
  derived from Sharpe or Sortino optimization.
- Deep parametric portfolio-policy work on SSRN uses investor utility directly
  as an economic regularizer for portfolio weights.

MCI-GRU interpretation: high upside, but not first-pass safe unless separated
into two types:

- per-date soft utility, which can fit the existing trainer,
- path-dependent Sharpe/Sortino/turnover/drawdown, which needs chronological
  batching and split-boundary state resets.

Main risk: annual Sharpe, drawdown, and turnover are path-dependent. Adding them
inside the current shuffled training loop would be conceptually wrong.

### 5. Probabilistic / Distributional Losses

Instead of predicting a point return, these losses predict quantiles or a full
distribution. They are useful when portfolio decisions should depend on
uncertainty, downside risk, or tail asymmetry.

- `Stock Index Prediction using Cointegration test and Quantile Loss`
  (arXiv:2109.15045, 2021-09-29) applies quantile loss to stock-index prediction
  and evaluates cumulative return and Sharpe.
- `Learning Probability Distributions in Macroeconomics and Finance`
  (arXiv:2204.06848, 2022-04-14) argues for probabilistic forecasting in finance,
  including stock return distributions with heavy tails, asymmetry, and low
  signal-to-noise.
- `Forecasting Probability Distributions of Financial Returns with Deep Neural
  Networks` (arXiv:2508.18921, 2025-08-26) trains CNN/LSTM models with custom
  negative log-likelihood losses for Normal, Student-t, and skewed Student-t
  return distributions, then evaluates with LPS, CRPS, PIT, and VaR metrics.
- `ProbFM` (arXiv:2601.10591, 2026-01-15) compares Deep Evidential Regression,
  Gaussian NLL, Student-t NLL, quantile loss, and conformal prediction for
  financial time-series uncertainty.

MCI-GRU interpretation: likely useful as an auxiliary head or later model
variant, not as a direct replacement for IC. A Student-t NLL, skew-t NLL, or
multi-quantile pinball loss could help identify high-uncertainty names or
calibrate top-k confidence.

Main risk: better calibrated return distributions do not automatically improve
rank quality. Promotion should require both calibration evidence and portfolio
replay evidence.

### 6. Direction / Trend-Aware Hybrid Losses

Some stock-index forecasting papers add directional or trend terms to MSE-like
losses.

- `Galformer` (Scientific Reports, 2024) combines quantitative forecast error
  with trend accuracy for multi-step stock market index prediction.
- Knowledge-graph stock prediction work from 2022 uses a piecewise loss around
  mutation points, arguing that standard MSE under-penalizes errors around
  abrupt stock-price moves.

MCI-GRU interpretation: direction-aware losses are less attractive for
cross-sectional ranking than RankIC/listwise losses, but a sign-consistency or
tail-event term may be useful if diagnostics show the model gets direction wrong
near large positive/negative labels.

Main risk: direction objectives can throw away magnitude and ordering
information. They should not replace rank-aware objectives for top-k selection.

### 7. Contrastive and Self-Supervised Auxiliary Losses

Contrastive losses are not usually final trading losses. They pretrain asset,
regime, or temporal embeddings so the supervised model starts from a better
representation.

- `Contrastive Learning of Asset Embeddings from Financial Time Series`
  (arXiv:2407.18645, 2024-07-26) explores contrastive loss functions for asset
  embeddings and evaluates downstream sector classification, risk management,
  and portfolio optimization tasks.
- Newer 2026 stock/portfolio papers combine KL alignment, NT-Xent contrastive
  losses, and downstream portfolio utility objectives.

MCI-GRU interpretation: more of a model/representation project than a loss
replacement. Interesting if graph or sector embeddings become stale or noisy,
but not the fastest way to improve the current training objective.

Main risk: pretraining can look impressive while adding little to PIT daily
portfolio replay. It needs a tight ablation.

## Recommended Experiment Order for MCI-GRU

1. Prototype a `rank_ic` or `lambdarank_ic` loss.
   Keep it per-date, PIT-mask-aware, and disabled by default. Compare against
   current `ic` and `portfolio_ic` on a tiny repeated-seed grid.

2. Add a generic `listwise_rank` or `pairwise_rank` loss.
   This gives a broader benchmark against the finance-specific RankIC objective.
   It should include pair sampling to control runtime.

3. Add a probabilistic auxiliary experiment, not a replacement loss.
   Candidate heads: multi-quantile pinball loss or Student-t NLL. Evaluate
   calibration, top-k confidence, and whether uncertainty filtering improves net
   replay.

4. Defer path-dependent Sharpe/Sortino/turnover/drawdown losses.
   They need a chronological trainer mode or a saved-prediction decision layer.
   The current shuffled trainer is not the right control plane for those losses.

5. Treat direct profit/action losses as high-risk research.
   They are worth reading, but MCI-GRU already has the safer soft top-k utility
   variant. Any more aggressive action loss should first reproduce a saved
   prediction replay advantage without changing the model.

## Source Table

| Date | Source | Loss idea |
| --- | --- | --- |
| 2026-05-01 | [LambdaRankIC](https://arxiv.org/abs/2605.00501) | Direct Rank IC / LambdaRank pairwise gradients |
| 2026-03-02 | [Deep Learning for Financial Time Series](https://arxiv.org/abs/2603.01820) | Sharpe/risk-adjusted optimization benchmark |
| 2026-01-15 | [ProbFM](https://arxiv.org/abs/2601.10591) | DER, Gaussian NLL, Student-t NLL, quantile, conformal |
| 2025-10-15 | [On Evaluating Loss Functions for Stock Ranking](https://arxiv.org/abs/2510.14156) | Pointwise, pairwise, listwise stock-ranking losses |
| 2025-08-26 | [Forecasting Probability Distributions of Financial Returns](https://arxiv.org/abs/2508.18921) | Distributional NLL, CRPS/PIT/VaR evaluation |
| 2025-07-25 | [Profit Guided Loss Functions](https://arxiv.org/abs/2507.19639) | Direct buy/short profit-guided losses |
| 2025-03-16 | [Decision by Supervised Learning with Deep Ensembles](https://arxiv.org/abs/2503.13544) | Cross-entropy to Sharpe/Sortino-optimal weights |
| 2025-02-20 | [A Novel Loss Function for Daily Stock Trading](https://arxiv.org/abs/2502.17493) | Return-weighted top-growth loss |
| 2024-09-15 | [Return Prediction for Mean-Variance Portfolio Selection](https://arxiv.org/abs/2409.09684) | Decision-focused MVO loss |
| 2024-07-26 | [Contrastive Learning of Asset Embeddings](https://arxiv.org/abs/2407.18645) | Contrastive representation losses |
| 2024 | [Galformer](https://www.nature.com/articles/s41598-024-72045-3) | Hybrid error plus trend-accuracy loss |
| 2022-04-14 | [Learning Probability Distributions in Macroeconomics and Finance](https://arxiv.org/abs/2204.06848) | Probabilistic return distributions |
| 2021-09-29 | [Stock Index Prediction using Cointegration test and Quantile Loss](https://arxiv.org/abs/2109.15045) | Quantile/pinball loss |

## Bottom Line

The best next idea for MCI-GRU is not a generic "better portfolio loss." It is a
rank-aligned loss, especially direct Rank IC or a LambdaRankIC-inspired surrogate,
because it matches the model's cross-sectional use case while preserving the
repo's no-lookahead and PIT masking constraints. Probabilistic losses are the
second most interesting family, but probably as auxiliary confidence/calibration
machinery rather than the primary ranking objective. Full Sharpe, Sortino,
drawdown, and turnover losses should wait for a chronological training design.
