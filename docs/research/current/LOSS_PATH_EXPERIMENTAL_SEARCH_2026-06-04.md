---
title: "Experimental Loss Path Search for MCI-GRU"
evaluated_on: "2026-06-04"
status: "current research evidence"
decision: "If the current-data constraint is removed, prioritize uncertainty-adjusted ranking before full decision-focused portfolio optimization"
primary_landing_zone: "Training/evaluation research"
---

# Experimental Loss Path Search for MCI-GRU

## Decision

This note removes the first criterion from
`docs/research/current/LOSS_PATH_DECISION_2026-06-04.md`:

> The loss must use information MCI-GRU already has.

With that constraint removed, the best experimental direction is no longer just
"make the current return score rank better." The more interesting direction is:

1. Teach MCI-GRU to forecast return uncertainty as well as expected return.
2. Train that extra output with stable probabilistic losses.
3. Rank stocks by a confidence-adjusted score, such as a lower confidence bound
   or median-to-spread score.

The recommended experimental branch is therefore:

**Uncertainty-adjusted ranking with a distributional alpha head.**

This sits between LambdaRankIC and full end-to-end portfolio optimization. It is
more adventurous than LambdaRankIC because it changes the model output contract,
but it is less brittle than direct Sharpe, drawdown, or optimizer-layer losses
because the training signal can still be a proper scoring loss with stable,
local gradients.

## Revised Criteria

The new criteria are:

1. The loss should produce stable gradients.
2. The loss metric should relate to backtesting performance.

Allowed now:

- New model heads, such as `mu` plus `sigma`, Student-t parameters, or quantiles.
- New data, such as option-implied volatility, analyst dispersion, news, or
  macro uncertainty.
- New training modes, such as chronological mini-episodes or decision-focused
  optimizer layers.

Still required:

- PIT and no-lookahead discipline.
- Fair comparison against `ic`, `portfolio_ic`, and LambdaRankIC.
- Backtest evidence before changing any production-style default.

## External Evidence

Recent evidence points to five experimental families.

### 1. Uncertainty-Adjusted Ranking

Uncertainty-adjusted sorting argues that stock ranking should use prediction
bounds rather than point predictions alone. The January 2026 arXiv paper reports
portfolio-performance gains from sorting on uncertainty-adjusted prediction
bounds, with improvements driven mainly by lower volatility.

Source: <https://arxiv.org/abs/2601.00593>.

This maps cleanly onto MCI-GRU because the trading layer is already rank-driven.
The missing piece is a calibrated uncertainty output. Training can use:

- Gaussian or Student-t negative log likelihood.
- Quantile pinball loss.
- CRPS-style distributional scoring.
- A hybrid distributional plus Rank IC objective.

The rank score could be:

```text
score = mu - lambda_uncertainty * sigma
```

or, for quantiles:

```text
score = q50 / (q90 - q10 + eps)
```

### 2. Distributional Return Forecasting

A 2025 arXiv paper on forecasting financial-return distributions trains CNN and
LSTM models to output Normal, Student-t, and skewed Student-t distribution
parameters using custom negative log-likelihood losses. It evaluates with log
predictive score, CRPS, PIT calibration, and VaR behavior.

Source: <https://arxiv.org/abs/2508.18921>.

A 2026 SSRN paper on quantile regression for stock trading finds that signals
using predicted median relative to predicted dispersion can outperform pure
location signals under realistic costs in a German large- and mid-cap equity
setting.

Source: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6542060>.

The key implication is that the loss does not need to be a direct portfolio
ratio to become portfolio-useful. A stable distributional loss can produce a
better ranking signal when the portfolio rule uses both location and dispersion.

### 3. Decision-Focused Portfolio Optimization

Decision-focused learning trains the predictor according to downstream
portfolio decision quality rather than pointwise forecast error. The 2024 arXiv
paper, revised in 2025, shows that decision-focused gradients for mean-variance
portfolio selection can be interpreted as tilting prediction errors by the
inverse covariance matrix.

Source: <https://arxiv.org/abs/2409.09684>.

The 2026 NBER/RePEc "Machine Learning Meets Markowitz" abstract makes the same
bigger point: the forecast and optimizer should be integrated because investors
care about the forecast precision of assets that matter to the final portfolio,
not average error across all securities.

Source: <https://ideas.repec.org/p/nbr/nberwo/34861.html>.

This family is backtest-aligned but engineering-heavy. It needs a covariance
model, a differentiable optimizer or implicit gradient, position constraints,
and a careful treatment of transaction costs and turnover.

### 4. SPO / Predict-Then-Optimize Surrogates

SPO-style losses are attractive because they directly target regret or decision
quality, but a May 2026 arXiv paper is an important warning sign. It argues that
SPO-based portfolio optimization can induce prediction inflation and excessive
turnover, and it evaluates clipping, rescaling, and partial portfolio adjustment
as stabilizers.

Source: <https://arxiv.org/abs/2605.01176>.

This is exactly why direct optimizer losses should not be the first experimental
branch in MCI-GRU. The method can align with backtests, but the model can learn
distorted scores unless the portfolio layer is realistic and constrained.

### 5. Multi-Objective Portfolio Loss Discovery

AlphaLoss is a 2025 SSRN / 2026 OpenReview workshop line of work that uses an
LLM-driven evolutionary loop to discover portfolio objectives. The reported
components include downside risk, maximum drawdown, recovery duration, GARCH-like
conditional volatility, diversification entropy, L1/L2/max-weight constraints,
VaR/CVaR, higher moments, and temporal stability penalties.

Sources:

- <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5263279>
- <https://openreview.net/forum?id=y0YEGgZnEC>

This is useful as an idea generator, not as a first implementation. It is too
meta-objective-heavy for the current training loop, and many terms are
path-dependent.

## Scorecard

Scores use 1 to 5, where 5 is strongest. "Infrastructure risk" is also 1 to 5,
where 5 means hardest to implement and validate safely.

| Candidate | Gradient stability | Backtest relation | Infrastructure risk | New requirements | Recommendation |
| --- | ---: | ---: | ---: | --- | --- |
| Uncertainty-adjusted ranking with distributional head | 4.5 | 4.5 | 3 | `mu/sigma` or quantile outputs, calibration diagnostics, confidence-adjusted portfolio ranking | Best experimental branch |
| LambdaRankIC-style pairwise Rank IC | 4 | 5 | 2 | Current output contract, Rank IC diagnostics | Still best conservative next loss |
| Quantile location-to-spread score | 4.5 | 4 | 3 | Quantile head, pinball loss, interval calibration | Good first version of uncertainty branch |
| Student-t NLL return distribution | 4 | 4 | 3 | Distributional head, scale constraints, PIT/CRPS diagnostics | Good if heavy tails matter |
| Risk-adjusted forecast-error loss | 4 | 4 | 4 | Covariance or precision estimate, investor objective definition | Promising but needs design |
| Differentiable MVO / decision-focused optimizer | 3 | 5 | 5 | Covariance model, optimizer layer, constraints, transaction-cost policy | Later high-upside project |
| SPO / decision regret surrogate | 3 | 5 | 5 | Regret target, optimizer solve, score constraints, turnover controls | Treat cautiously |
| DeepDow-style Sharpe/Sortino/MaxDD losses | 2.5 | 5 | 5 | Chronological horizon tensors, portfolio weights, path-state resets | Not first in current trainer |
| Neural covariance conditioning / shrinkage | 4 | 4 | 4 | Covariance labels, risk-loss pipeline, portfolio evaluator | Useful auxiliary research |
| AlphaLoss-style discovered multi-objective loss | 2.5 | 5 | 5 | Evolution loop, backtest-in-the-loop, many path-dependent terms | Idea source, not direct build |
| Reinforcement-learning portfolio policy | 2 | 5 | 5 | Environment, action space, costs, episodes, policy evaluation | Reject for now |
| Multimodal/news/text auxiliary losses | 4 | 2.5 | 5 | PIT text/news feed, embeddings, leakage controls | Data research, not loss-first |

## Practical First Experiment

The least brittle experimental path is a staged uncertainty experiment.

### Stage A: No-training replay

Before changing the model, test the decision rule using uncertainty proxies:

- Ensemble prediction dispersion from existing multi-model runs.
- Rolling residual volatility by stock.
- Cross-sectional score instability across seeds.

Replay saved predictions with:

```text
experimental_score = pred_mean - lambda_uncertainty * uncertainty_proxy
```

Evaluate top-k return, turnover, rank-drop behavior, costs, drawdown, Sharpe,
and year-by-year stability. This is cheap and tells us whether uncertainty
adjustment has signal before changing training.

### Stage B: Distributional alpha head

Add a disabled-by-default head that outputs either:

- `mu, log_sigma` for Gaussian or Student-t NLL.
- `q10, q50, q90` for quantile pinball loss.

Train with one of:

```text
loss = nll
loss = pinball(q10, q50, q90)
loss = nll + alpha * ICLoss(mu, target)
loss = pinball + alpha * LambdaRankIC(q50, target)
```

The portfolio score should be confidence-adjusted, not raw `mu` alone.

### Stage C: PIT repeated-seed comparison

Compare against:

- `ic`
- `portfolio_ic`
- LambdaRankIC
- distributional raw mean
- distributional uncertainty-adjusted rank

Use the same PIT, top-k, transaction-cost, rank-drop, and year-by-year reporting
surface already used for loss comparisons.

## Why This Beats Jumping Straight to Optimizer Losses

Full optimizer losses have the strongest theoretical backtest link, but they
also introduce the largest number of ways to fool ourselves:

- They can create inflated return scores.
- They can induce excessive turnover.
- They depend on the covariance estimate and the portfolio constraints.
- They require chronological training blocks and state reset rules.
- Their gradients may optimize a simplified portfolio layer instead of the
  actual backtest implementation.

Uncertainty-adjusted ranking is more modest but cleaner. It changes the model
contract in a meaningful way, uses stable training losses, and can be tested at
the saved-prediction layer before any invasive trainer rewrite.

## Recommended Path

Use this ordering:

1. Keep pure IC as the production-style default.
2. Prototype LambdaRankIC as the conservative next loss.
3. In parallel, run a no-training uncertainty-adjusted replay using ensemble or
   residual uncertainty.
4. If the replay improves net performance or turnover stability, build a
   distributional alpha head.
5. Revisit decision-focused MVO, SPO, or AlphaLoss-style objectives only after
   the uncertainty branch has a fair PIT repeated-seed result.

Bottom line: removing the current-data criterion makes the best experimental
idea **uncertainty-aware ranking**, not immediate direct Sharpe or optimizer
training. It is the most attractive middle path between stable gradients and
backtest relevance.

## Source Links

- Uncertainty-adjusted sorting: <https://arxiv.org/abs/2601.00593>
- Financial distribution forecasting with NLL, CRPS, and PIT: <https://arxiv.org/abs/2508.18921>
- Quantile regression for stock trading: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6542060>
- Decision-focused MVO return prediction: <https://arxiv.org/abs/2409.09684>
- Machine Learning Meets Markowitz: <https://ideas.repec.org/p/nbr/nberwo/34861.html>
- SPO turnover warning: <https://arxiv.org/abs/2605.01176>
- AlphaLoss SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5263279>
- AlphaLoss OpenReview: <https://openreview.net/forum?id=y0YEGgZnEC>
- Neural nonlinear shrinkage for minimum-variance portfolios: <https://arxiv.org/abs/2601.15597>
- DeepDow portfolio losses: <https://deepdow.readthedocs.io/en/v0.2.2/source/losses.html>
- CVXPYlayers differentiable convex optimization layers: <https://github.com/cvxpy/cvxpylayers>
- Torchsort differentiable sorting/ranking: <https://github.com/teddykoker/torchsort>
- Google fast-soft-sort: <https://github.com/google-research/fast-soft-sort>
