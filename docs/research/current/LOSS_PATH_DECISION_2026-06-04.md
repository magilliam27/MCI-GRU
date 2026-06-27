---
title: "Next Loss Path Decision for MCI-GRU"
evaluated_on: "2026-06-04"
status: "current research evidence"
decision: "Keep pure IC as the launch default; explore LambdaRankIC-style pairwise Rank IC next"
primary_landing_zone: "Training/evaluation"
---

# Next Loss Path Decision for MCI-GRU

## Decision

Keep the current frozen production-style default, `training.loss_type=ic` with
`selection_metric=val_ic`, as the best launch-safe loss today.

The next best loss path to explore is a disabled-by-default
`lambdarank_ic` / pairwise Rank IC surrogate in `mci_gru/training/losses.py`,
paired with explicit Spearman Rank IC diagnostics and a PIT repeated-seed
comparison against `ic` and `portfolio_ic`.

This is a recommendation to explore, not to replace the current launch default
yet. MCI-GRU already has a better-tested launch loss than the new candidate.
The candidate earns the next experiment slot because it best satisfies the
requested criteria:

1. It uses information MCI-GRU already has.
2. It can give the model a smoother pairwise logistic gradient than hard ranking.
3. Its target metric, Rank IC, is closer to ranking/backtest behavior than MSE
   and less path-dependent than direct Sharpe, turnover, or drawdown losses.

## Criteria

Ranked by importance from the goal:

1. The loss must use information MCI-GRU already has.
2. The loss should produce a stable gradient for the model.
3. The loss metric should relate to backtesting performance.

Secondary constraints:

- Preserve PIT masked-panel behavior and no-lookahead invariants.
- Avoid path-dependent losses inside the current shuffled static-graph trainer.
- Keep paper-trade frozen until offline PIT evidence exists.
- Prefer a PyTorch-native prototype over a new model stack.

## Current Repo Evidence

Current losses in `mci_gru/training/losses.py`:

- `MaskedMSELoss`
- `ICLoss`: negative Pearson cross-sectional IC
- `CombinedMSEICLoss`
- `SoftTopKForwardReturnLoss`
- `PortfolioICLoss`: `ICLoss` plus soft top-k forward-return utility

Current config in `mci_gru/config.py` supports:

- `mse`
- `ic`
- `combined`
- `portfolio_ic`

The trainer already has the right seam:

- `Trainer.train()` builds a criterion with `build_training_loss(...)`.
- `_train_epoch()` calls `loss = criterion(outputs, labels)`.
- `_validate()` reports `val_ic` through `mean_information_coefficient(...)`.

Evaluation already has useful backtest-facing metrics:

- `mci_gru/evaluation/statistics.py`: Pearson and Spearman daily IC helpers.
- `mci_gru/evaluation/portfolio.py`: top-k selection, turnover, and rank-drop
  gate helpers.
- `mci_gru/training/metrics.py`: Spearman correlation, top-k returns, Sharpe,
  and multiple top-k metrics in saved prediction evaluation.

## External Evidence

LambdaRankIC is the strongest research fit. The arXiv paper argues that finance
models are commonly evaluated by Rank IC, but trained with regression or generic
ranking losses; it derives pairwise LambdaRank-style pseudo-gradients for Rank
IC and reports stronger Rank IC, ICIR, monthly return, and Sharpe in its setting.
Source: <https://arxiv.org/abs/2605.00501>.

The broader stock-ranking loss benchmark argues that ordinary pointwise losses
do not directly teach stock-return ordering and benchmarks pointwise, pairwise,
and listwise losses for daily S&P 500 stock ranking with Transformer models.
Source: <https://arxiv.org/abs/2510.14156>.

Qlib's public benchmark framing matches MCI-GRU's practical split between
signal metrics and portfolio metrics: alpha scores are evaluated by correlation
with future returns and by constructing portfolios; benchmark tables report IC,
ICIR, Rank IC, Rank ICIR, annualized return, information ratio, and max drawdown.
Source: <https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md>.

Qlib's TopkDropout strategy also confirms that a common production-style ranking
strategy cares about score order more than score scale. This matches MCI-GRU's
rank-drop gate and top-k behavior. Source:
<https://github.com/microsoft/qlib/blob/main/docs/component/strategy.rst>.

Torchsort provides a public PyTorch implementation of differentiable soft ranks
and shows a differentiable Spearman example. It is useful as a later alternative
or implementation reference, but adding a compiled dependency is heavier than a
first PyTorch-native pairwise surrogate. Source:
<https://github.com/teddykoker/torchsort>.

Google Research's `fast-soft-sort` provides differentiable sorting and ranking
operations in `O(n log n)` with optional PyTorch support. It is useful research
evidence but less convenient for a first MCI-GRU loss because the repo is not a
simple packaged dependency. Source:
<https://github.com/google-research/fast-soft-sort>.

allRank is a PyTorch learning-to-rank framework with pointwise, pairwise, and
listwise losses such as RankNet, LambdaRank, LambdaLoss, ApproxNDCG, ListNet,
and ListMLE. It is useful for implementation shape, but its built-in objectives
are not finance-native Rank IC. Source: <https://github.com/allegro/allRank>.

DeepDow and CVXPYlayers show credible paths for Sharpe, Sortino, maximum
drawdown, risk parity, and differentiable optimizer layers, but those losses are
path-dependent or solver-dependent and should not be first in the current
trainer. Sources:

- <https://deepdow.readthedocs.io/en/v0.2.2/source/losses.html>
- <https://github.com/cvxpy/cvxpylayers>

## Gradient Sanity Check

A scratch local check compared current losses against a minimal pairwise
LambdaRankIC-style surrogate and a soft-rank Spearman surrogate on a tiny masked
batch. This was not a performance test; it only checked whether gradients are
finite and nonzero on representative `(batch, n_stocks)` tensors with NaNs.

Observed output:

```text
ic: loss=-0.730079 grad_norm=12.874610 max_abs_grad=4.967046 finite_grad=True
portfolio_ic: loss=-0.750173 grad_norm=15.844465 max_abs_grad=8.443178 finite_grad=True
lambdarank_ic_surrogate: loss=+0.216268 grad_norm=0.990428 max_abs_grad=0.588641 finite_grad=True
soft_spearman_tau_0p05: loss=-0.728994 grad_norm=12.814550 max_abs_grad=4.926031 finite_grad=True
```

Interpretation:

- Existing `ic` and `portfolio_ic` are differentiable and usable.
- The minimal LambdaRankIC-style pairwise surrogate produced finite gradients
  with a much smaller gradient norm on this tiny example.
- This does not prove backtest performance, but it supports the "stable
  gradient" criterion enough to justify a focused prototype.

## Candidate Scorecard

Scores use 1 to 5, where 5 is strongest for the criterion.

| Candidate | Uses MCI-GRU data | Gradient stability | Backtest relation | Implementation risk | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure IC, current default | 5 | 3 | 4 | 1 | 16 |
| Portfolio-IC, current experimental | 5 | 3 | 5 | 2 | 15 |
| LambdaRankIC-style pairwise Rank IC | 5 | 4 | 5 | 3 | 17 |
| Soft Spearman / differentiable soft rank | 5 | 3 | 5 | 4 | 15 |
| Generic pairwise/listwise LTR | 5 | 4 | 3 | 3 | 14 |
| Direct profit/action loss | 4 | 3 | 5 | 5 | 12 |
| Sharpe/Sortino/MaxDD loss | 3 | 2 | 5 | 5 | 10 |
| Decision-focused MVO / optimizer layer | 3 | 2 | 5 | 5 | 10 |
| Probabilistic NLL / quantile | 3 | 5 | 2 | 4 | 10 |

Notes:

- "Implementation risk" is reverse-scored: lower risk receives a higher
  contribution to overall score.
- `PortfolioICLoss` is the most directly top-k/backtest-adjacent current loss,
  but previous experiment context did not prove it beats pure IC after costs.
- LambdaRankIC wins the next-exploration slot because it adds a missing
  full-order rank objective without requiring chronological batching or new data.

## Why Not Make Portfolio-IC the Next Main Path?

Portfolio-IC is still useful, but it is already in the repo and already being
swept. It answers a different question: can a soft top-k return term improve the
portfolio layer?

The open gap is full-rank score quality. MCI-GRU's trading layer is rank-driven:
top-k selection, rank-drop gates, and score order matter more than calibrated
return magnitude. LambdaRankIC targets that rank order directly while staying
inside the current per-date loss contract.

## Why Not Direct Sharpe, Turnover, or Drawdown?

These metrics relate strongly to backtests, but they fail the current setup
criterion.

They are path-dependent. A turnover, cost, Sharpe, Sortino, or max-drawdown loss
needs chronological training blocks, previous-holding state, and split-boundary
resets. The current static-graph training path shuffles by default, so inserting
these objectives into the existing criterion seam would be conceptually wrong.

They remain later candidates after a chronological trainer design exists.

## Recommended Loss Path

### Step 1: Add Rank IC diagnostics

Make Spearman Rank IC explicit beside Pearson `val_ic`.

Expected changes:

- Add validation Rank IC reporting without changing current `val_ic`.
- Add or document `selection_metric=val_rank_ic` only after tests define it.
- Ensure saved summaries distinguish Pearson IC from Spearman Rank IC.

Why first:

- It proves whether the current launch loss is already strong on the metric that
  LambdaRankIC would target.
- It gives a fair comparison surface before any new loss changes training.

### Step 2: Prototype `lambdarank_ic`

Add a disabled-by-default PyTorch loss with this first-pass behavior:

- Input: current `(pred, target)` tensors.
- Scope: per batch row, same-date cross-section only.
- Mask: finite predictions and labels only.
- Ranking: derive label ranks from current return labels inside the loss.
- Pair loss: logistic pairwise ordering.
- Pair weights: RankIC-inspired separation weight,
  `12 * |pred_rank_j - pred_rank_i| * |label_rank_i - label_rank_j| / (n * (n^2 - 1))`.
- Runtime control: start with either all pairs for small breadth or deterministic
  sampled pairs via `lambdarank_ic_max_pairs_per_day`.
- Defaults: disabled in base config; no paper-trade effects.

Tests should cover:

- perfect order versus reversed order,
- partially swapped order,
- all-NaN and one-valid-stock rows,
- tied predictions and tied labels,
- finite gradient flow,
- pair sampling determinism,
- config validation and `build_training_loss(...)` label readback.

### Step 3: Compare Against Current Launch Candidate

Use a small fixed grid before any full ensemble:

- Current launch default: `loss_type=ic`, `selection_metric=val_ic`.
- Current portfolio branch: `loss_type=portfolio_ic`, fixed top-k and weight.
- New candidate: `loss_type=lambdarank_ic`.

Hold fixed:

- PIT masked-panel setup,
- 5-day raw return labels,
- frozen feature/graph recipe,
- model architecture,
- seeds,
- backtest return timing,
- transaction costs,
- rank-drop gate.

Promotion evidence should include:

- Pearson IC and Spearman Rank IC,
- ICIR and Rank ICIR,
- top-k gross return,
- net return after transaction costs,
- turnover / rank-drop churn,
- Newey-West Sharpe,
- drawdown,
- year-by-year behavior.

## Practical Answer

MCI-GRU probably already has the best launch-safe loss in the current setup:
pure IC. It is simple, PIT-safe, already wired into the frozen recipe, and its
metric is strongly related to the way the system scores and ranks stocks.

The most advantageous next loss to explore is LambdaRankIC-style pairwise Rank
IC, not another portfolio utility loss. It uses existing labels and scores,
offers a smoother pairwise gradient path than hard Spearman ranking, and targets
a metric that sits closer to rank-driven backtesting than MSE while avoiding
the path-dependence of Sharpe, turnover, and drawdown.

The go/no-go question after implementation is direct:

> Does `lambdarank_ic` improve Rank IC and net top-k/rank-drop backtests versus
> pure IC without increasing turnover, drawdown, or year instability?

Until that is answered, pure IC remains the launch default.
