---
paper_title: "Machine Learning Meets Markowitz"
source: "C:\\Users\\magil\\Downloads\\Machine Learning Meets Markowitz.pdf"
evaluated_on: "2026-05-13"
status: "evaluated"
decision: "pursue"
primary_landing_zone: "Training/evaluation"
data_gate: "partial"
recommended_next_action: "Add portfolio-objective evaluation diagnostics before adding any end-to-end optimizer layer."
github_issue_urls: []
---

# Research-to-Implementation Brief: Machine Learning Meets Markowitz

## Intake

The PDF is a 93-page academic paper titled "Machine Learning Meets Markowitz" by Yijie Wang, Hao Gao, Campbell R. Harvey, Yan Liu, and Xinyuan Tao. The current draft is dated January 1, 2026.

The paper argues that the standard "predict then optimize" workflow is misaligned with investor utility. Instead of training a return model on global MSE and only later applying portfolio constraints, it proposes end-to-end learning where the prediction model is trained through a portfolio optimization layer.

Empirical choices include China A-share stocks from 2010-2023, 5,489 stocks, 415 firm characteristics plus industry dummies, rolling 39-month training plus 9-month validation windows, daily rebalancing, long-only top-10% portfolios, transaction-cost stress tests, and risk-aversion sweeps. These choices are not the mechanism.

## Mechanisms

1. Decision-aware training: optimize model parameters for realized portfolio utility, not generic cross-sectional forecast error.
2. Constraints/frictions-aware portfolio construction: include long-only constraints, max-position caps, turnover penalties, transaction costs, and risk aversion in the objective being learned.
3. Sequential cross-section training: when portfolio weights depend on previous holdings, batches must preserve chronological order instead of randomly sampling independent stock-date rows.

## Data Readiness Gate

| Input | Status | MCI-GRU interpretation |
|---|---|---|
| Daily OHLCV and returns | already available | Core CSV/LSEG path already supports stock-date panels. |
| Model scores and ranks | already available | Produced under `averaged_predictions/` and used by evaluation/paper-trade. |
| Long-only top-k weights | already available | `mci_gru/evaluation/portfolio.py` has top-k and rank utilities. |
| Turnover and previous holdings | derivable | Existing turnover and paper-trade state logic can be reused offline. |
| Diagonal risk or volatility estimate | derivable | Compute from trailing returns; full covariance is heavier. |
| Transaction cost rates | derivable/configurable | Use scenario parameters first; exact live costs are external. |
| China A-share universe and 415 characteristics | external dependency | Blocks exact paper replication. |
| Differentiable convex solver layer | unavailable in repo | Requires dependency and architecture decision. |

Exact replication is blocked by market, characteristic, and execution data. A project-native adaptation is feasible using current MCI-GRU scores, labels, returns, top-k evaluation, and configurable cost/risk assumptions.

## MCI-GRU Landing Zone Ranking

1. Primary: `mci_gru/evaluation/portfolio.py` and `mci_gru/evaluation/statistics.py` - evaluate whether current MCI-GRU predictions are aligned with top-k portfolio utility, turnover, transaction costs, and risk-adjusted returns.
2. Primary later: `mci_gru/training/losses.py` and `mci_gru/training/trainer.py` - add a portfolio-aware loss only after evaluation shows objective mismatch.
3. Secondary: `configs/config.yaml` and `configs/experiment/` - expose risk aversion, cost rate, top-k cap, and objective sweeps as Hydra experiment presets.
4. Premature: `mci_gru/models/` and `mci_gru/graph/builder.py` - the mechanism is an objective/training change, not a graph or architecture change.
5. Rejected for now: `paper_trade/` - frozen inference should only use this after offline validation and retraining.

Repo evidence:

- `mci_gru/training/losses.py`: current objectives are MSE, IC, and combined MSE/IC.
- `mci_gru/training/trainer.py`: current trainer consumes batches independently and optimizes per-batch prediction loss.
- `mci_gru/data/data_manager.py`: static training shuffles by default; dynamic graph mode disables shuffle, which matters for sequential cost-aware training.
- `mci_gru/evaluation/portfolio.py`: rank, top-k, turnover, and rank-drop primitives already exist.
- `docs/BACKTEST_FAIRNESS_AUDIT.md`: execution timing and return attribution must be explicit.
- `paper_trade/scripts/infer.py`: paper-trade uses frozen checkpoints and frozen graph data.

## Invariant Check

Train-only normalization must remain unchanged. Any portfolio-aware loss can use training labels during training, but validation/test evaluation must only use out-of-sample predictions and realized returns.

If the loss uses previous portfolio weights or transaction costs, training batches must be chronological blocks and must reset state at train/validation/test boundaries. Random shuffled batches are invalid for path-dependent turnover objectives.

Risk and covariance estimates must use returns strictly before the decision date. Graph snapshots, if ever used, must still go through `GraphSchedule`. Label embargo gaps must stay strictly greater than `model.label_t`. Backtests must use tradeable returns consistent with prediction time and entry time. Paper-trade remains frozen-checkpoint only.

## Feasibility Opinion

| Idea | Effort | Confidence | Main blocker |
|---|---|---|---|
| Portfolio-objective evaluation diagnostics | easy win | high | validation cost |
| Configurable cost/risk/top-k evaluation sweep | easy win | high | validation cost |
| Smooth portfolio-aware loss without solver | medium | medium | no-lookahead risk |
| Sequential cost-aware trainer path | long-term | medium-low | code complexity |
| Full cvxpylayers/Clarabel optimization layer | long-term | low | dependency and production readiness |
| Exact China A-share replication | long-term | low | data |

## GitHub-Ready Slices

### Training/evaluation: Add portfolio-objective alignment diagnostics

- Problem: Current training optimizes MSE/IC-style prediction quality, but the paper shows this may not align with realized constrained portfolio utility.
- Proposed scope: Add an offline evaluator that compares IC/MSE against top-k return, turnover, cost-adjusted return, Newey-West Sharpe, and rank concentration.
- Acceptance criteria: Produces a report showing where high IC does or does not translate into better long-only portfolio utility.
- Suggested tests: Synthetic predictions where IC improves but top-k utility worsens; deterministic top-k and turnover tests.
- Out of scope: New training loss or optimizer layer.
- Feasibility Opinion: easy win, high confidence, main blocker validation cost.

### Config/experiment: Add Markowitz-style utility sweep

- Problem: The repo lacks a simple way to sweep investor preferences and frictions over existing predictions.
- Proposed scope: Add Hydra-configured evaluation parameters for top-k cap, long-only mode, transaction cost rate, risk aversion, and diagonal-volatility lookback.
- Acceptance criteria: One experiment can compare current MCI-GRU predictions across multiple cost/risk settings without retraining.
- Suggested tests: Config load test; small fixed prediction matrix with known utility ordering.
- Out of scope: Full covariance optimization and live trading.
- Feasibility Opinion: easy win, high confidence, main blocker validation cost.

### Training/evaluation: Prototype smooth portfolio-aware loss

- Problem: MSE/IC treats all stocks equally, while MCI-GRU mostly acts on top-ranked names.
- Proposed scope: Add a config-gated loss that uses differentiable soft top-k or smooth capped long-only weights to optimize realized training-period portfolio utility.
- Acceptance criteria: Loss is disabled by default, supports CPU smoke tests, and proves gradients flow without a convex solver dependency.
- Suggested tests: Gradient-flow test; no-lookahead label-window test; train-step smoke test.
- Out of scope: cvxpylayers, transaction-cost state, and paper-trade.
- Feasibility Opinion: medium effort, medium confidence, main blocker no-lookahead risk.

### ADR/Training: Decide whether to support a true differentiable optimizer layer

- Problem: A faithful E2E Markowitz layer needs solver dependencies, sequential batching, warm starts, and new failure modes.
- Proposed scope: Write an ADR comparing smooth surrogate loss, cvxpylayers/Clarabel, PyTorch-native quadratic/soft-top-k approximations, and evaluation-only usage.
- Acceptance criteria: ADR chooses a path, names dependency implications, and defines the minimum validation bar before implementation.
- Suggested tests: Not applicable until a path is selected.
- Out of scope: Implementing the optimizer layer in the ADR slice.
- Feasibility Opinion: long-term, low confidence, main blocker production readiness.

## ADR Candidates

- ADR: Whether MCI-GRU should stay prediction-first with portfolio-aware evaluation or introduce portfolio-aware training losses.
- ADR: Whether to permit solver dependencies such as `cvxpylayers` and Clarabel in the training stack.
- ADR: Whether path-dependent objectives require a chronological-block trainer separate from the existing default trainer.

## Rejected Ideas

- Do not implement a full differentiable convex optimization layer as the first step.
- Do not treat the paper's China A-share results as evidence that MCI-GRU stock-universe results will transfer.
- Do not add paper-trade behavior before offline retraining and fairness validation.
- Do not use shuffled batches for transaction-cost-aware training.
- Do not evaluate cost-aware strategies with non-tradeable close-to-close return attribution.

## Open Questions

- Should MCI-GRU's decision objective be top-k long-only return, mean-standard-deviation utility, rank-drop utility, or cost-adjusted utility?
- Should the first training prototype use a smooth soft-top-k surrogate or a real solver layer?
- What execution return should be canonical for offline evaluation: open-to-open, open-to-close, or another tradeable convention?
- Should `model.label_t=5` remain the horizon for this work, or should a separate quarterly-style experiment be created?
- What transaction cost rates are realistic for the user's intended market and turnover profile?
