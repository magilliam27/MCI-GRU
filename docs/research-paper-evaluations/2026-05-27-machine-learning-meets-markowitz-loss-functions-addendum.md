---
paper_title: "Machine Learning Meets Markowitz - loss-function addendum"
source: "C:\\Users\\magil\\Downloads\\ssrn-5947774.pdf; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5947774; arXiv scans listed in Intake"
evaluated_on: "2026-05-27"
status: "evaluated"
decision: "pursue"
primary_landing_zone: "Training/evaluation"
data_gate: "partial"
recommended_next_action: "Prototype rank-aware and soft top-k utility losses after a saved-prediction alignment diagnostic proves the IC-versus-PnL gap."
github_issue_urls: []
---

# Research-to-Implementation Brief: Machine Learning Meets Markowitz - Loss-Function Addendum

## Intake

The local PDF is an 85-page April 17, 2026 draft of "Machine Learning Meets Markowitz" by Yijie Wang, Hao Gao, Campbell R. Harvey, Yan Liu, and Xinyuan Tao. It updates the earlier MCI-GRU evaluation dated 2026-05-13 and is directly relevant to the user's observation that similar IC can map to very different trading performance.

The paper's mechanism is not "use Markowitz everywhere." The useful claim for MCI-GRU is that a statistical prediction objective can be misaligned with an economic decision objective when the final portfolio is long-only, top-heavy, cost-sensitive, or risk-sensitive. The paper trains through a portfolio construction layer so the model learns which prediction mistakes matter most for the actual portfolio.

Current literature scan:

- SSRN: "Machine Learning Meets Markowitz", SSRN abstract 5947774.
- arXiv: "LambdaRankIC: Directly Optimizing Rank IC for Financial Prediction", submitted 2026-05-01.
- arXiv: "On Evaluating Loss Functions for Stock Ranking: An Empirical Analysis With Transformer Model", submitted 2025-10-15.
- arXiv: "Differentiable Fast Top-K Selection for Large-Scale Recommendation", submitted 2025-10-13, revised 2025-11-04.
- arXiv: "Fast Differentiable Sorting and Ranking", submitted 2020-02-20.
- arXiv: "End-to-End Risk Budgeting Portfolio Optimization with Neural Networks", submitted 2021-07-09.
- DeepDow docs: practical PyTorch portfolio losses including negative Sharpe, Sortino, and maximum drawdown.

## Mechanisms

1. Rank-objective alignment. MCI-GRU currently optimizes Pearson IC. If promotion is judged partly by rank IC, ICIR, and top-k outcomes, then a Rank IC or listwise ranking loss may better match the full cross-sectional ordering objective.

2. Top-k concentration. MCI-GRU trades a small long-only top set, not the whole cross-section. A soft top-k utility loss would weight mistakes near the investable cutoff more heavily than mistakes deep in the middle or bottom of the ranking.

3. Cost-aware signal formation. 2024 diagnostics show trading costs and churn can dominate a valid gross signal. A turnover-aware loss should not only penalize the final portfolio churn; it should encourage lower-churn score dynamics during training.

4. Regime-risk utility. 2022 appears more like a regime-stress failure than a cost-only failure. A utility loss with volatility, downside, drawdown, or regime-conditioned penalty may be more appropriate than simply forcing higher average IC.

## Data Readiness Gate

| Input | Status | MCI-GRU interpretation |
| --- | --- | --- |
| Cross-sectional predictions and returns | already available | Current loss/evaluation paths already consume `(batch, n_stocks)` predictions and labels. |
| PIT masks | already available | Losses must continue to ignore non-finite and non-PIT-active labels. |
| Top-k portfolio returns | already available | `mci_gru/evaluation/portfolio.py` has deterministic top-k and rank helpers. |
| Rank-drop gate state | derivable | Existing portfolio helpers and saved prediction backtests can provide prior holdings/ranks offline. |
| Turnover and transaction costs | derivable | Existing replay/backtest outputs already track cost and turnover; training needs chronological batches for path-dependent costs. |
| Volatility or diagonal risk estimate | derivable | Can use trailing train-only returns; full covariance is not first-pass necessary. |
| Differentiable top-k/sort operator | external dependency or local implementation | Use a small PyTorch-native surrogate first; avoid adding heavy solver dependencies early. |
| Differentiable optimizer layer | external dependency | Solver-backed E2E Markowitz remains long-term and ADR-worthy. |

## MCI-GRU Landing Zone Ranking

Primary first: `mci_gru/evaluation/` and saved-prediction diagnostics. Before adding losses, quantify where daily Pearson IC, Spearman IC, top-k gross return, net return, turnover, and drawdown disagree by year and seed.

Primary second: `mci_gru/training/losses.py`, `mci_gru/training/trainer.py`, and `mci_gru/config.py`. Add config-gated experimental losses after the diagnostic chooses the failure mode.

Secondary: `configs/experiment/` for a small repeated-seed loss grid. Keep the frozen recipe fixed except for the loss and selection metric.

Premature: `paper_trade/`. No paper-trade change until offline PIT repeated-seed validation shows improvement in 2022/2024 without damaging 2023/2025.

Rejected for now: full cvxpylayers or optimizer-layer E2E training. It is conceptually relevant, but too much dependency and batching complexity for the first MCI-GRU slice.

## Invariant Check

Train-only normalization, graph timing, PIT masks, and label embargo must remain unchanged. Any loss operating on realized returns may use only training labels during training, and validation/test must remain out-of-sample.

Soft top-k and rank losses can preserve shuffled batches because they are per-cross-section. Turnover-aware and cost-aware losses are path-dependent and require chronological training blocks, reset at split boundaries. A shuffled cost-aware trainer would be invalid.

Risk estimates must be computed from trailing train-available data only. Backtest fairness must continue to use the existing tradeable return convention. Paper-trade remains frozen-checkpoint only.

## Feasibility Opinion

| Idea | Effort | Confidence | Main blocker |
| --- | --- | --- | --- |
| IC-versus-portfolio alignment diagnostic by year/seed | easy win | high | validation cost |
| Rank IC or listwise ranking loss | medium | medium-high | implementation details |
| Soft top-k utility loss | medium | medium | no-lookahead risk |
| Mean-SD or downside-risk utility loss | medium | medium | calibration and overfit risk |
| Turnover-aware sequential loss | long-term | medium-low | chronological trainer complexity |
| Solver-backed E2E Markowitz layer | long-term | low | dependency and production readiness |

## GitHub-Ready Slices

### Training/evaluation: Add IC-versus-portfolio alignment diagnostics

- Problem: Similar validation/test IC can map to different net trading outcomes across 2022 and 2025.
- Proposed scope: Add a replay-only report over saved predictions with daily Pearson IC, Spearman IC, top-k gross, net, turnover, drawdown, rank-gate churn, and cost attribution by year/seed.
- Acceptance criteria: The report identifies days/months where IC is positive but top-k net return is poor, and separates 2022 regime stress from 2024 churn/cost drag.
- Suggested tests: Synthetic arrays where IC improves while top-k utility worsens; fixed turnover/cost examples with known outputs.
- Out of scope: New model training loss.
- Feasibility Opinion: easy win, high confidence, main blocker validation cost.

### Training/evaluation: Prototype RankIC/Listwise loss

- Problem: Current `ICLoss` optimizes Pearson correlation, while the trading layer mostly consumes ordinal ranks.
- Proposed scope: Add an experimental `rank_ic` or `listwise_rank` loss that approximates Spearman/Rank IC or pairwise/listwise ordering quality on same-day PIT-valid names.
- Acceptance criteria: CPU gradient-flow test, PIT mask test, and a one-epoch smoke run pass; loss is disabled by default.
- Suggested tests: Monotone-score synthetic example, tie/NaN handling, and comparison against current `ICLoss` on tiny batches.
- Out of scope: Portfolio utility, costs, and solver layers.
- Feasibility Opinion: medium effort, medium-high confidence, main blocker implementation details.

### Training/evaluation: Prototype soft top-k utility loss

- Problem: IC weights the full cross-section equally, but MCI-GRU acts through top-20 style long-only selection.
- Proposed scope: Convert scores to smooth top-k weights and maximize realized same-horizon training returns, optionally with entropy/concentration control.
- Acceptance criteria: Top-k surrogate focuses gradients near the cutoff, preserves PIT masks, and can be compared against pure IC in a repeated-seed smoke grid.
- Suggested tests: Known top-k examples, NaN masking, gradient-flow, and no-lookahead label alignment.
- Out of scope: Transaction-cost state and paper-trade behavior.
- Feasibility Opinion: medium effort, medium confidence, main blocker no-lookahead risk.

### Training/evaluation: Design chronological turnover-aware loss path

- Problem: 2024 shows that a valid gross signal can be weakened by churn and costs.
- Proposed scope: Write a design/ADR for a sequential trainer mode that processes consecutive date blocks and adds smooth turnover/cost penalties to soft top-k weights.
- Acceptance criteria: Design names batching contract, split reset behavior, cost convention, and why shuffled training is invalid for this path.
- Suggested tests: Not applicable until the design is approved; later tests should use two- or three-day synthetic paths with exact turnover.
- Out of scope: Immediate implementation.
- Feasibility Opinion: long-term, medium-low confidence, main blocker chronological trainer complexity.

## ADR Candidates

- Whether MCI-GRU should add differentiable soft portfolio losses before a full optimizer layer.
- Whether path-dependent cost-aware training requires a separate chronological trainer.
- Whether solver-backed differentiable optimization dependencies are acceptable in the training stack.

## Rejected Ideas

- Do not replace the frozen recipe because 2022 was weak. Diagnose the mismatch first.
- Do not optimize a single annual Sharpe or drawdown metric directly as the first loss; it is path-dependent, noisy, and easy to overfit.
- Do not use rank-label top-k "return" metrics for promotion until the existing scale issue is audited.
- Do not route anything into `paper_trade/` until offline PIT repeated-seed evidence clears 2022, 2024, and preservation of 2023/2025.

## Open Questions

- Should the first alternative optimize Rank IC, soft top-k return, or a composite `IC + soft top-k` loss?
- Should checkpoint selection stay `val_ic`, move to `val_loss`, or use a composite validation score for experimental losses?
- What is the canonical training utility: gross top-k return, net top-k return, mean-SD utility, downside penalty, or regime-conditioned utility?
- How much turnover reduction is worth sacrificing IC?
- Should the first diagnostic use saved Option A predictions only, or also include a cheap retraining smoke grid?
