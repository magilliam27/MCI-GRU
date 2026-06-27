---
paper_title: "Investment Base Pairs"
source: "C:\\Users\\magil\\Downloads\\656_Base_pairs_2026.pdf"
evaluated_on: "2026-05-13"
status: "evaluated"
decision: "pursue"
primary_landing_zone: "Training/evaluation"
data_gate: "partial"
recommended_next_action: "Add pair-decomposition diagnostics for MCI-GRU ranked predictions before changing graph or model code."
github_issue_urls: []
---

# Research-to-Implementation Brief: Investment Base Pairs

## Intake

`656_Base_pairs_2026.pdf` is a 79-page research deck. The extractor metadata was noisy, but the cover and content indicate the title is "Investment Base Pairs" by Campbell R. Harvey, based on joint research with Christian Goulding.

The deck studies cross-sectional long/short strategies across futures asset classes and argues that rank-based strategies can be decomposed into simpler long/short base-pair trades.

## Mechanisms

1. Rank-based cross-sectional strategies can be decomposed into long/short base-pair strategies.
2. Pair quality is explained by five drivers: signal correlation, own signal-return correlation, cross signal-return correlation, signal mean imbalance, and signal variance imbalance.
3. Selectively trading only stronger pairs can remove junk exposure, improving many return and Sharpe profiles while increasing concentration risk.

## Data Readiness Gate

| Input | Status | MCI-GRU interpretation |
|---|---|---|
| Model scores or ranks | derivable | Use MCI-GRU predictions as the signal. |
| Forward returns | already available | Use existing labels and realized returns. |
| Pair decomposition | derivable | Compute from ranks and realized returns. |
| Momentum-style signals | partly available | Existing feature registry has momentum and volatility-style features. |
| Futures value and carry signals | external dependency | Blocks exact paper replication. |
| Asset-class leverage assumptions | external dependency | Not needed for first MCI-GRU diagnostic. |

Exact paper replication is blocked by futures data and signal-provenance requirements. An MCI-GRU-native adaptation using existing stock predictions and returns is feasible.

## MCI-GRU Landing Zone Ranking

1. Primary: `mci_gru/evaluation/` - add pair-decomposition diagnostics for ranked prediction portfolios.
2. Secondary: notebooks/reports - sweep pair selectivity thresholds before productionizing.
3. Long-term: `mci_gru/graph/builder.py` - consider pair-quality edge attributes or pruning only after offline evidence.
4. Rejected for now: `paper_trade/` - paper-trade should not receive pair logic until offline validation and retraining exist.

Repo evidence:

- `docs/ARCHITECTURE.md`: train-only normalization, graph schedule, evaluation layer.
- `mci_gru/graph/builder.py`: correlation graph, top-k, multi-feature edge attributes.
- `mci_gru/evaluation/portfolio.py`: ranking, top-k returns, turnover, rank-drop.
- `mci_gru/evaluation/statistics.py`: IC, Newey-West Sharpe, bootstrap CI.
- `paper_trade/scripts/infer.py`: frozen inference path.

## Invariant Check

Pair scores must obey strict no-lookahead. For prediction date `t`, pair-quality history may only use signal-return observations whose forward-return window is fully known before `t`.

If the model uses `label_t=5`, a pair score at `t` must not include observations whose 5-day return overlaps `t`. Any graph use must route through `GraphSchedule`, not ad hoc per-batch graph recomputation. Normalization remains train-period only.

## Feasibility Opinion

| Idea | Effort | Confidence | Main blocker |
|---|---|---|---|
| Pair decomposition diagnostics | easy win | high | validation cost |
| Pair-selectivity notebook | easy win | high | validation cost |
| Pair-filtered evaluator | medium | medium | no-lookahead risk |
| Pair-quality graph edges or pruning | long-term | medium-low | code complexity |
| Exact futures replication | long-term | low | data |

## GitHub-Ready Slices

### Data: Define paper-faithful replication requirements

- Problem: Exact replication needs futures value, momentum, carry, leverage, and universe data not currently in MCI-GRU.
- Proposed scope: Document required vendors, fields, transformations, and provenance.
- Acceptance criteria: The issue separates exact replication from MCI-GRU-native adaptation.
- Suggested tests: None until data is selected.
- Out of scope: Building model or graph changes from proxies.
- Feasibility Opinion: long-term, low confidence, main blocker data.

### Training/evaluation: Add base-pair decomposition diagnostics

- Problem: MCI-GRU rank outputs are evaluated at portfolio level, not pair-contribution level.
- Proposed scope: Given model scores and realized returns, compute pair-level contributions and rank good/junk pairs.
- Acceptance criteria: Produces pair contribution tables, good/junk rankings, and no-lookahead tests.
- Suggested tests: Synthetic ranked portfolio with known pair contributions; label-window overlap guard.
- Out of scope: Changing training, graph construction, or paper-trade.
- Feasibility Opinion: easy win, high confidence, main blocker validation cost.

### Notebook: Run base-pair selectivity study on MCI-GRU outputs

- Problem: Pair selectivity should be evaluated empirically before production changes.
- Proposed scope: Sweep 10%, 20%, 50%, and 100% pair inclusion and compare return, IC, Sharpe, turnover, and concentration.
- Acceptance criteria: Report identifies whether selectivity improves out-of-sample behavior.
- Suggested tests: Notebook smoke path over a small saved prediction sample.
- Out of scope: Exact futures replication.
- Feasibility Opinion: easy win, high confidence, main blocker validation cost.

### Graph/config experiment: Prototype pair-quality edge signal

- Problem: Pair quality may improve the correlation graph, but only if diagnostics show stable signal.
- Proposed scope: Add Hydra-gated experiment using trailing pair quality as edge attribute or pruning score.
- Acceptance criteria: Uses `GraphSchedule`, preserves edge-dimension wiring, and passes no-lookahead tests.
- Suggested tests: Dynamic snapshot timing test; edge feature dimension test; graph batch collation test.
- Out of scope: Paper-trade deployment.
- Feasibility Opinion: long-term, medium-low confidence, main blocker code complexity.

## ADR Candidates

- ADR: Pair-quality logic belongs in evaluation first, graph second. This records the decision to validate base-pair behavior as an analysis layer before allowing it to influence model training.

## Rejected Ideas

- Do not put pair selectivity directly into paper-trade.
- Do not use futures value or carry proxies without a data/provenance issue.
- Do not compute pair scores from the same future returns being evaluated.
- Do not change MCI-GRU architecture before proving pair selectivity helps offline.
- Do not rebuild graphs inside paper-trade inference.

## Open Questions

- Should the MCI-GRU signal be model prediction, raw momentum feature, or a named alpha component?
- Which universe should be used first: current stock universe, PIT universe only, or a smaller stable test universe?
- What trailing window is realistic given MCI-GRU's available train history?
- Should the first output be a notebook, a reusable evaluator module, or both?
