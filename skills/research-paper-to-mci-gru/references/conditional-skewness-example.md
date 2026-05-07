# Conditional Skewness Calibration Example

Use this as a compact style example for Harvey and Siddique, "Conditional Skewness in Asset Pricing: 25 Years of Out-of-Sample Evidence." Do not copy it blindly for other papers.

## Mechanisms

1. **Systematic skewness / coskewness risk premium**
   - Main claim: assets that worsen the skewness of a diversified portfolio can require compensation beyond variance risk.
   - MCI-GRU relevance: the mechanism may appear as stock-level higher-moment features, portfolio diagnostics, or experimental edge attributes.
   - Empirical choices: estimation window length, minimum observations, breakpoints, equal/value weighting, sample filters, and robustness variants.

2. **Research-choice sensitivity of higher moments**
   - Main claim: skewness and coskewness estimates are noisy and sensitive to design choices.
   - MCI-GRU relevance: this argues for diagnostics and notebooks before model-facing production features.
   - Empirical choices: out-of-sample windows, missing observation thresholds, small-cap filters, and weighting scheme.

## Data Readiness Gate

| Requirement | Status | Notes | Blocks? |
| --- | --- | --- | --- |
| Daily stock returns | Derivable | Current OHLCV flow can derive returns from `close` and `prev_close`/group shifts | No |
| Market or portfolio return | Derivable or external dependency | Could derive equal-weight universe return; paper-faithful value-weighting needs market cap | Blocks value-weighted replication |
| Market cap / size filters | External dependency | Not in the core OHLCV contract | Blocks small-cap/value-weighted paper-faithful slices |
| Options or downside-tail data | Unavailable unless separately sourced | Not needed for basic coskewness, but needed for option-implied variants | Blocks option-implied ideas |

## Landing Zone Ranking

| Surface | Rank | Rationale |
| --- | --- | --- |
| `mci_gru/evaluation/` | Primary | Portfolio skewness, downside-return, and coskewness diagnostics can evaluate whether predictions load on skew risk without changing model inputs. |
| `notebooks/` | Primary | Sensitivity analysis is central to the paper and should explore windows, missing-data thresholds, and weighting before productionization. |
| `mci_gru/features/` | Secondary | Rolling stock-level skewness or coskewness features are derivable from returns but need warmup, missing-data, and no-lookahead tests. |
| `configs/experiment/` | Secondary | A feature or diagnostic experiment should have Hydra presets and baseline comparisons. |
| `mci_gru/graph/builder.py` | Long-term | Coskewness edge attributes are plausible but interact with graph timing, edge dimensions, and noisy estimates. |
| `paper_trade/` | Rejected for now | Paper-trade uses frozen artifacts; skewness ideas need offline validation before inference changes. |

## Feasibility Examples

| Slice | Effort | Confidence | Main blocker | Why |
| --- | --- | --- | --- | --- |
| Add portfolio skewness diagnostics to evaluation outputs | easy win | high | validation cost | Uses prediction/backtest returns and does not alter training data. |
| Notebook sensitivity study for rolling coskewness estimates | easy win | high | research design | Best way to explore noisy estimates and empirical choices. |
| Add rolling skewness/coskewness feature family | medium | medium | no-lookahead risk | Derivable from returns, but needs strict rolling windows and warmup handling. |
| Add coskewness graph edge attributes | long-term | low | code complexity | Requires graph timing, edge feature width, and estimation-noise validation. |

## GitHub-Ready Slice Sketches

### 1. training/evaluation: Add skewness-aware portfolio diagnostics

- Problem: Current evaluation can miss whether predicted portfolios take hidden negative-skew exposure.
- Scope: Compute realized portfolio skewness, downside-return share, and optional coskewness against an equal-weight universe return from existing outputs.
- Acceptance criteria: Metrics are deterministic, documented in evaluation summary, and tested on synthetic returns.
- Suggested tests: Synthetic positive/negative skew arrays; NaN handling; top-k portfolio return alignment.
- Out of scope: Model training changes, paper-trade deployment, value-weighted replication.

### 2. notebook: Explore conditional-skewness sensitivity

- Problem: Higher-moment estimates are noisy and sensitive to empirical design.
- Scope: Notebook compares estimation windows, minimum observations, equal-weight market proxy, and missing-data policies.
- Acceptance criteria: Notebook states which inputs are derivable vs blocked and reports sensitivity tables.
- Suggested tests: Notebook smoke execution or extracted pure helper tests.
- Out of scope: Production feature registry changes.

### 3. feature: Add rolling skewness/coskewness feature family

- Problem: The model has no higher-moment node-level feature candidate.
- Scope: Add train-safe rolling skewness/coskewness features behind config flags after notebook evidence.
- Acceptance criteria: Features use only past data, declare warmup behavior, and integrate through `FeatureEngineer`.
- Suggested tests: No-lookahead rolling calculation, missing-data threshold, config wiring.
- Out of scope: Graph edge attributes and paper-trade adoption.

## ADR Candidates

None initially. Consider an ADR only if the project decides that higher-moment paper ideas must validate as diagnostics before becoming model inputs, or if coskewness is assigned permanently to features rather than graph edges.

## Rejected Ideas

- Paper-trade inference change: rejected until offline metrics and diagnostics justify production adoption.
- Paper-faithful value-weighting: blocked until market cap data and point-in-time universe treatment are available.
