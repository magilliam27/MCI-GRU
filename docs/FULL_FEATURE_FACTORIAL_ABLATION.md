# MCI-GRU Full Feature Factorial Ablation

Last updated: 2026-04-29

This document captures the experiment design behind
`notebooks/full_feature_factorial_ablation_colab.ipynb`.

The goal is to move from one-off ablations to a full interaction test around the
three levers now worth isolating:

- dynamic momentum
- dynamic graph construction
- global regime variables

The notebook follows the structure of
`notebooks/ablation_evaluation_loop_colab.ipynb`, but every primary run uses the
full decision budget:

```text
training.num_models = 20
training.num_epochs = 100
training.early_stopping_patience = 15
evaluation.bootstrap_resamples = 1000
```

## Architecture Findings

The architecture reference in `docs/ARCHITECTURE.md` separates the model into
three relevant intervention points.

First, feature engineering happens before normalization and windowing through
`FeatureEngineer`. That means momentum and regime variables are true input
features, normalized with train-period statistics only. This makes feature
ablations clean as long as each run keeps the same date split.

Second, graph construction happens inside `prepare_data`. Static graph mode
builds one Pearson-correlation graph. Dynamic graph mode precomputes
`GraphSchedule` snapshots and the collate function resolves the correct graph
per batch date. This means dynamic graph ablations should use normal batch sizes;
they do not require batch size 1.

Third, the model consumes correlation graph edges in the GAT streams. With
`graph.use_multi_feature_edges=true`, the edge attributes are at least four
channels: `corr`, `abs(corr)`, `corr^2`, and `rank_pct`. Because the new default
already uses these multi-feature edge attributes, the main graph question is no
longer scalar versus multi-feature edges. The more important question is whether
correlations should update through time and whether top-K neighbour selection is
better than the legacy global threshold.

## Primary Factorial Design

The primary notebook matrix is:

```text
2 momentum modes x 4 graph modes x 3 regime modes = 24 primary runs
```

### Momentum Factor

`momentum_static`

- Binary momentum.
- Weekly momentum enabled.
- Static fast/slow blend using `momentum_blend_fast_weight=0.5`.
- This matches the new default.

`momentum_dynamic`

- Binary momentum.
- Weekly momentum enabled.
- Dynamic fast/slow blend using expanding historical state-conditioned returns.
- Uses the current defaults:
  - `momentum_dynamic_correction_fast_weight=0.15`
  - `momentum_dynamic_rebound_fast_weight=0.7`
  - `momentum_dynamic_min_history=252`
  - `momentum_dynamic_min_state_observations=3`

### Graph Factor

`graph_static_threshold`

- Static graph.
- `graph.update_frequency_months=0`
- `graph.top_k=0`
- Multi-feature edges enabled.
- This matches the new default.

`graph_dynamic_threshold_6mo`

- Dynamic graph snapshots every 6 months.
- Keeps the legacy global threshold path.
- Tests whether stale correlations are hurting the static graph.

`graph_dynamic_topk20_pos_6mo`

- Dynamic graph snapshots every 6 months.
- Per-node `top_k=20`.
- Ranks neighbours by positive correlation with `top_k_metric=corr`.
- Tests whether stable positive peer relationships beat the global threshold.

`graph_dynamic_topk20_signed_6mo`

- Dynamic graph snapshots every 6 months.
- Per-node `top_k=20`.
- Ranks neighbours by absolute correlation with `top_k_metric=abs_corr`.
- Preserves signed edge features.
- Tests whether negative-correlation relationships carry useful cross-sectional
  information.

### Regime Factor

`regime_off`

- No global regime variables.
- This matches the new default.

`regime_current_only`

- Enables global regime similarity features.
- Disables historical subsequent-return context with
  `features.regime_include_subsequent_returns=false`.
- Tests whether the current macro similarity state helps without forward-return
  conditioning.

`regime_with_forward_context`

- Enables the full regime feature set.
- Includes similar-history subsequent-return features for 1-month and 3-month
  horizons.
- Uses `regime_exclusion_months=1`, which the implementation requires to avoid
  near-term lookahead.

## Diagnostic Controls

The notebook also includes non-factorial diagnostic controls. These are not used
for main-effect averages, but they help interpret failures or surprising wins.

`control__base-only__graph-static-threshold__regime-off`

- Disables momentum.
- Measures the value of all momentum features relative to OHLCV plus turnover.

`control__momentum-static-no-weekly__graph-static-threshold__regime-off`

- Static momentum without weekly momentum.
- Isolates whether the weekly momentum columns add signal.

`control__momentum-dynamic-no-weekly__graph-static-threshold__regime-off`

- Dynamic momentum without weekly momentum.
- Tests whether the dynamic speed selector still helps without weekly terms.

`control__momentum-static__graph-static-scalar-edges__regime-off`

- Uses legacy scalar edge weights.
- Isolates the value of the default multi-feature edge representation.

## Regime Data Finding

Regime runs can silently become zero-filled if auxiliary data cannot be loaded and
`features.regime_strict=false`. That is useful for robustness in production, but
bad for ablation interpretation. The new notebook therefore defaults to:

```text
REGIME_STRICT_FOR_REGIME_RUNS = True
```

For regime runs to complete, provide either:

- `FRED_API_KEY`, plus any required LSEG/FRED access for configured series, or
- `REGIME_INPUTS_CSV`, following `docs/REGIME_DATA_CONTRACT.md`.

If neither is available, regime runs should fail rather than masquerade as valid
zero-regime experiments.

## Decision Tables

The notebook writes:

- `full_feature_factorial_manifest.json`
- `ablation_results_raw.csv`
- `ablation_decision_table.csv`
- `ablation_decision_table.html`
- `momentum_factor_main_effects.csv`
- `graph_factor_main_effects.csv`
- `regime_factor_main_effects.csv`
- pairwise interaction tables for momentum/graph, momentum/regime, and graph/regime
- `ablation_summary_report.md`

The decision score matches the older notebook's spirit:

```text
35% average IC
25% average Spearman correlation
25% top-20 Newey-West Sharpe
15% top-20 return
```

The score is a ranking aid, not a replacement for checking confidence intervals.
The most important confirmatory columns are:

- `evaluation.metrics.avg_ic`
- `evaluation.metrics.avg_ic_ci_lower`
- `evaluation.metrics.return_top_20`
- `evaluation.metrics.top_20_return_ci_lower`
- `evaluation.metrics.sharpe_top_20_newey_west`

## How To Read The Results

Start with the control/default row:

```text
momentum_static x graph_static_threshold x regime_off
```

This is the clean comparison point for the new default settings.

Then read main effects:

- If `momentum_dynamic` beats `momentum_static` across graph and regime settings,
  dynamic speed selection is broadly useful.
- If it only wins under specific regime settings, the dynamic blend may need
  macro-state context rather than being a universal upgrade.
- If dynamic graph modes beat `graph_static_threshold`, stale graph edges are
  likely costing signal.
- If top-K signed beats top-K positive, negative correlations are useful enough
  to keep in the graph budget.
- If `regime_current_only` beats `regime_with_forward_context`, the forward
  context features may be noisy or too slow for the 5-day label horizon.

Then read interactions:

- Momentum x regime answers whether dynamic momentum and macro regime are
  substitutes or complements.
- Graph x regime answers whether changing correlations already capture macro
  state, or whether regime variables add something orthogonal.
- Momentum x graph answers whether the temporal signal and cross-sectional graph
  signal are reinforcing each other.

## Current Recommendation Before Empirical Results

Run the full 24-run factorial plus the diagnostic controls. Do not add VIX,
credit spreads, RSI, MA, price-derived, or volume-derived feature families to the
same factorial yet. Those features are valid future candidates, but crossing them
with momentum, graph, and regime would turn this into a much larger search and
make interpretation harder.

After this run finishes, promote only the best two or three configurations to a
fresh holdout or walk-forward confirmation.
