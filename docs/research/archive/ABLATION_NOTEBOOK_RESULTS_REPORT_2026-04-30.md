# MCI-GRU Ablation Notebook Results Report

Date prepared: 2026-04-30

Source Drive folder: [MCI-GRU-Ablations](https://drive.google.com/drive/folders/1KUIj06ekfNpZa1IkkcAdhHXbVZt-PYT5)

This report covers the two latest notebook result folders:

- Dynamic graph methods: [dynamic_graph_methods/20260430_024339](https://drive.google.com/drive/folders/1cWnZZqOoTeu9GtsR4rNP01431yQ9JAFY)
- Training/feature-factor ablation: [training_factor_ablation/20260430_024344](https://drive.google.com/drive/folders/1mEVQ-ILnjzUwY8xsdfvWLpgwZLC2ZOX9)

Both notebooks used `sp500_2019_universe_data_through_2026.csv`, 20-model ensembles, 100 epochs, early stopping patience 15, and 1,000 bootstrap resamples.

## Executive Summary

The dynamic-graph notebook does not show that "dynamic" is automatically better than the static control. The best dynamic configuration was a 6-month dynamic threshold graph with a longer 504-trading-day correlation lookback and forward-regime context. It achieved the top decision score in that notebook, but the static threshold shuffled control remained very competitive and had the best single-run average IC and top-20 Newey-West Sharpe.

The training/feature-factor notebook gives a much clearer signal: pure IC loss is the strongest objective family by decision score and average IC. Pure MSE is weak, especially with rank labels. Combined MSE+IC loss is respectable for portfolio-return-shaped metrics but generally weaker on IC confidence. Selecting checkpoints by validation loss appears more stable than selecting by validation IC in the control slice.

The main action item is to promote a raw-return pure-IC candidate as the next production-style contender, while treating the pure-IC rank-label result as interesting but not directly comparable to raw-return results. Rank-label runs produced top-k "return" values around 0.52 to 0.54, while raw-return runs are around 0.008; that scale mismatch should be audited before using rank-label portfolio metrics for model selection.

## Dynamic Graph Notebook

### Best Individual Runs

| Rank | Run | Decision score | Avg IC | IC CI lower | Top-20 return | Top-20 CI lower | Notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `dynamic-threshold-6mo-lookback504-shuffle__regime-with-forward-context` | 0.9426 | 0.0352 | 0.0026 | 0.0083 | 0.0013 | Best decision score; only dynamic variant with positive top-20 lower CI in the top group. |
| 2 | `static-threshold-shuffle__regime-current-only` | 0.8569 | 0.0361 | 0.0046 | 0.0078 | 0.0005 | Best IC and top-20 Sharpe among reported dynamic-graph table rows. Strong baseline. |
| 3 | `dynamic-threshold-6mo-leadlag-shuffle__regime-with-forward-context` | 0.7565 | 0.0364 | 0.0027 | 0.0067 | -0.0008 | Highest listed avg IC, but weaker portfolio CI. |
| 4 | `dynamic-threshold-6mo-shuffle__regime-with-forward-context` | 0.7249 | 0.0350 | 0.0035 | 0.0080 | 0.0007 | Solid raw 6-month dynamic threshold result. |
| 5 | `dynamic-threshold-6mo-snapshot-age-shuffle__regime-with-forward-context` | 0.6554 | 0.0357 | 0.0055 | 0.0070 | -0.0003 | Strong IC lower bound, but no portfolio CI support. |

### Main Effects

Static threshold remains a serious benchmark:

- `static_threshold_shuffle`: decision score 0.8569, avg IC 0.0361, top-20 return 0.0078, top-20 CI lower 0.0005, top-20 Newey-West Sharpe 2.0487.
- `graph_method` as a family averaged much lower: mean decision score -0.0612 across 14 successful graph-method rows, avg IC 0.0337, top-20 return 0.0063, top-20 CI lower -0.0008.

Dynamic graph settings were highly sensitive:

- The best dynamic setting was `dynamic_threshold_6mo_lookback504_shuffle`: decision score 0.7827 on average across two regime variants, avg IC 0.0357, top-20 return 0.0075, top-20 CI lower 0.0004.
- Plain `dynamic_threshold_6mo_shuffle` was positive but weaker: decision score 0.4339, avg IC 0.0347, top-20 return 0.0075, top-20 CI lower 0.0002.
- `dynamic_threshold_12mo_shuffle` was mediocre: decision score 0.2184, avg IC 0.0337, top-20 CI lower -0.0003.
- Lead-lag edge features helped decision score and IC versus the broad dynamic average, but the top-20 CI lower bound was still negative.
- Snapshot-age edge features did not add enough to justify promotion.
- Top-K selection underperformed badly. `top_k=20` had mean decision score -1.3704, avg IC 0.0314, and top-20 CI lower -0.0024. This was materially worse than threshold graphs.
- `top_k_metric=abs_corr` was especially poor, with decision score -1.2019 and top-20 CI lower -0.0023.

Regime context had only a modest dynamic-graph effect:

- `regime_with_forward_context`: median decision score 0.6554, median avg IC 0.0350, median top-20 return 0.0070.
- `regime_current_only`: median decision score 0.0678, median avg IC 0.0341, median top-20 return 0.0067.

The forward-context version appears helpful in the dynamic graph notebook, especially by median decision score, but the mean difference is small because graph-method failures and weak top-K variants dominate parts of the distribution.

### Failures

Five dynamic-graph notebook rows failed or produced no usable metrics:

- Static threshold shuffled with forward context.
- Static threshold sequential, current-only and forward-context.
- Dynamic threshold 6-month sequential, current-only and forward-context.

This points to the sequential/no-shuffle path as a reliability risk, not merely a weak-performing setup. It should be debugged separately before interpreting sequential controls as true negative evidence.

### Dynamic Graph Recommendation

Keep the static threshold shuffled graph as the main baseline. Promote only this dynamic graph candidate for the next confirmation run:

`graph.update_frequency_months=6`, `graph.corr_lookback_days=504`, `graph.top_k=0`, `graph.top_k_metric=corr`, `graph.use_multi_feature_edges=true`, no lead-lag, no snapshot-age, shuffled training, with forward-regime context.

Do not promote top-K graph variants yet. The notebook evidence says threshold graphs are substantially safer than top-K for this setup.

## Training / Feature-Factor Notebook

### Best Individual Runs

| Rank | Run | Decision score | Avg IC | IC CI lower | Top-20 return | Top-20 CI lower | Notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `pure-ic__rank-label-5d__regime-with-forward-context` | 0.9277 | 0.0340 | -0.0030 | 0.5376 | 0.5054 | Best decision score, but rank-label top-k metrics are on a different scale and need audit. |
| 2 | `pure-ic__returns-label-5d__regime-current-only` | 0.8635 | 0.0424 | 0.0087 | 0.0084 | 0.0009 | Best raw-return-label IC result. Strongest deployable-looking candidate. |
| 3 | `pure-ic__returns-label-5d__regime-with-forward-context` | 0.8098 | 0.0410 | 0.0071 | 0.0083 | 0.0011 | Nearly as strong as current-only; slightly better top-20 lower CI. |
| 4 | `pure-ic__rank-label-5d__regime-current-only` | 0.3027 | 0.0295 | -0.0096 | 0.5217 | 0.4878 | Rank-label result weaker than forward-context version. |
| 5 | `selection-val-loss__combined-alpha-050__returns-label-5d__regime-with-forward-context` | 0.1794 | 0.0375 | 0.0041 | 0.0086 | 0.0011 | Lower decision score but stable raw-return portfolio metrics. |

### Objective and Loss Effects

Pure IC loss was the clear winner:

- `pure_ic`: decision score 0.7259, avg IC 0.0367, IC CI lower 0.0008.
- `combined`: decision score -0.0376, avg IC 0.0313, IC CI lower -0.0043.
- `mse`: decision score -1.3783, avg IC 0.0224, IC CI lower -0.0120.

Among combined loss alphas:

- `combined_alpha_075` had the best decision score among combined settings: 0.0073.
- `combined_alpha_025` had the best top-20 return mean among combined alphas: 0.2677, though this aggregate mixes label scales.
- `combined_alpha_050` was not clearly superior in this notebook.

The loss result is directionally strong: optimize directly for IC if the goal is cross-sectional ranking. MSE alone should not be used as the primary objective for this model family.

### Label Effects

Raw return labels and rank labels are telling different stories:

- `returns_label_5d`: avg IC 0.0347, IC CI lower 0.0013, top-20 return 0.0075, top-20 CI lower 0.0004.
- `rank_label_5d`: avg IC 0.0275, IC CI lower -0.0107, top-20 "return" 0.5257, top-20 CI lower 0.4953.

The rank-label portfolio metrics are not numerically comparable to raw-return portfolio metrics. The scale mismatch is too large to interpret as true raw portfolio return. Rank labels may still be useful for training, but the evaluation code path should be checked before using rank-label top-k returns in decision scoring.

### Selection Metric Effects

Validation-loss checkpoint selection looked more stable than validation-IC selection in the notebook:

- `val_loss`: decision score 0.0227, avg IC 0.0365, IC CI lower 0.0034, top-20 return 0.0083, top-20 CI lower 0.0008.
- `val_ic`: decision score -0.0079, avg IC 0.0326, IC CI lower -0.0024, median top-20 return 0.0083, top-20 CI lower 0.0010.

This does not mean `val_loss` is universally better, because the selection-control rows are narrower than the full primary grid. But it is strong enough to justify a controlled confirmation of `selection_metric=val_loss` for the raw-return pure-IC and combined-loss candidates.

### Edge Dropout Effects

Edge dropout was not monotonic:

- `drop_edge_p=0.2`: decision score 0.0646, avg IC 0.0368, top-20 return 0.0081.
- `drop_edge_p=0.05`: decision score 0.0465, avg IC 0.0367, top-20 return 0.0080.
- `drop_edge_p=0.0`: decision score -0.1007, avg IC 0.0361, top-20 return 0.0082.
- `drop_edge_p=0.1`: many more rows, mixed scales, lower mean decision score.

Small-to-moderate edge dropout still looks useful. The p=0.2 result is promising, but it only has two rows; do not promote it without a replication run.

### Regime Feature Effects

In the training-factor notebook, forward regime context was not a decisive win:

- `regime_current_only`: decision score -0.0360, avg IC 0.0316, top-20 return 0.2361.
- `regime_with_forward_context`: decision score 0.0023, avg IC 0.0314, top-20 return 0.2395.

The forward-context version slightly improved decision score and top-20 aggregates, but not average IC. The best raw-return pure-IC run used current-only regime context, while the second-best raw-return pure-IC run used forward context. Treat regime context as a secondary tuning dimension, not the headline driver.

### Failures

The training-factor notebook had at least three failed top-table rows:

- `pure-mse__rank-label-5d__regime-current-only`
- `pure-mse__rank-label-5d__regime-with-forward-context`
- `drop-edge-0p0__combined-alpha-050__returns-label-5d__regime-current-only`

The first two reinforce that pure MSE on rank labels is not a usable path. The third is more likely a control-run reliability issue, since the p=0 aggregate did report metrics elsewhere.

## Cross-Notebook Interpretation

The strongest common configuration shape is:

- Shuffled training rather than sequential controls.
- Threshold correlation graph rather than top-K graph.
- Multi-feature correlation edges enabled.
- Pure IC objective for rank-ordering strength.
- Raw 5-day return labels for deployable portfolio metrics.
- Static threshold graph as a robust baseline; dynamic 6-month graph only when using a longer 504-day lookback.

The dynamic graph notebook says "dynamic can work if smoothed with a longer correlation window." The training-factor notebook says "objective choice matters more than graph schedule." Put together, the next high-value experiment is not a broad graph search. It is a small confirmation grid combining the best training objective with the two credible graph choices.

## Recommended Next Experiment Grid

Run a focused confirmation grid with repeated seeds:

1. Static threshold shuffled graph, raw 5-day returns, pure IC loss, current-only regime.
2. Static threshold shuffled graph, raw 5-day returns, pure IC loss, forward-context regime.
3. Dynamic threshold 6-month graph, 504-day lookback, raw 5-day returns, pure IC loss, forward-context regime.
4. Dynamic threshold 6-month graph, 504-day lookback, raw 5-day returns, pure IC loss, current-only regime.
5. Static threshold shuffled graph, raw 5-day returns, combined loss alpha 0.75, `selection_metric=val_loss`.
6. Dynamic 6-month 504-day graph, raw 5-day returns, combined loss alpha 0.75, `selection_metric=val_loss`.

For each run:

- Use at least 3 seeds or repeated 20-model ensembles.
- Keep `graph.top_k=0`.
- Keep `graph.use_multi_feature_edges=true`.
- Test `graph.drop_edge_p` in `{0.05, 0.1, 0.2}` only after the primary comparison is stable.
- Report both raw IC and raw-return top-k portfolio metrics.
- Exclude rank-label top-k return metrics from decision scoring until the evaluation-scale issue is resolved.

## Decision

Near-term default to beat:

`static_threshold_shuffle + pure_ic + returns_label_5d + regime_current_only`

Best dynamic challenger:

`dynamic_threshold_6mo_lookback504_shuffle + pure_ic + returns_label_5d`

Do not advance:

- `top_k=20` graph variants.
- `top_k_metric=abs_corr`.
- Pure MSE rank-label training.
- Sequential/no-shuffle controls until the failures are debugged.

## Source Artifacts

- [Dynamic graph summary report](https://drive.google.com/file/d/15gNobc0OAwDlh0ydMeogHEN5mQ5gira0/view?usp=drivesdk)
- [Dynamic graph decision table](https://drive.google.com/file/d/13i-o60kypTb3nq11TGnWPSOEPgFyU3tE/view?usp=drivesdk)
- [Training factor summary report](https://drive.google.com/file/d/1CkkoEEl9JKpBzTvf55wcK12DTBvF-1di/view?usp=drivesdk)
- [Training factor decision table](https://drive.google.com/file/d/1jFs6S8s3UcLhiUh4lzQ5GPzWilV6JQXd/view?usp=drivesdk)
