# MCI-GRU Architecture Review (2026-04)

> Grand-overview evaluation of the MCI-GRU codebase against current ML best
> practices, with a focus on model design, information transfer between
> sub-systems, training dynamics, and MLOps hygiene. Sister document to
> `docs/ARCHITECTURE.md` (which describes **what is**); this document is a
> reasoned critique plus prioritised upgrade roadmap.

Scope audited: `mci_gru/` package, `run_experiment.py`, `configs/`, `paper_trade/`,
`tests/`, and `docs/ARCHITECTURE.md` / `AGENTS.md` / `CLAUDE.md`.

## Implementation status (April 2026)

**Phase 1 (“free lunch”) from §9 is shipped.** The narrative sections below (§2–§7) were written as a *pre-change* audit; where they contradict this box, **this box wins**.

- **Optimizer / schedule / AMP:** `AdamW`, linear warmup + cosine LR (`TrainingConfig.lr_scheduler`, `warmup_steps`), `torch.amp` + `GradScaler` when `use_amp` and CUDA (no gradient accumulation yet).
- **Selection / loss:** `selection_metric` (default **`val_ic`**), combined MSE+IC loss default (`loss_type=combined`); `paper_faithful` pins paper-style MSE / scalar edges / MLflow off as needed.
- **Graph:** `use_multi_feature_edges=true` by default; `(E, 4)` edge attributes.
- **Splits / leakage:** `ExperimentConfig` enforces **> `label_t`** calendar-day gaps between train→val and val→test unless `data.skip_embargo_check=true`. `pipeline._build_tensors` aligns `stock_features_*` to label dates so embargo **gaps** do not misalign tensors vs graph inputs.
- **Speed:** `generate_time_series_features` is **vectorised** (no `iterrows` hotspot).
- **Repro / MLOps:** Per-ensemble-member seeds `seed + model_id`; `run_metadata.json` includes **`data_file_sha256`** (+ size, mtime) when `data.filename` exists; **MLflow `tracking.enabled=true`** by default.

Remaining large items (still accurate in the sections below): **regularisation in the trunk**, **fused / vectorised temporal encoder**, **stronger cross-stream fusion**, **walk-forward training**, **survivorship / PIT universe**, **Sharpe on overlapping `label_t` returns**, **shared portfolio utilities**, **locked dependencies**.

## TL;DR

The system is a clean, well-documented, four-stream ensemble over a dynamic
correlation graph. The *plumbing* (Hydra configs, typed dataclasses,
GraphSchedule, paper-trade isolation, no-lookahead discipline) is genuinely
good and above the median for research code. The *learning stack itself*
(optimizer, scheduler, normalisation layers, information flow between
streams, label design, temporal encoder implementation) lags modern best
practice and is where almost all the cheap alpha lives.

Highest remaining ROI (after Phase 1 — see **Implementation status** above), in rough order:

1. **Add LayerNorm + Dropout + residuals** to the four-stream trunk. Currently none of the three exist in the model.
2. **Vectorise or replace `AttentionResetGRUCell`** — the Python `for t in range(seq_len)` loop forfeits CuDNN fusion and dominates epoch time at larger `his_t`.
3. **Add explicit cross-stream information transfer** — A1 (temporal) and A2 (graph) still interact only via concat+self-attn; bilinear / cross-attention fusion is a well-known uplift.
4. **Walk-forward retraining** instead of a single fixed split — required to claim robustness about forward windows; embargo validation alone does not replace refitting.
5. **Gradient accumulation** (not yet) if you want larger effective batch on IC without blowing VRAM.

The rest of the document justifies the original audit findings; **§9** tracks what has already landed vs what is still open.

---

## 1. What the system does well

- **Separation of concerns.** `data/`, `features/`, `graph/`, `models/`, `training/` boundaries are respected. `pipeline.prepare_data` is the single orchestrator. `paper_trade/` is enforced as frozen-checkpoint-only via `.cursor/rules/paper-trade-isolation.mdc`.
- **Typed config surface.** `ExperimentConfig` + Hydra YAML + `__post_init__` validation catches bad runs before training starts.
- **No-lookahead discipline.**
  - `_compute_norm_stats` uses `train_end` only.
  - `GraphSchedule` snapshots use data *strictly before* their valid-from date.
  - `compute_labels` uses `close[t+label_t]/close[t+1] - 1` (first forward day is *after* prediction date).
  - `apply_rank_labels` uses same-day cross-section only.
- **Dynamic graph without `batch_size=1`.** `GraphSchedule.get_graph_for_date` + `combined_collate_fn` resolve edges per-sample via `bisect`, so any batch size works.
- **Ensemble averaging.** `train_multiple_models` with `num_models=10` is a simple and robust variance reducer.
- **MLflow integration** (**on by default**; disable with `tracking.enabled=false`) with parent/child runs, per-epoch metrics, checkpoint and prediction artefacts.
- **Feature registry plugin pattern** (`momentum`, `volatility`, `credit`, `regime`) with typed config and pre-defined `FEATURE_SETS`.
- **Backtest realism.** Transaction-cost accounting (`bid_ask=5bps`, `slippage=5bps`), turnover stats, gross/net ARR, rank-drop exit gate (`min_rank_drop=30`).
- **Tests cover invariants.** Lookahead, regime data contract, momentum blend modes, dynamic graph wiring, output-management — not just "does it run".

These are the reasons the system deserves an upgrade rather than a rewrite.

---

## 2. Model architecture — issues and opportunities

File: `mci_gru/models/mci_gru.py`.

### 2.1 Missing modern regularisation primitives

`rg -n 'LayerNorm|BatchNorm|Dropout' mci_gru/models` returns **zero** hits.
For a 4-stream model that processes noisy, fat-tailed financial data with a
stacked attention + GAT trunk, this is surprising.

Recommended minimum:
- `nn.LayerNorm(self.align_dim)` on the output of `proj_temporal` and `proj_cross`.
- `nn.LayerNorm(concat_size)` before the final GAT.
- `nn.Dropout(p ≈ 0.1-0.2)` on the concat `Z` before `self_attention`, and between the two GAT layers in `GATBlock`.
- Consider residual projection: `A1 = A1 + proj_temporal(A1_raw)` (requires matched dims, which you already have via `align_dim`).

Expected effect: smaller generalisation gap, less seed-to-seed variance.

### 2.2 `AttentionResetGRUCell` is Python-loop, not fused

```71:107:mci_gru/models/mci_gru.py
class ImprovedGRU(nn.Module):
    ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
        for t in range(seq_len):
            h = layer(layer_input[:, :, t, :], h)
            outputs.append(h)
```

This forfeits CuDNN kernels and Python-level dispatch per timestep. At
`his_t=10` it's tolerable; at `his_t=60` (a sweep recommended below) it is
the dominant cost.

Options (from least to most invasive):

1. **Quick win.** Replace `ImprovedGRU` with `nn.GRU(..., num_layers=2, batch_first=True)` on the `(batch*stocks, seq, features)` tensor and post-hoc add a single scaled-dot-product attention head on the final hidden state. Empirically keeps the spirit of the paper's reset-gate-as-attention idea at a fraction of the cost.
2. **Vectorised custom cell.** Unroll the cell into a batched tensor op by pre-computing `W_k(x), W_v(x), W_h(x)` for all `t`, then doing a streaming reduction. Retains exact math.
3. **Transformer encoder.** Swap `ImprovedGRU` for `nn.TransformerEncoder` (2 layers, 4 heads). Causal mask keeps no-lookahead. On small sequences (`his_t ≤ 60`) this is competitive with GRUs and easier to scale.

The "reset-gate attention" is really `sigmoid(q·k / √d) * v` — a single-key
attention. Calling it attention is a stretch; your own code comment
acknowledges this (`# The score is a scalar ... softmax would always yield 1.0`).

### 2.3 Cross-stream information transfer is weak

The four streams share information only via `torch.cat([A1, A2, B1, B2])` and
a single self-attention over the *set* of four group vectors per stock. This
is roughly a "late fusion" pattern.

Concrete improvements:

- **Bilinear / cross-attention fusion.** Let the graph stream attend over the
  temporal stream: `A2' = CrossAttn(query=A2, kv=A1)`. Today the GAT only
  sees the most recent day's features for each node; letting it attend over
  the full temporal representation is a strictly richer signal with minimal
  compute cost.
- **Group-type embedding in `SelfAttention`.** The 4-group sequence
  `[A1, A2, B1, B2]` is treated as a permutation-invariant set. Add a learned
  `nn.Embedding(4, concat_size)` type-embed and add it before self-attention
  so the model can *tell which stream a vector came from*.
- **Regime/VIX as conditioning, not as per-stock feature.** Right now global
  scalars (VIX, credit, regime_global_score) are broadcast to every
  `(stock, day)` row and then z-scored along with stock features. This
  inflates feature count and forces the model to re-discover that these
  values are identical across stocks. Better: use FiLM on the latent states
  `R1/R2` conditioned on a market token. That's 1 embedding per batch, not
  `num_stocks` copies.

### 2.4 `MarketLatentStateLearner` re-implements MultiheadAttention

The class is structurally `nn.MultiheadAttention` with learned queries. A
one-liner swap (`nn.MultiheadAttention(embed_dim=align_dim,
num_heads=cross_attn_heads, batch_first=True)`) gives you:

- Fused kernels (faster).
- Native flash-attention when available.
- Automatic weight init (currently you rely on default `nn.Linear` init on
  8 projections).
- Automatic broadcasting semantics.

Keep the learned `R1, R2` as the `key/value` tensor passed in.

### 2.5 `SelfAttention` has no masking or scaling factor bug

```236:245:mci_gru/models/mci_gru.py
self.scale = embed_dim ** -0.5
...
attn_weights = F.softmax(
    torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1
)
```

Two small issues:

- `self.scale = embed_dim ** -0.5` — but the correct scale for multi-head is
  `head_dim ** -0.5`. `SelfAttention` here is single-headed on the full
  `concat_size` embedding, so `embed_dim` and `head_dim` coincide and the
  code is correct — but this will silently break if anyone makes it
  multi-head. Worth documenting.
- Applied across the stock dimension (N ≈ 500). At larger universes this is
  `O(N²)` per batch element. Consider `FlashAttention` or a chunked variant
  above N≈2k.

### 2.6 Final activation clips predictions asymmetrically

```409|out = self.output_act(out)```

Default activation is `elu`, which floors scores at `-1` and is
asymmetrically saturating on the negative side. For a **ranking** model
that is later sorted cross-sectionally, any monotonic transform is a
no-op; for a **regression** model (MSE loss against forward returns), the
negative-side saturation is a real bias.

Recommendation: drop the final activation entirely. If you want a sigmoid-
for-rank-labels path, add it explicitly when `label_type='rank'`.

### 2.7 Edge attributes (implemented April 2026)

**Resolved:** `use_multi_feature_edges` now defaults to **`true`** in `GraphConfig` / `configs/config.yaml`, emitting `(E, 4)` `[corr, |corr|, corr², rank_pct]`. `configs/experiment/paper_faithful.yaml` pins **`false`** for paper-aligned runs. The `(E,)` scalar path remains for ablations.

---

## 3. Training loop — issues and opportunities

File: `mci_gru/training/trainer.py`.

### 3.1 Optimizer and schedule (updated April 2026)

**Shipped:** `AdamW`, **linear warmup + cosine** LR via `SequentialLR` stepped **per optimizer step** (`TrainingConfig.lr_scheduler`, `warmup_steps`), and **CUDA AMP** (`use_amp`, `autocast` + `GradScaler` with correct `unscale_` before grad clip).

**Still open:** gradient accumulation for larger effective batch; `ReduceLROnPlateau` as an alternative schedule; optional second-pass cosine tied to epochs only (current design is step-based for short epochs).

### 3.2 Early stopping vs ranking objective (updated April 2026)

**Shipped:** validation **IC (Spearman)** is computed each epoch; **`selection_metric`** chooses best checkpoint (`val_ic` vs `val_loss`). Default training still minimises the configured **loss** (often `combined`); checkpoint selection can prioritise IC.

**Residual risk:** if you force `selection_metric=val_loss` with `loss_type=mse`, you are back to the old behaviour — that is intentional for ablations / `paper_faithful`.

### 3.3 Ensemble diversity

**Shipped:** each ensemble member calls **`set_seed(config.seed + model_id)`** before `train()`, so initial weights and Python/NumPy/Torch RNGs differ by construction.

**Still open:** hyperparameter / architecture jitter and real **deep-ensemble** diversity (needs Phase 2 regularisation + DropEdge etc.). Averaging remains a variance reducer, not a full Bayesian model average.

### 3.4 Walk-forward vs single split

**Still the main structural gap:** training is still a **single** train/val/test configuration in typical runs. Production-style deployment wants rolling or expanding windows and concatenated OOS metrics.

**Embargo (April 2026):** `ExperimentConfig.__post_init__` rejects configs whose **calendar gaps** between train/val and val/test are **not strictly greater than `model.label_t`**, unless `data.skip_embargo_check=true`. **YAML date ranges must be shifted** to satisfy this (e.g. val starts at least `label_t+1` calendar days after `train_end`). That removes the specific “last train labels peek into val calendar” footgun when dates are configured correctly; it does **not** replace walk-forward refits.

### 3.5 Loss design (updated April 2026)

- **Default is `combined`** with `ic_loss_alpha: 0.5` in `configs/config.yaml`. `paper_faithful` pins `loss_type: mse` for replication-style runs.
- `ICLoss` averages per-day IC — still correct cross-sectionally.
- `CombinedMSEICLoss` still uses a fixed `eps`; monitor scale if you change normalisation.

### 3.6 No HPO / sweep framework

There is no Optuna, Ray Tune, or MLflow-driven sweep. `configs/experiment/`
has a handful of presets (`lookback_sweep.yaml`, `paper_faithful.yaml`)
that must be invoked manually via Hydra `--multirun`. Given that each run
takes 10 × 100 = 1000 model-epochs, an automated sweep over `his_t`,
`learning_rate`, `judge_value`, and `num_hidden_states` is the single
biggest alpha lever available.

Minimal recipe:

```bash
pip install optuna
python run_experiment.py --multirun \
    model.his_t=10,20,40 \
    training.learning_rate=3e-5,5e-5,1e-4 \
    graph.top_k=0,10,20 \
    hydra/sweeper=optuna
```

With MLflow **enabled by default**, each trial’s metrics are logged automatically (override with `tracking.enabled=false` if you want a silent sweep).

---

## 4. Data and feature pipeline — issues and opportunities

Files: `mci_gru/pipeline.py`, `mci_gru/data/data_manager.py`,
`mci_gru/data/preprocessing.py`, `mci_gru/features/*`.

### 4.1 Survivorship bias is the #1 data hazard

```394:427:mci_gru/data/data_manager.py
def filter_complete_stocks(self, df):
    ...
    kdcode_list = kdcode_counts[kdcode_counts == len(period_dates)].index.tolist()
```

This keeps only stocks with **every single day present** across the entire
train+val+test period. Effects:

- Dropped stocks are exactly the ones that failed, were acquired, delisted,
  or changed ticker — the *informative* tail of return distributions.
- Test-period metrics are upward-biased by the amount of negative tail
  survivorship.
- A Russell-1000 universe effectively collapses to ~700 names with this
  filter.

Mitigations (in increasing effort):
- **Document.** Add a warning in `run_metadata.json` noting survivorship universe size vs. nominal universe.
- **Relax to per-split completeness.** A stock only needs complete data for the split(s) in which it's used.
- **Point-in-time universes.** Use an as-of snapshot of the universe at each rebalance date, with proper delisting return treatment (Centre for Research in Security Prices convention).

### 4.2 Time-series feature tensor build (updated April 2026)

**Resolved:** `generate_time_series_features` was rewritten to a **vectorised** path (pivot / unstack + `reindex` on the date × stock grid). The old `iterrows()` loop is gone; equivalence is covered by tests. **§4.3+** below still applies to NaN handling in `pipeline.py`.

### 4.3 NaN handling is lossy

```200:207:mci_gru/pipeline.py
for _, df_day in df.groupby("dt"):
    df_day = df_day.copy()
    for col in feature_cols:
        if col in df_day.columns:
            df_day[col] = df_day[col].fillna(df_day[col].mean())
    df_day = df_day.fillna(0.0)
    parts.append(df_day)
```

For a missing momentum value on stock X, you substitute the
cross-sectional mean — which is always close to 0 for z-scored momentum.
For features that can be strongly directional cross-sectionally (e.g.
`regime_similar_subsequent_return_1m`), mean-imputation injects
cross-stock contamination.

Better pattern:
- Forward-fill within `kdcode` first (same-stock history is the right prior).
- Then cross-sectional mean as a fallback.
- Add a `{feature}_isnan` indicator column so the model can learn to treat
  imputed rows differently.

### 4.4 Normalisation scheme

Currently: `3-sigma clip → z-score` using train-period mean/std. This is
fine for OHLCV but poor for fat-tailed return-like features (momentum,
credit spreads, VIX changes). A rank-gauss (Gaussian-quantile transform on
per-stock or per-day ranks) transform is the de-facto standard for
quantitative cross-sectional models because it is invariant to monotone
re-scalings and robust to outliers without losing ordering.

### 4.5 Feature engineering is 100% pandas

All feature modules (`momentum.py`, `volatility.py`, `credit.py`,
`regime.py`) operate on pandas DataFrames. For Russell 3000 or daily-
refreshed MSCI World universes, `polars` or `pyarrow`-backed pandas would
give 5–20× speed-up at no model-side cost.

### 4.6 Graph stream only sees the latest day

```142:148:mci_gru/pipeline.py
x_graph_train = generate_graph_features(
    train_df, kdcode_list, feature_cols, train_dates[his_t:]
)
```

`generate_graph_features` returns a `(num_dates, num_stocks, num_features)`
tensor — a single snapshot per date. The GAT (stream A2) therefore only
sees a one-day cross-section per prediction, while the temporal encoder
(stream A1) sees `his_t` days. This asymmetry is by design in the paper,
but it means most of the temporal signal is forced through one 10-feature
vector per stock at the graph layer.

Two upgrades:
- Pass the GAT a *pooled* temporal summary (e.g. last 5-day mean of features) instead of just the last day.
- Stream a small temporal window through the GAT using `TemporalGATConv` (each node has a seq, message passing happens at each timestep, then reduce).

---

## 5. Graph module — issues and opportunities

File: `mci_gru/graph/builder.py`.

### 5.1 Only Pearson correlation on returns

The plan (`.cursor/plans/graph_signal_upgrades_c28cf640.plan.md`) already
acknowledges this and lists levers 1b (signed two-relation / RGATConv),
2a (lead-lag edges), 3a (graph-aware temporal encoder), 4c (rate-of-change
edge feature), and a 5th edge column (snapshot_age_days) as **pending**.

These are real uplifts and the plan is correct. Priority order in my view:

1. **Lead-lag edges** (`2a`). Cross-correlation at lag=1,5 between pairs. Recovers ETF-rebalance and mean-reversion patterns that same-day Pearson misses.
2. **Sector / industry one-hot edges** as a separate relation (RGATConv with 2 relations: correlation + sector).
3. **`snapshot_age_days`** as a 5th edge feature column — lets the model down-weight stale graphs during long holding periods.
4. **DropEdge regularisation** at train time (`rg -n 'dropout_edge' mci_gru` → no hits).

### 5.2 Graph cadence is too slow by default

`update_frequency_months=0` (static) in `configs/config.yaml`, and the
typical "dynamic" configs use 6 months. With `corr_lookback_days=252`
(1 year), a 6-month cadence means the graph is up to 18 months stale at
the end of its validity window. Monthly cadence with a 63-day lookback
is more reactive and closer to standard industry practice; the audit in
the plan file (`correlation_dynamic_topk20_*.yaml`) is a good starting
point but should be promoted to the default dynamic preset.

### 5.3 Graph snapshots are not cached across runs

`precompute_snapshots` runs inside `prepare_data()` every time. For a
hyperparameter sweep over `model.*` and `training.*` (which do not touch
the graph), this is wasted work. A content-addressable cache keyed on
`(kdcode_list_hash, corr_lookback_days, judge_value/top_k, update_dates)`
stored under `data/cache/graphs/` would collapse sweep wall-clock time
significantly.

---

## 6. Evaluation and metrics

File: `mci_gru/training/metrics.py`.

### 6.1 Sharpe annualisation is wrong for `label_t > 1`

```74:76:mci_gru/training/metrics.py
metrics["sharpe_ratio"] = float(
    np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
)
```

This is correct for daily returns but wrong for `label_t=5` forward
returns computed on overlapping daily windows: the returns are
auto-correlated (overlapping), so `std` is deflated and Sharpe is
inflated. Options:

- Use non-overlapping evaluation windows (take every 5th day).
- Newey-West correct the variance with `q = label_t - 1` lags.
- Annualise by `sqrt(252 / label_t)` and explicitly document that the
  resulting number is a lower bound on overlapping-Sharpe.

### 6.2 No uncertainty quantification on metrics

Single-point metrics per run. Bootstrap the per-day IC / portfolio return
series (block bootstrap with block size = `label_t`) and report 95% CIs
in the training summary — makes "did this change actually help?" a
testable question rather than a vibe.

### 6.3 Portfolio logic is simplistic

`compute_metrics` top-k = equal-weighted mean of the top 50 predictions.
There is no:
- Vol targeting / risk budgeting.
- Sector or factor neutralisation.
- Short side (only long top-k).

This is acceptable for the training loop's internal metric but is the
reason paper-trade and backtest have to re-implement everything. A shared
`portfolio.py` utility used both inside training metrics and in the
backtest would remove ~300 LOC of duplication between
`tests/backtest_sp500.py` and `paper_trade/scripts/portfolio.py`.

---

## 7. MLOps, reproducibility, and repo hygiene

- **MLflow defaults on.** `tracking.enabled=true` in `configs/config.yaml`; set `tracking.enabled=false` for local smoke runs.
- **No dependency lockfile.** `pyproject.toml` + `requirements.txt` without pinned hashes means CI / reproducibility is fragile. Add `uv.lock` or `poetry.lock`.
- **Data fingerprint (April 2026).** `run_metadata.json` now records **`data_file_sha256`**, size, and mtime for `data.filename` when the file exists (graceful skip if missing).
- **No CI config inspected.** `.github/` exists; a smoke-run job on every PR (`python run_experiment.py training.num_epochs=2 training.num_models=1`) should be near-free and would catch the kind of collate-tuple-shape regression that `AGENTS.md` invariant #3 warns about.
- **Repo-root clutter.** `656_MTP_2026.pdf`, `Seed_test (1).ipynb` (99 KB), `lseg_env/`, and `_uncertain/` should live outside the package import path.
- **Determinism.** `set_seed` covers `random`, `numpy`, `torch`, `torch.cuda`, but not `torch.backends.cudnn.deterministic` / `benchmark = False`. For exact reproducibility on CUDA, set both.
- **`tests/` mixes unit tests and scripts.** `backtest_sp500.py` (133 KB) is not a pytest test; move to `scripts/` so `pytest tests/` runs fast.

---

## 8. Information-transfer opportunities (summary)

Because the user specifically asked about information transfers, the
highest-leverage architecture changes, ranked:

| # | Change | Where | Cost | Upside |
|---|--------|-------|------|--------|
| 1 | Cross-attention `A2' = CrossAttn(Q=A2, KV=A1_full_seq)` | `StockPredictionModel.forward` | Low | High — graph stream gains temporal context |
| 2 | FiLM-condition `R1, R2` on a market/regime token | `MarketLatentStateLearner` | Low | Medium — latent states specialise by regime |
| 3 | Group-type embedding in `SelfAttention` | `SelfAttention` | Trivial | Medium — model can distinguish streams |
| 4 | Multi-relation graph (correlation + sector + lead-lag) via `RGATConv` | `graph/builder.py` + `GATBlock` | High | High — modality-diverse edges |
| 5 | Multi-horizon prediction head (1d/5d/20d shared trunk) | `StockPredictionModel` + loss | Medium | Medium — regularises encoder |
| 6 | Global macro tokens as conditioning, not per-stock features | `FeatureEngineer` + model | Medium | Medium — parameter-efficient |
| 7 | Pool last `k` days into graph stream instead of 1 day | `generate_graph_features` | Low | Medium — evens up A1/A2 depth |
| 8 | Cross-sectional self-attention with relative-rank bias | `SelfAttention` | Medium | Low-Medium — rank-aware prior |

---

## 9. Concrete, ordered upgrade roadmap

### Phase 1 — "free lunch" (≤ 1 week, no model surgery)

- [x] `Adam → AdamW` + `CosineAnnealingLR` + 1000-step linear warmup.
- [x] Enable `torch.amp.autocast` + `GradScaler` on CUDA.
- [x] Early-stop on validation IC, not val MSE.
- [x] `loss_type: combined` with `ic_loss_alpha: 0.5` as default.
- [x] `use_multi_feature_edges: true` as default.
- [x] Seed each ensemble member with `config.seed + model_id`.
- [x] Embargo `label_t` days between train/val and val/test in `ExperimentConfig.__post_init__` (`data.skip_embargo_check` opt-out).
- [x] Vectorise `generate_time_series_features` (kill `iterrows`).
- [x] Record `sha256` (+ size, mtime) of `data.filename` in `run_metadata.json`.
- [x] `tracking.enabled: true` by default.

Expected combined effect: 15–40% faster training, better IC, fewer silent data-version footguns.

### Phase 2 — architecture uplift (1–3 weeks)

- [ ] `LayerNorm` + `Dropout(0.1)` in the trunk (`proj_temporal`, `proj_cross`, before final GAT).
- [ ] Replace custom `MarketLatentStateLearner` with `nn.MultiheadAttention`.
- [ ] Replace `AttentionResetGRUCell` Python loop with either (a) native `nn.GRU` + a single post-hoc attention head or (b) a Transformer encoder with causal mask.
- [ ] Add cross-attention between A1 and A2 (`A2' = CrossAttn(Q=A2, KV=A1_seq)`).
- [ ] Add DropEdge at train time for the GAT.
- [ ] Ship walk-forward retraining in `run_experiment.py` (new `training.walkforward` config block).

### Phase 3 — data and graph modernisation (2–4 weeks)

- [ ] Lead-lag edges (lever 2a from the graph plan) and `snapshot_age_days` edge feature.
- [ ] Multi-relation graph via `RGATConv` (correlation + sector).
- [ ] Rank-gauss normaliser alongside the existing 3-sigma z-score.
- [ ] Point-in-time universe (relax survivorship filter to per-split, then to CRSP-style).
- [ ] Polars-backed feature engineering for universes > 1000 stocks.

### Phase 4 — eval and MLOps (parallelisable)

- [ ] Block-bootstrap CIs on per-day IC and top-k return, logged to MLflow.
- [ ] Shared `portfolio.py` used by both `tests/backtest_sp500.py` and `paper_trade/`.
- [ ] Optuna sweep config + example over `his_t`, `learning_rate`, `top_k`, `ic_loss_alpha`.
- [ ] CI smoke-run on PR: 2-epoch, 1-model sanity check that asserts the 7-tuple collate shape.
- [ ] Distribution-shift monitor on live features (KS / PSI vs. train window) surfaced in the nightly `report.py` output.

---

## 10. What NOT to change (at least not yet)

- **Four-stream structure.** Even with the issues above, the `[A1, A2, B1, B2]` concat+self-attn pattern is a defensible inductive bias (temporal × cross-sectional × latent). Don't rip it out; augment it.
- **Typed dataclass config.** Don't move to a looser dict-based config just because "it's simpler". The validation in `__post_init__` catches real bugs.
- **`GraphSchedule` precomputation.** This was hard-won. Do not regress to per-batch rebuilds.
- **`paper_trade/` isolation.** The rule that inference doesn't import `GraphBuilder` is worth keeping forever — it's the reason you can retire training code without breaking production.
- **Ensemble averaging.** Per-member seeds (Phase 1) help; Phase 2 regularisation / DropEdge / jitter are still needed for a “real” deep ensemble.

---

*Reviewer note.* This is a code-level architecture review; it does not
claim to predict P&L. The Phase-1 changes are almost all strict Pareto
improvements (same code paths, better defaults). Phase 2+ should be
gated on A/B backtests against the current `paper_faithful` baseline
using block-bootstrap CIs before being promoted to the default config.
