# Features

Every model input column, the window it is computed over, and when its inputs are
knowable. This is the only feature reference in the repository; the calculations live in
`mci_gru/features/`, and this document describes them as they are written. Where the two
disagree, the code wins ([agents/domain.md](agents/domain.md)).

`FeatureEngineer` (`mci_gru/features/registry.py`) composes the families below in the
order listed. Each family is switched by a `FeatureConfig` field (`mci_gru/config.py`),
selected in Hydra YAML under `configs/features/`, and lives in the module named in its
heading. Column order in the model input follows the same order.

## The timing rule

A row dated `D` for stock `k` is computed from that stock's rows dated `D` or earlier, or
from a market-wide series at `D` or earlier, with two families deliberately lagged further:
volatility targeting uses returns ending at `D − 2`, and the global regime compares a month
only to months that ended before it. Rolling windows are trailing and never centred.
Three related rules live in [ARCHITECTURE.md](ARCHITECTURE.md) and are only named here:
normalization statistics are fit on the training window and nothing later; the label,
`close[D + label_t] / close[D + 1] − 1` per stock, is not a feature and never enters one;
and the correlation graph is fit on trailing returns up to the training cutoff or, when
dynamic, up to each snapshot's valid-from date.

Every timing-sensitive family carries the no-lookahead canary described in
[TESTING_GUIDE.md](TESTING_GUIDE.md): compute twice, mutate a future row, assert earlier
rows are unchanged. `tests/test_momentum_blend_modes.py` and
`tests/test_regime_features.py` are the models; [agents/guide.md](agents/guide.md) lists
the guard surfaces for the no-lookahead invariant.

## Base (always on)

`mci_gru/features/base.py`. `FeatureConfig.base_features`.

| Column | Definition | Window |
| --- | --- | --- |
| `open`, `high`, `low`, `close`, `volume` | as loaded from the panel | `D` |
| `turnover` | `close × volume`, added when the panel does not carry it | `D` |

## Momentum (`include_momentum`, default on)

`mci_gru/features/momentum.py`, after Goulding, Harvey and Mazzoleni, *Momentum Turning
Points*. Daily return is `close / previous close − 1` per stock. A momentum column is the
sum of daily returns over its trailing window and is missing until the window is full.

| Column | Definition | Window |
| --- | --- | --- |
| `slow_momentum` | trailing sum of daily returns | 252 sessions |
| `fast_momentum` | trailing sum of daily returns | 21 sessions |
| `weekly_momentum` | trailing sum of daily returns (`include_weekly_momentum`, default on) | 5 sessions |
| `slow_signal`, `fast_signal`, `weekly_signal` | encoding of the momentum column, below | as the momentum column |
| `momentum_blend` | `(1 − w) × slow_signal + w × fast_signal`, where `w` is the FAST allocation; 0 while either window is not yet full | as above |
| `cycle_bull` | 1 when slow and fast momentum are both `≥ 0` | as above |
| `cycle_correction` | 1 when slow `≥ 0` and fast `< 0` | as above |
| `cycle_bear` | 1 when slow and fast are both `< 0` | as above |

The fourth state, Rebound (slow `< 0`, fast `≥ 0`), has no column: all three indicators
are 0 there, as they are on a row whose windows are not yet full.

**Encoding** (`momentum_encoding`):

- `binary` (default): a signal is `+1` when its momentum is `≥ 0`, `−1` otherwise.
- `continuous`: the momentum columns keep their values (missing filled with 0), and each
  signal is the day's cross-sectional z-score of its momentum column, clipped to
  `[−3, 3]`.
- `buffered`: each signal is a linear map of the day's cross-sectional percentile rank
  onto `[−1, +1]`, set to 0 outside `[momentum_buffer_low, momentum_buffer_high]`
  (defaults 0.1 and 0.9). This encoding adds one column, `trade_signal`: the sign of
  `momentum_blend` where its magnitude is at least 0.2, else 0.

**Blend weight** (`momentum_blend_mode`):

- `static` (default): `w = momentum_blend_fast_weight`, 0.5.
- `dynamic`: the paper's speed-selection estimate of `w` per cycle state, computed for
  each stock from strictly prior observations (`momentum_dynamic_lookback_periods`, 0 for
  expanding history). It falls back to `momentum_dynamic_correction_fast_weight` (0.15)
  and `momentum_dynamic_rebound_fast_weight` (0.70) until `momentum_dynamic_min_history`
  (252) observations and `momentum_dynamic_min_state_observations` (3) per state have
  accumulated. Bull and Bear rows use 0.5.

## Volatility (`include_volatility`)

`mci_gru/features/volatility.py`. Annualised by `√252`.

| Column | Definition | Window |
| --- | --- | --- |
| `volatility_5d` | standard deviation of daily returns, annualised | 5 sessions |
| `volatility_21d` | standard deviation of daily returns, annualised | 21 sessions |
| `vol_ratio` | `volatility_5d / volatility_21d`, clipped to `[0.1, 10]` | 21 sessions |

Missing values are filled from the expanding median of strictly earlier observations, and
with 0.2 when there are none.

## Volatility targeting (`include_volatility_targeting`)

`mci_gru/features/volatility.py`. Harvey-style ex ante inputs: a row dated `D` uses stock
returns ending no later than `D − 2`. Half-lives come from
`volatility_targeting_half_lives` (default `[20, 60, 90]`); the first and last listed
drive the dynamics columns. Column names carry the half-life.

| Column | Definition | Window |
| --- | --- | --- |
| `vol_target_ewm_vol_hl{h}` | exponentially weighted standard deviation of the two-day-lagged daily return, half-life `h`, annualised; filled with `volatility_target_vol` (0.10) until two observations exist | EWM, half-life `h` |
| `vol_target_scale_hl{h}` | `volatility_target_vol / vol_target_ewm_vol_hl{h}`, clipped to `volatility_target_scale_clip` (`[0.25, 4.0]`) | as above |
| `vol_target_vol_change_hl20_hl90` | `vol_target_ewm_vol_hl20 / vol_target_ewm_vol_hl90 − 1` | as above |
| `vol_target_vol_of_vol_hl20` | exponentially weighted standard deviation of the day-over-day change in `vol_target_ewm_vol_hl20`, annualised | EWM, half-life 20 |
| `vol_target_ret21_lag2_x_scale_hl20` | 21-session trailing return, lagged two sessions, times `vol_target_scale_hl20` (`volatility_targeting_interaction_return_window`) | 21 sessions ending `D − 2` |

`volatility_targeting_components` selects the groups: `ewm_vol`, `scale`, `dynamics`
(the change and vol-of-vol columns), `scaled_return`. The clip on the scale column is an
input-stability guard, not a portfolio rule; nothing here changes position sizing.

## VIX (`include_vix`)

`mci_gru/features/volatility.py`. Market-wide, joined to every stock by date. The series
comes from LSEG when `data.source=lseg`, otherwise from `data/raw/market/vix_data.csv`.

| Column | Definition | Window |
| --- | --- | --- |
| `vix` | index level, forward-filled; 20 when no value is available yet | `D` |
| `vix_change` | day-over-day percentage change of the level | `D` |
| `vix_regime` | 1 when the level is above its trailing mean | 10 observations |

## Credit spreads (`include_credit_spread`)

`mci_gru/features/credit.py`. Market-wide, joined by date. ICE BofA option-adjusted
spreads from FRED (`BAMLC0A0CM` investment grade, `BAMLH0A0HYM2` high yield; needs
`FRED_API_KEY`). The loader assigns date `T` the value published for `T − 1` and
forward-fills non-trading days. If the fetch fails the columns are zero-filled so the
feature width stays fixed.

| Column | Definition | Window |
| --- | --- | --- |
| `ig_spread`, `hy_spread` | option-adjusted spread, one-day lagged | `D − 1` |
| `ig_spread_change`, `hy_spread_change` | day-over-day percentage change | `D − 1` |
| `ig_spread_zscore`, `hy_spread_zscore` | z-score against the trailing mean and standard deviation, clipped to `[−3, 3]` | 63 observations |
| `credit_spread_diff` | `hy_spread − ig_spread` | `D − 1` |

## Global regime (`include_global_regime`)

`mci_gru/features/regime.py`. Market-wide, monthly, broadcast to every stock-day. Inputs
and their sources are the regime contract in
[REGIME_DATA_CONTRACT.md](REGIME_DATA_CONTRACT.md): market, yield curve, oil, copper and
stock-bond correlation are required; monetary policy and volatility are optional. The
frozen recipe enables this family with `regime_strict=true` and
`regime_include_subsequent_returns=false`.

Each input is sampled at month end, differenced over `regime_change_months` (12),
z-scored against a trailing `regime_norm_months` (120) window that needs at least
`regime_min_history_months` (24) of history, and clipped to `±regime_clip_z` (3). Month
`T` is then compared, by Euclidean distance over the jointly available inputs, with every
earlier month `i` where `i < T − regime_exclusion_months`: with the default of 1, the
month just before `T` is excluded along with `T` itself. Features are missing until at
least 24 comparable months exist. A stock-day dated `D` receives the features of the latest
month end at or before `D`, then forward-fills, then 0.

| Column | Definition | Window |
| --- | --- | --- |
| `regime_global_score` | mean distance from month `T` to every eligible prior month | all prior months |
| `regime_similarity_q20_mean` | mean distance to the nearest `regime_similarity_quantile` (20%) of prior months | all prior months |
| `regime_dissimilarity_q80_mean` | mean distance to the farthest 20% | all prior months |
| `regime_similarity_spread` | `regime_dissimilarity_q80_mean − regime_similarity_q20_mean` | all prior months |
| `regime_similar_subsequent_return_{h}m` | mean of the market's `h`-month return that followed the nearest 20% of prior months, over the prior months `i` with `i + h ≤ T`, so every return used had completed by `T` (`regime_subsequent_return_horizons`, default `[1, 3]`; `regime_include_subsequent_returns`) | all prior months |
| `regime_subsequent_return_spread_1m` | the 1-month version of the line above, similar minus dissimilar | all prior months |

## Additional families

All in `mci_gru/features/base.py` or `mci_gru/features/volatility.py`; each flag is off
by default and on in `configs/features/full.yaml`.

| Flag | Column | Definition | Window |
| --- | --- | --- | --- |
| `include_rsi` | `rsi_14` | relative strength index on close changes, simple averages; 50 until the window is full | 14 sessions |
| `include_rsi` | `rsi_normalized` | `(rsi_14 − 50) / 50` | 14 sessions |
| `include_ma_features` | `dist_ma50`, `dist_ma200` | `close / trailing mean close − 1`; early rows use the sessions available | 50 and 200 sessions |
| `include_ma_features` | `ma_cross` | 1 when the 50-session mean is above the 200-session mean | 200 sessions |
| `include_price_features` | `daily_range` | `(high − low) / close` | `D` |
| `include_price_features` | `body_ratio` | `|close − open| / (high − low)` | `D` |
| `include_price_features` | `overnight_return` | `open / previous close − 1` | `D − 1` to `D` |
| `include_price_features` | `intraday_return` | `close / open − 1` | `D` |
| `include_volume_features` | `volume_ma20` | trailing mean volume; early rows use the sessions available | 20 sessions |
| `include_volume_features` | `volume_ratio` | `volume / volume_ma20` | 20 sessions |
| `include_volume_features` | `dollar_volume` | `close × volume` | `D` |

## Feature presets

`configs/features/`, selected with `features=<name>`.

| Preset | Families |
| --- | --- |
| `base` | base only |
| `with_momentum` | base + momentum with weekly terms, binary encoding, static 0.5 blend. The base config's default. |
| `full` | every family above, including volatility targeting, VIX, credit, global regime, RSI, moving averages, price and volume |

Experiment presets under `configs/experiment/` can switch individual flags on top of a
feature preset; [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) lists them. Adding a
family means a feature function, the `FeatureEngineer` wiring, `build_feature_list`, a
`FeatureConfig` field with validation, the Hydra YAML, and a row here; the wiring checks
in [TESTING_GUIDE.md](TESTING_GUIDE.md) cover each surface.

Every run writes `feature_reference.json` beside its checkpoints: training-window
quantile bins and histogram counts for each feature column, for drift monitoring.
