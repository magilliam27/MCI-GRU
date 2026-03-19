"""
Global scalar regime features for MCI-GRU.

Implements a Phase-1 scalar regime signal from a small macro/market subset:
- market
- yield curve
- oil
- copper
- stock-bond correlation

Design goals:
- No look-ahead: month T compares only to historical months i < T.
- Configurable exclusion window to avoid near-term similarity leakage.
- Broadcast market-level regime features to all stocks by date.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


REGIME_REQUIRED_VARIABLES: List[str] = [
    "regime_market",
    "regime_yield_curve",
    "regime_oil",
    "regime_copper",
    "regime_stock_bond_corr",
]

REGIME_OPTIONAL_VARIABLES: List[str] = [
    "regime_monetary_policy",
    "regime_volatility",
]

REGIME_VARIABLES: List[str] = REGIME_REQUIRED_VARIABLES + REGIME_OPTIONAL_VARIABLES

REGIME_BASE_FEATURES: List[str] = [
    "regime_global_score",
    "regime_similarity_q20_mean",
    "regime_dissimilarity_q80_mean",
    "regime_similarity_spread",
]

DEFAULT_SUBSEQUENT_RETURN_HORIZONS: List[int] = [1, 3]


def get_regime_features(
    include_subsequent_returns: bool = True,
    horizons: Optional[Sequence[int]] = None,
) -> List[str]:
    features = list(REGIME_BASE_FEATURES)
    if not include_subsequent_returns:
        return features

    normalized_horizons = _normalize_horizons(horizons)
    for horizon in normalized_horizons:
        features.append(f"regime_similar_subsequent_return_{horizon}m")
    if 1 in normalized_horizons:
        features.append("regime_subsequent_return_spread_1m")
    return features


def _monthly_transform(
    monthly_raw: pd.DataFrame,
    change_months: int = 12,
    norm_window_months: int = 120,
    clip_z: float = 3.0,
    min_periods: int = 24,
) -> pd.DataFrame:
    """
    Create transformed monthly regime inputs per variable.

    Steps:
    1) month-over-month level aligned to month-end input
    2) 12m change (default)
    3) rolling z-like normalization using historical window only
    4) clip to reduce outlier dominance
    """
    transformed = pd.DataFrame(index=monthly_raw.index)
    for col in monthly_raw.columns:
        series = monthly_raw[col]
        delta = series - series.shift(change_months)
        rolling_mean = delta.rolling(window=norm_window_months, min_periods=min_periods).mean()
        rolling_std = delta.rolling(window=norm_window_months, min_periods=min_periods).std()
        z = (delta - rolling_mean) / (rolling_std + 1e-8)
        transformed[col] = z.clip(-clip_z, clip_z)
    return transformed


def _normalize_horizons(horizons: Optional[Sequence[int]]) -> List[int]:
    raw_horizons = list(DEFAULT_SUBSEQUENT_RETURN_HORIZONS if horizons is None else horizons)
    normalized: List[int] = []
    seen = set()
    for horizon in raw_horizons:
        horizon_int = int(horizon)
        if horizon_int <= 0:
            raise ValueError("subsequent_return_horizons must contain only positive integers")
        if horizon_int not in seen:
            seen.add(horizon_int)
            normalized.append(horizon_int)
    return normalized


def _empty_feature_row(dt: pd.Timestamp, feature_columns: Sequence[str]) -> dict[str, float]:
    row = {"dt": dt}
    row.update({col: np.nan for col in feature_columns})
    return row


def _nanmean_or_nan(values: np.ndarray) -> float:
    if values.size == 0 or np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


REGIME_FEATURES: List[str] = get_regime_features()


def compute_regime_monthly_features(
    regime_df: pd.DataFrame,
    change_months: int = 12,
    norm_window_months: int = 120,
    clip_z: float = 3.0,
    exclusion_months: int = 1,
    similarity_quantile: float = 0.2,
    min_history_months: int = 24,
    include_subsequent_returns: bool = True,
    subsequent_return_horizons: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Compute monthly regime scalar features from daily regime inputs.

    Expected input columns:
    - dt
    - regime_market
    - regime_yield_curve
    - regime_oil
    - regime_copper
    - regime_stock_bond_corr
    """
    required = {"dt"} | set(REGIME_REQUIRED_VARIABLES)
    missing = sorted(required - set(regime_df.columns))
    if missing:
        raise ValueError(f"regime_df is missing required columns: {missing}")

    if not (0 < similarity_quantile < 0.5):
        raise ValueError("similarity_quantile must be in (0, 0.5)")

    horizons = _normalize_horizons(subsequent_return_horizons) if include_subsequent_returns else []
    if include_subsequent_returns and exclusion_months < 1:
        raise ValueError("include_subsequent_returns=True requires exclusion_months >= 1 to avoid look-ahead")

    feature_columns = get_regime_features(
        include_subsequent_returns=include_subsequent_returns,
        horizons=horizons,
    )

    work = regime_df.copy()
    for col in REGIME_OPTIONAL_VARIABLES:
        if col not in work.columns:
            work[col] = np.nan
    work = work[["dt"] + REGIME_VARIABLES].copy()
    work["dt"] = pd.to_datetime(work["dt"])
    work = work.sort_values("dt")

    # Monthly sampling at month-end keeps one market-wide state per month.
    monthly_raw = (
        work.set_index("dt")[REGIME_VARIABLES]
        .resample("ME")
        .last()
        .dropna(how="all")
    )

    transformed = _monthly_transform(
        monthly_raw=monthly_raw,
        change_months=change_months,
        norm_window_months=norm_window_months,
        clip_z=clip_z,
        min_periods=max(min_history_months, change_months + 1),
    )

    market_series = pd.to_numeric(monthly_raw["regime_market"], errors="coerce")
    forward_returns = {
        horizon: (market_series.shift(-horizon) / market_series) - 1.0
        for horizon in horizons
    }

    out_rows = []
    for idx in range(len(transformed)):
        dt = transformed.index[idx]
        current = transformed.iloc[idx]

        # Strict historical-only set to prevent look-ahead bias.
        history_end = idx - exclusion_months
        if history_end <= 0:
            out_rows.append(_empty_feature_row(dt, feature_columns))
            continue

        historical = transformed.iloc[:history_end]
        indexed_distances = []
        for hist_idx in range(len(historical)):
            hist = historical.iloc[hist_idx]
            valid_mask = (~current.isna()) & (~hist.isna())
            valid_count = int(valid_mask.sum())
            if valid_count == 0:
                continue
            diff = (current[valid_mask] - hist[valid_mask]).to_numpy(dtype=float)
            indexed_distances.append((hist_idx, float(np.linalg.norm(diff))))

        if len(indexed_distances) < min_history_months:
            out_rows.append(_empty_feature_row(dt, feature_columns))
            continue

        indexed_distances.sort(key=lambda item: item[1])
        distances = np.array([distance for _, distance in indexed_distances], dtype=float)
        q_count = max(1, int(np.floor(len(indexed_distances) * similarity_quantile)))
        similar_indices = [hist_idx for hist_idx, _ in indexed_distances[:q_count]]
        dissimilar_indices = [hist_idx for hist_idx, _ in indexed_distances[-q_count:]]
        similar = distances[:q_count]
        dissimilar = distances[-q_count:]

        sim_mean = float(np.mean(similar))
        dis_mean = float(np.mean(dissimilar))
        row = {
            "dt": dt,
            "regime_global_score": float(np.mean(distances)),
            "regime_similarity_q20_mean": sim_mean,
            "regime_dissimilarity_q80_mean": dis_mean,
            "regime_similarity_spread": dis_mean - sim_mean,
        }
        for horizon in horizons:
            horizon_returns = forward_returns[horizon]
            row[f"regime_similar_subsequent_return_{horizon}m"] = _nanmean_or_nan(
                horizon_returns.iloc[similar_indices].to_numpy(dtype=float)
            )
            if horizon == 1:
                dissimilar_ret_1m = _nanmean_or_nan(
                    horizon_returns.iloc[dissimilar_indices].to_numpy(dtype=float)
                )
                row["regime_subsequent_return_spread_1m"] = (
                    row["regime_similar_subsequent_return_1m"] - dissimilar_ret_1m
                    if not np.isnan(row["regime_similar_subsequent_return_1m"]) and not np.isnan(dissimilar_ret_1m)
                    else np.nan
                )
        out_rows.append(row)

    monthly_features = pd.DataFrame(out_rows)
    expected_columns = ["dt"] + list(feature_columns)
    for col in expected_columns:
        if col not in monthly_features.columns:
            monthly_features[col] = np.nan
    monthly_features = monthly_features[expected_columns]
    monthly_features["dt"] = pd.to_datetime(monthly_features["dt"])
    return monthly_features


def add_regime_features(
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
    change_months: int = 12,
    norm_window_months: int = 120,
    clip_z: float = 3.0,
    exclusion_months: int = 1,
    similarity_quantile: float = 0.2,
    min_history_months: int = 24,
    include_subsequent_returns: bool = True,
    subsequent_return_horizons: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Merge monthly regime scalar features onto stock-level daily rows.

    Uses backward as-of merge by date so each stock-day only sees the latest
    already-computed regime month (no future month leakage).
    """
    monthly_features = compute_regime_monthly_features(
        regime_df=regime_df,
        change_months=change_months,
        norm_window_months=norm_window_months,
        clip_z=clip_z,
        exclusion_months=exclusion_months,
        similarity_quantile=similarity_quantile,
        min_history_months=min_history_months,
        include_subsequent_returns=include_subsequent_returns,
        subsequent_return_horizons=subsequent_return_horizons,
    )

    out = df.copy()
    out["dt"] = pd.to_datetime(out["dt"])
    out = out.sort_values(["dt", "kdcode"]).reset_index(drop=True)

    features_sorted = monthly_features.sort_values("dt").reset_index(drop=True)
    out = pd.merge_asof(
        out,
        features_sorted,
        on="dt",
        direction="backward",
    )

    # Keep behavior stable when regime series starts later than stock data.
    feature_columns = get_regime_features(
        include_subsequent_returns=include_subsequent_returns,
        horizons=subsequent_return_horizons,
    )
    for col in feature_columns:
        if col in out.columns:
            out[col] = out[col].ffill().fillna(0.0)

    out["dt"] = out["dt"].dt.strftime("%Y-%m-%d")
    return out
