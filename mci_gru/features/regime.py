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

from typing import List

import numpy as np
import pandas as pd


REGIME_VARIABLES: List[str] = [
    "regime_market",
    "regime_yield_curve",
    "regime_oil",
    "regime_copper",
    "regime_stock_bond_corr",
]

REGIME_FEATURES: List[str] = [
    "regime_global_score",
    "regime_similarity_q20_mean",
    "regime_dissimilarity_q80_mean",
    "regime_similarity_spread",
]


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


def compute_regime_monthly_features(
    regime_df: pd.DataFrame,
    change_months: int = 12,
    norm_window_months: int = 120,
    clip_z: float = 3.0,
    exclusion_months: int = 1,
    similarity_quantile: float = 0.2,
    min_history_months: int = 24,
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
    required = {"dt"} | set(REGIME_VARIABLES)
    missing = sorted(required - set(regime_df.columns))
    if missing:
        raise ValueError(f"regime_df is missing required columns: {missing}")

    if not (0 < similarity_quantile < 0.5):
        raise ValueError("similarity_quantile must be in (0, 0.5)")

    work = regime_df[["dt"] + REGIME_VARIABLES].copy()
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

    out_rows = []
    for idx in range(len(transformed)):
        dt = transformed.index[idx]
        current = transformed.iloc[idx]

        # Strict historical-only set to prevent look-ahead bias.
        history_end = idx - exclusion_months
        if history_end <= 0:
            out_rows.append(
                {
                    "dt": dt,
                    "regime_global_score": np.nan,
                    "regime_similarity_q20_mean": np.nan,
                    "regime_dissimilarity_q80_mean": np.nan,
                    "regime_similarity_spread": np.nan,
                }
            )
            continue

        historical = transformed.iloc[:history_end]
        distances = []
        for hist_idx in range(len(historical)):
            hist = historical.iloc[hist_idx]
            valid_mask = (~current.isna()) & (~hist.isna())
            valid_count = int(valid_mask.sum())
            if valid_count == 0:
                continue
            diff = (current[valid_mask] - hist[valid_mask]).to_numpy(dtype=float)
            distances.append(float(np.linalg.norm(diff)))

        if len(distances) < min_history_months:
            out_rows.append(
                {
                    "dt": dt,
                    "regime_global_score": np.nan,
                    "regime_similarity_q20_mean": np.nan,
                    "regime_dissimilarity_q80_mean": np.nan,
                    "regime_similarity_spread": np.nan,
                }
            )
            continue

        distances = np.array(sorted(distances), dtype=float)
        q_count = max(1, int(np.floor(len(distances) * similarity_quantile)))
        similar = distances[:q_count]
        dissimilar = distances[-q_count:]

        sim_mean = float(np.mean(similar))
        dis_mean = float(np.mean(dissimilar))
        out_rows.append(
            {
                "dt": dt,
                "regime_global_score": float(np.mean(distances)),
                "regime_similarity_q20_mean": sim_mean,
                "regime_dissimilarity_q80_mean": dis_mean,
                "regime_similarity_spread": dis_mean - sim_mean,
            }
        )

    monthly_features = pd.DataFrame(out_rows)
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
    for col in REGIME_FEATURES:
        if col in out.columns:
            out[col] = out[col].ffill().fillna(0.0)

    out["dt"] = out["dt"].dt.strftime("%Y-%m-%d")
    return out
