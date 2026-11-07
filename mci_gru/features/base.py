"""Base OHLCV features and turnover."""

import numpy as np
import pandas as pd

from mci_gru.utils.returns import padded_pct_change

# Base feature columns
BASE_FEATURES = ["close", "open", "high", "low", "volume", "turnover"]

# Per-stock replacements for the six levels (features.base_inputs=relative, #323).
# The price panel is split-adjusted as of its end, so a level before a split already
# reflects it; a ratio of two rows that share the adjustment factor does not.
RELATIVE_BASE_FEATURES = [
    "log_return",
    "overnight_return",
    "intraday_return",
    "daily_range",
    "log_volume_rel",
    "log_turnover_rel",
]
RELATIVE_VOLUME_WINDOW = 20


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Turnover = Close x Volume per paper specification
    if "turnover" not in df.columns:
        df["turnover"] = df["close"] * df["volume"]
        print("  Added turnover feature (close * volume)")

    return df


def _log_relative_to_trailing_mean(values: pd.Series, groups: pd.Series, window: int) -> pd.Series:
    """ln(value / the stock's own trailing mean, today included); non-positive -> NaN."""
    positive = values.where(values > 0)
    trailing = positive.groupby(groups).transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    return np.log(positive / trailing)


def add_relative_base_features(
    df: pd.DataFrame, volume_window: int = RELATIVE_VOLUME_WINDOW
) -> pd.DataFrame:
    """Scale-free per-stock inputs that replace the six raw levels.

    - log_return: ln(close / previous close), with the #245 gap policy
    - overnight_return: open / previous close - 1
    - intraday_return: close / open - 1
    - daily_range: (high - low) / close
    - log_volume_rel, log_turnover_rel: ln(value / the stock's trailing mean)

    A stock's first row has no previous close and stays NaN, as does a row with
    no positive volume; the pipeline's per-day imputation fills them.
    """
    df = df.copy()
    df = df.sort_values(["kdcode", "dt"])
    groups = df["kdcode"]
    df["log_return"] = np.log1p(padded_pct_change(df["close"], groups))
    prev_close = df.groupby("kdcode")["close"].ffill().groupby(groups).shift(1)
    df["overnight_return"] = df["open"] / prev_close - 1
    df["intraday_return"] = df["close"] / df["open"] - 1
    df["daily_range"] = (df["high"] - df["low"]) / df["close"]
    df["log_volume_rel"] = _log_relative_to_trailing_mean(df["volume"], groups, volume_window)
    df["log_turnover_rel"] = _log_relative_to_trailing_mean(df["turnover"], groups, volume_window)
    for col in RELATIVE_BASE_FEATURES:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    print(f"  Added relative base inputs: {', '.join(RELATIVE_BASE_FEATURES)}")

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["kdcode", "dt"])
    df["daily_range"] = (df["high"] - df["low"]) / df["close"]
    range_val = df["high"] - df["low"]
    df["body_ratio"] = np.abs(df["close"] - df["open"]) / (range_val + 1e-8)
    df["prev_close"] = df.groupby("kdcode")["close"].shift(1)
    df["overnight_return"] = df["open"] / df["prev_close"] - 1
    df["overnight_return"] = df["overnight_return"].fillna(0)
    df["intraday_return"] = df["close"] / df["open"] - 1
    df["intraday_return"] = df["intraday_return"].fillna(0)
    df = df.drop(columns=["prev_close"], errors="ignore")

    print("  Added price features: daily_range, body_ratio, overnight_return, intraday_return")

    return df


def add_volume_features(df: pd.DataFrame, ma_window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["kdcode", "dt"])

    ma_col = f"volume_ma{ma_window}"
    df[ma_col] = df.groupby("kdcode")["volume"].transform(
        lambda x: x.rolling(window=ma_window, min_periods=1).mean()
    )
    df["volume_ratio"] = df["volume"] / (df[ma_col] + 1e-8)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
    df["dollar_volume"] = df["close"] * df["volume"]

    print(f"  Added volume features: volume_ma{ma_window}, volume_ratio, dollar_volume")

    return df
