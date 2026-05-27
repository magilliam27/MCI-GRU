"""Volatility features: realized volatility, vol ratio, VIX, RSI, moving averages."""

import numpy as np
import pandas as pd

# Volatility feature columns
VOLATILITY_FEATURES = ["volatility_5d", "volatility_21d", "vol_ratio"]
VIX_FEATURES = ["vix", "vix_change", "vix_regime"]
DEFAULT_VOLATILITY_TARGETING_HALF_LIVES = [20, 60, 90]
DEFAULT_VOLATILITY_TARGET_VOL = 0.10
DEFAULT_VOLATILITY_TARGET_SCALE_CLIP = (0.25, 4.0)
DEFAULT_VOLATILITY_TARGET_INTERACTION_RETURN_WINDOW = 21


def _validate_volatility_targeting_controls(
    half_lives: list[int] | None,
    target_vol: float,
    scale_clip: tuple[float, float],
    interaction_return_window: int,
) -> list[int]:
    resolved_half_lives = list(half_lives or DEFAULT_VOLATILITY_TARGETING_HALF_LIVES)
    if len(resolved_half_lives) < 2:
        raise ValueError("volatility_targeting_half_lives must contain at least two values")
    if any(half_life <= 0 for half_life in resolved_half_lives):
        raise ValueError("volatility_targeting_half_lives must contain positive integers")
    if len(set(resolved_half_lives)) != len(resolved_half_lives):
        raise ValueError("volatility_targeting_half_lives must not contain duplicates")
    if target_vol <= 0:
        raise ValueError("volatility_target_vol must be > 0")
    clip_low, clip_high = scale_clip
    if clip_low <= 0 or clip_high <= clip_low:
        raise ValueError("volatility_target_scale_clip must be positive and increasing")
    if interaction_return_window <= 0:
        raise ValueError("volatility_target_interaction_return_window must be > 0")
    return resolved_half_lives


def get_volatility_targeting_features(
    half_lives: list[int] | None = None,
    interaction_return_window: int = DEFAULT_VOLATILITY_TARGET_INTERACTION_RETURN_WINDOW,
    include_ewm_vol: bool = True,
    include_scale: bool = True,
    include_dynamics: bool = True,
    include_scaled_return: bool = True,
) -> list[str]:
    """Return feature names for the Harvey-style volatility-targeting family."""
    if not any((include_ewm_vol, include_scale, include_dynamics, include_scaled_return)):
        return []

    resolved_half_lives = _validate_volatility_targeting_controls(
        half_lives,
        DEFAULT_VOLATILITY_TARGET_VOL,
        DEFAULT_VOLATILITY_TARGET_SCALE_CLIP,
        interaction_return_window,
    )
    short_half_life = resolved_half_lives[0]
    long_half_life = resolved_half_lives[-1]

    features: list[str] = []
    if include_ewm_vol:
        features.extend(f"vol_target_ewm_vol_hl{half_life}" for half_life in resolved_half_lives)
    if include_scale:
        features.extend(f"vol_target_scale_hl{half_life}" for half_life in resolved_half_lives)
    if include_dynamics:
        features.extend(
            [
                f"vol_target_vol_change_hl{short_half_life}_hl{long_half_life}",
                f"vol_target_vol_of_vol_hl{short_half_life}",
            ]
        )
    if include_scaled_return:
        features.append(
            f"vol_target_ret{interaction_return_window}_lag2_x_scale_hl{short_half_life}"
        )

    return features


VOLATILITY_TARGETING_FEATURES = get_volatility_targeting_features()


def add_volatility_features(
    df: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 21,
    annualization_factor: float = np.sqrt(252),
) -> pd.DataFrame:
    print("Computing volatility features...")
    df = df.sort_values(["kdcode", "dt"]).copy()

    if "_daily_return" not in df.columns:
        df["_daily_return"] = df.groupby("kdcode")["close"].pct_change()

    short_col = f"volatility_{short_window}d"
    df[short_col] = df.groupby("kdcode")["_daily_return"].transform(
        lambda x: (
            x.rolling(window=short_window, min_periods=short_window).std() * annualization_factor
        )
    )

    long_col = f"volatility_{long_window}d"
    df[long_col] = df.groupby("kdcode")["_daily_return"].transform(
        lambda x: (
            x.rolling(window=long_window, min_periods=long_window).std() * annualization_factor
        )
    )

    df[short_col] = df.groupby("kdcode")[short_col].transform(
        lambda x: x.fillna(x.median() if not x.isna().all() else 0.2)
    )
    df[long_col] = df.groupby("kdcode")[long_col].transform(
        lambda x: x.fillna(x.median() if not x.isna().all() else 0.2)
    )

    df["vol_ratio"] = df[short_col] / (df[long_col] + 1e-8)
    df["vol_ratio"] = df["vol_ratio"].clip(0.1, 10)

    print(f"  Added volatility features: {short_col}, {long_col}, vol_ratio")

    return df


def add_volatility_targeting_features(
    df: pd.DataFrame,
    half_lives: list[int] | None = None,
    target_vol: float = DEFAULT_VOLATILITY_TARGET_VOL,
    scale_clip: tuple[float, float] = DEFAULT_VOLATILITY_TARGET_SCALE_CLIP,
    interaction_return_window: int = DEFAULT_VOLATILITY_TARGET_INTERACTION_RETURN_WINDOW,
    annualization_factor: float = np.sqrt(252),
) -> pd.DataFrame:
    """Add Harvey-style ex ante volatility-targeting model-input features.

    A row dated D uses stock returns ending no later than D-2, matching the
    chapter's ex ante volatility-scaling setup while keeping this as a feature
    family rather than a portfolio exposure rule.
    """
    resolved_half_lives = _validate_volatility_targeting_controls(
        half_lives,
        target_vol,
        scale_clip,
        interaction_return_window,
    )
    clip_low, clip_high = scale_clip
    short_half_life = resolved_half_lives[0]
    long_half_life = resolved_half_lives[-1]

    print("Computing volatility-targeting features...")
    df = df.sort_values(["kdcode", "dt"]).copy()
    group_keys = df["kdcode"]
    daily_return = df.groupby("kdcode")["close"].pct_change()
    harvey_return = daily_return.groupby(group_keys).shift(2)

    for half_life in resolved_half_lives:
        vol_col = f"vol_target_ewm_vol_hl{half_life}"
        scale_col = f"vol_target_scale_hl{half_life}"
        df[vol_col] = harvey_return.groupby(group_keys).transform(
            lambda x, hl=half_life: (
                x.ewm(halflife=hl, min_periods=2, adjust=False).std(bias=False)
                * annualization_factor
            )
        )
        df[vol_col] = df[vol_col].fillna(target_vol)
        df[scale_col] = (target_vol / (df[vol_col] + 1e-8)).clip(clip_low, clip_high)

    short_vol_col = f"vol_target_ewm_vol_hl{short_half_life}"
    long_vol_col = f"vol_target_ewm_vol_hl{long_half_life}"
    short_scale_col = f"vol_target_scale_hl{short_half_life}"

    vol_change_col = f"vol_target_vol_change_hl{short_half_life}_hl{long_half_life}"
    df[vol_change_col] = df[short_vol_col] / (df[long_vol_col] + 1e-8) - 1.0
    df[vol_change_col] = df[vol_change_col].fillna(0.0)

    vol_of_vol_col = f"vol_target_vol_of_vol_hl{short_half_life}"
    df[vol_of_vol_col] = df.groupby("kdcode")[short_vol_col].transform(
        lambda x: (
            x.diff()
            .ewm(halflife=short_half_life, min_periods=2, adjust=False)
            .std(bias=False)
            * annualization_factor
        )
    )
    df[vol_of_vol_col] = df[vol_of_vol_col].fillna(0.0)

    interaction_col = (
        f"vol_target_ret{interaction_return_window}_lag2_x_scale_hl{short_half_life}"
    )
    trailing_return = df.groupby("kdcode")["close"].pct_change(
        periods=interaction_return_window
    )
    lagged_trailing_return = trailing_return.groupby(group_keys).shift(2).fillna(0.0)
    df[interaction_col] = lagged_trailing_return * df[short_scale_col]

    added = get_volatility_targeting_features(
        resolved_half_lives,
        interaction_return_window=interaction_return_window,
    )
    print(f"  Added volatility-targeting features: {', '.join(added)}")

    return df


def add_vix_features(
    df: pd.DataFrame, vix_df: pd.DataFrame, vix_ma_window: int = 10
) -> pd.DataFrame:
    print("Merging VIX features...")
    df = df.copy()
    vix = vix_df.copy()

    if "vix" not in vix.columns:
        if "close" in vix.columns:
            vix = vix.rename(columns={"close": "vix"})
        elif "CLOSE" in vix.columns:
            vix = vix.rename(columns={"CLOSE": "vix"})

    if "dt" not in vix.columns and (vix.index.name == "dt" or vix.index.name == "Date"):
        vix = vix.reset_index()
        vix = vix.rename(columns={vix.columns[0]: "dt"})

    vix["dt"] = pd.to_datetime(vix["dt"]).dt.strftime("%Y-%m-%d")
    vix = vix.sort_values("dt")
    vix["vix_change"] = vix["vix"].pct_change().fillna(0)
    vix["vix_ma"] = vix["vix"].rolling(window=vix_ma_window, min_periods=1).mean()
    vix["vix_regime"] = (vix["vix"] > vix["vix_ma"]).astype(float)

    vix_cols = ["dt", "vix", "vix_change", "vix_regime"]
    vix_merge = vix[vix_cols]
    df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
    df = df.merge(vix_merge, on="dt", how="left")
    df["vix"] = df["vix"].fillna(method="ffill").fillna(20)
    df["vix_change"] = df["vix_change"].fillna(0)
    df["vix_regime"] = df["vix_regime"].fillna(0)

    print("  Added VIX features: vix, vix_change, vix_regime")
    print(f"  VIX range: {df['vix'].min():.1f} to {df['vix'].max():.1f}")

    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    print(f"Computing RSI ({period}-day)...")
    df = df.sort_values(["kdcode", "dt"]).copy()

    def compute_rsi(series):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    rsi_col = f"rsi_{period}"
    df[rsi_col] = df.groupby("kdcode")["close"].transform(compute_rsi)
    df["rsi_normalized"] = (df[rsi_col] - 50) / 50
    df[rsi_col] = df[rsi_col].fillna(50)
    df["rsi_normalized"] = df["rsi_normalized"].fillna(0)

    print(f"  Added RSI features: {rsi_col}, rsi_normalized")

    return df


def add_moving_average_features(
    df: pd.DataFrame, short_window: int = 50, long_window: int = 200
) -> pd.DataFrame:
    print(f"Computing moving average features (MA{short_window}, MA{long_window})...")
    df = df.sort_values(["kdcode", "dt"]).copy()

    ma_short = f"ma_{short_window}"
    ma_long = f"ma_{long_window}"

    df[ma_short] = df.groupby("kdcode")["close"].transform(
        lambda x: x.rolling(window=short_window, min_periods=1).mean()
    )
    df[ma_long] = df.groupby("kdcode")["close"].transform(
        lambda x: x.rolling(window=long_window, min_periods=1).mean()
    )

    df[f"dist_ma{short_window}"] = (df["close"] - df[ma_short]) / df[ma_short]
    df[f"dist_ma{long_window}"] = (df["close"] - df[ma_long]) / df[ma_long]
    df["ma_cross"] = (df[ma_short] > df[ma_long]).astype(float)
    df = df.drop(columns=[ma_short, ma_long])

    print(f"  Added MA features: dist_ma{short_window}, dist_ma{long_window}, ma_cross")

    return df
