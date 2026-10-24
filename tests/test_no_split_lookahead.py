"""Stock inputs on a date never reflect a split that happens after it (#323, PRD #319 D8).

The EODHD panel is split-adjusted as of its end, so every price and volume before a
split is rescaled by a split nobody knew about yet. The legacy base inputs are those
raw levels (``close, open, high, low, volume, turnover``), so a row before Netflix's
10-for-1 split on 2025-11-17 already showed a tenth of the price that day. That breaks
invariant 1 (no look-ahead).

``features.base_inputs=relative`` replaces the six levels with per-stock ratios: a log
return, the overnight and intraday returns, the high-low range over the close, and log
volume and turnover against the stock's own trailing average. A split rescales every
row before it by one factor, and ratios of rows that share a factor do not move.

The guard compares two copies of one stock: the panel as an observer saw it on each
date (unadjusted, a real price drop on the split day), and the same panel adjusted
for that later split, as the vendor ships it. Every input row before the split must
be identical. The legacy levels fail that, which the companion test pins so the
guard is known to detect the bug it exists for.
"""

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
)
from mci_gru.features import FeatureEngineer
from mci_gru.features.base import RELATIVE_BASE_FEATURES
from mci_gru.pipeline import prepare_data

N_DAYS = 80
SPLIT_DAY = 60  # index of the first post-split session


def _dates(n: int = N_DAYS) -> list[str]:
    return pd.bdate_range("2021-01-04", periods=n).strftime("%Y-%m-%d").tolist()


def _economic_path(seed: int, n: int = N_DAYS) -> pd.DataFrame:
    """One stock's prices and share volume in pre-split units, with no split applied."""
    rng = np.random.default_rng(seed)
    close = 400.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * np.exp(rng.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.02, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.02, n))
    volume = rng.uniform(1.0e6, 3.0e6, n)
    return pd.DataFrame({"close": close, "open": open_, "high": high, "low": low, "volume": volume})


def _stock_rows(kdcode: str, path: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    out = path.copy()
    out.insert(0, "dt", dates)
    out.insert(0, "kdcode", kdcode)
    return out


def _split_views(ratio: float, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same split stock as an observer saw it, and as an end-of-panel adjusted file.

    ``seen``: prices before ``SPLIT_DAY`` are what traded; from the split day the share
    price is ``1/ratio`` of the economic price and the share volume ``ratio`` times.
    ``adjusted``: every row is on the post-split scale, as the vendor rescales history.
    Rows from the split day on are the same in both.
    """
    dates = _dates()
    path = _economic_path(seed)
    post = path.copy()
    for col in ("close", "open", "high", "low"):
        post[col] = post[col] / ratio
    post["volume"] = post["volume"] * ratio

    seen = path.copy()
    seen.iloc[SPLIT_DAY:] = post.iloc[SPLIT_DAY:].to_numpy()
    adjusted = post

    peer = _economic_path(seed + 1)
    seen_df = pd.concat(
        [_stock_rows("SPLT", seen, dates), _stock_rows("PEER", peer, dates)], ignore_index=True
    )
    adjusted_df = pd.concat(
        [_stock_rows("SPLT", adjusted, dates), _stock_rows("PEER", peer, dates)],
        ignore_index=True,
    )
    return seen_df, adjusted_df


def _engineer(base_inputs: str | None, **flags) -> FeatureEngineer:
    kwargs = dict(
        include_momentum=True,
        include_weekly_momentum=True,
        momentum_encoding="binary",
        momentum_blend_mode="static",
        momentum_blend_fast_weight=0.5,
    )
    kwargs.update(flags)
    if base_inputs is not None:
        kwargs["base_inputs"] = base_inputs
    return FeatureEngineer(FeatureConfig(**kwargs))


def _inputs_before_split(df: pd.DataFrame, engineer: FeatureEngineer) -> pd.DataFrame:
    out = engineer.transform(df.copy())
    cols = engineer.get_feature_columns()
    pre_split = _dates()[SPLIT_DAY - 1]
    out = out[(out["kdcode"] == "SPLT") & (out["dt"] <= pre_split)]
    return out.sort_values("dt")[cols].reset_index(drop=True)


@pytest.mark.parametrize("ratio", [4.0, 10.0])
def test_relative_inputs_before_a_split_ignore_the_later_split(ratio: float) -> None:
    seen, adjusted = _split_views(ratio)
    engineer = _engineer("relative")

    before_seen = _inputs_before_split(seen, engineer)
    before_adjusted = _inputs_before_split(adjusted, engineer)

    assert len(before_seen) == SPLIT_DAY
    pd.testing.assert_frame_equal(before_seen, before_adjusted, check_exact=False, rtol=1e-12)


def test_legacy_level_inputs_carry_the_later_split() -> None:
    """The guard's control: the base default still shows the bug it exists for."""
    seen, adjusted = _split_views(10.0)
    engineer = _engineer(None)

    before_seen = _inputs_before_split(seen, engineer)
    before_adjusted = _inputs_before_split(adjusted, engineer)

    for col in ("close", "open", "high", "low"):
        np.testing.assert_allclose(before_adjusted[col], before_seen[col] / 10.0, rtol=1e-12)
    np.testing.assert_allclose(before_adjusted["volume"], before_seen["volume"] * 10.0, rtol=1e-12)


def test_relative_inputs_replace_the_six_levels() -> None:
    engineer = _engineer("relative")
    cols = engineer.get_feature_columns()

    assert cols[: len(RELATIVE_BASE_FEATURES)] == RELATIVE_BASE_FEATURES
    assert len(RELATIVE_BASE_FEATURES) == 6
    assert not {"close", "open", "high", "low", "volume", "turnover"} & set(cols)
    assert len(cols) == len(set(cols))


def test_relative_inputs_do_not_repeat_when_price_features_are_also_on() -> None:
    engineer = _engineer("relative", include_price_features=True)
    cols = engineer.get_feature_columns()

    assert len(cols) == len(set(cols))
    assert {"overnight_return", "intraday_return", "daily_range"} <= set(cols)


def test_relative_definitions_win_over_the_price_features_for_shared_names() -> None:
    """With both on, the shared columns follow the relative definitions.

    ``add_price_features`` fills a stock's first ``overnight_return`` with 0 and does
    not carry the previous close over a gap; the relative step leaves the first row
    missing and spans the gap. A gap day is where the two disagree.
    """
    seen, _ = _split_views(4.0)
    gap_day = _dates()[20]
    seen.loc[(seen["kdcode"] == "PEER") & (seen["dt"] == gap_day), ["close", "open"]] = np.nan
    engineer = _engineer(
        "relative",
        include_momentum=False,
        include_weekly_momentum=False,
        include_price_features=True,
    )
    out = engineer.transform(seen.copy())
    stock = out[out["kdcode"] == "PEER"].sort_values("dt").reset_index(drop=True)

    assert np.isnan(stock.loc[0, "overnight_return"])
    assert stock.loc[21, "overnight_return"] == pytest.approx(
        stock.loc[21, "open"] / stock.loc[19, "close"] - 1, rel=1e-12
    )


def test_relative_inputs_ignore_a_later_split_across_gaps_in_the_history() -> None:
    """A missing session and an all-missing row before the split change nothing."""
    seen, adjusted = _split_views(10.0)
    dates = _dates()
    dropped, blank = dates[25], dates[40]
    price_cols = ["close", "open", "high", "low", "volume"]
    views = []
    for df in (seen, adjusted):
        df = df[~((df["kdcode"] == "SPLT") & (df["dt"] == dropped))].copy()
        df.loc[(df["kdcode"] == "SPLT") & (df["dt"] == blank), price_cols] = np.nan
        views.append(df)
    engineer = _engineer("relative")

    before_seen = _inputs_before_split(views[0], engineer)
    before_adjusted = _inputs_before_split(views[1], engineer)

    assert len(before_seen) == SPLIT_DAY - 1
    pd.testing.assert_frame_equal(before_seen, before_adjusted, check_exact=False, rtol=1e-12)


def test_relative_inputs_are_ratios_of_the_stock_s_own_prices() -> None:
    seen, _ = _split_views(4.0)
    engineer = _engineer("relative", include_momentum=False, include_weekly_momentum=False)
    out = engineer.transform(seen.copy())
    stock = out[out["kdcode"] == "PEER"].sort_values("dt").reset_index(drop=True)

    close, open_ = stock["close"], stock["open"]
    np.testing.assert_allclose(stock["log_return"][1:], np.log(close / close.shift(1))[1:])
    np.testing.assert_allclose(stock["overnight_return"][1:], (open_ / close.shift(1) - 1)[1:])
    np.testing.assert_allclose(stock["intraday_return"], close / open_ - 1)
    np.testing.assert_allclose(stock["daily_range"], (stock["high"] - stock["low"]) / close)
    trailing = stock["volume"].rolling(20, min_periods=1).mean()
    np.testing.assert_allclose(stock["log_volume_rel"], np.log(stock["volume"] / trailing))
    turnover = close * stock["volume"]
    np.testing.assert_allclose(
        stock["log_turnover_rel"],
        np.log(turnover / turnover.rolling(20, min_periods=1).mean()),
    )


def test_relative_inputs_leave_zero_volume_missing_rather_than_infinite() -> None:
    seen, _ = _split_views(4.0)
    seen.loc[(seen["kdcode"] == "PEER") & (seen["dt"] == _dates()[30]), "volume"] = 0.0
    engineer = _engineer("relative", include_momentum=False, include_weekly_momentum=False)
    out = engineer.transform(seen.copy())

    values = out[RELATIVE_BASE_FEATURES].to_numpy(dtype=float)
    assert not np.isinf(values).any()
    stock = out[out["kdcode"] == "PEER"].sort_values("dt").reset_index(drop=True)
    assert np.isnan(stock.loc[30, "log_volume_rel"])
    # The zero is missing, not a real day of no trading, so it stays out of the
    # trailing mean that the next sessions are measured against.
    window = stock.loc[12:31, "volume"].drop(index=30)
    assert stock.loc[31, "log_volume_rel"] == pytest.approx(
        np.log(stock.loc[31, "volume"] / window.mean()), rel=1e-12
    )


def test_unknown_base_inputs_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="base_inputs"):
        FeatureConfig(base_inputs="returns")


def test_base_default_keeps_the_legacy_levels() -> None:
    assert FeatureConfig().base_inputs == "levels"
    cols = _engineer(None).get_feature_columns()
    assert cols[:6] == ["close", "open", "high", "low", "volume", "turnover"]


# --- Through the pipeline: the normaliser fits the new inputs on train rows only ---


def _write_panel(df: pd.DataFrame, tmp_path, name: str) -> str:
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _pipeline_config(panel_csv: str, pit_csv: str, base_inputs: str) -> ExperimentConfig:
    dates = _dates()
    return ExperimentConfig(
        data=DataConfig(
            source="csv",
            filename=panel_csv,
            train_start=dates[0],
            train_end=dates[39],
            val_start=dates[45],
            val_end=dates[52],
            test_start=dates[SPLIT_DAY - 2],
            test_end=dates[-1],
            skip_embargo_check=True,
            use_pit_universe=True,
            pit_universe_csv=pit_csv,
            pit_universe_mode="masked_panel",
            pit_min_scoreable_stocks=0,
        ),
        features=FeatureConfig(
            base_inputs=base_inputs,
            include_momentum=False,
            include_weekly_momentum=False,
        ),
        graph=GraphConfig(judge_value=0.9999, use_multi_feature_edges=False),
        model=ModelConfig(his_t=5, label_t=2),
        training=TrainingConfig(num_epochs=1, num_models=1, label_type="returns"),
        tracking=TrackingConfig(enabled=False),
    )


def _prepare(df: pd.DataFrame, tmp_path, name: str, base_inputs: str) -> dict:
    dates = _dates()
    pit = pd.DataFrame(
        [
            {"kdcode": kd, "valid_from": dates[0], "valid_to": dates[-1]}
            for kd in sorted(df["kdcode"].unique())
        ]
    )
    pit_csv = _write_panel(pit, tmp_path, f"{name}_pit")
    panel_csv = _write_panel(df, tmp_path, name)
    cfg = _pipeline_config(panel_csv, pit_csv, base_inputs)
    return prepare_data(cfg, FeatureEngineer(cfg.features))


def _with_turnover(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["turnover"] = df["close"] * df["volume"]
    return df


def test_normaliser_fits_the_relative_inputs_on_train_rows_only(tmp_path) -> None:
    seen, adjusted = _split_views(10.0)
    seen, adjusted = _with_turnover(seen), _with_turnover(adjusted)

    data_seen = _prepare(seen, tmp_path, "seen", "relative")
    data_adjusted = _prepare(adjusted, tmp_path, "adjusted", "relative")

    assert data_seen["feature_cols"] == RELATIVE_BASE_FEATURES
    assert list(data_seen["norm_means"]) == RELATIVE_BASE_FEATURES
    assert list(data_seen["norm_stds"]) == RELATIVE_BASE_FEATURES
    # The split falls in the test span. Train statistics and train inputs are the same
    # whether or not history was rescaled for it.
    for col in RELATIVE_BASE_FEATURES:
        assert data_seen["norm_means"][col] == pytest.approx(
            data_adjusted["norm_means"][col], rel=1e-9, abs=1e-12
        )
        assert data_seen["norm_stds"][col] == pytest.approx(
            data_adjusted["norm_stds"][col], rel=1e-9, abs=1e-12
        )
    np.testing.assert_allclose(
        data_seen["stock_features_train"], data_adjusted["stock_features_train"], atol=1e-6
    )


def test_normaliser_statistics_ignore_rows_after_train_end(tmp_path) -> None:
    seen, _ = _split_views(10.0)
    seen = _with_turnover(seen)
    shocked = seen.copy()
    late = shocked["dt"] > _dates()[39]
    shocked.loc[late, "volume"] *= 50.0
    shocked.loc[late, "turnover"] = shocked.loc[late, "close"] * shocked.loc[late, "volume"]

    base = _prepare(seen, tmp_path, "base", "relative")
    after = _prepare(shocked, tmp_path, "shocked", "relative")

    assert base["norm_means"] == pytest.approx(after["norm_means"], rel=1e-12)
    assert base["norm_stds"] == pytest.approx(after["norm_stds"], rel=1e-12)
