"""Behavioral tests for ordinary rolling-volatility features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mci_gru.features.volatility import add_volatility_features


def _price_panel(closes: list[float], *, start: str = "2020-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kdcode": "AAA",
            "dt": pd.date_range(start, periods=len(closes), freq="D").strftime("%Y-%m-%d"),
            "close": closes,
        }
    )


def test_volatility_features_are_prefix_invariant() -> None:
    prefix_closes = [100.0]
    for i in range(1, 30):
        prefix_closes.append(prefix_closes[-1] * (1.0 + (0.004 if i % 3 else -0.002)))
    prefix = _price_panel(prefix_closes)

    future_closes: list[float] = []
    last_close = prefix_closes[-1]
    for i in range(40):
        last_close *= 1.5 if i % 2 == 0 else 0.6
        future_closes.append(last_close)
    extended = _price_panel(prefix_closes + future_closes)

    prefix_result = add_volatility_features(prefix)
    extended_result = add_volatility_features(extended)
    historical = extended_result[extended_result["dt"].isin(prefix_result["dt"])]

    for column in ("volatility_5d", "volatility_21d", "vol_ratio"):
        np.testing.assert_allclose(
            prefix_result[column].to_numpy(),
            historical[column].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
