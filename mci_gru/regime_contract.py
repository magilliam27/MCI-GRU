"""Regime input column contract (stdlib-only; shared by data and features layers)."""

REGIME_REQUIRED_VARIABLES: list[str] = [
    "regime_market",
    "regime_yield_curve",
    "regime_oil",
    "regime_copper",
    "regime_stock_bond_corr",
]

REGIME_OPTIONAL_VARIABLES: list[str] = [
    "regime_monetary_policy",
    "regime_volatility",
]

REGIME_VARIABLES: list[str] = REGIME_REQUIRED_VARIABLES + REGIME_OPTIONAL_VARIABLES
