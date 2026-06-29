"""Saved-prediction execution and capacity replay diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mci_gru.evaluation.artifacts import write_json_artifact
from mci_gru.evaluation.portfolio import apply_rank_drop_gate, calculate_turnover, rank_scores


def add_lagged_capacity_inputs(
    market_data: pd.DataFrame,
    *,
    lookback_days: int,
) -> pd.DataFrame:
    """Add rolling ADV and volatility known by each prediction-date close."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    missing = {"dt", "kdcode", "close", "volume"} - set(market_data.columns)
    if missing:
        raise ValueError(f"market_data missing columns: {sorted(missing)}")

    market = market_data.copy()
    market["dt"] = pd.to_datetime(market["dt"])
    market["dollar_volume"] = market["close"] * market["volume"]
    market = market.sort_values(["kdcode", "dt"]).reset_index(drop=True)
    grouped = market.groupby("kdcode", group_keys=False)
    market["daily_return"] = grouped["close"].pct_change()
    market["lagged_adv"] = grouped["dollar_volume"].transform(
        lambda series: series.shift(1).rolling(lookback_days, min_periods=1).mean()
    )
    volatility_min_periods = min(2, lookback_days)
    market["lagged_volatility"] = grouped["daily_return"].transform(
        lambda series: (
            series.shift(1).rolling(lookback_days, min_periods=volatility_min_periods).std()
        )
    )
    return market


def compute_capacity_replay(
    predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    aum_values: list[float],
    top_k_values: list[int],
    adv_lookback_days: int,
    max_adv_participation: float,
    spread_bps_values: list[float] | None = None,
    slippage_bps_values: list[float] | None = None,
    rank_drop_values: list[int | None] | None = None,
    max_lagged_volatility_values: list[float | None] | None = None,
) -> dict[str, Any]:
    """Replay saved scores through T+1 open execution, costs, and lagged capacity."""
    if max_adv_participation <= 0.0:
        raise ValueError("max_adv_participation must be positive")
    if any(aum <= 0.0 for aum in aum_values):
        raise ValueError("aum_values must all be positive")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("top_k_values must all be positive")
    missing = {"dt", "kdcode", "score"} - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    market_missing = {"dt", "kdcode", "open", "close", "volume"} - set(market_data.columns)
    if market_missing:
        raise ValueError(f"market_data missing columns: {sorted(market_missing)}")

    spreads = spread_bps_values or [0.0]
    slippages = slippage_bps_values or [0.0]
    rank_drop_grid = rank_drop_values if rank_drop_values is not None else [None]
    volatility_grid = (
        max_lagged_volatility_values if max_lagged_volatility_values is not None else [None]
    )
    if any(value < 0.0 for value in spreads):
        raise ValueError("spread_bps_values must be non-negative")
    if any(value < 0.0 for value in slippages):
        raise ValueError("slippage_bps_values must be non-negative")
    if any(value is not None and value < 0 for value in rank_drop_grid):
        raise ValueError("rank_drop_values must be non-negative")
    if any(value is not None and value <= 0.0 for value in volatility_grid):
        raise ValueError("max_lagged_volatility_values must be positive when provided")

    preds = predictions.copy()
    preds["dt"] = pd.to_datetime(preds["dt"])
    market = add_lagged_capacity_inputs(market_data, lookback_days=adv_lookback_days)
    market = market.sort_values(["dt", "kdcode"]).reset_index(drop=True)
    market_dates = sorted(market["dt"].dropna().unique())
    merged = preds.merge(
        market[["dt", "kdcode", "lagged_adv", "lagged_volatility"]],
        on=["dt", "kdcode"],
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for top_k in top_k_values:
        for aum in aum_values:
            for spread_bps in spreads:
                for slippage_bps in slippages:
                    for rank_drop_min in rank_drop_grid:
                        for max_lagged_volatility in volatility_grid:
                            rows.extend(
                                _replay_scenario(
                                    merged,
                                    market,
                                    market_dates=market_dates,
                                    top_k=top_k,
                                    aum=aum,
                                    spread_bps=spread_bps,
                                    slippage_bps=slippage_bps,
                                    rank_drop_min=rank_drop_min,
                                    max_adv_participation=max_adv_participation,
                                    max_lagged_volatility=max_lagged_volatility,
                                )
                            )

    return {
        "schema_version": 1,
        "policy": {
            "timing": "score_at_t_close_enter_t_plus_1_open_hold_open_to_open",
            "uses_lagged_adv": True,
            "uses_lagged_volatility": True,
            "uses_lagged_volatility_gate": any(value is not None for value in volatility_grid),
            "realized_t_plus_1_volume_ex_post_only": True,
            "adv_lookback_days": adv_lookback_days,
            "max_adv_participation": max_adv_participation,
            "cost_model": "one_way_turnover_times_round_trip_spread_plus_two_slippage",
        },
        "grid": {
            "aum_values": [float(value) for value in aum_values],
            "top_k_values": [int(value) for value in top_k_values],
            "spread_bps_values": [float(value) for value in spreads],
            "slippage_bps_values": [float(value) for value in slippages],
            "rank_drop_values": [None if value is None else int(value) for value in rank_drop_grid],
            "max_lagged_volatility_values": [
                None if value is None else float(value) for value in volatility_grid
            ],
        },
        "rows": rows,
    }


def write_capacity_replay(
    report: dict[str, Any], output_dir: str | Path, *, force: bool = False
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    json_path = out_dir / "capacity_replay.json"
    csv_path = out_dir / "capacity_replay.csv"
    if json_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {json_path}")
    if csv_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {csv_path}")
    write_json_artifact(json_path, report, force=force)
    pd.DataFrame(report["rows"]).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}


def _replay_scenario(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    *,
    market_dates: list[pd.Timestamp],
    top_k: int,
    aum: float,
    spread_bps: float,
    slippage_bps: float,
    rank_drop_min: int | None,
    max_adv_participation: float,
    max_lagged_volatility: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_holdings: list[dict[str, Any]] | None = None
    prev_ranks: dict[str, int] | None = None

    for date, day in predictions.groupby("dt", sort=True):
        ranked = rank_scores(day[["kdcode", "score"]])
        if rank_drop_min is None:
            target_stocks = ranked.head(top_k)["kdcode"].tolist()
            decision = {
                "survivors": [],
                "exits": [],
                "new_entries": target_stocks,
                "exit_details": [],
                "is_initial": prev_holdings is None,
            }
        else:
            decision = apply_rank_drop_gate(
                ranked,
                prev_holdings,
                prev_ranks,
                top_k=top_k,
                min_rank_drop=rank_drop_min,
            )
            target_stocks = decision["target_stocks"]

        entry_date = _next_market_date(date, market_dates)
        exit_date = _next_market_date(entry_date, market_dates) if entry_date is not None else None
        current_holdings = [{"kdcode": kdcode} for kdcode in target_stocks]
        turnover = calculate_turnover(prev_holdings, current_holdings, target_k=top_k)
        cost = _transaction_cost(turnover, spread_bps=spread_bps, slippage_bps=slippage_bps)
        selected = _target_frame(day, target_stocks)
        returns = _execution_returns(market, selected, entry_date=entry_date, exit_date=exit_date)
        gross_return = _safe_float(returns["open_to_open_return"].mean(skipna=True))
        net_return = None if gross_return is None else gross_return - cost
        target_notional = aum / top_k
        capacity = _capacity_metrics(
            selected,
            returns,
            target_notional=target_notional,
            max_adv_participation=max_adv_participation,
            max_lagged_volatility=max_lagged_volatility,
        )

        rows.append(
            {
                "dt": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "entry_dt": _format_date(entry_date),
                "exit_dt": _format_date(exit_date),
                "top_k": int(top_k),
                "selected_count": int(len(target_stocks)),
                "aum": float(aum),
                "target_notional_per_name": float(target_notional),
                "rank_drop_enabled": rank_drop_min is not None,
                "min_rank_drop": None if rank_drop_min is None else int(rank_drop_min),
                "survivor_count": int(len(decision["survivors"])),
                "exit_count": int(len(decision["exits"])),
                "new_entry_count": int(len(decision["new_entries"])),
                "spread_bps": float(spread_bps),
                "slippage_bps": float(slippage_bps),
                "turnover": float(turnover),
                "total_cost": float(cost),
                "gross_return": gross_return,
                "net_return": _safe_float(net_return),
                **capacity,
            }
        )
        prev_holdings = current_holdings
        prev_ranks = ranked.set_index("kdcode")["rank"].astype(int).to_dict()

    return rows


def _target_frame(day: pd.DataFrame, target_stocks: list[str]) -> pd.DataFrame:
    targets = pd.DataFrame({"kdcode": target_stocks})
    return targets.merge(day, on="kdcode", how="left")


def _execution_returns(
    market: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    entry_date: pd.Timestamp | None,
    exit_date: pd.Timestamp | None,
) -> pd.DataFrame:
    result = selected[["kdcode"]].copy()
    if entry_date is None or exit_date is None:
        result["entry_open"] = np.nan
        result["entry_volume"] = np.nan
        result["exit_open"] = np.nan
        result["open_to_open_return"] = np.nan
        return result

    entry = market.loc[market["dt"] == entry_date, ["kdcode", "open", "volume"]].rename(
        columns={"open": "entry_open", "volume": "entry_volume"}
    )
    exit_frame = market.loc[market["dt"] == exit_date, ["kdcode", "open"]].rename(
        columns={"open": "exit_open"}
    )
    result = result.merge(entry, on="kdcode", how="left").merge(exit_frame, on="kdcode", how="left")
    valid = (result["entry_open"] > 0) & result["exit_open"].notna()
    result["open_to_open_return"] = np.where(
        valid,
        result["exit_open"] / result["entry_open"] - 1.0,
        np.nan,
    )
    return result


def _capacity_metrics(
    selected: pd.DataFrame,
    execution: pd.DataFrame,
    *,
    target_notional: float,
    max_adv_participation: float,
    max_lagged_volatility: float | None,
) -> dict[str, Any]:
    lagged_adv = selected["lagged_adv"].replace([np.inf, -np.inf], np.nan)
    lagged_volatility = selected["lagged_volatility"].replace([np.inf, -np.inf], np.nan)
    valid_adv = lagged_adv.where(lagged_adv > 0.0)
    participation = target_notional / valid_adv
    adv_breaches = (participation > max_adv_participation).fillna(False)
    if max_lagged_volatility is None:
        volatility_breaches = pd.Series(False, index=selected.index)
    else:
        volatility_breaches = (lagged_volatility > max_lagged_volatility).fillna(False)
    max_trade_notional = max_adv_participation * valid_adv
    clipped_notional = (target_notional - max_trade_notional).clip(lower=0.0)
    entry_dollar_volume = execution["entry_open"] * execution["entry_volume"]
    ex_post_participation = target_notional / entry_dollar_volume.where(entry_dollar_volume > 0.0)
    fillable = execution["open_to_open_return"].notna()

    return {
        "fillable_count": int(fillable.sum()),
        "unfillable_count": int((~fillable).sum()),
        "missing_entry_open_count": int(execution["entry_open"].isna().sum()),
        "missing_exit_open_count": int(execution["exit_open"].isna().sum()),
        "max_participation": _safe_float(participation.max(skipna=True)),
        "median_participation": _safe_float(participation.median(skipna=True)),
        "capacity_breach_count": int(adv_breaches.sum()),
        "missing_adv_count": int(valid_adv.isna().sum()),
        "clipped_count": int((clipped_notional > 0.0).sum()),
        "clipped_total_notional": _safe_float(clipped_notional.sum(skipna=True)),
        "max_lagged_volatility": _safe_float(lagged_volatility.max(skipna=True)),
        "max_lagged_volatility_threshold": _safe_float(max_lagged_volatility),
        "volatility_breach_count": int(volatility_breaches.sum()),
        "gate_breach_count": int((adv_breaches | volatility_breaches).sum()),
        "missing_volatility_count": int(lagged_volatility.isna().sum()),
        "max_ex_post_entry_participation": _safe_float(ex_post_participation.max(skipna=True)),
    }


def _next_market_date(
    date: pd.Timestamp | None, market_dates: list[pd.Timestamp]
) -> pd.Timestamp | None:
    if date is None:
        return None
    timestamp = pd.Timestamp(date)
    for candidate in market_dates:
        if pd.Timestamp(candidate) > timestamp:
            return pd.Timestamp(candidate)
    return None


def _transaction_cost(turnover: float, *, spread_bps: float, slippage_bps: float) -> float:
    spread = spread_bps / 10_000.0
    slippage = slippage_bps / 10_000.0
    return float(turnover * (spread + 2.0 * slippage))


def _safe_float(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _format_date(date: pd.Timestamp | None) -> str | None:
    return None if date is None else pd.Timestamp(date).strftime("%Y-%m-%d")
