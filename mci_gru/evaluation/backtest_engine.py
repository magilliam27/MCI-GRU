"""
Saved-prediction replay for MCI-GRU runs (#339 inputs, #340 strategy and metrics).

The replay trades a run's averaged predictions on its own data package and reports the
book's performance. The book itself lives in ``mci_gru/evaluation/replay_book.py``.

Strategy (#340, Mac's E2-E4):
- Default ``close_tranches``: buy at close D+1, sell at close D+5 (the training label's
  legs), in four staggered tranches with buy-and-hold weights inside each tranche.
  This is the return the model was trained on, tradable with market-on-close orders.
- ``open_to_open`` (labelled variant): the paper's buy at open D+1, sell at open D+2.
  ``--holding_period N`` holds N sessions in N staggered sleeves, or one sleeve with
  ``--rebalance_style block``.
- Returns, exits and ties come from the evaluation core (``outcomes.py``): prices on
  the panel's own session axis; a holding with no exit price books its last price and
  holds cash to the exit (E3), and the event is logged; ties by score, then kdcode.
- At most one share class per issuer is held; the benchmark weights the issuer once (E4).
- Costs (``--transaction_costs``) are half the round-trip spread plus slippage on each
  side, charged on the session the trade happens.
- Price returns only (E6): the data package carries no dividends.

Benchmark: the equal-weight PIT universe, on the candidates' signal date and the same
tranche schedule. Not the published S&P 500 (cap-weighted).

Metrics, all on one daily series (the book's sessions from first entry to last exit):
- ARR (geometric), AVoL, ``sharpe`` (mean / sd x sqrt 252), ``ASR`` (the paper's
  ARR / AVoL), MDD (from a starting value of 1.0), CR, IR (annualised).
- ``top_k_excess_period_return``: mean over booked tranches of the top-k hold return
  over the universe's, gross of costs, with a Hansen-Hodrick t-stat; with one tranche
  per held session (the default) it is the in-run summary's number for the same legs.
- Multiple testing: Holm or Bonferroni for one strategy; BHY only with a trial ledger
  holding every other trial's p-value.
- No MSE or MAE: they measure the scale of IC-loss scores, not accuracy (E7).

Inputs (#339):
- ``--run_dir`` (one window's run_experiment.py output) supplies the data file, PIT
  file and its export cutoff, test window, label, k and members, each checked against
  what the run recorded (``mci_gru/evaluation/replay_inputs.py``). Other input flags
  are checked overrides.
- Without ``--run_dir`` every input must be named explicitly; nothing defaults to a
  paper-era file, window or k, and nothing is checked against a run.
- Every output records the SHA-256 of the data file, the PIT file and each prediction file.

Usage:
    # Replay a run from its own directory (no transaction costs)
    python scripts/backtest_sp500.py --run_dir <run>/ --top_k 50 --auto_save

    # With transaction costs (retail investor model), in basis points
    python scripts/backtest_sp500.py --run_dir <run>/ --top_k 50 --transaction_costs \
        --spread 15 --slippage 3

    # The paper's daily open-to-open variant
    python scripts/backtest_sp500.py --run_dir <run>/ --top_k 50 --strategy open_to_open
"""

import argparse
import hashlib
import json
import logging
import os
import warnings
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from mci_gru.data.path_resolver import resolve_project_data_path  # noqa: E402
from mci_gru.data.pit import normalise_pit_intervals  # noqa: E402
from mci_gru.data.preprocessing import return_basis_panel  # noqa: E402
from mci_gru.data.quality_contract import fill_open_valid_to  # noqa: E402
from mci_gru.evaluation.paired_inference import bhy_adjusted_p_values  # noqa: E402
from mci_gru.evaluation.replay_book import (  # noqa: E402
    CLOSE_TRANCHES,
    OPEN_TO_OPEN,
    STRATEGIES,
    CostModel,
    ReplayBook,
    ReplayStrategy,
    build_replay_book,
)
from mci_gru.evaluation.replay_inputs import (  # noqa: E402
    ReplayInputError,
    iso_date,
    resolve_replay_inputs,
)
from mci_gru.evaluation.statistics import hansen_hodrick_mean_inference  # noqa: E402
from mci_gru.tracking import (  # noqa: E402
    MLflowTrackingManager,
    load_run_metadata_from_predictions_dir,
)
from mci_gru.utils.hashing import sha256_file  # noqa: E402

# ============================================================================
# CONFIGURATION
# ============================================================================

# Strategy defaults only. Inputs (data file, PIT file, window, label, k) have no
# defaults: they come from the run (--run_dir) or are named explicitly (#339).
DEFAULT_CONFIG = {
    "trading_days_per_year": 252,
    "strategy": CLOSE_TRANCHES,
    # Transaction costs for retail investor
    "transaction_costs": {
        "enabled": False,  # Disabled by default (paper excludes costs)
        "bid_ask_spread": 0.0010,  # 10 bps round-trip (5 bps each side)
        "slippage": 0.0005,  # 5 bps per trade
    },
    # Rank-drop sell gate: a held name is kept unless its rank worsened by
    # >= min_rank_drop since its sleeve's previous signal date
    "rank_drop_gate": {
        "enabled": False,
        "min_rank_drop": 10,
    },
}

RETURN_BASIS = "price"
BENCHMARK = "equal_weight_pit_universe_issuer_once"


# ============================================================================
# METRICS (one daily series)
# ============================================================================


def _finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def calculate_arr(daily_returns: Any, trading_days: int = 252) -> float:
    """Annualised rate of return, geometric: ``prod(1 + r) ** (252 / T) - 1``."""
    r = _finite(daily_returns)
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) ** (trading_days / r.size) - 1.0)


def calculate_avol(daily_returns: Any, trading_days: int = 252) -> float:
    """Annualised volatility: ``std(r, ddof=1) * sqrt(252)``."""
    r = _finite(daily_returns)
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(trading_days))


def calculate_sharpe(daily_returns: Any, trading_days: int = 252) -> float:
    """Standard annualised Sharpe, no risk-free rate: ``mean / std(ddof=1) * sqrt(252)``."""
    r = _finite(daily_returns)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(trading_days))


def calculate_mdd(daily_returns: Any) -> float:
    """Maximum drawdown of the wealth path that starts at 1.0, as a negative number.

    The starting value counts as a peak, so a loss on the first session is a drawdown.
    """
    r = _finite(daily_returns)
    if r.size == 0:
        return 0.0
    values = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peaks = np.maximum.accumulate(values)
    return -float(np.max((peaks - values) / peaks))


def drawdown_path(daily_returns: Any) -> np.ndarray:
    """Drawdown after each session, from a starting value of 1.0 (<= 0)."""
    values = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(daily_returns, dtype=float))])
    peaks = np.maximum.accumulate(values)
    return (values / peaks - 1.0)[1:]


def calculate_asr(arr: float, avol: float) -> float:
    """The paper's ASR: geometric ARR over AVoL. Not a standard Sharpe; see ``sharpe``."""
    if avol == 0 or np.isnan(avol):
        return 0.0
    return arr / avol


def calculate_cr(arr: float, mdd: float) -> float:
    """Calmar ratio: ARR / |MDD|."""
    if mdd == 0 or np.isnan(mdd):
        return 0.0
    return arr / abs(mdd)


def calculate_ir(portfolio_returns: Any, benchmark_returns: Any, trading_days: int = 252) -> float:
    """Annualised information ratio: ``mean(r - b) / std(r - b) * sqrt(252)``.

    The two series must share sessions; they are never truncated to align.
    """
    r = np.asarray(portfolio_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    if r.shape != b.shape:
        raise ValueError(f"portfolio {r.shape} and benchmark {b.shape} must share sessions")
    return calculate_sharpe(r - b, trading_days)


def portfolio_metrics(
    daily_returns: Any, benchmark_returns: Any, trading_days: int = 252
) -> dict[str, float]:
    """The paper's metrics and a standard Sharpe, all on one daily series."""
    r = np.asarray(daily_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    arr = calculate_arr(r, trading_days)
    avol = calculate_avol(r, trading_days)
    mdd = calculate_mdd(r)
    total = float(np.prod(1.0 + r) - 1.0) if r.size else 0.0
    bench = float(np.prod(1.0 + b) - 1.0) if b.size else 0.0
    return {
        "ARR": arr,
        "AVoL": avol,
        "sharpe": calculate_sharpe(r, trading_days),
        "ASR": calculate_asr(arr, avol),
        "MDD": mdd,
        "CR": calculate_cr(arr, mdd),
        "IR": calculate_ir(r, b, trading_days),
        "total_return": total,
        "benchmark_return": bench,
        # Difference of cumulative returns, and relative wealth (the in-run summary's
        # ``top_<k>_excess_cumulative_return``).
        "excess_return": total - bench,
        "excess_cumulative_return": (1.0 + total) / (1.0 + bench) - 1.0,
    }


def monthly_performance(
    sessions: list[str], portfolio_returns: Any, benchmark_returns: Any
) -> pd.DataFrame:
    """Calendar-month returns, compounded from the daily series."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(list(sessions)),
            "portfolio_return": np.asarray(portfolio_returns, dtype=np.float64),
            "benchmark_return": np.asarray(benchmark_returns, dtype=np.float64),
        }
    )
    frame["year_month"] = frame["date"].dt.to_period("M")
    grouped = frame.groupby("year_month")
    out = pd.DataFrame(
        {
            "portfolio_return": grouped["portfolio_return"].apply(lambda r: np.prod(1 + r) - 1),
            "portfolio_volatility": grouped["portfolio_return"].std(),
            "trading_days": grouped["portfolio_return"].count(),
            "benchmark_return": grouped["benchmark_return"].apply(lambda r: np.prod(1 + r) - 1),
        }
    )
    out["excess_return"] = out["portfolio_return"] - out["benchmark_return"]
    return out.reset_index().assign(year_month=lambda f: f["year_month"].astype(str))


# ============================================================================
# MULTIPLE TESTING (Harvey & Liu, 2014)
# ============================================================================
# One strategy: Holm or Bonferroni (identical for the best of M). BHY is a step-up
# procedure over every trial's p-value, so it is applied only with a trial ledger.


def calculate_t_statistic(sharpe_ratio: float, num_years: float) -> float:
    """
    Convert Sharpe Ratio to t-statistic.

    Paper formula: t-statistic = Sharpe Ratio × √(Number of years)

    This allows us to assess statistical significance of the strategy.
    A t-stat > 2.0 is traditionally considered significant for a single test,
    but with multiple tests, higher thresholds are required.

    Args:
        sharpe_ratio: Annualized Sharpe Ratio
        num_years: Number of years in the test period

    Returns:
        t-statistic value
    """
    if num_years <= 0:
        return 0.0
    return sharpe_ratio * np.sqrt(num_years)


def t_stat_to_p_value(t_stat: float) -> float:
    """
    Convert t-statistic to two-tailed p-value.

    Args:
        t_stat: t-statistic value

    Returns:
        Two-tailed p-value
    """
    # Use large degrees of freedom (approximates normal distribution)
    return 2 * (1 - stats.norm.cdf(abs(t_stat)))


def p_value_to_t_stat(p_value: float) -> float:
    """
    Convert p-value to t-statistic.

    Args:
        p_value: p-value (two-tailed)

    Returns:
        t-statistic value
    """
    if p_value >= 1.0:
        return 0.0
    if p_value <= 0.0:
        return np.inf
    return stats.norm.ppf(1 - p_value / 2)


def adjust_p_value_bonferroni(p_value: float, num_tests: int) -> float:
    """
    Bonferroni p-value adjustment (most stringent FWER method).

    The Bonferroni method controls the Family-Wise Error Rate (FWER),
    meaning it's unacceptable to make even a single false discovery.

    This is the most conservative approach - appropriate for high-stakes
    decisions (e.g., space missions) but may be too stringent for trading.

    Paper formula: adjusted_p = original_p × M
    Equivalently: significance threshold = α / M

    Example: With 200 tests and α=0.05, you need p < 0.00025 to be significant.

    Args:
        p_value: Original p-value
        num_tests: Total number of tests (M)

    Returns:
        Adjusted p-value
    """
    if num_tests <= 1:
        return p_value

    adjusted_p = min(p_value * num_tests, 1.0)
    return adjusted_p


def adjust_p_value_holm(p_value: float, rank: int, num_tests: int) -> float:
    """
    Holm p-value adjustment (step-down FWER method).

    The Holm method is less stringent than Bonferroni but still controls
    the Family-Wise Error Rate. It uses information from the distribution
    of test statistics to provide a sharper test.

    Paper formula: threshold_k = α / (M + 1 - k)
    where k is the rank (1 = most significant)

    The procedure:
    1. Sort tests by p-value (lowest to highest)
    2. Starting from the MOST significant, compare to threshold
    3. When a test fails its threshold, reject it and all less significant tests

    Args:
        p_value: Original p-value
        rank: Rank of this test when sorted by p-value (1 = most significant)
        num_tests: Total number of tests (M)

    Returns:
        Adjusted p-value
    """
    if num_tests <= 1:
        return p_value

    # Holm adjustment factor: (M + 1 - rank)
    # Note: For rank=1 (most significant), this equals Bonferroni
    adjustment_factor = num_tests + 1 - rank
    adjusted_p = min(p_value * adjustment_factor, 1.0)

    return adjusted_p


ADJUSTMENT_METHODS = ("holm", "bonferroni", "bhy")


def read_trial_ledger_p_values(path: str, column: str = "p_value") -> np.ndarray:
    """Every other trial's p-value from a trial ledger CSV (one row per trial)."""
    frame = pd.read_csv(path)
    if column not in frame.columns:
        raise ValueError(f"trial ledger {path} has no {column!r} column")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError(f"trial ledger {path}: every {column!r} must be a p-value in [0, 1]")
    return values


def multiple_testing(
    sharpe: float,
    num_years: float,
    num_tests: int = 1,
    method: str = "holm",
    *,
    trial_ledger: str | None = None,
    p_value_column: str = "p_value",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Harvey-Liu haircut of an annualised Sharpe tested among ``num_tests`` trials.

    ``holm`` and ``bonferroni`` need only the count (for the best of M they agree).
    ``bhy`` needs every trial's p-value: ``trial_ledger`` lists the other trials, and
    the count is the ledger's rows plus this one.
    """
    method = method.lower()
    if method not in ADJUSTMENT_METHODS:
        raise ValueError(f"adjustment method must be one of {ADJUSTMENT_METHODS}, got {method!r}")
    t_stat = calculate_t_statistic(sharpe, num_years)
    original_p = t_stat_to_p_value(t_stat)
    if method == "bhy":
        if trial_ledger is None:
            raise ValueError(
                "BHY adjusts a family of p-values; pass the trial ledger with every other "
                "trial's p-value (--trial_ledger), or use holm/bonferroni for one strategy"
            )
        others = read_trial_ledger_p_values(trial_ledger, p_value_column)
        if num_tests not in (1, others.size + 1):
            raise ValueError(
                f"num_tests {num_tests} differs from the trial ledger's {others.size + 1} trials"
            )
        num_tests = others.size + 1
        adjusted_p = float(bhy_adjusted_p_values(np.concatenate([[original_p], others]))[0])
    elif num_tests <= 1:
        adjusted_p = original_p
    elif method == "bonferroni":
        adjusted_p = adjust_p_value_bonferroni(original_p, num_tests)
    else:
        adjusted_p = adjust_p_value_holm(original_p, 1, num_tests)

    adjusted_t = p_value_to_t_stat(adjusted_p)
    haircut = adjusted_t / np.sqrt(num_years) if num_years > 0 else 0.0
    haircut = max(0.0, min(haircut, abs(sharpe)))
    if sharpe < 0:
        haircut = -haircut
    return {
        "haircutted_sharpe": haircut,
        "original_sharpe": sharpe,
        "t_statistic": t_stat,
        "original_p_value": original_p,
        "adjusted_p_value": adjusted_p,
        "adjusted_t_statistic": adjusted_t,
        "is_significant": adjusted_p < alpha,
        "haircut_pct": (abs(sharpe) - abs(haircut)) / abs(sharpe) * 100 if sharpe else 0.0,
        "method": method,
        "num_tests": num_tests,
        "num_years": num_years,
    }


# ============================================================================
# DATA LOADING
# ============================================================================


def resolve_data_file(filename: str) -> str:
    """Resolve the named data file, as given or relative to the project root.

    A same-named file elsewhere is never substituted (#339, the #223 class).
    """
    return str(resolve_project_data_path(filename, allow_basename_fallback=False))


def load_stock_data(
    filename: str, start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    """
    Load stock data from CSV file.

    Args:
        filename: Path to stock data CSV
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)

    Returns:
        DataFrame with stock data
    """
    resolved_file = resolve_data_file(filename)
    print(f"Loading stock data from {resolved_file}...")
    df = pd.read_csv(resolved_file)

    # Ensure date format
    df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")

    # Filter by date range if specified
    if start_date:
        df = df[df["dt"] >= start_date]
    if end_date:
        df = df[df["dt"] <= end_date]

    # Sort by stock and date
    df = df.sort_values(["kdcode", "dt"]).reset_index(drop=True)

    print(f"  Loaded {len(df)} rows, {df['kdcode'].nunique()} stocks")
    print(f"  Date range: {df['dt'].min()} to {df['dt'].max()}")

    return df


def load_pit_universe_for_backtest(
    csv_path: str | None, *, open_valid_to: str | None = None
) -> pd.DataFrame | None:
    """Load optional PIT intervals used to restrict candidates and benchmark rows.

    A blank ``valid_to`` is membership through ``open_valid_to`` (the run's
    ``data.pit_export_cutoff``), as in training. With open-ended rows and no cutoff
    the load is refused: dropping them would remove every current member.
    """
    if not csv_path:
        return None
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"PIT universe CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    valid_to = next((c for c in frame.columns if str(c).strip().lower() == "valid_to"), None)
    if valid_to is not None and open_valid_to is None:
        open_ended = frame[valid_to].isna() | (frame[valid_to].astype(str).str.strip() == "")
        if open_ended.any():
            raise ValueError(
                f"PIT universe CSV {csv_path} has {int(open_ended.sum())} open-ended rows "
                "(blank valid_to); pass the run's pit_export_cutoff so they keep their members"
            )
    return normalise_pit_intervals(fill_open_valid_to(frame, open_valid_to))


def replay_universe(config: dict, predictions_df: pd.DataFrame) -> pd.DataFrame | None:
    """The universe a replay admits candidates from and benchmarks against.

    The PIT file's intervals (restricted to the run's members when known); with no
    PIT file, the run's members over the whole panel; with neither, the names the
    predictions score. So the benchmark never averages a name the run could not rank.
    """
    members = config.get("members")
    pit = load_pit_universe_for_backtest(
        config.get("pit_universe_csv"), open_valid_to=config.get("pit_export_cutoff")
    )
    if pit is not None:
        if members is not None:
            pit = pit[pit["kdcode"].isin({str(m) for m in members})].reset_index(drop=True)
        return pit
    if members is None:
        if "kdcode" not in predictions_df.columns or predictions_df.empty:
            return None
        members = sorted(predictions_df["kdcode"].astype(str).unique())
    return pd.DataFrame(
        {"kdcode": [str(m) for m in members], "valid_from": "0000-01-01", "valid_to": "9999-12-31"}
    )


def _pit_active_kdcodes_on_date(pit_universe_df: pd.DataFrame | None, date: str) -> set[str] | None:
    if pit_universe_df is None:
        return None
    mask = (pit_universe_df["valid_from"] <= date) & (pit_universe_df["valid_to"] >= date)
    return set(pit_universe_df.loc[mask, "kdcode"].astype(str))


def load_predictions(predictions_dir: str) -> pd.DataFrame:
    """
    Load model predictions from CSV files.

    Each file is named as YYYY-MM-DD.csv with columns: kdcode, dt, score

    Args:
        predictions_dir: Directory containing prediction CSV files

    Returns:
        DataFrame with all predictions
    """
    print(f"Loading predictions from {predictions_dir}...")

    pred_files = sorted(glob(os.path.join(predictions_dir, "*.csv")))

    if len(pred_files) == 0:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    all_preds = []
    for f in pred_files:
        df = pd.read_csv(f)
        all_preds.append(df)

    predictions_df = pd.concat(all_preds, ignore_index=True)
    predictions_df["dt"] = pd.to_datetime(predictions_df["dt"]).dt.strftime("%Y-%m-%d")

    print(f"  Loaded {len(predictions_df)} predictions")
    print(f"  {len(pred_files)} dates, {predictions_df['kdcode'].nunique()} stocks")
    print(f"  Date range: {predictions_df['dt'].min()} to {predictions_df['dt'].max()}")

    return predictions_df


def read_issuer_map(path: str) -> dict[str, str]:
    """kdcode -> issuer from a CSV with ``kdcode`` and ``issuer`` columns."""
    frame = pd.read_csv(path, dtype=str)
    missing = {"kdcode", "issuer"} - set(frame.columns)
    if missing:
        raise ValueError(f"issuer map {path} lacks column(s) {sorted(missing)}")
    return dict(zip(frame["kdcode"].str.strip(), frame["issuer"].str.strip(), strict=True))


def prediction_file_digests(predictions_dir: str) -> dict[str, str]:
    """SHA-256 of every prediction CSV, by file name."""
    return {
        Path(f).name: sha256_file(f) for f in sorted(glob(os.path.join(predictions_dir, "*.csv")))
    }


def input_record(predictions_dir: str, config: dict) -> dict[str, Any]:
    """What the replay read, by path and SHA-256, for every output it writes."""
    pit = config.get("pit_universe_csv")
    issuer_map = config.get("issuer_map")
    return {
        "run_dir": config.get("run_dir"),
        "data_file": str(config["data_file"]),
        "data_file_sha256": sha256_file(resolve_data_file(config["data_file"])),
        "pit_universe_csv": str(pit) if pit else None,
        "pit_universe_sha256": sha256_file(pit) if pit else None,
        "pit_export_cutoff": config.get("pit_export_cutoff"),
        "issuer_map": str(issuer_map) if issuer_map else None,
        "issuer_map_sha256": sha256_file(issuer_map) if issuer_map else None,
        "predictions_dir": str(predictions_dir),
        "prediction_files": prediction_file_digests(predictions_dir),
    }


# ============================================================================
# THE REPLAY
# ============================================================================


def replay_strategy(config: dict) -> ReplayStrategy:
    """The strategy ``config`` names; ``close_tranches`` unless told otherwise."""
    name = config.get("strategy") or CLOSE_TRANCHES
    holding_period = config.get("holding_period")
    rebalance_style = config.get("rebalance_style")
    if name == OPEN_TO_OPEN:
        if config.get("tranches") is not None:
            raise ValueError(
                "--tranches shapes close_tranches; the open_to_open variant takes "
                "--holding_period and --rebalance_style"
            )
        return ReplayStrategy.open_to_open(holding_period or 1, rebalance_style or "staggered")
    if name != CLOSE_TRANCHES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {name!r}")
    if holding_period not in (None, 1) or rebalance_style is not None:
        raise ValueError(
            "--holding_period and --rebalance_style shape the open_to_open variant; "
            "close_tranches holds the label's sessions in --tranches sleeves"
        )
    return ReplayStrategy.close_tranches(int(config["label_t"]), config.get("tranches"))


def _cost_model(config: dict) -> CostModel:
    tc = config.get("transaction_costs") or {}
    if not tc.get("enabled", False):
        return CostModel()
    return CostModel(
        bid_ask_spread=float(tc.get("bid_ask_spread", 0.001)),
        slippage=float(tc.get("slippage", 0.0005)),
    )


def _min_rank_drop(config: dict) -> int | None:
    gate = config.get("rank_drop_gate") or {}
    return int(gate.get("min_rank_drop", 10)) if gate.get("enabled", False) else None


def replay(predictions_dir: str, config: dict) -> tuple[dict[str, Any], ReplayBook]:
    """Trade the saved predictions under ``config`` and score the book.

    Returns the metrics and the book every output file is written from.
    """
    if config is None:
        raise ValueError(
            "replay() needs a config naming its inputs (data_file, test_start, test_end, "
            "top_k, label_t); build one from a run with resolve_replay_inputs (#339)"
        )
    strategy = replay_strategy(config)
    start, end = iso_date(config["test_start"]), iso_date(config["test_end"])

    predictions = load_predictions(predictions_dir)
    predictions["kdcode"] = predictions["kdcode"].astype(str)
    predictions = predictions[(predictions["dt"] >= start) & (predictions["dt"] <= end)]
    if predictions.empty:
        raise ValueError(f"no predictions inside the test window {start}..{end}")
    universe = replay_universe(config, predictions)
    kdcodes = sorted(universe["kdcode"].astype(str).unique())
    dates = sorted(predictions["dt"].unique())

    # Exits of the window's last signals fall after test_end, so the panel runs on. The
    # session axis is the universe's own, as in training and the in-run summary.
    panel = load_stock_data(config["data_file"], start_date=start)
    panel = panel[panel["kdcode"].astype(str).isin(kdcodes)]
    # Closes on the run's return basis, so the book measures what its labels did (#361).
    panel = return_basis_panel(panel, config.get("return_basis", "price"))
    scores = (
        predictions.pivot_table(index="dt", columns="kdcode", values="score", aggfunc="mean")
        .reindex(index=dates, columns=kdcodes)
        .to_numpy(dtype=np.float64)
    )
    members = np.zeros(scores.shape, dtype=bool)
    position = {k: j for j, k in enumerate(kdcodes)}
    for i, date in enumerate(dates):
        for kdcode in _pit_active_kdcodes_on_date(universe, date) or ():
            members[i, position[kdcode]] = True

    print(f"\nReplaying {strategy.describe()}; top-{config['top_k']}")
    book = build_replay_book(
        panel,
        scores,
        dates,
        kdcodes,
        strategy=strategy,
        top_k=int(config["top_k"]),
        members=members,
        costs=_cost_model(config),
        min_rank_drop=_min_rank_drop(config),
        issuers=read_issuer_map(config["issuer_map"]) if config.get("issuer_map") else None,
    )
    results = replay_metrics(book, config)
    results["inputs"] = input_record(predictions_dir, config)
    return results, book


def evaluate(predictions_dir: str, config: dict | None = None) -> dict[str, Any]:
    """The replay's metrics (see :func:`replay` for the book as well)."""
    results, _ = replay(predictions_dir, config)
    return results


def replay_metrics(book: ReplayBook, config: dict) -> dict[str, Any]:
    """Every reported number, on the book's one daily series."""
    days = int(config.get("trading_days_per_year", 252))
    tc_enabled = bool((config.get("transaction_costs") or {}).get("enabled", False))
    results: dict[str, Any] = {
        **book.strategy.to_dict(),
        "strategy_description": book.strategy.describe(),
        "return_basis": RETURN_BASIS,
        "benchmark": BENCHMARK,
        "top_k": book.top_k,
    }
    if not book.sessions:
        raise ValueError("the replay opened no tranche: no signal date has a resolved hold")
    results.update(portfolio_metrics(book.net_returns, book.benchmark_returns, days))

    excess = book.excess_period_returns
    finite = excess[np.isfinite(excess)]
    inference = hansen_hodrick_mean_inference(finite, label_horizon=book.strategy.tranches)
    events = {"missing_entry": 0, "missing_exit": 0}
    for event in book.events:
        events[event["kind"]] += 1
    results.update(
        {
            # Per-hold returns are gross of costs; the daily metrics above are net.
            "period_return_basis": "gross",
            "top_k_abs_period_return": _mean(book.period_returns),
            "benchmark_period_return": _mean(book.benchmark_period_returns),
            "top_k_excess_period_return": _mean(excess),
            "excess_t_stat": inference.t_stat,
            "excess_p_value": inference.p_value,
            "excess_inference": {"method": inference.method, "lags": inference.lags},
            "num_trading_days": len(book.sessions),
            "start_date": book.sessions[0],
            "end_date": book.sessions[-1],
            "first_signal_date": book.signal_dates[0] if book.signal_dates else None,
            "last_signal_date": book.signal_dates[-1] if book.signal_dates else None,
            "tranches_opened": book.tranches_opened,
            "tranches_unbooked": book.tranches_unbooked,
            "missing_close_events": events,
            "avg_daily_turnover": float(np.mean(book.turnover)),
            "rank_gate_enabled": book.rank_gate.get("enabled", False),
            "min_rank_drop": book.rank_gate.get("min_rank_drop"),
            "rank_gate_exits": book.rank_gate.get("gate_exits", 0),
            "transaction_costs_enabled": tc_enabled,
        }
    )
    if tc_enabled:
        tc = config["transaction_costs"]
        gross = portfolio_metrics(book.gross_returns, book.benchmark_returns, days)
        results.update(
            {
                "tc_bid_ask_spread_bps": float(tc.get("bid_ask_spread", 0.0)) * 10000,
                "tc_slippage_bps": float(tc.get("slippage", 0.0)) * 10000,
                "total_transaction_cost": float(np.sum(book.costs)),
                "avg_daily_cost_bps": float(np.mean(book.costs)) * 10000,
                "total_trades": len(book.trades),
                "gross_ARR": gross["ARR"],
                "gross_total_return": gross["total_return"],
                "net_ARR": results["ARR"],
                "net_total_return": results["total_return"],
                "cost_drag_ARR": gross["ARR"] - results["ARR"],
            }
        )
    return results


def _mean(values: Any) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite.size else float("nan")


def print_results(
    results: dict[str, Any],
    model_name: str = "MCI-GRU",
    num_tests: int = 1,
    adjustment_method: str = "holm",
    trial_ledger: str | None = None,
) -> None:
    """Print the replay's metrics."""
    line = "  " + "-" * 66
    print("\n" + "=" * 70)
    print(f"  REPLAY RESULTS: {model_name}")
    print("=" * 70)
    print(f"  Strategy: {results['strategy_description']}")
    print(f"  Benchmark: {results['benchmark']} ({results['return_basis']} returns)")
    print(
        f"  Sessions: {results['start_date']} to {results['end_date']} "
        f"({results['num_trading_days']})"
    )
    print(line)
    rows = [
        ("ARR (geometric)", "ARR"),
        ("AVoL", "AVoL"),
        ("Sharpe (mean/sd)", "sharpe"),
        ("ASR (paper: ARR/AVoL)", "ASR"),
        ("MDD (from 1.0)", "MDD"),
        ("CR (Calmar)", "CR"),
        ("IR (annualised)", "IR"),
        ("Total return", "total_return"),
        ("Benchmark return", "benchmark_return"),
        ("Excess (relative wealth)", "excess_cumulative_return"),
        ("Top-k excess per hold", "top_k_excess_period_return"),
        ("  HH t-stat", "excess_t_stat"),
    ]
    for label, key in rows:
        value = results.get(key)
        shown = "n/a" if value is None else f"{value:12.4f}"
        print(f"  {label:<28} {shown:>12}")
    print(line)
    significance = multiple_testing(
        results["sharpe"],
        results["num_trading_days"] / 252,
        num_tests,
        adjustment_method,
        trial_ledger=trial_ledger,
    )
    print(f"  Sharpe t-statistic           {significance['t_statistic']:12.4f}  (SR x sqrt(years))")
    print(f"  p-value                      {significance['original_p_value']:12.6f}")
    if significance["num_tests"] > 1:
        print(
            f"  {significance['method'].upper()} over {significance['num_tests']} tests: "
            f"adjusted p {significance['adjusted_p_value']:.6f}, haircut Sharpe "
            f"{significance['haircutted_sharpe']:.4f}"
        )
    events = results["missing_close_events"]
    print(
        f"  Missing closes among holdings: {events['missing_exit']} exits (booked last close), "
        f"{events['missing_entry']} entries (held cash)"
    )
    if results.get("transaction_costs_enabled"):
        print(
            f"  Costs: {results['tc_bid_ask_spread_bps']:.1f} bps spread, "
            f"{results['tc_slippage_bps']:.1f} bps slippage; gross ARR "
            f"{results['gross_ARR']:.4f}, net ARR {results['net_ARR']:.4f}"
        )
    else:
        print("  Transaction costs not included (use --transaction_costs to enable)")
    print("=" * 70 + "\n")


def significance_fields(
    results: dict[str, Any],
    num_tests: int = 1,
    adjustment_method: str = "holm",
    trial_ledger: str | None = None,
) -> dict[str, Any]:
    """The multiple-testing columns saved beside the metrics."""
    s = multiple_testing(
        results["sharpe"],
        results["num_trading_days"] / 252,
        num_tests,
        adjustment_method,
        trial_ledger=trial_ledger,
    )
    return {
        "t_statistic": s["t_statistic"],
        "original_p_value": s["original_p_value"],
        "num_tests": s["num_tests"],
        "adjustment_method": s["method"],
        "adjusted_p_value": s["adjusted_p_value"],
        "haircutted_sharpe": s["haircutted_sharpe"],
        "haircut_pct": s["haircut_pct"],
        "is_significant": s["is_significant"],
    }


def save_results(
    results: dict[str, Any],
    output_path: str,
    num_tests: int = 1,
    adjustment_method: str = "holm",
    trial_ledger: str | None = None,
) -> None:
    """Save the metrics and their significance as a one-row CSV."""
    flat = {k: v for k, v in results.items() if not isinstance(v, dict)}
    flat.update(results["missing_close_events"])
    inputs = results.get("inputs") or {}
    flat.update(
        {
            "data_file_sha256": inputs.get("data_file_sha256"),
            "pit_universe_sha256": inputs.get("pit_universe_sha256"),
            "issuer_map_sha256": inputs.get("issuer_map_sha256"),
            # One digest over every prediction file's name and SHA-256 (listed in full
            # in backtest_metrics.json).
            "prediction_files_sha256": hashlib.sha256(
                json.dumps(inputs.get("prediction_files") or {}, sort_keys=True).encode()
            ).hexdigest(),
        }
    )
    flat.update(significance_fields(results, num_tests, adjustment_method, trial_ledger))
    pd.DataFrame([flat]).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def derive_holdings_summary(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """Per-stock statistics across the tranches that held it."""
    output_cols = [
        "kdcode",
        "times_held",
        "avg_score",
        "avg_return",
        "total_contribution",
        "win_rate",
    ]
    if holdings_df is None or len(holdings_df) == 0:
        return pd.DataFrame(columns=output_cols)
    summary_df = holdings_df.groupby("kdcode", as_index=False).agg(
        times_held=("stock_return", "count"),
        avg_score=("score", "mean"),
        avg_return=("stock_return", "mean"),
        total_contribution=("contribution", "sum"),
        win_rate=("stock_return", lambda x: float((x > 0).mean())),
    )
    summary_df = summary_df.sort_values("total_contribution", ascending=False)
    return summary_df.reset_index(drop=True).reindex(columns=output_cols)


def _jsonable_config(config: dict) -> dict:
    out = {}
    for key, value in config.items():
        out[key] = {str(k): v for k, v in value.items()} if isinstance(value, dict) else value
    return out


def save_backtest_results(
    results: dict,
    backtest_dir: str,
    book: ReplayBook | None = None,
    config: dict | None = None,
    num_tests: int = 1,
    adjustment_method: str = "holm",
    trial_ledger: str | None = None,
) -> str:
    """Write every output from one book: metrics, daily, cumulative, monthly, holdings."""
    print(f"\nSaving backtest results to: {backtest_dir}")
    save_results(
        results,
        os.path.join(backtest_dir, "backtest_results.csv"),
        num_tests,
        adjustment_method,
        trial_ledger,
    )
    with open(os.path.join(backtest_dir, "backtest_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    if config:
        with open(os.path.join(backtest_dir, "backtest_config.json"), "w") as f:
            json.dump(
                {**_jsonable_config(config), "inputs": results.get("inputs")},
                f,
                indent=2,
                default=str,
            )

    if book is not None and book.sessions:
        daily = pd.DataFrame(
            {
                "date": book.sessions,
                "portfolio_return": book.net_returns,
                "gross_portfolio_return": book.gross_returns,
                "transaction_cost": book.costs,
                "turnover": book.turnover,
                "benchmark_return": book.benchmark_returns,
                "excess_return": book.net_returns - book.benchmark_returns,
            }
        )
        daily.to_csv(os.path.join(backtest_dir, "daily_returns.csv"), index=False)
        portfolio_value = np.cumprod(1.0 + book.net_returns)
        benchmark_value = np.cumprod(1.0 + book.benchmark_returns)
        pd.DataFrame(
            {
                "date": book.sessions,
                "portfolio_value": portfolio_value,
                "benchmark_value": benchmark_value,
                "relative_performance": portfolio_value / benchmark_value,
                "portfolio_drawdown": drawdown_path(book.net_returns),
            }
        ).to_csv(os.path.join(backtest_dir, "cumulative_performance.csv"), index=False)
        monthly_performance(book.sessions, book.net_returns, book.benchmark_returns).to_csv(
            os.path.join(backtest_dir, "monthly_performance.csv"), index=False
        )
        holdings = pd.DataFrame(book.holdings)
        if len(holdings):
            holdings.to_csv(os.path.join(backtest_dir, "holdings.csv"), index=False)
            derive_holdings_summary(holdings).to_csv(
                os.path.join(backtest_dir, "holdings_summary.csv"), index=False
            )
        if book.trades:
            pd.DataFrame(book.trades).to_csv(
                os.path.join(backtest_dir, "trade_journal.csv"), index=False
            )
        pd.DataFrame(
            book.events,
            columns=["signal_date", "kdcode", "kind", "scheduled_session", "booked_session"],
        ).to_csv(os.path.join(backtest_dir, "missing_close_events.csv"), index=False)

    significance = significance_fields(results, num_tests, adjustment_method, trial_ledger)
    with open(os.path.join(backtest_dir, "summary.txt"), "w") as f:
        f.write("=" * 80 + "\nBACKTEST SUMMARY\n" + "=" * 80 + "\n\n")
        f.write(f"Strategy: {results['strategy_description']}\n")
        f.write(f"Benchmark: {results['benchmark']} ({results['return_basis']} returns)\n")
        f.write(f"Top-K: {results['top_k']}\n")
        f.write(f"Sessions: {results['start_date']} to {results['end_date']}\n")
        if results.get("transaction_costs_enabled"):
            f.write(
                f"Transaction Costs: {results['tc_bid_ask_spread_bps']:.1f} bps spread, "
                f"{results['tc_slippage_bps']:.1f} bps slippage\n"
            )
        else:
            f.write("Transaction Costs: Disabled\n")
        f.write(f"Rank-Drop Gate: {'Enabled' if results['rank_gate_enabled'] else 'Disabled'}\n\n")
        for key in (
            "ARR",
            "AVoL",
            "sharpe",
            "ASR",
            "MDD",
            "CR",
            "IR",
            "total_return",
            "benchmark_return",
            "excess_cumulative_return",
            "top_k_excess_period_return",
        ):
            f.write(f"  {key:<28} {results[key]:.6f}\n")
        f.write(
            f"\n  {'multiple testing':<28} {significance['adjustment_method']} over "
            f"{significance['num_tests']}: adjusted p {significance['adjusted_p_value']:.4f}\n"
        )
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
    print("All backtest outputs saved.")
    return backtest_dir


def plot_equity_curve(
    book: ReplayBook, output_path: str | None = None, title_suffix: str = ""
) -> None:
    """Portfolio, benchmark and relative wealth from the book's one daily series."""
    if output_path:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dates = pd.to_datetime(book.sessions)
    portfolio = np.cumprod(1.0 + book.net_returns)
    benchmark = np.cumprod(1.0 + book.benchmark_returns)
    relative = portfolio / benchmark
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    ax1 = axes[0]
    ax1.plot(dates, portfolio, "r-", label=f"Top-{book.top_k} book", linewidth=2)
    ax1.plot(dates, benchmark, "b-", label="Equal-weight PIT universe", linewidth=1.5)
    ax1.plot(dates, relative, "orange", label="Relative wealth", linewidth=1.5)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Value (start 1.0)")
    ax1.set_title(f"{book.strategy.describe()}{title_suffix}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    drawdown = drawdown_path(relative / np.concatenate([[1.0], relative[:-1]]) - 1.0)
    ax2.fill_between(dates, 0, drawdown * 100, color="red", alpha=0.3)
    ax2.set_ylabel("Relative drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Equity curve saved to {output_path}")
    else:
        plt.show()
    plt.close()


def setup_backtest_output_dir(predictions_dir: str, suffix: str = "") -> str:
    """
    Create backtest output directory alongside predictions.

    Args:
        predictions_dir: Path to averaged_predictions folder
        suffix: Optional suffix for multiple backtests (e.g., "_with_costs")

    Returns:
        Path to backtest output directory
    """
    # predictions_dir is typically: {output_dir}/{experiment_name}/{timestamp}/averaged_predictions
    # We want: {output_dir}/{experiment_name}/{timestamp}/backtest{suffix}

    parent_dir = os.path.dirname(predictions_dir)
    backtest_dir = os.path.join(parent_dir, f"backtest{suffix}")
    os.makedirs(backtest_dir, exist_ok=True)

    return backtest_dir


def setup_backtest_logging(backtest_dir: str) -> logging.Logger:
    """
    Setup logging for backtest evaluation.

    Args:
        backtest_dir: Directory for backtest outputs

    Returns:
        Configured logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(backtest_dir, f"backtest_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("backtest")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Backtest logging initialized: {log_file}")

    return logger


def _infer_experiment_name(predictions_dir: str) -> str:
    """Infer experiment name from a standard averaged_predictions path."""
    predictions_path = Path(predictions_dir).resolve()
    if predictions_path.parent.parent.exists():
        return predictions_path.parent.parent.name
    return "backtest"


def setup_backtest_tracking(
    predictions_dir: str,
    config: dict,
    enable_mlflow: bool = False,
    disable_mlflow_autolink: bool = False,
    tracking_uri: str = None,
    experiment_name: str = None,
    backtest_suffix: str = "",
) -> tuple:
    """Create an optional MLflow child run for backtest tracking.

    If the predictions directory belongs to a tracked training run, the
    backtest automatically links itself beneath that parent run.
    """
    linked_metadata = (
        None if disable_mlflow_autolink else load_run_metadata_from_predictions_dir(predictions_dir)
    )
    tracking_enabled = enable_mlflow or linked_metadata is not None

    if not tracking_enabled:
        return MLflowTrackingManager(enabled=False), linked_metadata

    resolved_tracking_uri = tracking_uri
    parent_run_id = None
    resolved_experiment_name = experiment_name

    if linked_metadata is not None:
        resolved_tracking_uri = resolved_tracking_uri or linked_metadata.get("tracking_uri")
        resolved_experiment_name = (
            resolved_experiment_name
            or linked_metadata.get("experiment_name")
            or _infer_experiment_name(predictions_dir)
        )
        parent_run_id = linked_metadata.get("run_id")
    else:
        resolved_experiment_name = resolved_experiment_name or _infer_experiment_name(
            predictions_dir
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"backtest{backtest_suffix or ''}-{timestamp}"

    manager = MLflowTrackingManager(
        enabled=True,
        tracking_uri=resolved_tracking_uri,
        experiment_name=resolved_experiment_name,
        run_name=run_name,
        output_path=Path(predictions_dir).resolve().parent,
        parent_run_id=parent_run_id,
        tags={
            "run_kind": "backtest",
            "linked_training_run": linked_metadata is not None,
            "predictions_dir": str(Path(predictions_dir).resolve()),
            "top_k": config.get("top_k"),
            "label_t": config.get("label_t"),
        },
    )
    return manager, linked_metadata


def _log_backtest_artifacts(tracking_manager: MLflowTrackingManager, backtest_dir: str):
    """Log the highest-value backtest artifacts."""
    if not tracking_manager.is_active:
        return

    artifact_names = [
        "backtest_results.csv",
        "backtest_metrics.json",
        "backtest_config.json",
        "summary.txt",
        "daily_returns.csv",
        "cumulative_performance.csv",
        "monthly_performance.csv",
        "equity_curve.png",
    ]
    for name in artifact_names:
        tracking_manager.log_artifact(Path(backtest_dir) / name, artifact_path="backtest_artifacts")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a run's saved predictions as a book")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="One window's run_experiment.py output directory. Supplies the data file, PIT "
        "file and export cutoff, test window, label, k and members, each checked against "
        "the run's records; the input flags below become checked overrides.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default=None,
        help="Directory containing prediction CSV files (default with --run_dir: "
        "<run_dir>/averaged_predictions; required without it)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Stock data CSV. With --run_dir it must carry the bytes the run consumed.",
    )
    parser.add_argument(
        "--pit_universe_csv",
        type=str,
        default=None,
        help="PIT kdcode/valid_from/valid_to CSV for candidate and benchmark membership. "
        "With --run_dir it must carry the bytes the run consumed.",
    )
    parser.add_argument(
        "--pit_export_cutoff",
        type=str,
        default=None,
        help="Membership end for PIT rows with a blank valid_to. With --run_dir it must "
        "equal the run's data.pit_export_cutoff.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Number of top stocks each tranche holds. With --run_dir it must be one of "
        "the run's evaluation.top_k_values, and is required unless the run reported one.",
    )
    parser.add_argument(
        "--test_start",
        type=str,
        default=None,
        help="First signal date. With --run_dir it must not precede the run's.",
    )
    parser.add_argument(
        "--test_end",
        type=str,
        default=None,
        help="Last signal date (exits after it are still booked). With --run_dir it must "
        "not follow the run's.",
    )
    parser.add_argument(
        "--label_t",
        type=int,
        default=None,
        help="The run's label horizon: close D+1 to close D+label_t. With --run_dir it "
        "must equal the run's.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=list(STRATEGIES),
        help="close_tranches (default): close D+1 to close D+label_t in staggered "
        "tranches, the trained return. open_to_open: the paper's open D+1 to open D+2 "
        "hold (implied by --holding_period or --rebalance_style).",
    )
    parser.add_argument(
        "--tranches",
        type=int,
        default=None,
        help="close_tranches: staggered sleeves; must divide the hold (default: one per "
        "held session, 4 for label_t=5). 1 rebalances the whole book every hold.",
    )
    parser.add_argument(
        "--holding_period",
        type=int,
        default=None,
        help="open_to_open: sessions each tranche holds (default 1, daily).",
    )
    parser.add_argument(
        "--rebalance_style",
        type=str,
        default=None,
        choices=["staggered", "block"],
        help="open_to_open with --holding_period > 1: 'staggered' = one sleeve per held "
        "session; 'block' = one sleeve rebalanced every holding_period signal dates.",
    )
    parser.add_argument(
        "--issuer_map",
        type=str,
        default=None,
        help="CSV of kdcode,issuer for share-class pairs beyond the built-in map "
        "(Alphabet, Fox, News Corp, Discovery, Under Armour, Liberty Global)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output path for results CSV")
    parser.add_argument("--plot", action="store_true", help="Generate equity curve plot")
    parser.add_argument(
        "--num_tests",
        type=int,
        default=1,
        help="Number of strategies/configurations tested, for the multiple-testing "
        "haircut (default: 1 = no adjustment)",
    )
    parser.add_argument(
        "--adjustment_method",
        type=str,
        default="holm",
        choices=["holm", "bonferroni", "bhy"],
        help="Multiple-testing adjustment (default: holm). holm and bonferroni need only "
        "--num_tests; bhy adjusts a family of p-values and needs --trial_ledger.",
    )
    parser.add_argument(
        "--trial_ledger",
        type=str,
        default=None,
        help="CSV with one row per OTHER trial in the family and its p-value (for bhy)",
    )
    parser.add_argument(
        "--trial_p_value_column",
        type=str,
        default="p_value",
        help="Column of --trial_ledger holding each trial's p-value (default: p_value)",
    )
    parser.add_argument(
        "--transaction_costs",
        action="store_true",
        help="Charge transaction costs (bid-ask spread + slippage) on each trade's session",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=10.0,
        help="Bid-ask spread in basis points (default: 10 bps = 0.10%% round-trip). "
        "For S&P 500 large-caps, typical spreads are 5-15 bps.",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=5.0,
        help="Slippage in basis points per side traded (default: 5 bps = 0.05%%).",
    )
    parser.add_argument(
        "--auto_save",
        action="store_true",
        help="Automatically save all outputs in organized structure",
    )
    parser.add_argument(
        "--backtest_suffix",
        type=str,
        default="",
        help='Suffix for backtest directory (e.g., "_with_costs" or "_tc")',
    )
    parser.add_argument(
        "--enable_mlflow",
        action="store_true",
        help="Enable MLflow tracking for this backtest. Auto-links to training "
        "run if mlflow_run.json exists in the predictions directory.",
    )
    parser.add_argument(
        "--disable_mlflow_autolink",
        action="store_true",
        help="Do not auto-enable MLflow tracking from mlflow_run.json beside the "
        "predictions directory. Use this for portable Drive/Colab backtests where "
        "the original training MLflow run may not exist in the current file store.",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=None,
        help="Optional MLflow tracking URI override",
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        default=None,
        help="Optional MLflow experiment name override",
    )
    parser.add_argument(
        "--enable_rank_drop_gate",
        action="store_true",
        help="Enable rank-drop sell gate: a held stock is kept unless its rank worsened "
        "by at least --min_rank_drop since its sleeve's previous signal date",
    )
    parser.add_argument(
        "--min_rank_drop",
        type=int,
        default=10,
        help="Minimum rank worsening (current_rank - prev_rank) required to exit a held stock (default: 10)",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.strategy is None and (args.holding_period is not None or args.rebalance_style):
        args.strategy = OPEN_TO_OPEN
        print("NOTE: --holding_period/--rebalance_style select the open_to_open variant.")
    if args.adjustment_method == "bhy" and not args.trial_ledger:
        parser.error(
            "--adjustment_method bhy adjusts a family of p-values: pass --trial_ledger with "
            "every other trial's p-value, or use holm/bonferroni for one strategy"
        )

    if args.run_dir:
        try:
            replay_inputs = resolve_replay_inputs(
                args.run_dir,
                predictions_dir=args.predictions_dir,
                data_file=args.data_file,
                pit_universe_csv=args.pit_universe_csv,
                pit_export_cutoff=args.pit_export_cutoff,
                test_start=args.test_start,
                test_end=args.test_end,
                label_t=args.label_t,
                top_k=args.top_k,
            )
        except ReplayInputError as error:
            parser.error(f"--run_dir {args.run_dir}: {error}")
        args.predictions_dir = str(replay_inputs.predictions_dir)
        input_config = replay_inputs.backtest_config()
    else:
        required = {
            "--predictions_dir": args.predictions_dir,
            "--data_file": args.data_file,
            "--test_start": args.test_start,
            "--test_end": args.test_end,
            "--top_k": args.top_k,
            "--label_t": args.label_t,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            parser.error(
                "pass --run_dir to replay a run from its own records, or name every input: "
                f"missing {' '.join(missing)}"
            )
        try:
            args.test_start, args.test_end = iso_date(args.test_start), iso_date(args.test_end)
        except ReplayInputError as error:
            parser.error(str(error))
        print(
            "WARNING: no --run_dir, so the inputs below are not checked against any run's records."
        )
        input_config = {
            "data_file": args.data_file,
            "pit_universe_csv": args.pit_universe_csv,
            "pit_export_cutoff": args.pit_export_cutoff,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "label_t": args.label_t,
            "top_k": args.top_k,
        }

    config = {
        **input_config,
        "trading_days_per_year": 252,
        "strategy": args.strategy or CLOSE_TRANCHES,
        "tranches": args.tranches,
        "holding_period": args.holding_period,
        "rebalance_style": args.rebalance_style,
        "issuer_map": args.issuer_map,
        "transaction_costs": {
            "enabled": args.transaction_costs,
            "bid_ask_spread": args.spread / 10000.0,  # Convert bps to decimal
            "slippage": args.slippage / 10000.0,  # Convert bps to decimal
        },
        "rank_drop_gate": {
            "enabled": args.enable_rank_drop_gate,
            "min_rank_drop": args.min_rank_drop,
        },
    }
    try:
        replay_strategy(config)
    except ValueError as error:
        parser.error(str(error))
    testing = {
        "num_tests": args.num_tests,
        "adjustment_method": args.adjustment_method,
        "trial_ledger": args.trial_ledger,
    }

    backtest_dir = None
    logger = None
    if args.auto_save:
        backtest_dir = setup_backtest_output_dir(args.predictions_dir, args.backtest_suffix)
        logger = setup_backtest_logging(backtest_dir)
        logger.info("Starting backtest evaluation with auto-save")
        logger.info(f"Predictions directory: {args.predictions_dir}")
        logger.info(f"Backtest output directory: {backtest_dir}")

    tracking_manager, _linked_metadata = setup_backtest_tracking(
        predictions_dir=args.predictions_dir,
        config=config,
        enable_mlflow=args.enable_mlflow,
        disable_mlflow_autolink=args.disable_mlflow_autolink,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        backtest_suffix=args.backtest_suffix,
    )
    if tracking_manager.enabled:
        tracking_manager.log_params({"backtest": config, **testing})

    results, book = replay(args.predictions_dir, config)
    print_results(results, "MCI-GRU", **testing)
    if args.auto_save and backtest_dir:
        save_backtest_results(results, backtest_dir, book=book, config=config, **testing)
        plot_file = os.path.join(backtest_dir, "equity_curve.png")
        plot_equity_curve(book, output_path=plot_file)
    elif args.output:
        save_results(results, args.output, **testing)
        if args.plot:
            plot_equity_curve(book, args.output.replace(".csv", "_equity.png"))

    if tracking_manager.enabled:
        tracking_manager.log_metrics(
            {k: v for k, v in results.items() if isinstance(v, (int, float))},
            prefix="backtest.",
        )
        if backtest_dir:
            _log_backtest_artifacts(tracking_manager, backtest_dir)
        tracking_manager.close()

    if logger:
        logger.info("Backtest evaluation completed successfully")
    return results


if __name__ == "__main__":
    main()
