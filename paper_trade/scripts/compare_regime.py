"""
Regime vs. No-Regime comparison over a recent date window.

Runs inference for each trading day in the window twice:
  A) With live FRED regime data
  B) With zero-filled regime columns (regime_df=None)

Then evaluates both via backtest_sp500.py and prints a side-by-side
performance table.

Usage:
    python paper_trade/scripts/compare_regime.py
    python paper_trade/scripts/compare_regime.py --days 10
    python paper_trade/scripts/compare_regime.py --model-dir paper_trade/Model/seed7_w_regime
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mci_gru.models import create_model  # noqa: E402
from paper_trade.scripts.infer import (  # noqa: E402
    load_config,
    load_metadata,
    prepare_inference_data,
    prepare_inference_regime_df,
)

DEFAULT_MODEL_DIR = "paper_trade/Model/seed7_w_regime"
DEFAULT_CSV = "data/raw/market/sp500_2019_universe_data_through_2026.csv"
BACKTEST_SCRIPT = PROJECT_ROOT / "tests" / "backtest_sp500.py"


def get_trading_dates(csv_path: str, num_days: int) -> list[str]:
    """Return the last `num_days` trading dates from the master CSV."""
    dates = pd.read_csv(csv_path, usecols=["dt"])["dt"].dropna().unique()
    dates = sorted(dates)
    return dates[-num_days:]


def load_models(model_dir: Path, num_features: int, model_cfg: dict):
    """Load all checkpoint models once, return list of (name, model)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = model_dir / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("model_*_best.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    graph_data = torch.load(
        str(model_dir / "graph_data.pt"), map_location=device, weights_only=True
    )
    edge_index = graph_data["edge_index"].to(device)
    edge_weight = graph_data["edge_weight"].to(device)

    models = []
    for ckpt_path in ckpt_files:
        model = create_model(num_features, model_cfg)
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        models.append(model)

    return models, edge_index, edge_weight, device


def run_ensemble(
    models,
    edge_index,
    edge_weight,
    device,
    time_series: np.ndarray,
    graph_features: np.ndarray,
) -> np.ndarray:
    """Run all models and return averaged predictions."""
    ts_tensor = torch.from_numpy(time_series).float().to(device)
    gf_tensor = torch.from_numpy(graph_features).float().to(device)
    n_stocks = ts_tensor.shape[1]
    batched_gf = gf_tensor.view(-1, gf_tensor.shape[-1])

    all_preds = []
    for model in models:
        with torch.no_grad():
            output = model(ts_tensor, batched_gf, edge_index, edge_weight, n_stocks)
            all_preds.append(output.squeeze().cpu().numpy())

    return np.mean(all_preds, axis=0)


def save_prediction(scores: np.ndarray, kdcode_list: list, pred_date: str, out_dir: Path):
    """Save a single date's scores in backtest_sp500.py format: <dir>/<date>.csv"""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "kdcode": kdcode_list,
            "dt": pred_date,
            "score": np.round(scores, 5),
        }
    )
    df.to_csv(out_dir / f"{pred_date}.csv", index=False)


def run_backtest(
    predictions_dir: Path,
    test_start: str,
    test_end: str,
    data_file: str,
    python_exe: str,
    label: str,
) -> dict:
    """Run backtest_sp500.py and return parsed metrics."""
    suffix = f"_{label}"
    cmd = [
        python_exe,
        str(BACKTEST_SCRIPT),
        "--predictions_dir",
        str(predictions_dir),
        "--data_file",
        data_file,
        "--test_start",
        test_start,
        "--test_end",
        test_end,
        "--top_k",
        "20",
        "--transaction_costs",
        "--spread",
        "5",
        "--slippage",
        "5",
        "--enable_rank_drop_gate",
        "--min_rank_drop",
        "30",
        "--holding_period",
        "1",
        "--rebalance_style",
        "staggered",
        "--auto_save",
        "--backtest_suffix",
        suffix,
    ]
    print(f"\n  Running backtest ({label})...")
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    if result.returncode != 0:
        print(f"  Backtest FAILED for {label}:")
        for line in (result.stderr or "").strip().split("\n")[-15:]:
            print(f"    {line}")
        return {}

    for line in (result.stdout or "").strip().split("\n"):
        print(f"    {line}")

    # backtest_sp500.py saves to: parent_of(predictions_dir) / f"backtest{suffix}"
    backtest_dir = predictions_dir.parent / f"backtest{suffix}"
    metrics_path = backtest_dir / "backtest_metrics.json"

    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)

    print(f"  WARNING: metrics file not found at {metrics_path}")
    return {}


def print_comparison(metrics_regime: dict, metrics_no_regime: dict):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 75)
    print("  REGIME vs NO-REGIME COMPARISON")
    print("=" * 75)

    rows = [
        ("Total Return", "total_return", "{:+.2%}"),
        ("Benchmark Return", "benchmark_return", "{:+.2%}"),
        ("Excess Return", "excess_return", "{:+.2%}"),
        ("Ann. Return (ARR)", "ARR", "{:+.2%}"),
        ("Ann. Volatility", "AVoL", "{:.2%}"),
        ("Max Drawdown", "MDD", "{:.2%}"),
        ("Sharpe Ratio", "ASR", "{:.4f}"),
        ("Calmar Ratio", "CR", "{:.4f}"),
        ("Info Ratio", "IR", "{:.4f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Avg Turnover", "avg_turnover", "{:.1%}"),
        ("Trading Days", "num_trading_days", "{:.0f}"),
    ]

    print(f"\n  {'Metric':<22s}  {'With Regime':>14s}  {'No Regime':>14s}  {'Delta':>14s}")
    print(f"  {'-' * 22}  {'-' * 14}  {'-' * 14}  {'-' * 14}")

    for label, key, fmt in rows:
        val_r = metrics_regime.get(key)
        val_n = metrics_no_regime.get(key)
        s_r = fmt.format(val_r) if val_r is not None else "N/A"
        s_n = fmt.format(val_n) if val_n is not None else "N/A"
        if val_r is not None and val_n is not None:
            delta = val_r - val_n
            s_d = fmt.format(delta)
        else:
            s_d = "N/A"
        print(f"  {label:<22s}  {s_r:>14s}  {s_n:>14s}  {s_d:>14s}")

    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Compare regime vs no-regime inference over a date window.",
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of recent trading days to compare (default: 10)",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / args.model_dir
    csv_path = str(PROJECT_ROOT / args.csv)
    out_regime = PROJECT_ROOT / "paper_trade" / "compare_regime" / "with_regime"
    out_no_regime = PROJECT_ROOT / "paper_trade" / "compare_regime" / "no_regime"

    print("=" * 75)
    print("  Regime vs No-Regime Comparison")
    print("=" * 75)

    metadata = load_metadata(model_dir)
    config = load_config(model_dir)
    his_t = metadata["his_t"]
    feature_cols = metadata["feature_cols"]
    features_cfg = config.get("features", {})
    model_cfg = config["model"]

    dates = get_trading_dates(csv_path, args.days)
    print(f"\n  Model:       {model_dir.name}")
    print(f"  Date window: {dates[0]} to {dates[-1]} ({len(dates)} trading days)")
    print(
        f"  Features:    {len(feature_cols)} (include_global_regime={features_cfg.get('include_global_regime')})"
    )

    # Load regime data once for the full window.
    regime_df = None
    if features_cfg.get("include_global_regime", False):
        try:
            regime_df = prepare_inference_regime_df(config, dates[-1])
            print(f"  Regime data: {len(regime_df)} rows from FRED")
        except Exception as exc:
            print(f"  WARNING: Could not load regime data: {exc}")
            print("  Cannot run comparison without regime data. Exiting.")
            sys.exit(1)

    # Load models once.
    print("\n  Loading model checkpoints...")
    models, edge_index, edge_weight, device = load_models(
        model_dir,
        len(feature_cols),
        model_cfg,
    )
    print(f"  Loaded {len(models)} checkpoints on {device}")

    # Score each date twice.
    print(f"\n{'=' * 75}")
    print("  GENERATING SCORES")
    print(f"{'=' * 75}")

    t0 = time.time()
    for i, target_date in enumerate(dates):
        print(f"\n  [{i + 1}/{len(dates)}] {target_date}")

        for label, rdf, out_dir in [
            ("regime", regime_df, out_regime),
            ("no_regime", None, out_no_regime),
        ]:
            ts, gf, kdcode_list, pred_date = prepare_inference_data(
                csv_path=csv_path,
                metadata=metadata,
                features_cfg=features_cfg,
                his_t=his_t,
                target_date=target_date,
                regime_df=rdf,
            )
            scores = run_ensemble(models, edge_index, edge_weight, device, ts, gf)
            save_prediction(scores, kdcode_list, pred_date, out_dir)

            top3 = sorted(zip(kdcode_list, scores, strict=False), key=lambda x: -x[1])[:3]
            top3_str = ", ".join(f"{k}={s:.4f}" for k, s in top3)
            print(f"    {label:>10s}: top3 = {top3_str}")

    elapsed = time.time() - t0
    print(f"\n  Scoring complete in {elapsed:.1f}s")

    # Run backtests.
    print(f"\n{'=' * 75}")
    print("  RUNNING BACKTESTS")
    print(f"{'=' * 75}")

    data_file = str(PROJECT_ROOT / args.csv)
    metrics_regime = run_backtest(
        out_regime,
        dates[0],
        dates[-1],
        data_file,
        args.python,
        "with_regime",
    )
    metrics_no_regime = run_backtest(
        out_no_regime,
        dates[0],
        dates[-1],
        data_file,
        args.python,
        "no_regime",
    )

    if metrics_regime and metrics_no_regime:
        print_comparison(metrics_regime, metrics_no_regime)
    elif not metrics_regime and not metrics_no_regime:
        print("\n  Both backtests failed to produce metrics.")
    else:
        print("\n  One backtest failed; cannot compare.")
        if metrics_regime:
            print("  With-regime metrics available.")
        if metrics_no_regime:
            print("  No-regime metrics available.")


if __name__ == "__main__":
    main()
