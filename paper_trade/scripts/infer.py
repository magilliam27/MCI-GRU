"""
Standalone inference for the paper trading pipeline.

Loads frozen model checkpoints and run_metadata.json, applies the same
feature engineering and normalization as training, and outputs a scores
CSV for the latest prediction date.

Does NOT train -- only runs forward passes on saved checkpoints.

Usage:
    python paper_trade/scripts/infer.py
    python paper_trade/scripts/infer.py --model-dir paper_trade/Model/Seed73_trained_to_2062026
    python paper_trade/scripts/infer.py --date 2026-03-07
"""

import argparse
import json
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from functools import partial
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from mci_gru.features import FeatureEngineer
from mci_gru.models import create_model
from mci_gru.data.data_manager import (
    CombinedDataset,
    combined_collate_fn,
)

DEFAULT_MODEL_DIR = "paper_trade/Model/Seed73_trained_to_2062026"
DEFAULT_CSV = "data/raw/market/sp500_2019_universe_data_through_2026.csv"
DEFAULT_OUTPUT_DIR = "paper_trade/results"


def load_metadata(model_dir: Path) -> dict:
    meta_path = model_dir / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_metadata.json not found in {model_dir}")
    with open(meta_path) as f:
        return json.load(f)


def load_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {model_dir}")
    cfg = OmegaConf.load(str(cfg_path))
    return OmegaConf.to_container(cfg, resolve=True)


def build_feature_engineer(features_cfg: dict) -> FeatureEngineer:
    return FeatureEngineer(
        include_momentum=features_cfg.get("include_momentum", True),
        include_weekly_momentum=features_cfg.get("include_weekly_momentum", False),
        momentum_encoding=features_cfg.get("momentum_encoding", "binary"),
        momentum_buffer_low=features_cfg.get("momentum_buffer_low", 0.1),
        momentum_buffer_high=features_cfg.get("momentum_buffer_high", 0.9),
        include_volatility=features_cfg.get("include_volatility", False),
        include_vix=features_cfg.get("include_vix", False),
        include_credit_spread=features_cfg.get("include_credit_spread", False),
        include_global_regime=features_cfg.get("include_global_regime", False),
        regime_change_months=features_cfg.get("regime_change_months", 12),
        regime_norm_months=features_cfg.get("regime_norm_months", 120),
        regime_clip_z=features_cfg.get("regime_clip_z", 3.0),
        regime_exclusion_months=features_cfg.get("regime_exclusion_months", 1),
        regime_similarity_quantile=features_cfg.get("regime_similarity_quantile", 0.2),
        regime_min_history_months=features_cfg.get("regime_min_history_months", 24),
        regime_strict=features_cfg.get("regime_strict", False),
        include_rsi=features_cfg.get("include_rsi", False),
        include_ma_features=features_cfg.get("include_ma_features", False),
        include_price_features=features_cfg.get("include_price_features", False),
        include_volume_features=features_cfg.get("include_volume_features", False),
    )


def prepare_inference_data(
    csv_path: str,
    metadata: dict,
    features_cfg: dict,
    his_t: int,
    target_date: str = None,
):
    """
    Load CSV, engineer features, normalize with saved stats, and build
    the time-series + graph tensors needed for a single-date forward pass.

    Returns (time_series, graph_features, kdcode_list, pred_date)
    where time_series has shape (1, n_stocks, his_t, n_features)
    and graph_features has shape (1, n_stocks, n_features).
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows, date range {df['dt'].min()} to {df['dt'].max()}")

    feature_engineer = build_feature_engineer(features_cfg)
    df = feature_engineer.transform(df, None, None, None)

    feature_cols = metadata["feature_cols"]
    kdcode_list = metadata["kdcode_list"]
    means = metadata["norm_means"]
    stds = metadata["norm_stds"]

    actual_cols = feature_engineer.get_feature_columns()
    missing_feats = [c for c in feature_cols if c not in actual_cols]
    if missing_feats:
        raise ValueError(
            f"Feature mismatch: metadata expects {missing_feats} "
            f"but FeatureEngineer produced {actual_cols}"
        )

    df = df[df["kdcode"].isin(kdcode_list)].copy()

    all_dates = sorted(df["dt"].unique())
    if target_date is None:
        target_date = all_dates[-1]
    if target_date not in all_dates:
        raise ValueError(f"Target date {target_date} not found in data")

    target_idx = all_dates.index(target_date)
    if target_idx < his_t:
        raise ValueError(
            f"Need at least {his_t} days before target date; "
            f"only {target_idx} available"
        )

    window_dates = all_dates[target_idx - his_t: target_idx + 1]
    df_window = df[df["dt"].isin(window_dates)].copy()

    print("Filling NaN values...")
    grouped = df_window.groupby("dt")
    filled_parts = []
    for dt_val, df_day in grouped:
        df_day = df_day.copy()
        for col in feature_cols:
            if col in df_day.columns:
                df_day[col] = df_day[col].fillna(df_day[col].mean())
        df_day = df_day.fillna(0.0)
        filled_parts.append(df_day)
    df_window = pd.concat(filled_parts)

    print("Normalizing with saved training statistics...")
    for col in feature_cols:
        if col in df_window.columns:
            m, s = means[col], stds[col]
            df_window[col] = np.clip(df_window[col], m - 3 * s, m + 3 * s)
            df_window[col] = (df_window[col] - m) / s

    print("Building time-series tensor...")
    stock_to_idx = {kd: i for i, kd in enumerate(kdcode_list)}
    n_stocks = len(kdcode_list)
    n_features = len(feature_cols)

    lookback_dates = window_dates[:his_t]
    pivot = np.zeros((his_t, n_stocks, n_features), dtype=np.float32)

    for _, row in df_window[df_window["dt"].isin(lookback_dates)].iterrows():
        kd = row["kdcode"]
        dt_val = row["dt"]
        if kd in stock_to_idx and dt_val in lookback_dates:
            s_idx = stock_to_idx[kd]
            d_idx = lookback_dates.index(dt_val)
            pivot[d_idx, s_idx, :] = row[feature_cols].values.astype(np.float32)

    time_series = pivot.transpose(1, 0, 2)
    time_series = time_series[np.newaxis, ...]

    print("Building graph features tensor...")
    graph_date = window_dates[-1]
    graph_features = np.zeros((1, n_stocks, n_features), dtype=np.float32)
    df_graph_day = df_window[df_window["dt"] == graph_date]
    for _, row in df_graph_day.iterrows():
        kd = row["kdcode"]
        if kd in stock_to_idx:
            s_idx = stock_to_idx[kd]
            graph_features[0, s_idx, :] = row[feature_cols].values.astype(np.float32)

    print(f"  Prediction date: {target_date}")
    print(f"  Lookback window: {lookback_dates[0]} to {lookback_dates[-1]}")
    print(f"  Stocks: {n_stocks}, Features: {n_features}")

    return time_series, graph_features, kdcode_list, target_date


def run_inference(
    model_dir: Path,
    time_series: np.ndarray,
    graph_features: np.ndarray,
    kdcode_list: list,
    model_cfg: dict,
    num_features: int,
) -> np.ndarray:
    """
    Load each checkpoint, run a forward pass, and average predictions.
    Returns scores array of shape (n_stocks,).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning inference on {device}...")

    graph_data_path = model_dir / "graph_data.pt"
    if not graph_data_path.exists():
        raise FileNotFoundError(f"graph_data.pt not found in {model_dir}")

    graph_data = torch.load(str(graph_data_path), weights_only=True)
    edge_index = graph_data["edge_index"].to(device)
    edge_weight = graph_data["edge_weight"].to(device)

    ts_tensor = torch.from_numpy(time_series).float().to(device)
    gf_tensor = torch.from_numpy(graph_features).float().to(device)

    n_stocks = ts_tensor.shape[1]

    batched_gf = gf_tensor.view(-1, gf_tensor.shape[-1])
    batched_ei = edge_index
    batched_ew = edge_weight

    ckpt_dir = model_dir / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("model_*_best.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    print(f"  Found {len(ckpt_files)} checkpoints")

    all_preds = []
    for ckpt_path in ckpt_files:
        model = create_model(num_features, model_cfg)
        model.load_state_dict(
            torch.load(str(ckpt_path), map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(ts_tensor, batched_gf, batched_ei, batched_ew, n_stocks)
            preds = output.squeeze().cpu().numpy()

        all_preds.append(preds)
        print(f"    {ckpt_path.name}: done")

    avg_preds = np.mean(all_preds, axis=0)
    print(f"  Averaged {len(all_preds)} model predictions")
    return avg_preds


def save_scores(
    scores: np.ndarray,
    kdcode_list: list,
    pred_date: str,
    output_dir: Path,
):
    """Save scores CSV with kdcode, dt, score, rank columns."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "kdcode": kdcode_list,
        "dt": pred_date,
        "score": np.round(scores, 5),
    })
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    date_dir = output_dir / pred_date
    date_dir.mkdir(parents=True, exist_ok=True)
    out_path = date_dir / "scores.csv"
    df.to_csv(str(out_path), index=False)

    print(f"\nScores saved to {out_path}")
    print(f"  Top 5:")
    print(df.head().to_string(index=False))
    print(f"  Bottom 5:")
    print(df.tail().to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description="MCI-GRU standalone inference.")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Path to model directory with checkpoints, config, metadata",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Path to master OHLCV CSV",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Prediction date (default: latest date in CSV)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for scores CSV",
    )
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / args.model_dir
    csv_path = PROJECT_ROOT / args.csv
    output_dir = PROJECT_ROOT / args.output_dir

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    print("=" * 70)
    print("  MCI-GRU Standalone Inference")
    print("=" * 70)

    metadata = load_metadata(model_dir)
    config = load_config(model_dir)

    his_t = metadata["his_t"]
    feature_cols = metadata["feature_cols"]
    model_cfg = config["model"]

    time_series, graph_features, kdcode_list, pred_date = prepare_inference_data(
        csv_path=str(csv_path),
        metadata=metadata,
        features_cfg=config.get("features", {}),
        his_t=his_t,
        target_date=args.date,
    )

    scores = run_inference(
        model_dir=model_dir,
        time_series=time_series,
        graph_features=graph_features,
        kdcode_list=kdcode_list,
        model_cfg=model_cfg,
        num_features=len(feature_cols),
    )

    save_scores(scores, kdcode_list, pred_date, output_dir)

    print("\n" + "=" * 70)
    print("  Inference complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
