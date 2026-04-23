"""
Data pipeline for MCI-GRU experiments.

Consolidates the data loading → feature engineering → normalisation →
tensor construction flow that was previously split across run_experiment.py
(``prepare_data`` / ``prepare_data_index_level``) and
paper_trade/scripts/infer.py.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from mci_gru.data.data_manager import DataManager
from mci_gru.data.preprocessing import (
    apply_rank_gaussian,
    apply_rank_labels,
    compute_labels,
    fit_rank_gaussian_reference,
    generate_graph_features,
    generate_time_series_features,
)
from mci_gru.graph import GraphBuilder
from mci_gru.graph.sector_edges import build_sector_edges, load_sector_map_csv

if TYPE_CHECKING:
    from mci_gru.config import ExperimentConfig
    from mci_gru.features import FeatureEngineer

# ── helpers ──────────────────────────────────────────────────────────────


def _load_auxiliary_data(
    data_manager: DataManager,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Load VIX, credit spread, and regime DataFrames as configured."""
    vix_df: pd.DataFrame | None = None
    credit_df: pd.DataFrame | None = None
    regime_df: pd.DataFrame | None = None

    if config.features.include_vix:
        try:
            vix_df = data_manager.load_vix()
            print(f"Loaded VIX data: {len(vix_df)} observations")
        except Exception as exc:
            print(f"Warning: Could not load VIX data: {exc}")

    if config.features.include_credit_spread:
        try:
            credit_df = data_manager.load_credit_spreads()
            print(f"Loaded credit spread data: {len(credit_df)} observations")
        except Exception as exc:
            print(f"Warning: Could not load credit spread data: {exc}")

    if config.features.include_global_regime:
        try:
            regime_df = data_manager.load_regime_inputs(
                lseg_market_ric=config.features.regime_lseg_market_ric,
                lseg_copper_ric=config.features.regime_lseg_copper_ric,
                lseg_yield_10y_ric=config.features.regime_lseg_yield_10y_ric,
                lseg_yield_3m_ric=config.features.regime_lseg_yield_3m_ric,
                lseg_oil_ric=config.features.regime_lseg_oil_ric,
                lseg_vix_ric=config.features.regime_lseg_vix_ric,
                regime_inputs_csv=config.features.regime_inputs_csv or None,
                regime_enforce_lag_days=config.features.regime_enforce_lag_days,
            )
            print(f"Loaded regime input data: {len(regime_df)} observations")
        except Exception as exc:
            if config.features.regime_strict:
                raise
            print(f"Warning: Could not load regime input data: {exc}")
            print("Continuing with zero-filled regime features (soft-fail)")

    return vix_df, credit_df, regime_df


def _compute_norm_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_end: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-feature mean/std from the training period."""
    train_df = df[df["dt"] <= train_end]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in feature_cols:
        if col in train_df.columns:
            means[col] = train_df[col].mean()
            stds[col] = train_df[col].std()
            if stds[col] == 0:
                stds[col] = 1.0
    return means, stds


def _apply_pit_universe(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """Filter rows to kdcode/date pairs covered by [valid_from, valid_to] in *csv_path*."""
    pit = pd.read_csv(csv_path)
    pit.columns = [str(c).strip().lower() for c in pit.columns]
    if not {"kdcode", "valid_from", "valid_to"}.issubset(pit.columns):
        raise ValueError("pit_universe_csv must have columns kdcode, valid_from, valid_to")
    pit["vf"] = pd.to_datetime(pit["valid_from"]).dt.strftime("%Y-%m-%d")
    pit["vt"] = pd.to_datetime(pit["valid_to"]).dt.strftime("%Y-%m-%d")
    merged = df.merge(pit[["kdcode", "vf", "vt"]], on="kdcode", how="inner")
    mask = (merged["dt"] >= merged["vf"]) & (merged["dt"] <= merged["vt"])
    out = merged.loc[mask, df.columns]
    return out.reset_index(drop=True)


def _apply_normalisation(
    df: pd.DataFrame,
    feature_cols: list[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> pd.DataFrame:
    """3-sigma clipping followed by z-score normalisation."""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        m = means.get(col, 0.0)
        s = stds.get(col, 1.0)
        df[col] = np.clip(df[col], m - 3 * s, m + 3 * s)
        df[col] = (df[col] - m) / s
    return df


def _stock_feature_row_slice(
    all_dates: list[str], his_t: int, label_dates: list[str]
) -> tuple[int, int]:
    """Map consecutive label dates to [start, end) row indices in ``stock_features``.

    Row ``r`` corresponds to the window ending at ``all_dates[r + his_t - 1]``, so the
    label date ``D`` (aligned with ``train_dates[his_t:]``, ``val_dates``, ``test_dates``)
    sits at row ``all_dates.index(D) - his_t``.
    """
    if not label_dates:
        return 0, 0
    start = all_dates.index(label_dates[0]) - his_t
    end = start + len(label_dates)
    return start, end


def _build_tensors(
    df_filtered: pd.DataFrame,
    kdcode_list: list[str],
    feature_cols: list[str],
    train_dates: list[str],
    val_dates: list[str],
    test_dates: list[str],
    his_t: int,
    label_t: int,
    label_type: str,
    df_for_labels: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_polars: bool = False,
) -> dict[str, Any]:
    """Build time-series tensors, graph features, and labels."""
    print("Generating time series features...")
    stock_features = generate_time_series_features(
        df_filtered, kdcode_list, feature_cols, his_t, use_polars=use_polars
    )
    all_dates = sorted(df_filtered["dt"].unique())

    train_label_dates = train_dates[his_t:]
    tr0, tr1 = _stock_feature_row_slice(all_dates, his_t, train_label_dates)
    va0, va1 = _stock_feature_row_slice(all_dates, his_t, val_dates)
    te0, te1 = _stock_feature_row_slice(all_dates, his_t, test_dates)

    stock_features_train = stock_features[tr0:tr1]
    stock_features_val = stock_features[va0:va1]
    stock_features_test = stock_features[te0:te1]

    print("Generating graph features...")
    x_graph_train = generate_graph_features(
        train_df, kdcode_list, feature_cols, train_dates[his_t:]
    )
    x_graph_val = generate_graph_features(val_df, kdcode_list, feature_cols, val_dates)
    x_graph_test = generate_graph_features(test_df, kdcode_list, feature_cols, test_dates)

    print("Computing labels...")
    train_labels = compute_labels(df_for_labels, kdcode_list, train_dates[his_t:], label_t)
    val_labels = compute_labels(df_for_labels, kdcode_list, val_dates, label_t)

    if label_type == "rank":
        print("Converting labels to cross-sectional rank percentiles...")
        train_labels = apply_rank_labels(train_labels)
        val_labels = apply_rank_labels(val_labels)

    return {
        "train_dates": train_dates[his_t:],
        "val_dates": val_dates,
        "test_dates": test_dates,
        "stock_features_train": stock_features_train,
        "stock_features_val": stock_features_val,
        "stock_features_test": stock_features_test,
        "x_graph_train": x_graph_train,
        "x_graph_val": x_graph_val,
        "x_graph_test": x_graph_test,
        "train_labels": train_labels,
        "val_labels": val_labels,
    }


# ── public API ───────────────────────────────────────────────────────────


def prepare_data(
    config: ExperimentConfig,
    feature_engineer: FeatureEngineer,
) -> dict[str, Any]:
    """Load and prepare stock-level cross-sectional data for training.

    Returns a dict consumed by the training loop and metric evaluation.
    """
    print("=" * 80)
    print("Preparing Data")
    print("=" * 80)

    data_manager = DataManager(config.data)
    df = data_manager.load()

    vix_df, credit_df, regime_df = _load_auxiliary_data(data_manager, config)

    df = feature_engineer.transform(df, vix_df, credit_df, regime_df)
    feature_cols = feature_engineer.get_feature_columns()
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # Per-day mean imputation
    print("Filling NaN values...")
    parts = []
    for _, df_day in df.groupby("dt"):
        df_day = df_day.copy()
        for col in feature_cols:
            if col in df_day.columns:
                df_day[col] = df_day[col].fillna(df_day[col].mean())
        df_day = df_day.fillna(0.0)
        parts.append(df_day)
    df_filled = pd.concat(parts)
    del parts
    gc.collect()

    if config.data.use_pit_universe and config.data.pit_universe_csv:
        print("Applying PIT universe filter...")
        df_filled = _apply_pit_universe(df_filled, config.data.pit_universe_csv)

    rank_gauss_reference: dict[str, np.ndarray] | None = None
    if config.data.normalisation == "zscore":
        means, stds = _compute_norm_stats(df_filled, feature_cols, config.data.train_end)
        df_norm = _apply_normalisation(df_filled, feature_cols, means, stds)
    elif config.data.normalisation == "rank_gauss":
        print("Applying rank-Gaussian normalisation (train fit)...")
        train_mask = df_filled["dt"] <= config.data.train_end
        train_slice = df_filled.loc[train_mask]
        rank_gauss_reference = fit_rank_gaussian_reference(train_slice, feature_cols)
        df_norm = apply_rank_gaussian(df_filled, feature_cols, rank_gauss_reference)
        means, stds = {}, {}
    else:
        raise ValueError(f"Unknown normalisation: {config.data.normalisation!r}")
    del df_filled
    gc.collect()

    if config.data.filter_stocks_per_split:
        df_filtered, kdcode_list = data_manager.filter_complete_stocks_per_split(df_norm)
    else:
        df_filtered, kdcode_list = data_manager.filter_complete_stocks(df_norm)
    train_df, val_df, test_df = data_manager.split_by_period(df_filtered)

    train_dates = sorted(train_df["dt"].unique())
    val_dates = sorted(val_df["dt"].unique())
    test_dates = sorted(test_df["dt"].unique())

    tensors = _build_tensors(
        df_filtered,
        kdcode_list,
        feature_cols,
        train_dates,
        val_dates,
        test_dates,
        config.model.his_t,
        config.model.label_t,
        config.training.label_type,
        df,  # use un-normalised df for forward-return labels
        train_df,
        val_df,
        test_df,
        use_polars=config.data.use_polars,
    )

    print("Building correlation graph...")
    graph_builder = GraphBuilder(
        judge_value=config.graph.judge_value,
        update_frequency_months=config.graph.update_frequency_months,
        corr_lookback_days=config.graph.corr_lookback_days,
        top_k=config.graph.top_k,
        top_k_metric=config.graph.top_k_metric,
        use_multi_feature_edges=config.graph.use_multi_feature_edges,
        use_lead_lag_features=config.graph.use_lead_lag_features,
        lead_lag_days=config.graph.lead_lag_days,
    )
    edge_index, edge_weight = graph_builder.build_graph(df, kdcode_list, config.data.train_start)

    graph_schedule = None
    if config.graph.update_frequency_months > 0:
        graph_schedule = graph_builder.precompute_snapshots(
            df, kdcode_list, config.data.train_start, config.data.test_end
        )

    edge_index_sector = None
    edge_weight_sector = None
    if config.graph.use_sector_relation and config.graph.sector_map_csv:
        sector_map = load_sector_map_csv(config.graph.sector_map_csv)
        edge_index_sector, edge_weight_sector = build_sector_edges(
            kdcode_list,
            sector_map,
            config.graph.sector_top_k,
        )

    return {
        "kdcode_list": kdcode_list,
        **tensors,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "feature_cols": feature_cols,
        "graph_schedule": graph_schedule,
        "df": df,
        "norm_means": means,
        "norm_stds": stds,
        "graph_static_valid_from": config.data.train_start,
        "edge_index_sector": edge_index_sector,
        "edge_weight_sector": edge_weight_sector,
        "rank_gauss_reference": rank_gauss_reference,
    }


def prepare_data_index_level(
    config: ExperimentConfig,
    feature_engineer: FeatureEngineer,
) -> dict[str, Any]:
    """Prepare data for index-level mode (single series, no survivorship bias).

    Uses a trivial 1-node / 0-edge graph so the rest of the pipeline runs
    unchanged.
    """
    print("=" * 80)
    print("Preparing Data (index-level mode; no stock-level survivorship bias)")
    print("=" * 80)

    data_manager = DataManager(config.data)
    df = data_manager.load_index_series()
    kdcode_list = ["INDEX"]

    vix_df, credit_df, regime_df = _load_auxiliary_data(data_manager, config)

    df = feature_engineer.transform(df, vix_df, credit_df, regime_df)
    feature_cols = feature_engineer.get_feature_columns()
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    df = df.fillna(0.0)

    date_mask = (df["dt"] >= config.data.train_start) & (df["dt"] <= config.data.test_end)
    df_norm = df[date_mask].copy()

    train_df = df_norm[
        (df_norm["dt"] >= config.data.train_start) & (df_norm["dt"] <= config.data.train_end)
    ]
    val_df = df_norm[
        (df_norm["dt"] >= config.data.val_start) & (df_norm["dt"] <= config.data.val_end)
    ]
    test_df = df_norm[
        (df_norm["dt"] >= config.data.test_start) & (df_norm["dt"] <= config.data.test_end)
    ]

    means, stds = _compute_norm_stats(df_norm, feature_cols, config.data.train_end)
    df_norm = _apply_normalisation(df_norm, feature_cols, means, stds)

    df_filtered = df_norm.copy()

    train_dates = sorted(train_df["dt"].unique())
    val_dates = sorted(val_df["dt"].unique())
    test_dates = sorted(test_df["dt"].unique())

    tensors = _build_tensors(
        df_filtered,
        kdcode_list,
        feature_cols,
        train_dates,
        val_dates,
        test_dates,
        config.model.his_t,
        config.model.label_t,
        config.training.label_type,
        df_filtered,  # use normalised df (single series, no cross-section)
        train_df,
        val_df,
        test_df,
        use_polars=config.data.use_polars,
    )

    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_weight = torch.empty(0, dtype=torch.float32)

    return {
        "kdcode_list": kdcode_list,
        **tensors,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "feature_cols": feature_cols,
        "graph_builder": None,
        "graph_schedule": None,
        "df": df_filtered,
        "norm_means": means,
        "norm_stds": stds,
        "graph_static_valid_from": config.data.train_start,
        "edge_index_sector": None,
        "edge_weight_sector": None,
        "rank_gauss_reference": None,
    }
