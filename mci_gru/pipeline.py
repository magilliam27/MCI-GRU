"""
Data pipeline for MCI-GRU experiments.

Consolidates the data loading → feature engineering → normalisation →
tensor construction flow that was previously split across run_experiment.py
(``prepare_data`` / ``prepare_data_index_level``) and
paper_trade/scripts/infer.py.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from mci_gru.data.data_manager import DataManager
from mci_gru.data.pit import (
    active_kdcodes_in_period,
    apply_label_mask,
    build_pit_masks,
    candidate_breadth,
    load_pit_intervals,
)
from mci_gru.data.preprocessing import (
    apply_rank_gaussian,
    apply_rank_labels,
    assert_training_labels_respect_embargo,
    compute_labels,
    fit_rank_gaussian_reference,
    generate_graph_features,
    generate_time_series_features,
    purge_training_sessions_for_embargo,
)
from mci_gru.data.transforms import (
    compute_zscore_norm_stats,
    impute_feature_nans_by_day,
    normalize_features_zscore,
)
from mci_gru.graph import GraphBuilder
from mci_gru.graph.sector_edges import build_sector_edges, load_sector_map_csv

if TYPE_CHECKING:
    from mci_gru.config import ExperimentConfig, GraphConfig
    from mci_gru.features import FeatureEngineer
    from mci_gru.graph.schedule import GraphSchedule

logger = logging.getLogger(__name__)

# ── staged pipeline dataclasses ──────────────────────────────────────────


@dataclass
class PipelineFrames:
    """Carries the three DataFrame variants through the staged pipeline."""

    raw: pd.DataFrame  # post-engineer, pre-impute (labels, PIT masks, graph)
    normalized: pd.DataFrame  # pre-universe-filter
    filtered: pd.DataFrame  # post-universe-filter (windows/tensors)


@dataclass(frozen=True)
class PitContext:
    intervals: pd.DataFrame | None
    masked_panel: bool
    csv_path: str | None


@dataclass(frozen=True)
class NormFit:
    means: dict[str, float]
    stds: dict[str, float]
    rank_gauss_reference: dict[str, np.ndarray] | None


@dataclass
class TensorBundle:
    train_dates: list[str]
    val_dates: list[str]
    test_dates: list[str]
    stock_features_train: np.ndarray
    stock_features_val: np.ndarray
    stock_features_test: np.ndarray
    x_graph_train: np.ndarray
    x_graph_val: np.ndarray
    x_graph_test: np.ndarray
    train_labels: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray
    train_active_member_mask: np.ndarray | None = None
    val_active_member_mask: np.ndarray | None = None
    test_active_member_mask: np.ndarray | None = None
    train_feature_ready_mask: np.ndarray | None = None
    val_feature_ready_mask: np.ndarray | None = None
    test_feature_ready_mask: np.ndarray | None = None
    train_loss_mask: np.ndarray | None = None
    val_loss_mask: np.ndarray | None = None
    test_loss_mask: np.ndarray | None = None
    train_tradable_mask: np.ndarray | None = None
    val_tradable_mask: np.ndarray | None = None
    test_tradable_mask: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        core = {
            "train_dates",
            "val_dates",
            "test_dates",
            "stock_features_train",
            "stock_features_val",
            "stock_features_test",
            "x_graph_train",
            "x_graph_val",
            "x_graph_test",
            "train_labels",
            "val_labels",
            "test_labels",
        }
        out: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in core or value is not None:
                out[field.name] = value
        return out


@dataclass(frozen=True)
class GraphArtifacts:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    graph_schedule: GraphSchedule | None
    edge_index_sector: torch.Tensor | None
    edge_weight_sector: torch.Tensor | None


# ── helpers ──────────────────────────────────────────────────────────────


def load_auxiliary_data(
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
            logger.info(f"Loaded VIX data: {len(vix_df)} observations")
        except Exception as exc:
            logger.warning(f"Warning: Could not load VIX data: {exc}")

    if config.features.include_credit_spread:
        try:
            credit_df = data_manager.load_credit_spreads()
            logger.info(f"Loaded credit spread data: {len(credit_df)} observations")
        except Exception as exc:
            logger.warning(f"Warning: Could not load credit spread data: {exc}")

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
            logger.info(f"Loaded regime input data: {len(regime_df)} observations")
        except Exception as exc:
            if config.features.regime_strict:
                raise
            logger.warning(f"Warning: Could not load regime input data: {exc}")
            logger.warning("Continuing with zero-filled regime features (soft-fail)")

    return vix_df, credit_df, regime_df


def _compute_norm_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_end: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-feature mean/std from the training period."""
    return compute_zscore_norm_stats(df, feature_cols, train_end)


def _build_feature_reference(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Build train-only quantile bins and histogram counts for drift monitoring."""
    features: dict[str, Any] = {}
    for col in feature_cols:
        if col not in train_df.columns:
            continue
        values = pd.to_numeric(train_df[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        bins = np.quantile(values, np.linspace(0.0, 1.0, 11))
        if np.unique(bins).size < 2:
            center = float(bins[0])
            bins = np.linspace(center - 1e-6, center + 1e-6, 11)
        else:
            bins = np.maximum.accumulate(bins)
            for i in range(1, len(bins)):
                if bins[i] <= bins[i - 1]:
                    bins[i] = bins[i - 1] + 1e-9
        counts, _ = np.histogram(values, bins=bins)
        features[col] = {
            "bins": [float(v) for v in bins],
            "counts": [int(v) for v in counts],
        }
    return {"features": features}


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


def _filter_to_masked_pit_panel(
    df: pd.DataFrame,
    kdcode_list: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Keep PIT-union rows in the experiment calendar without requiring full coverage."""
    mask = df["kdcode"].isin(kdcode_list) & (df["dt"] >= start) & (df["dt"] <= end)
    return df.loc[mask].sort_values(["dt", "kdcode"]).reset_index(drop=True)


def _audit_pit_breadth(
    split_name: str,
    dates: list[str],
    tradable_mask: np.ndarray,
    min_scoreable: int,
    policy: str,
) -> list[dict[str, int | str]]:
    """Return daily breadth diagnostics and enforce the configured threshold."""
    summary = candidate_breadth(dates, tradable_mask)
    if policy == "off" or min_scoreable <= 0:
        return summary
    low = [row for row in summary if int(row["scoreable_count"]) < min_scoreable]
    if not low:
        return summary
    preview = ", ".join(f"{row['date']}={row['scoreable_count']}" for row in low[:5])
    message = (
        f"PIT {split_name} breadth below {min_scoreable} on {len(low)} dates "
        f"(first: {preview}). Check ticker mapping/data outages or lower "
        "data.pit_min_scoreable_stocks with an explicit explanation."
    )
    if policy == "error":
        raise ValueError(message)
    logger.warning(f"Warning: {message}")
    return summary


def _pit_mask_summary(dates: list[str], masks) -> list[dict[str, int | str]]:
    """Daily active/readiness/loss/tradable counts for run metadata."""
    return [
        {
            "date": str(date),
            "active_count": int(masks.active_member[i].sum()),
            "feature_ready_count": int(masks.feature_ready[i].sum()),
            "loss_count": int(masks.loss[i].sum()),
            "scoreable_count": int(masks.tradable[i].sum()),
        }
        for i, date in enumerate(dates)
    ]


def _apply_normalisation(
    df: pd.DataFrame,
    feature_cols: list[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> pd.DataFrame:
    """3-sigma clipping followed by z-score normalisation."""
    return normalize_features_zscore(df, feature_cols, means, stds)


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


def _embargo_training_sessions(
    train_dates: list[str],
    df_for_labels: pd.DataFrame,
    kdcode_list: list[str],
    val_start: str,
    his_t: int,
    label_t: int,
) -> list[str]:
    """Purge the training tail for the embargo, then verify it on the real session axis.

    The purge keeps the configured split dates untouched, so no YAML needs editing. The
    verification is unconditional -- ``data.skip_embargo_check`` governs the cheap
    calendar-day check in ``ExperimentConfig`` only, not this data-backed one.
    """
    purged = purge_training_sessions_for_embargo(train_dates, his_t, label_t)
    dropped = len(train_dates) - len(purged)
    if dropped:
        logger.info(
            f"Embargo purge: dropped the last {dropped} training session(s) "
            f"({train_dates[len(purged)]}..{train_dates[-1]}) so training labels mature "
            f"before val_start={val_start} (label_t={label_t} sessions)"
        )
    summary = assert_training_labels_respect_embargo(
        df_for_labels, kdcode_list, purged[his_t:], val_start, label_t
    )
    logger.info(f"Session-level train/val embargo verified: {summary}")
    return purged


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
    fill_missing_labels: bool = True,
) -> dict[str, Any]:
    """Build time-series tensors, graph features, and labels."""
    logger.info("Generating time series features...")
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

    logger.info("Generating graph features...")
    x_graph_train = generate_graph_features(
        train_df, kdcode_list, feature_cols, train_dates[his_t:]
    )
    x_graph_val = generate_graph_features(val_df, kdcode_list, feature_cols, val_dates)
    x_graph_test = generate_graph_features(test_df, kdcode_list, feature_cols, test_dates)

    logger.info("Computing labels...")
    train_labels = compute_labels(
        df_for_labels,
        kdcode_list,
        train_dates[his_t:],
        label_t,
        fill_missing=fill_missing_labels,
    )
    val_labels = compute_labels(
        df_for_labels,
        kdcode_list,
        val_dates,
        label_t,
        fill_missing=fill_missing_labels,
    )
    test_labels = compute_labels(
        df_for_labels,
        kdcode_list,
        test_dates,
        label_t,
        fill_missing=fill_missing_labels,
    )

    if label_type == "rank":
        logger.info("Converting labels to cross-sectional rank percentiles...")
        train_labels = apply_rank_labels(train_labels)
        val_labels = apply_rank_labels(val_labels)
        test_labels = apply_rank_labels(test_labels)

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
        "test_labels": test_labels,
    }


# ── staged pipeline functions ────────────────────────────────────────────


def load_raw_data(config: ExperimentConfig) -> tuple[DataManager, pd.DataFrame]:
    data_manager = DataManager(config.data)
    df = data_manager.load()
    return data_manager, df


def engineer_features(
    df: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    vix_df: pd.DataFrame | None,
    credit_df: pd.DataFrame | None,
    regime_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    df = feature_engineer.transform(df, vix_df, credit_df, regime_df)
    feature_cols = feature_engineer.get_feature_columns()
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    return df, feature_cols


def resolve_pit_context(config: ExperimentConfig) -> PitContext:
    masked_panel = config.data.use_pit_universe and config.data.pit_universe_mode == "masked_panel"
    csv_path = config.data.pit_universe_csv if config.data.use_pit_universe else None
    intervals: pd.DataFrame | None = None
    if config.data.use_pit_universe:
        if not config.data.pit_universe_csv:
            raise ValueError("data.use_pit_universe=true requires data.pit_universe_csv")
        intervals = load_pit_intervals(config.data.pit_universe_csv)
    return PitContext(intervals=intervals, masked_panel=masked_panel, csv_path=csv_path)


def fit_normalisation(
    df_filled: pd.DataFrame,
    feature_cols: list[str],
    train_end: str,
    mode: str,
    pit: PitContext,
) -> tuple[NormFit, pd.DataFrame]:
    rank_gauss_reference: dict[str, np.ndarray] | None = None
    if mode == "zscore":
        norm_source = df_filled
        if pit.masked_panel:
            norm_source = _apply_pit_universe(df_filled, pit.csv_path)
        means, stds = _compute_norm_stats(norm_source, feature_cols, train_end)
        df_norm = _apply_normalisation(df_filled, feature_cols, means, stds)
    elif mode == "rank_gauss":
        logger.info("Applying rank-Gaussian normalisation (train fit)...")
        rank_source = df_filled
        if pit.masked_panel:
            rank_source = _apply_pit_universe(df_filled, pit.csv_path)
        train_mask = rank_source["dt"] <= train_end
        train_slice = rank_source.loc[train_mask]
        rank_gauss_reference = fit_rank_gaussian_reference(train_slice, feature_cols)
        df_norm = apply_rank_gaussian(df_filled, feature_cols, rank_gauss_reference)
        means, stds = {}, {}
    else:
        raise ValueError(f"Unknown normalisation: {mode!r}")
    return (
        NormFit(means=means, stds=stds, rank_gauss_reference=rank_gauss_reference),
        df_norm,
    )


def select_universe(
    df_norm: pd.DataFrame,
    data_manager: DataManager,
    config: ExperimentConfig,
    pit: PitContext,
) -> tuple[pd.DataFrame, list[str]]:
    if pit.masked_panel:
        assert pit.intervals is not None
        kdcode_list = active_kdcodes_in_period(
            pit.intervals,
            config.data.train_start,
            config.data.test_end,
            available_kdcodes=set(df_norm["kdcode"].astype(str).unique()),
        )
        if not kdcode_list:
            raise ValueError("No PIT-union stocks overlap the configured experiment dates")
        df_filtered = _filter_to_masked_pit_panel(
            df_norm,
            kdcode_list,
            config.data.train_start,
            config.data.test_end,
        )
        data_manager.kdcode_list = kdcode_list
        logger.info(
            f"  PIT union axis: {len(kdcode_list)} stocks with any membership interval "
            f"from {config.data.train_start} to {config.data.test_end}"
        )
    elif config.data.filter_stocks_per_split:
        df_filtered, kdcode_list = data_manager.filter_complete_stocks_per_split(df_norm)
    else:
        df_filtered, kdcode_list = data_manager.filter_complete_stocks(df_norm)
    return df_filtered, kdcode_list


def build_tensors(
    frames: PipelineFrames,
    kdcode_list: list[str],
    feature_cols: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_dates: list[str],
    val_dates: list[str],
    test_dates: list[str],
    his_t: int,
    label_t: int,
    label_type: str,
    *,
    use_polars: bool = False,
    fill_missing_labels: bool = True,
) -> TensorBundle:
    result = _build_tensors(
        frames.filtered,
        kdcode_list,
        feature_cols,
        train_dates,
        val_dates,
        test_dates,
        his_t,
        label_t,
        label_type,
        frames.raw,
        train_df,
        val_df,
        test_df,
        use_polars=use_polars,
        fill_missing_labels=fill_missing_labels,
    )
    return TensorBundle(**result)


def apply_pit_masks_to_tensors(
    tensors: TensorBundle,
    frames: PipelineFrames,
    kdcode_list: list[str],
    pit: PitContext,
    his_t: int,
    label_t: int,
    label_type: str,
    min_scoreable: int,
    breadth_policy: str,
) -> tuple[TensorBundle, dict[str, list[dict[str, int | str]]]]:
    assert pit.intervals is not None
    train_masks = build_pit_masks(
        frames.filtered,
        frames.raw,
        kdcode_list,
        tensors.train_dates,
        his_t,
        label_t,
        pit.intervals,
    )
    val_masks = build_pit_masks(
        frames.filtered,
        frames.raw,
        kdcode_list,
        tensors.val_dates,
        his_t,
        label_t,
        pit.intervals,
    )
    test_masks = build_pit_masks(
        frames.filtered,
        frames.raw,
        kdcode_list,
        tensors.test_dates,
        his_t,
        label_t,
        pit.intervals,
    )

    train_labels = apply_label_mask(tensors.train_labels, train_masks.loss)
    val_labels = apply_label_mask(tensors.val_labels, val_masks.loss)
    test_labels = apply_label_mask(tensors.test_labels, test_masks.loss)
    if label_type == "rank":
        logger.info("Converting masked labels to PIT cross-sectional rank percentiles...")
        train_labels = apply_rank_labels(train_labels, train_masks.loss)
        val_labels = apply_rank_labels(val_labels, val_masks.loss)
        test_labels = apply_rank_labels(test_labels, test_masks.loss)

    _audit_pit_breadth(
        "train",
        tensors.train_dates,
        train_masks.tradable,
        min_scoreable,
        breadth_policy,
    )
    _audit_pit_breadth(
        "val",
        tensors.val_dates,
        val_masks.tradable,
        min_scoreable,
        breadth_policy,
    )
    _audit_pit_breadth(
        "test",
        tensors.test_dates,
        test_masks.tradable,
        min_scoreable,
        breadth_policy,
    )
    pit_breadth = {
        "train": _pit_mask_summary(tensors.train_dates, train_masks),
        "val": _pit_mask_summary(tensors.val_dates, val_masks),
        "test": _pit_mask_summary(tensors.test_dates, test_masks),
    }

    masked = TensorBundle(
        train_dates=tensors.train_dates,
        val_dates=tensors.val_dates,
        test_dates=tensors.test_dates,
        stock_features_train=tensors.stock_features_train,
        stock_features_val=tensors.stock_features_val,
        stock_features_test=tensors.stock_features_test,
        x_graph_train=tensors.x_graph_train,
        x_graph_val=tensors.x_graph_val,
        x_graph_test=tensors.x_graph_test,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        train_active_member_mask=train_masks.active_member,
        val_active_member_mask=val_masks.active_member,
        test_active_member_mask=test_masks.active_member,
        train_feature_ready_mask=train_masks.feature_ready,
        val_feature_ready_mask=val_masks.feature_ready,
        test_feature_ready_mask=test_masks.feature_ready,
        train_loss_mask=train_masks.loss,
        val_loss_mask=val_masks.loss,
        test_loss_mask=test_masks.loss,
        train_tradable_mask=train_masks.tradable,
        val_tradable_mask=val_masks.tradable,
        test_tradable_mask=test_masks.tradable,
    )
    return masked, pit_breadth


def build_correlation_graph(
    frames: PipelineFrames,
    kdcode_list: list[str],
    graph_config: GraphConfig,
    train_start: str,
    test_end: str,
) -> GraphArtifacts:
    logger.info("Building correlation graph...")
    graph_builder = GraphBuilder(
        judge_value=graph_config.judge_value,
        update_frequency_months=graph_config.update_frequency_months,
        corr_lookback_days=graph_config.corr_lookback_days,
        top_k=graph_config.top_k,
        top_k_metric=graph_config.top_k_metric,
        use_multi_feature_edges=graph_config.use_multi_feature_edges,
        use_lead_lag_features=graph_config.use_lead_lag_features,
        lead_lag_days=graph_config.lead_lag_days,
    )
    edge_index, edge_weight = graph_builder.build_graph(frames.raw, kdcode_list, train_start)

    graph_schedule = None
    if graph_config.update_frequency_months > 0:
        graph_schedule = graph_builder.precompute_snapshots(
            frames.raw, kdcode_list, train_start, test_end
        )

    edge_index_sector = None
    edge_weight_sector = None
    if graph_config.use_sector_relation and graph_config.sector_map_csv:
        sector_map = load_sector_map_csv(graph_config.sector_map_csv)
        edge_index_sector, edge_weight_sector = build_sector_edges(
            kdcode_list,
            sector_map,
            graph_config.sector_top_k,
        )

    return GraphArtifacts(
        edge_index=edge_index,
        edge_weight=edge_weight,
        graph_schedule=graph_schedule,
        edge_index_sector=edge_index_sector,
        edge_weight_sector=edge_weight_sector,
    )


# ── public API ───────────────────────────────────────────────────────────


def prepare_data(
    config: ExperimentConfig,
    feature_engineer: FeatureEngineer,
) -> dict[str, Any]:
    """Load and prepare stock-level cross-sectional data for training.

    Returns a dict consumed by the training loop and metric evaluation.
    """
    logger.info("=" * 80)
    logger.info("Preparing Data")
    logger.info("=" * 80)

    data_manager, df = load_raw_data(config)
    vix_df, credit_df, regime_df = load_auxiliary_data(data_manager, config)
    df, feature_cols = engineer_features(df, feature_engineer, vix_df, credit_df, regime_df)

    logger.info("Filling NaN values...")
    df_filled = impute_feature_nans_by_day(df, feature_cols)
    gc.collect()

    pit = resolve_pit_context(config)
    if pit.intervals is not None:
        if pit.masked_panel:
            logger.info("Using true PIT masked-panel mode (fixed union axis + daily masks)...")
        else:
            logger.info("Applying legacy PIT universe row filter...")
            df_filled = _apply_pit_universe(df_filled, pit.csv_path)

    norm_fit, df_norm = fit_normalisation(
        df_filled,
        feature_cols,
        config.data.train_end,
        config.data.normalisation,
        pit,
    )
    del df_filled
    gc.collect()

    df_filtered, kdcode_list = select_universe(df_norm, data_manager, config, pit)
    frames = PipelineFrames(raw=df, normalized=df_norm, filtered=df_filtered)

    train_df, val_df, test_df = data_manager.split_by_period(df_filtered)
    feature_reference = _build_feature_reference(train_df, feature_cols)

    train_dates = sorted(train_df["dt"].unique())
    val_dates = sorted(val_df["dt"].unique())
    test_dates = sorted(test_df["dt"].unique())

    train_dates = _embargo_training_sessions(
        train_dates,
        frames.raw,
        kdcode_list,
        config.data.val_start,
        config.model.his_t,
        config.model.label_t,
    )

    tensor_bundle = build_tensors(
        frames,
        kdcode_list,
        feature_cols,
        train_df,
        val_df,
        test_df,
        train_dates,
        val_dates,
        test_dates,
        config.model.his_t,
        config.model.label_t,
        "returns" if pit.masked_panel else config.training.label_type,
        use_polars=config.data.use_polars,
        fill_missing_labels=not pit.masked_panel,
    )

    pit_breadth: dict[str, list[dict[str, int | str]]] | None = None
    if pit.masked_panel:
        tensor_bundle, pit_breadth = apply_pit_masks_to_tensors(
            tensor_bundle,
            frames,
            kdcode_list,
            pit,
            config.model.his_t,
            config.model.label_t,
            config.training.label_type,
            config.data.pit_min_scoreable_stocks,
            config.data.pit_breadth_policy,
        )

    graphs = build_correlation_graph(
        frames,
        kdcode_list,
        config.graph,
        config.data.train_start,
        config.data.test_end,
    )

    return {
        "kdcode_list": kdcode_list,
        **tensor_bundle.to_dict(),
        "edge_index": graphs.edge_index,
        "edge_weight": graphs.edge_weight,
        "feature_cols": feature_cols,
        "graph_schedule": graphs.graph_schedule,
        "df": df,
        "norm_means": norm_fit.means,
        "norm_stds": norm_fit.stds,
        "graph_static_valid_from": config.data.train_start,
        "edge_index_sector": graphs.edge_index_sector,
        "edge_weight_sector": graphs.edge_weight_sector,
        "rank_gauss_reference": norm_fit.rank_gauss_reference,
        "feature_reference": feature_reference,
        "pit_breadth": pit_breadth,
        "pit_universe_mode": config.data.pit_universe_mode
        if config.data.use_pit_universe
        else None,
    }


def prepare_data_index_level(
    config: ExperimentConfig,
    feature_engineer: FeatureEngineer,
) -> dict[str, Any]:
    """Prepare data for index-level mode (single series, no survivorship bias).

    Uses a trivial 1-node / 0-edge graph so the rest of the pipeline runs
    unchanged.
    """
    logger.info("=" * 80)
    logger.info("Preparing Data (index-level mode; no stock-level survivorship bias)")
    logger.info("=" * 80)

    data_manager = DataManager(config.data)
    df = data_manager.load_index_series()
    kdcode_list = ["INDEX"]

    vix_df, credit_df, regime_df = load_auxiliary_data(data_manager, config)

    df = feature_engineer.transform(df, vix_df, credit_df, regime_df)
    feature_cols = feature_engineer.get_feature_columns()
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

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
    train_df = df_filtered[
        (df_filtered["dt"] >= config.data.train_start)
        & (df_filtered["dt"] <= config.data.train_end)
    ]
    val_df = df_filtered[
        (df_filtered["dt"] >= config.data.val_start) & (df_filtered["dt"] <= config.data.val_end)
    ]
    test_df = df_filtered[
        (df_filtered["dt"] >= config.data.test_start) & (df_filtered["dt"] <= config.data.test_end)
    ]
    feature_reference = _build_feature_reference(train_df, feature_cols)

    train_dates = sorted(train_df["dt"].unique())
    val_dates = sorted(val_df["dt"].unique())
    test_dates = sorted(test_df["dt"].unique())

    train_dates = _embargo_training_sessions(
        train_dates,
        df_filtered,
        kdcode_list,
        config.data.val_start,
        config.model.his_t,
        config.model.label_t,
    )

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
        "feature_reference": feature_reference,
    }
