"""Point-in-time universe masks for fixed-axis stock panels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class PITMaskSet:
    active_member: np.ndarray
    feature_ready: np.ndarray
    loss: np.ndarray
    tradable: np.ndarray


def normalise_pit_intervals(pit_intervals: pd.DataFrame) -> pd.DataFrame:
    """Return ``kdcode, valid_from, valid_to`` intervals with normalised dates."""
    frame = pit_intervals.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "constituent_ric" in frame.columns and "kdcode" not in frame.columns:
        frame = frame.rename(columns={"constituent_ric": "kdcode"})
    required = {"kdcode", "valid_from", "valid_to"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"PIT intervals missing columns: {sorted(missing)}")
    frame = frame[["kdcode", "valid_from", "valid_to"]].copy()
    frame = frame.dropna(subset=["kdcode", "valid_from", "valid_to"])
    frame["kdcode"] = frame["kdcode"].astype(str).str.strip()
    frame["valid_from"] = pd.to_datetime(frame["valid_from"]).dt.strftime("%Y-%m-%d")
    frame["valid_to"] = pd.to_datetime(frame["valid_to"]).dt.strftime("%Y-%m-%d")
    return frame[frame["kdcode"] != ""].reset_index(drop=True)


def load_pit_intervals(csv_path: str) -> pd.DataFrame:
    return normalise_pit_intervals(pd.read_csv(csv_path))


def active_kdcodes_in_period(
    pit_intervals: pd.DataFrame,
    start: str,
    end: str,
    available_kdcodes: set[str] | None = None,
) -> list[str]:
    """Return tickers whose PIT interval overlaps ``[start, end]``."""
    intervals = normalise_pit_intervals(pit_intervals)
    mask = (intervals["valid_from"] <= end) & (intervals["valid_to"] >= start)
    values = set(intervals.loc[mask, "kdcode"].astype(str))
    if available_kdcodes is not None:
        values &= {str(k) for k in available_kdcodes}
    return sorted(values)


def active_membership_mask(
    kdcode_list: list[str],
    dates: list[str],
    pit_intervals: pd.DataFrame,
) -> np.ndarray:
    """Boolean ``(dates, stocks)`` membership mask from PIT intervals."""
    intervals = normalise_pit_intervals(pit_intervals)
    by_kdcode: dict[str, list[tuple[str, str]]] = {}
    for row in intervals.itertuples(index=False):
        by_kdcode.setdefault(str(row.kdcode), []).append((row.valid_from, row.valid_to))

    out = np.zeros((len(dates), len(kdcode_list)), dtype=bool)
    for j, kdcode in enumerate(kdcode_list):
        ranges = by_kdcode.get(str(kdcode), [])
        if not ranges:
            continue
        for i, date in enumerate(dates):
            out[i, j] = any(start <= date <= end for start, end in ranges)
    return out


def feature_ready_mask(
    df_for_features: pd.DataFrame,
    kdcode_list: list[str],
    sample_dates: list[str],
    his_t: int,
) -> np.ndarray:
    """True when a stock has a complete pre-sample lookback window."""
    all_dates = sorted(df_for_features["dt"].astype(str).unique())
    date_to_idx = {date: i for i, date in enumerate(all_dates)}
    date_index = {date: idx for idx, date in enumerate(all_dates)}
    stock_index = {kdcode: idx for idx, kdcode in enumerate(kdcode_list)}
    presence = np.zeros((len(all_dates), len(kdcode_list)), dtype=bool)

    subset = df_for_features[["kdcode", "dt"]].drop_duplicates()
    for row in subset.itertuples(index=False):
        kdcode = str(row.kdcode)
        date = str(row.dt)
        if kdcode in stock_index and date in date_index:
            presence[date_index[date], stock_index[kdcode]] = True

    out = np.zeros((len(sample_dates), len(kdcode_list)), dtype=bool)
    for i, date in enumerate(sample_dates):
        date = str(date)
        end_idx = date_to_idx.get(date)
        if end_idx is None or end_idx < his_t:
            continue
        window = presence[end_idx - his_t : end_idx, :]
        out[i, :] = window.all(axis=0)
    return out


def label_available_mask(
    df_for_labels: pd.DataFrame,
    kdcode_list: list[str],
    sample_dates: list[str],
    label_t: int,
) -> np.ndarray:
    """True when the existing forward-return label formula is observable."""
    subset = df_for_labels[df_for_labels["kdcode"].isin(kdcode_list)].copy()
    subset = subset.sort_values(["kdcode", "dt"])
    subset["future_close"] = subset.groupby("kdcode")["close"].shift(-label_t)
    subset["next_close"] = subset.groupby("kdcode")["close"].shift(-1)
    subset["forward_return"] = subset["future_close"] / subset["next_close"] - 1
    subset = subset[subset["dt"].isin(sample_dates)]
    pivot = subset.pivot_table(index="dt", columns="kdcode", values="forward_return")
    pivot = pivot.reindex(index=sample_dates, columns=kdcode_list)
    return np.isfinite(pivot.to_numpy(dtype=np.float64))


def build_pit_masks(
    df_for_features: pd.DataFrame,
    df_for_labels: pd.DataFrame,
    kdcode_list: list[str],
    sample_dates: list[str],
    his_t: int,
    label_t: int,
    pit_intervals: pd.DataFrame,
) -> PITMaskSet:
    active = active_membership_mask(kdcode_list, sample_dates, pit_intervals)
    ready = feature_ready_mask(df_for_features, kdcode_list, sample_dates, his_t)
    labels = label_available_mask(df_for_labels, kdcode_list, sample_dates, label_t)
    tradable = active & ready
    loss = tradable & labels
    return PITMaskSet(
        active_member=active,
        feature_ready=ready,
        loss=loss,
        tradable=tradable,
    )


def apply_label_mask(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(labels, dtype=np.float32).copy()
    out[~np.asarray(mask, dtype=bool)] = np.nan
    return out


def candidate_breadth(dates: list[str], tradable_mask: np.ndarray) -> list[dict[str, int | str]]:
    mask = np.asarray(tradable_mask, dtype=bool)
    return [
        {"date": str(date), "scoreable_count": int(mask[i].sum())} for i, date in enumerate(dates)
    ]


def filter_edges_by_stock_mask(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    stock_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove edges whose source or destination node is inactive."""
    if edge_index.numel() == 0:
        return edge_index, edge_weight
    mask = stock_mask.to(dtype=torch.bool, device=edge_index.device)
    edge_keep = mask[edge_index[0]] & mask[edge_index[1]]
    return edge_index[:, edge_keep], edge_weight[edge_keep]
