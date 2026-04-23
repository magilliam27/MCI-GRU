"""Static same-sector edges for dual-GAT fusion (Phase 3)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


def load_sector_map_csv(path: str) -> dict[str, str]:
    """Load ``kdcode -> sector`` from a CSV with headers ``kdcode`` and ``sector``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"sector_map_csv not found: {path}")
    out: dict[str, str] = {}
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        if "kdcode" not in fields or "sector" not in fields:
            raise ValueError("sector_map_csv must have columns kdcode, sector")
        kc_col = fields["kdcode"]
        sec_col = fields["sector"]
        for row in reader:
            k = str(row[kc_col]).strip()
            s = str(row[sec_col]).strip()
            if k:
                out[k] = s
    return out


def build_sector_edges(
    kdcode_list: list[str],
    sector_by_kdcode: dict[str, str],
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed sector edges: each node links to up to ``top_k`` peers sharing sector.

    Returns ``(edge_index (2, E), edge_weight (E,))`` with scalar weight 1.0.
    """
    n = len(kdcode_list)
    if n == 0:
        z = torch.zeros((2, 0), dtype=torch.long)
        return z, torch.zeros(0, dtype=torch.float)

    sectors = [sector_by_kdcode.get(k, "UNKNOWN") for k in kdcode_list]
    buckets: dict[str, list[int]] = {}
    for idx, sec in enumerate(sectors):
        buckets.setdefault(sec, []).append(idx)

    rows: list[int] = []
    cols: list[int] = []
    for group in buckets.values():
        if len(group) <= 1:
            continue
        g_sorted = sorted(group)
        for i, src in enumerate(g_sorted):
            others = [x for x in g_sorted if x != src]
            if not others:
                continue
            take = others[:top_k]
            for dst in take:
                rows.append(src)
                cols.append(dst)

    if not rows:
        z = torch.zeros((2, 0), dtype=torch.long)
        return z, torch.zeros(0, dtype=torch.float)

    ei = torch.tensor([rows, cols], dtype=torch.long)
    ew = torch.ones(len(rows), dtype=torch.float)
    return ei, ew
