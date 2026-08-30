"""Static same-sector edges for dual-GAT fusion (Phase 3)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Values that state "no sector is known" rather than naming one.
_MISSING_SECTOR_VALUES = frozenset({"", "nan", "none", "null", "na", "n/a", "unknown"})

# Legacy curated map first, then the universe metadata export.
_SECTOR_COLUMNS = ("sector", "gics_sector")


def _is_missing(value: str) -> bool:
    return value.strip().lower() in _MISSING_SECTOR_VALUES


def load_sector_map_csv(path: str) -> dict[str, str]:
    """Load ``kdcode -> sector`` from a curated map or a universe metadata export.

    Two schemas are accepted:

    * the legacy curated map, with columns ``kdcode`` and ``sector``;
    * the universe metadata export (``as_of_date, kdcode, company_name,
      company_market_cap, gics_sector, gics_sector_field``), from which the map is
      **derived** rather than hand-maintained. The export carries one row per
      ``(as_of_date, kdcode)``; sector assignment is stable across snapshots, so
      the newest non-missing value wins and any disagreement is logged.

    Names whose sector is blank or a "no sector known" placeholder are omitted
    from the map entirely, so :func:`build_sector_edges` isolates them.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"sector_map_csv not found: {path}")
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        if "kdcode" not in fields:
            raise ValueError("sector_map_csv must have columns kdcode, sector")
        sector_col = next((fields[name] for name in _SECTOR_COLUMNS if name in fields), None)
        if sector_col is None:
            raise ValueError("sector_map_csv must have columns kdcode, sector")
        kdcode_col = fields["kdcode"]
        as_of_col = fields.get("as_of_date")

        out: dict[str, str] = {}
        seen_as_of: dict[str, str] = {}
        conflicts: set[str] = set()
        for row in reader:
            kdcode = str(row[kdcode_col]).strip()
            sector = str(row[sector_col] or "").strip()
            if not kdcode or _is_missing(sector):
                continue
            as_of = str(row[as_of_col] or "").strip() if as_of_col else ""
            previous = out.get(kdcode)
            if previous is not None and previous != sector:
                conflicts.add(kdcode)
            if previous is None or as_of >= seen_as_of.get(kdcode, ""):
                out[kdcode] = sector
                seen_as_of[kdcode] = as_of

    if conflicts:
        logger.warning(
            f"Sector map: {len(conflicts)} kdcode(s) carry more than one sector across "
            f"snapshots (newest wins), e.g. {sorted(conflicts)[:5]}"
        )
    logger.info(f"Sector map: {len(out)} kdcode(s) with a known sector from {p.name}")
    return out


def build_sector_edges(
    kdcode_list: list[str],
    sector_by_kdcode: dict[str, str],
    exclude_pairs: Sequence[Sequence[str]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed sector edges: every ordered pair of distinct names sharing a sector.

    Sectors are fully connected internally. The previous ``sector_top_k`` cap kept
    the first K peers by ascending node index, which on an alphabetically ordered
    panel encodes ticker spelling rather than any economic ranking; full connection
    is order-independent and the edge count stays modest at realistic sector sizes.

    Names with no sector assignment are **isolated** — they get no sector edges at
    all. Bucketing them together under a shared ``UNKNOWN`` label would wire every
    unmapped name to every other one, which is pure noise.

    *exclude_pairs* names kdcode pairs whose edges are withheld in both
    directions (issue 164 hygiene rule: a same-company twin shares a sector by
    construction, so the correlation-side exclusion alone would leave the pair
    wired here).

    Returns ``(edge_index (2, E), edge_weight (E,))`` with scalar weight 1.0.
    """
    n = len(kdcode_list)
    if n == 0:
        z = torch.zeros((2, 0), dtype=torch.long)
        return z, torch.zeros(0, dtype=torch.float)

    excluded = {frozenset(pair) for pair in exclude_pairs} if exclude_pairs else set()

    buckets: dict[str, list[int]] = {}
    isolated = 0
    for idx, kdcode in enumerate(kdcode_list):
        sector = str(sector_by_kdcode.get(kdcode, "") or "").strip()
        if _is_missing(sector):
            isolated += 1
            continue
        buckets.setdefault(sector, []).append(idx)

    rows: list[int] = []
    cols: list[int] = []
    skipped = 0
    for group in buckets.values():
        for src in group:
            for dst in group:
                if src == dst:
                    continue
                if excluded and frozenset((kdcode_list[src], kdcode_list[dst])) in excluded:
                    skipped += 1
                    continue
                rows.append(src)
                cols.append(dst)

    logger.info(
        f"Sector edges: {n - isolated}/{n} name(s) mapped "
        f"({100.0 * (n - isolated) / n:.1f}% coverage) across {len(buckets)} sector(s); "
        f"{isolated} name(s) isolated with no sector edges; {len(rows)} directed edge(s)"
        + (f"; {skipped} excluded-pair edge(s) withheld" if skipped else "")
    )

    if not rows:
        z = torch.zeros((2, 0), dtype=torch.long)
        return z, torch.zeros(0, dtype=torch.float)

    ei = torch.tensor([rows, cols], dtype=torch.long)
    ew = torch.ones(len(rows), dtype=torch.float)
    return ei, ew
