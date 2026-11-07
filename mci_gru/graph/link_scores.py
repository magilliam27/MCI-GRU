"""Pairwise company-link scores as a stand-in for the correlation matrix (#377).

A link-score file names how closely two companies are tied, scored outside this
repository (the Jev project's "Broad R" question on two 10-K business
descriptions). The graph builder ranks neighbours on the scores instead of on
past return correlation; every other part of edge construction is unchanged.

A file may carry several score sets, one per ``score_year`` (the year whose
10-Ks were read). Each graph snapshot then uses, for every pair, the newest
score whose two filings both precede the snapshot's valid-from date, so a
yearly schedule picks up each year's filings as they become public.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mci_gru.data.path_resolver import resolve_project_data_path

if TYPE_CHECKING:
    from mci_gru.data.input_observations import InputObservationContext

logger = logging.getLogger(__name__)

LINK_SCORE_COLUMNS = ("kdcode_a", "kdcode_b", "score", "filing_date_a", "filing_date_b")
_TEXT_COLUMNS = (
    ("kdcode_a", "str"),
    ("kdcode_b", "str"),
    ("filing_date_a", "str"),
    ("filing_date_b", "str"),
)
SCORE_YEAR_COLUMN = "score_year"


def load_link_scores_csv(
    path: str, *, input_observations: InputObservationContext | None = None
) -> pd.DataFrame:
    """Load and validate a link-score file: one row per unordered pair of kdcodes.

    Required columns are :data:`LINK_SCORE_COLUMNS`; an optional integer
    ``score_year`` column holds several yearly score sets; others are ignored.
    Scores must be finite and non-negative, filing dates ``YYYY-MM-DD``, and each
    pair of distinct kdcodes must appear once in either order (once per
    ``score_year`` when that column is present).
    """
    resolved = resolve_project_data_path(path, allow_basename_fallback=False)
    if input_observations is not None:
        scores = input_observations.read_file(
            resolved,
            role="graph.link_scores_csv",
            configured_path=path,
            parse=lambda content: _parse_link_scores(content, path),
            parser={"name": "pandas.read_csv", "options": {"dtype": dict(_TEXT_COLUMNS)}},
        )
    else:
        scores = _parse_link_scores(Path(resolved).read_bytes(), path)
    logger.info(
        "Link scores: %d pair(s) over %d kdcode(s) from %s",
        len(scores),
        len(set(scores["kdcode_a"]) | set(scores["kdcode_b"])),
        Path(resolved).name,
    )
    return scores


def _parse_link_scores(content: bytes, path: str) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(content), dtype=dict(_TEXT_COLUMNS))
    missing = [column for column in LINK_SCORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"link_scores_csv {path} lacks column(s) {missing}")
    columns = list(LINK_SCORE_COLUMNS)
    if SCORE_YEAR_COLUMN in frame.columns:
        columns.append(SCORE_YEAR_COLUMN)
    frame = frame[columns].copy()
    if frame.isna().any().any():
        raise ValueError(f"link_scores_csv {path} has blank cells")
    if SCORE_YEAR_COLUMN in frame.columns:
        years = pd.to_numeric(frame[SCORE_YEAR_COLUMN], errors="coerce")
        if years.isna().any() or (years != years.round()).any():
            raise ValueError(f"link_scores_csv {path}: {SCORE_YEAR_COLUMN} must be whole years")
        frame[SCORE_YEAR_COLUMN] = years.astype(int)
    frame["score"] = frame["score"].astype(float)
    if not np.isfinite(frame["score"]).all() or (frame["score"] < 0).any():
        raise ValueError(f"link_scores_csv {path} has a negative or non-finite score")
    for column in ("filing_date_a", "filing_date_b"):
        canonical = pd.to_datetime(frame[column], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
        if not canonical.equals(frame[column]):
            raise ValueError(f"link_scores_csv {path}: {column} must be YYYY-MM-DD")
    if (frame["kdcode_a"] == frame["kdcode_b"]).any():
        raise ValueError(f"link_scores_csv {path} links a kdcode to itself")
    keys = pd.DataFrame(
        [sorted(pair) for pair in zip(frame["kdcode_a"], frame["kdcode_b"], strict=True)],
        columns=["low", "high"],
    )
    if SCORE_YEAR_COLUMN in frame.columns:
        keys[SCORE_YEAR_COLUMN] = frame[SCORE_YEAR_COLUMN].to_numpy()
    if keys.duplicated().any():
        per = f" in one {SCORE_YEAR_COLUMN}" if SCORE_YEAR_COLUMN in frame.columns else ""
        raise ValueError(f"link_scores_csv {path} scores a pair more than once{per}")
    return frame.reset_index(drop=True)


def link_score_matrix(
    scores: pd.DataFrame, kdcode_list: list[str], valid_from: str
) -> pd.DataFrame:
    """Symmetric ``kdcode x kdcode`` matrix of scores scaled to ``(0, 1]``, for *valid_from*.

    Only a score whose two filings are both dated before *valid_from* is usable:
    the graph serves samples from that date on, so a later filing would be
    lookahead. Of a pair's usable scores the one with the newest filings wins
    (newest ``score_year`` when the file has one). Raises when no score is
    usable at all, which means the file does not cover this snapshot.

    Unscored pairs, zero scores and the diagonal are NaN, which edge selection
    skips; names absent from *kdcode_list* are dropped. Scores are divided by the
    largest score this snapshot places on the axis, so column 0 of the edge
    features reads like a correlation.
    """
    usable = (scores["filing_date_a"] < valid_from) & (scores["filing_date_b"] < valid_from)
    if not usable.any():
        raise ValueError(
            f"no link score has both filings dated before the graph's valid-from {valid_from}"
        )
    chosen = scores.loc[usable].copy()
    chosen["low"] = chosen[["kdcode_a", "kdcode_b"]].min(axis=1)
    chosen["high"] = chosen[["kdcode_a", "kdcode_b"]].max(axis=1)
    chosen["newest_filing"] = chosen[["filing_date_a", "filing_date_b"]].max(axis=1)
    order = (
        [SCORE_YEAR_COLUMN, "newest_filing"] if SCORE_YEAR_COLUMN in chosen else ["newest_filing"]
    )
    chosen = chosen.sort_values(order, kind="stable").drop_duplicates(["low", "high"], keep="last")

    position = {code: index for index, code in enumerate(kdcode_list)}
    on_axis = chosen["kdcode_a"].isin(position) & chosen["kdcode_b"].isin(position)
    kept = chosen.loc[on_axis]
    rows = kept["kdcode_a"].map(position).to_numpy()
    cols = kept["kdcode_b"].map(position).to_numpy()
    scale = float(kept["score"].max()) if len(kept) else 1.0
    # A zero score says the two companies are not linked, so it is no candidate at all.
    raw = kept["score"].to_numpy(dtype=float)
    values = np.where(raw > 0, raw, np.nan) / (scale or 1.0)
    matrix = np.full((len(kdcode_list), len(kdcode_list)), np.nan)
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    logger.info(
        "Link-score matrix at %s: %d scored pair(s) on a %d-name axis "
        "(%d of %d row(s) filed before it)%s",
        valid_from,
        len(kept),
        len(kdcode_list),
        int(usable.sum()),
        len(scores),
        (
            f", score_year {chosen[SCORE_YEAR_COLUMN].min()}-{chosen[SCORE_YEAR_COLUMN].max()}"
            if SCORE_YEAR_COLUMN in chosen
            else ""
        ),
    )
    return pd.DataFrame(matrix, index=kdcode_list, columns=kdcode_list)
