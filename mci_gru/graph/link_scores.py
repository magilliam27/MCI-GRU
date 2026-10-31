"""Pairwise company-link scores as a stand-in for the correlation matrix (#377).

A link-score file names how closely two companies are tied, scored outside this
repository (the Jev project's "Broad R" question on two 10-K business
descriptions). The graph builder ranks neighbours on the scores instead of on
past return correlation; every other part of edge construction is unchanged.
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


def load_link_scores_csv(
    path: str, *, input_observations: InputObservationContext | None = None
) -> pd.DataFrame:
    """Load and validate a link-score file: one row per unordered pair of kdcodes.

    Required columns are :data:`LINK_SCORE_COLUMNS`; others are ignored. Scores
    must be finite and non-negative, filing dates ``YYYY-MM-DD``, and each pair
    of distinct kdcodes must appear once in either order.
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
    frame = frame[list(LINK_SCORE_COLUMNS)].copy()
    if frame.isna().any().any():
        raise ValueError(f"link_scores_csv {path} has blank cells")
    frame["score"] = frame["score"].astype(float)
    if not np.isfinite(frame["score"]).all() or (frame["score"] < 0).any():
        raise ValueError(f"link_scores_csv {path} has a negative or non-finite score")
    for column in ("filing_date_a", "filing_date_b"):
        canonical = pd.to_datetime(frame[column], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
        if not canonical.equals(frame[column]):
            raise ValueError(f"link_scores_csv {path}: {column} must be YYYY-MM-DD")
    if (frame["kdcode_a"] == frame["kdcode_b"]).any():
        raise ValueError(f"link_scores_csv {path} links a kdcode to itself")
    pairs = pd.Series(
        [tuple(sorted(pair)) for pair in zip(frame["kdcode_a"], frame["kdcode_b"], strict=True)]
    )
    if pairs.duplicated().any():
        raise ValueError(f"link_scores_csv {path} scores a pair more than once")
    return frame.reset_index(drop=True)


def link_score_matrix(
    scores: pd.DataFrame, kdcode_list: list[str], valid_from: str
) -> pd.DataFrame:
    """Symmetric ``kdcode x kdcode`` matrix of scores scaled to ``(0, 1]``, for *valid_from*.

    Refuses a score built on a filing dated on or after *valid_from*: the graph
    serves samples from that date on, so a later filing would be lookahead.
    Unscored pairs and the diagonal are NaN, which edge selection skips; names
    absent from *kdcode_list* are dropped. Scores are divided by the file's
    largest score, so column 0 of the edge features reads like a correlation.
    """
    late = (scores["filing_date_a"] >= valid_from) | (scores["filing_date_b"] >= valid_from)
    if late.any():
        first = scores.loc[late].iloc[0]
        raise ValueError(
            f"{int(late.sum())} link score(s) use a filing dated on or after the graph's "
            f"valid-from {valid_from}, e.g. {first['kdcode_a']} / {first['kdcode_b']}"
        )
    position = {code: index for index, code in enumerate(kdcode_list)}
    on_axis = scores["kdcode_a"].isin(position) & scores["kdcode_b"].isin(position)
    kept = scores.loc[on_axis]
    rows = kept["kdcode_a"].map(position).to_numpy()
    cols = kept["kdcode_b"].map(position).to_numpy()
    scale = float(scores["score"].max()) or 1.0
    values = kept["score"].to_numpy(dtype=float) / scale
    matrix = np.full((len(kdcode_list), len(kdcode_list)), np.nan)
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    logger.info(
        "Link-score matrix at %s: %d of %d scored pair(s) on a %d-name axis",
        valid_from,
        len(kept),
        len(scores),
        len(kdcode_list),
    )
    return pd.DataFrame(matrix, index=kdcode_list, columns=kdcode_list)
