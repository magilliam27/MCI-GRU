"""Company-link scores as the neighbour ranking for the static graph (#377)."""

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import GraphConfig
from mci_gru.graph.builder import GraphBuilder
from mci_gru.graph.link_scores import link_score_matrix, load_link_scores_csv
from mci_gru.pipeline import PipelineFrames, build_correlation_graph

HEADER = "kdcode_a,kdcode_b,score,filing_date_a,filing_date_b,jev_version\n"


def _write(tmp_path, body, header=HEADER):
    path = tmp_path / "links.csv"
    path.write_text(header + body, encoding="utf-8")
    return str(path)


def _scores(rows):
    return pd.DataFrame(
        rows, columns=["kdcode_a", "kdcode_b", "score", "filing_date_a", "filing_date_b"]
    )


def test_a_valid_file_loads_with_float_scores(tmp_path):
    path = _write(tmp_path, "A,B,2.0,2019-02-01,2019-03-01,v\nA,C,0.5,2019-02-01,2019-04-01,v\n")
    scores = load_link_scores_csv(path)
    assert list(scores.columns) == [
        "kdcode_a",
        "kdcode_b",
        "score",
        "filing_date_a",
        "filing_date_b",
    ]
    assert scores["score"].tolist() == [2.0, 0.5]


@pytest.mark.parametrize(
    ("body", "header", "match"),
    [
        ("A,B,1.0,2019-02-01\n", "kdcode_a,kdcode_b,score,filing_date_a\n", "lacks column"),
        (
            "A,B,1.0,2019-02-01,2019-03-01,v\nB,A,2.0,2019-02-01,2019-03-01,v\n",
            HEADER,
            "more than once",
        ),
        ("A,A,1.0,2019-02-01,2019-03-01,v\n", HEADER, "to itself"),
        ("A,B,-0.1,2019-02-01,2019-03-01,v\n", HEADER, "negative or non-finite"),
        ("A,B,1.0,2019-2-1,2019-03-01,v\n", HEADER, "YYYY-MM-DD"),
        ("A,B,,2019-02-01,2019-03-01,v\n", HEADER, "blank cells"),
    ],
)
def test_a_malformed_file_is_refused(tmp_path, body, header, match):
    with pytest.raises(ValueError, match=match):
        load_link_scores_csv(_write(tmp_path, body, header))


def test_the_matrix_is_symmetric_scaled_and_leaves_unscored_pairs_empty():
    scores = _scores(
        [
            ("A", "B", 4.0, "2019-01-02", "2019-01-03"),
            ("C", "A", 1.0, "2019-01-02", "2019-01-03"),
            ("A", "Z", 2.0, "2019-01-02", "2019-01-03"),
        ]
    )
    matrix = link_score_matrix(scores, ["A", "B", "C"], "2020-01-02")
    assert matrix.loc["A", "B"] == matrix.loc["B", "A"] == 1.0
    assert matrix.loc["A", "C"] == matrix.loc["C", "A"] == 0.25
    assert np.isnan(matrix.loc["B", "C"])
    assert np.isnan(np.diag(matrix.values)).all()
    assert list(matrix.index) == ["A", "B", "C"]


@pytest.mark.parametrize("late", ["2020-01-02", "2020-06-30"])
def test_a_filing_on_or_after_the_valid_from_date_is_refused(late):
    scores = _scores([("A", "B", 1.0, "2019-01-02", late)])
    with pytest.raises(ValueError, match="on or after the graph's valid-from 2020-01-02"):
        link_score_matrix(scores, ["A", "B"], "2020-01-02")


def test_a_filing_the_day_before_valid_from_is_accepted():
    scores = _scores([("A", "B", 1.0, "2019-01-02", "2020-01-01")])
    assert link_score_matrix(scores, ["A", "B"], "2020-01-02").loc["A", "B"] == 1.0


def _returns_frame():
    """A and B move together, C moves against them: correlation pairs A with B."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 60)
    noise = rng.normal(0, 0.001, (3, 60))
    returns = {"A": base + noise[0], "B": base + noise[1], "C": -base + noise[2]}
    dates = pd.bdate_range("2019-09-02", periods=61).strftime("%Y-%m-%d")
    rows = []
    for code, series in returns.items():
        close = 100 * np.cumprod(np.concatenate([[1.0], 1 + series]))
        rows += [{"kdcode": code, "dt": d, "close": c} for d, c in zip(dates, close, strict=True)]
    return pd.DataFrame(rows)


def test_link_scores_not_correlation_choose_the_neighbours():
    df = _returns_frame()
    codes = ["A", "B", "C"]
    by_corr = GraphBuilder(top_k=1, use_multi_feature_edges=True)
    edge_index, _ = by_corr.build_graph(df, codes, "2020-01-02", show_progress=False)
    assert sorted(edge_index.T.tolist())[:2] == [[0, 1], [1, 0]]

    scores = _scores(
        [
            ("A", "C", 3.0, "2019-02-01", "2019-03-01"),
            ("B", "C", 1.0, "2019-02-01", "2019-03-01"),
            ("A", "B", 0.5, "2019-02-01", "2019-03-01"),
        ]
    )
    by_links = GraphBuilder(top_k=1, use_multi_feature_edges=True, link_scores=scores)
    edge_index, edge_attr = by_links.build_graph(df, codes, "2020-01-02", show_progress=False)
    assert sorted(edge_index.T.tolist()) == [[0, 2], [1, 2], [2, 0]]
    neighbour_of = {row: col for row, col in edge_index.T.tolist()}
    weight_of = dict(zip(edge_index[0].tolist(), edge_attr[:, 0].tolist(), strict=True))
    assert neighbour_of[1] == 2 and weight_of[1] == pytest.approx(1 / 3)
    assert weight_of[0] == pytest.approx(1.0)


def test_link_scores_need_top_k_and_a_static_graph():
    with pytest.raises(ValueError, match="requires graph.top_k > 0"):
        GraphConfig(link_scores_csv="links.csv", top_k=0)
    with pytest.raises(ValueError, match="requires a static graph"):
        GraphConfig(link_scores_csv="links.csv", top_k=8, update_frequency_months=6)
    assert GraphConfig(link_scores_csv="links.csv", top_k=8).link_scores_csv == "links.csv"


def test_the_pipeline_builds_the_static_graph_from_the_configured_scores(tmp_path):
    df = _returns_frame()
    frames = PipelineFrames(raw=df, normalized=df, filtered=df)
    path = _write(
        tmp_path,
        "A,C,3.0,2019-02-01,2019-03-01,v\nB,C,1.0,2019-02-01,2019-03-01,v\n"
        "A,B,0.5,2019-02-01,2019-03-01,v\n",
    )
    config = GraphConfig(top_k=1, link_scores_csv=path)
    art = build_correlation_graph(frames, ["A", "B", "C"], config, "2020-01-02", "2020-06-30")
    assert sorted(art.edge_index.T.tolist()) == [[0, 2], [1, 2], [2, 0]]
    assert art.edge_weight.shape == (3, 4)
