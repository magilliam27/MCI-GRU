"""Run summaries count ties among graph-linked names (#324).

The first run gave graph-linked names identical scores (GOOG/GOOGL, MA/V,
HD/LOW) and nothing reported it (#319). ``evaluation_summary.json`` now carries
a ``linked_score_ties`` block: per test date and in total, how many
correlation-linked pairs score within 1e-6, for the ensemble average and for
each member, with the score precision stated.
"""

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir

import run_experiment
from mci_gru.evaluation.experiment_summary import (
    LINKED_SCORE_TIE_TOLERANCE,
    correlation_edges_for_date,
    count_linked_score_ties,
    linked_pairs,
    linked_score_tie_summary,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATES = ["2025-01-02", "2025-01-03", "2025-01-06"]

# Stocks 0-1, 1-2 and 3-4 are linked; 5 is linked to nothing. Each edge is listed
# in both directions, one twice, and there are self-loops, as a GAT edge list has.
EDGES = np.array(
    [
        [0, 1, 1, 2, 3, 4, 3, 0, 5],
        [1, 0, 2, 1, 4, 3, 4, 0, 5],
    ]
)


def _scores() -> np.ndarray:
    return np.array(
        [
            # 0-1 exact tie, 1-2 apart, 3-4 within tolerance; 5 ties 0 but is unlinked.
            [0.10, 0.10, 0.30, 0.50, 0.50 + 4e-7, 0.10],
            # 0-1 just outside tolerance, 1-2 tie, 3-4 apart.
            [0.20, 0.20 + 3e-6, 0.20 + 3e-6, 0.10, 0.90, 0.20],
            # Stock 1 is masked (non-tradable): 0-1 and 1-2 are not scored; 3-4 tie.
            [0.40, np.nan, 0.40, -0.70, -0.70, 0.40],
        ],
        dtype=np.float32,
    )


def test_linked_pairs_drop_direction_duplicates_and_self_loops():
    assert linked_pairs(EDGES).tolist() == [[0, 1], [1, 2], [3, 4]]
    assert linked_pairs(torch.as_tensor(EDGES)).tolist() == [[0, 1], [1, 2], [3, 4]]


def test_an_empty_or_missing_graph_links_nothing():
    assert linked_pairs(np.empty((2, 0), dtype=np.int64)).shape == (0, 2)
    assert linked_pairs(np.array([[2, 3], [2, 3]])).shape == (0, 2)
    assert linked_pairs(None).shape == (0, 2)


def test_an_edge_list_laid_out_as_rows_is_refused():
    with pytest.raises(ValueError, match=r"shape \(2, E\)"):
        linked_pairs(EDGES.T)


def test_ties_are_counted_only_between_linked_finite_scores_within_tolerance():
    pairs = [linked_pairs(EDGES)] * len(DATES)

    evaluated, tied = count_linked_score_ties(_scores(), pairs)

    assert evaluated == [3, 3, 1]
    assert tied == [2, 1, 1]


def test_the_summary_reports_ensemble_and_each_member_per_date_and_in_total():
    avg = _scores()
    member_a = avg.copy()
    member_a[0, 1] = 0.15  # breaks the day-one 0-1 tie for this member only
    member_b = avg.copy()
    member_b[1] = [0.0, 1.0, 2.0, 3.0, 3.0, 4.0]  # day two: only 3-4 ties

    summary = linked_score_tie_summary(
        avg, [member_a, member_b], DATES, lambda date: EDGES, eval_autocast=False
    )

    assert summary == {
        "graph": "correlation",
        "tolerance": LINKED_SCORE_TIE_TOLERANCE,
        "scores_dtype": "float32",
        "eval_autocast": False,
        "linked_pairs": {"per_date": dict(zip(DATES, [3, 3, 1], strict=True)), "total": 7},
        "ensemble": {"per_date": dict(zip(DATES, [2, 1, 1], strict=True)), "total": 4},
        "members": [
            {"model_id": 0, "per_date": dict(zip(DATES, [1, 1, 1], strict=True)), "total": 3},
            {"model_id": 1, "per_date": dict(zip(DATES, [2, 1, 1], strict=True)), "total": 4},
        ],
    }
    assert LINKED_SCORE_TIE_TOLERANCE == 1e-6
    json.dumps(summary)


def test_each_date_is_counted_against_its_own_graph_snapshot():
    scores = np.zeros((2, 4), dtype=np.float32)  # every name ties every other
    snapshots = {
        DATES[0]: np.array([[0, 1], [1, 0]]),
        DATES[1]: np.array([[0, 1, 2], [1, 2, 3]]),
    }

    summary = linked_score_tie_summary(
        scores, [], DATES[:2], snapshots.__getitem__, eval_autocast=True
    )

    assert summary["ensemble"]["per_date"] == {DATES[0]: 1, DATES[1]: 3}
    assert summary["members"] == []
    assert summary["eval_autocast"] is True


class _Schedule:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def get_graph_for_date(self, date):
        return self.snapshots[date], torch.ones(self.snapshots[date].shape[1])


def test_a_dynamic_run_counts_against_the_schedule_snapshot_not_the_initial_graph():
    initial = torch.tensor([[0], [1]])
    later = torch.tensor([[2], [3]])
    schedule = _Schedule({DATES[0]: initial, DATES[1]: later})

    edges_for_date = correlation_edges_for_date({"graph_schedule": schedule, "edge_index": initial})

    assert edges_for_date(DATES[1]) is later


def test_a_static_run_counts_against_its_one_graph():
    static = torch.tensor([[0], [1]])

    edges_for_date = correlation_edges_for_date({"graph_schedule": None, "edge_index": static})

    assert all(edges_for_date(d) is static for d in DATES)


# --- Runner wiring --------------------------------------------------------------

RUNNER_OVERRIDES = [
    "features=base",
    "data.source=csv",
    "data.use_pit_universe=false",
    "data.train_start=2020-01-01",
    "data.train_end=2020-02-14",
    "data.val_start=2020-02-18",
    "data.val_end=2020-02-28",
    "data.test_start=2020-03-03",
    "data.test_end=2020-03-20",
    "model.his_t=3",
    "model.label_t=2",
    "model.gru_hidden_sizes=[8,4]",
    "model.hidden_size_gat1=8",
    "model.output_gat1=4",
    "model.gat_heads=1",
    "model.hidden_size_gat2=8",
    "model.num_hidden_states=4",
    "model.cross_attn_heads=1",
    "model.use_self_attention=false",
    "training.num_epochs=1",
    "training.num_models=2",
    "training.batch_size=2",
    "training.warmup_steps=0",
    "tracking.enabled=false",
]


def _write_panel(path: Path, days: int = 105) -> None:
    """Four stocks; ``B.N`` is an exact copy of ``A.N``, so the two always score alike.

    The panel starts before ``train_start`` so the static graph has returns to
    correlate; the copied pair is then the only one above the threshold.
    """
    dates = pd.bdate_range("2019-11-01", periods=days)
    rows = []
    for s_idx, stock in enumerate(["A.N", "C.N", "D.N"]):
        price = 100.0 + s_idx
        for d_idx, dt in enumerate(dates):
            open_px = price * (1.0 + 0.002 * np.sin(d_idx / 4.0 + s_idx))
            close_px = open_px * (1.0 + 0.001 * (s_idx + 1) * np.cos(d_idx / 7.0))
            row = {
                "dt": dt.strftime("%Y-%m-%d"),
                "open": open_px,
                "high": max(open_px, close_px) * 1.001,
                "low": min(open_px, close_px) * 0.999,
                "close": close_px,
                "volume": 1_000_000.0 + d_idx,
            }
            rows.append({"kdcode": stock, **row})
            if stock == "A.N":
                rows.append({"kdcode": "B.N", **row})
            price = close_px
    pd.DataFrame(rows).to_csv(path, index=False)


@contextmanager
def _restored_process_state(directory: Path):
    root = logging.getLogger()
    handlers, level, cwd = list(root.handlers), root.level, os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(cwd)
        for handler in root.handlers:
            if handler not in handlers:
                handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)


@pytest.fixture(scope="module")
def runner_ties(tmp_path_factory):
    """Run the runner once on CPU, recording what it handed the tie counter."""
    base = tmp_path_factory.mktemp("linked_ties_run")
    panel = base / "panel.csv"
    _write_panel(panel)
    out = base / "out"
    out.mkdir()
    calls = []

    def spy(avg, members, dates, edges_for_date, **kwargs):
        calls.append(
            {
                "avg": np.array(avg),
                "members": [np.array(m) for m in members],
                "dates": list(dates),
                "edges": {d: np.asarray(edges_for_date(d)) for d in dates},
                "kwargs": kwargs,
            }
        )
        return linked_score_tie_summary(avg, members, dates, edges_for_date, **kwargs)

    with initialize_config_dir(config_dir=str(REPOSITORY_ROOT / "configs"), version_base=None):
        cfg = compose(
            "config",
            overrides=[
                *RUNNER_OVERRIDES,
                "graph.corr_lookback_days=20",
                # Asked for, but a CPU run cannot autocast: the summary must say so.
                "training.eval_amp=true",
                f"data.filename={panel.as_posix()}",
            ],
        )
    patch = pytest.MonkeyPatch()
    patch.setattr(run_experiment, "linked_score_tie_summary", spy)
    patch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        with _restored_process_state(out):
            run_experiment.main.__wrapped__(cfg)
    finally:
        patch.undo()
    (call,) = calls
    summary = json.loads((out / "evaluation_summary.json").read_text())
    graph = torch.load(out / "graph_data.pt", weights_only=False)
    test_files = sorted(p.stem for p in (out / "averaged_predictions").glob("*.csv"))
    return call, summary, graph, test_files


def test_the_runner_counts_each_members_own_scores_on_the_graph_it_used(runner_ties):
    call, _, graph, _ = runner_ties

    assert len(call["members"]) == 2
    assert not np.array_equal(call["members"][0], call["members"][1])
    np.testing.assert_allclose(np.mean(call["members"], axis=0), call["avg"], rtol=0, atol=1e-7)
    for edges in call["edges"].values():
        np.testing.assert_array_equal(edges, graph["edge_index"].numpy())
    assert call["kwargs"] == {"eval_autocast": False}
    assert call["avg"].dtype == np.float32


def test_the_runner_writes_the_counts_into_the_evaluation_summary(runner_ties):
    call, summary, graph, test_files = runner_ties
    ties = summary["linked_score_ties"]
    graph_pairs = linked_pairs(graph["edge_index"]).tolist()

    assert list(ties["ensemble"]["per_date"]) == call["dates"] == test_files
    assert [m["model_id"] for m in ties["members"]] == [0, 1]
    assert ties["scores_dtype"] == "float32"
    assert "metrics" in summary
    # The only edge is A.N-B.N, and B.N copies A.N: one linked pair, tied on every date.
    assert graph_pairs == [[0, 1]]
    every_date = {date: 1 for date in call["dates"]}
    assert ties["linked_pairs"] == {"per_date": every_date, "total": len(every_date)}
    for scorer in [ties["ensemble"], *ties["members"]]:
        assert scorer["per_date"] == every_date
        assert scorer["total"] == len(every_date)
