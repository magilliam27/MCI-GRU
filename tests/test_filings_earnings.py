"""E17: earnings releases read against the previous one, from the pairing rule to the run.

Nothing here calls Jev or SEC: a fake client answers, and records every call so
the tests can check what was asked and that cached answers are never asked again.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.data.score_earnings_releases as script
from mci_gru.filings.earnings import (
    QUESTION_LABELS,
    SETUP_WEIGHTS,
    draw,
    previous_release,
    question_order,
    release_state,
    setup_score,
)
from mci_gru.filings.jev import ModelDriftError, load_question_set
from mci_gru.filings.labelling import (
    EARNINGS_COLUMNS,
    blind_sheet,
    earnings_markdown,
    items_markdown,
    load_label_set,
    read_blind_table,
)
from scripts.data.export_sec_member_10k import read_spells
from scripts.data.export_sec_member_earnings import check_drawn, company_candidates

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "filings"
QUESTIONS = CONFIGS / "questions_earnings_v1.yaml"
MODEL = "jev-1.13.0"
SETUP_PROBABILITIES = {
    "much_worse": 0.5,
    "worse": 0.3,
    "in_line": 0.1,
    "better": 0.1,
    "much_better": 0.0,
}


def _t(day: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} 21:05", tz="UTC")


# --- Which release is the previous one -----------------------------------------


def test_the_previous_release_is_the_latest_earlier_earnings_8k():
    eight_ks = [
        (_t("2024-02-01"), "q4", ("2.02", "9.01")),
        (_t("2024-04-01"), "deal", ("1.01",)),
        (_t("2024-05-01"), "q1", ("2.02",)),
        (_t("2024-08-01"), "q2", ("2.02", "7.01")),
        (_t("2024-09-01"), "later", ("2.02",)),
    ]
    found, reason = previous_release(eight_ks, _t("2024-08-01"), "q2")
    assert (found, reason) == ((_t("2024-05-01"), "q1"), "")


@pytest.mark.parametrize(
    ("days", "reason"),
    [
        (10, "previous_within_clear_days"),
        (14, "previous_within_clear_days"),
        (30, "previous_too_recent"),
        (44, "previous_too_recent"),
        (45, ""),
        (91, ""),
        (120, ""),
        (121, "previous_too_old"),
    ],
)
def test_a_pair_counts_only_one_quarter_apart(days, reason):
    now = _t("2024-08-01")
    eight_ks = [(now - pd.Timedelta(days=days), "before", ("2.02",)), (now, "now", ("2.02",))]
    found, why = previous_release(eight_ks, now, "now")
    assert why == reason
    assert (found is None) == bool(reason)


def test_a_company_with_no_earlier_release_has_no_pair():
    now = _t("2024-08-01")
    assert previous_release([(now, "now", ("2.02",))], now, "now") == (None, "no_previous")


# --- Which candidates are drawn -------------------------------------------------


def test_a_draw_never_depends_on_the_order_of_the_candidates():
    ids = [f"a{i:03d}" for i in range(200)]
    draws = [("reading", 10, 1700), ("test", 50, 17)]
    assert draw(ids, draws) == draw(list(reversed(ids)), draws)


def test_later_draws_never_repeat_an_earlier_one_and_minus_one_takes_the_rest():
    ids = [f"a{i:03d}" for i in range(100)]
    drawn = draw(ids, [("reading", 10, 1700), ("test", 50, 17), ("rest", -1, 0)])
    names = pd.Series(drawn).value_counts().to_dict()
    assert names == {"reading": 10, "test": 50, "rest": 40}
    assert sorted(drawn) == ids


def test_a_bigger_pool_keeps_the_events_a_smaller_one_drew():
    small = [f"a{i:03d}" for i in range(100)]
    big = small + [f"b{i:03d}" for i in range(100)]
    picked = set(draw(big, [("test", 20, 17)]))
    kept = {k for k in picked if k.startswith("a")}
    assert kept and kept <= set(draw(small, [("test", 20, 17)]))


def test_a_repeated_candidate_stops_the_draw():
    with pytest.raises(ValueError, match="repeats"):
        draw(["a", "a"], [("test", 1, 17)])


# --- What Jev reads and how it is scored ----------------------------------------


def test_the_frozen_questions_offer_the_scouted_options_each_described():
    question_set = load_question_set(QUESTIONS)
    assert question_set.model == MODEL
    for name, labels in QUESTION_LABELS.items():
        options = question_set.questions[name].options
        assert tuple(n for n, _ in options) == labels
        assert all(description for _, description in options)
    assert "CHANGE" in question_set.questions["setup"].instructions


def test_each_question_has_its_own_shuffle_the_same_on_every_run():
    question_set = load_question_set(QUESTIONS)
    setup = question_set.questions["setup"]
    orders = {question_order(setup, f"e{i}")[0] for i in range(40)}
    assert len(orders) > 1
    assert question_order(setup, "e1") == question_order(setup, "e1")
    assert sorted(question_order(setup, "e1")) == sorted(SETUP_WEIGHTS)
    twin = replace(setup, question_id="twin")
    shapes = [
        [setup.options.index(next(o for o in setup.options if o[0] == n)) for n in order]
        for order in (question_order(q, f"e{i}") for i in range(10) for q in (setup, twin))
    ]
    assert shapes[0::2] != shapes[1::2]


def test_setup_score_weights_much_worse_minus_two_to_much_better_plus_two():
    assert setup_score({"much_worse": 1.0}) == -2
    assert setup_score({"much_better": 1.0}) == 2
    assert setup_score(SETUP_PROBABILITIES) == pytest.approx(-1.0 - 0.3 + 0.1)


def test_the_state_holds_the_company_both_days_and_both_texts_only():
    state = release_state("Aaa Corp", "2024-08-01", "now text", "2024-05-01", "then text")
    assert state == {
        "company": "Aaa Corp",
        "filed": "2024-08-01",
        "current_results_release": "now text",
        "previous_results_release": {"filed": "2024-05-01", "text": "then text"},
    }


# --- The pull's candidates and checks -------------------------------------------


def _args(**changes) -> argparse.Namespace:
    values = {"accepted_from": "2014-01-01", "accepted_to": "2026-09-09"}
    values.update(changes)
    return argparse.Namespace(**values)


def _listing(accession: str, accepted: str, items: tuple[str, ...], form: str = "8-K") -> dict:
    return {
        "cik": "0000000001",
        "accession": accession,
        "form": form,
        "accepted": pd.Timestamp(accepted, tz="UTC"),
        "document": f"{accession}.htm",
        "items": items,
    }


def test_candidates_are_members_earnings_8ks_with_a_previous_release():
    company = SimpleNamespace(key="aaa", spells="AAA:2015-01-01:2016-06-30")
    listings = [
        _listing("q1", "2015-02-10 21:00", ("2.02",)),
        _listing("q2", "2015-05-10 21:00", ("2.02", "9.01")),
        _listing("q3a", "2015-11-10 21:00", ("2.02",), form="8-K/A"),
        _listing("deal", "2015-06-01 21:00", ("1.01",)),
        _listing("q3", "2015-08-10 21:00", ("2.02",)),
        _listing("late", "2016-11-10 21:00", ("2.02",)),
    ]
    rows = company_candidates(company, listings, _args())
    assert [(r["event_id"], r["previous_id"]) for r in rows] == [("q2", "q1"), ("q3", "q2")]
    assert rows[0]["items"] == "2.02;9.01"


def _read(text: str = "x" * 600, accepted: str | None = None) -> dict:
    return {"text": text, "index_accepted": pd.Timestamp(accepted, tz="UTC") if accepted else None}


def _drawn_row() -> dict:
    return {
        "listed_accepted_utc": "2015-05-11T02:00:00+00:00",
        "previous_listed_accepted_utc": "2015-02-11T02:00:00+00:00",
    }


def test_a_drawn_event_is_checked_again_on_its_index_page_times():
    spells = read_spells("AAA:2015-01-01:2015-05-10")
    row = _drawn_row()
    # Listed after midnight UTC (the 10th Eastern); the index page says the 10th too.
    assert check_drawn(row, _read(), _read(), spells, _args()) == (
        "outside_membership_or_window_by_index_page",
        "2015-05-11T02:00:00+00:00",
    )
    inside = _read(accepted="2015-05-09 21:00")
    assert check_drawn(row, inside, _read(), spells, _args()) == ("", "")
    early = _read(accepted="2015-03-20 21:00")
    assert check_drawn(row, early, _read(), spells, _args())[0] == "previous_gap_by_index_page"


def test_a_release_too_short_to_compare_is_left_out():
    spells = read_spells("AAA:2015-01-01:")
    reason, _ = check_drawn(_drawn_row(), _read(), _read("see exhibit"), spells, _args())
    assert reason == "previous_text_short"


# --- The run --------------------------------------------------------------------


class FakeClient:
    """Answers setup with fixed probabilities and the side questions with their last option."""

    def __init__(self, model: str = MODEL) -> None:
        self.calls: list[dict] = []
        self.model = model

    def system_one(self, *, state, questions):
        self.calls.append({"state": state, "questions": questions})
        choices = {}
        for name, (_, criteria) in questions.items():
            if name == "setup":
                probabilities = dict(SETUP_PROBABILITIES)
            else:
                probabilities = {k: (0.7 if k == list(criteria)[-1] else 0.1) for k in criteria}
            pick = max(probabilities, key=probabilities.get)
            choices[name] = SimpleNamespace(choice=pick, probabilities=probabilities)
        return SimpleNamespace(
            model=self.model,
            request_id=f"req-{len(self.calls)}",
            usage=SimpleNamespace(input_tokens=60),
            choices=choices,
        )


def _event(event_id: str, draw_name: str = "test", text: str = "A loss.") -> dict:
    return {
        "event_id": event_id,
        "company": "aaa",
        "company_name": "Aaa Corp",
        "draw": draw_name,
        "accepted_utc": "2024-08-01T21:05:00+00:00",
        "text": text,
        "previous": {"accepted_utc": "2024-05-02T01:05:00+00:00", "text": "A strong gain."},
    }


def _lm_dictionary(tmp_path: Path) -> Path:
    path = tmp_path / "lm.csv"
    pd.DataFrame(
        {"Word": ["LOSS", "STRONG", "GAIN"], "Negative": [2009, 0, 0], "Positive": [0, 2009, 2009]}
    ).to_csv(path, index=False)
    return path


def _run(tmp_path: Path, monkeypatch, *extra: str, client=None, events=None) -> int:
    dictionary = _lm_dictionary(tmp_path)
    monkeypatch.setattr(script, "LM_SHA256", script._sha256(dictionary))
    records = events or [_event("e1"), _event("e2", text="Strong."), _event("r1", "reading")]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    argv = ["--events", str(path), "--draw", "test", "--lm-dictionary", str(dictionary)]
    argv += ["--output-dir", str(tmp_path / "jev")]
    return script.main([*argv, *extra], client=client, sdk_question_for=lambda i, c: (i, c))


def test_the_baseline_is_the_tone_change_of_the_draws_events(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch) == 0
    table = pd.read_csv(tmp_path / "jev" / "baselines.csv")
    assert table["event_id"].tolist() == ["e1", "e2"]
    # "A loss." is all negative (+1); "A strong gain." is two-thirds positive (-2/3).
    assert table.loc[0, "tone_change"] == pytest.approx(0.5 - (-2 / 3))
    meta = json.loads((tmp_path / "jev" / "baselines.meta.json").read_text())
    assert meta["events"] == 2 and meta["estimated_input_tokens"] > 0


def test_three_questions_in_one_call_each_event_once_then_collect(tmp_path, monkeypatch):
    client = FakeClient()
    assert _run(tmp_path, monkeypatch, "--score", "--limit", "1", client=client) == 0
    assert len(client.calls) == 1
    call = client.calls[0]
    assert sorted(call["questions"]) == sorted(QUESTION_LABELS)
    assert call["state"]["filed"] == "2024-08-01"
    assert call["state"]["previous_results_release"]["filed"] == "2024-05-01"
    assert _run(tmp_path, monkeypatch, "--score", "--workers", "2", client=client) == 0
    assert len(client.calls) == 2  # e1 came from the answers file
    assert _run(tmp_path, monkeypatch, "--collect") == 0
    scores = pd.read_csv(tmp_path / "jev" / "scores.csv")
    assert scores["event_id"].tolist() == ["e1", "e2"]
    assert scores["setup_choice"].tolist() == ["much_worse", "much_worse"]
    assert scores["setup_score"].iloc[0] == pytest.approx(-1.2)
    for name in QUESTION_LABELS:
        assert scores[f"{name}_order"].str.split("|").map(sorted).map(tuple).nunique() == 1


def test_collect_refuses_while_an_event_has_no_answer(tmp_path, monkeypatch):
    client = FakeClient()
    assert _run(tmp_path, monkeypatch, "--score", "--limit", "1", client=client) == 0
    assert _run(tmp_path, monkeypatch, "--collect") == 1
    assert not (tmp_path / "jev" / "scores.csv").exists()


def test_an_answer_from_another_model_stops_the_run(tmp_path, monkeypatch):
    with pytest.raises(ModelDriftError):
        _run(tmp_path, monkeypatch, "--score", client=FakeClient(model="jev-1.14.0"))


def test_a_text_over_the_limit_is_refused(tmp_path, monkeypatch):
    events = [_event("e1", text="x" * 20_001)]
    assert _run(tmp_path, monkeypatch, events=events) == 1


# --- The reading check's sheet ---------------------------------------------------


def test_the_reading_check_asks_jevs_setup_question_word_for_word():
    label_set = load_label_set(CONFIGS / "labels" / "earnings_setup_v1.yaml")
    setup = load_question_set(QUESTIONS).questions["setup"]
    (question,) = label_set.questions
    assert label_set.items == "earnings_pair"
    assert label_set.item_columns == EARNINGS_COLUMNS
    assert question.text == setup.instructions
    assert tuple(question.options) == setup.options
    assert question.on_split == "middle"
    assert label_set.category({"setup": "much_worse"}) == "worse"
    assert label_set.category({"setup": "in_line"}) == "in_line"


def test_an_earnings_sheet_shows_the_previous_release_then_the_current(tmp_path):
    table = pd.DataFrame(
        {
            "label_id": ["a", "b"],
            "company": ["Aaa Corp", "Bbb Inc"],
            "current_filed": ["2024-08-01", "2024-08-02"],
            "current_release": ["Q2 revenue fell.", "Q2 revenue rose."],
            "previous_filed": ["2024-05-01", "2024-05-02"],
            "previous_release": ["Q1 revenue rose.", "Q1 revenue fell."],
            "setup_choice": ["worse", "better"],
        }
    )
    path = tmp_path / "pairs.csv"
    table.to_csv(path, index=False)
    read = read_blind_table(path, "label_id", EARNINGS_COLUMNS)
    items, _ = blind_sheet(read, "label_id", sample=None, seed=3, columns=EARNINGS_COLUMNS)
    text = items_markdown(items, "earnings_pair")
    assert text == earnings_markdown(items)
    first = items.loc[items["company"] == "Aaa Corp"].iloc[0]
    block = text.split(f"## Item {first.item_id}")[1].split("## Item")[0]
    assert block.index("filed 2024-05-01") < block.index("Q1 revenue rose.")
    assert block.index("Q1 revenue rose.") < block.index("filed 2024-08-01")
    assert block.index("filed 2024-08-01") < block.index("Q2 revenue fell.")
    assert "worse" not in text and "better" not in text
