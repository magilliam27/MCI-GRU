"""E10 (#305): what Jev reads for an 8-K event, the baseline, the scores and the run.

Nothing here calls Jev: a fake client answers, and records every call so the
tests can check what was asked and that cached answers are never asked again.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.data.score_8k_events as script
from mci_gru.filings.change import lm_words
from mci_gru.filings.events import (
    RATING_LABELS,
    SIZE_LABELS,
    Event,
    EventDataError,
    bad_score,
    baseline_badness,
    event_state,
    item_lean,
    load_events,
    rating_score,
    revenue_words,
    tone_shares,
)
from mci_gru.filings.jev import load_question_set

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "filings"
MODEL = "jev-1.13.0"
RATING_PROBABILITIES = {"very_bad": 0.1, "bad": 0.2, "neutral": 0.3, "good": 0.3, "very_good": 0.1}


class FakeClient:
    """Answers the rating with fixed probabilities and the size with 'no_amount_stated'."""

    def __init__(self, model: str = MODEL) -> None:
        self.calls: list[dict] = []
        self.model = model

    def system_one(self, *, state, questions):
        self.calls.append({"state": state, "questions": questions})
        choices = {}
        for name, criteria in questions.items():
            if name == "rating":
                probabilities = dict(RATING_PROBABILITIES)
            else:
                probabilities = {k: (0.7 if k == "no_amount_stated" else 0.1) for k in criteria}
            pick = max(probabilities, key=probabilities.get)
            choices[name] = SimpleNamespace(
                choice=pick, confidence=probabilities[pick], probabilities=probabilities
            )
        return SimpleNamespace(
            model=self.model,
            request_id=f"req-{len(self.calls)}",
            usage=SimpleNamespace(input_tokens=60, output_tokens=0),
            choices=choices,
        )


def _event(event_id: str = "e1", **changes) -> dict:
    record = {
        "event_id": event_id,
        "company": "aaa",
        "company_name": "Aaa Corp",
        "items": ["5.02", "9.01"],
        "text": "The chief executive officer resigned effective immediately.",
        "prior_revenue_usd": 211_915_000_000,
        "prior_revenue_fiscal_year": 2023,
        "prior_90d_items": [["2.02", "9.01"], ["8.01"]],
    }
    record.update(changes)
    return record


def _events_file(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _lm_dictionary(tmp_path: Path) -> Path:
    """A tiny master dictionary; ``gain`` left the positive list, ``loss`` joined the negative."""
    path = tmp_path / "lm.csv"
    pd.DataFrame(
        {
            "Word": ["LOSS", "IMPAIRMENT", "RESIGNED", "STRONG", "GAIN", "AGREEMENT"],
            "Negative": [2009, 2009, -2020, 0, 0, 0],
            "Positive": [0, 0, 0, 2009, -2014, 0],
        }
    ).to_csv(path, index=False)
    return path


def test_v5_questions_offer_the_cards_option_sets_each_described():
    question_set = load_question_set(CONFIGS / "questions_v5.yaml")
    assert question_set.model == MODEL
    assert RATING_LABELS == ("very_bad", "bad", "neutral", "good", "very_good")
    assert SIZE_LABELS == (
        "under_1_percent",
        "between_1_and_5_percent",
        "over_5_percent",
        "no_amount_stated",
    )
    for name, labels in (("rating", RATING_LABELS), ("size", SIZE_LABELS)):
        question = question_set.questions[name]
        assert question.kind == "choice"
        assert [option for option, _ in question.options] == list(labels)
        assert all(description for _, description in question.options)


@pytest.mark.parametrize(
    ("items", "lean"),
    [
        (["1.02"], 2),
        (["5.02", "4.02"], 2),
        (["4.01", "2.06"], 2),
        (["4.01", "9.01"], 1),
        (["5.02", "9.01"], 0),
        (["1.01", "2.01"], 0),
    ],
)
def test_item_lean_puts_negative_items_over_auditor_changes_over_the_rest(items, lean):
    assert item_lean(items) == lean


def test_baseline_badness_orders_by_lean_first_then_net_tone():
    assert baseline_badness(0, 1.0) < baseline_badness(1, -1.0) < baseline_badness(2, -1.0)
    assert baseline_badness(1, 0.02) > baseline_badness(1, -0.01)


def test_tone_shares_count_every_occurrence_on_the_current_lists(tmp_path):
    dictionary = _lm_dictionary(tmp_path)
    negative, positive = lm_words(dictionary, "Negative"), lm_words(dictionary, "Positive")
    assert negative == {"loss", "impairment"}
    assert positive == {"strong"}
    neg, pos = tone_shares("Loss, loss and a strong gain resigned.", negative, positive)
    assert neg == pytest.approx(2 / 7)
    assert pos == pytest.approx(1 / 7)
    assert tone_shares("", negative, positive) == (0.0, 0.0)


def test_rating_score_weights_minus_two_to_plus_two_and_bad_score_sums_the_bad_side():
    assert rating_score(RATING_PROBABILITIES) == pytest.approx(0.1)
    assert rating_score({"very_bad": 1.0}) == -2
    assert rating_score({"very_good": 1.0}) == 2
    assert bad_score(RATING_PROBABILITIES) == pytest.approx(0.3)


def test_event_state_holds_only_context_known_at_filing_and_the_text(tmp_path):
    (event,) = load_events(_events_file(tmp_path, [_event()]))
    state = event_state(event)
    assert state == {
        "company": "Aaa Corp",
        "item_codes": [
            "5.02 Departure of Directors or Certain Officers; Election of Directors; "
            "Appointment of Certain Officers; Compensatory Arrangements of Certain Officers",
            "9.01 Financial Statements and Exhibits",
        ],
        "prior_annual_revenue": "$211.9 billion (fiscal 2023 10-K)",
        "other_8k_item_codes_in_prior_90_days": ["2.02, 9.01", "8.01"],
        "event_text": "The chief executive officer resigned effective immediately.",
    }


def test_event_state_says_when_context_is_missing():
    event = Event("e", "aaa", "Aaa Corp", ("7.77",), "text", None, None, ())
    state = event_state(event)
    assert state["item_codes"] == ["7.77"]
    assert state["prior_annual_revenue"] == "not available"
    assert state["other_8k_item_codes_in_prior_90_days"] == ["none"]


def test_revenue_words_scale_to_billions_or_millions():
    assert revenue_words(4_250_000_000, 2015) == "$4.2 billion (fiscal 2015 10-K)"
    assert revenue_words(912_300_000, None) == "$912.3 million"


@pytest.mark.parametrize(
    "bad",
    [
        {"drop": "prior_90d_items"},
        {"items": []},
        {"text": "x" * 20_001},
        {"repeat": True},
    ],
)
def test_events_breaking_the_contract_are_refused(tmp_path, bad):
    record = _event()
    if "drop" in bad:
        del record[bad["drop"]]
    else:
        record.update({k: v for k, v in bad.items() if k != "repeat"})
    records = [record, _event()] if bad.get("repeat") else [record]
    with pytest.raises(EventDataError):
        load_events(_events_file(tmp_path, records))


def _records() -> list[dict]:
    return [
        _event("e1", items=["5.02"], text="Our chief executive resigned; a loss."),
        _event("e2", items=["2.05", "9.01"], text="Restructuring with an impairment loss."),
        _event("e3", items=["4.01"], text="We changed auditors. A strong year."),
        _event("e4", items=["1.01"], text="A strong agreement with strong terms."),
    ]


def _run(
    tmp_path: Path,
    monkeypatch,
    *extra: str,
    client=None,
    sample=("e3", "e1", "e2"),
    frozen: bool = True,
) -> int:
    dictionary = _lm_dictionary(tmp_path)
    if frozen:
        monkeypatch.setattr(script, "LM_SHA256", script._sha256(dictionary))
    argv = [
        "--events",
        str(_events_file(tmp_path, _records())),
        "--lm-dictionary",
        str(dictionary),
        "--output-dir",
        str(tmp_path / "e10"),
    ]
    if sample is not None:
        path = tmp_path / "sample.csv"
        pd.DataFrame({"event_id": list(sample), "item_group": "x"}).to_csv(path, index=False)
        argv += ["--sample", str(path)]
    return script.main([*argv, *extra], client=client, sdk_question_for=lambda c: c)


def test_e10_baselines_follow_the_sample_with_lean_first(tmp_path, monkeypatch):
    client = FakeClient()
    assert _run(tmp_path, monkeypatch, client=client) == 0
    assert client.calls == []
    table = pd.read_csv(tmp_path / "e10" / "baselines.csv")
    assert table["event_id"].tolist() == ["e3", "e1", "e2"]
    assert list(table.columns) == list(script.BASELINE_COLUMNS)
    by_id = table.set_index("event_id")
    assert by_id["lean"].to_dict() == {"e3": 1, "e1": 0, "e2": 2}
    assert by_id.loc["e1", "net_tone"] == pytest.approx(1 / 6)  # "resigned" left the list
    assert by_id.loc["e3", "net_tone"] == pytest.approx(-1 / 6)
    worst_first = table.sort_values("baseline_badness", ascending=False)["event_id"].tolist()
    assert worst_first == ["e2", "e3", "e1"]
    meta = json.loads((tmp_path / "e10" / "baselines.meta.json").read_text(encoding="utf-8"))
    assert meta["events"] == 3 and meta["lm_words"] == {"negative": 2, "positive": 1}


def test_e10_baselines_cover_every_event_before_the_sample_exists(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, sample=None) == 0
    table = pd.read_csv(tmp_path / "e10" / "baselines.csv")
    assert table["event_id"].tolist() == ["e1", "e2", "e3", "e4"]


def test_e10_refuses_any_dictionary_but_the_frozen_one(tmp_path, monkeypatch):
    assert script.LM_SHA256.startswith("147cb24a")
    assert _run(tmp_path, monkeypatch, frozen=False) == 1
    assert not (tmp_path / "e10").exists()


@pytest.mark.parametrize("sample", [None, ("e1", "e9"), ("e1", "e1")])
def test_e10_scoring_needs_the_leads_sample_and_every_id_in_it(tmp_path, monkeypatch, sample):
    client = FakeClient()
    assert _run(tmp_path, monkeypatch, "--score", client=client, sample=sample) == 1
    assert client.calls == [] and not (tmp_path / "e10").exists()


def test_e10_test_call_then_each_question_on_its_own_then_collect(tmp_path, monkeypatch):
    client = FakeClient()
    loud = FakeClient()

    def run(*extra, c=client):
        return _run(tmp_path, monkeypatch, *extra, client=c)

    assert run("--score", "--question", "rating", "--limit", "1") == 0
    assert len(client.calls) == 1
    (call,) = client.calls
    assert list(call["questions"]) == ["rating"]
    assert sorted(call["questions"]["rating"]) == sorted(RATING_LABELS)
    assert call["state"]["item_codes"][0].startswith("4.01 ")  # the sample's first event
    assert not (tmp_path / "e10" / "scores.csv").exists()
    assert run("--collect", c=loud) == 1
    assert loud.calls == [] and not (tmp_path / "e10" / "scores.csv").exists()
    assert run("--score") == 0
    assert len(client.calls) == 6
    assert all(len(c["questions"]) == 1 for c in client.calls)
    sizes = [c for c in client.calls if "size" in c["questions"]]
    assert len(sizes) == 3
    assert all(sorted(c["questions"]["size"]) == sorted(SIZE_LABELS) for c in sizes)
    assert run("--collect", c=loud) == 0
    assert loud.calls == [] and len(client.calls) == 6
    scores = pd.read_csv(tmp_path / "e10" / "scores.csv")
    assert scores["event_id"].tolist() == ["e3", "e1", "e2"]
    assert scores["rating_score"].tolist() == pytest.approx([0.1] * 3)
    assert scores["bad_score"].tolist() == pytest.approx([0.3] * 3)
    assert scores["rating_p_very_bad"].eq(0.1).all() and scores["rating_choice"].eq("neutral").all()
    assert scores["size_choice"].eq("no_amount_stated").all()
    assert {f"size_p_{label}" for label in SIZE_LABELS} <= set(scores.columns)
    meta = json.loads((tmp_path / "e10" / "scores.meta.json").read_text(encoding="utf-8"))
    assert meta["distinct_calls"] == {"rating": 3, "size": 3}
    assert meta["models_answering"] == [MODEL]
