"""E18: an 8-K rated with the company's latest earnings release, from the rule to the run.

Nothing here calls Jev or SEC: a fake client answers, and records every call so
the tests can check what was asked and that cached answers are never asked again.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.data.score_8k_with_context as script
from mci_gru.filings.change import option_order, score_states
from mci_gru.filings.context import (
    RELEASE_KEY,
    context_state,
    draw_strata,
    latest_release,
)
from mci_gru.filings.events import RATING_LABELS, event_state, load_events
from mci_gru.filings.jev import AnswerCache, cache_key, load_question_set
from scripts.data.export_sec_context_releases import check_release, company_releases

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "filings"
MODEL = "jev-1.13.0"


def _t(day: str, time: str = "21:05") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {time}", tz="UTC")


# --- Which release is the latest ---------------------------------------------


def test_the_latest_release_is_the_last_earnings_8k_accepted_before_the_event():
    eight_ks = [
        (_t("2024-02-01"), "q4", ("2.02", "9.01")),
        (_t("2024-04-01"), "deal", ("1.01",)),
        (_t("2024-05-01"), "q1", ("2.02",)),
        (_t("2024-06-03", "12:00"), "event", ("5.02",)),
        (_t("2024-06-03", "13:00"), "q1_late", ("2.02",)),
    ]
    found, reason = latest_release(eight_ks, _t("2024-06-03", "12:00"), "event")
    assert (found, reason) == ((_t("2024-05-01"), "q1"), "")


@pytest.mark.parametrize(("days", "reason"), [(0.5, ""), (120, ""), (120.01, "release_too_old")])
def test_a_release_counts_up_to_120_days_before(days, reason):
    accepted = _t("2024-09-01")
    eight_ks = [(accepted - pd.Timedelta(days=days), "q", ("2.02",))]
    assert latest_release(eight_ks, accepted, "event")[1] == reason


def test_an_event_with_no_earlier_release_has_no_context():
    eight_ks = [(_t("2024-09-02"), "after", ("2.02",)), (_t("2024-08-01"), "deal", ("1.01",))]
    assert latest_release(eight_ks, _t("2024-09-01"), "event") == (None, "no_release")


# --- Which events are drawn -----------------------------------------------------

FRAME = [(f"e{i}", "neutral" if i % 3 else "bad") for i in range(60)]


def test_a_draw_takes_only_its_stratum_and_never_depends_on_the_frames_order():
    drawn = draw_strata(FRAME, [("test", "bad", 5, 1)])
    assert len(drawn) == 5 and {dict(FRAME)[e] for e in drawn} == {"bad"}
    assert draw_strata(FRAME[::-1], [("test", "bad", 5, 1)]) == drawn


def test_later_draws_never_repeat_an_earlier_one_and_minus_one_takes_the_rest():
    drawn = draw_strata(FRAME, [("probe", "neutral", 4, 1), ("test", "neutral", -1, 2)])
    assert sum(name == "probe" for name in drawn.values()) == 4
    assert sum(name == "test" for name in drawn.values()) == 40 - 4


def test_a_repeated_event_or_an_unknown_stratum_stops_the_draw():
    with pytest.raises(ValueError, match="repeats"):
        draw_strata([("e1", "bad"), ("e1", "bad")], [("test", "bad", 1, 1)])
    with pytest.raises(ValueError, match="unknown strata"):
        draw_strata(FRAME, [("test", "awful", 1, 1)])


# --- What Jev reads -------------------------------------------------------------


def _event(event_id: str = "e1", text: str = "The CFO resigned.") -> dict:
    return {
        "event_id": event_id,
        "company": "aaa",
        "company_name": "Aaa Corp",
        "items": ["5.02"],
        "text": text,
        "prior_revenue_usd": 2.0e8,
        "prior_revenue_fiscal_year": 2023,
        "prior_90d_items": [["2.02", "9.01"]],
    }


def _loaded(tmp_path: Path, records: list[dict]) -> list:
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return load_events(path)


def test_the_release_goes_just_before_the_event_text_and_nothing_else_changes(tmp_path):
    (event,) = _loaded(tmp_path, [_event()])
    plain = event_state(event)
    state = context_state(plain, "2024-05-01", "Revenue fell 30%.")
    assert list(state) == [*list(plain)[:-1], RELEASE_KEY, "event_text"]
    assert state[RELEASE_KEY] == {"filed": "2024-05-01", "text": "Revenue fell 30%."}
    assert {k: v for k, v in state.items() if k != RELEASE_KEY} == plain
    with pytest.raises(ValueError):
        context_state(state, "2024-05-01", "again")


def test_the_context_question_is_e16s_with_only_the_instructions_changed():
    v5 = load_question_set(CONFIGS / "questions_v5.yaml").questions["rating"]
    context = load_question_set(CONFIGS / "questions_context_v1.yaml")
    assert context.model == MODEL and list(context.questions) == ["rating"]
    rating = context.questions["rating"]
    assert rating.options == v5.options
    assert rating.instructions.startswith(v5.instructions.split(", and its other")[0])
    assert "latest earnings release" in rating.instructions
    assert option_order(rating, "e1") == option_order(v5, "e1")


# --- A repeat ask: draw numbers in the cache key -------------------------------


class FakeClient:
    """Answers bad with 0.6 and neutral with 0.4, and records every call."""

    def __init__(self, model: str = MODEL) -> None:
        self.calls: list[dict] = []
        self.model = model

    def system_one(self, *, state, questions):
        self.calls.append({"state": state, "questions": questions})
        probabilities = {label: 0.0 for label in RATING_LABELS} | {"bad": 0.6, "neutral": 0.4}
        answer = SimpleNamespace(choice="bad", confidence=0.6, probabilities=probabilities)
        return SimpleNamespace(
            model=self.model,
            request_id=f"req-{len(self.calls)}",
            usage=SimpleNamespace(input_tokens=60),
            choices=dict.fromkeys(questions, answer),
        )


def test_a_draw_number_asks_again_and_leaves_the_undrawn_key_as_it_was(tmp_path):
    question = load_question_set(CONFIGS / "questions_v5.yaml").questions["rating"]
    state = {"event_text": "x"}
    cache = AnswerCache(tmp_path / "answers.jsonl")
    client = FakeClient()

    def ask(draw):
        return score_states(
            [("e1", state)],
            question,
            sdk_question_for=lambda criteria: criteria,
            client=client,
            cache=cache,
            model=MODEL,
            labels=RATING_LABELS,
            draw=draw,
        )

    ask(None)
    ask(None)
    ask(2)
    assert len(client.calls) == 2
    undrawn = {**question.spec(), "option_order": option_order(question, "e1")}
    assert cache_key(state, undrawn, MODEL) in cache
    assert cache_key(state, {**undrawn, "draw": 2}, MODEL) in cache


# --- The pull's checks ----------------------------------------------------------


def _listing(accession: str, accepted: pd.Timestamp, items: tuple[str, ...]) -> dict:
    return {
        "cik": "0000000001",
        "accession": accession,
        "form": "8-K",
        "accepted": accepted,
        "document": f"{accession}.htm",
        "items": items,
    }


def test_each_frame_event_gets_its_latest_release_or_the_reason_it_has_none():
    listings = [
        _listing("q1", _t("2024-05-01"), ("2.02", "9.01")),
        _listing("e1", _t("2024-06-03"), ("5.02",)),
        _listing("e0", _t("2023-01-03"), ("5.02",)),
    ]
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e0"],
            "company": ["aaa", "aaa"],
            "stratum": ["neutral", "bad"],
            "accepted_utc": [_t("2024-06-03").isoformat(), _t("2023-01-03").isoformat()],
        }
    )
    rows = company_releases(events, listings)
    assert rows[0]["release_id"] == "q1" and rows[0]["release_items"] == "2.02;9.01"
    assert rows[1]["no_release_reason"] == "no_release" and "release_id" not in rows[1]


def _row() -> dict:
    return {
        "accepted_utc": _t("2024-06-03", "12:00").isoformat(),
        "release_listed_accepted_utc": _t("2024-06-03", "11:00").isoformat(),
    }


@pytest.mark.parametrize(
    ("index_time", "text", "reason"),
    [
        (_t("2024-06-03", "10:00"), "x" * 600, ""),
        (_t("2024-06-03", "12:00"), "x" * 600, "release_not_before_8k_by_index_page"),
        (_t("2024-01-03", "10:00"), "x" * 600, "release_too_old_by_index_page"),
        (_t("2024-06-03", "10:00"), "x" * 499, "release_text_short"),
        (None, "x" * 600, ""),
    ],
)
def test_a_drawn_release_is_checked_again_on_its_index_page_time_and_text(index_time, text, reason):
    read = {"index_accepted": index_time, "text": text}
    assert check_release(_row(), read)[0] == reason


# --- The scorer, end to end -----------------------------------------------------


def _release(event_id: str, draw_name: str = "test", text: str = "A big loss.") -> dict:
    return {
        "event_id": event_id,
        "company": "aaa",
        "draw": draw_name,
        "stratum": "neutral",
        "accepted_utc": "2024-06-03T21:05:00+00:00",
        "release": {"accepted_utc": "2024-05-02T01:05:00+00:00", "text": text},
        "release_age_days": 32.8,
    }


def _lm_dictionary(tmp_path: Path) -> Path:
    path = tmp_path / "lm.csv"
    pd.DataFrame(
        {
            "Word": ["LOSS", "RESIGNED", "GAIN"],
            "Negative": [2009, 2009, 0],
            "Positive": [0, 0, 2009],
        }
    ).to_csv(path, index=False)
    return path


def _run(tmp_path: Path, monkeypatch, *extra: str, client=None, releases=None) -> int:
    dictionary = _lm_dictionary(tmp_path)
    monkeypatch.setattr(script, "LM_SHA256", script._sha256(dictionary))
    records = releases or [_release("e1"), _release("e2", text="A gain."), _release("p1", "probe")]
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join(json.dumps(_event(r["event_id"])) for r in records) + "\n")
    path = tmp_path / "releases.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    argv = ["--events", str(events), "--releases", str(path), "--lm-dictionary", str(dictionary)]
    argv += ["--output-dir", str(tmp_path / "jev")]
    return script.main([*argv, *extra], client=client, sdk_question_for=lambda c: c)


def test_the_baselines_are_the_tone_of_the_8k_the_release_and_both(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch) == 0
    table = pd.read_csv(tmp_path / "jev" / "baselines.csv").set_index("event_id")
    # "The CFO resigned." is one negative word in three; "A big loss." one in three.
    assert table.loc["e1", "net_tone_8k"] == pytest.approx(1 / 3)
    assert table.loc["e2", "net_tone_release"] == pytest.approx(-1 / 2)
    # Read as one, "The CFO resigned. A gain." has one negative and one positive word.
    assert table.loc["e2", "net_tone_both"] == pytest.approx(0.0)
    meta = json.loads((tmp_path / "jev" / "baselines.meta.json").read_text())
    assert set(meta["estimated_input_tokens_context_by_draw"]) == {"test", "probe"}


def test_the_context_arm_asks_once_per_event_with_the_release_then_collects(tmp_path, monkeypatch):
    client = FakeClient()
    assert (
        _run(tmp_path, monkeypatch, "--score", "--draw", "test", "--limit", "1", client=client) == 0
    )
    assert len(client.calls) == 1
    state = client.calls[0]["state"]
    assert state[RELEASE_KEY] == {"filed": "2024-05-01", "text": "A big loss."}
    assert (
        _run(tmp_path, monkeypatch, "--score", "--draw", "test", "--workers", "2", client=client)
        == 0
    )
    assert len(client.calls) == 2  # e1 came from the answers file
    assert _run(tmp_path, monkeypatch, "--collect", "--draw", "test") == 0
    scores = pd.read_csv(tmp_path / "jev" / "context_test_scores.csv")
    assert scores["event_id"].tolist() == ["e1", "e2"]
    assert scores["rating_score"].iloc[0] == pytest.approx(-0.6)


def test_the_plain_arm_asks_e16s_exact_question_and_state_as_a_new_draw(tmp_path, monkeypatch):
    client = FakeClient()
    argv = ["--score", "--arm", "plain", "--draw", "test", "--plain-sample", "1:7"]
    assert _run(tmp_path, monkeypatch, *argv, client=client) == 0
    (call,) = client.calls
    (record,) = [json.loads(x) for x in (tmp_path / "jev/answers/plain_rating.jsonl").open()]
    events = {e.event_id: e for e in _loaded(tmp_path / "jev", [_event("e1"), _event("e2")])}
    assert call["state"] == event_state(events[record["pair_id"]])
    v5 = load_question_set(CONFIGS / "questions_v5.yaml").questions["rating"]
    assert record["question"]["instructions"] == v5.instructions
    assert record["question"]["draw"] == 2


def test_collect_refuses_while_an_event_has_no_answer(tmp_path, monkeypatch):
    assert (
        _run(
            tmp_path, monkeypatch, "--score", "--draw", "test", "--limit", "1", client=FakeClient()
        )
        == 0
    )
    assert _run(tmp_path, monkeypatch, "--collect", "--draw", "test") == 1


def test_events_and_releases_must_name_the_same_8ks(tmp_path, monkeypatch):
    dictionary = _lm_dictionary(tmp_path)
    monkeypatch.setattr(script, "LM_SHA256", script._sha256(dictionary))
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps(_event("e1")) + "\n")
    releases = tmp_path / "releases.jsonl"
    releases.write_text(json.dumps(_release("e2")) + "\n")
    argv = [
        "--events",
        str(events),
        "--releases",
        str(releases),
        "--lm-dictionary",
        str(dictionary),
    ]
    assert script.main([*argv, "--output-dir", str(tmp_path / "jev")]) == 1
