"""The frozen question set, the answer cache and the blind label sheet (#289).

Jev itself is never called here: a fake client stands in for the TypeSafe SDK
and answers in the SDK's response shape (``nouls[name].noul``, ``model``,
``request_id``, ``usage.input_tokens``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mci_gru.filings.jev import (
    AnswerCache,
    ModelDriftError,
    QuestionSetError,
    blind_label_sheet,
    cache_key,
    load_question_set,
    pair_state,
    score_pairs,
)
from mci_gru.filings.pairing import RiskPair
from mci_gru.filings.risk_factors import RiskFactor

REPO_QUESTIONS = Path(__file__).resolve().parents[1] / "configs" / "filings" / "questions_v1.yaml"
MODEL = "jev-1.13.0"


class FakeClient:
    """Answers every Noul with a fixed probability, in the SDK's response shape."""

    def __init__(self, probability: float = 0.8, model: str = MODEL) -> None:
        self.calls: list[dict] = []
        self.probability = probability
        self.model = model

    def system_one(self, *, state, questions):
        self.calls.append({"state": state, "questions": questions})
        return SimpleNamespace(
            model=self.model,
            request_id=f"req-{len(self.calls)}",
            usage=SimpleNamespace(input_tokens=123, output_tokens=0),
            nouls={name: SimpleNamespace(noul=self.probability) for name in questions},
        )


def _risk(year: int, index: int, heading: str) -> RiskFactor:
    return RiskFactor(f"acc-{year}", year, index, None, heading, f"Body of {heading}")


def _pairs(years=(2024, 2025), per_year: int = 4, k: int = 3) -> list[RiskPair]:
    pairs = []
    for year in years:
        for i in range(1, per_year + 1):
            for rank in range(1, k + 1):
                pairs.append(
                    RiskPair(
                        _risk(year, i, f"Risk {year}-{i}"),
                        _risk(year - 1, rank, f"Risk {year - 1}-{rank}"),
                        rank,
                        1.0 / rank,
                    )
                )
    return pairs


@pytest.fixture
def question_set():
    return load_question_set(REPO_QUESTIONS)


def test_repository_question_set_pins_a_versioned_model(question_set):
    assert question_set.model == MODEL
    assert question_set.version == 1
    assert question_set.questions["same_risk"].kind == "noul"


@pytest.mark.parametrize("alias", ["jev", "jev-latest", "jev-preview"])
def test_moving_model_aliases_are_refused(tmp_path, alias):
    path = tmp_path / "q.yaml"
    path.write_text(
        f"version: 1\nmodel: {alias}\nquestions:\n  same_risk:\n    kind: noul\n"
        "    instructions: Same?\n",
        encoding="utf-8",
    )
    with pytest.raises(QuestionSetError):
        load_question_set(path)


def test_cached_pairs_are_never_sent_again(tmp_path, question_set):
    question = question_set.questions["same_risk"]
    pairs = _pairs(years=(2025,), per_year=2, k=2)
    client = FakeClient()
    first = score_pairs(
        pairs,
        question,
        sdk_question="Q",
        client=client,
        cache=AnswerCache(tmp_path / "c.jsonl"),
        model=MODEL,
    )
    assert len(client.calls) == len(pairs)

    again = FakeClient(probability=0.1)
    second = score_pairs(
        pairs,
        question,
        sdk_question="Q",
        client=again,
        cache=AnswerCache(tmp_path / "c.jsonl"),
        model=MODEL,
    )
    assert again.calls == []
    assert [s.probability for s in second] == [s.probability for s in first] == [0.8] * len(pairs)
    assert all(s.cached for s in second) and not any(s.cached for s in first)


def test_a_different_model_is_a_different_cache_entry(question_set):
    question = question_set.questions["same_risk"]
    state = pair_state(_pairs(years=(2025,), per_year=1, k=1)[0])
    assert cache_key(state, question.spec(), MODEL) != cache_key(
        state, question.spec(), "jev-1.14.0"
    )


def test_a_different_question_wording_is_a_different_cache_entry(question_set):
    question = question_set.questions["same_risk"]
    state = pair_state(_pairs(years=(2025,), per_year=1, k=1)[0])
    reworded = dict(question.spec(), instructions="Is it the same?")
    assert cache_key(state, question.spec(), MODEL) != cache_key(state, reworded, MODEL)


def test_an_answer_from_another_model_version_is_refused_and_not_cached(tmp_path, question_set):
    question = question_set.questions["same_risk"]
    cache = AnswerCache(tmp_path / "c.jsonl")
    with pytest.raises(ModelDriftError):
        score_pairs(
            _pairs(years=(2025,), per_year=1, k=1),
            question,
            sdk_question="Q",
            client=FakeClient(model="jev-1.14.0"),
            cache=cache,
            model=MODEL,
        )
    assert len(AnswerCache(tmp_path / "c.jsonl")) == 0


def test_each_answer_records_model_request_and_tokens(tmp_path, question_set):
    question = question_set.questions["same_risk"]
    scored = score_pairs(
        _pairs(years=(2025,), per_year=1, k=1),
        question,
        sdk_question="Q",
        client=FakeClient(),
        cache=AnswerCache(tmp_path / "c.jsonl"),
        model=MODEL,
    )
    assert (scored[0].model, scored[0].request_id, scored[0].input_tokens) == (MODEL, "req-1", 123)


def test_state_holds_both_risks_and_none_of_the_shortlist_score():
    pair = _pairs(years=(2025,), per_year=1, k=1)[0]
    state = pair_state(pair)
    assert state["current_risk_factor"]["heading"] == "Risk 2025-1"
    assert state["prior_year_risk_factor"]["heading"] == "Risk 2024-1"
    assert "similarity" not in str(state) and "rank" not in str(state)


def test_label_sheet_shows_no_scores_ranks_or_ids():
    sheet, _ = blind_label_sheet(_pairs(), n=12, seed=7)
    assert list(sheet.columns) == [
        "label_id",
        "current_heading",
        "current_body",
        "prior_heading",
        "prior_body",
        "same_risk",
    ]
    assert sheet["same_risk"].eq("").all()
    assert not sheet["label_id"].str.contains("acc-").any()


def test_label_sheet_columns_are_fixed_whatever_the_pair_carries():
    sheet, _ = blind_label_sheet(_pairs(), n=12, seed=7)
    assert set(sheet.columns).isdisjoint({"similarity", "rank", "pair_id", "fiscal_year"})


def test_label_rows_are_shuffled_across_strata():
    # Rows in stratum order would tell the labeller which pairs ranked top.
    _, design = blind_label_sheet(_pairs(), n=12, seed=7)
    assert design["stratum"].tolist() != sorted(design["stratum"])


def test_label_sample_covers_every_year_and_both_rank_strata_with_weights():
    pairs = _pairs()
    sheet, design = blind_label_sheet(pairs, n=12, seed=7)
    assert len(sheet) == 12
    assert set(design["stratum"]) == {"2024|top", "2024|rest", "2025|top", "2025|rest"}
    # Weights scale each stratum's sample back to the stratum's size.
    for stratum, rows in design.groupby("stratum"):
        size = sum(
            1
            for pair in pairs
            if f"{pair.current.fiscal_year}|" in stratum
            and (pair.rank == 1) == stratum.endswith("|top")
        )
        assert rows["weight"].sum() == pytest.approx(size)


def test_label_sample_is_reproducible_for_a_seed():
    pairs = _pairs()
    first, _ = blind_label_sheet(pairs, n=12, seed=7)
    second, _ = blind_label_sheet(pairs, n=12, seed=7)
    assert first.equals(second)
