"""Milestone 2 of #289: how each risk factor changed, the word baseline and the grade.

Also E4 (#295): changed or not, from the same answers and word shares.

Jev itself is never called: a fake client answers each Choice in the SDK's
response shape (``choices[name].choice``, ``.confidence``, ``.probabilities``).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.data.score_risk_change_flags as flags_script
import scripts.data.score_risk_changes as script
from mci_gru.filings.change import (
    CHANGE_LABELS,
    JEV_SAME_CUT,
    WORD_SCORE_CUT,
    ChangeGradingError,
    balanced_accuracy,
    baseline_change,
    change_flags,
    change_label_sheet,
    grade_changes,
    option_order,
    pair_table,
    read_change_labels,
    score_choices,
    word_change,
)
from mci_gru.filings.jev import (
    AnswerCache,
    ModelDriftError,
    Question,
    QuestionSetError,
    load_question_set,
)
from mci_gru.filings.pairing import RiskPair
from mci_gru.filings.risk_factors import RiskFactor

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "filings"
MODEL = "jev-1.13.0"


class FakeChoiceClient:
    """Picks a fixed option for every Choice and records what it was asked."""

    def __init__(self, pick: str = "same", model: str = MODEL) -> None:
        self.calls: list[dict] = []
        self.pick = pick
        self.model = model

    def system_one(self, *, state, questions):
        self.calls.append({"state": state, "questions": questions})
        return SimpleNamespace(
            model=self.model,
            request_id=f"req-{len(self.calls)}",
            usage=SimpleNamespace(input_tokens=50, output_tokens=0),
            choices={
                name: SimpleNamespace(
                    choice=self.pick,
                    confidence=0.7,
                    probabilities={k: (0.7 if k == self.pick else 0.075) for k in criteria},
                )
                for name, criteria in questions.items()
            },
        )


@pytest.fixture
def question() -> Question:
    return load_question_set(CONFIGS / "questions_v2.yaml").questions["change_type"]


def _risk(year: int, index: int, heading: str, body: str) -> RiskFactor:
    return RiskFactor(f"acc-{year}", year, index, None, heading, body)


def _pairs(n: int = 6) -> list[RiskPair]:
    return [
        RiskPair(
            _risk(2025, i, f"Risk {i}", f"current words {i}"),
            _risk(2024, i, f"Risk {i}", f"prior words {i}"),
            1,
            0.9,
        )
        for i in range(1, n + 1)
    ]


# --- the frozen question -------------------------------------------------


def test_v2_question_offers_exactly_the_five_change_kinds_each_described(question):
    assert question.kind == "choice"
    assert [name for name, _ in question.options] == list(CHANGE_LABELS)
    assert all(description for _, description in question.options)
    assert question.spec()["options"] == dict(question.options)


def test_a_noul_spec_is_unchanged_so_milestone_one_cache_keys_hold():
    v1 = load_question_set(CONFIGS / "questions_v1.yaml").questions["same_risk"]
    assert set(v1.spec()) == {"question_id", "kind", "instructions"}


@pytest.mark.parametrize(
    "options",
    [{"same": "Unchanged."}, {"same": "Unchanged.", "new": ""}, None],
)
def test_a_choice_needs_two_or_more_described_options(tmp_path, options):
    entry = {"kind": "choice", "instructions": "Pick one."}
    if options is not None:
        entry["options"] = options
    path = tmp_path / "q.yaml"
    path.write_text(json.dumps({"version": 9, "model": MODEL, "questions": {"q": entry}}))
    with pytest.raises(QuestionSetError, match="two options"):
        load_question_set(path)


# --- the word baseline ---------------------------------------------------


def test_word_change_counts_distinct_words_added_and_removed():
    added, removed = word_change("alpha beta gamma delta", "alpha beta epsilon")
    assert added == pytest.approx(2 / 4)
    assert removed == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("similarity", "added", "removed", "expected"),
    [
        (0.24, 0.0, 0.0, "new"),
        (0.25, 0.0, 0.0, "same"),
        (0.9, 0.20, 0.0, "broadened"),
        (0.9, 0.19, 0.0, "same"),
        (0.9, 0.0, 0.20, "narrowed"),
        (0.9, 0.20, 0.20, "reworked"),
        (0.9, 0.19, 0.19, "same"),
    ],
)
def test_baseline_rule_and_its_frozen_thresholds(similarity, added, removed, expected):
    assert baseline_change(similarity, added, removed) == expected


def test_pair_table_carries_both_years_and_the_baseline():
    pair = RiskPair(
        _risk(2025, 1, "Risk", "same old words plus brand new scope"),
        _risk(2024, 1, "Risk", "same old words"),
        1,
        0.8,
    )
    row = pair_table([pair]).iloc[0]
    assert (row["fiscal_year"], row["prior_fiscal_year"]) == (2025, 2024)
    assert row["baseline"] == "broadened"


# --- option order and scoring --------------------------------------------


def test_option_order_is_a_fixed_shuffle_per_pair(question):
    orders = {tuple(option_order(question, f"p{i}")) for i in range(20)}
    assert option_order(question, "p1") == option_order(question, "p1")
    assert all(sorted(order) == sorted(CHANGE_LABELS) for order in orders)
    assert len(orders) > 1


def test_jev_sees_the_options_in_the_pairs_shuffled_order(tmp_path, question):
    client = FakeChoiceClient()
    pairs = _pairs()
    score_choices(
        pairs,
        question,
        sdk_question_for=lambda criteria: criteria,
        client=client,
        cache=AnswerCache(tmp_path / "a.jsonl"),
        model=MODEL,
    )
    for pair, call in zip(pairs, client.calls, strict=True):
        criteria = call["questions"]["change_type"]
        assert list(criteria) == option_order(question, pair.pair_id)
        assert criteria == {name: dict(question.options)[name] for name in criteria}


def test_cached_choices_are_never_asked_again(tmp_path, question):
    client = FakeChoiceClient(pick="broadened")
    kwargs = dict(
        sdk_question_for=lambda criteria: criteria,
        client=client,
        cache=AnswerCache(tmp_path / "a.jsonl"),
        model=MODEL,
    )
    first = score_choices(_pairs(), question, **kwargs)
    second = score_choices(_pairs(), question, **kwargs)
    assert len(client.calls) == 6
    assert all(item.cached for item in second)
    assert [item.choice for item in second] == ["broadened"] * 6
    assert second[0].probabilities == first[0].probabilities


def test_an_answer_from_another_model_stops_the_choice_run(tmp_path, question):
    with pytest.raises(ModelDriftError):
        score_choices(
            _pairs(1),
            question,
            sdk_question_for=lambda criteria: criteria,
            client=FakeChoiceClient(model="jev-1.14.0"),
            cache=AnswerCache(tmp_path / "a.jsonl"),
            model=MODEL,
        )


def test_a_choice_without_the_five_change_kinds_is_refused(tmp_path, question):
    partial = Question("change_type", "choice", "Pick.", question.options[:4])
    with pytest.raises(QuestionSetError, match="exactly"):
        score_choices(
            _pairs(1),
            partial,
            sdk_question_for=lambda criteria: criteria,
            client=FakeChoiceClient(),
            cache=AnswerCache(tmp_path / "a.jsonl"),
            model=MODEL,
        )


# --- the label sample ----------------------------------------------------


def _table(counts: dict[str, int]) -> pd.DataFrame:
    rows, i = [], 0
    for kind, count in counts.items():
        for _ in range(count):
            i += 1
            rows.append(
                {
                    "pair_id": f"p{i}",
                    "fiscal_year": 2025,
                    "prior_fiscal_year": 2024,
                    "current_heading": f"h{i}",
                    "current_body": f"b{i}",
                    "prior_heading": f"ph{i}",
                    "prior_body": f"pb{i}",
                    "baseline": kind,
                }
            )
    return pd.DataFrame(rows)


def test_label_sample_oversamples_rare_kinds_and_weights_undo_it():
    counts = {"same": 60, "broadened": 10, "narrowed": 4, "reworked": 3, "new": 2}
    sheet, design = change_label_sheet(_table(counts), n=20, seed=1)
    assert len(sheet) == len(design) == 20
    taken = design["stratum"].value_counts().to_dict()
    assert taken["new"] == 2 and taken["reworked"] == 3
    weights = design.groupby("stratum")["weight"].sum().to_dict()
    assert weights == pytest.approx({kind: float(n) for kind, n in counts.items()})


def test_label_sheet_shows_years_and_texts_but_no_answer_hints():
    sheet, design = change_label_sheet(_table({"same": 5, "new": 5}), n=6, seed=1)
    assert list(sheet.columns) == [
        "label_id",
        "current_year",
        "current_heading",
        "current_body",
        "prior_year",
        "prior_heading",
        "prior_body",
        "change",
    ]
    assert (sheet["change"] == "").all()
    assert set(sheet["label_id"]) == set(design["label_id"])


def test_labels_outside_the_five_kinds_stop_the_grade():
    sheet = pd.DataFrame({"label_id": ["C001", "C002"], "change": ["Same ", "changed"]})
    with pytest.raises(ChangeGradingError, match=r"^1 labels .*\['C002'\]"):
        read_change_labels(sheet)


# --- the grade -----------------------------------------------------------


def test_calling_everything_same_earns_one_over_the_kinds_present():
    truth = np.array(["same", "same", "same", "new", "broadened"])
    guess = np.array(["same"] * 5)
    assert balanced_accuracy(truth, guess, np.ones(5)) == pytest.approx(1 / 3)


def test_balanced_accuracy_uses_the_design_weights():
    truth = np.array(["same", "same", "new"])
    guess = np.array(["same", "new", "new"])
    weight = np.array([3.0, 1.0, 1.0])
    assert balanced_accuracy(truth, guess, weight) == pytest.approx((0.75 + 1.0) / 2)


def _graded(jev: list[str], baseline: list[str], label: list[str]) -> pd.DataFrame:
    n = len(label)
    return pd.DataFrame(
        {
            "label": label,
            "jev": jev,
            "baseline": baseline,
            "weight": np.ones(n),
            "stratum": baseline,
        }
    )


def test_jev_wins_only_when_its_whole_interval_is_above_zero():
    label = ["same", "broadened", "narrowed", "new"] * 10
    perfect = _graded(label, ["same"] * 40, label)
    summary = grade_changes(perfect, seed=1)
    assert summary["balanced_accuracy"]["jev"] == pytest.approx(1.0)
    assert summary["jev_minus_baseline"]["jev_wins"] is True
    tied = grade_changes(_graded(label, label, label), seed=1)
    assert tied["jev_minus_baseline"]["interval_95"] == [0.0, 0.0]
    assert tied["jev_minus_baseline"]["jev_wins"] is False


def test_a_gap_whose_interval_spans_zero_is_not_a_win():
    label = ["same", "new"] * 20
    jev = ["same", "same"] * 10 + ["same", "new"] * 10
    baseline = ["same", "new"] * 10 + ["same", "same"] * 10
    gap = grade_changes(_graded(jev, baseline, label), seed=1)["jev_minus_baseline"]
    low, high = gap["interval_95"]
    assert low < 0 < high
    assert gap["jev_wins"] is False


def test_per_kind_recall_and_precision():
    summary = grade_changes(
        _graded(["new", "new", "same"], ["same"] * 3, ["new", "same", "same"]), seed=1
    )
    jev_new = summary["per_kind"]["jev"]["new"]
    assert (jev_new["recall"], jev_new["precision"]) == (1.0, 0.5)
    assert summary["per_kind"]["jev"]["reworked"] == {
        "labelled": 0,
        "recall": None,
        "precision": None,
    }


# --- the script, end to end ----------------------------------------------


def _risk_file(tmp_path: Path) -> Path:
    shared = "users advertisers revenue growth competition search mobile cloud devices"
    risks = []
    for year in (2023, 2024, 2025):
        for i in range(6):
            body = f"{shared} topic{i} detail{i} more{i}"
            if year == 2025 and i == 0:
                body += " artificial intelligence generative models regulation scrutiny"
            risks.append(_risk(year, i + 1, f"Risk heading {i}", body).to_dict())
    path = tmp_path / "risks.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in risks) + "\n", encoding="utf-8")
    return path


def _run(tmp_path, *extra, client=None):
    argv = ["--risk-factors", str(_risk_file(tmp_path)), "--output-dir", str(tmp_path / "out")]
    return script.main(
        [*argv, "--label-sample", "6", *extra],
        client=client,
        sdk_question_for=lambda criteria: criteria,
    )


def test_step_one_writes_the_sheet_and_guide_and_seals_the_rest(tmp_path):
    client = FakeChoiceClient()
    assert _run(tmp_path, client=client) == 0
    out = tmp_path / "out"
    assert client.calls == []
    assert {p.name for p in out.iterdir()} == {"change_sheet.csv", "label_guide.md", "sealed"}
    assert "broadened" in (out / "label_guide.md").read_text(encoding="utf-8")
    pairs = pd.read_csv(out / "sealed" / "pairs.csv")
    assert len(pairs) == 12 and "baseline" in pairs


def test_score_then_grade_end_to_end(tmp_path):
    client = FakeChoiceClient(pick="same")
    assert _run(tmp_path, "--score", client=client) == 0
    out = tmp_path / "out"
    assert len(client.calls) == 12
    sheet = pd.read_csv(out / "change_sheet.csv", dtype=str)
    assert _run(tmp_path, "--grade") == 1
    sheet["change"] = "same"
    sheet.to_csv(out / "change_sheet.csv", index=False)
    assert _run(tmp_path, "--grade") == 0
    summary = json.loads((out / "grading" / "summary.json").read_text(encoding="utf-8"))
    assert summary["labelled_risks"] == 6
    assert summary["balanced_accuracy"]["jev"] == pytest.approx(1.0)


def test_an_existing_change_sheet_is_kept(tmp_path):
    assert _run(tmp_path) == 0
    sheet = tmp_path / "out" / "change_sheet.csv"
    labelled = pd.read_csv(sheet, dtype=str).assign(change="new")
    labelled.to_csv(sheet, index=False)
    assert _run(tmp_path) == 0
    assert (pd.read_csv(sheet, dtype=str)["change"] == "new").all()


# --- E4: changed or not (#295) -------------------------------------------


def test_e4_cuts_are_the_values_frozen_before_its_labels():
    assert (JEV_SAME_CUT, WORD_SCORE_CUT) == (0.05, 0.04)


def _flag_inputs(rows: list[tuple[float, float, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"p{i}" for i in range(len(rows))]
    table = pd.DataFrame(
        {
            "pair_id": ids,
            "words_added": [added for added, _, _ in rows],
            "words_removed": [removed for _, removed, _ in rows],
        }
    )
    return table, pd.DataFrame({"pair_id": ids, "p_same": [p for _, _, p in rows]})


def test_jev_calls_changed_only_below_its_cut():
    table, choices = _flag_inputs([(0, 0, 0.049), (0, 0, 0.05), (0, 0, 0.5)])
    assert change_flags(table, choices)["jev_changed"].tolist() == [True, False, False]


def test_word_score_is_the_larger_share_and_changed_from_its_cut_up():
    table, choices = _flag_inputs(
        [(0.04, 0.0, 1), (0.0, 0.04, 1), (0.039, 0.039, 1), (0.1, 0.3, 1)]
    )
    flags = change_flags(table, choices)
    assert flags["word_score"].tolist() == [0.04, 0.04, 0.039, 0.3]
    assert flags["word_changed"].tolist() == [True, True, False, True]


def test_flags_keep_every_pair_in_order_with_the_agreed_columns():
    table, choices = _flag_inputs([(0, 0, 0.9), (0.5, 0, 0.0)])
    flags = change_flags(table, choices.iloc[::-1])
    assert list(flags.columns) == ["pair_id", "p_same", "word_score", "jev_changed", "word_changed"]
    assert flags["pair_id"].tolist() == ["p0", "p1"]
    assert flags["p_same"].tolist() == [0.9, 0.0]


def test_a_pair_without_a_jev_answer_stops_the_flags():
    table, choices = _flag_inputs([(0, 0, 0.9), (0, 0, 0.1)])
    with pytest.raises(ChangeGradingError, match="1 pairs have no Jev answer"):
        change_flags(table, choices.iloc[:1])


def _flag_run(tmp_path: Path, answers: Path) -> int:
    return flags_script.main(
        [
            "--risk-factors",
            str(tmp_path / "risks.jsonl"),
            "--answers",
            str(answers),
            "--output-dir",
            str(tmp_path / "e4"),
        ]
    )


def test_e4_scores_come_from_the_e2_answers_without_calling_jev(tmp_path):
    assert _run(tmp_path, "--score", client=FakeChoiceClient(pick="broadened")) == 0
    answers = tmp_path / "out" / "sealed" / "answers.jsonl"
    before = answers.read_bytes()
    assert _flag_run(tmp_path, answers) == 0
    assert answers.read_bytes() == before
    scores = pd.read_csv(tmp_path / "e4" / "scores.csv")
    pairs = pd.read_csv(tmp_path / "out" / "sealed" / "pairs.csv")
    assert scores["pair_id"].tolist() == pairs["pair_id"].tolist()
    assert scores["p_same"].eq(0.075).all() and not scores["jev_changed"].any()
    word = pairs[["words_added", "words_removed"]].max(axis=1)
    assert scores["word_score"].tolist() == pytest.approx(word.tolist())
    assert scores["word_changed"].tolist() == (word >= WORD_SCORE_CUT).tolist()
    meta = json.loads((tmp_path / "e4" / "scores.meta.json").read_text(encoding="utf-8"))
    assert meta["pairs"] == 12 and meta["model"] == MODEL


def test_a_pair_missing_from_the_answers_stops_e4_with_no_output(tmp_path):
    assert _run(tmp_path, "--score", "--limit", "11", client=FakeChoiceClient()) == 0
    assert _flag_run(tmp_path, tmp_path / "out" / "sealed" / "answers.jsonl") == 1
    assert not (tmp_path / "e4" / "scores.csv").exists()


def test_an_unfinished_answers_file_is_refused_and_left_alone(tmp_path):
    assert _run(tmp_path, "--score", client=FakeChoiceClient()) == 0
    answers = tmp_path / "out" / "sealed" / "answers.jsonl"
    torn = answers.read_bytes() + b'{"key": "unfinished'
    answers.write_bytes(torn)
    assert _flag_run(tmp_path, answers) == 1
    assert answers.read_bytes() == torn
