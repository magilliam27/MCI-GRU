"""The Labelling desk (#291): label sets, blind sheets, AI reviewers, agreement.

No subagent and no labelling page runs here: reviewer answers and page label
documents are written by hand in the shapes those produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.data.label_desk as desk
from mci_gru.filings.labelling import (
    EVENT_COLUMNS,
    SHEET_COLUMNS,
    LabelError,
    LabelSetError,
    agreement,
    blind_sheet,
    cohen_kappa,
    combine_answers,
    combine_reviewers,
    complete_answer_sets,
    events_markdown,
    load_label_set,
    page_documents,
    page_label_rows,
    read_answer_file,
    read_blind_table,
    render_reviewer_prompt,
    reviewer_batches,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = REPO_ROOT / "configs" / "filings" / "labels"
CHANGE = LABELS / "change_v1.yaml"
SAME_RISK = LABELS / "same_risk_v1.yaml"
PROMPT = LABELS / "reviewer_v1.md"
PROMPT_V2 = LABELS / "reviewer_v2.md"
DIRECTION = LABELS / "change_direction_v1.yaml"
EVENTS = LABELS / "event_rating_v1.yaml"
EVENT_PROMPT = LABELS / "reviewer_events_v1.md"


def _events_table(n=6):
    return pd.DataFrame(
        {
            "label_id": [f"K{i:03d}" for i in range(1, n + 1)],
            "company": [f"Company {i}" for i in range(1, n + 1)],
            "item_codes": ["5.02 Departure of Directors"] * n,
            "prior_annual_revenue": ["$10.0 billion (fiscal 2015 10-K)"] * n,
            "other_8k_item_codes_in_prior_90_days": ["2.02, 9.01"] + [""] * (n - 1),
            "event_text": [f"The chief financial officer resigned, event {i}." for i in range(n)],
            "jev_rating": ["bad"] * n,
            "item_group": ["officer"] * n,
        }
    )


CHOICE_SET = """id: choice_test
questions:
  - id: kind
    text: Which kind?
    options:
      - {id: worse, text: it got worse}
      - {id: better, text: it got better}
      - {id: neutral, text: no change}
  - id: sure
    text: Are you sure?
    ask_if: {kind: worse}
categories:
  - {label: bad, when: {kind: worse}}
  - {label: good, when: {kind: better}}
  - {label: flat, when: {}}
"""
PRACTICE = Path(__file__).parent / "fixtures" / "filings" / "label_practice.csv"


@pytest.fixture(scope="module")
def change():
    return load_label_set(CHANGE)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- Label sets ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("answers", "category"),
    [
        ({"same_risk": "no"}, "new"),
        ({"same_risk": "yes", "added": "yes", "dropped": "yes"}, "reworked"),
        ({"same_risk": "yes", "added": "yes", "dropped": "no"}, "broadened"),
        ({"same_risk": "yes", "added": "no", "dropped": "yes"}, "narrowed"),
        ({"same_risk": "yes", "added": "no", "dropped": "no"}, "same"),
    ],
)
def test_change_answers_derive_the_five_change_options(change, answers, category):
    assert change.category(answers) == category


def test_change_categories_are_the_options_of_the_jev_change_question(change):
    import yaml

    jev = yaml.safe_load((REPO_ROOT / "configs" / "filings" / "questions_v2.yaml").read_text())
    options = set(jev["questions"]["change_type"]["options"])
    assert set(change.categories) == options


def test_every_label_set_in_the_repo_loads():
    for path in sorted(LABELS.glob("*.yaml")):
        label_set = load_label_set(path)
        assert {label_set.category(a) for a in complete_answer_sets(label_set)} == set(
            label_set.categories
        )
    assert load_label_set(SAME_RISK).category({"same_risk": "no"}) == "different"


def test_a_skipped_question_must_stay_unanswered(change):
    with pytest.raises(LabelError, match="answered but not asked"):
        change.category({"same_risk": "no", "added": "yes"})
    with pytest.raises(LabelError, match="unanswered"):
        change.category({"same_risk": "yes", "added": "yes"})
    with pytest.raises(LabelError, match="yes or no"):
        change.category({"same_risk": "maybe"})


def test_an_unquoted_yes_in_yaml_is_refused(tmp_path):
    text = CHANGE.read_text(encoding="utf-8").replace('{same_risk: "no"}', "{same_risk: no}")
    with pytest.raises(LabelSetError, match="quote it in YAML"):
        load_label_set(_write(tmp_path, "bad.yaml", text))


def test_a_category_no_answers_reach_is_refused(tmp_path):
    text = CHANGE.read_text(encoding="utf-8").replace(
        "- {label: same, when: {}}",
        '- {label: odd, when: {same_risk: "yes", added: "yes"}}\n  - {label: same, when: {}}',
    )
    with pytest.raises(LabelSetError, match="no answers lead to"):
        load_label_set(_write(tmp_path, "bad.yaml", text))


def test_the_last_rule_must_match_everything(tmp_path):
    text = CHANGE.read_text(encoding="utf-8").replace(
        "- {label: same, when: {}}", '- {label: same, when: {added: "no"}}'
    )
    with pytest.raises(LabelSetError, match="last category rule"):
        load_label_set(_write(tmp_path, "bad.yaml", text))


def test_a_choice_question_takes_exactly_one_of_its_own_options(tmp_path):
    label_set = load_label_set(_write(tmp_path, "choice.yaml", CHOICE_SET))
    assert label_set.category({"kind": "better"}) == "good"
    assert label_set.category({"kind": "Worse", "sure": "yes"}) == "bad"
    with pytest.raises(LabelError, match="one of each question's options"):
        label_set.check({"kind": "yes"})
    with pytest.raises(LabelError, match="one of each question's options"):
        label_set.check({"kind": "worse", "sure": "maybe"})
    with pytest.raises(LabelError, match="answered but not asked"):
        label_set.check({"kind": "neutral", "sure": "yes"})
    assert len(complete_answer_sets(label_set)) == 4


def test_a_category_reached_only_through_a_choice_option_loads(tmp_path):
    # "good" needs kind=better; enumerating yes/no alone would call it unreachable.
    label_set = load_label_set(_write(tmp_path, "choice.yaml", CHOICE_SET))
    assert label_set.categories == ("bad", "good", "flat")


def test_a_condition_naming_an_option_a_question_lacks_is_refused(tmp_path):
    # A follow-up asked after an option its question lacks would never be asked.
    never = CHOICE_SET.replace("ask_if: {kind: worse}", "ask_if: {kind: 'yes'}")
    with pytest.raises(LabelSetError, match="must be the string of one of"):
        load_label_set(_write(tmp_path, "x.yaml", never))
    later = CHOICE_SET.replace("ask_if: {kind: worse}", "ask_if: {sure: 'yes'}")
    with pytest.raises(LabelSetError, match="earlier question"):
        load_label_set(_write(tmp_path, "y.yaml", later))


def test_the_change_direction_set_keeps_the_e8_card_options():
    label_set = load_label_set(DIRECTION)
    direction, finer = label_set.questions
    assert direction.answers == ("worse", "better", "neutral")
    assert finer.answers == ("materialized", "new_exposure", "escalated", "eased", "neutral")
    assert all(text for _, text in direction.options + finer.options)
    assert label_set.category({"direction": "better", "finer": "eased"}) == "better"


def test_the_e9_set_asks_the_e5_and_e8_questions_word_for_word():
    combined = load_label_set(LABELS / "new_changed_direction_v1.yaml")
    e5 = load_label_set(LABELS / "new_changed_v1.yaml")
    e8 = load_label_set(DIRECTION)
    assert combined.questions == e5.questions + e8.questions
    assert combined.categories == e5.categories
    answers = {"new_risk": "no", "changed": "yes", "direction": "worse", "finer": "escalated"}
    assert combined.category(answers) == "changed"


def test_the_e10_set_rates_8k_events_on_a_scale():
    label_set = load_label_set(EVENTS)
    rating, size = label_set.questions
    assert label_set.items == "event_8k"
    assert label_set.item_columns == EVENT_COLUMNS
    assert rating.answers == ("very_bad", "bad", "neutral", "good", "very_good")
    assert rating.on_split == "middle" and size.on_split == ""
    assert size.answers[-1] == "no_amount_stated"
    assert all(text for _, text in rating.options + size.options)
    assert label_set.category({"rating": "very_bad", "size": "over_5_percent"}) == "bad"


def test_a_split_rule_needs_a_scale_and_a_known_name(tmp_path):
    with pytest.raises(LabelSetError, match="scale of three"):
        load_label_set(
            _write(
                tmp_path,
                "a.yaml",
                CHOICE_SET.replace(
                    "    ask_if: {kind: worse}", "    ask_if: {kind: worse}\n    on_split: middle"
                ),
            )
        )
    with pytest.raises(LabelSetError, match="on_split must be"):
        load_label_set(
            _write(
                tmp_path,
                "b.yaml",
                CHOICE_SET.replace(
                    "    text: Which kind?", "    text: Which kind?\n    on_split: mean"
                ),
            )
        )
    with pytest.raises(LabelSetError, match="items must be one of"):
        load_label_set(_write(tmp_path, "c.yaml", "items: tweets\n" + CHOICE_SET))


def test_an_event_sheet_reads_only_the_context_and_text(tmp_path):
    path = tmp_path / "events.csv"
    _events_table().to_csv(path, index=False)
    read = read_blind_table(path, "label_id", EVENT_COLUMNS)
    assert sorted(read.columns) == sorted(["label_id", *EVENT_COLUMNS])
    items, _ = blind_sheet(read, "label_id", sample=None, seed=3, columns=EVENT_COLUMNS)
    assert list(items.columns) == ["item_id", *EVENT_COLUMNS]
    text = events_markdown(items.sort_values("item_id"))
    first = items.loc[items["company"] == "Company 1"].iloc[0]
    block = text.split(f"## Item {first.item_id}")[1].split("## Item")[0]
    assert "- Company: Company 1" in block
    assert "- Item codes: 5.02 Departure of Directors" in block
    assert "- Prior annual revenue: $10.0 billion (fiscal 2015 10-K)" in block
    assert "in the previous 90 days: 2.02, 9.01" in block
    assert "event 0." in block
    assert "jev" not in text.lower() and "officer\n" not in text


# --- Blind sheets -------------------------------------------------------------


def test_a_sheet_never_reads_an_answer_column(tmp_path):
    table = pd.read_csv(PRACTICE)
    table["jev_choice"] = "broadened"
    table["baseline"] = "same"
    table["stratum"] = "new"
    path = tmp_path / "handover.csv"
    table.to_csv(path, index=False)
    read = read_blind_table(path, "pair_id")
    assert sorted(read.columns) == sorted(["pair_id", *SHEET_COLUMNS])


def test_a_hand_over_missing_a_text_column_is_refused(tmp_path):
    path = tmp_path / "handover.csv"
    pd.read_csv(PRACTICE).drop(columns="prior_body").to_csv(path, index=False)
    with pytest.raises(LabelError, match="prior_body"):
        read_blind_table(path, "pair_id")


def test_sheet_ids_are_fresh_and_the_order_is_shuffled_by_seed():
    table = pd.DataFrame(
        {
            "src": [f"S{i:02d}" for i in range(30)],
            **{c: [f"{c}{i}" for i in range(30)] for c in SHEET_COLUMNS},
        }
    )
    items, source = blind_sheet(table, "src", sample=None, seed=7)
    again, _ = blind_sheet(table, "src", sample=None, seed=7)
    other, _ = blind_sheet(table, "src", sample=None, seed=8)
    assert items["item_id"].tolist() == [f"R{i:03d}" for i in range(1, 31)]
    assert source["source_id"].tolist() != table["src"].tolist()
    assert items.equals(again)
    assert not items.equals(other)
    joined = items.merge(source, on="item_id").merge(table, left_on="source_id", right_on="src")
    assert (joined["prior_body_x"] == joined["prior_body_y"]).all()
    sampled, _ = blind_sheet(table, "src", sample=5, seed=7)
    assert len(sampled) == 5


def test_the_page_gets_questions_but_never_the_category_rules(change):
    items, _ = blind_sheet(pd.read_csv(PRACTICE, dtype=str), "pair_id", sample=None, seed=0)
    docs = page_documents("demo", change, items, {"title": "Demo"})
    collection, doc_id, sheet = docs[0]
    assert (collection, doc_id) == ("sheets", "demo")
    assert [q["id"] for q in sheet["questions"]] == ["same_risk", "added", "dropped"]
    assert sheet["questions"][1]["ask_if"] == {"same_risk": "yes"}
    payload = json.dumps(sheet)
    assert "categories" not in sheet and "broadened" not in payload
    assert [d[1] for d in docs[1:]] == items["item_id"].tolist()
    assert all(d[0] == "sheets/demo/items" for d in docs[1:])


# --- AI reviewers -------------------------------------------------------------


def test_each_reviewer_sees_every_item_once_in_its_own_order():
    items = pd.DataFrame({"item_id": [f"R{i:03d}" for i in range(1, 91)]})
    batches = reviewer_batches(items, reviewers=3, batch_size=40, seed=0)
    orders = []
    for parts in batches.values():
        assert [len(p) for p in parts] == [40, 40, 10]
        seen = pd.concat(parts)["item_id"].tolist()
        assert sorted(seen) == items["item_id"].tolist()
        orders.append(seen)
    assert len({tuple(o) for o in orders}) == 3


def test_the_reviewer_prompt_states_every_question_and_the_header(change, tmp_path):
    prompt = render_reviewer_prompt(
        PROMPT.read_text(encoding="utf-8"),
        change,
        items_path=tmp_path / "items.md",
        answers_path=tmp_path / "answers.csv",
        item_count=40,
        example_id="R001",
    )
    for q in change.questions:
        assert f"`{q.question_id}`: {q.text}" in prompt
    assert "item_id,same_risk,added,dropped,note" in prompt
    assert "$" not in prompt
    assert "Answer only when `same_risk` is yes" in prompt


def test_the_reviewer_prompt_lists_a_choice_question_and_its_options(tmp_path):
    label_set = load_label_set(_write(tmp_path, "choice.yaml", CHOICE_SET))
    prompt = render_reviewer_prompt(
        PROMPT_V2.read_text(encoding="utf-8"),
        label_set,
        items_path=tmp_path / "items.md",
        answers_path=tmp_path / "answers.csv",
        item_count=3,
        example_id="R001",
    )
    assert "1. `kind`: Which kind?\n   Pick exactly one of these" in prompt
    for option, text in label_set.questions[0].options:
        assert f"   - `{option}`: {text}" in prompt
    assert "2. `sure`: Are you sure? Answer only when `kind` is worse" in prompt
    assert "unless the question lists its own options" in prompt
    assert "Pick exactly one" not in prompt.split("2. `sure`")[1]


def test_reviewer_packets_name_their_files_by_the_path_given(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    out = tmp_path / "link" / "labels"
    args = ["sheet", "--table", str(PRACTICE), "--id-column", "pair_id", "--label-set", str(CHANGE)]
    assert desk.main([*args, "--sheet-id", "practice", "--out-dir", str(out)]) == 0
    assert desk.main(["reviewers", "--labels-dir", str(out)]) == 0
    prompt = (out / "reviewers" / "reviewer_1" / "batch_1" / "prompt.md").read_text()
    assert f"`{(out / 'reviewers' / 'reviewer_1' / 'batch_1' / 'items.md').as_posix()}`" in prompt
    assert str(real) not in prompt


def test_a_reviewer_answering_a_skipped_question_is_rejected(change, tmp_path):
    path = _write(
        tmp_path,
        "answers.csv",
        "item_id,same_risk,added,dropped,note\nR001,no,yes,,different risk\n",
    )
    with pytest.raises(LabelError, match="R001: answered but not asked"):
        read_answer_file(change, path)
    dupes = _write(
        tmp_path,
        "dupes.csv",
        "item_id,same_risk,added,dropped,note\nR001,no,,,a\nR001,no,,,b\n",
    )
    with pytest.raises(LabelError, match="answered twice"):
        read_answer_file(change, dupes)


def test_combining_takes_the_majority_and_leaves_a_split_empty():
    def review(cats):
        return pd.DataFrame({"item_id": ["R1", "R2", "R3"], "category": cats})

    table = combine_reviewers(
        ["R1", "R2", "R3"],
        {
            "reviewer_1": review(["same", "new", "same"]),
            "reviewer_2": review(["same", "new", "broadened"]),
            "reviewer_3": review(["same", "same", "narrowed"]),
        },
    )
    assert table["ai_label"].tolist() == ["same", "new", ""]
    assert table["votes"].tolist() == [3, 2, 1]
    assert table["unanimous"].tolist() == [True, False, False]


def test_each_question_takes_its_own_majority(tmp_path):
    label_set = load_label_set(_write(tmp_path, "choice.yaml", CHOICE_SET))

    def review(kinds, sures):
        return pd.DataFrame({"item_id": ["R1", "R2"], "kind": kinds, "sure": sures})

    table = combine_answers(
        label_set,
        ["R1", "R2"],
        {
            "reviewer_1": review(["worse", "better"], ["yes", ""]),
            "reviewer_2": review(["worse", "neutral"], ["no", ""]),
            "reviewer_3": review(["worse", "better"], ["no", ""]),
        },
    )
    assert table["kind"].tolist() == ["worse", "better"]
    assert table["kind_votes"].tolist() == [3, 2]
    assert table["kind_unanimous"].tolist() == [True, False]
    assert table["sure"].tolist() == ["no", ""]
    assert table["reviewer_2_kind"].tolist() == ["worse", "neutral"]


def test_a_three_way_rating_split_takes_the_middle_and_is_marked():
    label_set = load_label_set(EVENTS)
    ids = ["E1", "E2", "E3", "E4"]

    def review(ratings, sizes):
        return pd.DataFrame({"item_id": ids, "rating": ratings, "size": sizes})

    table = combine_answers(
        label_set,
        ids,
        {
            "reviewer_1": review(
                ["very_bad", "very_good", "bad", "bad"],
                ["under_1_percent", "no_amount_stated", "over_5_percent", "over_5_percent"],
            ),
            "reviewer_2": review(
                ["good", "bad", "bad", "neutral"],
                ["over_5_percent", "no_amount_stated", "over_5_percent", "under_1_percent"],
            ),
            "reviewer_3": review(
                ["neutral", "very_bad", "bad", "good"],
                ["no_amount_stated", "no_amount_stated", "over_5_percent", "no_amount_stated"],
            ),
        },
    )
    assert table["rating"].tolist() == ["neutral", "bad", "bad", "neutral"]
    assert table["rating_by_rule"].tolist() == [True, True, False, True]
    assert table["rating_votes"].tolist() == [1, 1, 3, 1]
    assert table["size"].tolist() == ["", "no_amount_stated", "over_5_percent", ""]
    assert "size_by_rule" not in table


def test_combining_refuses_a_reviewer_who_skipped_an_item():
    full = pd.DataFrame({"item_id": ["R1", "R2"], "category": ["same", "new"]})
    with pytest.raises(LabelError, match="reviewer_2: missing items"):
        combine_reviewers(["R1", "R2"], {"reviewer_1": full, "reviewer_2": full.iloc[:1]})


# --- The owner's labels and agreement -----------------------------------------


def test_page_labels_derive_a_category_only_when_finished(change):
    rows = page_label_rows(
        change,
        [
            {"item_id": "R2", "answers": {"same_risk": "yes", "added": "yes"}, "flagged": True},
            {"item_id": "R1", "answers": {"same_risk": "yes", "added": "no", "dropped": "yes"}},
        ],
    )
    assert rows["item_id"].tolist() == ["R1", "R2"]
    assert rows["category"].tolist() == ["narrowed", ""]
    assert rows.loc[1, "unfinished"].startswith("unanswered")
    assert rows["flagged"].tolist() == [False, True]


def test_agreement_counts_only_items_both_labelled(change):
    owner = {"R1": "same", "R2": "new", "R3": "broadened", "R4": ""}
    ai = {"R1": "same", "R2": "same", "R3": "broadened", "R4": "same", "R5": "new"}
    result = agreement(change, owner, ai)
    assert result["items_compared"] == 3
    assert result["agreed"] == 2
    assert result["disagreements"] == [{"item_id": "R2", "owner": "new", "ai": "same"}]
    assert result["confusion_owner_rows_ai_columns"]["new"]["same"] == 1
    low, high = result["agreement_interval_95"]
    assert low < 2 / 3 < high


def test_cohen_kappa_matches_a_worked_example():
    # 50 items: both yes 20, both no 15, a-yes/b-no 5, a-no/b-yes 10.
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    # observed 0.70; expected 0.5*0.6 + 0.5*0.4 = 0.50; kappa 0.40.
    assert cohen_kappa(a, b) == pytest.approx(0.40)


# --- The script, end to end ---------------------------------------------------


def test_the_desk_runs_an_event_sheet_end_to_end(tmp_path):
    table = tmp_path / "events.csv"
    _events_table(3).to_csv(table, index=False)
    out = tmp_path / "labels"
    args = ["sheet", "--table", str(table), "--id-column", "label_id", "--label-set", str(EVENTS)]
    assert desk.main([*args, "--sheet-id", "e10", "--seed", "1", "--out-dir", str(out)]) == 0
    assert not (out / "page_seed").exists()
    assert desk.main(["reviewers", "--labels-dir", str(out), "--prompt", str(EVENT_PROMPT)]) == 0
    packet = out / "reviewers" / "reviewer_1" / "batch_1"
    assert "- Prior annual revenue:" in (packet / "items.md").read_text()
    prompt = (packet / "prompt.md").read_text()
    assert "8-K" in prompt and "   - `very_good`:" in prompt
    ratings = {"reviewer_1": "very_bad", "reviewer_2": "neutral", "reviewer_3": "good"}
    for name, rating in ratings.items():
        lines = ["item_id,rating,size,note"]
        lines += [f"R00{i},{rating},no_amount_stated,n" for i in range(1, 4)]
        (out / "reviewers" / name / "batch_1" / "answers.csv").write_text("\n".join(lines) + "\n")
    assert desk.main(["combine", "--labels-dir", str(out)]) == 0
    answers = pd.read_csv(out / "ai_answers.csv", keep_default_na=False)
    assert answers["rating"].tolist() == ["neutral"] * 3
    summary = json.loads((out / "ai_summary.json").read_text())
    assert summary["answers"]["rating"]["by_rule"] == ["R001", "R002", "R003"]


def test_the_desk_runs_end_to_end_on_the_practice_sheet(tmp_path, capsys):
    out = tmp_path / "labels"
    args = ["sheet", "--table", str(PRACTICE), "--id-column", "pair_id", "--label-set", str(CHANGE)]
    assert desk.main([*args, "--sheet-id", "practice", "--practice", "--out-dir", str(out)]) == 0
    assert desk.main([*args, "--sheet-id", "practice", "--out-dir", str(out)]) == 1
    assert "never rebuilt" in capsys.readouterr().err
    batches = json.loads((out / "page_seed" / "batches.json").read_text())
    assert [e["doc_id"] for e in batches[0]] == ["practice", "R001", "R002", "R003"]

    assert desk.main(["reviewers", "--labels-dir", str(out)]) == 0
    answers = {
        "reviewer_1": ["no,,", "yes,yes,no", "yes,no,yes"],
        "reviewer_2": ["no,,", "yes,yes,no", "yes,no,no"],
        "reviewer_3": ["no,,", "yes,no,no", "yes,no,no"],
    }
    for name, rows in answers.items():
        lines = ["item_id,same_risk,added,dropped,note"]
        lines += [f"R00{i},{row},note" for i, row in enumerate(rows, start=1)]
        (out / "reviewers" / name / "batch_1" / "answers.csv").write_text("\n".join(lines) + "\n")
    assert desk.main(["combine", "--labels-dir", str(out)]) == 0
    ai = pd.read_csv(out / "ai_labels.csv", keep_default_na=False)
    assert ai["ai_label"].tolist() == ["new", "broadened", "same"]
    assert ai["unanimous"].tolist() == [True, False, False]
    by_question = pd.read_csv(out / "ai_answers.csv", keep_default_na=False)
    assert by_question["same_risk"].tolist() == ["no", "yes", "yes"]
    assert by_question["added"].tolist() == ["", "yes", "no"]

    dump = tmp_path / "dump" / "sheets" / "practice" / "labels"
    dump.mkdir(parents=True)
    for item_id, given in {"R001": {"same_risk": "no"}, "R002": {"same_risk": "yes"}}.items():
        (dump / f"{item_id}.json").write_text(json.dumps({"item_id": item_id, "answers": given}))
    assert desk.main(["collect", "--labels-dir", str(out), "--dump", str(tmp_path / "dump")]) == 0
    assert desk.main(["agree", "--labels-dir", str(out)]) == 0
    result = json.loads((out / "agreement.json").read_text())
    assert (result["items_compared"], result["agreed"]) == (1, 1)


def test_a_changed_label_set_stops_the_later_steps(tmp_path):
    label_set = tmp_path / "change_copy.yaml"
    label_set.write_text(CHANGE.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "labels"
    desk.main(
        [
            "sheet",
            "--table",
            str(PRACTICE),
            "--id-column",
            "pair_id",
            "--label-set",
            str(label_set),
            "--sheet-id",
            "x",
            "--out-dir",
            str(out),
        ]
    )
    label_set.write_text(
        label_set.read_text(encoding="utf-8").replace("Updated figures", "New figures"),
        encoding="utf-8",
    )
    assert desk.main(["reviewers", "--labels-dir", str(out)]) == 1
