"""E7 (#310): each member 10-K's change signals against its pair partner, and the run.

Nothing here calls Jev: a fake client answers, and records every call so the
tests can check that only flagged risks are asked and cached answers never again.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import scripts.data.score_e7_signals as script
from mci_gru.filings.report_signals import (
    SignalError,
    risk_pair_table,
    risk_shares,
    section_change,
    worse_share,
)
from mci_gru.filings.risk_factors import RiskFactor

if TYPE_CHECKING:
    from pathlib import Path

MODEL = "jev-1.13.0"
P_WORSE = 0.8

COMPETITION = ("Competition", "We face intense competition from rivals in every market we serve.")
CYBER = ("Cyber attacks", "Hackers may breach our systems and steal customer data.")
CYBER_GREW = (
    "Cyber attacks",
    "Hackers may breach our systems and steal customer data. A breach last year "
    "disrupted billing for weeks.",
)
PANDEMIC = ("Pandemic", "A pandemic could close our stores and cut demand sharply.")


class FakeClient:
    """Answers every direction question with p(worse) = 0.8."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def system_one(self, *, state, questions):
        self.calls.append({"state": state, "questions": questions})
        probabilities = {"worse": P_WORSE, "better": 0.05, "neutral": 0.15}
        return SimpleNamespace(
            model=MODEL,
            request_id=f"req-{len(self.calls)}",
            usage=SimpleNamespace(input_tokens=50, output_tokens=0),
            choices={
                name: SimpleNamespace(
                    choice="worse", confidence=P_WORSE, probabilities=probabilities
                )
                for name in questions
            },
        )


def _risks(accession: str, year: int, texts: list[tuple[str, str]]) -> list[RiskFactor]:
    return [
        RiskFactor(accession, year, i, None, heading, body)
        for i, (heading, body) in enumerate(texts, start=1)
    ]


def test_section_change_is_one_minus_the_cosine_of_word_counts():
    assert section_change("Risk risk RISK.", "risk") == pytest.approx(0.0)
    assert section_change("alpha beta", "gamma delta") == pytest.approx(1.0)
    # every occurrence counts: (2*1 + 1*1) / (sqrt(5) * sqrt(2))
    assert section_change("a a b", "a b") == pytest.approx(1 - 3 / (5**0.5 * 2**0.5))
    # no stop words removed: "the" is a word like any other
    assert section_change("the the cat", "the cat cat") == pytest.approx(1 - 4 / 5)
    with pytest.raises(SignalError):
        section_change("123", "words")


def test_risk_pairs_flag_new_under_the_similarity_cut_and_changed_by_word_score_or_new():
    prior = _risks("p", 2019, [COMPETITION, CYBER])
    current = _risks("c", 2020, [COMPETITION, CYBER_GREW, PANDEMIC])
    table = risk_pair_table(current, prior, prior_accession="p").set_index("current_risk_id")
    assert table["prior_risk_id"].tolist() == [
        "p#001",
        "p#002",
        table.loc["c#003", "prior_risk_id"],
    ]
    assert table["new"].tolist() == [False, False, True]
    assert table["changed"].tolist() == [False, True, True]
    assert table.loc["c#001", "word_score"] == 0.0
    shares = risk_shares(table)
    assert shares == {
        "n_risks": 3,
        "n_new": 1,
        "n_changed": 2,
        "new_share": pytest.approx(1 / 3),
        "changed_share": pytest.approx(2 / 3),
    }


def test_a_risk_whose_word_score_is_exactly_the_cut_counts_as_changed():
    # 25 distinct words with the heading, one of them new: words added = 1/25 = 0.04
    words = [f"w{chr(97 + i)}" for i in range(23)]
    prior = _risks("p", 2019, [("Head", " ".join([*words, "old"]))])
    current = _risks("c", 2020, [("Head", " ".join([*words, "new"]))])
    table = risk_pair_table(current, prior, prior_accession="p")
    assert table.loc[0, "word_score"] == pytest.approx(0.04)
    assert bool(table.loc[0, "changed"]) and not bool(table.loc[0, "new"])


def test_worse_share_counts_unflagged_risks_as_zero():
    assert worse_share([0.8, 0.5], 4) == pytest.approx(1.3 / 4)
    assert worse_share([], 3) == 0.0
    with pytest.raises(SignalError):
        worse_share([0.1, 0.2], 1)


# --- the run, on a two-release fake of the Data lane's E7 package -------------------------


def _section(texts: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"{h}\n\n{b}" for h, b in texts)


REPORTS = [
    # release, company, accession, fiscal year, accepted (UTC), role, partner, risks or None
    ("r1", "aaa", "A0", 2019, "2020-02-01T15:00:00+00:00", "prior", "", [COMPETITION, CYBER]),
    (
        "r1",
        "aaa",
        "A1",
        2020,
        "2021-02-01T15:00:00+00:00",
        "member",
        "A0",
        [COMPETITION, CYBER_GREW, PANDEMIC],
    ),
    # accepted 22:00 Eastern on 31 Dec 2021, so it counts in 2021
    (
        "r1",
        "aaa",
        "A2",
        2021,
        "2022-01-01T03:00:00+00:00",
        "member",
        "A1",
        [COMPETITION, CYBER_GREW, PANDEMIC],
    ),
    ("r2", "bbb", "B1", 2020, "2021-03-01T15:00:00+00:00", "member", "", None),
    ("r2", "bbb", "B2", 2021, "2022-03-01T15:00:00+00:00", "member", "B1", [PANDEMIC]),
]


def _write_data(root: Path) -> Path:
    data = root / "data"
    for release in script.RELEASES:
        folder = data / "e7_sp500_10k" / release
        folder.mkdir(parents=True)
        index, sections, risks = [], [], []
        for rel, company, accession, year, accepted, role, partner, texts in REPORTS:
            if rel != release:
                continue
            index.append(
                {
                    "company": company,
                    "cik": "0000000001",
                    "accession_number": accession,
                    "form": "10-K",
                    "accepted_utc": accepted,
                    "report_date": f"{year}-12-31",
                    "fiscal_year": year,
                    "role": role,
                    "prior_accession": partner,
                    "sic": "1000",
                    "risks_published": texts is not None,
                    "section_published": True,
                }
            )
            section_texts = texts or [PANDEMIC, CYBER]
            sections.append(
                {
                    "company": company,
                    "accession_number": accession,
                    "section_words": 40,
                    "risks_passed": texts is not None,
                    "text": _section(section_texts),
                }
            )
            for risk in _risks(accession, year, texts or []):
                risks.append({"company": company, **risk.to_dict()})
        pd.DataFrame(index).to_csv(folder / "member_10k_index.csv", index=False)
        for name, rows in (
            ("member_item_1a.jsonl", sections),
            ("member_risk_factors.jsonl", risks),
        ):
            (folder / name).write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
            )
        manifest = {
            "files": [
                {"path": name, "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()}
                for name in script.DATA_FILES
            ]
        }
        (data / f"{script.PACKAGE}.{release}.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return data


def _run(root: Path, *extra: str, client: FakeClient | None = None) -> int:
    argv = ["--data-dir", str(root / "data"), "--output-dir", str(root / "E7"), *extra]
    # The SDK is optional (CI has none): the fake client takes the criteria as the question.
    return script.main(
        argv,
        client_factory=(lambda: client) if client else None,
        sdk_question_for=lambda criteria: criteria,
    )


def test_e7_refuses_data_that_differs_from_its_manifest(tmp_path):
    data = _write_data(tmp_path)
    with (data / "e7_sp500_10k" / "r2" / "member_item_1a.jsonl").open("a") as handle:
        handle.write("\n")
    assert _run(tmp_path) == 1
    assert not (tmp_path / "E7" / "signals").exists()


def test_e7_signals_pair_each_member_report_with_its_partner(tmp_path):
    _write_data(tmp_path)
    assert _run(tmp_path) == 0
    reports = pd.read_csv(tmp_path / "E7" / "signals" / "report_signals.csv").set_index(
        "accession_number"
    )
    assert reports.index.tolist() == ["A1", "A2", "B1", "B2"]  # members only
    a1 = reports.loc["A1"]
    assert a1["section_change"] == pytest.approx(
        section_change(
            _section([COMPETITION, CYBER_GREW, PANDEMIC]), _section([COMPETITION, CYBER])
        )
    )
    assert (a1["n_risks"], a1["n_prior_risks"], a1["n_new"], a1["n_changed"]) == (3, 2, 1, 2)
    assert reports.loc["A2", "section_change"] == pytest.approx(0.0)
    assert reports.loc["A2", "changed_share"] == 0.0
    # no partner: nothing; partner without published risks: Signal 1 only
    assert reports.loc["B1", ["section_change", "changed_share"]].isna().all()
    assert reports.loc["B2", "section_change"] == pytest.approx(
        section_change(_section([PANDEMIC]), _section([PANDEMIC, CYBER]))
    )
    assert pd.isna(reports.loc["B2", "changed_share"])
    pairs = pd.read_csv(tmp_path / "E7" / "signals" / "risk_pairs.csv")
    assert sorted(set(pairs["accession_number"])) == ["A1", "A2"]
    assert (pairs.loc[pairs["accession_number"] == "A2", "prior_risk_id"].str[:2] == "A1").all()
    coverage = pd.read_csv(tmp_path / "E7" / "signals" / "coverage_by_year.csv").set_index("year")
    assert coverage["member_reports"].to_dict() == {2021: 3, 2022: 1}
    assert coverage["signals_2_3"].to_dict() == {2021: 2, 2022: 0}


def test_e7_jev_is_asked_only_about_flagged_risks_then_signal_4_is_collected(tmp_path):
    _write_data(tmp_path)
    assert _run(tmp_path) == 0
    client = FakeClient()
    assert _run(tmp_path, "--collect") == 1  # nothing scored yet; collect never calls Jev
    assert _run(tmp_path, "--score", "--limit", "1", client=client) == 0
    assert len(client.calls) == 1
    assert _run(tmp_path, "--score", client=client) == 0
    assert _run(tmp_path, "--score", client=client) == 0  # all cached now
    asked = [call["state"]["current_risk_factor"]["heading"] for call in client.calls]
    assert sorted(asked) == ["Cyber attacks", "Pandemic"]  # A1's two flagged risks only
    assert all(list(call["questions"]) == ["direction"] for call in client.calls)
    assert _run(tmp_path, "--collect") == 0
    worse = pd.read_csv(tmp_path / "E7" / "jev" / "worse_share.csv").set_index("accession_number")
    assert worse.index.tolist() == ["A1", "A2"]
    assert worse.loc["A1", "worse_share"] == pytest.approx(2 * P_WORSE / 3)
    assert worse.loc["A2", "worse_share"] == 0.0
    meta = json.loads((tmp_path / "E7" / "jev" / "jev.meta.json").read_text())
    assert (meta["flagged_risks"], meta["distinct_calls"]) == (2, 2)


def test_e7_stops_before_any_call_when_the_estimate_is_over_the_budget(tmp_path):
    _write_data(tmp_path)
    assert _run(tmp_path) == 0
    client = FakeClient()
    assert _run(tmp_path, "--score", "--max-cost", "0", client=client) == 1
    assert client.calls == []


def test_e7_a_left_out_report_gets_no_signal_4_and_needs_no_answers(tmp_path):
    _write_data(tmp_path)
    assert _run(tmp_path) == 0
    leave_out = tmp_path / "left_out.csv"
    leave_out.write_text("accession_number,reason\nB2,no risk signals\n", encoding="utf-8")
    assert _run(tmp_path, "--collect", "--leave-out", str(leave_out)) == 1  # B2 has none
    leave_out.write_text("accession_number,reason\nA1,too long for Jev\n", encoding="utf-8")
    client = FakeClient()
    assert _run(tmp_path, "--score", "--leave-out", str(leave_out), client=client) == 0
    assert client.calls == []  # A1 held the only flagged risks
    assert _run(tmp_path, "--collect", "--leave-out", str(leave_out)) == 0
    worse = pd.read_csv(tmp_path / "E7" / "jev" / "worse_share.csv")
    assert worse["accession_number"].tolist() == ["A2"]
    meta = json.loads((tmp_path / "E7" / "jev" / "jev.meta.json").read_text())
    assert meta["left_out"] == [{"accession_number": "A1", "reason": "too long for Jev"}]
