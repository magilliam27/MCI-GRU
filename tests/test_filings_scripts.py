"""The filings pull and the pair scoring, end to end with canned EDGAR and Jev (#289)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.data.export_sec_risk_factors as export
import scripts.data.score_risk_factor_pairs as score
from mci_gru.data.input_manifest import read_input_manifest
from mci_gru.filings.edgar import ALPHABET_CIK, GOOGLE_INC_CIK

FIXTURE = Path(__file__).parent / "fixtures" / "filings" / "tenk_item1a_sample.htm"
AGENT = "MCI-GRU tests tests@example.com"


def _block(rows):
    return {
        "accessionNumber": [r[0] for r in rows],
        "form": [r[1] for r in rows],
        "filingDate": [r[3][:10] for r in rows],
        "reportDate": [r[2] for r in rows],
        "acceptanceDateTime": [r[3] for r in rows],
        "primaryDocument": [r[4] for r in rows],
    }


def _document(year: int, extra_risk: bool) -> bytes:
    html = FIXTURE.read_text(encoding="utf-8").replace("in 2025", f"in {year}")
    if extra_risk:
        html = html.replace(
            '<div id="i1b">',
            '<div><span style="font-weight:700;font-style:italic">'
            "Generative AI may expose us to new liabilities.</span></div>"
            "<div><span>AI systems may produce inaccurate or harmful content.</span></div>"
            '<div id="i1b">',
        )
    return html.encode("utf-8")


def _edgar(broken: bool = False) -> dict[str, bytes]:
    google_main = {
        "filings": {
            "recent": _block(
                [
                    (
                        "0001288776-15-000008",
                        "10-K",
                        "2014-12-31",
                        "2015-02-06T21:10:00.000Z",
                        "g14.htm",
                    ),
                ]
            ),
            "files": [{"name": "CIK0001288776-submissions-001.json"}],
        }
    }
    # The older file holds fiscal 2013, inside the window, and fiscal 2012, before it.
    google_old = _block(
        [
            ("0001288776-14-000020", "10-K", "2013-12-31", "2014-02-12T21:10:00.000Z", "g13.htm"),
            ("0001288776-13-000010", "10-K", "2012-12-31", "2013-01-29T21:10:00.000Z", "g12.htm"),
        ]
    )
    alphabet_main = {
        "filings": {
            "recent": _block(
                [
                    (
                        "0001652044-16-000012",
                        "10-K",
                        "2015-12-31",
                        "2016-02-11T21:10:00.000Z",
                        "a15.htm",
                    )
                ]
            )
        }
    }
    base = "https://www.sec.gov/Archives/edgar/data"
    return {
        f"https://data.sec.gov/submissions/CIK{GOOGLE_INC_CIK}.json": json.dumps(
            google_main
        ).encode(),
        "https://data.sec.gov/submissions/CIK0001288776-submissions-001.json": json.dumps(
            google_old
        ).encode(),
        f"https://data.sec.gov/submissions/CIK{ALPHABET_CIK}.json": json.dumps(
            alphabet_main
        ).encode(),
        f"{base}/1288776/000128877614000020/g13.htm": _document(2013, extra_risk=False),
        f"{base}/1288776/000128877615000008/g14.htm": _document(2014, extra_risk=False),
        f"{base}/1652044/000165204416000012/a15.htm": (
            b"<html><body><div>ITEM 1. BUSINESS</div></body></html>"
            if broken
            else _document(2015, extra_risk=True)
        ),
    }


class Transport:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses
        self.seen: list[tuple[str, dict]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> bytes:
        self.seen.append((url, headers))
        return self.responses[url]


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch):
    monkeypatch.setattr(export, "MIN_INTERVAL_SECONDS", 0.0)


def _export(tmp_path: Path, transport: Transport, manifest: bool = True) -> int:
    argv = [
        "--user-agent",
        AGENT,
        "--package-root",
        str(tmp_path / "package"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    if manifest:
        argv += ["--manifest-output", str(tmp_path / "manifest.json")]
    return export.main(argv, transport=transport)


def test_export_reads_both_filers_and_older_files_and_publishes_a_manifest(tmp_path):
    transport = Transport(_edgar())
    assert _export(tmp_path, transport) == 0
    out = tmp_path / "package" / "filings"
    index = pd.read_csv(out / "alphabet_10k_index.csv")
    assert index["fiscal_year"].tolist() == [2013, 2014, 2015]
    assert index["document_sha256"].str.len().eq(64).all()
    risks = [
        json.loads(line) for line in (out / "alphabet_risk_factors.jsonl").read_text().splitlines()
    ]
    assert {risk["fiscal_year"] for risk in risks} == {2013, 2014, 2015}
    manifest = read_input_manifest(tmp_path / "manifest.json")
    assert {record.path for record in manifest.manifest.files} == {
        "filings/alphabet_10k_index.csv",
        "filings/alphabet_10k_duplicates.csv",
        "filings/alphabet_risk_factors.jsonl",
        "filings/alphabet_extraction_report.csv",
        "filings/alphabet_risk_factors.meta.json",
    }


def test_every_request_names_the_client_as_sec_asks(tmp_path):
    transport = Transport(_edgar())
    _export(tmp_path, transport, manifest=False)
    assert transport.seen and all(headers["User-Agent"] == AGENT for _, headers in transport.seen)


def test_user_agent_without_contact_email_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        export.main(
            [
                "--user-agent",
                "anonymous",
                "--package-root",
                str(tmp_path),
                "--cache-dir",
                str(tmp_path),
            ],
            transport=Transport({}),
        )


def test_a_second_pull_reads_everything_from_the_cache(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    again = Transport(_edgar())
    _export(tmp_path, again, manifest=False)
    assert again.seen == []


def test_an_unreadable_item_1a_blocks_the_manifest(tmp_path):
    assert _export(tmp_path, Transport(_edgar(broken=True))) == 1
    assert not (tmp_path / "manifest.json").exists()
    report = pd.read_csv(tmp_path / "package" / "filings" / "alphabet_extraction_report.csv")
    assert report.loc[report["fiscal_year"] == 2015, "error"].str.contains("Item 1A").all()


class FakeJev:
    def __init__(self) -> None:
        self.calls = 0

    def system_one(self, *, state, questions):
        self.calls += 1
        return SimpleNamespace(
            model="jev-1.13.0",
            request_id=f"r{self.calls}",
            usage=SimpleNamespace(input_tokens=50),
            nouls={name: SimpleNamespace(noul=0.9) for name in questions},
        )


def _score(tmp_path: Path, *extra: str, client=None) -> int:
    risks = tmp_path / "package" / "filings" / "alphabet_risk_factors.jsonl"
    argv = [
        "--risk-factors",
        str(risks),
        "--output-dir",
        str(tmp_path / "pairs"),
        "--top-k",
        "2",
        "--label-sample",
        "6",
        *extra,
    ]
    return score.main(argv, client=client, sdk_question="same_risk")


def test_pairs_and_blind_sheet_are_written_without_calling_jev(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    client = FakeJev()
    assert _score(tmp_path, client=client) == 0
    assert client.calls == 0
    pairs = pd.read_csv(tmp_path / "pairs" / "pairs.csv")
    assert sorted(pairs["fiscal_year"].unique()) == [2014, 2015]
    sheet = pd.read_csv(tmp_path / "pairs" / "label_sheet.csv", keep_default_na=False)
    assert "similarity" not in sheet.columns and sheet["same_risk"].eq("").all()
    assert not (tmp_path / "pairs" / "scores.csv").exists()


def test_scoring_twice_calls_jev_once_per_distinct_pair(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    first = FakeJev()
    assert _score(tmp_path, "--score", client=first) == 0
    pairs = pd.read_csv(tmp_path / "pairs" / "pairs.csv")
    # Risks reworded nowhere between years make identical pairs; Jev sees each once.
    texts = ["current_heading", "current_body", "prior_heading", "prior_body"]
    distinct = len(pairs.drop_duplicates(subset=texts))
    assert distinct < len(pairs)
    assert first.calls == distinct
    second = FakeJev()
    assert _score(tmp_path, "--score", client=second) == 0
    assert second.calls == 0
    scores = pd.read_csv(tmp_path / "pairs" / "scores.csv")
    assert scores["cached"].all() and scores["probability"].eq(0.9).all()


def test_an_existing_label_sheet_is_never_overwritten(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    _score(tmp_path, client=FakeJev())
    sheet_path = tmp_path / "pairs" / "label_sheet.csv"
    labelled = pd.read_csv(sheet_path, keep_default_na=False)
    labelled["same_risk"] = "yes"
    labelled.to_csv(sheet_path, index=False)
    _score(tmp_path, "--seed", "1", client=FakeJev())
    assert pd.read_csv(sheet_path, keep_default_na=False)["same_risk"].eq("yes").all()
