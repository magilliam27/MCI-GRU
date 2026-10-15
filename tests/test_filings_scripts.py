"""The filings pull and the pair scoring, end to end with canned EDGAR and Jev (#289)."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.data.export_sec_risk_factors as export
import scripts.data.score_risk_factor_pairs as score
from mci_gru.data.input_manifest import read_input_manifest
from mci_gru.filings.edgar import ALPHABET_CIK, GOOGLE_INC_CIK
from mci_gru.filings.filers import FILERS, Filer

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


def _export(
    tmp_path: Path, transport: Transport, manifest: bool = True, extra: list[str] | None = None
) -> int:
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
    return export.main(argv + (extra or []), transport=transport)


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
        "filings/alphabet_extraction_checks.csv",
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


def test_a_second_pull_rereads_only_the_filers_main_submissions_files(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    again = Transport(_edgar())
    _export(tmp_path, again, manifest=False)
    assert sorted(url for url, _ in again.seen) == sorted(
        f"https://data.sec.gov/submissions/CIK{cik}.json" for cik in (GOOGLE_INC_CIK, ALPHABET_CIK)
    )


class Flaky(Transport):
    """Fails the first request for each URL with the given error, then answers."""

    def __init__(self, responses, error) -> None:
        super().__init__(responses)
        self.error = error
        self.failed: set[str] = set()

    def __call__(self, url, headers):
        if url not in self.failed:
            self.failed.add(url)
            self.seen.append((url, headers))
            raise self.error(url)
        return super().__call__(url, headers)


def _http(code):
    return lambda url: urllib.error.HTTPError(url, code, "error", None, None)


@pytest.mark.parametrize(
    ("error", "wait"),
    [
        (_http(429), export.RATE_LIMITED_WAIT_SECONDS),
        (_http(503), export.RETRY_WAIT_SECONDS),
        (lambda url: urllib.error.URLError("connection reset"), export.RETRY_WAIT_SECONDS),
        (lambda url: TimeoutError("read timed out"), export.RETRY_WAIT_SECONDS),
    ],
)
def test_rate_limits_server_errors_and_dropped_connections_are_retried_after_a_wait(
    tmp_path, monkeypatch, error, wait
):
    sleeps: list[float] = []
    monkeypatch.setattr(export.time, "sleep", sleeps.append)
    transport = Flaky(_edgar(), error)
    assert _export(tmp_path, transport, manifest=False) == 0
    assert len(transport.seen) == 2 * len(transport.failed)
    assert sleeps and all(seconds == wait for seconds in sleeps)


def test_a_request_refused_outright_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(export.time, "sleep", lambda seconds: None)
    transport = Flaky(_edgar(), _http(403))
    with pytest.raises(urllib.error.HTTPError):
        _export(tmp_path, transport, manifest=False)
    assert len(transport.seen) == 1


def test_a_document_that_is_not_utf8_is_read_as_cp1252_and_reported(tmp_path):
    responses = _edgar()
    url = "https://www.sec.gov/Archives/edgar/data/1288776/000128877615000008/g14.htm"
    responses[url] = responses[url].replace(b"in 2014", b"in 2014 \x96 worldwide")
    assert _export(tmp_path, Transport(responses), manifest=False) == 0
    out = tmp_path / "package" / "filings"
    report = pd.read_csv(out / "alphabet_extraction_report.csv")
    assert report.set_index("fiscal_year")["encoding"].to_dict() == {
        2013: "utf-8",
        2014: "cp1252",
        2015: "utf-8",
    }
    risks = (out / "alphabet_risk_factors.jsonl").read_text(encoding="utf-8")
    assert "in 2014 \u2013 worldwide" in risks


def test_an_unreadable_item_1a_blocks_the_manifest(tmp_path):
    assert _export(tmp_path, Transport(_edgar(broken=True))) == 1
    assert not (tmp_path / "manifest.json").exists()
    report = pd.read_csv(tmp_path / "package" / "filings" / "alphabet_extraction_report.csv")
    assert report.loc[report["fiscal_year"] == 2015, "error"].str.contains("Item 1A").all()


def test_every_filing_is_checked_and_a_clean_pull_lists_no_finding(tmp_path):
    assert _export(tmp_path, Transport(_edgar())) == 0
    out = tmp_path / "package" / "filings"
    checks = pd.read_csv(out / "alphabet_extraction_checks.csv")
    assert list(checks.columns) == ["accession_number", "fiscal_year", "check", "risk_id", "detail"]
    assert checks.empty
    report = pd.read_csv(out / "alphabet_extraction_report.csv")
    assert report["check_findings"].tolist() == [0, 0, 0]


def test_a_failed_check_is_reported_and_blocks_the_manifest(tmp_path, capsys):
    # A page number left in a body, as Alphabet's "11." ones were before #289's fix.
    responses = _edgar()
    url = "https://www.sec.gov/Archives/edgar/data/1288776/000128877615000008/g14.htm"
    responses[url] = responses[url].replace(
        b"intensely competitive.</span>", b"intensely competitive. 12. Table of Contents</span>"
    )
    assert _export(tmp_path, Transport(responses)) == 1
    assert not (tmp_path / "manifest.json").exists()
    out = tmp_path / "package" / "filings"
    checks = pd.read_csv(out / "alphabet_extraction_checks.csv")
    assert checks[["fiscal_year", "check", "risk_id"]].values.tolist() == [
        [2014, "page_furniture_in_text", "0001288776-15-000008#002"]
    ]
    report = pd.read_csv(out / "alphabet_extraction_report.csv").set_index("fiscal_year")
    assert report["check_findings"].to_dict() == {2013: 0, 2014: 1, 2015: 0}
    assert "BLOCKING 0001288776-15-000008: page_furniture_in_text" in capsys.readouterr().out


ACME_CIK = "0000000042"
ACME = Filer("acme", "Acme Corp.", (ACME_CIK,), layout="run_in")
ACME_DOCUMENT = (
    "<html><body><div>ITEM 1A. RISK FACTORS</div>"
    "<div><b>Supply and Demand</b></div>"
    "<div><i>Economic conditions.</i> The demand for energy follows growth.</div>"
    "<div><i>Competition.</i> Rivals are many.</div>"
    "<div>ITEM 1B. UNRESOLVED STAFF COMMENTS</div></body></html>"
)


def _acme() -> dict[str, bytes]:
    main = {
        "filings": {
            "recent": _block(
                [
                    (
                        "0000000042-25-000001",
                        "10-K",
                        "2024-12-31",
                        "2025-02-20T21:10:00.000Z",
                        "x.htm",
                    )
                ]
            )
        }
    }
    return {
        f"https://data.sec.gov/submissions/CIK{ACME_CIK}.json": json.dumps(main).encode(),
        "https://www.sec.gov/Archives/edgar/data/42/000000004225000001/x.htm": (
            ACME_DOCUMENT.encode()
        ),
    }


@pytest.fixture
def acme(monkeypatch):
    monkeypatch.setitem(FILERS, "acme", ACME)


def test_a_named_filer_is_read_with_its_own_layout_under_its_own_key(tmp_path, acme):
    argv = ["--filer", "acme", "--manifest-output", str(tmp_path / "manifest.json")]
    assert _export(tmp_path, Transport(_acme()), manifest=False, extra=argv) == 0
    out = tmp_path / "package" / "filings"
    risks = [
        json.loads(line) for line in (out / "acme_risk_factors.jsonl").read_text().splitlines()
    ]
    assert [risk["heading"] for risk in risks] == ["Economic conditions.", "Competition."]
    meta = json.loads((out / "acme_risk_factors.meta.json").read_text())
    assert (meta["filer"], meta["layout"], meta["filers"]) == ("acme", "run_in", [ACME_CIK])
    assert meta["network_requests"] == 2
    manifest = read_input_manifest(tmp_path / "manifest.json")
    assert manifest.manifest.package_id == "acme_10k_risk_factors"
    assert not (out / "alphabet_risk_factors.jsonl").exists()


def test_the_fiscal_year_rule_and_each_years_period_are_recorded(tmp_path, acme):
    argv = ["--filer", "acme"]
    assert _export(tmp_path, Transport(_acme()), extra=argv) == 0
    meta = json.loads(
        (tmp_path / "package" / "filings" / "acme_risk_factors.meta.json").read_text()
    )
    assert meta["period_of_report_by_fiscal_year"] == {"2024": "2024-12-31"}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["metadata"]["fiscal_year_rule"] == export.FISCAL_YEAR_RULE


def test_two_reports_in_one_fiscal_year_block_the_manifest(tmp_path, acme, capsys):
    # A changed year end would give one fiscal year two reports, and pairing
    # takes the previous fiscal year as the prior filing.
    responses = _acme()
    main = json.loads(responses[f"https://data.sec.gov/submissions/CIK{ACME_CIK}.json"])
    main["filings"]["recent"] = _block(
        [
            ("0000000042-25-000001", "10-K", "2024-12-31", "2025-02-20T21:10:00.000Z", "x.htm"),
            ("0000000042-24-000001", "10-K", "2024-06-30", "2024-08-20T21:10:00.000Z", "y.htm"),
        ]
    )
    responses[f"https://data.sec.gov/submissions/CIK{ACME_CIK}.json"] = json.dumps(main).encode()
    responses["https://www.sec.gov/Archives/edgar/data/42/000000004224000001/y.htm"] = (
        ACME_DOCUMENT.encode()
    )
    assert _export(tmp_path, Transport(responses), extra=["--filer", "acme"]) == 1
    assert not (tmp_path / "manifest.json").exists()
    assert "BLOCKING acme: fiscal 2024 has 2 reports" in capsys.readouterr().out


def test_several_filers_share_one_manifest_named_by_package_id(tmp_path, acme):
    responses = {**_edgar(), **_acme()}
    argv = ["--filer", "alphabet", "--filer", "acme", "--package-id", "ten_companies"]
    assert _export(tmp_path, Transport(responses), extra=argv) == 0
    manifest = read_input_manifest(tmp_path / "manifest.json")
    assert manifest.manifest.package_id == "ten_companies"
    paths = {record.path for record in manifest.manifest.files}
    assert {"filings/alphabet_risk_factors.jsonl", "filings/acme_risk_factors.jsonl"} <= paths
    assert len(paths) == 12
    acme_meta = json.loads(
        (tmp_path / "package" / "filings" / "acme_risk_factors.meta.json").read_text()
    )
    assert acme_meta["network_requests"] == 2


def test_several_filers_without_a_package_id_are_refused(tmp_path, acme):
    with pytest.raises(SystemExit):
        _export(tmp_path, Transport({}), extra=["--filer", "alphabet", "--filer", "acme"])


def test_only_filings_in_the_fiscal_year_range_are_read(tmp_path):
    transport = Transport(_edgar())
    argv = ["--first-fiscal-year", "2014", "--last-fiscal-year", "2014"]
    assert _export(tmp_path, transport, manifest=False, extra=argv) == 0
    index = pd.read_csv(tmp_path / "package" / "filings" / "alphabet_10k_index.csv")
    assert index["fiscal_year"].tolist() == [2014]
    assert not any(url.endswith(("g13.htm", "a15.htm")) for url, _ in transport.seen)


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
    pairs = pd.read_csv(tmp_path / "pairs" / "sealed" / "pairs.csv")
    assert sorted(pairs["fiscal_year"].unique()) == [2014, 2015]
    sheet = pd.read_csv(tmp_path / "pairs" / "label_sheet.csv", keep_default_na=False)
    assert "similarity" not in sheet.columns and sheet["same_risk"].eq("").all()
    assert not (tmp_path / "pairs" / "sealed" / "scores.csv").exists()


def test_scoring_twice_calls_jev_once_per_distinct_pair(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    first = FakeJev()
    assert _score(tmp_path, "--score", client=first) == 0
    pairs = pd.read_csv(tmp_path / "pairs" / "sealed" / "pairs.csv")
    # Risks reworded nowhere between years make identical pairs; Jev sees each once.
    texts = ["current_heading", "current_body", "prior_heading", "prior_body"]
    distinct = len(pairs.drop_duplicates(subset=texts))
    assert distinct < len(pairs)
    assert first.calls == distinct
    second = FakeJev()
    assert _score(tmp_path, "--score", client=second) == 0
    assert second.calls == 0
    scores = pd.read_csv(tmp_path / "pairs" / "sealed" / "scores.csv")
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


def test_only_the_label_sheet_sits_outside_the_sealed_folder(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    _score(tmp_path, "--score", client=FakeJev())
    out = tmp_path / "pairs"
    assert sorted(path.name for path in out.iterdir()) == ["label_sheet.csv", "sealed"]
    assert {path.name for path in (out / "sealed").iterdir()} == {
        "pairs.csv",
        "label_design.csv",
        "answers.jsonl",
        "scores.csv",
        "risk_verdicts.csv",
    }


def test_a_rerun_that_changes_the_pairs_under_a_label_sheet_stops(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    assert _score(tmp_path, client=FakeJev()) == 0
    risks = tmp_path / "package" / "filings" / "alphabet_risk_factors.jsonl"
    argv = ["--risk-factors", str(risks), "--output-dir", str(tmp_path / "pairs"), "--top-k", "1"]
    assert score.main(argv, client=FakeJev(), sdk_question="same_risk") == 1


class HeadingJev(FakeJev):
    """Says 'same risk' only when the two headings match, as a careful reader would here."""

    def system_one(self, *, state, questions):
        self.calls += 1
        same = state["current_risk_factor"]["heading"] == state["prior_year_risk_factor"]["heading"]
        return SimpleNamespace(
            model="jev-1.13.0",
            request_id=f"r{self.calls}",
            usage=SimpleNamespace(input_tokens=50),
            nouls={name: SimpleNamespace(noul=0.95 if same else 0.05) for name in questions},
        )


def _risk_counts(tmp_path: Path) -> dict[int, int]:
    risks = tmp_path / "package" / "filings" / "alphabet_risk_factors.jsonl"
    years = [json.loads(line)["fiscal_year"] for line in risks.read_text().splitlines()]
    return {year: years.count(year) for year in set(years)}


def test_jev_is_asked_about_every_prior_year_risk_by_default(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    assert _score(tmp_path, "--score", client=HeadingJev()) == 0
    counts = _risk_counts(tmp_path)
    pairs = pd.read_csv(tmp_path / "pairs" / "sealed" / "pairs.csv")
    assert len(pairs) == sum(counts[year] * counts[year - 1] for year in (2014, 2015))
    # The label sheet still comes from the --top-k 2 shortlist.
    design = pd.read_csv(tmp_path / "pairs" / "sealed" / "label_design.csv")
    shortlisted = set(pairs.loc[pairs["rank"] <= 2, "pair_id"])
    assert set(design["pair_id"]) <= shortlisted


def test_shortlist_scope_asks_jev_only_about_the_top_k(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    client = HeadingJev()
    assert _score(tmp_path, "--score", "--jev-scope", "shortlist", client=client) == 0
    pairs = pd.read_csv(tmp_path / "pairs" / "sealed" / "pairs.csv")
    assert pairs["rank"].max() == 2
    assert len(pd.read_csv(tmp_path / "pairs" / "sealed" / "scores.csv")) == len(pairs)


def test_risk_verdicts_name_jevs_best_match_and_flag_a_new_risk(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    assert _score(tmp_path, "--score", client=HeadingJev()) == 0
    verdicts = pd.read_csv(tmp_path / "pairs" / "sealed" / "risk_verdicts.csv")
    counts = _risk_counts(tmp_path)
    assert len(verdicts) == counts[2014] + counts[2015]
    assert verdicts["complete"].all()
    new = verdicts["current_heading"].str.startswith("Generative AI")
    assert new.sum() == 1
    assert verdicts.loc[new, "jev_probability"].item() == pytest.approx(0.05)
    carried = verdicts.loc[~new]
    assert (carried["jev_best_prior_heading"] == carried["current_heading"]).all()
    assert carried["jev_probability"].eq(0.95).all()


def test_a_limited_run_marks_risks_it_did_not_finish(tmp_path):
    _export(tmp_path, Transport(_edgar()), manifest=False)
    assert _score(tmp_path, "--score", "--limit", "2", client=HeadingJev()) == 0
    verdicts = pd.read_csv(tmp_path / "pairs" / "sealed" / "risk_verdicts.csv")
    assert len(verdicts) == 1 and not verdicts["complete"].any()
