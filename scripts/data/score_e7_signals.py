"""E7: each S&P 500 member 10-K's change signals for the returns test (#310).

The research lead's frozen design (``E7-returns/CARD.md``) asks for four signals
per member report, keyed by accession, each against the company's previous
report (its pair partner). :mod:`mci_gru.filings.report_signals` defines them.
No price or return is read here. Three steps::

    # 1. Signals 1 to 3, the per-risk pair table and coverage per year. No Jev
    #    calls, no API key. Refuses data that differs from the published manifests.
    python -m scripts.data.score_e7_signals \\
        --data-dir /mnt/project-files/filings/data \\
        --output-dir /mnt/project-files/filings/experiments/E7-returns

    # 2. Jev on the flagged risks only (questions_v4 direction, unchanged): one
    #    test call, then the rest. Stops before any call if the estimated cost of
    #    every flagged risk is over --max-cost. Companies run side by side, each
    #    with its own answer cache.
    ... --score --limit 1
    ... --score [--workers 8]

    # 3. Signal 4 from the caches. Never calls Jev: a flagged risk with no answer
    #    stops it, unless its report is on the --leave-out list (accession_number,
    #    reason), which the metadata records.
    ... --collect [--leave-out left_out.csv]

What it writes under ``--output-dir``:

- ``signals/report_signals.csv``: one row per member report: its index facts,
  Signal 1 (``section_change``), the risk counts and Signals 2 and 3
  (``new_share``, ``changed_share``); blank where a signal does not apply.
- ``signals/risk_pairs.csv``: one row per risk of every report with risk
  signals: its closest partner risk, similarity, word change and the two flags.
- ``signals/coverage_by_year.csv`` and ``signals/signals.meta.json``.
- ``jev/answers/<company>.jsonl``: every Jev answer, appended, never rewritten.
- With ``--collect``: ``jev/answers.csv`` (one row per flagged risk),
  ``jev/worse_share.csv`` (Signal 4 per report) and ``jev/jev.meta.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.change import DIRECTION_LABELS, ChangeGradingError, score_states
from mci_gru.filings.jev import AnswerCache, load_question_set, pair_state
from mci_gru.filings.pairing import RiskPair
from mci_gru.filings.report_signals import risk_pair_table, risk_shares, section_change
from mci_gru.filings.risk_factors import RiskFactor
from scripts.data.score_company_risk_changes import NoJev, _commit
from scripts.data.score_risk_changes import CHARS_PER_TOKEN

try:
    from typesafe_sdk import Choice, TypeSafeClient
except ImportError:  # the [filings] extra is needed only for --score
    Choice = TypeSafeClient = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = REPO_ROOT / "configs" / "filings" / "questions_v4.yaml"
QUESTION_ID = "direction"
RELEASES = ("r1", "r2")
PACKAGE = "e7_member_10k_risk_factors"
DATA_FILES = ("member_10k_index.csv", "member_item_1a.jsonl", "member_risk_factors.jsonl")
#: Jev's list price per input token (docs.typesafe.ai/models, jev-1.13.0: $0.042 per million).
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
#: Jev's per-question context; a state estimated over this would be refused.
MAX_QUESTION_TOKENS = 32_000
EVENT_ZONE = "America/New_York"
REPORT_COLUMNS = (
    "accession_number",
    "company",
    "release",
    "cik",
    "form",
    "accepted_utc",
    "report_date",
    "fiscal_year",
    "sic",
    "prior_accession",
    "section_words",
    "prior_section_words",
    "section_change",
    "n_risks",
    "n_prior_risks",
    "n_new",
    "n_changed",
    "new_share",
    "changed_share",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="filings/data")
    parser.add_argument("--output-dir", type=Path, required=True, help="E7-returns")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--score", action="store_true", help="Call Jev (costs money)")
    step.add_argument("--collect", action="store_true", help="Write Signal 4 from the caches")
    parser.add_argument("--company", default=None, help="Score this company only")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N risks")
    parser.add_argument("--workers", type=int, default=8, help="Companies scored side by side")
    parser.add_argument("--max-cost", type=float, default=8.0, help="US dollars, all flagged risks")
    parser.add_argument(
        "--leave-out",
        type=Path,
        default=None,
        help="CSV of accession_number and reason: reports given no Signal 4",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_inputs(data_dir: Path) -> dict[str, str]:
    """Each file read, with its digest, after checking it against its release's manifest."""
    found = {}
    for release in RELEASES:
        manifest_path = data_dir / f"{PACKAGE}.{release}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
        found[manifest_path.name] = _sha256(manifest_path)
        for name in DATA_FILES:
            digest = _sha256(data_dir / "e7_sp500_10k" / release / name)
            if listed.get(name) != digest:
                raise ValueError(f"{release}/{name} differs from {manifest_path.name}")
            found[f"{release}/{name}"] = digest
    return found


def load_index(data_dir: Path) -> pd.DataFrame:
    """Every report of both releases; accessions must be unique and partners present."""
    index = pd.concat(
        [
            pd.read_csv(data_dir / "e7_sp500_10k" / r / "member_10k_index.csv", dtype=str).assign(
                release=r
            )
            for r in RELEASES
        ],
        ignore_index=True,
    )
    if not index["accession_number"].is_unique:
        raise ValueError("an accession repeats across the index")
    partners = index["prior_accession"].dropna()
    if not partners.isin(index["accession_number"]).all():
        raise ValueError("a pair partner is missing from the index")
    return index


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_sections(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Each published Item 1A by accession: its text and word count."""
    return {
        row["accession_number"]: {"text": row["text"], "words": row["section_words"]}
        for r in RELEASES
        for row in _jsonl(data_dir / "e7_sp500_10k" / r / "member_item_1a.jsonl")
    }


def load_risks(data_dir: Path) -> dict[str, list[RiskFactor]]:
    """Each report's published risks by accession, in reading order."""
    risks: dict[str, list[RiskFactor]] = {}
    for r in RELEASES:
        for row in _jsonl(data_dir / "e7_sp500_10k" / r / "member_risk_factors.jsonl"):
            risk = RiskFactor.from_dict(row)
            risks.setdefault(risk.accession_number, []).append(risk)
    for items in risks.values():
        items.sort(key=lambda risk: risk.index)
    return risks


def signals(
    index: pd.DataFrame,
    sections: dict[str, dict[str, Any]],
    risks: dict[str, list[RiskFactor]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signals 1 to 3 for every member report, and the per-risk pair table."""
    published = set(index.loc[index["risks_published"] == "True", "accession_number"])
    if published != set(risks):
        raise ValueError("the risk file and the index disagree on which reports have risks")
    rows, tables = [], []
    for report in index[index["role"] == "member"].itertuples(index=False):
        accession, prior = report.accession_number, report.prior_accession
        row = {column: getattr(report, column, None) for column in REPORT_COLUMNS[:10]}
        row["prior_accession"] = prior if isinstance(prior, str) else None
        now, before = sections.get(accession), sections.get(prior)
        row["section_words"] = now["words"] if now else None
        row["prior_section_words"] = before["words"] if before else None
        if now and before:
            row["section_change"] = section_change(now["text"], before["text"])
        if accession in risks and prior in risks:
            table = risk_pair_table(risks[accession], risks[prior], prior_accession=prior)
            tables.append(table)
            row |= risk_shares(table)
            row["n_prior_risks"] = len(risks[prior])
        rows.append(row)
    report_table = pd.DataFrame(rows).reindex(columns=list(REPORT_COLUMNS))
    for column in ("n_risks", "n_prior_risks", "n_new", "n_changed"):
        report_table[column] = report_table[column].astype("Int64")
    return report_table, pd.concat(tables, ignore_index=True)


def acceptance_year(accepted_utc: pd.Series) -> pd.Series:
    """The calendar year of acceptance in US Eastern time."""
    return pd.to_datetime(accepted_utc, utc=True).dt.tz_convert(EVENT_ZONE).dt.year


def coverage_by_year(reports: pd.DataFrame) -> pd.DataFrame:
    """Member reports per acceptance year, and how many each signal covers."""
    table = (
        reports.assign(year=acceptance_year(reports["accepted_utc"]))
        .groupby("year")
        .agg(
            member_reports=("accession_number", "size"),
            signal_1=("section_change", "count"),
            signals_2_3=("changed_share", "count"),
        )
        .reset_index()
    )
    table["signal_1_share"] = (table["signal_1"] / table["member_reports"]).round(3)
    table["signals_2_3_share"] = (table["signals_2_3"] / table["member_reports"]).round(3)
    return table


def flagged_items(
    pairs: pd.DataFrame, risks: dict[str, list[RiskFactor]], companies: dict[str, str]
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Each company's flagged risks as (pair id, state), in the pair table's order."""
    by_id = {risk.risk_id: risk for items in risks.values() for risk in items}
    items: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for row in pairs[pairs["changed"]].itertuples(index=False):
        pair = RiskPair(by_id[row.current_risk_id], by_id[row.prior_risk_id], 1, row.similarity)
        if pair.pair_id != row.pair_id:
            raise ValueError(f"{row.pair_id}: the pair table and the risk file disagree")
        items.setdefault(companies[row.accession_number], []).append(
            (row.pair_id, pair_state(pair))
        )
    return items


def estimated_tokens(state: dict[str, Any], question: Any) -> int:
    chars = len(json.dumps(state, ensure_ascii=False)) + len(question.instructions)
    return (chars + sum(len(n) + len(d) for n, d in question.options)) // CHARS_PER_TOKEN


def _write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")


def main(
    argv: list[str] | None = None, *, client_factory: Any = None, sdk_question_for: Any = None
) -> int:
    args = parse_args(argv)
    signals_dir, jev_dir = args.output_dir / "signals", args.output_dir / "jev"
    try:
        inputs = checked_inputs(args.data_dir)
        index = load_index(args.data_dir)
    except ValueError as error:
        print(f"Cannot use the data: {error}. Refusing to go on.")
        return 1
    risks = load_risks(args.data_dir)

    if not (args.score or args.collect):
        reports, pairs = signals(index, load_sections(args.data_dir), risks)
        signals_dir.mkdir(parents=True, exist_ok=True)
        reports.to_csv(signals_dir / "report_signals.csv", index=False)
        pairs.to_csv(signals_dir / "risk_pairs.csv", index=False)
        coverage = coverage_by_year(reports)
        coverage.to_csv(signals_dir / "coverage_by_year.csv", index=False)
        meta = {
            "experiment": "E7-returns",
            "commit": _commit(),
            "member_reports": len(reports),
            "signal_1_reports": int(reports["section_change"].notna().sum()),
            "signals_2_3_reports": int(reports["changed_share"].notna().sum()),
            "risk_pairs": len(pairs),
            "new_risks": int(pairs["new"].sum()),
            "changed_risks": int(pairs["changed"].sum()),
            "definitions": {
                "section_change": "1 - cosine similarity of [a-z]+ word counts, lower-cased",
                "new": "TF-IDF similarity to the closest partner risk < 0.25",
                "changed": "word_score >= 0.04, or new",
                "year": f"acceptance year in {EVENT_ZONE}",
            },
            "sha256": {
                **inputs,
                **{
                    f"signals/{name}": _sha256(signals_dir / name)
                    for name in ("report_signals.csv", "risk_pairs.csv", "coverage_by_year.csv")
                },
            },
        }
        _write_meta(signals_dir / "signals.meta.json", meta)
        print(json.dumps({k: v for k, v in meta.items() if k != "sha256"}, indent=2, default=str))
        print(coverage.to_string(index=False))
        return 0

    question_set = load_question_set(args.questions)
    question = question_set.questions[QUESTION_ID]
    reports = pd.read_csv(signals_dir / "report_signals.csv", dtype={"accession_number": str})
    pairs = pd.read_csv(signals_dir / "risk_pairs.csv", dtype=str)
    pairs["changed"] = pairs["changed"] == "True"
    pairs["similarity"] = pairs["similarity"].astype(float)
    left_out = pd.DataFrame(columns=["accession_number", "reason"])
    if args.leave_out is not None:
        left_out = pd.read_csv(args.leave_out, usecols=["accession_number", "reason"], dtype=str)
        with_risks = set(reports.loc[reports["changed_share"].notna(), "accession_number"])
        unknown = sorted(set(left_out["accession_number"]) - with_risks)
        if unknown or not left_out["accession_number"].is_unique:
            print(f"--leave-out must name reports with risk signals once each: {unknown[:3]}")
            return 1
    pairs = pairs[~pairs["accession_number"].isin(left_out["accession_number"])]
    reports = reports[~reports["accession_number"].isin(left_out["accession_number"])]
    companies = dict(zip(index["accession_number"], index["company"], strict=True))
    items = flagged_items(pairs, risks, companies)
    tokens = {
        company: [estimated_tokens(state, question) for _, state in found]
        for company, found in items.items()
    }
    total_tokens = sum(sum(found) for found in tokens.values())
    estimate = total_tokens * USD_PER_INPUT_TOKEN
    too_long = sum(t > MAX_QUESTION_TOKENS for found in tokens.values() for t in found)

    def sdk_question(criteria: dict[str, str]) -> Any:
        if sdk_question_for is not None:
            return sdk_question_for(criteria)
        return Choice(instructions=question.instructions, criteria=criteria)

    def run(company: str, found: list, client: Any, asked: Any = sdk_question) -> list:
        return score_states(
            found,
            question,
            sdk_question_for=asked,
            client=client,
            cache=AnswerCache(jev_dir / "answers" / f"{company}.jsonl"),
            model=question_set.model,
            labels=DIRECTION_LABELS,
        )

    if args.score:
        print(
            json.dumps(
                {
                    "flagged_risks": sum(len(found) for found in items.values()),
                    "companies": len(items),
                    "estimated_input_tokens": total_tokens,
                    "estimated_cost_usd": round(estimate, 2),
                    "estimated_over_question_context": too_long,
                },
                indent=2,
            )
        )
        if estimate > args.max_cost:
            print(f"Estimated ${estimate:.2f} is over --max-cost ${args.max_cost:.2f}. Stopping.")
            return 1
        if client_factory is None:
            if TypeSafeClient is None:
                print('Scoring needs the TypeSafe SDK: pip install -e ".[filings]"')
                return 1
            if not os.environ.get("TYPESAFE_API_KEY"):
                print("Scoring needs TYPESAFE_API_KEY in the environment.")
                return 1

            def client_factory() -> Any:
                return TypeSafeClient(model=question_set.model)

        chosen = {args.company: items.get(args.company, [])} if args.company else items
        if args.limit is not None:
            left, limited = args.limit, {}
            for company, found in chosen.items():
                if left <= 0:
                    break
                limited[company], left = found[:left], left - len(found[:left])
            chosen = limited
        new_calls = new_tokens = done = 0
        failed: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(run, company, found, client_factory()): company
                for company, found in chosen.items()
            }
            for future in as_completed(futures):
                company = futures[future]
                try:
                    answered = future.result()
                except Exception as error:  # report every company, then fail the run
                    failed[company] = f"{type(error).__name__}: {error}"
                    continue
                done += 1
                fresh = [item for item in answered if not item.cached]
                new_calls += len(fresh)
                new_tokens += sum(item.input_tokens or 0 for item in fresh)
                print(f"{done}/{len(chosen)} {company}: {len(fresh)} new", flush=True)
                if args.limit is not None:
                    for item in answered:
                        print(
                            json.dumps(
                                {"pair_id": item.pair_id, "choice": item.choice}
                                | item.probabilities
                            )
                        )
        print(
            json.dumps(
                {
                    "new_calls": new_calls,
                    "input_tokens_new": new_tokens,
                    "cost_usd_new": round(new_tokens * USD_PER_INPUT_TOKEN, 4),
                    "failed": failed,
                },
                indent=2,
            )
        )
        return 1 if failed else 0

    rows = []
    try:
        for company, found in items.items():
            # Collecting builds no SDK question: NoJev refuses any call first.
            for item in run(company, found, NoJev(), lambda criteria: criteria):
                rows.append(
                    {"company": company, "pair_id": item.pair_id, "choice": item.choice}
                    | {f"p_{label}": item.probabilities.get(label) for label in DIRECTION_LABELS}
                    | {"option_order": "|".join(item.option_order)}
                )
    except ChangeGradingError as error:
        print(f"Cannot collect: {error}")
        return 1
    answer_columns = ["company", "pair_id", "choice"]
    answer_columns += [f"p_{label}" for label in DIRECTION_LABELS] + ["option_order"]
    answers = pd.DataFrame(rows, columns=answer_columns)
    answers = pairs.loc[pairs["changed"], ["accession_number", "pair_id"]].merge(
        answers, on="pair_id", how="left", validate="one_to_one"
    )
    covered = reports[reports["changed_share"].notna()]
    flagged_p = answers.groupby("accession_number")["p_worse"].agg(["sum", "size"])
    worse = covered[["accession_number", "company", "n_risks", "n_changed"]].merge(
        flagged_p, left_on="accession_number", right_index=True, how="left"
    )
    if (worse["size"].fillna(0).astype(int) != worse["n_changed"].astype(int)).any():
        print("Cannot collect: a report's flagged risks and its answers differ in number")
        return 1
    worse["worse_share"] = worse["sum"].fillna(0.0) / worse["n_risks"].astype(int)
    worse = worse.drop(columns=["sum", "size"])
    jev_dir.mkdir(parents=True, exist_ok=True)
    answers.to_csv(jev_dir / "answers.csv", index=False)
    worse.to_csv(jev_dir / "worse_share.csv", index=False)
    records = [
        json.loads(line)
        for path in sorted((jev_dir / "answers").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    input_tokens = sum(r.get("input_tokens") or 0 for r in records)
    meta = {
        "experiment": "E7-returns",
        "commit": _commit(),
        "model": question_set.model,
        "question_file": args.questions.name,
        "question": QUESTION_ID,
        "flagged_risks": len(answers),
        "reports": len(worse),
        "left_out": left_out.to_dict(orient="records"),
        "distinct_calls": len(records),
        "input_tokens_total": input_tokens,
        "cost_usd": round(input_tokens * USD_PER_INPUT_TOKEN, 2),
        "models_answering": sorted({r["model"] for r in records}),
        "choices": answers["choice"].value_counts().to_dict(),
        "worse_share": "sum of p_worse over the report's flagged risks / its number of risks",
        "sha256": {
            "signals/report_signals.csv": _sha256(signals_dir / "report_signals.csv"),
            "signals/risk_pairs.csv": _sha256(signals_dir / "risk_pairs.csv"),
            "jev/answers.csv": _sha256(jev_dir / "answers.csv"),
            "jev/worse_share.csv": _sha256(jev_dir / "worse_share.csv"),
            **({"leave_out": _sha256(args.leave_out)} if args.leave_out else {}),
            **{
                f"jev/answers/{path.name}": _sha256(path)
                for path in sorted((jev_dir / "answers").glob("*.jsonl"))
            },
        },
    }
    _write_meta(jev_dir / "jev.meta.json", meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "sha256"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
