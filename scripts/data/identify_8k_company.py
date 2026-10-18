"""E13 leak check: can Jev still tell which company filed a masked 8-K?

A name mask only answers the memory question if the filer stays hidden. This
asks Jev, for a fixed sample of events, to pick the filer from five companies:
the true one and four others drawn from filers in the same industry (same SIC
code, widening to its first two digits, then its first digit, then any). Chance
is one in five. The same five options, in the same order, are asked of the
original event and of each masked version, so only the text differs::

    python -m scripts.data.identify_8k_company \\
        --events <member_8k_events.jsonl> \\
        --masked-dir <E13 masked/> \\
        --sample <E13 identify_sample.csv> \\
        --companies <membership/sp500_companies.csv> \\
        --output-dir <E13 jev/identify> [--draw-sample 1000] [--score] [--workers 8]

``--draw-sample N`` writes the sample (seed 313) and the options for each event
(``identify_options.csv``) and stops; nothing is asked. ``--score`` asks Jev;
without it, the answers already cached are collected into ``identify.csv``,
one row per event and version, and any gap stops it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mci_gru.filings.change import ChangeGradingError, score_states
from mci_gru.filings.events import Event, event_state, load_events
from mci_gru.filings.jev import Question
from mci_gru.filings.masking import display_name
from scripts.data.score_8k_events import SharedAnswerCache, sampled_events
from scripts.data.score_company_risk_changes import NoJev, _commit, _sha256

try:
    from typesafe_sdk import Choice, TypeSafeClient
except ImportError:  # the [filings] extra is needed only for --score
    Choice = TypeSafeClient = None

MODEL = "jev-1.13.0"
SEED = 313
OPTIONS = ("company_a", "company_b", "company_c", "company_d", "company_e")
VERSIONS = ("original", "name", "deep")
INSTRUCTIONS = (
    "This 8-K was filed by one of five companies. The filer's name may be withheld "
    "in the text. Which company filed it?"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--masked-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True, help="Written by --draw-sample")
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draw-sample", type=int, default=None, metavar="N")
    parser.add_argument(
        "--from-events", type=Path, default=None, help="CSV to draw the sample from"
    )
    parser.add_argument("--score", action="store_true", help="Call Jev (costs money)")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def _rng(*parts: str) -> np.random.Generator:
    seed = int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng(seed)


def company_sic(raw_events: Path) -> dict[str, str]:
    """Each filer's most common SIC code across its events."""
    found: dict[str, dict[str, int]] = {}
    with raw_events.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            sic = record.get("sic") or ""
            tally = found.setdefault(record["company"], {})
            tally[sic] = tally.get(sic, 0) + 1
    return {company: max(t, key=lambda s: (t[s], s)) for company, t in found.items()}


def distractors(
    company: str, event_id: str, sic: dict[str, str], names: dict[str, str]
) -> list[str]:
    """Four other filers for one event, nearest industry first, drawn from the event's id."""
    own = sic.get(company, "")
    taken = {company}
    shown = {names[company]}
    chosen: list[str] = []
    for width in (4, 2, 1, 0):
        if len(chosen) == 4:
            break
        pool = sorted(
            c
            for c, code in sic.items()
            if c not in taken
            and c in names
            and names[c] not in shown
            and code[:width] == own[:width]
        )
        rng = _rng(event_id, str(width))
        for c in rng.permutation(pool)[: 4 - len(chosen)]:
            chosen.append(str(c))
            taken.add(str(c))
            shown.add(names[str(c)])
    return chosen


def options_table(events: list[Event], sic: dict[str, str], names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for event in events:
        companies = [event.company, *distractors(event.company, event.event_id, sic, names)]
        order = _rng(event.event_id, "slots").permutation(len(OPTIONS))
        row: dict[str, Any] = {"event_id": event.event_id, "company": event.company}
        for slot, index in zip(OPTIONS, order, strict=True):
            row[slot] = companies[index]
            if index == 0:
                row["truth"] = slot
        rows.append(row)
    return pd.DataFrame(rows)


def question_for(row: Any, names: dict[str, str]) -> Question:
    return Question(
        question_id="filer",
        kind="choice",
        instructions=INSTRUCTIONS,
        options=tuple((slot, f"The filer is {names[getattr(row, slot)]}.") for slot in OPTIONS),
    )


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    companies = pd.read_csv(args.companies, dtype=str)
    names = {row.key: display_name(str(row.name)) for row in companies.itertuples()}

    if args.draw_sample is not None:
        pool = pd.read_csv(args.from_events or args.sample, usecols=["event_id"], dtype=str)
        ids = sorted(pool["event_id"])
        picked = sorted(
            np.random.default_rng(SEED).choice(ids, size=args.draw_sample, replace=False)
        )
        pd.DataFrame({"event_id": picked}).to_csv(args.sample, index=False)
        events = sampled_events(load_events(args.events), pd.DataFrame({"event_id": picked}))
        table = options_table(events, company_sic(args.events), names)
        table.to_csv(out / "identify_options.csv", index=False)
        print(
            f"Sample of {len(picked)} and its options written (sha256 {_sha256(args.sample)[:8]})."
        )
        return 0

    sample = pd.read_csv(args.sample, usecols=["event_id"], dtype=str)
    table = pd.read_csv(out / "identify_options.csv", dtype=str)
    if list(table["event_id"]) != list(sample["event_id"]):
        print("identify_options.csv does not match the sample. Refusing to go on.")
        return 1
    versions = {
        "original": sampled_events(load_events(args.events), sample),
        "name": sampled_events(load_events(args.masked_dir / "name" / "events.jsonl"), sample),
        "deep": sampled_events(load_events(args.masked_dir / "deep" / "events.jsonl"), sample),
    }
    if args.score:
        if TypeSafeClient is None or not os.environ.get("TYPESAFE_API_KEY"):
            print(
                'Scoring needs the TypeSafe SDK and TYPESAFE_API_KEY: pip install -e ".[filings]"'
            )
            return 1
    else:
        client = NoJev()
    rows_by_id = {row.event_id: row for row in table.itertuples()}
    results = []
    try:
        for version, events in versions.items():
            cache = SharedAnswerCache(out / "answers" / f"{version}.jsonl")

            def ask(event: Event, cache: SharedAnswerCache = cache) -> Any:
                row = rows_by_id[event.event_id]
                question = question_for(row, names)
                return score_states(
                    [(event.event_id, event_state(event))],
                    question,
                    sdk_question_for=lambda criteria, q=question: Choice(
                        instructions=q.instructions, criteria=criteria
                    ),
                    client=client or TypeSafeClient(model=MODEL),
                    cache=cache,
                    model=MODEL,
                    labels=OPTIONS,
                )[0]

            with ThreadPoolExecutor(max_workers=max(1, args.workers if args.score else 1)) as pool:
                answers = list(pool.map(ask, events))
            for event, answer in zip(events, answers, strict=True):
                truth = rows_by_id[event.event_id].truth
                results.append(
                    {
                        "event_id": event.event_id,
                        "company": event.company,
                        "version": version,
                        "truth": truth,
                        "choice": answer.choice,
                        "correct": answer.choice == truth,
                        "p_truth": answer.probabilities.get(truth),
                        "input_tokens": answer.input_tokens,
                    }
                )
    except ChangeGradingError as error:
        print(f"Cannot collect: {error}")
        return 1
    if args.score:
        print(f"Asked {len(results)} questions.")
        return 0
    found = pd.DataFrame(results)
    found.to_csv(out / "identify.csv", index=False)
    meta = {
        "experiment": "E13-name-masking",
        "commit": _commit(),
        "model": MODEL,
        "events": len(sample),
        "chance": 1 / len(OPTIONS),
        "accuracy": found.groupby("version")["correct"].mean().round(4).to_dict(),
        "mean_p_truth": found.groupby("version")["p_truth"].mean().round(4).to_dict(),
        "input_tokens_total": int(found["input_tokens"].fillna(0).sum()),
        "sha256": {
            "sample": _sha256(args.sample),
            "identify_options.csv": _sha256(out / "identify_options.csv"),
            **{f"answers/{v}.jsonl": _sha256(out / "answers" / f"{v}.jsonl") for v in VERSIONS},
            "identify.csv": _sha256(out / "identify.csv"),
        },
    }
    (out / "identify.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
