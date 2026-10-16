"""E10: how good or bad is an 8-K event for shareholders? (#305).

Two frozen Choice questions (``configs/filings/questions_v5.yaml``), each asked
on its own for every event the research lead sampled: ``rating`` (very bad, bad,
neutral, good, very good) and ``size`` (the amount involved against prior
annual revenue). Jev reads :func:`mci_gru.filings.events.event_state`. Options
are shuffled per event. Three steps::

    # 1. Freeze the baseline for the sampled events (or every event, before the
    #    sample exists). No Jev calls, no API key.
    python -m scripts.data.score_8k_events \\
        --events <the Data lane's 8-K events .jsonl> \\
        --sample /mnt/project-files/filings/experiments/E10-8k-events/<sample>.csv \\
        --lm-dictionary <Loughran-McDonald_MasterDictionary_1993-2025.csv> \\
        --output-dir /mnt/project-files/filings/experiments/E10-8k-events/jev

    # 2. Score: one test call, then every sampled event, one question or both.
    ... --score --question rating --limit 1
    ... --score [--question size]

    # 3. Collect both questions' answers into scores.csv. Never calls Jev: an
    #    event with no answer stops it.
    ... --collect

The baseline, fixed on the card before any label: the item code's lean, then
the Loughran-McDonald net tone (negative minus positive words as a share of all
words), with E8's pinned dictionary. ``baseline_badness`` orders events exactly
that way, worst highest.

What it writes under ``--output-dir``: ``baselines.csv`` with
``baselines.meta.json``; ``answers/<question>.jsonl``, appended and never
rewritten; and with ``--collect``, ``scores.csv`` (each question's choice, its
probability for every option and the option order shown, plus the rating score
and the bad score) with ``scores.meta.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.change import ChangeGradingError, lm_words, score_states
from mci_gru.filings.events import (
    RATING_LABELS,
    SIZE_LABELS,
    EventDataError,
    bad_score,
    baseline_badness,
    event_state,
    item_lean,
    load_events,
    rating_score,
    tone_shares,
)
from mci_gru.filings.jev import AnswerCache, load_question_set
from scripts.data.score_change_direction import LM_SHA256
from scripts.data.score_company_risk_changes import NoJev, _commit, _sha256
from scripts.data.score_risk_changes import CHARS_PER_TOKEN

try:
    from typesafe_sdk import Choice, TypeSafeClient
except ImportError:  # the [filings] extra is needed only for --score
    Choice = TypeSafeClient = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = REPO_ROOT / "configs" / "filings" / "questions_v5.yaml"
#: Each question and the options it must offer, in the order scores.csv lists them.
QUESTIONS = {"rating": RATING_LABELS, "size": SIZE_LABELS}
BASELINE_COLUMNS = (
    "event_id",
    "company",
    "items",
    "lean",
    "neg_share",
    "pos_share",
    "net_tone",
    "baseline_badness",
    "text_chars",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=None, help="CSV with an event_id column")
    parser.add_argument("--lm-dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--score", action="store_true", help="Call Jev (costs money)")
    step.add_argument("--collect", action="store_true", help="Write scores.csv from the caches")
    parser.add_argument("--question", choices=sorted(QUESTIONS), default=None)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N events")
    return parser.parse_args(argv)


def sampled_events(events: list, sample: pd.DataFrame | None) -> list:
    """The sampled events in the sample's order, or every event; any gap or repeat stops it."""
    if sample is None:
        return events
    ids = sample["event_id"].tolist()
    if len(set(ids)) != len(ids):
        raise EventDataError("a sampled event id repeats")
    by_id = {event.event_id: event for event in events}
    missing = [event_id for event_id in ids if event_id not in by_id]
    if missing:
        raise EventDataError(f"{len(missing)} sampled events are not in the events file")
    return [by_id[event_id] for event_id in ids]


def baselines(events: list, negative: frozenset[str], positive: frozenset[str]) -> pd.DataFrame:
    """The frozen baseline for every event, in order."""
    rows = []
    for event in events:
        neg, pos = tone_shares(event.text, negative, positive)
        lean = item_lean(event.items)
        rows.append(
            {
                "event_id": event.event_id,
                "company": event.company,
                "items": "|".join(event.items),
                "lean": lean,
                "neg_share": neg,
                "pos_share": pos,
                "net_tone": neg - pos,
                "baseline_badness": baseline_badness(lean, neg - pos),
                "text_chars": len(event.text),
            }
        )
    return pd.DataFrame(rows, columns=list(BASELINE_COLUMNS))


def _inputs(args: argparse.Namespace) -> dict[str, str]:
    found = {"events": _sha256(args.events), "lm_dictionary": _sha256(args.lm_dictionary)}
    if args.sample is not None:
        found["sample"] = _sha256(args.sample)
    return found


def main(argv: list[str] | None = None, *, client: Any = None, sdk_question_for: Any = None) -> int:
    args = parse_args(argv)
    question_set = load_question_set(args.questions)
    if sorted(question_set.questions) != sorted(QUESTIONS):
        print(f"{args.questions.name} must hold exactly {', '.join(QUESTIONS)}.")
        return 1
    if _sha256(args.lm_dictionary) != LM_SHA256:
        print(f"{args.lm_dictionary.name} is not the frozen dictionary (sha256 {LM_SHA256}).")
        return 1
    if (args.score or args.collect) and args.sample is None:
        print("Scoring and collecting need --sample: only the lead's sample is scored.")
        return 1
    sample = None
    if args.sample is not None:
        sample = pd.read_csv(args.sample, usecols=["event_id"], dtype=str)
    try:
        events = sampled_events(load_events(args.events), sample)
    except EventDataError as error:
        print(f"Cannot use the events: {error}. Refusing to go on.")
        return 1
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if not (args.score or args.collect):
        negative = lm_words(args.lm_dictionary, "Negative")
        positive = lm_words(args.lm_dictionary, "Positive")
        table = baselines(events, negative, positive)
        table.to_csv(out / "baselines.csv", index=False)
        chars = sum(
            len(json.dumps(event_state(event), ensure_ascii=False))
            + len(question.instructions)
            + sum(len(n) + len(d) for n, d in question.options)
            for event in events
            for question in question_set.questions.values()
        )
        meta = {
            "experiment": "E10-8k-events",
            "commit": _commit(),
            "events": len(table),
            "events_by_lean": {int(k): int(v) for k, v in table["lean"].value_counts().items()},
            "lean": "2 if any of 1.02 1.03 2.05 2.06 3.01 4.02; 1 if 4.01; else 0",
            "net_tone": "Loughran-McDonald (negative - positive) words / all words",
            "baseline_badness": "lean + (1 + net_tone) / 3; higher is worse",
            "lm_words": {"negative": len(negative), "positive": len(positive)},
            "estimated_input_tokens": chars // CHARS_PER_TOKEN,
            "sha256": {**_inputs(args), "baselines.csv": _sha256(out / "baselines.csv")},
        }
        (out / "baselines.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(meta, indent=2))
        print("Baselines frozen. Re-run with --score --question rating --limit 1.")
        return 0

    if args.collect:
        client = NoJev()
    elif client is None:
        if TypeSafeClient is None:
            print('Scoring needs the TypeSafe SDK: pip install -e ".[filings]"')
            return 1
        if not os.environ.get("TYPESAFE_API_KEY"):
            print("Scoring needs TYPESAFE_API_KEY in the environment.")
            return 1
        client = TypeSafeClient(model=question_set.model)
    chosen = events if args.limit is None else events[: args.limit]
    items = [(event.event_id, event_state(event)) for event in chosen]
    names = list(QUESTIONS) if args.question is None or args.collect else [args.question]
    scored = {}
    try:
        for name in names:
            question = question_set.questions[name]

            def sdk_question(criteria: dict[str, str], question=question) -> Any:
                if sdk_question_for is not None:
                    return sdk_question_for(criteria)
                return Choice(instructions=question.instructions, criteria=criteria)

            scored[name] = score_states(
                items,
                question,
                sdk_question_for=sdk_question,
                client=client,
                cache=AnswerCache(out / "answers" / f"{name}.jsonl"),
                model=question_set.model,
                labels=QUESTIONS[name],
            )
    except ChangeGradingError as error:
        print(f"Cannot collect: {error}")
        return 1
    answered = [item for name in names for item in scored[name]]
    if not args.collect:
        run = {
            "questions": names,
            "scored": len(answered),
            "new_calls": sum(not item.cached for item in answered),
            "input_tokens_new": sum(i.input_tokens or 0 for i in answered if not i.cached),
        }
        print(json.dumps(run, indent=2))
        if args.limit is not None:
            print(
                json.dumps(
                    [
                        {"question": i.question_id, "event_id": i.pair_id, "choice": i.choice}
                        | i.probabilities
                        for i in answered
                    ],
                    indent=2,
                )
            )
        return 0

    rows = []
    for event, *answers in zip(events, *(scored[name] for name in names), strict=True):
        row: dict[str, Any] = {"event_id": event.event_id, "company": event.company}
        for name, answer in zip(names, answers, strict=True):
            row[f"{name}_choice"] = answer.choice
            for label in QUESTIONS[name]:
                row[f"{name}_p_{label}"] = answer.probabilities.get(label)
            if name == "rating":
                row["rating_score"] = rating_score(answer.probabilities)
                row["bad_score"] = bad_score(answer.probabilities)
            row[f"{name}_order"] = "|".join(answer.option_order)
        rows.append(row)
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "scores.csv", index=False)
    records = {
        name: [
            json.loads(line)
            for line in (out / "answers" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for name in names
    }
    meta = {
        "experiment": "E10-8k-events",
        "commit": _commit(),
        "model": question_set.model,
        "question_file": args.questions.name,
        "events": len(scores),
        "rating_score": "sum of p(option) x weight, very_bad -2 ... very_good +2",
        "bad_score": "p_bad + p_very_bad",
        "distinct_calls": {name: len(found) for name, found in records.items()},
        "input_tokens_total": sum(
            r.get("input_tokens") or 0 for found in records.values() for r in found
        ),
        "models_answering": sorted({r["model"] for found in records.values() for r in found}),
        "choices": {name: scores[f"{name}_choice"].value_counts().to_dict() for name in names},
        "sha256": {
            **_inputs(args),
            **{
                f"answers/{name}.jsonl": _sha256(out / "answers" / f"{name}.jsonl")
                for name in names
            },
            "scores.csv": _sha256(out / "scores.csv"),
        },
    }
    (out / "scores.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
