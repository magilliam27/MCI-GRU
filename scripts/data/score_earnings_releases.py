"""E17: an earnings release read against the company's previous one.

Three frozen Choice questions (``configs/filings/questions_earnings_v1.yaml``),
asked together in one call per event: ``setup`` (much worse .. much better than
the previous release set up), ``guidance`` and ``promise``. Jev reads
:func:`mci_gru.filings.earnings.release_state`; each question's options are
shuffled per event. Three steps::

    # 1. Freeze the baseline (word tone of both releases) and estimate the cost.
    #    No Jev calls, no API key.
    python -m scripts.data.score_earnings_releases \\
        --events <earnings_events.jsonl> --draw test \\
        --lm-dictionary <Loughran-McDonald_MasterDictionary_1993-2025.csv> \\
        --output-dir <experiment>/jev

    # 2. Score: one test call, then every event.
    ... --score --limit 1
    ... --score --workers 8

    # 3. Collect every answer into scores.csv. Never calls Jev: an event with no
    #    answer stops it.
    ... --collect

The baseline, fixed on the card before any score: the Loughran-McDonald net
tone (negative minus positive words as a share of all words) of the current
release minus that of the previous one, with E8's pinned dictionary.

What it writes under ``--output-dir``: ``baselines.csv`` with
``baselines.meta.json``; ``answers.jsonl``, one line per call, appended and
never rewritten; and with ``--collect``, ``scores.csv`` (each question's choice,
its probability for every option and the option order shown, plus the setup
score) with ``scores.meta.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.change import lm_words
from mci_gru.filings.earnings import (
    QUESTION_LABELS,
    question_order,
    release_state,
    setup_score,
)
from mci_gru.filings.eight_k import EDGAR_TIMEZONE, MAX_TEXT_CHARS
from mci_gru.filings.events import tone_shares
from mci_gru.filings.jev import AnswerCache, ModelDriftError, cache_key, load_question_set
from scripts.data.score_change_direction import LM_SHA256
from scripts.data.score_company_risk_changes import NoJev, _commit, _sha256
from scripts.data.score_e7_signals import USD_PER_INPUT_TOKEN
from scripts.data.score_risk_changes import CHARS_PER_TOKEN

try:
    from typesafe_sdk import Choice, TypeSafeClient
except ImportError:  # the [filings] extra is needed only for --score
    Choice = TypeSafeClient = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = REPO_ROOT / "configs" / "filings" / "questions_earnings_v1.yaml"
BASELINE_COLUMNS = (
    "event_id",
    "company",
    "net_tone",
    "previous_net_tone",
    "tone_change",
    "text_chars",
    "previous_text_chars",
)


class EarningsDataError(ValueError):
    """An events file that cannot be scored as it stands."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--draw", default=None, help="Only events of this draw (default all)")
    parser.add_argument("--lm-dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--score", action="store_true", help="Call Jev (costs money)")
    step.add_argument("--collect", action="store_true", help="Write scores.csv from the answers")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N events")
    parser.add_argument("--workers", type=int, default=1, help="Calls in flight at once")
    parser.add_argument("--experiment", default="E17-earnings-vs-previous")
    return parser.parse_args(argv)


def load_earnings(path: Path, draw: str | None) -> list[dict[str, Any]]:
    """The events of one draw (or all), in file order; a repeat or an over-long text stops it."""
    events = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if draw is not None:
        events = [event for event in events if event["draw"] == draw]
    ids = [event["event_id"] for event in events]
    if len(set(ids)) != len(ids):
        raise EarningsDataError("an event id repeats")
    for event in events:
        for text in (event["text"], event["previous"]["text"]):
            if len(text) > MAX_TEXT_CHARS:
                raise EarningsDataError(f"{event['event_id']}: a text is over {MAX_TEXT_CHARS}")
    return events


def filed_day(accepted_utc: str) -> str:
    """The Eastern calendar day of an acceptance time, as EDGAR dates a filing."""
    return pd.Timestamp(accepted_utc).tz_convert(EDGAR_TIMEZONE).strftime("%Y-%m-%d")


def event_release_state(event: dict[str, Any]) -> dict[str, Any]:
    previous = event["previous"]
    return release_state(
        event["company_name"],
        filed_day(event["accepted_utc"]),
        event["text"],
        filed_day(previous["accepted_utc"]),
        previous["text"],
    )


def baselines(
    events: list[dict[str, Any]], negative: frozenset[str], positive: frozenset[str]
) -> pd.DataFrame:
    rows = []
    for event in events:
        neg, pos = tone_shares(event["text"], negative, positive)
        prev_neg, prev_pos = tone_shares(event["previous"]["text"], negative, positive)
        rows.append(
            {
                "event_id": event["event_id"],
                "company": event["company"],
                "net_tone": neg - pos,
                "previous_net_tone": prev_neg - prev_pos,
                "tone_change": (neg - pos) - (prev_neg - prev_pos),
                "text_chars": len(event["text"]),
                "previous_text_chars": len(event["previous"]["text"]),
            }
        )
    return pd.DataFrame(rows, columns=list(BASELINE_COLUMNS))


def call_spec(question_set: Any, event_id: str) -> dict[str, Any]:
    """The questions as asked for one event, options in its own order; hashed into the key."""
    return {
        name: {**question.spec(), "option_order": question_order(question, event_id)}
        for name, question in question_set.questions.items()
    }


def ask_event(
    event: dict[str, Any],
    question_set: Any,
    *,
    client: Any,
    cache: AnswerCache,
    lock: threading.Lock,
    sdk_question_for: Any,
) -> tuple[dict[str, Any], bool]:
    """One event's answers, from the cache when held; whether they were cached."""
    state = event_release_state(event)
    spec = call_spec(question_set, event["event_id"])
    key = cache_key(state, spec, question_set.model)
    with lock:
        if key in cache:
            return cache.get(key), True
    descriptions = {name: dict(q.options) for name, q in question_set.questions.items()}
    questions = {
        name: sdk_question_for(
            question_set.questions[name].instructions,
            {option: descriptions[name][option] for option in spec[name]["option_order"]},
        )
        for name in spec
    }
    response = client.system_one(state=state, questions=questions)
    if response.model != question_set.model:
        raise ModelDriftError(f"asked {question_set.model}, answered by {response.model}")
    record = {
        "key": key,
        "event_id": event["event_id"],
        "question": spec,
        "model": response.model,
        "answers": {
            name: {
                "choice": str(response.choices[name].choice),
                "probabilities": {
                    k: float(v) for k, v in response.choices[name].probabilities.items()
                },
            }
            for name in spec
        },
        "request_id": response.request_id,
        "input_tokens": getattr(response.usage, "input_tokens", None),
    }
    with lock:
        cache.put(record)
    return record, False


def score_rows(records: list[dict[str, Any]], events: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for event, record in zip(events, records, strict=True):
        row: dict[str, Any] = {"event_id": event["event_id"], "company": event["company"]}
        for name, labels in QUESTION_LABELS.items():
            answer = record["answers"][name]
            row[f"{name}_choice"] = answer["choice"]
            for label in labels:
                row[f"{name}_p_{label}"] = answer["probabilities"].get(label)
            row[f"{name}_order"] = "|".join(record["question"][name]["option_order"])
        row["setup_score"] = setup_score(record["answers"]["setup"]["probabilities"])
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None, *, client: Any = None, sdk_question_for: Any = None) -> int:
    args = parse_args(argv)
    question_set = load_question_set(args.questions)
    if sorted(question_set.questions) != sorted(QUESTION_LABELS):
        print(f"{args.questions.name} must hold exactly {', '.join(QUESTION_LABELS)}.")
        return 1
    for name, labels in QUESTION_LABELS.items():
        if sorted(n for n, _ in question_set.questions[name].options) != sorted(labels):
            print(f"{name} must offer exactly {', '.join(labels)}.")
            return 1
    if _sha256(args.lm_dictionary) != LM_SHA256:
        print(f"{args.lm_dictionary.name} is not the frozen dictionary (sha256 {LM_SHA256}).")
        return 1
    try:
        events = load_earnings(args.events, args.draw)
    except EarningsDataError as error:
        print(f"Cannot use the events: {error}. Refusing to go on.")
        return 1
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    inputs = {"events": _sha256(args.events), "lm_dictionary": _sha256(args.lm_dictionary)}

    if not (args.score or args.collect):
        table = baselines(
            events,
            lm_words(args.lm_dictionary, "Negative"),
            lm_words(args.lm_dictionary, "Positive"),
        )
        table.to_csv(out / "baselines.csv", index=False)
        questions_chars = sum(
            len(q.instructions) + sum(len(n) + len(d) for n, d in q.options)
            for q in question_set.questions.values()
        )
        chars = sum(
            len(json.dumps(event_release_state(event), ensure_ascii=False)) + questions_chars
            for event in events
        )
        tokens = chars // CHARS_PER_TOKEN
        meta = {
            "experiment": args.experiment,
            "commit": _commit(),
            "draw": args.draw or "all",
            "events": len(table),
            "tone_change": "Loughran-McDonald net tone of this release minus the previous one",
            "estimated_input_tokens": tokens,
            "estimated_cost_usd": round(tokens * USD_PER_INPUT_TOKEN, 2),
            "sha256": {**inputs, "baselines.csv": _sha256(out / "baselines.csv")},
        }
        (out / "baselines.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))
        print("Baselines frozen. Re-run with --score --limit 1.")
        return 0

    cache = AnswerCache(out / "answers.jsonl")
    lock = threading.Lock()
    chosen = events if args.limit is None else events[: args.limit]
    if args.collect:
        missing = [
            e["event_id"]
            for e in chosen
            if cache_key(
                event_release_state(e), call_spec(question_set, e["event_id"]), question_set.model
            )
            not in cache
        ]
        if missing:
            print(f"Cannot collect: {len(missing)} events have no answer yet; score them first.")
            return 1
        client = NoJev()
    elif client is None:
        if TypeSafeClient is None:
            print('Scoring needs the TypeSafe SDK: pip install -e ".[filings]"')
            return 1
        if not os.environ.get("TYPESAFE_API_KEY"):
            print("Scoring needs TYPESAFE_API_KEY in the environment.")
            return 1
    if sdk_question_for is None:
        sdk_question_for = lambda instructions, criteria: Choice(  # noqa: E731
            instructions=instructions, criteria=criteria
        )
    local = threading.local()

    def worker_client() -> Any:
        if client is not None:
            return client
        if not hasattr(local, "client"):
            local.client = TypeSafeClient(model=question_set.model)
        return local.client

    def one(event: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        return ask_event(
            event,
            question_set,
            client=worker_client(),
            cache=cache,
            lock=lock,
            sdk_question_for=sdk_question_for,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        answered = list(pool.map(one, chosen))
    if not args.collect:
        new = [record for record, cached in answered if not cached]
        run = {
            "scored": len(answered),
            "new_calls": len(new),
            "input_tokens_new": sum(record["input_tokens"] or 0 for record in new),
        }
        print(json.dumps(run, indent=2))
        if args.limit is not None:
            print(json.dumps([record["answers"] for record, _ in answered], indent=2))
        return 0

    scores = score_rows([record for record, _ in answered], chosen)
    scores.to_csv(out / "scores.csv", index=False)
    records = [
        json.loads(line)
        for line in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = {
        "experiment": args.experiment,
        "commit": _commit(),
        "model": question_set.model,
        "question_file": args.questions.name,
        "draw": args.draw or "all",
        "events": len(scores),
        "setup_score": "sum of p(option) x weight, much_worse -2 ... much_better +2",
        "distinct_calls": len(records),
        "input_tokens_total": sum(r.get("input_tokens") or 0 for r in records),
        "models_answering": sorted({r["model"] for r in records}),
        "choices": {
            name: scores[f"{name}_choice"].value_counts().to_dict() for name in QUESTION_LABELS
        },
        "sha256": {
            **inputs,
            "answers.jsonl": _sha256(out / "answers.jsonl"),
            "scores.csv": _sha256(out / "scores.csv"),
        },
    }
    (out / "scores.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
