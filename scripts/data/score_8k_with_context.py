"""E18: E10's rating question asked with the company's latest earnings release.

``configs/filings/questions_context_v1.yaml`` is E16's rating question with the
release named in its context; the state is E16's with the release added
(:func:`mci_gru.filings.context.context_state`). E16's own answers are the "without
context" side, so the main arm costs one call per event. A second, small arm
re-asks E16's question exactly (``questions_v5.yaml``, no release) as a fresh
draw, to count how often a plain repeat moves an answer. Three steps::

    # 1. Freeze the baselines; estimate the cost. No Jev calls, no API key.
    python -m scripts.data.score_8k_with_context \\
        --events sample_events.jsonl --releases context_releases.jsonl \\
        --lm-dictionary <Loughran-McDonald_MasterDictionary_1993-2025.csv> \\
        --output-dir /mnt/project-files/filings/experiments/E18-neutral-context/jev

    # 2. Score one draw: a test call, then all; or the plain repeat arm.
    ... --score --draw test --limit 1
    ... --score --draw test --workers 8
    ... --score --arm plain --draw test --plain-sample 500:1820

    # 3. Collect an arm's answers into a scores file. Never calls Jev.
    ... --collect --draw test [--arm plain --plain-sample 500:1820]

The baselines, fixed on the card before any score: the Loughran-McDonald net
tone (negative minus positive words as a share of all words, E8's dictionary)
of the 8-K alone, of the release alone, and of the two texts read as one.

What it writes under ``--output-dir``: ``baselines.csv`` with
``baselines.meta.json``; ``answers/<arm>_rating.jsonl``, appended and never
rewritten; and with ``--collect``, ``<arm>_<draw>_scores.csv`` (the choice, every
option's probability, the option order shown, the rating score and the bad
score) with its ``.meta.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.change import ChangeGradingError, lm_words, score_states
from mci_gru.filings.context import context_state
from mci_gru.filings.earnings import draw_rank
from mci_gru.filings.events import (
    RATING_LABELS,
    EventDataError,
    bad_score,
    event_state,
    load_events,
    rating_score,
    tone_shares,
)
from mci_gru.filings.jev import load_question_set
from scripts.data.score_8k_events import SharedAnswerCache
from scripts.data.score_change_direction import LM_SHA256
from scripts.data.score_company_risk_changes import NoJev, _commit, _sha256
from scripts.data.score_e7_signals import USD_PER_INPUT_TOKEN
from scripts.data.score_risk_changes import CHARS_PER_TOKEN

try:
    from typesafe_sdk import Choice, TypeSafeClient
except ImportError:  # the [filings] extra is needed only for --score
    Choice = TypeSafeClient = None

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTION_FILES = {
    "context": REPO_ROOT / "configs" / "filings" / "questions_context_v1.yaml",
    "plain": REPO_ROOT / "configs" / "filings" / "questions_v5.yaml",
}
#: The plain arm repeats E16's call, so its cache key carries a draw number.
PLAIN_DRAW = 2
BASELINE_COLUMNS = (
    "event_id",
    "company",
    "draw",
    "stratum",
    "release_age_days",
    "net_tone_8k",
    "net_tone_release",
    "net_tone_both",
    "release_chars",
)


def parse_sample(value: str) -> tuple[int, int]:
    size, seed = value.split(":")
    return int(size), int(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--releases", type=Path, required=True)
    parser.add_argument("--lm-dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--score", action="store_true", help="Call Jev (costs money)")
    step.add_argument("--collect", action="store_true", help="Write a scores file from the cache")
    parser.add_argument("--arm", choices=sorted(QUESTION_FILES), default="context")
    parser.add_argument("--draw", default=None, help="Score or collect only this draw")
    parser.add_argument(
        "--plain-sample", type=parse_sample, default=None, help="size:seed within the draw"
    )
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N events")
    parser.add_argument("--workers", type=int, default=1, help="Calls in flight at once")
    parser.add_argument("--experiment", default="E18-neutral-context")
    args = parser.parse_args(argv)
    if (args.arm == "plain") != (args.plain_sample is not None):
        parser.error("--plain-sample goes with --arm plain, and only with it")
    return args


def read_releases(path: Path) -> dict[str, dict[str, Any]]:
    """Each event's release record, keyed by the 8-K's event id; a repeat stops it."""
    found: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record["event_id"] in found:
                raise EventDataError(f"{record['event_id']} has two releases")
            found[record["event_id"]] = record
    return found


def plain_sample(event_ids: list[str], size: int, seed: int) -> set[str]:
    """The plain arm's events: the ``size`` of ``event_ids`` ranked first by ``seed``."""
    return set(sorted(event_ids, key=lambda event_id: draw_rank(event_id, seed))[:size])


def arm_state(arm: str, event: Any, release: dict[str, Any]) -> dict[str, Any]:
    """What Jev reads in an arm: E16's state, with the release in the context arm."""
    state = event_state(event)
    if arm == "plain":
        return state
    filed = pd.Timestamp(release["release"]["accepted_utc"]).tz_convert("America/New_York")
    return context_state(state, filed.strftime("%Y-%m-%d"), release["release"]["text"])


def baselines(
    events: list, releases: dict[str, dict[str, Any]], negative: frozenset, positive: frozenset
) -> pd.DataFrame:
    rows = []
    for event in events:
        record = releases[event.event_id]
        text = record["release"]["text"]
        tones = [
            tone_shares(body, negative, positive)
            for body in (event.text, text, event.text + "\n\n" + text)
        ]
        rows.append(
            {
                "event_id": event.event_id,
                "company": event.company,
                "draw": record["draw"],
                "stratum": record["stratum"],
                "release_age_days": record["release_age_days"],
                "net_tone_8k": tones[0][0] - tones[0][1],
                "net_tone_release": tones[1][0] - tones[1][1],
                "net_tone_both": tones[2][0] - tones[2][1],
                "release_chars": len(text),
            }
        )
    return pd.DataFrame(rows, columns=list(BASELINE_COLUMNS))


def _chars(state: dict[str, Any], question: Any) -> int:
    return (
        len(json.dumps(state, ensure_ascii=False))
        + len(question.instructions)
        + sum(len(name) + len(text) for name, text in question.options)
    )


def main(argv: list[str] | None = None, *, client: Any = None, sdk_question_for: Any = None) -> int:
    args = parse_args(argv)
    if _sha256(args.lm_dictionary) != LM_SHA256:
        print(f"{args.lm_dictionary.name} is not the frozen dictionary (sha256 {LM_SHA256}).")
        return 1
    question_set = load_question_set(QUESTION_FILES[args.arm])
    question = question_set.questions["rating"]
    try:
        events = load_events(args.events)
        releases = read_releases(args.releases)
    except EventDataError as error:
        print(f"Cannot use the events: {error}. Refusing to go on.")
        return 1
    if {event.event_id for event in events} != set(releases):
        print("The events and the releases do not name the same 8-Ks. Refusing to go on.")
        return 1
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    inputs = {"events": _sha256(args.events), "releases": _sha256(args.releases)}

    if not (args.score or args.collect):
        negative = lm_words(args.lm_dictionary, "Negative")
        positive = lm_words(args.lm_dictionary, "Positive")
        table = baselines(events, releases, negative, positive)
        table.to_csv(out / "baselines.csv", index=False)
        context = load_question_set(QUESTION_FILES["context"]).questions["rating"]
        plain = load_question_set(QUESTION_FILES["plain"]).questions["rating"]
        by_draw: dict[str, int] = {}
        for event in events:
            name = releases[event.event_id]["draw"]
            state = arm_state("context", event, releases[event.event_id])
            by_draw[name] = by_draw.get(name, 0) + _chars(state, context) // CHARS_PER_TOKEN
        plain_tokens = sum(_chars(event_state(e), plain) for e in events) // CHARS_PER_TOKEN
        meta = {
            "experiment": args.experiment,
            "commit": _commit(),
            "events": len(table),
            "events_by_draw_and_stratum": {
                f"{d}|{s}": int(n)
                for (d, s), n in table.groupby(["draw", "stratum"]).size().items()
            },
            "net_tone": "Loughran-McDonald (negative - positive) words / all words",
            "estimated_input_tokens_context_by_draw": by_draw,
            "estimated_cost_usd_context_by_draw": {
                name: round(tokens * USD_PER_INPUT_TOKEN, 2) for name, tokens in by_draw.items()
            },
            "estimated_input_tokens_plain_per_event": plain_tokens // max(1, len(events)),
            "sha256": {
                **inputs,
                "lm_dictionary": _sha256(args.lm_dictionary),
                "baselines.csv": _sha256(out / "baselines.csv"),
            },
        }
        (out / "baselines.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(meta, indent=2))
        print("Baselines frozen. Re-run with --score --draw <name> --limit 1.")
        return 0

    chosen = [e for e in events if args.draw is None or releases[e.event_id]["draw"] == args.draw]
    if args.plain_sample is not None:
        keep = plain_sample([e.event_id for e in chosen], *args.plain_sample)
        chosen = [event for event in chosen if event.event_id in keep]
    if args.limit is not None:
        chosen = chosen[: args.limit]
    if args.collect:
        client = NoJev()
    elif client is None:
        if TypeSafeClient is None:
            print('Scoring needs the TypeSafe SDK: pip install -e ".[filings]"')
            return 1
        if not os.environ.get("TYPESAFE_API_KEY"):
            print("Scoring needs TYPESAFE_API_KEY in the environment.")
            return 1
    items = [(e.event_id, arm_state(args.arm, e, releases[e.event_id])) for e in chosen]
    draw = PLAIN_DRAW if args.arm == "plain" else None

    def sdk_question(criteria: dict[str, str]) -> Any:
        if sdk_question_for is not None:
            return sdk_question_for(criteria)
        return Choice(instructions=question.instructions, criteria=criteria)

    cache_path = out / "answers" / f"{args.arm}_rating.jsonl"
    cache = SharedAnswerCache(cache_path)
    workers = 1 if args.collect else max(1, args.workers)
    # Contiguous slices, so the answers come back in the events' order.
    size = -(-len(items) // workers) if items else 1
    slices = [items[i : i + size] for i in range(0, len(items), size)]

    def ask(part: list) -> list:
        return score_states(
            part,
            question,
            sdk_question_for=sdk_question,
            client=client or TypeSafeClient(model=question_set.model),
            cache=cache,
            model=question_set.model,
            labels=RATING_LABELS,
            draw=draw,
        )

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answered = [item for part in pool.map(ask, slices) for item in part]
    except ChangeGradingError as error:
        print(f"Cannot collect: {error}")
        return 1
    if not args.collect:
        run = {
            "arm": args.arm,
            "draw": args.draw,
            "scored": len(answered),
            "new_calls": sum(not item.cached for item in answered),
            "input_tokens_new": sum(i.input_tokens or 0 for i in answered if not i.cached),
        }
        print(json.dumps(run, indent=2))
        if args.limit is not None:
            print(
                json.dumps(
                    [
                        {"event_id": i.pair_id, "choice": i.choice} | i.probabilities
                        for i in answered
                    ],
                    indent=2,
                )
            )
        return 0

    rows = []
    for event, answer in zip(chosen, answered, strict=True):
        row: dict[str, Any] = {
            "event_id": event.event_id,
            "company": event.company,
            "draw": releases[event.event_id]["draw"],
            "stratum": releases[event.event_id]["stratum"],
            "rating_choice": answer.choice,
        }
        for label in RATING_LABELS:
            row[f"rating_p_{label}"] = answer.probabilities.get(label)
        row["rating_score"] = rating_score(answer.probabilities)
        row["bad_score"] = bad_score(answer.probabilities)
        row["rating_order"] = "|".join(answer.option_order)
        row["input_tokens"] = answer.input_tokens
        rows.append(row)
    scores = pd.DataFrame(rows)
    name = f"{args.arm}_scores" if args.draw is None else f"{args.arm}_{args.draw}_scores"
    scores.to_csv(out / f"{name}.csv", index=False)
    meta = {
        "experiment": args.experiment,
        "commit": _commit(),
        "arm": args.arm,
        "draw": args.draw,
        "plain_sample": list(args.plain_sample) if args.plain_sample else None,
        "model": question_set.model,
        "question_file": QUESTION_FILES[args.arm].name,
        "events": len(scores),
        "rating_score": "sum of p(option) x weight, very_bad -2 ... very_good +2",
        "bad_score": "p_bad + p_very_bad",
        "input_tokens_total": int(scores["input_tokens"].fillna(0).sum()),
        "models_answering": sorted({item.model for item in answered}),
        "choices": scores["rating_choice"].value_counts().to_dict(),
        "sha256": {
            **inputs,
            f"answers/{cache_path.name}": _sha256(cache_path),
            f"{name}.csv": _sha256(out / f"{name}.csv"),
        },
    }
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
