"""E8: is a risk's change bad, good or neutral for the company? (#303).

Two frozen Choice questions (``configs/filings/questions_v4.yaml``), each asked
on its own for every pair the research lead listed: ``direction`` (worse,
better, neutral) and ``change_kind`` (materialized, new_exposure, escalated,
eased, neutral). Each pair is a current risk beside its closest prior-year risk
from the same company, paired exactly as E5 pairs them. Options are shuffled per
pair. Three steps::

    # 1. Find the lead's pairs and freeze the baselines. No Jev calls, no API key.
    python -m scripts.data.score_change_direction \\
        --data-dir /mnt/project-files/filings/data/ten_companies_10k/filings \\
        --pairs /mnt/project-files/filings/experiments/E8-change-direction/pairs.csv \\
        --lm-dictionary <Loughran-McDonald_MasterDictionary_1993-2025.csv> \\
        --output-dir /mnt/project-files/filings/experiments/E8-change-direction/jev

    # 2. Score: one test call, then every pair, one question at a time or both.
    ... --score --question direction --limit 1
    ... --score [--question change_kind]

    # 3. Collect both questions' answers into scores.csv. Never calls Jev: a pair
    #    with no answer stops it.
    ... --collect

The baselines, frozen before any E8 label existed:

- ``neg_change``: the Loughran-McDonald negative-word share of the current text
  minus that of the prior text (more negative words: the risk got worse). The
  dictionary is pinned by its sha256, ``LM_SHA256``; any other file is refused.
- ``word_balance``: the share of distinct words added minus the share removed
  (the risk grew).

What it writes under ``--output-dir``: ``baselines.csv`` with
``baselines.meta.json``; ``answers/<question>.jsonl``, appended and never
rewritten; and with ``--collect``, ``scores.csv`` (each question's choice, its
probability for every option and the option order shown) with
``scores.meta.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.change import (
    DIRECTION_LABELS,
    KIND_LABELS,
    ChangeGradingError,
    lm_negative_words,
    negative_share,
    pair_table,
    score_choices,
)
from mci_gru.filings.jev import AnswerCache, load_question_set, pair_state
from scripts.data.score_company_risk_changes import (
    NoJev,
    _commit,
    _sha256,
    company_files,
    company_pairs,
)
from scripts.data.score_risk_changes import CHARS_PER_TOKEN

try:
    from typesafe_sdk import Choice, TypeSafeClient
except ImportError:  # the [filings] extra is needed only for --score
    Choice = TypeSafeClient = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = REPO_ROOT / "configs" / "filings" / "questions_v4.yaml"
#: Each question and the options it must offer, in the order scores.csv lists them.
QUESTIONS = {"direction": DIRECTION_LABELS, "change_kind": KIND_LABELS}
#: The Loughran-McDonald master dictionary, 1993-2025 release (updated March 2026),
#: from sraf.nd.edu, frozen 2026-10-15 before any E8 label existed.
LM_SHA256 = "147cb24a1687ccde60363a0fa29c62be4097debcc698c01a83e51bb8a176dbb3"
BASELINE_COLUMNS = (
    "company",
    "pair_id",
    "fiscal_year",
    "similarity",
    "words_added",
    "words_removed",
    "word_balance",
    "neg_share_prior",
    "neg_share_current",
    "neg_change",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--lm-dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--score", action="store_true", help="Call Jev (costs money)")
    step.add_argument("--collect", action="store_true", help="Write scores.csv from the caches")
    parser.add_argument("--question", choices=sorted(QUESTIONS), default=None)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N pairs")
    return parser.parse_args(argv)


def listed_pairs(data_dir: Path, listed: pd.DataFrame) -> list[tuple[str, Any]]:
    """The lead's pairs as (company, pair), in the list's order; any gap or repeat stops it."""
    keys = list(zip(listed["company"], listed["pair_id"], strict=True))
    if len(set(keys)) != len(keys):
        raise ChangeGradingError("a listed pair repeats")
    found = {
        (key, pair.pair_id): pair
        for key, items in company_pairs(data_dir).items()
        for pair in items
    }
    missing = [key for key in keys if key not in found]
    if missing:
        raise ChangeGradingError(f"{len(missing)} listed pairs are not produced: {missing[:3]}")
    return [(key[0], found[key]) for key in keys]


def baselines(pairs: list[tuple[str, Any]], negative: frozenset[str]) -> pd.DataFrame:
    """The frozen baselines for every listed pair, in the list's order."""
    table = pair_table([pair for _, pair in pairs])
    table["company"] = [key for key, _ in pairs]
    table["word_balance"] = table["words_added"] - table["words_removed"]
    table["neg_share_prior"] = [negative_share(p.prior.text, negative) for _, p in pairs]
    table["neg_share_current"] = [negative_share(p.current.text, negative) for _, p in pairs]
    table["neg_change"] = table["neg_share_current"] - table["neg_share_prior"]
    return table[list(BASELINE_COLUMNS)]


def _inputs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "pairs": _sha256(args.pairs),
        "lm_dictionary": _sha256(args.lm_dictionary),
        **{path.name: _sha256(path) for path in company_files(args.data_dir).values()},
    }


def main(argv: list[str] | None = None, *, client: Any = None, sdk_question_for: Any = None) -> int:
    args = parse_args(argv)
    question_set = load_question_set(args.questions)
    if sorted(question_set.questions) != sorted(QUESTIONS):
        print(f"{args.questions.name} must hold exactly {', '.join(QUESTIONS)}.")
        return 1
    if _sha256(args.lm_dictionary) != LM_SHA256:
        print(f"{args.lm_dictionary.name} is not the frozen dictionary (sha256 {LM_SHA256}).")
        return 1
    listed = pd.read_csv(args.pairs, usecols=["company", "pair_id"], dtype=str)
    try:
        pairs = listed_pairs(args.data_dir, listed)
    except ChangeGradingError as error:
        print(f"The pairs do not match {args.pairs.name}: {error}. Refusing to go on.")
        return 1
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if not (args.score or args.collect):
        table = baselines(pairs, lm_negative_words(args.lm_dictionary))
        table.to_csv(out / "baselines.csv", index=False)
        chars = sum(
            len(json.dumps(pair_state(pair), ensure_ascii=False))
            + len(question.instructions)
            + sum(len(n) + len(d) for n, d in question.options)
            for _, pair in pairs
            for question in question_set.questions.values()
        )
        meta = {
            "experiment": "E8-change-direction",
            "commit": _commit(),
            "pairs": len(table),
            "neg_change": "Loughran-McDonald negative-word share, current minus prior",
            "word_balance": "share of distinct words added minus share removed",
            "lm_negative_words": len(lm_negative_words(args.lm_dictionary)),
            "estimated_input_tokens": chars // CHARS_PER_TOKEN,
            "sha256": {**_inputs(args), "baselines.csv": _sha256(out / "baselines.csv")},
        }
        (out / "baselines.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(meta, indent=2))
        print("Baselines frozen. Re-run with --score --question direction --limit 1.")
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
    chosen = [pair for _, pair in pairs]
    if args.limit is not None:
        chosen = chosen[: args.limit]
    names = list(QUESTIONS) if args.question is None or args.collect else [args.question]
    scored = {}
    try:
        for name in names:
            question = question_set.questions[name]

            def sdk_question(criteria: dict[str, str], question=question) -> Any:
                if sdk_question_for is not None:
                    return sdk_question_for(criteria)
                return Choice(instructions=question.instructions, criteria=criteria)

            scored[name] = score_choices(
                chosen,
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
    items = [item for name in names for item in scored[name]]
    if not args.collect:
        run = {
            "questions": names,
            "scored": len(items),
            "new_calls": sum(not item.cached for item in items),
            "input_tokens_new": sum(item.input_tokens or 0 for item in items if not item.cached),
        }
        print(json.dumps(run, indent=2))
        if args.limit is not None:
            print(
                json.dumps(
                    [
                        {"question": i.question_id, "pair_id": i.pair_id, "choice": i.choice}
                        | i.probabilities
                        for i in items
                    ],
                    indent=2,
                )
            )
        return 0

    rows = []
    for (key, pair), *answers in zip(pairs, *(scored[name] for name in names), strict=True):
        row: dict[str, Any] = {"company": key, "pair_id": pair.pair_id}
        for name, answer in zip(names, answers, strict=True):
            row[f"{name}_choice"] = answer.choice
            for label in QUESTIONS[name]:
                row[f"{name}_p_{label}"] = answer.probabilities.get(label)
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
        "experiment": "E8-change-direction",
        "commit": _commit(),
        "model": question_set.model,
        "question_file": args.questions.name,
        "pairs": len(scores),
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
