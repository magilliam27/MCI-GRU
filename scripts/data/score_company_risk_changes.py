"""E5: the frozen five-way change question on many companies' 10-K risks (#299).

Each company's risk file is paired on its own, so a risk is only ever compared
with its own company's prior-year risks (the pairing groups by fiscal year
alone). The question is E2's, unchanged: ``configs/filings/questions_v2.yaml``,
each current risk against its closest prior-year risk, options shuffled per
pair. Three steps::

    # 1. Pair, check against the lead's list, and freeze the word baselines.
    #    No Jev calls, no API key.
    python -m scripts.data.score_company_risk_changes \\
        --data-dir /mnt/project-files/filings/data/ten_companies_10k/filings \\
        --population /mnt/project-files/filings/experiments/E5-ten-companies/population_pairs.csv \\
        --output-dir /mnt/project-files/filings/experiments/E5-ten-companies/jev

    # 2. Score: one test call, then every pair. --company scores one company,
    #    so companies can run side by side; each has its own answer cache.
    ... --score --company ba --limit 1
    ... --score [--company ba]

    # 3. Collect every answer into scores.csv. Never calls Jev: a pair with no
    #    answer stops it.
    ... --collect

What it writes under ``--output-dir``:

- ``baselines.csv``: per pair, its company, fiscal year, TF-IDF similarity,
  ``words_added``, ``words_removed`` and ``word_score`` (the larger of the two),
  with ``baselines.meta.json`` (commit, inputs and their digests).
- ``answers/<company>.jsonl``: every Jev answer, appended and never rewritten.
- With ``--collect``: ``scores.csv`` (pair id, company, Jev's choice and its
  probability for each of the five options, the option order shown) and
  ``scores.meta.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.change import (
    CHANGE_LABELS,
    ChangeGradingError,
    closest_prior_pairs,
    pair_table,
    score_choices,
    word_score,
)
from mci_gru.filings.jev import AnswerCache, load_question_set, pair_state
from scripts.data.score_risk_changes import (
    CHARS_PER_TOKEN,
    DEFAULT_QUESTIONS,
    QUESTION_ID,
    load_risks,
)

try:
    from typesafe_sdk import Choice, TypeSafeClient
except ImportError:  # the [filings] extra is needed only for --score
    Choice = TypeSafeClient = None

REPO_ROOT = Path(__file__).resolve().parents[2]
RISK_FILE_SUFFIX = "_risk_factors.jsonl"
BASELINE_COLUMNS = (
    "company",
    "pair_id",
    "fiscal_year",
    "similarity",
    "words_added",
    "words_removed",
    "word_score",
)


class NoJev:
    """Stands in for the Jev client when collecting: any call means an answer is missing."""

    def system_one(self, **_: Any) -> Any:
        raise ChangeGradingError("a pair has no answer yet; score it before collecting")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--score", action="store_true", help="Call Jev (costs money)")
    step.add_argument("--collect", action="store_true", help="Write scores.csv from the caches")
    parser.add_argument("--company", default=None, help="Score this company only")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N pairs")
    return parser.parse_args(argv)


def company_files(data_dir: Path) -> dict[str, Path]:
    """Each company's risk file, keyed by the name before ``_risk_factors.jsonl``."""
    files = sorted(data_dir.glob(f"*{RISK_FILE_SUFFIX}"))
    return {path.name[: -len(RISK_FILE_SUFFIX)]: path for path in files}


def company_pairs(data_dir: Path) -> dict[str, list]:
    """Every company's pairs, each company paired on its own."""
    return {
        key: closest_prior_pairs(load_risks(path)) for key, path in company_files(data_dir).items()
    }


def population_mismatch(pairs: dict[str, list], population: pd.DataFrame) -> str | None:
    """Why the pairs differ from the lead's list, or None when they match exactly."""
    ours = [(key, pair.pair_id) for key, items in pairs.items() for pair in items]
    theirs = list(zip(population["company"], population["pair_id"], strict=True))
    if len(set(ours)) != len(ours) or len(set(theirs)) != len(theirs):
        return "a pair id repeats"
    missing, extra = set(theirs) - set(ours), set(ours) - set(theirs)
    if missing or extra:
        return f"{len(missing)} listed pairs not produced, {len(extra)} produced pairs not listed"
    return None


def baselines(pairs: dict[str, list]) -> pd.DataFrame:
    """The frozen word baselines for every pair, company by company."""
    frames = []
    for key, items in pairs.items():
        table = pair_table(items)
        table["company"] = key
        table["word_score"] = word_score(table)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)[list(BASELINE_COLUMNS)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"sha": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def _inputs(args: argparse.Namespace) -> dict[str, str]:
    files = {f"{key}{RISK_FILE_SUFFIX}": path for key, path in company_files(args.data_dir).items()}
    return {
        "population": _sha256(args.population),
        **{name: _sha256(path) for name, path in files.items()},
    }


def main(argv: list[str] | None = None, *, client: Any = None, sdk_question_for: Any = None) -> int:
    args = parse_args(argv)
    question_set = load_question_set(args.questions)
    question = question_set.questions[QUESTION_ID]
    pairs = company_pairs(args.data_dir)
    if not pairs:
        print(f"No *{RISK_FILE_SUFFIX} files in {args.data_dir}")
        return 1
    population = pd.read_csv(args.population, usecols=["company", "pair_id"], dtype=str)
    mismatch = population_mismatch(pairs, population)
    if mismatch:
        print(f"The pairs do not reproduce {args.population.name}: {mismatch}. Refusing to go on.")
        return 1
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if not (args.score or args.collect):
        table = baselines(pairs)
        table.to_csv(out / "baselines.csv", index=False)
        words = len(question.instructions) + sum(len(n) + len(d) for n, d in question.options)
        chars = sum(
            len(json.dumps(pair_state(pair), ensure_ascii=False)) + words
            for items in pairs.values()
            for pair in items
        )
        meta = {
            "experiment": "E5-ten-companies",
            "commit": _commit(),
            "pairs": len(table),
            "pairs_by_company": {key: len(items) for key, items in pairs.items()},
            "word_score": "max(words_added, words_removed)",
            "similarity": "TF-IDF cosine, closest_prior_pairs, each company paired on its own",
            "estimated_input_tokens": chars // CHARS_PER_TOKEN,
            "sha256": {**_inputs(args), "baselines.csv": _sha256(out / "baselines.csv")},
        }
        (out / "baselines.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(meta, indent=2))
        print("Baselines frozen. Re-run with --score --company <key> --limit 1 for a test call.")
        return 0

    if args.company is not None and args.company not in pairs:
        print(f"No company {args.company!r}; known: {', '.join(pairs)}")
        return 1
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
    if sdk_question_for is None:

        def sdk_question_for(criteria: dict[str, str]) -> Any:
            return Choice(instructions=question.instructions, criteria=criteria)

    keys = list(pairs) if args.company is None else [args.company]
    scored = {}
    try:
        for key in keys:
            chosen = pairs[key] if args.limit is None else pairs[key][: args.limit]
            scored[key] = score_choices(
                chosen,
                question,
                sdk_question_for=sdk_question_for,
                client=client,
                cache=AnswerCache(out / "answers" / f"{key}.jsonl"),
                model=question_set.model,
            )
    except ChangeGradingError as error:
        print(f"Cannot collect: {error}")
        return 1
    items = [(key, item) for key, scored_items in scored.items() for item in scored_items]
    run = {
        "scored": len(items),
        "new_calls": sum(not item.cached for _, item in items),
        "input_tokens_new": sum(item.input_tokens or 0 for _, item in items if not item.cached),
    }
    if not args.collect:
        print(json.dumps({"companies": keys, **run}, indent=2))
        if args.limit is not None:
            print(
                json.dumps(
                    [
                        {"pair_id": i.pair_id, "choice": i.choice, **i.probabilities}
                        for _, i in items
                    ],
                    indent=2,
                )
            )
        return 0

    scores = pd.DataFrame(
        [
            {
                "company": key,
                "pair_id": item.pair_id,
                "choice": item.choice,
                **{f"p_{name}": item.probabilities.get(name) for name in CHANGE_LABELS},
                "option_order": "|".join(item.option_order),
            }
            for key, item in items
        ]
    )
    scores.to_csv(out / "scores.csv", index=False)
    records = [
        json.loads(line)
        for key in keys
        for line in (out / "answers" / f"{key}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = {
        "experiment": "E5-ten-companies",
        "commit": _commit(),
        "model": question_set.model,
        "question_file": args.questions.name,
        "question_id": QUESTION_ID,
        "pairs": len(scores),
        "distinct_calls": len(records),
        "input_tokens_total": sum(record.get("input_tokens") or 0 for record in records),
        "models_answering": sorted({record["model"] for record in records}),
        "choices": scores["choice"].value_counts().to_dict(),
        "sha256": {
            **_inputs(args),
            **{f"answers/{key}.jsonl": _sha256(out / "answers" / f"{key}.jsonl") for key in keys},
            "scores.csv": _sha256(out / "scores.csv"),
        },
    }
    (out / "scores.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
