"""Changed or not: E4's two-way call for every risk, from Jev's E2 answers (#295).

E4 asks whether each Alphabet risk materially changed from its closest
prior-year risk. It spends nothing on Jev: the five-way E2 question was already
asked of every pair, and its probability for ``same`` is the score. The word
score is the larger of E2's shares of words added and removed. Each is cut at
the value frozen in ``mci_gru.filings.change`` before any E4 label existed::

    python -m scripts.data.score_risk_change_flags \\
        --risk-factors /mnt/project-files/jev_pilot/sec_v2/filings/alphabet_risk_factors.jsonl \\
        --answers /mnt/project-files/jev_pilot/change/sealed/answers.jsonl \\
        --output-dir /mnt/project-files/filings/experiments/E4-changed-or-not/jev

Answers are looked up exactly as the E2 run looked them up, by cache key (the
pair's text, the question and its option order, and the model), so two pairs
whose request was identical share one answer. This script has no Jev client:
a pair the answers do not hold stops it. It never writes to ``--answers``.

What it writes under ``--output-dir``:

- ``scores.csv``: ``pair_id``, ``p_same``, ``word_score``, ``jev_changed`` and
  ``word_changed`` for every pair.
- ``scores.meta.json``: the commit it ran from, the cuts, the model, the question file, row counts, and
  SHA-256 digests of both inputs and the output, so the frozen run can be
  checked later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.change import (
    CHANGE_LABELS,
    JEV_SAME_CUT,
    WORD_SCORE_CUT,
    ChangeGradingError,
    change_flags,
    closest_prior_pairs,
    pair_table,
    score_choices,
)
from mci_gru.filings.jev import AnswerCache, load_question_set
from scripts.data.score_risk_changes import DEFAULT_QUESTIONS, QUESTION_ID, load_risks


class NoJev:
    """Stands in for the Jev client: any call means an answer is missing."""

    def system_one(self, **_: Any) -> Any:
        raise ChangeGradingError(
            "a pair has no answer in --answers, and this script never calls Jev"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--risk-factors", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> dict[str, Any]:
    """The repository commit this ran from, and whether the working tree had edits."""
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"sha": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.answers.is_file():
        print(f"No answers file at {args.answers}")
        return 1
    if not args.answers.read_bytes().endswith(b"\n"):
        # AnswerCache would trim an unfinished last line in place; refuse instead.
        print(f"{args.answers} does not end in a complete line; not reading it")
        return 1
    question_set = load_question_set(args.questions)
    question = question_set.questions[QUESTION_ID]
    pairs = closest_prior_pairs(load_risks(args.risk_factors))
    try:
        scored = score_choices(
            pairs,
            question,
            sdk_question_for=lambda criteria: criteria,
            client=NoJev(),
            cache=AnswerCache(args.answers),
            model=question_set.model,
        )
        choices = pd.DataFrame(
            {"pair_id": item.pair_id, "p_same": item.probabilities["same"]} for item in scored
        )
        flags = change_flags(pair_table(pairs), choices)
    except ChangeGradingError as error:
        print(f"Cannot score: {error}")
        return 1

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    scores = out / "scores.csv"
    flags.to_csv(scores, index=False)
    meta = {
        "experiment": "E4-changed-or-not",
        "commit": _commit(),
        "model": question_set.model,
        "question_file": args.questions.name,
        "question_id": QUESTION_ID,
        "options": list(CHANGE_LABELS),
        "jev_changed": f"p_same < {JEV_SAME_CUT}",
        "word_changed": f"word_score >= {WORD_SCORE_CUT}",
        "word_score": "max(words_added, words_removed)",
        "pairs": len(flags),
        "distinct_answers": len({item.request_id for item in scored}),
        "jev_changed_count": int(flags["jev_changed"].sum()),
        "word_changed_count": int(flags["word_changed"].sum()),
        "sha256": {
            "risk_factors": _sha256(args.risk_factors),
            "answers": _sha256(args.answers),
            "scores.csv": _sha256(scores),
        },
    }
    (out / "scores.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
