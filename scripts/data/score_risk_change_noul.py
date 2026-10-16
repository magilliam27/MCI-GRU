"""Did this risk materially change? E6's yes/no question, three draws per risk (#298).

Two steps, the first free::

    # 1. Check the pairs and estimate the cost. No Jev calls, no API key.
    python -m scripts.data.score_risk_change_noul \\
        --risk-factors /mnt/project-files/jev_pilot/sec_v2/filings/alphabet_risk_factors.jsonl \\
        --pair-ids /mnt/project-files/filings/experiments/E6-changed-noul/grading/design.csv \\
        --output-dir /mnt/project-files/filings/experiments/E6-changed-noul/jev

    # 2. Score: one test call first, then the full run.
    ... --score --limit 1 --draws 1
    ... --score

Each pair is the current risk beside its closest prior-year risk by TF-IDF, the
same state as E2. Only the ``pair_id`` column of ``--pair-ids`` is read, so a
design file holding anything else gives nothing away. Every draw is its own
cache key, so the cache never serves draw 1's answer as draw 2; a rerun asks
only what is missing.

What it writes under ``--output-dir`` with ``--score``:

- ``answers.jsonl``: every Jev answer, appended and never rewritten.
- After a full run (no ``--limit``): ``scores.csv`` with ``pair_id``,
  ``p_changed_draw1`` and on, ``p_changed_mean``, ``jev_changed`` (mean at least
  0.5), ``word_score`` and ``word_changed`` (at least 0.04, E4's word cut), and
  ``scores.meta.json`` with the commit, cuts, model, counts and digests.
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
    JEV_CHANGED_CUT,
    WORD_SCORE_CUT,
    closest_prior_pairs,
    pair_table,
    word_score,
)
from mci_gru.filings.jev import AnswerCache, load_question_set, pair_state, score_pairs
from scripts.data.score_risk_changes import CHARS_PER_TOKEN, load_risks

try:
    from typesafe_sdk import Noul, TypeSafeClient
except ImportError:  # the [filings] extra is needed only for --score
    Noul = TypeSafeClient = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = REPO_ROOT / "configs" / "filings" / "questions_v3.yaml"
QUESTION_ID = "materially_changed"
#: The tuning thread's frozen rule this question copies (jev_tuning/frozen/
#: changed_noul_v1.yaml in the project folder): same instructions, criteria,
#: three draws, mean, cut 0.5 and word baseline.
TUNING_RULE = {
    "file": "jev_tuning/frozen/changed_noul_v1.yaml",
    "sha256": "a6ef0cd9470bf062640e10815ca8f4ead8e0e105a6cac16fba044f2c332e8637",
}
DEFAULT_DRAWS = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--risk-factors", type=Path, required=True)
    parser.add_argument("--pair-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--score", action="store_true", help="Call Jev (costs money)")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N pairs")
    return parser.parse_args(argv)


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


def main(argv: list[str] | None = None, *, client: Any = None, sdk_question: Any = None) -> int:
    args = parse_args(argv)
    if args.draws < 1:
        print("--draws must be at least 1")
        return 1
    question_set = load_question_set(args.questions)
    question = question_set.questions[QUESTION_ID]
    wanted = pd.read_csv(args.pair_ids, usecols=["pair_id"], dtype=str)["pair_id"].tolist()
    by_id = {pair.pair_id: pair for pair in closest_prior_pairs(load_risks(args.risk_factors))}
    missing = [pair_id for pair_id in wanted if pair_id not in by_id]
    if missing or len(set(wanted)) != len(wanted):
        print(
            f"--pair-ids names {len(missing)} pairs the risk file does not produce "
            f"and {len(wanted) - len(set(wanted))} repeats; refusing to score."
        )
        return 1
    pairs = [by_id[pair_id] for pair_id in wanted]
    words = len(question.instructions) + sum(len(d) for _, d in question.criteria)
    chars = sum(len(json.dumps(pair_state(pair), ensure_ascii=False)) + words for pair in pairs)
    print(
        json.dumps(
            {
                "pairs": len(pairs),
                "draws": args.draws,
                "model": question_set.model,
                "estimated_input_tokens": args.draws * chars // CHARS_PER_TOKEN,
            },
            indent=2,
        )
    )
    if not args.score:
        print("Pairs checked. Re-run with --score --limit 1 --draws 1 for the test call.")
        return 0

    if client is None:
        if TypeSafeClient is None:
            print('Scoring needs the TypeSafe SDK: pip install -e ".[filings]"')
            return 1
        if not os.environ.get("TYPESAFE_API_KEY"):
            print("Scoring needs TYPESAFE_API_KEY in the environment.")
            return 1
        client = TypeSafeClient(model=question_set.model)
    if sdk_question is None:
        sdk_question = Noul(instructions=question.instructions, criteria=dict(question.criteria))

    out = args.output_dir
    cache = AnswerCache(out / "answers.jsonl")
    chosen = pairs if args.limit is None else pairs[: args.limit]
    draws = {}
    for draw in range(1, args.draws + 1):
        draws[draw] = score_pairs(
            chosen,
            question,
            sdk_question=sdk_question,
            client=client,
            cache=cache,
            model=question_set.model,
            draw=draw,
        )
    scored = [item for items in draws.values() for item in items]
    run = {
        "scored": len(scored),
        "new_calls": sum(not item.cached for item in scored),
        "input_tokens_new": sum(item.input_tokens or 0 for item in scored if not item.cached),
    }
    if args.limit is not None:
        answers = [
            {"pair_id": i.pair_id, "draw": i.draw, "p_changed": i.probability} for i in scored
        ]
        print(json.dumps({**run, "answers": answers}, indent=2))
        print("Limited run: no scores written.")
        return 0

    table = pair_table(pairs).set_index("pair_id").loc[wanted]
    scores = pd.DataFrame({"pair_id": wanted})
    for draw, items in draws.items():
        scores[f"p_changed_draw{draw}"] = [item.probability for item in items]
    draw_columns = [f"p_changed_draw{draw}" for draw in draws]
    scores["p_changed_mean"] = scores[draw_columns].mean(axis=1)
    scores["jev_changed"] = scores["p_changed_mean"] >= JEV_CHANGED_CUT
    scores["word_score"] = word_score(table).to_numpy()
    scores["word_changed"] = scores["word_score"] >= WORD_SCORE_CUT
    scores_path = out / "scores.csv"
    scores.to_csv(scores_path, index=False)
    meta = {
        "experiment": "E6-changed-noul",
        "commit": _commit(),
        "model": question_set.model,
        "question_file": args.questions.name,
        "question_id": QUESTION_ID,
        "tuning_rule": TUNING_RULE,
        "draws": args.draws,
        "jev_changed": f"p_changed_mean >= {JEV_CHANGED_CUT}",
        "word_changed": f"word_score >= {WORD_SCORE_CUT}",
        "word_score": "max(words_added, words_removed)",
        "pairs": len(scores),
        "jev_changed_count": int(scores["jev_changed"].sum()),
        "word_changed_count": int(scores["word_changed"].sum()),
        "draw_spread_max": float(
            (scores[draw_columns].max(axis=1) - scores[draw_columns].min(axis=1)).max()
        ),
        **run,
        "sha256": {
            "risk_factors": _sha256(args.risk_factors),
            "pair_ids": _sha256(args.pair_ids),
            "answers": _sha256(out / "answers.jsonl"),
            "scores.csv": _sha256(scores_path),
        },
    }
    (out / "scores.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
