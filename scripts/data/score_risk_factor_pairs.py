"""Pair each 10-K risk factor with last year's, then ask Jev whether each pair is the same risk (#289).

Two steps, run in this order so the hand labels stay blind::

    # 1. Pairs and the blind label sheet. No Jev calls, no API key needed.
    python -m scripts.data.score_risk_factor_pairs \\
        --risk-factors /content/sec_risk_factors/filings/alphabet_risk_factors.jsonl \\
        --output-dir /content/risk_pairs

    # 2. Label label_sheet.csv by hand (same_risk: yes or no), then score.
    #    Needs `pip install -e ".[filings]"` and TYPESAFE_API_KEY.
    python -m scripts.data.score_risk_factor_pairs \\
        --risk-factors /content/sec_risk_factors/filings/alphabet_risk_factors.jsonl \\
        --output-dir /content/risk_pairs --score [--limit 20]

    # Make the first paid run --limit 1: it shows the model ID the API reports,
    # and an ID other than the pinned one stops the run after that one call.

What it writes under ``--output-dir``:

- ``label_sheet.csv``: a stratified sample of pairs with an empty ``same_risk``
  column and nothing that hints at an answer. This is the only file to open
  until the labels are done.
- ``sealed/``: everything that could give an answer away, so it stays shut while
  labelling.
  - ``pairs.csv``: every current risk with its ``--top-k`` most similar
    prior-year risks (TF-IDF), with rank and similarity.
  - ``label_design.csv``: which pair each label row is, its stratum and weight.
  - With ``--score``: ``answers.jsonl`` (the cache: every request and answer,
    never rewritten) and ``scores.csv``.

Once a label sheet exists it is kept, and a rerun that would no longer produce
the pairs it was drawn from (a different ``--top-k`` or risk file) stops.

The question set is ``configs/filings/questions_v1.yaml``. It pins the model
version; an answer from any other version stops the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.jev import (
    AnswerCache,
    blind_label_sheet,
    load_question_set,
    pair_state,
    score_pairs,
)
from mci_gru.filings.pairing import DEFAULT_SHORTLIST, pair_consecutive_years
from mci_gru.filings.risk_factors import RiskFactor

try:
    from typesafe_sdk import Noul, TypeSafeClient
except ImportError:  # the [filings] extra is needed only for --score
    Noul = TypeSafeClient = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = REPO_ROOT / "configs" / "filings" / "questions_v1.yaml"
QUESTION_ID = "same_risk"
#: Rough characters per token, for the cost estimate printed before scoring.
CHARS_PER_TOKEN = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--risk-factors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_SHORTLIST)
    parser.add_argument("--label-sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=289)
    parser.add_argument("--score", action="store_true", help="Call Jev (costs money)")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N pairs")
    return parser.parse_args(argv)


def load_risks(path: Path) -> list[RiskFactor]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [RiskFactor.from_dict(json.loads(line)) for line in lines if line.strip()]


def main(argv: list[str] | None = None, *, client: Any = None, sdk_question: Any = None) -> int:
    args = parse_args(argv)
    question_set = load_question_set(args.questions)
    question = question_set.questions[QUESTION_ID]
    pairs = pair_consecutive_years(load_risks(args.risk_factors), k=args.top_k)

    out = args.output_dir
    sealed = out / "sealed"
    sealed.mkdir(parents=True, exist_ok=True)
    sheet_path = out / "label_sheet.csv"
    design_path = sealed / "label_design.csv"
    if sheet_path.exists():
        drawn = set(pd.read_csv(design_path)["pair_id"]) if design_path.exists() else None
        missing = None if drawn is None else drawn - {pair.pair_id for pair in pairs}
        if missing is None or missing:
            print(
                f"{sheet_path.name} was drawn from pairs this run does not produce "
                f"({'no design file' if missing is None else f'{len(missing)} missing'}). "
                "Re-run with the original --top-k and risk file, or label a new output dir."
            )
            return 1
        print(f"Keeping the existing {sheet_path.name}; it may already hold labels.")
    else:
        sheet, design = blind_label_sheet(pairs, n=args.label_sample, seed=args.seed)
        sheet.to_csv(sheet_path, index=False)
        design.to_csv(design_path, index=False)
    pd.DataFrame([pair.to_dict() for pair in pairs]).to_csv(sealed / "pairs.csv", index=False)

    chars = sum(
        len(json.dumps(pair_state(pair), ensure_ascii=False)) + len(question.instructions)
        for pair in pairs
    )
    print(
        json.dumps(
            {
                "pairs": len(pairs),
                "fiscal_years": sorted({pair.current.fiscal_year for pair in pairs}),
                "model": question_set.model,
                "estimated_input_tokens": chars // CHARS_PER_TOKEN,
            },
            indent=2,
        )
    )
    if not args.score:
        print("Pairs and label sheet written. Label the sheet, then re-run with --score.")
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
        sdk_question = Noul(instructions=question.instructions)

    chosen = pairs if args.limit is None else pairs[: args.limit]
    scored = score_pairs(
        chosen,
        question,
        sdk_question=sdk_question,
        client=client,
        cache=AnswerCache(sealed / "answers.jsonl"),
        model=question_set.model,
    )
    pd.DataFrame([vars(item) for item in scored]).to_csv(sealed / "scores.csv", index=False)
    print(
        json.dumps(
            {
                "scored": len(scored),
                "new_calls": sum(not item.cached for item in scored),
                "input_tokens_new": sum(
                    item.input_tokens or 0 for item in scored if not item.cached
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
