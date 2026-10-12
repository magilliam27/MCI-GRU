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
  - ``pairs.csv``: the pairs Jev is asked about, each with its TF-IDF rank and
    similarity. With ``--jev-scope all`` (the default) that is every current
    risk against every prior-year risk; with ``shortlist``, only the ``--top-k``
    most similar. The label sheet is always drawn from the ``--top-k``
    shortlist, which is also the text baseline Jev has to beat.
  - ``label_design.csv``: which pair each label row is, its stratum and weight.
  - With ``--score``: ``answers.jsonl`` (the cache: every request and answer,
    never rewritten), ``scores.csv``, and ``risk_verdicts.csv``: for each
    current risk, Jev's most likely prior-year match and probability beside
    the shortlist's top match.

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
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_SHORTLIST, help="Shortlist size for the label sheet"
    )
    parser.add_argument(
        "--jev-scope",
        choices=("all", "shortlist"),
        default="all",
        help="Ask Jev about every prior-year risk (all) or only the --top-k shortlist",
    )
    parser.add_argument("--label-sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=289)
    parser.add_argument("--score", action="store_true", help="Call Jev (costs money)")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N pairs")
    return parser.parse_args(argv)


def load_risks(path: Path) -> list[RiskFactor]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [RiskFactor.from_dict(json.loads(line)) for line in lines if line.strip()]


def risk_verdicts(pairs: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Per current risk: Jev's most likely prior-year match beside the shortlist's top one.

    ``jev_probability`` is Jev's highest same-risk probability over the prior
    risks it was asked about; a low value means Jev found no prior-year match.
    ``complete`` is false when ``--limit`` left some of the risk's pairs unasked.
    """
    asked = pairs.merge(scores[["pair_id", "probability"]], on="pair_id")
    top_text = pairs.loc[pairs["rank"] == 1].set_index("current_risk_id")["prior_risk_id"]
    rows = []
    for risk_id, group in asked.groupby("current_risk_id", sort=False):
        best = group.loc[group["probability"].idxmax()]
        expected = int((pairs["current_risk_id"] == risk_id).sum())
        rows.append(
            {
                "current_risk_id": risk_id,
                "fiscal_year": int(best["fiscal_year"]),
                "current_heading": best["current_heading"],
                "jev_best_prior_risk_id": best["prior_risk_id"],
                "jev_best_prior_heading": best["prior_heading"],
                "jev_probability": float(best["probability"]),
                "jev_best_text_rank": int(best["rank"]),
                "text_top_prior_risk_id": top_text.get(risk_id),
                "agrees_with_text_top": best["prior_risk_id"] == top_text.get(risk_id),
                "pairs_asked": len(group),
                "complete": len(group) == expected,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None, *, client: Any = None, sdk_question: Any = None) -> int:
    args = parse_args(argv)
    question_set = load_question_set(args.questions)
    question = question_set.questions[QUESTION_ID]
    risks = load_risks(args.risk_factors)
    pairs = pair_consecutive_years(risks, k=args.top_k)
    jev_pairs = pairs if args.jev_scope == "shortlist" else pair_consecutive_years(risks, k=None)

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
    pair_table = pd.DataFrame([pair.to_dict() for pair in jev_pairs])
    pair_table.to_csv(sealed / "pairs.csv", index=False)

    chars = sum(
        len(json.dumps(pair_state(pair), ensure_ascii=False)) + len(question.instructions)
        for pair in jev_pairs
    )
    print(
        json.dumps(
            {
                "jev_scope": args.jev_scope,
                "pairs": len(jev_pairs),
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

    chosen = jev_pairs if args.limit is None else jev_pairs[: args.limit]
    scored = score_pairs(
        chosen,
        question,
        sdk_question=sdk_question,
        client=client,
        cache=AnswerCache(sealed / "answers.jsonl"),
        model=question_set.model,
    )
    scores = pd.DataFrame([vars(item) for item in scored])
    scores.to_csv(sealed / "scores.csv", index=False)
    risk_verdicts(pair_table, scores).to_csv(sealed / "risk_verdicts.csv", index=False)
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
