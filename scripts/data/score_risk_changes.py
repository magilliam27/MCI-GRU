"""Ask Jev how each 10-K risk factor changed from last year, and grade it (#289, milestone 2).

Three steps, run in this order so the hand labels stay blind::

    # 1. Pairs, the baseline and the blind label sheet. No Jev calls, no API key.
    python -m scripts.data.score_risk_changes \\
        --risk-factors /content/sec/filings/alphabet_risk_factors.jsonl \\
        --output-dir /content/change

    # 2. Label change_sheet.csv by hand (change: same, broadened, narrowed,
    #    reworked or new; label_guide.md defines each), then score.
    #    Needs `pip install -e ".[filings]"` and TYPESAFE_API_KEY.
    python -m scripts.data.score_risk_changes \\
        --risk-factors /content/sec/filings/alphabet_risk_factors.jsonl \\
        --output-dir /content/change --score [--limit 1]

    # 3. Grade Jev and the baseline against the labels.
    python -m scripts.data.score_risk_changes \\
        --risk-factors /content/sec/filings/alphabet_risk_factors.jsonl \\
        --output-dir /content/change --grade

What it writes under ``--output-dir``:

- ``change_sheet.csv``: a sample of risks stratified by the baseline's answer,
  each beside its closest prior-year risk, with an empty ``change`` column.
  ``label_guide.md`` defines the five options in the frozen question's words.
  These are the only files to open until the labels are done.
- ``sealed/``: anything that could give an answer away.
  - ``pairs.csv``: every current risk with its closest prior-year risk, the
    TF-IDF similarity, the shares of words added and removed, and the
    baseline's answer.
  - ``label_design.csv``: which pair each label row is, its stratum and weight.
  - With ``--score``: ``answers.jsonl`` (the cache, never rewritten) and
    ``choices.csv`` (Jev's choice, confidence and option probabilities).
- With ``--grade``: ``grading/summary.json`` (balanced accuracy of Jev and the
  baseline, the gap with a bootstrap interval, recall and precision per option,
  confusion tables) and ``grading/graded_risks.csv``.

Once a label sheet exists it is kept, and a rerun that no longer produces the
pairs it was drawn from stops. The question set is
``configs/filings/questions_v2.yaml``; it pins the model version, and an answer
from any other version stops the run.
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
    CHANGE_LABELS,
    ChangeGradingError,
    change_label_sheet,
    closest_prior_pairs,
    grade_changes,
    graded_changes,
    pair_table,
    score_choices,
)
from mci_gru.filings.jev import AnswerCache, Question, load_question_set, pair_state
from mci_gru.filings.risk_factors import RiskFactor

try:
    from typesafe_sdk import Choice, TypeSafeClient
except ImportError:  # the [filings] extra is needed only for --score
    Choice = TypeSafeClient = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = REPO_ROOT / "configs" / "filings" / "questions_v2.yaml"
QUESTION_ID = "change_type"
#: Rough characters per token, for the cost estimate printed before scoring.
CHARS_PER_TOKEN = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--risk-factors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--label-sample", type=int, default=80)
    parser.add_argument("--seed", type=int, default=289)
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--score", action="store_true", help="Call Jev (costs money)")
    step.add_argument("--grade", action="store_true", help="Grade against the labels")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N pairs")
    return parser.parse_args(argv)


def load_risks(path: Path) -> list[RiskFactor]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [RiskFactor.from_dict(json.loads(line)) for line in lines if line.strip()]


def label_guide(question: Question) -> str:
    lines = [
        "# Labelling guide: how did this risk change?",
        "",
        "Each row of change_sheet.csv is one risk factor from a 10-K (current) beside "
        "the most similar risk factor in the previous year's 10-K (prior). Read both "
        "and write one word in the `change` column. This is Jev's question, word for word:",
        "",
        f"> {question.instructions}",
        "",
    ]
    lines += [f"- **{name}**: {description}" for name, description in question.options]
    return "\n".join(lines) + "\n"


def _grade(out: Path, seed: int) -> int:
    sealed = out / "sealed"
    try:
        table = graded_changes(
            pd.read_csv(out / "change_sheet.csv", dtype=str, encoding="utf-8-sig"),
            pd.read_csv(sealed / "label_design.csv"),
            pd.read_csv(sealed / "pairs.csv"),
            pd.read_csv(sealed / "choices.csv"),
        )
    except (ChangeGradingError, FileNotFoundError) as error:
        print(f"Cannot grade: {error}")
        return 1
    summary = grade_changes(table, seed=seed)
    grading = out / "grading"
    grading.mkdir(parents=True, exist_ok=True)
    (grading / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table["jev_disagrees"] = table["jev"] != table["label"]
    table.sort_values(["jev_disagrees", "label_id"], ascending=[False, True]).to_csv(
        grading / "graded_risks.csv", index=False
    )
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None, *, client: Any = None, sdk_question_for: Any = None) -> int:
    args = parse_args(argv)
    if args.grade:
        return _grade(args.output_dir, args.seed)
    question_set = load_question_set(args.questions)
    question = question_set.questions[QUESTION_ID]
    if sorted(name for name, _ in question.options) != sorted(CHANGE_LABELS):
        print(f"{QUESTION_ID} must offer exactly: {', '.join(CHANGE_LABELS)}")
        return 1
    pairs = closest_prior_pairs(load_risks(args.risk_factors))
    table = pair_table(pairs)

    out = args.output_dir
    sealed = out / "sealed"
    sealed.mkdir(parents=True, exist_ok=True)
    sheet_path = out / "change_sheet.csv"
    design_path = sealed / "label_design.csv"
    if sheet_path.exists():
        drawn = set(pd.read_csv(design_path)["pair_id"]) if design_path.exists() else None
        missing = None if drawn is None else drawn - set(table["pair_id"])
        if missing is None or missing:
            print(
                f"{sheet_path.name} was drawn from pairs this run does not produce "
                f"({'no design file' if missing is None else f'{len(missing)} missing'}). "
                "Re-run with the original risk file, or label a new output dir."
            )
            return 1
        print(f"Keeping the existing {sheet_path.name}; it may already hold labels.")
    else:
        sheet, design = change_label_sheet(table, n=args.label_sample, seed=args.seed)
        sheet.to_csv(sheet_path, index=False)
        design.to_csv(design_path, index=False)
        (out / "label_guide.md").write_text(label_guide(question), encoding="utf-8")
    table.to_csv(sealed / "pairs.csv", index=False)

    words = len(question.instructions) + sum(len(n) + len(d) for n, d in question.options)
    chars = sum(len(json.dumps(pair_state(pair), ensure_ascii=False)) + words for pair in pairs)
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
    if sdk_question_for is None:

        def sdk_question_for(criteria: dict[str, str]) -> Any:
            return Choice(instructions=question.instructions, criteria=criteria)

    chosen = pairs if args.limit is None else pairs[: args.limit]
    scored = score_choices(
        chosen,
        question,
        sdk_question_for=sdk_question_for,
        client=client,
        cache=AnswerCache(sealed / "answers.jsonl"),
        model=question_set.model,
    )
    rows = [
        {
            **{k: v for k, v in vars(item).items() if k not in ("probabilities", "option_order")},
            "option_order": "|".join(item.option_order),
            **{f"p_{name}": item.probabilities.get(name) for name in CHANGE_LABELS},
        }
        for item in scored
    ]
    pd.DataFrame(rows).to_csv(sealed / "choices.csv", index=False)
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
