"""The Labelling desk: blind sheets, the labelling page, three AI reviewers, agreement (#291).

One labels folder per sheet, normally ``filings/experiments/<name>/labels/`` in
the project's shared folder. Steps, in order::

    # 1. A blind sheet from a hand-over table. Only the id column and the item
    #    columns of the label set's kind are read (labelling.ITEM_COLUMNS).
    python -m scripts.data.label_desk sheet --table handover.csv --id-column pair_id \\
        --label-set configs/filings/labels/change_v1.yaml --sheet-id change-alphabet-1 \\
        --out-dir <labels> [--sample 80] [--seed 0]

    # 2. Load page_seed/ into the labelling page's store (ArtifactData batch,
    #    one call per entry of page_seed/batches.json). The owner labels there.

    # 3. Three blind AI reviewers: write their packets, run one fresh subagent
    #    per packet (see .claude/skills/blind-ai-reviewers), then combine.
    python -m scripts.data.label_desk reviewers --labels-dir <labels>
    python -m scripts.data.label_desk combine --labels-dir <labels>

    # 4. The owner's labels, read back from the page's store into a folder
    #    (ArtifactData list with out_dir), then compared with the AI majority.
    python -m scripts.data.label_desk collect --labels-dir <labels> --dump <dir>
    python -m scripts.data.label_desk agree --labels-dir <labels>

What a labels folder holds:

- ``sheet.json``: the sheet's id, label set and its digest, source file hash,
  seed and sample size.
- ``items.csv``: the blind items, shuffled, under fresh ids (R001, ...).
- ``key/source_ids.csv``: each fresh id's source id, for whoever grades.
- ``page_seed/``: the page's store documents and ``batches.json``.
- ``reviewers/reviewer_<k>/batch_<j>/``: ``prompt.md`` and ``items.md`` for one
  subagent, and the ``answers.csv`` it writes.
- ``ai_labels.csv`` and ``ai_summary.json``: each reviewer's category, the
  majority and whether all three agreed.
- ``ai_answers.csv``: the same for each question on its own, for a key that
  reads each question's majority rather than one derived category.
- ``owner_labels.csv`` and ``agreement.json``: the owner's labels and how often
  they match the AI majority.

Nothing here reads a Jev or baseline answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mci_gru.filings.labelling import (
    LabelError,
    LabelSet,
    agreement,
    answer_header,
    blind_sheet,
    combine_answers,
    combine_reviewers,
    items_markdown,
    load_label_set,
    page_documents,
    page_label_rows,
    read_answer_file,
    read_blind_table,
    render_reviewer_prompt,
    reviewer_batches,
    write_page_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = REPO_ROOT / "configs" / "filings" / "labels" / "reviewer_v1.md"
REVIEWERS = 3
BATCH_SIZE = 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_sheet(labels_dir: Path) -> tuple[dict, LabelSet, pd.DataFrame]:
    meta = json.loads((labels_dir / "sheet.json").read_text(encoding="utf-8"))
    path = Path(meta["label_set_path"])
    label_set = load_label_set(path if path.is_absolute() else REPO_ROOT / path)
    if label_set.digest != meta["label_set_digest"]:
        raise LabelError(
            f"{path} changed since this sheet was built; a changed question set is a new file"
        )
    items = pd.read_csv(labels_dir / "items.csv", dtype=str, keep_default_na=False)
    return meta, label_set, items


def cmd_sheet(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    if (out / "items.csv").exists():
        raise LabelError(f"{out / 'items.csv'} exists; a sheet is never rebuilt in place")
    label_set = load_label_set(Path(args.label_set))
    columns = label_set.item_columns
    table = read_blind_table(Path(args.table), args.id_column, columns)
    items, source = blind_sheet(
        table, args.id_column, sample=args.sample, seed=args.seed, columns=columns
    )
    meta = {
        "sheet_id": args.sheet_id,
        "title": args.title or label_set.title,
        "practice": bool(args.practice),
        "label_set_path": _repo_relative(Path(args.label_set)),
        "label_set": label_set.set_id,
        "label_set_digest": label_set.digest,
        "source_path": str(Path(args.table).resolve()),
        "source_sha256": _sha256(Path(args.table)),
        "id_column": args.id_column,
        "seed": args.seed,
        "sample": args.sample,
        "item_count": len(items),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "key").mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "items.csv", index=False)
    source.to_csv(out / "key" / "source_ids.csv", index=False)
    (out / "sheet.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if label_set.items != "risk_pair":
        # The labelling page shows risk pairs only; other items go to AI reviewers.
        print(f"{len(items)} items under {out}; no page seed for {label_set.items} items")
        return
    page_meta = {k: meta[k] for k in ("title", "practice", "created")}
    batches = write_page_seed(
        page_documents(args.sheet_id, label_set, items, page_meta), out / "page_seed"
    )
    print(
        f"{len(items)} items under {out}; page seed in {len(batches)} batch(es): "
        f"{out / 'page_seed' / 'batches.json'}"
    )


def cmd_reviewers(args: argparse.Namespace) -> None:
    labels_dir = Path(args.labels_dir)
    meta, label_set, items = _load_sheet(labels_dir)
    template = Path(args.prompt).read_text(encoding="utf-8")
    batches = reviewer_batches(
        items, reviewers=args.reviewers, batch_size=args.batch_size, seed=meta["seed"]
    )
    root = labels_dir / "reviewers"
    for name, parts in batches.items():
        for j, part in enumerate(parts, start=1):
            folder = root / name / f"batch_{j}"
            if (folder / "answers.csv").exists():
                raise LabelError(f"{folder / 'answers.csv'} exists; reviewers are never re-run")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "items.md").write_text(
                items_markdown(part, label_set.items), encoding="utf-8"
            )
            prompt = render_reviewer_prompt(
                template,
                label_set,
                items_path=(folder / "items.md").absolute(),
                answers_path=(folder / "answers.csv").absolute(),
                item_count=len(part),
                example_id=str(part["item_id"].iloc[0]),
            )
            (folder / "prompt.md").write_text(prompt, encoding="utf-8")
    run = {
        "reviewers": args.reviewers,
        "batch_size": args.batch_size,
        "prompt_path": _repo_relative(Path(args.prompt)),
        "prompt_sha256": _sha256(Path(args.prompt)),
        "label_set_digest": label_set.digest,
        "answer_header": answer_header(label_set),
    }
    (root / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    packets = sorted(str(p) for p in root.glob("reviewer_*/batch_*/prompt.md"))
    print(f"{len(packets)} packets; give each to its own fresh subagent:")
    for packet in packets:
        print(f"  {packet}")


def cmd_combine(args: argparse.Namespace) -> None:
    labels_dir = Path(args.labels_dir)
    _, label_set, items = _load_sheet(labels_dir)
    root = labels_dir / "reviewers"
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    reviews = {}
    for k in range(1, run["reviewers"] + 1):
        name = f"reviewer_{k}"
        files = sorted((root / name).glob("batch_*/answers.csv"))
        expected = len(list((root / name).glob("batch_*/prompt.md")))
        if len(files) != expected:
            raise LabelError(f"{name}: {len(files)} of {expected} answer files written")
        reviews[name] = pd.concat([read_answer_file(label_set, f) for f in files])
    table = combine_reviewers(items["item_id"], reviews).drop(columns="by_rule")
    table.to_csv(labels_dir / "ai_labels.csv", index=False)
    answers = combine_answers(label_set, items["item_id"], reviews)
    answers.to_csv(labels_dir / "ai_answers.csv", index=False)
    summary = {
        "label_set": label_set.set_id,
        "items": len(table),
        "unanimous": int(table["unanimous"].sum()),
        "majority_only": int(((table["ai_label"] != "") & ~table["unanimous"]).sum()),
        "no_majority": table.loc[table["ai_label"] == "", "item_id"].tolist(),
        "ai_label_counts": table["ai_label"].value_counts().to_dict(),
        "answers": {
            qid: {
                "counts": answers[qid].value_counts().to_dict(),
                "unanimous": int(answers[f"{qid}_unanimous"].sum()),
                "empty": answers.loc[answers[qid] == "", "item_id"].tolist(),
                **(
                    {"by_rule": answers.loc[answers[f"{qid}_by_rule"], "item_id"].tolist()}
                    if f"{qid}_by_rule" in answers
                    else {}
                ),
            }
            for qid in label_set.question_ids
        },
        "prompt_sha256": run["prompt_sha256"],
    }
    (labels_dir / "ai_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_collect(args: argparse.Namespace) -> None:
    labels_dir = Path(args.labels_dir)
    meta, label_set, items = _load_sheet(labels_dir)
    docs = []
    for path in sorted(Path(args.dump).rglob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.setdefault("id", path.stem)
        docs.append(doc)
    rows = page_label_rows(label_set, docs)
    stray = sorted(set(rows["item_id"]) - set(items["item_id"]))
    if stray:
        raise LabelError(f"labels for items not on sheet {meta['sheet_id']}: {stray[:5]}")
    rows.to_csv(labels_dir / "owner_labels.csv", index=False)
    done = int((rows["category"] != "").sum())
    print(f"{done} finished labels, {len(rows) - done} unfinished, of {len(items)} items")


def cmd_agree(args: argparse.Namespace) -> None:
    labels_dir = Path(args.labels_dir)
    _, label_set, _ = _load_sheet(labels_dir)
    owner = pd.read_csv(labels_dir / "owner_labels.csv", dtype=str, keep_default_na=False)
    ai = pd.read_csv(labels_dir / "ai_labels.csv", dtype=str, keep_default_na=False)
    result = agreement(
        label_set,
        dict(zip(owner["item_id"], owner["category"], strict=True)),
        dict(zip(ai["item_id"], ai["ai_label"], strict=True)),
    )
    (labels_dir / "agreement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("items_compared", "agreed", "agreement")}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sheet = sub.add_parser("sheet", help="Build a blind sheet and the page seed")
    sheet.add_argument("--table", required=True)
    sheet.add_argument("--id-column", required=True)
    sheet.add_argument("--label-set", required=True)
    sheet.add_argument("--sheet-id", required=True)
    sheet.add_argument("--out-dir", required=True)
    sheet.add_argument("--title")
    sheet.add_argument("--sample", type=int)
    sheet.add_argument("--seed", type=int, default=0)
    sheet.add_argument("--practice", action="store_true", help="Labels here are never graded")
    sheet.set_defaults(func=cmd_sheet)

    reviewers = sub.add_parser("reviewers", help="Write the AI reviewers' packets")
    reviewers.add_argument("--labels-dir", required=True)
    reviewers.add_argument("--reviewers", type=int, default=REVIEWERS)
    reviewers.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    reviewers.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    reviewers.set_defaults(func=cmd_reviewers)

    for name, func, text in (
        ("combine", cmd_combine, "Combine the reviewers' answers by majority"),
        ("agree", cmd_agree, "Compare the owner's labels with the AI majority"),
    ):
        step = sub.add_parser(name, help=text)
        step.add_argument("--labels-dir", required=True)
        step.set_defaults(func=func)

    collect = sub.add_parser("collect", help="Read the owner's labels from a store dump")
    collect.add_argument("--labels-dir", required=True)
    collect.add_argument("--dump", required=True, help="Folder of label documents (JSON)")
    collect.set_defaults(func=cmd_collect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except LabelError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
