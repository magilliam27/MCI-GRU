"""Blind label sheets, yes/no and choice answers, AI reviewers and agreement (#291).

The Labelling desk builds answer keys for filings experiments. Two rules shape
everything here:

- **Labellers answer fixed questions; code derives the category.** A label set
  (``configs/filings/labels/*.yaml``) lists the questions, which ones are asked
  only after a given answer, and the rules that turn a complete set of answers
  into a category. A question is yes/no unless it lists its own ``options``, in
  which case the labeller picks exactly one of them. The labelling page, the AI reviewers and the grading all
  read the same file, so the owner and the reviewers are asked the same thing.
- **The desk never sees an answer it is labelling against.** A sheet is read
  from a hand-over table through an allow-list of columns (``SHEET_COLUMNS``),
  so a Jev or baseline column in that table is never loaded. Item ids are
  minted afresh after a shuffle, so a source id cannot carry a sample stratum.

The three blind AI reviewers are a fixed step: each gets the instructions in
``configs/filings/labels/reviewer_v1.md`` rendered from the same label set, the
same items in its own shuffled order, and writes its own answer file. Their
category labels are combined by majority, and the owner's spot-check labels are
compared with that majority.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import string
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

ANSWERS = ("yes", "no")
#: The only columns a sheet reads from a hand-over table, beside its id column.
SHEET_COLUMNS = (
    "prior_year",
    "prior_heading",
    "prior_body",
    "current_year",
    "current_heading",
    "current_body",
)
NOTE_COLUMN = "note"


class LabelSetError(ValueError):
    """A label set file that cannot be used as written."""


class LabelError(ValueError):
    """An answer file or answer set that does not fit its label set."""


@dataclass(frozen=True)
class LabelQuestion:
    question_id: str
    text: str
    help: str
    #: Asked only when every earlier answer named here was given.
    ask_if: tuple[tuple[str, str], ...]
    #: The answers a labeller may give, each with its definition; yes/no by default.
    options: tuple[tuple[str, str], ...] = tuple((a, "") for a in ANSWERS)

    @property
    def answers(self) -> tuple[str, ...]:
        return tuple(option for option, _ in self.options)

    @property
    def is_choice(self) -> bool:
        return self.answers != ANSWERS


@dataclass(frozen=True)
class CategoryRule:
    label: str
    when: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class LabelSet:
    set_id: str
    title: str
    intro: str
    questions: tuple[LabelQuestion, ...]
    rules: tuple[CategoryRule, ...]
    digest: str

    @property
    def question_ids(self) -> tuple[str, ...]:
        return tuple(q.question_id for q in self.questions)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(rule.label for rule in self.rules))

    def asked(self, answers: Mapping[str, str]) -> tuple[str, ...]:
        """The questions that apply, given the answers so far."""
        return tuple(
            q.question_id
            for q in self.questions
            if all(answers.get(qid) == want for qid, want in q.ask_if)
        )

    def check(self, answers: Mapping[str, Any]) -> dict[str, str]:
        """The answers, cleaned, or a ``LabelError`` naming what is wrong."""
        given = {
            str(k): str(v).strip().lower()
            for k, v in answers.items()
            if v is not None and str(v).strip() != "" and not _is_nan(v)
        }
        unknown = sorted(set(given) - set(self.question_ids))
        if unknown:
            raise LabelError(f"not questions of {self.set_id}: {unknown}")
        allowed = {q.question_id: q.answers for q in self.questions}
        bad = {k: v for k, v in given.items() if v not in allowed[k]}
        if bad:
            raise LabelError(
                f"answers must be one of each question's options (yes or no unless it lists"
                f" its own): {bad}"
            )
        asked = self.asked(given)
        missing = [qid for qid in asked if qid not in given]
        if missing:
            raise LabelError(f"unanswered: {missing}")
        extra = sorted(set(given) - set(asked))
        if extra:
            raise LabelError(f"answered but not asked: {extra}")
        return {qid: given[qid] for qid in asked}

    def category(self, answers: Mapping[str, Any]) -> str:
        """The category a complete answer set derives to."""
        clean = self.check(answers)
        for rule in self.rules:
            if all(clean.get(qid) == want for qid, want in rule.when):
                return rule.label
        raise LabelError(f"no rule matches {clean}")  # load_label_set rules this out

    def page_spec(self) -> dict[str, Any]:
        """What the labelling page needs: the questions only, never the rules."""
        return {
            "label_set": self.set_id,
            "label_set_digest": self.digest,
            "title": self.title,
            "intro": self.intro,
            "questions": [
                {
                    "id": q.question_id,
                    "text": q.text,
                    "help": q.help,
                    "ask_if": dict(q.ask_if),
                    "options": [{"id": o, "text": t} for o, t in q.options],
                }
                for q in self.questions
            ],
        }


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _conditions(
    raw: Any, where: str, questions: Mapping[str, LabelQuestion]
) -> tuple[tuple[str, str], ...]:
    """Conditions on earlier answers, each naming a known question and one of its options."""
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise LabelSetError(f"{where}: conditions must be a mapping")
    out = []
    for qid, want in raw.items():
        question = questions.get(str(qid))
        if question is None:
            raise LabelSetError(f"{where}: {qid!r} is not an earlier question")
        if not isinstance(want, str) or want not in question.answers:
            raise LabelSetError(
                f"{where}: {qid} must be the string of one of {list(question.answers)}"
                f" (quote it in YAML), got {want!r}"
            )
        out.append((str(qid), want))
    return tuple(out)


def _options(raw: Any, where: str) -> tuple[tuple[str, str], ...]:
    """A choice question's options, or yes/no when it lists none."""
    if raw is None:
        return tuple((a, "") for a in ANSWERS)
    if not isinstance(raw, list) or len(raw) < 2:
        raise LabelSetError(f"{where}: options must be a list of at least two")
    out: list[tuple[str, str]] = []
    for entry in raw:
        option = (
            str((entry or {}).get("id") or "").strip().lower() if isinstance(entry, dict) else ""
        )
        if not option or option in {o for o, _ in out}:
            raise LabelSetError(f"{where}: every option needs a distinct id")
        out.append((option, " ".join(str(entry.get("text") or "").split())))
    return tuple(out)


def load_label_set(path: Path) -> LabelSet:
    """A label set, validated so every complete answer set derives a category."""
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise LabelSetError(f"{path}: not a mapping")
    set_id = str(raw.get("id") or "").strip()
    if not set_id:
        raise LabelSetError(f"{path}: no id")
    questions: list[LabelQuestion] = []
    for entry in raw.get("questions") or []:
        qid = str(entry.get("id") or "").strip()
        if not qid or not str(entry.get("text") or "").strip():
            raise LabelSetError(f"{path}: every question needs an id and text")
        if qid in {q.question_id for q in questions} or qid == NOTE_COLUMN:
            raise LabelSetError(f"{path}: question id {qid!r} is repeated or reserved")
        earlier = {q.question_id: q for q in questions}
        ask_if = _conditions(entry.get("ask_if"), f"{path}: {qid}.ask_if", earlier)
        questions.append(
            LabelQuestion(
                question_id=qid,
                text=" ".join(str(entry["text"]).split()),
                help=" ".join(str(entry.get("help") or "").split()),
                ask_if=ask_if,
                options=_options(entry.get("options"), f"{path}: {qid}.options"),
            )
        )
    if not questions:
        raise LabelSetError(f"{path}: no questions")
    known = {q.question_id: q for q in questions}
    rules = []
    for entry in raw.get("categories") or []:
        label = str(entry.get("label") or "").strip()
        if not label:
            raise LabelSetError(f"{path}: every category needs a label")
        when = _conditions(entry.get("when") or {}, f"{path}: category {label}", known)
        rules.append(CategoryRule(label=label, when=when))
    if not rules or rules[-1].when:
        raise LabelSetError(f"{path}: the last category rule must match everything")
    label_set = LabelSet(
        set_id=set_id,
        title=" ".join(str(raw.get("title") or set_id).split()),
        intro=" ".join(str(raw.get("intro") or "").split()),
        questions=tuple(questions),
        rules=tuple(rules),
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    )
    reachable = {label_set.category(a) for a in complete_answer_sets(label_set)}
    unreachable = [label for label in label_set.categories if label not in reachable]
    if unreachable:
        raise LabelSetError(f"{path}: no answers lead to {unreachable}")
    return label_set


def complete_answer_sets(label_set: LabelSet) -> list[dict[str, str]]:
    """Every distinct answer set a labeller can give, skipped questions left out."""
    seen: dict[str, dict[str, str]] = {}
    for combo in itertools.product(*(q.answers for q in label_set.questions)):
        full = dict(zip(label_set.question_ids, combo, strict=True))
        answers: dict[str, str] = {}
        for q in label_set.questions:
            if all(answers.get(dep) == want for dep, want in q.ask_if):
                answers[q.question_id] = full[q.question_id]
        seen[json.dumps(answers, sort_keys=True)] = answers
    return list(seen.values())


# --- Sheets -------------------------------------------------------------------


def read_blind_table(path: Path, id_column: str) -> pd.DataFrame:
    """A hand-over table, reading only its id column and ``SHEET_COLUMNS``."""
    wanted = {id_column, *SHEET_COLUMNS}
    table = pd.read_csv(path, usecols=lambda column: column in wanted, dtype=str)
    missing = sorted(wanted - set(table.columns))
    if missing:
        raise LabelError(f"{path}: missing columns {missing}")
    if table[id_column].isna().any() or table[id_column].duplicated().any():
        raise LabelError(f"{path}: {id_column} must be present and unique on every row")
    return table.fillna("")


def blind_sheet(
    table: pd.DataFrame, id_column: str, *, sample: int | None, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffled items under fresh ids, and the map back to the source ids.

    With ``sample`` a simple random sample is drawn; a stratified sample has to
    be drawn by whoever holds the strata and handed over as the table.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(table))
    if sample is not None:
        if not 0 < sample <= len(table):
            raise LabelError(f"sample must be between 1 and {len(table)}")
        order = order[:sample]
    picked = table.iloc[order].reset_index(drop=True)
    width = max(3, len(str(len(picked))))
    item_ids = [f"R{i + 1:0{width}d}" for i in range(len(picked))]
    items = picked[list(SHEET_COLUMNS)].copy()
    items.insert(0, "item_id", item_ids)
    source = pd.DataFrame({"item_id": item_ids, "source_id": picked[id_column].to_numpy()})
    return items, source


def page_documents(
    sheet_id: str, label_set: LabelSet, items: pd.DataFrame, meta: Mapping[str, Any]
) -> list[tuple[str, str, dict[str, Any]]]:
    """The labelling page's store documents for one sheet: (collection, doc id, body)."""
    sheet_doc = {
        **label_set.page_spec(),
        **{k: v for k, v in meta.items() if k not in {"questions"}},
        "sheet_id": sheet_id,
        "item_count": len(items),
    }
    docs = [("sheets", sheet_id, sheet_doc)]
    for position, row in enumerate(items.itertuples(index=False)):
        record = row._asdict()
        docs.append(
            (
                f"sheets/{sheet_id}/items",
                record["item_id"],
                {"order": position, **{k: str(v) for k, v in record.items()}},
            )
        )
    return docs


def write_page_seed(
    docs: list[tuple[str, str, dict[str, Any]]],
    out_dir: Path,
    *,
    max_entries: int = 50,
    max_bytes: int = 900_000,
) -> list[list[dict[str, str]]]:
    """Each document as a JSON file, grouped into store batches under the size caps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    batches: list[list[dict[str, str]]] = [[]]
    size = 0
    for collection, doc_id, body in docs:
        path = out_dir / f"{collection.replace('/', '__')}__{doc_id}.json"
        payload = json.dumps(body, ensure_ascii=False, indent=1)
        path.write_text(payload, encoding="utf-8")
        weight = len(payload.encode("utf-8"))
        if batches[-1] and (len(batches[-1]) >= max_entries or size + weight > max_bytes):
            batches.append([])
            size = 0
        batches[-1].append(
            {"op": "set", "collection": collection, "doc_id": doc_id, "file_path": str(path)}
        )
        size += weight
    (out_dir / "batches.json").write_text(json.dumps(batches, indent=1), encoding="utf-8")
    return batches


# --- AI reviewers -------------------------------------------------------------


def answer_header(label_set: LabelSet) -> list[str]:
    return ["item_id", *label_set.question_ids, NOTE_COLUMN]


def render_reviewer_prompt(
    template: str,
    label_set: LabelSet,
    *,
    items_path: Path,
    answers_path: Path,
    item_count: int,
    example_id: str,
) -> str:
    lines = []
    for number, q in enumerate(label_set.questions, start=1):
        condition = ""
        if q.ask_if:
            parts = [f"`{qid}` is {want}" for qid, want in q.ask_if]
            condition = f" Answer only when {' and '.join(parts)}; otherwise leave it empty."
        help_text = f" {q.help}" if q.help else ""
        lines.append(f"{number}. `{q.question_id}`: {q.text}{help_text}{condition}")
        if q.is_choice:
            lines.append("   Pick exactly one of these, written as shown:")
            lines.extend(f"   - `{option}`: {text}" for option, text in q.options)
    return string.Template(template).substitute(
        title=label_set.title,
        intro=label_set.intro,
        questions="\n".join(lines),
        header=",".join(answer_header(label_set)),
        items_path=items_path.as_posix(),
        answers_path=answers_path.as_posix(),
        item_count=item_count,
        example_id=example_id,
    )


def items_markdown(items: pd.DataFrame) -> str:
    """Items in full, plain text, nothing marked up between the two years."""
    blocks = []
    for row in items.itertuples(index=False):
        blocks.append(
            "\n".join(
                [
                    f"## Item {row.item_id}",
                    "",
                    f"### Last year (fiscal {row.prior_year})",
                    "",
                    f"**{row.prior_heading.strip()}**",
                    "",
                    row.prior_body.strip(),
                    "",
                    f"### This year (fiscal {row.current_year})",
                    "",
                    f"**{row.current_heading.strip()}**",
                    "",
                    row.current_body.strip(),
                    "",
                ]
            )
        )
    return "\n".join(blocks)


def reviewer_batches(
    items: pd.DataFrame, *, reviewers: int, batch_size: int, seed: int
) -> dict[str, list[pd.DataFrame]]:
    """For each reviewer, the items in its own shuffled order, cut into batches."""
    if reviewers < 1 or batch_size < 1:
        raise LabelError("reviewers and batch_size must be at least 1")
    out = {}
    for k in range(1, reviewers + 1):
        rng = np.random.default_rng([seed, k])
        shuffled = items.iloc[rng.permutation(len(items))].reset_index(drop=True)
        out[f"reviewer_{k}"] = [
            shuffled.iloc[start : start + batch_size]
            for start in range(0, len(shuffled), batch_size)
        ]
    return out


def read_answer_file(label_set: LabelSet, path: Path) -> pd.DataFrame:
    """One answer file, every row checked: item_id, the answers, note, category."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ["item_id", *label_set.question_ids] if c not in table.columns]
    if missing:
        raise LabelError(f"{path}: missing columns {missing}")
    if table["item_id"].duplicated().any():
        dupes = sorted(table.loc[table["item_id"].duplicated(), "item_id"])
        raise LabelError(f"{path}: items answered twice: {dupes}")
    rows = []
    for record in table.to_dict("records"):
        answers = {qid: record[qid] for qid in label_set.question_ids}
        try:
            clean = label_set.check(answers)
        except LabelError as error:
            raise LabelError(f"{path}: item {record['item_id']}: {error}") from error
        rows.append(
            {
                "item_id": record["item_id"].strip(),
                **{qid: clean.get(qid, "") for qid in label_set.question_ids},
                NOTE_COLUMN: record.get(NOTE_COLUMN, ""),
                "category": label_set.category(clean),
            }
        )
    return pd.DataFrame(rows, columns=["item_id", *label_set.question_ids, NOTE_COLUMN, "category"])


def combine_reviewers(
    item_ids: Iterable[str], reviews: Mapping[str, pd.DataFrame], *, column: str = "category"
) -> pd.DataFrame:
    """One row per item: each reviewer's category, the majority and whether all agreed.

    The majority is a category more than half the reviewers gave; with no such
    category ``ai_label`` is empty, and the item needs a human call. With
    ``column`` set to a question id, the same is done for that question's answers.
    """
    ids = list(item_ids)
    table = pd.DataFrame({"item_id": ids})
    for name, review in reviews.items():
        got = review.set_index("item_id")[column]
        absent = [i for i in ids if i not in got.index]
        stray = sorted(set(got.index) - set(ids))
        if absent or stray:
            raise LabelError(f"{name}: missing items {absent[:5]}, unknown items {stray[:5]}")
        table[name] = table["item_id"].map(got)
    names = list(reviews)
    labels, votes = [], []
    for row in table[names].itertuples(index=False):
        counts = pd.Series(list(row)).value_counts()
        top, n = counts.index[0], int(counts.iloc[0])
        winner = top if n * 2 > len(names) else ""
        labels.append(winner)
        votes.append(n)
    table["ai_label"] = labels
    table["votes"] = votes
    table["unanimous"] = table["votes"] == len(names)
    return table


def combine_answers(
    label_set: LabelSet, item_ids: Iterable[str], reviews: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """One row per item: each question's majority answer, its votes and whether all agreed.

    Each question is decided by its own majority, so a key with several
    questions does not have to go through one derived category.
    """
    ids = list(item_ids)
    table = pd.DataFrame({"item_id": ids})
    for qid in label_set.question_ids:
        one = combine_reviewers(ids, reviews, column=qid)
        for name in reviews:
            table[f"{name}_{qid}"] = one[name]
        table[qid] = one["ai_label"]
        table[f"{qid}_votes"] = one["votes"]
        table[f"{qid}_unanimous"] = one["unanimous"]
    return table


# --- The owner's labels and agreement -----------------------------------------


def page_label_rows(label_set: LabelSet, docs: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Label documents the page saved, one row each; an unfinished one has no category."""
    rows = []
    for doc in docs:
        body = doc.get("data", doc) if isinstance(doc.get("data"), dict) else doc
        item_id = str(body.get("item_id") or doc.get("id") or doc.get("doc_id") or "")
        answers = dict(body.get("answers") or {})
        try:
            category = label_set.category(answers)
            problem = ""
        except LabelError as error:
            category, problem = "", str(error)
        rows.append(
            {
                "item_id": item_id,
                **{qid: answers.get(qid, "") for qid in label_set.question_ids},
                "category": category,
                "unfinished": problem,
                "flagged": bool(body.get("flagged", False)),
                NOTE_COLUMN: str(body.get(NOTE_COLUMN) or ""),
                "updated_at": str(body.get("updated_at") or ""),
            }
        )
    columns = [
        "item_id",
        *label_set.question_ids,
        "category",
        "unfinished",
        "flagged",
        NOTE_COLUMN,
        "updated_at",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("item_id").reset_index(drop=True)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return math.nan
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    if expected == 1:
        return 1.0 if observed == 1 else math.nan
    return (observed - expected) / (1 - expected)


def agreement(
    label_set: LabelSet, human: Mapping[str, str], ai: Mapping[str, str]
) -> dict[str, Any]:
    """How often the owner's category matches the AI majority, on items both labelled."""
    shared = sorted(i for i in human if human[i] and ai.get(i))
    a = [human[i] for i in shared]
    b = [ai[i] for i in shared]
    agree = sum(x == y for x, y in zip(a, b, strict=True))
    low, high = wilson_interval(agree, len(shared))
    confusion = {
        h: {
            m: sum(1 for x, y in zip(a, b, strict=True) if x == h and y == m)
            for m in label_set.categories
        }
        for h in label_set.categories
    }
    return {
        "label_set": label_set.set_id,
        "items_compared": len(shared),
        "agreed": agree,
        "agreement": agree / len(shared) if shared else math.nan,
        "agreement_interval_95": [low, high],
        "cohen_kappa": cohen_kappa(a, b),
        "confusion_owner_rows_ai_columns": confusion,
        "disagreements": [
            {"item_id": i, "owner": h, "ai": m}
            for i, h, m in zip(shared, a, b, strict=True)
            if h != m
        ],
    }
