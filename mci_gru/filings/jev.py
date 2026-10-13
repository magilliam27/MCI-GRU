"""The frozen question set, the answer cache and the blind label sheet (#289).

What this module decides:

- **Which model answers.** A question set names a versioned Jev ID
  (``jev-1.13.0``); moving aliases such as ``jev-latest`` are refused, and an
  answer reporting any other model stops the run before it is cached. A new
  model version therefore gets its own cache entries and its own grading pass.
- **When Jev is asked.** The cache key is a hash of the state, the question's
  wording and the model. A key already in the cache is never sent again, so a
  rerun replays its answers instead of drawing new ones.
- **What a labeller sees.** The blind label sheet shows the two risk factors and
  an empty answer column: no Jev probability, no shortlist score or rank, and no
  identifiers that lead back to them. The sampling design and its weights go to
  a separate file.

The SDK itself is not imported here. ``scripts/data/score_risk_factor_pairs.py``
builds the SDK question objects and passes the client in.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mci_gru.filings.pairing import RiskPair

#: Milestone 1 of #289 asks one Noul; milestone 2 asks one Choice.
QUESTION_KINDS = frozenset({"noul", "choice"})
_VERSIONED_MODEL = re.compile(r"^jev-\d+\.\d+\.\d+$")
LABEL_COLUMNS = (
    "label_id",
    "current_heading",
    "current_body",
    "prior_heading",
    "prior_body",
    "same_risk",
)


class QuestionSetError(ValueError):
    """A question set cannot be used as written."""


class ModelDriftError(RuntimeError):
    """An answer came from a model other than the pinned one."""


@dataclass(frozen=True)
class Question:
    """One question as the model receives it."""

    question_id: str
    kind: str
    instructions: str
    #: A Choice's options as (name, description), in the order the file gives.
    options: tuple[tuple[str, str], ...] = ()

    def spec(self) -> dict[str, Any]:
        """The wording that defines the question, as hashed into the cache key.

        A Noul's spec has no options key, so milestone 1's cache keys are unchanged.
        """
        spec: dict[str, Any] = {
            "question_id": self.question_id,
            "kind": self.kind,
            "instructions": self.instructions,
        }
        if self.kind == "choice":
            spec["options"] = dict(self.options)
        return spec


@dataclass(frozen=True)
class QuestionSet:
    version: int
    model: str
    questions: dict[str, Question]


def load_question_set(path: Path) -> QuestionSet:
    """A frozen question set, validated."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise QuestionSetError(f"{path}: not a mapping")
    model = str(raw.get("model", ""))
    if not _VERSIONED_MODEL.match(model):
        raise QuestionSetError(f"{path}: model {model!r} is not a versioned Jev ID")
    version = raw.get("version")
    if not isinstance(version, int):
        raise QuestionSetError(f"{path}: version must be an integer")
    questions: dict[str, Question] = {}
    for question_id, entry in (raw.get("questions") or {}).items():
        kind = str(entry.get("kind", ""))
        if kind not in QUESTION_KINDS:
            raise QuestionSetError(f"{path}: {question_id} has unsupported kind {kind!r}")
        if not str(entry.get("instructions", "")).strip():
            raise QuestionSetError(f"{path}: {question_id} has no instructions")
        options: tuple[tuple[str, str], ...] = ()
        if kind == "choice":
            raw_options = entry.get("options") or {}
            options = tuple((str(k), str(v or "").strip()) for k, v in raw_options.items())
            if len(options) < 2 or not all(description for _, description in options):
                raise QuestionSetError(
                    f"{path}: {question_id} needs at least two options, each described"
                )
        questions[str(question_id)] = Question(
            question_id=str(question_id),
            kind=kind,
            instructions=str(entry.get("instructions", "")).strip(),
            options=options,
        )
    if not questions:
        raise QuestionSetError(f"{path}: no questions")
    return QuestionSet(version=version, model=model, questions=questions)


def pair_state(pair: RiskPair) -> dict[str, Any]:
    """The state a pair question is asked against: both risk factors and nothing else."""
    return {
        "current_risk_factor": {"heading": pair.current.heading, "text": pair.current.body},
        "prior_year_risk_factor": {"heading": pair.prior.heading, "text": pair.prior.body},
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(state: Any, question_spec: dict[str, Any], model: str) -> str:
    payload = _canonical({"state": state, "question": question_spec, "model": model})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AnswerCache:
    """Answers on disk as JSON lines, one per call, appended and never rewritten.

    A run cut off mid-write (a Colab disconnect) can leave the last line
    unfinished. That line is set aside in ``<name>.torn`` and dropped, so its pair
    is simply asked again; a damaged line anywhere else is refused.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return
        text = self.path.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        for position, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if position != len(lines) - 1 or text.endswith("\n"):
                    raise
                self._set_aside_torn_tail(text, line)
                break
            self._records[record["key"]] = record

    def _set_aside_torn_tail(self, text: str, line: str) -> None:
        torn = self.path.with_name(self.path.name + ".torn")
        with torn.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self.path.write_text(text[: len(text) - len(line)], encoding="utf-8")

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, key: str) -> dict[str, Any]:
        return self._records[key]

    def put(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            # ASCII on disk, so a torn write cannot split a character.
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._records[record["key"]] = record


@dataclass(frozen=True)
class ScoredPair:
    pair_id: str
    question_id: str
    probability: float
    model: str
    request_id: str | None
    input_tokens: int | None
    cached: bool


def score_pairs(
    pairs: Sequence[RiskPair],
    question: Question,
    *,
    sdk_question: Any,
    client: Any,
    cache: AnswerCache,
    model: str,
) -> list[ScoredPair]:
    """Ask one Noul of every pair, from the cache where it already holds the answer."""
    if question.kind != "noul":
        raise QuestionSetError(f"{question.question_id}: only Noul questions are scored here")
    scored: list[ScoredPair] = []
    for pair in pairs:
        state = pair_state(pair)
        key = cache_key(state, question.spec(), model)
        cached = key in cache
        if not cached:
            response = client.system_one(
                state=state, questions={question.question_id: sdk_question}
            )
            if response.model != model:
                raise ModelDriftError(f"asked {model}, answered by {response.model}")
            cache.put(
                {
                    "key": key,
                    "pair_id": pair.pair_id,
                    "question": question.spec(),
                    "model": response.model,
                    "probability": float(response.nouls[question.question_id].noul),
                    "request_id": response.request_id,
                    "input_tokens": getattr(response.usage, "input_tokens", None),
                    "state": state,
                }
            )
        record = cache.get(key)
        scored.append(
            ScoredPair(
                pair_id=pair.pair_id,
                question_id=question.question_id,
                probability=record["probability"],
                model=record["model"],
                request_id=record["request_id"],
                input_tokens=record["input_tokens"],
                cached=cached,
            )
        )
    return scored


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Equal allocation across strata, capped by each stratum's size."""
    allocation = dict.fromkeys(sizes, 0)
    remaining = min(n, sum(sizes.values()))
    while remaining:
        open_strata = [key for key in sorted(sizes) if allocation[key] < sizes[key]]
        for key in open_strata[:remaining]:
            allocation[key] += 1
            remaining -= 1
    return allocation


def blind_label_sheet(
    pairs: Sequence[RiskPair], *, n: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A stratified sample for blind labelling, and its design (pair ids, strata, weights).

    Strata are fiscal year by shortlist position (the most similar prior risk, or
    the rest), so both likely-same and likely-different pairs are labelled in
    every year. Each sampled pair's weight is its stratum's size over its
    stratum's sample, so weighted results describe all pairs.
    """
    strata: dict[str, list[RiskPair]] = {}
    for pair in pairs:
        position = "top" if pair.rank == 1 else "rest"
        strata.setdefault(f"{pair.current.fiscal_year}|{position}", []).append(pair)
    allocation = _allocate({key: len(value) for key, value in strata.items()}, n)
    rng = np.random.default_rng(seed)
    chosen: list[tuple[str, RiskPair, float]] = []
    for key in sorted(strata):
        members = strata[key]
        take = allocation[key]
        if not take:
            continue
        picks = rng.choice(len(members), size=take, replace=False)
        chosen.extend((key, members[i], len(members) / take) for i in sorted(picks))
    order = rng.permutation(len(chosen))
    sheet_rows, design_rows = [], []
    for position, index in enumerate(order, start=1):
        stratum, pair, weight = chosen[index]
        label_id = f"L{position:03d}"
        sheet_rows.append(
            {
                "label_id": label_id,
                "current_heading": pair.current.heading,
                "current_body": pair.current.body,
                "prior_heading": pair.prior.heading,
                "prior_body": pair.prior.body,
                "same_risk": "",
            }
        )
        design_rows.append(
            {"label_id": label_id, "pair_id": pair.pair_id, "stratum": stratum, "weight": weight}
        )
    sheet = pd.DataFrame(sheet_rows, columns=list(LABEL_COLUMNS))
    design = pd.DataFrame(design_rows, columns=["label_id", "pair_id", "stratum", "weight"])
    return sheet, design
