"""The one way into the locked 2026 holdout: a committed promotion record (#374).

PRD #369, decision 3. A candidate that a declared experiment adopted gets one look at
the holdout, scored together with the incumbent on the same seeds, and the holdout
allows three looks in all. After that it retires, and the next holdout is the data
after 2026-07-31.

A promotion is declared in ``docs/research/holdout/<id>.yaml`` and committed before
its run. The record names the declared experiment, its candidate arm, its incumbent
(the experiment's control) and the commit its verdict ran at. The pass rule is fixed:
the candidate's holdout IC minus the incumbent's, paired by date, must be above zero,
and it is reported with its interval.

``protocol.holdout_promotion=<id> protocol.arm=<candidate or incumbent>`` makes the
runner load that record at the run's commit and refuse unless:

- the record and every other tracked file match the commit (as for a declared run);
- the committed holdout ledger, ``docs/research/holdout/ledger.yaml``, does not
  already hold the id and holds fewer than three looks;
- the run's code is the verdict's commit: the two commits differ only under
  ``docs/research/holdout/``;
- the incumbent is the experiment's control and the candidate one of its treatment
  arms;
- the run is exactly the named arm, at the experiment's seed base, on the holdout
  window's six dates.

Recording the look is a commit to the ledger, made from the receipt that
``first_run_colab.py compare --promotion`` writes.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from mci_gru.evaluation.verdict import treatment_arms
from mci_gru.preregistration import (
    LoadedPreregistration,
    PreregistrationError,
    arm_overrides,
    config_differences,
    foreign_config_sources,
    preregistration_at_commit,
    read_committed,
    recipe_overrides,
    recipe_overrides_sha256,
    recipe_slug,
    require_committed_checkout,
)
from mci_gru.protocol import HOLDOUT_WINDOW

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

HOLDOUT_DIR = Path("docs") / "research" / "holdout"
LEDGER_PATH = HOLDOUT_DIR / "ledger.yaml"
#: Decision 3: three looks in total, one per candidate.
MAX_LOOKS = 3
#: The one pass rule (#374): holdout IC(candidate) - IC(incumbent), paired, above zero.
PASS_RULE = "holdout_paired_ic_difference_positive"
HOLDOUT_WINDOW_NAME = HOLDOUT_WINDOW.name
#: Where the looks already taken are recorded for everyone: the ledger on main.
CANONICAL_REF = "origin/main"

_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_FIELDS = ("id", "experiment_id", "candidate", "incumbent", "commit", "pass_rule")
_LOOK_FIELDS = (
    "id",
    "experiment_id",
    "candidate",
    "commit",
    "record_sha256",
    "receipt_sha256",
    "passed",
)


class PromotionError(PreregistrationError):
    """A holdout promotion the protocol does not allow."""


@dataclass(frozen=True)
class PromotionRecord:
    """One committed promotion: which adopted arm meets which incumbent on the holdout."""

    id: str
    experiment_id: str
    candidate: str
    incumbent: str
    commit: str
    pass_rule: str

    @classmethod
    def from_mapping(cls, data: Any, *, expected_id: str) -> PromotionRecord:
        if not isinstance(data, dict):
            raise PromotionError("a promotion record is a mapping of its fields")
        missing = [name for name in _RECORD_FIELDS if name not in data]
        unknown = sorted(set(data) - set(_RECORD_FIELDS))
        if missing or unknown:
            raise PromotionError(
                f"the promotion record is missing {missing} and has unknown keys {unknown}"
            )
        if data["id"] != expected_id:
            raise PromotionError(f"the record for {expected_id!r} declares id {data['id']!r}")
        for name in ("experiment_id", "candidate", "incumbent"):
            if not isinstance(data[name], str) or not data[name]:
                raise PromotionError(f"{name} must be a non-empty string")
        if not isinstance(data["commit"], str) or not _SHA.fullmatch(data["commit"]):
            raise PromotionError(f"commit {data['commit']!r} must be a full 40-character id")
        if data["pass_rule"] != PASS_RULE:
            raise PromotionError(
                f"pass_rule is fixed at {PASS_RULE!r} (#374), got {data['pass_rule']!r}"
            )
        if data["candidate"] == data["incumbent"]:
            raise PromotionError("the candidate and the incumbent are the same arm")
        return cls(**{name: data[name] for name in _RECORD_FIELDS})


@dataclass(frozen=True)
class LoadedPromotion:
    """A record the runner may act on, where it was read, and what it points at."""

    record: PromotionRecord
    path: str
    sha256: str
    commit: str
    looks_before: int
    preregistration: LoadedPreregistration


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args], capture_output=True, check=False
        )
    except OSError as error:
        raise PromotionError(f"git is needed to check a promotion: {error}") from error


def parse_ledger(raw: bytes) -> list[dict[str, Any]]:
    """The looks the committed holdout ledger records, validated."""
    where = LEDGER_PATH.as_posix()
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PromotionError(f"{where} is not valid YAML: {error}") from error
    if not isinstance(data, dict) or set(data) != {"looks"} or not isinstance(data["looks"], list):
        raise PromotionError(f"{where} must hold exactly a 'looks' list")
    looks = data["looks"]
    for look in looks:
        if not isinstance(look, dict) or set(look) != set(_LOOK_FIELDS):
            raise PromotionError(
                f"each look in {where} has exactly {list(_LOOK_FIELDS)}; got {look!r}"
            )
        problems = [
            name
            for name, ok in (
                ("id", isinstance(look["id"], str) and _ID.fullmatch(look["id"])),
                ("experiment_id", isinstance(look["experiment_id"], str) and look["experiment_id"]),
                ("candidate", isinstance(look["candidate"], str) and look["candidate"]),
                ("commit", isinstance(look["commit"], str) and _SHA.fullmatch(look["commit"])),
                *(
                    (name, isinstance(look[name], str) and _SHA256.fullmatch(look[name]))
                    for name in ("record_sha256", "receipt_sha256")
                ),
                ("passed", isinstance(look["passed"], bool)),
            )
            if not ok
        ]
        if problems:
            raise PromotionError(f"a look in {where} has malformed {problems}: {look!r}")
    ids = [look["id"] for look in looks]
    if len(set(ids)) != len(ids):
        raise PromotionError(f"{where} records an id twice: {ids}")
    return looks


def _ledger_at(repo_root: Path, rev: str) -> list[dict[str, Any]]:
    return parse_ledger(read_committed(repo_root, LEDGER_PATH.as_posix(), rev))


def load_promotion(
    promotion_id: str, repo_root: str | Path, *, canonical_ref: str = CANONICAL_REF
) -> LoadedPromotion:
    """Load ``<id>.yaml`` and the holdout ledger as committed at HEAD; refuse otherwise.

    The ledger HEAD carries must extend the one on ``canonical_ref`` (``origin/main``),
    so a branch cut from an old verdict commit cannot run on a ledger that forgot a
    look. And HEAD may carry no other record that the ledger has not recorded, so two
    records committed together cannot both run before either look is recorded.
    """
    if not isinstance(promotion_id, str) or not _ID.fullmatch(promotion_id):
        raise PromotionError(f"protocol.holdout_promotion {promotion_id!r} is not an id")
    repo_root = Path(repo_root)
    relative = (HOLDOUT_DIR / f"{promotion_id}.yaml").as_posix()
    if not (repo_root / relative).is_file():
        raise PromotionError(
            f"protocol.holdout_promotion={promotion_id}: {relative} does not exist; a "
            "promotion record is committed before its run"
        )
    head = require_committed_checkout(repo_root, relative)
    ledger = LEDGER_PATH.as_posix()
    if _git(repo_root, "ls-files", "--error-unmatch", "--", ledger).returncode != 0:
        raise PromotionError(f"{ledger} is not committed; the holdout ledger must be")
    looks = _ledger_at(repo_root, "HEAD")
    try:
        canonical = _ledger_at(repo_root, canonical_ref)
    except PreregistrationError as error:
        raise PromotionError(
            f"the holdout ledger on {canonical_ref} cannot be read ({error}); fetch it, "
            "since a promotion counts the looks recorded there"
        ) from error
    if looks[: len(canonical)] != canonical:
        raise PromotionError(
            f"this checkout's {ledger} does not extend the one on {canonical_ref} "
            f"({len(canonical)} looks); copy that ledger onto the promotion branch"
        )
    record, sha256 = _record_at(promotion_id, repo_root, head)
    if promotion_id in {look["id"] for look in looks}:
        raise PromotionError(
            f"{promotion_id} has already looked at the holdout ({ledger}); each candidate "
            "gets one look (PRD #369, decision 3)"
        )
    if (record.experiment_id, record.candidate) in {
        (look["experiment_id"], look["candidate"]) for look in looks
    }:
        raise PromotionError(
            f"{record.experiment_id}'s {record.candidate!r} has already looked at the "
            f"holdout under another id ({ledger}); each candidate gets one look"
        )
    if len(looks) >= MAX_LOOKS:
        raise PromotionError(
            f"the holdout has had its {MAX_LOOKS} looks ({ledger}) and is retired; the next "
            "holdout is the data after 2026-07-31 (decision 3)"
        )
    listed = _git(repo_root, "ls-tree", "--name-only", "HEAD", HOLDOUT_DIR.as_posix() + "/")
    recorded = {look["id"] for look in looks}
    pending = sorted(
        name.rsplit("/", 1)[-1].removesuffix(".yaml")
        for name in listed.stdout.decode().splitlines()
        if name.endswith(".yaml") and name != ledger
    )
    pending = [other for other in pending if other != promotion_id and other not in recorded]
    if pending:
        raise PromotionError(
            f"{promotion_id}: this checkout also carries the promotion record(s) {pending}, "
            f"which {ledger} has not recorded; record or remove them first, one look at a time"
        )
    if _git(repo_root, "merge-base", "--is-ancestor", record.commit, head).returncode != 0:
        raise PromotionError(
            f"{promotion_id}: the verdict's commit {record.commit[:8]} is not in the history "
            f"of this checkout's {head[:8]}"
        )
    # --no-renames: a file moved into the holdout folder still counts at its old path.
    changed = _git(
        repo_root, "diff", "--no-renames", "--name-only", record.commit, head
    ).stdout.decode()
    outside = [
        path for path in changed.splitlines() if not path.startswith(HOLDOUT_DIR.as_posix() + "/")
    ]
    if outside:
        raise PromotionError(
            f"{promotion_id}: this checkout differs from the verdict's commit "
            f"{record.commit[:8]} outside {HOLDOUT_DIR.as_posix()}/, e.g. {outside[0]}; a "
            "promotion runs the code the verdict ran"
        )
    loaded = _experiment_of(record, repo_root)
    return LoadedPromotion(
        record=record,
        path=relative,
        sha256=sha256,
        commit=head,
        looks_before=len(looks),
        preregistration=loaded,
    )


def promotion_at_commit(
    promotion_id: str, repo_root: str | Path, commit: str
) -> tuple[PromotionRecord, str, LoadedPreregistration]:
    """The record as committed at *commit*, its SHA-256, and its experiment at the verdict.

    No check of the working tree: the receipt reads the record at the runs' commit, as
    the verdict reads a declaration at its runs' commit. The incumbent must be the
    experiment's control and the candidate one of its treatment arms (not the control,
    not the seed-floor arm).
    """
    record, sha256 = _record_at(promotion_id, Path(repo_root), commit)
    return record, sha256, _experiment_of(record, Path(repo_root))


def _record_at(promotion_id: str, repo_root: Path, commit: str) -> tuple[PromotionRecord, str]:
    relative = (HOLDOUT_DIR / f"{promotion_id}.yaml").as_posix()
    raw = read_committed(repo_root, relative, commit)
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PromotionError(f"{relative} is not valid YAML: {error}") from error
    record = PromotionRecord.from_mapping(data, expected_id=promotion_id)
    return record, hashlib.sha256(raw).hexdigest()


def _experiment_of(record: PromotionRecord, repo_root: Path) -> LoadedPreregistration:
    """The record's experiment at the verdict's commit, its arms checked against it."""
    promotion_id = record.id
    loaded = preregistration_at_commit(record.experiment_id, repo_root, record.commit)
    preregistration = loaded.preregistration
    if record.incumbent != preregistration.control:
        raise PromotionError(
            f"{promotion_id}: the incumbent must be {record.experiment_id}'s control "
            f"{preregistration.control!r}, not {record.incumbent!r}"
        )
    if record.candidate not in treatment_arms(preregistration):
        raise PromotionError(
            f"{promotion_id}: {record.candidate!r} is not one of {record.experiment_id}'s "
            f"treatment arms {treatment_arms(preregistration)}"
        )
    return loaded


def promotion_overrides(loaded: LoadedPromotion, arm: str) -> list[str]:
    """The overrides one promotion arm composes: the experiment's arm on the holdout window."""
    if arm not in (loaded.record.candidate, loaded.record.incumbent):
        raise PromotionError(
            f"{loaded.record.id} runs only {loaded.record.candidate!r} and "
            f"{loaded.record.incumbent!r}, not {arm!r}"
        )
    prereg = loaded.preregistration
    return arm_overrides(
        prereg.preregistration,
        recipe_overrides(prereg.recipe_text),
        arm=arm,
        spans=HOLDOUT_WINDOW.spans(),
    )


def check_promotion_run(
    loaded: LoadedPromotion,
    resolved: Mapping[str, Any],
    *,
    arm: str | None,
    compose: Callable[[list[str]], Mapping[str, Any]],
) -> None:
    """Refuse a run that is not exactly *arm* of the promotion on the holdout window."""
    prereg = loaded.preregistration
    slug = recipe_slug(prereg.recipe_text)
    if slug != prereg.preregistration.recipe:
        raise PromotionError(
            f"{loaded.record.id}: {prereg.preregistration.id} was registered on recipe "
            f"{prereg.preregistration.recipe!r}, but the verdict's commit has {slug!r}"
        )
    want = compose(promotion_overrides(loaded, str(arm)))
    differ = config_differences(resolved, want)
    if differ:
        more = f" and {len(differ) - 8} more" if len(differ) > 8 else ""
        raise PromotionError(
            f"{loaded.record.id}: this run is not arm {arm!r} on the holdout window as "
            f"recorded: {', '.join(differ[:8])}{more}"
        )


def promotion_stamp(loaded: LoadedPromotion, *, arm: str) -> dict[str, Any]:
    """What a promotion run records about its standing in ``run_metadata.json``."""
    prereg = loaded.preregistration
    return {
        "exploratory": False,
        "experiment_id": None,
        "arm": arm,
        "window": HOLDOUT_WINDOW_NAME,
        "holdout_promotion": loaded.record.id,
        "promotion_record_path": loaded.path,
        "promotion_record_sha256": loaded.sha256,
        "promotion_commit": loaded.commit,
        "looks_before": loaded.looks_before,
        "candidate_experiment_id": loaded.record.experiment_id,
        "verdict_commit": loaded.record.commit,
        "preregistration_sha256": prereg.sha256,
        "recipe_slug": recipe_slug(prereg.recipe_text),
        "recipe_overrides_sha256": recipe_overrides_sha256(prereg.recipe_text),
    }


def resolve_promotion_standing(
    protocol: Any,
    resolved: Mapping[str, Any],
    *,
    repo_root: str | Path,
    compose: Callable[[list[str]], Mapping[str, Any]],
    config_sources: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The runner's gate for ``protocol.holdout_promotion``: the run's stamp, or a refusal."""
    foreign = foreign_config_sources(config_sources or [], Path(repo_root) / "configs")
    if foreign:
        raise PromotionError(
            f"{protocol.holdout_promotion}: a promotion composes from this checkout's configs "
            f"only, but Hydra also reads {foreign}"
        )
    loaded = load_promotion(protocol.holdout_promotion, repo_root)
    check_promotion_run(loaded, resolved, arm=protocol.arm, compose=compose)
    return promotion_stamp(loaded, arm=str(protocol.arm))
