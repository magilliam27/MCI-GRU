"""Pre-registered experiments: the file a protocol run must be declared in (#372).

PRD #369. An experiment's hypothesis, control, arms, seed base, windows, metric, alpha
and decision rule are written to ``docs/research/experiments/<id>.yaml`` and committed
before its first run. ``protocol.experiment_id=<id>`` then makes the runner load that
file at the run's own commit, check that the run is exactly one declared arm on one
declared window, and stamp the id, arm and window into ``run_metadata.json``.

``protocol.exploratory=true`` is the only way to run without a file; such a run is
stamped exploratory and never gets a verdict (decision 6).

The decision rule here is the PRD's default. A file may tighten it (a smaller alpha,
Bonferroni in place of Holm, a larger multiple of the seed floor) but never loosen it.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from mci_gru.protocol import WINDOWS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

EXPERIMENTS_DIR = Path("docs") / "research" / "experiments"
RECIPE_PATH = Path("docs") / "DEFAULT_EXPERIMENT_RECIPE.md"

#: The primary metric is fixed (decision 5): mean daily IC, arm minus control.
PRIMARY_METRIC = "ic"
#: The default rule (decision 4). A file may only move each of these the strict way.
DEFAULT_ALPHA = 0.05
DEFAULT_ADJUSTMENT = "holm"
DEFAULT_SEED_FLOOR_MULTIPLE = 1.0
#: Bonferroni rejects nothing Holm would not, so it is the one stricter choice.
ADJUSTMENTS = ("holm", "bonferroni")
#: A verdict needs all five windows (decision 2); a subset is exploratory.
REQUIRED_WINDOWS = tuple(WINDOWS)

#: Keys a declared run may set freely: where it writes and how it records itself,
#: never what it trains on or how. Every other resolved key must be the declared arm's.
OPERATIONAL_KEYS = (
    "experiment_name",
    "output_dir",
    "tracking",
    "protocol",
    "data.auxiliary_snapshot_mode",
    "data.auxiliary_snapshot_directory",
    "data.auxiliary_snapshot_references",
)
#: Keys an arm may not change: the window sets the dates, and the rest is operational.
_WINDOW_KEYS = tuple(WINDOWS["W1"].spans())
#: Prefixes an arm may not change either: walk-forward would replace the window's one
#: train/validate/test split, and ``hydra.*`` changes the run, not the model.
_ARM_FORBIDDEN_PREFIXES = ("training.walkforward", "hydra")

_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_ARM = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_OVERRIDE_BLOCK = re.compile(r"^## Hydra Overrides\n+```text\n(.*?)^```", re.M | re.S)
_SLUG_BLOCK = re.compile(r"^Recipe slug:\n+```text\n(.*?)^```", re.M | re.S)

_FIELDS = (
    "id",
    "family",
    "hypothesis",
    "recipe",
    "control",
    "arms",
    "seed_base",
    "windows",
    "primary_metric",
    "alpha",
    "decision_rule",
)
_RULE_FIELDS = ("adjustment", "seed_floor_multiple")


class PreregistrationError(Exception):
    """A pre-registration is malformed, missing, uncommitted, or does not match the run."""


@dataclass(frozen=True)
class DecisionRule:
    """How an arm's pooled IC difference is judged (PRD #369, Metric and rule).

    Adopt when the adjusted two-sided p is below ``alpha``, the difference is positive
    and it exceeds ``seed_floor_multiple`` times the seed floor; reject on the same test
    with a negative difference; otherwise inconclusive.
    """

    adjustment: str = DEFAULT_ADJUSTMENT
    seed_floor_multiple: float = DEFAULT_SEED_FLOOR_MULTIPLE


@dataclass(frozen=True)
class Preregistration:
    """One experiment's committed declaration."""

    id: str
    family: str
    hypothesis: str
    recipe: str
    control: str
    arms: dict[str, tuple[str, ...]]
    seed_base: int
    windows: tuple[str, ...]
    primary_metric: str
    alpha: float
    decision_rule: DecisionRule

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, expected_id: str) -> Preregistration:
        """Validate a parsed file; ``expected_id`` is the name the file was loaded by."""
        if not isinstance(data, dict):
            raise PreregistrationError("a pre-registration is a mapping of its fields")
        _require_exact_keys(data, _FIELDS, "the pre-registration")
        experiment_id = data["id"]
        if not isinstance(experiment_id, str) or not _ID.match(experiment_id):
            raise PreregistrationError(
                f"id {experiment_id!r} must be lower-case letters, digits and hyphens"
            )
        if experiment_id != expected_id:
            raise PreregistrationError(
                f"the file for {expected_id!r} declares id {experiment_id!r}"
            )
        for name in ("family", "hypothesis", "recipe"):
            if not isinstance(data[name], str) or not data[name].strip():
                raise PreregistrationError(f"{name} must be a non-empty string")
        arms = _validate_arms(data["arms"], data["control"])
        seed_base = data["seed_base"]
        if not isinstance(seed_base, int) or isinstance(seed_base, bool):
            raise PreregistrationError(f"seed_base must be an integer, got {seed_base!r}")
        windows = data["windows"]
        if not isinstance(windows, list) or tuple(windows) != REQUIRED_WINDOWS:
            raise PreregistrationError(
                f"windows must be {list(REQUIRED_WINDOWS)}: a verdict needs all five "
                f"(PRD #369, decision 2), got {windows!r}"
            )
        if data["primary_metric"] != PRIMARY_METRIC:
            raise PreregistrationError(
                f"primary_metric is fixed at {PRIMARY_METRIC!r} (decision 5), "
                f"got {data['primary_metric']!r}"
            )
        alpha = _validate_alpha(data["alpha"])
        return cls(
            id=experiment_id,
            family=data["family"].strip(),
            hypothesis=data["hypothesis"].strip(),
            recipe=data["recipe"].strip(),
            control=data["control"],
            arms=arms,
            seed_base=seed_base,
            windows=tuple(windows),
            primary_metric=PRIMARY_METRIC,
            alpha=alpha,
            decision_rule=_validate_rule(data["decision_rule"]),
        )


def _require_exact_keys(data: Mapping[str, Any], fields: Sequence[str], what: str) -> None:
    missing = [name for name in fields if name not in data]
    unknown = sorted(set(data) - set(fields))
    if missing or unknown:
        raise PreregistrationError(f"{what} is missing {missing} and has unknown keys {unknown}")


def _override_key(override: str) -> str:
    return override.partition("=")[0].lstrip("+~")


def _normalized(override: str) -> str:
    """``key=value`` without Hydra's ``+``/``++``/``~`` prefixes, for spotting repeats."""
    return f"{_override_key(override)}={override.partition('=')[2].strip()}"


def _validate_arms(arms: Any, control: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(arms, dict) or len(arms) < 2:
        raise PreregistrationError("arms must map at least the control and one arm to changes")
    if control not in arms:
        raise PreregistrationError(f"the control arm {control!r} is not among the arms")
    result: dict[str, tuple[str, ...]] = {}
    seen: dict[str, str] = {}
    for name, changes in arms.items():
        if not isinstance(name, str) or not _ARM.match(name):
            raise PreregistrationError(
                f"arm name {name!r} must be lower-case letters, digits and underscores"
            )
        if not isinstance(changes, list) or not all(isinstance(c, str) for c in changes):
            raise PreregistrationError(f"arm {name!r} must list its overrides as strings")
        expected = 0 if name == control else 1
        if len(changes) != expected:
            what = "the control changes nothing" if expected == 0 else "an arm makes one change"
            raise PreregistrationError(
                f"arm {name!r} lists {len(changes)} overrides, but {what} (PRD #369)"
            )
        for change in changes:
            key = _override_key(change)
            if "=" not in change or not key:
                raise PreregistrationError(f"arm {name!r}: {change!r} is not a Hydra override")
            if key in _WINDOW_KEYS or _is_operational(key):
                raise PreregistrationError(
                    f"arm {name!r} changes {key}, which the window or the launcher sets"
                )
            if _has_prefix(key, _ARM_FORBIDDEN_PREFIXES):
                raise PreregistrationError(
                    f"arm {name!r} changes {key}; an arm may not change walk-forward "
                    "or Hydra itself"
                )
            normalized = _normalized(change)
            if normalized in seen:
                raise PreregistrationError(
                    f"arms {seen[normalized]!r} and {name!r} make the same change {change!r}"
                )
            seen[normalized] = name
        result[name] = tuple(changes)
    return result


def _validate_alpha(alpha: Any) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise PreregistrationError(f"alpha must be a number, got {alpha!r}")
    if not 0.0 < float(alpha) <= DEFAULT_ALPHA:
        raise PreregistrationError(
            f"alpha {alpha} may tighten the default {DEFAULT_ALPHA} but never loosen it"
        )
    return float(alpha)


def _validate_rule(rule: Any) -> DecisionRule:
    if not isinstance(rule, dict):
        raise PreregistrationError("decision_rule must be a mapping")
    _require_exact_keys(rule, _RULE_FIELDS, "decision_rule")
    if rule["adjustment"] not in ADJUSTMENTS:
        raise PreregistrationError(
            f"decision_rule.adjustment must be one of {ADJUSTMENTS}: the default "
            f"{DEFAULT_ADJUSTMENT!r} or a stricter one, got {rule['adjustment']!r}"
        )
    multiple = rule["seed_floor_multiple"]
    if isinstance(multiple, bool) or not isinstance(multiple, (int, float)):
        raise PreregistrationError("decision_rule.seed_floor_multiple must be a number")
    if float(multiple) < DEFAULT_SEED_FLOOR_MULTIPLE:
        raise PreregistrationError(
            f"decision_rule.seed_floor_multiple {multiple} may tighten the default "
            f"{DEFAULT_SEED_FLOOR_MULTIPLE} but never loosen it"
        )
    return DecisionRule(adjustment=rule["adjustment"], seed_floor_multiple=float(multiple))


def _has_prefix(key: str, prefixes: Sequence[str]) -> bool:
    return any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes)


def _is_operational(key: str) -> bool:
    return _has_prefix(key, OPERATIONAL_KEYS)


# -- Loading at the run's commit ------------------------------------------------------


@dataclass(frozen=True)
class LoadedPreregistration:
    """A validated file and where it was read: its path, digest, the commit, and the
    recipe text at that commit."""

    preregistration: Preregistration
    path: str
    sha256: str
    commit: str
    recipe_text: str


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args], capture_output=True, check=False
        )
    except OSError as error:
        raise PreregistrationError(
            f"git is needed to check the pre-registration: {error}"
        ) from error


def _committed_bytes(repo_root: Path, relative: str) -> bytes:
    shown = _git(repo_root, "show", f"HEAD:{relative}")
    if shown.returncode != 0:
        raise PreregistrationError(f"{relative} is not in the commit at {repo_root}")
    return shown.stdout


def _hidden_paths(repo_root: Path) -> list[str]:
    """Tracked paths git is told not to compare: assume-unchanged or skip-worktree."""
    listing = _git(repo_root, "ls-files", "-v").stdout.decode("utf-8", "replace")
    return [line[2:] for line in listing.splitlines() if line[:1].islower() or line[:1] == "S"]


def load_preregistration(experiment_id: str, repo_root: str | Path) -> LoadedPreregistration:
    """Load ``<id>.yaml`` as committed at ``repo_root``'s HEAD; refuse it otherwise.

    The file must be tracked and identical to HEAD, and so must every other tracked
    file, with none hidden from git's comparison: the run's code, configs and recipe are
    then what the commit recorded in ``run_metadata.json`` holds. The file and the recipe
    are read from the commit itself.
    """
    if not isinstance(experiment_id, str) or not _ID.match(experiment_id):
        raise PreregistrationError(f"protocol.experiment_id {experiment_id!r} is not an id")
    repo_root = Path(repo_root)
    relative = (EXPERIMENTS_DIR / f"{experiment_id}.yaml").as_posix()
    path = repo_root / relative
    if not path.is_file():
        raise PreregistrationError(
            f"protocol.experiment_id={experiment_id}: {relative} does not exist. Commit the "
            "pre-registration before the first run, or run with protocol.exploratory=true."
        )
    head = _git(repo_root, "rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        raise PreregistrationError(
            f"{repo_root} is not a git checkout with a commit; a declared run records the "
            "commit it ran, so it runs from one"
        )
    if _git(repo_root, "ls-files", "--error-unmatch", "--", relative).returncode != 0:
        raise PreregistrationError(f"{relative} is not committed; commit it before the run")
    hidden = _hidden_paths(repo_root)
    if hidden:
        raise PreregistrationError(
            f"git is told not to compare {len(hidden)} tracked file(s), e.g. {hidden[0]} "
            "(assume-unchanged or skip-worktree); clear the flag before a declared run"
        )
    if _git(repo_root, "diff", "--quiet", "HEAD", "--", relative).returncode != 0:
        raise PreregistrationError(
            f"{relative} differs from the commit; a run reads the committed declaration only"
        )
    changed = _git(repo_root, "status", "--porcelain", "--untracked-files=no").stdout
    if changed.strip():
        first = changed.decode("utf-8", "replace").splitlines()[0][3:]
        raise PreregistrationError(
            f"tracked files differ from the commit, e.g. {first}; a declared run is the "
            "commit it records, so commit or set aside the change first"
        )
    raw = _committed_bytes(repo_root, relative)
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except yaml.YAMLError as error:
        raise PreregistrationError(f"{relative} is not valid YAML: {error}") from error
    return LoadedPreregistration(
        preregistration=Preregistration.from_mapping(data, expected_id=experiment_id),
        path=relative,
        sha256=hashlib.sha256(raw).hexdigest(),
        commit=head.stdout.decode("ascii").strip(),
        recipe_text=_committed_bytes(repo_root, RECIPE_PATH.as_posix()).decode("utf-8"),
    )


def foreign_config_sources(
    sources: Sequence[Mapping[str, Any]], config_dir: str | Path
) -> list[str]:
    """Hydra config sources other than Hydra's own, the schema, and ``config_dir``.

    A declared run must compose from this checkout's ``configs/`` only. ``--config-path``,
    ``--config-dir`` or ``hydra.searchpath`` elsewhere would let the run and its check
    both read the same foreign tree, and the check would pass.
    """
    home = Path(config_dir).resolve()
    foreign = []
    for source in sources:
        provider, schema, path = source.get("provider"), source.get("schema"), source.get("path")
        if provider == "hydra" or schema == "structured":
            continue
        if schema == "file" and path and Path(path).resolve() == home:
            continue
        foreign.append(f"{schema}://{path} ({provider})")
    return foreign


# -- Checking a run against its declaration -------------------------------------------


def recipe_overrides(text: str) -> list[str]:
    """The Hydra overrides in the recipe's override block, in order."""
    match = _OVERRIDE_BLOCK.search(text)
    if match is None:
        raise PreregistrationError("the recipe has no '## Hydra Overrides' text block")
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


def recipe_slug(text: str) -> str:
    """The one line in the block under 'Recipe slug:'."""
    match = _SLUG_BLOCK.search(text)
    lines = [line.strip() for line in match.group(1).splitlines() if line.strip()] if match else []
    if len(lines) != 1:
        raise PreregistrationError("the recipe has no one-line 'Recipe slug:' block")
    return lines[0]


def _set_override(overrides: list[str], override: str) -> list[str]:
    """``override`` in place of any override of the same key, else appended."""
    key = _override_key(override)
    if not any(_override_key(o) == key for o in overrides):
        return [*overrides, override]
    return [override if _override_key(o) == key else o for o in overrides]


def declared_overrides(
    preregistration: Preregistration, recipe: Sequence[str], *, arm: str, window: str
) -> list[str]:
    """The overrides the declared run composes: the recipe, the seed base, the arm, the window."""
    overrides = _set_override(list(recipe), f"seed={preregistration.seed_base}")
    for change in preregistration.arms[arm]:
        overrides = _set_override(overrides, change)
    for key, value in WINDOWS[window].spans().items():
        overrides = _set_override(overrides, f"{key}={value}")
    return overrides


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat or {prefix[:-1]: {}}
    return {prefix[:-1]: value}


def check_declared_run(
    preregistration: Preregistration,
    resolved: Mapping[str, Any],
    *,
    arm: str | None,
    window: str | None,
    recipe_text: str,
    compose: Callable[[list[str]], Mapping[str, Any]],
) -> None:
    """Refuse a run that is not exactly the declared ``arm`` on the declared ``window``.

    ``resolved`` is the run's resolved config as a plain mapping. ``compose`` builds the
    resolved config a list of overrides gives, the way the runner's Hydra tree does.
    Every key outside :data:`OPERATIONAL_KEYS` must equal the declared run's.
    """
    experiment = preregistration.id
    if arm not in preregistration.arms:
        raise PreregistrationError(
            f"{experiment}: protocol.arm={arm!r} is not declared; "
            f"the arms are {sorted(preregistration.arms)}"
        )
    if window not in preregistration.windows:
        raise PreregistrationError(
            f"{experiment}: protocol.window={window!r} is not declared; "
            f"the windows are {list(preregistration.windows)}"
        )
    slug = recipe_slug(recipe_text)
    if slug != preregistration.recipe:
        raise PreregistrationError(
            f"{experiment} was registered on recipe {preregistration.recipe!r}, but this "
            f"commit's recipe is {slug!r}; a changed recipe needs a new pre-registration"
        )
    recipe = recipe_overrides(recipe_text)
    want = _flatten(
        dict(compose(declared_overrides(preregistration, recipe, arm=arm, window=window)))
    )
    if arm != preregistration.control:
        control = declared_overrides(
            preregistration, recipe, arm=preregistration.control, window=window
        )
        if not _differing_keys(want, _flatten(dict(compose(control)))):
            raise PreregistrationError(
                f"{experiment}: arm {arm!r} composes the same config as the control "
                f"{preregistration.control!r}, so it tests nothing"
            )
    got = _flatten(dict(resolved))
    differ = _differing_keys(got, want)
    if differ:
        shown = ", ".join(
            f"{key}={got.get(key, '<unset>')!r} (declared {want.get(key, '<unset>')!r})"
            for key in differ[:8]
        )
        more = f" and {len(differ) - 8} more" if len(differ) > 8 else ""
        raise PreregistrationError(
            f"{experiment}: this run is not arm {arm!r} on {window} as declared: {shown}{more}"
        )


_MISSING = object()


def _differing_keys(got: Mapping[str, Any], want: Mapping[str, Any]) -> list[str]:
    """The non-operational flattened keys whose values differ, sorted."""
    return sorted(
        key
        for key in set(got) | set(want)
        if not _is_operational(key) and got.get(key, _MISSING) != want.get(key, _MISSING)
    )


def protocol_stamp(
    loaded: LoadedPreregistration | None, *, arm: str | None, window: str | None
) -> dict[str, Any]:
    """What a run records about its standing under the protocol.

    An exploratory run records the window its dates were checked against, or None.
    """
    if loaded is None:
        return {"exploratory": True, "experiment_id": None, "arm": None, "window": window}
    return {
        "exploratory": False,
        "experiment_id": loaded.preregistration.id,
        "arm": arm,
        "window": window,
        "preregistration_path": loaded.path,
        "preregistration_sha256": loaded.sha256,
        "preregistration_commit": loaded.commit,
        "recipe_slug": recipe_slug(loaded.recipe_text),
        "recipe_overrides_sha256": hashlib.sha256(
            "\n".join(recipe_overrides(loaded.recipe_text)).encode("utf-8")
        ).hexdigest(),
    }


def check_window_dates(resolved: Mapping[str, Any], window: str) -> None:
    """Refuse a run labelled ``window`` whose six ``data.*`` dates are not that window's."""
    if window not in WINDOWS:
        raise PreregistrationError(
            f"protocol.window={window!r} is not one of {list(WINDOWS)}; the holdout is "
            "never a window a run may name"
        )
    flat = _flatten(dict(resolved))
    differ = {
        key: flat.get(key)
        for key, value in WINDOWS[window].spans().items()
        if str(flat.get(key)) != value
    }
    if differ:
        raise PreregistrationError(
            f"this run is labelled {window} but its dates are not {window}'s: "
            + ", ".join(
                f"{key}={got!r} ({window} has {WINDOWS[window].spans()[key]})"
                for key, got in differ.items()
            )
        )


def resolve_run_standing(
    protocol: Any,
    resolved: Mapping[str, Any],
    *,
    repo_root: str | Path,
    compose: Callable[[list[str]], Mapping[str, Any]],
    config_sources: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The runner's gate: the run's stamp, or a refusal (#372).

    ``protocol`` is the run's :class:`~mci_gru.config.ProtocolConfig`. A run must name
    a committed pre-registration and be exactly one of its arms on one of its windows,
    or set ``protocol.exploratory=true``; an exploratory run that names a window must
    run on that window's dates (#371). ``config_sources`` is Hydra's
    ``runtime.config_sources`` for the run; a declared run with any source outside
    ``repo_root/configs`` is refused.
    """
    if protocol.exploratory:
        if protocol.window is not None:
            check_window_dates(resolved, protocol.window)
        return protocol_stamp(None, arm=None, window=protocol.window)
    if protocol.experiment_id is None:
        raise PreregistrationError(
            "This run names no pre-registration. Declare it in "
            f"{EXPERIMENTS_DIR.as_posix()}/<id>.yaml and pass protocol.experiment_id=<id> "
            "protocol.arm=<arm> protocol.window=<W1..W5>, or pass protocol.exploratory=true "
            "for a run that never gets a verdict (docs/research/EXPERIMENT_PROTOCOL.md)."
        )
    foreign = foreign_config_sources(config_sources or [], Path(repo_root) / "configs")
    if foreign:
        raise PreregistrationError(
            f"{protocol.experiment_id}: a declared run composes from this checkout's configs "
            f"only, but Hydra also reads {foreign}; drop --config-path, --config-dir and "
            "hydra.searchpath"
        )
    loaded = load_preregistration(protocol.experiment_id, repo_root)
    check_declared_run(
        loaded.preregistration,
        resolved,
        arm=protocol.arm,
        window=protocol.window,
        recipe_text=loaded.recipe_text,
        compose=compose,
    )
    return protocol_stamp(loaded, arm=protocol.arm, window=protocol.window)
