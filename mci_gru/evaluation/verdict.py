"""The verdict a pre-registered experiment gets from its complete trial ledger (#373).

PRD #369, decision 4. Each treatment arm's pooled paired IC difference against the
control is tested with the overlap-aware HAC t (``paired_inference``). The two-sided
p-values are adjusted across the treatment arms, Holm by default or Bonferroni when the
pre-registration asks for it. An arm is adopted when its adjusted p is below alpha, its
difference is positive, and the difference exceeds the seed floor times the declared
multiple. It is rejected on the same test with a negative difference. Anything else is
inconclusive, and the default stays.

The seed floor is the absolute pooled paired IC difference that changing only the seed
base gives on the same recipe: the noise two separately seeded ensembles put under any
comparison. It comes from the experiment's own seed arm, or from another declared
experiment's verdict on the same recipe, and a verdict without one is refused.

Everything here is pure; reading runs and writing files stay in the launcher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mci_gru.preregistration import Preregistration

ADOPT = "adopt"
REJECT = "reject"
INCONCLUSIVE = "inconclusive"
#: The one key whose change makes an arm the seed-floor arm.
SEED_KEY = "seed"


class VerdictError(ValueError):
    """A verdict the protocol does not allow: a missing floor, an unknown rule."""


def _override_key(override: str) -> str:
    return override.partition("=")[0].lstrip("+~")


def seed_floor_arm(preregistration: Preregistration) -> str | None:
    """The arm whose one change is the seed base, or None; two such arms are refused."""
    arms = [
        name
        for name, changes in preregistration.arms.items()
        if name != preregistration.control
        and len(changes) == 1
        and _override_key(changes[0]) == SEED_KEY
    ]
    if len(arms) > 1:
        raise VerdictError(
            f"{preregistration.id} declares {len(arms)} seed arms {sorted(arms)}; the seed "
            "floor comes from exactly one"
        )
    return arms[0] if arms else None


def treatment_arms(preregistration: Preregistration) -> list[str]:
    """The arms a verdict is about: every arm but the control and the seed-floor arm."""
    seed_arm = seed_floor_arm(preregistration)
    return [
        name for name in preregistration.arms if name not in (preregistration.control, seed_arm)
    ]


def adjusted_p_values(p_values: Mapping[str, float | None], adjustment: str) -> dict[str, float]:
    """Holm step-down or Bonferroni, capped at one, over every arm given.

    A missing or non-finite p counts toward the family and is adjusted as one, so an arm
    whose test could not be computed can never be adopted and never shrinks the family.
    """
    if adjustment not in ("holm", "bonferroni"):
        raise VerdictError(f"unknown p-value adjustment {adjustment!r}")
    if not p_values:
        return {}
    clean = {
        arm: (float(p) if p is not None and math.isfinite(float(p)) else 1.0)
        for arm, p in p_values.items()
    }
    m = len(clean)
    if adjustment == "bonferroni":
        return {arm: min(1.0, m * p) for arm, p in clean.items()}
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (arm, p) in enumerate(sorted(clean.items(), key=lambda item: item[1])):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[arm] = running
    return {arm: adjusted[arm] for arm in clean}


@dataclass(frozen=True)
class ArmDecision:
    """One treatment arm's verdict and every number that decided it."""

    arm: str
    mean_delta: float
    hac_t: float | None
    p_value: float | None
    adjusted_p_value: float
    significant: bool
    exceeds_seed_floor: bool
    verdict: str


def decide(
    tests: Mapping[str, Mapping[str, Any]],
    *,
    alpha: float,
    adjustment: str,
    seed_floor: float | None,
    seed_floor_multiple: float,
) -> list[ArmDecision]:
    """Adopt, reject or inconclusive for each arm in ``tests``.

    ``tests`` maps each treatment arm to its pooled paired test of IC(arm) minus
    IC(control): at least ``mean_delta``, ``hac_t`` and ``hac_p``. The adjustment runs
    across exactly these arms, so the caller passes every treatment arm and nothing else.
    """
    if seed_floor is None or not math.isfinite(float(seed_floor)):
        raise VerdictError(
            "a verdict needs the seed floor: declare a seed arm in the experiment or name "
            "another declared experiment's verdict on the same recipe"
        )
    if not tests:
        raise VerdictError("a verdict needs at least one treatment arm")
    floor = abs(float(seed_floor)) * float(seed_floor_multiple)
    adjusted = adjusted_p_values({arm: t.get("hac_p") for arm, t in tests.items()}, adjustment)
    decisions = []
    for arm, test in tests.items():
        delta = float(test["mean_delta"])
        significant = adjusted[arm] < alpha
        exceeds = math.isfinite(delta) and abs(delta) > floor
        if significant and exceeds and delta > 0:
            verdict = ADOPT
        elif significant and exceeds and delta < 0:
            verdict = REJECT
        else:
            verdict = INCONCLUSIVE
        decisions.append(
            ArmDecision(
                arm=arm,
                mean_delta=delta,
                hac_t=test.get("hac_t"),
                p_value=test.get("hac_p"),
                adjusted_p_value=adjusted[arm],
                significant=significant,
                exceeds_seed_floor=exceeds,
                verdict=verdict,
            )
        )
    return decisions


def seed_floor_from_verdict(
    payload: Mapping[str, Any], *, recipe_slug: str, recipe_overrides_sha256: str
) -> dict[str, Any]:
    """The seed floor another experiment's ``verdict.json`` measured, for the same recipe.

    Refused unless that verdict came from a complete ledger, measured a floor itself, and
    ran the same recipe slug with the same override lines.
    """
    floor = payload.get("seed_floor") or {}
    if payload.get("ledger_complete") is not True:
        raise VerdictError("the named seed-floor verdict does not come from a complete ledger")
    if floor.get("source") != "seed_arm":
        raise VerdictError(
            "the named verdict did not measure a seed floor itself; name the verdict of "
            "the experiment that ran the seed arm"
        )
    for key, want in (
        ("recipe_slug", recipe_slug),
        ("recipe_overrides_sha256", recipe_overrides_sha256),
    ):
        if floor.get(key) != want:
            raise VerdictError(
                f"the named seed floor was measured on {key} {floor.get(key)!r}, not this "
                f"experiment's {want!r}; the floor is measured again for each recipe"
            )
    value = floor.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise VerdictError(f"the named seed floor has no finite value: {value!r}")
    return dict(floor)
