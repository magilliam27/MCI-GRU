"""The verdict rule a pre-registered experiment is judged by (#373, PRD #369 decision 4)."""

from __future__ import annotations

import copy

import pytest

from mci_gru.evaluation.verdict import (
    ADOPT,
    INCONCLUSIVE,
    REJECT,
    VerdictError,
    adjusted_p_values,
    decide,
    seed_arm_problem,
    seed_floor_arm,
    seed_floor_from_verdict,
    treatment_arms,
)
from mci_gru.preregistration import Preregistration

_SLUG = "some-recipe"
_ARMS = {"recipe": [], "graph_zeroed": ["+experiment=graph_zeroed"], "other_seeds": ["seed=2729"]}


def _prereg(arms=None) -> Preregistration:
    return Preregistration.from_mapping(
        {
            "id": "exp",
            "family": "graph",
            "hypothesis": "h",
            "recipe": _SLUG,
            "control": "recipe",
            "arms": copy.deepcopy(_ARMS if arms is None else arms),
            "seed_base": 1729,
            "windows": ["W1", "W2", "W3", "W4", "W5"],
            "primary_metric": "ic",
            "alpha": 0.05,
            "decision_rule": {"adjustment": "holm", "seed_floor_multiple": 1.0},
        },
        expected_id="exp",
    )


def _decide(tests, *, floor=0.001, multiple=1.0, adjustment="holm", alpha=0.05):
    decisions = decide(
        tests,
        alpha=alpha,
        adjustment=adjustment,
        seed_floor=floor,
        seed_floor_multiple=multiple,
    )
    return {d.arm: d for d in decisions}


def _test(delta, p):
    return {"mean_delta": delta, "hac_t": None, "hac_p": p}


# -- The adjustment ---------------------------------------------------------------------


def test_holm_steps_down_and_stays_monotone():
    adjusted = adjusted_p_values({"a": 0.01, "b": 0.04, "c": 0.03}, "holm")
    # Sorted 0.01, 0.03, 0.04 times 3, 2, 1 is 0.03, 0.06, 0.04; monotone makes the last 0.06.
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_bonferroni_multiplies_every_p_by_the_family_size_and_caps_at_one():
    adjusted = adjusted_p_values({"a": 0.01, "b": 0.4}, "bonferroni")
    assert adjusted == pytest.approx({"a": 0.02, "b": 0.8})
    assert adjusted_p_values({"a": 0.6, "b": 0.7}, "holm") == pytest.approx({"a": 1.0, "b": 1.0})


def test_a_missing_p_still_counts_toward_the_family():
    """An arm whose test failed cannot shrink the family and so cheapen the others."""
    adjusted = adjusted_p_values({"a": 0.02, "b": None, "c": float("nan")}, "holm")
    assert adjusted["a"] == pytest.approx(0.06)
    assert adjusted["b"] == adjusted["c"] == 1.0


def test_an_unknown_adjustment_is_refused():
    with pytest.raises(VerdictError, match="unknown"):
        adjusted_p_values({"a": 0.01}, "none")


# -- The decision -----------------------------------------------------------------------


def test_adopt_reject_and_inconclusive():
    decisions = _decide(
        {
            "up": _test(0.02, 0.001),
            "down": _test(-0.02, 0.001),
            "noise": _test(0.02, 0.4),
        }
    )
    assert decisions["up"].verdict == ADOPT
    assert decisions["down"].verdict == REJECT
    assert decisions["noise"].verdict == INCONCLUSIVE
    assert decisions["noise"].adjusted_p_value == pytest.approx(0.4)


def test_two_arms_each_significant_alone_are_inconclusive_after_holm():
    """Holm's job: raw p 0.03 twice passes alpha alone and fails it as a family of two."""
    decisions = _decide({"a": _test(0.02, 0.03), "b": _test(0.02, 0.03)})
    assert {d.verdict for d in decisions.values()} == {INCONCLUSIVE}
    assert decisions["a"].adjusted_p_value == pytest.approx(0.06)
    alone = _decide({"a": _test(0.02, 0.03)})
    assert alone["a"].verdict == ADOPT


def test_a_difference_inside_the_seed_floor_is_inconclusive_however_small_its_p():
    decisions = _decide({"a": _test(0.004, 1e-6), "b": _test(-0.004, 1e-6)}, floor=-0.005)
    assert decisions["a"].verdict == decisions["b"].verdict == INCONCLUSIVE
    assert not decisions["a"].exceeds_seed_floor
    assert decisions["a"].significant


def test_the_declared_multiple_scales_the_floor():
    assert _decide({"a": _test(0.008, 1e-6)}, floor=0.005)["a"].verdict == ADOPT
    assert _decide({"a": _test(0.008, 1e-6)}, floor=0.005, multiple=2.0)["a"].verdict == (
        INCONCLUSIVE
    )


def test_a_verdict_without_the_seed_floor_is_refused():
    for floor in (None, float("nan")):
        with pytest.raises(VerdictError, match="seed floor"):
            _decide({"a": _test(0.02, 0.001)}, floor=floor)


# -- Which arms are tested --------------------------------------------------------------


def test_the_seed_arm_measures_the_floor_and_is_not_tested():
    prereg = _prereg()
    assert seed_floor_arm(prereg) == "other_seeds"
    assert treatment_arms(prereg) == ["graph_zeroed"]
    without = _prereg({"recipe": [], "graph_zeroed": ["+experiment=graph_zeroed"]})
    assert seed_floor_arm(without) is None
    assert treatment_arms(without) == ["graph_zeroed"]
    assert seed_floor_arm(_prereg({"recipe": [], "s": ["++seed=5"]})) == "s"


def test_two_seed_arms_are_refused():
    prereg = _prereg({"recipe": [], "s1": ["seed=2729"], "s2": ["seed=3729"]})
    with pytest.raises(VerdictError, match="2 seed arms"):
        seed_floor_arm(prereg)


# -- A floor measured by another experiment ---------------------------------------------


def _floor_verdict(**changes):
    payload = {
        "ledger_complete": True,
        "seed_floor": {
            "source": "seed_arm",
            "value": 0.003,
            "recipe_slug": _SLUG,
            "recipe_overrides_sha256": "a" * 64,
        },
    }
    for key, value in changes.items():
        if key in payload["seed_floor"]:
            payload["seed_floor"][key] = value
        else:
            payload[key] = value
    return payload


def test_another_verdicts_floor_is_read_for_the_same_recipe_only():
    kwargs = {"recipe_slug": _SLUG, "recipe_overrides_sha256": "a" * 64}
    assert seed_floor_from_verdict(_floor_verdict(), **kwargs)["value"] == 0.003
    refusals = {
        "complete ledger": _floor_verdict(ledger_complete=False),
        "did not measure": _floor_verdict(source="another_verdict"),
        "measured again for each recipe": _floor_verdict(recipe_overrides_sha256="b" * 64),
        "no finite value": _floor_verdict(value=None),
    }
    for message, payload in refusals.items():
        with pytest.raises(VerdictError, match=message):
            seed_floor_from_verdict(payload, **kwargs)
    with pytest.raises(VerdictError, match="recipe_slug"):
        seed_floor_from_verdict(
            _floor_verdict(), recipe_slug="other", recipe_overrides_sha256="a" * 64
        )


def test_another_verdicts_floor_must_share_the_label_horizon_and_windows():
    kwargs = {"recipe_slug": _SLUG, "recipe_overrides_sha256": "a" * 64}
    payload = _floor_verdict(label_t=5, windows=["W1", "W2", "W3", "W4", "W5"])
    windows = ["W1", "W2", "W3", "W4", "W5"]
    assert seed_floor_from_verdict(payload, label_t=5, windows=windows, **kwargs)
    with pytest.raises(VerdictError, match="label_t"):
        seed_floor_from_verdict(payload, label_t=10, windows=windows, **kwargs)
    with pytest.raises(VerdictError, match="windows"):
        seed_floor_from_verdict(payload, label_t=5, windows=["W5"], **kwargs)


def test_a_seed_arm_must_use_no_member_seed_the_control_uses():
    """Member i trains on seed + i, so a base fewer than num_models away overlaps."""
    recipe = ["training.num_models=20", "seed=1729"]

    def problem(change):
        return seed_arm_problem(_prereg({"recipe": [], "s": [change]}), recipe)

    assert problem("seed=2729") is None
    assert problem("seed=1749") is None
    assert problem("seed=1709") is None
    assert "shares member seeds" in problem("seed=1748")
    assert "shares member seeds" in problem("seed=1729")
    assert "shares member seeds" in problem("seed=1710")
    assert "not an integer" in problem("seed=abc")
    assert "cannot be checked" in seed_arm_problem(_prereg({"recipe": [], "s": ["seed=9"]}), [])
    assert seed_arm_problem(_prereg({"recipe": [], "g": ["+experiment=x"]}), recipe) is None
