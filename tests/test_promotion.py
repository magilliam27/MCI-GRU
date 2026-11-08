"""The one way into the locked 2026 holdout: a committed promotion record (#374).

PRD #369, decision 3. These tests pin the runner's gate:

* the record: every field required and no other, a full commit id, the one pass rule;
* the unlock: the record must exist and be committed with every other tracked file,
  and the committed holdout ledger must hold neither its id nor three looks already;
* the code: the run's commit differs from the verdict's only under
  ``docs/research/holdout/``, and the incumbent is the experiment's control and the
  candidate one of its treatment arms;
* the run: exactly the named arm, at the experiment's seed base, on the holdout window;
* the runner: a promotion goes through this gate, before anything is written.

The holdout window's dates themselves are pinned in ``tests/test_protocol_holdout.py``,
and the launcher's receipt in ``tests/test_first_run_colab.py``.
"""

from __future__ import annotations

import copy
import hashlib
import re
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import run_experiment
from mci_gru.config import ProtocolConfig
from mci_gru.preregistration import PreregistrationError, recipe_slug
from mci_gru.promotion import (
    LEDGER_PATH,
    MAX_LOOKS,
    PASS_RULE,
    PromotionError,
    PromotionRecord,
    check_promotion_run,
    load_promotion,
    parse_ledger,
    promotion_overrides,
    resolve_promotion_standing,
)
from mci_gru.protocol import HOLDOUT_WINDOW, WINDOWS

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE_TEXT = (REPO_ROOT / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md").read_text(encoding="utf-8")
EXPERIMENT = "graph-test"
PROMOTION = "graph-zeroed-2026"


def _declaration() -> dict:
    return {
        "id": EXPERIMENT,
        "family": "graph",
        "hypothesis": "Zeroing the graph adds IC.",
        "recipe": recipe_slug(RECIPE_TEXT),
        "control": "recipe",
        "arms": {
            "recipe": [],
            "graph_zeroed": ["+experiment=graph_zeroed"],
            "other_seeds": ["seed=2729"],
        },
        "seed_base": 1729,
        "windows": list(WINDOWS),
        "primary_metric": "ic",
        "alpha": 0.05,
        "decision_rule": {"adjustment": "holm", "seed_floor_multiple": 1.0},
    }


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return done.stdout.decode().strip()


def _record(sha: str, **changes) -> dict:
    data = {
        "id": PROMOTION,
        "experiment_id": EXPERIMENT,
        "candidate": "graph_zeroed",
        "incumbent": "recipe",
        "commit": sha,
        "pass_rule": PASS_RULE,
    }
    data.update(changes)
    return data


def _look(look_id: str, passed: bool = False, candidate: str = "other_arm") -> dict:
    return {
        "id": look_id,
        "experiment_id": EXPERIMENT,
        "candidate": candidate,
        "commit": "a" * 40,
        "record_sha256": "b" * 64,
        "receipt_sha256": "c" * 64,
        "passed": passed,
    }


def _write_holdout(repo: Path, *, record: dict | None, looks: list[dict] | None = None) -> None:
    holdout = repo / "docs" / "research" / "holdout"
    holdout.mkdir(parents=True, exist_ok=True)
    (holdout / "ledger.yaml").write_text(
        yaml.safe_dump({"looks": looks or []}, sort_keys=False), encoding="utf-8"
    )
    if record is not None:
        (holdout / f"{record['id']}.yaml").write_text(
            yaml.safe_dump(record, sort_keys=False), encoding="utf-8"
        )


def _commit(repo: Path, message: str = "holdout", *, publish: bool = True) -> str:
    """Commit everything; ``publish`` also moves ``origin/main`` there, as a merge would."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    if publish:
        _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def verdict_repo(tmp_path: Path) -> tuple[Path, str]:
    """A checkout at the commit an experiment's verdict ran: the recipe and its file."""
    root = tmp_path / "repo"
    (root / "docs" / "research" / "experiments").mkdir(parents=True)
    (root / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md").write_text(RECIPE_TEXT, encoding="utf-8")
    (root / "docs" / "research" / "experiments" / f"{EXPERIMENT}.yaml").write_text(
        yaml.safe_dump(_declaration(), sort_keys=False), encoding="utf-8"
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    return root, _commit(root, "verdict")


@pytest.fixture
def repo(verdict_repo) -> Path:
    """The verdict's checkout plus a committed record and an empty holdout ledger."""
    root, verdict = verdict_repo
    _write_holdout(root, record=_record(verdict))
    _commit(root)
    return root


def _compose(overrides: list[str]) -> dict:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


#: What the launcher adds to a promotion run: where it writes and its protocol stamp.
def _launched(loaded, arm: str, *extra: str) -> dict:
    return _compose(
        [
            *promotion_overrides(loaded, arm),
            "tracking.enabled=false",
            "experiment_name=promotion-run",
            f"protocol.holdout_promotion={PROMOTION}",
            f"protocol.arm={arm}",
            *extra,
        ]
    )


# -- The record -------------------------------------------------------------------------


def test_a_committed_record_loads_against_the_verdict_and_the_ledger(repo, verdict_repo):
    _root, verdict = verdict_repo
    loaded = load_promotion(PROMOTION, repo)
    assert loaded.record == PromotionRecord(**_record(verdict))
    assert loaded.commit == _git(repo, "rev-parse", "HEAD")
    assert loaded.looks_before == 0
    assert loaded.preregistration.commit == verdict
    raw = (repo / "docs" / "research" / "holdout" / f"{PROMOTION}.yaml").read_bytes()
    assert loaded.sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"pass_rule": "holdout_ic_above_zero"}, "pass_rule is fixed"),
        ({"commit": "abc1234"}, "40-character"),
        ({"commit": "A" * 40}, "40-character"),
        ({"commit": "a" * 40 + "\n"}, "40-character"),
        ({"id": "other-id"}, "declares id"),
        ({"candidate": "recipe"}, "same arm"),
        ({"experiment_id": ""}, "non-empty"),
        ({"note": "x"}, "unknown keys"),
    ],
)
def test_a_malformed_record_is_refused(change, message):
    with pytest.raises(PromotionError, match=message):
        PromotionRecord.from_mapping(_record("a" * 40, **change), expected_id=PROMOTION)
    data = _record("a" * 40)
    del data["incumbent"]
    with pytest.raises(PromotionError, match="missing"):
        PromotionRecord.from_mapping(data, expected_id=PROMOTION)


def test_the_unlock_without_a_record_is_refused(verdict_repo):
    """Mutation case: the unlock without a record."""
    root, _verdict = verdict_repo
    _write_holdout(root, record=None)
    _commit(root)
    with pytest.raises(PromotionError, match="does not exist"):
        load_promotion(PROMOTION, root)
    with pytest.raises(PromotionError, match="not an id"):
        load_promotion("../experiments/graph-test", root)


def test_a_record_not_committed_is_refused(verdict_repo):
    """Mutation case: a record not committed."""
    root, verdict = verdict_repo
    _write_holdout(root, record=None)
    _commit(root)
    _write_holdout(root, record=_record(verdict))
    with pytest.raises(PreregistrationError, match="is not committed; commit it before the run"):
        load_promotion(PROMOTION, root)


def test_a_record_edited_after_its_commit_is_refused(repo):
    path = repo / "docs" / "research" / "holdout" / f"{PROMOTION}.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    with pytest.raises(PreregistrationError, match="differs from the commit"):
        load_promotion(PROMOTION, repo)


# -- The ledger -------------------------------------------------------------------------


def test_the_same_id_scored_twice_is_refused(verdict_repo):
    """Mutation case: an id already in the committed ledger has had its look."""
    root, verdict = verdict_repo
    _write_holdout(root, record=_record(verdict), looks=[_look(PROMOTION)])
    _commit(root)
    with pytest.raises(PromotionError, match="already looked"):
        load_promotion(PROMOTION, root)


def test_a_fourth_look_is_refused(verdict_repo):
    """Mutation case: three looks retire the holdout, whoever took them."""
    root, verdict = verdict_repo
    looks = [_look(f"earlier-{n}") for n in range(MAX_LOOKS)]
    _write_holdout(root, record=_record(verdict), looks=looks)
    _commit(root)
    with pytest.raises(PromotionError, match="retired"):
        load_promotion(PROMOTION, root)
    # Control: two earlier looks leave the third.
    _write_holdout(root, record=_record(verdict), looks=looks[:2])
    _commit(root)
    assert load_promotion(PROMOTION, root).looks_before == 2


def test_the_ledger_must_be_committed_and_well_formed(verdict_repo):
    root, verdict = verdict_repo
    _write_holdout(root, record=_record(verdict))
    (root / LEDGER_PATH).unlink()
    _commit(root)
    with pytest.raises(PromotionError, match="not committed"):
        load_promotion(PROMOTION, root)
    for raw, message in (
        (b"looks: {}\n", "exactly a 'looks' list"),
        (b"looks: []\nextra: 1\n", "exactly a 'looks' list"),
        (b"looks:\n- id: x\n", "has exactly"),
        (yaml.safe_dump({"looks": [_look("x"), _look("x")]}).encode(), "records an id twice"),
    ):
        with pytest.raises(PromotionError, match=message):
            parse_ledger(raw)


def test_a_branch_ledger_that_forgets_a_look_on_main_is_refused(verdict_repo):
    """A promotion branch starts at an old verdict commit; main's looks still count."""
    root, verdict = verdict_repo
    _write_holdout(root, record=None, looks=[_look("earlier-0")])
    _commit(root, "a look recorded on main")
    _write_holdout(root, record=_record(verdict), looks=[])
    _commit(root, "a branch that forgot it", publish=False)
    with pytest.raises(PromotionError, match="does not extend"):
        load_promotion(PROMOTION, root)
    # Control: carrying main's ledger over passes, and counts its look.
    _write_holdout(root, record=_record(verdict), looks=[_look("earlier-0")])
    _commit(root, "main's ledger copied", publish=False)
    assert load_promotion(PROMOTION, root).looks_before == 1


def test_a_checkout_without_the_canonical_ledger_is_refused(repo):
    _git(repo, "update-ref", "-d", "refs/remotes/origin/main")
    with pytest.raises(PromotionError, match="origin/main cannot be read"):
        load_promotion(PROMOTION, repo)


def test_the_same_candidate_under_another_id_is_refused(verdict_repo):
    root, verdict = verdict_repo
    _write_holdout(
        root, record=_record(verdict), looks=[_look("first-try", candidate="graph_zeroed")]
    )
    _commit(root)
    with pytest.raises(PromotionError, match="under another id"):
        load_promotion(PROMOTION, root)


def test_a_second_unrecorded_record_in_the_checkout_is_refused(repo, verdict_repo):
    """Two records committed together could both run before either look is recorded."""
    _root, verdict = verdict_repo
    _write_holdout(repo, record=_record(verdict, id="another-promotion"))
    _commit(repo)
    with pytest.raises(PromotionError, match="another-promotion"):
        load_promotion(PROMOTION, repo)
    # Control: once the ledger records it, it no longer blocks (and the budget counts it).
    _write_holdout(repo, record=None, looks=[_look("another-promotion")])
    _commit(repo)
    assert load_promotion(PROMOTION, repo).looks_before == 1


def test_a_malformed_look_is_refused():
    for change in (
        {"passed": "yes"},
        {"commit": "abc"},
        {"record_sha256": "b" * 63},
        {"id": "Not An Id"},
        {"candidate": ""},
    ):
        with pytest.raises(PromotionError, match="malformed"):
            parse_ledger(yaml.safe_dump({"looks": [{**_look("x"), **change}]}).encode())


# -- The code is the verdict's ----------------------------------------------------------


def test_code_that_differs_from_the_verdicts_commit_is_refused(verdict_repo):
    root, verdict = verdict_repo
    _write_holdout(root, record=_record(verdict))
    (root / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md").write_text(
        RECIPE_TEXT + "\nA later note.\n", encoding="utf-8"
    )
    _commit(root)
    with pytest.raises(PromotionError, match="differs from the verdict's commit"):
        load_promotion(PROMOTION, root)


def test_a_file_moved_into_the_holdout_folder_still_counts_as_a_code_change(repo):
    _git(repo, "mv", "docs/DEFAULT_EXPERIMENT_RECIPE.md", "docs/research/holdout/recipe.md")
    _commit(repo)
    with pytest.raises(PromotionError, match="DEFAULT_EXPERIMENT_RECIPE"):
        load_promotion(PROMOTION, repo)


def test_changes_under_the_holdout_folder_are_allowed(repo):
    (repo / "docs" / "research" / "holdout" / "README.md").write_text("notes\n", encoding="utf-8")
    _commit(repo)
    assert load_promotion(PROMOTION, repo).record.id == PROMOTION


def test_a_verdict_commit_outside_the_history_is_refused(verdict_repo):
    root, _verdict = verdict_repo
    _write_holdout(root, record=_record("0" * 40))
    _commit(root)
    with pytest.raises(PromotionError, match="not in the history"):
        load_promotion(PROMOTION, root)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"incumbent": "graph_zeroed", "candidate": "other_seeds"}, "incumbent must be"),
        ({"candidate": "other_seeds"}, "treatment arms"),
        ({"candidate": "levels_inputs"}, "treatment arms"),
        ({"experiment_id": "absent"}, "absent"),
    ],
)
def test_the_arms_must_be_the_experiments_control_and_a_treatment(verdict_repo, change, message):
    root, verdict = verdict_repo
    _write_holdout(root, record=_record(verdict, **change))
    _commit(root)
    with pytest.raises(PreregistrationError, match=message):
        load_promotion(PROMOTION, root)


# -- The run ----------------------------------------------------------------------------


def test_a_promotion_arm_composes_on_the_holdout_window_and_the_seed_base(repo):
    loaded = load_promotion(PROMOTION, repo)
    for arm in ("graph_zeroed", "recipe"):
        overrides = promotion_overrides(loaded, arm)
        assert "seed=1729" in overrides
        for key, value in HOLDOUT_WINDOW.spans().items():
            assert f"{key}={value}" in overrides
        check_promotion_run(loaded, _launched(loaded, arm), arm=arm, compose=_compose)
    with pytest.raises(PromotionError, match="runs only"):
        promotion_overrides(loaded, "other_seeds")


def test_a_run_that_is_not_the_named_arm_on_the_holdout_window_is_refused(repo):
    loaded = load_promotion(PROMOTION, repo)
    candidate = _launched(loaded, "graph_zeroed")
    with pytest.raises(PromotionError, match="not arm 'recipe'"):
        check_promotion_run(loaded, candidate, arm="recipe", compose=_compose)
    for extra, key in (("seed=2729", "seed"), ("data.test_end=2026-06-30", "data.test_end")):
        changed = _launched(loaded, "recipe", extra)
        with pytest.raises(PromotionError, match=re.escape(key)):
            check_promotion_run(loaded, changed, arm="recipe", compose=_compose)


def test_the_gate_stamps_the_run_with_its_record_and_the_verdict(repo, verdict_repo):
    _root, verdict = verdict_repo
    protocol = ProtocolConfig(holdout_promotion=PROMOTION, arm="graph_zeroed")
    loaded = load_promotion(PROMOTION, repo)
    stamp = resolve_promotion_standing(
        protocol, _launched(loaded, "graph_zeroed"), repo_root=repo, compose=_compose
    )
    assert stamp["exploratory"] is False
    assert stamp["experiment_id"] is None
    assert (stamp["arm"], stamp["window"]) == ("graph_zeroed", "holdout")
    assert stamp["holdout_promotion"] == PROMOTION
    assert stamp["promotion_record_sha256"] == loaded.sha256
    assert stamp["promotion_commit"] == _git(repo, "rev-parse", "HEAD")
    assert stamp["verdict_commit"] == verdict
    assert stamp["looks_before"] == 0
    assert stamp["candidate_experiment_id"] == EXPERIMENT
    foreign = {"path": "/elsewhere", "provider": "command-line", "schema": "file"}
    with pytest.raises(PromotionError, match="configs only"):
        resolve_promotion_standing(
            protocol,
            _launched(loaded, "graph_zeroed"),
            repo_root=repo,
            compose=_compose,
            config_sources=[foreign],
        )


def test_the_runner_sends_a_promotion_through_its_gate_before_writing_anything(
    tmp_path, verdict_repo, monkeypatch
):
    root, _verdict = verdict_repo
    _write_holdout(root, record=None)
    _commit(root)
    monkeypatch.setattr(run_experiment, "REPO_ROOT", root)
    out = tmp_path / "out"
    out.mkdir()
    overrides = [
        *(f"{key}={value}" for key, value in HOLDOUT_WINDOW.spans().items()),
        f"protocol.holdout_promotion={PROMOTION}",
        "protocol.arm=graph_zeroed",
        "tracking.enabled=false",
    ]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose("config", overrides=overrides)
    monkeypatch.chdir(out)
    with pytest.raises(PromotionError, match="does not exist"):
        run_experiment.main.__wrapped__(cfg)
    assert not list(out.iterdir())


def test_the_committed_ledger_starts_empty_and_parses():
    raw = (REPO_ROOT / LEDGER_PATH).read_bytes()
    assert parse_ledger(raw) == []


def test_the_template_record_in_the_protocol_doc_is_a_valid_record():
    doc = (REPO_ROOT / "docs" / "research" / "EXPERIMENT_PROTOCOL.md").read_text("utf-8")
    match = re.search(r"^ *Promotion record:\n+ *```yaml\n(.*?)^ *```", doc, re.M | re.S)
    assert match, "EXPERIMENT_PROTOCOL.md carries no promotion record template"
    data = yaml.safe_load(textwrap.dedent(match.group(1)))
    record = PromotionRecord.from_mapping(copy.deepcopy(data), expected_id=data["id"])
    assert record.pass_rule == PASS_RULE
