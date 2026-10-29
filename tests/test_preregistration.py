"""Pre-registered experiments and the runner's standing gate (#372, PRD #369).

A run must name a committed ``docs/research/experiments/<id>.yaml`` and be exactly one
of its arms on one of its windows, or say ``protocol.exploratory=true``. These tests
pin:

* the schema: every field required and no other; the control changes nothing and
  every other arm makes exactly one change; all five windows; IC fixed; a rule that
  may tighten the default but never loosen it;
* the loader: the file must exist, be tracked and match the commit;
* the run check: an undeclared arm or window, an extra change, the wrong arm's
  change, another window's dates, another seed or another recipe are refused;
* the runner: a run with no standing is refused before anything is written, and a
  run's ``run_metadata.json`` carries its stamp;
* the template in ``docs/research/EXPERIMENT_PROTOCOL.md`` loads and is accepted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import run_experiment
from mci_gru import preregistration
from mci_gru.config import ProtocolConfig, create_config_from_dict
from mci_gru.preregistration import (
    Preregistration,
    PreregistrationError,
    check_declared_run,
    declared_overrides,
    load_preregistration,
    recipe_overrides,
    recipe_slug,
    resolve_run_standing,
)
from mci_gru.protocol import WINDOWS, Window

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE_TEXT = (REPO_ROOT / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md").read_text(encoding="utf-8")
PROTOCOL_DOC = REPO_ROOT / "docs" / "research" / "EXPERIMENT_PROTOCOL.md"


def _template() -> dict:
    match = re.search(
        r"^Template:\n+```yaml\n(.*?)^```", PROTOCOL_DOC.read_text("utf-8"), re.M | re.S
    )
    assert match, "EXPERIMENT_PROTOCOL.md carries no template"
    return yaml.safe_load(match.group(1))


def _valid(**changes) -> dict:
    data = copy.deepcopy(_template())
    data.update(changes)
    return data


def _prereg(data: dict | None = None) -> Preregistration:
    data = _template() if data is None else data
    return Preregistration.from_mapping(data, expected_id=data["id"])


def _compose(overrides: list[str]) -> dict:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


#: What the Colab launcher adds to a run: where it writes and how it records itself.
LAUNCHER = [
    "tracking.enabled=false",
    "experiment_name=some-run",
    "data.auxiliary_snapshot_directory=/content/run/input_snapshots",
]


def _run_config(prereg: Preregistration, arm: str, window: str, *extra: str) -> dict:
    """The resolved config a launcher gives for ``arm`` on ``window``, plus ``extra``."""
    overrides = declared_overrides(prereg, recipe_overrides(RECIPE_TEXT), arm=arm, window=window)
    stamp = [
        f"protocol.experiment_id={prereg.id}",
        f"protocol.arm={arm}",
        f"protocol.window={window}",
    ]
    return _compose([*overrides, *LAUNCHER, *stamp, *extra])


def _check(prereg: Preregistration, resolved: dict, arm: str, window: str) -> None:
    check_declared_run(
        prereg, resolved, arm=arm, window=window, recipe_text=RECIPE_TEXT, compose=_compose
    )


# -- The schema -------------------------------------------------------------------------


def test_the_documented_template_is_a_valid_preregistration():
    prereg = _prereg()
    assert prereg.control == "recipe"
    assert prereg.arms == {"recipe": (), "graph_zeroed": ("+experiment=graph_zeroed",)}
    assert prereg.windows == ("W1", "W2", "W3", "W4", "W5")
    assert (prereg.alpha, prereg.decision_rule.adjustment) == (0.05, "holm")


def test_the_template_names_the_current_recipe():
    """The template must not rot: a stale slug would be refused on every run."""
    assert _template()["recipe"] == recipe_slug(RECIPE_TEXT)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"id": "Other_ID"}, "lower-case"),
        ({"id": "another-id"}, "declares id"),
        ({"family": ""}, "family"),
        ({"control": "missing"}, "control arm"),
        ({"arms": {"recipe": []}}, "at least"),
        ({"arms": {"recipe": ["seed=1"], "x": ["graph.top_k=5"]}}, "control changes nothing"),
        (
            {"arms": {"recipe": [], "two": ["graph.top_k=5", "graph.top_k_metric=abs_corr"]}},
            "one change",
        ),
        ({"arms": {"recipe": [], "none": []}}, "one change"),
        ({"arms": {"recipe": [], "a": ["graph.top_k=5"], "b": ["graph.top_k=5"]}}, "same change"),
        ({"arms": {"recipe": [], "dates": ["data.test_end=2025-06-30"]}}, "window or the launcher"),
        ({"arms": {"recipe": [], "name": ["experiment_name=x"]}}, "window or the launcher"),
        ({"arms": {"recipe": [], "stamp": ["protocol.window=W2"]}}, "window or the launcher"),
        ({"arms": {"recipe": [], "flag": ["graph.top_k"]}}, "not a Hydra override"),
        ({"arms": {"recipe": [], "Bad-Name": ["graph.top_k=5"]}}, "arm name"),
        ({"windows": ["W5"]}, "all five"),
        ({"windows": ["W1", "W2", "W3", "W4", "W5", "holdout"]}, "all five"),
        ({"primary_metric": "rank_ic"}, "fixed"),
        ({"alpha": 0.10}, "never loosen"),
        ({"alpha": 0}, "never loosen"),
        ({"alpha": "0.05"}, "number"),
        ({"seed_base": "1729"}, "integer"),
        ({"decision_rule": {"adjustment": "bh", "seed_floor_multiple": 1.0}}, "stricter"),
        ({"decision_rule": {"adjustment": "none", "seed_floor_multiple": 1.0}}, "stricter"),
        ({"decision_rule": {"adjustment": "holm", "seed_floor_multiple": 0.5}}, "never loosen"),
        ({"decision_rule": {"adjustment": "holm"}}, "missing"),
        ({"decision_rule": {"adjustment": "holm", "seed_floor_multiple": 1, "x": 1}}, "unknown"),
    ],
)
def test_a_malformed_or_looser_preregistration_is_refused(change, message):
    with pytest.raises(PreregistrationError, match=message):
        Preregistration.from_mapping(_valid(**change), expected_id="graph-zeroed-control")


def test_a_missing_or_unknown_field_is_refused():
    data = _valid()
    del data["hypothesis"]
    with pytest.raises(PreregistrationError, match="missing \\['hypothesis'\\]"):
        _prereg(data)
    with pytest.raises(PreregistrationError, match="unknown keys \\['notes'\\]"):
        _prereg(_valid(notes="free text"))


def test_a_stricter_rule_is_accepted():
    prereg = _prereg(
        _valid(alpha=0.01, decision_rule={"adjustment": "bonferroni", "seed_floor_multiple": 2})
    )
    assert (prereg.alpha, prereg.decision_rule.adjustment) == (0.01, "bonferroni")
    assert prereg.decision_rule.seed_floor_multiple == 2.0


# -- Loading at the commit -------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A checkout holding the recipe and nothing committed under experiments/ yet."""
    root = tmp_path / "repo"
    (root / "docs" / "research" / "experiments").mkdir(parents=True)
    (root / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md").write_text(RECIPE_TEXT, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "recipe")
    return root


def _write(repo: Path, data: dict) -> Path:
    path = repo / "docs" / "research" / "experiments" / f"{data['id']}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _commit(repo: Path) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "register")


def test_a_committed_preregistration_loads_with_its_digest_and_commit(repo):
    path = _write(repo, _template())
    _commit(repo)
    loaded = load_preregistration("graph-zeroed-control", repo)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded.commit == head
    assert loaded.path == "docs/research/experiments/graph-zeroed-control.yaml"
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_missing_preregistration_is_refused(repo):
    with pytest.raises(PreregistrationError, match="does not exist"):
        load_preregistration("graph-zeroed-control", repo)


def test_an_uncommitted_preregistration_is_refused(repo):
    _write(repo, _template())
    with pytest.raises(PreregistrationError, match="not committed"):
        load_preregistration("graph-zeroed-control", repo)


def test_a_preregistration_edited_after_its_commit_is_refused(repo):
    _write(repo, _template())
    _commit(repo)
    _write(repo, _valid(hypothesis="Rewritten after the results came in."))
    with pytest.raises(PreregistrationError, match="differs from the commit"):
        load_preregistration("graph-zeroed-control", repo)


def test_an_experiment_id_that_is_a_path_is_refused(repo):
    with pytest.raises(PreregistrationError, match="not an id"):
        load_preregistration("../../secrets", repo)


# -- Checking a run against its declaration --------------------------------------------


@pytest.mark.parametrize("arm", ["recipe", "graph_zeroed"])
@pytest.mark.parametrize("window", ["W1", "W5"])
def test_a_declared_arm_on_a_declared_window_is_accepted(arm, window):
    prereg = _prereg()
    _check(prereg, _run_config(prereg, arm, window), arm, window)


def test_the_declared_run_carries_the_windows_dates_and_the_seed_base():
    resolved = _run_config(_prereg(), "recipe", "W2")
    spans = {key: resolved["data"][key.removeprefix("data.")] for key in WINDOWS["W2"].spans()}
    assert spans == WINDOWS["W2"].spans()
    assert resolved["seed"] == 1729


def test_an_undeclared_arm_is_refused():
    prereg = _prereg()
    with pytest.raises(PreregistrationError, match="protocol.arm='levels_inputs' is not declared"):
        _check(prereg, _run_config(prereg, "recipe", "W1"), "levels_inputs", "W1")


@pytest.mark.parametrize("window", ["W6", "holdout", None])
def test_an_undeclared_window_is_refused(window):
    prereg = _prereg()
    with pytest.raises(PreregistrationError, match="is not declared"):
        _check(prereg, _run_config(prereg, "recipe", "W1"), "recipe", window)


def test_a_run_with_a_second_change_is_refused():
    prereg = _prereg()
    resolved = _run_config(prereg, "graph_zeroed", "W3", "graph.drop_edge_p=0.2")
    with pytest.raises(PreregistrationError, match="graph.drop_edge_p=0.2 \\(declared 0.1\\)"):
        _check(prereg, resolved, "graph_zeroed", "W3")


def test_a_run_labelled_with_the_wrong_arm_is_refused():
    """The treatment's change on a run stamped as the control."""
    prereg = _prereg()
    resolved = _run_config(prereg, "graph_zeroed", "W3")
    with pytest.raises(PreregistrationError, match="graph.zero_edges=True"):
        _check(prereg, resolved, "recipe", "W3")


def test_a_run_on_another_windows_dates_is_refused():
    prereg = _prereg()
    resolved = _run_config(prereg, "recipe", "W2")
    with pytest.raises(PreregistrationError, match="data.test_end='2022-12-31'"):
        _check(prereg, resolved, "recipe", "W3")


def test_a_run_on_another_seed_base_is_refused():
    prereg = _prereg()
    resolved = _run_config(prereg, "recipe", "W1", "seed=2729")
    with pytest.raises(PreregistrationError, match="seed=2729"):
        _check(prereg, resolved, "recipe", "W1")


def test_the_declared_seed_base_is_the_one_checked():
    """A base other than the recipe's: its runs pass, a run on the recipe's 1729 does not."""
    prereg = _prereg(_valid(seed_base=4729))
    _check(prereg, _run_config(prereg, "recipe", "W1"), "recipe", "W1")
    on_recipe_seed = _run_config(prereg, "recipe", "W1", "seed=1729")
    with pytest.raises(PreregistrationError, match="seed=1729 \\(declared 4729\\)"):
        _check(prereg, on_recipe_seed, "recipe", "W1")


def test_a_seed_floor_arm_may_change_the_seed():
    prereg = _prereg(
        _valid(arms={"recipe": [], "other_seeds": ["seed=2729"]}, id="graph-zeroed-control")
    )
    _check(prereg, _run_config(prereg, "other_seeds", "W4"), "other_seeds", "W4")


def test_the_launcher_keys_are_free():
    """Where a run writes and how it records itself are not part of the arm."""
    prereg = _prereg()
    resolved = _run_config(
        prereg, "recipe", "W1", "output_dir=/elsewhere", "data.auxiliary_snapshot_mode=replay"
    )
    _check(prereg, resolved, "recipe", "W1")


def test_a_changed_recipe_is_refused():
    prereg = _prereg(_valid(recipe="an-older-recipe-slug"))
    with pytest.raises(PreregistrationError, match="needs a new pre-registration"):
        _check(prereg, _run_config(prereg, "recipe", "W1"), "recipe", "W1")


# -- The standing gate -----------------------------------------------------------------


def test_an_exploratory_run_is_stamped_exploratory(repo):
    stamp = resolve_run_standing(
        ProtocolConfig(exploratory=True), {}, repo_root=repo, compose=_compose
    )
    assert stamp == {"exploratory": True, "experiment_id": None, "arm": None, "window": None}


def test_a_run_with_no_standing_is_refused(repo):
    with pytest.raises(PreregistrationError, match="names no pre-registration"):
        resolve_run_standing(ProtocolConfig(), {}, repo_root=repo, compose=_compose)


def test_a_declared_run_is_stamped_with_its_declaration(repo):
    _write(repo, _template())
    _commit(repo)
    prereg = _prereg()
    protocol = ProtocolConfig(experiment_id=prereg.id, arm="graph_zeroed", window="W2")
    stamp = resolve_run_standing(
        protocol, _run_config(prereg, "graph_zeroed", "W2"), repo_root=repo, compose=_compose
    )
    loaded = load_preregistration(prereg.id, repo)
    assert stamp == {
        "exploratory": False,
        "experiment_id": "graph-zeroed-control",
        "arm": "graph_zeroed",
        "window": "W2",
        "preregistration_path": loaded.path,
        "preregistration_sha256": loaded.sha256,
        "preregistration_commit": loaded.commit,
    }


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"exploratory": True, "experiment_id": "x", "arm": "a", "window": "W1"}, "two standings"),
        ({"experiment_id": "x"}, "needs protocol.arm"),
        ({"experiment_id": "x", "arm": "a"}, "needs protocol.arm"),
        ({"arm": "a"}, "need a protocol.experiment_id"),
        ({"window": "W1", "exploratory": True}, "need a protocol.experiment_id"),
        ({"exploratory": "yes"}, "true or false"),
    ],
)
def test_an_incoherent_protocol_section_is_refused(fields, message):
    with pytest.raises(ValueError, match=message):
        ProtocolConfig(**fields)


def test_a_saved_run_without_a_protocol_section_still_loads():
    """Rescoring rebuilds old runs' configs; only the runner demands a standing."""
    config = create_config_from_dict({"protocol": {"holdout_promotion": None}})
    assert config.protocol == ProtocolConfig()


# -- The runner ------------------------------------------------------------------------

RUNNER_OVERRIDES = [
    "features=base",
    "data.source=csv",
    "data.use_pit_universe=false",
    "model.his_t=3",
    "model.label_t=2",
    "model.gru_hidden_sizes=[8,4]",
    "model.hidden_size_gat1=8",
    "model.output_gat1=4",
    "model.gat_heads=1",
    "model.hidden_size_gat2=8",
    "model.num_hidden_states=4",
    "model.cross_attn_heads=1",
    "model.use_self_attention=false",
    "training.num_epochs=1",
    "training.num_models=1",
    "training.batch_size=2",
    "training.warmup_steps=0",
    "tracking.enabled=false",
]
#: A two-month stand-in for W1, so a declared run trains in seconds.
TINY_W1 = Window(
    "W1", "2020-01-01", "2020-02-14", "2020-02-18", "2020-02-28", "2020-03-03", "2020-03-20"
)


def _write_panel(path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=60)
    rows = []
    for s_idx, stock in enumerate(["A.N", "B.N", "C.N", "D.N"]):
        for d_idx, dt in enumerate(dates):
            price = 100.0 + s_idx + 0.5 * np.sin(d_idx / 4.0 + s_idx)
            rows.append(
                {
                    "kdcode": stock,
                    "dt": dt.strftime("%Y-%m-%d"),
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price * (1.0 + 0.001 * np.cos(d_idx / 7.0 + s_idx)),
                    "volume": 1_000_000.0 + d_idx,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


@contextmanager
def _restored_process_state(directory: Path):
    root = logging.getLogger()
    handlers, level, cwd = list(root.handlers), root.level, os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(cwd)
        for handler in root.handlers:
            if handler not in handlers:
                handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)


def _runner(out: Path, overrides: list[str]) -> None:
    out.mkdir(parents=True)
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose("config", overrides=overrides)
    with _restored_process_state(out):
        run_experiment.main.__wrapped__(cfg)


def test_the_runner_refuses_a_run_with_no_standing_before_writing_anything(tmp_path):
    panel = tmp_path / "panel.csv"
    _write_panel(panel)
    out = tmp_path / "out"
    with pytest.raises(PreregistrationError, match="names no pre-registration"):
        _runner(out, [*RUNNER_OVERRIDES, f"data.filename={panel.as_posix()}"])
    assert not (out / "config.yaml").exists()
    assert not list(out.rglob("run_metadata.json"))


def test_the_runner_stamps_a_declared_run(tmp_path, repo, monkeypatch):
    """A tiny recipe and a stand-in W1 in a scratch checkout, through the real runner."""
    panel = tmp_path / "panel.csv"
    _write_panel(panel)
    slug = "tiny-recipe"
    tiny_recipe = (
        f"Recipe slug:\n\n```text\n{slug}\n```\n\n## Hydra Overrides\n\n```text\n"
        + "\n".join([*RUNNER_OVERRIDES, f"data.filename={panel.as_posix()}"])
        + "\n```\n"
    )
    (repo / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md").write_text(tiny_recipe, encoding="utf-8")
    _write(repo, _valid(recipe=slug, arms={"recipe": [], "no_drop": ["graph.drop_edge_p=0.0"]}))
    _commit(repo)
    monkeypatch.setitem(preregistration.WINDOWS, "W1", TINY_W1)
    monkeypatch.setattr(run_experiment, "REPO_ROOT", repo)

    out = tmp_path / "out"
    run = [
        *RUNNER_OVERRIDES,
        f"data.filename={panel.as_posix()}",
        "graph.drop_edge_p=0.0",
        *(f"{key}={value}" for key, value in TINY_W1.spans().items()),
        "protocol.experiment_id=graph-zeroed-control",
        "protocol.arm=no_drop",
        "protocol.window=W1",
    ]
    _runner(out, run)
    metadata = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    loaded = load_preregistration("graph-zeroed-control", repo)
    assert metadata["protocol"] == {
        "exploratory": False,
        "experiment_id": "graph-zeroed-control",
        "arm": "no_drop",
        "window": "W1",
        "preregistration_path": loaded.path,
        "preregistration_sha256": loaded.sha256,
        "preregistration_commit": loaded.commit,
    }

    # Control: the same run stamped as the control arm is refused before it trains.
    refused = tmp_path / "refused"
    with pytest.raises(PreregistrationError, match="graph.drop_edge_p"):
        _runner(refused, [*run[:-2], "protocol.arm=recipe", "protocol.window=W1"])
    assert not list(refused.rglob("run_metadata.json"))


def test_the_runner_stamps_an_exploratory_run(tmp_path):
    panel = tmp_path / "panel.csv"
    _write_panel(panel)
    out = tmp_path / "out"
    dates = [f"{key}={value}" for key, value in TINY_W1.spans().items()]
    _runner(
        out,
        [
            *RUNNER_OVERRIDES,
            f"data.filename={panel.as_posix()}",
            *dates,
            "protocol.exploratory=true",
        ],
    )
    metadata = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["protocol"]["exploratory"] is True
    assert metadata["protocol"]["experiment_id"] is None
