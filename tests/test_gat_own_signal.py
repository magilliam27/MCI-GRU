"""Graph layers must let a linked stock be scored on its own inputs.

``GATBlock`` is two PyG ``GATConv`` (GAT v1) layers. In GAT v1 a node's own
features reach its output only as a self-loop neighbour, and the attention term
that depends on the target node cancels inside the softmax. So nodes with the
same closed neighbourhood get the same output whatever their own inputs. In the
first run (``20261010T044328Z-full-ec77f4b7``) GOOG/GOOGL, HD/LOW and MA/V
tied exactly on every date checked (PRD #319, problem A).

``model.gat_layer_form=residual`` gives both layers of every ``GATBlock`` an
own-feature path (``GATConv(residual=True)``): the final head, the A2 graph
stream and the sector stream (#319 decisions D2 and D3). The base default stays
``legacy`` so first-run checkpoints still load.

These tests assert behaviour (outputs respond to a node's own input), at the
block, model and config seams named in #319. They do not assert on attention
weights or on how the own-feature path is built.
"""

import re
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import ModelConfig, create_config_from_dict
from mci_gru.graph.utils import edge_feature_dim
from mci_gru.models import create_model
from mci_gru.models.graph import GATBlock

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
RECIPE = REPO_ROOT / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md"
OVERRIDE_BLOCK = re.compile(r"^## Hydra Overrides\n+```text\n(.*?)^```", re.M | re.S)

# Multi-feature edges, as the recipe builds them: corr, |corr|, corr^2, sign flag.
EDGE_DIM = 4
_EDGE = [0.95, 0.95, 0.9025, 0.0]
SEEDS = range(5)
# Float32 round-off between two nodes that sum the same terms in a different
# order is about 1e-7; a real difference here is 1e-2 or more.
TIE_ATOL = 1e-6
DIFFERENT = 1e-3


def _graph(
    pairs: list[tuple[int, int]], edge_dim: int = EDGE_DIM
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undirected edges with identical weights, as a linked group has in a static graph."""
    src = [a for a, b in pairs] + [b for a, b in pairs]
    dst = [b for a, b in pairs] + [a for a, b in pairs]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor([_EDGE[:edge_dim]] * edge_index.shape[1])
    return edge_index, edge_weight


PAIR = _graph([(0, 1)])
TRIANGLE = _graph([(0, 1), (1, 2), (0, 2)])


def _gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


# --- Block seam -------------------------------------------------------------


def _block(residual: bool, seed: int, out_channels: int) -> GATBlock:
    torch.manual_seed(seed)
    # The legacy block is built without the new argument, so these tests also
    # pin that leaving it out means legacy.
    kwargs = {"residual": True} if residual else {}
    block = GATBlock(
        in_channels=8,
        hidden=16,
        out_channels=out_channels,
        heads=4,
        activation="elu",
        edge_feature_dim=EDGE_DIM,
        **kwargs,
    )
    return block.eval()


def _pair_counterfactual(block, in_channels: int, seed: int):
    """Outputs for a linked pair, before and after redrawing node 0's input alone."""
    edge_index, edge_weight = _graph([(0, 1)], block.edge_feature_dim)
    generator = torch.Generator().manual_seed(100 + seed)
    x = torch.randn(2, in_channels, generator=generator)
    moved = x.clone()
    moved[0] = torch.randn(in_channels, generator=generator)
    with torch.no_grad():
        before = block(x, edge_index, edge_weight)
        after = block(moved, edge_index, edge_weight)
    return before, after


def _linked_share(block, x: torch.Tensor, pairs, i: int, j: int) -> float:
    """How much of the difference between nodes i and j survives linking them.

    The same block, inputs and weights are run twice: with the group's edges,
    and with no edges (each node then sees only itself). 1 means linking
    changed nothing; 0 means the link erased the difference entirely.
    """
    edge_dim = block.edge_feature_dim
    unlinked = (torch.empty(2, 0, dtype=torch.long), torch.empty(0, edge_dim))
    with torch.no_grad():
        linked_out = block(x, *_graph(pairs, edge_dim))
        unlinked_out = block(x, *unlinked)
    return _gap(linked_out[i], linked_out[j]) / _gap(unlinked_out[i], unlinked_out[j])


# GAT v1's target term cancels in the softmax except through the LeakyReLU's
# kink, so a linked pair ties exactly while no attention logit changes sign (as
# GOOG/GOOGL did in the first run) and nearly ties otherwise. Measured over 50
# seeds: legacy keeps at most 0.028 of the unlinked difference.
LEGACY_SHARE = 0.05


@pytest.mark.parametrize("out_channels", [1, 4])
def test_legacy_block_ties_a_linked_pair_whatever_their_inputs(out_channels: int) -> None:
    """Guard: the fixture reproduces the bug, so the tests below can fail.

    Two nodes linked only to each other, with different inputs, (nearly) tie,
    before and after one node's input is redrawn.
    """
    for seed in SEEDS:
        block = _block(False, seed, out_channels)
        generator = torch.Generator().manual_seed(100 + seed)
        x = torch.randn(2, 8, generator=generator)
        moved = x.clone()
        moved[0] = torch.randn(8, generator=generator)
        assert _linked_share(block, x, [(0, 1)], 0, 1) < LEGACY_SHARE
        assert _linked_share(block, moved, [(0, 1)], 0, 1) < LEGACY_SHARE
        before, after = _pair_counterfactual(block, 8, seed)
        assert _gap(after[0], before[0]) > DIFFERENT, "the perturbation moved nothing"


@pytest.mark.parametrize("out_channels", [1, 4])
def test_residual_block_scores_a_linked_pair_on_its_own_inputs(out_channels: int) -> None:
    for seed in SEEDS:
        before, after = _pair_counterfactual(_block(True, seed, out_channels), 8, seed)
        assert _gap(before[0], before[1]) > DIFFERENT, f"seed {seed}: linked pair tied"
        assert _gap(after[0], before[0]) > DIFFERENT, f"seed {seed}: own input ignored"
        # The moved node must not carry its partner to the same value.
        assert _gap(after[0], after[1]) > DIFFERENT, f"seed {seed}: partner moved in lockstep"
        assert _gap(after[0] - before[0], after[1] - before[1]) > DIFFERENT


def _triangle_outputs(block) -> torch.Tensor:
    edge_index, edge_weight = TRIANGLE
    x = torch.randn(3, 8, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        return block(x, edge_index, edge_weight)


@pytest.mark.parametrize("out_channels", [1, 4])
def test_legacy_block_ties_every_node_of_a_linked_triangle(out_channels: int) -> None:
    """Guard: ties are not only a two-node effect.

    Every node of a fully linked group has the same closed neighbourhood, as
    a tight cluster of banks or utilities does in a high-threshold graph.
    """
    triangle = [(0, 1), (1, 2), (0, 2)]
    x = torch.randn(3, 8, generator=torch.Generator().manual_seed(7))
    for seed in SEEDS:
        block = _block(False, seed, out_channels)
        for i, j in ((0, 1), (1, 2), (0, 2)):
            assert _linked_share(block, x, triangle, i, j) < LEGACY_SHARE


@pytest.mark.parametrize("out_channels", [1, 4])
def test_residual_block_separates_every_node_of_a_linked_triangle(out_channels: int) -> None:
    for seed in SEEDS:
        out = _triangle_outputs(_block(True, seed, out_channels))
        for i, j in ((0, 1), (1, 2), (0, 2)):
            assert _gap(out[i], out[j]) > DIFFERENT, f"seed {seed}: nodes {i} and {j} tied"


# --- Model seam -------------------------------------------------------------

N_STOCKS, HIS_T, N_FEATURES = 12, 10, 8


def _recipe_overrides() -> list[str]:
    block = OVERRIDE_BLOCK.search(RECIPE.read_text(encoding="utf-8"))
    assert block, "the recipe has no Hydra override block"
    return [line.strip() for line in block.group(1).splitlines() if line.strip()]


def _recipe_model(gat_layer_form: str, **extra) -> torch.nn.Module:
    """The recipe's model with the given graph-layer form, built as run_experiment.py builds it."""
    overrides = [*_recipe_overrides(), f"model.gat_layer_form={gat_layer_form}"]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    config = create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))
    model_cfg = {
        **config.model.to_dict(),
        "edge_feature_dim": edge_feature_dim(config.graph),
        "drop_edge_p": config.graph.drop_edge_p,
        "isolate_edge_dropout_rng": config.graph.isolate_edge_dropout_rng,
        "use_sector_relation": config.graph.use_sector_relation,
        **extra,
    }
    assert model_cfg["edge_feature_dim"] == EDGE_DIM
    return create_model(N_FEATURES, model_cfg).eval()


def _scores(model, x_ts: torch.Tensor, x_graph: torch.Tensor | None = None) -> torch.Tensor:
    edge_index, edge_weight = PAIR
    if x_graph is None:
        x_graph = x_ts[0, :, -1, :].clone()
    with torch.no_grad():
        return model(x_ts, x_graph, edge_index, edge_weight, N_STOCKS)[0]


def test_legacy_recipe_model_ties_a_linked_pair() -> None:
    """Guard: the recipe's model reproduces the first run's pair ties."""
    torch.manual_seed(0)
    model = _recipe_model("legacy")
    x_ts = torch.randn(1, N_STOCKS, HIS_T, N_FEATURES)
    scores = _scores(model, x_ts)
    assert abs(scores[0] - scores[1]) < TIE_ATOL
    assert abs(scores[0] - scores[2]) > DIFFERENT, "control: unlinked stocks should not tie"


def test_residual_recipe_model_scores_a_linked_pair_on_its_own_inputs() -> None:
    """User story 1 and 2 of #319, through ``create_model`` with the recipe's settings."""
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = _recipe_model("residual")
        x_ts = torch.randn(1, N_STOCKS, HIS_T, N_FEATURES)
        before = _scores(model, x_ts)
        assert abs(before[0] - before[1]) > DIFFERENT, f"seed {seed}: linked pair tied"

        moved = x_ts.clone()
        moved[0, 0] = torch.randn(HIS_T, N_FEATURES)
        after = _scores(model, moved)
        own_shift = after[0] - before[0]
        partner_shift = after[1] - before[1]
        assert abs(own_shift) > DIFFERENT, f"seed {seed}: own input ignored"
        assert abs(own_shift - partner_shift) > DIFFERENT, f"seed {seed}: partner in lockstep"


@pytest.mark.parametrize(("form", "ties"), [("legacy", True), ("residual", False)])
def test_graph_stream_carries_each_stocks_own_graph_features(form: str, ties: bool) -> None:
    """The A2 graph stream, not only the head, must keep a linked stock's own signal.

    The pair gets identical price histories and different graph-node features,
    so only the A2 stream can tell them apart. If A2 averages them away, ``z`` is
    the same for both and even a fixed head must tie them. Fixing the head alone
    therefore fails the residual case; that is the point of D3 in #319.
    """
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = _recipe_model(form)
        x_ts = torch.randn(1, N_STOCKS, HIS_T, N_FEATURES)
        x_ts[0, 1] = x_ts[0, 0]
        x_graph = x_ts[0, :, -1, :].clone()
        x_graph[1] = torch.randn(N_FEATURES)
        scores = _scores(model, x_ts, x_graph)
        if ties:
            assert abs(scores[0] - scores[1]) < TIE_ATOL
        else:
            assert abs(scores[0] - scores[1]) > DIFFERENT, f"seed {seed}: A2 averaged the pair"


def _graph_blocks(model) -> dict[str, GATBlock]:
    return {name: m for name, m in model.named_modules() if isinstance(m, GATBlock)}


@pytest.mark.parametrize(("form", "ties"), [("legacy", True), ("residual", False)])
def test_every_graph_block_the_model_builds_keeps_own_signal(form: str, ties: bool) -> None:
    """Final head, A2 graph stream and sector stream all get the chosen form.

    The sector stream is built only with ``use_sector_relation``, so it is
    switched on here. Each block is probed with the pair counterfactual at its
    own input width.
    """
    model = _recipe_model(form, use_sector_relation=True)
    blocks = _graph_blocks(model)
    assert set(blocks) == {"gat_layer", "gat_layer_sector", "final_gat"}
    for name, block in blocks.items():
        before, after = _pair_counterfactual(block, block.gat1.in_channels, seed=0)
        if ties:
            x = torch.randn(2, block.gat1.in_channels, generator=torch.Generator().manual_seed(3))
            assert _linked_share(block, x, [(0, 1)], 0, 1) < LEGACY_SHARE, name
        else:
            assert _gap(before[0], before[1]) > DIFFERENT, f"{name}: linked pair tied"
            assert _gap(after[0], after[1]) > DIFFERENT, f"{name}: partner in lockstep"


# --- Config and checkpoint seam ---------------------------------------------

_SMALL_MODEL_CONFIG = {
    "gru_hidden_sizes": [4, 4],
    "hidden_size_gat1": 8,
    "output_gat1": 4,
    "gat_heads": 2,
    "hidden_size_gat2": 8,
    "num_hidden_states": 4,
    "cross_attn_heads": 2,
    "use_multi_scale": False,
    "use_self_attention": True,
    "activation": "relu",
    "temporal_encoder": "legacy",
    "edge_feature_dim": EDGE_DIM,
    "use_sector_relation": True,
}

# Measured on origin/main at ecc1021, before this change, for _SMALL_MODEL_CONFIG
# with input_size=8 (torch-geometric 2.8.0, the locked version). Literals, so a
# regression in the default path cannot appear on both sides and cancel out.
_LEGACY_STATE_DICT_KEY_COUNT = 94
_LEGACY_PARAMETER_COUNT = 5576
_LEGACY_GAT_PARAMETER_SUFFIXES = {
    "att_dst",
    "att_edge",
    "att_src",
    "bias",
    "lin.weight",
    "lin_edge.weight",
}
_GRAPH_BLOCKS = ("gat_layer", "gat_layer_sector", "final_gat")
_LEGACY_GAT_KEYS = {
    f"{block}.{layer}.{suffix}"
    for block in _GRAPH_BLOCKS
    for layer in ("gat1", "gat2")
    for suffix in _LEGACY_GAT_PARAMETER_SUFFIXES
}


def _gat_keys(state_dict) -> set[str]:
    return {k for k in state_dict if k.startswith(_GRAPH_BLOCKS)}


def test_default_config_keeps_the_pre_change_checkpoint_shape() -> None:
    """First-run checkpoints must keep loading, so the default may not move."""
    model = create_model(8, dict(_SMALL_MODEL_CONFIG))
    state_dict = model.state_dict()

    assert len(state_dict) == _LEGACY_STATE_DICT_KEY_COUNT
    assert sum(p.numel() for p in model.parameters()) == _LEGACY_PARAMETER_COUNT
    assert _gat_keys(state_dict) == _LEGACY_GAT_KEYS


def test_explicit_legacy_form_keeps_the_pre_change_checkpoint_shape() -> None:
    model = create_model(8, {**_SMALL_MODEL_CONFIG, "gat_layer_form": "legacy"})
    assert _gat_keys(model.state_dict()) == _LEGACY_GAT_KEYS


def test_residual_form_adds_an_own_feature_map_to_every_graph_layer() -> None:
    """The residual form is a different parameter set, so it is a different checkpoint."""
    model = create_model(8, {**_SMALL_MODEL_CONFIG, "gat_layer_form": "residual"})
    added = _gat_keys(model.state_dict()) - _LEGACY_GAT_KEYS

    assert _gat_keys(model.state_dict()) > _LEGACY_GAT_KEYS
    assert added == {f"{b}.{layer}.res.weight" for b in _GRAPH_BLOCKS for layer in ("gat1", "gat2")}


def test_a_legacy_checkpoint_does_not_load_into_the_residual_form() -> None:
    """A silent partial load would leave the own-feature maps at their random init."""
    legacy = create_model(8, dict(_SMALL_MODEL_CONFIG)).state_dict()
    residual = create_model(8, {**_SMALL_MODEL_CONFIG, "gat_layer_form": "residual"})

    with pytest.raises(RuntimeError, match=r"Missing key\(s\).*res\.weight"):
        residual.load_state_dict(legacy)


def test_a_residual_checkpoint_does_not_load_into_the_legacy_form() -> None:
    residual = create_model(8, {**_SMALL_MODEL_CONFIG, "gat_layer_form": "residual"}).state_dict()
    legacy = create_model(8, dict(_SMALL_MODEL_CONFIG))

    with pytest.raises(RuntimeError, match=r"Unexpected key\(s\).*res\.weight"):
        legacy.load_state_dict(residual)


def test_create_model_rejects_an_unknown_gat_layer_form() -> None:
    """``create_model`` also takes a checkpoint's plain ``config.yaml`` dict,
    which never passes through ``ModelConfig``, so a misspelt form must not
    quietly build legacy."""
    with pytest.raises(ValueError, match="gat_layer_form"):
        create_model(8, {**_SMALL_MODEL_CONFIG, "gat_layer_form": "Residual"})


def test_model_config_rejects_an_unknown_gat_layer_form() -> None:
    with pytest.raises(ValueError, match="gat_layer_form"):
        ModelConfig(gat_layer_form="gatv2")


def test_model_config_round_trips_the_gat_layer_form() -> None:
    assert ModelConfig().to_dict()["gat_layer_form"] == "legacy"
    serialised = ModelConfig(gat_layer_form="residual").to_dict()
    assert serialised["gat_layer_form"] == "residual"
    assert ModelConfig(**serialised).gat_layer_form == "residual"


def test_base_config_yaml_ships_the_legacy_graph_layers() -> None:
    """Real runs take the default from ``configs/config.yaml``, not from ``ModelConfig``."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=[])
    assert cfg.model.gat_layer_form == "legacy"


def test_residual_form_reaches_the_built_model_from_hydra() -> None:
    """The override must survive Hydra, ``ExperimentConfig`` and the factory."""
    model = _recipe_model("residual")
    assert all(
        k in model.state_dict() for k in ("final_gat.gat1.res.weight", "gat_layer.gat2.res.weight")
    )


def test_dependency_floor_has_gatconv_residual() -> None:
    """``GATConv(residual=...)`` first shipped in torch-geometric 2.6.0.

    Below that, building any ``GATBlock`` raises, legacy included, because the
    argument is always passed. ``requirements.txt`` (Colab) must agree with
    ``pyproject.toml``.
    """
    pattern = re.compile(r"torch-geometric>=(\d+)\.(\d+)")
    for path in ("pyproject.toml", "requirements.txt"):
        match = pattern.search((REPO_ROOT / path).read_text(encoding="utf-8"))
        assert match, f"{path} has no torch-geometric floor"
        assert (int(match.group(1)), int(match.group(2))) >= (2, 6), path
