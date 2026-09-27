"""Cross-stock block behaviour: cross-section survival, masking, and compatibility.

The shipped ``SelfAttention`` block replaces ``z`` with its output. With
near-uniform attention that hands every stock the same cross-sectional average,
so almost nothing that distinguished one stock from another reaches the score
head. Issue #197 measures it retaining 0.024 of a 0.433 cross-sectional
variance share on real data.

These tests specify the behaviour a cross-stock block must have, at the module
and model seams named in that issue. They do not assert on attention weights,
entropy, or effective rank: those are diagnostics, and asserting on them would
couple the suite to the implementation.
"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from mci_gru.config import ModelConfig
from mci_gru.models import SelfAttention, create_model
from mci_gru.models.attention import ResidualCrossSectionBlock

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

_BASE_MODEL_CONFIG = {
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
}


def _cross_sectional_variance_ratio(block, num_stocks: int, embed_dim: int, seed: int) -> float:
    """Fraction of across-stock variance that survives the block.

    A block that returns the same vector for every stock scores ~0. A block
    that carries each stock's own representation through scores ~1.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, num_stocks, embed_dim, generator=generator)
    y = block(x)
    return (y.var(dim=1).mean() / x.var(dim=1).mean()).item()


@pytest.mark.parametrize("num_stocks", [20, 110, 201])
def test_residual_block_preserves_the_cross_section(num_stocks: int) -> None:
    """Most of the across-stock variation must survive the block.

    Stock counts span the real universe: ``gics_top10_110_2016`` carries about
    110 admissible names on a 201-node PIT union axis.
    """
    torch.manual_seed(0)
    block = ResidualCrossSectionBlock(SelfAttention(embed_dim=16, align_dim=4), embed_dim=16)

    retained = _cross_sectional_variance_ratio(block, num_stocks, embed_dim=16, seed=100)

    assert retained > 0.5


def test_residual_block_zeroes_inactive_nodes() -> None:
    """PIT-inactive names must leave the block at exactly zero.

    ``SelfAttention`` already guarantees this for its own output. The residual
    add reintroduces the raw input, so the wrapper has to re-apply the mask or
    inactive union nodes carry values into the score head.
    """
    torch.manual_seed(0)
    block = ResidualCrossSectionBlock(SelfAttention(embed_dim=8, align_dim=2), embed_dim=8)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])

    out = block(x, stock_mask=mask)

    assert torch.all(out[:, 2:, :] == 0)


def test_residual_block_inactive_node_cannot_influence_active_ones() -> None:
    """Changing a PIT-inactive name must not move any active name."""
    torch.manual_seed(0)
    block = ResidualCrossSectionBlock(SelfAttention(embed_dim=8, align_dim=2), embed_dim=8)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, True, False]])

    before = block(x, stock_mask=mask)
    changed = x.clone()
    changed[:, 3, :] = 999.0
    after = block(changed, stock_mask=mask)

    assert torch.allclose(before[:, :3, :], after[:, :3, :], atol=1e-6)


def test_residual_block_active_nodes_influence_each_other() -> None:
    """Positive control for the test above: moving other active names moves this one.

    An inert attention branch, multiplied by zero or fed a constant, turns the
    block into a masked identity. That passes every isolation, retention, and
    offset test in this file trivially, because an identity lets nothing
    influence anything. This test fails for it.
    """
    torch.manual_seed(0)
    block = ResidualCrossSectionBlock(SelfAttention(embed_dim=8, align_dim=2), embed_dim=8)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, True, False]])

    before = block(x, stock_mask=mask)
    changed = x.clone()
    generator = torch.Generator().manual_seed(7)
    changed[:, 1:3, :] = 3.0 * torch.randn(1, 2, 8, generator=generator)
    after = block(changed, stock_mask=mask)

    assert torch.equal(changed[:, 0, :], x[:, 0, :])
    # Measured at about 0.33; an inert branch moves it by exactly 0.
    assert (after[:, 0, :] - before[:, 0, :]).abs().max() > 1e-2


# Measured on origin/main at 125abda, before this change, for _BASE_MODEL_CONFIG with
# input_size=8. Pinned literals rather than a value recomputed from today's code, so
# this guard can actually disagree with a regression in the default path.
_LEGACY_STATE_DICT_KEY_COUNT = 80
_LEGACY_PARAMETER_COUNT = 5137
_LEGACY_CROSS_SECTION_KEYS = {
    "self_attention.W_k.bias",
    "self_attention.W_k.weight",
    "self_attention.W_q.bias",
    "self_attention.W_q.weight",
    "self_attention.W_v.bias",
    "self_attention.W_v.weight",
}


def test_default_config_keeps_the_pre_change_checkpoint_shape() -> None:
    """Checkpoints saved before this change must keep loading, so the default may not move.

    The expected values are literals measured on ``origin/main`` before this
    change, not quantities recomputed the way the code computes them.
    """
    model = create_model(8, dict(_BASE_MODEL_CONFIG))
    state_dict = model.state_dict()

    assert len(state_dict) == _LEGACY_STATE_DICT_KEY_COUNT
    assert sum(p.numel() for p in model.parameters()) == _LEGACY_PARAMETER_COUNT
    assert {k for k in state_dict if "self_attention" in k} == _LEGACY_CROSS_SECTION_KEYS


def test_config_without_the_new_key_keeps_the_pre_change_checkpoint_shape() -> None:
    """A ``config.yaml`` written before this change has no such key at all."""
    legacy_only = dict(_BASE_MODEL_CONFIG)
    assert "cross_section_block" not in legacy_only

    state_dict = create_model(8, legacy_only).state_dict()

    assert {k for k in state_dict if "self_attention" in k} == _LEGACY_CROSS_SECTION_KEYS


def test_residual_flag_changes_the_cross_section_parameters() -> None:
    """Selecting the residual form must actually reach the built model.

    The wrapper nests the attention under ``inner`` and adds a norm, so the two
    forms have disjoint parameter names. A checkpoint therefore belongs to the
    form that produced it; they are not interchangeable.
    """
    model = create_model(8, {**_BASE_MODEL_CONFIG, "cross_section_block": "residual"})

    keys = {k for k in model.state_dict() if "self_attention" in k}

    assert keys.isdisjoint(_LEGACY_CROSS_SECTION_KEYS)
    assert "self_attention.inner.W_q.weight" in keys
    assert "self_attention.norm.weight" in keys


def _forward_inputs(num_stocks: int = 4, num_features: int = 7, seq_len: int = 4):
    torch.manual_seed(0)
    time_series = torch.randn(1, num_stocks, seq_len, num_features)
    graph_features = torch.randn(num_stocks, num_features)
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    edge_weight = torch.randn(3, 1)
    return time_series, graph_features, edge_index, edge_weight


def _residual_model(num_features: int = 7):
    # Without an explicit head activation the head inherits ReLU from
    # ``activation``, and a negative score would then pass no gradient. The
    # scores happen to be positive at this seed; the tests should not rely on it.
    # ``_BASE_MODEL_CONFIG`` keeps the inherited head, because the legacy
    # literals below were measured with it.
    torch.manual_seed(0)
    return create_model(
        num_features,
        {**_BASE_MODEL_CONFIG, "cross_section_block": "residual", "output_activation": "none"},
    )


# Scores of the default model, measured on origin/main at 0ecf723 and again at
# 125abda (identical at both), before this change existed: torch.manual_seed(0),
# create_model(7, _BASE_MODEL_CONFIG), eval mode, _forward_inputs(). Literals
# rather than a second model built by today's code, so a regression in the
# default path cannot appear on both sides and cancel out.
_LEGACY_SCORES_UNMASKED = [0.0462127551, 0.0462127551, 0.0462319329, 0.0461926572]
_LEGACY_SCORES_MASKED = [0.0189649239, 0.0189649239, 0.0189655237, 0.0]


def test_default_model_scores_match_the_pre_change_model() -> None:
    """With the flag off, the model must compute exactly what it computed before.

    The key and parameter-count guards above catch a change of module graph.
    They cannot catch a default path that keeps the same parameters but uses
    them differently (a residual add with no new weights, a mask applied in a
    different place) or that consumes the initialisation RNG differently. The
    scores can. The tolerance is float32 round-off, far below the cross-stock
    spread of the pinned scores.
    """
    torch.manual_seed(0)
    model = create_model(7, dict(_BASE_MODEL_CONFIG))
    model.eval()
    time_series, graph_features, edge_index, edge_weight = _forward_inputs()
    mask = torch.tensor([[True, True, True, False]])

    with torch.no_grad():
        unmasked = model(time_series, graph_features, edge_index, edge_weight, 4)
        masked = model(time_series, graph_features, edge_index, edge_weight, 4, stock_mask=mask)

    expected_unmasked = torch.tensor([_LEGACY_SCORES_UNMASKED])
    expected_masked = torch.tensor([_LEGACY_SCORES_MASKED])
    torch.testing.assert_close(unmasked, expected_unmasked, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(masked, expected_masked, rtol=1e-5, atol=1e-7)


def test_create_model_rejects_an_unknown_cross_section_block() -> None:
    """A misspelt form must not quietly build the legacy block.

    ``ModelConfig`` validates the field, but ``create_model`` also accepts a
    plain dict such as a checkpoint's ``config.yaml``, which never passes
    through ``ModelConfig``.
    """
    with pytest.raises(ValueError, match="cross_section_block"):
        create_model(8, {**_BASE_MODEL_CONFIG, "cross_section_block": "Residual"})


def test_residual_model_zeroes_inactive_nodes_end_to_end() -> None:
    model = _residual_model()
    time_series, graph_features, edge_index, edge_weight = _forward_inputs()
    mask = torch.tensor([[True, True, True, False]])

    out = model(time_series, graph_features, edge_index, edge_weight, 4, stock_mask=mask)

    assert out.shape == (1, 4)
    assert torch.all(out[:, 3] == 0)


def test_residual_model_inactive_stock_cannot_move_active_scores() -> None:
    model = _residual_model()
    model.eval()
    time_series, graph_features, edge_index, edge_weight = _forward_inputs()
    mask = torch.tensor([[True, True, True, False]])

    with torch.no_grad():
        before = model(time_series, graph_features, edge_index, edge_weight, 4, stock_mask=mask)
        changed_ts = time_series.clone()
        changed_ts[:, 3] = changed_ts[:, 3] + 99.0
        changed_graph = graph_features.clone()
        changed_graph[3] = changed_graph[3] + 99.0
        after = model(changed_ts, changed_graph, edge_index, edge_weight, 4, stock_mask=mask)

    assert torch.allclose(before[:, :3], after[:, :3], atol=1e-5)


# ``inner.W_k.bias`` adds the same term, q_i . b_k, to every key's score for a
# given query. Softmax is shift-invariant, so its gradient is zero up to rounding
# (measured 1.5e-8) and it is exempt from the nonzero requirement below.
_SOFTMAX_SHIFT_INVARIANT_PARAMETERS = {"inner.W_k.bias"}
# The smallest real gradient here is about 2e-2 (``inner.W_q.bias``).
_MIN_GRADIENT = 1e-4


def test_residual_block_parameters_receive_gradients() -> None:
    """Every block parameter must receive a real gradient, not merely a tensor.

    A branch multiplied by zero, or fed a constant, still leaves a finite
    gradient on every parameter: it is just zero. Existence and finiteness
    therefore cannot tell a working block from an inert one, so each parameter
    must carry a gradient well above rounding.
    """
    model = _residual_model()
    time_series, graph_features, edge_index, edge_weight = _forward_inputs()

    model(time_series, graph_features, edge_index, edge_weight, 4).sum().backward()

    grads = {name: p.grad for name, p in model.self_attention.named_parameters()}
    assert set(grads) > _SOFTMAX_SHIFT_INVARIANT_PARAMETERS
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    too_small = {
        name: g.abs().max().item()
        for name, g in grads.items()
        if name not in _SOFTMAX_SHIFT_INVARIANT_PARAMETERS and g.abs().max() <= _MIN_GRADIENT
    }
    assert not too_small


def test_residual_model_is_finite_under_autocast() -> None:
    model = _residual_model()
    time_series, graph_features, edge_index, edge_weight = _forward_inputs()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(time_series, graph_features, edge_index, edge_weight, 4)

    assert torch.isfinite(out.float()).all()


def test_attention_correction_ignores_a_per_stock_constant_offset() -> None:
    """The pre-norm must make the correction depend on shape, not on level.

    Shifting one stock's whole vector by a constant changes its level and
    nothing else. With the input normalised before attending, the attention's
    contribution is unmoved and only the residual carries the offset through.
    Without the norm this fails, so the guard pins the LayerNorm behaviourally
    rather than by asserting the module is present.
    """
    torch.manual_seed(0)
    block = ResidualCrossSectionBlock(SelfAttention(embed_dim=8, align_dim=2), embed_dim=8)
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(1, 20, 8, generator=generator)
    per_stock_offset = torch.randn(1, 20, 1, generator=generator) * 7.0

    correction = block(x) - x
    shifted_correction = block(x + per_stock_offset) - (x + per_stock_offset)

    assert torch.allclose(correction, shifted_correction, atol=1e-5)


def test_base_config_yaml_ships_the_legacy_block() -> None:
    """Real runs take the default from ``configs/config.yaml``, not from ``ModelConfig``.

    The "no default change" criterion rests on this YAML value, and no other
    test reads it.
    """
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=[])
    assert cfg.model.cross_section_block == "legacy"


def test_model_config_rejects_an_unknown_cross_section_block() -> None:
    with pytest.raises(ValueError, match="cross_section_block"):
        ModelConfig(cross_section_block="not-a-mode")


def test_model_config_round_trips_the_new_field() -> None:
    assert ModelConfig().to_dict()["cross_section_block"] == "legacy"
    serialised = ModelConfig(cross_section_block="residual").to_dict()
    assert serialised["cross_section_block"] == "residual"
    assert ModelConfig(**serialised).cross_section_block == "residual"
