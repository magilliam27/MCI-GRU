"""Market latent state behaviour: does the "market" stream see the market?

``MarketLatentStateLearner`` holds ``R1`` and ``R2`` as plain parameters and its
forward takes only per-stock vectors, so ``B1 = f(A1)`` pointwise and the
latents are frozen after training. The streams the architecture calls market
latent states cannot observe the market on any date (issue #198).

These tests specify that behaviour at the module and model seams the issue
declared. They assert nothing about attention weights or the latent values.
One test also reads ``configs/config.yaml`` through Hydra ``compose``, the
boundary already used to pin the other model switches' YAML defaults.
"""

import hashlib
import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from mci_gru.config import ModelConfig
from mci_gru.models import MarketLatentStateLearner, create_model

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
    # Without this the head inherits ``activation`` and ends in a ReLU, which
    # clamps these small models' scores to zero, and every model-level equality
    # below would then hold whatever the latents did.
    "output_activation": "none",
    "temporal_encoder": "legacy",
}

# Measured on origin/main at 125abda, before this change, for feature_dim=8,
# num_latent_states=4, num_heads=2. Literals rather than values recomputed the
# way the code computes them, so the guard can disagree with a regression.
_STATIC_PARAMETER_COUNT = 640
_STATIC_LEGACY_KEYS = {
    "R1",
    "R2",
    "W_K1.bias",
    "W_K1.weight",
    "W_K2.bias",
    "W_K2.weight",
    "W_O1.bias",
    "W_O1.weight",
    "W_O2.bias",
    "W_O2.weight",
    "W_Q1.bias",
    "W_Q1.weight",
    "W_Q2.bias",
    "W_Q2.weight",
    "W_V1.bias",
    "W_V1.weight",
    "W_V2.bias",
    "W_V2.weight",
}


def _data_dependent_learner(feature_dim: int = 8) -> MarketLatentStateLearner:
    torch.manual_seed(0)
    return MarketLatentStateLearner(
        feature_dim=feature_dim,
        num_latent_states=4,
        num_heads=2,
        market_latent_mode="data_dependent",
    )


def _stream_inputs(num_stocks: int = 6, feature_dim: int = 8, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    a1 = torch.randn(num_stocks, feature_dim, generator=generator)
    a2 = torch.randn(num_stocks, feature_dim, generator=generator)
    return a1, a2


def _move_every_stock_except_the_first(a1: torch.Tensor) -> torch.Tensor:
    changed = a1.clone()
    changed[1:] = changed[1:] + 3.0
    return changed


def _move_one_stream(
    a1: torch.Tensor, a2: torch.Tensor, stream: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move every stock but the first in stream ``stream`` (0 for A1, 1 for A2) only."""
    if stream == 0:
        return _move_every_stock_except_the_first(a1), a2
    return a1, _move_every_stock_except_the_first(a2)


@pytest.mark.parametrize("stream", [0, 1], ids=["B1_reads_A1", "B2_reads_A2"])
def test_data_dependent_latents_respond_to_the_rest_of_the_cross_section(stream: int) -> None:
    """A stock's market latent state must depend on the market it sits in.

    Stock 0's own inputs are held fixed and every other stock on the date is
    moved, in one stream at a time. That stream's latent-state output for stock
    0 must move, because its latents are gathered from the date's cross-section
    before the stock reads them. One stream at a time, so B1 must read A1's
    cross-section and B2 must read A2's; a gather wired to the other stream, or
    one that is inert for either, fails a case.
    """
    torch.manual_seed(0)
    learner = MarketLatentStateLearner(
        feature_dim=8,
        num_latent_states=4,
        num_heads=2,
        market_latent_mode="data_dependent",
    )
    a1, a2 = _stream_inputs()

    before = learner(a1, a2, num_stocks=6)
    after = learner(*_move_one_stream(a1, a2, stream), num_stocks=6)

    assert not torch.allclose(before[stream][0], after[stream][0], atol=1e-6)


@pytest.mark.parametrize("stream", [0, 1], ids=["A1_moved", "A2_moved"])
def test_static_latents_ignore_the_rest_of_the_cross_section(stream: int) -> None:
    """The shipped behaviour, pinned so the default path cannot drift.

    In static mode the latents are fixed parameters, so a stock's output is a
    function of its own vector alone. This is the defect issue #198 describes;
    it is pinned because ``static`` remains the default and existing runs must
    keep reproducing. Neither B1 nor B2 of stock 0 may move.
    """
    torch.manual_seed(0)
    learner = MarketLatentStateLearner(feature_dim=8, num_latent_states=4, num_heads=2)
    a1, a2 = _stream_inputs()

    before_b1, before_b2 = learner(a1, a2, num_stocks=6)
    after_b1, after_b2 = learner(*_move_one_stream(a1, a2, stream), num_stocks=6)

    assert torch.allclose(before_b1[0], after_b1[0], atol=1e-6)
    assert torch.allclose(before_b2[0], after_b2[0], atol=1e-6)


def test_gathered_latents_ignore_pit_inactive_names() -> None:
    """An inactive union node must not reach the market state of an active one."""
    learner = _data_dependent_learner()
    a1, a2 = _stream_inputs()
    mask = torch.tensor([[True, True, True, True, True, False]])

    before, _ = learner(a1, a2, num_stocks=6, stock_mask=mask)
    changed = a1.clone()
    changed[5] = 999.0
    after, _ = learner(changed, a2, num_stocks=6, stock_mask=mask)

    assert torch.allclose(before[:5], after[:5], atol=1e-5)


def test_gathered_latents_read_the_active_names_not_the_inactive_ones() -> None:
    """With a mask supplied, the gather must still read the *active* cross-section.

    Moving other active names has to move stock 0's market state. A mask applied
    the wrong way round would build the latents from the excluded names instead,
    leaving stock 0 unmoved by the market it actually sits in.
    """
    learner = _data_dependent_learner()
    a1, a2 = _stream_inputs()
    mask = torch.tensor([[True, True, True, True, True, False]])

    before, _ = learner(a1, a2, num_stocks=6, stock_mask=mask)
    moved = a1.clone()
    moved[1:5] = moved[1:5] + 3.0
    after, _ = learner(moved, a2, num_stocks=6, stock_mask=mask)

    assert not torch.allclose(before[0], after[0], atol=1e-6)


def test_gathered_latents_ignore_how_many_names_are_inactive() -> None:
    """Padding a date with more inactive names must not move the active ones.

    Zeroing an inactive row is not enough on its own: a zero row still occupies
    a slot in the gather's softmax and dilutes every weight, so the active
    names' market state would drift with the size of the union axis. Only
    excluding those keys from the attention keeps it stable. This is the guard
    that distinguishes the two masking steps, which content-only tests cannot.
    """
    learner = _data_dependent_learner()
    a1, a2 = _stream_inputs(num_stocks=6)
    padding = torch.Generator().manual_seed(11)
    wide_a1 = torch.cat([a1[:5], torch.randn(5, 8, generator=padding)])
    wide_a2 = torch.cat([a2[:5], torch.randn(5, 8, generator=padding)])

    narrow, _ = learner(a1, a2, num_stocks=6, stock_mask=torch.tensor([[True] * 5 + [False]]))
    wide, _ = learner(
        wide_a1, wide_a2, num_stocks=10, stock_mask=torch.tensor([[True] * 5 + [False] * 5])
    )

    assert torch.allclose(narrow[:5], wide[:5], atol=1e-5)


def _two_dates(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two dates of six names each, flattened date-major as the trunk passes them."""
    generator = torch.Generator().manual_seed(seed)
    a1 = torch.randn(12, 8, generator=generator)
    a2 = torch.randn(12, 8, generator=generator)
    return a1, a2


@pytest.mark.parametrize(("moved_date", "held_date"), [(1, 0), (0, 1)])
def test_gathered_latents_do_not_read_other_dates_in_the_batch(
    moved_date: int, held_date: int
) -> None:
    """A date's market state is gathered from that date's cross-section alone.

    A batch stacks several dates, so a gather over the whole batch would let one
    date's latents read another date's names, a later one included: lookahead.
    Moving the names on one date must leave the other date's outputs where they
    were, in both directions. The same move must still reach the held-fixed
    first name on its own date, or this would pass with the gather switched off.
    """
    learner = _data_dependent_learner()
    a1, a2 = _two_dates()
    moved = slice(6 * moved_date, 6 * moved_date + 6)
    held = slice(6 * held_date, 6 * held_date + 6)
    moved_a1, moved_a2 = a1.clone(), a2.clone()
    moved_a1[moved.start + 1 : moved.stop] += 3.0
    moved_a2[moved.start + 1 : moved.stop] += 3.0

    before_b1, before_b2 = learner(a1, a2, num_stocks=6)
    after_b1, after_b2 = learner(moved_a1, moved_a2, num_stocks=6)

    assert torch.allclose(before_b1[held], after_b1[held], atol=1e-6)
    assert torch.allclose(before_b2[held], after_b2[held], atol=1e-6)
    assert not torch.allclose(before_b1[moved.start], after_b1[moved.start], atol=1e-6)
    assert not torch.allclose(before_b2[moved.start], after_b2[moved.start], atol=1e-6)


def test_each_date_in_the_batch_is_masked_by_its_own_row() -> None:
    """The PIT mask is per date: row ``d`` of ``stock_mask`` governs date ``d`` only.

    Name 2 is inactive on the second date but active on the first. Moving it on
    the second date must not reach that date's active names, while moving an
    active name there must. A mask read from the wrong date's row gets at least
    one of the two wrong.
    """
    learner = _data_dependent_learner()
    a1, a2 = _two_dates()
    mask = torch.tensor(
        [[True, True, True, True, True, False], [True, True, False, True, True, True]]
    )
    second_date_active = [6, 7, 9, 10, 11]
    before, _ = learner(a1, a2, num_stocks=6, stock_mask=mask)

    def b1_after_moving(flat_row: int) -> torch.Tensor:
        moved = a1.clone()
        moved[flat_row] += 3.0
        after, _ = learner(moved, a2, num_stocks=6, stock_mask=mask)
        return after

    after_inactive_move = b1_after_moving(8)
    after_active_move = b1_after_moving(10)

    assert torch.allclose(
        before[second_date_active], after_inactive_move[second_date_active], atol=1e-6
    )
    assert not torch.allclose(before[6], after_active_move[6], atol=1e-6)


@pytest.mark.parametrize("training", [True, False])
def test_a_date_with_no_active_names_does_not_produce_nan(training: bool) -> None:
    """The gather's softmax would see every key masked; it must not divide by nothing.

    Both modes, because inference runs the attention under ``eval`` and
    ``no_grad``, which can take a different kernel from training.
    """
    learner = _data_dependent_learner()
    learner.train(training)
    a1, a2 = _stream_inputs()
    mask = torch.zeros(1, 6, dtype=torch.bool)

    with torch.set_grad_enabled(training):
        b1, b2 = learner(a1, a2, num_stocks=6, stock_mask=mask)

    assert torch.isfinite(b1).all()
    assert torch.isfinite(b2).all()


# Measured spread of B1 and B2 across the six names of an empty date for the
# learner below: about 0.26. With the gather's residual dropped it is exactly 0.
_EMPTY_DATE_MIN_SPREAD = 1e-3


@pytest.mark.parametrize("training", [True, False])
def test_a_date_with_no_active_names_keeps_the_learned_latents(training: bool) -> None:
    """With nothing to read, a date's latents must fall back on the learned ones.

    #198 specifies ``R_d = R + MHA(query=R, key/value=A_active)``: the gather
    adds what the date's names say to the learned latents rather than replacing
    them. On a date with no active names there is nothing to add, so the names
    still read ``k`` distinct learned latents and keep distinct B1 and B2.
    Replace the sum by the gather alone and the ``k`` latents collapse to one
    vector, because every latent's query then gets the same answer from an empty
    key set, and every name on the date receives the same state.

    The latents start at unit scale so that the spread they give the names is
    far from rounding; at the default ``0.02`` it is about ``1e-4``.
    """
    torch.manual_seed(0)
    learner = MarketLatentStateLearner(
        feature_dim=8,
        num_latent_states=4,
        num_heads=2,
        latent_init_scale=1.0,
        market_latent_mode="data_dependent",
    )
    learner.train(training)
    a1, a2 = _stream_inputs()
    mask = torch.zeros(1, 6, dtype=torch.bool)

    with torch.set_grad_enabled(training):
        b1, b2 = learner(a1, a2, num_stocks=6, stock_mask=mask)

    for stream in (b1, b2):
        spread = (stream - stream.mean(dim=0)).abs().max().item()
        assert spread > _EMPTY_DATE_MIN_SPREAD


def test_data_dependent_mode_refuses_to_guess_the_date_grouping() -> None:
    """Without num_stocks the flattened stream cannot be grouped by date."""
    learner = _data_dependent_learner()
    a1, a2 = _stream_inputs()

    with pytest.raises(ValueError, match="num_stocks"):
        learner(a1, a2)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="market_latent_mode"):
        MarketLatentStateLearner(feature_dim=8, market_latent_mode="not-a-mode")


def test_static_mode_ignores_the_new_arguments() -> None:
    """Passing the new arguments must not change the shipped computation."""
    torch.manual_seed(0)
    learner = MarketLatentStateLearner(feature_dim=8, num_latent_states=4, num_heads=2)
    a1, a2 = _stream_inputs()
    mask = torch.tensor([[True, True, True, True, True, False]])

    plain_b1, plain_b2 = learner(a1, a2)
    with_args_b1, with_args_b2 = learner(a1, a2, num_stocks=6, stock_mask=mask)

    assert torch.equal(plain_b1, with_args_b1)
    assert torch.equal(plain_b2, with_args_b2)


# Captured by running origin/main's mci_gru/models/latent.py at 125abda with
# torch.manual_seed(0), feature_dim=8, num_latent_states=4, num_heads=2, and the
# _stream_inputs batch. Golden values from the pre-change code, so the static
# path is pinned against what it actually used to produce.
_PRE_CHANGE_B1_ROW0 = {
    False: [-0.3019599, 0.3648441, 0.025169, 0.1454739, -0.4389973, 0.1422492, 0.047831, 0.075839],
    True: [0.003509, 0.0017126, -0.0034129, 0.0020406, -0.0009026, -0.005144, 9.52e-05, -0.0031499],
}
_PRE_CHANGE_B2_SUM = {False: 0.0382016, True: -0.0076822}


@pytest.mark.parametrize("use_nn_multihead_attention", [False, True])
def test_static_mode_reproduces_the_pre_change_outputs(use_nn_multihead_attention: bool) -> None:
    """Static mode must be bitwise what it was, not merely self-consistent.

    Both static branches are pinned: the legacy 8-Linear cross-attention and the
    ``nn.MultiheadAttention`` one. The expected numbers were produced by the
    pre-change module, so this can disagree with a regression in either.
    """
    torch.manual_seed(0)
    learner = MarketLatentStateLearner(
        feature_dim=8,
        num_latent_states=4,
        num_heads=2,
        use_nn_multihead_attention=use_nn_multihead_attention,
    )
    a1, a2 = _stream_inputs()

    with torch.no_grad():
        b1, b2 = learner(a1, a2)

    assert b1[0].tolist() == pytest.approx(
        _PRE_CHANGE_B1_ROW0[use_nn_multihead_attention], abs=1e-6
    )
    assert float(b2.sum()) == pytest.approx(
        _PRE_CHANGE_B2_SUM[use_nn_multihead_attention], abs=1e-6
    )


# The production model block of configs/config.yaml at this change, plus the
# graph-derived keys run_experiment.py adds before calling create_model. Written
# out rather than read from the YAML so that a later, deliberate default change
# does not silently move what this pins.
_PRODUCTION_MODEL_CONFIG = {
    "his_t": 10,
    "label_t": 5,
    "gru_hidden_sizes": [32, 10],
    "hidden_size_gat1": 32,
    "output_gat1": 4,
    "gat_heads": 4,
    "hidden_size_gat2": 32,
    "num_hidden_states": 32,
    "cross_attn_heads": 4,
    "slow_kernel": 5,
    "slow_stride": 2,
    "use_multi_scale": True,
    "use_self_attention": True,
    "activation": "elu",
    "output_activation": "none",
    "latent_init_scale": 0.02,
    "use_group_type_embed": True,
    "use_trunk_regularisation": True,
    "trunk_dropout": 0.1,
    "use_nn_multihead_attention": True,
    "temporal_encoder": "gru_attn",
    "use_a1_a2_cross_attention": False,
    "cross_a2_num_heads": 4,
    "edge_feature_dim": 4,
    "drop_edge_p": 0.1,
    "isolate_edge_dropout_rng": False,
    "use_sector_relation": False,
}

# Captured by running origin/main's mci_gru package at 0ecf723, which does not
# contain this change, through exactly the construction and forward below, and
# re-checked bitwise against abe1f81, which adds ``cross_section_block``. The
# signature is the sha256 of the sorted ``key:shape`` lines of the state dict,
# which is what ``load_state_dict(strict=True)`` checks, so matching it means a
# checkpoint written by main loads here. ``legacy_empty`` is a config with no
# model keys at all, the shape of a checkpoint directory older than every flag.
_MAIN_DEFAULT_MODELS = {
    "production": {
        "config": _PRODUCTION_MODEL_CONFIG,
        "edge_dim": 4,
        "state_dict_keys": 75,
        "parameters": 84026,
        "signature": "3d03a4f34307fda6c9793a2ec366e921bd640e5173b0806c8777c8b6b4b5d96e",
        "scores": [
            -0.4530891,
            -0.4530891,
            -0.4535756,
            -0.4535756,
            0.0,
            -0.428826,
            -0.428826,
            0.0,
            -0.4347683,
            -0.4341896,
        ],
    },
    "legacy_empty": {
        "config": {},
        "edge_dim": 1,
        "state_dict_keys": 112,
        "parameters": 91915,
        "signature": "0490ceb61e6d3cb1b112540ceabbff58685ca7044ff6355a1aaa985dd1a9fb61",
        "scores": [
            0.0883905,
            0.0883905,
            0.0884054,
            0.0884054,
            0.0,
            0.0045736,
            0.0045736,
            0.0,
            0.0044855,
            0.0044812,
        ],
    },
}


def _state_dict_signature(model: torch.nn.Module) -> str:
    lines = [f"{key}:{tuple(value.shape)}" for key, value in sorted(model.state_dict().items())]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _two_date_masked_batch(edge_dim: int):
    generator = torch.Generator().manual_seed(7)
    time_series = torch.randn(2, 5, 10, 7, generator=generator)
    graph_features = torch.randn(10, 7, generator=generator)
    edge_index = torch.tensor([[0, 1, 2, 3, 5, 6, 8], [1, 0, 3, 2, 6, 5, 9]], dtype=torch.long)
    edge_weight = torch.rand(7, edge_dim, generator=generator)
    stock_mask = torch.tensor([[True, True, True, True, False], [True, True, False, True, True]])
    return time_series, graph_features, edge_index, edge_weight, stock_mask


@pytest.mark.parametrize("spell_out_static", [False, True])
@pytest.mark.parametrize("case", sorted(_MAIN_DEFAULT_MODELS))
def test_default_model_is_unchanged_from_main(case: str, spell_out_static: bool) -> None:
    """Default-off must leave the whole model what ``main`` builds, not just the module.

    The same seed must give the same state-dict signature, parameter count and
    scores as ``main`` did, whether the new key is absent or spelled out as
    ``static``. That covers checkpoint loading and the seeded initialisation
    order, which the module-level golden values cannot see: a change that drew
    one extra random number while building the trunk would move every later
    layer's weights and still pass them.

    The tolerance is about fifty times the float32 rounding measured for this
    forward against float64, so it absorbs platform differences, and far below
    what any change to initialisation or wiring moves these scores by. On the
    machine that captured them, ``main`` and this branch agreed bitwise.
    """
    expected = _MAIN_DEFAULT_MODELS[case]
    config = dict(expected["config"])
    if spell_out_static:
        config["market_latent_mode"] = "static"

    torch.manual_seed(0)
    model = create_model(7, config)
    model.eval()
    time_series, graph_features, edge_index, edge_weight, stock_mask = _two_date_masked_batch(
        expected["edge_dim"]
    )
    with torch.no_grad():
        scores = model(
            time_series, graph_features, edge_index, edge_weight, 5, stock_mask=stock_mask
        )

    assert len(model.state_dict()) == expected["state_dict_keys"]
    assert sum(p.numel() for p in model.parameters()) == expected["parameters"]
    assert _state_dict_signature(model) == expected["signature"]
    assert scores.flatten().tolist() == pytest.approx(expected["scores"], abs=1e-5)


def test_a_static_checkpoint_loads_strictly_into_a_default_model() -> None:
    """A default-model checkpoint reloads with ``load_state_dict(strict=True)``.

    This only shows the default model agrees with itself. Compatibility with a
    checkpoint written *before* this change is pinned separately, against
    ``origin/main``'s state-dict signature, in
    ``test_default_model_is_unchanged_from_main``.
    """
    saved = create_model(8, dict(_BASE_MODEL_CONFIG)).state_dict()

    reloaded = create_model(8, dict(_BASE_MODEL_CONFIG))
    missing, unexpected = reloaded.load_state_dict(saved, strict=True)

    assert not missing
    assert not unexpected


def test_model_config_refuses_data_dependent_without_multihead_attention() -> None:
    """The legacy 8-Linear path cannot take per-date keys, so refuse rather than override."""
    with pytest.raises(ValueError, match="use_nn_multihead_attention"):
        ModelConfig(market_latent_mode="data_dependent", use_nn_multihead_attention=False)


def test_static_mode_keeps_the_pre_change_parameter_set() -> None:
    """Frozen checkpoints must keep loading, so the default may not move."""
    torch.manual_seed(0)
    learner = MarketLatentStateLearner(feature_dim=8, num_latent_states=4, num_heads=2)

    assert set(learner.state_dict()) == _STATIC_LEGACY_KEYS
    assert sum(p.numel() for p in learner.parameters()) == _STATIC_PARAMETER_COUNT


def test_data_dependent_mode_holds_different_parameters() -> None:
    """The two modes are not checkpoint-interchangeable, and that is asserted."""
    learner = _data_dependent_learner()

    keys = set(learner.state_dict())

    assert keys.isdisjoint({k for k in _STATIC_LEGACY_KEYS if k.startswith("W_")})
    assert any(k.startswith("gather1.") for k in keys)


def _forward_inputs(num_stocks: int = 4, num_features: int = 7, seq_len: int = 4):
    torch.manual_seed(0)
    time_series = torch.randn(1, num_stocks, seq_len, num_features)
    graph_features = torch.randn(num_stocks, num_features)
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    edge_weight = torch.randn(3, 1)
    return time_series, graph_features, edge_index, edge_weight


def _data_dependent_model(num_features: int = 7):
    torch.manual_seed(0)
    return create_model(
        num_features,
        {**_BASE_MODEL_CONFIG, "market_latent_mode": "data_dependent"},
    )


def test_default_config_still_builds_static_latents() -> None:
    model = create_model(8, dict(_BASE_MODEL_CONFIG))

    assert model.latent_learner.market_latent_mode == "static"
    assert {
        k.removeprefix("latent_learner.") for k in model.state_dict() if "latent_learner" in k
    } == (_STATIC_LEGACY_KEYS)


def test_config_without_the_new_key_builds_static_latents() -> None:
    """A ``config.yaml`` written before this change has no such key at all."""
    legacy_only = dict(_BASE_MODEL_CONFIG)
    assert "market_latent_mode" not in legacy_only

    model = create_model(8, legacy_only)

    assert model.latent_learner.market_latent_mode == "static"


def test_data_dependent_flag_reaches_the_built_model() -> None:
    model = _data_dependent_model(num_features=8)

    assert model.latent_learner.market_latent_mode == "data_dependent"


def _isolated_stock_zero_model(mode: str):
    """Model whose ONLY cross-stock path is the latent gather.

    Cross-stock self-attention is off and the graph carries a single edge
    between stocks 2 and 3, so stock 0 has no graph or attention route to the
    stocks that get perturbed. Without that isolation the test passes in static
    mode too, because the GAT and the self-attention both mix stocks.
    """
    torch.manual_seed(0)
    return create_model(
        7,
        {
            **_BASE_MODEL_CONFIG,
            "use_self_attention": False,
            "market_latent_mode": mode,
        },
    )


def _isolated_forward_inputs():
    time_series, graph_features, _, _ = _forward_inputs()
    edge_index = torch.tensor([[2], [3]], dtype=torch.long)
    edge_weight = torch.ones(1, 1)
    return time_series, graph_features, edge_index, edge_weight


def _stock_zero_shift(mode: str) -> float:
    model = _isolated_stock_zero_model(mode)
    model.eval()
    time_series, graph_features, edge_index, edge_weight = _isolated_forward_inputs()

    with torch.no_grad():
        before = model(time_series, graph_features, edge_index, edge_weight, 4)
        changed_ts = time_series.clone()
        changed_ts[:, 2:] = changed_ts[:, 2:] + 3.0
        changed_graph = graph_features.clone()
        changed_graph[2:] = changed_graph[2:] + 3.0
        after = model(changed_ts, changed_graph, edge_index, edge_weight, 4)

    return (before[:, 0] - after[:, 0]).abs().max().item()


def test_model_latents_respond_to_other_stocks_end_to_end() -> None:
    """The trunk must pass the date grouping down, or the mode is inert.

    Paired against the static control on the same isolated graph, so this
    cannot pass by way of the GAT or the cross-stock attention.
    """
    assert _stock_zero_shift("static") == pytest.approx(0.0, abs=1e-6)
    assert _stock_zero_shift("data_dependent") > 1e-4


def test_data_dependent_model_zeroes_inactive_nodes() -> None:
    model = _data_dependent_model()
    time_series, graph_features, edge_index, edge_weight = _forward_inputs()
    mask = torch.tensor([[True, True, True, False]])

    out = model(time_series, graph_features, edge_index, edge_weight, 4, stock_mask=mask)

    assert out.shape == (1, 4)
    assert torch.all(out[:, 3] == 0)


def test_data_dependent_model_inactive_stock_cannot_move_active_scores() -> None:
    model = _data_dependent_model()
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


def test_model_active_scores_ignore_how_many_names_are_inactive() -> None:
    """The trunk must hand the PIT mask to the gather, not just zero the streams.

    The trunk already zeroes A1 and A2 for inactive nodes, so withholding the
    mask from the latent learner leaves their *content* out but still lets them
    dilute the gather's softmax. Widening the inactive padding is what makes
    that visible. Cross-stock attention is off and the graph is empty, so the
    latents are the only route between stocks.
    """
    torch.manual_seed(0)
    model = create_model(
        7,
        {
            **_BASE_MODEL_CONFIG,
            "use_self_attention": False,
            "market_latent_mode": "data_dependent",
        },
    )
    model.eval()
    generator = torch.Generator().manual_seed(4)
    active_ts = torch.randn(1, 3, 4, 7, generator=generator)
    active_graph = torch.randn(3, 7, generator=generator)
    no_edges = torch.empty((2, 0), dtype=torch.long)
    no_weights = torch.empty((0, 1))

    def score(total_stocks: int) -> torch.Tensor:
        pad = total_stocks - 3
        time_series = torch.cat([active_ts, torch.randn(1, pad, 4, 7, generator=generator)], dim=1)
        graph_features = torch.cat([active_graph, torch.randn(pad, 7, generator=generator)], dim=0)
        mask = torch.tensor([[True] * 3 + [False] * pad])
        with torch.no_grad():
            out = model(
                time_series, graph_features, no_edges, no_weights, total_stocks, stock_mask=mask
            )
        return out[:, :3]

    assert torch.allclose(score(4), score(6), atol=1e-5)


@pytest.mark.parametrize(("moved_date", "held_date"), [(1, 0), (0, 1)])
def test_model_scores_do_not_read_other_dates_in_the_batch(moved_date: int, held_date: int) -> None:
    """End to end, the trunk must group the gather by date, not by batch.

    Cross-stock attention is off and there are no edges, so the latent gather is
    the only route between names, and any route between dates would have to run
    through it. Moving names on one date must leave the other date's scores
    untouched, including the earlier date when the later one moves. Moving the
    other names on the moved date must still reach its held-fixed first name,
    so the test cannot pass with the data-dependent path inert.
    """
    torch.manual_seed(0)
    model = create_model(
        7,
        {
            **_BASE_MODEL_CONFIG,
            "use_self_attention": False,
            "market_latent_mode": "data_dependent",
        },
    )
    model.eval()
    generator = torch.Generator().manual_seed(5)
    time_series = torch.randn(2, 4, 4, 7, generator=generator)
    graph_features = torch.randn(8, 7, generator=generator)
    no_edges = torch.empty((2, 0), dtype=torch.long)
    no_weights = torch.empty((0, 1))
    moved_ts = time_series.clone()
    moved_ts[moved_date, 1:] += 3.0
    moved_graph = graph_features.clone()
    moved_graph[4 * moved_date + 1 : 4 * moved_date + 4] += 3.0

    with torch.no_grad():
        before = model(time_series, graph_features, no_edges, no_weights, 4)
        after = model(moved_ts, moved_graph, no_edges, no_weights, 4)

    assert torch.allclose(before[held_date], after[held_date], atol=1e-6)
    assert (before[moved_date, 0] - after[moved_date, 0]).abs().item() > 1e-4


def test_data_dependent_latent_parameters_receive_gradients() -> None:
    model = _data_dependent_model()
    time_series, graph_features, edge_index, edge_weight = _forward_inputs()

    model(time_series, graph_features, edge_index, edge_weight, 4).sum().backward()

    grads = {name: p.grad for name, p in model.latent_learner.named_parameters()}
    assert any(name.startswith("gather1.") for name in grads)
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    for gather in ("gather1.", "gather2."):
        assert any(g.abs().sum() > 0 for name, g in grads.items() if name.startswith(gather))


def test_data_dependent_model_is_finite_under_autocast() -> None:
    model = _data_dependent_model()
    time_series, graph_features, edge_index, edge_weight = _forward_inputs()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(time_series, graph_features, edge_index, edge_weight, 4)

    assert torch.isfinite(out.float()).all()


def test_model_config_rejects_an_unknown_market_latent_mode() -> None:
    with pytest.raises(ValueError, match="market_latent_mode"):
        ModelConfig(market_latent_mode="not-a-mode")


def test_model_config_round_trips_the_new_field() -> None:
    assert ModelConfig().to_dict()["market_latent_mode"] == "static"

    # The dataclass default for use_nn_multihead_attention is False for legacy
    # safety, while configs/config.yaml ships it true, so data_dependent has to
    # opt in explicitly here.
    data_dependent = ModelConfig(
        market_latent_mode="data_dependent", use_nn_multihead_attention=True
    )
    serialised = data_dependent.to_dict()

    assert serialised["market_latent_mode"] == "data_dependent"
    assert ModelConfig(**serialised).market_latent_mode == "data_dependent"


def test_base_config_yaml_ships_static_market_latents() -> None:
    """Real runs take the default from ``configs/config.yaml``, not from ``ModelConfig``.

    "No default changes" rests on this YAML value, and no other test reads it.
    """
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=[])
    assert cfg.model.market_latent_mode == "static"


# -- both switches on ---------------------------------------------------------
#
# ``model.cross_section_block`` (issue #197) can wrap the cross-stock attention
# as a residual block. It and ``market_latent_mode`` edit the same trunk, and a
# run may turn both on, so these tests build that model the way a run does,
# through ``ModelConfig.to_dict()`` and ``create_model``, and check that each
# switch keeps its guarantees with the other on.

_SMALL_MODEL = {
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
    "output_activation": "none",
    "temporal_encoder": "legacy",
    "use_nn_multihead_attention": True,
}

_NO_EDGES = (torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1)))

# Measured smallest of the per-component largest gradients below: about 0.14.
_BOTH_SWITCHES_MIN_GRADIENT = 1e-4


def _switched_model(
    cross_section_block: str = "legacy", market_latent_mode: str = "static"
) -> torch.nn.Module:
    config = ModelConfig(
        **_SMALL_MODEL,
        cross_section_block=cross_section_block,
        market_latent_mode=market_latent_mode,
    ).to_dict()
    torch.manual_seed(0)
    return create_model(7, config)


def _both_switches_model() -> torch.nn.Module:
    return _switched_model(cross_section_block="residual", market_latent_mode="data_dependent")


def _three_dates(seed: int = 3):
    """Three dates of five names, one name inactive on each, and a ring graph per date."""
    generator = torch.Generator().manual_seed(seed)
    time_series = torch.randn(3, 5, 4, 7, generator=generator)
    graph_features = torch.randn(15, 7, generator=generator)
    ring = [(5 * d + i, 5 * d + (i + 1) % 5) for d in range(3) for i in range(5)]
    src, dst = (list(side) for side in zip(*ring, strict=True))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    edge_weight = torch.rand(edge_index.shape[1], 1, generator=generator)
    stock_mask = torch.tensor(
        [
            [True, True, True, False, True],
            [True, False, True, True, True],
            [True, True, True, True, False],
        ]
    )
    return time_series, graph_features, edge_index, edge_weight, stock_mask


def _move_dates(time_series, graph_features, dates, first_name: int = 0):
    """Shift every name from ``first_name`` onwards on each of ``dates``."""
    moved_ts, moved_graph = time_series.clone(), graph_features.clone()
    for date in dates:
        moved_ts[date, first_name:] += 3.0
        moved_graph[5 * date + first_name : 5 * date + 5] += 3.0
    return moved_ts, moved_graph


def test_both_switches_each_make_exactly_their_own_change() -> None:
    """Neither switch may suppress the other, or add anything when combined.

    The combined state dict must be the default one with the residual block's
    edits and the data-dependent latents' edits both applied, and nothing else.
    A factory or trunk that let one switch override the other builds a model
    that is missing one set of edits.
    """
    default = set(_switched_model().state_dict())
    residual = set(_switched_model(cross_section_block="residual").state_dict())
    data_dependent = set(_switched_model(market_latent_mode="data_dependent").state_dict())

    both = set(_both_switches_model().state_dict())

    assert residual - default
    assert data_dependent - default
    kept = default & residual & data_dependent
    assert both == kept | (residual - default) | (data_dependent - default)


def test_both_switches_keep_inactive_names_out() -> None:
    """Inactive names score exactly zero and cannot move an active name's score.

    The graph is empty, so the gather and the residual block are the only
    routes between names.
    """
    model = _both_switches_model()
    model.eval()
    time_series, graph_features, _, _, mask = _three_dates()
    inactive = ~mask
    moved_ts, moved_graph = time_series.clone(), graph_features.clone()
    moved_ts[inactive] += 99.0
    moved_graph[inactive.flatten()] += 99.0

    with torch.no_grad():
        before = model(time_series, graph_features, *_NO_EDGES, 5, stock_mask=mask)
        after = model(moved_ts, moved_graph, *_NO_EDGES, 5, stock_mask=mask)

    assert torch.all(before[inactive] == 0)
    assert torch.allclose(before[mask], after[mask], atol=1e-5)


@pytest.mark.parametrize("held_date", [0, 1, 2])
def test_both_switches_do_not_read_other_dates_in_the_batch(held_date: int) -> None:
    """With both switches on, a date's scores come from that date's names alone.

    Both switches mix names: the gather reads the date's cross-section and the
    residual block attends across it. Either one grouped by batch rather than by
    date would let a date read the others, later ones included: lookahead. Every
    other date's inputs and mask row are changed, and the held date must not
    move. The graph is empty, so the two switches are the only routes between
    names; moving the held date's other names must still reach its first name,
    or this would pass with both routes inert.
    """
    model = _both_switches_model()
    model.eval()
    time_series, graph_features, _, _, mask = _three_dates()
    others = [date for date in range(3) if date != held_date]
    other_ts, other_graph = _move_dates(time_series, graph_features, others)
    other_mask = mask.clone()
    other_mask[others] = ~other_mask[others]
    own_ts, own_graph = _move_dates(time_series, graph_features, [held_date], first_name=1)

    with torch.no_grad():
        before = model(time_series, graph_features, *_NO_EDGES, 5, stock_mask=mask)
        after_others = model(other_ts, other_graph, *_NO_EDGES, 5, stock_mask=other_mask)
        after_own = model(own_ts, own_graph, *_NO_EDGES, 5, stock_mask=mask)

    assert torch.allclose(before[held_date], after_others[held_date], atol=1e-6)
    assert (before[held_date, 0] - after_own[held_date, 0]).abs().item() > 1e-4


def test_both_switches_train_both_new_components() -> None:
    """A backward pass must reach the gathers and the residual block, with real gradients."""
    model = _both_switches_model()
    model.train()
    time_series, graph_features, edge_index, edge_weight, mask = _three_dates()
    weights = torch.linspace(-1.0, 1.0, int(mask.sum()))

    scores = model(time_series, graph_features, edge_index, edge_weight, 5, stock_mask=mask)
    (scores[mask] * weights).sum().backward()

    grads = {name: p.grad for name, p in model.named_parameters()}
    assert all(g is None or torch.isfinite(g).all() for g in grads.values())
    for component in (
        "latent_learner.gather1.",
        "latent_learner.gather2.",
        "self_attention.inner.",
        "self_attention.norm.",
    ):
        component_grads = [g for name, g in grads.items() if name.startswith(component)]
        assert component_grads
        assert all(g is not None for g in component_grads)
        largest = max(g.abs().max().item() for g in component_grads)
        assert largest > _BOTH_SWITCHES_MIN_GRADIENT


def test_both_switches_stay_finite_under_autocast_with_an_empty_date() -> None:
    model = _both_switches_model()
    model.train()
    time_series, graph_features, edge_index, edge_weight, mask = _three_dates()
    mask[1] = False

    with torch.autocast("cpu", dtype=torch.bfloat16):
        scores = model(time_series, graph_features, edge_index, edge_weight, 5, stock_mask=mask)
    scores.float()[mask].sum().backward()

    assert torch.isfinite(scores.float()).all()
    assert torch.all(scores[1] == 0)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_both_switches_train_through_a_batch_with_an_empty_date() -> None:
    """A few optimiser steps run, stay finite, and reduce the loss.

    The middle date has no active names, so its gather attends to nothing. That
    must not reach the loss or the weights of the dates that do have names.
    """
    model = _both_switches_model()
    model.train()
    time_series, graph_features, edge_index, edge_weight, mask = _three_dates()
    mask[1] = False
    targets = torch.randn(3, 5, generator=torch.Generator().manual_seed(9))
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(10):
        optimiser.zero_grad()
        scores = model(time_series, graph_features, edge_index, edge_weight, 5, stock_mask=mask)
        loss = (scores - targets)[mask].pow(2).mean()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0]
    assert all(torch.isfinite(p).all() for p in model.parameters())
