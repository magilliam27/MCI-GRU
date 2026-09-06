"""Market latent state behaviour: does the "market" stream see the market?

``MarketLatentStateLearner`` holds ``R1`` and ``R2`` as plain parameters and its
forward takes only per-stock vectors, so ``B1 = f(A1)`` pointwise and the
latents are frozen after training. The streams the architecture calls market
latent states cannot observe the market on any date (issue #198).

These tests specify that behaviour at the module and model seams the issue
declared. They assert nothing about attention weights or the latent values.
"""

import pytest
import torch

from mci_gru.config import ModelConfig
from mci_gru.models import MarketLatentStateLearner, create_model

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


def test_data_dependent_latents_respond_to_the_rest_of_the_cross_section() -> None:
    """A stock's market latent state must depend on the market it sits in.

    Stock 0's own inputs are held fixed and every other stock on the date is
    moved. Its latent-state output must move, because the latents are gathered
    from the date's cross-section before the stock reads them.
    """
    torch.manual_seed(0)
    learner = MarketLatentStateLearner(
        feature_dim=8,
        num_latent_states=4,
        num_heads=2,
        market_latent_mode="data_dependent",
    )
    a1, a2 = _stream_inputs()

    before, _ = learner(a1, a2, num_stocks=6)
    after, _ = learner(_move_every_stock_except_the_first(a1), a2, num_stocks=6)

    assert not torch.allclose(before[0], after[0], atol=1e-6)


def test_static_latents_ignore_the_rest_of_the_cross_section() -> None:
    """The shipped behaviour, pinned so the default path cannot drift.

    In static mode the latents are fixed parameters, so a stock's output is a
    function of its own vector alone. This is the defect issue #198 describes;
    it is pinned because ``static`` remains the default and existing runs must
    keep reproducing.
    """
    torch.manual_seed(0)
    learner = MarketLatentStateLearner(feature_dim=8, num_latent_states=4, num_heads=2)
    a1, a2 = _stream_inputs()

    before, _ = learner(a1, a2, num_stocks=6)
    after, _ = learner(_move_every_stock_except_the_first(a1), a2, num_stocks=6)

    assert torch.allclose(before[0], after[0], atol=1e-6)


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


def test_a_date_with_no_active_names_does_not_produce_nan() -> None:
    """The gather's softmax would see every key masked; it must not divide by nothing."""
    learner = _data_dependent_learner()
    a1, a2 = _stream_inputs()
    mask = torch.zeros(1, 6, dtype=torch.bool)

    b1, b2 = learner(a1, a2, num_stocks=6, stock_mask=mask)

    assert torch.isfinite(b1).all()
    assert torch.isfinite(b2).all()


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

    b1, b2 = learner(a1, a2)

    assert b1[0].tolist() == pytest.approx(
        _PRE_CHANGE_B1_ROW0[use_nn_multihead_attention], abs=1e-6
    )
    assert float(b2.sum()) == pytest.approx(
        _PRE_CHANGE_B2_SUM[use_nn_multihead_attention], abs=1e-6
    )


def test_a_static_checkpoint_loads_strictly_into_a_default_model() -> None:
    """Frozen paper-trade bundles load by ``load_state_dict(strict=True)``."""
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


def test_data_dependent_latent_parameters_receive_gradients() -> None:
    model = _data_dependent_model()
    time_series, graph_features, edge_index, edge_weight = _forward_inputs()

    model(time_series, graph_features, edge_index, edge_weight, 4).sum().backward()

    grads = {name: p.grad for name, p in model.latent_learner.named_parameters()}
    assert any(name.startswith("gather1.") for name in grads)
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())


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

    assert data_dependent.to_dict()["market_latent_mode"] == "data_dependent"
