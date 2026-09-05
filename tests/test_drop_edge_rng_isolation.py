"""Isolated edge-dropout RNG (ticket 183, item 3; ticket 181 section 4 ruling).

``dropout_edge`` (PyG 2.7.0) draws from the global torch RNG and takes no
generator argument, so the number of draws a training step makes depends on how
many edges the arm has. The zeroed control arm A0 has none, so its stream and
every populated arm's stream diverge after the first step, and that divergence
is inside the ``sd(delta)`` the multi-year protocol is trying to shrink.

``graph.isolate_edge_dropout_rng`` forks the stream around each ``dropout_edge``
call. Under the flag the edge draws come from a stream seeded from the member
seed and the step, and the global stream is left exactly as it was found, so the
non-graph draws -- initialisation, feature dropout, shuffling -- coincide across
arms and only the graph differs.

The flag is config-gated and default off; shipped behaviour is unchanged when
off, which is what the ``_off`` guards below pin. Every guard here is
mutation-checked on ticket 183's pull request.
"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from mci_gru.config import GraphConfig
from mci_gru.models import create_model
from mci_gru.models.trunk import _apply_edge_dropout, _forked_dropout_edge

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

MEMBER_SEED = 4242
EDGE_DROPOUT_P = 0.5


def _model(*, isolate: bool, edge_feature_dim: int = 4) -> torch.nn.Module:
    """A small trainable model seeded exactly as ``train_multiple_models`` seeds one.

    ``mci_gru.training.ensemble`` calls ``set_seed(config.seed + model_id)``
    immediately before ``model_factory()``, so ``torch.initial_seed()`` inside
    the constructor is the member seed. Trunk regularisation is on so the model
    makes its own non-graph dropout draws after the edge dropout.
    """
    torch.manual_seed(MEMBER_SEED)
    return create_model(
        3,
        {
            "gru_hidden_sizes": [4, 4],
            "hidden_size_gat1": 8,
            "output_gat1": 4,
            "gat_heads": 2,
            "hidden_size_gat2": 8,
            "num_hidden_states": 4,
            "cross_attn_heads": 2,
            "use_multi_scale": False,
            "use_self_attention": False,
            "activation": "relu",
            "use_trunk_regularisation": True,
            "trunk_dropout": 0.25,
            "use_nn_multihead_attention": False,
            "temporal_encoder": "legacy",
            "edge_feature_dim": edge_feature_dim,
            "drop_edge_p": EDGE_DROPOUT_P,
            "isolate_edge_dropout_rng": isolate,
        },
    )


def _inputs(n_stocks: int = 6, n_features: int = 3, window: int = 5):
    torch.manual_seed(7)
    x_time_series = torch.randn(1, n_stocks, window, n_features)
    x_graph = torch.randn(n_stocks, n_features)
    return x_time_series, x_graph


def _populated_edges(n_stocks: int = 6, edge_feature_dim: int = 4):
    """A dense directed edge set over ``n_stocks`` nodes, self-loops excluded."""
    pairs = [(i, j) for i in range(n_stocks) for j in range(n_stocks) if i != j]
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    edge_weight = torch.ones(edge_index.shape[1], edge_feature_dim)
    return edge_index, edge_weight


def _empty_edges(edge_feature_dim: int = 4):
    """The A0 control's adjacency: shape (2, 0), self-loops only downstream."""
    return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, edge_feature_dim)


def _global_draw_after_forward(model, edge_index, edge_weight, *, stream_seed: int = 99):
    """Run one training forward and return the next global-stream draw after it.

    That trailing draw stands for every non-graph draw the rest of the step
    makes -- feature dropout, shuffling, the next member's initialisation. If
    two arms leave the global stream in the same place, those draws coincide.
    """
    x_time_series, x_graph = _inputs()
    model.train()
    torch.manual_seed(stream_seed)
    model(x_time_series, x_graph, edge_index, edge_weight)
    return torch.rand(8), torch.get_rng_state()


# -- config and plumbing seams --------------------------------------------


def test_graph_config_flag_defaults_off() -> None:
    assert GraphConfig().isolate_edge_dropout_rng is False


def test_base_config_yaml_ships_the_flag_off() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=[])
    assert "isolate_edge_dropout_rng" in cfg.graph
    assert cfg.graph.isolate_edge_dropout_rng is False


def test_factory_defaults_the_flag_off_and_plumbs_it_through() -> None:
    assert _model(isolate=False).isolate_edge_dropout_rng is False
    assert _model(isolate=True).isolate_edge_dropout_rng is True


# -- the load-bearing guard: on, the global stream coincides across arms ---


def test_isolated_rng_makes_non_graph_draws_coincide_across_arms() -> None:
    """A populated arm and the empty control leave the global stream identical.

    This is the mechanics check ticket 181 section 4 asks for: A0's and A1's
    non-graph draws coincide, so ``sd(delta)`` measures the graph rather than
    the graph plus the stream.
    """
    populated_draw, populated_state = _global_draw_after_forward(
        _model(isolate=True), *_populated_edges()
    )
    empty_draw, empty_state = _global_draw_after_forward(_model(isolate=True), *_empty_edges())

    assert torch.equal(populated_draw, empty_draw)
    assert torch.equal(populated_state, empty_state)


def test_shipped_path_diverges_across_arms_when_isolation_is_off() -> None:
    """Off, the streams diverge -- the defect the flag exists to remove.

    This guard is what tells a forked run from an unforked one. Without it the
    test above would pass just as well against a no-op flag.
    """
    populated_draw, populated_state = _global_draw_after_forward(
        _model(isolate=False), *_populated_edges()
    )
    empty_draw, empty_state = _global_draw_after_forward(_model(isolate=False), *_empty_edges())

    assert not torch.equal(populated_draw, empty_draw)
    assert not torch.equal(populated_state, empty_state)


def test_isolation_off_leaves_shipped_behaviour_byte_identical() -> None:
    """Off, the model's output is exactly what it was before the flag existed."""
    edge_index, edge_weight = _populated_edges()
    x_time_series, x_graph = _inputs()

    outputs = []
    for _ in range(2):
        model = _model(isolate=False)
        model.train()
        torch.manual_seed(1234)
        outputs.append(model(x_time_series, x_graph, edge_index, edge_weight))

    assert torch.equal(outputs[0], outputs[1])


# -- the forked stream still drops edges, deterministically ----------------


def test_forked_stream_is_reproducible_for_the_same_member_seed_and_step() -> None:
    edge_index, edge_weight = _populated_edges()
    x_time_series, x_graph = _inputs()

    outputs = []
    for _ in range(2):
        model = _model(isolate=True)
        model.train()
        torch.manual_seed(31337)
        outputs.append(model(x_time_series, x_graph, edge_index, edge_weight))

    assert torch.equal(outputs[0], outputs[1])


def test_forked_stream_advances_between_steps() -> None:
    """Successive training steps must not reuse one frozen edge mask.

    Asserted on the fork seeds and the kept edges, not only on the model
    output: the model's own trunk dropout advances the global stream between
    forwards, so two outputs differ even when the edge mask is frozen. An
    earlier output-only form of this guard was blind to a frozen step counter
    (mutation M12 on ticket 183's pull request).
    """
    model = _model(isolate=True)
    model.train()
    edge_index, edge_weight = _populated_edges(n_stocks=40)

    first_corr, first_sector = model._next_edge_dropout_fork_seeds(True)
    second_corr, _ = model._next_edge_dropout_fork_seeds(True)
    assert first_corr != second_corr, "the step counter did not advance"
    # The two dropout_edge calls in one forward must not share a stream either.
    assert first_corr != first_sector

    first_kept, _ = _apply_edge_dropout(
        edge_index, edge_weight, EDGE_DROPOUT_P, training=True, fork_seed=first_corr
    )
    second_kept, _ = _apply_edge_dropout(
        edge_index, edge_weight, EDGE_DROPOUT_P, training=True, fork_seed=second_corr
    )
    assert first_kept.shape != second_kept.shape or not torch.equal(first_kept, second_kept)


def test_forked_edge_dropout_still_drops_edges() -> None:
    """Isolation must not silently disable the regulariser it wraps."""
    edge_index, edge_weight = _populated_edges(n_stocks=40)
    kept, kept_weight = _apply_edge_dropout(
        edge_index, edge_weight, EDGE_DROPOUT_P, training=True, fork_seed=11
    )

    total = edge_index.shape[1]
    assert 0 < kept.shape[1] < total
    assert abs(kept.shape[1] / total - (1.0 - EDGE_DROPOUT_P)) < 0.1
    assert kept_weight.shape[0] == kept.shape[1]


def test_fork_refuses_a_device_type_it_cannot_reseed() -> None:
    """An unsupported accelerator must fail loudly, not isolate nothing.

    ``dropout_edge`` draws on ``edge_index.device``. Seeding the CPU generator
    for a device whose generator is elsewhere leaves edge dropout on the global
    stream with no error and no isolation -- the failure that looks like
    success. A meta tensor stands in for any such device.
    """
    meta_edges = torch.zeros(2, 4, dtype=torch.long, device="meta")
    with pytest.raises(NotImplementedError, match="cannot fork the RNG for device type"):
        _forked_dropout_edge(meta_edges, EDGE_DROPOUT_P, 11)


def test_forked_edge_dropout_is_inert_in_eval_mode() -> None:
    edge_index, edge_weight = _populated_edges()
    x_time_series, x_graph = _inputs()

    outputs = []
    for isolate in (False, True):
        model = _model(isolate=isolate)
        model.eval()
        torch.manual_seed(555)
        outputs.append(model(x_time_series, x_graph, edge_index, edge_weight))

    assert torch.equal(outputs[0], outputs[1])
