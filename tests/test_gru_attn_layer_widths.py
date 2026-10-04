"""Issue 131: how ``gru_attn`` reads ``model.gru_hidden_sizes``.

The shipped ``gru_attn`` encoder builds ``[32, 10]`` as two layers of width 10:
the list length is the layer count and only the last entry is a width.
``model.gru_attn_layer_widths=per_layer`` builds one layer per entry at that
entry's width, so ``[32, 10]`` is a 32-wide layer feeding a 10-wide one. The
default stays ``shared`` so checkpoints from the shipped form still load.

Tested at ``GRUWithAttention``, ``create_model`` and ``ModelConfig``. Every
assertion that the new form builds 32-then-10 is paired with one that the
default still builds 10-and-10, so neither passes for the other's reason.
"""

import pytest
import torch
from torch import nn

from mci_gru.config import ModelConfig
from mci_gru.models import GRUWithAttention, create_model

INPUT_SIZE = 6


def _layer_shapes(encoder: GRUWithAttention) -> list[tuple[int, int]]:
    """(input width, hidden width) of every recurrent layer, in order."""
    grus = [m for m in encoder.modules() if isinstance(m, nn.GRU)]
    shapes = []
    for gru in grus:
        for layer in range(gru.num_layers):
            in_width = gru.input_size if layer == 0 else gru.hidden_size
            shapes.append((in_width, gru.hidden_size))
    return shapes


def _model_config(**overrides) -> dict:
    cfg = ModelConfig(temporal_encoder="gru_attn", use_nn_multihead_attention=True, **overrides)
    return {**cfg.to_dict(), "edge_feature_dim": 4}


def test_per_layer_builds_each_entry_at_its_own_width():
    encoder = GRUWithAttention(INPUT_SIZE, [32, 10], layer_widths="per_layer")
    assert _layer_shapes(encoder) == [(INPUT_SIZE, 32), (32, 10)]
    assert encoder.output_size == 10


def test_shared_still_builds_two_layers_of_the_last_width():
    """Control: the shipped form is unchanged, so old checkpoints still match it."""
    encoder = GRUWithAttention(INPUT_SIZE, [32, 10])
    assert _layer_shapes(encoder) == [(INPUT_SIZE, 10), (10, 10)]
    assert {name for name, _ in encoder.named_parameters()} >= {
        "gru.weight_ih_l0",
        "gru.weight_ih_l1",
    }
    assert not any(name.startswith("grus.") for name, _ in encoder.named_parameters())


def test_per_layer_forward_and_sequence_shapes_and_gradients():
    torch.manual_seed(0)
    encoder = GRUWithAttention(INPUT_SIZE, [32, 10], layer_widths="per_layer")
    x = torch.randn(2, 5, 7, INPUT_SIZE, requires_grad=True)
    y = encoder(x)
    assert y.shape == (2, 5, 10)
    assert encoder.forward_sequence(x).shape == (2, 5, 7, 10)
    y.sum().backward()
    assert torch.isfinite(y).all()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_per_layer_output_depends_on_the_first_layer():
    """The 32-wide layer must feed the 10-wide one, not sit beside it."""
    torch.manual_seed(0)
    encoder = GRUWithAttention(INPUT_SIZE, [32, 10], layer_widths="per_layer")
    x = torch.randn(1, 3, 7, INPUT_SIZE)
    before = encoder(x)
    with torch.no_grad():
        encoder.grus[0].weight_ih_l0.add_(1.0)
    assert not torch.allclose(before, encoder(x))


def test_create_model_routes_the_setting_to_both_multi_scale_branches():
    model = create_model(INPUT_SIZE, _model_config(gru_attn_layer_widths="per_layer"))
    for branch in (model.temporal_encoder.fast_gru, model.temporal_encoder.slow_gru):
        assert _layer_shapes(branch) == [(INPUT_SIZE, 32), (32, 10)]


def test_create_model_default_keeps_the_shared_form():
    """Control for the routing test: without the key the model is the shipped one."""
    model = create_model(INPUT_SIZE, _model_config())
    for branch in (model.temporal_encoder.fast_gru, model.temporal_encoder.slow_gru):
        assert _layer_shapes(branch) == [(INPUT_SIZE, 10), (10, 10)]


def test_create_model_routes_the_setting_without_multi_scale():
    model = create_model(
        INPUT_SIZE, _model_config(gru_attn_layer_widths="per_layer", use_multi_scale=False)
    )
    assert _layer_shapes(model.temporal_encoder) == [(INPUT_SIZE, 32), (32, 10)]


def test_model_config_validates_and_serialises_the_setting():
    with pytest.raises(ValueError, match="gru_attn_layer_widths"):
        ModelConfig(gru_attn_layer_widths="wide")
    assert ModelConfig().gru_attn_layer_widths == "shared"
    cfg = ModelConfig(gru_attn_layer_widths="per_layer")
    assert cfg.to_dict()["gru_attn_layer_widths"] == "per_layer"


def test_encoder_rejects_an_unknown_layer_widths_value():
    with pytest.raises(ValueError, match="layer_widths"):
        GRUWithAttention(INPUT_SIZE, [32, 10], layer_widths="wide")
