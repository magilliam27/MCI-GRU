"""Phase 2 model flags: self-attention type embed, MHA path shapes, encoders."""
import torch

from mci_gru.models import create_model, GRUWithAttention, ImprovedGRU, MarketLatentStateLearner


def test_group_type_embed_on_has_embedding_params():
    m = create_model(
        5,
        {
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
            "use_group_type_embed": True,
            "use_trunk_regularisation": False,
            "use_nn_multihead_attention": False,
            "temporal_encoder": "legacy",
        },
    )
    assert m.self_attention is not None
    assert m.self_attention.type_embed is not None
    names = [k for k, _ in m.self_attention.named_parameters()]
    assert any("type_embed" in k for k in names)


def test_market_learner_legacy_and_mha_same_shape():
    f = 16
    nq = 10
    a1 = torch.randn(nq, f)
    a2 = torch.randn(nq, f)
    leg = MarketLatentStateLearner(
        f, num_latent_states=4, num_heads=4, use_nn_multihead_attention=False
    )
    mha = MarketLatentStateLearner(
        f, num_latent_states=4, num_heads=4, use_nn_multihead_attention=True, attn_dropout=0.0
    )
    b1l, b2l = leg(a1, a2)
    b1h, b2h = mha(a1, a2)
    assert b1l.shape == (nq, f) and b1h.shape == (nq, f)


def test_temporal_legacy_vs_gru_attn_shape_and_grad():
    b, n, t, f = 1, 4, 5, 6
    x = torch.randn(b, n, t, f, requires_grad=True)
    leg = ImprovedGRU(f, [4, 4])
    y1 = leg(x)
    y1.sum().backward()
    assert y1.shape == (b, n, 4)
    x2 = torch.randn(b, n, t, f, requires_grad=True)
    g = GRUWithAttention(f, [4, 4])
    y2 = g(x2)
    y2.sum().backward()
    assert y2.shape == (b, n, 4)
    assert x2.grad is not None


def test_paper_faithful_create_model_no_mha_no_selfattn():
    cfg = {
        "gru_hidden_sizes": [32, 10],
        "hidden_size_gat1": 32,
        "output_gat1": 4,
        "gat_heads": 4,
        "hidden_size_gat2": 32,
        "num_hidden_states": 32,
        "cross_attn_heads": 4,
        "use_multi_scale": False,
        "use_self_attention": False,
        "activation": "relu",
        "output_activation": "relu",
        "use_group_type_embed": False,
        "use_trunk_regularisation": False,
        "use_nn_multihead_attention": False,
        "temporal_encoder": "legacy",
    }
    m = create_model(8, cfg)
    assert m.self_attention is None
    assert m.latent_learner.mha1 is None


def test_forward_smoke_with_drop_edge_p():
    m = create_model(
        7,
        {
            "gru_hidden_sizes": [4, 4],
            "hidden_size_gat1": 8,
            "use_multi_scale": False,
            "use_self_attention": False,
            "activation": "relu",
            "use_trunk_regularisation": False,
            "use_nn_multihead_attention": False,
            "temporal_encoder": "legacy",
            "drop_edge_p": 0.0,
        },
    )
    m.train()
    x = torch.randn(1, 3, 4, 7)
    g = torch.randn(3, 7)
    ei = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    ew = torch.randn(3, 1)
    y = m(x, g, ei, ew, 3)
    assert y.shape == (1, 3)
    m.eval()
    y2 = m(x, g, ei, ew, 3)
    assert y2.shape == (1, 3)
