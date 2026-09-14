from typing import Any

from mci_gru.models.trunk import StockPredictionModel


def create_model(input_size: int, config: dict[str, Any]) -> StockPredictionModel:
    """
    Build a model. Missing keys default to *legacy* shapes so old ``config.yaml``
    in checkpoint dirs still load; new runs should set all keys in Hydra.
    """
    act = config.get("activation", "elu")
    out = config.get("output_activation")
    if out is None:
        out = act
    tr = config.get("use_trunk_regularisation", False)
    tdrop = config.get("trunk_dropout", 0.1) if tr else 0.0
    if not tr:
        tdrop = 0.0
    return StockPredictionModel(
        input_size=input_size,
        gru_hidden_sizes=config.get("gru_hidden_sizes", [32, 10]),
        hidden_size_gat1=config.get("hidden_size_gat1", 32),
        output_gat1=config.get("output_gat1", 4),
        gat_heads=config.get("gat_heads", 4),
        hidden_size_gat2=config.get("hidden_size_gat2", 32),
        num_hidden_states=config.get("num_hidden_states", 32),
        cross_attn_heads=config.get("cross_attn_heads", 4),
        slow_kernel=config.get("slow_kernel", 5),
        slow_stride=config.get("slow_stride", 2),
        use_multi_scale=config.get("use_multi_scale", True),
        use_self_attention=config.get("use_self_attention", True),
        activation=act,
        output_activation=out,
        latent_init_scale=config.get("latent_init_scale", 0.02),
        edge_feature_dim=config.get("edge_feature_dim", 1),
        use_group_type_embed=config.get("use_group_type_embed", False),
        use_trunk_regularisation=tr,
        trunk_dropout=tdrop,
        use_nn_multihead_attention=config.get("use_nn_multihead_attention", False),
        temporal_encoder=config.get("temporal_encoder", "legacy"),
        drop_edge_p=float(config.get("drop_edge_p", 0.0)),
        isolate_edge_dropout_rng=bool(config.get("isolate_edge_dropout_rng", False)),
        use_sector_relation=bool(config.get("use_sector_relation", False)),
        use_a1_a2_cross_attention=bool(config.get("use_a1_a2_cross_attention", False)),
        cross_a2_num_heads=int(config.get("cross_a2_num_heads", 4)),
        market_latent_mode=config.get("market_latent_mode", "static"),
    )
