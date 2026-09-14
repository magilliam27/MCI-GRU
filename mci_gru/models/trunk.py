import torch
import torch.nn as nn
from torch_geometric.utils import dropout_edge

from mci_gru.models.attention import SelfAttention
from mci_gru.models.graph import GATBlock
from mci_gru.models.latent import MarketLatentStateLearner
from mci_gru.models.temporal import (
    CausalTransformerEncoder,
    GRUWithAttention,
    ImprovedGRU,
    MultiScaleTemporalEncoder,
)


def _make_output_activation(name: str) -> nn.Module:
    if name == "none" or name == "identity":
        return nn.Identity()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(
        f"Unsupported output activation: {name!r}. Choose 'none', 'elu', 'relu', or 'sigmoid'."
    )


def _maybe_ln_ch(dim: int, on: bool) -> nn.Module:
    return nn.LayerNorm(dim) if on else nn.Identity()


def _maybe_drop(p: float, on: bool) -> nn.Module:
    if not on or p <= 0.0:
        return nn.Identity()
    return nn.Dropout(p)


#: Odd stride mixing (step, stream) into a fork seed. Two ``dropout_edge`` calls
#: happen per training forward -- correlation then sector -- and consecutive
#: steps must not reuse a mask, so the stride separates both axes.
_EDGE_DROPOUT_STREAM_STRIDE = 2_654_435_761
_EDGE_DROPOUT_SEED_MODULUS = 1 << 63
#: Which of the forward's two edge-dropout calls a fork seed belongs to.
_EDGE_DROPOUT_STREAM_CORRELATION = 0
_EDGE_DROPOUT_STREAM_SECTOR = 1
_EDGE_DROPOUT_STREAMS_PER_STEP = 2
#: Device types whose generator ``_forked_dropout_edge`` knows how to reseed.
#: Anything else would seed the CPU generator while ``dropout_edge`` drew from
#: the accelerator's, so isolation would silently become a no-op -- which is the
#: failure that looks like success. Raise instead; add the device here when one
#: is actually supported.
_EDGE_DROPOUT_FORKABLE_DEVICES = ("cpu", "cuda")


def _edge_dropout_fork_seed(base_seed: int, step: int, stream: int) -> int:
    """Seed for one forked edge-dropout draw, from the member seed and position."""
    offset = (step * _EDGE_DROPOUT_STREAMS_PER_STEP + stream) * _EDGE_DROPOUT_STREAM_STRIDE
    return (int(base_seed) + offset) % _EDGE_DROPOUT_SEED_MODULUS


def _forked_dropout_edge(
    edge_index: torch.Tensor, p: float, fork_seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``dropout_edge`` with the global RNG saved, reseeded, and restored.

    PyG 2.7.0's ``dropout_edge`` takes no generator argument -- it calls
    ``torch.rand`` on ``edge_index.device`` -- so the isolation has to fork
    around the call rather than be passed into it. Only the one generator that
    call consults is reseeded, so a multi-device process keeps every other
    stream intact.
    """
    device = edge_index.device
    if device.type not in _EDGE_DROPOUT_FORKABLE_DEVICES:
        raise NotImplementedError(
            f"graph.isolate_edge_dropout_rng cannot fork the RNG for device type "
            f"{device.type!r}; supported: {_EDGE_DROPOUT_FORKABLE_DEVICES}. Refusing "
            "rather than silently leaving edge dropout on the global stream."
        )
    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices, enabled=True):
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(fork_seed)
        else:
            torch.default_generator.manual_seed(fork_seed)
        return dropout_edge(edge_index, p=p, force_undirected=False, training=True)


def _apply_edge_dropout(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    p: float,
    training: bool,
    fork_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (not training) or p <= 0.0 or p >= 1.0 or edge_index.numel() == 0:
        return edge_index, edge_weight
    if fork_seed is None:
        e_new, edge_mask = dropout_edge(edge_index, p=p, force_undirected=False, training=True)
    else:
        e_new, edge_mask = _forked_dropout_edge(edge_index, p, fork_seed)
    w = edge_weight[edge_mask] if edge_weight is not None else edge_weight
    return e_new, w


class StockPredictionModel(nn.Module):
    """
    MCI-GRU: temporal encoder + GAT + latent cross-attn + (optional) self-attn
    + prediction GAT. Four streams [A1, A2, B1, B2] are built in that order
    in ``forward``; do not change without updating :class:`SelfAttention` docs.
    """

    def __init__(
        self,
        input_size: int,
        gru_hidden_sizes: list[int] = None,
        hidden_size_gat1: int = 32,
        output_gat1: int = 4,
        gat_heads: int = 4,
        hidden_size_gat2: int = 32,
        num_hidden_states: int = 32,
        cross_attn_heads: int = 4,
        slow_kernel: int = 5,
        slow_stride: int = 2,
        use_multi_scale: bool = True,
        use_self_attention: bool = True,
        activation: str = "elu",
        output_activation: str = "none",
        latent_init_scale: float = 0.02,
        edge_feature_dim: int = 1,
        use_group_type_embed: bool = False,
        use_trunk_regularisation: bool = False,
        trunk_dropout: float = 0.1,
        use_nn_multihead_attention: bool = False,
        temporal_encoder: str = "legacy",
        drop_edge_p: float = 0.0,
        isolate_edge_dropout_rng: bool = False,
        use_sector_relation: bool = False,
        use_a1_a2_cross_attention: bool = False,
        cross_a2_num_heads: int = 4,
        market_latent_mode: str = "static",
    ):
        super().__init__()
        if gru_hidden_sizes is None:
            gru_hidden_sizes = [32, 10]

        self._align_dim = hidden_size_gat1
        tr = use_trunk_regularisation
        tdrop = trunk_dropout if tr else 0.0
        self._use_trunk_regularisation = tr
        gat_inter_drop = tdrop if tr else 0.0
        self.use_sector_relation = use_sector_relation
        self.use_a1_a2_cross_attention = use_a1_a2_cross_attention

        # Temporal: legacy ImprovedGRU vs GRU+attention (or multi-scale variants)
        if use_multi_scale:
            self.temporal_encoder = MultiScaleTemporalEncoder(
                input_size,
                hidden_sizes=gru_hidden_sizes,
                slow_kernel=slow_kernel,
                slow_stride=slow_stride,
                temporal_encoder=temporal_encoder,
            )
        elif temporal_encoder == "legacy":
            self.temporal_encoder = ImprovedGRU(input_size, hidden_sizes=gru_hidden_sizes)
        elif temporal_encoder == "transformer":
            self.temporal_encoder = CausalTransformerEncoder(input_size, gru_hidden_sizes[-1])
        else:
            self.temporal_encoder = GRUWithAttention(input_size, gru_hidden_sizes)

        gru_output_size = self.temporal_encoder.output_size
        self.edge_feature_dim = edge_feature_dim
        self.gat_layer = GATBlock(
            in_channels=input_size,
            hidden=hidden_size_gat1,
            out_channels=output_gat1,
            heads=gat_heads,
            activation=activation,
            edge_feature_dim=edge_feature_dim,
            inter_layer_dropout=gat_inter_drop,
        )
        if use_sector_relation:
            self.gat_layer_sector = GATBlock(
                in_channels=input_size,
                hidden=hidden_size_gat1,
                out_channels=output_gat1,
                heads=gat_heads,
                activation=activation,
                edge_feature_dim=1,
                inter_layer_dropout=gat_inter_drop,
            )
            self.gat_stream_fuse = nn.Linear(output_gat1 * 2, output_gat1)
        else:
            self.gat_layer_sector = None
            self.gat_stream_fuse = None

        self.align_dim = hidden_size_gat1
        self.proj_temporal = nn.Linear(gru_output_size, self.align_dim)
        self.proj_cross = nn.Linear(output_gat1, self.align_dim)

        if use_a1_a2_cross_attention:
            self.proj_a1_seq = nn.Linear(gru_output_size, self.align_dim)
            self.cross_a1_a2 = nn.MultiheadAttention(
                self.align_dim,
                cross_a2_num_heads,
                batch_first=True,
                dropout=tdrop,
            )
        else:
            self.proj_a1_seq = None
            self.cross_a1_a2 = None

        self.ln_a1 = _maybe_ln_ch(self.align_dim, tr)
        self.ln_a2 = _maybe_ln_ch(self.align_dim, tr)

        self.latent_learner = MarketLatentStateLearner(
            feature_dim=self.align_dim,
            num_latent_states=num_hidden_states,
            num_heads=cross_attn_heads,
            latent_init_scale=latent_init_scale,
            use_nn_multihead_attention=use_nn_multihead_attention,
            attn_dropout=tdrop,
            market_latent_mode=market_latent_mode,
        )
        self.concat_size = 4 * self.align_dim
        self.ln_z = _maybe_ln_ch(self.concat_size, tr)
        self.drop_z = _maybe_drop(tdrop, tr)

        if use_self_attention:
            self.self_attention: SelfAttention | None = SelfAttention(
                embed_dim=self.concat_size,
                align_dim=self.align_dim,
                use_group_type_embed=use_group_type_embed,
            )
        else:
            self.self_attention = None

        self.final_gat = GATBlock(
            in_channels=self.concat_size,
            hidden=hidden_size_gat2,
            out_channels=1,
            heads=gat_heads,
            activation=activation,
            edge_feature_dim=edge_feature_dim,
            inter_layer_dropout=gat_inter_drop,
        )
        self.output_act = _make_output_activation(output_activation)
        self.drop_edge_p = float(drop_edge_p)
        self.isolate_edge_dropout_rng = bool(isolate_edge_dropout_rng)
        # ``train_multiple_models`` calls ``set_seed(config.seed + model_id)``
        # immediately before ``model_factory()``, so reading the seed here
        # captures this member's seed without consuming a draw from the global
        # stream -- which is the point: arms with the flag on must consume the
        # stream identically whatever their edge count. The step counter is
        # transient training state, deliberately not a buffer: a resumed model
        # restarts the edge-dropout sequence and its non-graph draws still
        # coincide with every other arm's.
        self._edge_dropout_seed = int(torch.initial_seed())
        self._edge_dropout_step = 0

    def _temporal_fast_sequence(self, x_time_series: torch.Tensor) -> torch.Tensor:
        enc = self.temporal_encoder
        if isinstance(enc, MultiScaleTemporalEncoder):
            return enc.forward_fast_sequence(x_time_series)
        if hasattr(enc, "forward_sequence"):
            return enc.forward_sequence(x_time_series)
        raise TypeError("Temporal encoder does not expose a sequence for cross-attention")

    def _next_edge_dropout_fork_seeds(self, training: bool) -> tuple[int | None, int | None]:
        """Fork seeds for this step's correlation and sector edge dropout.

        ``(None, None)`` when isolation is off or outside training, which is the
        shipped path: ``_apply_edge_dropout`` then draws from the global stream
        exactly as it always has.
        """
        if not (training and self.isolate_edge_dropout_rng):
            return None, None
        # The counter advances once per training forward whatever the arm's edge
        # count, including when a seed goes unused because the arm has no edges
        # or dropout is off. That is the point rather than an oversight: it is a
        # position in the forward, not a count of masks drawn, and every arm
        # must be at the same position after the same number of steps.
        step = self._edge_dropout_step
        self._edge_dropout_step = step + 1
        return (
            _edge_dropout_fork_seed(
                self._edge_dropout_seed, step, _EDGE_DROPOUT_STREAM_CORRELATION
            ),
            _edge_dropout_fork_seed(self._edge_dropout_seed, step, _EDGE_DROPOUT_STREAM_SECTOR),
        )

    def forward(
        self,
        x_time_series: torch.Tensor,
        x_graph: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_stocks: int | None = None,
        edge_index_sector: torch.Tensor | None = None,
        edge_weight_sector: torch.Tensor | None = None,
        stock_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        training = self.training
        batch_size = x_time_series.shape[0]
        if num_stocks is None:
            num_stocks = x_time_series.shape[1]

        node_mask = None
        if stock_mask is not None:
            stock_mask = stock_mask.to(dtype=torch.bool, device=x_time_series.device)
            node_mask = stock_mask.reshape(batch_size * num_stocks, 1).to(x_graph.dtype)
            x_time_series = x_time_series * stock_mask[:, :, None, None].to(x_time_series.dtype)
            x_graph = x_graph * node_mask

        corr_fork_seed, sector_fork_seed = self._next_edge_dropout_fork_seeds(training)
        e_idx, e_wt = _apply_edge_dropout(
            edge_index, edge_weight, self.drop_edge_p, training=training, fork_seed=corr_fork_seed
        )
        # Sector fusion falls back to correlation-only when the sector tensors are
        # absent. That fallback is deliberate and no reachable training config hits
        # it: prepare_data always builds the tensors, and index-level runs (which
        # cannot build them) are rejected in ExperimentConfig validation.
        if (
            self.use_sector_relation
            and self.gat_layer_sector is not None
            and edge_index_sector is not None
            and edge_weight_sector is not None
        ):
            e_idx_s, e_wt_s = _apply_edge_dropout(
                edge_index_sector,
                edge_weight_sector,
                self.drop_edge_p,
                training=training,
                fork_seed=sector_fork_seed,
            )
        else:
            e_idx_s, e_wt_s = None, None

        a1_raw = self.temporal_encoder(x_time_series)
        a1_raw = a1_raw.reshape(batch_size * num_stocks, -1)
        a1 = self.proj_temporal(a1_raw)
        a1 = self.ln_a1(a1)
        if node_mask is not None:
            a1 = a1 * node_mask.to(a1.dtype)

        a2_corr = self.gat_layer(x_graph, e_idx, e_wt)
        if self.gat_layer_sector is not None and e_idx_s is not None and e_wt_s is not None:
            a2_sec = self.gat_layer_sector(x_graph, e_idx_s, e_wt_s)
            a2_raw = self.gat_stream_fuse(torch.cat([a2_corr, a2_sec], dim=-1))
        else:
            a2_raw = a2_corr

        a2 = self.proj_cross(a2_raw)
        if (
            self.use_a1_a2_cross_attention
            and self.cross_a1_a2 is not None
            and self.proj_a1_seq is not None
        ):
            seq = self._temporal_fast_sequence(x_time_series)
            seq = self.proj_a1_seq(seq)
            bn = batch_size * num_stocks
            tlen = seq.shape[2]
            q = a2.view(batch_size, num_stocks, -1).reshape(bn, 1, self.align_dim)
            kv = seq.reshape(bn, tlen, self.align_dim)
            cross_out, _ = self.cross_a1_a2(q, kv, kv, need_weights=False)
            a2 = a2 + cross_out.reshape(batch_size * num_stocks, -1)
        a2 = self.ln_a2(a2)
        if node_mask is not None:
            a2 = a2 * node_mask.to(a2.dtype)

        b1, b2 = self.latent_learner(a1, a2, num_stocks=num_stocks, stock_mask=stock_mask)
        if node_mask is not None:
            b1 = b1 * node_mask.to(b1.dtype)
            b2 = b2 * node_mask.to(b2.dtype)
        # Contract: A1, A2, B1, B2 order for SelfAttention group_type_embed slots 0..3
        z = torch.cat([a1, a2, b1, b2], dim=-1)
        z = self.ln_z(z)
        z = self.drop_z(z)
        if node_mask is not None:
            z = z * node_mask.to(z.dtype)

        if self.self_attention is not None:
            z = z.view(batch_size, num_stocks, -1)
            z = self.self_attention(z, stock_mask=stock_mask)
            z = z.view(batch_size * num_stocks, -1)
        out = self.final_gat(z, e_idx, e_wt)
        out = self.output_act(out)
        if node_mask is not None:
            out = out * node_mask.to(out.dtype)
        return out.view(batch_size, num_stocks)
