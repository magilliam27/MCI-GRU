import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from mci_gru.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
)
from mci_gru.data.data_manager import CombinedDataset, combined_collate_fn
from mci_gru.data.pit import (
    apply_label_mask,
    build_pit_masks,
    candidate_breadth,
    filter_edges_by_stock_mask,
)
from mci_gru.evaluation.portfolio import top_k_returns
from mci_gru.models.mci_gru import SelfAttention
from mci_gru.pipeline import prepare_data
from mci_gru.training.losses import ICLoss, MaskedMSELoss, mean_information_coefficient
from mci_gru.training.trainer import prediction_rows_for_date


def _panel(kdcodes: list[str], dates: list[str]) -> pd.DataFrame:
    rows = []
    for sidx, kdcode in enumerate(kdcodes):
        for didx, date in enumerate(dates):
            close = 100.0 + sidx * 10.0 + didx
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": date,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0 + didx,
                    "turnover": close * (1000.0 + didx),
                }
            )
    return pd.DataFrame(rows)


def test_future_joiner_uses_price_history_but_enters_masks_on_valid_from() -> None:
    dates = [f"2020-01-{i:02d}" for i in range(1, 9)]
    kdcodes = ["AAA", "NEW"]
    panel = _panel(kdcodes, dates)
    intervals = pd.DataFrame(
        [
            {"kdcode": "AAA", "valid_from": "2020-01-01", "valid_to": "2020-01-08"},
            {"kdcode": "NEW", "valid_from": "2020-01-06", "valid_to": "2020-01-08"},
        ]
    )

    masks = build_pit_masks(
        df_for_features=panel,
        df_for_labels=panel,
        kdcode_list=kdcodes,
        sample_dates=["2020-01-05", "2020-01-06", "2020-01-07"],
        his_t=3,
        label_t=1,
        pit_intervals=intervals,
    )

    assert masks.feature_ready.tolist() == [[True, True], [True, True], [True, True]]
    assert masks.active_member.tolist() == [[True, False], [True, True], [True, True]]
    assert masks.tradable.tolist() == [[True, False], [True, True], [True, True]]
    assert masks.loss.tolist() == [[True, False], [True, True], [True, True]]


def test_leaver_rejoiner_masks_turn_off_and_back_on() -> None:
    dates = [f"2020-01-{i:02d}" for i in range(1, 9)]
    panel = _panel(["AAA"], dates)
    intervals = pd.DataFrame(
        [
            {"kdcode": "AAA", "valid_from": "2020-01-01", "valid_to": "2020-01-03"},
            {"kdcode": "AAA", "valid_from": "2020-01-06", "valid_to": "2020-01-08"},
        ]
    )

    masks = build_pit_masks(
        df_for_features=panel,
        df_for_labels=panel,
        kdcode_list=["AAA"],
        sample_dates=["2020-01-03", "2020-01-04", "2020-01-06"],
        his_t=2,
        label_t=1,
        pit_intervals=intervals,
    )

    assert masks.active_member[:, 0].tolist() == [True, False, True]
    assert masks.tradable[:, 0].tolist() == [True, False, True]


def test_apply_label_mask_preserves_invalid_targets_as_nan() -> None:
    labels = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    mask = np.array([[True, False, True]])

    masked = apply_label_mask(labels, mask)

    assert masked[0, 0] == pytest.approx(0.1)
    assert np.isnan(masked[0, 1])
    assert masked[0, 2] == pytest.approx(0.3)


def test_candidate_breadth_reports_scoreable_counts() -> None:
    summary = candidate_breadth(
        ["2020-01-01", "2020-01-02"],
        np.array([[True, False, True], [True, True, True]]),
    )

    assert summary == [
        {"date": "2020-01-01", "scoreable_count": 2},
        {"date": "2020-01-02", "scoreable_count": 3},
    ]


def test_filter_edges_by_stock_mask_removes_inactive_nodes() -> None:
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 2]], dtype=torch.long)
    edge_weight = torch.arange(4, dtype=torch.float32)
    mask = torch.tensor([True, True, False])

    out_ei, out_ew = filter_edges_by_stock_mask(edge_index, edge_weight, mask)

    assert out_ei.tolist() == [[0, 1], [1, 0]]
    assert out_ew.tolist() == [0.0, 1.0]


def test_collate_keeps_9_tuple_and_carries_stock_mask_in_batch_meta() -> None:
    ts = torch.zeros((2, 3, 2, 1))
    graph = torch.zeros((2, 3, 1))
    labels = torch.zeros((2, 3))
    stock_mask = torch.tensor([[True, True, False], [True, False, True]])
    dataset = CombinedDataset(
        ts,
        graph,
        labels,
        sample_dates=["2020-01-03", "2020-01-04"],
        stock_masks=stock_mask,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda batch: combined_collate_fn(
            batch,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_weight=torch.zeros(0),
        ),
    )

    batch = next(iter(loader))
    assert len(batch) == 9
    assert batch[6]["dates"] == ["2020-01-03", "2020-01-04"]
    assert torch.equal(batch[6]["stock_mask"], stock_mask)


def test_masked_losses_ignore_nan_targets() -> None:
    pred = torch.tensor([[1.0, 2.0, 9.0], [1.0, 1.0, 1.0]])
    target = torch.tensor([[1.5, 2.5, float("nan")], [float("nan"), float("nan"), 3.0]])

    mse = MaskedMSELoss()(pred, target)
    ic = ICLoss()(pred, target)
    mean_ic = mean_information_coefficient(pred, target)

    assert mse.item() == pytest.approx(((1.0 - 1.5) ** 2 + (2.0 - 2.5) ** 2 + (1.0 - 3.0) ** 2) / 3)
    assert ic.item() == pytest.approx(-1.0)
    assert mean_ic.item() == pytest.approx(1.0)


def test_top_k_returns_ignores_invalid_candidates() -> None:
    predictions = np.array([[100.0, 2.0, 1.0]])
    returns = np.array([[np.nan, 0.2, 0.1]])

    out = top_k_returns(predictions, returns, top_k=1)

    assert out.tolist() == [pytest.approx(0.2)]


def test_self_attention_mask_prevents_inactive_node_influence() -> None:
    torch.manual_seed(7)
    attn = SelfAttention(embed_dim=4, align_dim=1)
    x = torch.randn(1, 3, 4)
    mask = torch.tensor([[True, True, False]])

    out_a = attn(x, stock_mask=mask)
    changed = x.clone()
    changed[:, 2, :] = 999.0
    out_b = attn(changed, stock_mask=mask)

    assert torch.allclose(out_a[:, :2, :], out_b[:, :2, :], atol=1e-6)
    assert torch.all(out_a[:, 2, :] == 0)


def test_prediction_rows_for_date_filters_to_tradable_mask() -> None:
    rows = prediction_rows_for_date(
        predictions=np.array([0.4, 0.9, -0.2], dtype=np.float32),
        kdcode_list=["AAA", "NEW", "OLD"],
        date="2020-01-06",
        prediction_mask=np.array([True, True, False]),
    )

    assert rows == [["AAA", "2020-01-06", 0.4], ["NEW", "2020-01-06", 0.9]]


class _PassThroughFeatureEngineer:
    def __init__(self, feature_cols: list[str]):
        self._feature_cols = feature_cols

    def transform(self, df, *_args):
        return df

    def get_feature_columns(self) -> list[str]:
        return self._feature_cols


def test_prepare_data_masked_panel_keeps_union_axis_without_complete_stock_filter(tmp_path) -> None:
    dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
    rows = []
    for kdcode in ["AAA", "NEW", "DROP"]:
        stock_dates = dates if kdcode != "DROP" else dates[:4]
        for didx, date in enumerate(stock_dates):
            close = 100.0 + didx + (10.0 if kdcode == "NEW" else 0.0)
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": date,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1000.0,
                    "turnover": close * 1000.0,
                }
            )
    data_path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    pit_path = tmp_path / "pit.csv"
    pd.DataFrame(
        [
            {"kdcode": "AAA", "valid_from": "2020-01-01", "valid_to": "2020-01-10"},
            {"kdcode": "NEW", "valid_from": "2020-01-05", "valid_to": "2020-01-10"},
            {"kdcode": "DROP", "valid_from": "2020-01-01", "valid_to": "2020-01-04"},
        ]
    ).to_csv(pit_path, index=False)

    cfg = ExperimentConfig(
        data=DataConfig(
            source="csv",
            filename=str(data_path),
            train_start="2020-01-01",
            train_end="2020-01-06",
            val_start="2020-01-07",
            val_end="2020-01-08",
            test_start="2020-01-09",
            test_end="2020-01-10",
            skip_embargo_check=True,
            use_pit_universe=True,
            pit_universe_csv=str(pit_path),
            pit_universe_mode="masked_panel",
            pit_min_scoreable_stocks=0,
        ),
        features=FeatureConfig(
            base_features=["close", "open", "high", "low", "volume", "turnover"],
            include_momentum=False,
            include_weekly_momentum=False,
        ),
        graph=GraphConfig(judge_value=0.9999, use_multi_feature_edges=False),
        model=ModelConfig(his_t=2, label_t=1),
        training=TrainingConfig(num_epochs=1, num_models=1, label_type="returns"),
        tracking=TrackingConfig(enabled=False),
    )

    data = prepare_data(
        cfg,
        _PassThroughFeatureEngineer(["close", "open", "high", "low", "volume", "turnover"]),
    )

    assert data["kdcode_list"] == ["AAA", "DROP", "NEW"]
    assert data["stock_features_train"].shape[1] == 3
    new_idx = data["kdcode_list"].index("NEW")
    assert data["train_dates"] == ["2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"]
    assert data["train_feature_ready_mask"][:, new_idx].tolist() == [True, True, True, True]
    assert data["train_tradable_mask"][:, new_idx].tolist() == [False, False, True, True]
    assert data["pit_breadth"]["train"][2]["scoreable_count"] == 2
