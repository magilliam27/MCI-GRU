from mci_gru.config import GraphConfig
from mci_gru.graph.utils import edge_feature_dim


def test_shared_edge_dim_helper_covers_phase3_edge_widths():
    assert edge_feature_dim(GraphConfig(use_multi_feature_edges=False)) == 1
    assert edge_feature_dim(GraphConfig(use_multi_feature_edges=True)) == 4
    assert (
        edge_feature_dim(
            GraphConfig(use_multi_feature_edges=True, use_lead_lag_features=True)
        )
        == 6
    )
    assert (
        edge_feature_dim(
            GraphConfig(
                use_multi_feature_edges=True,
                use_lead_lag_features=True,
                append_snapshot_age_days=True,
            )
        )
        == 7
    )
