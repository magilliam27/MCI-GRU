from pathlib import Path

import torch

from paper_trade.scripts import infer


def test_load_frozen_model_and_graph_reads_graph_data_pt(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor(
        [[0.80, 0.80, 0.64, 1.00], [0.25, 0.25, 0.0625, 0.50]],
        dtype=torch.float32,
    )
    edge_index_sector = torch.tensor([[0], [1]], dtype=torch.long)
    edge_weight_sector = torch.tensor([1.0], dtype=torch.float32)
    torch.save(
        {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "edge_index_sector": edge_index_sector,
            "edge_weight_sector": edge_weight_sector,
        },
        model_dir / "graph_data.pt",
    )

    loaded = infer.load_frozen_model_and_graph(model_dir, torch.device("cpu"))

    loaded_edge_index, loaded_edge_weight, loaded_sector_index, loaded_sector_weight = loaded
    assert torch.equal(loaded_edge_index, edge_index)
    assert torch.equal(loaded_edge_weight, edge_weight)
    assert loaded_sector_index is not None
    assert loaded_sector_weight is not None
    assert torch.equal(loaded_sector_index, edge_index_sector)
    assert torch.equal(loaded_sector_weight, edge_weight_sector)


def test_paper_trade_infer_does_not_rebuild_graph_from_training_code() -> None:
    source = Path(infer.__file__).read_text(encoding="utf-8")

    assert "GraphBuilder" not in source
