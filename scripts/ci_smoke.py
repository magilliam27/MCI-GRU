"""Tiny end-to-end CI smoke for MCI-GRU."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_csv(path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=70)
    stocks = ["A.N", "B.N", "C.N", "D.N"]
    rows = []
    for s_idx, stock in enumerate(stocks):
        price = 100.0 + s_idx
        for d_idx, dt in enumerate(dates):
            drift = 0.001 * (s_idx + 1)
            seasonal = 0.002 * np.sin(d_idx / 4.0 + s_idx)
            open_px = price * (1.0 + seasonal)
            close_px = open_px * (1.0 + drift)
            rows.append(
                {
                    "kdcode": stock,
                    "dt": dt.strftime("%Y-%m-%d"),
                    "open": round(open_px, 4),
                    "high": round(max(open_px, close_px) * 1.001, 4),
                    "low": round(min(open_px, close_px) * 0.999, 4),
                    "close": round(close_px, 4),
                    "volume": 1_000_000 + s_idx * 1000 + d_idx,
                }
            )
            price = close_px
    pd.DataFrame(rows).to_csv(path, index=False)


def _assert_collate_contract() -> None:
    from functools import partial

    from torch.utils.data import DataLoader

    from mci_gru.data.data_manager import CombinedDataset, combined_collate_fn

    ts = torch.randn(2, 4, 3, 2)
    gf = torch.randn(2, 4, 2)
    labels = torch.randn(2, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.ones((2, 4), dtype=torch.float32)
    ds = CombinedDataset(ts, gf, labels, sample_dates=["2020-01-10", "2020-01-13"])
    loader = DataLoader(
        ds,
        batch_size=2,
        collate_fn=partial(
            combined_collate_fn,
            edge_index=edge_index,
            edge_weight=edge_weight,
            append_snapshot_age_days=False,
        ),
    )
    batch = next(iter(loader))
    if len(batch) != 9:
        raise AssertionError(f"combined_collate_fn returned {len(batch)} items, expected 9")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mci_gru_ci_") as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "synthetic_market.csv"
        run_dir = tmp_path / "run"
        _write_synthetic_csv(csv_path)

        cmd = [
            sys.executable,
            "run_experiment.py",
            "features=base",
            f"data.filename={str(csv_path).replace(chr(92), '/')}",
            "data.train_start=2020-01-01",
            "data.train_end=2020-02-14",
            "data.val_start=2020-02-18",
            "data.val_end=2020-02-28",
            "data.test_start=2020-03-03",
            "data.test_end=2020-03-20",
            "model.his_t=3",
            "model.label_t=1",
            "model.gru_hidden_sizes=[8,4]",
            "model.hidden_size_gat1=8",
            "model.output_gat1=4",
            "model.gat_heads=1",
            "model.hidden_size_gat2=8",
            "model.num_hidden_states=4",
            "model.cross_attn_heads=1",
            "model.use_self_attention=false",
            "training.num_epochs=1",
            "training.num_models=1",
            "training.batch_size=2",
            "training.warmup_steps=0",
            "tracking.enabled=false",
            f"hydra.run.dir={str(run_dir).replace(chr(92), '/')}",
        ]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=180)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode

        required = [
            "run_metadata.json",
            "feature_reference.json",
            "graph_data.pt",
            "training_summary.json",
            "evaluation_summary.json",
        ]
        for name in required:
            path = run_dir / name
            if not path.exists():
                raise AssertionError(f"Missing smoke artifact: {path}")
        pred_dir = run_dir / "averaged_predictions"
        if not pred_dir.exists() or not list(pred_dir.glob("*.csv")):
            raise AssertionError("Missing averaged prediction CSVs")

        _assert_collate_contract()
    print("CI smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
