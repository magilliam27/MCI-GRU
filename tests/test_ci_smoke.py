from __future__ import annotations

from pathlib import Path

from scripts.ci_smoke import _build_smoke_command


def test_ci_smoke_uses_synthetic_csv_loader() -> None:
    cmd = _build_smoke_command(Path("tmp/synthetic_market.csv"), Path("tmp/run"))

    assert "data.source=csv" in cmd
    assert "data.filename=tmp/synthetic_market.csv" in cmd
    assert not any(arg == "data.source=lseg" for arg in cmd)
