from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert NumPy and non-finite values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_artifact(path: str | Path, payload: Any, *, force: bool = False) -> Path:
    """Write a strict JSON artifact, refusing to overwrite existing evidence by default."""
    artifact_path = Path(path)
    if artifact_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact_path}")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(to_jsonable(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return artifact_path
