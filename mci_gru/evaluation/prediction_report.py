"""TSFM-style saved-prediction evaluation reports."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mci_gru.evaluation.metrics import evaluate_predictions

KEY_COLUMNS = ["dt", "kdcode"]
DEFAULT_MODEL_NAME = "mci_gru"


def compute_oos_r2_zero(predictions: np.ndarray, true_returns: np.ndarray) -> float:
    """Out-of-sample R2 against a zero-return forecast benchmark."""
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    ret = np.asarray(true_returns, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(ret)
    if not valid.any():
        return float("nan")

    errors = ret[valid] - pred[valid]
    model_sse = float(np.dot(errors, errors))
    benchmark_sse = float(np.dot(ret[valid], ret[valid]))
    if benchmark_sse <= 0.0:
        return 1.0 if model_sse <= 0.0 else float("nan")
    return float(1.0 - model_sse / benchmark_sse)


def compute_sign_metrics(
    predictions: np.ndarray,
    true_returns: np.ndarray,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Directional accuracy and macro-F1 over negative, zero, and positive signs."""
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    ret = np.asarray(true_returns, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(ret)
    if not valid.any():
        return {
            "direction_accuracy": float("nan"),
            "macro_f1": float("nan"),
        }

    pred_sign = _sign_classes(pred[valid], threshold=threshold)
    true_sign = _sign_classes(ret[valid], threshold=threshold)
    accuracy = float(np.mean(pred_sign == true_sign))

    f1_values = []
    for label in (-1, 0, 1):
        tp = int(np.sum((pred_sign == label) & (true_sign == label)))
        fp = int(np.sum((pred_sign == label) & (true_sign != label)))
        fn = int(np.sum((pred_sign != label) & (true_sign == label)))
        denom = 2 * tp + fp + fn
        f1_values.append(float(2 * tp / denom) if denom > 0 else 0.0)

    return {
        "direction_accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
    }


def align_prediction_comparison(
    primary_predictions: pd.DataFrame,
    realized_returns: pd.DataFrame,
    baseline_predictions: dict[str, pd.DataFrame] | None = None,
    primary_name: str = DEFAULT_MODEL_NAME,
    prediction_col: str = "score",
    return_col: str = "realized_return",
) -> pd.DataFrame:
    """Inner-align predictions, realized returns, and optional baselines by date/ticker."""
    aligned = _normalise_value_frame(realized_returns, return_col, "realized_return")
    primary_col = _score_column(primary_name)
    aligned = aligned.merge(
        _normalise_value_frame(primary_predictions, prediction_col, primary_col),
        on=KEY_COLUMNS,
        how="inner",
    )

    for baseline_name, baseline_df in (baseline_predictions or {}).items():
        baseline_col = _score_column(baseline_name)
        aligned = aligned.merge(
            _normalise_value_frame(baseline_df, prediction_col, baseline_col),
            on=KEY_COLUMNS,
            how="inner",
        )

    value_cols = [col for col in aligned.columns if col not in KEY_COLUMNS]
    finite_mask = np.ones(len(aligned), dtype=bool)
    for col in value_cols:
        finite_mask &= np.isfinite(pd.to_numeric(aligned[col], errors="coerce").to_numpy())

    aligned = aligned.loc[finite_mask].copy()
    aligned = aligned.sort_values(KEY_COLUMNS).reset_index(drop=True)
    return aligned


def compute_tsfm_prediction_report(
    aligned_predictions: pd.DataFrame,
    model_score_columns: list[str] | None = None,
    top_k_values: list[int] | None = None,
    label_t: int = 1,
) -> dict[str, Any]:
    """Compute a repeatable TSFM-style report from an already aligned comparison frame."""
    if "realized_return" not in aligned_predictions.columns:
        raise ValueError("aligned_predictions must include 'realized_return'")

    frame = aligned_predictions.copy()
    frame["dt"] = pd.to_datetime(frame["dt"]).dt.strftime("%Y-%m-%d")
    model_score_columns = model_score_columns or [
        col for col in frame.columns if col.endswith("_score") and col != "zero_score"
    ]
    if not model_score_columns:
        raise ValueError("No model score columns were provided or found")

    report: dict[str, Any] = {
        "comparison": _comparison_summary(frame),
        "zero_benchmark": {
            "name": "zero_return",
            "description": "Forecast return is 0.0 for every aligned observation.",
        },
        "models": {},
        "yearly_decay": {},
    }

    for score_col in model_score_columns:
        model_name = _model_name_from_score_col(score_col)
        report["models"][model_name] = _model_metrics(
            frame,
            score_col=score_col,
            top_k_values=top_k_values,
            label_t=label_t,
        )
        report["yearly_decay"][model_name] = _yearly_metrics(
            frame,
            score_col=score_col,
            top_k_values=top_k_values,
            label_t=label_t,
        )

    return report


def load_prediction_files(predictions_dir: str | Path) -> pd.DataFrame:
    """Load saved MCI-GRU ``averaged_predictions/*.csv`` files."""
    path = Path(predictions_dir)
    files = sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction CSV files found in {path}")
    frames = [pd.read_csv(file) for file in files]
    predictions = pd.concat(frames, ignore_index=True)
    missing = {"dt", "kdcode", "score"} - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction files missing columns: {sorted(missing)}")
    return predictions[["dt", "kdcode", "score"]].copy()


def load_prediction_input(path: str | Path) -> pd.DataFrame:
    """Load either a prediction directory of daily CSVs or one prediction CSV."""
    input_path = Path(path)
    if input_path.is_dir():
        return load_prediction_files(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Prediction input not found: {input_path}")
    predictions = pd.read_csv(input_path)
    missing = {"dt", "kdcode", "score"} - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {sorted(missing)}")
    return predictions[["dt", "kdcode", "score"]].copy()


def realized_returns_from_market_data(
    market_data: pd.DataFrame,
    label_t: int,
    return_col: str = "realized_return",
) -> pd.DataFrame:
    """Compute forward close-to-close labels aligned to prediction dates."""
    required = {"dt", "kdcode", "close"}
    missing = required - set(market_data.columns)
    if missing:
        raise ValueError(f"market_data missing columns: {sorted(missing)}")
    if label_t <= 0:
        raise ValueError("label_t must be positive")

    df = market_data.copy()
    df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["kdcode", "dt"]).reset_index(drop=True)
    close_t1 = df.groupby("kdcode")["close"].shift(-1)
    close_tn = df.groupby("kdcode")["close"].shift(-label_t)
    df[return_col] = close_tn / close_t1 - 1.0
    return df[["dt", "kdcode", return_col]]


def write_tsfm_prediction_report(
    predictions_dir: str | Path,
    market_data_path: str | Path,
    output_dir: str | Path,
    label_t: int,
    baseline_prediction_paths: dict[str, str | Path] | None = None,
    top_k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Generate JSON, Markdown, and aligned-row outputs from saved predictions."""
    primary = load_prediction_files(predictions_dir)
    market_data = pd.read_csv(market_data_path)
    realized = realized_returns_from_market_data(market_data, label_t=label_t)
    baselines = {
        name: load_prediction_input(path)
        for name, path in (baseline_prediction_paths or {}).items()
    }
    aligned = align_prediction_comparison(
        primary_predictions=primary,
        realized_returns=realized,
        baseline_predictions=baselines,
    )
    model_score_columns = [
        col for col in aligned.columns if col.endswith("_score") and col != "zero_score"
    ]
    report = compute_tsfm_prediction_report(
        aligned,
        model_score_columns=model_score_columns,
        top_k_values=top_k_values,
        label_t=label_t,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = out_dir / "tsfm_aligned_predictions.csv"
    json_path = out_dir / "tsfm_prediction_report.json"
    markdown_path = out_dir / "tsfm_prediction_report.md"

    aligned.to_csv(aligned_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_for_file(report), handle, indent=2)
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")

    return {
        "report": report,
        "aligned_predictions": aligned,
        "paths": {
            "aligned_csv": aligned_path,
            "json": json_path,
            "markdown": markdown_path,
        },
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    """Render a compact Markdown report for saved prediction comparisons."""
    comparison = report["comparison"]
    lines = [
        "# TSFM-Style Prediction Evaluation",
        "",
        "This report evaluates saved predictions without retraining. Forecast metrics are computed on the same aligned date/ticker rows as the IC and portfolio-oriented fields.",
        "",
        "## Aligned Sample",
        "",
        f"- Observations: {comparison['aligned_observations']}",
        f"- Dates: {comparison['aligned_dates']}",
        f"- Stocks: {comparison['aligned_stocks']}",
        f"- Window: {comparison['first_date']} to {comparison['last_date']}",
        "",
        "## Model Summary",
        "",
        _summary_table(report["models"]),
        "",
        "## Yearly Decay",
        "",
    ]
    for model_name, yearly_rows in report["yearly_decay"].items():
        lines.extend([f"### {model_name}", "", _yearly_table(yearly_rows), ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "- OOS R2 is measured against a zero-return forecast; values above 0 mean lower squared forecast error than predicting zero.",
            "- Direction accuracy and macro-F1 use negative, zero, and positive return signs, so zero-return handling is explicit.",
            "- IC and top-k return fields remain ranking and portfolio diagnostics; read them beside R2 rather than replacing them.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _normalise_value_frame(df: pd.DataFrame, source_col: str, output_col: str) -> pd.DataFrame:
    missing = set(KEY_COLUMNS + [source_col]) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")
    out = df[KEY_COLUMNS + [source_col]].copy()
    out["dt"] = pd.to_datetime(out["dt"]).dt.strftime("%Y-%m-%d")
    out["kdcode"] = out["kdcode"].astype(str)
    if out.duplicated(KEY_COLUMNS).any():
        duplicates = out.loc[out.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
        sample = duplicates.head(5).to_dict("records")
        raise ValueError(f"Duplicate date/ticker rows found: {sample}")
    out[output_col] = pd.to_numeric(out[source_col], errors="coerce")
    return out[KEY_COLUMNS + [output_col]]


def _score_column(name: str) -> str:
    return f"{_safe_name(name)}_score"


def _safe_name(name: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    return value or "model"


def _model_name_from_score_col(score_col: str) -> str:
    return score_col[: -len("_score")] if score_col.endswith("_score") else score_col


def _sign_classes(values: np.ndarray, threshold: float) -> np.ndarray:
    signs = np.zeros(values.shape, dtype=int)
    signs[values > threshold] = 1
    signs[values < -threshold] = -1
    return signs


def _comparison_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "aligned_observations": int(len(frame)),
        "aligned_dates": int(frame["dt"].nunique()),
        "aligned_stocks": int(frame["kdcode"].nunique()),
        "first_date": str(frame["dt"].min()) if len(frame) else None,
        "last_date": str(frame["dt"].max()) if len(frame) else None,
    }


def _model_metrics(
    frame: pd.DataFrame,
    score_col: str,
    top_k_values: list[int] | None,
    label_t: int,
) -> dict[str, Any]:
    pred = frame[score_col].to_numpy(dtype=np.float64)
    ret = frame["realized_return"].to_numpy(dtype=np.float64)
    metrics: dict[str, Any] = {
        "n_observations": int(len(frame)),
        "n_dates": int(frame["dt"].nunique()),
        "n_stocks": int(frame["kdcode"].nunique()),
        "oos_r2_zero": compute_oos_r2_zero(pred, ret),
    }
    metrics.update(compute_sign_metrics(pred, ret))

    prediction_matrix, return_matrix = _to_aligned_matrices(frame, score_col)
    ranking_metrics = evaluate_predictions(
        prediction_matrix,
        return_matrix,
        top_k_values=top_k_values,
        label_t=label_t,
    )
    metrics.update(ranking_metrics)
    return _json_ready(metrics)


def _yearly_metrics(
    frame: pd.DataFrame,
    score_col: str,
    top_k_values: list[int] | None,
    label_t: int,
) -> list[dict[str, Any]]:
    dated = frame.copy()
    dated["year"] = pd.to_datetime(dated["dt"]).dt.year
    rows = []
    for year, year_df in dated.groupby("year", sort=True):
        row = {"year": int(year)}
        row.update(
            _model_metrics(
                year_df.drop(columns=["year"]),
                score_col=score_col,
                top_k_values=top_k_values,
                label_t=label_t,
            )
        )
        rows.append(row)
    return rows


def _to_aligned_matrices(frame: pd.DataFrame, score_col: str) -> tuple[np.ndarray, np.ndarray]:
    predictions = frame.pivot(index="dt", columns="kdcode", values=score_col).sort_index()
    returns = frame.pivot(index="dt", columns="kdcode", values="realized_return").reindex_like(
        predictions
    )
    return predictions.to_numpy(dtype=np.float64), returns.to_numpy(dtype=np.float64)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value
    return value


def _json_for_file(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_for_file(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_for_file(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return _json_for_file(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def _summary_table(models: dict[str, dict[str, Any]]) -> str:
    columns = [
        "model",
        "n_observations",
        "oos_r2_zero",
        "direction_accuracy",
        "macro_f1",
        "avg_ic",
        "avg_spearman_corr",
    ]
    rows = []
    for model_name, metrics in models.items():
        rows.append([model_name] + [_format_cell(metrics.get(col)) for col in columns[1:]])
    return _markdown_table(columns, rows)


def _yearly_table(rows: list[dict[str, Any]]) -> str:
    columns = ["year", "n_observations", "oos_r2_zero", "direction_accuracy", "macro_f1", "avg_ic"]
    values = [[_format_cell(row.get(col)) for col in columns] for row in rows]
    return _markdown_table(columns, values)


def _markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "No rows."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6g}"
    return str(value)
