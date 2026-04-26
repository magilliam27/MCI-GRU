"""Feature drift calculations for paper-trade monitoring."""

from __future__ import annotations

import numpy as np

PSI_WARN = 0.10
PSI_ALERT = 0.25
KS_WARN = 0.10
KS_ALERT = 0.20
EPS = 1e-8


def _status(psi: float, ks: float) -> str:
    if psi >= PSI_ALERT or ks >= KS_ALERT:
        return "ALERT"
    if psi >= PSI_WARN or ks >= KS_WARN:
        return "WARN"
    return "OK"


def compute_feature_drift(
    observed_features: np.ndarray,
    feature_cols: list[str],
    reference: dict,
) -> list[dict]:
    """Compute PSI and KS-like CDF distance for each observed feature."""
    obs = np.asarray(observed_features, dtype=np.float64)
    if obs.ndim != 2:
        raise ValueError("observed_features must be 2-D (rows, features)")
    refs = reference.get("features", {}) if reference else {}
    rows: list[dict] = []
    for i, feature in enumerate(feature_cols):
        ref = refs.get(feature)
        if ref is None:
            rows.append(
                {
                    "feature": feature,
                    "psi": float("nan"),
                    "ks": float("nan"),
                    "status": "NOT_AVAILABLE",
                    "observed_count": int(np.isfinite(obs[:, i]).sum()),
                }
            )
            continue

        bins = np.asarray(ref.get("bins", []), dtype=np.float64)
        expected_counts = np.asarray(ref.get("counts", []), dtype=np.float64)
        valid = obs[:, i][np.isfinite(obs[:, i])]
        if bins.size < 2 or expected_counts.size != bins.size - 1 or valid.size == 0:
            rows.append(
                {
                    "feature": feature,
                    "psi": float("nan"),
                    "ks": float("nan"),
                    "status": "NOT_AVAILABLE",
                    "observed_count": int(valid.size),
                }
            )
            continue

        observed_counts, _ = np.histogram(valid, bins=bins)
        expected_pct = expected_counts / max(float(expected_counts.sum()), EPS)
        observed_pct = observed_counts.astype(np.float64) / max(float(observed_counts.sum()), EPS)
        expected_safe = np.clip(expected_pct, EPS, None)
        observed_safe = np.clip(observed_pct, EPS, None)
        psi = float(np.sum((observed_safe - expected_safe) * np.log(observed_safe / expected_safe)))
        ks = float(np.max(np.abs(np.cumsum(observed_pct) - np.cumsum(expected_pct))))
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "ks": ks,
                "status": _status(psi, ks),
                "observed_count": int(valid.size),
            }
        )
    return rows


def summarize_drift(rows: list[dict]) -> dict:
    """Summarize per-feature drift rows into an overall status."""
    if not rows:
        return {
            "overall_status": "NOT_AVAILABLE",
            "warn_features": 0,
            "alert_features": 0,
            "features_evaluated": 0,
        }
    alert = sum(1 for row in rows if row.get("status") == "ALERT")
    warn = sum(1 for row in rows if row.get("status") == "WARN")
    ok = sum(1 for row in rows if row.get("status") == "OK")
    if alert:
        status = "ALERT"
    elif warn:
        status = "WARN"
    elif ok:
        status = "OK"
    else:
        status = "NOT_AVAILABLE"
    return {
        "overall_status": status,
        "warn_features": warn,
        "alert_features": alert,
        "features_evaluated": ok + warn + alert,
    }
