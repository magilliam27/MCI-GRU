import numpy as np

from mci_gru.evaluation.drift import compute_feature_drift, summarize_drift


def test_feature_drift_is_near_zero_for_matching_distribution():
    reference = {
        "features": {
            "x": {
                "bins": [-2.0, -1.0, 0.0, 1.0, 2.0],
                "counts": [10, 10, 10, 10],
            }
        }
    }
    observed = np.array([[-1.5], [-0.5], [0.5], [1.5]] * 10)

    rows = compute_feature_drift(observed, ["x"], reference)

    assert rows[0]["feature"] == "x"
    assert rows[0]["status"] == "OK"
    assert rows[0]["psi"] < 0.01
    assert rows[0]["ks"] < 0.01


def test_feature_drift_flags_shifted_distribution():
    reference = {
        "features": {
            "x": {
                "bins": [-2.0, -1.0, 0.0, 1.0, 2.0],
                "counts": [10, 10, 10, 10],
            }
        }
    }
    observed = np.array([[1.5], [1.6], [1.7], [1.8]] * 10)

    rows = compute_feature_drift(observed, ["x"], reference)
    summary = summarize_drift(rows)

    assert rows[0]["status"] == "ALERT"
    assert summary["overall_status"] == "ALERT"
    assert summary["alert_features"] == 1


def test_missing_feature_reference_returns_not_available():
    rows = compute_feature_drift(np.array([[1.0]]), ["missing"], {"features": {}})
    summary = summarize_drift(rows)

    assert rows[0]["status"] == "NOT_AVAILABLE"
    assert summary["overall_status"] == "NOT_AVAILABLE"
