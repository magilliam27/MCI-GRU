"""Unit tests for mci_gru.evaluation.experiment_summary (WS-M M2 move)."""

import hashlib
import logging
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mci_gru.config import create_config_from_dict
from mci_gru.evaluation.experiment_summary import (
    compute_evaluation_summary,
    data_file_fingerprint,
    resolved_evaluation_kwargs,
    select_training_objective_value,
)

logger = logging.getLogger(__name__)


def test_data_file_fingerprint_hashes_existing_file(tmp_path):
    payload = b"date,kdcode,close\n2026-01-02,AAPL,100\n"
    data_file = tmp_path / "market.csv"
    data_file.write_bytes(payload)

    result = data_file_fingerprint(str(data_file), logger)

    assert result["data_file_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["data_file_size_bytes"] == len(payload)
    # ISO-8601 UTC timestamp must parse
    assert datetime.fromisoformat(result["data_file_mtime_iso"]).tzinfo is not None


def test_data_file_fingerprint_missing_file_returns_nulls(tmp_path):
    result = data_file_fingerprint(str(tmp_path / "does_not_exist.csv"), logger)
    assert result == {
        "data_file_sha256": None,
        "data_file_size_bytes": None,
        "data_file_mtime_iso": None,
    }


def test_resolved_evaluation_kwargs_derives_defaults_from_label_t():
    config = create_config_from_dict({"model": {"label_t": 5}})

    kwargs = resolved_evaluation_kwargs(config)

    assert kwargs["label_t"] == 5
    assert kwargs["block_size"] == 5  # eval block_size unset -> max(1, label_t)
    assert kwargs["newey_west_lags"] == 4  # unset -> max(0, label_t - 1)
    assert kwargs["top_k_values"] == config.evaluation.top_k_values
    assert kwargs["bootstrap_enabled"] == config.evaluation.bootstrap_enabled
    assert kwargs["bootstrap_resamples"] == config.evaluation.bootstrap_resamples
    assert kwargs["bootstrap_seed"] == config.evaluation.bootstrap_seed
    assert kwargs["ci_level"] == config.evaluation.ci_level


def test_resolved_evaluation_kwargs_label_t_one_floors():
    config = create_config_from_dict({"model": {"label_t": 1}})

    kwargs = resolved_evaluation_kwargs(config)

    assert kwargs["block_size"] == 1
    assert kwargs["newey_west_lags"] == 0


def test_resolved_evaluation_kwargs_explicit_values_win():
    config = create_config_from_dict(
        {
            "model": {"label_t": 5},
            "evaluation": {"block_size": 7, "newey_west_lags": 2},
        }
    )

    kwargs = resolved_evaluation_kwargs(config)

    assert kwargs["block_size"] == 7
    assert kwargs["newey_west_lags"] == 2


def test_compute_evaluation_summary_shape_and_metrics():
    config = create_config_from_dict(
        {
            "model": {"label_t": 5},
            "evaluation": {"top_k_values": [5, 10], "bootstrap_enabled": False},
        }
    )
    rng = np.random.default_rng(1729)
    predictions = rng.normal(size=(6, 30))
    labels = rng.normal(size=(6, 30))

    summary = compute_evaluation_summary(predictions, labels, config)

    assert summary["label_t"] == 5
    assert summary["top_k_values"] == [5, 10]
    assert isinstance(summary["metrics"], dict)
    assert summary["metrics"]  # non-empty metric dict from evaluate_predictions


def test_select_training_objective_key_mapping_last_window():
    final_summary = {
        "mean_best_val_loss": 0.7,
        "mean_best_val_ic": 0.1,
        "mean_best_val_rank_ic": 0.2,
    }

    assert select_training_objective_value("val_loss", [final_summary], None) == 0.7
    assert select_training_objective_value("val_ic", [final_summary], None) == 0.1
    assert select_training_objective_value("val_rank_ic", [final_summary], None) == 0.2
    # Unknown metric falls back to the val_ic key
    assert select_training_objective_value("anything_else", [final_summary], None) == 0.1


def test_select_training_objective_merged_summary_wins():
    final_summary = {"mean_best_val_ic": 0.1}
    merged_summary = {
        "mean_best_val_loss_across_windows": 0.6,
        "mean_best_val_ic_across_windows": 0.3,
        "mean_best_val_rank_ic_across_windows": 0.4,
    }

    assert select_training_objective_value("val_loss", [final_summary], merged_summary) == 0.6
    assert select_training_objective_value("val_ic", [final_summary], merged_summary) == 0.3
    assert select_training_objective_value("val_rank_ic", [final_summary], merged_summary) == 0.4


def test_select_training_objective_no_summaries_returns_none():
    assert select_training_objective_value("val_ic", [], None) is None
