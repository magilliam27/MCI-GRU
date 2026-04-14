"""Shared pytest fixtures and configuration for MCI-GRU tests."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests that need external resources."""
    for item in items:
        if (
            "fred" in item.nodeid.lower() or "credit" in item.nodeid.lower()
        ) and not os.environ.get("FRED_API_KEY"):
            item.add_marker(pytest.mark.skip(reason="FRED_API_KEY not set"))

        if "lseg" in item.nodeid.lower() or "refinitiv" in item.nodeid.lower():
            item.add_marker(pytest.mark.skip(reason="LSEG/Refinitiv requires desktop app"))


@pytest.fixture
def sample_config():
    """Minimal ExperimentConfig for unit tests that don't need real data."""
    from mci_gru.config import ExperimentConfig, ModelConfig, TrainingConfig

    return ExperimentConfig(
        training=TrainingConfig(num_epochs=2, num_models=1, batch_size=4),
        model=ModelConfig(his_t=5, label_t=3),
        experiment_name="test_run",
    )
