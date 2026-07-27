"""Shared pytest fixtures for MCI-GRU tests.

External capabilities are declared with the explicit `requires_data`,
`requires_fred`, and `requires_lseg` markers registered in `pyproject.toml` and
selected with `-m`. Test and file names never imply vendor access, so nothing
here infers a capability from a node id.
"""

import pytest


@pytest.fixture
def sample_config():
    """Minimal ExperimentConfig for unit tests that don't need real data."""
    from mci_gru.config import ExperimentConfig, ModelConfig, TrainingConfig

    return ExperimentConfig(
        training=TrainingConfig(num_epochs=2, num_models=1, batch_size=4),
        model=ModelConfig(his_t=5, label_t=3),
        experiment_name="test_run",
    )
