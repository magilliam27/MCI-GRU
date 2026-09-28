"""Admissible range for `model.label_t` (issue 107).

`compute_labels` builds the label for signal date ``t`` as
``close[t + label_t] / close[t + 1] - 1`` over per-stock session shifts. Which
values that formula can carry decides the boundary:

* ``label_t = 1`` divides the entry close by itself, so every label is exactly
  zero. Training, selection and evaluation then run against a flat target panel,
  which is how the CI smoke once passed on it.
* ``label_t <= 0`` puts the exit close at or before the entry close, so the
  label runs backwards in time. The session embargo treats these values as
  having no horizon and returns early, while the label still reads
  ``close[t + 1]``.
* ``label_t >= 2`` is a genuine forward return from ``close[t + 1]``.

`ModelConfig` is the only gate on this value, so the tests construct it the two
ways a run does: directly, and through `create_config_from_dict` over the real
Hydra tree, as `run_experiment.py` does.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import ExperimentConfig, ModelConfig, create_config_from_dict
from mci_gru.data.preprocessing import compute_labels

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
LABEL_FORMULA = r"close\[t \+ label_t\] / close\[t \+ 1\] - 1"


def _config_from_hydra(overrides: list[str]) -> ExperimentConfig:
    """Build the config the way `run_experiment.py` does for a CLI override."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))


@pytest.mark.parametrize("label_t", [1, 0, -1])
def test_label_t_below_two_is_rejected_naming_the_label_formula(label_t: int) -> None:
    """1 is the all-zero panel; 0 and -1 are the backward labels the embargo skips."""
    with pytest.raises(ValueError, match=LABEL_FORMULA):
        ModelConfig(label_t=label_t)


@pytest.mark.parametrize("label_t", [2, 3, 5, 21])
def test_label_t_of_two_and_above_still_constructs(label_t: int) -> None:
    """Control. 2 is the smallest genuine forward label; 5 is the shipped value."""
    assert ModelConfig(label_t=label_t).label_t == label_t


def test_a_cli_override_of_label_t_one_is_rejected() -> None:
    """`python run_experiment.py model.label_t=1` must stop at config construction."""
    with pytest.raises(ValueError, match=LABEL_FORMULA):
        _config_from_hydra(["model.label_t=1"])


def test_the_shipped_hydra_config_and_the_smallest_override_still_build() -> None:
    """Control for the CLI path: the rejection is about the value, not the route."""
    assert _config_from_hydra([]).model.label_t == 5
    assert _config_from_hydra(["model.label_t=2"]).model.label_t == 2


def test_the_boundary_sits_where_the_label_panel_degenerates() -> None:
    """Ties the guard to the formula it protects, measured through `compute_labels`.

    If the label computation changes so that ``label_t = 1`` carries a real
    return, this fails and the guard above needs revisiting with it.
    """
    rng = np.random.default_rng(107)
    dates = [f"2024-01-{day:02d}" for day in range(1, 21)]
    kdcodes = ["AAA", "BBB", "CCC", "DDD"]
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (len(kdcodes), len(dates))), axis=1))
    panel = pd.DataFrame(
        {
            "kdcode": np.repeat(kdcodes, len(dates)),
            "dt": np.tile(dates, len(kdcodes)),
            "close": closes.ravel(),
        }
    )
    sample_dates = dates[:10]

    rejected = compute_labels(panel, kdcodes, sample_dates, label_t=1, fill_missing=False)
    accepted = compute_labels(panel, kdcodes, sample_dates, label_t=2, fill_missing=False)

    assert np.isfinite(rejected).all() and np.isfinite(accepted).all()
    assert np.all(rejected == 0.0)
    assert np.all(accepted.std(axis=1) > 0.0)
