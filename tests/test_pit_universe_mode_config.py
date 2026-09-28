"""``row_filter`` can no longer be selected as ``data.pit_universe_mode`` (#139).

The legacy ``row_filter`` mode dropped panel rows outside PIT membership while
the correlation graph, which reads the unfiltered frame, still linked names
outside the universe. Nothing in a run's output showed the difference. It was
also the ``DataConfig`` default, so any config that omitted the key selected it
silently, and seven shipped data configs declared it outright.

``DataConfig`` now rejects it and defaults to ``masked_panel``. These tests pin
both halves at the two public surfaces a run's configuration passes through:
``DataConfig`` construction, and Hydra composition of every shipped config,
converted by ``create_config_from_dict`` the way ``run_experiment.py`` does it.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from mci_gru.config import DataConfig, ExperimentConfig, create_config_from_dict

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DATA_GROUPS = sorted(path.stem for path in (CONFIG_DIR / "data").glob("*.yaml"))
EXPERIMENT_PRESETS = sorted(path.stem for path in (CONFIG_DIR / "experiment").glob("*.yaml"))

# The data configs that declared ``pit_universe_mode: row_filter`` until #139.
FORMER_ROW_FILTER_GROUPS = {
    "csv_sp500",
    "lseg_sp500",
    "sp500",
    "temporal_2016",
    "temporal_2017",
    "temporal_2018",
    "temporal_2019",
}


def _typed_config(overrides: list[str]) -> ExperimentConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    # lseg_sp500 declares ``api_key``, which is not a DataConfig field and whose
    # ${oc.env:...} interpolation raises when the variable is unset. That defect
    # is tracked separately (#151) and is unrelated to the PIT mode, so the key
    # is dropped unresolved here rather than leaving that config unchecked.
    # Every other config is converted exactly as composed.
    with open_dict(cfg):
        if "api_key" in cfg.data:
            del cfg.data["api_key"]
    return create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))


def test_default_pit_universe_mode_is_masked_panel() -> None:
    assert DataConfig().pit_universe_mode == "masked_panel"
    # A config that omits the key receives the same default.
    config = create_config_from_dict({"data": {"use_pit_universe": False}})
    assert config.data.pit_universe_mode == "masked_panel"


@pytest.mark.parametrize("use_pit_universe", [False, True])
def test_row_filter_is_rejected_at_construction(use_pit_universe: bool) -> None:
    fields = {
        "use_pit_universe": use_pit_universe,
        "pit_universe_csv": "pit.csv",
        "pit_universe_mode": "row_filter",
    }
    with pytest.raises(ValueError, match="got 'row_filter'"):
        DataConfig(**fields)
    with pytest.raises(ValueError, match="got 'row_filter'"):
        create_config_from_dict({"data": fields})


def test_command_line_override_to_row_filter_is_rejected() -> None:
    # The base data config turns PIT on, so this is the combination that
    # filtered rows while leaving the graph unrestricted.
    with pytest.raises(ValueError, match="got 'row_filter'"):
        _typed_config(["data.pit_universe_mode=row_filter"])


def test_discovery_covers_every_config_that_declared_row_filter() -> None:
    # Guards the parametrisations below against silently shrinking to nothing.
    assert set(DATA_GROUPS) >= FORMER_ROW_FILTER_GROUPS
    assert "pit_temporal_2022" in EXPERIMENT_PRESETS


@pytest.mark.parametrize("group", DATA_GROUPS)
def test_shipped_data_config_composes_to_masked_panel(group: str) -> None:
    config = _typed_config([f"data={group}"])
    assert config.data.pit_universe_mode == "masked_panel"


@pytest.mark.parametrize("preset", EXPERIMENT_PRESETS)
def test_shipped_experiment_preset_composes_to_masked_panel(preset: str) -> None:
    config = _typed_config([f"+experiment={preset}"])
    assert config.data.pit_universe_mode == "masked_panel"
