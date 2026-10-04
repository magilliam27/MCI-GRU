"""Contract tests for docs/DEFAULT_EXPERIMENT_RECIPE.md.

The recipe named no data config, so it inherited whatever `configs/config.yaml`
composed. When that base default moved from `data: sp500` to
`data: gics_top10_110_2016`, the recipe's effective universe moved with it and
the document did not change. A recipe whose data moves when a default moves is
not frozen. See issue 152.

Every assertion here is paired with a case in which it must fail.
"""

import re
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import create_config_from_dict
from mci_gru.graph.utils import edge_feature_dim
from mci_gru.models import ResidualCrossSectionBlock, create_model

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE = REPO_ROOT / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md"

# `data=<group>` on its own line inside the Hydra overrides block.
DATA_SELECTOR = re.compile(r"^data=([A-Za-z0-9_]+)$", re.M)
OVERRIDE_BLOCK = re.compile(r"^## Hydra Overrides\n+```text\n(.*?)^```", re.M | re.S)


def _pinned_data_group(text: str) -> str | None:
    match = DATA_SELECTOR.search(text)
    return match.group(1) if match else None


def test_recipe_pins_a_data_config_explicitly():
    """Without this, the recipe silently inherits configs/config.yaml."""
    group = _pinned_data_group(RECIPE.read_text(encoding="utf-8"))
    assert group is not None, "the recipe names no data config, so its universe floats"


def test_the_pinned_data_config_exists():
    group = _pinned_data_group(RECIPE.read_text(encoding="utf-8"))
    assert (REPO_ROOT / f"configs/data/{group}.yaml").exists()


def test_the_selector_detector_actually_detects():
    """Control: the two tests above pass vacuously if the regex never matches."""
    assert _pinned_data_group("data=gics_top10_110_2016") == "gics_top10_110_2016"
    assert _pinned_data_group("data=sp500") == "sp500"
    # A recipe with no selector must read as unpinned, not as some default.
    assert _pinned_data_group("seed=1729\ntraining.num_models=20\n") is None
    # `data.` key overrides are not group selectors and must not be mistaken for one.
    assert _pinned_data_group("data.source=csv\n") is None


def test_the_pinned_universe_is_a_csv_source_not_lseg():
    """The recipe must not silently depend on the live-LSEG path."""
    group = _pinned_data_group(RECIPE.read_text(encoding="utf-8"))
    data_cfg = OmegaConf.load(REPO_ROOT / f"configs/data/{group}.yaml")
    assert data_cfg.source == "csv", f"recipe pins source={data_cfg.source}"

    # Control: the config it used to inherit really is the lseg one, so the
    # assertion above discriminates rather than holding for any config.
    assert OmegaConf.load(REPO_ROOT / "configs/data/sp500.yaml").source == "lseg"


def test_recipe_records_that_the_universe_changed():
    """The change must be stated, or pre- and post-change evidence gets compared."""
    text = RECIPE.read_text(encoding="utf-8")
    assert "2026-08-08" in text, "the universe change date is not recorded"
    assert "not directly comparable" in text, "the evidence-comparability warning is missing"


def test_recipe_last_updated_is_not_stale_relative_to_the_change():
    text = RECIPE.read_text(encoding="utf-8")
    match = re.search(r"^Last updated:\s*(\d{4}-\d{2}-\d{2})$", text, re.M)
    assert match, "the recipe carries no Last updated line"
    assert match.group(1) >= "2026-08-08", (
        f"Last updated is {match.group(1)}, older than the universe change it now describes"
    )


def _recipe_overrides() -> list[str]:
    block = OVERRIDE_BLOCK.search(RECIPE.read_text(encoding="utf-8"))
    assert block, "the recipe has no Hydra override block"
    return [line.strip() for line in block.group(1).splitlines() if line.strip()]


def _compose(overrides: list[str]):
    """Build the typed config the way run_experiment.py does."""
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))


def _build_model(config):
    """Mirror run_experiment.py's model_cfg_dict, so the test sees what a run builds."""
    model_cfg = {
        **config.model.to_dict(),
        "edge_feature_dim": edge_feature_dim(config.graph),
        "drop_edge_p": config.graph.drop_edge_p,
        "isolate_edge_dropout_rng": config.graph.isolate_edge_dropout_rng,
        "use_sector_relation": config.graph.use_sector_relation,
    }
    return create_model(8, model_cfg)


def test_the_recipe_builds_data_dependent_market_latents():
    """Issue 198: static latents cannot see the date, so the first run must not use them."""
    model = _build_model(_compose(_recipe_overrides()))
    assert model.latent_learner.market_latent_mode == "data_dependent"
    assert hasattr(model.latent_learner, "gather1"), "no per-date gather was built"


def test_the_recipe_builds_the_residual_cross_stock_block():
    """Issue 197: the legacy block replaces z and discards most cross-sectional variation."""
    model = _build_model(_compose(_recipe_overrides()))
    assert isinstance(model.self_attention, ResidualCrossSectionBlock)


def test_the_recipe_builds_gru_attn_layers_at_32_then_10():
    """Issue 131: [32, 10] must mean a 32-wide layer then a 10-wide one."""
    model = _build_model(_compose(_recipe_overrides()))
    for branch in (model.temporal_encoder.fast_gru, model.temporal_encoder.slow_gru):
        assert [layer.hidden_size for layer in branch.grus] == [32, 10]


def test_the_base_config_alone_still_builds_the_legacy_forms():
    """Control: the two tests above must come from the recipe's own pins.

    configs/config.yaml keeps the legacy forms so older checkpoint directories
    rebuild. If it ever moved, the recipe tests would pass without the pins, and
    this control says so rather than letting them pass for the wrong reason.
    """
    without_model_pins = [line for line in _recipe_overrides() if not line.startswith("model.")] + [
        "model.label_t=5"
    ]
    model = _build_model(_compose(without_model_pins))
    assert model.latent_learner.market_latent_mode == "static"
    assert not isinstance(model.self_attention, ResidualCrossSectionBlock)
    assert model.temporal_encoder.fast_gru.grus is None
    assert model.temporal_encoder.fast_gru.gru.hidden_size == 10


def test_the_override_block_parser_reads_the_block():
    """Control: an empty parse would make every composition test vacuous."""
    overrides = _recipe_overrides()
    assert "data=gics_top10_110_2016" in overrides
    assert "model.market_latent_mode=data_dependent" in overrides
    assert all("```" not in line for line in overrides)
