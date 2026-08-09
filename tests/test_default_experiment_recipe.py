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

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE = REPO_ROOT / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md"

# `data=<group>` on its own line inside the Hydra overrides block.
DATA_SELECTOR = re.compile(r"^data=([A-Za-z0-9_]+)$", re.M)


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
