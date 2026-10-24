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
from mci_gru.features.base import BASE_FEATURES, RELATIVE_BASE_FEATURES
from mci_gru.features.registry import FeatureEngineer
from mci_gru.graph.utils import edge_feature_dim
from mci_gru.models import ResidualCrossSectionBlock, create_model
from mci_gru.models.graph import GATBlock

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE = REPO_ROOT / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md"

# `data=<group>` on its own line inside the Hydra overrides block.
DATA_SELECTOR = re.compile(r"^data=([A-Za-z0-9_]+)$", re.M)
OVERRIDE_BLOCK = re.compile(r"^## Hydra Overrides\n+```text\n(.*?)^```", re.M | re.S)
SNAPSHOT_MODE = re.compile(r"^data\.auxiliary_snapshot_mode=(\w+)$", re.M)
SLUG_BLOCK = re.compile(r"^Recipe slug:\n+```text\n(.*?)^```", re.M | re.S)

#: The slug the first admitted run (#187) ran under. It stays on record for that run.
FIRST_RUN_SLUG = (
    "static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble"
    "__drop-edge-0p1__latents-data__xsec-residual__gru-32-10"
)
#: One slug part per #319 fix the recipe adopts (#325).
FIX_SLUG_PARTS = ("gat-residual", "inputs-relative", "eval-fp32")


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


def _graph_blocks(model) -> dict[str, GATBlock]:
    return {name: m for name, m in model.named_modules() if isinstance(m, GATBlock)}


def test_the_recipe_builds_residual_graph_layers():
    """Issue 320: legacy graph layers tie linked names; every layer needs an own-feature path."""
    config = _compose(_recipe_overrides())
    config.graph.use_sector_relation = True  # build the sector stream too, so all three are seen
    blocks = _graph_blocks(_build_model(config))
    assert set(blocks) == {"gat_layer", "gat_layer_sector", "final_gat"}
    for name, block in blocks.items():
        for conv in (block.gat1, block.gat2):
            assert conv.residual and conv.res is not None, f"{name} has no own-feature path"


def test_the_recipe_feeds_relative_base_inputs():
    """Issue 323: raw split-adjusted levels encode later splits; the recipe must not feed them."""
    config = _compose(_recipe_overrides())
    columns = FeatureEngineer(config.features).get_feature_columns()
    assert columns[: len(RELATIVE_BASE_FEATURES)] == RELATIVE_BASE_FEATURES
    assert not set(BASE_FEATURES) & set(columns), "a raw level is still an input"


def test_the_recipe_scores_in_full_precision():
    """Issue 321: the first run scored under autocast, so its scores sat on an fp16 grid.

    The base default is already false, so the composed value alone cannot tell a
    pin from an inherited default. The recipe must say it, so a move of the
    default cannot put scoring back under autocast.
    """
    assert "training.eval_amp=false" in _recipe_overrides()
    assert _compose(_recipe_overrides()).training.eval_amp is False


def test_the_base_config_alone_still_builds_the_legacy_forms():
    """Control: the two tests above must come from the recipe's own pins.

    configs/config.yaml keeps the legacy forms so older checkpoint directories
    rebuild. If it ever moved, the recipe tests would pass without the pins, and
    this control says so rather than letting them pass for the wrong reason.
    """
    without_model_pins = [
        line
        for line in _recipe_overrides()
        if not line.startswith(("model.", "features.base_inputs="))
    ] + ["model.label_t=5"]
    config = _compose(without_model_pins)
    model = _build_model(config)
    assert model.latent_learner.market_latent_mode == "static"
    assert not isinstance(model.self_attention, ResidualCrossSectionBlock)
    assert model.temporal_encoder.fast_gru.grus is None
    assert model.temporal_encoder.fast_gru.gru.hidden_size == 10
    for name, block in _graph_blocks(model).items():
        assert not block.gat1.residual and not block.gat2.residual, name
    columns = FeatureEngineer(config.features).get_feature_columns()
    assert columns[: len(BASE_FEATURES)] == BASE_FEATURES


def _recipe_slug() -> str:
    block = SLUG_BLOCK.search(RECIPE.read_text(encoding="utf-8"))
    assert block, "the recipe has no slug block"
    lines = [line.strip() for line in block.group(1).splitlines() if line.strip()]
    assert len(lines) == 1, lines
    return lines[0]


def test_the_slug_names_each_fix_the_recipe_adopts():
    """PRD 319 D6: the fixed model gets its own id, so its results never mix with the first run's."""
    slug = _recipe_slug()
    assert slug == FIRST_RUN_SLUG + "".join(f"__{part}" for part in FIX_SLUG_PARTS)
    assert slug != FIRST_RUN_SLUG


def test_every_document_naming_the_current_slug_names_this_one():
    """The slug is quoted outside the recipe; a stale copy would label new runs with the old id."""
    slug = _recipe_slug()
    for relative in ("AGENTS.md", "docs/CONFIGURATION_GUIDE.md", "docs/NOTEBOOK_BEST_PRACTICES.md"):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert slug in text, f"{relative} does not name the recipe slug"


def test_recipe_records_the_model_fix_and_keeps_the_first_run_apart():
    """PRD 319 D7: the first run stays on record under its own slug and is not this baseline."""
    text = RECIPE.read_text(encoding="utf-8")
    assert "The model and its inputs changed on 2026-10-24" in text
    assert FIRST_RUN_SLUG in text, "the first run's slug is no longer recorded"
    assert re.search(r"It is not the\s+(> )?baseline for this recipe", text)
    match = re.search(r"^Last updated:\s*(\d{4}-\d{2}-\d{2})$", text, re.M)
    assert match and match.group(1) >= "2026-10-24"


def test_the_override_block_parser_reads_the_block():
    """Control: an empty parse would make every composition test vacuous."""
    overrides = _recipe_overrides()
    assert "data=gics_top10_110_2016_eodhd" in overrides
    assert "model.market_latent_mode=data_dependent" in overrides
    assert all("```" not in line for line in overrides)


def test_recipe_captures_its_provider_inputs():
    """The owner chose capture for the first run (2026-10-04); source leaves inputs unverifiable."""
    match = SNAPSHOT_MODE.search(RECIPE.read_text(encoding="utf-8"))
    assert match is not None, "the recipe does not select an auxiliary snapshot mode"
    assert match.group(1) == "capture"
    # Control: the detector reads another mode as that mode, and absence as absence.
    assert SNAPSHOT_MODE.search("data.auxiliary_snapshot_mode=source\n").group(1) == "source"
    assert SNAPSHOT_MODE.search("seed=1729\n") is None
    # Control: the base default is source, so the recipe line is what selects capture.
    assert (
        OmegaConf.load(REPO_ROOT / "configs/config.yaml").data.auxiliary_snapshot_mode == "source"
    )


def test_the_recipe_trains_on_the_eodhd_price_panel():
    """Issue 283: the first run reads the EODHD package, not the LSEG one (owner, 2026-10-08)."""
    eodhd = OmegaConf.load(REPO_ROOT / "configs/data/gics_top10_110_2016_eodhd.yaml")
    data = _compose(_recipe_overrides()).data
    assert data.filename == eodhd.filename
    assert "_eodhd_" in data.filename
    assert data.input_package_manifest == eodhd.input_package_manifest
    assert data.input_package_manifest_sha256 == eodhd.input_package_manifest_sha256
    # Same universe as the LSEG config: only the price panel and its manifest move.
    lseg = OmegaConf.load(REPO_ROOT / "configs/data/gics_top10_110_2016.yaml")
    assert data.pit_universe_csv == lseg.pit_universe_csv
    assert (data.train_start, data.test_end) == (lseg.train_start, lseg.test_end)


def test_without_its_data_line_the_recipe_would_read_the_lseg_panel():
    """Control: the test above must come from the recipe's selector, not the base default."""
    without_data = [line for line in _recipe_overrides() if not line.startswith("data=")]
    data = _compose(without_data).data
    assert "_lseg_" in data.filename
    assert "_eodhd_" not in data.filename


def test_recipe_records_the_price_panel_change():
    """The 2026-10-08 EODHD switch is dated, so LSEG- and EODHD-panel evidence stay apart."""
    text = RECIPE.read_text(encoding="utf-8")
    assert "The price panel changed on 2026-10-08" in text
    match = re.search(r"^Last updated:\s*(\d{4}-\d{2}-\d{2})$", text, re.M)
    assert match and match.group(1) >= "2026-10-08"
