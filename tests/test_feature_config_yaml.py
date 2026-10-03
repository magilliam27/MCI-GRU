"""Every feature group must accept the regime overrides that launchers pass, without ``+``.

In April 2026 every regime-enabled arm of the full-feature factorial ablation failed
before training: ``FeatureConfig`` had ``regime_include_subsequent_returns``, but the
``configs/features`` groups did not declare it, and Hydra refuses to override an
undeclared key unless the override carries a ``+`` prefix (issue 5). The composition
tests below go through the real ``configs/`` tree the way ``run_experiment.py`` does
(``compose`` then ``create_config_from_dict``), so a regime key dropped from any feature
group fails here instead of on a GPU runtime.
"""

import dataclasses
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from mci_gru.config import ExperimentConfig, FeatureConfig, create_config_from_dict

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
FEATURE_GROUPS = sorted(path.stem for path in (CONFIG_DIR / "features").glob("*.yaml"))
REGIME_KEYS = sorted(
    field.name
    for field in dataclasses.fields(FeatureConfig)
    if field.name == "include_global_regime" or field.name.startswith("regime_")
)
FEATURE_KEYS = sorted(field.name for field in dataclasses.fields(FeatureConfig))
REGIME_INPUTS_CSV = "data/raw/market/regime_inputs.csv"


def _april_regime_arm_overrides(include_subsequent_returns: bool) -> list[str]:
    """One regime-enabled arm of the April factorial launcher.

    The shape of ``regime_overrides(True, ...)`` in
    ``notebooks/full_feature_factorial_ablation_colab.ipynb`` at tag
    ``archive/pre-cleanup-2026-09``, with ``regime_inputs_csv`` set so that every
    regime key an in-repo launcher passes today is exercised as well.
    """
    return [
        "features.include_global_regime=true",
        "features.regime_strict=true",
        f"features.regime_include_subsequent_returns={str(include_subsequent_returns).lower()}",
        "features.regime_subsequent_return_horizons=[1,3]",
        "features.regime_change_months=12",
        "features.regime_norm_months=120",
        "features.regime_exclusion_months=1",
        "features.regime_similarity_quantile=0.2",
        "features.regime_min_history_months=24",
        "features.regime_enforce_lag_days=0",
        f"features.regime_inputs_csv={REGIME_INPUTS_CSV}",
    ]


def _compose(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def _compose_experiment_config(overrides: list[str]) -> ExperimentConfig:
    cfg = _compose(overrides)
    return create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))


def test_feature_yaml_declares_regime_subsequent_return_keys():
    """Regime ablation overrides must target keys declared in feature YAML groups."""
    config_dir = Path("configs/features")
    required_keys = {
        "regime_include_subsequent_returns",
        "regime_subsequent_return_horizons",
    }

    for path in config_dir.glob("*.yaml"):
        cfg = OmegaConf.load(path)
        missing = required_keys - set(cfg.keys())
        assert not missing, f"{path} missing keys: {sorted(missing)}"

        feature_cfg = FeatureConfig(**OmegaConf.to_container(cfg, resolve=True))
        assert feature_cfg.regime_subsequent_return_horizons == [1, 3]


def test_regime_override_guard_discovers_the_default_group_and_ticket_keys() -> None:
    """The regime override guards are not vacuous: feature groups and regime keys are found."""
    assert "with_momentum" in FEATURE_GROUPS
    assert {
        "include_global_regime",
        "regime_include_subsequent_returns",
        "regime_subsequent_return_horizons",
    } <= set(REGIME_KEYS)


@pytest.mark.parametrize(
    ("arm", "include_subsequent_returns"),
    [("regime_current_only", False), ("regime_with_forward_context", True)],
)
@pytest.mark.parametrize("group", FEATURE_GROUPS)
def test_april_regime_ablation_arms_compose_without_append_prefix(
    group: str, arm: str, include_subsequent_returns: bool
) -> None:
    """Both April regime arms compose, without ``+``, and reach the typed config."""
    overrides = [f"features={group}", *_april_regime_arm_overrides(include_subsequent_returns)]
    assert not [override for override in overrides if override.startswith("+")]

    features = _compose_experiment_config(overrides).features

    where = f"features={group} {arm}"
    assert features.include_global_regime is True, where
    assert features.regime_strict is True, where
    assert features.regime_include_subsequent_returns is include_subsequent_returns, where
    assert features.regime_subsequent_return_horizons == [1, 3], where
    assert features.regime_inputs_csv == REGIME_INPUTS_CSV, where


@pytest.mark.parametrize("group", FEATURE_GROUPS)
def test_every_feature_group_declares_every_regime_key(group: str) -> None:
    """Each ``FeatureConfig`` regime key is declared, so its override needs no ``+``."""
    composed = _compose([f"features={group}"])

    missing = sorted(set(REGIME_KEYS) - set(composed.features.keys()))
    assert not missing, f"features={group} does not declare {missing}"

    override = _compose_experiment_config(
        [f"features={group}", "features.regime_subsequent_return_horizons=[2,6]"]
    )
    assert override.features.regime_subsequent_return_horizons == [2, 6], group


@pytest.mark.parametrize("group", FEATURE_GROUPS)
def test_every_feature_group_declares_every_feature_config_key(group: str) -> None:
    """Every ``FeatureConfig`` field is declared in every group, so no override needs ``+``.

    ``features=base`` and ``features=full`` once lacked the seven momentum-blend keys that
    ``with_momentum`` declares, so ``features.momentum_blend_mode=dynamic`` failed to
    compose with them (issue 247).
    """
    composed = _compose([f"features={group}"])

    missing = sorted(set(FEATURE_KEYS) - set(composed.features.keys()))
    assert not missing, f"features={group} does not declare {missing}"


@pytest.mark.parametrize("group", FEATURE_GROUPS)
def test_momentum_blend_overrides_compose_with_every_feature_group(group: str) -> None:
    """The momentum-blend overrides reach the typed config from every group, without ``+``."""
    features = _compose_experiment_config(
        [
            f"features={group}",
            "features.momentum_blend_mode=dynamic",
            "features.momentum_dynamic_min_history=126",
        ]
    ).features

    assert features.momentum_blend_mode == "dynamic", group
    assert features.momentum_dynamic_min_history == 126, group


@pytest.mark.parametrize("group", FEATURE_GROUPS)
def test_feature_group_declarations_match_the_dataclass_defaults_for_momentum_blend(
    group: str,
) -> None:
    """Declaring the momentum-blend keys leaves each group's default behaviour unchanged."""
    features = _compose_experiment_config([f"features={group}"]).features
    defaults = FeatureConfig()

    for key in FEATURE_KEYS:
        if key.startswith("momentum_blend") or key.startswith("momentum_dynamic"):
            assert getattr(features, key) == getattr(defaults, key), f"{group}: {key}"
