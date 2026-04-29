from pathlib import Path

from omegaconf import OmegaConf

from mci_gru.config import FeatureConfig


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
