import pytest

from mci_gru.config import TrainingConfig, create_config_from_dict


def test_training_config_accepts_lambdarank_ic_loss_defaults() -> None:
    cfg = TrainingConfig(loss_type="lambdarank_ic", selection_metric="val_rank_ic")

    assert cfg.loss_type == "lambdarank_ic"
    assert cfg.selection_metric == "val_rank_ic"
    assert cfg.lambdarank_ic_max_pairs_per_day == 4096
    assert cfg.lambdarank_ic_temperature == 1.0


def test_lambdarank_ic_is_disabled_by_default() -> None:
    cfg = TrainingConfig()

    assert cfg.loss_type != "lambdarank_ic"


def test_create_config_from_dict_accepts_lambdarank_ic_overrides() -> None:
    cfg = create_config_from_dict(
        {
            "training": {
                "loss_type": "lambdarank_ic",
                "selection_metric": "val_rank_ic",
                "lambdarank_ic_max_pairs_per_day": 128,
                "lambdarank_ic_temperature": 0.75,
            }
        }
    )

    assert cfg.training.loss_type == "lambdarank_ic"
    assert cfg.training.selection_metric == "val_rank_ic"
    assert cfg.training.lambdarank_ic_max_pairs_per_day == 128
    assert cfg.training.lambdarank_ic_temperature == 0.75


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lambdarank_ic_max_pairs_per_day": 0}, "lambdarank_ic_max_pairs_per_day"),
        ({"lambdarank_ic_temperature": 0.0}, "lambdarank_ic_temperature"),
    ],
)
def test_training_config_rejects_invalid_lambdarank_ic_knobs(
    kwargs: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        TrainingConfig(loss_type="lambdarank_ic", **kwargs)
