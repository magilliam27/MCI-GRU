import pytest

from mci_gru.config import TrainingConfig, create_config_from_dict


def test_training_config_accepts_portfolio_ic_loss_defaults() -> None:
    cfg = TrainingConfig(loss_type="portfolio_ic")

    assert cfg.loss_type == "portfolio_ic"
    assert cfg.portfolio_ic_top_k == 10
    assert cfg.portfolio_ic_weight == 0.25
    assert cfg.portfolio_ic_temperature == 0.25


def test_create_config_from_dict_accepts_portfolio_ic_overrides() -> None:
    cfg = create_config_from_dict(
        {
            "training": {
                "loss_type": "portfolio_ic",
                "portfolio_ic_top_k": 5,
                "portfolio_ic_weight": 0.40,
                "portfolio_ic_temperature": 0.50,
                "selection_metric": "val_loss",
            }
        }
    )

    assert cfg.training.loss_type == "portfolio_ic"
    assert cfg.training.portfolio_ic_top_k == 5
    assert cfg.training.portfolio_ic_weight == 0.40
    assert cfg.training.portfolio_ic_temperature == 0.50
    assert cfg.training.selection_metric == "val_loss"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"portfolio_ic_top_k": 0}, "portfolio_ic_top_k"),
        ({"portfolio_ic_weight": -0.01}, "portfolio_ic_weight"),
        ({"portfolio_ic_weight": 1.01}, "portfolio_ic_weight"),
        ({"portfolio_ic_temperature": 0.0}, "portfolio_ic_temperature"),
    ],
)
def test_training_config_rejects_invalid_portfolio_ic_knobs(
    kwargs: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        TrainingConfig(loss_type="portfolio_ic", **kwargs)
