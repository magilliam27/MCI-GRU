import pytest

from mci_gru.config import TrainingConfig, create_config_from_dict


def test_training_config_accepts_efficiency_knobs() -> None:
    cfg = create_config_from_dict(
        {
            "training": {
                "test_batch_size": 16,
                "dataloader_num_workers": 2,
                "dataloader_pin_memory": True,
                "dataloader_persistent_workers": True,
                "dataloader_prefetch_factor": 3,
                "save_member_predictions": False,
                "save_checkpoints": False,
            }
        }
    )

    assert cfg.training.test_batch_size == 16
    assert cfg.training.dataloader_num_workers == 2
    assert cfg.training.dataloader_pin_memory is True
    assert cfg.training.dataloader_persistent_workers is True
    assert cfg.training.dataloader_prefetch_factor == 3
    assert cfg.training.save_member_predictions is False
    assert cfg.training.save_checkpoints is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"test_batch_size": 0}, "test_batch_size"),
        ({"dataloader_num_workers": -1}, "dataloader_num_workers"),
        (
            {"dataloader_num_workers": 0, "dataloader_persistent_workers": True},
            "dataloader_persistent_workers",
        ),
        ({"dataloader_prefetch_factor": 0}, "dataloader_prefetch_factor"),
    ],
)
def test_training_config_rejects_invalid_efficiency_knobs(
    kwargs: dict[str, int | bool], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        TrainingConfig(**kwargs)
