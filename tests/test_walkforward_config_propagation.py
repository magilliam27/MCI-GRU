"""Every walk-forward window must inherit the base configuration it was cloned from."""

from mci_gru.config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    WalkforwardConfig,
)
from mci_gru.walkforward import generate_walkforward_configs


def _base_config(evaluation: EvaluationConfig) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            train_start="2020-01-01",
            train_end="2022-12-31",
            val_start="2023-02-10",
            val_end="2023-08-31",
            test_start="2023-11-10",
            test_end="2024-06-30",
            skip_embargo_check=False,
        ),
        model=ModelConfig(label_t=5),
        training=TrainingConfig(
            walkforward=WalkforwardConfig(
                enabled=True,
                window_train_years=2,
                window_val_months=3,
                test_span_months=2,
                step_months=6,
                max_windows=2,
            )
        ),
        evaluation=evaluation,
    )


def test_walkforward_windows_carry_configured_evaluation() -> None:
    evaluation = EvaluationConfig(
        bootstrap_resamples=17,
        bootstrap_seed=4242,
        ci_level=0.8,
        sharpe_method="naive",
        top_k_values=[3, 7],
    )

    windows = generate_walkforward_configs(_base_config(evaluation))

    assert windows
    for window in windows:
        assert window.evaluation.bootstrap_resamples == 17
        assert window.evaluation.bootstrap_seed == 4242
        assert window.evaluation.ci_level == 0.8
        assert window.evaluation.sharpe_method == "naive"
        assert window.evaluation.top_k_values == [3, 7]


def test_walkforward_windows_do_not_silently_fall_back_to_evaluation_defaults() -> None:
    defaults = EvaluationConfig()
    configured = EvaluationConfig(bootstrap_resamples=defaults.bootstrap_resamples + 1)

    windows = generate_walkforward_configs(_base_config(configured))

    assert windows
    for window in windows:
        assert window.evaluation.bootstrap_resamples != defaults.bootstrap_resamples
