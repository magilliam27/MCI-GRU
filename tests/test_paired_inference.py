"""Tests for ``mci_gru.evaluation.paired_inference`` (ticket 179).

These guard the paired re-analysis of the graph-specification ablation: the
daily difference of an arm's cross-sectional IC against the control's on the
same dates, overlap-aware inference on that difference, BHY correction across
arms, and the power arithmetic that decides whether a multi-year protocol can
separate arms at all. Every function is pure; I/O stays in the notebook.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mci_gru.evaluation.paired_inference import (
    PairedMeanInference,
    align_daily_series,
    bhy_adjusted_p_values,
    minimum_detectable_effect,
    paired_daily_differences,
    paired_mean_inference,
    required_days,
    sharpe_block_bootstrap_ci,
    tail_share,
    winsorize_rows,
)


def _dated(values: list[float], start: str = "2025-01-22") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values)).strftime("%Y-%m-%d")
    return pd.Series(values, index=idx, dtype=float)


# ── alignment and differencing ─────────────────────────────────────────────


def test_align_daily_series_inner_joins_on_dates_and_keeps_date_order() -> None:
    a0 = _dated([0.1, 0.2, 0.3, 0.4])
    a1 = _dated([1.0, 2.0, 3.0], start="2025-01-23")  # misses the first date
    frame = align_daily_series({"A0_zeroed": a0, "A1_shipped": a1})
    assert list(frame.columns) == ["A0_zeroed", "A1_shipped"]
    assert list(frame.index) == list(a1.index)
    assert frame["A0_zeroed"].tolist() == [0.2, 0.3, 0.4]
    assert frame["A1_shipped"].tolist() == [1.0, 2.0, 3.0]


def test_align_daily_series_drops_dates_where_any_arm_is_non_finite() -> None:
    a0 = _dated([0.1, float("nan"), 0.3, 0.4])
    a1 = _dated([1.0, 2.0, float("inf"), 4.0])
    frame = align_daily_series({"A0_zeroed": a0, "A1_shipped": a1})
    assert list(frame.index) == [a0.index[0], a0.index[3]]
    assert frame["A0_zeroed"].tolist() == [0.1, 0.4]
    assert frame["A1_shipped"].tolist() == [1.0, 4.0]


def test_align_daily_series_refuses_an_empty_intersection() -> None:
    a0 = _dated([0.1, 0.2], start="2025-01-22")
    a1 = _dated([0.3, 0.4], start="2025-03-03")
    with pytest.raises(ValueError, match="no common dates"):
        align_daily_series({"A0_zeroed": a0, "A1_shipped": a1})


def test_paired_differences_are_arm_minus_control_and_exclude_the_control() -> None:
    frame = pd.DataFrame(
        {"A0_zeroed": [0.10, 0.20], "A1_shipped": [0.15, 0.10], "A2_thr05": [0.05, 0.25]},
        index=["2025-01-22", "2025-01-23"],
    )
    delta = paired_daily_differences(frame, control="A0_zeroed")
    assert list(delta.columns) == ["A1_shipped", "A2_thr05"]
    assert delta["A1_shipped"].tolist() == pytest.approx([0.05, -0.10])
    assert delta["A2_thr05"].tolist() == pytest.approx([-0.05, 0.05])


def test_paired_differences_require_the_control_column() -> None:
    frame = pd.DataFrame({"A1_shipped": [0.1]}, index=["2025-01-22"])
    with pytest.raises(KeyError, match="A0_zeroed"):
        paired_daily_differences(frame, control="A0_zeroed")


# ── paired inference on the difference ─────────────────────────────────────


def _inference(delta: pd.Series, **overrides) -> PairedMeanInference:
    kwargs = dict(
        arm="A3_topk20",
        control="A0_zeroed",
        label_horizon=5,
        block_size=5,
        n_resamples=400,
        seed=1729,
        ci_level=0.95,
    )
    kwargs.update(overrides)
    return paired_mean_inference(delta, **kwargs)


def test_paired_mean_inference_recovers_a_known_positive_shift() -> None:
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.002, size=240)
    delta = _dated((0.01 + noise).tolist())
    result = _inference(delta)
    assert result.arm == "A3_topk20" and result.control == "A0_zeroed"
    assert result.n_days == 240
    assert result.mean_delta == pytest.approx(0.01, abs=5e-4)
    assert result.median_delta == pytest.approx(0.01, abs=5e-4)
    assert result.sd_delta == pytest.approx(0.002, rel=0.25)
    assert result.win_rate == 1.0
    assert result.hac_p is not None and result.hac_p < 1e-6
    assert result.ci_lower is not None and result.ci_lower > 0.0
    assert result.ci_upper is not None and result.ci_upper > result.ci_lower


def test_paired_mean_inference_under_the_null_brackets_zero() -> None:
    rng = np.random.default_rng(11)
    delta = _dated(rng.normal(0.0, 0.05, size=240).tolist())
    result = _inference(delta)
    assert result.ci_lower is not None and result.ci_upper is not None
    assert result.ci_lower < 0.0 < result.ci_upper
    assert result.hac_p is not None and result.hac_p > 0.05
    assert 0.3 < result.win_rate < 0.7


def test_paired_mean_inference_uses_overlap_aware_defaults() -> None:
    delta = _dated(np.linspace(-0.01, 0.01, 30).tolist())
    result = _inference(delta)
    # Newey-West lags default to label_horizon - 1 and the block is label_horizon.
    assert result.hac_lags == 4
    assert result.block_size == 5


def test_paired_mean_inference_drops_non_finite_days_and_reports_the_count() -> None:
    values = [0.01] * 10 + [float("nan")] + [0.01] * 10
    result = _inference(_dated(values), n_resamples=50)
    assert result.n_days == 20
    assert result.mean_delta == pytest.approx(0.01)


def test_paired_mean_inference_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(3)
    delta = _dated(rng.normal(0.003, 0.03, size=120).tolist())
    first = _inference(delta)
    second = _inference(delta)
    assert first == second


# ── multiple comparisons ───────────────────────────────────────────────────


def test_bhy_step_up_matches_a_hand_computation() -> None:
    # m = 4, c(4) = 1 + 1/2 + 1/3 + 1/4 = 25/12.
    p = [0.01, 0.04, 0.03, 0.20]
    adjusted = bhy_adjusted_p_values(p)
    c_m = 25.0 / 12.0
    raw_sorted = {
        0.01: 0.01 * 4 * c_m / 1,  # rank 1
        0.03: 0.03 * 4 * c_m / 2,  # rank 2
        0.04: 0.04 * 4 * c_m / 3,  # rank 3
        0.20: 0.20 * 4 * c_m / 4,  # rank 4
    }
    # Monotone enforcement from the largest rank down pulls rank 2 down to rank 3's value.
    expected = [
        min(raw_sorted[0.01], raw_sorted[0.03], raw_sorted[0.04], raw_sorted[0.20]),
        min(raw_sorted[0.04], raw_sorted[0.20]),
        min(raw_sorted[0.04], raw_sorted[0.20]),
        raw_sorted[0.20],
    ]
    assert adjusted.tolist() == pytest.approx(expected)
    assert adjusted.tolist() == pytest.approx([0.083333, 0.111111, 0.111111, 0.416667], rel=1e-4)


def test_bhy_with_a_single_test_returns_the_raw_p_value() -> None:
    assert bhy_adjusted_p_values([0.037]).tolist() == pytest.approx([0.037])


def test_bhy_caps_at_one_and_preserves_nan_positions() -> None:
    adjusted = bhy_adjusted_p_values([0.9, float("nan"), 0.8])
    assert adjusted[0] == 1.0 and adjusted[2] == 1.0
    assert math.isnan(adjusted[1])


# ── power arithmetic ───────────────────────────────────────────────────────


def test_minimum_detectable_effect_matches_the_normal_approximation() -> None:
    # (z_0.975 + z_0.80) * sd / sqrt(n) = 2.80158 * 0.05 / sqrt(950).
    mde = minimum_detectable_effect(sd=0.05, n_days=950, power=0.8, alpha=0.05)
    assert mde == pytest.approx(2.80158 * 0.05 / math.sqrt(950), rel=1e-4)


def test_minimum_detectable_effect_scales_with_inverse_root_n() -> None:
    one = minimum_detectable_effect(sd=0.05, n_days=238)
    four = minimum_detectable_effect(sd=0.05, n_days=4 * 238)
    assert one / four == pytest.approx(2.0, rel=1e-9)


def test_required_days_inverts_minimum_detectable_effect() -> None:
    sd, n = 0.05, 950
    mde = minimum_detectable_effect(sd=sd, n_days=n)
    assert required_days(sd=sd, mde=mde) == n


def test_power_helpers_reject_nonsense_inputs() -> None:
    with pytest.raises(ValueError):
        minimum_detectable_effect(sd=-0.1, n_days=10)
    with pytest.raises(ValueError):
        minimum_detectable_effect(sd=0.1, n_days=0)
    with pytest.raises(ValueError):
        required_days(sd=0.1, mde=0.0)


# ── descriptive helpers ────────────────────────────────────────────────────


def test_tail_share_is_the_top_fraction_of_days_share_of_the_total() -> None:
    delta = np.array([1.0] * 9 + [11.0])
    assert tail_share(delta, top_fraction=0.1) == pytest.approx(11.0 / 20.0)


def test_tail_share_is_nan_when_the_total_is_zero() -> None:
    assert math.isnan(tail_share(np.array([1.0, -1.0]), top_fraction=0.5))


def test_winsorize_rows_clips_per_row_and_keeps_nan() -> None:
    rows = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 100.0],
            [float("nan"), -50.0, 0.0, 1.0, 2.0],
        ]
    )
    out = winsorize_rows(rows, lower_q=0.25, upper_q=0.75)
    # Row 0 quantiles at 25%/75% are 1.0 and 3.0.
    assert out[0].tolist() == pytest.approx([1.0, 1.0, 2.0, 3.0, 3.0])
    # Row 1 ignores the NaN when computing quantiles and keeps it in place.
    assert math.isnan(out[1][0])
    assert out[1][1] == pytest.approx(np.nanquantile(rows[1], 0.25))
    assert out[1][4] == pytest.approx(np.nanquantile(rows[1], 0.75))
    assert rows[0][4] == 100.0  # input untouched


def test_sharpe_block_bootstrap_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0008, 0.01, size=240)
    result = sharpe_block_bootstrap_ci(
        returns, nw_lags=4, block_size=5, n_resamples=300, seed=1729, ci_level=0.95
    )
    assert result["lower"] <= result["point"] <= result["upper"]
    assert result["n_days"] == 240
    # An annualised daily Sharpe of ~0.08 * sqrt(252) ~ 1.27 with 240 days has an
    # interval of order one; the point of the helper is to show that width.
    assert result["upper"] - result["lower"] > 0.5
