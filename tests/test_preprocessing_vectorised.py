"""Regression tests for vectorised preprocessing (matches legacy semantics).

Also covers the session-level train/val embargo (issue #115): labels are per-stock row
shifts over a session-indexed panel, so the embargo must be counted in trading sessions
rather than the calendar days ``ExperimentConfig._validate_embargo`` compares.
"""

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
)
from mci_gru.data.preprocessing import (
    assert_training_labels_respect_embargo,
    compute_labels,
    generate_time_series_features,
    purge_training_sessions_for_embargo,
)
from mci_gru.pipeline import prepare_data


def _legacy_generate_time_series_features(
    df: pd.DataFrame,
    kdcode_list: list[str],
    feature_cols: list[str],
    his_t: int,
) -> np.ndarray:
    """Reference implementation (iterrows) for equivalence checks."""
    all_dates = sorted(df["dt"].unique())
    num_stocks = len(kdcode_list)
    num_features = len(feature_cols)
    num_usable_days = len(all_dates) - his_t

    stock_features = np.zeros((num_usable_days, num_stocks, his_t, num_features), dtype=np.float32)

    stock_to_idx = {stock: idx for idx, stock in enumerate(kdcode_list)}
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}

    df_subset = df[df["kdcode"].isin(kdcode_list)][["kdcode", "dt"] + feature_cols].copy()
    pivot_data = np.zeros((len(all_dates), num_stocks, num_features), dtype=np.float32)

    for _, row in df_subset.iterrows():
        kdcode = row["kdcode"]
        dt = row["dt"]
        if kdcode in stock_to_idx and dt in date_to_idx:
            stock_idx = stock_to_idx[kdcode]
            date_idx = date_to_idx[dt]
            pivot_data[date_idx, stock_idx, :] = row[feature_cols].values.astype(np.float32)

    for day_offset in range(num_usable_days):
        stock_features[day_offset, :, :, :] = pivot_data[
            day_offset : day_offset + his_t, :, :
        ].transpose(1, 0, 2)

    return stock_features


def test_generate_time_series_features_matches_legacy_small_grid():
    """Vectorised path must match iterrows reference on a tiny panel."""
    dates = [f"2020-01-{d:02d}" for d in range(1, 11)]
    kdcodes = ["A", "B", "C"]
    feature_cols = ["f1", "f2"]
    rows = []
    rng = np.random.default_rng(0)
    for dt in dates:
        for k in kdcodes:
            rows.append(
                {
                    "kdcode": k,
                    "dt": dt,
                    "f1": float(rng.random()),
                    "f2": float(rng.random()),
                }
            )
    df = pd.DataFrame(rows)
    his_t = 3

    got = generate_time_series_features(df, kdcodes, feature_cols, his_t)
    want = _legacy_generate_time_series_features(df, kdcodes, feature_cols, his_t)

    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_generate_time_series_features_duplicate_dt_kdcode_last_wins():
    """When duplicate (dt, kdcode) rows exist, last row wins (legacy iterrows behaviour)."""
    df = pd.DataFrame(
        [
            {"kdcode": "X", "dt": "2020-01-01", "f1": 1.0, "f2": 0.0},
            {"kdcode": "X", "dt": "2020-01-01", "f1": 99.0, "f2": 1.0},
            {"kdcode": "X", "dt": "2020-01-02", "f1": 2.0, "f2": 0.0},
            {"kdcode": "X", "dt": "2020-01-03", "f1": 3.0, "f2": 0.0},
            {"kdcode": "X", "dt": "2020-01-04", "f1": 4.0, "f2": 0.0},
        ]
    )
    kdcode_list = ["X"]
    feature_cols = ["f1", "f2"]
    his_t = 2

    got = generate_time_series_features(df, kdcode_list, feature_cols, his_t)
    want = _legacy_generate_time_series_features(df, kdcode_list, feature_cols, his_t)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def _label_panel(
    sessions: list[str], kdcodes: list[str], skip: dict[str, set[str]] | None = None
) -> pd.DataFrame:
    """One row per (kdcode, session); ``skip`` removes sessions for individual stocks."""
    skip = skip or {}
    rows = []
    for kdcode in kdcodes:
        for idx, session in enumerate(sessions):
            if session in skip.get(kdcode, set()):
                continue
            close = 100.0 + idx + len(kdcode)
            rows.append({"kdcode": kdcode, "dt": session, "close": close})
    return pd.DataFrame(rows)


def test_purge_training_sessions_drops_exactly_label_t_sessions():
    """The purge removes the final label_t sessions of training signal, nothing more."""
    sessions = [f"2020-01-{d:02d}" for d in range(1, 11)]

    kept = purge_training_sessions_for_embargo(sessions, his_t=2, label_t=3)

    assert kept == sessions[:-3]
    assert purge_training_sessions_for_embargo(sessions, his_t=2, label_t=0) == sessions


def test_purge_training_sessions_refuses_when_no_labels_would_remain():
    """Too-short training windows must fail loudly, not silently yield an empty axis."""
    sessions = [f"2020-01-{d:02d}" for d in range(1, 6)]

    with pytest.raises(ValueError, match="leaves no training labels"):
        purge_training_sessions_for_embargo(sessions, his_t=2, label_t=4)


def test_embargo_validator_flags_label_maturing_on_first_validation_session():
    """A calendar gap wider than label_t is not enough: the panel gap is in sessions."""
    sessions = [
        "2023-12-26",
        "2023-12-27",
        "2023-12-28",
        "2023-12-29",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
    ]
    panel = _label_panel(sessions, ["AAA", "BBB"])
    # 2023-12-31 -> 2024-01-08 is an 8 calendar-day gap, which clears the config-level
    # calendar check for label_t=5, but only 4 sessions separate the two windows.
    train_label_dates = sessions[:4]

    with pytest.raises(ValueError, match="Train/val embargo violated"):
        assert_training_labels_respect_embargo(
            panel, ["AAA", "BBB"], train_label_dates, "2024-01-08", label_t=5
        )

    # Same panel, same val_start: signal dates whose labels mature earlier are accepted.
    summary = assert_training_labels_respect_embargo(
        panel, ["AAA", "BBB"], sessions[:2], "2024-01-08", label_t=5
    )
    assert summary["last_union_outcome_date"] == "2024-01-04"
    assert summary["rows_without_matured_label"] == 0


def test_embargo_validator_flags_stock_whose_own_sessions_reach_into_validation():
    """Per-stock row shifts, not the union axis, are what compute_labels consumes."""
    sessions = [f"2020-01-{d:02d}" for d in range(1, 15)]
    val_start = "2020-01-11"
    train_label_dates = sessions[:5]
    # GAPPY is missing two mid-panel sessions, so its 5th subsequent row is 2020-01-11.
    panel = _label_panel(sessions, ["AAA", "GAPPY"], skip={"GAPPY": {"2020-01-06", "2020-01-07"}})

    # The union axis alone says every training label matures by 2020-01-10.
    union_only = assert_training_labels_respect_embargo(
        _label_panel(sessions, ["AAA"]), ["AAA"], train_label_dates, val_start, label_t=5
    )
    assert union_only["last_union_outcome_date"] == "2020-01-10"

    with pytest.raises(ValueError, match="on their own session axis"):
        assert_training_labels_respect_embargo(
            panel, ["AAA", "GAPPY"], train_label_dates, val_start, label_t=5
        )


def test_embargo_validator_refuses_panel_that_ends_before_labels_mature():
    """A truncated panel must abort, not count unmatured labels as compliant."""
    sessions = [f"2020-01-{d:02d}" for d in range(1, 10)]
    panel = _label_panel(sessions, ["AAA"])

    with pytest.raises(ValueError, match="ends before the last training label matures"):
        assert_training_labels_respect_embargo(
            panel, ["AAA"], sessions[:5], "2020-01-20", label_t=5
        )


def test_embargo_validator_rejects_label_date_missing_from_panel():
    """An unknown training label date means the axes disagree; refuse to guess."""
    sessions = [f"2020-01-{d:02d}" for d in range(1, 15)]
    panel = _label_panel(sessions, ["AAA"])

    with pytest.raises(ValueError, match="absent from the label panel"):
        assert_training_labels_respect_embargo(
            panel, ["AAA"], ["2019-12-31"], "2020-01-11", label_t=5
        )


class _PassThroughFeatureEngineer:
    def __init__(self, feature_cols: list[str]):
        self._feature_cols = feature_cols

    def transform(self, df, *_args):
        return df

    def get_feature_columns(self) -> list[str]:
        return self._feature_cols


_OHLC_COLS = ["close", "open", "high", "low", "volume", "turnover"]
_HOLIDAYS = {"2023-11-23", "2023-12-25", "2024-01-01", "2024-01-15", "2024-02-19"}


def _sessions_2024() -> list[str]:
    """Business days around the shipped 2023-12-31 / 2024-01-08 split, holidays removed."""
    days = pd.bdate_range("2023-11-01", "2024-02-29").strftime("%Y-%m-%d").tolist()
    return [d for d in days if d not in _HOLIDAYS]


def _write_ohlc_panel(path, sessions: list[str], kdcodes: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for kdcode in kdcodes:
        price = 100.0
        for session in sessions:
            price *= 1.0 + float(rng.normal(0.0005, 0.02))
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": session,
                    "open": price * 0.999,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000.0,
                    "turnover": price * 1_000_000.0,
                }
            )
    panel = pd.DataFrame(rows)
    panel.to_csv(path, index=False)
    return panel


@pytest.mark.parametrize("skip_embargo_check", [False, True])
def test_prepare_data_training_labels_never_consume_validation_closes(
    tmp_path, skip_embargo_check: bool
):
    """End-to-end: no training label may read a close at or after val_start.

    The split is the shipped shape from issue #115 -- train_end 2023-12-31, val_start
    2024-01-08, label_t=5 -- where the 8 calendar-day gap clears the config-level check
    but only four sessions separate the windows, so the label for the last training
    session consumed the first validation close.

    The oracle is independent of the purge: labels recomputed from a panel truncated
    before val_start must equal the labels prepare_data returned. Any label reaching
    into validation shows up as a mismatch. ``skip_embargo_check`` is parametrised
    because it governs the calendar check only -- the session-level guarantee holds
    either way.
    """
    sessions = _sessions_2024()
    kdcodes = ["AAA", "BBB", "CCC"]
    data_path = tmp_path / "panel.csv"
    panel = _write_ohlc_panel(data_path, sessions, kdcodes)

    label_t = 5
    val_start = "2024-01-08"
    cfg = ExperimentConfig(
        data=DataConfig(
            source="csv",
            filename=str(data_path),
            train_start="2023-11-01",
            train_end="2023-12-31",
            val_start=val_start,
            val_end="2024-01-19",
            test_start="2024-01-29",
            test_end="2024-02-29",
            skip_embargo_check=skip_embargo_check,
        ),
        features=FeatureConfig(
            base_features=_OHLC_COLS,
            include_momentum=False,
            include_weekly_momentum=False,
        ),
        graph=GraphConfig(judge_value=0.9999, use_multi_feature_edges=False),
        model=ModelConfig(his_t=2, label_t=label_t),
        training=TrainingConfig(num_epochs=1, num_models=1, label_type="returns"),
        tracking=TrackingConfig(enabled=False),
    )

    data = prepare_data(cfg, _PassThroughFeatureEngineer(_OHLC_COLS))

    train_label_dates = data["train_dates"]
    assert train_label_dates[-1] == "2023-12-21"
    assert not set(train_label_dates) & {
        "2023-12-22",
        "2023-12-26",
        "2023-12-27",
        "2023-12-28",
        "2023-12-29",
    }
    assert data["train_labels"].shape[0] == len(train_label_dates)

    truncated = panel[panel["dt"] < val_start]
    expected = compute_labels(
        truncated, data["kdcode_list"], train_label_dates, label_t, fill_missing=True
    )
    np.testing.assert_allclose(data["train_labels"], expected, rtol=1e-6, atol=1e-8)
    assert np.abs(expected).sum() > 0.0


@pytest.mark.parametrize("his_t", [1, 2, 5])
def test_generate_time_series_features_shape(his_t: int):
    """Output shape matches (num_dates - his_t, n_stocks, his_t, n_features)."""
    dates = [f"2020-01-{d:02d}" for d in range(1, 21)]
    stocks = ["S1", "S2"]
    feature_cols = ["a", "b"]
    rows = []
    for dt in dates:
        for s in stocks:
            rows.append({"kdcode": s, "dt": dt, "a": 1.0, "b": 2.0})
    df = pd.DataFrame(rows)
    out = generate_time_series_features(df, stocks, feature_cols, his_t)
    assert out.shape == (len(dates) - his_t, len(stocks), his_t, len(feature_cols))
