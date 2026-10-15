"""Each risk factor paired with its most similar prior-year risks (#289)."""

from __future__ import annotations

import pytest

from mci_gru.filings.pairing import PairingError, pair_consecutive_years, shortlist_prior_risks
from mci_gru.filings.risk_factors import RiskFactor


def _risk(year: int, index: int, heading: str, body: str) -> RiskFactor:
    return RiskFactor(
        accession_number=f"acc-{year}",
        fiscal_year=year,
        index=index,
        category=None,
        heading=heading,
        body=body,
    )


PRIOR = [
    _risk(
        2024, 1, "Advertising revenue concentration.", "Most revenue comes from advertising spend."
    ),
    _risk(2024, 2, "Intense competition.", "Competitors in search and cloud compete with us."),
    _risk(2024, 3, "Data privacy regulation.", "Privacy laws such as GDPR restrict our data use."),
    _risk(2024, 4, "Exchange rate risk.", "Currency movements affect international revenue."),
]
CURRENT = [
    _risk(
        2025, 1, "Privacy and data protection laws.", "New privacy laws restrict how we use data."
    ),
    _risk(2025, 2, "Artificial intelligence risks.", "AI products may produce harmful output."),
]


def test_most_similar_prior_risk_ranks_first():
    pairs = shortlist_prior_risks(CURRENT, PRIOR, k=2)
    privacy = [pair for pair in pairs if pair.current.risk_id == "acc-2025#001"]
    assert privacy[0].prior.risk_id == "acc-2024#003"
    assert [pair.rank for pair in privacy] == [1, 2]
    assert privacy[0].similarity >= privacy[1].similarity


def test_every_current_risk_gets_k_pairs_even_with_no_word_overlap():
    pairs = shortlist_prior_risks(CURRENT, PRIOR, k=3)
    by_current = {}
    for pair in pairs:
        by_current.setdefault(pair.current.risk_id, []).append(pair)
    assert {key: len(value) for key, value in by_current.items()} == {
        "acc-2025#001": 3,
        "acc-2025#002": 3,
    }


def test_k_larger_than_the_prior_year_takes_every_prior_risk():
    pairs = shortlist_prior_risks(CURRENT[:1], PRIOR, k=10)
    assert len(pairs) == len(PRIOR)


def test_k_none_pairs_every_current_risk_with_every_prior_risk_in_rank_order():
    pairs = shortlist_prior_risks(CURRENT, PRIOR, k=None)
    assert len(pairs) == len(CURRENT) * len(PRIOR)
    privacy = [pair for pair in pairs if pair.current.risk_id == "acc-2025#001"]
    assert [pair.rank for pair in privacy] == [1, 2, 3, 4]
    assert privacy[0].prior.risk_id == "acc-2024#003"


def test_pair_ids_are_stable():
    pair = shortlist_prior_risks(CURRENT[:1], PRIOR, k=1)[0]
    assert pair.pair_id == "acc-2025#001|acc-2024#003"


def test_consecutive_years_pair_each_year_with_the_one_before_only():
    older = [_risk(2023, 1, "Old risk.", "Old body text about privacy laws.")]
    pairs = pair_consecutive_years(older + PRIOR + CURRENT, k=1)
    assert {pair.current.fiscal_year for pair in pairs} == {2024, 2025}
    assert all(pair.prior.fiscal_year == pair.current.fiscal_year - 1 for pair in pairs)


def test_a_missing_prior_year_is_refused_rather_than_skipped():
    with pytest.raises(PairingError):
        pair_consecutive_years([_risk(2021, 1, "A.", "a b")] + CURRENT, k=1)
