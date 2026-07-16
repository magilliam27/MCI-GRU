# MCI-GRU Research Evidence Ledger Implementation Plan

> **Status:** Implemented and locally verified on
> `codex/research-evidence-ledger-20260715`; ready for publication review. R0-A
> and R1-A change only research-evaluation code, tests, and documentation. No
> training, campaign backtest, paper trading, or Drive mutation was performed.
>
> **Verification:** A retained, gitignored 60-date synthetic CLI smoke produced
> the expected `PRELIMINARY_SIGNAL_EVIDENCE` result, independently reproduced
> every T+1-to-T+h label from the raw fixture, verified all manifest hashes, and
> reran byte-identically. All 35 focused research-ledger tests passed, including
> the saved-prediction no-training guard; after transplanting the ledger-only
> commit onto current `origin/main`, the full repository suite passed with 421
> tests, 2 skips, and 10 pre-existing `PytestReturnNotNoneWarning` warnings.
> Changed-file Ruff check and format check passed. The optimized SHA-null
> benchmark completed 1 date x 500 instruments x 1,000 draws in 0.42 seconds on
> the local reference venv.
>
> **Reviewed:** 2026-07-15
>
> **Publication baseline:**
> `811f4db54ad6542aa08fcf79c08ce99a4b79fc29` (`origin/main` at final rebase).
> The original implementation baseline was unpublished local commit
> `d6b0f60ba414d2152dfb0cc6ef715e0860d6a1fe`; it remains outside this
> workstream and is not part of the publication diff.
>
> **Supersedes:**
> `2026-07-15-mci-gru-retail-backtest-paper-shadow-implementation-plan.md`
>
> **Goal:** Determine whether frozen MCI-GRU scores contain repeatable
> out-of-sample stock-selection information beyond matched random labels.

## 1. Decision

Do not build a realistic trading simulator inside MCI-GRU now.

The active implementation has only two phases:

1. **R0 — Freeze the research protocol and validate the daily evidence set.**
2. **R1 — Compare dated MCI-GRU ranking evidence with a matched permutation
   null and produce one auditable report.**

Only if R1 finds preliminary signal should a later plan repair and extend the
economic portfolio replay. Prospective logging and external paper trading are
also later decisions. They are not part of this implementation.

The narrow claim under test is:

> For a frozen prediction rule and predeclared point-in-time scorable universe,
> MCI-GRU score ranks have positive out-of-sample cross-sectional association
> with the exact forward-return target and outperform exchangeable random score
> assignments.

This is evidence about information content. It is not proof of persistent alpha,
after-cost profitability, or executable retail performance.

## 2. What stays from the larger plan

Retain these useful decisions:

- Saved-prediction-only evaluation; no retraining.
- Arbitrary predeclared periods; no special 2026 code path.
- Strict point-in-time and no-lookahead validation.
- A canonical session calendar and explicit realized-outcome cutoff.
- An expected-scorable denominator independent of prediction rows.
- Deterministic ranking and null assignments.
- Verified adjusted-price research data only.
- Serial-correlation-aware inference.
- A trial-family reference so previously inspected variants are not forgotten.
- Versioned, deterministic, machine-readable evidence plus one human report.
- The existing backtest remains available for later economic context.

Move these ideas out of the active plan:

- $20,000/$50,000/$100,000 account scenarios.
- Cash, shares, buying power, no-leverage guards, orders, fills, and rejects.
- Raw-price corporate-action and delisting accounting.
- ADV, capacity, market impact, and partial-fill modeling.
- Broker or paper-account state, reconciliation, locks, recovery, and export.
- A new trading framework or a rewrite of `backtest_engine.py`.

If the signal is eventually confirmed, the preferred next execution test is an
external paper-trading platform. The deferred retail policy remains long-only,
no intentional leverage, and retail-scale; no endogenous market-impact model is
justified without evidence that the user's orders could move the market.

## 3. Evidence classes and claim statuses

### 3.1 Evidence labels

- **Verified repository fact:** directly visible in current code, tests, or a
  maintained repository document.
- **Design inference:** a consequence inferred from verified behavior.
- **Protocol default:** a setting frozen before new OOS outcomes are inspected.

When sources disagree, use current code/tests, then `AGENTS.md`, canonical docs,
current research evidence, and finally historical audits or plans.

### 3.2 Claim statuses

| Status | Meaning |
|---|---|
| `INVALID_EVIDENCE` | Timing, PIT, price, coverage, provenance, or protocol failed. |
| `INSUFFICIENT_EVIDENCE` | Valid mechanics but too few dates, outcomes, or null draws. |
| `NO_DETECTABLE_SIGNAL` | The frozen primary test does not reject the noise null. |
| `PRELIMINARY_SIGNAL_EVIDENCE` | Positive evidence from a valid but previously inspected, short, or multiplicity-incomplete sample. |
| `CONFIRMATORY_SIGNAL_EVIDENCE` | A future untouched/prospective study passes a separately frozen confirmation protocol. |

This active plan can produce at most `PRELIMINARY_SIGNAL_EVIDENCE`. Promotion to
confirmatory status belongs to a later plan using untouched or prospective data.

The default R1 positive-evidence rule is all of:

- Mean daily Rank IC is positive.
- Its 95% moving-block-bootstrap interval excludes zero.
- The one-sided empirical permutation p-value is at most 0.05 using the
  plus-one correction.
- At least 60 valid dates exist. This is a reporting floor, not a guarantee of
  statistical adequacy.
- No PIT, expected-score, outcome, price, timing, or provenance guard failed.
- Multiplicity is either supported by a complete trial ledger or explicitly
  labeled `UNADJUSTED_EXPLORATORY`.

## 4. Current repository evidence

### 4.1 Current paths

```text
averaged_predictions/*.csv
    -> scripts/backtest_sp500_daily.py
    -> mci_gru/evaluation/backtest_engine.py
       -> PIT/rank selection
       -> T+1-open to T+2-open policy return
       -> approximate turnover and fixed costs

averaged_predictions/*.csv
    -> scripts/run_saved_prediction_selection_audit.py
    -> mci_gru/evaluation/selection_audit.py
       -> exact trained-label returns
       -> daily IC and top-k returns
       -> Newey-West/bootstrap helpers
       -> JSON summary
```

### 4.2 Useful code to reuse

| Surface | Verified behavior | Decision |
|---|---|---|
| `mci_gru/data/preprocessing.py:174-195` | Training target is `close[T+label_t] / close[T+1] - 1`. | This exact target is the primary outcome. |
| `mci_gru/evaluation/selection_audit.py:35-112` | Saved predictions already produce IC and top-k label diagnostics. | Extend this path; do not create a second evaluator. |
| `scripts/run_saved_prediction_selection_audit.py` | A dedicated saved-prediction CLI already exists. | Extend it; do not add another research CLI. |
| `mci_gru/evaluation/statistics.py:45-76` | Daily Pearson/Spearman IC exists. | Preserve dates and invalid-date reasons. |
| `mci_gru/evaluation/statistics.py:79-151` | Newey-West and moving-block helpers exist. | Reuse them and add explicit dated mean inference where needed. |
| `mci_gru/evaluation/portfolio.py:13-22` | Score order is deterministic by score descending, then `kdcode`. | Preserve tie behavior. |
| `mci_gru/data/pit.py:20-35` | PIT normalization currently retains only `kdcode`, `valid_from`, and `valid_to`. | Preserve an optional `known_from` field and classify evidence without inventing it. |
| `mci_gru/evaluation/artifacts.py:14-43` | JSON is strict and refuses overwrite by default. | Extend only enough to write the five-file bundle. |
| `mci_gru/evaluation/trial_ledger.py:29-69` | A trial ledger already exists. | Read and hash it when available; do not build a new ledger system now. |
| `configs/config.yaml:88-96` | Evaluation bootstrap/Newey-West settings exist. | Reuse one statistical configuration surface. |

### 4.3 Gaps that affect the research conclusion

| Severity | Current behavior | Required correction |
|---|---|---|
| Critical | `prediction_report.py:69-103` inner-joins and removes nonfinite rows. | Emit the daily denominator and every exclusion before calculating metrics. |
| Critical | `selection_audit.py` uses matrices built only from aligned finite rows. | Select the evidence set using T-known eligibility, then validate complete outcomes. |
| Critical | Current PIT intervals contain effective dates but may lack `known_from`. | Classify PIT evidence; only known-as-of membership can support later confirmation. |
| High | `daily_ic_series()` returns an undated compact array. | Return one dated row per requested session, including nulls and reasons. |
| High | `top_k_returns()` filters finite score/outcome pairs before selecting. | Rank first from T-known scores; a missing matured outcome invalidates the date. |
| High | `selection_audit.py` trusts caller-supplied `trial_count`. | Treat caller counts as legacy only; read/hash a ledger or label inference unadjusted. |
| High | Grouped `shift()` can advance to the next observed stock row rather than the next canonical session. | Resolve T+1 and T+h from the study calendar with `fill_missing=False`. |

The existing backtest's missing selected-return reweighting, missing-benchmark
zero fill, and observed-row calendar are real defects, but they belong to the
conditional economic-replay follow-up, not R0/R1.

## 5. Minimum data contract

### 5.1 Required inputs

| Input | Minimum contract |
|---|---|
| Frozen predictions | `dt`, `kdcode`, `score`, ordered file hashes, row counts, source run, ensemble rule, member count, and seed/base-seed metadata |
| Adjusted market panel | `dt`, `kdcode`, `close`, declared adjustment provenance, content hash, and data-as-of date |
| PIT membership | `kdcode`, `valid_from`, `valid_to`, plus `known_from` or an explicit evidence downgrade |
| Expected-scorable sets | Per signal date, instruments expected to receive scores using only T-close-known facts, with exclusion reasons |
| Session calendar | Ordered exchange sessions, timezone, source, and content hash |
| Study protocol | Primary endpoint, horizon, dates, null, draw count/seed, block rule, alpha, top-k secondary metric, and trial-family reference |

OHLCV remains sufficient. R0/R1 need adjusted close; open/high/low/volume may
remain in the source file but are not used by the primary test. No quote,
order-book, broker, ADV, borrow, or corporate-action feed is required.

### 5.2 Price and PIT evidence classes

Price basis:

- `ADJUSTED_RESEARCH`: adjustment provenance is declared and hashed; valid for
  research returns only.
- `UNKNOWN`: diagnostics only; no headline result.

Universe knowledge:

- `KNOWN_AS_OF`: membership was known by signal close; eligible for a future
  confirmatory claim.
- `EFFECTIVE_ONLY`: effective interval is known but information timing is not;
  exploratory only.
- `UNKNOWN`: invalid.

Score denominator:

- `EXPECTED_SCORABLE`: independently constructed expected set; eligible for a
  future confirmatory claim.
- `SCORED_SET_ONLY`: prediction rows define the observed set; exploratory only.
- `UNKNOWN`: invalid.

R1 may run on `EFFECTIVE_ONLY` or `SCORED_SET_ONLY` evidence to diagnose current
saved runs, but the report must not imply strict PIT confirmation.

### 5.3 Daily validity rule

For a signal date T to enter the primary statistic:

1. T is in the canonical calendar.
2. PIT and expected-scorable sets are available as of T close.
3. Every expected instrument has exactly one finite score.
4. The exact canonical T+1 and T+h adjusted closes exist for every expected
   instrument after the outcome has matured.
5. No price is forward-filled, backfilled, or replaced with a later observed
   row; label construction uses `fill_missing=False`.
6. At least two instruments have nonconstant scores and outcomes.

If a matured expected instrument lacks either required close, invalidate the
whole date for the primary confirmatory statistic. An exploratory partial-set
metric may be reported separately, but it cannot affect the primary result.

Missing, duplicate, extra, or nonfinite expected scores likewise invalidate the
date. Missing outcomes and unmatured tail labels are never zero.

## 6. Frozen R1 protocol

### 6.1 Primary endpoint

For each valid signal date T and instrument i:

```text
label_return(T, i) =
    adjusted_close(canonical_session(T, +h), i)
    / adjusted_close(canonical_session(T, +1), i)
    - 1

daily_rank_ic(T) =
    Spearman(score(T, i), label_return(T, i))
    across the expected-scorable set

observed_statistic = mean_T(daily_rank_ic(T))
```

`h` is the frozen model `label_t`. This matches the target implemented in
`mci_gru/data/preprocessing.py:174-195` while making session alignment explicit.

The date is the inference unit. The number of stocks does not multiply the OOS
sample size. Ties use average ranks.

### 6.2 Secondary endpoint

For each valid date:

```text
top_k_spread(T) =
    mean(label_return of the score-ranked top k)
    - mean(label_return of the full expected-scorable set)
```

The default is top 10. This connects ranking information to the stock-selection
question without introducing entry prices, turnover, costs, cash, or fills.

Median Rank IC, Pearson IC, top-minus-bottom spread, quantiles, annualized
returns, Sharpe, and drawdown are not part of the MVP claim. They may be added
later only when they answer a specific decision.

### 6.3 Primary null

`WITHIN_DATE_SCORE_PERMUTATION_V1` keeps the same date, expected set, breadth,
score multiset, ties, and realized outcomes, but permutes score ownership among
instruments.

Deterministic assignment for each draw/date:

1. Sort source scores by instrument key.
2. Sort destination instruments by
   `SHA256(null_seed | draw_id | signal_dt | instrument_key)`.
3. Assign the ordered source scores to the ordered destinations.

The assignment never reads realized returns.

Protocol defaults:

- 5,000 valid draws; hard minimum 1,000.
- One fixed null seed.
- One-sided empirical p-value:

```text
p = (1 + count(null_mean_rank_ic >= observed_mean_rank_ic))
    / (1 + valid_draw_count)
```

The same null draws also produce the secondary top-k-spread distribution.
Per-draw rows need not be persisted; the algorithm version, seed, draw count,
input hashes, observed statistic, null quantiles, and assignment digest are
sufficient to reproduce the result.

### 6.4 Dependence-aware uncertainty

- For overlapping horizon `h`, Newey-West lag is at least `h - 1`.
- Moving-block-bootstrap block length is at least `h`.
- The protocol records the exact lag, block length, resamples, and seed.
- Blocks resample complete dated evidence rows, never unordered stock rows.
- Fewer than 60 valid dates cannot receive preliminary positive status.

HAC and bootstrap quantify date dependence. The permutation test answers the
primary exchangeable-label null. Report both; do not substitute a naïve t-test
or annualized Sharpe.

### 6.5 Seeds and periods

One study consumes one predeclared frozen prediction set. If five trained base
seeds exist, run five separate descriptive studies with the same protocol.

Report seed dispersion and selection overlap outside the primary inference, but
do not average seed p-values, treat seeds as independent histories, or select the
best seed. A future campaign-level estimand requires a separate protocol.

The same rule applies to periods: report predeclared period slices, but do not
combine or cherry-pick p-values. The 2026 reference is an exploratory study
because it has already been inspected.

### 6.6 Trial-family handling

`protocol.json` records:

- `trial_family_id`.
- Path/stable reference and SHA-256 of the existing trial ledger when one is
  complete.
- Whether the current OOS period was previously accessed.
- The registered multiplicity method, if available.

R0/R1 do not redesign `trial_ledger.py`. If the known trial family is incomplete,
the result is labeled `UNADJUSTED_EXPLORATORY` and cannot be promoted.

### 6.7 Adversarial-review hardening implemented

Before handoff, the implementation was tightened in these ways without changing
the research question:

- PIT knowledge is evaluated at an explicit per-session close when supplied,
  otherwise at a recorded exchange timezone and local close time. Naive
  `known_from` timestamps use a separately recorded timezone.
- An independent expected-scorable file must represent every active PIT member
  exactly once as scored or excluded, and exclusions require reasons.
- A source prediction commit, trained-label contract ID, trained-label horizon,
  price-adjustment provenance, ensemble rule/count, and seed ID are recorded;
  missing or mismatched label provenance invalidates the headline claim.
- Mechanical invalidity takes precedence over a short sample. A study with bad
  coverage is `INVALID_EVIDENCE`, not merely `INSUFFICIENT_EVIDENCE`.
- The approved per-instrument SHA-256 assignment order is preserved. Fixed
  score/outcome ranks are precomputed and the aggregate assignment digest uses
  `SHA256_ASSIGNMENT_INDEX_STREAM_V1`, materially reducing null runtime without
  changing assignments.
- A byte-identical bundle rerun is verified and returned. A conflicting bundle
  under the same semantic study ID is rejected.
- Trial-ledger completeness remains a declaration backed by a hashed ledger and
  matching family ID, so its output label is
  `DECLARED_COMPLETE_TRIAL_LEDGER`, not an independently proven completeness
  claim.

## 7. Minimal implementation design

### 7.1 Keep the current surface

Extend:

- `mci_gru/evaluation/selection_audit.py`
- `scripts/run_saved_prediction_selection_audit.py`
- `mci_gru/evaluation/statistics.py`
- `mci_gru/evaluation/artifacts.py`
- `mci_gru/data/pit.py` only to preserve/classify optional `known_from`
- Focused tests and one tiny fixture

Do not add a second research orchestrator or CLI. Split a small null helper from
`selection_audit.py` only if the existing Module becomes difficult to test; do
not create a general null-model framework.

### 7.2 Configuration

Add one narrow `SelectionResearchProtocol` record, located with the selection
audit unless reuse proves necessary. Its statistical core is:

```text
research_semantics_version
study_name
trial_family_id
prediction_input
market_input
PIT_input
expected_scorable_input
calendar_input
label_horizon
test_start
test_end
data_as_of
top_k
null_family
null_draws
null_seed
HAC_lag
bootstrap_block_length
bootstrap_resamples
bootstrap_seed
ci_level
alpha
```

The implemented record also carries the minimum provenance needed to validate
those fields: exchange/close and PIT timestamp timezones; price-adjustment
provenance; prediction source run/commit, ensemble rule/count, seed, trained
label contract/horizon; and an optional hashed trial-ledger reference.

There is no capital, account, execution, order, fill, liquidity, or broker
configuration.

The five `*_input` fields are runtime locators only. The canonical
`protocol.json` replaces them with stable source IDs, content hashes, and an
optional normalized repository-relative path. Operational absolute paths are
never written into or hashed by the canonical bundle, so identical content in
different worktrees has identical evidence bytes.

### 7.3 Required code changes

1. Load and hash all frozen inputs before evaluation.
2. Preserve and validate an optional PIT `known_from` field; never infer one
   from `valid_from`.
3. Build the requested date grid from the canonical session calendar.
4. Construct PIT/expected/scored/outcome coverage before any inner join.
5. Compute labels against exact canonical session offsets with
   `fill_missing=False`.
6. Emit one dated evidence row for every requested signal session.
7. Compute the primary statistic from only `VALID_PRIMARY` dates.
8. Run the deterministic within-date permutation over those same dates and
   denominators.
9. Compute HAC, block interval, empirical p-value, and claim status.
10. Write one five-file bundle without overwriting existing evidence.
11. Preserve the current selection-audit output as a compatibility mode during
    migration; mark its caller-supplied trial count and deflated-Sharpe output as
    legacy exploratory diagnostics.

No change to `backtest_engine.py` is required for R0/R1.

## 8. Five-file evidence bundle

```text
research_evidence/v1/<study_id>/
|-- protocol.json
|-- date_evidence.csv
|-- result.json
|-- report.md
|-- manifest.json
```

### 8.1 `protocol.json`

Contains:

- Schema and `research_semantics_version`.
- Frozen protocol fields from section 7.2.
- Prediction, market, PIT, expected-scorable, calendar, trial-ledger, and code
  hashes.
- Source run, ensemble, member-count, seed, price-basis, universe-knowledge,
  score-denominator, and data-as-of metadata.
- Requested dates and realized-outcome cutoff.

### 8.2 `date_evidence.csv`

One row per requested signal date:

```text
signal_dt
label_start_dt
label_end_dt
PIT_active_count
expected_scorable_count
prediction_count
finite_score_count
complete_outcome_count
daily_rank_ic
top_k_label_return
expected_set_label_return
top_k_spread
date_status
reason_codes
```

Invalid and unmatured dates remain in the file with null metrics and explicit
reasons.

### 8.3 `result.json`

Contains:

```text
schema
study_id
claim_status
evidence_class
valid_date_count
invalid_date_count
observed_mean_rank_ic
observed_mean_top_k_spread
HAC_method_lag_standard_error_t_p
bootstrap_method_block_resamples_seed_interval
null_family_draws_seed_assignment_digest_quantiles
empirical_p_value
multiplicity_status
adjusted_p_value_or_null
failed_guards
limitation_codes
```

### 8.4 `report.md`

Short outline:

1. Conclusion and claim status.
2. Frozen hypothesis and null.
3. Input/source hashes and evidence class.
4. Coverage and invalid dates.
5. Mean daily Rank IC, block interval, and permutation result.
6. Secondary top-10 spread.
7. Multiplicity status and limitations.
8. Go/no-go recommendation for an economic-replay follow-up.

### 8.5 `manifest.json`

Hashes the other four artifacts and is written last.

Canonical rules:

- JSON: UTF-8, LF, sorted keys, strict finite values, terminal newline.
- CSV: fixed columns/order, chronological rows, stable numeric formatting, LF,
  terminal newline, empty field for null.
- `study_id` hashes the research semantics version, canonical protocol, input
  hashes, and code commit/dirty-diff hash.
- Absolute paths and wall-clock timestamps are not written into the canonical
  bundle and do not affect identity.
- Existing canonical evidence cannot be overwritten with `force=True`.
- Identical semantic inputs verify or return the existing byte-identical bundle.
- Invalid studies still write the bundle with null headline fields.

No nested campaign store, per-draw warehouse, state lock, head pointer, or hash
chain is needed.

## 9. Active implementation slices

### R0-A — Denominator, calendar, and perfect-signal tracer

Extend the existing selection audit and CLI with the smallest end-to-end
fixture:

- Three valid signal dates plus required future sessions.
- Four stocks.
- One PIT entry and one PIT exit.
- One expected non-scorable exclusion.
- One missing expected score.
- One missing middle-session price.
- One unmatured tail outcome.
- One valid perfect-ranking date with score ties elsewhere.

Deliver:

- `SelectionResearchProtocol`.
- Input hashes and evidence classifications.
- Canonical-session label construction with `fill_missing=False`.
- `date_evidence.csv`.
- Five-file deterministic bundle with a non-promotional status.

Acceptance:

- PIT and score denominators are independent of realized outcomes.
- Missing expected scores and missing matured outcomes invalidate their dates.
- The perfect date has exactly the expected Rank IC.
- Future rows cannot alter earlier scores, eligibility, or date assignments.
- Identical inputs produce byte-identical artifacts.

### R1-A — Matched permutation, uncertainty, and report

Add:

- `WITHIN_DATE_SCORE_PERMUTATION_V1`.
- 5,000-draw deterministic null with exact assignment digest.
- Plus-one empirical p-value.
- HAC mean inference and moving-block interval.
- Claim-state rules and concise report.
- Legacy compatibility output for the existing selection-audit CLI.

Acceptance:

- Every draw preserves the date's expected set, score multiset, ties, and
  breadth.
- Assignments are deterministic and independent of realized outcomes.
- Inference operates on dates only.
- The block length and HAC lag respect the frozen horizon.
- The report cannot promote an invalid, short, multiplicity-incomplete, or
  nonsignificant study.

## 10. Exact MVP regression tests

1. `test_saved_prediction_selection_research_never_invokes_training`
2. `test_expected_scorable_set_is_independent_of_prediction_rows`
3. `test_optional_known_from_survives_pit_normalization_and_is_checked_at_signal_close`
4. `test_label_uses_exact_canonical_t_plus_1_and_t_plus_h_sessions`
5. `test_label_construction_never_fills_missing_middle_session`
6. `test_missing_expected_score_invalidates_date_without_reweighting`
7. `test_missing_matured_outcome_invalidates_whole_primary_date`
8. `test_unmatured_tail_outcome_is_reported_not_zeroed`
9. `test_future_rows_do_not_change_prior_denominator_or_scores`
10. `test_daily_spearman_is_computed_per_date_with_average_ties`
11. `test_perfect_ranking_fixture_has_exact_daily_rank_ic_one`
12. `test_top_k_spread_uses_the_same_expected_set_denominator`
13. `test_permutation_preserves_daily_set_score_multiset_and_ties`
14. `test_permutation_is_deterministic_and_outcome_independent`
15. `test_empirical_p_value_uses_plus_one_correction`
16. `test_overlap_aware_hac_and_block_settings_are_enforced`
17. `test_identical_inputs_produce_byte_identical_five_file_bundle`
18. `test_study_id_changes_with_research_semantics_or_code_hash`
19. `test_invalid_study_writes_null_result_and_manifest`
20. `test_report_never_promotes_nonsignificant_or_exploratory_evidence`
21. `test_identical_content_from_different_roots_has_same_study_id_and_bundle_bytes`

These tests are the active contract. Economic backtest timing, turnover, costs,
and missing-open regression tests belong to the conditional follow-up.

## 11. Go/no-go after R1

### No-go

Stop active engineering when:

- Evidence is invalid and required source data cannot be repaired cheaply.
- Mean Rank IC is not positive.
- The block interval includes zero.
- The permutation result is not significant.
- The apparent result depends on partial-set outcomes, an incomplete
  denominator, or untracked OOS selection.

Preserve the negative result in the evidence bundle. Do not respond by tuning k,
horizon, periods, gates, or seeds on the same OOS sample.

### Go

If R1 produces credible `PRELIMINARY_SIGNAL_EVIDENCE`, write a separate scoped
plan for:

1. Correcting the existing T-close/T+1-open/T+2-open economic replay where its
   calendar, missing selected prices, and missing benchmark can bias results.
2. Comparing ungated top 10 and the predeclared rank-drop policy gross first,
   then under the existing simple fixed-cost sensitivity.
3. Descriptive multi-seed and multi-period robustness without treating seeds as
   independent histories.
4. Freezing one future confirmation protocol.
5. Using an external paper simulator if prospective execution evidence is still
   desired.

That follow-up must not reopen account-ledger, broker-runtime, corporate-action,
market-impact, or fill-simulator work without a demonstrated research need.

## 12. Reference 2026 replay

The completed LambdaRankIC 2026-YTD replay remains a generic exploratory
fixture, not the architecture target.

Reference contract:

- Campaign commit `9bd17d5`; replay notebook commit `212de7f`.
- Base seeds `314159`, `271828`, `161803`, `141421`, and `173205`.
- `averaged_predictions` only; 20 members per seed.
- Test window 2026-01-01 through 2026-07-13.
- Realized t+5 label cutoff 2026-07-06.
- T-close signal, T+1-open entry, T+2-open policy return.
- Long-only top 10, PIT universe, rank-drop 30.
- 10 bps round-trip spread plus 5 bps slippage per side.
- Supplied mean net return 9.76%, benchmark 9.85%, and no seed significant at
  5%.

Use it to verify input discovery, generic dates, evidence downgrades, and report
interpretation. Do not special-case its 129-session-scale window or use it as an
untouched confirmation sample.

## 13. Best first implementation slice

Implement **R0-A — Denominator, calendar, and perfect-signal tracer** first.

It answers the prerequisite question: do we know exactly which stocks should
have been scored on each date, and can we align their complete trained-label
outcomes without future-dependent row dropping? Until that is true, a
permutation p-value or attractive backtest is not trustworthy. R0-A changes no
training, trading, paper-account, or Drive behavior and creates the evidence
spine needed by the only other active slice, R1-A.
