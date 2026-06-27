---
paper_title: "Re(Visiting) Time Series Foundation Models in Finance"
authors:
  - Eghbal Rahimikia
  - Hao Ni
  - Weiguan Wang
paper_date: "2025-11"
evaluated_on: "2026-05-13"
source_pdf: "C:\\Users\\magil\\Downloads\\ssrn-5770562.pdf"
source: "C:\\Users\\magil\\Downloads\\ssrn-5770562.pdf"
status: "evaluated"
decision: "pursue"
primary_landing_zone: "training/evaluation"
data_gate: "partial"
recommended_next_action: "Add TSFM-style evaluation metrics and long-window baseline sweeps before integrating external foundation-model dependencies."
github_issue_urls:
  - "https://github.com/magilliam27/MCI-GRU/issues/22"
  - "https://github.com/magilliam27/MCI-GRU/issues/23"
  - "https://github.com/magilliam27/MCI-GRU/issues/24"
---

# Research-to-Implementation Brief: Re(Visiting) Time Series Foundation Models in Finance

## Intake

This paper evaluates time series foundation models (TSFMs) for cross-sectional stock-return forecasting across a very large global daily equity panel. It compares zero-shot use of generic pre-trained TSFMs, fine-tuning on finance data, and finance-native pretraining from scratch against strong classical and machine-learning baselines.

The core result is useful for MCI-GRU because it is skeptical in the right direction: generic off-the-shelf TSFMs do poorly in finance, finance-specific training helps, longer historical windows help TSFMs more than short windows, and tree ensembles remain hard baselines to beat. The paper is not a reason to replace MCI-GRU with a foundation model. It is a reason to tighten evaluation, add stronger baseline discipline, and test whether longer temporal context improves the current architecture.

Important paper setup:

- Forecast target: one-day-ahead firm-level excess return.
- Input regime: univariate daily return histories, commonly tested at 5, 21, 252, and 512-day windows.
- Evaluation: out-of-sample R2 against zero, directional accuracy, macro-F1, and daily long-short decile portfolios.
- Main finding: off-the-shelf zero-shot and naive fine-tuned TSFMs underperform; finance-domain pretraining and data scaling improve results, but strong boosted-tree baselines remain competitive or better.
- Compute reality: full finance-native TSFM pretraining is expensive and depends on external data and model stacks not currently present in this repo.

## Mechanisms

1. **Finance-native temporal pretraining beats generic transfer, but generic zero-shot is not reliable.**

   The paper rejects the easy version of the TSFM story. Models pre-trained on broad non-financial time series do not automatically transfer to return prediction, and fine-tuning can deteriorate performance. Finance-domain pretraining helps because return series have domain-specific noise, weak signal, heavy tails, regime shifts, and low signal-to-noise structure.

   MCI-GRU implication: do not wire a generic Chronos/TimesFM style model into paper trading as a trusted alpha source without a provenance and backtest gate. If TSFM work enters the project, the first version should be an evaluation harness or external-prediction adapter, not a production inference dependency.

2. **Longer return histories matter for TSFM-like models.**

   The paper finds that TSFMs benefit substantially from longer input windows such as 252 or 512 trading days, while shorter windows favor many benchmark models. MCI-GRU currently defaults to a much shorter sequence length (`model.his_t=10` in the base Hydra config), though the architecture already supports configurable history length and an optional transformer temporal encoder.

   MCI-GRU implication: a staged long-window ablation is a better first experiment than importing a new foundation model. Test whether 21, 63, 126, and possibly 252-day windows improve IC and portfolio metrics under existing no-lookahead contracts.

3. **Forecast metrics and portfolio metrics can diverge.**

   The paper reports cases where hyperparameter choices worsen forecasting metrics but improve economic metrics such as Sharpe by changing volatility, turnover, or tail behavior. This is directly relevant because MCI-GRU already optimizes and selects on validation IC in some configurations, then evaluates ranked portfolios downstream.

   MCI-GRU implication: evaluation should report statistical forecast quality and economic usefulness together. A model that wins on R2 but fails on IC, turnover-adjusted return, or drawdown should not be promoted.

## Data Readiness Gate

**Already available in MCI-GRU**

- Daily stock-level OHLCV-derived features and labels through the existing pipeline.
- Configurable input history length via Hydra (`model.his_t`).
- Existing validation/test splits with strict train-period cutoffs.
- IC-style training and selection support (`loss_type=ic` / `combined`, `selection_metric=val_ic`).
- Portfolio evaluation surfaces, including ranking, top-k selection, rank-drop gating, turnover, Newey-West Sharpe, and bootstrap statistics.
- Optional transformer temporal encoder already referenced by existing tests/config paths.

**Derivable inside the repo**

- Out-of-sample R2 against a zero-return benchmark.
- Directional accuracy and macro-F1 for sign prediction.
- Daily high-minus-low or top-minus-bottom portfolio diagnostics from saved predictions.
- Performance decay by calendar year or rolling window.
- Long-window MCI-GRU and transformer-encoder sweeps using existing data loaders, subject to memory and runtime limits.

**External or blocked**

- Exact paper replication requires CRSP, Compustat Global, delisting returns, market-cap filtering, global country universes, and JKP factor data.
- Off-the-shelf TSFM inference requires external model dependencies and weights such as Chronos, TimesFM, or similar Hugging Face models.
- Finance-native TSFM pretraining at paper scale requires large GPU resources and a much larger data corpus than the current project appears to manage.
- Strong tree baselines such as CatBoost, XGBoost, and LightGBM are not currently repo dependencies.
- Exact excess-return replication needs a risk-free-rate source and point-in-time treatment. Any global version would require country-specific data decisions.

**Gate result**

Partial. The paper is actionable for evaluation design and history-window ablations now. Exact global replication and foundation-model training are blocked by data, dependency, and compute constraints.

## MCI-GRU Landing Zone Ranking

1. **Data/provenance boundary - prerequisite for exact replication and external baselines.**

   Define what can be evaluated with current MCI-GRU data, what requires external prediction files, and what is blocked pending CRSP/Compustat/JKP/risk-free provenance. This keeps the paper's global replication claims separate from MCI-GRU's current tradable universe.

2. **Training/evaluation diagnostics - highest value, easiest win.**

   Add TSFM-paper-style metrics to MCI-GRU reports: OOS R2 versus zero, directional accuracy, macro-F1, long-short/top-bottom portfolio returns, turnover-adjusted Sharpe, drawdown, and year-by-year decay. This directly improves model selection discipline without changing architecture.

3. **Config/experiment sweeps - high value, medium cost.**

   Run controlled `his_t` sweeps before importing foundation-model dependencies. Start with 21, 63, and 126 trading days, then attempt 252 only after memory and runtime are understood. Compare existing GRU-attention against the optional transformer temporal encoder.

4. **External score adapter - medium value, medium cost.**

   Define a CSV/parquet schema for external model predictions from CatBoost, LightGBM, Chronos, TimesFM, or other offline systems, then evaluate those predictions through the same MCI-GRU metrics and portfolio layer. This lets the project benchmark TSFM or tree-model claims without making those stacks production dependencies.

5. **Finance-native TSFM pretraining - long-term, high cost.**

   Consider only after evaluation and baseline discipline are in place. A realistic version would begin as self-supervised or supervised pretraining of an MCI-GRU-compatible temporal encoder, not as paper-scale replication.

Rejected as first landing zones:

- Directly adding a generic zero-shot TSFM to paper-trade inference.
- Replacing the current graph-aware architecture with a foundation model.
- Starting with synthetic data augmentation before baseline/evaluation gaps are closed.

## Invariant Check

- **No lookahead:** Any pretraining, fine-tuning, or benchmark fitting must be chronological. A zero-shot model is only valid for backtest comparison if its training corpus and release date cannot include future market data relative to the tested period.
- **Normalization:** Long-window experiments must preserve train-period normalization cutoffs. Longer `his_t` must not cause validation/test statistics to leak into feature scaling.
- **Graph timing:** Dynamic graph snapshots must still be resolved by sample date through `GraphSchedule`. Longer temporal windows cannot reuse future correlation windows.
- **Paper-trade boundary:** `paper_trade/` must continue using frozen checkpoint artifacts and frozen graph data. External TSFM or baseline scores should be evaluated offline first, then frozen explicitly before any paper-trade pathway is considered.
- **Augmentation:** Synthetic data or factor augmentation, if ever used, must be generated from training-window information only. Test-period distribution fitting would invalidate the experiment.
- **Comparability:** Statistical metrics and economic metrics should be computed on the same prediction dates and asset universe to avoid selection bias.

## Feasibility Opinion

**Easy wins**

- Add R2, direction accuracy, macro-F1, and yearly decay diagnostics to existing prediction evaluation. Created as [Issue #22](https://github.com/magilliam27/MCI-GRU/issues/22).
- Add a report template that compares MCI-GRU predictions to a zero benchmark and any supplied external baseline predictions. Created as [Issue #22](https://github.com/magilliam27/MCI-GRU/issues/22).
- Create Hydra experiment presets for `his_t` sweeps. Created as [Issue #23](https://github.com/magilliam27/MCI-GRU/issues/23).

**Medium-term**

- Evaluate 63/126/252-day windows with the existing GRU-attention and transformer temporal encoder. Created as [Issue #23](https://github.com/magilliam27/MCI-GRU/issues/23).
- Add external prediction ingestion for tree baselines and TSFM outputs. Created as [Issue #24](https://github.com/magilliam27/MCI-GRU/issues/24).
- Add a small, controlled CatBoost or LightGBM baseline only if the team accepts the new dependency.

**Long-term**

- Pretrain an MCI-GRU-compatible temporal encoder on finance data.
- Explore a TSFM adapter as an auxiliary alpha stream, not a replacement for graph-aware modeling.
- Use global or factor-augmented data only after point-in-time data provenance is solved.

My opinion: pursue the paper, but keep it evaluation-first. The most useful lesson is not "use foundation models"; it is "do not trust foundation models until they survive finance-specific baselines, long-window tests, and portfolio metrics."

## GitHub-Ready Slices

### 1. Data/provenance: Define TSFM comparison inputs

**Labels:** `data`, `evaluation`, `research-paper`

**GitHub issue:** [#24 Add external prediction ingestion for TSFM and tree baselines](https://github.com/magilliam27/MCI-GRU/issues/24)

**Problem**

The paper's exact empirical setup depends on external global equity data, delisting treatment, risk-free rates, JKP factors, and TSFM model provenance. MCI-GRU can still learn from the paper, but only if current-data experiments are separated from exact-replication claims.

**Proposed scope**

- Define the allowed current-data comparison universe for MCI-GRU experiments.
- Document external inputs required for exact replication: CRSP, Compustat Global, JKP factors, risk-free rates, delisting returns, and model-weight provenance.
- Define a prediction-file schema with date, ticker, prediction, optional model name, optional training cutoff, and optional realized return.
- Refuse external prediction files with duplicate date/ticker rows or missing prediction dates.
- State that external foundation-model weights are comparison-only until training data provenance is known.

**Acceptance criteria**

- A documented schema exists for external baseline and TSFM predictions.
- The schema includes a training-cutoff or provenance field.
- Exact paper replication is explicitly marked blocked unless external data requirements are met.

**Suggested tests**

- Unit test schema validation for required columns.
- Unit test duplicate date/ticker rejection.
- Unit test rejection or warning when provenance/training cutoff is missing.

**Out of scope**

- Installing Chronos, TimesFM, CatBoost, LightGBM, or XGBoost.
- Building CRSP/Compustat/JKP ingestion.
- Feeding external predictions into paper-trade inference.

**Feasibility Opinion**

Effort: easy win. Confidence: high. Rationale: this is mostly schema and documentation around existing evaluation flows. Main blocker: data.

### 2. Training/evaluation: Add TSFM-paper metrics to prediction evaluation

**Labels:** `training`, `evaluation`, `research-paper`

**GitHub issue:** [#22 Add TSFM-style prediction evaluation report](https://github.com/magilliam27/MCI-GRU/issues/22)

**Problem**

The TSFM paper evaluates models with both statistical and economic metrics. MCI-GRU already has IC and portfolio surfaces, but it does not yet package the paper's full comparison set in a single repeatable report.

**Proposed scope**

- Add OOS R2 versus zero-return benchmark.
- Add directional accuracy and macro-F1 for predicted sign.
- Add top-minus-bottom or decile-style portfolio diagnostics where prediction coverage allows it.
- Add year-by-year metric breakdowns to expose performance decay.
- Keep all metrics aligned to the same prediction dates and universe.

**Acceptance criteria**

- Existing tests pass.
- New metric tests cover zero benchmark, sign edge cases, and date alignment.
- Report can run on saved MCI-GRU predictions without retraining.

**Suggested tests**

- Unit test OOS R2 against a zero benchmark.
- Unit test directional accuracy and macro-F1 on positive, negative, and zero returns.
- Unit test date/universe alignment before portfolio metrics are computed.

**Out of scope**

- Changing the training objective.
- Adding new model dependencies.
- Promoting any baseline into paper trading.

**Feasibility Opinion**

Effort: easy win. Confidence: high. Rationale: repo already has prediction arrays, IC metrics, and portfolio utilities. Main blocker: validation cost.

### 3. Config/experiment: Create long-history MCI-GRU experiment presets

**Labels:** `config`, `experiment`, `research-paper`

**GitHub issue:** [#23 Create and evaluate long-history MCI-GRU presets](https://github.com/magilliam27/MCI-GRU/issues/23)

**Problem**

The paper finds TSFM-style models benefit from longer input windows, while MCI-GRU defaults to short history. This needs a controlled ablation before larger architecture work.

**Proposed scope**

- Add Hydra experiment presets for `his_t=21`, `63`, `126`, and optionally `252`.
- Include a memory/runtime note for long-window runs.
- Compare default temporal encoder against the existing transformer encoder where supported.
- Preserve train-only normalization and dynamic graph timing invariants.

**Acceptance criteria**

- Smoke run works for at least the 21-day preset.
- Config validation catches incompatible settings.
- Documentation states how to compare IC, R2, Sharpe, turnover, and drawdown across windows.

**Suggested tests**

- Smoke test the 21-day preset with a tiny epoch/model count.
- Unit or integration test that long-history windows still respect label/date alignment.
- Existing no-lookahead tests continue passing.

**Out of scope**

- Full 512-day production runs.
- Architecture replacement.
- Claims about TSFM superiority.

**Feasibility Opinion**

Effort: medium. Confidence: medium. Rationale: configuration is simple, but long windows can stress memory and runtime. Main blocker: validation cost.

### 4. ADR: Decide whether TSFM work belongs as dependency, adapter, or pretraining path

**Labels:** `adr`, `architecture`, `research-paper`

**Problem**

The paper suggests finance-specific TSFM training can help, but direct integration has data, compute, provenance, and paper-trade risks.

**Proposed scope**

- Document three options: no TSFM dependency, external-score adapter only, or in-repo temporal pretraining.
- Evaluate each option against no-lookahead, reproducibility, dependency burden, compute, and paper-trade deployment.
- Decide what evidence is required before TSFM outputs can influence paper-trade portfolios.

**Acceptance criteria**

- ADR names the chosen near-term path.
- ADR explicitly rejects direct zero-shot paper-trade integration unless provenance and backtest gates are satisfied.
- ADR links back to this evaluation note.

**Suggested tests**

- Documentation-only change; no code tests required.
- Review checklist confirms no paper-trade dependency is introduced by the ADR.

**Out of scope**

- Implementing TSFM training.
- Installing external model libraries.
- Creating live trading hooks.

**Feasibility Opinion**

Effort: easy win. Confidence: high. Rationale: the decision is architectural and should be captured before dependency work begins. Main blocker: production readiness.

## ADR Candidates

- **TSFM Integration Boundary:** Foundation-model outputs should enter first as external evaluated predictions, not as live paper-trade dependencies.
- **Long-Window Experiment Policy:** Long `his_t` experiments are allowed only through Hydra presets with explicit memory notes and no-lookahead validation.
- **Baseline Comparison Standard:** Research-inspired model work must compare against zero, current MCI-GRU, and at least one non-neural or external baseline before being considered for production.

## Rejected Ideas

- **Direct zero-shot TSFM alpha in paper trading:** Rejected for now because the paper itself shows weak zero-shot transfer and because model-pretraining provenance can create hidden lookahead risk.
- **Paper-scale finance TSFM pretraining immediately:** Rejected for now because the compute and data requirements are far outside an easy MCI-GRU iteration.
- **Synthetic data augmentation as the next step:** Rejected for now because evaluation and baseline gaps should be closed first, and augmentation introduces additional leakage risks.
- **Exact global replication:** Rejected for this project phase because it requires CRSP, Compustat Global, JKP factors, delisting handling, and global point-in-time data decisions.

## Open Questions

- What is the largest `his_t` that can train acceptably on the current hardware and dataset?
- Should the first external baseline be CatBoost, LightGBM, or an offline TSFM prediction file?
- Do we want exact excess-return labels using a risk-free-rate source, or is raw/relative return sufficient for the next MCI-GRU experiment?
- Should MCI-GRU selection remain centered on validation IC, or should a composite selection metric include portfolio Sharpe, turnover, and drawdown?
- What provenance standard is required before any external foundation-model weights are allowed into backtest comparison?
