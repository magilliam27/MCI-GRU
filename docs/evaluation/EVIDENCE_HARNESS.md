# Evidence Harness

This harness implements the first no-retraining evidence wave from
`docs/research/current/MCI_GRU_TOP_UNIVERSITY_RESEARCH_SCAN_2026-06-21.md`.
It is an additive operator layer around existing run folders, saved
predictions, PIT inputs, and market data.

## Outputs

| Artifact | Writer | Purpose |
| --- | --- | --- |
| `run_manifest.json` | `scripts/build_run_bundle_manifest.py` | Hashes core artifacts plus config, command, repo-dir git state, graph, checkpoint, data/PIT, environment, seed, MLflow, and paper-trade eligibility provenance for an existing run folder. |
| `artifact_validation.json` | `scripts/build_run_bundle_manifest.py` | Reports missing core artifacts without modifying the run. |
| `trial_ledger.csv` / `trial_ledger.jsonl` | `scripts/build_trial_ledger.py` | Lists candidate, failed, skipped, abandoned, and promoted trials. |
| `selection_audit_summary.json` | `scripts/run_saved_prediction_selection_audit.py` | Computes saved-prediction IC, top-k returns, bootstrap CIs, BHY multiple-testing haircut, deflated-Sharpe evidence for Sharpe-style claims, and explicit insufficient-evidence status. |
| `protocol.json`, `date_evidence.csv`, `result.json`, `report.md`, `manifest.json` | `scripts/run_saved_prediction_selection_audit.py --research-evidence` | Canonical research ledger for one frozen prediction set: exact calendar-aligned labels, complete PIT/expected-score coverage, dated Rank IC, matched within-date permutation, dependence-aware inference, claim guards, and immutable input/code identity. |
| `pit_availability_report.json` | `scripts/write_pit_availability_report.py` | Reports PIT breadth, missingness, staleness, and tradability without changing masks; optional calendars catch full-day market-panel outages. |
| `capacity_replay.json` / `capacity_replay.csv` | `scripts/run_saved_prediction_capacity_replay.py` | Replays saved scores through T+1 open, open-to-open gross/net returns, turnover, cost grid, rank-drop grid, lagged-ADV capacity, optional lagged-volatility gates, clipping, and unfillable diagnostics. |

## Invariants

- The harness does not retrain models or launch Colab.
- The harness does not change model, graph, feature, loss, PIT, or paper-trade defaults.
- The harness does not rewrite saved predictions, backtest outputs, `run_metadata.json`, `training_summary.json`, or `evaluation_summary.json`.
- Canonical research bundles have no `--force` path. A byte-identical rerun is
  verified and returned; conflicting bytes under the same study ID are rejected.
- Older evidence writers continue to refuse overwrite unless `--force` is passed.
- JSON artifacts are strict JSON; non-finite numeric values are written as `null`.
- Filesystem artifacts remain the source of truth; MLflow IDs and links are additive context.
- PIT masked-panel breadth stays intact; tradability and staleness are reported as separate evidence.
- Capacity calculations use lagged ADV and optional lagged-volatility thresholds known by the prediction date; realized T+1 volume is diagnostic only.
- Capacity replay scores at T close, enters at the next market open, and holds open-to-open unless a different future contract is named.
- Research-evidence labels use exact canonical T+1 and T+h sessions with no
  forward/back fill. The expected-scorable manifest must account for every
  active PIT member as scored or explicitly excluded.
- PIT `known_from` is evaluated at each session's declared signal close, with
  explicit exchange and naive-timestamp timezones recorded in `protocol.json`.
- The active research path can claim at most `PRELIMINARY_SIGNAL_EVIDENCE`;
  it does not claim executable returns, profitability, or paper-trading readiness.

## Operator Flow

1. Build or validate a run manifest for each existing run folder.
2. Record all sibling trials in a ledger, including failed and skipped variants.
3. Run the canonical saved-prediction research ledger before promoting any
   signal claim. Use the legacy selection audit only for compatibility.
4. Add PIT availability and tradability reports for masked-panel runs; pass an explicit calendar when full-day data outages are in scope.
5. Run economic or capacity replay only after the frozen research ledger finds
   preliminary signal and a separate follow-up protocol justifies it.

Keep outputs beside the run or in a clearly named evidence directory. Do not
treat ignored run folders as durable research evidence unless their exact
artifacts are preserved, hashed, and referenced by a current report.

## Command Sketches

```powershell
.\.venv\Scripts\python.exe scripts\build_run_bundle_manifest.py --run-dir outputs\run_id --repo-dir . --selection-rule "validation rank IC" --paper-trade-eligible false
.\.venv\Scripts\python.exe scripts\run_saved_prediction_selection_audit.py --predictions-dir outputs\run_id\averaged_predictions --market-data-path data\market.csv --output-dir outputs\run_id\evidence --trial-count 12 --top-k 20
.\.venv\Scripts\python.exe scripts\run_saved_prediction_selection_audit.py --research-evidence --predictions-dir outputs\run_id\averaged_predictions --market-data-path data\market.csv --pit-universe-csv data\pit_universe.csv --expected-scorable-csv data\expected_scorable.csv --calendar-csv data\trading_calendar.csv --output-dir research_evidence\v1 --study-name frozen-model-rank-ic --trial-family-id family-001 --test-start 2026-01-01 --test-end 2026-07-13 --data-as-of 2026-07-13 --label-t 5 --top-k 10 --price-basis ADJUSTED_RESEARCH --price-adjustment-provenance "declared adjusted-close source" --prediction-source-run-id run-id --prediction-source-code-commit commit --prediction-ensemble-rule averaged_predictions --prediction-ensemble-member-count 20 --prediction-seed-id 314159 --prediction-label-contract MCI_GRU_FORWARD_CLOSE_V1 --prediction-label-horizon 5
.\.venv\Scripts\python.exe scripts\write_pit_availability_report.py --market-data data\market.csv --pit-universe data\pit_universe.csv --calendar data\trading_calendar.csv --output outputs\run_id\evidence\pit_availability_report.json
.\.venv\Scripts\python.exe scripts\run_saved_prediction_capacity_replay.py --predictions outputs\run_id\averaged_predictions --market-data data\market.csv --output-dir outputs\run_id\evidence --aum 1000000 --top-k 20 --spread-bps 5 --slippage-bps 0 --min-rank-drop 30 --max-lagged-volatility 0.08
```
