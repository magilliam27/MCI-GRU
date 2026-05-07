# MCI-GRU Surfaces

Use this reference to classify finance-paper ideas into MCI-GRU work without turning every paper into a feature request.

## Required Repo Evidence

Use path plus concept evidence. Exact line numbers are optional.

- `mci_gru/features/registry.py`: `FeatureEngineer` orchestrates feature modules and `build_feature_list` composes feature columns from config flags.
- `mci_gru/pipeline.py`: `prepare_data` loads data, applies features, computes train-period normalization stats, builds tensors, and constructs static or dynamic graphs.
- `mci_gru/config.py`: `ExperimentConfig` validates train/validation and validation/test embargo gaps against `model.label_t`.
- `mci_gru/graph/builder.py`: `GraphBuilder` builds Pearson-correlation edges and `GraphSchedule` serves snapshots valid from dates using only data before each valid-from date.
- `mci_gru/data/data_manager.py`: `combined_collate_fn` returns the 9-tuple batch contract and resolves dynamic graph snapshots by sample date.
- `mci_gru/evaluation/statistics.py`: evaluation helpers cover IC, Newey-West Sharpe, and moving-block bootstrap confidence intervals.
- `mci_gru/evaluation/portfolio.py`: portfolio helpers cover top-k returns, deterministic ranking, turnover, and rank-drop gate logic.
- `docs/CONFIGURATION_GUIDE.md`: Hydra configs and typed dataclasses are the entrypoint for experiment presets and ablations.
- `docs/BACKTEST_FAIRNESS_AUDIT.md`: backtest review highlights execution timing, label leakage, graph timing, and return attribution risks.
- `paper_trade/scripts/infer.py`: paper-trade inference loads frozen checkpoints, `run_metadata.json`, and `graph_data.pt`; it does not rebuild research graphs on the fly.

## Slice Categories

- **Data issue**: new source, column, provenance, universe metadata, lag policy, point-in-time handling, missing-data behavior.
- **Feature issue**: new `mci_gru/features/` feature family, registry flag, warmup behavior, config wiring, and tests.
- **Graph issue**: `GraphBuilder`, `GraphSchedule`, edge attributes, graph timing, sector/relation graph, or graph diagnostics.
- **Model issue**: architecture changes in `mci_gru/models/`, stream fusion, attention, temporal encoder, or output head changes.
- **Training/evaluation issue**: losses, metrics, IC, bootstrap, Sharpe, top-k returns, portfolio diagnostics, or statistical tests.
- **Config/experiment issue**: Hydra preset, ablation matrix, train/val/test split, sweep, or smoke-run wiring.
- **Notebook issue**: exploratory reproduction, sensitivity analysis, diagnostics, or paper-faithful calculation outside production code.
- **Paper-trade issue**: frozen inference, monitoring, portfolio, reporting, or nightly pipeline. Use only after offline validation.
- **ADR issue**: durable architecture choice with hard-to-reverse consequences and real alternatives.

## Data Readiness Gate

Classify each required input:

| Status | Meaning | Action |
| --- | --- | --- |
| Already available | Present in current data flow or metadata | Proceed if invariants can be met |
| Derivable | Computable from current columns without new source access | Proceed with warmup/missing-data tests |
| External dependency | Requires FRED, LSEG field, WRDS/CRSP/Compustat, options data, accounting data, forecasts, or universe metadata | Draft a data issue first |
| Unavailable | No responsible source or proxy is available | Block implementation and list as rejected/open |

Do not use proxies by default. If exact data is unavailable, propose a data/provenance slice before any model-facing slice.

## Invariant Checklist

Every brief must explicitly check:

- Train-only normalization and reference statistics.
- Strict temporal cutoffs for feature estimates, graph edges, labels, and evaluation windows.
- Dynamic graph snapshot timing through `GraphSchedule`.
- Label embargo gaps between train/validation/test splits.
- Backtest fairness and return timing.
- Paper-trade frozen checkpoint and frozen `graph_data.pt` rule.
- Missing-data behavior, warmup periods, and survivorship/point-in-time universe risk.

## Landing Zone Ranking

Rank surfaces for each Research Mechanism:

- **Primary**: best first landing zone.
- **Secondary**: plausible later work.
- **Rejected or premature**: tempting but wrong now, with a reason.

Prefer evaluation/notebook diagnostics before production features when the paper idea is noisy, fragile, or sensitive to empirical choices. Prefer feature work before graph or model work when the same mechanism can be represented as stable node-level inputs. Treat paper-trade as late-stage unless the idea has already passed offline validation.
