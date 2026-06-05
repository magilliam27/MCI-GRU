# Portfolio-IC Training Metrics Handoff

Last updated: 2026-05-29

## Resume Here

- Start with the clean scoped worktree: `C:\Users\magil\.codex\worktrees\portfolio-ic-hybrid-testing\MCI-GRU`.
- Current objective: add metric instrumentation for the new `portfolio_ic` loss so training logs explain what the blended loss is doing, without changing the loss behavior or the live full-grid recipe.
- Immediate next move: inspect `mci_gru/training/losses.py` and `mci_gru/training/trainer.py`, then add a small component-metric API around `PortfolioICLoss` and focused tests.

## Current Objective

- The user wants a second workstream to improve observability for the Portfolio-IC Hybrid Loss.
- The specific problem from this chat: current epoch logs only show `Train Loss`, `Val Loss`, and `Val IC`. For `loss_type=portfolio_ic`, `Val Loss` is a blended scalar, so it hides whether movement comes from IC loss or the soft top-10 forward-return utility term.

## What Changed

- `portfolio_ic` already exists on branch `codex/portfolio-ic-hybrid-testing`.
- Commit: `373dc18 Add portfolio IC hybrid loss`.
- The branch is clean at last local check:
  - `git -c safe.directory=C:/Users/magil/.codex/worktrees/portfolio-ic-hybrid-testing/MCI-GRU -C C:/Users/magil/.codex/worktrees/portfolio-ic-hybrid-testing/MCI-GRU status --short --branch`
  - Output: `## codex/portfolio-ic-hybrid-testing...origin/codex/portfolio-ic-hybrid-testing`
- Main checkout is dirty with unrelated volatility-targeting and local artifacts. Do not work there unless intentionally creating docs only.

## Key Decisions

- Keep `portfolio_ic` as the canonical term.
- Do not change the v1 loss formula while adding metrics:
  - `(1 - weight) * ICLoss + weight * SoftTopKForwardReturnLoss`
  - Default knobs: `top_k=10`, `weight=0.25`, `temperature=0.25`
- Do not add Sharpe, drawdown, turnover, or transaction-cost terms to the training loss in this metric workstream.
- Use the static regime CSV for current Colab experiments while FRED observations are timing out from Colab:
  - `data/raw/regime/pit_repeated_seed_regime_inputs_20260520_183538.csv`
  - Source in Drive: `/content/drive/MyDrive/MCI-GRU-Ablations/pit_repeated_seed_replication/20260520_183538/inputs/pit_repeated_seed_regime_inputs_20260520_183538.csv`

## Important Files

- `mci_gru/training/losses.py`: defines `SoftTopKForwardReturnLoss` and `PortfolioICLoss`; best place to add reusable component computation.
- `mci_gru/training/trainer.py`: currently logs only `Train Loss`, `Val Loss`, and `Val IC`; `_train_epoch` and `_validate` return scalars only.
- `mci_gru/config.py`: owns `training.loss_type`, `portfolio_ic_top_k`, `portfolio_ic_weight`, and `portfolio_ic_temperature` validation.
- `tests/test_portfolio_ic_loss.py`: extend with component metric tests.
- `tests/test_portfolio_ic_trainer.py`: add a smoke assertion that trainer can surface Portfolio-IC component metrics.
- `scripts/gen_portfolio_ic_pit_nb.py` and `notebooks/portfolio_ic_pit_colab.ipynb`: update only after core logging/tests are stable.

## Verification

- Prior implementation verification from the Portfolio-IC branch: focused Portfolio-IC tests passed, full suite passed, and one-epoch local smoke runs completed.
- In this chat, strict regime-enabled Colab smoke with the static regime CSV passed for both variants:
  - Pure IC baseline: return code `0`, train loss `-0.001669`, val loss `0.001878`, val IC `-0.001878`.
  - Portfolio-IC hybrid: return code `0`, train loss `-0.001004`, val loss `0.005654`, val IC `-0.000572`.
- Live full grid was started in Colab:
  - Run root: `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid/20260529_021358_static_regime_full`
  - Manifest: `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid/20260529_021358_static_regime_full/portfolio_ic_pit_static_regime_full_manifest.json`
  - Results: `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid/20260529_021358_static_regime_full/portfolio_ic_pit_static_regime_full_results.json`
  - At last browser check, 3/24 jobs had returned and the next active job was training models in the 20-model ensemble.
- No metric-instrumentation code has been implemented yet in this handoff.

## Open Risks

- Live full-grid Colab run is ongoing; do not interrupt the notebook cell unless the user explicitly asks.
- Component metrics could increase log volume. Keep per-epoch output concise.
- Train component metrics require accumulation across batches; avoid doubling model forward passes.
- `portfolio_ic` uses standardized forward-return labels inside the loss, not actual trade PnL. Name metrics accordingly to avoid overclaiming.
- If adding MLflow or JSON summary fields, preserve backward compatibility for existing result readers.

## Next Actions

1. Add a `PortfolioICLoss.components(pred, target)` or standalone helper that returns at least:
   - `ic_loss`
   - `soft_topk_loss`
   - `soft_topk_utility`
   - `blended_loss`
2. Extend trainer aggregation so Portfolio-IC epochs can log:
   - `Train Loss`, `Train IC Loss`, `Train SoftTop10 Utility`
   - `Val Loss`, `Val IC`, `Val IC Loss`, `Val SoftTop10 Utility`
3. Add optional diagnostic metrics that are useful but not selection criteria:
   - hard top-10 standardized forward-return utility
   - hard top-10 raw forward-return mean
   - top-10 hit rate, if label sign is meaningful
   - soft inclusion effective breadth: `1 / sum(weights^2)`
4. Add tests for metric decomposition:
   - blended value equals `(1 - weight) * ic_loss + weight * soft_topk_loss`
   - metrics ignore NaNs/PIT-masked labels consistently with the loss
   - constant predictions/labels produce finite component metrics
5. Update the Portfolio-IC Colab generator after core tests pass so future notebooks display the component metrics in the run logs or saved summaries.

## Commands Run

- `git status --short --branch`
- `git -c safe.directory=C:/Users/magil/.codex/worktrees/portfolio-ic-hybrid-testing/MCI-GRU -C C:/Users/magil/.codex/worktrees/portfolio-ic-hybrid-testing/MCI-GRU status --short --branch`
- `rg -n "class SoftTopKForwardReturnLoss|class PortfolioICLoss|portfolio_ic|Selection metric|Train Loss|Val Loss|Val IC|compute_ic|criterion" ...`
- Browser/Colab checks confirmed static regime smoke results and live full-grid progress.

## Data/Experiment State

- Static regime source is available and validated:
  - Drive path: `/content/drive/MyDrive/MCI-GRU-Ablations/pit_repeated_seed_replication/20260520_183538/inputs/pit_repeated_seed_regime_inputs_20260520_183538.csv`
  - Schema: `dt`, `regime_market`, `regime_yield_curve`, `regime_oil`, `regime_copper`, `regime_stock_bond_corr`, `regime_monetary_policy`, `regime_volatility`
  - Coverage: `2001-01-04` through `2025-12-31`, `6606` rows.
- FRED was not a bad-key issue. Direct Colab HTTP to `fred/series/observations` returned `504`/timeouts; `fredapi` surfaced that as `ValueError: None`.
- The full static-regime run uses 2022-2025, seeds `314159`, `271828`, `161803`, both objective variants, `num_models=20`, `num_epochs=100`, `patience=15`.

## User Preferences

- Keep scope tight and preserve unrelated dirty work.
- For Colab, use the logged-in session and avoid noisy polling; report only meaningful progress or blockers.
- For this research track, prefer realistic first steps over big architecture changes.
- Keep Portfolio-IC v1 reversible and dependency-light.

## Do Not Do

- Do not alter the loss formula while adding observability.
- Do not disable regime features and call that canonical evidence.
- Do not restart/delete the active Colab runtime unless the user explicitly asks.
- Do not rely on live FRED for this run while the observations endpoint is timing out.
- Do not mix this metric work into the main checkout's volatility-targeting edits.

## References

- Branch: `codex/portfolio-ic-hybrid-testing`
- Commit: `373dc18 Add portfolio IC hybrid loss`
- Notebook URL: `https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/portfolio-ic-hybrid-testing/notebooks/portfolio_ic_pit_colab.ipynb`
- Live full-grid run root: `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid/20260529_021358_static_regime_full`
- Static-regime smoke run root: `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid/20260529_020617_static_regime_smoke`
