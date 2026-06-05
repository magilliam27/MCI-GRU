# Portfolio-IC Weight Sweep Handoff

Last updated: 2026-05-31

## Resume Here

- Continue in the clean Portfolio-IC worktree: `C:\Users\magil\.codex\worktrees\portfolio-ic-hybrid-testing\MCI-GRU`.
- Immediate next action: launch a Colab G4 full-grid training run for `portfolio_ic_weight=0.75` and `portfolio_ic_weight=1.0`, then replay saved predictions through the same PIT daily transaction-cost/rank-gate backtest used for pure IC, 25%, and 50%.
- Inspect first:
  - `scripts/gen_portfolio_ic_pit_nb.py`
  - `notebooks/portfolio_ic_pit_colab.ipynb`
  - `scripts/run_pit_saved_prediction_backtests.py`

## Current Objective

- The user wants to test whether leaning further into the portfolio utility term improves the Portfolio-IC Hybrid Loss, or whether the 50/50 version has conflicting signals between broad cross-sectional IC and soft top-10 utility.
- Next target weights are `0.75` and `1.0`; keep `top_k=10` and `temperature=0.25` fixed so the sweep isolates blend weight.

## What Changed

- Portfolio-IC implementation already exists on branch `codex/portfolio-ic-hybrid-testing`.
- Known implementation branch state from this handoff pass:
  - Worktree: `C:\Users\magil\.codex\worktrees\portfolio-ic-hybrid-testing\MCI-GRU`
  - Branch: `codex/portfolio-ic-hybrid-testing`
  - Status: clean relative to `origin/codex/portfolio-ic-hybrid-testing`
  - Commit recorded in prior handoff: `373dc18 Add portfolio IC hybrid loss`
- The main checkout `C:\Users\magil\MCI-GRU` is dirty and includes unrelated volatility-targeting changes plus local Colab screenshots/artifacts. Do not use the main checkout for experiment edits unless intentionally writing docs only.
- No code changes were made by this handoff beyond adding this Markdown file.

## Key Decisions

- Run only the new upward sweep variants unless the existing Drive results are missing:
  - `portfolio_ic_weight75`
  - `portfolio_ic_weight100`
- Do not rerun pure IC, weight 25, or weight 50 by default. Existing comparable results are already available.
- Keep the training recipe aligned with the frozen PIT recipe:
  - years `2022, 2023, 2024, 2025`
  - seeds `314159, 271828, 161803`
  - `num_models=20`
  - `num_epochs=100`
  - `early_stopping_patience=15`
  - `loss_type=portfolio_ic`
  - `selection_metric=val_loss`
  - `portfolio_ic_top_k=10`
  - `portfolio_ic_temperature=0.25`
- User explicitly wants Colab on a G4-class GPU. Confirm with `nvidia-smi`; do not accept a T4 runtime for full training.
- Keep Sharpe, drawdown, turnover, and transaction costs out of the v1 training loss. Evaluate them post-training through saved-prediction backtests.

## Important Files

- `mci_gru/training/losses.py`: defines `SoftTopKForwardReturnLoss` and `PortfolioICLoss`; `weight=1.0` should become pure soft top-10 utility because the blended formula is `(1 - weight) * ICLoss + weight * SoftTopKForwardReturnLoss`.
- `mci_gru/config.py`: validates `portfolio_ic_weight` in `[0, 1]`, so both `0.75` and `1.0` are legal.
- `mci_gru/training/trainer.py`: wires `loss_type=portfolio_ic` to `PortfolioICLoss` and logs the selected loss settings.
- `scripts/gen_portfolio_ic_pit_nb.py`: current generator compares pure IC vs the 25% hybrid. Either duplicate it for an upward sweep or modify a Colab scratch copy so the job grid contains only 75% and 100%.
- `notebooks/portfolio_ic_pit_colab.ipynb`: existing Colab notebook generated from the script above.
- `scripts/run_pit_saved_prediction_backtests.py`: orchestration script for saved PIT prediction backtests with transaction costs and rank-drop gate.
- `tests/test_portfolio_ic_config.py`: confirms `portfolio_ic_weight=1.0` is valid via the `[0, 1]` validation boundary.
- `docs/handoffs/2026-05-29-portfolio-ic-training-metrics.md`: prior handoff about adding component metrics; useful background but not required to launch the weight sweep.

## Verification

- This handoff pass inspected:
  - `git status --short`
  - `git branch --show-current`
  - `git diff --stat`
  - `git worktree list --porcelain`
  - portfolio worktree status via `git -c safe.directory=... status --short --branch`
  - `scripts/gen_portfolio_ic_pit_nb.py`
  - `scripts/run_pit_saved_prediction_backtests.py`
  - Portfolio-IC tests/config files
- No tests were run during this handoff-only turn.
- Current local evidence: Portfolio-IC code and tests are present in the main checkout, but the main checkout is dirty with unrelated volatility files. The clean continuation point is the Portfolio-IC worktree above.

## Data/Experiment State

- Existing pure IC and 25% hybrid post-training backtest:
  - Run root: `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid/20260529_021358_static_regime_full`
  - Summary folder: `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid/20260529_021358_static_regime_full/summaries/portfolio_ic_pure_ic_and_weight25_pit_daily_tc_rank_gate/`
  - Results CSV: `portfolio_ic_pure_ic_and_weight25_pit_daily_tc_rank_gate_results.csv`
  - Status from prior Colab output: `Rows: 24`, `Failures: 0`
- Existing 50% hybrid post-training backtest:
  - Drive file title fetched earlier in this thread: `portfolio_ic_weight50_pit_daily_tc_rank_gate_results.csv`
  - Drive URL: `https://drive.google.com/file/d/19wdZwmPJ2v-ywreR6Dm1l9D5HX0NT6EO/view?usp=drivesdk`
  - Source run root recorded earlier: `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid_weight50/20260529_235430_static_regime_full`
- Comparable post-training scenario:
  - PIT daily saved-prediction backtest
  - top-k `10`
  - label horizon `5`
  - transaction costs enabled: spread `10` bps, slippage `5` bps
  - rank-drop gate enabled: `min_rank_drop=30`
  - one backtest pass with BHY adjustment

Known averaged results across 2022-2025 x 3 seeds:

| Variant | Total Return | ARR | ASR | MDD | Turnover | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure IC | 17.63% | 19.19% | 0.741 | -23.17% | 9.98% | 4.71% |
| Portfolio-IC 25% | 14.67% | 15.96% | 0.683 | -25.52% | 4.53% | 2.14% |
| Portfolio-IC 50% | 15.30% | 16.63% | 0.710 | -26.33% | 3.40% | 1.60% |

Interpretation so far:

- `0.50` improved over `0.25` on average return, ARR, ASR, turnover, and cost.
- Pure IC still has the best average raw return and ASR in this backtest.
- The 50% hybrid has much lower turnover and transaction-cost drag than pure IC.
- The open question is whether `0.75` continues the improvement, and whether `1.0` shows that IC was helping as regularization or was creating conflict.

## Open Risks

- `portfolio_ic_weight=1.0` removes the broad IC anchor. It may overfit top-k utility, lose cross-sectional ranking quality, or become less stable across regimes.
- `selection_metric=val_loss` is still the correct metric for Portfolio-IC variants, but the scalar loss is not directly comparable across different weights without component metrics.
- Training logs still do not expose IC-loss and soft-top-k-loss components unless the metric-instrumentation workstream has landed separately.
- G4 availability in Colab may drift. Confirm the actual GPU before launching full jobs.
- Existing Drive result paths are based on prior conversation and should be rechecked before final comparison if a file is missing or moved.

## Next Actions

1. In Colab, open or create an upward-sweep notebook from `notebooks/portfolio_ic_pit_colab.ipynb`; set the objective variants to only `portfolio_ic_weight75` and `portfolio_ic_weight100`.
2. Confirm runtime is G4-class with `nvidia-smi`, mount Google Drive, and confirm the market CSV, PIT universe CSV, and static regime inputs are available.
3. Launch the full grid with `SMOKE_MODE=False`: 4 years x 3 seeds x 2 weights x 20 models.
4. After training completes, run `scripts/run_pit_saved_prediction_backtests.py` in Colab on the new run root using the same backtest settings as previous results.
5. Compare pure IC, 25%, 50%, 75%, and 100% by total return, ARR, ASR, MDD, turnover, cost, and year-sliced behavior, with special attention to 2022 stress behavior and 2025 continuation.

## Commands Run

- `Get-Content -Path C:\Users\magil\.codex\skills\handoff\SKILL.md`
- `rg -n "portfolio_ic|Portfolio-IC|handoff|handoffs|workstream" C:\Users\magil\.codex\memories\MEMORY.md`
- `git status --short`
- `git branch --show-current`
- `git diff --stat`
- `rg -n "portfolio_ic|Portfolio-IC|PortfolioIC|portfolio_ic_weight" mci_gru configs tests notebooks scripts docs CONTEXT.md`
- `Get-ChildItem -Path docs\handoffs -Force`
- `git worktree list --porcelain`
- `git -c safe.directory=C:/Users/magil/.codex/worktrees/portfolio-ic-hybrid-testing/MCI-GRU -C C:/Users/magil/.codex/worktrees/portfolio-ic-hybrid-testing/MCI-GRU status --short --branch`
- `Get-Content -Path scripts\gen_portfolio_ic_pit_nb.py -TotalCount 360`
- `Get-Content -Path scripts\run_pit_saved_prediction_backtests.py -TotalCount 150`

## User Preferences

- Prefer running the experiment in Colab rather than locally.
- Use a G4-class GPU for full training; avoid T4 for this run.
- The user is comfortable with Google Drive access/clicks when needed.
- Keep the experiment realistic and scoped; performance judgment should come from actual PIT backtests, not just training loss.

## Do Not Do

- Do not touch unrelated volatility-targeting changes in the main checkout.
- Do not rerun expensive pure/25/50 jobs unless their existing artifacts are unavailable or corrupted.
- Do not change `top_k`, temperature, regime setup, cost assumptions, or rank gate during the 75/100 sweep.
- Do not call the 100% loss a replacement for IC until its post-training backtest and year-sliced behavior are reviewed.

## References

- Portfolio-IC branch: `codex/portfolio-ic-hybrid-testing`
- Prior Portfolio-IC metric handoff: `docs/handoffs/2026-05-29-portfolio-ic-training-metrics.md`
- Existing 25% run root: `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid/20260529_021358_static_regime_full`
- Existing 50% results Drive URL: `https://drive.google.com/file/d/19wdZwmPJ2v-ywreR6Dm1l9D5HX0NT6EO/view?usp=drivesdk`
- Default recipe doc: `docs/DEFAULT_EXPERIMENT_RECIPE.md`
