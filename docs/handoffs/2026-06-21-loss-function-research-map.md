# Loss Function Research Map Handoff

Date: 2026-06-21

Workspace: `C:\Users\magil\.codex\worktrees\81be\MCI-GRU`

Base state checked: detached `HEAD` at `0d6b7c4da89b32858c13d8c06aa1816ddfc982b5`, which is the tip of `codex/colab-gpu-utilization-hardening-20260620`.

## Salvage Review

This handoff was recovered from the detached `0d6b7c4` decision surface after
`C:\Users\magil\.codex\worktrees\81be\MCI-GRU` no longer appeared in
`git worktree list --porcelain` and the directory only contained pytest
cache/temp files. The document is kept because it preserves the loss-path
rationale that connects the June 4 loss research, the early LambdaRankIC cap
evidence, the saved-prediction diagnostic need, and the decision to defer
path-dependent portfolio losses until chronological infrastructure exists.

Current `origin/main` contains newer LambdaRankIC diagnostics under
`docs/handoffs/2026-06-30-*` and `docs/handoffs/2026-07-01-*`. Those later docs
supersede this note for default-readiness: LambdaRankIC remains experimental,
pure IC remains the launch/default objective, and the current blocker is staging
saved prediction/trade artifacts for rank-stability, cross-seed agreement, and
rank-drop sensitivity checks. They do not fully replace this note's decision
logic for why higher pair caps, saved-prediction alignment diagnostics, and an
uncertainty-adjusted replay were the next loss-path ladder.

Consequence if this were dumped: the repo would lose the compact bridge between
the original loss-function literature/decision reports and the later
LambdaRankIC stability handoffs, including the explicit kill criteria and the
reason Sharpe/drawdown/turnover/optimizer-layer losses should stay out of the
current per-date training criterion.

## Executive Recommendation

Keep pure IC as the production-style default. The next loss work should be:

1. Run a cheap LambdaRankIC cap sweep at breadth-respecting caps before rejecting the implemented LambdaRankIC path.
2. Add or run a saved-prediction alignment diagnostic that joins Rank IC, top-k gross return, net return after costs, turnover, rank-drop behavior, Sharpe, and drawdown by year/seed.
3. Start an uncertainty-adjusted ranking replay using existing ensemble dispersion or residual-volatility proxies before adding a distributional head.

Do not promote direct Sharpe, drawdown, turnover, or optimizer-layer losses into the current training criterion yet. They are portfolio-aligned in principle, but they require chronological training blocks, previous-holding state, transaction-cost conventions, and split-boundary resets. Adding them to the current per-date criterion seam would be easy to make invalid.

## Current Evidence From Latest LambdaRankIC Tranche

The source-thread results make the 512-pair LambdaRankIC result a weak rejection test, not a clean rejection of the objective:

| Variant | Mean best val IC | Mean best val Rank IC | Elapsed seconds |
| --- | ---: | ---: | ---: |
| Pure IC baseline | 0.0116745624 | 0.0114966107 | 1472.696 |
| Portfolio-IC hybrid | 0.0092855887 | 0.0063391431 | 2361.768 |
| LambdaRankIC pairs512 | 0.0063347626 | 0.0088068908 | 1696.275 |

The same source thread found that 512 pairs covers only about 0.4%-0.5% of same-day PIT S&P breadth when 450-509 names are valid. A synthetic cap-vs-full probe reported gradient cosine around 0.61 at 512, 0.91 at 4096, and 0.96 at 8192 versus an effectively all-pairs reference. That strongly argues for testing 4096/8192 before concluding that LambdaRankIC itself is misaligned.

The optimized LambdaRankIC hot path appears semantics-preserving relative to the old triu/filter/cap path: local test/benchmark evidence reported `abs_loss_diff=0.0`.

## Implemented Loss Inventory

Implemented in `mci_gru/training/losses.py` and wired by `build_training_loss(...)`:

| Loss | Alignment to portfolio goal | Strength | Risk / gap |
| --- | --- | --- | --- |
| `mse` / `MaskedMSELoss` | Weak. Optimizes point-return scale, while the strategy consumes ranks. | Simple finite-mask behavior. | Misaligned with top-k/rank-drop portfolio construction; likely overvalues noisy magnitude. |
| `ic` / `ICLoss` | Good conservative default. Optimizes daily cross-sectional Pearson ordering signal. | Same-date, PIT-mask-compatible, stable, already used by frozen recipe. | Full cross-section metric, not top-k/cost aware; Pearson not Spearman. |
| `combined` / `CombinedMSEICLoss` | Mixed. Adds point estimation back into a rank-driven task. | Backward-compatible bridge. | MSE term can pull away from portfolio rank quality. |
| `portfolio_ic` / `PortfolioICLoss` | Direct top-k utility proxy plus IC anchor. | Most top-k-adjacent current criterion; no chronological state needed. | Prior sweep context showed lower turnover/cost drag but weaker aggregate return/ASR than pure IC; not cost-aware inside training. |
| `lambdarank_ic` / `LambdaRankICLoss` | Strong full-order rank alignment. Optimizes a RankIC-inspired pairwise ordering surrogate. | Same-date, PIT-mask-compatible, `val_rank_ic` selection supported, no model contract change. | Sensitive to pair cap; 512-pair tranche under-sampled the PIT cross-section severely. |

The trainer already validates both Pearson IC and Rank IC, and supports checkpoint selection by `val_loss`, `val_ic`, or `val_rank_ic`. This means the immediate gap is experiment design and objective alignment, not missing trainer plumbing.

## Local Research Map Synthesis

Files inspected:

- `docs/research/current/LOSS_PATH_DECISION_2026-06-04.md`
- `docs/research/current/LOSS_PATH_EXPERIMENTAL_SEARCH_2026-06-04.md`
- `docs/research/current/LOSS_FUNCTION_LITERATURE_SCAN_2026-06-03.md`
- `docs/research-paper-evaluations/2026-06-04-lambdarankic.md`
- `docs/research-paper-evaluations/2026-05-27-machine-learning-meets-markowitz-loss-functions-addendum.md`
- `docs/handoffs/2026-06-03-portfolio-ic-no-rank-gate-worst-seed.md`

Two requested June 19 files were not present in this worktree:

- `docs/research/current/MCI_GRU_PROGRAM_MAP_2026-06-19.md`
- `docs/research/current/MCI_GRU_RESEARCH_OPPORTUNITY_SCAN_2026-06-19.md`

Memory indicates those June 19 files existed in a prior checkout and named ListNet-style rank loss, execution-cost/capacity replay, tradability overlays, uncertainty ranking, and saved-prediction selection audits as high-priority opportunities. Treat that as memory-derived and potentially branch-stale until verified in a checkout that contains the files.

The research-map direction is consistent:

- Conservative next path: LambdaRankIC-style pairwise Rank IC, disabled by default.
- More adventurous next path: uncertainty-adjusted ranking with a distributional alpha head.
- Later-stage path: decision-focused optimizer, Sharpe, Sortino, drawdown, turnover, and cost-aware objectives after chronological training infrastructure exists.

## Recommended Candidate Paths

### 1. LambdaRankIC cap sweep - first priority

Why: It is already implemented and tested, directly targets Rank IC, preserves PIT masks, and fits the current `(pred, target)` loss contract. The 512-pair result is not enough evidence because it saw less than 1% of normal same-day pairs and had poor gradient agreement with the all-pairs surrogate.

Cheap experiment:

- Same tranche shape as the latest full tranche where possible.
- Keep pure IC and Portfolio-IC controls fixed.
- Test LambdaRankIC with `lambdarank_ic_max_pairs_per_day` at 4096 and 8192.
- Use `selection_metric=val_rank_ic`.
- Keep the frozen recipe otherwise unchanged: raw 5-day labels, PIT masked panel, same graph/features, same seeds.

Promotion signal:

- Rank IC improves materially versus pure IC, and net top-k/rank-drop backtest does not worsen turnover, drawdown, or year stability.

Kill criteria:

- At 4096/8192, mean best val Rank IC still trails pure IC by more than roughly 10%-15% and net portfolio metrics remain worse.
- Runtime grows enough that 8192 cannot finish comparable smoke/tranche runs without offsetting validation gains.
- Improvements show only in validation Rank IC but not in top-k net return, drawdown, or turnover diagnostics.

### 2. Portfolio-alignment diagnostic - second priority, and needed for every loss

Why: The Markowitz addendum and Portfolio-IC handoffs both point to the same failure mode: similar signal metrics can map to different economic outcomes. Before adding another objective, the project needs a replay-level report that tells whether the miss is full-order rank, cutoff/top-k concentration, churn/cost, or regime drawdown.

Cheap experiment:

- Use saved predictions from pure IC, Portfolio-IC, and LambdaRankIC runs.
- Report by year/seed: Pearson IC, Spearman Rank IC, ICIR/Rank ICIR, top-k gross return, net return after spread/slippage, turnover, rank-drop exits, Newey-West Sharpe, max drawdown.
- Add synthetic tests where Rank IC improves but top-k worsens, and where top-k improves while full-order Rank IC is flat.

Promotion signal:

- The diagnostic identifies one dominant mismatch, such as "Rank IC gains fail near the top-k cutoff" or "gross signal is fine but rank-drop/cost churn destroys net return."

Kill criteria:

- If the report cannot separate signal quality from portfolio rule effects, do not use it to choose a loss.
- If no saved-prediction set has durable provenance, rerun a smaller controlled grid rather than mixing artifact lineages.

### 3. Uncertainty-adjusted ranking replay - third priority, no model change first

Why: The June 4 experimental search argues that uncertainty-aware ranking is the best middle path if the current-output constraint is relaxed. It aligns with portfolio construction by selecting robust names, but can be tested first without changing model architecture.

Cheap experiment:

- Compute uncertainty proxies from existing outputs: ensemble prediction dispersion, cross-seed score instability, or trailing residual volatility.
- Replay scores like `score = pred_mean - lambda_uncertainty * uncertainty_proxy`.
- Sweep a small set of lambda values and compare net return, turnover, drawdown, Sharpe, and year stability.

Promotion signal:

- Net performance or drawdown/turnover stability improves without a large loss of gross return or Rank IC.

Kill criteria:

- Uncertainty adjustment merely suppresses scores and reduces turnover while worsening risk-adjusted returns.
- Gains appear only in the short 2025 partial window and fail in full-year 2022/2024 stress windows.

## Paths To Defer

Listwise Rank IC / differentiable Spearman / NeuralSort / SoftRank:

- Worth keeping as a backup if LambdaRankIC higher caps fail.
- Do not add a compiled differentiable-sorting dependency until the cheaper pairwise path is fairly tested.
- If pursued, prefer a tiny local soft-rank prototype and compare gradients/ordering on synthetic panels before wiring into `TrainingConfig`.

Direct soft long-short or top-bottom decile return:

- More aligned with long-short research portfolios than the current long-only top-k/rank-drop stack.
- Could be useful as a diagnostic loss, but first prove whether the strategy wants full-order rank quality or cutoff utility.

Sharpe, Sortino, drawdown, turnover, transaction-cost-aware objectives:

- Conceptually aligned but not valid in the current shuffled per-date criterion.
- Requires a separate chronological trainer or episode sampler, previous holdings, split-boundary reset rules, and a cost model matching `tests/backtest_sp500_daily.py`.

Decision-focused MVO / SPO / optimizer layers:

- High upside, but high infrastructure risk.
- Needs covariance/risk modeling, constraints, solver or implicit gradient, and safeguards against prediction inflation and excessive turnover.

## Suggested Next Commands

Targeted local verification for current loss plumbing:

```powershell
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m pytest tests/test_lambdarank_ic_loss.py tests/test_lambdarank_ic_config.py tests/test_lambdarank_ic_trainer.py tests/test_portfolio_ic_loss.py tests/test_portfolio_ic_config.py tests/test_portfolio_ic_trainer.py -v --basetemp .tmp_pytest\pytest
```

LambdaRankIC local benchmark sanity at higher caps:

```powershell
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe scripts/benchmark_lambdarank_ic_loss.py --batch-size 4 --n-stocks 512 --max-pairs 4096 --reps 20 --warmup 5 --device cpu
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe scripts/benchmark_lambdarank_ic_loss.py --batch-size 4 --n-stocks 512 --max-pairs 8192 --reps 20 --warmup 5 --device cpu
```

If moving to Colab, use the existing LambdaRankIC notebook generator/contract and set pair caps to 4096 and 8192 before any promotion claim.

## Verification Status

Commands run during this investigation:

```powershell
git status --short --branch
git rev-parse HEAD
git branch --contains HEAD
git show-ref --heads codex/colab-gpu-utilization-hardening-20260620
Test-Path -LiteralPath 'C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe'; Test-Path -LiteralPath '.\.venv\Scripts\python.exe'
Get-Content -LiteralPath 'AGENTS.md'
rg -n "LambdaRankIC|portfolio_ic|LOSS_PATH|MCI_GRU_PROGRAM_MAP|loss function|Rank IC|rank" 'C:\Users\magil\.codex\memories\MEMORY.md'
rg -n "loss|portfolio|rank|IC|Spearman|Sharpe|turnover|drawdown|selection_metric|daily_ic|training" docs/ARCHITECTURE.md
Get-Content -LiteralPath 'mci_gru\training\losses.py'
Get-Content -LiteralPath 'mci_gru\training\trainer.py'
rg -n "loss_type|selection_metric|portfolio_ic|lambdarank|rank_ic|val_rank_ic|build_training_loss|criterion|calculate_metrics|best_val" mci_gru\config.py mci_gru\training\metrics.py run_experiment.py mci_gru\walkforward.py
Get-Content -LiteralPath 'docs\research-paper-evaluations\2026-06-04-lambdarankic.md'
rg --files | rg "LOSS_PATH|portfolio-ic|lambdarank|rank|loss-function|loss_path|handoffs"
rg --files docs\research docs\handoffs docs\research-paper-evaluations
Get-Content -LiteralPath 'docs\research\current\LOSS_PATH_DECISION_2026-06-04.md'
Get-Content -LiteralPath 'docs\research\current\LOSS_PATH_EXPERIMENTAL_SEARCH_2026-06-04.md'
Get-Content -LiteralPath 'docs\handoffs\2026-06-03-portfolio-ic-no-rank-gate-worst-seed.md'
Get-Content -LiteralPath 'docs\research-paper-evaluations\2026-05-27-machine-learning-meets-markowitz-loss-functions-addendum.md'
Get-Content -LiteralPath 'mci_gru\evaluation\portfolio.py'
Get-Content -LiteralPath 'mci_gru\evaluation\statistics.py'
Get-Content -LiteralPath 'mci_gru\training\metrics.py'
Test-Path -LiteralPath 'docs\handoffs\2026-06-21-lambdarankic-pair-cap-investigation.md'
rg -n "pair cap|pairs512|4096|8192|LambdaRankIC|mean_best_val|gradient cosine|full tranche|pair" docs scripts tests notebooks mci_gru -g "*.md" -g "*.py"
Get-Content -LiteralPath 'tests\test_lambdarank_ic_loss.py'
Get-Content -LiteralPath 'tests\test_portfolio_ic_loss.py'
rg -n "loss_type|selection_metric|portfolio_ic|lambdarank_ic|num_models|num_epochs|early_stopping|label_type" configs docs\DEFAULT_EXPERIMENT_RECIPE.md docs\CONFIGURATION_GUIDE.md
```

No production code was changed by this investigation. This handoff is a docs-only addition.

Targeted verification after writing this note:

```powershell
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m pytest tests/test_lambdarank_ic_loss.py tests/test_lambdarank_ic_config.py tests/test_lambdarank_ic_trainer.py tests/test_portfolio_ic_loss.py tests/test_portfolio_ic_config.py tests/test_portfolio_ic_trainer.py -v --basetemp .tmp_pytest\pytest
```

Result: first run collected 38 items and reached 32 passed / 1 skipped before 5 trainer-test setup errors because `.tmp_pytest\pytest`'s parent directory did not exist. After creating `.tmp_pytest`, the same command passed with `37 passed, 1 skipped, 3 warnings in 4.96s`.
