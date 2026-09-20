# MCI-GRU

Cross-sectional stock ranking on a point-in-time S&P 500 universe.

The model is the four-stream network from *MCI-GRU: Stock Prediction Model Based on
Multi-Head Cross-Attention and Improved GRU* (Neurocomputing 2025). The repository is
what grew around it: a point-in-time data layer, a frozen experiment recipe, an evidence
ledger, and a tracker on which every default is either a measured choice or labelled as
an inherited one.

[![CI](https://github.com/magilliam27/MCI-GRU/actions/workflows/ci.yml/badge.svg)](https://github.com/magilliam27/MCI-GRU/actions/workflows/ci.yml)
Python 3.10+ · PyTorch 2 · PyTorch Geometric · Hydra

## What it does

Each trading day, every stock in the universe is scored from its own recent history, from
its neighbours on a correlation graph, and from two sets of learned market latents. Twenty
independently seeded models are trained and their scores averaged. The daily
cross-sectional correlation between the ensemble's scores and 5-day forward returns is the
arbiter for every comparison in this repository; portfolio replays are reported alongside
it, never in its place.

| Stream | What it sees | Module |
| --- | --- | --- |
| A1, temporal | `his_t` days of per-stock features through an attention-gated GRU, on a fast path and a strided-convolution slow path | `mci_gru/models/temporal.py` |
| A2, cross-sectional | the day's features for all stocks, through two graph-attention layers over a trailing-return correlation graph | `mci_gru/models/graph.py` |
| B1, B2, market latents | A1 and A2 re-expressed by cross-attention against 32 learned state vectors | `mci_gru/models/latent.py` |
| Head | `[A1, A2, B1, B2]`, cross-stock self-attention, a final graph-attention layer, one score per stock | `mci_gru/models/trunk.py` |

The paper covers the streams. Most of the work here is in what it does not cover.

**A universe that changes under the model.** Membership is an interval table
(`kdcode, valid_from, valid_to`), not a constituent list. The panel keeps a fixed union of
every name that was ever a member and carries four daily masks, so a name accumulates
lookback history before it joins, becomes tradable on its first day of membership, and
disappears after it leaves. Losses, exported predictions and backtests only see names that
were members on that day. Normalization statistics, correlation graphs and labels are fit
on the training window and nothing later; tests fail if that changes.

**A frozen recipe.** Confirmation runs use one recipe verbatim
([`docs/DEFAULT_EXPERIMENT_RECIPE.md`](docs/DEFAULT_EXPERIMENT_RECIPE.md)): 20 models,
100 epochs, patience 15, pure-IC loss on raw 5-day return labels, validation-IC checkpoint
selection, a static 0.8-threshold correlation graph with four-channel edge features. An
experiment varies one factor against it, so a difference can be attributed.

**An evidence ledger.** Every report under [`docs/research/`](docs/research/README.md) is
current or superseded. Every figure in a current report is `[Verified]`, read from a run
artifact, or `[Inferred]`, reasoning on top of one. A run folder carries a manifest with
the SHA-256 of its config, data, graph and checkpoints, and a selection audit applies a
multiple-testing haircut before any signal claim is made. CI refuses a dated report that
lands outside the lifecycle.

## What the evidence says

As of 2026-09-02. Each item links the report it is read from; the caveats are the report's.

- **The correlation graph does not beat an empty one.** Five graph specifications ran at
  the frozen recipe on the 110-name point-in-time universe, including a control with every
  correlation edge removed. The control ranked first on the arbiter at both the screening
  and confirmation stages. Paired on the same 238 test days, every specification's mean
  daily difference from the control is negative and none is distinguishable from zero
  (BHY-adjusted p = 1.00 for all four), and four test years could not separate these arms
  at the 2025 effect sizes.
  [`GRAPH_SPECIFICATION_ABLATION_2026-09-01`](docs/research/current/GRAPH_SPECIFICATION_ABLATION_2026-09-01.md)
  · [`GRAPH_PAIRED_REANALYSIS_2026-09-02`](docs/research/current/GRAPH_PAIRED_REANALYSIS_2026-09-02.md)
- **The point-in-time panel runs end to end at full breadth.** Rolling yearly trainings
  for 2022 through 2025 on the roughly 700-name point-in-time union completed with every
  breadth check passed and every prediction file matching the same-day scoreable mask.
  Point-in-time daily top-K replays show strong positive excess return in 2023, 2024 and
  2025 and a weak 2022; compounded across the four windows the model returns about 92%
  against 20% for the benchmark, with no transaction costs, no rank gate, and no single
  year significant on its own.
  [`PIT_MASKED_PANEL_2022_2025_FULL_RUN_REPORT_2026-05-16`](docs/research/current/PIT_MASKED_PANEL_2022_2025_FULL_RUN_REPORT_2026-05-16.md)
- **The reduced universe is promising and not uniformly robust.** On the 110-name GICS
  top-10 universe extended back to 2016, out-of-sample test IC was +0.056 in 2023 and
  +0.039 in 2024, with 30 and 23 points of excess return over the benchmark, and −0.058
  in 2022 with 22 points of underperformance, again with no costs and no gate.
  [`SP500_PIT_GICS_TOP10_MULTIYEAR_BASELINE_2026-06-23`](docs/research/current/SP500_PIT_GICS_TOP10_MULTIYEAR_BASELINE_2026-06-23.md)

![Left: excess return over the benchmark by test year on the 110-name universe, beside the roughly 700-name panel's compounded return against the benchmark. Right: each graph specification's mean daily IC difference from the graph-zeroed control, with intervals that all straddle zero.](docs/assets/readme_evidence.svg)

*Left: percentage points of excess return over the benchmark by test year on the 110-name point-in-time universe, a PIT daily top-10, no transaction costs, no rank-drop gate ([report](docs/research/current/SP500_PIT_GICS_TOP10_MULTIYEAR_BASELINE_2026-06-23.md)); beside it, the roughly 700-name panel's return compounded across its four yearly test windows against the benchmark's ([report](docs/research/current/PIT_MASKED_PANEL_2022_2025_FULL_RUN_REPORT_2026-05-16.md)). Transaction costs were disabled in the reviewed backtest artifact. Rank-drop gating was disabled. Year-by-year statistical significance is absent. Right: each graph specification's mean daily IC difference from the graph-zeroed control on the same 238 test days with its 95% block-bootstrap interval: every arm's mean daily difference is negative, none is distinguishable from zero, and the BHY-adjusted p-value is 1.00 for all four comparisons ([report](docs/research/current/GRAPH_PAIRED_REANALYSIS_2026-09-02.md)). Every number, label, and caveat is read from those reports ([`docs/assets/readme_evidence.json`](docs/assets/readme_evidence.json)); `python scripts/gen_readme_figure.py` regenerates the figure.*

None of this is a deployable performance claim, and the reports say so. The next decision
on the graph is a human-gated ticket on the graph-specification map
([#157](https://github.com/magilliam27/MCI-GRU/issues/157)); the 2022 failure has its own
diagnostic ticket ([#30](https://github.com/magilliam27/MCI-GRU/issues/30)).

## Run it

The market panel is LSEG-licensed and is not in the repository. Everything else is.

```bash
git clone https://github.com/magilliam27/MCI-GRU.git && cd MCI-GRU
pip install -e ".[dev,fred]"
python scripts/ci_smoke.py   # a synthetic four-stock panel through the whole pipeline; what CI runs
```

With a panel of your own (`kdcode, dt, open, high, low, close, volume`):

```bash
python run_experiment.py data.filename=/path/to/panel.csv data.use_pit_universe=false \
    training.num_epochs=2 training.num_models=1 tracking.enabled=false
```

Confirmation-scale runs are generated as Colab notebooks by the `scripts/gen_*_nb.py`
generators; [`docs/NOTEBOOK_BEST_PRACTICES.md`](docs/NOTEBOOK_BEST_PRACTICES.md) describes
the contract they follow.

## Working conventions

- Work is planned on the tracker. A map issue names a destination and its child tickets;
  each ticket says whether it resolves through live exchange with the maintainer or can be
  taken unattended, and declares the paths it owns before a branch exists.
- Every pull request is a draft until the maintainer merges it, and carries its owned
  paths, a review on two axes (the repository's standards, the ticket's spec), and a
  mutation table: each load-bearing test is broken on purpose and must fail.
- A green test is not evidence, a handoff is not evidence, and a notebook contract test is
  not a live run. Research claims come from `docs/research/current/` only.
- When a document disagrees with the code, the code wins;
  [`docs/agents/domain.md`](docs/agents/domain.md) gives the order.

[`CONTRIBUTING.md`](CONTRIBUTING.md) has the full set, including how tickets, branches and
labels are shaped.

## Where to look

| | |
| --- | --- |
| [`docs/index.md`](docs/index.md) | the documentation map |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | data flow, batch contract, model, graph, evaluation surfaces |
| [`docs/DEFAULT_EXPERIMENT_RECIPE.md`](docs/DEFAULT_EXPERIMENT_RECIPE.md) | the frozen recipe, as Hydra overrides |
| [`docs/FEATURES.md`](docs/FEATURES.md) | every feature column, its window, and its no-lookahead rule |
| [`docs/CONFIGURATION_GUIDE.md`](docs/CONFIGURATION_GUIDE.md) | Hydra groups, presets, data sources, override syntax |
| [`docs/research/README.md`](docs/research/README.md) | which reports are current and which are superseded |
| [`docs/evaluation/EVIDENCE_HARNESS.md`](docs/evaluation/EVIDENCE_HARNESS.md) | run manifests, trial ledgers, selection audits, capacity replay |
| [`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md) · [`docs/TEST_REGISTRY.md`](docs/TEST_REGISTRY.md) | the verification ladder, and the generated registry of every test |
| [`AGENTS.md`](AGENTS.md) | the entrypoint for anyone working in the repository |

## References

1. *MCI-GRU: Stock Prediction Model Based on Multi-Head Cross-Attention and Improved GRU*,
   Neurocomputing 2025. [arXiv:2410.20679](https://arxiv.org/abs/2410.20679).
2. Goulding, Harvey, Mazzoleni, *Momentum Turning Points*.
   [SSRN 3489539](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3489539). Basis for
   the slow/fast momentum signals, the Bull/Correction/Bear cycle states, and the dynamic
   speed selection.
3. Veličković et al., *Graph Attention Networks*, ICLR 2018.
   [arXiv:1710.10903](https://arxiv.org/abs/1710.10903).
