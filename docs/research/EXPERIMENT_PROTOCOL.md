# Experiment Protocol

The rules every MCI-GRU experiment runs under. The design and its seven decisions
are PRD #369, approved by the owner on 2026-10-28; this file is the working
summary, and the code named here enforces it. Where the two disagree, the code
and #369 win.

## Every run has a standing

`run_experiment.py` refuses a run unless it is one of these:

- **Pre-registered.** `protocol.experiment_id=<id> protocol.arm=<arm>
  protocol.window=<W1..W5>` names a committed
  `docs/research/experiments/<id>.yaml` (below). The run must start from a git
  checkout whose tracked files all match its commit, with none marked
  assume-unchanged or skip-worktree; untracked files such as data pulls are
  fine. The runner reads the file and the recipe from that commit, and refuses
  the file if it is missing, untracked or edited. It composes only from the
  checkout's `configs/`, so `--config-path`, `--config-dir` and
  `hydra.searchpath` elsewhere are refused. It also refuses a run that is not
  exactly the declared arm on the declared window. `run_metadata.json` records
  the id, arm, window, the file's path, SHA-256 and commit, the recipe slug,
  and the SHA-256 of the recipe's override lines.
- **Exploratory.** `protocol.exploratory=true`. Smoke runs, debugging, a
  W5-only run, and any run whose design was not written down first all go here.
  The run is stamped `exploratory` and never gets a verdict (decision 6).

The gate is `mci_gru/preregistration.py` (`resolve_run_standing`, #372).

## Windows

The model trains on four calendar years, validates on the next and tests on the
year after. Each window is its own full run with the same recipe; only the six
`data.*` dates change. The schedule is `WINDOWS` in `mci_gru/protocol.py`, and
the table is in [`../CONFIGURATION_GUIDE.md`](../CONFIGURATION_GUIDE.md).

| Window | Train | Validate | Test |
|---|---|---|---|
| W1 | 2016 to 2019 | 2020 | 2021 |
| W2 | 2017 to 2020 | 2021 | 2022 |
| W3 | 2018 to 2021 | 2022 | 2023 |
| W4 | 2019 to 2022 | 2023 | 2024 |
| W5 (the recipe) | 2020 to 2023 | 2024 | 2025 |

A verdict needs all five windows (decision 2). So a pre-registration declares
all five.

## The holdout

The 2026 signal dates from 2026-01-22 to 2026-07-24 are locked. No run, rescore,
backtest or saved-prediction score may read them or let a label reach into them.
Everything after them is the next holdout (#370; `require_outside_holdout` in
`mci_gru/protocol.py`). The only way in will be a committed promotion record,
one look per candidate and three looks in total (decision 3, #374). Until #374
lands, nothing unlocks it.

## Seeds

- The seed base is 1729, and ensemble member i uses 1729 + i.
- Every arm and every window in an experiment shares the base, and the runner
  checks it.
- The seed-floor arm is the one arm whose change is `seed=<another base>`. The
  paired difference it gives is the noise floor the other arms are read against,
  measured again for each new recipe id.
- Rerunning on another base after seeing results is a new trial, never a
  replacement.

## Metric and decision rule

- **Primary metric.** The mean daily IC of the arm minus the control, pooled
  over the W1 to W5 test dates, with a Hansen-Hodrick t on `label_t - 2` lags.
  The per-window differences are reported beside it. The metric is fixed: IC
  (decision 5).
- **Adopt** when the Holm-adjusted two-sided p is below alpha (0.05), the
  difference is positive, and it exceeds the seed floor (decision 4).
- **Reject** on the same test with a negative difference.
- **Otherwise inconclusive**, and the default stays.
- Rank IC, top-50 excess, tranche Sharpe, ties and per-window signs are
  reported but never decisive.

A pre-registration may make the rule **stricter** but never looser:

- a smaller `alpha`;
- `bonferroni` in place of `holm`;
- a `seed_floor_multiple` above 1.

The loader refuses anything looser.

## Trial ledger

Every launched run of an experiment gets a ledger row, including failed and
aborted runs. `compare` gives no verdict unless the ledger holds exactly the
declared runs (#373).

## Pre-registration

Commit `docs/research/experiments/<id>.yaml` before the experiment's first run.
The file's name is its id: lower-case letters, digits and hyphens. Every field
is required and no other field is allowed. The control arm lists no changes.
Every other arm lists exactly one Hydra override, and no two arms make the same
change (`++key=value` and `key=value` count as the same). An arm may not change
the window's dates, `training.walkforward.*`, `hydra.*`, or a launcher key such
as `experiment_name`, `output_dir`, `tracking.*` or `protocol.*`. An arm whose
change composes to the same config as the control, such as one setting a value
the recipe already sets, is refused when it runs.

The runner composes the declared run from these parts, in order:

1. the recipe's override block, at the run's commit;
2. `seed=<seed_base>`;
3. the arm's one change;
4. the window's six dates.

It then compares the result with the run's resolved config. Any key that
differs is refused, except the launcher keys above and the auxiliary-snapshot
location keys. `recipe` must equal the recipe slug at the run's commit, so a
recipe change between windows needs a new pre-registration.

Template:

```yaml
id: graph-zeroed-control
family: graph
hypothesis: >-
  The correlation graph adds IC over the same model with its edges suppressed.
recipe: static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1__latents-data__xsec-residual__gru-32-10__gat-residual__inputs-relative__eval-fp32__train-4y
control: recipe
arms:
  recipe: []
  graph_zeroed: ["+experiment=graph_zeroed"]
seed_base: 1729
windows: [W1, W2, W3, W4, W5]
primary_metric: ic
alpha: 0.05
decision_rule:
  adjustment: holm
  seed_floor_multiple: 1.0
```

A run of its treatment arm on the third window:

```text
python run_experiment.py <the recipe's overrides> +experiment=graph_zeroed \
    <W3's six data.* dates> \
    protocol.experiment_id=graph-zeroed-control protocol.arm=graph_zeroed protocol.window=W3
```

Until #371 gives the launcher a `--window` flag, the window dates are passed by
hand. `tests/test_preregistration.py` checks that this template loads and that a
run built this way is accepted.
