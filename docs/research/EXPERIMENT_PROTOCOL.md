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
  The run is stamped `exploratory` and never gets a verdict (decision 6). It may
  also name `protocol.window=<W1..W5>`, and is then refused unless its six dates
  are that window's; the stamp records the window (#371).

The gate is `mci_gru/preregistration.py` (`resolve_run_standing`, #372). The
one run that is neither is a holdout promotion, below, which has its own gate.

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
`mci_gru/protocol.py`). The only way in is a committed promotion record: one look
per candidate, three looks in total, and then the holdout retires and the next one
is the data after 2026-07-31 (decision 3, #374).

### Promoting a candidate

A promotion scores an arm that a declared experiment adopted against that
experiment's control, on the holdout window: train 2021 to 2024, validate 2025,
test 2026-01-22 to 2026-07-24 (`HOLDOUT_WINDOW`). Both arms run on the
experiment's seed base, from the code the verdict ran.

1. **Write the record.** `docs/research/holdout/<id>.yaml`, committed before
   either run, on a branch whose only change since the verdict's commit is under
   `docs/research/holdout/`:

   Promotion record:

   ```yaml
   id: graph-zeroed-2026
   experiment_id: graph-zeroed-control
   candidate: graph_zeroed
   incumbent: recipe
   commit: 0123456789abcdef0123456789abcdef01234567
   pass_rule: holdout_paired_ic_difference_positive
   ```

   `commit` is the full id of the commit the experiment's verdict ran at.
   `incumbent` must be the experiment's control and `candidate` one of its
   treatment arms (not the seed-floor arm). The pass rule is fixed: the
   candidate's holdout IC minus the incumbent's, paired by date, above zero. It
   is reported with its HAC interval, which decides nothing.
2. **Run both arms.** `first_run_colab.py run --mode full --promotion <id> --arm
   <candidate> --verdict-runs-root <experiment runs>`, then the same with `--arm
   <incumbent>`. The launcher first recomputes the experiment's verdict from its
   runs and refuses unless it ran at the record's commit and adopted the
   candidate (`--seed-floor` as for the verdict). The runner takes
   `protocol.holdout_promotion=<id> protocol.arm=<arm>` and nothing else of
   `protocol.*`. Its gate (`resolve_promotion_standing` in
   `mci_gru/promotion.py`) refuses unless:
   - the record exists and, like every tracked file, matches the checkout's
     commit;
   - the committed `docs/research/holdout/ledger.yaml` extends the ledger on
     `origin/main` (so copy main's ledger onto the promotion branch), holds
     neither the id nor the same candidate of the same experiment, and holds
     fewer than three looks;
   - the checkout carries no other record the ledger has not recorded: one
     look at a time;
   - the verdict's commit is an ancestor of the checkout, and the two differ only
     under `docs/research/holdout/` (a file moved in from elsewhere counts at its
     old path);
   - the run is exactly the named arm, at the seed base, on the holdout window's
     six dates (and with any other dates the config itself refuses it);
   - its labels close by 2026-07-31.

   `run_metadata.json` records the promotion id, the record's SHA-256, the
   commit, the verdict's commit and how many looks the ledger held.
3. **Write the receipt.** `first_run_colab.py compare --promotion <id>
   --runs-root <folder> --out-dir <dir>` reads every run in the folder that names
   the promotion, as the verdict does. It writes `promotion_receipt.json` and
   `ledger_entry.yaml`. When the pair is complete, the receipt holds the paired
   test and `passed`. When a run scored the holdout but the runs give no scored
   result (a scored run that crashed, a pair scored twice, a stamp that does not
   match), the receipt is `void` and `passed` is false: the look is spent. When
   nothing scored, nothing looked, and no receipt is written, so the pair may run
   again.
4. **Record the look.** In one commit, add the receipt as
   `docs/research/holdout/<id>.receipt.json` and the entry (id, experiment,
   candidate, commit, the record's and the receipt's SHA-256, passed) under
   `looks:` in the ledger, and merge it to main. A look is never removed or
   edited.

The rescore, replay, backtest and saved-prediction paths keep refusing the
holdout, a promotion run included: the receipt is the score.

What the gate cannot see:

- A look not yet merged to main. Two promotion branches run before either
  ledger entry lands would each pass. The receipt refuses more than one scored
  run of each arm, and the ledger merge is what closes the id, so merge it as
  soon as the receipt is written.
- Code other than `run_experiment.py`. `ExperimentConfig` accepts the holdout
  window whenever `protocol.holdout_promotion` is set; the record, the ledger and
  the verdict commit are checked by the runner's gate only. A script that builds
  its own config with that key set reads the holdout unchecked.
- The ledger's digests are checked for form only, not against the committed
  record and receipt.

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

The ledger is read from a folder of runs, not from a list of runs, so leaving one
out means moving its folder away. A run belongs to the experiment when its
launcher record (`colab_run_record.json`) or its runner stamp names the
experiment. Each row gets one of these statuses:

- `COMPLETE`: a clean exit, with a protocol stamp and test scores.
- `FAILED`: a non-zero exit, or a clean exit without scores.
- `ABORTED`: no exit code recorded, because the launcher never finished.
- `INVALID`: the folder cannot be trusted. That covers no launcher record, a
  record and stamp that disagree, a stamp that says exploratory, more than one
  run in one folder, or a record that cannot be read.

A run counts as **scored** once anything computed on the test set is in its
folder. That includes one member's test predictions (`predictions_model_*`),
which are written long before the summary.

The verdict is refused when any of these hold:

- a row is `INVALID`;
- a row is an undeclared arm or window;
- a run that scored the test set did not complete;
- the complete runs do not hold each declared arm and window exactly once
  (`validate_trial_family`).

A launch that failed or was aborted before scoring stays in the ledger, and its
arm and window may be launched again. A second scored run of the same pair is a
second look and refuses the verdict.

## Verdict

`scripts/first_run_colab.py compare --experiment <id> --runs-root <folder>`
writes `trial_ledger.csv` and `trial_ledger.jsonl` first, so the ledger exists
even when the verdict is refused, then `verdict.json` if the ledger allows one.
It reads the pre-registration at the commit the complete runs ran (they must
share one). Every run's stamp must carry that file's SHA-256, the recipe slug
and the SHA-256 of the recipe's override lines. All runs must use mode `full`,
one staged package and one label horizon.

The verdict is computed in these steps:

- **Each arm's paired difference.** For every arm except the control, the
  arm's daily IC is joined with the control's over all five windows. The paired
  difference is tested with the HAC t on `label_t - 2` lags.
- **The seed floor.** It is the absolute pooled difference of the seed arm (the
  one arm whose change is `seed=...`). Member i trains on `seed + i`, so the
  seed arm's base must be at least `training.num_models` away from `seed_base`;
  otherwise the arm shares member seeds with the control, and both the launch
  and the verdict are refused. An experiment without a seed arm names another
  declared experiment and its runs with `--seed-floor <id>=<runs folder>`. That
  experiment's verdict is computed from its own complete ledger, never read from
  a file. It must have measured the floor with its own seed arm, on the same
  recipe slug, override lines, label horizon and windows. A verdict with
  neither is refused.
- **The adjustment.** The treatment arms (every arm but the control and the seed
  arm) are adjusted together, with Holm or with the declared Bonferroni.
- **The decision.** Each treatment arm is adopted, rejected or inconclusive by
  the rule above (`mci_gru/evaluation/verdict.py`).

`verdict.json` records these numbers:

- the attempts by status;
- each run's IC, and whether its captured FRED inputs match the control run's on
  that window;
- the rule as declared;
- the seed floor and the test that measured it;
- for each treatment arm: its difference, t, raw and adjusted p, whether it
  cleared alpha and the floor, and its verdict.

Rank IC and the per-window differences sit under `reported_not_decisive`.

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
recipe: static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1__latents-data__xsec-residual__gru-32-10__gat-residual__inputs-relative__eval-fp32__train-4y__graph-quarterly-0p5
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

The same run through the launcher is
`scripts/first_run_colab.py run --mode full --experiment graph-zeroed-control
--arm graph_zeroed --window W3` (#373). It composes exactly these parts, adds
only where the run writes and the `protocol.*` stamp, and records the experiment
id and the file's SHA-256 so the ledger counts the launch. Without
`--experiment`, `run --window` builds an exploratory run (#371).
`tests/test_preregistration.py` checks that this template loads and that a run
built this way is accepted.

## Pooled comparison

`scripts/first_run_colab.py compare` takes one run folder per arm and window. Per
arm it joins the windows' dated IC into one series, so the paired difference is
pooled over every window's test dates, and it reports each window's difference
beside it. It refuses arms with different window sets, a window given twice, and a
run that scored dates outside its window's test span (#371). That comparison is
exploratory and gives no verdict. A declared experiment's verdict comes from
`compare --experiment` (above).
