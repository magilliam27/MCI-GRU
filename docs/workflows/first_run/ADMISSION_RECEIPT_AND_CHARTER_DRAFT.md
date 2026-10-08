# First admitted run: admission receipt and run charter (DRAFT)

> **Draft for the owner's review. Not posted on #187, and not an admission.**
> Prepared 2026-10-04 by a Claude Code session working while the owner was away
> (ticket #278). Admission stays on **HOLD** (#187, comment 5862836271) until the
> owner edits this, signs the release in section 3, and posts it on #187.
> Ticket states below were read from GitHub on 2026-10-04 around 06:25 UTC, and
> updated on 2026-10-08 for #279, #275, #280 and the move to EODHD stock prices.

The run this governs is the first admitted real-data run of the frozen recipe
(`docs/DEFAULT_EXPERIMENT_RECIPE.md`) on the 110-name universe with EODHD stock
prices (owner decision, 2026-10-08), launched from `notebooks/first_run_colab.ipynb`. Runbook:
[`RUNBOOK.md`](RUNBOOK.md).

## 1. Receipt items

Each row is an item the run depends on, the work that delivers it, and its state.

| Item | Ticket | Pull requests | State, 2026-10-08 | What still closes it |
| --- | --- | --- | --- | --- |
| Run input attachment: each window keeps the package manifest and binds every read to it | #208 | #270 | **Merged** at `6d02778` | Done |
| Capture of the regime inputs, kept in the run folder; EODHD file declared as a required input role | #208 | #274 | **Merged** at `1dcd818` | Done |
| Two EODHD record gaps: a rule stop leaves the file read recorded as used, and the `eodhd.regime_market` verdict carries no `observation_id` | #208 | #279 | **Merged** at `24d1d2a` (gaps logged on #208, comment 5976533692) | Done |
| Execution provenance: code, config, environment and member facts captured at execution time | #144 | #261, #265, #266 merged | Done; issue open for the owner to close (status comment 5972582665) | Owner closes the issue |
| Interpreter and lock policy | #143 | #263 merged | Done; issue open for the owner to close (evidence comment 5975806936) | Owner closes the issue |
| Recipe pins the fixed model forms: data-dependent latents (#198), residual cross-stock block (#197), 32-then-10 `gru_attn` layers (#131); edge channels kept as they are | #273 | #275 | **Merged** at `f2beea0` | Done |
| S&P 500 regime input read fully from EODHD (`GSPC.INDX`), no FRED splice | #276 | #277 | **Merged** at `1a0a17b`; #276 closed | Done |
| Raw-input validation and `run_failure.json` | #223 | #268 merged | First-run rules in force; issue stays open for the deferred index, sector and exporter rules | Nothing for this run |
| Historical availability of the regime inputs | #224 | #272 merged (with #269, the back-fill removal) | Closed | Done |
| Dated PIT eligibility and fixed-session labels | #225 | #267 merged | Closed | Done |
| Never-finite selection metric fails the run; walk-forward means skip NaN windows with coverage | #252, #253 | #260 merged | Closed | Done |
| Retrieval and two-copy preservation | #207 | none | Stock package: two new copies, each restored on its own 10/10 (comment 5964915801). The rest is **deferred until after the run** by the owner (2026-10-04 02:33 UTC) | Nothing for this run; see limitation 2.6 |
| Colab launcher, EODHD staging in the regime notebooks, this draft | #278 | #280 | **Merged** at `46889bd` | Done |
| Stock prices from EODHD: the same 110-name membership, priced from an EODHD pull | #281 | #282 | Draft. The r1 package published on 2026-10-08 is replaced by r2, which drops carried zero-volume final rows on delisted codes; r2's manifest is not published yet | r2 published and pinned in the data config; owner merges |
| Recipe selects the EODHD data config (`data=gics_top10_110_2016_eodhd`) | #283 | #284 | Draft, lands after #282. Its delisting check found a carried zero-volume final row for `ATVI.OQ^J23`, `HES.N^G25` and `WBA.OQ^H25` on r1 | The check is clean on r2; owner merges |
| The notebook stages the package the recipe's data config declares, and no other | #285 | #286 | Draft | r2's entry added once published; owner merges |

Merged: #270 at `6d02778`, #274 at `1dcd818`, #279 at `24d1d2a`, #275 at `f2beea0`
and #280 at `46889bd`. Still open before the run: #282, #284 and #286.

## 2. The six receipt sections of #187 (comment 5862836271)

### 2.1 Exact campaign inputs

| Input | Identity | Where the run gets it |
| --- | --- | --- |
| Stock package `sp500_pit_gics_top10_mcap_monthly_20160104_20260731`, EODHD prices, r2 | Manifest path and SHA-256: _to fill when r2 is published_. Its point-in-time membership file is byte-identical to the LSEG package's. Pre-merger DuPont (`DD.N^I17`) has no EODHD price history and is declared absent in `data.pit_absent_kdcodes` (#282) | Drive, the r2 package folder under `MCI_GRU_shared/preservation/` (_to fill_). The notebook stages the package the recipe's data config declares, requires the copy of the manifest at the folder's top to have the pinned bytes, and runs `validate_input_package` on the staged files |
| EODHD S&P 500 index (`GSPC.INDX`), pull of 2026-09-19 | `data/raw/market/eodhd_sp500_2010_20260919/sp500_index.csv`, 351,815 bytes, SHA-256 `9f5cfaf5e9f057065b83619b01568164e1f61a45df9453c66269fb54f080f5ac` (matches that pull's `inventory.json`) | Drive `MCI_GRU_shared/preservation/eodhd_sp500_2010_20260919/`, beside the package folder. The notebook stops on any other bytes |
| Five FRED regime series: `DGS10`, `DGS3MO`, `DCOILWTICO`, `VIXCLS`, `PCOPPUSDM` | Values as FRED serves them on the run date; revisions unchecked (agenda answer 3) | Requested live with the owner's key from Colab secrets, captured byte-exact into `input_snapshots/` in the run folder (#274) |

No basename fallback for required selected files (#223), and no provider or symbol
substitution (#187 ruling of 2026-09-21).

The LSEG package (manifest
`data/manifests/sp500_pit_gics_top10_mcap_monthly_20160104_20260731.r1.json`,
SHA-256 `1e2043dcb4f00a88de128985cc94423c122e480034c16ec741710a2460653ecf`, Drive
`MCI_GRU_shared/preservation/2026-10-02-110-universe-2016`) stays the base default
in `configs/config.yaml`, and the notebook stages it when the recipe names
`data=gics_top10_110_2016`. It is not this run's input. The EODHD r1 package is
replaced by r2 and must not be staged; the notebook carries no pin for it.

### 2.2 Admission findings

The run records its own verdicts. These must be present and clean in the run
folder before the owner signs section 3:

- `admission.json` for the window, with every role admitted: the market panel and
  PIT file (#223), the six regime roles `fred.yield_10y`, `fred.yield_3m`,
  `fred.regime_oil`, `fred.regime_volatility`, `fred.regime_copper` and
  `eodhd.regime_market` (#224, #276), and PIT eligibility and label coverage (#225).
- No `run_failure.json`.
- The rules in force are the owner's answers 1 to 14 in the project's admission
  decision agenda, as recorded on #223 (comment 5971665014), #224 (comment
  5971662000) and #225 (comment 5971662968), with answer 7 superseded by the
  EODHD decision of 2026-10-04 03:17 UTC.

### 2.3 Restoration

The stock package has two preserved copies, each restored on its own into an
empty destination with a 10/10 hash and size match (#207, comment 5964915801). The
notebook's staging is a third, independent byte check against the Drive copy at
run time. The regime inputs have one copy, the run's own `input_snapshots/`,
copied to Drive by the notebook. A second copy and independent restore of those
is #207 work the owner deferred until after the run.

### 2.4 Run evidence

From the run folder, copied whole to
`MyDrive/MCI_GRU_shared/runs/first_run/<run tag>/`:

- `colab_run_record.json`: commit, mode, prerequisite check, staged input hashes,
  the exact overrides, start and end, exit code; `pip_freeze.txt` beside it.
- The execution plan, execution start and run receipt (#144), and per-window
  `run_metadata.json` with `input_attachment` status (#208). The attachment
  should read `complete`; in `source` mode it would read `incomplete`.
- `input_snapshots/`: what FRED returned and the index file the run read.
- Checkpoints, predictions and the training log.

### 2.5 Comparable outcomes

Member and window failures stay visible with their denominators (#252, #253 via
#260). This run becomes the baseline for its recipe slug. A later EODHD-only
recipe gets a new recipe id and is compared against this one (owner decision,
2026-10-03 17:09 UTC, amended 2026-10-04).

### 2.6 Known limitations to carry in the release

1. **No cessation event file is declared** (`data.pit_cessation_events_csv` is
   null). Under answer 9, a cessation with no dated evidence rejects the run;
   with no file, none are applied. This is an open owner question from the
   first-run config thread.
2. FRED values are today's values standing in for history (answer 3).
3. The EODHD file is a fixed vintage ending 2026-09-18. The recipe's test window
   ends 2025-12-31, so it is covered.
4. Regime inputs have a single preserved copy (2.3).
5. Colab trains on a GPU build of `torch==2.12.1` (a `+cu…` local version of the
   pin); the #143 qualification is the Linux CPU reference. GPU training is not
   bit-reproducible run to run. The CUDA packages torch pulls in are not in the
   lock; the run folder's `pip_freeze.txt` records them.
6. #223's deferred index, sector and exporter rules are not in force; the recipe
   uses none of those inputs.
7. MLflow tracking is off for the run (`tracking.enabled=false`), because the lock
   does not carry the `tracking` extra.
8. **Pre-merger DuPont is absent** (`DD.N^I17`, owner decision 2026-10-08 02:32
   UTC). EODHD has no price history for it, and no LSEG prices are mixed in, so
   from January 2016 to August 2017 the panel scores at most 109 names (session
   floor 104). The data config declares the gap in `data.pit_absent_kdcodes`.

## 3. Run charter

| Field | Value |
| --- | --- |
| Purpose | First admitted real-data run of the frozen recipe; becomes the baseline for its recipe slug |
| Recipe | `docs/DEFAULT_EXPERIMENT_RECIPE.md` at the run commit. Since #275 the slug is `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1__latents-data__xsec-residual__gru-32-10` |
| Data | `data=gics_top10_110_2016_eodhd`: EODHD prices on the same PIT membership file and windows as `gics_top10_110_2016`; PIT masked panel, 104-name session floor, train 2016-01-04 to 2023-12-31, validation 2024-01-22 to 2024-12-31, test 2025-01-22 to 2025-12-31 |
| Model and training | `gru_attn`, 20 models x 100 epochs, patience 15, pure IC loss on raw 5-day returns, selection on validation IC |
| Inputs | Section 2.1, captured (`data.auxiliary_snapshot_mode=capture`) |
| Compute | Google Colab GPU runtime, `notebooks/first_run_colab.ipynb` in `full` mode |
| Commit | _to fill: the `main` commit after the last of #282, #284 and #286 merges; set as `EXPECTED_COMMIT`_ |
| Run tag and Drive folder | _to fill from `colab_run_record.json`_ |
| Success | Exit code 0; no `run_failure.json`; every role admitted in `admission.json`; input attachment `complete`; 20 member records |
| Stop rules | Any `run_failure.json` stops the run and is reported as it stands, without editing inputs. A lost runtime leaves an incomplete run: start again from the top under a new tag, never resume into the same folder |
| What it may claim | Only results of this slug on the 2025 test window, with the limitations in 2.6. No claim against runs of other slugs or the pre-2026-10-04 recipe |
| Not authorised by this charter | Other recipes, other universes, vendor pulls, data edits, further GPU campaigns |

**Release** (owner, after the run):

- [ ] Section 1 items merged or accepted as limitations
- [ ] Section 2.2 verdicts read in the run folder
- [ ] Run tag, commit and Drive folder filled in above
- [ ] Posted on #187 by the owner on: ____

---
_Drafted by Claude Code for the owner's review._
