# Graph-specification ablation: the graph-zeroed control is not beaten

**Date:** 2026-09-01
**Ticket:** magilliam27/MCI-GRU issue #167 (`wayfinder:task`, AFK with maintainer-supervised compute), child of map #157
**Protocol:** issue #164 resolution, executed through the issue #166 harness (`notebooks/graph_specification_ablation_colab.ipynb` at commit `46ea7b0`)
**Status:** Current research evidence. Live Colab execution with Drive-backed artifacts; no code change, no config change, no recipe change.
**Prior result reconciled:** `docs/ABLATION_NOTEBOOK_RESULTS_REPORT_2026-04-30.md`, on `top_k=20` specifically (section 8).

Every figure is labelled `[Verified]` (read from a run artifact on Drive, or reproduced from
one) or `[Inferred]` (reasoning on top of those figures). Section 10 is entirely
interpretation and is labelled as such. Nothing from the mechanics smoke appears in any
table in this document; see section 3.

---

## 1. Summary

**The control arm first.** `[Verified]` At the frozen recipe — 20 models × 100 epochs,
patience 15, 1000 bootstrap resamples, seed 1729 — **A0, the graph-zeroed control, ranks
first of five arms on the decided arbiter**, test-span pooled daily IC:

| rank | arm | avg_ic | 95% CI |
|---:|---|---:|---|
| 1 | **A0_zeroed** — empty correlation edges, GAT intact | **0.04238** | [−0.01426, 0.09699] |
| 2 | A1_shipped — threshold 0.8, the incumbent | 0.04191 | [−0.01575, 0.09580] |
| 3 | A3_topk20 — top-K 20 (corr), static | 0.03857 | [−0.01279, 0.08601] |
| 4 | A4_sector_only — exact sector relation, no correlation edges | 0.03765 | [−0.01503, 0.08771] |
| 5 | A2_thr05 — threshold 0.5 | 0.03479 | [−0.01485, 0.08147] |

`[Verified]` **No arm separates from the control, and no arm separates from zero.** Every
95% interval contains zero. The largest gap to the control, A0 − A2 = 0.0076, is about 14%
of A0's own half-width. A0 − A1 = 0.0005.

`[Verified]` A0 also ranked first on the arbiter at the screen stage (3 seeds × 20 epochs),
and the harness's pre-registered promotion rule promoted nothing beyond the always-confirmed
pair, because no screen CI was disjoint from A0's.

`[Verified]` The hard hygiene invariant held in every arm at both stages:
`twin_edge_count == 0` for GOOG.OQ/GOOGL.OQ.

`[Inferred]` Against the outcome map pre-registered on ticket 164 — *"nothing beats A0 —
the graph leaves the recipe; only A4 beats A0 — replace estimation with the exact sector map;
A2 or A3 beats A0 and A4 — re-specify on the winning rule"* — the arbiter lands this result
in the **first branch**. Whether to act on that branch is decision `D1`, which is human-gated
and outside this map; this report does not take it.

**Eight further findings**, all `[Verified]` unless marked:

1. **All five arms ran at both stages.** Nothing was dropped for budget. The conditional
   disambiguation arm (neighbours vs channels) was pre-registered to run only if a top-K arm
   won the screen; none did, so it was correctly not run.
2. **The confirm stage was widened from the two promoted arms to all five**, at the
   maintainer's direction, after the screen numbers were visible. This is a disclosed
   deviation from the ticket-166 default and is discussed in section 3.
3. **From screen to confirm the arms converged.** The spread of `avg_ic` across arms
   narrowed from 0.0130 to 0.0076; the two arms with no correlation edges (A0, A4) fell,
   the two with populated correlation graphs (A2, A3) rose.
4. **The alongside metrics order the arms differently from the arbiter, and differently
   from each other.** Sharpe of the top-20 portfolio puts A0 fourth and the incumbent A1
   last; Sharpe of the top-10 portfolio puts A0 first; median rank IC puts A0 last and A3
   first. Section 7 tabulates this. None of it moves the arbiter, which was decided before
   any number was seen; all of it is why a new grilling ticket is warranted before `D1`.
5. **No portfolio return separates from zero for any arm at any K** in {10, 20, 50, 100}.
6. **April's composite score, computed alongside, still ranks top-K 20 last** — so April's
   anti-top-K conclusion survives on April's own metric while the arbiter puts A3 third and
   within 0.0038 of the control. Section 8.
7. **Density and isolation are identical between stages** (the graph does not depend on the
   training budget) and reproduce the picture from the density child: the incumbent leaves
   75–78% of admissible names isolated in every span; A4's sector graph isolates none.
8. **The manifest's `promotion_rule` prose is stale** relative to what ran (section 3).

---

## 2. What ran, and what did not

`[Verified]` Five arms, two stages, one run root on Drive:
`MyDrive/MCI-GRU-Ablations/graph_specification_ablation/20260901_015032/`.

| arm | configuration (Hydra overrides on top of the frozen recipe) | screen | confirm |
|---|---|---|---|
| A0_zeroed | `+experiment=graph_zeroed` — `graph.zero_edges=true`; edge tensor forced empty, GAT and parameters intact, self-loops only | ran, rc 0 | ran, rc 0 |
| A1_shipped | `graph.judge_value=0.8 graph.top_k=0` — the shipped default | ran, rc 0 | ran, rc 0 |
| A2_thr05 | `+experiment=graph_thr05` — `judge_value=0.5` | ran, rc 0 | ran, rc 0 |
| A3_topk20 | `+experiment=graph_topk20_static` — `top_k=20 top_k_metric=corr` | ran, rc 0 | ran, rc 0 |
| A4_sector_only | `+experiment=graph_sector_only` — `zero_edges=true use_sector_relation=true`, sector map derived from the universe metadata export | ran, rc 0 | ran, rc 0 |

`[Verified]` Held fixed across every arm and stage, exactly as ticket 164 ruled:
`graph.corr_lookback_days=252 graph.update_frequency_months=0
graph.use_multi_feature_edges=true graph.append_snapshot_age_days=false
graph.use_lead_lag_features=false graph.drop_edge_p=0.1`, and the twin exclusion
`+graph.exclude_edge_pairs=[["GOOG.OQ","GOOGL.OQ"]]` applied at candidate level to every
arm including the sector path. Recipe side: `data=gics_top10_110_2016 features=with_momentum
training.loss_type=ic training.label_type=returns training.selection_metric=val_ic
training.shuffle_train=true model.label_t=5 training.learning_rate=5e-5
training.lr_scheduler=cosine`, strict current-only global regime features.

`[Verified]` Budgets: screen `training.num_models=3 training.num_epochs=20`; confirm
`training.num_models=20 training.num_epochs=100`; both `early_stopping_patience=15
evaluation.bootstrap_resamples=1000 seed=1729`. Every confirm run reports
`models_trained: 20`.

**Not run, and why:**

- `[Verified]` The **disambiguation arm** (top-K neighbours with the shipped edge channels
  vs top-K channels) was pre-registered on ticket 164 to run *only if a top-K arm won the
  screen*. A3 did not separate from A0 (screen CI [−0.0061, 0.0730] vs A0's
  [−0.0102, 0.1003]), so the arm was not triggered. This is the protocol working, not an
  omission.
- Nothing else was planned, and nothing planned was dropped.

---

## 3. Protocol as executed, and every deviation from it

### 3.1 Staging `[Verified]`

The harness stages three files from a flat `MyDrive/MCI_GRU_shared/data/`. The `20260731`
vintage existed on Drive only inside a preservation bundle one directory deeper, so the
committed notebook could not have run unattended (recorded on ticket 167 before any runtime
was allocated). At the maintainer's direction the three files were **copied server-side** into
the flat location — the preservation bundle was left untouched. The notebook's own SHA-256 of
each staged file matched the bundle's `MANIFEST.txt`, so the copies are byte-identical:

| role | bytes | sha256 |
|---|---:|---|
| market panel | 40,035,626 | `d64c4d041ef4c1632ed76e1456885ffe8301a477c8e27c4a73805f94ff97aeb4` |
| PIT universe | 16,058 | `15721bb3c8d17a16901c02d352799cfa7b38bbb65b6f5a7e362c7d6424703200` |
| sector map (A4) | 5,034,067 | `31673ec69e5306e6d923623dc857c5be9dda805eabd7643648dbaacca90cec90` |

The sector export reproduces every figure `configs/data/gics_top10_110_2016.yaml` records
for it: 206 universe names, 545 validity intervals, 128 monthly snapshots, 11 GICS sectors,
206 of 206 names mapped, 3706 directed intra-sector edges.

### 3.2 Mechanics smoke `[Verified]` — excluded from evidence

A `SMOKE_MODE = True` pass (1 model × 2 epochs × 5 arms, 25 bootstrap resamples) ran first
under run tag `20260901_014022` and completed every cell, including the twin assertion. It
proved staging, Hydra composition of all five arms, artifact wiring, and that the harness
refuses to write a promotion record from a smoke (`Promotion record not written (stage:
screen | smoke: True)`). **Its metric outputs are not evidence and are not reproduced
anywhere in this report.** Anyone reading Drive should treat that run tag as a wiring proof.

### 3.3 Screen `[Verified]`

`RUN_STAGE="screen"`, `SMOKE_MODE=False`, fresh run tag `20260901_015032`. Section 9 of the
notebook wrote `graph_specification_ablation_promotion.json` with
`PROMOTED_ARMS: ["A0_zeroed", "A1_shipped"]` — the always-confirmed pair only — and, for each
other arm, `separated_from_A0: false`.

### 3.4 Confirm — the widening `[Verified]`, its reasons `[Inferred]`

The ticket-166 default confirms A0, A1, and any arm separating from A0 on the screen; that
resolved to A0 and A1. A two-arm confirm was started (`RUN_TAG_OVERRIDE="20260901_015032"`
so screen and confirm share a run root) and **interrupted after about 16 minutes**, with
A0 part-way through and nothing completed. Its partial directory,
`training/confirm/graphspec_confirm_A0_zeroed_seed1729/20260901_022112/`, exists on Drive,
holds no `evaluation_summary.json`, and is referenced by nothing.

The maintainer then directed that **all five arms be confirmed**, set via
`CONFIRM_ARMS_OVERRIDE=['A0_zeroed','A1_shipped','A2_thr05','A3_topk20','A4_sector_only']`.
Two reasons, the second the stronger:

- the screen was shallow and every CI overlapped, so the promotion rule had no
  discriminating power to exercise — "not promoted" carried no information;
- confirming only A0 and A1 would have set a 20 × 100 control against 3 × 20 graph arms in
  the headline comparison — a comparison across training budgets, which is the class of
  confound this map exists to remove.

This decision was taken **after the screen numbers were visible**. It is recorded here as a
deviation for that reason. It is sanctioned rather than a quiet substitution because ticket 164
explicitly labelled the screening and promotion defaults *"left to ticket 166, overridable"*,
and because it widens coverage rather than selecting a favourable subset. **The arbiter was
not touched.**

### 3.5 Manifest prose is stale `[Verified]`

`graph_specification_ablation_manifest_confirm.json` carries a hard-coded `promotion_rule`
string — *"A0 and A1 always confirmed … additionally any arm whose test-span pooled daily IC
95% bootstrap CI does not overlap A0's"* — that no longer describes how the confirm arm set was
chosen. The manifest's `arms` and `jobs` keys record the true five, so the record is not
false, but a reader taking the prose field at face value would be misled. The screen manifest's
string is accurate.

### 3.6 Runtime disconnect after completion `[Verified]`

All five confirm trainings and every downstream cell completed; the last analysis artifact was
written at 04:19:30 UTC. The Colab runtime then idled out. Subsequent `Run all` attempts on a
reconnected runtime failed at the Drive mount in the setup cell and therefore **wrote nothing**
under the run root (the notebook creates its run directories only in section 4, which those
attempts never reached). The on-page outputs were cleared by those attempts, which is why the
results below are read from the Drive artifacts rather than from the notebook display.

### 3.7 Not a deviation, but recorded

The GPU the "G4" runtime type resolved to was `NVIDIA RTX PRO 6000 Blackwell Server Edition`,
accepted by the notebook's gate via its `RTX PRO` / `BLACKWELL` allow-markers. `T4` was the
runtime's default and is blocked by the gate; it was switched before any cell ran.

---

## 4. The arbiter: test-span pooled daily IC with 95% bootstrap CI

The arbiter is read from each run's `evaluation_summary.json` (`avg_ic`, `avg_ic_ci_lower`,
`avg_ic_ci_upper`), exactly as notebook section 6 reads it. Test span 2025-01-22 … 2025-12-31.

### 4.1 Confirm — 20 models × 100 epochs `[Verified]`

| arm | avg_ic | CI lower | CI upper | IC IR | mean best val IC |
|---|---:|---:|---:|---:|---:|
| A0_zeroed | 0.042384 | −0.014258 | 0.096988 | 0.1714 | 0.037686 |
| A1_shipped | 0.041912 | −0.015748 | 0.095797 | 0.1660 | 0.036182 |
| A3_topk20 | 0.038573 | −0.012790 | 0.086014 | 0.1716 | 0.035610 |
| A4_sector_only | 0.037645 | −0.015033 | 0.087708 | 0.1647 | 0.037440 |
| A2_thr05 | 0.034792 | −0.014848 | 0.081470 | 0.1463 | 0.033916 |

Gaps to the control: A1 −0.0005, A3 −0.0038, A4 −0.0047, A2 −0.0076.

### 4.2 Screen — 3 seeds × 20 epochs `[Verified]`

| arm | avg_ic | CI lower | CI upper | mean best val IC |
|---|---:|---:|---:|---:|
| A0_zeroed | 0.046388 | −0.010158 | 0.100253 | 0.040108 |
| A4_sector_only | 0.045491 | −0.009062 | 0.097234 | 0.040779 |
| A1_shipped | 0.043005 | −0.015653 | 0.099234 | 0.039014 |
| A3_topk20 | 0.035859 | −0.006061 | 0.072995 | 0.034144 |
| A2_thr05 | 0.033401 | −0.004655 | 0.071404 | 0.034509 |

### 4.3 Movement from screen to confirm `[Verified]`

| arm | Δ avg_ic (confirm − screen) |
|---|---:|
| A0_zeroed | −0.0040 |
| A1_shipped | −0.0011 |
| A2_thr05 | +0.0014 |
| A3_topk20 | +0.0027 |
| A4_sector_only | −0.0078 |

`[Inferred]` The two arms carrying no correlation edges lost ground with more training; the
two with populated correlation graphs gained. The across-arm spread narrowed from 0.0130 to
0.0076. Whether that is convergence toward a common ceiling or the graph arms needing more
epochs to use their edges is not distinguishable from two budgets; it is a question, not a
finding.

---

## 5. Per-year disclosure `[Verified]`

Notebook section 8 recomputes the daily IC series from each run's `averaged_predictions/`
against panel-derived 5-session forward returns and bootstraps a CI per calendar year. The
test span is a single calendar year, so this is one row per arm, not a table across years.
It is an independent recomputation, so it differs slightly from section 4.

**Confirm**, 2025, 238 days:

| arm | pooled daily IC | CI lower | CI upper |
|---|---:|---:|---:|
| A0_zeroed | 0.046589 | −0.007856 | 0.105519 |
| A1_shipped | 0.045834 | −0.011886 | 0.104820 |
| A3_topk20 | 0.042165 | −0.007005 | 0.095462 |
| A4_sector_only | 0.041852 | −0.004288 | 0.097079 |
| A2_thr05 | 0.038840 | −0.011174 | 0.094179 |

Same ordering as the arbiter. All lower bounds negative.

**Screen**, 2025, 238 days: A0 0.051191 [−0.003130, 0.110020]; A4 0.051278
[−0.000232, 0.107930]; A1 0.047548 [−0.010587, 0.109339]; A3 0.039292 [0.000889, 0.081612];
A2 0.036649 [0.000206, 0.079023]. At the screen the two densest arms had lower bounds just
above zero; at confirm they do not.

---

## 6. Density, isolation, and the twin check `[Verified]`

Measured by notebook section 7 from each run's frozen `graph_data.pt` against the PIT
admissible mask per span, on 201 names. Identical at screen and confirm, as expected — the
graph is built from the panel, not from training.

| arm | twin edges | train density | train isolated | val density | val isolated | test density | test isolated |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0_zeroed | 0 | 0.000000 | 100.0% | 0.000000 | 100.0% | 0.000000 | 100.0% |
| A1_shipped | 0 | 0.005720 | 75.5% | 0.004581 | 77.1% | 0.004537 | 78.5% |
| A2_thr05 | 0 | 0.206253 | 20.9% | 0.149368 | 31.4% | 0.140912 | 34.2% |
| A3_topk20 | 0 | 0.121581 | 19.6% | 0.089999 | 30.3% | 0.084089 | 33.2% |
| A4_sector_only (corr channel) | 0 | 0.000000 | 100.0% | 0.000000 | 100.0% | 0.000000 | 100.0% |
| A4_sector_only (sector channel) | 0 | 0.082402 | 0.0% | 0.082402 | 0.0% | 0.082402 | 0.0% |

`Twin check passed: twin_edge_count == 0 in every arm.` at both stages.

`[Inferred]` A1's 75–78% isolation reproduces the density child's finding (74.0% / 88.9% /
77.2% on its own axis and date grid) that the shipped graph leaves most tradable names with
no neighbours. The notebook's arbiter therefore compares an empty graph (A0) against a
near-empty one (A1) against three populated ones — which is the comparison ticket 164 asked
for, and the closeness of A0 and A1 on every metric in this report is the measured form of
the worry that ticket recorded.

---

## 7. Alongside, and explicitly not the arbiter

Ticket 164 ruled the arbiter before any number was seen and named April's composite as
"computed alongside for reconciliation, not the arbiter." Everything in this section is in
that category. It is reported because ticket 167 requires agreement and disagreement both to
be said, and because the disagreements here are material.

### 7.1 April composite `[Verified]`

Weights `avg_ic 0.35, avg_spearman_corr 0.25, sharpe_top_20_newey_west 0.25,
return_top_20 0.15`, each z-scored across the five arms.

| arm | confirm | screen |
|---|---:|---:|
| A0_zeroed | 0.7920 | 0.7635 |
| A4_sector_only | 0.0724 | 0.4755 |
| A1_shipped | −0.0153 | 0.0766 |
| A2_thr05 | −0.4114 | −1.1798 |
| A3_topk20 | −0.4378 | −0.1359 |

A0 first at both stages; A3 last at confirm.

### 7.2 Portfolio Sharpe and return by K (confirm, Newey–West) `[Verified]`

| arm | Sharpe top-10 | Sharpe top-20 | Sharpe top-50 | Sharpe top-100 |
|---|---:|---:|---:|---:|
| A0_zeroed | **1.876** | 1.595 | 1.339 | 1.050 |
| A1_shipped | 1.800 | **1.439** | 1.502 | 1.100 |
| A2_thr05 | 1.552 | 1.652 | 1.338 | 1.052 |
| A3_topk20 | 1.845 | 1.612 | 1.286 | **1.135** |
| A4_sector_only | 1.624 | **1.712** | **1.639** | 1.082 |

| arm | return top-10 | return top-20 | return top-50 | return top-100 |
|---|---:|---:|---:|---:|
| A0_zeroed | 0.006900 | **0.005008** | 0.002912 | 0.001891 |
| A1_shipped | 0.006755 | 0.004509 | **0.003313** | 0.001953 |
| A2_thr05 | 0.006145 | 0.004861 | 0.003106 | 0.001885 |
| A3_topk20 | **0.006953** | 0.004634 | 0.002743 | **0.002005** |
| A4_sector_only | 0.005938 | 0.004799 | 0.003279 | 0.001936 |

`[Verified]` **Every top-K return CI lower bound is negative for every arm at every K.** No
portfolio return separates from zero.

`[Verified]` The Sharpe ordering **depends on K**. At top-10 the control leads; at top-20 it
is fourth and the incumbent is last; at top-100 the control is last. The Sharpe ordering is
also **not stable between stages**: at the screen, top-20 Sharpe ran A3 1.790 > A4 1.618 >
A0 1.573 > A1 1.430 > A2 1.342.

### 7.3 Rank IC, mean against median (confirm) `[Verified]`

| arm | mean rank IC | median rank IC | hit rate | long–short spread |
|---|---:|---:|---:|---:|
| A0_zeroed | 0.04182 | **0.02678** | 0.4968 | 0.003913 |
| A1_shipped | **0.04212** | 0.03888 | **0.5115** | 0.003844 |
| A2_thr05 | 0.03352 | 0.03253 | 0.4980 | 0.003200 |
| A3_topk20 | **0.02975** | **0.04031** | 0.4898 | 0.002545 |
| A4_sector_only | 0.03493 | 0.03786 | 0.4660 | 0.003200 |

`[Verified]` On **mean** rank IC A0 is second and A3 last; on **median** rank IC **A0 is last
and A3 is first**. The control's mean is carried by a minority of strong days; its typical day
is the weakest of the five.

`[Verified]` Raw prediction scale differs by more than an order of magnitude across arms —
test MSE: A2 0.031, A3 0.066, A1 0.113, A0 0.191, A4 0.741. IC is scale-invariant so the
arbiter is unaffected; the scale itself is a property of the graph stream worth knowing.

### 7.4 What this section does and does not do `[Inferred]`

It does not change the answer. The arbiter was chosen before the numbers, and choosing a
different one now because it likes a different arm is exactly what ticket 167 forbids. It
does establish that "the control wins" is **metric-dependent in a structured way** — the
control is strongest where the ranking is sharpest (top of the book, mean IC) and weakest
where it is broadest (top-100, median day). Per ticket 167, that is a new grilling ticket,
not a substitution, and it should be opened before `D1` is taken.

---

## 8. Reconciliation with the April 2026 report on `top_k=20`

`docs/ABLATION_NOTEBOOK_RESULTS_REPORT_2026-04-30.md` reported, for `top_k=20`, a mean
decision score of **−1.3704**, avg IC **0.0314**, and top-20 return CI lower **−0.0024**,
called it *"materially worse than threshold graphs"*, and ruled *"do not promote top-K graph
variants yet."* Those runs used `sp500_2019_universe_data_through_2026.csv` — a different
panel and universe, before the PIT-admissibility fix — at 20 × 100. Ticket 164 ruled that a
measurement of a different configuration and directed a re-run under the corrected protocol.

**Where April and this run agree** `[Verified]`:

- On April's own composite, top-K 20 is **last of five** here (−0.4378) as it was worst
  there. The composite's verdict on top-K survives the protocol change.
- Top-K 20's top-20 return CI lower bound is negative here too (−0.0014).
- Top-K 20 has the **lowest mean rank IC** of the five arms (0.0298).

**Where they disagree** `[Verified]`:

- On the arbiter, A3 is **third**, above both A4 and the recalibrated threshold A2, and
  0.0038 below the control — well inside every interval. April's "materially worse than
  threshold" does not reproduce on IC: A3 beats the 0.5 threshold and trails the 0.8
  threshold by 0.0033.
- A3's avg IC here (0.0386) exceeds April's top-K figure (0.0314) and April's *static
  threshold baseline* (0.0361).
- A3 has the **highest median rank IC** and the highest IC IR (tied with A0) of the five.

`[Inferred]` The reconciliation is therefore partial, and it is metric-shaped rather than
arm-shaped: April's composite still dislikes top-K 20, the decided arbiter does not single it
out, and the two disagree for the reasons section 7 tabulates. Ticket 164's grounds for
treating April as a different configuration (PIT-blind adjacency, contamination scaling with
K) stand; this run neither confirms nor overturns them directly, since it did not re-run the
April configuration.

---

## 9. Provenance

| | |
|---|---|
| Repository commit | `46ea7b0ec24ac38ef52bd4e1bc7b40dc804e8c74` — `origin/main` at the time the notebook cloned `main`. `[Verified]` that `origin/main` was this commit both before the run began and after it completed; `main` is never force-pushed in this repository, so the clone can only have been this commit. |
| Notebook | `notebooks/graph_specification_ablation_colab.ipynb`, opened from GitHub `main`; only `RUN_STAGE`, `SMOKE_MODE`, `RUN_TAG_OVERRIDE`, and `CONFIRM_ARMS_OVERRIDE` were edited in-session, as sections 3.3–3.4 describe. No cell body was changed. |
| Run root (Drive) | `MyDrive/MCI-GRU-Ablations/graph_specification_ablation/20260901_015032/` (folder id `1GBTfB8k1EmAs3z5KVf3Tsku5iyigYe5n`) |
| Runtime | Google Colab, runtime type "G4 GPU, High-RAM"; `nvidia-smi` reported `NVIDIA RTX PRO 6000 Blackwell Server Edition` |
| Interpreter | `/usr/bin/python3` (Python 3.13, from the `dist-packages` path in the setup traceback), `torch 2.11.0+cu128`, `torch.cuda.is_available() == True` |
| Panel | `sp500_pit_gics_top10_mcap_monthly_20160104_20260731_lseg_20150101_20260731.csv`, sha256 `d64c4d04…97aeb4`, 40,035,626 bytes (full digests in section 3.1) |
| PIT universe | `…_pit_universe.csv`, sha256 `15721bb3…703200`, 16,058 bytes |
| Sector map | `…_all_metadata_snapshots.csv`, sha256 `31673ec6…0cec90`, 5,034,067 bytes |
| `resolved_config.json` (A0 confirm) | 5,617 bytes, sha256 `f77e56e6388745c33f82bfbc6cb9353f4af8a3002611f08b39f2c0aff4b25857`. Key fields: `graph.zero_edges=true`, `graph.judge_value=0.8`, `graph.top_k=0`, `graph.exclude_edge_pairs=[["GOOG.OQ","GOOGL.OQ"]]`, `training.num_models=20`, `training.num_epochs=100`, `training.early_stopping_patience=15`, `evaluation.bootstrap_resamples=1000`, `evaluation.bootstrap_seed=1729`, `evaluation.ci_level=0.95`, `evaluation.sharpe_method=newey_west`, `evaluation.block_size=null`, `seed=1729`. |
| `run_metadata.json` | Present in every run directory (A0 confirm: 426,194 bytes). Not read in full for this report; the commit above is established from the `origin/main` observations, not from this file. |
| Per-run timing (confirm) | A0 1126.9 s, A1 1088.5 s, A2 1229.5 s, A3 1163.8 s, A4 1428.0 s elapsed; ≈78–80 s of each is `prepare_data`, the rest model training and prediction export. Total ≈ 100.6 min. |
| Manifests | `graph_specification_ablation_manifest_screen.json`, `…_manifest_confirm.json` (see 3.5 for the stale prose field) |
| Promotion record | `graph_specification_ablation_promotion.json` (screen stage only, by design) |
| Confirm run directories | `training/confirm/graphspec_confirm_{A0_zeroed,A1_shipped,A2_thr05,A3_topk20,A4_sector_only}_seed1729/20260901_{023833,025724,031536,033610,035537}/` |
| Screen run directories | `training/screen/graphspec_screen_*_seed1729/20260901_{015036,015353,015713,020046,020443}/` |
| Aggregate artifacts | `graph_specification_ablation_{results,density_disclosure,per_year_ic}_{screen,confirm}.csv`, `graph_specification_ablation_summary_{screen,confirm}.md`, `gpu_util_{screen,confirm}.csv` |
| Discarded | `training/confirm/graphspec_confirm_A0_zeroed_seed1729/20260901_022112/` — the interrupted two-arm attempt; no `evaluation_summary.json`; referenced by nothing |
| Smoke | run tag `20260901_014022` — wiring proof only; not evidence |

`[Verified]` `evaluation.block_size` is `null` in the frozen config, and `null` resolves to
`max(1, label_t) = 5` at `mci_gru/evaluation/metrics.py:68` and
`mci_gru/evaluation/experiment_summary.py:107`; `mci_gru/evaluation/statistics.py:367` refuses
any block shorter than the label horizon. The arbiter's CI is therefore a 5-session block
bootstrap over days, as the notebook's prose states. Checked at commit `46ea7b0`, the commit
the run used.

Gitignored `results/`, `outputs/`, `*.pth`, and `*.pt` were not used as a source for any figure
here; every number is read from the Drive artifacts above or reproduced from them.

---

## 10. Interpretation — entirely `[Inferred]`

**What the arbiter says.** The graph, as specified by any of the four non-empty arms, does not
earn its edges on test-span pooled daily IC at the frozen recipe. That is the measurement this
map has lacked since it was charted, and it is now made. The result is not "the graph hurts" —
no arm separates in either direction — it is "the graph does not measurably help on the
decided metric, at this budget, on this panel, in this single test year."

**What it does not say.** It does not say the graph is inert. Section 7 shows the arms are
distinguishable on other statistics in a way that has structure: the empty-graph control is
best at the very top of the ranking and on mean IC, and worst on the typical day and in broad
portfolios. Two families of explanation fit, and this run cannot separate them:

1. *The metric is doing the choosing.* Pooled daily IC averaged over days rewards an arm with a
   few very strong days; median rank IC and top-100 Sharpe reward consistency. If the
   control's advantage is concentrated in a handful of days, "A0 wins" is a statement about
   the arbiter's sensitivity to tail days. The CI already respects the 5-session label
   overlap (section 9), so this is about the point estimate, not the interval: a mean over
   238 days can be moved by a few of them in a way a median cannot.
2. *The graph is changing what the model predicts, not how well.* The order-of-magnitude
   spread in raw MSE across arms says the GAT stream's output scale depends strongly on
   density. IC ignores scale; portfolio construction does not entirely. An arm can rank
   slightly worse yet size its top book better.

Neither explanation is established. Both are testable without new training, from the
`averaged_predictions/` already on Drive.

**On `D1`.** The pre-registered outcome map routes this result to "the graph leaves the
recipe." That routing was written for the arbiter, and the arbiter delivered it. The
alongside evidence does not override the routing — it was declared non-arbiter before the
run — but it is strong enough that taking `D1` on the arbiter alone would be deciding while
looking away from section 7. The correct next step under this repository's rules is a
grilling ticket on the metric–pipeline tension, opened before `D1`, so that `D1` is taken
knowing whether the IC verdict and the Sharpe/median verdicts are measuring the same thing.

**On the incumbent.** A1 tracks A0 within 0.0005 on the arbiter, within 0.001 on mean rank
IC, and within 0.02 on hit rate, while carrying a graph that isolates three quarters of the
names. On this evidence the shipped default is an expensive way of computing the control.

---

## 11. Rules this run was held to

- Draft pull request only; nothing pushed to `main`; no branch, ref, worktree, stash, or
  Drive file deleted. The preservation bundle on Drive is untouched.
- `C:\Users\magil\MCI-GRU` fingerprinted at session start and end: `codex/paper_trade_scrape`
  @ `e286649`, 42 dirty entries, unchanged.
- No production code, config, or recipe file was edited. `docs/DEFAULT_EXPERIMENT_RECIPE.md`
  is untouched.
- The contract test `tests/test_graph_specification_ablation_notebook.py` was not run for
  this report and is not cited as run evidence.
