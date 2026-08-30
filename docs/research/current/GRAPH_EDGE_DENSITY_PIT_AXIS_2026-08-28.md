# Correlation-graph edge density and node isolation on the PIT-admissible axis

**Date:** 2026-08-28
**Ticket:** magilliam27/MCI-GRU issue #163 (`wayfinder:research`, AFK), child of map #157
**Status:** Current research evidence. Read-only measurement; no training, no GPU, no code change.
**Supersedes as production figures:** the density table preserved in
[map #157's original body](https://github.com/magilliam27/MCI-GRU/issues/157#issuecomment-5235023475).
That table was measured on the panel **union** axis. The figures here are measured on the
**PIT-admissible** axis and are the ones that describe production. The union-axis table is
reproduced exactly below as a control, and is not wrong — it answers a different question.

Every figure is labelled `[Verified]` (measured in this run) or `[Inferred]` (reasoning on
top of measurements). Section 9 is entirely interpretation and is labelled as such.

---

## 1. Summary

Nine findings, all `[Verified]` unless marked.

1. **Moving to the admissible axis removes 55–62% of the edges and makes isolation worse
   in every window.** Density *rises* — but only because the denominator shrinks faster
   than the numerator. The fraction of tradable names with no neighbours goes **up**.
2. **On the axis production uses, `judge_value=0.8` sits above the 99th percentile of the
   correlation distribution on 108 of 120 monthly build dates (90%)** — including **100% of
   validation dates and 100% of test dates**.
3. **The validation span is the emptiest, not the test span.** Mean isolated-node fraction
   under the shipped default: train **74.0%**, test **77.2%**, validation **88.9%**. The
   frozen recipe selects models on `val_ic`, on the window where the graph does least.
4. **The full degree distribution is not merely "median 0" — it is a spike at zero with a
   thin tail.** In validation, 88.9% of node-dates have degree 0, 10.3% have degree 1, and
   0.8% have degree ≥ 2. Maximum degree observed all year: **4**.
5. **The threshold arm saturates.** At `judge_value ≤ -0.2` the graph is the complete
   admissible graph on ≥ 52% of dates and at `-0.5` on 100% of dates. `0.0` already has
   Jaccard 0.95 against complete. Opening the threshold below ≈ `0.1` buys progressively
   nothing.
6. **Top-K edge sets are strictly nested in K**, verified directly (48/48 adjacent-depth
   checks). Adjacent-depth Jaccard is therefore the algebraic identity `K₁/K₂` —
   0.500, 0.667, 0.750, 0.800 — with **zero variance across all 120 dates**. Jaccard cannot
   distinguish top-K depths on data grounds; the overlap is fixed by the grid.
7. **The threshold family is also strictly nested** as the threshold falls (132/132 checks).
8. **Top-K and threshold never coincide, even at matched density.** Best cross-family
   Jaccard anywhere on the grid is 0.61. At *per-date* density-matched settings it is
   0.485 (K=10) rising to 0.706 (K=50) — the two rules disagree on 29–51% of edges while
   holding edge count equal.
9. **No fixed threshold density-matches top-K through time.** The matching threshold for
   K=10 ranges from 0.36 to 0.73 across build dates.

**The bound this places on the arm count for #164:** the proposed grid of five top-K depths
plus a widened threshold arm contains at most **13 distinct threshold points plus one
saturated point**, and **five nested top-K depths that form a monotone ladder rather than
five independent arms**. The two families are genuinely different adjacencies and cannot be
collapsed into one another.

---

## 2. What was measured, and how

| | |
|---|---|
| Panel | `data/raw/market/sp500_pit_gics_top10_mcap_monthly_20160104_20260731_lseg_20150101_20260731.csv` — 40,035,626 bytes, 573,403 rows, 206 names, 2015-01-02 … 2026-07-30 |
| PIT intervals | `data/raw/constituents/sp500_pit_gics_top10_mcap_monthly_20160104_20260731_pit_universe.csv` — 545 intervals |
| Node axis | PIT union over the experiment span, via `active_kdcodes_in_period(intervals, train_start, test_end, available)` → **201 names** |
| Admissible mask | `active_membership_mask(kdcode_list, [date], intervals)` — the same call `GraphBuilder` makes through `admissible_mask_for_date` |
| Correlation | the real `mci_gru.graph.correlation.compute_correlation_matrix`, `corr_lookback_days=252` |
| Selection | the real `mci_gru.graph.correlation.build_edges`, `use_multi_feature_edges=True`, `top_k_metric="corr"` |
| Build dates | 120 monthly dates, `2016-01-04 + k months`, matching `GraphBuilder.get_update_dates`, which steps by `relativedelta` and does not snap to sessions |
| Splits | train 2016-01-04…2023-12-31 (96 dates), val 2024-01-22…2024-12-31 (11), test 2025-01-22…2025-12-31 (11), plus 2 dates in the embargo gaps, reported separately |

`n_adm = 110` on **all 120 build dates** — min, median and max are all 110. The universe is
flat at its design size across the whole span.

### Two things done deliberately

**Selection ran through the shipped selector, not a reimplementation.** #163 notes that a
negative threshold is measurable without a `GraphConfig`, and that routing through
`GraphConfig` hits its validator. That is correct, and the route taken here is one step
better: `build_edges` is a free function taking `judge_value: float` and performs no bound
check, and `GraphBuilder.__init__` validates only `top_k` and `top_k_metric`. So the
production selection code measured the negative-threshold arm directly. Nothing was
reimplemented in numpy, and the `[-1, 1)` bound opened by #162 was not needed for this run.

**The import path was pinned.** `sys.path.insert(0, <worktree root>)`, and every script
asserts `mci_gru.__file__.startswith(<worktree root>)` before doing any work. Recorded
output: `C:\Users\magil\.claude\worktrees\mci-gru-163-density\mci_gru\__init__.py`. Without
this the venv's editable install silently resolves to the protected checkout.

### Degree convention

Edges are directed: `edge_index[0] = rows`, `edge_index[1] = cols`, and PyG's default
`flow="source_to_target"` means a node aggregates from its **in**-neighbours. **In-degree is
therefore the quantity that decides whether a node receives any cross-sectional information
at all**, and "isolated" below means in-degree 0.

- Threshold arms are **symmetric on every date** (verified, 2,160/2,160 arm-dates), so
  in-degree and out-degree coincide and the distinction does not matter there.
- Top-K arms are **never symmetric** (0/600 arm-dates). Out-degree is exactly K for every
  admissible node by construction, so out-degree isolation is 0 by definition and carries
  no information. In-degree is the real measure and is reported.

---

## 3. Finding 1 — the axis change, measured as a paired difference

The preserved table's five dates, re-measured on both axes in the same run. The union-axis
column is the control: it reproduces the preserved table **exactly** — same edge counts,
same isolated counts, same p99s — which is what licenses reading the difference as the axis
change and not as a pipeline difference. `[Verified]`

| date | axis | n | corr p99 | edges @0.8 | possible pairs | density | isolated | median deg |
|---|---|---|---|---|---|---|---|---|
| 2017-01-03 | union | 191 | +0.714 | 152 | 36,290 | 0.419% | 146 (76.4%) | 0 |
| 2017-01-03 | **admissible** | **110** | **+0.728** | **64** | **11,990** | **0.534%** | **85 (77.3%)** | **0** |
| 2019-01-02 | union | 196 | +0.738 | 138 | 38,220 | 0.361% | 143 (73.0%) | 0 |
| 2019-01-02 | **admissible** | **110** | **+0.759** | **60** | **11,990** | **0.500%** | **86 (78.2%)** | **0** |
| 2021-01-04 | union | 202 | +0.821 | 596 | 40,602 | 1.468% | 80 (39.6%) | 1 |
| 2021-01-04 | **admissible** | **110** | **+0.834** | **218** | **11,990** | **1.818%** | **47 (42.7%)** | **1** |
| 2023-01-03 | union | 204 | +0.767 | 272 | 41,412 | 0.657% | 123 (60.3%) | 0 |
| 2023-01-03 | **admissible** | **110** | **+0.784** | **100** | **11,990** | **0.834%** | **72 (65.5%)** | **0** |
| 2025-01-22 | union | 206 | +0.598 | 36 | 42,230 | 0.085% | 173 (84.0%) | 0 |
| 2025-01-22 | **admissible** | **110** | **+0.589** | **14** | **11,990** | **0.117%** | **96 (87.3%)** | **0** |

**The direction the preserved caveat predicted holds, and one it did not predict also holds.**
The caveat said the qualifying-pair count could not rise. It did not — it fell by 55–62% at
every date. What the caveat left open, and what this table settles, is the other two columns:

- **Density rises**, by 24–43% relative. This is arithmetic, not signal: removing an
  inadmissible name deletes its ~2×109 candidate pairs from the denominator but only its
  handful of qualifying edges from the numerator. **Density is the wrong headline number for
  this graph** and should not be quoted without the isolation figure beside it.
- **Isolation gets worse at every single date**, by +0.9 to +5.2 percentage points. On the
  axis production actually uses, a *larger* share of the names that can be traded have no
  neighbours than the union-axis table suggested.

So the preserved table understated the problem in the dimension that matters and overstated
the edge count. Neither of its five figures should now be quoted as a production number.

---

## 4. Finding 2 — where `judge_value=0.8` sits in the admissible distribution

`[Verified]`, over all 120 build dates, off-diagonal admissible pairs only.

| span | median | p90 | p95 | **p99** | max | share negative |
|---|---|---|---|---|---|---|
| train | +0.337 | +0.533 | +0.593 | **+0.744** | +0.992 | 3.7% |
| val | +0.188 | +0.373 | +0.439 | **+0.631** | +0.998 | 7.7% |
| test | +0.252 | +0.465 | +0.525 | **+0.687** | +0.998 | 9.6% |

- **0.8 exceeds the admissible p99 on 108 of 120 dates (90.0%)** — by span: train 88%,
  validation **100%**, test **100%**.
- 0.8 never exceeds the admissible *maximum* (0/120), so the graph is never literally empty.
- Correlations are overwhelmingly positive: only 3.7–9.6% of admissible pairs are negative.

`[Inferred]` The threshold is not calibrated to this universe. A cutoff above p99 means
fewer than 1% of pairs can qualify *by construction*, before any market behaviour is
consulted — and in the two spans that decide model selection and the reported result, that
is true on every single build date.

---

## 5. Finding 3 — the shipped default, in full

### Edge count and isolation by span, `judge_value=0.8`, top_k=0 `[Verified]`

| span | dates | mean edges | min | max | mean density | **mean isolated** | worst date isolated | mean max degree |
|---|---|---|---|---|---|---|---|---|
| train | 96 | 83.9 | 18 | 384 | 0.700% | **74.0%** | 89.1% | 6.3 |
| val | 11 | 13.3 | 8 | 28 | 0.111% | **88.9%** | **92.7%** | 1.4 |
| test | 11 | 43.8 | 14 | 64 | 0.365% | **77.2%** | 87.3% | 4.1 |
| gap | 2 | 21.0 | 12 | 30 | 0.175% | 85.9% | 89.1% | 2.0 |

**Validation is the sparsest span**, by a clear margin. Map #157 recorded the 2025/test
window as the sparsest, which was true of the five single dates then measured; across the
full monthly grid it is not. The three emptiest dates in the entire run are all validation
dates: 2024-12-04 and 2024-05-04 (8 directed edges, 92.7% isolated, **max degree 1**) and
2024-04-04 (10 edges, 90.9% isolated, max degree 1).

The densest dates are the COVID window — 2020-05-04 at 384 edges and 37.3% isolated is the
best the shipped default ever achieves, and it is still worse than one node in three having
no neighbours.

### Full in-degree distribution, not the median `[Verified]`

Percentage of admissible node-dates at each degree, pooled within span. This is the figure
#163 asked for in place of the median, which is 0 nearly everywhere and hides the shape.

| span | node-dates | deg 0 | deg 1 | deg 2 | deg 3 | deg 4 | deg 5 | deg 6–10 | deg 11+ | max |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 10,560 | **74.0%** | 10.4% | 4.3% | 3.3% | 2.2% | 2.5% | 2.7% | 0.5% | 18 |
| val | 1,210 | **88.9%** | 10.3% | 0.6% | 0.1% | 0.1% | 0.0% | 0.0% | 0.0% | **4** |
| test | 1,210 | **77.2%** | 15.9% | 2.4% | 1.2% | 1.2% | 1.9% | 0.2% | 0.0% | **6** |

The distribution is a spike at zero with a thin tail. In validation, **99.2% of node-dates
have degree 0 or 1**, and the busiest node in the entire validation year has four neighbours.

**The edges that exist are concentrated on a few names.** Share of all edge endpoints held
by the top 10% of nodes by degree: train **70.5%**, validation **89.9%**, test **67.6%**.
So the graph is not "sparse but even" — it is a small clique-ish core plus a large isolated
remainder.

### Why this is a degeneration rather than a crash `[Verified]` as a code fact

`GATBlock` constructs `GATConv(...)` at `mci_gru/models/graph.py:36` and `:39` without
passing `add_self_loops`, so PyG's default `True` applies. An in-degree-0 node still
receives its own self-loop message. `[Inferred]` For that node the attention is over a
single element, softmax returns 1.0, and the block reduces to a node-wise linear transform
plus activation — the same conclusion map #157 drew, now checked against the code and
attached to a measured 74–89% of node-dates rather than to five single dates.

`drop_edge_p: 0.1` in the base config applies at training time only
(`trunk.py:226`, gated on `training`), so it does not touch the validation and test figures
above. `[Inferred]` On a training date at the low end (18 edges) it removes ~2 edges; the
regulariser's meaning is very different at 18 edges and at 11,990, which is the open
question map #157 already records under *Not yet specified*.

---

## 6. Finding 4 — the threshold curve

Mean over all 120 build dates. Density is over admissible directed pairs
(110 × 109 = 11,990). `[Verified]`

| judge_value | mean edges | median | min | max | density | **isolated** | mean median deg | mean max deg |
|---|---|---|---|---|---|---|---|---|
| 0.90 | 7 | 5 | 2 | 32 | 0.06% | 94.5% | 0 | 1.5 |
| 0.85 | 30 | 22 | 4 | 128 | 0.25% | 85.2% | 0 | 3.3 |
| **0.80** ← shipped | **73** | **52** | **8** | **384** | **0.61%** | **75.9%** | **0.1** | **5.6** |
| 0.75 | 180 | 100 | 26 | 1,230 | 1.50% | 63.7% | 0.7 | 9.3 |
| 0.70 | 360 | 158 | 50 | 2,612 | 3.01% | 51.9% | 2.0 | 13.8 |
| 0.65 | 639 | 260 | 66 | 4,520 | 5.33% | 39.4% | 4.6 | 20.0 |
| 0.60 | 1,017 | 453 | 118 | 6,558 | 8.48% | 28.0% | 8.0 | 27.8 |
| 0.55 | 1,524 | 733 | 180 | 8,380 | 12.71% | 19.3% | 12.5 | 36.8 |
| 0.50 | 2,190 | 1,265 | 236 | 9,780 | 18.26% | 12.8% | 18.7 | 46.8 |
| 0.40 | 3,921 | 3,228 | 472 | 11,122 | 32.71% | 4.2% | 36.2 | 67.3 |
| 0.30 | 6,076 | 5,846 | 1,214 | 11,656 | 50.68% | 0.5% | 58.1 | 86.2 |
| 0.20 | 8,372 | 8,299 | 3,000 | 11,888 | 69.83% | 0.1% | 79.1 | 100.9 |
| 0.10 | 10,310 | 10,336 | 6,412 | 11,984 | 85.99% | 0.0% | 95.7 | 108.0 |
| 0.00 | 11,431 | 11,559 | 9,886 | 11,990 | 95.33% | 0.0% | 105.9 | 109.0 |
| −0.10 | 11,846 | 11,903 | 11,260 | 11,990 | 98.80% | 0.0% | 108.8 | 109.0 |
| −0.20 | 11,960 | 11,990 | 11,746 | 11,990 | 99.75% | 0.0% | 109.0 | 109.0 |
| −0.30 | 11,987 | 11,990 | 11,930 | 11,990 | 99.97% | 0.0% | 109.0 | 109.0 |
| −0.50 | 11,990 | 11,990 | 11,990 | 11,990 | **100.00%** | 0.0% | 109.0 | 109.0 |

### Saturation `[Verified]`

Fraction of build dates on which the arm is the **complete** admissible graph:

| judge_value | 0.10 | 0.00 | −0.10 | −0.20 | −0.30 | −0.50 |
|---|---|---|---|---|---|---|
| complete on | 0% | 8.3% | 25.8% | 52.5% | 76.7% | **100%** |

**The threshold arm has an effective floor at roughly `0.0`.** Below it the arm is
increasingly just "connect everything", and at `−0.5` it is exactly that on every date. The
negative range #162 opened is measurable and behaves as expected — it simply has little left
to buy by the time it is reached.

### Isolation by span across the curve `[Verified]`

| judge_value | 0.90 | 0.85 | **0.80** | 0.75 | 0.70 | 0.65 | 0.60 | 0.55 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 93.9% | 83.8% | **74.0%** | 61.2% | 48.7% | 35.9% | 24.3% | 15.9% | 9.5% | 2.8% | 0.3% | 0.1% |
| val | 98.2% | 95.0% | **88.9%** | 76.8% | 69.6% | 59.5% | 50.4% | 40.3% | 34.0% | 15.0% | 1.7% | 0.0% |
| test | 95.7% | 85.8% | **77.2%** | 69.5% | 58.8% | 46.6% | 34.4% | 24.6% | 17.3% | 4.9% | 1.0% | 0.2% |

Threshold required to hold mean isolation below a stated level:

| span | isolation < 10% | isolation < 1% |
|---|---|---|
| train | `judge_value ≤ 0.50` | `judge_value ≤ 0.30` |
| val | `judge_value ≤ 0.30` | `judge_value ≤ 0.20` |
| test | `judge_value ≤ 0.40` | `judge_value ≤ 0.30` |

`[Inferred]` A single fixed threshold cannot hold a stable isolation level across the three
spans — the val curve sits 10–20 points above train throughout the usable range. Any arm
specified as a constant `judge_value` is implicitly accepting a different graph regime in
each span.

---

## 7. Finding 5 — the top-K grid

`[Verified]`, mean over 120 dates, `top_k_metric="corr"`.

| K | edges | density | **isolated (in-deg 0)** | worst date | in-deg min | median | p90 | max | symmetric |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 1,100 | 9.17% | **5.6%** | 12.7% | 0.0 | 8.3 | 21.5 | 43.7 | never |
| 20 | 2,200 | 18.35% | **1.5%** | 4.6% | 0.3 | 17.0 | 41.0 | 69.5 | never |
| 30 | 3,300 | 27.52% | **0.7%** | 2.7% | 1.0 | 26.4 | 58.7 | 84.2 | never |
| 40 | 4,400 | 36.70% | **0.5%** | 1.8% | 2.3 | 35.6 | 72.4 | 96.3 | never |
| 50 | 5,500 | 45.87% | **0.3%** | 1.8% | 4.2 | 46.7 | 84.9 | 105.0 | never |

Edge count is exactly `110 × K` on every date — every admissible node always finds K
candidates, so no date is degree-starved.

**Top-K does produce isolated nodes, but only via in-degree.** Out-degree is K for every
admissible node by construction, so out-degree isolation is 0 everywhere and is not a
meaningful statistic. In-degree isolation is 5.6% at K=10, falling to 0.3% at K=50 — one to
two orders of magnitude below the shipped threshold arm's 75.9%.

Pooled in-degree distribution: at K=10, 5.6% of node-dates have in-degree 0 and 19.3% have
≤ 2, with a long right tail to 70. `[Inferred]` Top-K redistributes rather than equalises —
it guarantees every node *sends*, but popular names still absorb most of the incoming edges.

---

## 8. Finding 6 — which grid points are actually distinct

Jaccard on the directed edge set, computed per date and averaged over all 120 dates. This
is the section that bounds the arm count for #164.

### Adjacent threshold points `[Verified]`

| pair | mean J | min | max | | pair | mean J | min | max |
|---|---|---|---|---|---|---|---|---|
| 0.90 → 0.85 | 0.280 | 0.056 | 1.000 | | 0.50 → 0.40 | 0.473 | 0.324 | 0.881 |
| 0.85 → 0.80 | 0.441 | 0.143 | 0.923 | | 0.40 → 0.30 | 0.572 | 0.337 | 0.956 |
| 0.80 → 0.75 | 0.483 | 0.200 | 0.720 | | 0.30 → 0.20 | 0.683 | 0.398 | 0.983 |
| 0.75 → 0.70 | 0.570 | 0.381 | 0.786 | | 0.20 → 0.10 | 0.796 | 0.449 | 0.992 |
| 0.70 → 0.65 | 0.613 | 0.447 | 0.796 | | 0.10 → 0.00 | **0.899** | 0.649 | 0.999 |
| 0.65 → 0.60 | 0.617 | 0.459 | 0.828 | | 0.00 → −0.10 | **0.965** | 0.861 | 1.000 |
| 0.60 → 0.55 | 0.619 | 0.477 | 0.866 | | −0.10 → −0.20 | **0.990** | 0.958 | 1.000 |
| 0.55 → 0.50 | 0.631 | 0.520 | 0.857 | | −0.20 → −0.30 | **0.998** | 0.983 | 1.000 |
| | | | | | −0.30 → −0.50 | **1.000** | 0.995 | 1.000 |

Every threshold point from 0.90 down to 0.10 is distinct from its neighbour (J ≤ 0.90).
Every point from 0.00 down collapses: all pairs among `{0.0, −0.1, −0.2, −0.3, −0.5}` have
mean J ≥ 0.953, and `−0.3` vs `−0.5` is 1.000 to three decimals.

**Threshold arms that coincide at mean J ≥ 0.90** — these are one arm, not several:

```
thr0.0 ~ thr-0.1 (0.965)   thr0.0 ~ thr-0.2 (0.956)   thr0.0 ~ thr-0.3 (0.954)   thr0.0 ~ thr-0.5 (0.953)
thr-0.1 ~ thr-0.2 (0.990)  thr-0.1 ~ thr-0.3 (0.988)  thr-0.1 ~ thr-0.5 (0.988)
thr-0.2 ~ thr-0.3 (0.998)  thr-0.2 ~ thr-0.5 (0.998)  thr-0.3 ~ thr-0.5 (1.000)
```

### Adjacent top-K depths — a structural identity, not a measurement `[Verified]`

| pair | mean J | min | max | train | val | test |
|---|---|---|---|---|---|---|
| K10 → K20 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| K20 → K30 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |
| K30 → K40 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| K40 → K50 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 |

Min equals max equals mean, on every date and in every span. **This is because top-K edge
sets are strictly nested in K** — verified directly rather than inferred from the ratio:
across 12 sampled dates and 48 adjacent-depth checks, `E(K−10) ⊆ E(K)` held every time, with
zero edges ever lost as K grows. Non-adjacent pairs confirm it: J(K10,K30) = 0.333 = 10/30,
J(K10,K50) = 0.200 = 10/50, J(K20,K40) = 0.500 = 20/40.

`[Inferred]` **Jaccard is therefore uninformative about top-K depth.** Overlap between two
depths is `K₁/K₂` by algebra and tells you nothing about the market. Each increment of K
appends the next-ranked neighbours to every node and removes nothing, so the question "are
K=30 and K=40 distinct arms?" is not answerable by set overlap — the sets *are* different,
necessarily and by a known amount. Whether that difference matters is a question only a
training run answers. This is a different situation from the threshold family, where overlap
genuinely varies with the data (min 0.056 to max 1.000 on a single pair).

The threshold family is also strictly nested as the threshold falls (132/132 checks), but
its *step sizes* are data-dependent, which is why its Jaccards carry information.

### Top-K against the threshold arm `[Verified]`

Best-matching threshold for each depth, by mean Jaccard over the whole grid:

| K | mean edges | best match | 2nd | 3rd |
|---|---|---|---|---|
| 10 | 1,100 | thr0.50 (J=0.368) | thr0.55 (0.352) | thr0.40 (0.311) |
| 20 | 2,200 | thr0.40 (J=0.425) | thr0.30 (0.381) | thr0.50 (0.367) |
| 30 | 3,300 | thr0.30 (J=0.486) | thr0.40 (0.462) | thr0.20 (0.411) |
| 40 | 4,400 | thr0.30 (J=0.549) | thr0.20 (0.520) | thr0.40 (0.468) |
| 50 | 5,500 | thr0.20 (J=0.607) | thr0.30 (0.586) | thr0.10 (0.538) |

**The highest cross-family Jaccard anywhere on the grid is 0.607.** No top-K depth is a
restatement of any threshold value.

### The matched-density control `[Verified]`

The comparison above confounds selection rule with edge count. Re-run with the threshold
chosen **per date** to match that date's top-K edge count exactly (search grid 0.95 → −0.55
in steps of 0.01, 12 sampled dates):

| K | K edges | matched threshold (mean) | range | matched edges | **Jaccard** | min | max |
|---|---|---|---|---|---|---|---|
| 10 | 1,100 | 0.512 | 0.36 – 0.73 | 1,102 | **0.485** | 0.431 | 0.540 |
| 20 | 2,200 | 0.443 | 0.29 – 0.69 | 2,221 | **0.565** | 0.485 | 0.649 |
| 30 | 3,300 | 0.398 | 0.24 – 0.66 | 3,303 | **0.622** | 0.552 | 0.702 |
| 40 | 4,400 | 0.358 | 0.20 – 0.63 | 4,456 | **0.665** | 0.606 | 0.741 |
| 50 | 5,500 | 0.325 | 0.17 – 0.60 | 5,463 | **0.706** | 0.659 | 0.775 |

Holding edge count equal, the two rules still disagree on **29% to 51% of edges**. Density
and selection rule are separable on this universe, and top-K's difference from threshold is
not reducible to it being denser.

**The matching threshold is not stable through time.** For K=10 it ranges from 0.36 to 0.73
across build dates — a spread of 0.37 in `judge_value`. `[Inferred]` A control arm defined
as a *fixed* threshold cannot hold density constant against a top-K arm; matching would have
to be done per snapshot, which is a different configuration shape than
`graph.judge_value: <float>` currently permits.

---

## 9. What this bounds for #164 — inputs, not answers

`[Inferred]` throughout. #164 is a `wayfinder:grilling` ticket and is `ready-for-human`; it
resolves only through live exchange with the maintainer. Nothing below is a decision, and
none of #164's questions are answered here. These are the measured constraints the decision
now has available.

**On arm count.** The proposed grid contains fewer distinct configurations than it has
points:

- The **threshold family** yields distinct edge sets from 0.90 down to about 0.10 — roughly
  13 usable points — after which everything collapses into a single "complete admissible
  graph" arm. Opening the threshold "much wider" is worth doing to about `0.1`, and past
  `0.0` it stops buying anything.
- The **top-K family**'s five depths are five distinct edge sets, but they form a *nested
  ladder*, not five independent configurations. Their pairwise overlap is fixed by the grid.
- The **two families do not overlap**, at matched density or otherwise.

**On the confound #164 names.** The April report's finding that top-K underperformed cannot
be explained by "top-K was denser" alone: at equal edge count the two rules select largely
different edges (J = 0.49–0.71). The confound is real but it is not total, and a
matched-density control is measurable — though only as a per-snapshot threshold, not a
constant.

**On the control arm (#165).** The shipped default already leaves 74–89% of nodes with no
in-edges and reduces the GAT to a node-wise transform for them. `[Inferred]` A graph-zeroed
control and the shipped threshold arm may therefore be closer in behaviour than the arm
count suggests, which makes the control's separation from the shipped default an
informative measurement in its own right rather than a formality.

**On where the measurement is weakest.** The validation span is the emptiest of the three
and the frozen recipe selects on `val_ic`. Any protocol that reports a mean across spans
will average over a validation window where the graph is doing close to nothing.

---

## 10. Reproduction

Scripts were run from a scratchpad, not committed — this ticket owns no code path. Each
pins the import and asserts the tree before measuring. The essential form:

```python
import sys; sys.path.insert(0, r"<worktree root>")
import numpy as np, pandas as pd
import mci_gru
assert mci_gru.__file__.startswith(r"<worktree root>"), mci_gru.__file__

from mci_gru.graph.correlation import compute_correlation_matrix, build_edges
from mci_gru.data.pit import (
    load_pit_intervals, active_kdcodes_in_period, active_membership_mask,
)

df = pd.read_csv(PANEL, usecols=["dt", "kdcode", "close"])
df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
intervals = load_pit_intervals(PIT_UNIVERSE_CSV)

kdcode_list = active_kdcodes_in_period(
    intervals, "2016-01-04", "2025-12-31",
    available_kdcodes=set(df["kdcode"].astype(str).unique()),
)                                              # 201 names
corr = compute_correlation_matrix(df, kdcode_list, DATE, 252)
adm = active_membership_mask(kdcode_list, [DATE], intervals)[0]   # 110 True

# Threshold arm; judge_value is a plain float here, so negative values need no
# GraphConfig and are unaffected by its validator.
ei, _ = build_edges(corr, kdcode_list, False, None, JUDGE_VALUE, 0, "corr",
                    True, False, [1, 2, 3, 5], admissible_mask=adm)
# Top-K arm: pass K in place of 0 above.

N = len(kdcode_list)
indeg = np.bincount(ei.numpy()[1], minlength=N)[np.where(adm)[0]]
print(ei.shape[1], (indeg == 0).mean(), np.bincount(indeg))
```

The union-axis control in section 3 is the preserved body's own snippet, unchanged.

Recorded environment: `mci_gru.__file__` =
`C:\Users\magil\.claude\worktrees\mci-gru-163-density\mci_gru\__init__.py`. Full run:
120 build dates × 23 arms = 2,760 arm-date records and 30,360 Jaccard pairs, in 33 seconds.

---

## 11. Limits of this report

- **Monthly grid, not per-session.** 120 build dates on the production `relativedelta`
  cadence. Between-date behaviour is not measured. Validation and test contribute 11 dates
  each, so their means rest on a small sample even though every date in them is included.
- **Nesting and the matched-density control were verified on 12 sampled dates**, not all
  120 — 48 and 132 checks respectively, all passing. The full-grid Jaccards (section 8)
  are over all 120 dates.
- **`top_k_metric="corr"` only.** The `abs_corr` path, which recovers strong negative
  correlations, was not measured; it is a different adjacency and would need its own row.
- **Edge *features* were not evaluated.** This is adjacency only. The degeneracy recorded
  in #114 concerns the feature channels and is untouched here.
- **No training was run**, so nothing here says whether any arm predicts better. That is
  what #165–#167 are for.
- Sector-relation, lead–lag and snapshot-age edge families are out of scope for map #157
  and were left off.

---

## 12. Provenance

- Produced under issue #163, a `wayfinder:research` child of map #157.
- Protected checkout `C:\Users\magil\MCI-GRU` was read-only throughout (`git -C` reads and
  data-file reads only); it remained on `codex/paper_trade_scrape` @ `e286649`.
- Branch `claude/163-density`, worktree
  `C:\Users\magil\.claude\worktrees\mci-gru-163-density`, off `origin/main` @ `5fa04a4`.
