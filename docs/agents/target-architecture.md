# MCI-GRU Target Architecture Workspace

Status: **human-led design workspace. Not current-state authority.**

This document does not describe what the code does today. Nothing in it may be
cited as evidence of implemented behaviour, and an agent must never satisfy a
factual question about the repository from this file.

- For implemented architecture, read [../ARCHITECTURE.md](../ARCHITECTURE.md).
- For current-state routing — owning modules, adjacent contracts, focused tests,
  and engineering constraints — read [guide.md](guide.md).
- For the source-of-truth order, read [domain.md](domain.md).

This file is the shared workspace where the project owner and an agent define
MCI-GRU's target architecture together. It is intentionally incomplete. It is
not a roadmap, not an implementation authorisation, and not a summary of
historical plans under `docs/agent_references/`.

## Working Agreement

- The project owner supplies goals, priorities, risk tolerance, and acceptable
  trade-offs.
- The agent supplies current-code evidence, alternatives, consequences,
  migration options, and verification strategies.
- No historical plan, handoff, research report, or agent proposal becomes a
  decision by default.
- Design-gated work in progress elsewhere is not a decision either. Record it as
  an open question, not as a target.
- Record unresolved questions as unresolved. Do not silently choose a target.
- A decision that changes a repository invariant listed in `AGENTS.md` must name
  its migration and validation path before implementation begins.

## Goals

The owner wants repeatable Colab runs, model components that can be varied
without rebuilding the execution workflow, and documented results that another
agent can reconstruct and assess. Packet A begins with the reproducibility
decisions recorded under [issue #142](https://github.com/magilliam27/MCI-GRU/issues/142)
and map [#140](https://github.com/magilliam27/MCI-GRU/issues/140).

The agreed target is **environment-equivalent reruns with complete provenance
and measured numerical tolerances**, including stability of selected stocks.
This is a target contract, not an implemented guarantee. Model interfaces and
other architectural boundaries remain undecided where indicated below.

## Non-Goals

This milestone does not require bitwise training equality. Optional strict
determinism is deferred until baseline repeatability measurements show whether
it is needed. Revisit it if same-seed variation materially changes selections
or conclusions; do not widen tolerances merely to excuse instability.

These decisions do not change the frozen experiment recipe, model defaults,
training loss, stock-allocation rule, or graph research protocol. They do not
authorize new compute runs, data acquisition, or retirement of old evidence.

## Design Forces And Constraints

Preserve the existing no-lookahead, PIT masked-panel breadth, GraphSchedule,
9-tuple collate, arithmetic-mean ensemble, and frozen paper-trade invariants in
`AGENTS.md`. Reproducibility comparisons must preserve the experiment's data,
selection and timing semantics rather than modifying them to obtain agreement.

Cost and runtime matter alongside repeatability. A cheaper GPU per hour is not
necessarily cheaper per completed experiment. Qualification needs measured
resource use, wall time, and cost evidence, with missing measurements labelled
unknown. A mechanics smoke proves wiring, not full-recipe reproducibility or
model performance.

## Target System Boundaries

> To decide together.

## Target Data And Timing Contracts

> To decide together.

## Target Model And Graph Boundaries

> To decide together.

## Target Training, Evaluation, And Promotion Flow

### G4 reference and T4 qualification

The owner selected **one initial reference Colab environment: G4**, based on
their experience of its cost/speed balance. Prepare a **T4 qualification plan
alongside it**, then qualify T4 separately against the established reference.
The [existing G4 smoke record](https://github.com/magilliam27/MCI-GRU/issues/185#issuecomment-5555870542)
identifies an NVIDIA RTX PRO 6000 Blackwell Server Edition. That is the starting
hardware candidate, not proof that the full repeatability contract has passed.

Record both the Colab runtime label and actual GPU identity. Software versions,
GPU memory, driver, CUDA, precision settings, and host RAM also form part of
the environment record. A different or faster GPU does not automatically
inherit qualification. Select and verify software pins through #143; the
historical smoke's package versions are evidence, not newly approved pins.

The requested T4 plan uses the same execution and artifact workflow with
explicit environment profiles, not a second model or notebook implementation:

| Stage | Agent deliverable | Evidence needed to advance |
| --- | --- | --- |
| Prepare alongside G4 | Profile fields, preflight checks, shared comparison report and a bounded benchmark proposal | Explicit environment identity, workload, outputs, and run budget |
| Compatibility and capacity | Required-kernel/setup check, data preparation, then a bounded T4 training smoke | Peak GPU memory and host RAM, setup/epoch timing, valid artifacts; smoke-only status |
| Comparable cost and feasibility | Run the same named benchmark workload on G4 and T4 | Wall time and observed compute-unit consumption where available; separate setup from training and label extrapolations |
| Qualification | Fresh-session same-seed T4 reruns plus comparisons against G4 | Separate within-T4 repeatability and cross-device portability; full comparison artifacts and verified recovery behavior |
| Supported workload decision | A report stating the workload T4 can actually support | Unchanged reference workload qualified, explicitly limited smoke/development support, or unqualified status |

Do not silently shrink the PIT universe, batch, precision, epochs, or ensemble
to fit T4. A behavior-affecting adaptation is a distinct named configuration
requiring review and its own evidence. Record whether recovery replays a whole
model or restores validated optimizer/RNG/data-loader state; interrupted
training continuation is not assumed equivalent to a fresh run.

The existing [Colab runbook](../workflows/COLAB_CHROME_CONTROL_GUIDE.md) remains
the operational authority. Its broad "G4-class or better" check is not an
environment-equivalence test. A future scoped T4 task must amend the relevant
profile/runbook gates explicitly; this design does not globally relax them.

### Stock-selection distribution comparison

The owner accepted **date-by-date, base-2 Jensen-Shannon divergence (JSD)** as
the primary selected-stock distribution comparison, with selection-frequency
views alongside it. Similar aggregate IC or Sharpe alone is insufficient.

Compare the same declared selection rule on matching signal dates and stock
identifiers. For equal-weight top-K baskets, each selected stock carries `1/K`
mass and other eligible stocks carry zero. K belongs to the named experiment;
this policy selects no new K. Use the saved arithmetic-mean ensemble scores as
the primary output; member-level diagnostics do not replace full-run repeats.

For distributions `p` and `q` on the same stock axis:

```text
m = (p + q) / 2
JSD(p, q) = [KL_base2(p || m) + KL_base2(q || m)] / 2
```

JSD is symmetric and finite with zero selection mass in one basket. Its range
is 0 for identical distributions to 1 for disjoint support. In contrast, raw
KL can be infinite at a support mismatch. SciPy's
[`jensenshannon`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html)
returns the square-root **distance**: square its result with `base=2` when
reporting divergence. Validate distributions before calling it; automatic
normalization must not hide malformed input.

For two equal-weight baskets of the same size, JSD equals
`1 - intersection_size / K`. It is therefore another expression of overlap,
not independent evidence beyond overlap. Two top-10 baskets sharing nine
stocks have JSD 0.1; this is a synthetic illustration, **not a pass threshold**.

Supporting diagnostics should preserve:

- date-level divergence and typical, upper-tail, and worst-period summaries;
- each run's per-stock selection frequencies over the declared date window;
- stock-by-date inclusion frequencies across reruns, with the run count shown;
- basket overlap, ranking agreement, score differences, and metric deltas;
- the dates and names responsible for disagreement, in tables and heatmaps.

An inclusion frequency across runs is a probability for each stock separately;
these rates need not sum to one across stocks. Any categorical distribution
built from them needs an explicit normalization definition. Pooled frequencies
cannot replace dated comparisons: the same overall frequencies can conceal
different selections on every date.

Implementation must reuse the selection/ranking semantics in
`mci_gru/evaluation/portfolio.py` and the relevant evaluation surface. Do not
compare pre-execution top-K picks in one run with realized holdings after
missing-price filtering or stateful retention in another. Specify ties and the
selection/holding date boundary before implementation. Missing expected dates,
duplicate identifiers, invalid scores, or mismatched eligibility masks are
contract failures, not zero divergence. Do not silently intersect away missing
observations or use future returns to decide which picks enter the comparison.

### Baseline variation before acceptance thresholds

The owner accepted this sequence:

1. Specify a bounded calibration workload using a fixed code/input identity,
   resolved recipe, evaluation window, selection rule, and seed schedule in the
   G4 reference environment. Scope repeat count and cost before execution.
2. Measure same-seed reruns. Report typical disagreement, unstable dates and
   affected stocks, plus companion numerical and performance differences.
3. Propose thresholds with practical interpretations for owner review. A small
   calibration sample is not proof about the full tail of possible outcomes;
   report sample size, variability, and that limitation.
4. Freeze the approved limits and comparison protocol before independent
   qualification reruns that were not used to set those limits.
5. Evaluate qualification against the frozen contract. Investigate excessive
   baseline variation or failures; do not adjust limits merely to obtain a pass.

Keep three questions distinct: same-seed/same-environment execution
repeatability, sensitivity to different training seeds, and G4/T4 portability.
Do not pool them into a single tolerance distribution. Passing repeatability
does not establish profitability, generalization, or a model improvement.

Numerical thresholds and qualification runs have **not** been approved or
performed by this decision record. Cheap smokes do not qualify the full frozen
recipe. The exact calibration job, period, K, repeat count, and budget must be
specified in a follow-up task without silently changing the recipe.

## Target Artifact And Provenance Contract

Every qualifying comparison must identify the code, all consumed raw inputs,
resolved configuration, seed schedule, environment, selection rule, and output
artifacts for each run. Report the actual determinism/precision settings rather
than treating a seed as a guarantee of equality. Reuse the existing run-bundle,
resolved-config, and selection-evidence mechanisms; MLflow can mirror evidence
but is not a prerequisite for inspecting the retained artifacts.

The implementation proposal is to retain a machine-readable comparison report
and a human-readable explanation with dated divergence rows, frequency views,
coverage/validation status, metric deltas, and the protocol/threshold version.
Every attempt, including failures and incomplete runs, must remain visible;
an absent or invalid run must not be presented as a successful comparison.
The exact schema and writer are follow-up engineering decisions, not features
already delivered here. Live Colab claims require visible execution and the
expected durable Drive artifacts per the runbook.

## Target Paper-Trade And Live-Operations Boundary

> To decide together.

## Migration Strategy

Implement incrementally through the existing maps. The table routes follow-up
work; it does not change ticket states, merge draft PRs, or grant run budgets.

| Existing owner | Connection to the accepted target |
| --- | --- |
| #140 / #142 | Reproducibility policy and canonical decision record; keep the map and child comments consistent |
| #143 | Interpreter/dependency role and reference software qualification; retain `pyproject.toml` as range authority and the documented Colab range file until a reviewed change |
| #144, coordinated with #191 / PR #201 | Runtime/code provenance and all-consumed-input identities; avoid parallel metadata implementations |
| #132 | Strict-mode deferral is answered; comparable-condition artifacts still require delivery |
| #187 / #192 | Raw-input identity mechanism remains an owner decision; both GPUs need the same verifiable inputs |
| #157 / #185 / PR #186 | Existing graph experiment and mechanics evidence; this policy does not expand their charter |

Provenance is needed before calibration evidence can be trusted. A comparator
and calibration/qualification run work must receive bounded tickets, owned
paths, public test seams, and appropriate run authorization through the
existing workflow. TDD and mutation checks belong to those implementations,
not to a documentation-only statement of future behavior.

## Decision Log

Rows below record explicit owner decisions from the Packet A exchange. They
do not assert implementation completion. Subsequent rows also require review.

| Date | Decision | Rationale | Alternatives considered | Invariants affected | Validation required |
| --- | --- | --- | --- | --- | --- |
| 2026-09-06 | [Environment-equivalent reruns with full provenance and measured tolerances](https://github.com/magilliam27/MCI-GRU/issues/142#issuecomment-5563299365) | Reconstruct and assess experiments | Provenance only; bitwise target | None changed | Fresh-session reference reruns with identities and comparison evidence |
| 2026-09-06 | [Defer optional strict determinism until repeatability evidence warrants it](https://github.com/magilliam27/MCI-GRU/issues/142#issuecomment-5563322123) | Measure the actual limitation before adding execution cost | Require strict mode immediately | None changed | Investigate material selection/conclusion instability |
| 2026-09-07 | [G4 first, with a T4 qualification plan alongside it](https://github.com/magilliam27/MCI-GRU/issues/142#issuecomment-5565150044) | Owner's cost/speed experience; prepare a lower-cost option | Qualify multiple environments immediately | None changed | Runtime identity, capacity/cost evidence and separate T4 qualification |
| 2026-09-07 | [Date-aligned base-2 JSD plus selection-frequency views](https://github.com/magilliam27/MCI-GRU/issues/142#issuecomment-5565211151) | Detect changed selections hidden by similar summary metrics | Raw KL; summary metrics or pooled frequencies alone | None changed | Aligned selections, defined mass, divergence and companion diagnostics |
| 2026-09-07 | [Baseline variation before threshold approval and independent qualification](https://github.com/magilliam27/MCI-GRU/issues/142#issuecomment-5565236846) | Avoid arbitrary thresholds and post-hoc tolerance inflation | Fix limits before measurements; adapt limits to qualification results | None changed | Bounded calibration, owner-reviewed limits, fresh qualifying reruns |

## Open Questions

- Calibration job, date window, K/selection policy, repeat count, compute budget,
  tail summaries, and numeric tolerances remain to be specified and reviewed.
- The G4 software profile and precise runtime validation fields must be tested;
  T4 capacity, cost, repeatability, and supported workload are not established.
- Dataset identity choice remains with #192; this document selects neither
  hardened sidecars nor DVC, LFS, or a custom store.
- Comparator schema, tie handling, empty/insufficient-candidate behavior,
  recovery granularity, and validation seams need bounded implementation specs.
- Model interfaces and the other architecture sections left as placeholders
  remain human-led design work, not decisions inherited from historical plans.

## Parked Or Rejected Ideas

- Optional strict-determinism support is deferred with the evidence trigger
  above. No permanent rejection of deterministic execution is implied.
- Additional environment qualification follows the G4 reference; the T4 plan
  is requested, but T4 is not currently certified by this policy.
