# MCI-GRU Agentic Engineering Technical Spec

Date: 2026-06-30

Status: proposed implementation blueprint

Source plan: `docs/agents/agentic-engineering-process-plan.md`

This spec synthesizes five focused technical reviews into one implementable plan
for turning the agentic-engineering process plan into MCI-GRU repository
behavior. It is intentionally additive: the first implementation should not
change model defaults, PIT semantics, graph behavior, paper-trade inference, or
live Colab behavior.

## Current State

The current checkout already has:

- `docs/agents/agentic-engineering-process-plan.md`
- `docs/agents/domain.md`
- `docs/TESTING_GUIDE.md`
- `docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md`
- `docs/NOTEBOOK_BEST_PRACTICES.md`
- `scripts/ci_smoke.py`, including a synthetic run and a collate 9-tuple check
- basic cockpit evidence, model, render, and runner modules under
  `mci_gru/cockpit/`
- `.codex/agents/mci-evidence-curator.toml`

The current checkout does not have:

- `docs/agents/harness.md`
- `scripts/check_agentic_invariants.py`
- `tests/test_agentic_invariant_gates.py`
- structured cockpit evidence levels
- reusable workflow docs for the individual invariant surfaces
- repo-local skills for no-lookahead review or experiment promotion review

There is unrelated untracked scratch state at `pytest_tmp_review/`; it should not
be touched by this implementation.

## Goals

1. Make `docs/agents/harness.md` the canonical operational process router.
2. Convert invariant gates into a mix of docs, tests, and cheap static checks.
3. Add an evidence-level and closeout contract that can be used by humans,
   agents, PRs, handoffs, cockpit packets, and final answers.
4. Add focused workflow runbooks and only the two repo-local skills that need
   trigger semantics.
5. Extend live-operation observability for Colab and cockpit without launching
   expensive jobs.

## Non-Goals

- Do not retrain models.
- Do not launch Colab.
- Do not mutate GitHub except during a later explicit live cockpit-sync
  verification pass.
- Do not change the default experiment recipe.
- Do not change PIT masked-panel behavior.
- Do not change dynamic graph construction semantics.
- Do not change paper-trade inference to build graphs.

## Architecture

Use a four-layer implementation:

1. **Process docs:** `docs/agents/harness.md` owns lanes, routing, evidence
   levels, invariant gates, verification ladder, and AI Change Packet.
2. **Deterministic checks:** pytest protects behavioral invariants; a cheap
   static script protects deterministic text/AST boundaries.
3. **Reporting model:** cockpit gains typed evidence-level and verification
   records as additive metadata.
4. **Workflow runbooks and skills:** detailed operating guides live under
   `docs/workflows/`; only cross-cutting review/promotion behaviors become
   repo-local Codex skills.

The existing source-of-truth hierarchy remains unchanged:

1. current code and tests,
2. `AGENTS.md` invariants,
3. canonical docs,
4. current research evidence,
5. handoffs and historical references.

## Phase 0: Canonical Harness Document

### Scope

Create the canonical process router and point the existing entrypoints at it
without bloating `AGENTS.md`.

### Files

- Create: `docs/agents/harness.md`
- Modify: `AGENTS.md`
- Modify: `docs/index.md`
- Modify: `docs/agents/agentic-engineering-process-plan.md`
- Modify: `docs/agents/domain.md`
- Modify: `docs/TESTING_GUIDE.md`
- Modify: `docs/agents/cockpit/RUNBOOK.md`

### `docs/agents/harness.md` Structure

Use these exact top-level headings:

```md
# Agent Harness

## Purpose
## Source-Of-Truth Order
## Process Lanes
## Agent Routing Rules
## Invariant Gate Matrix
## Evidence Levels
## Evidence Transitions
## Verification Ladder
## AI Change Packet Template
## Usage Requirements
## Closeout Review Checklist
```

The `Process Lanes` table must include:

| Lane | Examples | Default Mode | Required Evidence | Minimum Evidence Level |
| --- | --- | --- | --- | --- |
| Exploration | Research scans, idea comparison, notebook sketches, factor hypotheses | Conductor or read-only background reviewers | Source list, assumptions, no production claims | E1 |
| Validated Experiment | New feature family, loss screen, graph preset, Colab proof | Terminal agent with conductor review | Targeted tests, config diff, experiment recipe, evidence grade | E2 |
| Production/Paper-Trade | Defaults, `paper_trade/`, inference boundaries, release candidates | Conductor-first, tightly scoped terminal agent | Invariant gates, repo-health proof, frozen artifact proof | E3/E4 |
| Live Operations | Colab full runs, cockpit sync, GitHub sync, cloud work | Conductor plus explicit runbook | Visible runtime state, exact commands, Drive/GitHub artifacts | E5/E6 |
| Documentation/Harness | Agent docs, skills, runbooks, templates, eval gates | Terminal agent | Readback, `git diff --check`, source-of-truth alignment | E1/E2 |

The `Agent Routing Rules` table must include:

| Task Shape | Route | Stop/Ask Trigger |
| --- | --- | --- |
| Ambiguous correctness work touching no-lookahead, PIT, graph timing, labels, or `paper_trade/` | Conductor | Scope or invariant risk is unclear |
| Bounded multi-file change with known acceptance criteria | Terminal agent | Write scope overlaps another active worker |
| Independent read-only reviews or separate failures | Parallel background agents | Findings require shared mutable state |
| Artifact closeout or run evidence review | `mci_evidence_curator` or equivalent | Artifact target is missing |
| Credentials, unexpected auth, expensive relaunch, account recovery, or unclear cloud spend | Stop and ask | Always |

### AGENTS Routing

Add exactly one compact bullet under `How to Work in This Repo`:

```md
- **For nontrivial agent-authored work**, classify the process lane,
  invariant gates, and evidence level using `docs/agents/harness.md`; keep
  closeout commands and residual risk in the AI Change Packet.
```

Do not duplicate the harness tables in `AGENTS.md`.

### Index Routing

Add one row under `Agent Docs And Skills` in `docs/index.md`:

```md
| [agents/harness.md](agents/harness.md) | Agent process lanes, routing rules, invariant gates, evidence levels, and AI Change Packet template. |
```

### Process Plan Update

Keep `docs/agents/agentic-engineering-process-plan.md` as the strategic plan,
but add this pointer near the top:

```md
The canonical, current harness is `docs/agents/harness.md`. Keep lane routing,
evidence levels, invariant gates, and the AI Change Packet template there.
```

The plan may retain rationale and rollout phases, but it must not be the
operative template source.

### AI Change Packet

Canonical location: `docs/agents/harness.md`.

Exact template:

```md
## AI Change Packet

### Scope
- Lane:
- Mode: conductor / terminal-agent / background-agent / evidence-curator
- Branch/worktree:
- Files or modules in scope:
- Files or modules out of scope:

### Source Of Truth
- Code/tests inspected:
- Canonical docs inspected:
- Research evidence or handoffs used:
- Stale or conflicting sources:

### Invariant Gates
- No lookahead:
- PIT masked panel breadth:
- GraphSchedule/dynamic graph:
- Collate 9-tuple:
- Ensemble averaging:
- Paper-trade frozen graph:
- Default experiment recipe:
- Colab/live-run evidence:

### Verification
- Commands run:
- Exit status:
- Pass/fail/skip summary:
- Artifacts produced or checked:
- Evidence level:

### Residual Risk
- Skipped checks:
- Assumptions:
- Recommended next action:
```

### Acceptance Criteria

- `docs/agents/harness.md` exists and is the only canonical home for lanes,
  routing, evidence levels, invariant gates, and the AI Change Packet template.
- `AGENTS.md` remains short.
- `docs/index.md` links the harness.
- `docs/agents/domain.md` remains the source-of-truth hierarchy and points to
  the harness only as process policy.
- `docs/TESTING_GUIDE.md` maps its verification ladder to the evidence levels.
- `docs/agents/cockpit/RUNBOOK.md` states that run color and evidence level are
  separate concepts.

### Verification

```powershell
git diff --check -- AGENTS.md docs/index.md docs/agents/domain.md docs/TESTING_GUIDE.md docs/agents/cockpit/RUNBOOK.md docs/agents/agentic-engineering-process-plan.md docs/agents/harness.md
rg -n "harness.md|AI Change Packet|Process Lanes|Evidence Levels" AGENTS.md docs/index.md docs/agents docs/TESTING_GUIDE.md
```

## Phase 1: Invariant Gates And Static Checks

### Scope

Back deterministic process risks with tests or a cheap static checker. Keep live
or expensive evidence out of CI.

### Files

- Create: `scripts/check_agentic_invariants.py`
- Create: `tests/test_agentic_invariant_gates.py`
- Modify: `scripts/ci_smoke.py`
- Modify: `docs/TESTING_GUIDE.md`
- Modify: `docs/agents/harness.md`

### Gate Classification

| Gate | Implementation Level |
| --- | --- |
| No lookahead | Behavioral tests when timing/data code changes; docs checklist for unrelated changes |
| PIT masked-panel breadth | Existing and extended pytest coverage |
| Dynamic `GraphSchedule` | Existing and extended pytest coverage |
| Collate 9-tuple | Existing `scripts/ci_smoke.py` plus pytest |
| Ensemble averaging | New deterministic pytest |
| Paper-trade frozen graph | New static/AST check plus focused pytest |
| Default experiment recipe | Static consistency check plus targeted notebook/preset tests |
| Colab evidence boundary | Docs/static checks only; live proof remains non-CI |

### Static Checker

Create `scripts/check_agentic_invariants.py` with this public surface:

```python
from pathlib import Path

def run_static_ci(root: Path) -> list[InvariantFinding]: ...
def check_paper_trade_frozen_graph(root: Path) -> list[InvariantFinding]: ...
def check_default_recipe_contract(root: Path) -> list[InvariantFinding]: ...
def check_colab_evidence_docs(root: Path) -> list[InvariantFinding]: ...
```

Expected behavior:

- scan `paper_trade/**/*.py` with `ast` and fail on `GraphBuilder` or
  `mci_gru.graph.builder` imports,
- allow graph utility imports such as edge feature dimension helpers,
- assert `paper_trade/scripts/infer.py` requires `graph_data.pt` and reads
  `edge_index` / `edge_weight` from the frozen artifact,
- assert `AGENTS.md` and `docs/DEFAULT_EXPERIMENT_RECIPE.md` share the canonical
  default recipe slug,
- assert Colab docs contain the evidence boundary phrases:
  - visible G4 or L4-class runtime,
  - reject or do not accept T4 for full-preset runs,
  - `heartbeat.json`,
  - notebook contract tests are not live-run proof.

The script must support:

```powershell
.\.venv\Scripts\python.exe scripts\check_agentic_invariants.py --profile static-ci
```

The static profile must not run pytest, access the network, call GitHub, open
Chrome, inspect Drive, load market data, or launch notebooks.

### Behavioral Tests

Create `tests/test_agentic_invariant_gates.py`.

It should cover:

- paper-trade static checker findings are empty for the current tree,
- default recipe slug is consistent between `AGENTS.md` and
  `docs/DEFAULT_EXPERIMENT_RECIPE.md`,
- Colab evidence docs preserve visible runtime and Drive artifact language,
- ensemble averaging is the arithmetic mean of independently produced model
  predictions.

The ensemble test should monkeypatch the narrow prediction-producing internals
around `train_multiple_models` rather than running real training. It should
verify both the returned prediction data and any saved `averaged_predictions`
rows when the local implementation writes them.

Existing behavioral proof sources remain:

- `tests/test_dynamic_graph_updates.py`
- `tests/test_pit_masked_panel.py`
- collate check inside `scripts/ci_smoke.py`
- paper-trade focused tests if present

### CI Smoke Integration

Modify `scripts/ci_smoke.py` after `_assert_collate_contract()`:

- import `run_static_ci` from `scripts/check_agentic_invariants.py`,
- call it with `PROJECT_ROOT`,
- print actionable findings and return nonzero if any findings exist.

The runtime should remain dominated by the existing synthetic experiment. The
new static pass should be text/AST only.

### Acceptance Criteria

- `scripts/check_agentic_invariants.py --profile static-ci` exits nonzero with
  actionable findings if `paper_trade/` imports `GraphBuilder`.
- The static checker catches default-recipe or Colab-evidence doc drift.
- `tests/test_agentic_invariant_gates.py` covers static checks and ensemble mean
  aggregation.
- `scripts/ci_smoke.py` still performs the synthetic smoke run and collate
  9-tuple check.

### Verification

```powershell
.\.venv\Scripts\python.exe scripts\check_agentic_invariants.py --profile static-ci
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_invariant_gates.py tests/test_dynamic_graph_updates.py tests/test_pit_masked_panel.py -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\python.exe scripts\ci_smoke.py
.\.venv\Scripts\ruff.exe check scripts/check_agentic_invariants.py tests/test_agentic_invariant_gates.py scripts/ci_smoke.py
```

## Phase 2: Evidence Levels And Closeout Reporting

### Scope

Make evidence levels a typed reporting concept for cockpit while keeping the
canonical taxonomy in `docs/agents/harness.md`.

### Files

- Modify: `docs/agents/harness.md`
- Modify: `docs/TESTING_GUIDE.md`
- Modify: `docs/agents/cockpit/RUNBOOK.md`
- Modify: `mci_gru/cockpit/models.py`
- Modify: `mci_gru/cockpit/render.py`
- Modify: `mci_gru/cockpit/runner.py`
- Modify: `tests/test_cockpit_render.py`
- Modify: `tests/test_cockpit_runner.py`

### Evidence Levels

Canonical levels:

| Level | Name | Minimum Proof | Promotion Rule |
| --- | --- | --- | --- |
| E0 | Inference only | Reasoning from memory, stale docs, or unverified assumptions | Promote only after current checkout readback |
| E1 | Readback | Current files, refs, docs, GitHub metadata, or Drive metadata inspected | Promote with targeted command or artifact inspection |
| E2 | Targeted local proof | Focused pytest, grep guard, script, or artifact check passed | Promote with repo-health proof |
| E3 | Repo health proof | Non-slow suite plus lint, or focused invariant suite for touched surface | Promote with full local validation |
| E4 | Full local proof | Full test suite, release-level smoke, or complete local validation passed | Promote only through live remote verification |
| E5 | Live remote proof | GitHub, Colab, or cloud operation directly observed or queried | Promote only when durable artifacts are inspectable |
| E6 | Durable artifact proof | Drive, GitHub, or run artifacts prove completion and are linked | Terminal unless artifacts disappear or contradict claims |

Allowed transitions:

- monotonic promotions,
- same-level refreshes,
- explicit downgrades.

Downgrades must record one cause:

- missing artifact,
- stale evidence,
- failed rerun,
- contradictory source,
- scope expansion.

A broad claim inherits the weakest required subclaim.

### Cockpit Data Model

Add these to `mci_gru/cockpit/models.py`:

```python
class EvidenceLevel(StrEnum):
    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    E6 = "E6"


@dataclass(frozen=True)
class VerificationRecord:
    evidence_level: EvidenceLevel
    commands: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    exit_status: str = ""
    residual_risk: str = ""
```

Add defaulted fields to `CockpitReport`:

```python
verification_records: list[VerificationRecord] = field(default_factory=list)
residual_risks: list[str] = field(default_factory=list)
skipped_checks: list[str] = field(default_factory=list)
```

Do not remove `verification_notes` or `evidence_gaps`; keep existing tests and
packet structure compatible.

### Rendering

Append new packet sections in `mci_gru/cockpit/render.py`:

- `## Evidence Level`
- `## Verification Records`
- `## Skipped Checks`
- `## Residual Risk`

Do not rename existing headings, especially:

- `## GitHub Actions Proposed Or Skipped`
- `## Verification Notes`
- `## Evidence Gaps And Contradictions`

### Reporting Contract

Final answers, handoffs, PR text, and cockpit comments should include:

```md
Evidence Level: E# - name
Commands:
- <exact command> -> <exit status/result>
Artifacts:
- <path, URL, PR, issue, Drive ID, run root, or "none">
Residual Risk:
- <skipped checks, unavailable systems, stale/contradictory evidence>
Closeout Verdict:
- supports_closeout | needs_more_evidence | blocked_missing_artifact | not_supported
```

The `.codex/agents/mci-evidence-curator.toml` verdict tokens remain closeout
decisions; they are not evidence levels.

### Acceptance Criteria

- Harness defines E0-E6 and downgrade rules.
- `docs/TESTING_GUIDE.md` maps targeted, repo-health, and full-confidence proof
  to E2/E3/E4.
- Cockpit packet includes evidence-level and residual-risk sections.
- Existing direct `CockpitReport(...)` constructions in tests still work because
  new fields are defaulted.

### Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cockpit_render.py tests/test_cockpit_runner.py -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\ruff.exe check mci_gru/cockpit tests/test_cockpit_render.py tests/test_cockpit_runner.py
rg -n "Evidence Level|Closeout Verdict|AI Change Packet|docs/agents/harness.md" docs mci_gru .codex/agents
```

## Phase 3: Reusable Workflow Docs And Skills

### Scope

Move detailed operational guidance into focused workflow docs. Add Codex skills
only where a trigger and fixed output contract provide real value.

### Files

Create workflow docs:

- `docs/workflows/no-lookahead-review.md`
- `docs/workflows/pit-masked-panel-change.md`
- `docs/workflows/graph-schedule-change.md`
- `docs/workflows/paper-trade-repro.md`
- `docs/workflows/colab-live-run.md`
- `docs/workflows/experiment-promotion-review.md`

Create skills:

- `skills/no-lookahead-review/SKILL.md`
- `skills/experiment-promotion-review/SKILL.md`

Modify:

- `AGENTS.md`
- `docs/index.md`
- `skills/research-paper-to-mci-gru/SKILL.md` only if a short routing pointer
  is useful and does not expand its scope.

### Docs Versus Skills

| Proposed Item | Implementation | Rationale |
| --- | --- | --- |
| `no-lookahead-review` | skill plus workflow doc | Cross-cutting review trigger |
| `pit-masked-panel-change` | workflow doc | Scoped change checklist |
| `graph-schedule-change` | workflow doc | Scoped graph-change checklist |
| `paper-trade-repro` | workflow doc | Repro checklist, not live trading agent |
| `colab-live-run` | thin workflow wrapper over Chrome guide | Existing Chrome guide remains canonical |
| `experiment-promotion-review` | skill plus workflow doc | Needs fixed verdict/evidence packet |

### Required Runbook Headings

Each workflow doc must use:

```md
# <Name>

## Triggers
## Required Files
## Forbidden Shortcuts
## Commands
## Evidence Level
## Closeout Format
```

### Required Runbook Content

`no-lookahead-review.md`:

- triggers: features, labels, normalization, graph construction, backtests,
  regime data,
- required files: `AGENTS.md`, `docs/agents/domain.md`, `docs/TESTING_GUIDE.md`,
  touched modules/tests,
- forbidden: claiming safety from prose, using handoffs as proof, accepting
  future-row mutation,
- command: `.\.venv\Scripts\python.exe -m pytest tests/ -k "test_no_lookahead" -v --basetemp .tmp_pytest\pytest`,
- evidence: E1 minimum for review, E2 or higher for code changes.

`pit-masked-panel-change.md`:

- required files: PIT reports, `mci_gru/data/pit.py`, `mci_gru/pipeline.py`,
  `mci_gru/data/data_manager.py`, `tests/test_pit_masked_panel.py`,
- forbidden: complete-stock filtering, stayer filtering, collapsing fixed PIT
  union axis,
- command: `.\.venv\Scripts\python.exe -m pytest tests/test_pit_masked_panel.py -v --basetemp .tmp_pytest\pytest`,
- evidence: E2 for local change, E5/E6 for Colab/full-run claims.

`graph-schedule-change.md`:

- required files: `docs/ARCHITECTURE.md`, `mci_gru/graph/builder.py`,
  `mci_gru/data/data_manager.py`, dynamic graph plan reference,
  `tests/test_dynamic_graph_updates.py`,
- forbidden: per-batch graph recomputation, sample-date lookahead, breaking
  9-tuple or edge-width contract,
- command: `.\.venv\Scripts\python.exe -m pytest tests/test_dynamic_graph_updates.py tests/test_phase3_graph_and_walkforward.py -v --basetemp .tmp_pytest\pytest`,
- evidence: E2 minimum, E3 if shared model/data-loader contracts are touched.

`paper-trade-repro.md`:

- required files: `AGENTS.md`, `docs/ARCHITECTURE.md`,
  `docs/OUTPUT_MANAGEMENT.md`, `paper_trade/scripts/infer.py`,
  `paper_trade/scripts/portfolio.py`, `paper_trade/scripts/run_nightly.py`,
- forbidden: importing `GraphBuilder`, rebuilding research graphs, using
  gitignored artifacts as source of truth without provenance,
- command: `rg -n "GraphBuilder" paper_trade`,
- evidence: E2 for local repro, E6 for production artifact claims.

`colab-live-run.md`:

- required files: existing Chrome guide, notebook-specific handoff/issue,
  `docs/NOTEBOOK_BEST_PRACTICES.md`, `docs/DEFAULT_EXPERIMENT_RECIPE.md`,
- forbidden: hidden kernels for full runs, accepting T4 for full-preset without
  explicit user approval, claiming success from notebook contract tests or
  browser cleanup,
- evidence: E5 for visible runtime, E6 for Drive heartbeat/results,
- closeout: reuse the Colab Run Review format.

`experiment-promotion-review.md`:

- required files: `AGENTS.md`, `docs/agents/domain.md`,
  `docs/research/README.md`, default recipe, run report/artifact target,
  relevant invariant runbook,
- forbidden: promoting from fluent summary, stale handoff, notebook scrollback,
  missing artifact target, or cheap proof mislabeled as full validation,
- evidence: declared E-level; production/paper-trade promotion should normally
  require E4 plus E6 where artifacts matter,
- closeout: AI Change Packet plus `promote`, `defer`, or `block` verdict.

### Skill Requirements

Each new `SKILL.md` should be small:

- frontmatter with `name` and `description`,
- required runbook read,
- required output format,
- explicit prohibition on expensive jobs unless the parent task asks.

`experiment-promotion-review` may call `mci_evidence_curator` only after a
specific artifact target and issue/report decision are supplied. The curator
must remain artifact-only and fail closed.

### Acceptance Criteria

- all six workflow docs exist,
- both skill files have valid frontmatter,
- every runbook has the six required headings,
- `docs/index.md` links the workflow family compactly,
- `AGENTS.md` remains brief,
- no runbook treats handoffs, notebook contract tests, or browser state as
  artifact proof.

### Verification

```powershell
Test-Path docs/workflows/no-lookahead-review.md
Test-Path docs/workflows/pit-masked-panel-change.md
Test-Path docs/workflows/graph-schedule-change.md
Test-Path docs/workflows/paper-trade-repro.md
Test-Path docs/workflows/colab-live-run.md
Test-Path docs/workflows/experiment-promotion-review.md
Test-Path skills/no-lookahead-review/SKILL.md
Test-Path skills/experiment-promotion-review/SKILL.md
rg -n "## Triggers|## Required Files|## Forbidden Shortcuts|## Commands|## Evidence Level|## Closeout Format" docs/workflows
rg -n "^---|^name:|^description:|docs/workflows/" skills/no-lookahead-review/SKILL.md skills/experiment-promotion-review/SKILL.md
rg -n "GraphBuilder|masked_panel|GraphSchedule|heartbeat.json|supports_closeout|AI Change Packet" docs/workflows skills
git diff --check
```

## Phase 4: Live Operations Observability

### Scope

Represent Colab run reviews and cockpit sync evidence using the shared evidence
contract, while keeping live operations out of routine CI.

### Files

Docs:

- Modify: `docs/agents/harness.md`
- Modify: `docs/agents/cockpit/RUNBOOK.md`
- Modify: `docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md`
- Modify: `docs/NOTEBOOK_BEST_PRACTICES.md`
- Modify: `docs/workflows/colab-live-run.md`

Code:

- Modify: `mci_gru/cockpit/models.py`
- Modify: `mci_gru/cockpit/evidence.py`
- Modify: `mci_gru/cockpit/runner.py`
- Modify: `mci_gru/cockpit/github.py`
- Modify: `mci_gru/cockpit/render.py`
- Modify: `scripts/refresh_cockpit.py`

Tests:

- Modify: `tests/test_cockpit_runner.py`
- Modify: `tests/test_cockpit_render.py`
- Modify: `tests/test_cockpit_cli.py`
- Modify: `tests/test_cockpit_github.py`

### Colab Run Review Schema

Every live Colab attempt should leave a run-review artifact with:

```md
## Colab Run Review
- Evidence level: E5 visible runtime / E6 durable Drive artifacts
- Branch and notebook URL:
- Surface: chrome:control-chrome or fallback reason
- Runtime accepted from matrix:
- Visible runtime evidence:
- In-notebook GPU gate evidence:
- Drive artifact root:
- Drive artifacts checked: heartbeat.json, ensemble_progress.json, training_results.csv/json, summary/manifest
- Prompts handled:
- Cells/phases run:
- Outcome: succeeded / failed / blocked / stopped proof / handed off
- Failure taxonomy:
- Cleanup state:
- Residual risk:
```

Evidence boundaries:

- local notebook JSON or contract checks are E1/E2 only,
- visible Colab runtime and GPU gate are E5,
- Drive heartbeat/results/manifest evidence is E6,
- browser tab state alone is never E6,
- a killed proof run is automation proof, not model validation.

### Cockpit Observability

Extend cockpit refresh without running training, Colab, data-vendor probes, or
other expensive jobs.

Add read-only evidence collection:

- `git status --short --branch`,
- `git worktree list --porcelain`,
- `git branch --all --no-merged origin/main`,
- required-doc gaps,
- current branch,
- ahead/behind when available,
- dirty/detached worktree summary.

For GitHub-sync mode, verify after mutation:

- PR URL/state,
- cockpit issue comment,
- PR checks status,
- skipped label or check actions.

Do not equate GitHub sync success with green CI.

### Run Color Rules

Use topology-aware colors:

- `green`: no dirty paths, no topology attention, required docs present, no sync
  contradictions,
- `yellow`: skipped non-critical checks, stale/superseded cockpit PRs,
  local-only branches needing review, GitHub checks pending/unstable,
- `red`: dirty non-cockpit paths, missing required docs, failed sync, ambiguous
  branch continuation, main divergence blocking trustworthy refresh, or failed
  required verification.

### Backward Compatibility

- Do not change the workstream register table columns.
- Do not rename existing packet headings.
- Add new `CockpitReport` fields with defaults only.
- Update tests that construct `CockpitReport` directly only when they assert the
  new sections.
- After `sync_github()`, rewrite the cockpit packet with post-sync evidence so a
  synced packet does not still say "GitHub sync skipped."

### Acceptance Criteria

- process docs define Colab Run Review and Cockpit Sync Review schemas,
- cockpit packet shows evidence level, skipped checks, residual risk, GitHub
  verification, and git topology,
- local-only mode remains non-mutating and clearly says GitHub sync skipped,
- GitHub-sync mode verifies PR, issue comment, and PR checks separately,
- no expensive jobs are run by cockpit refresh,
- existing packet/register rendering remains compatible.

### Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cockpit_runner.py tests/test_cockpit_render.py tests/test_cockpit_cli.py tests/test_cockpit_github.py -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\ruff.exe check mci_gru/cockpit scripts/refresh_cockpit.py tests/test_cockpit_runner.py tests/test_cockpit_render.py tests/test_cockpit_cli.py tests/test_cockpit_github.py
git diff --check
```

Optional live acceptance after explicit user approval:

```powershell
.\.venv\Scripts\python.exe scripts\refresh_cockpit.py --date YYYY-MM-DD --github-sync
gh pr view <cockpit-pr>
gh issue view 38 --comments
gh pr checks <cockpit-pr>
```

## Rollout Order

Implement in this order:

1. Phase 0 docs harness and routing.
2. Phase 1 static checker and invariant tests.
3. Phase 2 evidence-level model and cockpit rendering.
4. Phase 3 workflow runbooks and two small repo-local skills.
5. Phase 4 cockpit and live-ops observability.

This order keeps the source-of-truth router stable before code starts depending
on it, then adds cheap deterministic safeguards, then expands generated
operational reporting.

## Combined Verification Matrix

| Phase | Minimum Verification |
| --- | --- |
| Phase 0 | `git diff --check` plus `rg` readback |
| Phase 1 | static checker, `tests/test_agentic_invariant_gates.py`, dynamic graph/PIT tests, `scripts/ci_smoke.py`, ruff on touched files |
| Phase 2 | cockpit render/runner tests, ruff on cockpit modules, readback for evidence-level docs |
| Phase 3 | `Test-Path`, `rg` heading/frontmatter checks, `git diff --check` |
| Phase 4 | cockpit runner/render/CLI/GitHub tests, ruff on cockpit modules, optional live GitHub sync only with approval |

Before publishing a full implementation branch, run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -m "not slow" -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe scripts\ci_smoke.py
git diff --check
```

## Open Decisions

1. Whether `paper-trade-repro` should eventually become a skill. Start as a
   workflow doc because live trading/repro work should remain explicit and
   parent-authorized.
2. Whether `scripts/check_agentic_invariants.py` should be wired into GitHub
   Actions immediately. Start with `scripts/ci_smoke.py` and local verification;
   promote to CI only after it proves stable.
3. Whether cockpit local-only refresh should ever reach E2 automatically. The
   safest default is E1 for generation alone and E2 only when focused tests or a
   local script verification ran in the same closeout.
