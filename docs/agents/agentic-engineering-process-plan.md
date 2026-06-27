# MCI-GRU Agentic Engineering Process Plan

Date: 2026-06-27

This plan adapts the AI engineering practices from `Day_1_v3 (1).pdf` to the
MCI-GRU workflow. The synthesis used focused reviewers for five areas of the
guide: context engineering, AI-driven SDLC, harness engineering, developer
orchestration, and economics/adoption.

## Executive Thesis

MCI-GRU should treat AI work as a controlled engineering factory, not as ad hoc
prompting. The useful shift is not "let agents write more code"; it is "design a
process where agents receive the right context, operate inside explicit
constraints, produce verifiable evidence, and preserve financial ML invariants."

The highest-leverage change is a risk-tiered operating model:

- Classify each task before work starts.
- Route it to the right working mode: conductor, terminal agent, background
  agent, or evidence curator.
- Attach invariant gates and evidence requirements to the task.
- Close out with reproducible commands, artifact paths, source-of-truth tier,
  and residual risk.

## Process Lanes

| Lane | Examples | Default Mode | Required Evidence |
| --- | --- | --- | --- |
| Exploration | Research scans, idea comparison, notebook sketches, factor hypotheses | Conductor or read-only background reviewers | Source list, assumptions, no production claims |
| Validated Experiment | New feature family, loss function screen, graph preset, Colab proof | Terminal agent with conductor review | Targeted tests, config diff, experiment recipe, evidence grade |
| Production/Paper-Trade | Defaults, paper_trade, inference boundaries, release candidates | Conductor-first, tightly scoped terminal agent | Invariant gates, non-slow or full suite, frozen artifact proof |
| Live Operations | Colab full runs, cockpit sync, GitHub sync, costly cloud work | Conductor plus explicit runbook | Visible runtime state, exact commands, Drive/GitHub artifacts |
| Documentation/Harness | Agent docs, skills, runbooks, templates, eval gates | Terminal agent | Readback, `git diff --check`, source-of-truth alignment |

Exploration can move quickly, but it must not be promoted by tone or plausibility.
Promotion requires evidence. A good experiment note says what it proves and what
it does not prove.

## Agent Routing Rules

Use the guide's conductor/orchestrator split deliberately:

| Task Shape | Route | Why |
| --- | --- | --- |
| Ambiguous correctness work touching no-lookahead, PIT, graph timing, labels, or paper_trade | Conductor | Human judgment and tight iteration are needed for the last 20 percent |
| Bounded multi-file code change with known acceptance criteria | Terminal agent | The agent can read, edit, test, and report within a scoped surface |
| Independent read-only reviews or separate test failures | Parallel background agents | They can gather evidence without conflicting edits |
| Artifact closeout or run evidence review | `mci_evidence_curator` or equivalent | Keeps evidence verification separate from implementation enthusiasm |
| Credentials, unexpected auth, expensive Colab relaunch, account recovery, or unclear cloud spend | Stop and ask | The risk is operational, not just technical |

Every delegated task should carry an invariant bundle, an allowed file scope, and
expected output. Do not delegate "make this better" on high-risk model/data
surfaces.

## Invariant Gate Matrix

These gates should become the default checklist for agent-authored MCI-GRU work.

| Gate | Trigger | Required Context | Preferred Proof |
| --- | --- | --- | --- |
| No lookahead | Features, labels, normalization, regimes, graph construction, backtests | `AGENTS.md`, `docs/TESTING_GUIDE.md`, relevant module | Future-row mutation test, train-period cutoff assertion, targeted pytest |
| PIT masked panel breadth | PIT universe, data loaders, panel filtering, masks | `AGENTS.md`, PIT reports, data manager code | Fixed PIT union axis preserved; no stayer-only or complete-stock filtering |
| Dynamic graph schedule | `graph.update_frequency_months`, graph builder, collate, snapshots | `docs/ARCHITECTURE.md`, `mci_gru/graph/builder.py`, data manager | `GraphSchedule.get_graph_for_date` behavior and snapshot cutoff tests |
| Collate 9-tuple contract | Data loader or graph feature changes | `AGENTS.md`, `mci_gru/data/data_manager.py` | Shape/arity test for `(time_series, labels, graph_features, edge_index, edge_weight, n_stocks, batch_dates, edge_index_sector, edge_weight_sector)` |
| Ensemble averaging | Training loop, prediction aggregation, evaluation | trainer code and tests | N independent models; prediction equals mean |
| Frozen paper-trade graph | `paper_trade/`, inference, checkpoint loading | `paper_trade/` docs/code | `GraphBuilder` is not imported; frozen `graph_data.pt` is loaded |
| Default experiment recipe | Production-style notebooks, PIT validation, confirmation runs | `docs/DEFAULT_EXPERIMENT_RECIPE.md`, configs | Deviations explicitly named as experimental factors |
| Colab evidence | Full-preset or costly notebook execution | `docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md`, `docs/NOTEBOOK_BEST_PRACTICES.md` | Visible G4/L4-class runtime evidence plus Drive heartbeat/results |

## Evidence Levels

Use evidence levels in plans, PRs, handoffs, and final run notes. Do not claim a
higher level than the artifacts support.

| Level | Name | What It Means |
| --- | --- | --- |
| E0 | Inference only | Reasoned from docs or memory; not currently verified |
| E1 | Readback | Current files, refs, or docs inspected in this checkout |
| E2 | Targeted local proof | Specific pytest, script, or grep guard passed |
| E3 | Repo health proof | Non-slow tests plus lint, or equivalent focused suite |
| E4 | Full local proof | Full test suite or release-level local validation passed |
| E5 | Live remote proof | Visible Colab/GitHub/cloud operation verified directly |
| E6 | Durable artifact proof | Drive/GitHub/run artifacts prove completion and are linked |

For Colab, notebook contract tests are not live-run proof. Browser state is not
artifact proof. Drive heartbeat/results and visible runtime evidence are the
truth for long training runs.

## AI Change Packet Template

Use this for every nontrivial agent task, PR description, or handoff.

```md
## Scope
- Lane:
- Mode: conductor / terminal-agent / background-agent / evidence-curator
- Branch/worktree:
- Files or modules in scope:
- Files or modules out of scope:

## Source Of Truth
- Code/tests inspected:
- Canonical docs inspected:
- Research evidence or handoffs used:
- Any stale or conflicting source:

## Invariant Gates
- No lookahead:
- PIT masked panel breadth:
- GraphSchedule/dynamic graph:
- Collate 9-tuple:
- Ensemble averaging:
- Paper-trade frozen graph:
- Colab/run evidence:

## Verification
- Commands run:
- Exit status:
- Pass/fail/skip summary:
- Artifacts produced or checked:
- Evidence level:

## Residual Risk
- Skipped checks:
- Assumptions:
- Recommended next action:
```

## SDLC Workflow

### 1. Intake

Start by classifying the lane and risk. The agent or human should answer:

- What invariant could this break?
- Which source of truth wins if docs disagree?
- Is this exploration, validation, production, or live operation?
- What evidence level is required before closeout?

### 2. Specification

For risky work, prose requirements are not enough. Translate intent into one or
more of:

- a saved regression test,
- a Hydra config or preset diff,
- a run matrix,
- a notebook contract,
- a checklist item in a run review,
- an artifact acceptance rule.

### 3. Execution

Keep context dense and scoped. Agents should load `AGENTS.md`, the relevant
canonical docs, and the exact files for the surface being changed. They should
avoid broad context dumps, stale handoffs, and unrelated refactors.

Use parallel agents only for independent questions. Each subtask needs a clear
scope, no shared write set, and an expected output format.

### 4. Verification

Use the smallest proof that directly exercises the change first, then broaden
when the surface justifies it.

Preferred Windows commands:

```powershell
New-Item -ItemType Directory -Force .tmp_pytest | Out-Null
$env:TMP = (Resolve-Path .tmp_pytest).Path
$env:TEMP = $env:TMP
.\.venv\Scripts\python.exe -m pytest tests/ -k "test_no_lookahead" -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\python.exe -m pytest tests/ -m "not slow" -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest tests/ -v --basetemp .tmp_pytest\pytest
```

### 5. Review

Review agent output for trajectory, not just diff quality:

- Did it load the right docs?
- Did it preserve source-of-truth hierarchy?
- Did it name touched invariants?
- Did it avoid stale run artifacts as proof?
- Did it run the right checks?
- Did it state skipped checks and residual risk?

### 6. Promotion

Promotion should be explicit:

- Exploration becomes a validated experiment only after focused local proof.
- A validated experiment becomes a production candidate only after invariant
  gates and reproducible run evidence.
- Paper-trade changes require frozen inference proof.
- Colab claims require live runtime and durable Drive artifacts.

## Harness Improvements To Implement

### Phase 0: Documentation and Templates

1. Add `docs/agents/harness.md` as the durable router from task type to context,
   verification, agent mode, and invariant gate.
2. Add the AI Change Packet to PR descriptions, handoffs, and agent task prompts.
3. Update `docs/index.md` to point agents to the harness document.
4. Keep `AGENTS.md` short; route deeper policy through `docs/agents/`.

### Phase 1: Deterministic Gates

1. Add or consolidate tests for:
   - collate 9-tuple shape,
   - dynamic graph schedule date resolution,
   - PIT masked-panel breadth preservation,
   - paper_trade frozen `graph_data.pt` boundary,
   - no-lookahead canaries for high-risk feature families.
2. Add a lightweight invariant check command or pytest marker bundle for agent
   closeout.
3. Add CI/pre-commit guardrails only for deterministic high-risk violations,
   starting with paper_trade importing `GraphBuilder`.

### Phase 2: Reusable Skills And Runbooks

Create or refine task-specific skills/runbooks:

- `no-lookahead-review`
- `pit-masked-panel-change`
- `graph-schedule-change`
- `paper-trade-repro`
- `colab-live-run`
- `experiment-promotion-review`

Each skill should state required files, forbidden shortcuts, commands, evidence
level, and closeout format.

### Phase 3: Operational Observability

1. Treat recurring Colab and cockpit workflows as production agents with scoped
   permissions, heartbeat artifacts, direct verification, and run reviews.
2. Track skipped checks and residual risks in cockpit or issue comments when
   they affect promotion decisions.
3. Periodically review agent harness docs like production code, because stale
   process docs are a real source of model/data risk.

## Anti-Patterns To Avoid

- Shipping from a fluent diff, demo, or notebook scrollback.
- Letting background agents handle ambiguous correctness work without close
  supervision.
- Treating handoffs or generated status docs as equal to code/tests.
- Mixing exploratory label scales, graph timings, or universe definitions into
  production scoring without an explicit audit.
- Using huge unstructured context instead of the source-of-truth hierarchy.
- Allowing "smoke proof" language to drift into "full validation" language.

## Adoption Checklist

This process is working when:

- Every nontrivial agent task has a lane, mode, invariant map, and evidence
  level.
- Handoffs and PRs state exact commands, artifact paths, skipped checks, and
  residual risk.
- Agents default to current code/tests and canonical docs over stale handoffs.
- Colab claims separate local notebook validation, visible runtime proof, and
  Drive artifact proof.
- Production/paper-trade changes cannot bypass frozen graph and no-lookahead
  checks.
- The harness itself is versioned, reviewed, and kept synchronized with the repo.
